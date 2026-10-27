"""
RNN sequence packing utilities.
All computation uses the C++ engine — no numpy.
"""

import struct
from collections.abc import Sequence
from itertools import pairwise
from typing import NamedTuple

import lucid
from lucid._tensor.tensor import Tensor as _Tensor
from lucid._tensor.tensor import Tensor  # public alias for NamedTuple field annotations
from lucid._C import engine as _C_engine

# Integer-dtype → struct format-code map, used by _int_tensor_to_list to
# unpack a 1-D int tensor's raw bytes into a Python list without going
# through numpy.  Apple Silicon is little-endian; ``=`` uses native byte
# order without struct padding, which matches the engine's contiguous
# bytes layout exactly.
_INT_DTYPE_STRUCT: dict[_C_engine.Dtype, str] = {
    _C_engine.Dtype.I8: "b",
    _C_engine.Dtype.I16: "h",
    _C_engine.Dtype.I32: "i",
    _C_engine.Dtype.I64: "q",
    _C_engine.Dtype.Bool: "?",
}


def _int_tensor_to_list(t: Tensor) -> list[int]:
    """Convert a 1-D integer tensor to a flat Python list of ints — numpy-free.

    Reads the tensor's contiguous bytes through ``to_bytes`` (which exists
    precisely to keep paths off the numpy bridge) and unpacks them with
    ``struct.unpack``.  Used by ``pack_padded_sequence`` /
    ``pad_packed_sequence`` to inspect ``lengths`` / ``batch_sizes`` /
    ``unsorted_indices`` — small metadata tensors where numpy was vastly
    overkill.

    Raises ``TypeError`` for non-integer dtypes (we intentionally don't
    silently coerce floats — the caller should pass an int tensor).
    """
    n = t._impl.numel()
    if n == 0:
        return []
    fmt = _INT_DTYPE_STRUCT.get(t._impl.dtype)
    if fmt is None:
        raise TypeError(
            f"_int_tensor_to_list expects an integer tensor; got dtype {t._impl.dtype!r}"
        )
    raw = t._impl.to_bytes()
    return list(struct.unpack(f"={n}{fmt}", raw))


def _length_list(lengths: Tensor | Sequence[int]) -> list[int]:
    """``lengths`` as Python ints — a 1-D sequence on any device.

    A list goes through the same tensor conversion as a tensor so both
    spellings meet one rank rule; non-integer values truncate toward zero,
    as an int64 cast does.
    """
    t = lengths if isinstance(lengths, _Tensor) else lucid.tensor(lengths)
    if t.ndim != 1:
        raise ValueError(
            f"'lengths' must be a 1-D sequence of per-sample lengths, "
            f"got {t.ndim}-D with shape {tuple(t.shape)}"
        )
    if t._impl.dtype not in _INT_DTYPE_STRUCT:
        t = t.to(lucid.int64)
    return [int(v) for v in _int_tensor_to_list(t)]


def _check_lengths(lengths: list[int], batch: int, steps: int, enforce_sorted: bool) -> None:
    """Refuse a ``lengths`` that would build a corrupt ``batch_sizes``.

    Every packing entry point validates here, before anything is gathered:
    a length of 0 or less, or longer than the padded time axis, silently
    produced a ``batch_sizes`` that no longer matched ``data``.
    """
    if len(lengths) != batch:
        raise ValueError(
            f"Expected `len(lengths)` to be equal to batch_size, but got "
            f"{len(lengths)} (batch_size={batch})"
        )
    if any(n <= 0 for n in lengths):
        raise ValueError(
            "Length of all samples has to be greater than 0, but found an "
            "element in 'lengths' that is <= 0"
        )
    longest = max(lengths)
    if longest > steps:
        raise ValueError(
            f"Found a length of {longest} in 'lengths', longer than the "
            f"padded time axis ({steps} steps)"
        )
    if enforce_sorted and any(a < b for a, b in pairwise(lengths)):
        raise ValueError(
            "`lengths` array must be sorted in decreasing order when "
            "`enforce_sorted` is True. Pass `enforce_sorted=False` to "
            "pack_padded_sequence and/or pack_sequence to sort internally."
        )


def _index(values: list[int], like: Tensor) -> Tensor:
    """An int64 index tensor on ``like``'s device.

    Every index these helpers build — the sort permutation, its inverse, the
    unsort gather — goes through here, so none of them lands on the default
    device while the data sits on another.
    """
    return lucid.tensor(values, dtype=lucid.int64, device=like.device)


def _cpu_int64(values: list[int]) -> Tensor:
    """Length metadata (``batch_sizes``, returned lengths) — CPU int64 like the reference."""
    return lucid.tensor(values, dtype=lucid.int64, device="cpu")


def _batch_size_list(batch_sizes: Tensor, rows: int) -> list[int]:
    """A packed sequence's ``batch_sizes`` as ints, refused unless it describes ``rows`` rows."""
    if batch_sizes.ndim != 1:
        raise ValueError(f"batch_sizes must be 1-D, got shape {tuple(batch_sizes.shape)}")
    sizes = [int(v) for v in _int_tensor_to_list(batch_sizes)]
    if not sizes or sizes[-1] <= 0 or any(a < b for a, b in pairwise(sizes)):
        raise ValueError(
            f"batch_sizes must be a non-empty, non-increasing run of positive "
            f"counts, got {sizes}"
        )
    if sum(sizes) != rows:
        raise ValueError(f"batch_sizes sums to {sum(sizes)} but the packed data has {rows} rows")
    return sizes


def _check_sequences(sequences: Sequence[Tensor]) -> None:
    """Refuse a sequence list that cannot stack into one padded batch."""
    if not sequences:
        raise ValueError("received an empty list of sequences")
    first = sequences[0]
    for i, s in enumerate(sequences):
        if s.ndim == 0:
            raise ValueError(f"sequence {i} is 0-d; every sequence needs a leading time axis")
        if tuple(s.shape[1:]) != tuple(first.shape[1:]):
            raise _C_engine.ShapeMismatch(
                f"sequence {i} has trailing shape {tuple(s.shape[1:])}, but "
                f"sequence 0 has {tuple(first.shape[1:])}; only the leading "
                f"time axis may differ"
            )
        if s.device != first.device:
            raise _C_engine.DeviceMismatch(
                f"sequence {i} is on {s.device}, but sequence 0 is on " f"{first.device}"
            )


class PackedSequence(NamedTuple):
    r"""Compact representation of a batch of variable-length sequences.

    A ``PackedSequence`` interleaves all surviving time-steps of every
    sequence in the batch into a single flat tensor, dropping the
    padding entries entirely.  RNN cells consume this form to avoid
    wasting compute on padded positions.

    Attributes
    ----------
    data : Tensor
        Concatenation of the active features at each time-step in
        descending-length order, shape ``(sum(batch_sizes), *feat)``.
    batch_sizes : Tensor
        1-D int tensor giving the number of sequences still alive at
        each successive time-step.  Strictly non-increasing.
    sorted_indices : Tensor or None
        Permutation that took the original batch order to the packed
        (descending-length) order; ``None`` if the input was already
        sorted.
    unsorted_indices : Tensor or None
        Inverse of ``sorted_indices``; used by
        :func:`pad_packed_sequence` to restore the caller's original
        batch order on unpack.

    Notes
    -----
    Conceptually, with sequence lengths :math:`\ell_1 \geq \dots \geq
    \ell_B`, the packed layout walks time-major:

    .. math::

        \text{data} = \bigl[\,x^{(1)}_0, \dots, x^{(B)}_0,\;
                            x^{(1)}_1, \dots, x^{(B_1)}_1,\;
                            \dots\,\bigr],

    where :math:`B_t` (= ``batch_sizes[t]``) is the count of sequences
    with length :math:`> t`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    >>> # padded (B=3, T=4, F=5), sorted-by-length [4, 3, 2]
    >>> x_padded = lucid.randn(3, 4, 5)
    >>> packed = pack_padded_sequence(x_padded, lengths=[4, 3, 2], batch_first=True)
    >>> packed.data.shape                   # 4 + 3 + 2 real time-steps
    (9, 5)
    >>> packed.batch_sizes.tolist()         # sequences alive at each step
    [3, 3, 2, 1]
    >>> # ... feed to RNN ...
    >>> unpacked, lengths = pad_packed_sequence(packed, batch_first=True)
    >>> unpacked.shape, lengths.tolist()
    ((3, 4, 5), [4, 3, 2])
    """

    data: Tensor
    batch_sizes: Tensor
    sorted_indices: Tensor | None
    unsorted_indices: Tensor | None


def pack_padded_sequence(
    input: Tensor,
    lengths: Tensor | list[int],
    batch_first: bool = False,
    enforce_sorted: bool = True,
) -> PackedSequence:
    r"""Pack a padded :math:`(T, B, *)` batch into a :class:`PackedSequence`.

    Strips out the padding cells so downstream RNN kernels iterate only
    over genuine time-steps.  Reduces both compute and (with masked
    losses) accidental gradient flow through pad positions.

    Parameters
    ----------
    input : Tensor
        Padded batch.  Default layout is ``(T, B, *)`` with time on
        axis 0; set ``batch_first=True`` for ``(B, T, *)``.
    lengths : Tensor or list of int
        Per-sequence true lengths, 1-D with one entry per batch element,
        each in ``[1, T]``.  A tensor may live on any device;
        non-integer values truncate toward zero.
    batch_first : bool, optional
        Whether ``input`` is laid out batch-first.  Default ``False``.
    enforce_sorted : bool, optional
        If ``True`` (the default), assume the caller already supplied
        sequences in descending-length order — cheap but raises
        :class:`ValueError` on violation, and the result carries no
        permutation (``sorted_indices`` / ``unsorted_indices`` are
        ``None``).  Set to ``False`` to have the function sort
        internally; the sort permutation is stored on the returned
        :class:`PackedSequence` so :func:`pad_packed_sequence` can undo
        it.

    Returns
    -------
    PackedSequence
        Packed view of the batch.  ``data`` and the index tensors live on
        ``input``'s device; ``batch_sizes`` is CPU ``int64``.

    Raises
    ------
    ValueError
        If ``input`` has fewer than two axes or no elements; if
        ``lengths`` is not 1-D, does not have one entry per batch
        element, or has an entry ``<= 0`` or longer than ``T``; or with
        ``enforce_sorted=True`` and an unsorted ``lengths``.

    Notes
    -----
    Total packed length equals :math:`\sum_b \ell_b`, the sum of true
    sequence lengths — strictly less than :math:`T \cdot B` whenever any
    sequence is shorter than the max.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.utils.rnn import pack_padded_sequence
    >>> x = lucid.randn(5, 3, 8)                 # (T=5, B=3, F=8)
    >>> packed = pack_padded_sequence(x, lengths=[5, 3, 2], batch_first=False)
    >>> packed.data.shape                        # 5 + 3 + 2 real time-steps
    (10, 8)
    """
    if input.ndim < 2:
        raise ValueError(
            f"pack_padded_sequence: input needs a time and a batch axis, got "
            f"shape {tuple(input.shape)}"
        )
    if input.numel() == 0:
        raise ValueError(
            f"pack_padded_sequence: cannot pack an empty tensor (shape " f"{tuple(input.shape)})"
        )
    if batch_first:
        input = input.permute(1, 0, *range(2, input.ndim))

    T, B = int(input.shape[0]), int(input.shape[1])
    lengths_list = _length_list(lengths)
    _check_lengths(lengths_list, B, T, enforce_sorted)

    sorted_t: Tensor | None = None
    unsorted_t: Tensor | None = None
    if not enforce_sorted:
        order = sorted(range(B), key=lambda i: -lengths_list[i])
        inverse = [0] * B
        for new_pos, orig_pos in enumerate(order):
            inverse[orig_pos] = new_pos
        sorted_t = _index(order, input)
        unsorted_t = _index(inverse, input)
        input = input.index_select(1, sorted_t)
        lengths_list = [lengths_list[i] for i in order]

    # Sorted descending, so the sequences alive at step t are a prefix.
    batch_sizes_list = [sum(1 for n in lengths_list if n > t) for t in range(lengths_list[0])]
    data = lucid.cat([input[t, :bs] for t, bs in enumerate(batch_sizes_list)], 0)
    return PackedSequence(data, _cpu_int64(batch_sizes_list), sorted_t, unsorted_t)


def pad_packed_sequence(
    sequence: PackedSequence,
    batch_first: bool = False,
    padding_value: float = 0.0,
    total_length: int | None = None,
) -> tuple[Tensor, Tensor]:
    r"""Inverse of :func:`pack_padded_sequence` — produce a padded tensor.

    Re-inflates a :class:`PackedSequence` into a dense
    :math:`(T_\text{max}, B, *)` (or :math:`(B, T_\text{max}, *)`)
    tensor with the original batch order restored.  The freshly created
    cells beyond each sequence's true length are filled with
    ``padding_value``.

    Parameters
    ----------
    sequence : PackedSequence
        Packed batch to unpack.
    batch_first : bool, optional
        Output layout.  ``False`` (default) yields ``(T, B, *)``,
        ``True`` yields ``(B, T, *)``.
    padding_value : float, optional
        Fill value for padded entries.  Default ``0.0``.
    total_length : int, optional
        If given, pad up to this length instead of the longest packed
        sequence.  Useful when downstream code expects a fixed
        ``T_max`` (e.g. fully-static export graphs).  Must be ``>=`` the
        longest sequence.

    Returns
    -------
    Tensor
        Padded batch tensor, on the packed data's device.
    Tensor
        1-D CPU ``int64`` tensor of per-sequence true lengths in the
        *original* batch order.

    Raises
    ------
    ValueError
        If ``total_length`` is shorter than the longest sequence, if
        ``batch_sizes`` is not a non-empty non-increasing run of positive
        counts summing to the packed row count, or if
        ``unsorted_indices`` is not a permutation of the batch.

    Notes
    -----
    Output time dimension is :math:`\max_b \ell_b` (or ``total_length``
    when given).  The function honours ``unsorted_indices`` on the
    packed sequence so the rows of the returned tensor match the order
    the caller used at pack time.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    >>> x = lucid.randn(3, 4, 5)                 # (B=3, T=4, F=5)
    >>> packed = pack_padded_sequence(x, lengths=[4, 3, 2], batch_first=True)
    >>> padded, lens = pad_packed_sequence(packed, batch_first=True)
    >>> padded.shape, lens.tolist()
    ((3, 4, 5), [4, 3, 2])
    """
    data = sequence.data
    batch_sizes_list = _batch_size_list(sequence.batch_sizes, int(data.shape[0]))
    longest = len(batch_sizes_list)
    if total_length is not None and total_length < longest:
        raise ValueError(
            f"Expected total_length to be at least the length of the longest "
            f"sequence in input, but got total_length={total_length} and max "
            f"sequence length being {longest}"
        )
    T = longest if total_length is None else total_length
    B = batch_sizes_list[0]

    out_impl = _C_engine.full(
        [T, B] + list(data.shape[1:]), padding_value, data._impl.dtype, data._impl.device
    )
    out_t = _Tensor.__new_from_impl__(out_impl)

    offset = 0
    for t, bs in enumerate(batch_sizes_list):
        # ``out_t[t, :bs]`` materialises a NEW tensor, so ``.copy_()`` on it
        # writes into that temporary and never reaches ``out_t``; assign
        # through __setitem__, which writes back.
        out_t[t, :bs] = data[offset : offset + bs]
        offset += bs

    # Packed row k is alive for every step whose active set exceeds k.
    lengths_list = [sum(1 for bs in batch_sizes_list if bs > k) for k in range(B)]

    if sequence.unsorted_indices is not None:
        order = _int_tensor_to_list(sequence.unsorted_indices)
        if sorted(order) != list(range(B)):
            raise ValueError(
                f"pad_packed_sequence: unsorted_indices must be a permutation "
                f"of range({B}), got {order}"
            )
        out_t = out_t.index_select(1, _index(order, out_t))
        lengths_list = [lengths_list[i] for i in order]

    if batch_first:
        out_t = out_t.permute(1, 0, *range(2, out_t.ndim))
    return out_t, _cpu_int64(lengths_list)


def pad_sequence(
    sequences: list[Tensor], batch_first: bool = False, padding_value: float = 0.0
) -> Tensor:
    r"""Stack a list of variable-length tensors into a padded batch.

    Each input is padded along its leading time dimension up to the
    longest sequence in ``sequences``, then the padded versions are
    stacked along a new batch axis.  Useful as the first step of an
    RNN training loop when sequences are produced one at a time (per-
    example tokenisation, audio framing, etc.).

    Parameters
    ----------
    sequences : list of Tensor
        Non-empty list of per-example tensors.  Each must share the
        same trailing feature shape and dtype / device; only the
        leading length axis may differ.
    batch_first : bool, optional
        Output layout.  ``False`` (default) yields ``(T_max, B, *feat)``,
        ``True`` yields ``(B, T_max, *feat)``.
    padding_value : float, optional
        Fill value for padded entries.  Default ``0.0``.

    Returns
    -------
    Tensor
        Padded batch tensor.

    Raises
    ------
    ValueError
        If ``sequences`` is empty or holds a 0-d tensor.
    ShapeMismatch
        If two sequences differ in a trailing (non-time) axis.  Also a
        :class:`ValueError`.
    DeviceMismatch
        If the sequences do not share one device.  Also a
        :class:`ValueError`.

    Notes
    -----
    Padding is constructed by concatenating a freshly allocated tail
    of shape ``(T_max - T_i, *feat)`` onto each input — this keeps the
    operation differentiable end-to-end (view-based mutation would
    silently drop gradients through Lucid's autograd-aware
    :class:`~lucid._tensor.tensor.Tensor`).

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.utils.rnn import pad_sequence
    >>> a, b, c = lucid.ones(5, 3), lucid.ones(3, 3), lucid.ones(2, 3)
    >>> batch = pad_sequence([a, b, c], batch_first=True)
    >>> batch.shape                              # (B, T_max, F)
    (3, 5, 3)
    """
    _check_sequences(sequences)
    T_max: int = max(int(s.shape[0]) for s in sequences)
    feat_shape: list[int] = list(sequences[0].shape[1:])
    device = sequences[0]._impl.device

    padded_each: list[Tensor] = []
    for s in sequences:
        T_i: int = int(s.shape[0])
        if T_i == T_max:
            padded_each.append(s)
            continue
        # Build a constant-fill tail and concat.
        tail_shape: list[int] = [T_max - T_i] + feat_shape
        tail_impl = _C_engine.full(tail_shape, padding_value, s._impl.dtype, device)
        tail: Tensor = _Tensor.__new_from_impl__(tail_impl)
        padded_each.append(lucid.cat([s, tail], 0))

    # Stack along axis 1 (T_max, B, *feat) or axis 0 with batch_first.
    if batch_first:
        return lucid.stack(padded_each, 0)
    return lucid.stack(padded_each, 1)


def pack_sequence(sequences: list[Tensor], enforce_sorted: bool = True) -> PackedSequence:
    r"""One-shot pack-from-list — equivalent to :func:`pad_sequence` then :func:`pack_padded_sequence`.

    Convenience wrapper for the common case where a list of per-example
    tensors needs to become a :class:`PackedSequence` without going
    through an intermediate padded tensor.  Each entry's leading axis
    is treated as the time dimension; remaining axes carry the feature
    shape (which must be identical across the list).

    Parameters
    ----------
    sequences : list of Tensor
        Per-example tensors.  Leading axis is time; trailing axes are
        the feature shape and must match across entries.
    enforce_sorted : bool, optional
        If ``True`` (the default), the caller must supply sequences in
        non-increasing length order; mismatches raise
        :class:`ValueError`.  With ``False`` the batch is sorted for you
        and the permutation is recorded on the result, so
        :func:`pad_packed_sequence` returns the batch in the order you
        passed it.

    Returns
    -------
    PackedSequence
        Packed view of the input list.

    Raises
    ------
    ValueError
        If ``sequences`` is empty, holds a 0-d or zero-length tensor, or
        is not sorted while ``enforce_sorted=True``.  Trailing-shape and
        device mismatches raise as in :func:`pad_sequence`.

    Notes
    -----
    Implemented as :func:`pad_sequence` followed by
    :func:`pack_padded_sequence`.  The intermediate padded tensor lives
    only for the duration of the call.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn.utils.rnn import pack_sequence
    >>> long_seq = lucid.ones(4, 2)
    >>> mid_seq = lucid.ones(3, 2)
    >>> short_seq = lucid.ones(1, 2)
    >>> packed = pack_sequence([long_seq, mid_seq, short_seq])
    >>> packed.batch_sizes.tolist()              # sequences alive at each step
    [3, 2, 2, 1]
    """
    padded: Tensor = pad_sequence(sequences, batch_first=False)
    lengths: list[int] = [int(s.shape[0]) for s in sequences]
    return pack_padded_sequence(padded, lengths, batch_first=False, enforce_sorted=enforce_sorted)
