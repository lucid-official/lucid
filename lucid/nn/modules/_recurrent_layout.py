"""The input layouts every recurrent forward takes apart and puts back.

RNN / GRU / LSTM accept a ``(T, B, *)``, ``(B, T, *)`` or unbatched
``(T, *)`` tensor, or a packed sequence; the cells a ``(B, *)`` or
unbatched ``(*,)`` tensor.  The recurrences in :mod:`lucid.nn.modules.rnn`
run on one layout only — ``(T, B, *)`` in descending-length batch order,
``(B, *)`` for a cell step — so each forward enters through
:func:`_sequence_major` (cells: :func:`_cell_batch`) and leaves through
:func:`_restore_layout` (cells: :func:`_cell_unbatch`).  The initial
states are checked against the shape the input implies on the way in.

Between stacked layers every module applies its ``dropout`` through
:func:`_inter_layer_dropout`, after checking it once with
:func:`_check_dropout` when it is built.
"""

import warnings
from collections.abc import Sequence
from typing import NamedTuple

import lucid as _lucid
from lucid import _promotion, stack
from lucid._dispatch import _unwrap, _wrap
from lucid._factories.creation import zeros
from lucid._ops._autocast import _autocast_entry
from lucid._tensor.tensor import Tensor
from lucid.nn.functional.activations import sigmoid, tanh
from lucid.nn.functional.dropout import dropout as _dropout
from lucid.nn.functional.linear import linear
from lucid.nn.utils.rnn import PackedSequence, _int_tensor_to_list

# The engine's ``lstm`` entry refuses an empty sequence in the same words.
_EMPTY_SEQUENCE = "expected sequence length to be larger than 0"


class _Layout(NamedTuple):
    """A recurrent input in run order, and how to give the outputs back.

    Attributes:
        seq: The input as ``(T, B, *)``.
        batch_sizes: The active batch size at each step, non-increasing.
        packed: The packed source, ``None`` for a dense input.
        unbatched: The input was ``(T, *)``; the batch axis of 1 the
            recurrence ran with is not part of any output.
        batch_first: The output goes back batch-major.
    """

    seq: Tensor
    batch_sizes: list[int]
    packed: PackedSequence | None
    unbatched: bool
    batch_first: bool


def _sequence_major(
    kind: str,
    x: Tensor | PackedSequence,
    batch_first: bool,
    states: Sequence[tuple[str, Tensor | None, int]],
    rows: int,
) -> tuple[_Layout, list[Tensor]]:
    """The input in run order and every initial state as ``(rows, B, width)``.

    Every recurrent forward enters here and leaves through
    :func:`_restore_layout`, so the input layouts — packed, batch-first,
    unbatched — are taken apart in one place and put back in one place.
    An unbatched ``(T, *)`` input runs as a batch of one, whatever
    ``batch_first`` says, and its states are then ``(rows, width)``.  A
    missing state is zeros.

    Args:
        kind: The module's name, for the error messages.
        x: The input — a ``(T, B, *)`` / ``(B, T, *)`` / ``(T, *)`` tensor
            or a packed sequence.
        batch_first: Whether a batched dense input is ``(B, T, *)``.
        states: ``(name, tensor or None, width)`` for each initial state.
        rows: ``num_layers * num_directions``.

    Raises:
        ValueError: The input is neither 2-D nor 3-D, has no timesteps, or
            an initial state does not have the shape the input implies.
    """
    layout = _input_layout(kind, x, batch_first)
    return layout, [
        _initial_state(kind, layout, name, hx, (rows, width))
        for name, hx, width in states
    ]


def _input_layout(kind: str, x: Tensor | PackedSequence, batch_first: bool) -> _Layout:
    """The input half of :func:`_sequence_major`.

    A sequence with no timesteps is refused as the reference refuses it:
    the recurrences build their output by stacking per-step results, and
    there are none to stack.
    """
    if isinstance(x, PackedSequence):
        if int(x.batch_sizes.shape[0]) == 0:
            raise ValueError(f"{_EMPTY_SEQUENCE}, got a packed sequence with no steps")
        seq, batch_sizes = _packed_to_padded(x)
        return _Layout(seq, batch_sizes, x, False, batch_first)
    if x.ndim not in (2, 3):
        raise ValueError(
            f"{kind}: expected input to be 2-D (unbatched) or 3-D (batched), "
            f"got {x.ndim}-D input of shape {tuple(x.shape)}"
        )
    unbatched = x.ndim == 2
    time_axis = 1 if batch_first and not unbatched else 0
    if int(x.shape[time_axis]) == 0:
        raise ValueError(f"{_EMPTY_SEQUENCE}, got input of shape {tuple(x.shape)}")
    if unbatched:
        seq = x.unsqueeze(1)
    elif batch_first:
        seq = x.permute([1, 0, 2])
    else:
        seq = x
    batch_sizes = [int(seq.shape[1])] * int(seq.shape[0])
    return _Layout(seq, batch_sizes, None, unbatched, batch_first)


def _initial_state(
    kind: str, layout: _Layout, name: str, hx: Tensor | None, size: tuple[int, int]
) -> Tensor:
    """One initial state of :func:`_sequence_major` as ``(rows, B, width)``
    in run order — ``size`` is ``(rows, width)``."""
    rows, width = size
    batch = int(layout.seq.shape[1])
    if hx is None:
        like = layout.seq
        return zeros(rows, batch, width, device=like.device, dtype=like.dtype)
    expected = size if layout.unbatched else (rows, batch, width)
    if tuple(hx.shape) != expected:
        source = "an unbatched 2-D" if layout.unbatched else "a batched"
        raise ValueError(
            f"{kind}: expected {name} of shape {expected} for {source} input, "
            f"got {tuple(hx.shape)}"
        )
    if layout.unbatched:
        return hx.unsqueeze(1)
    if layout.packed is not None:
        return _reorder_batch(hx, layout.packed, to_packed=True)
    return hx


def _restore_layout(
    layout: _Layout, out: Tensor, states: list[Tensor]
) -> tuple[Tensor | PackedSequence, list[Tensor]]:
    """Give a ``(T, B, *)`` output and ``(rows, B, *)`` final states back in
    the layout the input came in — the inverse of :func:`_sequence_major`."""
    if layout.packed is not None:
        return (
            _padded_to_packed(out, layout.batch_sizes, layout.packed),
            [_reorder_batch(s, layout.packed, to_packed=False) for s in states],
        )
    if layout.unbatched:
        return out.squeeze(1), [s.squeeze(1) for s in states]
    if layout.batch_first:
        return out.permute([1, 0, 2]), states
    return out, states


def _packed_to_padded(packed: PackedSequence) -> tuple[Tensor, list[int]]:
    """Inflate a packed sequence to ``(T, B, *)`` in *packed* batch order.

    Deliberately not :func:`~lucid.nn.utils.rnn.pad_packed_sequence`: that
    restores the caller's original batch order, and the recurrence needs
    the descending-length order, because that is what makes the active set
    at each timestep a prefix of the batch.

    Args:
        packed: The packed sequence, with at least one step.

    Returns:
        ``(padded, batch_sizes)`` — padded is ``(T, B, *)`` with zeros past
        each sequence's length, ``batch_sizes[t]`` is how many sequences
        are still running at step ``t``.
    """
    batch_sizes = _int_tensor_to_list(packed.batch_sizes)
    T = len(batch_sizes)
    B = batch_sizes[0]
    feat = tuple(int(v) for v in packed.data.shape[1:])

    rows: list[Tensor] = []
    offset = 0
    for bs in batch_sizes:
        step = packed.data[offset : offset + bs]
        if bs < B:
            pad = zeros(
                B - bs,
                *feat,
                device=packed.data.device,
                dtype=packed.data.dtype,
            )
            step = _lucid.cat([step, pad], 0)
        rows.append(step)
        offset += bs

    assert len(rows) == T
    return stack(rows, 0), batch_sizes


def _padded_to_packed(
    padded: Tensor, batch_sizes: list[int], src: PackedSequence
) -> PackedSequence:
    """Repack ``(T, B, *)`` back into a packed sequence.

    Reuses the input's ``sorted_indices`` / ``unsorted_indices`` verbatim,
    so unpacking the result returns the batch in the caller's own order.
    """
    rows = [padded[t, :bs] for t, bs in enumerate(batch_sizes)]
    return PackedSequence(
        _lucid.cat(rows, 0),
        src.batch_sizes,
        src.sorted_indices,
        src.unsorted_indices,
    )


def _packed_segments(batch_sizes: list[int]) -> list[tuple[int, int, int]]:
    """Maximal runs of constant batch size: ``[(t_start, t_end, bs), …]``.

    The engine takes a whole ``(T, B, *)`` block per call, so the packed
    path runs one call per run of timesteps that share an active-set size
    rather than one per timestep.
    """
    segments: list[tuple[int, int, int]] = []
    start = 0
    for t in range(1, len(batch_sizes) + 1):
        if t == len(batch_sizes) or batch_sizes[t] != batch_sizes[start]:
            segments.append((start, t, batch_sizes[start]))
            start = t
    return segments


def _reorder_batch(state: Tensor, src: PackedSequence, *, to_packed: bool) -> Tensor:
    """Permute the batch axis of a ``(N, B, H)`` state across a packed input.

    The recurrence runs the batch sorted by descending length.
    ``to_packed=True`` takes a caller's initial state into that order — row
    ``i`` becomes the caller's row ``sorted_indices[i]``.  ``to_packed=False``
    takes a final state back to the caller's order by ``unsorted_indices``.
    """
    indices = src.sorted_indices if to_packed else src.unsorted_indices
    if indices is None:
        return state
    order = _int_tensor_to_list(indices)
    if order == list(range(len(order))):
        return state
    return stack([state[:, i] for i in order], 1)


def _cell_batch(
    kind: str, x: Tensor, states: Sequence[tuple[str, Tensor | None]], width: int
) -> tuple[bool, Tensor, list[Tensor]]:
    """A cell's input and states as ``(B, *)``, and whether the input was a
    single unbatched ``(input_size,)`` vector.

    Every cell forward enters here; a ``True`` flag means the cell squeezes
    the batch axis of 1 back off what it returns (:func:`_cell_unbatch`).
    A missing state is zeros.  The multi-step modules step a cell once per
    timestep, so this checks shapes only — it adds no op to a batched call.

    Args:
        kind: The cell's name, for the error messages.
        x: ``(B, input_size)`` or ``(input_size,)``.
        states: ``(name, tensor or None)`` for each state.
        width: The state width, ``hidden_size``.

    Raises:
        ValueError: The input is neither 1-D nor 2-D, or a state does not
            have the shape the input implies.
    """
    if x.ndim not in (1, 2):
        raise ValueError(
            f"{kind}: expected input to be 1-D (unbatched) or 2-D (batched), "
            f"got {x.ndim}-D input of shape {tuple(x.shape)}"
        )
    unbatched = x.ndim == 1
    batch = 1 if unbatched else int(x.shape[0])
    expected = (width,) if unbatched else (batch, width)
    out: list[Tensor] = []
    for name, hx in states:
        if hx is None:
            out.append(zeros(batch, width, device=x.device, dtype=x.dtype))
            continue
        if tuple(hx.shape) != expected:
            source = "an unbatched 1-D" if unbatched else "a batched"
            raise ValueError(
                f"{kind}: expected {name} of shape {expected} for {source} input, "
                f"got {tuple(hx.shape)}"
            )
        out.append(hx.unsqueeze(0) if unbatched else hx)
    return unbatched, (x.unsqueeze(0) if unbatched else x), out


def _cell_unbatch(unbatched: bool, state: Tensor) -> Tensor:
    """A cell's ``(B, H)`` result in the input's layout — :func:`_cell_batch`
    reversed."""
    return state.squeeze(0) if unbatched else state


def _lstm_cell_gates(
    x: Tensor,
    h0: Tensor,
    c0: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor | None,
    b_hh: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Compute the four LSTM gates and next ``(h1, c1)`` states."""
    gates = linear(x, w_ih, b_ih) + linear(h0, w_hh, b_hh)
    hs = int(w_hh.shape[1])
    i_gate = sigmoid(gates[:, :hs])
    f_gate = sigmoid(gates[:, hs : 2 * hs])
    g_gate = tanh(gates[:, 2 * hs : 3 * hs])
    o_gate = sigmoid(gates[:, 3 * hs :])
    c1 = f_gate * c0 + i_gate * g_gate
    return o_gate * tanh(c1), c1


def _lstm_cell_step(
    x: Tensor,
    state: tuple[Tensor, Tensor],
    weights: tuple[Tensor, Tensor, Tensor | None, Tensor | None],
) -> tuple[Tensor, Tensor]:
    """One ``LSTMCell`` step on ``(B, I)`` and ``(B, H)`` states: ``(h1, c1)``.

    The ``lstm_cell`` row (LOWER on Metal; the CPU decomposes into ``linear``
    and the gate elementwise ops, like ``GRUCell``) applies at the entry to
    every operand at once on Metal — the float32 weights and zero state with
    the input — so the gates compute in one dtype, as the reference's
    ``lstm_cell`` does; applied gate by gate, the float32 state widened the
    result again.
    """
    h0, c0 = state
    w_ih, w_hh, b_ih, b_hh = weights
    if x.device.type == "metal":
        biases = [b for b in (b_ih, b_hh) if b is not None]
        operands = (x, h0, c0, w_ih, w_hh, *biases)
        with _autocast_entry("lstm_cell", *(_unwrap(t) for t in operands)) as cast:
            x, h0, c0, w_ih, w_hh, *b = (_wrap(t) for t in cast)
            b_ih, b_hh = b or (None, None)
            return _lstm_cell_gates(x, h0, c0, w_ih, w_hh, b_ih, b_hh)
    return _lstm_cell_gates(x, h0, c0, w_ih, w_hh, b_ih, b_hh)


def _check_dropout(dropout: object, num_layers: int) -> float:
    """The inter-layer ``dropout`` of RNN / GRU / LSTM, checked once.

    Dropout runs only *between* stacked layers, so a single-layer module
    has nowhere to apply it — the reference warns rather than refuses,
    and so does this.

    Raises:
        ValueError: ``dropout`` is not a number in ``[0, 1]``.
    """
    d = _promotion.as_number(dropout)
    if isinstance(d, bool) or not isinstance(d, (int, float)) or not 0.0 <= d <= 1.0:
        raise ValueError(
            "dropout should be a number in range [0, 1] representing the "
            f"probability of an element being zeroed, got {dropout!r}"
        )
    if d > 0 and num_layers == 1:
        warnings.warn(
            "dropout option adds dropout after all but last recurrent layer, "
            "so non-zero dropout expects num_layers greater than 1, but got "
            f"dropout={dropout} and num_layers={num_layers}",
            UserWarning,
            stacklevel=3,
        )
    return float(d)


def _inter_layer_dropout(
    x: Tensor, p: float, layer: int, num_layers: int, training: bool
) -> Tensor:
    """Dropout on the output of ``layer`` before it feeds the next one.

    The one place RNN, GRU and LSTM apply their ``dropout``: in training
    only, and never after the last layer, whose output is the module's.
    """
    if p == 0.0 or not training or layer >= num_layers - 1:
        return x
    return _dropout(x, p, training=True)
