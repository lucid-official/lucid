"""
Index ranges: the one place that decides whether an integer index lies
inside the axis it indexes.

Every ``nn.functional`` op that reads with an integer index is checked
here, before the gather or kernel that would read with it. That covers a
class target of a class-index loss, a ``one_hot`` class, an embedding row,
``max_unpool{1,2,3}d``'s flat spatial indices, and ``ctc_loss``'s labels,
``blank`` and sequence lengths. Each used to have a policy of its own:

- the class-index losses checked after their gather;
- the embedding table was read on the host on every device;
- ``max_unpool{1,2,3}d`` let ``scatter_add`` wrap negative indices into the
  output and caught positive ones only inside the CPU kernel;
- ``one_hot`` and ``ctc_loss`` did not check at all.

As a result, ``one_hot`` quietly gave a zero row for a class past the end,
and ``ctc_loss`` with an ``input_lengths`` past the input read beyond it
and returned a number.

Policy
------
**CPU.** An index outside its axis raises ``IndexError`` before any gather
or kernel runs.

- The index is on the host already, so reading it waits on no device.
- The read is the engine's own comparison and reduction over the index
  (:func:`_refuse`). A Python scan of the values costs about 25 ns an
  entry, which at a segmentation target's millions of entries takes
  longer than the loss itself. A refusal therefore dispatches the guard's
  reads (the clamp, the comparison, ``all``, and any mask they need) and
  none of the consumer's kernels.
- The guard is not a value: it raises or passes, and nothing downstream
  depends on what it read. So it runs outside any active compile trace,
  and a compiled step is validated when it is traced.

**Metal.** The index is never read back, because a host read in every
call would stall every training step. An out-of-range index is isolated
inside the graph instead, and never reaches memory outside its buffer:

- The class-index losses clamp before every gather and make the sample's
  factor NaN (:func:`_poison`). A bad label then makes the loss NaN,
  rather than being scored as class ``0`` or ``C - 1``.
- The engine's index kernels clamp and mask the rest: a float gather
  (``embedding`` and ``embedding_bag`` included) reads NaN, an integer
  gather reads 0, a scatter drops the update, and ``one_hot`` gives a
  zero row.

Two consumers read on Metal anyway, each for its own reason:

- ``ctc_loss`` runs its recursions on the CPU, so its labels and lengths
  make the round trip regardless.
- ``embedding_bag``'s 1-D ``offsets`` (:func:`_check_offsets`) are bag
  boundaries (starting at 0 and never decreasing) that its backward walks
  on the CPU.

``ctc_loss`` is stricter than the reference in one place: the reference
accepts a label outside ``[0, C)`` and reads past the class axis. The loss
it gives is meaningless, so such a label is refused here.
"""

from collections.abc import Callable
from itertools import pairwise
from typing import TYPE_CHECKING, NamedTuple, cast, overload

import lucid as _lucid
from lucid._dtype import dtype as _dtype
from lucid.nn.functional._spatial_args import _spatial_extent

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _host_read[T](read: Callable[[], T]) -> T:
    """Run ``read``, a guard's read of an index, outside any active compile
    trace.

    A read inside the trace would mark it unsupported (a traced value read
    on the host) and send every step that holds a guard to eager.
    """
    # Imported here: ``lucid.compile`` imports ``lucid.nn``.
    from lucid.compile import _host_check

    with _host_check():
        return read()


def _as_index(index: Tensor) -> Tensor:
    """``index`` as an integer tensor.

    A float or bool index is truncated to ``int64``, as its consumer's own
    cast would do, and is checked in that form. An integer index is
    returned untouched, so it is checked at its own width: an ``int64``
    value past ``int32``'s range is refused rather than wrapped into the
    range by a later cast (the one exception is a Metal class target; see
    :func:`_class_targets`).
    """
    if index.dtype in (_lucid.int8, _lucid.int16, _lucid.int32, _lucid.int64):
        return index
    return index.to(dtype=_lucid.int64)


def _in_range(index: Tensor, extent: int, counted: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """``index`` clamped into ``[0, extent)``, and where that changed nothing.

    Returns ``(clamped, usable)``. ``clamped`` is what a gather can read
    with safely. ``usable`` is ``False`` exactly at an entry outside the
    axis, unless ``counted`` is ``False`` there: an ``ignore_index``
    sentinel and the padding after a label list are allowed to be anything.
    """
    if extent <= 0:
        # No entry fits an empty axis, and clip would be handed an upper
        # bound below its lower one.
        clamped = _lucid.zeros_like(index)
        usable = _lucid.zeros_like(index, dtype=_lucid.bool_)
    else:
        clamped = _lucid.clip(index, 0, extent - 1)
        usable = clamped == index
    if counted is not None:
        usable = usable | ~counted
    return clamped, usable


def _guard(index: Tensor, extent: int, counted: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """``(index, usable)`` for a guard whose clamp nothing reads: ``index``
    as an integer tensor, and where it lies inside ``[0, extent)``.

    Run it through :func:`_host_read`, so that a compile trace or an
    export records none of it.
    """
    index = _as_index(index)
    return index, _in_range(index, extent, counted)[1]


def _flat(values: object) -> list[object]:
    """``Tensor.tolist()``'s nested lists, or its scalar, as one flat list."""
    out: list[object] = []
    stack: list[object] = [values]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            out.append(item)
    return out


def _refuse(index: Tensor, usable: Tensor, message: Callable[[object], str]) -> None:
    """The guard's one read: raise ``IndexError(message(v))`` for the first
    entry ``v`` of ``index`` that is not ``usable``, or return.

    Every range check of an index tensor ends here; only ``ctc_loss``'s
    lengths, which are host metadata, are read elsewhere
    (:func:`_check_lengths`). The pass is one ``all`` over the mask the
    clamp already needed. The failure path finds the first offending entry
    with ``tolist``, which dispatches no op.
    """
    if usable._impl.is_phantom or _host_read(lambda: bool(usable.all().item())):
        return
    first = _host_read(
        lambda: next(
            value
            for value, ok in zip(_flat(index.tolist()), _flat(usable.tolist()), strict=True)
            if not ok
        )
    )
    raise IndexError(message(first))


# ── the class-index losses ──────────────────────────────────────────────────


def _class_index(target: Tensor) -> Tensor:
    """``target`` at the width :func:`_class_targets` checks it at: its own
    integer width on the CPU, ``int32`` on Metal (see there).

    A loss that derives a mask from its targets before the check (the
    labels before a list's first negative entry) builds it from this, so the
    mask and the check read the same values.
    """
    index = _as_index(target)
    return index if target.device == "cpu" else index.to(dtype=_lucid.int32)


class _ClassTargets(NamedTuple):
    """The class targets of a class-index loss, checked
    (:func:`_class_targets`)."""

    #: The targets clamped into the class range, as ``int32``: what every
    #: gather reads with.
    safe: Tensor
    #: ``False`` at the entries the loss does not read as a class, or
    #: ``None`` when it reads every entry.
    counted: Tensor | None
    #: Metal only: ``False`` at each out-of-range counted target, for
    #: :func:`_poison`.  ``None`` on the CPU, where such a target raised.
    usable: Tensor | None


def _class_targets(
    target: Tensor,
    num_classes: int,
    op: str,
    *,
    ignore_index: int | None = None,
    counted: Tensor | None = None,
) -> _ClassTargets:
    """Check the class targets of a class-index loss, before its gather.

    The guard of ``cross_entropy``, ``nll_loss``, ``multi_margin_loss`` and
    ``multilabel_margin_loss``.  The entries a loss does not read as a
    class may hold anything: the ``ignore_index`` positions (pass
    ``ignore_index``), or the padding after a label list (pass the
    ``counted`` mask).  Any other target outside ``[0, num_classes)``
    raises ``IndexError("<op>: Target N is out of bounds.")`` for a CPU
    ``target``, and is flagged in ``usable`` for a Metal one.

    A CPU target is checked at its own width, so an ``int64`` target past
    ``int32``'s range is refused rather than wrapped into the range.  A
    Metal target is checked and gathered as ``int32``, the width these
    losses always used there, and an ``int64`` target that wide wraps.  It
    must not be compared at its own width: an ``int64`` comparison aborts
    the graph compiler when the ``int64`` came from an ``argmax`` inside a
    compiled step, because the graph holds that result as ``int32``
    (LCD-270).
    """
    index = _class_index(target)
    on_cpu = target.device == "cpu"
    if ignore_index is not None:
        counted = index != ignore_index
    clamped, usable = _in_range(index, num_classes, counted)
    if on_cpu:
        _refuse(index, usable, lambda v: f"{op}: Target {v} is out of bounds.")
        return _ClassTargets(clamped.to(dtype=_lucid.int32), counted, None)
    return _ClassTargets(clamped, counted, usable)


@overload
def _poison(scale: Tensor, usable: Tensor | None, like: Tensor, dtype: _dtype) -> Tensor: ...
@overload
def _poison(scale: None, usable: Tensor | None, like: Tensor, dtype: _dtype) -> Tensor | None: ...
def _poison(
    scale: Tensor | None,
    usable: Tensor | None,
    like: Tensor,
    dtype: _dtype,
) -> Tensor | None:
    """The per-sample factor ``scale``, NaN at each out-of-range target.

    ``usable`` comes from :func:`_class_targets`. ``None``, which is the CPU
    case where the guard has raised already, returns ``scale`` unchanged.
    ``scale`` of ``None`` means all ones, of the shape of ``like`` and the
    dtype ``dtype``.
    """
    if usable is None:
        return scale
    ones = scale if scale is not None else _lucid.ones_like(like, dtype=dtype)
    return _lucid.where(usable, ones, _lucid.full_like(ones, float("nan")))


# ── tables and class axes ───────────────────────────────────────────────────


def _check_indices(
    index: Tensor,
    extent: int,
    op: str,
    *,
    what: str,
    axis: str,
) -> None:
    """Refuse an entry of ``index`` outside ``[0, extent)`` with
    ``IndexError``, before the consumer reads with it.

    The message is ``"<op>: <what> <v> is out of range for <axis> (valid
    range [0, extent - 1])"``. A Metal ``index`` is isolated in the graph
    and not read back (see the module docstring). An empty ``index`` has
    nothing to check.
    """
    if index.numel() == 0 or index.device != "cpu":
        return
    index, usable = _host_read(lambda: _guard(index, extent))
    _refuse(
        index,
        usable,
        lambda v: (
            f"{op}: {what} {v} is out of range for {axis} " f"(valid range [0, {extent - 1}])"
        ),
    )


#: The integer dtypes an embedding table can be indexed with.
_TABLE_INDEX_DTYPES = (
    _lucid.int8,
    _lucid.int16,
    _lucid.int32,
    _lucid.int64,
    _lucid.bool_,
)


def _check_table(x: Tensor, weight: Tensor, op: str) -> None:
    """Check the indices ``x`` into the rows of an embedding table
    ``weight``, for ``embedding`` and ``embedding_bag``.

    A float ``x`` raises ``TypeError``. It would pass the range check
    trivially, since its values sit in ``[0, 1)``, and the engine gather
    would then read its bits as integers: ``F.embedding(float_tensor,
    weight)`` used to end in a segmentation fault. A CPU index outside
    ``[0, num_embeddings)`` raises ``IndexError``; a Metal one is isolated
    in the table gather (see the module docstring).
    """
    if x.dtype not in _TABLE_INDEX_DTYPES:
        raise TypeError(
            f"{op}: indices must be an integer tensor, got {x.dtype}; "
            f"cast with .to(lucid.int64) first"
        )
    rows = int(weight.shape[0])
    _check_indices(
        x,
        rows,
        op,
        what="index",
        axis=f"a table with {rows} entries",
    )


def _check_unpool(
    x: Tensor,
    indices: Tensor,
    output_spatial: tuple[int, ...],
    op: str,
) -> tuple[Tensor, int]:
    """Check the flat spatial ``indices`` and ``output_spatial`` extents of
    ``max_unpool{1,2,3}d``, before its ``scatter_add``.

    ``indices`` must match ``x`` in shape, hold integer values, and address
    positions in ``[0, prod(output_spatial))``.  ``scatter_add`` wraps a
    negative index ``[-extent, 0)`` into the axis on both devices, so a
    negative ``indices`` entry used to write into the tail of the output
    silently, and a positive one past the end only failed inside
    ``cpu_backend::scatter_add``.  Here a CPU entry outside
    ``[0, spatial_numel)`` raises ``IndexError`` before any buffer is
    allocated, and a Metal negative entry is mapped on device to
    ``spatial_numel`` so ``scatter_add`` drops it alongside entries
    ``>= spatial_numel`` without a host sync.
    """
    _spatial_extent(x, len(output_spatial), op)
    if x.shape != indices.shape:
        raise ValueError(
            f"{op}: expected input and indices shapes to match, got "
            f"{tuple(int(d) for d in x.shape)} vs {tuple(int(d) for d in indices.shape)}"
        )
    spatial_numel = 1
    for extent in output_spatial:
        if int(extent) <= 0:
            raise ValueError(
                f"{op}: output_size must be positive in every dimension, "
                f"got {tuple(int(v) for v in output_spatial)}"
            )
        spatial_numel *= int(extent)
    if indices.dtype not in (_lucid.int8, _lucid.int16, _lucid.int32, _lucid.int64):
        raise TypeError(
            f"{op}: indices must be an integer tensor, got {indices.dtype}; "
            f"cast with .to(lucid.int64) first"
        )
    _check_indices(
        indices,
        spatial_numel,
        op,
        what="index",
        axis=f"output_size {output_spatial} ({spatial_numel} elements)",
    )
    idx = (
        indices if indices.dtype in (_lucid.int32, _lucid.int64) else indices.to(dtype=_lucid.int64)
    )
    if idx.device != "cpu" and idx.numel() > 0:
        idx = _lucid.where(idx < 0, spatial_numel, idx)
    return idx, spatial_numel


def _padding_index(padding_idx: int | None, weight: Tensor, op: str) -> int | None:
    """``padding_idx`` as a row of the table ``weight``, or ``None``.

    It must lie in ``[-num_embeddings, num_embeddings)``, as for
    :class:`~lucid.nn.Embedding`, else ``ValueError``.  A negative one counts
    from the end and comes back as its row; the engine reads ``-1`` as
    "no padding row", so ``padding_idx=-1`` was taken for none.
    """
    if padding_idx is None:
        return None
    rows = int(weight.shape[0])
    if not -rows <= padding_idx < rows:
        raise ValueError(f"{op}: padding_idx must be within [-{rows}, {rows}); got {padding_idx}")
    return padding_idx + rows if padding_idx < 0 else padding_idx


def _check_offsets(offsets: Tensor, length: int, op: str) -> None:
    """The bag boundaries ``offsets`` into ``length`` indices, as
    ``embedding_bag`` reads them.

    ``offsets`` is 1-D, starts at 0, never decreases, and every entry lies
    in ``[0, length]``.  A boundary past the end, which the kernel read past
    the indices for, raises ``IndexError``.  A first boundary that is not 0,
    or one that goes backwards, raises ``ValueError``.  Both used to be
    scored silently.  ``offsets`` is read on the host on both devices (see
    the module docstring).
    """
    if offsets.ndim != 1:
        raise ValueError(
            f"{op}: offsets must be 1-D, got shape " f"{tuple(int(d) for d in offsets.shape)}"
        )
    if offsets._impl.is_phantom:
        return
    bounds = [int(cast(int, v)) for v in _flat(_host_read(offsets.tolist))]
    for v in bounds:
        if not 0 <= v <= length:
            raise IndexError(
                f"{op}: offset {v} is out of range for {length} indices "
                f"(valid range [0, {length}])"
            )
    if bounds and bounds[0] != 0:
        raise ValueError(f"{op}: offsets[0] must be 0, got {bounds[0]}")
    for before, after in pairwise(bounds):
        if after < before:
            raise ValueError(f"{op}: offsets must not decrease, got {after} after {before}")


# ── ctc_loss ────────────────────────────────────────────────────────────────


def _check_lengths(
    lengths: Tensor, name: str, batch: int, longest: int | None, op: str
) -> list[int]:
    """One length per sequence, each in ``[0, longest]`` (``longest`` of
    ``None`` bounds it below only). Returns the lengths as Python ints.

    A wrong count raises ``ValueError``, because the kernel reads one
    length per sequence. A length outside its range raises ``IndexError``,
    because the kernel would read that far along the axis.
    """
    shape = tuple(int(d) for d in lengths.shape)
    if shape != (batch,):
        raise ValueError(
            f"{op}: {name} must hold one length per sequence, shape " f"({batch},), got {shape}"
        )
    values = [int(cast(int, v)) for v in _flat(_host_read(lengths.tolist))]
    for v in values:
        if v < 0 or (longest is not None and v > longest):
            bound = "be non-negative" if longest is None else f"lie in [0, {longest}]"
            raise IndexError(f"{op}: each of {name} must {bound}, got {v}")
    return values


def _check_ctc(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
    blank: int,
    op: str,
) -> list[int]:
    """Check every index ``ctc_loss`` reads with, before its kernel.

    - ``log_probs`` must be ``(T, N, C)``.
    - ``blank`` must lie in ``[0, C)``.
    - ``input_lengths`` must hold one length per sequence, each in
      ``[0, T]``.
    - ``targets`` must be padded ``(N, S)`` with each ``target_lengths``
      entry in ``[0, S]``, or concatenated ``(sum(target_lengths),)``.
    - Every label the kernel reads must lie in ``[0, C)``. The padding past
      a sequence's length is not read, so it can hold anything.

    These are read on the host on both devices; the module docstring gives
    the reason. A wrong shape or count raises ``ValueError``. An index
    outside its axis raises ``IndexError``.

    Returns
    -------
    list of int
        ``target_lengths`` as Python ints, which the caller uses to
        gather a padded target into one list.
    """
    if log_probs.ndim != 3:
        raise ValueError(
            f"{op}: log_probs must be (T, N, C), or (T, C) for one sequence, "
            f"got shape {tuple(int(d) for d in log_probs.shape)}"
        )
    time, batch, classes = (int(d) for d in log_probs.shape)
    if not 0 <= blank < classes:
        raise IndexError(
            f"{op}: blank {blank} is out of range for {classes} classes "
            f"(valid range [0, {classes - 1}])"
        )
    if targets.ndim not in (1, 2):
        raise ValueError(
            f"{op}: targets must be padded (N, S) or concatenated (sum of "
            f"target_lengths,), got shape {tuple(int(d) for d in targets.shape)}"
        )
    padded = targets.ndim == 2
    if padded and int(targets.shape[0]) != batch:
        raise ValueError(
            f"{op}: padded targets must hold one row per sequence ({batch}), "
            f"got {int(targets.shape[0])}"
        )
    _check_lengths(input_lengths, "input_lengths", batch, time, op)
    width = int(targets.shape[1]) if padded else None
    lengths = _check_lengths(target_lengths, "target_lengths", batch, width, op)
    if not padded and sum(lengths) != int(targets.numel()):
        raise ValueError(
            f"{op}: concatenated targets hold {int(targets.numel())} labels, "
            f"but target_lengths add up to {sum(lengths)}"
        )
    if targets.numel() == 0:
        return lengths
    counted: Tensor | None = None
    if padded and width is not None:
        # Row b reads its first target_lengths[b] entries; the rest is
        # padding, which may hold anything.
        def real() -> Tensor:
            positions = _lucid.arange(width, device=targets.device)
            sizes = _lucid.tensor(lengths, device=targets.device)
            return positions.reshape([1, width]) < sizes.reshape([batch, 1])

        counted = _host_read(real)
    labels, usable = _host_read(lambda: _guard(targets, classes, counted))
    _refuse(
        labels,
        usable,
        lambda v: (
            f"{op}: target {v} is out of range for {classes} classes "
            f"(valid range [0, {classes - 1}])"
        ),
    )
    return lengths
