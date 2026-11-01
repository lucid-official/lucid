"""How a weight is stored in the package: float, int8, palettized or sparse.

Split from :mod:`lucid.coreml._build`.  Each storage format has the same
three steps: decide whether the tensor is worth compressing, write its bytes
into the weight blob (``_place_*``, run once per tensor when several
functions share a package, see :mod:`lucid.coreml._shared`), and declare the
constant that reads them (``_emit_*``).
"""

import math
import struct
from collections.abc import Callable
from typing import TYPE_CHECKING

import lucid
import lucid.quantization as _quant
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap
from lucid.coreml import _shared, _spec
from lucid.coreml._emit import _as_float
from lucid.coreml._spec import WeightPrecision

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor
    from lucid.coreml._build import _BuildCtx


def _flatten[T](tensor: Tensor, convert: Callable[[object], T]) -> list[T]:
    """Each element of ``tensor``, row-major, through ``convert`` (numpy-free, H4)."""
    flat: list[T] = []

    def walk(value: object) -> None:
        if isinstance(value, list):
            for item in value:
                walk(item)
        else:
            flat.append(convert(value))

    walk(tensor.tolist())
    return flat


# Below this many elements a weight is not worth quantizing: the
# per-channel scale is itself stored uncompressed, so a small tensor can
# come out larger, and the error it adds buys nothing. The same threshold
# the reference tooling uses, stated here rather than inherited.
_QUANTIZE_MIN_ELEMENTS = 2048


def _worth_storing_compressed(
    tensor: Tensor, weights: WeightPrecision | _spec.Palettize | _spec.Sparsify
) -> bool:
    """Whether a folded constant should go through the weight writer.

    The same rule the compressors apply themselves — rank at least two,
    enough elements to pay for the table — asked before the value is
    routed, so a small folded constant keeps travelling inline rather
    than acquiring a blob entry it does not need.
    """
    if weights is WeightPrecision.FLOAT:
        return False
    return len(tensor.shape) >= 2 and int(tensor.numel()) >= _QUANTIZE_MIN_ELEMENTS


#: A channel is tested for an existing grid on at most this many of its
#: elements. An ordinary float weight has as many distinct values as
#: elements, so a sample rejects it immediately; one that really sits on
#: an 8-bit grid cannot have more than 256 however much is looked at.
_GRID_SAMPLE = 1 << 12


def _grid_already_there(flat: Tensor) -> Tensor | None:
    """The step a weight is already quantized on, or ``None``.

    A quantization-aware model's weights have been trained onto a grid,
    and deriving a fresh one throws that away: this writer's scale is
    ``max / 127.5``, so re-quantizing a weight whose extreme is
    ``127 * step`` lands it on a grid 0.4% finer that its values do not
    sit on. Measured on a QAT classifier, that cost 1.4e-03 against a
    model the export could have reproduced exactly — the whole claim of
    training with quantization, given away at the last step.

    Rejected cheaply: a sample of each channel is sorted and its distinct
    values counted, and an ordinary weight has as many as it has
    elements. Only a weight that survives that is examined in full.

    Parameters
    ----------
    flat : Tensor
        ``(channels, elements)`` — the weight, one row per output
        channel.

    Returns
    -------
    Tensor or None
        ``(channels, 1)`` steps, or ``None`` when any channel holds more
        distinct values than eight bits can index, or when the values are
        not multiples of a common step.
    """
    width = int(flat.shape[1])
    sample = flat[:, :: (width // _GRID_SAMPLE + 1)] if width > _GRID_SAMPLE else flat
    ordered = lucid.sort(sample, dim=-1).values
    fresh = (ordered[:, 1:] != ordered[:, :-1]).to(lucid.float32)
    if float(fresh.sum(dim=1).max().item()) + 1 > 256:
        return None

    ordered = lucid.sort(flat, dim=-1).values
    gaps = ordered[:, 1:] - ordered[:, :-1]
    # The step is the smallest gap that is not two identical values. A
    # grid with a code missing in the middle would give a multiple of the
    # step instead, which the integrality check below then rejects.
    positive = lucid.where(gaps > 0, gaps, lucid.full_like(gaps, float("inf")))
    step = positive.min(dim=1).values.reshape(-1, 1)
    if not bool((step < float("inf")).all().item()):
        return None

    codes = (flat / step).round()
    # Measured against the step rather than against the weight's range:
    # the fake-quantize that produced these computed ``round(w / s) * s``
    # in float32, so a value sits within an ulp of its lattice point, and
    # an ulp of the largest weight is far larger than an ulp of the
    # smallest. A thousandth of a step is decisively on the lattice.
    off = (flat - codes * step).abs() / step
    # A hundredth of a step. The discriminating work is done by the
    # distinct-value count above — an ordinary weight has as many values
    # as elements and never gets here — so this only has to tolerate the
    # float32 round trip that produced the lattice, which on a trained
    # weight runs to about a thousandth of a step and on an untrained one
    # to a tenth of that.
    if float(off.max().item()) > 1e-2:
        return None
    # int8 runs from -128 to 127, and a symmetric fake-quantize reaches
    # the negative end — rejecting on magnitude alone would throw away
    # the grid for a code that is perfectly representable.
    if float(codes.max().item()) > 127.0 or float(codes.min().item()) < -128.0:
        return None
    return step


def _quantize_weight(tensor: Tensor, scale_mil: int) -> tuple[Tensor, bytes, bytes, int] | None:
    """A weight as int8 codes with one scale per output channel.

    Returns ``None`` when the tensor should stay in floating point — a
    bias, a norm's parameters, anything below the threshold, or a rank-1
    value that has no channel axis to scale along.

    Symmetric rather than affine: a symmetric grid puts zero exactly on a
    lattice point, which matters because padding and masking write real
    zeros that an affine grid would round to something else.

    Parameters
    ----------
    tensor : Tensor
        Float weight to quantize.
    scale_mil : int
        MIL dtype the scale is stored as — the body's precision, so the
        dequantized value lands in the type the operation wants.

    Returns
    -------
    tuple or None
        ``(codes, scale bytes, zero-point bytes, channels)``.
    """
    if len(tensor.shape) < 2:
        return None
    count = 1
    for dim in tensor.shape:
        count *= int(dim)
    if count < _QUANTIZE_MIN_ELEMENTS:
        return None

    channels = int(tensor.shape[0])
    flat = tensor.reshape(channels, -1)
    existing = _grid_already_there(flat)
    if existing is not None:
        # Already on a grid, so it is written on that one. Deriving a new
        # one would round values that were trained to sit exactly where
        # they are.
        scale = existing.reshape(-1)
        zero_point = lucid.zeros_like(scale)
        codes = (flat / existing).round().reshape(*[int(d) for d in tensor.shape]).to(lucid.int8)
    else:
        scale, zero_point = _quant.calculate_qparams(
            flat.min(dim=1)[0], flat.max(dim=1)[0], _quant.per_channel_symmetric, _quant.qint8
        )
        codes = _quant.quantize(tensor, scale, zero_point, _quant.qint8, ch_axis=0)

    # MIL has no float16 or int8 immediate list; both travel as raw
    # little-endian bytes, so they are packed here rather than handed to
    # the engine as numbers it would have to re-encode.
    fmt = "<e" if scale_mil == _spec.FLOAT16 else "<f"
    scale_bytes = b"".join(struct.pack(fmt, v) for v in _flatten(scale, _as_float))
    zero_bytes = b"".join(struct.pack("<b", int(v)) for v in _flatten(zero_point, _as_float))
    return codes, scale_bytes, zero_bytes, channels


#: A palette is fitted from at most this many weights.  The table has at
#: most 256 entries, so past a certain sample the centroids stop moving
#: and every extra element is time spent for nothing — an AlexNet
#: classifier weight has 37 million of them.
_PALETTE_SAMPLE = 1 << 16


def _assign(rows: Tensor, table: Tensor, count: int) -> Tensor:
    """Which palette entry each value falls in, a row at a time.

    ``lucid.bucketize`` answers this for one shared edge list, but it
    walks the whole list per element — a 256-entry table costs 255 passes
    over the weight — and it cannot give each row its own edges, which is
    the whole point of a per-channel palette. The edges are sorted and
    ``count`` is a power of two, so bisection answers in ``log2(count)``
    passes and the running index needs no bounds check: the steps sum to
    ``count - 1`` exactly.

    ``table`` is ``(groups, count)``: column zero is a placeholder that
    nothing probes (the first candidate is already ``count // 2``), and
    the rest are the midpoints between neighbouring palette entries. The
    convention matches ``bucketize`` — the entry is the number of edges
    strictly below the value — which is what the unit test compares
    against.
    """
    found = lucid.zeros_like(rows).to(lucid.int64)
    step = count // 2
    while step >= 1:
        candidate = found + step
        picked = lucid.gather(table, 1, candidate)
        found = lucid.where(picked < rows, candidate, found)
        step //= 2
    return found


def _edge_table(centres: Tensor) -> Tensor:
    """Midpoints between neighbouring palette entries, per row.

    Column zero is the placeholder ``_assign`` never reads, kept so the
    table indexes the same way the palette does.
    """
    groups = int(centres.shape[0])
    lower = centres[:, :-1]
    upper = centres[:, 1:]
    head = lucid.full((groups, 1), -math.inf)
    return lucid.concat([head, (lower + upper) / 2], dim=1)


#: A palette is fitted from at most this many values per group.  The
#: table has at most 256 entries, so past a certain sample the centroids
#: stop moving and every extra element is time spent for nothing.
_PALETTE_SAMPLE = 1 << 14

#: Lloyd passes.  Well past where the movement stops mattering at these
#: sizes, and a fixed count keeps export time predictable.
_PALETTE_PASSES = 10


def _values_already_there(ordered: Tensor, count: int) -> Tensor | None:
    """The exact palette of a weight that already sits on few values.

    ``None`` when any row holds more distinct values than the table has
    entries, which is every ordinary weight.

    Lloyd's algorithm does not reliably find this palette even though it
    is the optimum with zero error. Two initial centres can land inside
    one repeated value, leaving another value unclaimed — and an entry no
    value landed on keeps the centre it had and never moves again, so the
    iteration cannot recover. Measured on a fine-tuned model whose rows
    held exactly sixteen values, the refit came back 9.3e-03 away from
    the palette the model had been trained against.

    That case is not exotic. It is what a compression-aware fine-tune
    produces by construction, and the whole point of that fine-tune is
    that the export writes the palette the model learned rather than a
    nearby one.

    Parameters
    ----------
    ordered : Tensor
        ``(groups, span)``, each row sorted ascending.
    count : int
        Entries the table has.

    Returns
    -------
    Tensor or None
        ``(groups, count)`` palettes, ascending, with any unused tail
        entry filled by the row's largest value so the table stays
        monotone for the bisection.
    """
    groups = int(ordered.shape[0])
    fresh = lucid.concat(
        [lucid.ones(groups, 1), (ordered[:, 1:] != ordered[:, :-1]).to(lucid.float32)], dim=1
    )
    slot = (lucid.cumsum(fresh, dim=1) - 1.0).to(lucid.int64)
    if int(slot.max().item()) >= count:
        return None

    empty = lucid.zeros(groups, count)
    totals = lucid.scatter_add(empty, 1, slot, ordered)
    hits = lucid.scatter_add(empty, 1, slot, lucid.ones_like(ordered))
    safe = lucid.where(hits > 0, hits, lucid.ones_like(hits))
    return lucid.where(hits > 0, totals / safe, ordered[:, -1:])


def _palettes_for(rows: Tensor, count: int) -> Tensor:
    """One palette per row, by Lloyd's algorithm in one dimension.

    Every row is fitted at once rather than in a Python loop over
    channels: a bisection for the assignment and two ``scatter_add``
    passes for the means, all batched. A per-channel fit over a 2048
    channel weight is 2048 independent clusterings, and doing them one at
    a time is the difference between an export that takes seconds and one
    that takes an hour.

    Started from evenly spaced order statistics rather than from the
    range, so a row whose mass sits near zero does not spend most of its
    table on tails it barely uses.
    """
    groups = int(rows.shape[0])
    width = int(rows.shape[1])
    sample = rows[:, :: (width // _PALETTE_SAMPLE + 1)] if width > _PALETTE_SAMPLE else rows
    span = int(sample.shape[1])

    ordered = lucid.sort(sample, dim=-1).values
    settled = _values_already_there(ordered, count)
    if settled is not None:
        # Exact, and the iteration below would not find it — see the
        # helper. Every ordinary weight falls straight through this.
        return settled
    marks = [float(min(span - 1, int((index + 0.5) * span / count))) for index in range(count)]
    centres = lucid.gather(
        ordered, 1, (lucid.zeros(groups, 1) + lucid.tensor(marks)).to(lucid.int64)
    )

    ones = lucid.ones_like(sample)
    empty = lucid.zeros(groups, count)
    for _ in range(_PALETTE_PASSES):
        assigned = _assign(sample, _edge_table(centres), count)
        totals = lucid.scatter_add(empty, 1, assigned, sample)
        weights = lucid.scatter_add(empty, 1, assigned, ones)
        # An entry no value landed on keeps the centre it had; dividing by
        # its zero population would put a NaN in the table and take the
        # whole row's bisection with it.
        moved = lucid.where(
            weights > 0, totals / lucid.where(weights > 0, weights, ones[:, :1]), centres
        )
        # Re-sorted because the bisection needs a monotone table, and a
        # row whose entries crossed would otherwise answer nonsense.
        centres = lucid.sort(moved, dim=-1).values
    return centres


def _palette_groups(channels: int, count: int, elements: int) -> int:
    """How many palettes to fit along the output-channel axis.

    One per channel is what makes palettization usable — a convolution's
    channels differ in magnitude by orders of magnitude, and a single
    shared table spends nearly all of its resolution on the loudest ones.
    But the tables are stored, so at 256 entries and 2048 channels they
    would cost more than the keys they index. Groups are halved until the
    table is a small fraction of the payload it serves.
    """
    key_bytes = elements * (count.bit_length() - 1) // 8
    groups = channels
    while groups > 1 and groups * count * 2 > key_bytes // 4:
        groups //= 2
    # The group count has to divide the channel count for the operation
    # to lay the tables out over the axis.
    while groups > 1 and channels % groups:
        groups -= 1
    return max(groups, 1)


def _palettize_weight(tensor: Tensor, bits: int) -> tuple[bytes, Tensor, list[int], int] | None:
    """A weight as packed keys into one palette per group of channels.

    ``None`` when the weight is too small to be worth the tables, or has
    no channel axis — the same two refusals the int8 path makes, for the
    same reasons: the tables cost something fixed, and a bias or a norm's
    parameters have no output channel to group along and are ruinous to
    approximate.

    Returns the packed keys, the tables, the shape the operation wants
    them in, and the width the keys were packed at — the last so the
    caller picks its dtype tags from the same number that did the
    packing rather than from the request.
    """
    if len(tensor.shape) < 2 or int(tensor.numel()) < _QUANTIZE_MIN_ELEMENTS:
        return None
    count = 1 << bits
    channels = int(tensor.shape[0])
    elements = int(tensor.numel())
    groups = _palette_groups(channels, count, elements)
    # Fitted in single precision even when the body is half. The weight
    # has already been rounded to float16 by then, so this loses nothing
    # — and Lloyd's algorithm on half-precision centroids both converges
    # worse and mixes dtypes against the edge table's sentinel, which is
    # how a float16 palettized export used to fail outright.
    rows = tensor.to(lucid.float32).reshape(groups, -1)

    palettes = _palettes_for(rows, count)
    keys = _assign(rows, _edge_table(palettes), count)
    # ``[groups, 1, ..., 1, count, 1]``: one table per group along the
    # output-channel axis, then the entries, then the vector width — one,
    # because these are scalar palettes.
    lut_shape = [groups] + [1] * (len(tensor.shape) - 1) + [count, 1]
    return (
        _C_engine.coreml.pack_bits(_unwrap(keys.reshape(-1)), bits),
        palettes.reshape(-1),
        lut_shape,
        bits,
    )


def _sparsify_weight(tensor: Tensor, ratio: float) -> tuple[Tensor, bytes, int] | None:
    """A weight as its surviving values plus one bit per element.

    ``None`` when nothing would be dropped, or when the weight is small
    enough that the mask costs more than the zeros save.
    """
    if len(tensor.shape) < 2 or ratio <= 0.0:
        return None
    if int(tensor.numel()) < _QUANTIZE_MIN_ELEMENTS:
        return None
    flat = tensor.reshape(-1)
    threshold = float(lucid.quantile(flat.abs(), ratio).item())
    keep = flat.abs() > threshold
    kept = int(keep.to(lucid.int64).sum().item())
    if kept == int(flat.numel()) or kept == 0:
        # Nothing to drop, or nothing to keep. The second happens for
        # real: a constant whose elements are all the same value has no
        # element above its own median, and a folded batch norm leaves
        # whole zero vectors behind. Such a constant would be a sparse
        # const with an empty payload, which is not something to write —
        # it goes out dense, at a cost of one small tensor.
        return None

    # Both of these are tensor work rather than Python loops: a weight
    # carries tens of millions of elements and iterating them here cost
    # minutes per layer.
    values = lucid.masked_select(flat, keep)
    return values, _C_engine.coreml.pack_bits(_unwrap(keep), 1), kept


def _place[P: _shared.Placement](
    ctx: _BuildCtx,
    source: _C_engine.TensorImpl | None,
    kind: type[P],
    write: Callable[[], P | None],
) -> P | None:
    """Write ``source`` as ``kind`` once per package; ``None`` source is never shared."""
    if ctx.into is None or source is None:
        return write()
    return ctx.into.place(source, kind, (ctx.half, ctx.target), write)


def _place_palettized(
    ctx: _BuildCtx, tensor: Tensor, palettize: _spec.Palettize
) -> _shared.Palettized | None:
    palettized = _palettize_weight(tensor, palettize.bits)
    if palettized is None:
        return None
    keys, palettes, lut_shape, key_bits = palettized
    offset = ctx.blob.append_bytes(keys, _C_engine.coreml.BLOB_SUBBYTE[str(key_bits)])
    lut_offset = ctx.blob.append_tensor(
        _unwrap(palettes.half() if ctx.half else palettes), ctx.body_blob
    )
    return _shared.Palettized(offset, lut_offset, lut_shape, key_bits)


def _emit_palettized_const(
    ctx: _BuildCtx, name: str, shape: list[int], placed: _shared.Palettized
) -> None:
    ctx.program.add_grouped_lut_const(
        name,
        (ctx.body_mil, shape),
        placed.offset,
        _C_engine.coreml.MIL_SUBBYTE[str(placed.key_bits)],
        placed.lut_offset,
        ctx.body_mil,
        placed.lut_shape,
    )
    ctx.quantized_count += 1


def _place_sparse(
    ctx: _BuildCtx, tensor: Tensor, sparsify: _spec.Sparsify
) -> _shared.Sparse | None:
    sparse = _sparsify_weight(tensor, sparsify.ratio)
    if sparse is None:
        return None
    values, mask, kept = sparse
    if ctx.half:
        values = values.half()
    value_offset = ctx.blob.append_tensor(_unwrap(values), ctx.body_blob)
    if ctx.target is _spec.DeploymentTarget.IOS18:
        mask_offset = ctx.blob.append_bytes(mask, _C_engine.coreml.BLOB_SUBBYTE["1"])
    else:
        mask_offset = ctx.blob.append_bytes(mask, _spec.BLOB_UINT8)
    return _shared.Sparse(value_offset, mask_offset, kept, len(mask))


def _emit_sparse_const(ctx: _BuildCtx, name: str, shape: list[int], placed: _shared.Sparse) -> None:
    if ctx.target is _spec.DeploymentTarget.IOS18:
        ctx.program.add_sparse_const_extended(
            name,
            (ctx.body_mil, shape),
            placed.value_offset,
            placed.kept,
            ctx.body_mil,
            placed.mask_offset,
        )
    else:
        ctx.program.add_sparse_const(
            name,
            (ctx.body_mil, shape),
            placed.value_offset,
            placed.kept,
            placed.mask_offset,
            placed.mask_len,
        )
    ctx.quantized_count += 1


def _place_int8(ctx: _BuildCtx, tensor: Tensor) -> _shared.Int8 | None:
    quantized = _quantize_weight(tensor, ctx.body_mil)
    if quantized is None:
        return None
    codes, scale_bytes, zero_bytes, channels = quantized
    offset = ctx.blob.append_tensor(_unwrap(codes), _spec.BLOB_INT8)
    return _shared.Int8(offset, scale_bytes, zero_bytes, channels)


def _emit_int8_const(ctx: _BuildCtx, name: str, shape: list[int], placed: _shared.Int8) -> None:
    ctx.program.add_quantized_const(
        name,
        (ctx.body_mil, shape),
        placed.offset,
        placed.scale_bytes,
        ctx.body_mil,
        placed.zero_bytes,
        placed.channels,
        0,
    )
    ctx.quantized_count += 1


def _write_compressed_const(
    ctx: _BuildCtx, name: str, tensor: Tensor, shape: list[int], source: _C_engine.TensorImpl | None
) -> bool:
    """Write ``tensor`` as ``ctx.weights`` asks; False means it stays a float blob."""
    weights = ctx.weights
    if isinstance(weights, _spec.Palettize):
        palettized = _place(
            ctx, source, _shared.Palettized, lambda: _place_palettized(ctx, tensor, weights)
        )
        if palettized is not None:
            _emit_palettized_const(ctx, name, shape, palettized)
            return True
    elif isinstance(weights, _spec.Sparsify):
        sparse = _place(ctx, source, _shared.Sparse, lambda: _place_sparse(ctx, tensor, weights))
        if sparse is not None:
            _emit_sparse_const(ctx, name, shape, sparse)
            return True
    elif weights is WeightPrecision.INT8:
        quantized = _place(ctx, source, _shared.Int8, lambda: _place_int8(ctx, tensor))
        if quantized is not None:
            _emit_int8_const(ctx, name, shape, quantized)
            return True
    return False


def _write_float_const(
    ctx: _BuildCtx, name: str, tensor: Tensor, source: _C_engine.TensorImpl | None
) -> None:
    """Write one float constant, compressed the way ``ctx.weights`` asks.

    ``source`` is the traced tensor behind ``tensor`` when it may be shared
    between the functions of one package, else ``None``.
    """
    shape = [int(d) for d in tensor.shape]
    if _write_compressed_const(ctx, name, tensor, shape, source):
        return
    is_f16 = tensor.dtype == lucid.float16
    mil_dt = _spec.FLOAT16 if is_f16 else _spec.FLOAT32
    blob_dt = _spec.BLOB_FLOAT16 if is_f16 else _spec.BLOB_FLOAT32
    written = _place(
        ctx,
        source,
        _shared.Float,
        lambda: _shared.Float(ctx.blob.append_tensor(_unwrap(tensor), blob_dt)),
    )
    assert written is not None  # a float blob is always writable
    ctx.program.add_blob_const(name, (mil_dt, shape), written.offset)
