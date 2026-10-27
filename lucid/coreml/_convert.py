"""Every conversion between element types a package makes (LCD-388).

MIL does not promote, so operands are converted explicitly — and two
conversions are not what they look like. A ``cast`` to or from bool is
folded wrongly by Core ML when its operand is constant, and the gathers
and scatters take no bool data at all. :func:`emit_cast` is the one place
a conversion is spelled, so neither reaches a package.
"""

import math
from typing import TYPE_CHECKING

from lucid.coreml import _flex, _spec

if TYPE_CHECKING:
    from lucid.coreml._build import Builder
    from lucid.coreml._emit import Bindings, EmitResult, TracedOp

__all__ = [
    "MIL_CAST_NAMES",
    "emit_cast",
    "emit_opmath_binary",
    "moved",
    "scalar_like",
    "scatter_along_axis",
    "widened",
]

_MIL_BOOL = _spec.BOOL
_MIL_INT32 = _spec.INT32
_MIL_UINT8 = _spec.UINT8

#: MIL's spelling of each element type a conversion can land on — the
#: ``out_dtype`` :func:`emit_cast` takes.
MIL_CAST_NAMES = {
    _spec.FLOAT16: "fp16",
    _spec.FLOAT32: "fp32",
    _MIL_INT32: "int32",
    _MIL_BOOL: "bool",
}


def scalar_like(b: Builder, value: str, number: int) -> str:
    """A scalar ``number`` of ``value``'s type: MIL refuses int32 beside a float.

    Parameters
    ----------
    b : Builder
        Builder minting the constant.
    value : str
        The operand the scalar will meet.
    number : int
        The scalar.

    Returns
    -------
    str
        The constant's name: int32 beside an int32 or bool value, otherwise
        a float at the body's precision.
    """
    if b.dtype_of(value) in (_MIL_INT32, _MIL_BOOL):
        return b.const_int(number)
    return b.const_float(float(number))


def emit_cast(b: Builder, value: str, out_dtype: str) -> EmitResult:
    """``value`` converted to ``out_dtype``: every conversion the program makes.

    The driver's own casts bracket an fp16 body, which still presents fp32
    inputs and outputs; ``astype`` and the operations that need another
    operand type make the rest.

    A conversion to or from bool is not a MIL ``cast``. Core ML folds such
    a cast wrongly when its operand is constant — built from the model's
    own constants, as a causal mask is: ``ones(5, 5).triu(1).bool()`` came
    back all false in a float16 package, and a float32 package lost the
    mask once a scatter cast it on to int32 (LCD-388). ``not_equal`` and
    ``select`` say the same thing and are folded correctly.

    Parameters
    ----------
    b : Builder
        Builder minting the constants.
    value : str
        Name of the value to convert.
    out_dtype : str
        MIL's spelling of the target type — ``"fp16"``, ``"fp32"``,
        ``"int32"`` or ``"bool"``.

    Returns
    -------
    tuple[str, list]
        The last operation, for the caller to append and declare.
    """
    from_bool = b.dtype_of(value) == _MIL_BOOL
    if out_dtype == "bool" and not from_bool:
        return "not_equal", [("x", value), ("y", scalar_like(b, value, 0))]
    if from_bool and out_dtype != "bool":
        picked: Bindings = [
            ("cond", value),
            ("a", b.const_int(1)),
            ("b", b.const_int(0)),
        ]
        if out_dtype == "int32":
            return "select", picked
        # On through int32, whose cast to a float is not a bool cast.
        value = b.emit("select", picked, b.shape_of(value), dtype=_MIL_INT32)
    return "cast", [("x", value), ("dtype", b.const_str(out_dtype))]


def widened(b: Builder, value: str) -> str:
    """``value`` as int32 1 / 0 when it is bool, else itself.

    MIL's gathers and scatters take no bool data: the package fails to
    parse, naming the operand (LCD-388).  A bool tensor is moved as int32
    and turned back into bool afterwards, which 1 and 0 survive exactly.

    Parameters
    ----------
    b : Builder
        Builder appending the conversion.
    value : str
        The operand.

    Returns
    -------
    str
        ``value`` itself, or an int32 copy of it.
    """
    if b.dtype_of(value) != _MIL_BOOL:
        return value
    return b.emit(*emit_cast(b, value, "int32"), b.shape_of(value), dtype=_MIL_INT32)


def moved(
    b: Builder, op: TracedOp, mil_type: str, bindings: Bindings, data: tuple[str, ...]
) -> EmitResult:
    """``mil_type`` with its ``data`` parameters widened when the result is bool.

    Parameters
    ----------
    b : Builder
        Builder appending the int32 operation.
    op : TracedOp
        The traced gather or scatter; its declared result type decides.
    mil_type : str
        The MIL operation.
    bindings : Bindings
        Its operands, as an emitter returns them.
    data : tuple of str
        The parameters that carry the tensor being moved.

    Returns
    -------
    tuple[str, list]
        The operation itself for a non-bool result; otherwise the
        conversion of its int32 result back to bool.
    """
    if b.result_mil_dtype(op) != _MIL_BOOL:
        return mil_type, bindings
    operands: Bindings = [
        (
            param,
            widened(b, value) if param in data and isinstance(value, str) else value,
        )
        for param, value in bindings
    ]
    result = b.emit(mil_type, operands, b.result_dims(op), dtype=_MIL_INT32)
    return emit_cast(b, result, "bool")


def _split_prefix(
    b: Builder, value: str, shape: list[int], axis: int, head_size: int
) -> tuple[str, str, list[int]]:
    first, second = list(shape), list(shape)
    first[axis], second[axis] = head_size, shape[axis] - head_size
    head, tail = b.emit_multi(
        "split",
        [
            ("x", value),
            ("split_sizes", b.const_ints([first[axis], second[axis]])),
            ("axis", b.const_int(axis)),
        ],
        [first, second],
    )
    return head, tail, first


def scatter_along_axis(
    b: Builder, op: TracedOp, ins: list[str], axis: int, mode: str
) -> EmitResult:
    """Emit MIL ``scatter_along_axis``, narrowing non-scatter axes when ``indices`` is a sub-slice."""
    is_bool = b.result_mil_dtype(op) == _MIL_BOOL
    work_dtype = _MIL_INT32 if is_bool else b.result_mil_dtype(op)
    data = widened(b, ins[0]) if is_bool else ins[0]
    indices = ins[1]
    updates = widened(b, ins[2]) if is_bool else ins[2]
    cur_shape = b.shape_of(data).declared()
    idx_shape = b.shape_of(indices).declared()
    upd_shape = b.shape_of(updates).declared()
    rank = len(cur_shape)
    norm_axis = axis % rank if rank else 0
    for a in range(rank):
        if 0 < idx_shape[a] < upd_shape[a]:
            updates, _, upd_shape = _split_prefix(b, updates, upd_shape, a, idx_shape[a])
    tails: list[tuple[int, str, list[int]]] = []
    for a in range(rank):
        if a != norm_axis and 0 < idx_shape[a] < cur_shape[a]:
            full_shape = list(cur_shape)
            data, tail, cur_shape = _split_prefix(b, data, cur_shape, a, idx_shape[a])
            tails.append((a, tail, full_shape))
    bindings: Bindings = [
        ("data", data),
        ("indices", indices),
        ("updates", updates),
        ("axis", b.const_int(axis)),
        ("mode", b.const_str(mode)),
        ("validate_indices", b.const_bool(False)),
    ]
    if not tails and not is_bool:
        return "scatter_along_axis", bindings
    cur = b.emit("scatter_along_axis", bindings, cur_shape, dtype=work_dtype)
    for step, (a, tail, full_shape) in enumerate(reversed(tails)):
        c_bindings: Bindings = [
            ("values", [cur, tail]),
            ("axis", b.const_int(a)),
            ("interleave", b.const_bool(False)),
        ]
        if step == len(tails) - 1 and not is_bool:
            return "concat", c_bindings
        cur = b.emit("concat", c_bindings, full_shape, dtype=work_dtype)
    return emit_cast(b, cur, "bool")


def _broadcast_f32(b: Builder, val: str, peer: str, dims: _flex.Dims) -> str:
    """Broadcast ``val`` to ``dims`` by multiplying by a 1.0 tensor of ``peer``'s shape."""
    if b.shape_of(val) == dims.declared():
        return val
    peer_shape = b.shape_of(peer)
    less_peer = b.emit(
        "less", [("x", peer), ("y", b.const_float32(0.0))], peer_shape, dtype=_MIL_BOOL
    )
    true_peer = b.emit(
        "logical_or",
        [("x", less_peer), ("y", b.const_bool(True))],
        peer_shape,
        dtype=_MIL_BOOL,
    )
    one_peer = b.emit(*emit_cast(b, true_peer, "fp32"), peer_shape, dtype=_spec.FLOAT32)
    return b.emit("mul", [("x", val), ("y", one_peer)], dims, dtype=_spec.FLOAT32)


def _emit_f32_fmod_bindings(b: Builder, a: str, b_val: str, dims: _flex.Dims) -> EmitResult:
    """Exact ``fmod(a, b_val)`` in float32 for equal-shape operands ``a`` and ``b_val``."""
    f32 = _spec.FLOAT32
    c_zero = b.const_float32(0.0)
    c_one = b.const_float32(1.0)
    c_neg_one = b.const_float32(-1.0)
    c_inf = b.const_float32(math.inf)
    abs_a = b.emit("abs", [("x", a)], dims, dtype=f32)
    abs_b = b.emit("abs", [("x", b_val)], dims, dtype=f32)
    r = abs_a
    for k in (96, 80, 64, 48, 32, 16):
        c_2k = b.const_float32(float(2**k))
        bk = b.emit("mul", [("x", abs_b), ("y", c_2k)], dims, dtype=f32)
        bk_fin = b.emit("less", [("x", bk), ("y", c_inf)], dims, dtype=_MIL_BOOL)
        step = b.emit("select", [("cond", bk_fin), ("a", bk), ("b", abs_b)], dims, dtype=f32)
        rn = b.emit("mod", [("x", r), ("y", step)], dims, dtype=f32)
        rn_neg = b.emit("less", [("x", rn), ("y", c_zero)], dims, dtype=_MIL_BOOL)
        rn_fix = b.emit("add", [("x", rn), ("y", step)], dims, dtype=f32)
        r = b.emit("select", [("cond", rn_neg), ("a", rn_fix), ("b", rn)], dims, dtype=f32)
    rn = b.emit("mod", [("x", r), ("y", abs_b)], dims, dtype=f32)
    rn_neg = b.emit("less", [("x", rn), ("y", c_zero)], dims, dtype=_MIL_BOOL)
    rn_fix = b.emit("add", [("x", rn), ("y", abs_b)], dims, dtype=f32)
    r = b.emit("select", [("cond", rn_neg), ("a", rn_fix), ("b", rn)], dims, dtype=f32)
    a_diff = b.emit("sub", [("x", abs_a), ("y", abs_a)], dims, dtype=f32)
    a_fin = b.emit("equal", [("x", a_diff), ("y", c_zero)], dims, dtype=_MIL_BOOL)
    b_inf = b.emit("equal", [("x", abs_b), ("y", c_inf)], dims, dtype=_MIL_BOOL)
    fin_over_inf = b.emit("logical_and", [("x", a_fin), ("y", b_inf)], dims, dtype=_MIL_BOOL)
    r = b.emit("select", [("cond", fin_over_inf), ("a", abs_a), ("b", r)], dims, dtype=f32)
    a_lt = b.emit("less", [("x", a), ("y", c_zero)], dims, dtype=_MIL_BOOL)
    a_eq = b.emit("equal", [("x", a), ("y", c_zero)], dims, dtype=_MIL_BOOL)
    inv_a = b.emit("real_div", [("x", c_one), ("y", a)], dims, dtype=f32)
    inv_neg = b.emit("less", [("x", inv_a), ("y", c_zero)], dims, dtype=_MIL_BOOL)
    neg_zero = b.emit("logical_and", [("x", a_eq), ("y", inv_neg)], dims, dtype=_MIL_BOOL)
    a_sign = b.emit("logical_or", [("x", a_lt), ("y", neg_zero)], dims, dtype=_MIL_BOOL)
    neg_r = b.emit("mul", [("x", r), ("y", c_neg_one)], dims, dtype=f32)
    return "select", [("cond", a_sign), ("a", neg_r), ("b", r)]


def _emit_f32_floordiv_bindings(
    b: Builder,
    operands: tuple[str, str],
    mod_adjust: tuple[str, str],
    dims: _flex.Dims,
) -> EmitResult:
    """Exact ``floordiv(a, b_val)`` in float32 from ``mod_val`` and ``adjust``."""
    a, b_val = operands
    mod_val, adjust = mod_adjust
    f32 = _spec.FLOAT32
    c_zero = b.const_float32(0.0)
    c_one = b.const_float32(1.0)
    c_half = b.const_float32(0.5)
    raw_div = b.emit("real_div", [("x", a), ("y", b_val)], dims, dtype=f32)
    num = b.emit("sub", [("x", a), ("y", mod_val)], dims, dtype=f32)
    q_exact = b.emit("real_div", [("x", num), ("y", b_val)], dims, dtype=f32)
    q_minus_one = b.emit("sub", [("x", q_exact), ("y", c_one)], dims, dtype=f32)
    div = b.emit(
        "select",
        [("cond", adjust), ("a", q_minus_one), ("b", q_exact)],
        dims,
        dtype=f32,
    )
    whole = b.emit("floor", [("x", div)], dims, dtype=f32)
    frac = b.emit("sub", [("x", div), ("y", whole)], dims, dtype=f32)
    round_up = b.emit("greater", [("x", frac), ("y", c_half)], dims, dtype=_MIL_BOOL)
    whole_inc = b.emit("add", [("x", whole), ("y", c_one)], dims, dtype=f32)
    whole = b.emit("select", [("cond", round_up), ("a", whole_inc), ("b", whole)], dims, dtype=f32)
    signed_zero = b.emit("mul", [("x", raw_div), ("y", c_zero)], dims, dtype=f32)
    div_zero = b.emit("equal", [("x", div), ("y", c_zero)], dims, dtype=_MIL_BOOL)
    res = b.emit(
        "select",
        [("cond", div_zero), ("a", signed_zero), ("b", whole)],
        dims,
        dtype=f32,
    )
    b_zero = b.emit("equal", [("x", b_val), ("y", c_zero)], dims, dtype=_MIL_BOOL)
    return "select", [("cond", b_zero), ("a", raw_div), ("b", res)]


def _emit_float_divmod(b: Builder, op: TracedOp, lhs: str, rhs: str, target: int) -> EmitResult:
    """Emit floating-point ``floordiv``, ``remainder``, or ``fmod`` in float32."""
    f32 = _spec.FLOAT32
    dims = b.result_dims(op)
    if target == _spec.FLOAT16 and op.name != "floordiv":
        lhs, rhs = b.agree_on_dtype([lhs, rhs], _spec.FLOAT16)
    x_f32, y_f32 = b.agree_on_dtype([lhs, rhs], f32)
    a = _broadcast_f32(b, x_f32, y_f32, dims)
    b_val = _broadcast_f32(b, y_f32, x_f32, dims)
    fmod_res = _emit_f32_fmod_bindings(b, a, b_val, dims)
    if op.name == "fmod":
        if target == _spec.FLOAT16:
            return emit_cast(b, b.emit(*fmod_res, dims, dtype=f32), "fp16")
        return fmod_res
    mod_val = b.emit(*fmod_res, dims, dtype=f32)
    c_zero = b.const_float32(0.0)
    mod_nz = b.emit("not_equal", [("x", mod_val), ("y", c_zero)], dims, dtype=_MIL_BOOL)
    mod_neg = b.emit("less", [("x", mod_val), ("y", c_zero)], dims, dtype=_MIL_BOOL)
    b_neg = b.emit("less", [("x", b_val), ("y", c_zero)], dims, dtype=_MIL_BOOL)
    diff_sign = b.emit("logical_xor", [("x", mod_neg), ("y", b_neg)], dims, dtype=_MIL_BOOL)
    adjust = b.emit("logical_and", [("x", mod_nz), ("y", diff_sign)], dims, dtype=_MIL_BOOL)
    if op.name == "remainder":
        mod_plus_b = b.emit("add", [("x", mod_val), ("y", b_val)], dims, dtype=f32)
        rem_res: EmitResult = (
            "select",
            [("cond", adjust), ("a", mod_plus_b), ("b", mod_val)],
        )
        if target == _spec.FLOAT16:
            return emit_cast(b, b.emit(*rem_res, dims, dtype=f32), "fp16")
        return rem_res
    fdiv_res = _emit_f32_floordiv_bindings(b, (a, b_val), (mod_val, adjust), dims)
    if target == _spec.FLOAT16:
        return emit_cast(b, b.emit(*fdiv_res, dims, dtype=f32), "fp16")
    return fdiv_res


def _bnns_zero(b: Builder) -> str:
    """Emit a scalar int32 zero via uint8 gather so EirValidator rejects classic_cpu.

    Core ML's legacy Espresso CPU backend (``classic_cpu``) executes ``floor_div``
    and ``mod`` in float32, losing precision on ``int32`` operands above ``2^24``.
    On virtualized ``universal`` platforms (where decision-tree cost models are
    absent and ``classic_cpu`` ties ``bnns`` on op cost), ``EirValidator``
    explicitly rejects ``gather`` with ``uint8`` output and ``int32`` indices,
    causing the segmenter to keep the connected ``int32`` divmod subgraph on
    native 32-bit ``BNNSGraph`` (``main_bnns``) to avoid backend transfer cost.
    """
    c0 = b.const_int(0)
    table = b.emit(
        "cast",
        [("x", b.const_ints([0])), ("dtype", b.const_str("uint8"))],
        [1],
        dtype=_MIL_UINT8,
    )
    gathered = b.emit(
        "gather",
        [
            ("x", table),
            ("indices", c0),
            ("axis", c0),
            ("validate_indices", b.const_bool(False)),
        ],
        [],
        dtype=_MIL_UINT8,
    )
    return b.emit(
        "cast",
        [("x", gathered), ("dtype", b.const_str("int32"))],
        [],
        dtype=_MIL_INT32,
    )


def _emit_int_divmod(b: Builder, op: TracedOp, lhs: str, rhs: str) -> EmitResult:
    """Emit integer ``floordiv``, ``remainder``, or ``fmod`` in int32."""
    x, y = b.agree_on_dtype([lhs, rhs], _MIL_INT32)
    dims = b.result_dims(op)
    zero = _bnns_zero(b)
    if op.name == "floordiv":
        q = b.emit("floor_div", [("x", x), ("y", y)], dims, dtype=_MIL_INT32)
        return "add", [("x", q), ("y", zero)]
    r = b.emit("mod", [("x", x), ("y", y)], dims, dtype=_MIL_INT32)
    if op.name == "fmod":
        return "add", [("x", r), ("y", zero)]
    r_plus_y = b.emit("add", [("x", r), ("y", y)], dims, dtype=_MIL_INT32)
    r_nz = b.emit("not_equal", [("x", r), ("y", zero)], dims, dtype=_MIL_BOOL)
    r_neg = b.emit("less", [("x", r), ("y", zero)], dims, dtype=_MIL_BOOL)
    y_neg = b.emit("less", [("x", y), ("y", zero)], b.shape_of(y), dtype=_MIL_BOOL)
    diff_sign = b.emit("logical_xor", [("x", r_neg), ("y", y_neg)], dims, dtype=_MIL_BOOL)
    adjust = b.emit("logical_and", [("x", r_nz), ("y", diff_sign)], dims, dtype=_MIL_BOOL)
    return "select", [("cond", adjust), ("a", r_plus_y), ("b", r)]


def emit_opmath_binary(
    b: Builder,
    op: TracedOp,
    lucid_name: str,
    mil_name: str,
    lhs: str,
    rhs: str,
    target: int | None,
) -> EmitResult | None:
    """Emit exact ``floordiv``/``remainder``/``fmod`` or half op-math ``mul``/``div``.

    Matches eager ``DivMod.h`` and ``_Opmath.h`` rules across integer and
    floating-point operands, casting half-precision results back to fp16.

    Parameters
    ----------
    b : Builder
        Active MIL builder.
    op : TracedOp
        Traced binary operation.
    lucid_name : str
        Lucid operation name.
    mil_name : str
        MIL binary operation name.
    lhs : str
        Left-hand MIL value name.
    rhs : str
        Right-hand MIL value name.
    target : int or None
        Target MIL output data type.

    Returns
    -------
    EmitResult or None
        Emitted MIL operation and bindings if handled here, else ``None``.
    """
    if lucid_name in ("floordiv", "remainder", "fmod"):
        if target == _MIL_INT32:
            return _emit_int_divmod(b, op, lhs, rhs)
        if target in (_spec.FLOAT16, _spec.FLOAT32):
            return _emit_float_divmod(b, op, lhs, rhs, target)
    if target != _spec.FLOAT16 or lucid_name not in ("mul", "div"):
        return None
    is_pair = (b.dtype_of(lhs) == _spec.FLOAT16 and b.is_0d_float32(rhs)) or (
        b.dtype_of(rhs) == _spec.FLOAT16 and b.is_0d_float32(lhs)
    )
    if not is_pair:
        return None
    x_f32, y_f32 = b.agree_on_dtype([lhs, rhs], _spec.FLOAT32)
    res_f32 = b.emit(
        mil_name,
        [("x", x_f32), ("y", y_f32)],
        b.result_dims(op),
        dtype=_spec.FLOAT32,
    )
    return emit_cast(b, res_f32, "fp16")
