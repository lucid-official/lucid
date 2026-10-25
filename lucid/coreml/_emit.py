"""Lucid trace op → Core ML MIL operation.

One entry per Lucid op name.  An emitter receives a builder context (for
minting the constants MIL requires every scalar operand to be) and the
already-emitted value names of the op's inputs, and returns the MIL op
type plus its ``(parameter, value name)`` bindings.  The driver in
:mod:`lucid.coreml._build` supplies the output name and type from the
trace, so an emitter never has to reason about shapes.

Operand order is Lucid's, and it is not the order habit suggests.  The
table below was read off a trace rather than assumed:

    conv2d           (x, weight, bias)
    linear           (x, weight, bias)
    batch_norm_eval  (x, running_mean, running_var, weight, bias)

The third line is the dangerous one: the statistics come *before* the
affine parameters.  An emitter written from the usual
``(gamma, beta, mean, var)`` habit yields a model with correct shapes,
correct output ranks, and wrong values everywhere — nothing about it
fails loudly.
"""

import math
from typing import TYPE_CHECKING, Callable, NamedTuple, Protocol, Sequence

from lucid.coreml import _flex, _sized, _spec
from lucid.coreml._convert import (
    emit_cast,
    emit_opmath_binary,
    moved,
    scalar_like,
    scatter_along_axis,
    widened,
)
from lucid.coreml._linear import fused_linear_gelu, fused_linear_relu, linear, matmul

if TYPE_CHECKING:
    from lucid.coreml._build import Builder

_LN2 = 0.6931471805599453

# MIL's element type for a comparison's result.
_MIL_BOOL = 1
# ... and for a shape, an index or a count.
_MIL_INT32 = 23

__all__ = ["EMITTERS", "Bound", "Constant", "MultiOutput"]


class TracedValue(Protocol):
    """One result of a traced operation, as an emitter sees it.

    Three fields, because only three are ever read: the identity a
    flexible export keys its varying axes on, and two descriptions. ``reshape`` carries
    its target in the output's shape rather than in an attribute, and
    ``astype`` names its cast target by the output's dtype — everything
    else an emitter needs is an attribute or an operand.

    ``dtype`` is ``object`` rather than a Lucid dtype: the emitter turns
    it into MIL's spelling through its ``str``, so nothing here depends
    on which dtype class the tracer happens to hand over.
    """

    id: int
    shape: Sequence[int]
    dtype: object


class TracedOp(Protocol):
    """What an emitter is allowed to read off a traced operation.

    Stated as a protocol rather than taken as ``object`` so that reading
    a field the tracer does not carry is a type error here instead of an
    ``AttributeError`` halfway through an export.
    """

    name: str
    attrs: dict[str, object]
    outputs: list[TracedValue]


class Constant(NamedTuple):
    """A traced value that *is* a constant, with no operation to emit.

    ``zeros``, ``full`` and friends produce a value out of nothing, and
    the obvious translation is an ``identity`` reading a ``const``. Core
    ML does not take that in a stateful program: an identity on a
    constant, alongside a declared state, builds a package that compiles
    and then fails to plan, with an error naming neither. Binding the
    value straight to the constant avoids the operation entirely, and is
    one operation fewer everywhere else too.
    """

    name: str


class Bound(NamedTuple):
    """Traced results that are values the emitter already produced.

    ``Constant`` says one result is a constant; this says each result is
    whatever the emitter built for it, in order. An operation Lucid has
    and MIL does not sometimes becomes several MIL operations with
    several results — ``meshgrid`` is two independent broadcasts — and
    there is no single operation for the driver to append.
    """

    names: list[str]


# A parameter bound to one operand, or to several — ``concat`` is variadic.
Bindings = list[tuple[str, str | Sequence[str]]]
EmitResult = tuple[str, Bindings]

# Lucid dtype name -> the spelling MIL's ``cast`` expects.
_CAST_TARGETS = {
    "F32": "fp32",
    "F16": "fp16",
    "I64": "int32",
    "I32": "int32",
    "Bool": "bool",
}


class MultiOutput(NamedTuple):
    """One MIL operation that produces every one of a Lucid op's outputs.

    ``split`` is the case: the trace records one op with several
    results, and ``Operation.outputs`` is repeated to match.
    """

    mil_type: str
    bindings: Bindings


# Lucid op name -> emitter.
Emitter = Callable[..., EmitResult | MultiOutput | Constant | Bound]

# Lucid op name -> emitter.
EMITTERS: dict[str, Emitter] = {}


def _emitter(name: str) -> Callable[[Emitter], Emitter]:
    """Register one emitter under a Lucid op name.

    Stackable: an op that translates exactly like another shares the
    function rather than a second copy of it.

    Parameters
    ----------
    name : str
        Lucid op name as the tracer records it.

    Returns
    -------
    Callable
        Decorator returning the emitter unchanged.
    """

    def register(fn: Emitter) -> Emitter:
        EMITTERS[name] = fn
        return fn

    return register


def _attr(op: TracedOp, name: str) -> object:
    """An attribute the translation depends on, with no default.

    Defaults are how two of these went wrong silently: Lucid spells the
    leaky-ReLU slope ``slope`` and this read ``negative_slope``, quietly
    substituting 0.01 for 0.1 across every negative activation in YOLOv3;
    softmax is ``dim``, not ``axis``, and only happened to be right because
    the models seen so far normalise the last axis. A missing attribute is
    now a failure, not a guess.

    Parameters
    ----------
    op : trace operation
        Operation to read from.
    name : str
        Attribute name as the tracer records it.

    Returns
    -------
    object
        The attribute's value.

    Raises
    ------
    KeyError
        The attribute is absent — the tracer renamed it, or this emitter
        was written against a different operation.
    """
    attrs = op.attrs
    if name not in attrs:
        raise KeyError(
            f"lucid.coreml: op {op.name!r} has no attribute {name!r} "
            f"(it carries {sorted(attrs)}) — the emitter and the tracer disagree"
        )
    return attrs[name]


def _as_int(value: object) -> int:
    """A trace attribute read as an integer, or a named failure."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    raise TypeError(f"lucid.coreml: expected a number, got {type(value).__name__}")


def _as_float(value: object) -> float:
    """A trace attribute read as a float, or a named failure."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise TypeError(f"lucid.coreml: expected a number, got {type(value).__name__}")


def _as_seq(value: object) -> Sequence[object]:
    """A trace attribute read as a sequence, or a named failure."""
    if isinstance(value, (list, tuple)):
        return value
    raise TypeError(f"lucid.coreml: expected a sequence, got {type(value).__name__}")


def _flag(value: object, default: bool = False) -> bool:
    # Lucid records these as single-element int lists in the trace.
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return bool(value[0])
    return bool(value)


# ── elementwise unary: Lucid name -> the MIL op that is exactly it ───────────
#
# A table, because these differ only by a name and writing thirty of them
# out is thirty chances to pair the wrong two. Anything needing an
# attribute, a decomposition, or more than one operand is a real emitter
# below.
_UNARY_MIL = {
    "abs": "abs",
    "arccos": "acos",
    "arcsin": "asin",
    "arctan": "atan",
    "ceil": "ceil",
    "cos": "cos",
    "cosh": "cosh",
    "erf": "erf",
    "exp": "exp",
    "floor": "floor",
    "identity": "identity",
    "relu": "relu",
    "relu6": "relu6",
    "round": "round",
    "sigmoid": "sigmoid",
    "sign": "sign",
    "silu": "silu",
    "sin": "sin",
    "sinh": "sinh",
    "sqrt": "sqrt",
    "square": "square",
    "tan": "tan",
    "tanh": "tanh",
}
# MIL's ``relu`` and ``relu6`` refuse an int32 ``x``; Lucid runs both on integers.
_INTEGER_BOUNDS = {"relu": (0, None), "relu6": (0, 6)}


def _register_unary(lucid_name: str, mil_name: str) -> None:
    """Bind one Lucid unary op to the MIL op of the same meaning."""

    def emit(b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
        bounds = _INTEGER_BOUNDS.get(lucid_name)
        if bounds is not None and b.dtype_of(ins[0]) == _MIL_INT32:
            return _integer_clip(b, ins[0], *bounds)
        return mil_name, [("x", ins[0])]

    EMITTERS[lucid_name] = emit


for _lucid_name, _mil_name in _UNARY_MIL.items():
    _register_unary(_lucid_name, _mil_name)


# MIL's ``log`` and ``rsqrt`` take a mandatory ``epsilon`` that they add to
# the input; Lucid's do not. Passing zero keeps the two the same function
# rather than quietly shifting every value by coremltools' default.
_EPSILON_UNARY = {"log": "log", "rsqrt": "rsqrt", "reciprocal": "inverse"}


def _register_epsilon_unary(lucid_name: str, mil_name: str) -> None:
    """Bind a MIL unary that insists on an epsilon Lucid does not have."""

    def emit(b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
        return mil_name, [("x", ins[0]), ("epsilon", b.const_float(0.0))]

    EMITTERS[lucid_name] = emit


for _lucid_name, _mil_name in _EPSILON_UNARY.items():
    _register_epsilon_unary(_lucid_name, _mil_name)


@_emitter("neg")
def _neg(b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    # MIL has no negation op; multiplying by -1 is the whole of it.
    return "mul", [("x", ins[0]), ("y", scalar_like(b, ins[0], -1))]


def _integer_clip(b: Builder, value: str, low: int, high: int | None) -> EmitResult:
    """``value`` held to ``[low, high]``: MIL's ``maximum`` takes int32, ``clip`` not."""
    low, high = _spec.int32_clip_bounds(low, high)
    floor_binding: Bindings = [("x", value), ("y", b.const_int(low))]
    if high is None:
        return "maximum", floor_binding
    floored = b.emit("maximum", floor_binding, b.shape_of(value))
    return "minimum", [("x", floored), ("y", b.const_int(high))]


@_emitter("log2")
def _log2(b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    """``log2(x)`` as ``log(x) * 1/ln 2`` — MIL carries no base-2 log."""
    natural = b.emit(
        "log",
        [("x", ins[0]), ("epsilon", b.const_float(0.0))],
        b.shape_of(ins[0]),
    )
    return "mul", [("x", natural), ("y", b.const_float(1.0 / _LN2))]


# ── convolution and pooling, at every rank Lucid offers ──────────────────────
#
# MIL's ``conv``, ``max_pool`` and ``avg_pool`` are rank-agnostic: the
# length of the attribute lists is what says 1-D from 3-D. Lucid spells
# them apart, so the mapping is a family rather than three emitters that
# would differ only in how many elements they read.


def _ints(value: object) -> list[int]:
    """A trace attribute holding a per-spatial-axis list."""
    return [_as_int(v) for v in _as_seq(value)]


def _pad_pairs(padding: object) -> list[int]:
    """Lucid's one pad per axis as MIL's ``[before, after]`` per axis."""
    pairs: list[int] = []
    for pad in _ints(padding):
        pairs += [pad, pad]
    return pairs


def _register_conv(lucid_name: str) -> None:
    def emit(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
        attrs = op.attrs
        bindings: Bindings = [("x", ins[0]), ("weight", ins[1])]
        if len(ins) > 2:
            bindings.append(("bias", ins[2]))
        bindings += [
            ("strides", b.const_ints(_ints(attrs["stride"]))),
            ("pad_type", b.const_str("custom")),
            ("pad", b.const_ints(_pad_pairs(attrs["padding"]))),
            ("dilations", b.const_ints(_ints(attrs["dilation"]))),
            ("groups", b.const_int(_as_int(attrs["groups"]))),
        ]
        return "conv", bindings

    EMITTERS[lucid_name] = emit


for _name in ("conv1d", "conv2d", "conv3d"):
    _register_conv(_name)


def _register_pool(lucid_name: str, mil_name: str, counts_pad: bool) -> None:
    def emit(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
        attrs = op.attrs
        bindings: Bindings = [
            ("x", ins[0]),
            ("kernel_sizes", b.const_ints(_ints(attrs["kernel_size"]))),
            ("strides", b.const_ints(_ints(attrs["stride"]))),
            ("pad_type", b.const_str("custom")),
            ("pad", b.const_ints(_pad_pairs(attrs["padding"]))),
            ("ceil_mode", b.const_bool(_flag(attrs.get("ceil_mode")))),
        ]
        if counts_pad:
            bindings.append(
                (
                    "exclude_padding_from_average",
                    b.const_bool(not _flag(attrs.get("count_include_pad"), True)),
                )
            )
        return mil_name, bindings

    EMITTERS[lucid_name] = emit


for _name in ("max_pool1d", "max_pool2d", "max_pool3d"):
    _register_pool(_name, "max_pool", False)
for _name in ("avg_pool1d", "avg_pool2d", "avg_pool3d"):
    _register_pool(_name, "avg_pool", True)


# The fully connected layer and nn.FusedLinear's kernels: lucid/coreml/_linear.py.
EMITTERS["linear"] = linear
EMITTERS["_fused_linear_relu"] = fused_linear_relu
EMITTERS["_fused_linear_gelu"] = fused_linear_gelu


@_emitter("batch_norm_eval")
def _batch_norm_eval(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    x, mean, variance, gamma, beta = ins[0], ins[1], ins[2], ins[3], ins[4]
    return "batch_norm", [
        ("x", x),
        ("mean", mean),
        ("variance", variance),
        ("gamma", gamma),
        ("beta", beta),
        ("epsilon", b.const_float(_as_float(op.attrs["eps"]))),
    ]


# ── elementwise binary: same table, same reason as the unary one ─────────────
_BINARY_MIL = {
    "add": "add",
    "div": "real_div",
    "equal": "equal",
    "floordiv": "floor_div",
    "fmod": "mod",
    "greater": "greater",
    "greater_equal": "greater_equal",
    "less": "less",
    "less_equal": "less_equal",
    "maximum": "maximum",
    "minimum": "minimum",
    "mul": "mul",
    "not_equal": "not_equal",
    "pow": "pow",
    "remainder": "mod",
    "sub": "sub",
}


#: Binary operations whose result is boolean. Their operands have to
#: agree with each other rather than with the result — casting them to
#: ``bool`` to match the output would compare two truth values.
_COMPARISONS = frozenset(
    {"equal", "not_equal", "greater", "greater_equal", "less", "less_equal"}
)


def _runtime_base(b: Builder, op: TracedOp, base: str, exponent: str) -> str:
    """A constant base of ``pow``, made a value computed at run time.

    Core ML's optimizer rewrites a ``pow`` whose constant operand is 2 into a
    square — and does it for a constant *base* too, so ``2.0 ** x`` came back
    as ``x ** 2``: the right shape, the wrong numbers, every other base fine
    (macOS 27).  Multiplying the base by ones shaped like the exponent at run
    time leaves nothing for that rewrite to match; ``1 * base`` is exact.
    """
    shape = b.shape_of(exponent)
    dims = b.emit("shape", [("x", exponent)], [len(shape)], dtype=_MIL_INT32)
    ones = b.emit("fill", [("shape", dims), ("value", b.const_float(1.0))], shape)
    return b.emit("mul", [("x", ones), ("y", base)], b.result_dims(op))


def _register_binary(lucid_name: str, mil_name: str) -> None:
    """Bind one Lucid binary op to the MIL op of the same meaning."""

    def emit(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
        # Core ML does not promote, and says so only at parse time; Lucid
        # does, so a traced graph reaches here with mixed operands.
        target = None if lucid_name in _COMPARISONS else b.result_mil_dtype(op)
        if target is None and _MIL_BOOL in (b.dtype_of(ins[0]), b.dtype_of(ins[1])):
            # MIL compares numbers only; ``mask != False`` compares 0 and 1.
            target = _MIL_INT32
        opmath = emit_opmath_binary(b, op, lucid_name, mil_name, ins[0], ins[1], target)
        if opmath is not None:
            return opmath
        x, y = b.agree_on_dtype([ins[0], ins[1]], target)
        if lucid_name == "pow" and b.is_const(x) and not b.is_const(y):
            x = _runtime_base(b, op, x, y)
        return mil_name, [("x", x), ("y", y)]

    EMITTERS[lucid_name] = emit


for _lucid_name, _mil_name in _BINARY_MIL.items():
    _register_binary(_lucid_name, _mil_name)


@_emitter("reshape")
@_emitter("squeeze")
@_emitter("unsqueeze")
def _reshape(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    # The target is on the result — ``squeeze`` / ``unsqueeze`` record no
    # axes at all — and a Core ML tensor has no strides, so all three are a
    # reshape to it. The builder leaves a flexible export's open axes open.
    target = b.shape_operand(b.result_dims(op), op.outputs[0].id, ins[0])
    return "reshape", [("x", ins[0]), ("shape", target)]


@_emitter("dropout")
def _dropout(_b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    # An exported graph is an inference graph.  Lucid still records the op
    # under ``eval()``; a training-mode one is refused rather than
    # silently turned into an identity the caller did not ask for.
    if op.attrs.get("training"):
        raise NotImplementedError(
            "lucid.coreml: dropout was traced in training mode — call model.eval() "
            "before exporting, or the exported graph would differ from the traced one"
        )
    return "identity", [("x", ins[0])]


@_emitter("permute")
def _permute(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    perm = [_as_int(a) for a in _as_seq(op.attrs["permutation"])]
    return "transpose", [("x", ins[0]), ("perm", b.const_ints(perm))]


@_emitter("concatenate")
def _concatenate(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    axis = _as_int(_attr(op, "dim"))
    # ``values`` is variadic: one parameter bound to every input, each of
    # the result's type — MIL does not promote and Lucid did.
    return "concat", [
        ("values", b.agree_on_dtype(list(ins), b.result_mil_dtype(op))),
        ("axis", b.const_int(axis)),
        ("interleave", b.const_bool(False)),
    ]


# ── activations and arithmetic ───────────────────────────────────────


@_emitter("gelu_exact")
def _gelu_exact(b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    # Lucid's exact GELU is the erf form, which MIL calls EXACT; MIL's
    # default is the tanh approximation, a different function.
    return "gelu", [("x", ins[0]), ("mode", b.const_str("EXACT"))]


@_emitter("leaky_relu")
def _leaky_relu(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    alpha = _as_float(_attr(op, "slope"))
    return "leaky_relu", [("x", ins[0]), ("alpha", b.const_float(alpha))]


@_emitter("softmax")
def _softmax(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    axis = _as_int(_attr(op, "dim"))
    return "softmax", [("x", ins[0]), ("axis", b.const_int(axis))]


def _arg_reduce(b: Builder, op: TracedOp, ins: list[str], mil_type: str) -> EmitResult:
    """Index of the extreme value along one axis.

    Lucid allows ``dim=None``, meaning "over every element", which MIL's
    reduction cannot express — it always takes an axis. Flattening first
    turns that case into the ordinary one, and the flat index is what
    ``dim=None`` is defined to return anyway.
    """
    attrs = op.attrs
    raw_axis = attrs.get("axis", attrs.get("dim"))
    keep = bool(attrs.get("keepdim", False))
    source = ins[0]
    if raw_axis is None:
        source = b.emit(
            "reshape",
            [("x", source), ("shape", b.const_ints([-1]))],
            [-1],
        )
        axis = 0
    else:
        axis = _as_int(raw_axis)
    return mil_type, [
        ("x", widened(b, source)),
        ("axis", b.const_int(axis)),
        ("keep_dims", b.const_bool(keep)),
    ]


@_emitter("argmax")
def _argmax(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    return _arg_reduce(b, op, ins, "reduce_argmax")


@_emitter("argmin")
def _argmin(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    return _arg_reduce(b, op, ins, "reduce_argmin")


@_emitter("rand")
@_emitter("randn")
def _randn(_b: Builder, op: TracedOp, _ins: list[str]) -> EmitResult:
    """Refused, with the reason measured rather than assumed.

    A model that samples inside ``forward`` — a variational encoder
    drawing its latent — has no single answer to export, so the question
    is what Core ML would do with the draw. MIL has ``random_normal``,
    and emitting it produces a package that loads and runs. It also
    returns the same numbers on every call, from a fresh load, and from
    a second export made under a different seed: the operation's inputs
    are all constants, so the compiler folds it and the sample is drawn
    once, at build time.

    That is a deterministic model wearing a sampler's shape, and it is
    the failure this subsystem works hardest to avoid — no error, no
    fallback, plausible numbers, and the same ones forever. Better to
    refuse and say so than to hand over a variational encoder whose
    latent never moves.
    """
    from lucid.coreml._build import UnsupportedOp

    raise UnsupportedOp(
        op.name,
        "this model draws random numbers inside forward, and Core ML folds "
        "the draw at build time — the package would return one fixed sample "
        "on every call, for the life of the file. Export a deterministic "
        "path instead (a variational encoder's mean rather than a sample "
        "from it), or keep the sampling in the caller and export the network "
        "it wraps",
    )


@_emitter("one_hot")
def _one_hot(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """Indices to a basis vector each.

    The vector length comes from the traced result rather than from the
    attribute: Lucid's ``num_classes=-1`` means "as many as the largest
    index needs", which is a value the host worked out and the graph has
    no way to recompute.
    """
    size = b.result_dims(op)[-1]
    # Core ML requires the two fill values to carry the result's own
    # dtype, and refuses at parse time otherwise. Lucid's ``one_hot``
    # answers in the index type unless a caller casts it, so the choice
    # follows the trace rather than assuming floats.
    floating = b.result_is_float(op)
    on = b.const_float(1.0) if floating else b.const_int(1)
    off = b.const_float(0.0) if floating else b.const_int(0)
    # Lucid takes float indices that hold whole numbers; MIL takes int32
    # and refused the package at parse time.  A no-op when already int32.
    return "one_hot", [
        ("indices", b.narrow_to_int32(ins[0])),
        ("one_hot_vector_size", b.const_int(int(size))),
        ("axis", b.const_int(-1)),
        ("on_value", on),
        ("off_value", off),
    ]


@_emitter("cumsum")
def _cumsum(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """Running sum along one axis.

    MIL takes ``exclusive`` and ``reverse`` as well; Lucid's op is the
    inclusive forward direction, so both go out false rather than being
    left to whatever the runtime defaults to.
    """
    axis = _as_int(_attr(op, "axis"))
    return "cumsum", [
        ("x", ins[0]),
        ("axis", b.const_int(axis)),
        ("exclusive", b.const_bool(False)),
        ("reverse", b.const_bool(False)),
    ]


_emitter("matmul")(matmul)


@_emitter("layer_norm")
def _layer_norm(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """Normalised over the trailing axes the weight covers.

    Lucid's trace does not record which axes were normalised, but the
    affine weight's rank determines them: it broadcasts over exactly the
    normalised tail. MIL wants them as explicit (negative) axis indices.
    """
    x, weight = ins[0], ins[1]
    rank = len(b.shape_of(weight))
    axes = list(range(-rank, 0))
    bindings: Bindings = [
        ("x", x),
        ("axes", b.const_ints(axes)),
        ("gamma", weight),
    ]
    if len(ins) > 2:
        bindings.append(("beta", ins[2]))
    bindings.append(("epsilon", b.const_float(_as_float(op.attrs["eps"]))))
    return "layer_norm", bindings


@_emitter("mean")
def _mean(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    attrs = op.attrs
    axes = [_as_int(d) for d in _as_seq(attrs["dims"])]
    return "reduce_mean", [
        ("x", ins[0]),
        ("axes", b.const_ints(axes)),
        ("keep_dims", b.const_bool(bool(attrs.get("keepdim", False)))),
    ]


@_emitter("stack")
def _stack(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    axis = _as_int(_attr(op, "axis"))
    values = b.agree_on_dtype(list(ins), b.result_mil_dtype(op))
    return "stack", [("values", values), ("axis", b.const_int(axis))]


@_emitter("split")
@_emitter("split_at")
def _split_at(b: Builder, op: TracedOp, ins: list[str]) -> MultiOutput | Bound:
    """One MIL ``split`` producing every section the trace recorded.

    Lucid records the cut points; MIL wants the section sizes, which the
    output shapes already give — and taking them from the outputs keeps
    the two descriptions from disagreeing.
    """
    traced_axis = _as_int(op.attrs["axis"])
    sizes = [_sized.declared(o.shape)[traced_axis] for o in op.outputs]
    # The operand may be emitted a rank lower than the trace records —
    # see ``_unit_axis_rewrites`` — in which case every axis after the
    # dropped one has moved down by one. The sizes are read from the
    # trace's shapes and do not move.
    axis = traced_axis - 1 if b.leading_axis_dropped(ins[0]) else traced_axis
    if any(size < 0 for size in sizes):
        return Bound(_sized.split_at_run_time(b, op, ins[0], traced_axis, axis))
    return MultiOutput(
        "split",
        [
            ("x", ins[0]),
            ("split_sizes", b.const_ints(sizes)),
            ("axis", b.const_int(axis)),
        ],
    )


@_emitter("broadcast_to")
def _broadcast_to(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """Expressed as ``tile``, which MIL has, since ``broadcast_to`` it does not.

    A broadcast repeats each size-1 axis; the repetition counts come from
    dividing the result shape by the operand's, after leading axes match
    the ranks. An axis that follows the input — a class token spread over
    the batch — repeats as often as the input is long there, at run time.
    """
    src = b.shape_of(ins[0]).declared()
    out = b.result_dims(op).declared()
    # ``tile`` keeps its operand's type and MIL checks that it does.
    (value,) = b.agree_on_dtype([ins[0]], b.result_mil_dtype(op))
    if len(src) < len(out):
        lead = list(range(len(out) - len(src)))
        src = [1] * len(lead) + src
        value = b.emit("expand_dims", [("x", value), ("axes", b.const_ints(lead))], src)
    operand = _flex.Dims(src, op.name)
    reps = [
        (-1 if size == 1 else 1) if o < 0 else o // operand[axis]
        for axis, (size, o) in enumerate(zip(src, out, strict=True))
    ]
    return "tile", [("x", value), ("reps", b.sizes_operand(reps, op.outputs[0].id))]


@_emitter("scaled_dot_product_attention")
def _sdpa(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """Decomposed rather than mapped to ``ios18.scaled_dot_product_attention``.

    The fused operation exists only in a newer opset than the one this
    writer emits, and the decomposition is what Core ML's own converter
    produces for older targets: scores, scale, softmax, weighted sum.
    """
    attrs = op.attrs
    if attrs.get("has_mask") and len(ins) < 4:
        raise NotImplementedError(
            "lucid.coreml: scaled_dot_product_attention declares a mask but the "
            "trace carries no mask operand"
        )
    query, key, value = ins[0], ins[1], ins[2]
    scores_shape = b.shape_of(query).declared()[:-1] + [b.shape_of(key).declared()[-2]]

    scores = b.emit(
        "matmul",
        [
            ("x", query),
            ("y", key),
            ("transpose_x", b.const_bool(False)),
            ("transpose_y", b.const_bool(True)),
        ],
        scores_shape,
    )
    scaled = b.emit(
        "mul",
        [("x", scores), ("y", b.const_float(_as_float(attrs["scale"])))],
        scores_shape,
    )
    if attrs.get("has_mask"):
        # The mask is already an additive float tensor in the trace — it
        # broadcasts over the head axis — so it goes straight onto the
        # scores rather than being rebuilt here.
        scaled = b.emit("add", [("x", scaled), ("y", ins[3])], scores_shape)
    if attrs.get("is_causal"):
        # A causal mask is a constant here: the trace fixed the sequence
        # length, so the forbidden positions are known.  ``-1e4`` rather
        # than ``-inf`` because the softmax may run in float16, where
        # ``inf - inf`` is a NaN rather than a zero weight.
        import lucid

        rows, cols = _flex.Dims(scores_shape, op.name)[-2:]
        offset = cols - rows
        mask = lucid.tensor(
            [
                [0.0 if j <= i + offset else -1.0e4 for j in range(cols)]
                for i in range(rows)
            ]
        )
        scaled = b.emit(
            "add", [("x", scaled), ("y", b.const_from_tensor(mask))], scores_shape
        )
    weights = b.emit(
        "softmax", [("x", scaled), ("axis", b.const_int(-1))], scores_shape
    )
    return "matmul", [
        ("x", weights),
        ("y", value),
        ("transpose_x", b.const_bool(False)),
        ("transpose_y", b.const_bool(False)),
    ]


# ── values produced out of nothing ───────────────────────────────────
#
# These take no operands: the trace already fixed their contents, so each
# becomes a constant that an ``identity`` names as the op's output.


def _filled(b: Builder, op: TracedOp, value: float) -> Constant | EmitResult:
    """One value, shaped and typed like the result: a constant, or a ``fill``
    sized at run time when the shape follows the input (``zeros(x.shape[0])``)."""
    import lucid

    dims = b.result_dims(op)
    shape = dims.declared()
    kind = b.result_mil_dtype(op)
    whole = kind in (_MIL_INT32, _MIL_BOOL)
    made: Constant | EmitResult
    if dims.open:
        fill = b.const_int(int(value)) if whole else b.const_float(value)
        sizes = b.sizes_operand(shape, op.outputs[0].id)
        made = "fill", [("shape", sizes), ("value", fill)]
    elif whole:
        made = Constant(b.const_ints_shaped([int(value)] * math.prod(shape), shape))
    else:
        filled = lucid.zeros(*shape) + value if shape else lucid.tensor(value)
        made = Constant(b.const_from_tensor(filled))
    if kind != _MIL_BOOL:
        return made
    held = made.name if isinstance(made, Constant) else b.emit(*made, shape, _MIL_INT32)
    return emit_cast(b, held, "bool")


@_emitter("zeros")
def _zeros(b: Builder, op: TracedOp, _ins: list[str]) -> Constant | EmitResult:
    return _filled(b, op, 0.0)


@_emitter("full")
def _full(b: Builder, op: TracedOp, _ins: list[str]) -> Constant | EmitResult:
    return _filled(b, op, _as_float(_attr(op, "fill_value")))


@_emitter("arange")
def _arange(b: Builder, op: TracedOp, _ins: list[str]) -> Constant:
    """The traced sequence, as a constant of the type the trace gave it.

    ``const_from_tensor`` writes the body's float type, which was only
    right while ``arange`` defaulted to floats. ``lucid.arange(n)`` is
    int64 now, and declared float it would reach an index or a
    comparison as the wrong type — in an fp16 program, as a rounded one.
    An integer sequence goes out inline as int32, the integer width this
    opset's operations take.
    """
    import lucid

    attrs = op.attrs
    start = _as_float(attrs.get("start", 0.0))
    step = _as_float(attrs.get("step", 1.0))
    count = b.result_dims(op)[0]
    integer = _CAST_TARGETS.get(str(op.outputs[0].dtype).split(".")[-1]) == "int32"
    if integer and count > 0:
        # The attributes are doubles, which hold every int32 exactly.
        first, stride = int(start), int(step)
        last = first + stride * (count - 1)
        if not -(2**31) <= min(first, last) <= max(first, last) < 2**31:
            from lucid.coreml._build import UnsupportedOp

            raise UnsupportedOp(
                "arange",
                f"its values run from {first} to {last}, and Core ML holds "
                "integers as int32, so the package would wrap them",
            )
        values = [first + stride * i for i in range(count)]
        return Constant(b.const_ints_shaped(values, [count]))
    # An empty sequence goes this way too, and is refused there with the
    # reason.
    floats = lucid.tensor([start + step * i for i in range(count)])
    return Constant(b.const_from_tensor(floats))


# ── indexing, casting, reductions ────────────────────────────────────


@_emitter("embedding")
def _embedding(b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    """A row lookup, which MIL spells ``gather`` along the table's first axis.

    ``padding_idx`` is not honoured here: it only matters to the backward
    pass, and an exported graph has none.
    """
    table, indices = ins[0], ins[1]
    # This opset's ``gather`` takes int32 indices and smaller, and refuses
    # int64 at parse time rather than converting. A token id that arrives
    # as a model input is already narrowed on the way in, but one the
    # graph computes — DiT builds its timestep embedding's indices — is
    # not, so the narrowing happens here. Casting int32 to int32 is a
    # no-op the compiler drops.
    indices = b.narrow_to_int32(indices)
    return "gather", [
        ("x", table),
        ("indices", indices),
        ("axis", b.const_int(0)),
        # Required by this opset.  False: the trace's indices are already
        # in range, and checking them would cost a comparison per lookup.
        ("validate_indices", b.const_bool(False)),
    ]


@_emitter("contiguous")
def _contiguous(_b: Builder, _op: TracedOp, ins: list[str]) -> Bound:
    """Nothing, because a MIL value has no layout to make contiguous.

    Lucid's ``contiguous`` copies a strided tensor into packed storage.
    A MIL program has no strides to be packed against — a value is its
    shape and its contents — so the operation has no work to describe,
    and emitting ``identity`` for it asks Core ML to schedule a copy that
    means nothing.

    That is not free. Swin calls it after every permute, which put 109
    identity operations in the program, all of them scheduled on the CPU
    — and an operation on the CPU between two on the Neural Engine costs
    a round trip whatever the operation is.
    """
    return Bound([ins[0]])


@_emitter("detach")
def _detach(_b: Builder, _op: TracedOp, ins: list[str]) -> Bound:
    """The value itself: an exported graph has no gradient to stop.

    ``detach`` is traced (a straight-through estimator's ``x + (q - x).detach()``
    otherwise reached the compile path as a stale constant), so it reaches
    here too — as nothing, like ``contiguous``.
    """
    return Bound([ins[0]])


@_emitter("astype")
def _astype(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """A cast to the type the result is declared as.

    The trace says what the model asked for; the driver says what the
    program will hold. In a float16 export those differ for a cast to
    float32 — float intermediates follow the body — and Core ML refuses
    a ``cast`` whose named type is not its output's. The declared type
    wins, because it is the one the rest of the program agrees with.
    """
    target = b.result_cast_spelling(op)
    if target is None:
        name = str(op.outputs[0].dtype).split(".")[-1]
        target = _CAST_TARGETS.get(name)
    if target is None:
        name = str(op.outputs[0].dtype).split(".")[-1]
        raise NotImplementedError(f"lucid.coreml: no Core ML cast target for {name}")
    return emit_cast(b, ins[0], target)


@_emitter("max")
def _max(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    return _reduce(b, op, ins, "reduce_max")


@_emitter("min")
def _min(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    return _reduce(b, op, ins, "reduce_min")


def _reduce(b: Builder, op: TracedOp, ins: list[str], mil_type: str) -> EmitResult:
    attrs = op.attrs
    return mil_type, [
        ("x", ins[0]),
        ("axes", b.const_ints([_as_int(d) for d in _as_seq(attrs["dims"])])),
        ("keep_dims", b.const_bool(bool(attrs.get("keepdim", False)))),
    ]


@_emitter("gelu")
def _gelu(b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    # Lucid's plain ``gelu`` is the tanh approximation; ``gelu_exact`` is
    # the erf form and maps to MIL's EXACT mode.
    return "gelu", [("x", ins[0]), ("mode", b.const_str("TANH_APPROXIMATION"))]


# ── resampling and transposed convolution ────────────────────────────


@_emitter("interpolate_nearest_2d")
@_emitter("interpolate_bilinear")
def _resample(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    return _sized.resample(b, op, ins[0])


@_emitter("grid_sample")
def _grid_sample(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """MIL calls it ``resample``, and it is the same operation.

    Same operand order, same ``(x, y)`` coordinate order, same
    normalised ``[-1, 1]`` range. ``F.grid_sample`` folds ``reflection``
    padding into the grid in Python and hands the kernel ``border``
    (``padding_mode == 1``); only ``mode in (0, 1)`` and
    ``padding_mode in (0, 1)`` reach this op.
    """
    attrs = op.attrs
    mode, pad = _as_int(attrs.get("mode", 0)), _as_int(attrs.get("padding_mode", 0))
    if mode not in (0, 1) or pad not in (0, 1):
        raise ValueError(
            f"lucid.coreml: grid_sample mode={mode}, padding_mode={pad} out of {{0, 1}}"
        )
    return "resample", [
        ("x", ins[0]),
        ("coordinates", ins[1]),
        ("sampling_mode", b.const_str("nearest" if mode == 1 else "bilinear")),
        ("padding_mode", b.const_str("constant" if pad == 0 else "border")),
        ("padding_value", b.const_float32(0.0)),
        ("coordinates_mode", b.const_str("normalized_minus_one_to_one")),
        ("align_corners", b.const_bool(bool(attrs.get("align_corners", False)))),
    ]


# Same rank-agnostic story as ``conv``: every rank reaches one MIL op.
@_emitter("conv_transpose3d")
@_emitter("conv_transpose1d")
@_emitter("conv_transpose2d")
def _conv_transpose2d(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """The output shape is passed explicitly rather than derived.

    A transposed convolution's result size is ambiguous — ``output_padding``
    exists precisely because several inputs map to the same output — and
    the trace already recorded which one this is.
    """
    attrs = op.attrs
    strides = _ints(attrs["stride"])
    bindings: Bindings = [("x", ins[0]), ("weight", ins[1])]
    if len(ins) > 2:
        bindings.append(("bias", ins[2]))
    bindings += [
        ("strides", b.const_ints(strides)),
        ("pad_type", b.const_str("custom")),
        ("pad", b.const_ints(_pad_pairs(attrs["padding"]))),
        # ``dilation`` and ``groups`` are recorded by the trace, but the
        # defaults are kept: a graph captured before they were traced
        # still loads, and MIL's weight layout for ``conv_transpose`` is
        # ``(C_in, C_out / groups, *K)`` — the same one Lucid stores, so
        # grouping needs no relayout here.
        ("dilations", b.const_ints(_ints(attrs.get("dilation", [1] * len(strides))))),
        ("groups", b.const_int(_as_int(attrs.get("groups", 1)))),
        # ``output_shape`` disambiguates a transposed convolution's result
        # — several inputs map to the same output, which is what
        # ``output_padding`` exists for — and the trace recorded which one
        # this is.  Under a flexible export that number is no longer
        # knowable, so it is left out and MIL infers it from the input,
        # stride and padding.  Baking the traced one instead gives a
        # decoder whose skip connections stop lining up at every other
        # size, which the compiler reports as a concat of mismatched
        # tensors far from here.
        *(
            []
            if b.result_dims(op).open
            else [("output_shape", b.const_ints(list(b.result_dims(op))))]
        ),
    ]
    return "conv_transpose", bindings


# ── gather / scatter along an axis, comparison, selection ────────────


@_emitter("gather")
def _gather(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    # The result has the *indices'* shape, not the source's, which is the
    # along-axis form rather than the row-lookup ``gather`` uses.
    bindings: Bindings = [
        ("x", ins[0]),
        ("indices", ins[1]),
        ("axis", b.const_int(_as_int(_attr(op, "axis")))),
        # Added to this operation in iOS17, which is the opset the writer
        # declares.  Emitting the iOS15 shape of the payload produced a
        # package that failed to parse — "Required param
        # 'validate_indices' is missing" — and took every model with a
        # gather down with it.  False for the same reason ``embedding``
        # gives: the trace's indices are already in range.
        ("validate_indices", b.const_bool(False)),
    ]
    return moved(b, op, "gather_along_axis", bindings, ("x",))


@_emitter("pow_scalar")
def _pow_scalar(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """``x ** c`` with the exponent on the trace rather than in a tensor.

    Reached through LP pooling and the p-norms, which raise by a constant
    the caller chose; MIL's ``pow`` takes both operands as tensors, so
    the constant becomes one.
    """
    return "pow", [("x", ins[0]), ("y", b.const_float32(_as_float(_attr(op, "exp"))))]


@_emitter("unfold_dim")
def _unfold_dim(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """A sliding window along one axis, which MIL has as ``sliding_windows``.

    The two disagree about where the window axis goes: MIL inserts it
    directly after the axis it slid along, and Lucid appends it last.
    One transpose reconciles them, and it is the whole difference —
    ``sliding_windows`` does the sliding itself, so this does not become
    the slice-and-concatenate chain the MPSGraph emitter needs.
    """
    shape = b.shape_of(ins[0])
    rank = len(shape)
    axis = _as_int(_attr(op, "dim"))
    if axis < 0:
        axis += rank
    size = _as_int(_attr(op, "size"))
    step = _as_int(_attr(op, "step"))

    windowed = shape.declared()
    windowed[axis] = (shape[axis] - size) // step + 1
    windowed.insert(axis + 1, size)
    slid = b.emit(
        "sliding_windows",
        [
            ("x", ins[0]),
            ("axis", b.const_int(axis)),
            ("size", b.const_int(size)),
            ("stride", b.const_int(step)),
        ],
        windowed,
    )
    # Move the window axis from just after the slid one to the end.
    perm = [i for i in range(rank + 1) if i != axis + 1] + [axis + 1]
    return "transpose", [("x", slid), ("perm", b.const_ints(perm))]


@_emitter("scatter_add")
def _scatter_add(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    return _scatter_along_axis(b, op, ins, "add")


@_emitter("scatter_set")
def _scatter_set(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """The overwrite — ``x[key] = v``, ``scatter`` and ``index_copy``.

    MIL's ``update`` mode, the twin of ``scatter_add`` above.  A position
    named twice keeps one of its values, which one unspecified, as on Metal.
    """
    return _scatter_along_axis(b, op, ins, "update")


def _scatter_along_axis(
    b: Builder, op: TracedOp, ins: list[str], mode: str
) -> EmitResult:
    """``scatter_along_axis`` in ``mode``, for every result type Lucid has.

    A bool scatter (``mask[0] = True``) runs on int32 copies of 1 and 0
    (:func:`moved`) and comes back as "nonzero": an ``add`` of two trues
    is true, as eager's is.
    """
    return scatter_along_axis(b, op, ins, _as_int(_attr(op, "dim")), mode)


@_emitter("where")
def _where(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    # Both branches carry the result's type, which MIL checks and Lucid
    # promoted to: ``where(outside, full(...), index)`` meets an int index.
    a, other = b.agree_on_dtype([ins[1], ins[2]], b.result_mil_dtype(op))
    return "select", [("cond", ins[0]), ("a", a), ("b", other)]


@_emitter("roll")
def _roll(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """Cut and swap, once per axis — MIL has no ``roll``.

    ``roll(x, s)[i] == x[(i - s) mod n]``, so the tail of length ``s``
    moves to the front. Expressed as ``split`` plus a swapped ``concat``:
    both are already verified here, where a ``slice_by_index`` would add
    three mask vectors as new ways to be subtly wrong.
    """
    attrs = op.attrs
    axes = [_as_int(a) for a in _as_seq(attrs["axes"])]
    shifts = [_as_int(s) for s in _as_seq(attrs["shifts"])]
    value = ins[0]
    shape = b.shape_of(value)

    for step, (axis, shift) in enumerate(zip(axes, shifts, strict=True)):
        size = shape[axis]
        head = (size - shift % size) % size
        if head == 0:
            continue
        first, second = shape.declared(), shape.declared()
        first[axis], second[axis] = head, size - head
        pieces = b.emit_multi(
            "split",
            [
                ("x", value),
                ("split_sizes", b.const_ints([head, size - head])),
                ("axis", b.const_int(axis)),
            ],
            [first, second],
        )
        bindings: Bindings = [
            ("values", [pieces[1], pieces[0]]),
            ("axis", b.const_int(axis)),
            ("interleave", b.const_bool(False)),
        ]
        if step == len(axes) - 1:
            return "concat", bindings
        value = b.emit("concat", bindings, shape)
    return "identity", [("x", value)]


# ── activations MIL spells differently, or not at all ────────────────────────
#
# The SELU constants are the paper's (Klambauer 2017): the fixed point of
# the variance map, not anything Core ML can be asked for.
_SELU_ALPHA = 1.6732632423543772
_SELU_SCALE = 1.0507009873554805


@_emitter("elu")
def _elu(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    return "elu", [
        ("x", ins[0]),
        ("alpha", b.const_float(_as_float(_attr(op, "alpha")))),
    ]


@_emitter("selu")
def _selu(b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    """MIL's ``elu`` scaled — there is no ``selu`` in the opset."""
    shaped = b.emit(
        "elu",
        [("x", ins[0]), ("alpha", b.const_float(_SELU_ALPHA))],
        b.shape_of(ins[0]),
    )
    return "mul", [("x", shaped), ("y", b.const_float(_SELU_SCALE))]


@_emitter("softplus")
def _softplus(_b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    return "softplus", [("x", ins[0])]


@_emitter("mish")
def _mish(b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    """``x * tanh(softplus(x))`` — there is no ``mish`` in the opset."""
    shape = b.shape_of(ins[0])
    softened = b.emit("softplus", [("x", ins[0])], shape)
    gate = b.emit("tanh", [("x", softened)], shape)
    return "mul", [("x", ins[0]), ("y", gate)]


@_emitter("log_softmax")
def _log_softmax(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """``x - logsumexp(x)`` — there is no ``log_softmax`` in the opset.

    Not ``log(softmax(x))``: that underflows to ``-inf`` wherever a
    probability is small enough to round to zero, which is exactly where
    a log-softmax is being asked for.
    """
    axis = _as_int(_attr(op, "dim"))
    shape = b.shape_of(ins[0])
    kept = shape.declared()
    kept[axis if axis >= 0 else axis + len(kept)] = 1
    total = b.emit(
        "reduce_log_sum_exp",
        [
            ("x", ins[0]),
            ("axes", b.const_ints([axis])),
            ("keep_dims", b.const_bool(True)),
        ],
        kept,
    )
    return "sub", [("x", ins[0]), ("y", total)]


@_emitter("clip")
def _clip(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    # An int clamp (a histogram's bucket index) truncates bounds, as the kernel.
    if b.dtype_of(ins[0]) == _MIL_INT32:
        low, high = (math.trunc(_as_float(_attr(op, k))) for k in ("min", "max"))
        return _integer_clip(b, ins[0], low, high)
    return "clip", [
        ("x", ins[0]),
        ("alpha", b.const_float(_as_float(_attr(op, "min")))),
        ("beta", b.const_float(_as_float(_attr(op, "max")))),
    ]


@_emitter("ones")
def _ones(b: Builder, op: TracedOp, _ins: list[str]) -> Constant | EmitResult:
    return _filled(b, op, 1.0)


# ── reductions ───────────────────────────────────────────────────────────────


def _register_reduce(lucid_name: str, mil_name: str) -> None:
    """Bind a Lucid reduction; ``dims``/``keepdim`` are MIL's ``axes``/``keep_dims``."""

    def emit(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
        dims = [_as_int(d) for d in _as_seq(_attr(op, "dims"))]
        return mil_name, [
            ("x", ins[0]),
            ("axes", b.const_ints(dims)),
            ("keep_dims", b.const_bool(bool(_attr(op, "keepdim")))),
        ]

    EMITTERS[lucid_name] = emit


for _lucid_name, _mil_name in {"sum": "reduce_sum", "prod": "reduce_prod"}.items():
    _register_reduce(_lucid_name, _mil_name)


def _reduce_every_axis(b: Builder, value: str, mil_name: str) -> EmitResult:
    """``mil_name`` over every axis of ``value``, to a scalar."""
    axes = list(range(len(b.shape_of(value))))
    return mil_name, [
        ("x", value),
        ("axes", b.const_ints(axes)),
        ("keep_dims", b.const_bool(False)),
    ]


@_emitter("isnan")
def _isnan(_b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    """NaN is the one value unequal to itself."""
    return "not_equal", [("x", ins[0]), ("y", ins[0])]


@_emitter("isinf")
def _isinf(b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    magnitude = b.emit("abs", [("x", ins[0])], b.shape_of(ins[0]))
    return "equal", [("x", magnitude), ("y", b.const_float(math.inf))]


def _truth_reduce(b: Builder, ins: list[str], mil_name: str) -> EmitResult:
    """``all`` / ``any`` over the whole tensor: min / max of the truth values.

    MIL reduces numbers, not booleans, so the truth of each element is
    counted as an int32 first and the reduced count compared back to zero.
    """
    shape = b.shape_of(ins[0])
    truth = ins[0]
    if b.dtype_of(truth) != _MIL_BOOL:
        truth = b.emit(
            "not_equal",
            [("x", truth), ("y", scalar_like(b, truth, 0))],
            shape,
            dtype=_MIL_BOOL,
        )
    counted = b.emit(*emit_cast(b, truth, "int32"), shape, dtype=_MIL_INT32)
    name, bindings = _reduce_every_axis(b, counted, mil_name)
    reduced = b.emit(name, bindings, [], dtype=_MIL_INT32)
    return "not_equal", [("x", reduced), ("y", b.const_int(0))]


@_emitter("all")
def _all(b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    return _truth_reduce(b, ins, "reduce_min")


@_emitter("any")
def _any(b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    return _truth_reduce(b, ins, "reduce_max")


@_emitter("var")
def _var(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """Mean squared deviation from the mean, over ``dims``.

    MIL has no variance.  The divisor is the count (``unbiased`` false is
    what the engine records; a Bessel correction, when asked for, is applied
    outside the op) — or the count less one when the trace says so.
    """
    x = ins[0]
    shape = b.shape_of(x)
    rank = len(shape)
    dims = [_as_int(d) % rank for d in _as_seq(_attr(op, "dims"))]
    keep = bool(_attr(op, "keepdim"))
    kept = [1 if i in dims else s for i, s in enumerate(shape.declared())]
    axes = b.const_ints(dims)
    mean = b.emit(
        "reduce_mean",
        [("x", x), ("axes", axes), ("keep_dims", b.const_bool(True))],
        kept,
    )
    deviation = b.emit("sub", [("x", x), ("y", mean)], shape)
    squared = b.emit("mul", [("x", deviation), ("y", deviation)], shape)
    bindings: Bindings = [
        ("x", squared),
        ("axes", b.const_ints(dims)),
        ("keep_dims", b.const_bool(keep)),
    ]
    if not bool(op.attrs.get("unbiased", False)):
        return "reduce_mean", bindings
    count = math.prod(shape[i] for i in dims)
    biased = b.emit("reduce_mean", bindings, b.result_dims(op))
    return "mul", [("x", biased), ("y", b.const_float(count / max(count - 1, 1)))]


def _dot_product(b: Builder, left: str, right: str) -> EmitResult:
    product = b.emit("mul", [("x", left), ("y", right)], b.shape_of(left))
    return "reduce_sum", [
        ("x", product),
        ("axes", b.const_ints([0])),
        ("keep_dims", b.const_bool(False)),
    ]


@_emitter("dot")
def _dot(b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    return _dot_product(b, ins[0], ins[1])


def _as_matrix(b: Builder, value: str, rows: int, cols: int) -> str:
    if b.shape_of(value) == [rows, cols]:
        return value
    return b.emit(
        "reshape", [("x", value), ("shape", b.const_ints([rows, cols]))], [rows, cols]
    )


@_emitter("inner")
def _inner(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """Contraction of the last axes: ``a (..., n) · b (..., n)``."""
    sa, sb = b.shape_of(ins[0]), b.shape_of(ins[1])
    if len(sa) == 1 and len(sb) == 1:
        return _dot_product(b, ins[0], ins[1])
    n = sa[-1]
    left = _as_matrix(b, ins[0], math.prod(sa[:-1]), n)
    right = _as_matrix(b, ins[1], math.prod(sb[:-1]), n)
    rows, cols = math.prod(sa[:-1]), math.prod(sb[:-1])
    product = b.emit(
        "matmul",
        [
            ("x", left),
            ("y", right),
            ("transpose_x", b.const_bool(False)),
            ("transpose_y", b.const_bool(True)),
        ],
        [rows, cols],
    )
    return "reshape", [("x", product), ("shape", b.const_ints(list(b.result_dims(op))))]


@_emitter("outer")
def _outer(b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    n, m = b.shape_of(ins[0])[0], b.shape_of(ins[1])[0]
    column = _as_matrix(b, ins[0], n, 1)
    row = _as_matrix(b, ins[1], 1, m)
    return "mul", [("x", column), ("y", row)]


@_emitter("tensordot")
def _tensordot(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """Both operands laid out as matrices, one matmul, the free axes restored."""
    sa, sb = b.shape_of(ins[0]), b.shape_of(ins[1])
    axes_a = [_as_int(a) % len(sa) for a in _as_seq(_attr(op, "axes_a"))]
    axes_b = [_as_int(a) % len(sb) for a in _as_seq(_attr(op, "axes_b"))]
    free_a = [i for i in range(len(sa)) if i not in axes_a]
    free_b = [i for i in range(len(sb)) if i not in axes_b]

    def arranged(
        value: str, shape: Sequence[int], perm: list[int], rows: int, cols: int
    ) -> str:
        if perm != list(range(len(shape))):
            value = b.emit(
                "transpose",
                [("x", value), ("perm", b.const_ints(perm))],
                [shape[i] for i in perm],
            )
        return _as_matrix(b, value, rows, cols)

    m = math.prod(sa[i] for i in free_a)
    k = math.prod(sa[i] for i in axes_a)
    n = math.prod(sb[i] for i in free_b)
    left = arranged(ins[0], sa, free_a + axes_a, m, k)
    right = arranged(ins[1], sb, axes_b + free_b, k, n)
    product = b.emit(
        "matmul",
        [
            ("x", left),
            ("y", right),
            ("transpose_x", b.const_bool(False)),
            ("transpose_y", b.const_bool(False)),
        ],
        [m, n],
    )
    out = [sa[i] for i in free_a] + [sb[i] for i in free_b]
    return "reshape", [("x", product), ("shape", b.const_ints(out))]


def _loss_reduction(b: Builder, op: TracedOp, loss: str) -> EmitResult:
    """Lucid's loss ``reduction``: 0 none, 1 mean, 2 sum (``nn/Loss.h``)."""
    reduction = _as_int(_attr(op, "reduction"))
    if reduction == 0:
        return "identity", [("x", loss)]
    return _reduce_every_axis(
        b, loss, "reduce_mean" if reduction == 1 else "reduce_sum"
    )


@_emitter("mse_loss")
def _mse_loss(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    shape = b.shape_of(ins[0])
    diff = b.emit("sub", [("x", ins[0]), ("y", ins[1])], shape)
    squared = b.emit("mul", [("x", diff), ("y", diff)], shape)
    return _loss_reduction(b, op, squared)


@_emitter("huber_loss")
def _huber_loss(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """Quadratic within ``delta`` of the target, linear beyond it."""
    delta = _as_float(_attr(op, "delta"))
    shape = b.shape_of(ins[0])
    diff = b.emit("sub", [("x", ins[0]), ("y", ins[1])], shape)
    size = b.emit("abs", [("x", diff)], shape)
    squared = b.emit("mul", [("x", diff), ("y", diff)], shape)
    quadratic = b.emit("mul", [("x", squared), ("y", b.const_float(0.5))], shape)
    beyond = b.emit("sub", [("x", size), ("y", b.const_float(0.5 * delta))], shape)
    linear = b.emit("mul", [("x", beyond), ("y", b.const_float(delta))], shape)
    near = b.emit(
        "less", [("x", size), ("y", b.const_float(delta))], shape, dtype=_MIL_BOOL
    )
    loss = b.emit("select", [("cond", near), ("a", quadratic), ("b", linear)], shape)
    return _loss_reduction(b, op, loss)


@_emitter("trace")
def _trace(b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    """The diagonal read from the flattened matrix, every ``cols + 1``-th value."""
    rows, cols = b.shape_of(ins[0])
    flat = b.emit(
        "reshape",
        [("x", ins[0]), ("shape", b.const_ints([rows * cols]))],
        [rows * cols],
    )
    steps = [i * (cols + 1) for i in range(min(rows, cols))]
    diagonal = b.emit(
        "gather",
        [
            ("x", flat),
            ("indices", b.const_ints(steps)),
            ("axis", b.const_int(0)),
            ("validate_indices", b.const_bool(False)),
        ],
        [len(steps)],
    )
    return "reduce_sum", [
        ("x", diagonal),
        ("axes", b.const_ints([0])),
        ("keep_dims", b.const_bool(False)),
    ]


def _argsort_operands(b: Builder, op: TracedOp, x: str) -> Bindings:
    """Ascending ``argsort`` along the op's axis. MIL's refuses bool, so a
    mask is ordered as int32 1 / 0 (:func:`widened`): false first, as eager."""
    axis = _as_int(_attr(op, "axis")) % len(b.shape_of(x))
    up = b.const_bool(True)
    return [("x", widened(b, x)), ("axis", b.const_int(axis)), ("ascending", up)]


@_emitter("argsort")
def _argsort(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    return "argsort", _argsort_operands(b, op, ins[0])


@_emitter("sort")
def _sort(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    sorting = _argsort_operands(b, op, ins[0])
    order = b.emit("argsort", sorting, b.shape_of(ins[0]), dtype=_MIL_INT32)
    # The values gathered from the (possibly widened) operand; ``moved``
    # turns an int32 result back into bool.
    picked = [*sorting[:2], ("indices", order)]
    picked.append(("validate_indices", b.const_bool(False)))
    return moved(b, op, "gather_along_axis", picked, ())


@_emitter("topk")
def _topk(b: Builder, op: TracedOp, ins: list[str]) -> MultiOutput:
    """The ``k`` largest along ``axis``, values then indices, as MIL's own."""
    axis = _as_int(_attr(op, "axis")) % len(b.shape_of(ins[0]))
    return MultiOutput(
        "topk",
        [
            ("x", ins[0]),
            ("k", b.const_int(_as_int(_attr(op, "k")))),
            ("axis", b.const_int(axis)),
            ("ascending", b.const_bool(False)),
        ],
    )


@_emitter("unbind")
def _unbind(b: Builder, op: TracedOp, ins: list[str]) -> Bound:
    """One ``gather`` per slice: a scalar index drops the axis, as unbind does."""
    axis = _as_int(_attr(op, "axis")) % len(b.shape_of(ins[0]))
    source = widened(b, ins[0])
    pieces = []
    for i in range(len(op.outputs)):
        piece = b.emit(
            "gather",
            [
                ("x", source),
                ("indices", b.const_int(i)),
                ("axis", b.const_int(axis)),
                ("validate_indices", b.const_bool(False)),
            ],
            b.result_dims(op, i),
        )
        if source != ins[0]:
            piece = b.emit(
                *emit_cast(b, piece, "bool"), b.result_dims(op, i), _MIL_BOOL
            )
        pieces.append(piece)
    return Bound(pieces)


# ── shape ────────────────────────────────────────────────────────────────────


@_emitter("tile")
def _tile(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    reps = [_as_int(r) for r in _as_seq(_attr(op, "reps"))]
    return "tile", [("x", ins[0]), ("reps", b.const_ints(reps))]


@_emitter("flip")
def _flip(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    dims = [_as_int(d) for d in _as_seq(_attr(op, "dims"))]
    return "reverse", [("x", ins[0]), ("axes", b.const_ints(dims))]


@_emitter("pad")
def _pad(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """Constant padding; Lucid's ``pads`` is already MIL's per-axis pair list."""
    pads = [_as_int(v) for v in _as_seq(_attr(op, "pads"))]
    return "pad", [
        ("x", ins[0]),
        ("pad", b.const_ints(pads)),
        ("mode", b.const_str("constant")),
        ("constant_val", b.const_float(_as_float(_attr(op, "constant")))),
    ]


def _triangle(b: Builder, op: TracedOp, ins: list[str], lower: bool) -> EmitResult:
    """Keep one triangle of the trailing two axes, zero the other.

    MIL has ``band_part``, but its bounds are "negative means unbounded",
    so it cannot say *strictly* upper — and ``triu(x, 1)`` is how a
    causal mask is built, which makes the unexpressible case the common
    one. The mask depends only on the shape and the offset, both known
    while tracing, so it goes out as a constant and the operation is a
    multiply.
    """
    import lucid

    from lucid.coreml._build import UnsupportedOp

    if not b.result_is_float(op):
        raise UnsupportedOp(
            f"{op.name}: Core ML has no dtype-preserving triangle for a "
            "non-float tensor — cast to float before masking"
        )
    shape = b.shape_of(ins[0])
    if len(shape) < 2:
        raise UnsupportedOp(f"{op.name}: needs at least two axes, got {len(shape)}")
    offset = _as_int(_attr(op, "k"))
    ones = lucid.ones(*shape[-2:])
    mask = lucid.tril(ones, offset) if lower else lucid.triu(ones, offset)
    return "mul", [("x", ins[0]), ("y", b.const_from_tensor(mask))]


@_emitter("tril")
def _tril(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    return _triangle(b, op, ins, lower=True)


@_emitter("triu")
def _triu(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    return _triangle(b, op, ins, lower=False)


@_emitter("masked_fill")
def _masked_fill(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """``select(mask, fill, x)``; the mask is second, the fill a broadcast scalar."""
    value = _as_float(_attr(op, "fill_value"))
    integer = b.dtype_of(ins[0]) == _MIL_INT32
    fill = b.const_int(int(value)) if integer else b.const_float(value)
    return "select", [("cond", ins[1]), ("a", fill), ("b", ins[0])]


# ── the rest of what a traced graph reaches ──────────────────────────────────


@_emitter("isfinite")
def _isfinite(b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    """``x - x == 0`` — MIL has no ``isfinite``.

    Infinity minus itself is NaN and NaN compares false, so the one
    expression covers both non-finite cases.
    """
    shape = b.shape_of(ins[0])
    zeroed = b.emit("sub", [("x", ins[0]), ("y", ins[0])], shape)
    return "equal", [("x", zeroed), ("y", b.const_float(0.0))]


@_emitter("trunc")
def _trunc(b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    """``floor`` where ``x >= 0``, ``ceil`` elsewhere — MIL has no ``trunc``.

    The engine computes it the same way, so ``-0.0`` keeps its sign here too.
    """
    x = ins[0]
    shape = b.shape_of(x)
    non_negative = b.emit(
        "greater_equal", [("x", x), ("y", b.const_float(0.0))], shape, dtype=_MIL_BOOL
    )
    floored = b.emit("floor", [("x", x)], shape)
    ceiled = b.emit("ceil", [("x", x)], shape)
    return "select", [("cond", non_negative), ("a", floored), ("b", ceiled)]


@_emitter("nan_to_num")
def _nan_to_num(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """Three selects — MIL has no ``nan_to_num``.

    NaN is the one value unequal to itself; the infinities are found by
    equality rather than by ``x > FLT_MAX``, because in a float16 package
    that bound is itself infinite and the comparison is never true. The
    replacements are the values the trace resolved, so a default
    ``posinf`` is float32's largest — a float16 package holds it as
    infinity, as a cast of Lucid's result would.
    """
    x = ins[0]
    shape = b.shape_of(x)
    is_nan = b.emit("not_equal", [("x", x), ("y", x)], shape, dtype=_MIL_BOOL)
    is_pos = b.emit(
        "equal", [("x", x), ("y", b.const_float(math.inf))], shape, dtype=_MIL_BOOL
    )
    is_neg = b.emit(
        "equal", [("x", x), ("y", b.const_float(-math.inf))], shape, dtype=_MIL_BOOL
    )
    replaced = x
    for mask, key in ((is_nan, "nan"), (is_pos, "posinf")):
        value = b.const_float(_as_float(_attr(op, key)))
        replaced = b.emit(
            "select", [("cond", mask), ("a", value), ("b", replaced)], shape
        )
    neginf = b.const_float(_as_float(_attr(op, "neginf")))
    return "select", [("cond", is_neg), ("a", neginf), ("b", replaced)]


def _refuse_non_boolean(b: Builder, op: TracedOp, _ins: list[str], symbol: str) -> None:
    """Stop before writing a logical operation over integers.

    ``&`` on two integers combines their bit patterns; ``logical_and``
    reads each side as true-or-false. Emitting one for the other would
    load, run, and answer — ``5 & 3`` as ``True`` — so the difference has
    to be caught here, where there is still a name to put in the message.
    """
    from lucid.coreml._build import UnsupportedOp

    if b.result_mil_dtype(op) != _MIL_BOOL:
        raise UnsupportedOp(
            op.name,
            f"Core ML has logical operations and none at the bit level, so "
            f"``{symbol}`` is only exportable over booleans — this one joins "
            f"integers, whose bits it would silently read as true or false. "
            f"Compare them first if a boolean is what was meant",
        )


@_emitter("bitwise_and")
def _bitwise_and(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """Booleans only, which is the whole of what MIL offers.

    Lucid's ``&`` also takes integers and combines their bit patterns;
    MIL has ``logical_and`` and nothing at the bit level, so an integer
    pair is refused rather than answered with the wrong operation. The
    boolean case is the one models reach — an attention mask joined to a
    padding mask — and it is exactly ``logical_and``.
    """
    _refuse_non_boolean(b, op, ins, "&")
    return "logical_and", [("x", ins[0]), ("y", ins[1])]


@_emitter("bitwise_or")
def _bitwise_or(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    _refuse_non_boolean(b, op, ins, "|")
    return "logical_or", [("x", ins[0]), ("y", ins[1])]


@_emitter("bitwise_xor")
def _bitwise_xor(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    _refuse_non_boolean(b, op, ins, "^")
    return "logical_xor", [("x", ins[0]), ("y", ins[1])]


@_emitter("group_norm")
def _group_norm(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """Normalise per group, then scale per channel.

    MIL has ``layer_norm`` but no ``group_norm``. Splitting the groups
    out turns one into the other: reshape ``(N, C, ...)`` to ``(N, G, -1)``,
    normalise over the last axis, and reshape back. The affine weights stay
    per channel, so they are applied after the reshape rather than through
    ``layer_norm``.
    """
    shape = b.shape_of(ins[0])
    groups = _as_int(_attr(op, "num_groups"))
    channels = shape[1]
    trailing = shape.declared()[2:]
    # Each group one row: an open batch is the operand's own, read at run
    # time, and open spatial axes fold into the one ``reshape`` infers.
    rest = -1 if -1 in trailing else channels // groups * math.prod(trailing)
    grouped = [shape.declared()[0], groups, rest]
    target = b.shape_operand(grouped, known={0: (ins[0], 0)})
    folded = b.emit("reshape", [("x", ins[0]), ("shape", target)], grouped)
    normalised = b.emit(
        "layer_norm",
        [
            ("x", folded),
            ("axes", b.const_ints([2])),
            ("epsilon", b.const_float(_as_float(_attr(op, "eps")))),
        ],
        grouped,
    )
    target = b.shape_operand(b.result_dims(op), op.outputs[0].id, ins[0])
    restored = b.emit("reshape", [("x", normalised), ("shape", target)], shape)
    if len(ins) < 3:
        return "identity", [("x", restored)]

    # ``gamma``/``beta`` are per channel; broadcasting needs them shaped
    # (C, 1, ...) against the trailing axes.
    affine = [channels] + [1] * len(trailing)
    gamma = b.emit("reshape", [("x", ins[1]), ("shape", b.const_ints(affine))], affine)
    beta = b.emit("reshape", [("x", ins[2]), ("shape", b.const_ints(affine))], affine)
    scaled = b.emit("mul", [("x", restored), ("y", gamma)], shape)
    return "add", [("x", scaled), ("y", beta)]


@_emitter("rms_norm")
def _rms_norm(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """``x * rsqrt(mean(x^2) + eps) * weight`` — MIL has no ``rms_norm``.

    Unlike ``layer_norm`` this does not centre, so it cannot be borrowed
    from that op: subtracting the mean is the whole difference.
    """
    shape = b.shape_of(ins[0])
    weight_shape = b.shape_of(ins[1]) if len(ins) > 1 else []
    axes = list(range(len(shape) - len(weight_shape), len(shape)))
    kept = shape.declared()
    for axis in axes:
        kept[axis] = 1

    squared = b.emit("mul", [("x", ins[0]), ("y", ins[0])], shape)
    mean = b.emit(
        "reduce_mean",
        [
            ("x", squared),
            ("axes", b.const_ints(axes)),
            ("keep_dims", b.const_bool(True)),
        ],
        kept,
    )
    shifted = b.emit(
        "add", [("x", mean), ("y", b.const_float(_as_float(_attr(op, "eps"))))], kept
    )
    scale = b.emit("rsqrt", [("x", shifted), ("epsilon", b.const_float(0.0))], kept)
    normalised = b.emit("mul", [("x", ins[0]), ("y", scale)], shape)
    if len(ins) < 2:
        return "identity", [("x", normalised)]
    return "mul", [("x", normalised), ("y", ins[1])]


@_emitter("repeat")
def _repeat(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """Repeat each element along one axis, which ``tile`` does not do.

    ``tile`` lays the whole tensor down again; this interleaves. Adding a
    length-1 axis after the target, tiling that, and folding it back is
    the standard way to say it with the ops MIL has.
    """
    shape = b.shape_of(ins[0]).declared()
    repeats = _as_int(_attr(op, "repeats"))
    axis = _as_int(_attr(op, "axis"))
    axis = axis if axis >= 0 else axis + len(shape)

    spread = shape[: axis + 1] + [1] + shape[axis + 1 :]
    expanded = b.emit(
        "expand_dims", [("x", ins[0]), ("axes", b.const_ints([axis + 1]))], spread
    )
    reps = [1] * len(spread)
    reps[axis + 1] = repeats
    tiled_shape = list(spread)
    tiled_shape[axis + 1] = repeats
    tiled = b.emit("tile", [("x", expanded), ("reps", b.const_ints(reps))], tiled_shape)
    target = b.shape_operand(b.result_dims(op), op.outputs[0].id, ins[0])
    return "reshape", [("x", tiled), ("shape", target)]


@_emitter("invert")
def _invert(_b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    return "logical_not", [("x", ins[0])]


@_emitter("hard_sigmoid")
def _hard_sigmoid(b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    """``clip((x + 3) / 6, 0, 1)``, which MIL parameterises as alpha and beta."""
    return "sigmoid_hard", [
        ("x", ins[0]),
        ("alpha", b.const_float(1.0 / 6.0)),
        ("beta", b.const_float(0.5)),
    ]


@_emitter("hard_swish")
def _hard_swish(b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    """``x * hard_sigmoid(x)`` — there is no ``hard_swish`` in the opset."""
    gate = b.emit(
        "sigmoid_hard",
        [
            ("x", ins[0]),
            ("alpha", b.const_float(1.0 / 6.0)),
            ("beta", b.const_float(0.5)),
        ],
        b.shape_of(ins[0]),
    )
    return "mul", [("x", ins[0]), ("y", gate)]


@_emitter("diagonal")
def _diagonal(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """The diagonal as a gather along one of the two axes it spans.

    MIL has no ``diagonal``. Taking ``x[..., i, i + offset]`` is a gather
    along the second axis with the index equal to the position on the
    first, so the index tensor is a constant the trace already determines.
    """
    shape = b.shape_of(ins[0])
    rank = len(shape)
    axis1 = _as_int(_attr(op, "axis1"))
    axis2 = _as_int(_attr(op, "axis2"))
    offset = _as_int(_attr(op, "offset"))
    axis1 = axis1 if axis1 >= 0 else axis1 + rank
    axis2 = axis2 if axis2 >= 0 else axis2 + rank
    if axis2 != axis1 + 1 or axis2 != rank - 1:
        # Imported here: ``_build`` imports this module, so naming the
        # exception at module scope would close the cycle.
        from lucid.coreml._build import UnsupportedOp

        # Only the trailing pair is expressible this way; any other pair
        # would need a transpose the trace did not ask for.
        raise UnsupportedOp("diagonal")

    length = b.result_dims(op)[-1]
    # MIL wants the index tensor the same shape as the result, not a
    # broadcastable stand-in, so the leading axes are written out.
    leading = 1
    for dim in shape[:axis1]:
        leading *= dim
    picked = [i + offset for _ in range(leading) for i in range(length)]
    index_shape = list(shape[:axis1]) + [length, 1]
    indices = b.const_ints_shaped(picked, index_shape)
    gathered = list(shape[:axis1]) + [length, 1]
    source = widened(b, ins[0])
    taken = b.emit(
        "gather_along_axis",
        [
            ("x", source),
            ("indices", indices),
            ("axis", b.const_int(axis2)),
            ("validate_indices", b.const_bool(False)),
        ],
        gathered,
    )
    if source != ins[0]:
        taken = b.emit(*emit_cast(b, taken, "bool"), gathered, _MIL_BOOL)
    return "reshape", [("x", taken), ("shape", b.const_ints(list(b.result_dims(op))))]


@_emitter("meshgrid")
def _meshgrid(b: Builder, op: TracedOp, ins: list[str]) -> Bound:
    """Each input broadcast along every axis but its own.

    Two independent results with no single operation behind them, so the
    emitter appends both and says what they are — when the trace carries
    enough to build them, which today it does not.
    """
    grid = list(b.result_dims(op))
    if len(ins) != len(grid):
        from lucid.coreml._build import UnsupportedOp

        raise UnsupportedOp(
            "meshgrid",
            f"the trace recorded {len(ins)} operand(s) for a {len(grid)}-"
            "dimensional grid. Reconstructing the missing ones would put "
            "the wrong values in the grid, which is worse than not exporting",
        )
    # ``indexing="xy"`` swaps which axis the first two inputs vary along,
    # and the output shapes only show it when those inputs have different
    # lengths — so the trace carries it.
    swapped = bool(_as_int(op.attrs.get("indexing_xy", 0))) and len(ins) >= 2
    produced: list[str] = []
    for index, source in enumerate(ins):
        axis = (1 - index) if swapped and index < 2 else index
        spread = [1] * len(grid)
        spread[axis] = grid[axis]
        laid = b.emit(
            "reshape", [("x", source), ("shape", b.const_ints(spread))], spread
        )
        reps = [grid[i] if i != axis else 1 for i in range(len(grid))]
        produced.append(
            b.emit("tile", [("x", laid), ("reps", b.const_ints(reps))], grid)
        )
    return Bound(produced)


@_emitter("affine_grid")
def _affine_grid(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """The sampling grid an affine matrix produces, as one matmul.

    The grid before the transform is a constant: every output position
    written as ``(x, y, 1)`` in normalised coordinates. What ``theta``
    does to it is a matrix product, which MIL has.
    """
    import lucid

    width = _as_int(_attr(op, "W"))
    height = _as_int(_attr(op, "H"))
    corners = bool(_attr(op, "align_corners"))

    def coordinate(index: int, extent: int) -> float:
        # ``align_corners`` decides whether -1 and 1 name the centres of
        # the edge pixels or their outer edges.
        if corners:
            return -1.0 if extent == 1 else 2.0 * index / (extent - 1) - 1.0
        return (2.0 * index + 1.0) / extent - 1.0

    rows = [
        [coordinate(x, width), coordinate(y, height), 1.0]
        for y in range(height)
        for x in range(width)
    ]
    base = b.const_from_tensor(lucid.tensor([rows]))
    b.declare(base, [1, height * width, 3])
    out = b.result_dims(op).declared()
    flat = b.emit(
        "matmul",
        [
            ("x", base),
            ("y", ins[0]),
            ("transpose_x", b.const_bool(False)),
            ("transpose_y", b.const_bool(True)),
        ],
        [out[0], height * width, 2],
    )
    # The product is one row per position; the grid is those rows laid
    # back out over the image.
    target = b.shape_operand(out, op.outputs[0].id, ins[0])
    return "reshape", [("x", flat), ("shape", target)]


@_emitter("bilinear_layer")
def _bilinear_layer(b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    """``y[n,o] = x1[n,i] W[o,i,j] x2[n,j]``, as a matmul and a reduction.

    MIL has no bilinear form. Contracting ``j`` first leaves an ordinary
    product to sum over ``i``, and both halves are operations it does
    have.
    """
    x1, x2, weight = ins[0], ins[1], ins[2]
    left = b.shape_of(x1)
    right = b.shape_of(x2)
    outputs = int(b.shape_of(weight)[0])
    batch, inner_i, inner_j = left.declared()[0], left[1], right[1]

    # (N, J) -> (N, 1, J, 1) so the batched matmul lines up with
    # (1, O, I, J), contracting J.
    column = b.emit(
        "reshape",
        [("x", x2), ("shape", b.shape_operand([batch, 1, inner_j, 1]))],
        [batch, 1, inner_j, 1],
    )
    kernel = b.emit(
        "reshape",
        [("x", weight), ("shape", b.const_ints([1, outputs, inner_i, inner_j]))],
        [1, outputs, inner_i, inner_j],
    )
    contracted = b.emit(
        "matmul",
        [
            ("x", kernel),
            ("y", column),
            ("transpose_x", b.const_bool(False)),
            ("transpose_y", b.const_bool(False)),
        ],
        [batch, outputs, inner_i, 1],
    )
    folded = b.emit(
        "reshape",
        [("x", contracted), ("shape", b.shape_operand([batch, outputs, inner_i]))],
        [batch, outputs, inner_i],
    )
    spread = b.emit(
        "reshape",
        [("x", x1), ("shape", b.shape_operand([batch, 1, inner_i]))],
        [batch, 1, inner_i],
    )
    weighted = b.emit("mul", [("x", folded), ("y", spread)], [batch, outputs, inner_i])
    summed = b.emit(
        "reduce_sum",
        [
            ("x", weighted),
            ("axes", b.const_ints([2])),
            ("keep_dims", b.const_bool(False)),
        ],
        [batch, outputs],
    )
    if len(ins) < 4:
        return "identity", [("x", summed)]
    return "add", [("x", summed), ("y", ins[3])]


@_emitter("lp_normalize")
def _lp_normalize(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """``x / max(||x||_p, eps)`` along one axis.

    MIL's ``l2_norm`` normalises over everything but the batch, which is
    a different operation; this one names its axis.
    """
    shape = b.shape_of(ins[0])
    axis = _as_int(_attr(op, "axis"))
    order = _as_float(_attr(op, "ord"))
    epsilon = _as_float(_attr(op, "eps"))
    kept = shape.declared()
    kept[axis if axis >= 0 else axis + len(kept)] = 1

    magnitude = b.emit("abs", [("x", ins[0])], shape)
    if order == 2.0:
        total = b.emit(
            "reduce_l2_norm",
            [
                ("x", ins[0]),
                ("axes", b.const_ints([axis])),
                ("keep_dims", b.const_bool(True)),
            ],
            kept,
        )
    else:
        raised = b.emit("pow", [("x", magnitude), ("y", b.const_float(order))], shape)
        summed = b.emit(
            "reduce_sum",
            [
                ("x", raised),
                ("axes", b.const_ints([axis])),
                ("keep_dims", b.const_bool(True)),
            ],
            kept,
        )
        total = b.emit("pow", [("x", summed), ("y", b.const_float(1.0 / order))], kept)
    floored = b.emit("maximum", [("x", total), ("y", b.const_float(epsilon))], kept)
    return "real_div", [("x", ins[0]), ("y", floored)]


@_emitter("unfold")
def _unfold(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """Sliding blocks as columns, which MIL has no single operation for.

    One strided slice per kernel position, stacked so that the column
    index runs ``(channel, kernel row, kernel column)`` — the order Lucid
    lays them in. Concatenating the slices directly would give
    ``(kernel, channel)`` instead, which has the right shape and the
    wrong rows.
    """
    attrs = op.attrs
    shape = b.shape_of(ins[0])
    if len(shape) != 4:
        from lucid.coreml._build import UnsupportedOp

        raise UnsupportedOp("unfold")
    kernel = _ints(attrs["kernel_size"])
    stride = _ints(attrs["stride"])
    padding = _ints(attrs["padding"])
    dilation = _ints(attrs["dilation"])

    source = ins[0]
    padded = list(shape)
    if any(padding):
        pads = [0, 0, 0, 0, padding[0], padding[0], padding[1], padding[1]]
        padded = [
            shape[0],
            shape[1],
            shape[2] + 2 * padding[0],
            shape[3] + 2 * padding[1],
        ]
        source = b.emit(
            "pad",
            [
                ("x", source),
                ("pad", b.const_ints(pads)),
                ("mode", b.const_str("constant")),
                ("constant_val", b.const_float(0.0)),
            ],
            padded,
        )

    batch, channels = padded[0], padded[1]
    out = list(b.result_dims(op))
    length = out[2]
    rows = (padded[2] - dilation[0] * (kernel[0] - 1) - 1) // stride[0] + 1
    columns = (padded[3] - dilation[1] * (kernel[1] - 1) - 1) // stride[1] + 1

    blocks: list[str] = []
    for i in range(kernel[0]):
        for j in range(kernel[1]):
            begin = [0, 0, i * dilation[0], j * dilation[1]]
            end = [
                batch,
                channels,
                i * dilation[0] + (rows - 1) * stride[0] + 1,
                j * dilation[1] + (columns - 1) * stride[1] + 1,
            ]
            window = b.emit(
                "slice_by_index",
                [
                    ("x", source),
                    ("begin", b.const_ints(begin)),
                    ("end", b.const_ints(end)),
                    ("stride", b.const_ints([1, 1, stride[0], stride[1]])),
                ],
                [batch, channels, rows, columns],
            )
            blocks.append(
                b.emit(
                    "reshape",
                    [
                        ("x", window),
                        ("shape", b.const_ints([batch, channels, 1, length])),
                    ],
                    [batch, channels, 1, length],
                )
            )

    stacked = b.emit(
        "concat",
        [
            ("values", blocks),
            ("axis", b.const_int(2)),
            ("interleave", b.const_bool(False)),
        ],
        [batch, channels, len(blocks), length],
    )
    return "reshape", [("x", stacked), ("shape", b.const_ints(out))]


@_emitter("interpolate_nearest_3d")
def _interpolate_nearest_3d(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """Nearest resampling in three dimensions, without ever leaving rank 5.

    MIL's ``upsample_nearest_neighbor`` is two-dimensional, so the height
    and width go through it with depth folded in as channels. Depth is
    then repetition — which is what nearest resampling by a whole number
    is — done on a rank-3 view, because inserting an axis into a rank-5
    tensor would exceed the rank Core ML allows and fail at load with an
    error naming neither the operation nor the rank.

    A fractional factor is not repetition, and is refused rather than
    rounded into something that looks like it worked.
    """
    from lucid.coreml._build import UnsupportedOp

    shape = b.shape_of(ins[0])
    out = list(b.result_dims(op))
    if len(shape) != 5:
        raise UnsupportedOp("interpolate_nearest_3d")
    batch, channels, depth, height, width = shape
    if any(out[a] % shape[a] for a in (2, 3, 4)):
        raise UnsupportedOp("interpolate_nearest_3d")
    scale_d, scale_h, scale_w = (out[a] // shape[a] for a in (2, 3, 4))

    # Depth rides along as channels while the plane is resampled.
    planes = [batch * channels, depth, height, width]
    folded = b.emit("reshape", [("x", ins[0]), ("shape", b.const_ints(planes))], planes)
    grown = [batch * channels, depth, out[3], out[4]]
    resampled = b.emit(
        "upsample_nearest_neighbor",
        [
            ("x", folded),
            ("scale_factor_height", b.const_float32(float(scale_h))),
            ("scale_factor_width", b.const_float32(float(scale_w))),
        ],
        grown,
    )

    if scale_d != 1:
        area = out[3] * out[4]
        rows = [batch * channels, depth, area]
        flat = b.emit(
            "reshape", [("x", resampled), ("shape", b.const_ints(rows))], rows
        )
        spread = [batch * channels, depth, 1, area]
        spaced = b.emit(
            "reshape", [("x", flat), ("shape", b.const_ints(spread))], spread
        )
        repeated = [batch * channels, depth, scale_d, area]
        tiled = b.emit(
            "tile",
            [("x", spaced), ("reps", b.const_ints([1, 1, scale_d, 1]))],
            repeated,
        )
        resampled = tiled

    return "reshape", [("x", resampled), ("shape", b.const_ints(out))]


# Giles (2010), "Approximating the erfinv function" — the single-precision
# coefficients, one polynomial for the central region and one for the
# tails.  Written out rather than derived so the numbers can be checked
# against the paper.
_ERFINV_CENTRAL = (
    2.81022636e-08,
    3.43273939e-07,
    -3.5233877e-06,
    -4.39150654e-06,
    0.00021858087,
    -0.00125372503,
    -0.00417768164,
    0.246640727,
    1.50140941,
)
_ERFINV_TAIL = (
    -0.000200214257,
    0.000100950558,
    0.00134934322,
    -0.00367342844,
    0.00573950773,
    -0.0076224613,
    0.00943887047,
    1.00167406,
    2.83297682,
)


def _horner(
    b: Builder, coefficients: tuple[float, ...], w: str, shape: Sequence[int]
) -> str:
    """Evaluate a polynomial in ``w`` by Horner's rule."""
    value = b.const_float(coefficients[0])
    for coefficient in coefficients[1:]:
        scaled = b.emit("mul", [("x", value), ("y", w)], shape)
        value = b.emit("add", [("x", scaled), ("y", b.const_float(coefficient))], shape)
    return value


@_emitter("erfinv")
def _erfinv(b: Builder, _op: TracedOp, ins: list[str]) -> EmitResult:
    """The inverse error function, which MIL does not have.

    An approximation, and said so: Giles' single-precision rational form,
    two polynomials selected on how far into the tail the argument is.
    Both branches are evaluated and one is chosen, because MIL has no
    branch — which costs arithmetic and nothing in accuracy.

    Measured against Lucid's own ``erfinv`` on |x| < 1: agreement to
    about 1e-6 relative, which is the same order as a float32 export's
    other error rather than a new one.
    """
    x = ins[0]
    shape = b.shape_of(x)

    # w = -log((1 - x) * (1 + x)), the argument both polynomials take.
    one = b.const_float(1.0)
    lower = b.emit("sub", [("x", one), ("y", x)], shape)
    upper = b.emit("add", [("x", x), ("y", one)], shape)
    product = b.emit("mul", [("x", lower), ("y", upper)], shape)
    logged = b.emit("log", [("x", product), ("epsilon", b.const_float(0.0))], shape)
    w = b.emit("mul", [("x", logged), ("y", b.const_float(-1.0))], shape)

    central_w = b.emit("sub", [("x", w), ("y", b.const_float(2.5))], shape)
    central = _horner(b, _ERFINV_CENTRAL, central_w, shape)

    rooted = b.emit("sqrt", [("x", w)], shape)
    tail_w = b.emit("sub", [("x", rooted), ("y", b.const_float(3.0))], shape)
    tail = _horner(b, _ERFINV_TAIL, tail_w, shape)

    near = b.emit("less", [("x", w), ("y", b.const_float(5.0))], shape, dtype=_MIL_BOOL)
    chosen = b.emit("select", [("cond", near), ("a", central), ("b", tail)], shape)
    return "mul", [("x", chosen), ("y", x)]


@_emitter("interpolate_trilinear")
def _interpolate_trilinear(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """Linear resampling in three dimensions, which MIL stops short of.

    Separable, so it is two operations rather than one: MIL's
    ``upsample_bilinear`` resamples the plane with depth folded in as
    channels, and depth is then a blend of two slices whose indices and
    weights the output size fixes — both constants, so the blend is a
    pair of gathers and an add.

    ``align_corners`` decides which source coordinate each output sample
    reads, and it has to reach both halves.  It used to reach neither:
    the trace did not record it, so an ``align_corners=True`` model
    exported as its ``False`` counterpart — a well-formed package, a
    plausible volume, and values off by 23%.
    """
    import lucid

    from lucid.coreml._build import UnsupportedOp

    shape = b.shape_of(ins[0])
    out = list(b.result_dims(op))
    if len(shape) != 5:
        raise UnsupportedOp("interpolate_trilinear")
    batch, channels, depth, height, width = shape
    align = bool(op.attrs.get("align_corners", False))

    planes = [batch * channels, depth, height, width]
    folded = b.emit("reshape", [("x", ins[0]), ("shape", b.const_ints(planes))], planes)
    grown = [batch * channels, depth, out[3], out[4]]
    resampled = b.emit(
        "upsample_bilinear",
        [
            ("x", folded),
            ("scale_factor_height", b.const_float32(out[3] / height)),
            ("scale_factor_width", b.const_float32(out[4] / width)),
            ("align_corners", b.const_bool(align)),
        ],
        grown,
    )
    spatial = [batch, channels, depth, out[3], out[4]]
    restored = b.emit(
        "reshape", [("x", resampled), ("shape", b.const_ints(spatial))], spatial
    )
    if out[2] == depth:
        return "identity", [("x", restored)]

    # Where output sample d reads from, in input coordinates.  The two
    # conventions are the engine's own (see the ``src_coord_fn`` in
    # ``CpuBackend::interpolate_trilinear_forward``): with corners aligned
    # the endpoints are pinned to the endpoints, otherwise sample centres
    # map to sample centres.
    lower: list[int] = []
    upper: list[int] = []
    blend: list[float] = []
    for index in range(out[2]):
        if align:
            position = 0.0 if out[2] <= 1 else index * (depth - 1) / (out[2] - 1)
        else:
            position = (index + 0.5) * depth / out[2] - 0.5
        position = min(max(position, 0.0), float(depth - 1))
        low = min(int(position), depth - 1)
        high = min(low + 1, depth - 1)
        lower.append(low)
        upper.append(high)
        blend.append(position - low)

    taken = [batch, channels, out[2], out[3], out[4]]
    low_slice = b.emit(
        "gather",
        [
            ("x", restored),
            ("indices", b.const_ints_shaped(lower, [out[2]])),
            ("axis", b.const_int(2)),
            ("validate_indices", b.const_bool(False)),
        ],
        taken,
    )
    high_slice = b.emit(
        "gather",
        [
            ("x", restored),
            ("indices", b.const_ints_shaped(upper, [out[2]])),
            ("axis", b.const_int(2)),
            ("validate_indices", b.const_bool(False)),
        ],
        taken,
    )
    weights = b.const_from_tensor(lucid.tensor(blend).reshape(1, 1, out[2], 1, 1))
    difference = b.emit("sub", [("x", high_slice), ("y", low_slice)], taken)
    scaled = b.emit("mul", [("x", difference), ("y", weights)], taken)
    return "add", [("x", low_slice), ("y", scaled)]


@_emitter("fold")
def _fold(b: Builder, op: TracedOp, ins: list[str]) -> EmitResult:
    """Columns summed back into an image — ``unfold`` run backwards.

    Overlapping blocks have to be added, not written, and MIL has no
    scatter that would do it in one step at a sensible size. Each kernel
    position instead becomes a plane the size of the output — its blocks
    spaced out by the stride with zeros between, then shifted into place
    by a pad — and the planes are summed. Every offset is fixed by the
    attributes, so nothing here depends on the values.
    """
    from lucid.coreml._build import UnsupportedOp

    attrs = op.attrs
    shape = b.shape_of(ins[0])
    if len(shape) != 3:
        raise UnsupportedOp("fold")
    kernel = _ints(attrs["kernel_size"])
    stride = _ints(attrs["stride"])
    padding = _ints(attrs["padding"])
    dilation = _ints(attrs["dilation"])
    size = _ints(attrs["output_size"])

    batch, rows, length = shape
    out = list(b.result_dims(op))
    channels = out[1]
    padded = [size[0] + 2 * padding[0], size[1] + 2 * padding[1]]
    blocks_h = (padded[0] - dilation[0] * (kernel[0] - 1) - 1) // stride[0] + 1
    blocks_w = (padded[1] - dilation[1] * (kernel[1] - 1) - 1) // stride[1] + 1
    if blocks_h * blocks_w != length or rows != channels * kernel[0] * kernel[1]:
        raise UnsupportedOp("fold")

    plane = [batch * channels, padded[0], padded[1]]
    total: str | None = None
    for i in range(kernel[0]):
        for j in range(kernel[1]):
            offset = i * kernel[1] + j
            picked = b.emit(
                "slice_by_index",
                [
                    ("x", ins[0]),
                    ("begin", b.const_ints([0, offset, 0])),
                    ("end", b.const_ints([batch, rows, length])),
                    ("stride", b.const_ints([1, kernel[0] * kernel[1], 1])),
                ],
                [batch, channels, length],
            )
            grid = [batch * channels, blocks_h, blocks_w]
            block = b.emit(
                "reshape", [("x", picked), ("shape", b.const_ints(grid))], grid
            )

            # Space the blocks out by the stride, one axis at a time, on a
            # rank-4 view so the inserted axis stays inside Core ML's cap.
            span = [(blocks_h - 1) * stride[0] + 1, (blocks_w - 1) * stride[1] + 1]
            spread = block
            for axis, step in ((2, stride[1]), (1, stride[0])):
                if step == 1:
                    continue
                current = b.shape_of(spread)
                opened = list(current[: axis + 1]) + [1] + list(current[axis + 1 :])
                widened = b.emit(
                    "reshape",
                    [("x", spread), ("shape", b.const_ints(opened))],
                    opened,
                )
                pads = [0] * (2 * len(opened))
                pads[2 * (axis + 1) + 1] = step - 1
                filled = list(opened)
                filled[axis + 1] = step
                padded_block = b.emit(
                    "pad",
                    [
                        ("x", widened),
                        ("pad", b.const_ints(pads)),
                        ("mode", b.const_str("constant")),
                        ("constant_val", b.const_float(0.0)),
                    ],
                    filled,
                )
                merged = list(current)
                merged[axis] = current[axis] * step
                spread = b.emit(
                    "reshape",
                    [("x", padded_block), ("shape", b.const_ints(merged))],
                    merged,
                )
            stretched = b.shape_of(spread)
            if stretched[1] != span[0] or stretched[2] != span[1]:
                spread = b.emit(
                    "slice_by_index",
                    [
                        ("x", spread),
                        ("begin", b.const_ints([0, 0, 0])),
                        ("end", b.const_ints([batch * channels, span[0], span[1]])),
                        ("stride", b.const_ints([1, 1, 1])),
                    ],
                    [batch * channels, span[0], span[1]],
                )

            top = i * dilation[0]
            left = j * dilation[1]
            placed = b.emit(
                "pad",
                [
                    ("x", spread),
                    (
                        "pad",
                        b.const_ints(
                            [
                                0,
                                0,
                                top,
                                padded[0] - top - span[0],
                                left,
                                padded[1] - left - span[1],
                            ]
                        ),
                    ),
                    ("mode", b.const_str("constant")),
                    ("constant_val", b.const_float(0.0)),
                ],
                plane,
            )
            total = (
                placed
                if total is None
                else b.emit("add", [("x", total), ("y", placed)], plane)
            )

    assert total is not None
    if any(padding):
        total = b.emit(
            "slice_by_index",
            [
                ("x", total),
                ("begin", b.const_ints([0, padding[0], padding[1]])),
                (
                    "end",
                    b.const_ints(
                        [batch * channels, padding[0] + size[0], padding[1] + size[1]]
                    ),
                ),
                ("stride", b.const_ints([1, 1, 1])),
            ],
            [batch * channels, size[0], size[1]],
        )
    return "reshape", [("x", total), ("shape", b.const_ints(out))]
