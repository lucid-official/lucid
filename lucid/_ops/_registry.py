"""
Ops registry: maps op names to engine functions and Tensor method names.

Used by ``_methods.py`` to auto-inject Tensor methods and by
``_ops/__init__.py`` to expose free functions.

Adapter functions (signature normalisers, e.g. ``A._sum_adapter``) live in
``_adapters.py`` — this file references them so the registry stays a
declarative table of ``OpEntry`` records, not a mix of declarations and
plumbing.
"""

from dataclasses import dataclass, field
from typing import Any, Callable

from lucid import _dispatch
from lucid._C import engine as _C_engine
from lucid._ops import _adapters as A  # accessor for every signature-normaliser
from lucid._ops._division import _make_div_adapter, _make_round_adapter
from lucid._ops._reduce import (
    _all_adapter,
    _any_adapter,
    _argmax_adapter,
    _argmin_adapter,
    _cummax_adapter,
    _cummin_adapter,
    _cumprod_adapter,
    _cumsum_adapter,
    _log_softmax_adapter,
    _logsumexp_adapter,
    _max_adapter,
    _mean_adapter,
    _min_adapter,
    _prod_adapter,
    _softmax_adapter,
    _std_adapter,
    _sum_adapter,
    _var_adapter,
)


@dataclass(slots=True)
class OpEntry:
    """Descriptor for a single engine operation registered with the dispatcher.

    The op registry is the single source of truth for the
    Tensor-method / ``lucid.xxx`` / engine-binding triplet — every entry
    here generates (at import time) a method on :class:`Tensor` and a
    free function in :mod:`lucid._ops`, both routing through the same
    underlying ``engine_fn``.

    Attributes
    ----------
    name : str
        Canonical op name (e.g. ``"abs"``).
    engine_fn : Callable
        The C++ engine entry point (a pybind11-bound function).
    n_tensor_args : int
        How many positional Tensor arguments to unwrap before the call.
        ``-1`` is the special value for variadic ops where the first
        argument is a *list* of tensors (e.g. ``stack``, ``cat``).
    returns_tensor : bool, optional
        ``True`` (default) when the result is a Tensor and should be
        wrapped back via ``_wrap``.  ``False`` for ops returning
        scalars / tuples of metadata.
    inplace : bool, optional
        ``True`` for trailing-underscore ops that mutate their first
        operand.  Default ``False``.
    method_name : str, optional
        Name to expose on :class:`Tensor` as a method.  ``None`` means
        no method is generated.
    free_fn_name : str, optional
        Name to expose in :mod:`lucid` as a free function.  ``None``
        falls back to :attr:`name`.
    extra_kwargs : list[str], optional
        Keyword-only names appended after the unwrapped positional
        tensors.  Useful for ops that take both tensors and shape /
        dim parameters.
    """

    name: str
    engine_fn: Callable[..., Any]
    n_tensor_args: int  # positional Tensor args to unwrap; -1 = first arg is list
    returns_tensor: bool = True
    inplace: bool = False
    method_name: str | None = None  # Tensor method name (None = no method)
    free_fn_name: str | None = None  # lucid.xxx name (None = same as name)
    extra_kwargs: list[str] = field(default_factory=list)


def _bind_leading(
    op: str, names: tuple[str, ...], args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[tuple[object, ...], dict[str, object]]:
    """``args`` / ``kwargs`` with the operands named in ``names`` moved to position.

    ``names`` are the parameters ``args`` would fill from its first slot
    on.  An operand given both ways, or one given by keyword after a gap
    in the positions before it, is refused as an ordinary call would be.
    The free functions (``lucid._ops``) and the Tensor methods
    (``lucid._tensor._methods``) both bind through here.
    """
    if kwargs.keys().isdisjoint(names):
        return args, kwargs
    rest = dict(kwargs)
    filled = list(args)
    for slot, name in enumerate(names):
        if name not in rest:
            continue
        if slot < len(filled):
            raise TypeError(f"{op}() got multiple values for argument {name!r}")
        if slot > len(filled):
            raise TypeError(f"{op}() missing required argument {names[len(filled)]!r}")
        filled.append(rest.pop(name))
    return tuple(filled), rest


_Unary = Callable[[_C_engine.TensorImpl], _C_engine.TensorImpl]


def _float_valued(engine_fn: _Unary, *, engine_promotes: bool = True) -> _Unary:
    """``engine_fn``, a unary op whose answer is floating, taking a bool or
    integer operand at the default float (:func:`lucid._dispatch._float_operand`).

    The entry's free function and its Tensor method both call this, so
    ``lucid.exp2(i)`` and ``i.exp2()`` cast alike.

    ``engine_promotes`` says the engine op makes that cast itself when the
    target is float32 — its kernel's schema guard turns a bool or integer
    operand into float32 — so while the default float is float32 the operand
    goes straight through and the dtype is not read: a dtype read costs about
    a third of a small unary call, the flag test a hundredth.  It is False
    for the engine compositions that build the answer from other ops without
    that guard — ``exp2`` multiplied an integer by a ``log 2`` truncated to
    0, ``log1p`` refused to mix int and float — which take the cast on every
    call.
    """
    if engine_promotes:

        def _op(input: _C_engine.TensorImpl, /) -> _C_engine.TensorImpl:
            if _dispatch._DEFAULT_FLOAT_IS_F32:
                return engine_fn(input)
            return engine_fn(_dispatch._float_operand(input))

    else:

        def _op(input: _C_engine.TensorImpl, /) -> _C_engine.TensorImpl:
            return engine_fn(_dispatch._float_operand(input))

    A._keep_engine_signature(_op, engine_fn)
    _op.__qualname__ = _op.__name__
    return _op


def _real(input: _C_engine.TensorImpl, /) -> _C_engine.TensorImpl:
    """real(input): a complex tensor's real part as a view, a real tensor itself.

    The engine op refuses a real operand, where the reference hands it
    back — ``real(x) is x`` — so ``x.real`` and ``lucid.real(x)`` take any
    tensor.  ``imag`` keeps the refusal: a real tensor has no imaginary
    part to view, and the reference refuses it too.
    """
    if input.dtype in (_C_engine.C64, _C_engine.C128):
        return _C_engine.real(input)
    return input


A._keep_engine_signature(_real, _C_engine.real)

_R = _C_engine  # shorthand

# fmt: off
_REGISTRY: list[OpEntry] = [
    # ── unary ──────────────────────────────────────────────────────────────
    OpEntry("neg",       _R.neg,       1, method_name="neg",       free_fn_name="neg"),
    OpEntry("abs",       _R.abs,       1, method_name="abs",       free_fn_name="abs"),
    OpEntry("sign",      _R.sign,      1, method_name="sign",      free_fn_name="sign"),
    OpEntry("exp",       _float_valued(_R.exp),       1, method_name="exp",       free_fn_name="exp"),
    OpEntry("log",       _float_valued(_R.log),       1, method_name="log",       free_fn_name="log"),
    OpEntry("log2",      _float_valued(_R.log2),      1, method_name="log2",      free_fn_name="log2"),
    OpEntry("sqrt",      _float_valued(_R.sqrt),      1, method_name="sqrt",      free_fn_name="sqrt"),
    # rsqrt: not in engine — implemented in Python as reciprocal(sqrt(x))
    OpEntry("square",    _R.square,    1, method_name="square",    free_fn_name="square"),
    OpEntry("reciprocal",_float_valued(_R.reciprocal),1, method_name="reciprocal",free_fn_name="reciprocal"),
    OpEntry("floor",     _R.floor,     1, method_name="floor",     free_fn_name="floor"),
    OpEntry("ceil",      _R.ceil,      1, method_name="ceil",      free_fn_name="ceil"),
    OpEntry("round",     _make_round_adapter(), 1, method_name="round", free_fn_name="round",
            extra_kwargs=["decimals"]),
    OpEntry("sin",       _float_valued(_R.sin),       1, method_name="sin",       free_fn_name="sin"),
    OpEntry("cos",       _float_valued(_R.cos),       1, method_name="cos",       free_fn_name="cos"),
    OpEntry("tan",       _float_valued(_R.tan),       1, method_name="tan",       free_fn_name="tan"),
    OpEntry("arcsin",    _float_valued(_R.arcsin),    1, method_name="arcsin",    free_fn_name="arcsin"),
    OpEntry("arccos",    _float_valued(_R.arccos),    1, method_name="arccos",    free_fn_name="arccos"),
    OpEntry("arctan",    _float_valued(_R.arctan),    1, method_name="arctan",    free_fn_name="arctan"),
    OpEntry("sinh",      _float_valued(_R.sinh),      1, method_name="sinh",      free_fn_name="sinh"),
    OpEntry("cosh",      _float_valued(_R.cosh),      1, method_name="cosh",      free_fn_name="cosh"),
    OpEntry("tanh",      _float_valued(_R.tanh),      1, method_name="tanh",      free_fn_name="tanh"),
    OpEntry("relu",      _R.relu,      1, method_name="relu",      free_fn_name="relu"),
    OpEntry("sigmoid",   _float_valued(_R.sigmoid),   1, method_name="sigmoid",   free_fn_name="sigmoid"),
    OpEntry("silu",      _R.silu,      1, method_name="silu",      free_fn_name="silu"),
    # ``_R.gelu_exact``, not ``_R.gelu``.  The engine's ``gelu`` is the tanh
    # approximation, and ``F.gelu``'s default is the exact erf form — so
    # ``x.gelu()`` and ``F.gelu(x)`` were two different functions wearing one
    # name and disagreeing by 2e-4.  The approximation is still reachable,
    # spelled out, as ``F.gelu(x, approximate="tanh")``.
    OpEntry("gelu", _R.gelu_exact, 1, method_name="gelu", free_fn_name="gelu"),
    OpEntry("mish",      _R.mish,      1, method_name="mish",      free_fn_name="mish"),
    OpEntry("selu",      _R.selu,      1, method_name="selu",      free_fn_name="selu"),
    OpEntry("softplus",  _R.softplus,  1, method_name="softplus",  free_fn_name="softplus"),
    OpEntry("relu6",     _R.relu6,     1, method_name="relu6",     free_fn_name="relu6"),
    OpEntry("hard_sigmoid", _R.hard_sigmoid, 1, method_name="hard_sigmoid", free_fn_name="hard_sigmoid"),
    OpEntry("hard_swish",   _R.hard_swish,   1, method_name="hard_swish",   free_fn_name="hard_swish"),
    OpEntry("ravel",     _R.ravel,     1, method_name="ravel",     free_fn_name="ravel"),
    OpEntry("contiguous",A._contiguous_adapter,1, method_name="contiguous",free_fn_name="contiguous"),

    # ── in-place unary ─────────────────────────────────────────────────────
    OpEntry("neg_",        _R.neg_,        1, inplace=True, method_name="neg_"),
    OpEntry("abs_",        _R.abs_,        1, inplace=True, method_name="abs_"),
    OpEntry("sign_",       _R.sign_,       1, inplace=True, method_name="sign_"),
    OpEntry("reciprocal_", _R.reciprocal_, 1, inplace=True, method_name="reciprocal_"),
    OpEntry("square_",     _R.square_,     1, inplace=True, method_name="square_"),
    OpEntry("cube_",       _R.cube_,       1, inplace=True, method_name="cube_"),
    OpEntry("exp_",        _R.exp_,        1, inplace=True, method_name="exp_"),
    OpEntry("log_",        _R.log_,        1, inplace=True, method_name="log_"),
    OpEntry("log2_",       _R.log2_,       1, inplace=True, method_name="log2_"),
    OpEntry("sqrt_",       _R.sqrt_,       1, inplace=True, method_name="sqrt_"),
    OpEntry("sin_",        _R.sin_,        1, inplace=True, method_name="sin_"),
    OpEntry("cos_",        _R.cos_,        1, inplace=True, method_name="cos_"),
    OpEntry("tan_",        _R.tan_,        1, inplace=True, method_name="tan_"),
    OpEntry("arcsin_",     _R.arcsin_,     1, inplace=True, method_name="arcsin_"),
    OpEntry("arccos_",     _R.arccos_,     1, inplace=True, method_name="arccos_"),
    OpEntry("arctan_",     _R.arctan_,     1, inplace=True, method_name="arctan_"),
    OpEntry("sinh_",       _R.sinh_,       1, inplace=True, method_name="sinh_"),
    OpEntry("cosh_",       _R.cosh_,       1, inplace=True, method_name="cosh_"),
    OpEntry("tanh_",       _R.tanh_,       1, inplace=True, method_name="tanh_"),
    OpEntry("sigmoid_",    _R.sigmoid_,    1, inplace=True, method_name="sigmoid_"),
    OpEntry("relu_",       _R.relu_,       1, inplace=True, method_name="relu_"),
    OpEntry("floor_",      _R.floor_,      1, inplace=True, method_name="floor_"),
    OpEntry("ceil_",       _R.ceil_,       1, inplace=True, method_name="ceil_"),
    OpEntry("round_",      _make_round_adapter(inplace=True), 1, inplace=True, method_name="round_",
            extra_kwargs=["decimals"]),

    # ── binary ─────────────────────────────────────────────────────────────
    # add/sub/mul/pow/maximum/minimum are wrapped with _make_arith_adapter
    # so that mixed-dtype operands (e.g. float32 × bool) are automatically
    # promoted to their common type before entering the C++ kernel — matching
    # the reference framework's type-promotion semantics.
    # matmul/dot/inner bypass it: they have dtype contracts of their own that
    # the engine enforces.  outer has none — an elementwise product, nothing
    # contracted — so it promotes its two tensors (and only tensors) through
    # the adapter it shares with linalg.outer.
    OpEntry("add",      A._make_arith_adapter(_R.add),      2, method_name="add",      free_fn_name="add",
            extra_kwargs=["alpha"]),
    OpEntry("sub",      A._make_arith_adapter(_R.sub),      2, method_name="sub",      free_fn_name="sub",
            extra_kwargs=["alpha"]),
    OpEntry("mul",      A._make_arith_adapter(_R.mul),      2, method_name="mul",      free_fn_name="mul"),
    # div adds ``rounding_mode``: "floor" is floor_divide's path, "trunc"
    # rounds toward zero (``lucid._ops._division``).
    OpEntry("div",      _make_div_adapter(), 2, method_name="div",      free_fn_name="div"),
    OpEntry("pow",      A._make_arith_adapter(_R.pow),      2, method_name="pow",      free_fn_name="pow"),
    OpEntry("maximum",  A._make_arith_adapter(_R.maximum),  2, method_name="maximum",  free_fn_name="maximum"),
    OpEntry("minimum",  A._make_arith_adapter(_R.minimum),  2, method_name="minimum",  free_fn_name="minimum"),
    # Integer operands keep their dtype; a half tensor's scalar partner is
    # read in float32 (op-math), as for mul and div.
    OpEntry("floor_divide", A._make_arith_adapter(_R.floordiv), 2, method_name="floor_divide", free_fn_name="floor_divide"),
    OpEntry("matmul",   _R.matmul,   2, method_name="matmul",   free_fn_name="matmul"),
    OpEntry("dot",      _R.dot,      2, method_name="dot",      free_fn_name="dot"),
    OpEntry("inner",    _R.inner,    2, method_name="inner",    free_fn_name="inner"),
    OpEntry("outer",    A._outer_adapter, 2, method_name="outer", free_fn_name="outer"),

    # ── in-place binary ────────────────────────────────────────────────────
    OpEntry("add_",     A._make_arith_adapter(_R.add_, inplace=True),     2, inplace=True, method_name="add_",
            extra_kwargs=["alpha"]),
    OpEntry("sub_",     A._make_arith_adapter(_R.sub_, inplace=True),     2, inplace=True, method_name="sub_",
            extra_kwargs=["alpha"]),
    OpEntry("mul_",     A._make_arith_adapter(_R.mul_, inplace=True),     2, inplace=True, method_name="mul_"),
    OpEntry("div_",     _make_div_adapter(inplace=True), 2, inplace=True, method_name="div_"),
    OpEntry("pow_",     A._make_arith_adapter(_R.pow_, inplace=True),     2, inplace=True, method_name="pow_"),
    OpEntry("maximum_", A._make_arith_adapter(_R.maximum_, inplace=True), 2, inplace=True, method_name="maximum_"),
    OpEntry("minimum_", A._make_arith_adapter(_R.minimum_, inplace=True), 2, inplace=True, method_name="minimum_"),

    # ── reduction (with API-compat adapters) ───────────────────────────────
    OpEntry("sum",    _sum_adapter,    1, method_name="sum",    free_fn_name="sum",
            extra_kwargs=["dim", "keepdim", "dtype"]),
    OpEntry("mean",   _mean_adapter,   1, method_name="mean",   free_fn_name="mean",
            extra_kwargs=["dim", "keepdim", "dtype"]),
    OpEntry("prod",   _prod_adapter,   1, method_name="prod",   free_fn_name="prod",
            extra_kwargs=["dim", "keepdim", "dtype"]),
    OpEntry("max",    _max_adapter,    1, method_name="max",    free_fn_name="max",
            extra_kwargs=["dim", "keepdim"]),
    OpEntry("min",    _min_adapter,    1, method_name="min",    free_fn_name="min",
            extra_kwargs=["dim", "keepdim"]),
    OpEntry("var",    _var_adapter,      1, method_name="var",    free_fn_name="var",
            extra_kwargs=["dim", "keepdim", "correction", "unbiased"]),
    OpEntry("argmax", _argmax_adapter, 1, method_name="argmax", free_fn_name="argmax",
            extra_kwargs=["dim", "keepdim"]),
    OpEntry("argmin", _argmin_adapter, 1, method_name="argmin", free_fn_name="argmin",
            extra_kwargs=["dim", "keepdim"]),
    OpEntry("cumsum", _cumsum_adapter, 1, method_name="cumsum", free_fn_name="cumsum",
            extra_kwargs=["dim", "dtype"]),
    OpEntry("cumprod",_cumprod_adapter,1, method_name="cumprod",free_fn_name="cumprod",
            extra_kwargs=["dim", "dtype"]),
    OpEntry("cummax", _cummax_adapter, 1, method_name="cummax", free_fn_name="cummax",
            extra_kwargs=["dim"]),
    OpEntry("cummin", _cummin_adapter, 1, method_name="cummin", free_fn_name="cummin",
            extra_kwargs=["dim"]),
    OpEntry("erf",    _float_valued(_R.erf),    1, method_name="erf",    free_fn_name="erf"),
    OpEntry("erfinv", _float_valued(_R.erfinv), 1, method_name="erfinv", free_fn_name="erfinv"),
    OpEntry("trace",  _R.trace,  1, method_name="trace",  free_fn_name="trace"),

    # ── shape / layout ─────────────────────────────────────────────────────
    OpEntry("reshape",    A._reshape_adapter, 1, method_name="reshape",    free_fn_name="reshape"),
    OpEntry("squeeze",    A._squeeze_adapter, 1, method_name="squeeze",    free_fn_name="squeeze",
            extra_kwargs=["dim"]),
    OpEntry("squeeze_all",_R.squeeze_all,1, method_name="squeeze_all"),
    OpEntry("unsqueeze",  _R.unsqueeze,  1, method_name="unsqueeze",  free_fn_name="unsqueeze",
            extra_kwargs=["dim"]),
    OpEntry("flatten",    A._flatten_adapter, 1, method_name="flatten", free_fn_name="flatten",
            extra_kwargs=["start_dim", "end_dim"]),
    OpEntry("permute",    A._permute_adapter, 1, method_name="permute",    free_fn_name="permute"),
    OpEntry("transpose",  A._transpose_adapter, 1, method_name="transpose",  free_fn_name="transpose"),
    OpEntry("swapaxes",   A._swapaxes_adapter, 1, method_name="swapaxes",
            extra_kwargs=["axis0", "axis1"]),
    # broadcast_to is expand without the -1: a view on the CPU, as expand is.
    OpEntry("broadcast_to", A._broadcast_to_adapter, 1, method_name="broadcast_to",
            free_fn_name="broadcast_to", extra_kwargs=["size"]),
    OpEntry("expand",     A._expand_adapter, 1, method_name="expand",     free_fn_name="expand",
            extra_kwargs=["size", "implicit"]),
    OpEntry("as_strided", A._as_strided_adapter, 1, method_name="as_strided", free_fn_name="as_strided"),
    OpEntry("expand_dims",_R.expand_dims,1, method_name="expand_dims",
            extra_kwargs=["dim"]),
    # ``lucid.repeat(x, repeats, dim=None)`` — interleave semantics.  No
    # ``method_name`` because Tensor.repeat (below) follows the reference
    # framework's ``Tensor.repeat`` instead, which tiles copies.
    OpEntry("repeat",     A._repeat_adapter, 1, method_name=None,         free_fn_name="repeat",
            extra_kwargs=["dim"]),
    # ``Tensor.repeat(*sizes)`` — tile copies (separate semantics from the
    # free function above; see ``A._repeat_method_adapter``).
    OpEntry("repeat_method", A._repeat_method_adapter, 1,
            method_name="repeat", free_fn_name=None),
    OpEntry("tile",       A._tile_adapter, 1, method_name="tile",     free_fn_name="tile",
            extra_kwargs=["dims"]),
    OpEntry("flip",       A._flip_adapter,   1, method_name="flip",   free_fn_name="flip",
            extra_kwargs=["dims"]),
    OpEntry("fliplr",     A._fliplr_adapter, 1, method_name="fliplr", free_fn_name="fliplr"),
    OpEntry("flipud",     A._flipud_adapter, 1, method_name="flipud", free_fn_name="flipud"),
    OpEntry("roll",       A._roll_adapter, 1, method_name="roll",     free_fn_name="roll",
            extra_kwargs=["shifts", "dims"]),
    OpEntry("tril",       A._tril_adapter, 1, method_name="tril",     free_fn_name="tril",
            extra_kwargs=["diagonal"]),
    OpEntry("triu",       A._triu_adapter, 1, method_name="triu",     free_fn_name="triu",
            extra_kwargs=["diagonal"]),
    # No ``pad``: the reference has neither ``lucid.pad`` nor ``Tensor.pad``,
    # and ``nn.functional.pad`` is the one owner of every mode (LCD-473).

    # ── index / gather ─────────────────────────────────────────────────────
    # gather(input, dim, index), the reference order.  The engine takes
    # (a, indices, dim); the adapter reorders.
    OpEntry("gather",    A._gather_adapter, 1, method_name="gather", free_fn_name="gather",
            extra_kwargs=["dim", "index", "sparse_grad"]),
    OpEntry("sort",      A._sort_adapter,    1, method_name="sort",      free_fn_name="sort",
            extra_kwargs=["dim", "descending", "stable"]),
    OpEntry("argsort",   A._argsort_adapter, 1, method_name="argsort",   free_fn_name="argsort",
            extra_kwargs=["dim", "descending", "stable"]),
    OpEntry("nonzero",   A._nonzero_adapter, 1, method_name="nonzero", free_fn_name="nonzero"),
    OpEntry("unique",    A._unique_adapter,  1, method_name="unique",  free_fn_name="unique",
            extra_kwargs=["sorted", "return_inverse", "return_counts", "dim"]),
    OpEntry("unique_consecutive", A._unique_consecutive_adapter, 1,
            method_name="unique_consecutive", free_fn_name="unique_consecutive",
            extra_kwargs=["return_inverse", "return_counts", "dim"]),
    OpEntry("topk",      A._topk_adapter, 1, method_name="topk",      free_fn_name="topk",
            extra_kwargs=["k", "dim", "largest", "sorted"]),
    OpEntry("diagonal",  _R.diagonal,  1, method_name="diagonal",  free_fn_name="diagonal",
            extra_kwargs=["offset", "dim1", "dim2"]),

    # ── comparison ────────────────────────────────────────────────────────
    OpEntry("not_equal",     A._make_arith_adapter(_R.not_equal),     2, method_name="not_equal",     free_fn_name="not_equal"),
    OpEntry("greater",       A._make_arith_adapter(_R.greater),       2, method_name="greater",       free_fn_name="greater"),
    OpEntry("greater_equal", A._make_arith_adapter(_R.greater_equal), 2, method_name="greater_equal", free_fn_name="greater_equal"),
    OpEntry("less",          A._make_arith_adapter(_R.less),          2, method_name="less",          free_fn_name="less"),
    OpEntry("less_equal",    A._make_arith_adapter(_R.less_equal),    2, method_name="less_equal",    free_fn_name="less_equal"),

    # ── masking ────────────────────────────────────────────────────────────
    # ``where`` and ``masked_fill`` auto-cast their condition/mask to bool to
    # match reference behaviour, so they need adapters rather than direct
    # engine bindings.
    OpEntry("where",       A._where_adapter, 0, free_fn_name="where"),
    OpEntry("masked_fill", A._masked_fill_adapter, 1,
            method_name="masked_fill", free_fn_name="masked_fill",
            extra_kwargs=["value"]),

    # ── joining ────────────────────────────────────────────────────────────
    # The engine joins tensors of one dtype only; the join adapter brings
    # mixed inputs to their common dtype first, as the reference does.
    OpEntry("concatenate", A._make_join_adapter(_R.concatenate), -1, free_fn_name="cat",
            extra_kwargs=["dim"]),
    OpEntry("stack",       A._make_join_adapter(_R.stack),       -1, free_fn_name="stack",
            extra_kwargs=["dim"]),
    OpEntry("hstack",      A._make_join_adapter(_R.hstack),      -1, free_fn_name="hstack"),
    OpEntry("vstack",      A._make_join_adapter(_R.vstack),      -1, free_fn_name="vstack"),
    # The reference's method names the sizes ``split_size``, its free
    # function ``split_size_or_sections``: one entry each.
    OpEntry("split",       A._split_adapter, 1,  method_name=None,         free_fn_name="split",
            extra_kwargs=["dim"]),
    OpEntry("split_method", A._split_method_adapter, 1, method_name="split", free_fn_name=None,
            extra_kwargs=["dim"]),
    OpEntry("chunk",       A._chunk_adapter, 1, method_name="chunk",      free_fn_name="chunk",
            extra_kwargs=["chunks", "dim"]),
    OpEntry("unbind",      _R.unbind,      1,  method_name="unbind",     free_fn_name="unbind",
            extra_kwargs=["dim"]),
    OpEntry("meshgrid",    A._meshgrid_adapter, 0, free_fn_name="meshgrid",
            extra_kwargs=["indexing"]),

    # ── softmax / log_softmax (have dim and dtype kwargs) ──────────────────
    OpEntry("softmax",     _softmax_adapter,     1, method_name="softmax",     free_fn_name="softmax",
            extra_kwargs=["dim", "dtype"]),
    OpEntry("log_softmax", _log_softmax_adapter, 1, method_name="log_softmax", free_fn_name="log_softmax",
            extra_kwargs=["dim", "dtype"]),

    # ── rsqrt / std (engine-native) ────────────────────────────────────────
    OpEntry("rsqrt", _float_valued(_R.rsqrt), 1, method_name="rsqrt", free_fn_name="rsqrt"),
    OpEntry("std",   _std_adapter,   1, method_name="std",   free_fn_name="std",
            extra_kwargs=["dim", "keepdim", "correction", "unbiased"]),

    # ── boolean reductions ─────────────────────────────────────────────────
    OpEntry("any", _any_adapter, 1, method_name="any", free_fn_name="any",
            extra_kwargs=["dim", "keepdim"]),
    OpEntry("all", _all_adapter, 1, method_name="all", free_fn_name="all",
            extra_kwargs=["dim", "keepdim"]),

    # ── linear algebra ─────────────────────────────────────────────────────
    OpEntry("tensordot",  A._tensordot_adapter, 2, free_fn_name="tensordot",
            extra_kwargs=["dims"]),
    OpEntry("clip",       A._clip_adapter, 1, method_name="clip",    free_fn_name="clip",
            extra_kwargs=["min", "max"]),

    # ── floating-point predicates (output is always bool) ──────────────────
    OpEntry("isinf",      _R.isinf,     1, method_name="isinf",     free_fn_name="isinf"),
    OpEntry("isnan",      _R.isnan,     1, method_name="isnan",     free_fn_name="isnan"),
    OpEntry("isfinite",   _R.isfinite,  1, method_name="isfinite",  free_fn_name="isfinite"),
    OpEntry("nan_to_num", _R.nan_to_num,1, method_name="nan_to_num",free_fn_name="nan_to_num",
            extra_kwargs=["nan", "posinf", "neginf"]),

    # ── tensor lifecycle ────────────────────────────────────────────────────
    # detach: deep-copy without gradient tracking (uses contiguous + clone_with_grad).
    OpEntry("detach",      A._detach_adapter, 1, method_name="detach", free_fn_name="detach"),
    # clone: deep-copy preserving autograd history (contiguous = storage copy).
    OpEntry("clone",       _R.contiguous,   1, method_name="clone",  free_fn_name="clone"),
    # clamp is an alias for clip (same signature, same engine op).
    OpEntry("clamp",       A._clip_adapter, 1, method_name="clamp",  free_fn_name="clamp",
            extra_kwargs=["min", "max"]),
    OpEntry("clamp_min",   A._clamp_min_adapter, 1, method_name="clamp_min", free_fn_name="clamp_min",
            extra_kwargs=["min"]),
    OpEntry("clamp_max",   A._clamp_max_adapter, 1, method_name="clamp_max", free_fn_name="clamp_max",
            extra_kwargs=["max"]),
    # scatter_add: arg order differs — adapter reorders before calling engine.
    # Python call — scatter_add(x, dim, index, src)
    # Engine call — scatter_add(base, indices, src, dim)
    # n_tensor_args=1 auto-unwraps x; the adapter manually unwraps index/src.
    OpEntry("scatter_add", A._scatter_add_adapter, 1,
            method_name="scatter_add", free_fn_name="scatter_add"),

    # ══ composite ops (impl in _C/ops/composite/) ═══════════════════════════
    # Composition wrappers built atop primitives.  Autograd flows through the
    # underlying primitive backward nodes; no new schemas are registered.

    # ── elementwise math compositions (unary) ───────────────────────────────
    OpEntry("log10", _float_valued(_R.log10), 1, method_name="log10", free_fn_name="log10"),
    OpEntry("log1p", _float_valued(_R.log1p, engine_promotes=False), 1, method_name="log1p", free_fn_name="log1p"),
    OpEntry("exp2",  _float_valued(_R.exp2, engine_promotes=False),  1, method_name="exp2",  free_fn_name="exp2"),
    OpEntry("trunc", _R.trunc, 1, method_name="trunc", free_fn_name="trunc"),
    OpEntry("frac",  _R.frac,  1, method_name="frac",  free_fn_name="frac"),

    # ── elementwise math compositions (binary) ──────────────────────────────
    OpEntry("atan2",     A._make_arith_adapter(_R.atan2),     2, method_name="atan2",     free_fn_name="atan2"),
    OpEntry("fmod",      A._make_arith_adapter(_R.fmod),      2, method_name="fmod",      free_fn_name="fmod"),
    OpEntry("remainder", A._make_arith_adapter(_R.remainder), 2, method_name="remainder", free_fn_name="remainder"),
    OpEntry("hypot",     A._make_arith_adapter(_R.hypot),     2, method_name="hypot",     free_fn_name="hypot"),
    OpEntry("logaddexp", A._make_arith_adapter(_R.logaddexp), 2, method_name="logaddexp", free_fn_name="logaddexp"),
    OpEntry("nextafter", A._make_arith_adapter(_R.nextafter), 2, method_name="nextafter", free_fn_name="nextafter"),

    # ── reduction compositions ──────────────────────────────────────────────
    OpEntry("logsumexp", _logsumexp_adapter, 1,
            method_name="logsumexp", free_fn_name="logsumexp"),

    # ── linear-algebra compositions ─────────────────────────────────────────
    OpEntry("mm",   _R.mm,   2, method_name="mm",   free_fn_name="mm"),
    OpEntry("bmm",  _R.bmm,  2, method_name="bmm",  free_fn_name="bmm"),
    OpEntry("kron", A._kron_adapter, 2, method_name="kron", free_fn_name="kron"),

    # ── logical compositions ────────────────────────────────────────────────
    OpEntry("logical_and", A._make_arith_adapter(_R.logical_and), 2,
            method_name="logical_and", free_fn_name="logical_and"),
    OpEntry("logical_or",  A._make_arith_adapter(_R.logical_or), 2,
            method_name="logical_or",  free_fn_name="logical_or"),
    OpEntry("logical_xor", A._make_arith_adapter(_R.logical_xor), 2,
            method_name="logical_xor", free_fn_name="logical_xor"),
    OpEntry("logical_not", _R.logical_not, 1,
            method_name="logical_not", free_fn_name="logical_not"),

    # ── indexing compositions ───────────────────────────────────────────────
    OpEntry("take",         A._take_adapter, 1,
            method_name="take", free_fn_name="take"),
    OpEntry("index_select", A._index_select_adapter, 1,
            method_name="index_select", free_fn_name="index_select"),
    OpEntry("narrow",       A._narrow_adapter, 1,
            method_name="narrow", free_fn_name="narrow"),
    OpEntry("scatter",      A._scatter_adapter, 1,
            method_name="scatter", free_fn_name="scatter",
            extra_kwargs=["reduce"]),
    OpEntry("kthvalue",     A._kthvalue_adapter, 1,
            method_name="kthvalue", free_fn_name="kthvalue"),

    # ── layout compositions ─────────────────────────────────────────────────
    OpEntry("movedim",   A._movedim_adapter, 1,
            method_name="movedim", free_fn_name="movedim"),
    OpEntry("unflatten", A._unflatten_adapter, 1,
            method_name="unflatten", free_fn_name="unflatten"),

    # ── stats / search compositions ─────────────────────────────────────────
    OpEntry("histc",         A._histc_adapter, 1,
            method_name="histc", free_fn_name="histc"),
    OpEntry("cartesian_prod", A._cartesian_prod_adapter, 0,
            method_name=None, free_fn_name="cartesian_prod"),
    OpEntry("searchsorted",  A._searchsorted_adapter, 2,
            method_name=None, free_fn_name="searchsorted",
            extra_kwargs=["out_int32", "right", "side", "sorter"]),
    OpEntry("bucketize",     A._bucketize_adapter, 2,
            method_name=None, free_fn_name="bucketize",
            extra_kwargs=["out_int32", "right"]),

    # ══ aliases of existing primitives (no new C++ — share engine kernels) ══

    # Comparison short names — same kernels as equal/not_equal/less/...
    OpEntry("eq", A._make_arith_adapter(_R.eq), 2, method_name="eq", free_fn_name="eq"),
    OpEntry("ne", A._make_arith_adapter(_R.ne), 2, method_name="ne", free_fn_name="ne"),
    OpEntry("lt", A._make_arith_adapter(_R.lt), 2, method_name="lt", free_fn_name="lt"),
    OpEntry("le", A._make_arith_adapter(_R.le), 2, method_name="le", free_fn_name="le"),
    OpEntry("gt", A._make_arith_adapter(_R.gt), 2, method_name="gt", free_fn_name="gt"),
    OpEntry("ge", A._make_arith_adapter(_R.ge), 2, method_name="ge", free_fn_name="ge"),

    # Trig short names — share kernels with arcsin/arccos/arctan.
    OpEntry("asin", _float_valued(_R.asin), 1, method_name="asin", free_fn_name="asin"),
    OpEntry("acos", _float_valued(_R.acos), 1, method_name="acos", free_fn_name="acos"),
    OpEntry("atan", _float_valued(_R.atan), 1, method_name="atan", free_fn_name="atan"),

    # ``bitwise_not`` is the reference framework's name for ``invert``.
    OpEntry("bitwise_not", _R.bitwise_not, 1,
            method_name="bitwise_not", free_fn_name="bitwise_not"),
    OpEntry("bitwise_and", A._make_arith_adapter(_R.bitwise_and), 2,
            method_name="bitwise_and", free_fn_name="bitwise_and"),
    OpEntry("bitwise_or",  A._make_arith_adapter(_R.bitwise_or),  2,
            method_name="bitwise_or",  free_fn_name="bitwise_or"),
    OpEntry("bitwise_xor", A._make_arith_adapter(_R.bitwise_xor), 2,
            method_name="bitwise_xor", free_fn_name="bitwise_xor"),
    OpEntry("bitwise_left_shift", A._make_arith_adapter(_R.bitwise_left_shift), 2,
            method_name="bitwise_left_shift", free_fn_name="bitwise_left_shift"),
    OpEntry("bitwise_right_shift", A._make_arith_adapter(_R.bitwise_right_shift), 2,
            method_name="bitwise_right_shift", free_fn_name="bitwise_right_shift"),

    # masked_select: engine kernel returns a flat 1-D tensor of selected
    # elements.  Both inputs are tensors so n_tensor_args=2.
    OpEntry("masked_select", A._masked_select_adapter, 2,
            method_name="masked_select", free_fn_name="masked_select"),

    # isclose: composite op with rtol/atol/equal_nan keyword arguments.
    # n_tensor_args=2 so both operands get unwrapped before the adapter runs.
    OpEntry("isclose", A._isclose_adapter, 2,
            method_name="isclose", free_fn_name="isclose",
            extra_kwargs=["rtol", "atol", "equal_nan"]),

    # repeat_interleave: thin wrapper around engine.repeat that supports
    # ``dim=None`` (flatten first) like the reference framework's API.
    OpEntry("repeat_interleave", A._repeat_interleave_adapter, 1,
            method_name="repeat_interleave", free_fn_name="repeat_interleave",
            extra_kwargs=["repeats", "dim", "output_size"]),


    # Shape aliases.
    OpEntry("view",   A._view_adapter,   1,
            method_name="view", free_fn_name="view"),
    OpEntry("concat", A._concat_adapter, -1,
            method_name=None, free_fn_name="concat"),

    # No `cross` / `norm` / `einsum` top-level forwarders.  Lucid's API tree
    # gives each op exactly one canonical path: linalg ops live only under
    # ``lucid.linalg.*`` and einops-style ops only under ``lucid.einops.*``.
    # Adding shortcut aliases here would re-introduce the duplication this
    # policy is meant to avoid.

    # ── transpose shorthand ─────────────────────────────────────────────────
    # ``Tensor.t()`` is the standard 2-D transpose — same engine kernel as
    # the ``T`` property, exposed here as a method so registry-driven method
    # injection picks it up alongside everything else.
    OpEntry("t", _R.T, 1, method_name="t", free_fn_name=None),

    # ── complex viewing (engine ops, dual backend) ─────────────────────────
    # Each backend implements these natively: CPU walks the interleaved
    # ``[re, im, ...]`` storage; GPU dispatches to ``mlx::core::real`` /
    # ``imag`` / ``conjugate``.  ``complex`` builds a C64 from two F32 inputs.
    OpEntry("real",    _real,      1, method_name=None, free_fn_name="real"),
    OpEntry("imag",    _R.imag,    1, method_name=None, free_fn_name="imag"),
    OpEntry("complex", A._complex_adapter, 2, method_name=None, free_fn_name="complex"),
    OpEntry("conj",    _R.conj,    1, method_name="conj", free_fn_name="conj"),
]
# fmt: on
