"""
Lucid — Apple Silicon ML framework.
"""

from collections.abc import Callable as _Callable

# ── Save Python builtins before dtype aliases shadow them ────────────────────
_py_int = int
_py_float = float
_py_bool = bool

from lucid.version import __version__, _EXPECTED_ABI
from lucid._C import engine as _C_engine

try:
    _abi = _py_int(_C_engine.ABI_VERSION)
    if _abi != _EXPECTED_ABI:
        raise RuntimeError(
            f"lucid C++ engine ABI mismatch: expected {_EXPECTED_ABI}, "
            f"got {_abi}. The native extension is stale for this Python source. "
            "From the repository root, rebuild in the project environment with: "
            "uv pip install --python .venv/bin/python -e . --no-build-isolation. "
            "Diagnose the environment with: .venv/bin/python -m tools.doctor"
        )
except (TypeError, ValueError):  # fmt: skip
    pass  # Mocked during docs build — skip ABI check

# fmt: off
from lucid._dtype import (
    dtype,
    float16,
    float32,
    float64,
    bfloat16,
    int8,
    int16,
    int32,
    int64,
    bool_,
    complex64,
    complex128,
    # Existing aliases kept for backward compat
    half,
    double,
    short,
    long,
    # New API-compatible module-level aliases
    # lucid.float / lucid.int / lucid.bool shadow Python builtins
    # only as module attributes — no impact on user code
    float32 as float,   # lucid.float  == float32  == float32
    int32   as int,     # lucid.int    == int32    == int32
    bool_   as bool,    # lucid.bool   == bool_   == bool_
)
from lucid._device import device
from lucid._dtype import finfo, iinfo
from lucid._globals import (
    set_default_dtype,
    get_default_dtype,
    set_default_device,
    get_default_device,
)
from lucid._threads import (
    set_num_threads,
    get_num_threads,
    set_num_interop_threads,
    get_num_interop_threads,
)


# ── Determinism (top-level reference-framework names) ───────────────────────
# The engine already exposes ``set_deterministic`` / ``is_deterministic``;
# wrap them under the standard ``use_deterministic_algorithms`` /
# ``are_deterministic_algorithms_enabled`` names without an alias hop.
def use_deterministic_algorithms(mode: _py_bool, *, warn_only: _py_bool = False) -> None:
    """Globally toggle deterministic kernel selection.

    Lucid does not ship a separate ``warn_only`` mode — callers either
    get deterministic kernels or they don't.  The kwarg is accepted for
    API compatibility and ignored.

    Parameters
    ----------
    mode : bool
        ``True`` enables deterministic kernels (engine prefers
        reproducible paths for scatter, atomic reductions, some MLX
        kernels) at a small throughput cost.  ``False`` re-enables the
        fastest available path.
    warn_only : bool, optional
        Accepted for cross-framework API parity; ignored — Lucid has no
        "warn but continue" tier.  Default ``False``.
    """
    # ``bool`` here resolves to ``lucid.bool`` (the dtype alias) — use the
    # builtin saved at module top.
    del warn_only  # reference-signature parity; Lucid has no warn-only tier
    _C_engine.set_deterministic(_py_bool(mode))


def are_deterministic_algorithms_enabled() -> _py_bool:
    """Return whether deterministic kernel selection is currently active.

    Reflects the engine-side flag toggled by
    :func:`use_deterministic_algorithms`.  When ``True`` the engine
    prefers reproducible kernels for non-deterministic ops (scatter,
    atomic reductions, some MLX paths) at a usually-small throughput
    cost.

    Returns
    -------
    bool
        Current deterministic-mode flag from the C++ engine.
    """
    return _py_bool(_C_engine.is_deterministic())

# ── Public API ────────────────────────────────────────────────────────────────
# Organised by category — mirrors the standard tensor framework surface.
# Invariant: no Tier-2 symbols (Module, Parameter, Linear, Adam, ...) here.

__all__ = [
    # ── metadata ──────────────────────────────────────────────────────────
    "__version__",
    # ── dtype ─────────────────────────────────────────────────────────────
    "dtype", "dtypes",
    "float16", "bfloat16", "float32", "float64",
    "int8", "int16", "int32", "int64",
    "bool_", "complex64", "complex128",
    "half", "double", "short", "long",   # aliases (excluded from import *)
    # ── device / defaults ─────────────────────────────────────────────────
    "device",
    "set_default_dtype", "get_default_dtype",
    "set_default_device", "get_default_device",
    # ── dtype info ────────────────────────────────────────────────────────
    "finfo", "iinfo",
    # ── threading (advisory — see lucid/_threads.py) ──────────────────────
    "set_num_threads", "get_num_threads",
    "set_num_interop_threads", "get_num_interop_threads",
    # ── determinism ───────────────────────────────────────────────────────
    "use_deterministic_algorithms", "are_deterministic_algorithms_enabled",
    # ── core tensor ───────────────────────────────────────────────────────
    "Tensor", "eval",
    # ── factory — deterministic ───────────────────────────────────────────
    "tensor", "as_tensor", "from_numpy", "from_dlpack", "to_dlpack",
    "zeros", "ones", "empty", "empty_strided", "full", "eye", "arange", "linspace", "logspace",
    "zeros_like", "ones_like", "empty_like", "full_like",
    # ── factory — random ──────────────────────────────────────────────────
    "rand", "randn", "randint", "bernoulli", "normal",
    "rand_like", "randn_like", "randint_like", "manual_seed", "randperm",
    "Generator", "seed", "initial_seed", "get_rng_state", "set_rng_state",
    # ── ops — unary ───────────────────────────────────────────────────────
    "neg", "sign",
    "exp", "exp2", "log", "log2", "log10", "log1p",
    "sqrt", "square", "reciprocal", "rsqrt",
    "floor", "ceil", "trunc", "frac",
    "sin", "cos", "tan", "arcsin", "arccos", "arctan",
    "asin", "acos", "atan",
    "sinh", "cosh", "tanh",
    "relu", "sigmoid", "selu", "softmax", "log_softmax",
    "clip", "clamp", "clamp_min", "clamp_max",
    "isinf", "isnan", "isfinite", "nan_to_num",
    # ── ops — binary ──────────────────────────────────────────────────────
    "add", "sub", "mul", "div",
    "matmul", "mm", "bmm", "tensordot", "kron",
    "atan2", "fmod", "remainder", "hypot", "logaddexp", "nextafter",
    "maximum", "minimum",
    "equal", "not_equal", "greater", "greater_equal", "less", "less_equal",
    "eq", "ne", "lt", "le", "gt", "ge", "isclose",
    "logical_and", "logical_or", "logical_xor", "logical_not",
    "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not",
    "bitwise_left_shift", "bitwise_right_shift",
    # ── ops — reduction ───────────────────────────────────────────────────
    "mean", "prod",
    "argmax", "argmin", "cumsum", "cumprod",
    "std", "var", "trace",
    "logsumexp", "logcumsumexp", "aminmax", "renorm", "trapezoid", "cumulative_trapezoid",
    # ── tensor manipulation ───────────────────────────────────────────────
    "reshape", "view", "permute", "transpose", "unsqueeze", "squeeze", "flatten",
    "unflatten", "narrow", "select", "movedim",
    "expand", "broadcast_to", "as_strided", "repeat", "repeat_interleave", "tile",
    "cat", "concat", "stack", "hstack", "vstack",
    "split", "chunk", "unbind",
    "gather", "scatter", "scatter_add", "take", "index_select", "masked_select",
    "kthvalue",
    "where", "masked_fill", "roll", "flip", "fliplr", "flipud",
    "sort", "argsort", "topk",
    "nonzero", "unique", "unique_consecutive", "meshgrid",
    "tril", "triu",
    "contiguous", "detach", "clone",
    # ── stats / search ────────────────────────────────────────────────────
    "searchsorted", "bucketize", "histc", "cartesian_prod",
    # ── spectral (window math lives in lucid.signal.windows) ──────────────
    "stft", "istft",
    "hann_window", "hamming_window", "blackman_window", "bartlett_window",
    "kaiser_window",
    # ── Tensor-method ops surfaced as free functions ──────────────────────
    "lerp", "diff",
    # ── extras: constants ─────────────────────────────────────────────────
    "pi", "e", "inf", "nan", "newaxis",
    # ── extras: aliases / dtype promotion ─────────────────────────────────
    "absolute", "negative", "positive",
    "subtract", "multiply", "divide", "true_divide", "rsub",
    "arccosh", "acosh", "arcsinh", "asinh", "arctanh", "atanh", "arctan2",
    "swapaxes", "swapdims", "moveaxis", "adjoint", "t",
    "real", "imag", "angle", "polar",
    "view_as_real", "view_as_complex",
    "conj", "conj_physical", "resolve_conj", "resolve_neg",
    "result_type", "promote_types", "can_cast",
    # ── extras: special elementwise ───────────────────────────────────────
    "expm1", "sinc", "heaviside", "xlogy", "logit", "signbit",
    "float_power", "fmax", "fmin",
    # ── extras: nan-safe reductions ───────────────────────────────────────
    "nansum", "nanmean", "median", "nanmedian", "mode", "msort", "quantile", "nanquantile",
    # ── extras: BLAS variants ─────────────────────────────────────────────
    "addmm", "addbmm", "baddbmm", "addmv", "addr", "addcmul", "addcdiv",
    "mv", "ger", "vdot", "block_diag",
    # ── extras: shape ─────────────────────────────────────────────────────
    "broadcast_shapes", "broadcast_tensors",
    "column_stack", "row_stack", "dstack",
    "atleast_1d", "atleast_2d", "atleast_3d",
    "vsplit", "hsplit", "dsplit", "tensor_split",
    "take_along_dim", "rot90",
    # ── extras: predicates / introspection ────────────────────────────────
    "numel", "is_storage", "is_nonzero", "is_same_size", "is_neg", "is_conj",
    "isin", "isneginf", "isposinf", "isreal",
    # ── grad control ──────────────────────────────────────────────────────
    "no_grad", "enable_grad", "is_grad_enabled", "set_grad_enabled", "inference_mode",
    # ── type predicates ───────────────────────────────────────────────────
    "is_tensor", "is_floating_point", "is_complex", "is_signed",
    # ── serialization ─────────────────────────────────────────────────────
    "save", "load", "save_sharded", "load_sharded",
    "save_safetensors", "load_safetensors",
    # ── ops reachable via lazy __getattr__ but previously missing from
    # __all__ / dir() (IDE autocomplete + ``from lucid import *``) ────────
    "abs", "all", "any", "complex", "cummax", "cummin", "diagonal",
    "erf", "erfinv", "max", "min", "pow", "ravel", "round", "sum",
    # ── composites the module served but __all__ omitted (LCD-641) ───────
    "floor_divide", "copysign", "ldexp", "frexp", "gcd", "lcm",
    "digamma", "erfc", "i0", "lgamma",
    "amax", "amin", "std_mean", "var_mean", "count_nonzero", "allclose",
    "argwhere", "put", "index_add", "index_copy", "index_fill",
    "index_put", "index_put_", "masked_scatter", "scatter_reduce",
    "bincount", "histogram", "histogram2d", "histogramdd",
    "corrcoef", "cov", "cdist", "combinations",
    "tril_indices", "triu_indices", "diag_embed", "logdet",
    "multinomial", "poisson",
    # ── subpackages ───────────────────────────────────────────────────────
    "nn", "optim", "autograd", "func", "linalg", "fft", "signal", "special",
    "diffeq", "utils", "amp", "profiler", "einops", "compile", "quantization",
    "coreml", "peft",
    "metal", "backends", "test", "weights",
    # ── public type aliases ───────────────────────────────────────────────
    "Scalar", "TensorLike", "DeviceLike", "DTypeLike", "ShapeLike",
    "StateDict", "TensorOrScalar",
    "HasShape", "SupportsNumpyConversion", "SupportsGrad", "TensorLikeProtocol",
]

# ── Name sets used by the dispatch table ─────────────────────────────────────
_FACTORY_NAMES: frozenset[str] = frozenset([
    # ── deterministic ─────────────────────────────────────────────────────
    "tensor", "as_tensor", "from_numpy", "from_dlpack", "to_dlpack",
    "zeros", "ones", "empty", "empty_strided", "full", "eye", "arange", "linspace", "logspace",
    "zeros_like", "ones_like", "empty_like", "full_like",
    # ── random ────────────────────────────────────────────────────────────
    "rand", "randn", "randint", "bernoulli", "normal",
    "rand_like", "randn_like", "randint_like", "manual_seed", "randperm",
    "Generator", "seed", "initial_seed", "get_rng_state", "set_rng_state",
])

_SCATTER_NAMES: frozenset[str] = frozenset(["scatter_add"])

_OPS_NAMES: frozenset[str] = frozenset([
    # ── unary math ────────────────────────────────────────────────────────
    "abs", "neg", "sign",
    "exp", "exp2", "log", "log2", "log10", "log1p",
    "sqrt", "square", "reciprocal", "rsqrt",
    "floor", "ceil", "round", "trunc", "frac",
    "sin", "cos", "tan", "arcsin", "arccos", "arctan", "asin", "acos", "atan",
    "sinh", "cosh", "tanh",
    "relu", "sigmoid", "selu", "softmax", "log_softmax",
    "clip", "clamp", "clamp_min", "clamp_max",
    "isinf", "isnan", "isfinite", "nan_to_num",
    "erf", "erfinv",
    # ── binary math ───────────────────────────────────────────────────────
    "add", "sub", "mul", "div", "pow",
    "atan2", "fmod", "remainder", "floor_divide", "hypot", "logaddexp", "nextafter",
    "maximum", "minimum",
    # ── linear algebra ────────────────────────────────────────────────────
    "matmul", "mm", "bmm", "tensordot", "kron",
    # ── complex viewing (engine ops) ──────────────────────────────────────
    "real", "imag", "complex", "conj",
    # ── comparison ────────────────────────────────────────────────────────
    "not_equal", "greater", "greater_equal", "less", "less_equal",
    "eq", "ne", "lt", "le", "gt", "ge", "isclose",
    # ── logical / bitwise ─────────────────────────────────────────────────
    "logical_and", "logical_or", "logical_xor", "logical_not",
    "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not",
    "bitwise_left_shift", "bitwise_right_shift",
    # ── reduction ─────────────────────────────────────────────────────────
    "sum", "mean", "max", "min", "prod",
    "argmax", "argmin", "cumsum", "cumprod", "cummax", "cummin",
    "std", "var", "trace", "any", "all", "logsumexp",
    # ── shape / view ──────────────────────────────────────────────────────
    "reshape", "view", "permute", "transpose", "unsqueeze", "squeeze", "flatten",
    "unflatten", "narrow", "movedim",
    "expand", "broadcast_to", "as_strided", "repeat", "repeat_interleave", "tile",
    "cat", "concat", "stack", "hstack", "vstack",
    "split", "chunk", "unbind",
    "ravel", "diagonal", "tril", "triu",
    "flip", "fliplr", "flipud", "roll",
    "contiguous", "detach", "clone",
    # ── indexing / scatter ────────────────────────────────────────────────
    "gather", "scatter", "scatter_add",
    "take", "index_select", "masked_select",
    "where", "masked_fill",
    # ── sort / search ─────────────────────────────────────────────────────
    "sort", "argsort", "topk", "kthvalue",
    "nonzero", "unique", "unique_consecutive", "meshgrid",
    "searchsorted", "bucketize", "histc", "cartesian_prod",
])

_GRAD_NAMES: frozenset[str] = frozenset([
    "no_grad", "enable_grad", "is_grad_enabled", "set_grad_enabled", "inference_mode",
])

_PREDICATE_NAMES: frozenset[str] = frozenset([
    "is_tensor", "is_floating_point", "is_complex", "is_signed",
])

_SERIALIZATION_NAMES: frozenset[str] = frozenset([
    "save", "load", "save_sharded", "load_sharded",
    "save_safetensors", "load_safetensors",
])

_SUBPKG_NAMES: frozenset[str] = frozenset([
    # ── core ML stack ─────────────────────────────────────────────────────
    "nn", "optim", "autograd", "func",
    # ── numerical sub-packages ────────────────────────────────────────────
    "linalg", "fft", "signal", "special", "distributions", "diffeq",
    # ── model transforms / deployment ─────────────────────────────────────
    "quantization", "coreml", "peft",
    # ── infra / tooling ───────────────────────────────────────────────────
    "utils", "amp", "profiler", "einops", "compile",
    "metal", "backends", "test", "weights",
])

_METHOD_ALIASES: frozenset[str] = frozenset(["lerp", "diff"])

_TYPE_ALIAS_NAMES: frozenset[str] = frozenset([
    "Scalar", "TensorLike", "DeviceLike", "DTypeLike", "ShapeLike",
    "StateDict", "TensorOrScalar",
    "HasShape", "SupportsNumpyConversion", "SupportsGrad", "TensorLikeProtocol",
])

# ── Lazy group loaders ────────────────────────────────────────────────────────
# Each loader is called at most once per interpreter session: it populates
# globals() for every name in the group, then is never called again.
#
# To add a new lazily-imported group:
#   1. Define its name frozenset above.
#   2. Write a _load_<group>() function below.
#   3. Register it via @_register(<frozenset>).
#
# fmt: on

_GROUP_LOADERS: dict[str, _Callable[[], dict[str, object]]] = {}


def _register(
    names: frozenset[str],
) -> _Callable[[_Callable[[], dict[str, object]]], _Callable[[], dict[str, object]]]:
    """Register a batch loader for a set of lazily-imported names."""

    def decorator(fn: _Callable[[], dict[str, object]]) -> _Callable[[], dict[str, object]]:
        """Bind ``fn`` as the loader for every name in the enclosing group."""
        for name in names:
            _GROUP_LOADERS[name] = fn
        return fn

    return decorator


@_register(_FACTORY_NAMES)
def _load_factories() -> dict[str, object]:
    """Lazy loader for tensor factory functions (zeros / ones / arange / …)."""
    import lucid._factories as _fac

    return {n: getattr(_fac, n) for n in _FACTORY_NAMES}


@_register(_OPS_NAMES)
def _load_ops() -> dict[str, object]:
    """Lazy loader for the core engine-backed op surface (add / matmul / …)."""
    import lucid._ops as _ops

    return {n: getattr(_ops, n) for n in _OPS_NAMES}


@_register(_SCATTER_NAMES)
def _load_scatter() -> dict[str, object]:
    """Lazy loader for ``scatter_add`` (split out so it imports independently)."""
    from lucid._ops import scatter_add

    return {"scatter_add": scatter_add}


@_register(_GRAD_NAMES)
def _load_grad() -> dict[str, object]:
    """Lazy loader for autograd mode helpers (``no_grad`` / ``enable_grad`` / …)."""
    from lucid.autograd._grad_mode import (
        no_grad,
        enable_grad,
        is_grad_enabled,
        set_grad_enabled,
        inference_mode,
    )

    return dict(
        no_grad=no_grad,
        enable_grad=enable_grad,
        is_grad_enabled=is_grad_enabled,
        set_grad_enabled=set_grad_enabled,
        inference_mode=inference_mode,
    )


@_register(_PREDICATE_NAMES)
def _load_predicates() -> dict[str, object]:
    """Lazy loader for dtype predicate helpers (``is_tensor`` / ``is_floating_point`` / …)."""
    from lucid import _dtype as _lucid_dtype

    # Derived from the dtype registry, not written out by hand.  The
    # hand-written sets were the whole bug: ``bfloat16`` and
    # ``complex128`` became engine dtypes of their own and nobody came
    # back here, so ``lucid.is_floating_point(x)`` said ``False`` where
    # ``x.is_floating_point()`` said ``True`` — and the disagreement was
    # load-bearing, since ``autograd/_functional.py`` gates on the free
    # function and so refused to differentiate a bfloat16 input.
    # Reading the registry means a future dtype cannot drift out again.
    _KINDS = {
        engine_dtype: (
            "complex"
            if d._name.startswith("complex")
            else (
                "float"
                if d._name.startswith(("float", "bfloat"))
                else ("int" if d._name.startswith("int") else "bool")
            )
        )
        for engine_dtype, d in _lucid_dtype._ENGINE_TO_DTYPE.items()
    }
    _FLOAT_DTYPES = frozenset(e for e, k in _KINDS.items() if k == "float")
    _COMPLEX_DTYPES = frozenset(e for e, k in _KINDS.items() if k == "complex")
    _SIGNED_DTYPES = frozenset(e for e, k in _KINDS.items() if k in ("float", "complex", "int"))

    def is_tensor(obj: object, /) -> _py_bool:
        """Return True if *obj* is a lucid Tensor."""
        from lucid._tensor.tensor import Tensor as _T

        return isinstance(obj, _T)

    def is_floating_point(input: object) -> _py_bool:
        """Return True if *input* has a floating-point dtype."""
        from lucid._dispatch import _unwrap

        return _unwrap(input).dtype in _FLOAT_DTYPES  # type: ignore[arg-type]

    def is_complex(input: object) -> _py_bool:
        """Return True if *input* has a complex dtype."""
        from lucid._dispatch import _unwrap

        return _unwrap(input).dtype in _COMPLEX_DTYPES  # type: ignore[arg-type]

    def is_signed(input: object) -> _py_bool:
        """Return True if *input* has a signed numeric dtype."""
        from lucid._dispatch import _unwrap

        return _unwrap(input).dtype in _SIGNED_DTYPES  # type: ignore[arg-type]

    return dict(
        is_tensor=is_tensor,
        is_floating_point=is_floating_point,
        is_complex=is_complex,
        is_signed=is_signed,
    )


@_register(_SERIALIZATION_NAMES)
def _load_serialization() -> dict[str, object]:
    """Lazy loader for ``save`` / ``load`` (sharded and safetensors variants)."""
    import lucid.serialization as _ser

    return {
        "save": _ser.save,
        "load": _ser.load,
        "save_sharded": _ser.save_sharded,
        "load_sharded": _ser.load_sharded,
        "save_safetensors": _ser.save_safetensors,
        "load_safetensors": _ser.load_safetensors,
    }


def _get_composite_names() -> frozenset[str]:
    """Lazily query ``lucid._ops.composite`` for its export set (avoids cycles)."""
    from lucid._ops.composite import COMPOSITE_NAMES

    return COMPOSITE_NAMES


# Pre-populate the loader registry for the composite ops surface.  The actual
# loader runs once on first attribute access.
_COMPOSITE_NAMES_CACHED: frozenset[str] = _get_composite_names()


@_register(_COMPOSITE_NAMES_CACHED)
def _load_composite_ops() -> dict[str, object]:
    """Pure-Python composites layered on engine primitives."""
    import lucid._ops.composite as _comp

    return {n: getattr(_comp, n) for n in _COMPOSITE_NAMES_CACHED}


@_register(_METHOD_ALIASES)
def _load_method_aliases() -> dict[str, object]:
    """Surface select Tensor methods as free functions (reference-framework parity).

    ``lerp(a, b, w)`` etc. are commonly used standalone — we wrap each
    method so the call ``lucid.lerp(a, b, w)`` dispatches to ``a.lerp(b, w)``.
    """
    import inspect

    from lucid._tensor.tensor import Tensor as _T

    def _make(method_name: str) -> _Callable[..., object]:
        """Build a free-function wrapper that forwards to ``Tensor.<method_name>``."""

        def _fn(input: object, *args: object, **kwargs: object) -> object:
            """Dispatch ``lucid.<method_name>(input, ...)`` to ``input.<method_name>(...)``."""
            if not isinstance(input, _T):
                raise TypeError(
                    f"lucid.{method_name}() expects a Tensor as the first "
                    f"argument, got {type(input).__name__}"
                )
            return getattr(input, method_name)(*args, **kwargs)

        _fn.__name__ = method_name
        _fn.__qualname__ = method_name
        _fn.__doc__ = (
            f"Free-function alias for ``Tensor.{method_name}``. "
            f"Equivalent to ``input.{method_name}(*args, **kwargs)``."
        )
        # The method's parameters, ``self`` read as ``input``: what the
        # forwarding body accepts, for ``inspect`` and ``help()``.
        own = inspect.signature(getattr(_T, method_name))
        first, *rest = own.parameters.values()
        _fn.__dict__["__signature__"] = own.replace(parameters=[first.replace(name="input"), *rest])
        return _fn

    return {n: _make(n) for n in _METHOD_ALIASES}


@_register(_TYPE_ALIAS_NAMES)
def _load_type_aliases() -> dict[str, object]:
    """Lazy loader for the public type aliases / protocols re-exported from :mod:`lucid._types`."""
    from lucid._types import (
        Scalar,
        TensorLike,
        DeviceLike,
        DTypeLike,
        ShapeLike,
        StateDict,
        TensorOrScalar,
        HasShape,
        SupportsNumpyConversion,
        SupportsGrad,
        TensorLikeProtocol,
    )

    return dict(
        Scalar=Scalar,
        TensorLike=TensorLike,
        DeviceLike=DeviceLike,
        DTypeLike=DTypeLike,
        ShapeLike=ShapeLike,
        StateDict=StateDict,
        TensorOrScalar=TensorOrScalar,
        HasShape=HasShape,
        SupportsNumpyConversion=SupportsNumpyConversion,
        SupportsGrad=SupportsGrad,
        TensorLikeProtocol=TensorLikeProtocol,
    )


# ── lucid.eval(*tensors) ──────────────────────────────────────────────────────

# Loaded by now — the composite ops above import it — and named by the
# annotations below, which PEP 649 reads from this module's globals.
from lucid._tensor.tensor import Tensor


def eval(*tensors: Tensor) -> None:
    """Compute the values of ``tensors`` now, in one evaluation.

    A tensor on Metal holds a lazy computation until something reads it.
    ``lucid.eval`` runs the pending work of every tensor it is given in a
    single batched evaluation, which lets the GPU schedule them together —
    cheaper than calling :meth:`Tensor.eval` on each in turn.  A tensor on
    the CPU is computed already, and passes through.

    Parameters
    ----------
    *tensors : Tensor
        The tensors to evaluate, on either device; none at all is a no-op.

    Raises
    ------
    TypeError
        An argument is not a :class:`Tensor` — an output object or a list
        of tensors must be unpacked, ``lucid.eval(*outputs)``.

    Notes
    -----
    The training-loop pattern on Metal evaluates the forward graph before
    backward and the updated parameters after the step, keeping each
    evaluation scope small; deferring both to one call after the step lets
    forward, backward and the optimizer accumulate into a single graph,
    which is much slower to schedule::

        loss.eval()
        loss.backward()
        optimizer.step()
        lucid.eval(*model.parameters())

    :func:`lucid.metal.synchronize` runs every pending tensor instead.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.ones(3, device="metal") * 2
    >>> b = a + 1
    >>> lucid.eval(a, b)
    >>> b.tolist()
    [3.0, 3.0, 3.0]
    """
    for t in tensors:
        if not isinstance(t, Tensor):
            raise TypeError(
                f"lucid.eval() takes tensors, got {type(t).__name__} — unpack "
                "a collection of tensors, lucid.eval(*tensors)"
            )
    if tensors:
        _C_engine.eval_tensors([t._impl for t in tensors])


# ── Module __getattr__ ────────────────────────────────────────────────────────


def __getattr__(name: str) -> object:
    """Lazy attribute resolver for the top-level :mod:`lucid` namespace.

    Implements PEP 562 lazy imports — names listed in the lazy-loader
    registries are not imported at package init time, but only on first
    attribute access (e.g. ``lucid.zeros`` triggers the factories loader).
    Subsequent accesses go through the regular module ``__dict__``.
    """
    _g = globals()

    # ``Tensor`` is bound above, with ``eval``; the lazily-imported groups
    # share one dispatch pattern.
    loader = _GROUP_LOADERS.get(name)
    if loader is not None:
        _g.update(loader())
        return _g[name]

    # Subpackages: import on demand and cache.
    if name in _SUBPKG_NAMES:
        import importlib
        import sys as _sys

        pkg_key = f"lucid.{name}"
        mod = _sys.modules.get(pkg_key) or importlib.import_module(pkg_key)
        _g[name] = mod
        return mod

    # dtypes namespace object (separate from the dtype class).
    if name == "dtypes":
        import lucid.dtypes as _dtypes

        _g["dtypes"] = _dtypes
        return _dtypes

    raise AttributeError(f"module 'lucid' has no attribute '{name}'")


def _bind_annotation_names() -> None:
    """Give every module the names its annotations mention.

    PEP 649 evaluates a function's annotations in its own module globals
    at the moment something asks for them, so a name imported only under
    ``TYPE_CHECKING`` makes ``inspect.signature`` raise ``NameError`` —
    what ``help()``, an editor's tooltip and the docs build all call.
    :mod:`lucid._annotations` binds those names once their
    sources exist, here for what ``import lucid`` loaded and for every
    ``lucid`` module imported after it.  Called when everything this
    package defines is in place.
    """
    from lucid._annotations import install

    install()


_bind_annotation_names()


def _attach_free_fn_methods() -> None:
    """Give ``Tensor`` the method form of the free functions that lacked one.

    ``x.expm1()`` is ``lucid.expm1(x)`` (:mod:`lucid._tensor._free_fn_methods`).
    The methods are bound to what ``lucid.<name>`` resolves to and take its
    signature with the annotations evaluated, which holds only from here on:
    after ``Tensor``, the lazy loaders and the annotation names above.
    """
    from lucid._tensor._free_fn_methods import inject_free_fn_methods

    inject_free_fn_methods(Tensor)


_attach_free_fn_methods()
