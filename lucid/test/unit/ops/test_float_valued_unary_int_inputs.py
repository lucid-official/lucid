"""A float-valued unary op takes a bool or integer tensor at the default float.

``exp``, ``sin``, ``sigmoid``, ``lgamma``, ``special.i0e`` and the rest answer
in floating point whatever their input, so the reference computes an integer
or bool input at the default floating dtype.  Lucid did not, in three ways
(LCD-508):

* ``exp2(int)`` was all ones — the engine composition multiplied the integer
  by a ``log 2`` built on the integer, which truncated to 0 — and
  ``log1p(int)`` raised ``DtypeMismatch``;
* ``lgamma`` / ``digamma`` / ``gammaln`` / ``psi`` raised on an integer,
  ``erfc`` and ``angle`` on a bool or any real input;
* the ops that did promote promoted to float32 regardless of
  :func:`lucid.set_default_dtype`, where the reference follows it.

The one owner is ``lucid._dispatch._float_operand``; every spelling — the
free function, the Tensor method, the Python composites, a compiled trace —
is pinned here against the reference for dtype and values, under a float32
and a float64 default.  Off the CPU a float64 default yields float32, the
documented narrowing (Metal holds no float64).
"""

import warnings
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import Any

import pytest

import lucid
from lucid.compile._core.fallback import CompileFallbackWarning

pytestmark = pytest.mark.parity

_METAL = pytest.mark.skipif(not lucid.metal.is_available(), reason="no Metal device")
_DEVICES = ["cpu", pytest.param("metal", marks=_METAL)]
_DEFAULTS = ["float32", "float64"]
_INT_VALUES = [-1, 0, 1, 2, 3]
_BOOL_VALUES = [True, False, True]
_FLOATS = ("float16", "bfloat16", "float32", "float64")

# The registry's float-valued unary entries: a free function and a method.
_REGISTRY_OPS = [
    "exp", "exp2", "log", "log2", "log10", "log1p", "sqrt", "rsqrt",
    "reciprocal", "sin", "cos", "tan", "asin", "acos", "atan", "arcsin",
    "arccos", "arctan", "sinh", "cosh", "tanh", "sigmoid", "erf", "erfinv",
]  # fmt: skip

# Written in Python on top of the registry ops, as free functions only.
_COMPOSITE_OPS = [
    "expm1", "sinc", "logit", "erfc", "lgamma", "digamma", "i0", "angle",
    "asinh", "acosh", "atanh", "arcsinh", "arccosh", "arctanh",
]  # fmt: skip

_SPECIAL_OPS = [
    "expit", "ndtr", "ndtri", "log_ndtr", "entr", "erfcx", "i0e", "i1",
    "i1e", "gammaln", "psi", "bessel_j0", "bessel_j1", "bessel_y0",
    "bessel_y1", "modified_bessel_i0", "modified_bessel_i1",
    "modified_bessel_k0", "modified_bessel_k1", "scaled_modified_bessel_k0",
    "scaled_modified_bessel_k1", "spherical_bessel_j0",
]  # fmt: skip

# Of degree n in x: the order is fixed, the tensor is the argument.
_POLYNOMIALS = [
    "chebyshev_polynomial_t", "chebyshev_polynomial_u",
    "chebyshev_polynomial_v", "chebyshev_polynomial_w",
    "shifted_chebyshev_polynomial_t", "shifted_chebyshev_polynomial_u",
    "shifted_chebyshev_polynomial_v", "shifted_chebyshev_polynomial_w",
    "hermite_polynomial_h", "hermite_polynomial_he",
    "legendre_polynomial_p", "laguerre_polynomial_l",
]  # fmt: skip

# ``nn.functional`` forms the reference takes an integer for; softsign and
# tanhshrink refuse a bool there.
_FUNCTIONAL_OPS = {"sigmoid": True, "tanh": True, "softsign": False, "tanhshrink": False}


def _attr(root: Any, path: str) -> Callable[..., Any]:
    for part in path.split("."):
        root = getattr(root, part)
    return root


def _cases() -> Iterator[Any]:
    spellings: list[tuple[str, str, bool]] = []
    spellings += [("lucid", n, True) for n in _REGISTRY_OPS]
    spellings += [("method", n, True) for n in _REGISTRY_OPS]
    spellings += [("lucid", n, True) for n in _COMPOSITE_OPS]
    spellings += [("special", n, True) for n in _SPECIAL_OPS]
    spellings += [("special", n, True) for n in _POLYNOMIALS]
    spellings += [("nn.functional", n, b) for n, b in _FUNCTIONAL_OPS.items()]
    for spelling, name, takes_bool in spellings:
        for dtype in ("int32", "int64", "bool") if takes_bool else ("int32", "int64"):
            for device in ("cpu", "metal"):
                for default in _DEFAULTS:
                    marks = [_METAL] if device == "metal" else []
                    yield pytest.param(
                        spelling,
                        name,
                        dtype,
                        device,
                        default,
                        marks=marks,
                        id=f"{spelling}.{name}-{dtype}-{device}-{default}",
                    )


def _call(module: Any, spelling: str, name: str, x: Any) -> Any:
    if spelling == "method":
        return getattr(x, name)()
    fn = _attr(module, name if spelling == "lucid" else f"{spelling}.{name}")
    if name in _POLYNOMIALS:
        return fn(x, 3)
    return fn(x)


@pytest.fixture
def default_dtype(request: pytest.FixtureRequest, ref: ModuleType) -> Iterator[str]:
    """Set one default floating dtype on both sides, and restore both."""
    name = str(request.param)
    before, ref_before = lucid.get_default_dtype(), ref.get_default_dtype()
    lucid.set_default_dtype(getattr(lucid, name))
    ref.set_default_dtype(getattr(ref, name))
    try:
        yield name
    finally:
        lucid.set_default_dtype(before)
        ref.set_default_dtype(ref_before)


def _expected_dtype(ref: ModuleType, ref_dtype: Any, device: str) -> lucid.dtype:
    name = next(n for n in _FLOATS if getattr(ref, n) == ref_dtype)
    if device != "cpu" and name == "float64":
        return lucid.float32
    return getattr(lucid, name)


def _inputs(ref: ModuleType, dtype: str, device: str) -> tuple[lucid.Tensor, Any]:
    values = _BOOL_VALUES if dtype == "bool" else _INT_VALUES
    lucid_dtype = lucid.bool_ if dtype == "bool" else getattr(lucid, dtype)
    return (
        lucid.tensor(values, dtype=lucid_dtype, device=device),
        ref.tensor(values, dtype=getattr(ref, dtype)),
    )


def _assert_matches(ref: ModuleType, got: lucid.Tensor, want: Any, device: str) -> None:
    assert want.dtype.is_floating_point, "the reference answers in floating point"
    assert got.dtype == _expected_dtype(ref, want.dtype, device)
    assert got.tolist() == pytest.approx(want.tolist(), rel=1e-4, abs=1e-5, nan_ok=True)


@pytest.mark.parametrize(
    ("spelling", "name", "dtype", "device", "default_dtype"),
    list(_cases()),
    indirect=["default_dtype"],
)
def test_integer_input_matches_reference(
    spelling: str, name: str, dtype: str, device: str, default_dtype: str, ref: ModuleType
) -> None:
    x, rx = _inputs(ref, dtype, device)
    want = _call(ref, spelling, name, rx)
    _assert_matches(ref, _call(lucid, spelling, name, x), want, device)


@_METAL
@pytest.mark.parametrize("name", ["exp2", "log1p", "sin", "sigmoid", "lgamma", "angle"])
@pytest.mark.parametrize("dtype", ["int64", "bool"])
@pytest.mark.parametrize("default_dtype", _DEFAULTS, indirect=True)
def test_compiled_trace_matches_eager_and_reference(
    name: str, dtype: str, default_dtype: str, ref: ModuleType
) -> None:
    # The trace records the cast the eager call makes; it runs through the
    # same registry entry, so the two cannot drift apart.
    x, rx = _inputs(ref, dtype, "metal")
    fn = _attr(lucid, name)
    compiled = lucid.compile(lambda t: fn(t))
    want = _call(ref, "lucid", name, rx)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        traced, replayed = compiled(x), compiled(x)
    assert not compiled.cache_info()["eager_only"]
    for got in (traced, replayed, fn(x)):
        _assert_matches(ref, got, want, "metal")


@pytest.mark.parametrize("name", _POLYNOMIALS)
@pytest.mark.parametrize("n", [0, 1])
@pytest.mark.parametrize("device", _DEVICES)
def test_low_degree_polynomial_of_an_integer_is_floating(
    name: str, n: int, device: str, ref: ModuleType
) -> None:
    # P_0 and P_1 are not stepped through the recurrence, whose float
    # coefficients promoted the higher degrees: they kept the integer dtype.
    x, rx = _inputs(ref, "int64", device)
    got = _attr(lucid.special, name)(x, n)
    _assert_matches(ref, got, _attr(ref.special, name)(rx, n), device)


@pytest.mark.parametrize("name", ["exp", "sqrt", "exp2", "lgamma", "i0e"])
@pytest.mark.parametrize("device", _DEVICES)
def test_floating_input_is_not_cast(name: str, device: str) -> None:
    # A float16 input stays float16: the cast is for bool and integer only.
    x = lucid.tensor([0.5, 1.5], dtype=lucid.float16, device=device)
    fn = _attr(lucid, name if name != "i0e" else "special.i0e")
    assert fn(x).dtype == lucid.float16


@pytest.mark.parametrize("name", ["i0e", "entr", "erfcx", "ndtri", "bessel_j0", "gammaln"])
def test_complex_input_to_a_real_special_function_is_refused(name: str) -> None:
    # The cast that promoted an integer also dropped a complex input's
    # imaginary part, so ``i0e(1+1j)`` answered ``i0e(1)``.
    with pytest.raises(TypeError, match="real"):
        _attr(lucid.special, name)(lucid.tensor([1 + 1j]))
