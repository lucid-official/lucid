"""isnan / isinf / isfinite / nan_to_num over every dtype, on both devices.

CHA-34.  The CPU kernels tested only float32 and float64.  Every other
dtype got the integer answer: nothing NaN, nothing infinite, everything
finite.  So on the CPU a float16 or bfloat16 tensor full of infinities
passed as clean, ``nan_to_num`` handed it back unchanged, ``nansum`` /
``nanmean`` returned NaN, and the gradient of ``nan_to_num`` let a
gradient through every replaced slot.  ``GradScaler``, which asks
``isfinite`` of each gradient, never saw a half-precision overflow.
Metal was right throughout.  complex64 / complex128 had the same hole on
the CPU.

The ``nan_to_num`` defaults were wrong on both devices too.  The binding
filled them with float32's extremes whatever the dtype.  In float16 and
bfloat16 those round to infinity, so a default call left the infinities
in place.  float64 got 3.4e38 where the reference framework gives 1.8e308.
``None``, which the stub documents, was refused.

The expectations below are written from the IEEE definitions, so they hold
without the reference framework installed.  The ``ref`` tests check the
same cells against it.
"""

import math

import numpy as np
import pytest

import lucid
from lucid.test._fixtures.devices import devices_supporting

_HALVES = [lucid.float16, lucid.bfloat16]
_FLOATS = [lucid.float16, lucid.bfloat16, lucid.float32, lucid.float64]
_COMPLEX = [lucid.complex64, lucid.complex128]
_INTEGRAL = [lucid.bool, lucid.int8, lucid.int32, lucid.int64]
_PROBES = ["isnan", "isinf", "isfinite"]

#: The largest finite value and the smallest subnormal of each float format.
_LIMITS: dict[lucid.dtype, tuple[float, float]] = {
    lucid.float16: (65504.0, 2.0**-24),
    lucid.bfloat16: (3.3895313892515355e38, 2.0**-133),
    lucid.float32: (3.4028234663852886e38, 2.0**-149),
    lucid.float64: (1.7976931348623157e308, 5e-324),
}


def _cells(dtypes: list[lucid.dtype]) -> list[object]:
    return [
        pytest.param(dev, dt, id=f"{dev}-{str(dt).split('.')[-1]}")
        for dt in dtypes
        for dev in devices_supporting(dt)
    ]


def _specials(dtype: lucid.dtype) -> list[float]:
    """Every class of value the format has: both zeros, normals, both
    subnormals, both extremes, NaN of either sign, both infinities."""
    top, tiny = _LIMITS[dtype]
    nan = float("nan")
    return [0.0, -0.0, 1.0, -2.5, tiny, -tiny, top, -top, nan, -nan, math.inf, -math.inf]


def _make(values: list[float], dtype: lucid.dtype, device: str) -> lucid.Tensor:
    """``values`` held exactly in ``dtype`` on ``device``."""
    x = lucid.tensor(values, dtype=lucid.float64).to(dtype)
    # Every value is representable, so the round trip is exact; a lossy
    # construction would make the test check something other than intended.
    back = x.to(lucid.float64).numpy()
    np.testing.assert_array_equal(back, np.array(values))
    return x.to(device)


def _widened(t: lucid.Tensor) -> np.ndarray:
    """``t`` as float64 on the host — exact for every float dtype."""
    return t.to("cpu").to(lucid.float64).numpy()


def _assert_bits_equal(got: np.ndarray, want: np.ndarray) -> None:
    """Equal values, NaN where NaN, and the same sign on every non-NaN —
    so a ``-0.0`` that became ``+0.0`` fails."""
    np.testing.assert_array_equal(got, want)
    keep = ~np.isnan(want)
    np.testing.assert_array_equal(np.signbit(got[keep]), np.signbit(want[keep]))


# ── the probes, from the definitions ────────────────────────────────────────


@pytest.mark.parametrize("device,dtype", _cells(_FLOATS))
@pytest.mark.parametrize("op", _PROBES)
def test_probe_classifies_every_value(op: str, device: str, dtype: lucid.dtype) -> None:
    values = _specials(dtype)
    got = getattr(lucid, op)(_make(values, dtype, device))
    want = [getattr(math, op)(v) for v in values]
    assert got.dtype == lucid.bool
    assert got.to("cpu").tolist() == want


@pytest.mark.parametrize("device,dtype", _cells(_INTEGRAL))
@pytest.mark.parametrize("op", _PROBES)
def test_probe_on_integers_and_bool(op: str, device: str, dtype: lucid.dtype) -> None:
    values = [0, 1, 1, 0] if dtype == lucid.bool else [0, -3, 7, 100]
    got = getattr(lucid, op)(lucid.tensor(values).to(dtype).to(device))
    assert got.dtype == lucid.bool
    assert got.to("cpu").tolist() == [op == "isfinite"] * len(values)


@pytest.mark.parametrize("dtype", _COMPLEX, ids=str)
@pytest.mark.parametrize("op", _PROBES)
def test_probe_on_complex_cpu(op: str, dtype: lucid.dtype) -> None:
    """NaN or infinite when either part is; finite only when both are."""
    nan, inf = float("nan"), math.inf
    values = [
        complex(nan, 1.0),
        complex(1.0, inf),
        complex(-inf, nan),
        complex(1.0, -2.0),
        complex(0.0, -inf),
        complex(nan, nan),
        complex(-0.0, 5e-40),
    ]
    got = getattr(lucid, op)(lucid.tensor(values, dtype=dtype)).tolist()
    pieces = {
        "isnan": lambda z: math.isnan(z.real) or math.isnan(z.imag),
        "isinf": lambda z: math.isinf(z.real) or math.isinf(z.imag),
        "isfinite": lambda z: math.isfinite(z.real) and math.isfinite(z.imag),
    }
    assert got == [pieces[op](z) for z in values]


# ── nan_to_num, from the definitions ────────────────────────────────────────


@pytest.mark.parametrize("device,dtype", _cells(_FLOATS))
def test_nan_to_num_defaults_are_the_dtype_extremes(device: str, dtype: lucid.dtype) -> None:
    values = _specials(dtype)
    top, _ = _LIMITS[dtype]
    got = lucid.nan_to_num(_make(values, dtype, device))
    assert got.dtype == dtype
    want = [
        (0.0 if math.isnan(v) else top if v == math.inf else -top if v == -math.inf else v)
        for v in values
    ]
    _assert_bits_equal(_widened(got), np.array(want))


@pytest.mark.parametrize("device,dtype", _cells(_FLOATS))
def test_nan_to_num_none_means_default(device: str, dtype: lucid.dtype) -> None:
    x = _make(_specials(dtype), dtype, device)
    _assert_bits_equal(
        _widened(lucid.nan_to_num(x, nan=None, posinf=None, neginf=None)),
        _widened(lucid.nan_to_num(x)),
    )
    _assert_bits_equal(_widened(x.nan_to_num(posinf=None)), _widened(lucid.nan_to_num(x)))


@pytest.mark.parametrize("device,dtype", _cells(_FLOATS))
def test_nan_to_num_explicit_values(device: str, dtype: lucid.dtype) -> None:
    x = _make([1.0, float("nan"), math.inf, -math.inf, -0.0], dtype, device)
    got = lucid.nan_to_num(x, nan=0.5, posinf=7.0, neginf=-7.0)
    _assert_bits_equal(_widened(got), np.array([1.0, 0.5, 7.0, -7.0, -0.0]))


@pytest.mark.parametrize("device,dtype", _cells(_HALVES))
def test_nan_to_num_out_of_range_replacement_is_infinite(device: str, dtype: lucid.dtype) -> None:
    """An explicit replacement is cast, not clamped: one the dtype cannot
    hold becomes its infinity, as in the reference framework."""
    x = _make([float("nan"), math.inf, -math.inf], dtype, device)
    got = lucid.nan_to_num(x, nan=1e39, posinf=1e39, neginf=-1e39)
    _assert_bits_equal(_widened(got), np.array([math.inf, math.inf, -math.inf]))


@pytest.mark.parametrize("device,dtype", _cells(_INTEGRAL))
def test_nan_to_num_on_integers_and_bool_is_identity(device: str, dtype: lucid.dtype) -> None:
    values = [1, 0, 1] if dtype == lucid.bool else [0, -3, 7]
    x = lucid.tensor(values).to(dtype).to(device)
    got = lucid.nan_to_num(x, posinf=1e30)
    assert got.dtype == dtype
    assert got.to("cpu").tolist() == x.to("cpu").tolist()


@pytest.mark.parametrize("dtype", _COMPLEX, ids=str)
def test_nan_to_num_on_complex_cpu_replaces_each_part(dtype: lucid.dtype) -> None:
    top = _LIMITS[lucid.float32 if dtype == lucid.complex64 else lucid.float64][0]
    nan, inf = float("nan"), math.inf
    x = lucid.tensor([complex(nan, 1.0), complex(1.0, inf), complex(-inf, nan)], dtype=dtype)
    got = lucid.nan_to_num(x).tolist()
    assert got == [complex(0.0, 1.0), complex(1.0, top), complex(-top, 0.0)]


# ── shapes and layouts ──────────────────────────────────────────────────────


@pytest.mark.parametrize("device,dtype", _cells(_FLOATS))
def test_empty_and_zero_dim(device: str, dtype: lucid.dtype) -> None:
    empty = lucid.zeros(0, 3).to(dtype).to(device)
    for op in _PROBES:
        out = getattr(lucid, op)(empty)
        assert out.shape == (0, 3) and out.dtype == lucid.bool
    assert lucid.nan_to_num(empty).shape == (0, 3)
    scalar = lucid.tensor(math.inf).to(dtype).to(device)
    assert lucid.isinf(scalar).shape == ()
    assert lucid.isinf(scalar).item() is True
    assert lucid.isfinite(scalar).item() is False
    assert _widened(lucid.nan_to_num(scalar)).item() == _LIMITS[dtype][0]


@pytest.mark.parametrize("device,dtype", _cells(_FLOATS))
def test_non_contiguous_input(device: str, dtype: lucid.dtype) -> None:
    values = _specials(dtype)
    grid = _make(values, dtype, device).reshape(3, 4).transpose(0, 1)
    flat = np.array(values).reshape(3, 4).T
    assert lucid.isnan(grid).to("cpu").tolist() == np.isnan(flat).tolist()
    assert lucid.isinf(grid).to("cpu").tolist() == np.isinf(flat).tolist()
    assert lucid.isfinite(grid).to("cpu").tolist() == np.isfinite(flat).tolist()
    top = _LIMITS[dtype][0]
    _assert_bits_equal(
        _widened(lucid.nan_to_num(grid)), np.nan_to_num(flat, nan=0.0, posinf=top, neginf=-top)
    )


@pytest.mark.parametrize("dtype", _FLOATS + _COMPLEX[:1], ids=str)
def test_large_input_across_parallel_chunks_cpu(dtype: lucid.dtype) -> None:
    """Long enough that the CPU splits the loop across cores; special values
    sit on and around every chunk edge, and in the tail."""
    n = 3 * 65536 + 7
    base = np.linspace(-4.0, 4.0, n)
    for k, i in enumerate(range(0, n, 4093)):
        base[i] = (float("nan"), math.inf, -math.inf)[k % 3]
    base[-1] = math.inf
    if dtype == lucid.complex64:
        # The specials in the imaginary part, so the real part alone
        # would say finite.
        parts = np.zeros(n, dtype=np.complex64)
        parts.imag = base
        x = lucid.tensor(parts)
        assert lucid.isnan(x).numpy().tolist() == np.isnan(base).tolist()
        assert lucid.isinf(x).numpy().tolist() == np.isinf(base).tolist()
        assert lucid.isfinite(x).numpy().tolist() == np.isfinite(base).tolist()
        return
    x = lucid.tensor(base).to(dtype)
    wide = _widened(x)
    assert lucid.isnan(x).numpy().tolist() == np.isnan(wide).tolist()
    assert lucid.isinf(x).numpy().tolist() == np.isinf(wide).tolist()
    assert lucid.isfinite(x).numpy().tolist() == np.isfinite(wide).tolist()
    top = _LIMITS[dtype][0]
    _assert_bits_equal(
        _widened(lucid.nan_to_num(x, nan=-1.0)),
        np.nan_to_num(wide, nan=-1.0, posinf=top, neginf=-top),
    )


# ── what the probes feed ─────────────────────────────────────────────────────


@pytest.mark.parametrize("device,dtype", _cells(_HALVES))
def test_nan_to_num_gradient_is_zero_where_replaced(device: str, dtype: lucid.dtype) -> None:
    x = (
        lucid.tensor([1.0, float("nan"), math.inf, -math.inf, 2.0])
        .to(dtype)
        .to(device)
        .requires_grad_()
    )
    lucid.nan_to_num(x).sum().backward()
    assert x.grad.to("cpu").float().tolist() == [1.0, 0.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize("device,dtype", _cells(_HALVES))
def test_nansum_and_nanmean(device: str, dtype: lucid.dtype) -> None:
    nan = float("nan")
    x = lucid.tensor([1.0, nan, 2.0, nan, 3.5]).to(dtype).to(device)
    assert lucid.nansum(x).item() == 6.5
    assert lucid.nanmean(x).item() == lucid.tensor(6.5 / 3).to(dtype).item()
    rows = lucid.tensor([[1.0, nan], [nan, nan]]).to(dtype).to(device)
    assert lucid.nansum(rows, dim=1).to("cpu").float().tolist() == [1.0, 0.0]


class _CountingOptimizer:
    """What GradScaler touches of an optimizer, and a count of steps.

    A real optimizer would also test whether it can update a half
    parameter on this device, which is not the question here.
    """

    def __init__(self, params: list[lucid.Tensor]) -> None:
        self.param_groups = [{"params": params}]
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


def _half_grad(values: list[float], dtype: lucid.dtype, device: str) -> lucid.Tensor:
    """A half parameter whose gradient is ``values``."""
    p = lucid.nn.Parameter(lucid.ones(len(values)).to(dtype).to(device))
    (p * lucid.tensor(values).to(dtype).to(device)).sum().backward()
    assert p.grad.dtype == dtype
    return p


# bfloat16 only: like the reference framework, ``GradScaler`` refuses float16
# gradients with ``ValueError`` (CHA-72) — covered in
# lucid/test/unit/amp/test_grad_scaler_half_grads.py.
_SCALER_HALVES = [lucid.bfloat16]


@pytest.mark.parametrize("device,dtype", _cells(_SCALER_HALVES))
@pytest.mark.parametrize("bad", [math.inf, -math.inf, float("nan")], ids=str)
def test_grad_scaler_sees_half_overflow(device: str, dtype: lucid.dtype, bad: float) -> None:
    """The overflow step is skipped and the scale backs off."""
    opt = _CountingOptimizer([_half_grad([1.0, bad, 2.0], dtype, device)])
    scaler = lucid.amp.GradScaler(init_scale=4.0, backoff_factor=0.5)
    assert scaler.step(opt) is None  # type: ignore[arg-type]
    scaler.update()
    assert opt.steps == 0
    assert scaler.get_scale() == 2.0


@pytest.mark.parametrize("device,dtype", _cells(_SCALER_HALVES))
def test_grad_scaler_steps_on_finite_half_gradient(device: str, dtype: lucid.dtype) -> None:
    """The dtype's largest value is finite: no false alarm at the edge."""
    top = _LIMITS[dtype][0]
    p = _half_grad([4.0, 8.0, top], dtype, device)
    opt = _CountingOptimizer([p])
    scaler = lucid.amp.GradScaler(init_scale=4.0, growth_interval=100)
    scaler.step(opt)  # type: ignore[arg-type]
    scaler.update()
    assert opt.steps == 1
    assert scaler.get_scale() == 4.0


# ── against the reference framework ─────────────────────────────────────────

_INT_VALUES = [0, -3, 7, 2**31 - 1, -(2**31)]


def _ref_dtype(ref: object, dtype: lucid.dtype) -> object:
    return getattr(ref, str(dtype).split(".")[-1])


@pytest.mark.parametrize("device,dtype", _cells(_FLOATS + [lucid.int32, lucid.bool]))
@pytest.mark.parametrize("op", [*_PROBES, "nan_to_num"])
def test_matches_reference(ref: object, op: str, device: str, dtype: lucid.dtype) -> None:
    if dtype == lucid.bool:
        x = lucid.tensor([True, False, True]).to(device)
        r = ref.tensor([True, False, True])  # type: ignore[attr-defined]
    elif dtype == lucid.int32:
        x = lucid.tensor(_INT_VALUES).to(dtype).to(device)
        r = ref.tensor(_INT_VALUES, dtype=ref.int32)  # type: ignore[attr-defined]
    else:
        values = _specials(dtype)
        x = _make(values, dtype, device)
        r = ref.tensor(values, dtype=ref.float64).to(_ref_dtype(ref, dtype))  # type: ignore[attr-defined]
    got = getattr(lucid, op)(x).to("cpu")
    want = getattr(ref, op)(r)
    assert str(got.dtype).split(".")[-1] == str(want.dtype).split(".")[-1]
    if op == "nan_to_num" and dtype in _FLOATS:
        _assert_bits_equal(_widened(got), want.double().numpy())
    else:
        assert got.tolist() == want.tolist()


@pytest.mark.parametrize("device,dtype", _cells(_HALVES))
@pytest.mark.parametrize(
    "kwargs",
    [
        {"nan": 1e-7, "posinf": 1e10, "neginf": -65519.99},
        {"nan": 1.0009765625, "posinf": 3.4028234663852886e38, "neginf": -1.0},
        {"nan": -0.0, "posinf": 65520.0, "neginf": 1e-45},
    ],
    ids=["subnormal-and-overflow", "rounding", "ties-and-underflow"],
)
def test_explicit_replacements_round_like_reference(
    ref: object, device: str, dtype: lucid.dtype, kwargs: dict[str, float]
) -> None:
    values = [float("nan"), math.inf, -math.inf, 3.0]
    x = _make(values, dtype, device)
    r = ref.tensor(values, dtype=_ref_dtype(ref, dtype))  # type: ignore[attr-defined]
    got = _widened(lucid.nan_to_num(x, **kwargs))
    want = ref.nan_to_num(r, **kwargs).double().numpy()  # type: ignore[attr-defined]
    _assert_bits_equal(got, want)


@pytest.mark.parametrize("dtype", _COMPLEX, ids=str)
@pytest.mark.parametrize("op", [*_PROBES, "nan_to_num"])
def test_complex_matches_reference_cpu(ref: object, op: str, dtype: lucid.dtype) -> None:
    nan, inf = float("nan"), math.inf
    values = [complex(nan, 1.0), complex(1.0, inf), complex(-inf, nan), complex(1.0, 2.0)]
    got = getattr(lucid, op)(lucid.tensor(values, dtype=dtype)).tolist()
    want = getattr(ref, op)(ref.tensor(values, dtype=_ref_dtype(ref, dtype))).tolist()  # type: ignore[attr-defined]
    assert got == want
