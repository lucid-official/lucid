"""Inside an autocast scope ``sum`` / ``mean`` / ``prod`` never cast a float32 input down.

CHA-35.  The three reductions were registered ``AmpPolicy::Promote``, so an
autocast scope cast their input down to the autocast dtype before reducing:
``full((64, 1024), 1.5).sum()`` under ``autocast(device_type="metal")``
answered float16 inf where the reference gives float32 98304, and an
integer or float64 sum was demoted the same way.

Their rows in the policy table (LCD-500) are the reference's: ``prod`` is
FP32 on both devices, ``sum`` FP32 on Metal and KEEP on the CPU, and
``mean`` has no row.  So a float32 input stays float32, an integer or
float64 one keeps its dtype (only floating inputs are ever cast), and a
half input widens to float32 where the row is FP32 and otherwise stays
half, accumulated in float32 by the kernel.
"""

from types import ModuleType

import pytest

import lucid
from lucid.amp import autocast
from lucid.test._fixtures.devices import metal_available

_DEVICES = ["cpu", "metal"]


def _autocast_dtype(device: str) -> lucid.dtype:
    """The autocast dtype that changes something on ``device`` — float16 is
    computed as float32 on the CPU stream, so the CPU scope uses bfloat16."""
    return lucid.float16 if device == "metal" else lucid.bfloat16


@pytest.fixture(params=_DEVICES)
def device(request: pytest.FixtureRequest) -> str:
    if request.param == "metal" and not metal_available():
        pytest.skip("Metal not available on this host")
    return str(request.param)


def _name(dtype: lucid.dtype) -> str:
    return str(dtype).split(".")[-1]


@pytest.mark.parametrize("op", ["sum", "mean", "prod"])
def test_a_float32_reduction_stays_float32(op: str, device: str) -> None:
    x = lucid.full((64, 1024), 1.5 if op != "prod" else 1.0, device=device)
    with autocast(device_type=device, dtype=_autocast_dtype(device)):
        got = getattr(x, op)()
    assert got.dtype == lucid.float32
    assert got.item() == {"sum": 98304.0, "mean": 1.5, "prod": 1.0}[op]


def test_an_integer_sum_stays_integral(device: str) -> None:
    x = lucid.arange(100000, device=device)
    with autocast(device_type=device, dtype=_autocast_dtype(device)):
        got = x.sum()
    assert got.dtype == lucid.int64
    assert got.item() == 100000 * 99999 // 2


def test_a_float64_sum_stays_float64() -> None:
    x = lucid.full((1000,), 0.1, dtype=lucid.float64)
    with autocast(device_type="cpu", dtype=lucid.bfloat16):
        got = x.sum()
    assert got.dtype == lucid.float64
    assert got.item() == pytest.approx(100.0, rel=1e-12)


# The reductions whose row widens a half input, per device.
_WIDENS = {"metal": {"sum", "prod"}, "cpu": {"prod"}}


@pytest.mark.parametrize("op", ["sum", "mean", "prod"])
@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=_name)
def test_a_half_reduction_follows_its_row(device: str, dtype: lucid.dtype, op: str) -> None:
    x = lucid.full((64, 64), 1.0, dtype=dtype, device=device)
    with autocast(device_type=device, dtype=_autocast_dtype(device)):
        got = getattr(x, op)()
    assert got.dtype == (lucid.float32 if op in _WIDENS[device] else dtype)
    assert got.item() == {"sum": 4096.0, "mean": 1.0, "prod": 1.0}[op]


@pytest.mark.parametrize("half", [False, True], ids=["float32", "half"])
@pytest.mark.parametrize("op", ["sum", "mean", "prod"])
def test_autocast_reduction_dtype_matches_reference(
    ref: ModuleType, op: str, half: bool, device: str
) -> None:
    """A float32 input keeps its dtype under the reference's autocast on
    either stream; a half input widens where its row says so — ``sum`` and
    ``prod`` on the GPU stream, ``prod`` alone on the CPU."""
    ref_device = "mps" if device == "metal" else "cpu"
    if ref_device == "mps" and not ref.backends.mps.is_available():
        pytest.skip("the reference's GPU stream is not available")
    amp_dtype = _autocast_dtype(device)
    ref_dtype = getattr(ref, _name(amp_dtype))
    x = lucid.full((8, 8), 1.0, device=device, dtype=amp_dtype if half else lucid.float32)
    r = ref.full((8, 8), 1.0, device=ref_device, dtype=ref_dtype if half else ref.float32)
    with autocast(device_type=device, dtype=amp_dtype):
        got = getattr(x, op)()
    with ref.autocast(device_type=ref_device, dtype=ref_dtype):
        want = getattr(r, op)()
    assert _name(got.dtype) == str(want.dtype).split(".")[-1]
