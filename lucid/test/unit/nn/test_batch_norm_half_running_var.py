"""A half-precision BatchNorm keeps its running variance non-negative.

The training step rebuilds the batch variance from the forward's saved
``rstd`` as ``1/rstd^2 - eps``.  In float16 that ran at float16: with a batch
variance near 0, ``rstd`` is about ``1/sqrt(eps)``, its square overflowed, and
the running variance came out as ``-eps`` — then eval() took the square root
of a negative number (LCD-122).  It runs in float32 now, clamped at 0.
"""

import math
from types import ModuleType

import pytest

import lucid
from lucid.test._fixtures.devices import metal_available

_DEVICES = [
    "cpu",
    pytest.param("metal", marks=pytest.mark.skipif(not metal_available(), reason="needs Metal")),
]
_HALF = ["float16", "bfloat16"]

#: rank -> (module, input shape)
_RANKS = {1: ("BatchNorm1d", (8, 4)), 2: ("BatchNorm2d", (2, 4, 2, 2))}


def _trained(rank: int, dtype: str, device: str, data: lucid.Tensor) -> lucid.nn.Module:
    name, _ = _RANKS[rank]
    bn = getattr(lucid.nn, name)(4, momentum=1.0).to(device=device, dtype=getattr(lucid, dtype))
    bn.train()
    bn(data.to(device=device, dtype=getattr(lucid, dtype)))
    return bn


def _flat(t: lucid.Tensor) -> list[float]:
    return t.to("cpu").to(lucid.float32).reshape(-1).tolist()


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _HALF)
@pytest.mark.parametrize("rank", list(_RANKS))
@pytest.mark.parametrize("value", [0.0, 3.0])
def test_a_constant_batch_leaves_a_zero_running_variance(
    value: float, rank: int, dtype: str, device: str
) -> None:
    # momentum=1 makes the running variance the batch's own, which is 0.
    bn = _trained(rank, dtype, device, lucid.full(_RANKS[rank][1], value))
    running = _flat(bn.running_var)
    assert all(0.0 <= v <= 1e-7 for v in running), running


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _HALF)
def test_eval_after_a_constant_batch_is_finite(dtype: str, device: str) -> None:
    bn = _trained(1, dtype, device, lucid.zeros(8, 4))
    bn.eval()
    out = bn(lucid.ones(8, 4, device=device, dtype=getattr(lucid, dtype)))
    assert all(math.isfinite(v) for v in _flat(out))


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _HALF)
def test_a_nan_in_the_batch_still_reaches_the_running_variance(dtype: str, device: str) -> None:
    # The clamp at 0 must not turn a NaN variance into 0.
    data = lucid.ones(8, 4)
    data[0, 0] = float("nan")
    running = _flat(_trained(1, dtype, device, data).running_var)
    assert math.isnan(running[0])
    assert all(math.isfinite(v) for v in running[1:])


@pytest.mark.parity
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", _HALF)
def test_the_running_variance_matches_the_reference(
    dtype: str, device: str, ref: ModuleType
) -> None:
    rows = [[((i * 7 + j * 3) % 11) / 4.0 - 1.0 for j in range(4)] for i in range(8)]
    rows[3] = [0.0] * 4
    got = _flat(_trained(1, dtype, device, lucid.tensor(rows)).running_var)
    bn = ref.nn.BatchNorm1d(4, momentum=1.0).to(getattr(ref, dtype))
    bn.train()
    bn(ref.tensor(rows, dtype=getattr(ref, dtype)))
    want = bn.running_var.float().tolist()
    # One step of the half's own rounding: rstd is saved at that precision.
    rtol = 2e-3 if dtype == "float16" else 1.6e-2
    for g, w in zip(got, want, strict=True):
        assert math.isclose(g, w, rel_tol=rtol), (got, want)
