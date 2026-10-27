"""Cross-precision agreement — f32 vs f64.

f64 is the precision baseline.  f32 should agree with f64 within
expected float-precision tolerance for a wide span of common ops; if
it doesn't the f32 path has a real bug (lost mantissa bits in an
intermediate cast, fast-math reorder, etc.).

The f64 baseline always runs on CPU, the only stream that holds f64
(Metal refuses it — asserted in
``lucid/test/unit/device/test_metal_dtype_support.py``).  The f32 side
runs on every device, so Metal's f32 kernels are measured against the
same CPU double baseline as the CPU's own.

The cancellation watchdogs are about the f64 kernels themselves, so they
are collected only on devices that hold f64.
"""

import numpy as np
import pytest

import lucid
from lucid.test._fixtures.devices import devices_supporting


@pytest.mark.f64_only
class TestPrecisionParity:
    def test_sum(self, device: str) -> None:
        np.random.seed(0)
        x = np.random.uniform(-1.0, 1.0, size=(8, 8)).astype(np.float64)
        l32 = lucid.tensor(x.astype(np.float32), device=device).sum().item()
        l64 = lucid.tensor(x, dtype=lucid.float64, device="cpu").sum().item()
        assert abs(l32 - l64) < 1e-3

    def test_mean(self, device: str) -> None:
        np.random.seed(0)
        x = np.random.uniform(-1.0, 1.0, size=(64,)).astype(np.float64)
        l32 = lucid.tensor(x.astype(np.float32), device=device).mean().item()
        l64 = lucid.tensor(x, dtype=lucid.float64, device="cpu").mean().item()
        assert abs(l32 - l64) < 1e-4

    def test_matmul(self, device: str) -> None:
        np.random.seed(0)
        a = np.random.uniform(-1.0, 1.0, size=(4, 4)).astype(np.float64)
        b = np.random.uniform(-1.0, 1.0, size=(4, 4)).astype(np.float64)
        out32 = (
            (
                lucid.tensor(a.astype(np.float32), device=device)
                @ lucid.tensor(b.astype(np.float32), device=device)
            )
            .numpy()
            .astype(np.float64)
        )
        out64 = (
            lucid.tensor(a, dtype=lucid.float64, device="cpu")
            @ lucid.tensor(b, dtype=lucid.float64, device="cpu")
        ).numpy()
        np.testing.assert_allclose(out32, out64, atol=1e-5)

    def test_exp(self, device: str) -> None:
        x = np.linspace(-2.0, 2.0, 32, dtype=np.float64)
        out32 = lucid.tensor(x.astype(np.float32), device=device).exp().numpy().astype(np.float64)
        out64 = lucid.tensor(x, dtype=lucid.float64, device="cpu").exp().numpy()
        np.testing.assert_allclose(out32, out64, atol=1e-5)


# ── catastrophic cancellation watchdog ─────────────────────────────────


@pytest.mark.f64_only
@pytest.mark.parametrize("device", devices_supporting(lucid.float64))
class TestCancellation:
    def test_log1p_vs_log_for_small_x(self, device: str) -> None:
        # log1p(x) is the safe form near zero; log(1 + x) loses precision.
        x = lucid.tensor([1e-7], dtype=lucid.float64, device=device)
        good = lucid.log1p(x).item()
        # Reference: log1p(1e-7) ≈ 1e-7 - 5e-15.
        assert abs(good - 1e-7) < 1e-12

    def test_expm1_vs_exp_for_small_x(self, device: str) -> None:
        x = lucid.tensor([1e-7], dtype=lucid.float64, device=device)
        # expm1(1e-7) ≈ 1e-7 + 5e-15.
        assert abs(lucid.expm1(x).item() - 1e-7) < 1e-12


# ── f32 sanity floor ─────────────────────────────────────────────────────


class TestF32SanityFloor:
    """Cheap baseline so we notice if a kernel quietly halves precision."""

    def test_sum_relative_error(self, device: str) -> None:
        np.random.seed(0)
        x = np.random.uniform(-1.0, 1.0, size=(1024,)).astype(np.float32)
        l = lucid.tensor(x.copy(), device=device).sum().item()
        ref = float(np.sum(x.astype(np.float64)))
        rel = abs(l - ref) / max(abs(ref), 1e-9)
        # Kahan-free f32 sum on 1024 elements should comfortably hit 1e-4.
        assert rel < 1e-4

    def test_dot_relative_error(self, device: str) -> None:
        np.random.seed(0)
        x = np.random.uniform(-1.0, 1.0, size=(1024,)).astype(np.float32)
        y = np.random.uniform(-1.0, 1.0, size=(1024,)).astype(np.float32)
        out = (
            (lucid.tensor(x.copy(), device=device) * lucid.tensor(y.copy(), device=device))
            .sum()
            .item()
        )
        ref = float(np.sum((x.astype(np.float64) * y.astype(np.float64))))
        rel = abs(out - ref) / max(abs(ref), 1e-9)
        assert rel < 1e-4
