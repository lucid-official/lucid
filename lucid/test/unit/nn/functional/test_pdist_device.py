"""Regression tests for ``F.pdist`` and the ``cdist`` zero-distance gradient.

Two bugs, both found on 2026-07-22:

1. ``pdist`` built its pair-index tensor with ``triu_indices(n, n, offset=1)``
   and no ``device=``, so the indices defaulted to the CPU.  Feeding a CPU index
   tensor to ``gather`` on a Metal tensor made the engine's
   ``std::get<GpuStorage>`` throw ``bad_variant_access`` — in the *forward*
   pass, before autograd was involved.
2. ``cdist``'s p=2 path ended in ``sqrt``, whose derivative is infinite at 0.
   Any coincident pair (in particular every diagonal entry of ``cdist(x, x)``,
   which is exactly how ``pdist`` is built) backpropagated ``0 * inf = NaN``,
   so ``pdist``'s gradient was entirely NaN on *both* devices.
"""

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F

DEVICES = ["cpu", "metal"]


def _reference_pdist(x, p):
    n = x.shape[0]
    return np.array(
        [
            np.power(np.abs(x[i] - x[j]) ** p, 1.0).sum() ** (1.0 / p)
            for i in range(n)
            for j in range(i + 1, n)
        ]
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n,d", [(5, 4), (2, 3), (7, 1)])
def test_pdist_forward(device, n, d):
    rng = np.random.default_rng(0)
    base = rng.standard_normal((n, d)).astype(np.float32)
    got = F.pdist(lucid.tensor(base, device=device)).numpy()
    ref = _reference_pdist(base.astype(np.float64), 2.0)
    assert got.shape == (n * (n - 1) // 2,)
    assert np.abs(got - ref).max() < 1e-4


@pytest.mark.parametrize("device", DEVICES)
def test_pdist_device_parity(device):
    rng = np.random.default_rng(1)
    base = rng.standard_normal((6, 5)).astype(np.float32)
    ref = F.pdist(lucid.tensor(base, device="cpu")).numpy()
    got = F.pdist(lucid.tensor(base, device=device)).numpy()
    assert np.abs(got - ref).max() < 1e-5


@pytest.mark.parametrize("device", DEVICES)
def test_pdist_backward(device):
    rng = np.random.default_rng(2)
    base = rng.standard_normal((5, 4)).astype(np.float64)
    dt = lucid.float64 if device == "cpu" else lucid.float32
    data = base if device == "cpu" else base.astype(np.float32)

    probe = F.pdist(lucid.tensor(base, dtype=lucid.float64))
    w = rng.standard_normal(probe.shape).astype(np.float64)

    x = lucid.tensor(data, device=device, dtype=dt)
    x.requires_grad = True
    weight = w if device == "cpu" else w.astype(np.float32)
    (F.pdist(x) * lucid.tensor(weight, device=device, dtype=dt)).sum().backward()
    assert x.grad is not None
    got = np.asarray(x.grad.numpy(), dtype=np.float64)

    def value(b):
        t = lucid.tensor(b, dtype=lucid.float64)
        return float((F.pdist(t) * lucid.tensor(w, dtype=lucid.float64)).sum().item())

    ref = np.zeros_like(base)
    it = np.nditer(base, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        plus = base.copy()
        plus[idx] += 1e-6
        minus = base.copy()
        minus[idx] -= 1e-6
        ref[idx] = (value(plus) - value(minus)) / 2e-6
        it.iternext()

    scale = max(float(np.abs(ref).max()), 1e-12)
    tol = 1e-7 if device == "cpu" else 1e-4
    assert np.abs(got - ref).max() / scale < tol


# ── cdist zero-distance gradient (the root cause behind pdist's NaN) ─────────


@pytest.mark.parametrize("device", DEVICES)
def test_cdist_self_gradient_is_not_nan(device):
    """Every diagonal entry of cdist(x, x) is a zero distance."""
    rng = np.random.default_rng(3)
    x = lucid.tensor(rng.standard_normal((4, 3)).astype(np.float32), device=device)
    x.requires_grad = True
    out = lucid.cdist(x, x)
    (out * lucid.ones_like(out)).sum().backward()
    assert x.grad is not None
    assert not np.isnan(x.grad.numpy()).any()


@pytest.mark.parametrize("device", DEVICES)
def test_cdist_duplicate_rows_gradient_is_not_nan(device):
    """Coincident points across two *distinct* inputs hit the same path."""
    rng = np.random.default_rng(4)
    base = rng.standard_normal((3, 3)).astype(np.float32)
    other = np.concatenate([base[:1], rng.standard_normal((2, 3)).astype(np.float32)])
    x = lucid.tensor(base, device=device)
    x.requires_grad = True
    out = lucid.cdist(x, lucid.tensor(other, device=device))
    (out * lucid.ones_like(out)).sum().backward()
    assert x.grad is not None
    assert not np.isnan(x.grad.numpy()).any()


@pytest.mark.parametrize("device", DEVICES)
def test_cdist_forward_unchanged_by_the_zero_guard(device):
    """The guard routes around sqrt but must not perturb forward values.

    Up to 25 rows the default mode takes the difference form, whose
    self-distance is exact.  The ~1e-3 tolerance on it is what the
    ``‖a‖² + ‖b‖² − 2a·b`` expansion (past 25 rows, or by ``compute_mode``)
    cancels to for coincident points in float32 — behaviour of that form, not
    something the zero-guard introduced.  The guard fires on an exact zero,
    which is what the difference form and float64 produce (and exactly when
    the NaN used to appear).
    """
    rng = np.random.default_rng(5)
    a = rng.standard_normal((4, 3)).astype(np.float64)
    b = rng.standard_normal((5, 3)).astype(np.float64)

    ref = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    got = lucid.cdist(
        lucid.tensor(a, device=device, dtype=lucid.float32),
        lucid.tensor(b, device=device, dtype=lucid.float32),
    ).numpy()
    assert np.abs(got - ref).max() < 1e-5

    ref_self = np.sqrt(((a[:, None, :] - a[None, :, :]) ** 2).sum(-1))
    got_self = lucid.cdist(
        lucid.tensor(a, device=device, dtype=lucid.float32),
        lucid.tensor(a, device=device, dtype=lucid.float32),
    ).numpy()
    assert np.abs(got_self - ref_self).max() < 1e-3
    assert not np.isnan(got_self).any()


def test_cdist_self_distance_is_exactly_zero_in_float64():
    """float64 yields an exact zero — the input that used to NaN the backward."""
    rng = np.random.default_rng(6)
    a = rng.standard_normal((4, 3)).astype(np.float64)
    x = lucid.tensor(a, dtype=lucid.float64)
    got = lucid.cdist(x, x).numpy()
    assert np.abs(np.diag(got)).max() == 0.0
