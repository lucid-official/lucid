"""Regression tests for the unfold / fold gradient paths.

Covers four bugs found on 2026-07-22:

1. ``unfold_dim`` on Metal built its gather index broadcast-shaped, so MLX's
   ``take`` spliced *all* of the index axes in and the resulting array had rank
   ``2*ndim`` instead of ``ndim+1``.  For ``dim < ndim-1`` the window axis also
   landed in the wrong position, so the data itself was wrong; for
   ``dim == ndim-1`` only the rank desynced, which silently corrupted whichever
   op consumed the result next.
2. ``unfold_dim`` had no autograd node at all — every composite built on it
   (``Tensor.unfold``, ``LPPool1d/2d``, ``LocalResponseNorm``) silently dropped
   that gradient path on both devices.
3. ``fold`` had no autograd node either, so ``fold(unfold(x))`` returned a
   ``None`` gradient.
4. ``fold``'s block counts used the raw kernel size instead of the effective
   (dilated) extent, which mis-mapped positions and read past the end of the
   column buffer whenever ``dilation > 1``.

Ground truth is finite differences in float64 on the CPU; Metal is compared
against the same reference at float32 tolerance.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

DEVICES = ["cpu", "metal"]


def _numpy_unfold(t, dim, size, step):
    """Reference im2col along a single axis: (..., L, ..., size)."""
    length = (t.shape[dim] - size) // step + 1
    stacked = np.stack(
        [np.take(t, range(i * step, i * step + size), axis=dim) for i in range(length)], axis=dim
    )
    return np.moveaxis(stacked, dim + 1, -1)


def _grad(base, dev, fn, w):
    dt = lucid.float64 if dev == "cpu" else lucid.float32
    data = base if dev == "cpu" else base.astype(np.float32)
    weight = w if dev == "cpu" else w.astype(np.float32)
    x = lucid.tensor(data, device=dev, dtype=dt)
    x.requires_grad = True
    (fn(x) * lucid.tensor(weight, device=dev, dtype=dt)).sum().backward()
    assert x.grad is not None, "gradient did not reach the input"
    return np.asarray(x.grad.numpy(), dtype=np.float64)


def _finite_diff(base, fn, w, eps=1e-5):
    def value(b):
        x = lucid.tensor(b, dtype=lucid.float64)
        return float((fn(x) * lucid.tensor(w, dtype=lucid.float64)).sum().item())

    out = np.zeros_like(base)
    it = np.nditer(base, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        plus = base.copy()
        plus[idx] += eps
        minus = base.copy()
        minus[idx] -= eps
        out[idx] = (value(plus) - value(minus)) / (2 * eps)
        it.iternext()
    return out


def _check_grad(shape, fn, dev, seed=0):
    """Assert the autograd gradient matches finite differences."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(shape).astype(np.float64)
    probe = fn(lucid.tensor(base, dtype=lucid.float64))
    w = rng.standard_normal(probe.shape).astype(np.float64)

    got = _grad(base, dev, fn, w)
    ref = _finite_diff(base, fn, w)
    scale = max(float(np.abs(ref).max()), 1e-12)
    tol = 1e-7 if dev == "cpu" else 5e-5
    assert np.abs(got - ref).max() / scale < tol


# ── 1. unfold_dim forward correctness (device parity + absolute) ──────────────

UNFOLD_SPECS = [
    ((2, 16, 8), 2, 3, 1),  # last axis, overlapping windows
    ((1, 2, 8, 8), 2, 2, 2),  # mid axis — the case that corrupted data
    ((1, 2, 8, 8), 1, 2, 1),  # mid axis, overlapping
    ((2, 6, 4, 4), 3, 2, 2),  # last axis of a 4-D input
    ((3, 5, 7, 4), 0, 2, 1),  # first axis
    ((2, 9), 1, 2, 3),  # step > size (gappy: some inputs get no gradient)
]


@pytest.mark.parametrize("shape,dim,size,step", UNFOLD_SPECS)
@pytest.mark.parametrize("device", DEVICES)
def test_unfold_dim_forward_matches_reference(shape, dim, size, step, device):
    rng = np.random.default_rng(0)
    base = rng.standard_normal(shape).astype(np.float32)
    ref = _numpy_unfold(base, dim, size, step)
    got = lucid.tensor(base, device=device).unfold(dim, size, step).numpy()
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() == 0.0


@pytest.mark.parametrize("shape,dim,size,step", UNFOLD_SPECS)
def test_unfold_dim_consumer_op_not_corrupted(shape, dim, size, step):
    """The rank desync only bit whichever op consumed the unfold output."""
    rng = np.random.default_rng(1)
    base = rng.standard_normal(shape).astype(np.float32)
    ref = _numpy_unfold(base, dim, size, step).sum(-1)
    outs = {
        dev: lucid.tensor(base, device=dev).unfold(dim, size, step).sum(dim=-1).numpy()
        for dev in DEVICES
    }
    for dev, got in outs.items():
        assert np.abs(got - ref).max() < 1e-4, f"{dev} diverged from reference"


# ── 2. unfold_dim backward ───────────────────────────────────────────────────


@pytest.mark.parametrize("shape,dim,size,step", UNFOLD_SPECS)
@pytest.mark.parametrize("device", DEVICES)
def test_unfold_dim_backward(shape, dim, size, step, device):
    _check_grad(shape, lambda t: t.unfold(dim, size, step), device)


@pytest.mark.parametrize("device", DEVICES)
def test_lppool2d_backward(device):
    _check_grad((1, 2, 8, 8), lambda t: nn.LPPool2d(2, 2)(t), device)


@pytest.mark.parametrize("device", DEVICES)
def test_lppool1d_backward(device):
    _check_grad((1, 2, 8), lambda t: nn.LPPool1d(2, 2)(t), device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("alpha", [1e-4, 1.0])
def test_local_response_norm_backward(device, alpha):
    """The missing path scaled with ``alpha``, so a large alpha exposes it."""
    _check_grad((2, 6, 4, 4), lambda t: F.local_response_norm(t, 3, alpha=alpha), device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(2, 6, 4), (2, 6, 4, 4), (2, 6, 2, 3, 4)])
def test_local_response_norm_forward_device_parity(device, shape):
    rng = np.random.default_rng(2)
    base = rng.standard_normal(shape).astype(np.float32)
    ref = F.local_response_norm(lucid.tensor(base, device="cpu"), 3).numpy()
    got = F.local_response_norm(lucid.tensor(base, device=device), 3).numpy()
    assert np.abs(got - ref).max() < 1e-5


# ── 3 + 4. fold backward and dilated fold ────────────────────────────────────

FOLD_SPECS = [
    # (cols_shape, output_size, kernel, stride, padding, dilation)
    ((1, 18, 25), (7, 7), 3, 1, 0, 1),
    ((1, 8, 9), (7, 7), 2, 2, 0, 1),
    ((1, 8, 25), (7, 7), 2, 1, 0, 2),  # dilated — was wrong + over-read
    ((1, 18, 16), (7, 7), 3, 2, 1, 1),
    ((1, 8, 64), (7, 7), 2, 1, 1, 1),
]


@pytest.mark.parametrize("cols,out_hw,k,s,p,d", FOLD_SPECS)
@pytest.mark.parametrize("device", DEVICES)
def test_fold_backward(cols, out_hw, k, s, p, d, device):
    _check_grad(cols, lambda t: F.fold(t, out_hw, k, stride=s, padding=p, dilation=d), device)


@pytest.mark.parametrize("device", DEVICES)
def test_fold_of_unfold_has_gradient(device):
    """``fold(unfold(x))`` used to return a ``None`` gradient on both devices."""
    _check_grad((1, 2, 7, 7), lambda t: F.fold(F.unfold(t, 3), (7, 7), 3), device)


@pytest.mark.parametrize("cols,out_hw,k,s,p,d", FOLD_SPECS)
@pytest.mark.parametrize("device", DEVICES)
def test_fold_is_adjoint_of_unfold(cols, out_hw, k, s, p, d, device):
    """<fold(x), y> == <x, unfold(y)> — device- and FD-free correctness check."""
    rng = np.random.default_rng(3)
    dt = lucid.float64 if device == "cpu" else lucid.float32
    npdt = np.float64 if device == "cpu" else np.float32
    x = lucid.tensor(rng.standard_normal(cols).astype(npdt), device=device, dtype=dt)
    channels = cols[1] // (k * k)
    y = lucid.tensor(
        rng.standard_normal((cols[0], channels) + out_hw).astype(npdt), device=device, dtype=dt
    )
    lhs = float((F.fold(x, out_hw, k, stride=s, padding=p, dilation=d) * y).sum().item())
    rhs = float((x * F.unfold(y, k, stride=s, padding=p, dilation=d)).sum().item())
    tol = 1e-12 if device == "cpu" else 1e-5
    assert abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-12) < tol


def test_fold_rejects_mismatched_block_count():
    """A wrong column count used to read past the end of the buffer."""
    bad = lucid.tensor(np.zeros((1, 8, 36), dtype=np.float32))
    with pytest.raises(Exception):
        F.fold(bad, (7, 7), 2, padding=1)
