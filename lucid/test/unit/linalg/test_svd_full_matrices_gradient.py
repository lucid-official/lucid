"""LCD-943: ``svd(full_matrices=True)`` ignores the gradient of the null-space tail.

``U[..., :, k:]`` (tall) and ``Vh[..., k:, :]`` (wide) are an arbitrary basis
of a null space, not a function of ``A``.  The reference framework documents
that their gradient "will be ignored" — it does not raise, unlike
``qr(mode='complete')`` — and Lucid matches it: the tail is detached, every
other graph is untouched.  These tests pin that, so a refusal (the QR
behaviour) is never added here by analogy.

Singular vectors are only defined up to a sign, so every comparison against
the reference uses a loss that is invariant to it: ``U D Uᵀ`` and ``Vhᵀ D' Vh``
with ``D`` constant over the tail.
"""

from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.linalg as LA

_SHAPES = [(5, 3), (3, 5), (4, 4), (2, 5, 3), (2, 3, 5)]


def _weights(shape: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    m, n = shape[-2:]
    k = min(m, n)
    rng = np.random.default_rng(11)
    dm = np.r_[np.arange(1.0, k + 1), np.full(m - k, 7.0)]
    dn = np.r_[np.arange(1.0, k + 1) * 0.5, np.full(n - k, 3.0)]
    w = rng.standard_normal((*shape[:-2], m, m))
    v = rng.standard_normal((*shape[:-2], n, n))
    return dm, dn, w, v


def _lucid_dtype(device: str) -> lucid.dtype:
    return lucid.float32 if device == "metal" else lucid.float64


@pytest.mark.parity
@pytest.mark.parametrize("shape", _SHAPES)
def test_mixed_loss_gradient_matches_reference(
    ref: ModuleType, device: str, shape: tuple[int, ...]
) -> None:
    arr = np.random.default_rng(3).standard_normal(shape)
    dm, dn, w, v = _weights(shape)
    dt = _lucid_dtype(device)
    rdt = ref.float32 if device == "metal" else ref.float64

    ra = ref.tensor(arr, dtype=rdt, requires_grad=True)
    ru, rs, rvh = ref.linalg.svd(ra, full_matrices=True)
    r = lambda x: ref.tensor(x, dtype=rdt)  # noqa: E731 - local converter
    rloss = (ru * r(dm) @ ru.mT * r(w)).sum() + (rvh.mT * r(dn) @ rvh * r(v)).sum() + rs.sum()
    rloss.backward()

    a = lucid.tensor(arr, dtype=dt, device=device, requires_grad=True)
    u, s, vh = LA.svd(a, full_matrices=True)
    t = lambda x: lucid.tensor(x, dtype=dt, device=device)  # noqa: E731 - local converter
    loss = (u * t(dm) @ u.mT * t(w)).sum() + (vh.mT * t(dn) @ vh * t(v)).sum() + s.sum()
    loss.backward()

    tol = 1e-4 if device == "metal" else 1e-10
    assert a.grad is not None and ra.grad is not None
    np.testing.assert_allclose(a.grad.numpy(), ra.grad.numpy(), rtol=tol, atol=tol * 10)


@pytest.mark.parity
@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (2, 5, 3)])
def test_a_loss_reading_only_the_tail_has_zero_gradient_like_the_reference(
    ref: ModuleType, device: str, shape: tuple[int, ...]
) -> None:
    arr = np.random.default_rng(4).standard_normal(shape).astype(np.float32)
    m, n = shape[-2:]
    k = min(m, n)
    ra = ref.tensor(arr, requires_grad=True)
    ru, _, rvh = ref.linalg.svd(ra, full_matrices=True)
    (ru[..., k:].sum() + rvh[..., k:, :].sum()).backward()
    assert ra.grad is not None and float(ra.grad.abs().max()) == 0.0

    a = lucid.tensor(arr, device=device, requires_grad=True)
    u, _, vh = LA.svd(a, full_matrices=True)
    (u[..., k:].sum() + vh[..., k:, :].sum()).backward()
    assert a.grad is not None
    assert float(a.grad.abs().max().item()) == 0.0


@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (2, 5, 3)])
def test_singular_values_only_graph_is_identical_in_full_and_reduced_mode(
    device: str, shape: tuple[int, ...]
) -> None:
    arr = np.random.default_rng(5).standard_normal(shape).astype(np.float32)
    grads = []
    for full in (True, False):
        a = lucid.tensor(arr, device=device, requires_grad=True)
        LA.svd(a, full_matrices=full)[1].sum().backward()
        assert a.grad is not None
        grads.append(a.grad.numpy())
    np.testing.assert_array_equal(grads[0], grads[1])


@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (2, 3, 5)])
def test_leading_columns_gradient_is_identical_in_full_and_reduced_mode(
    device: str, shape: tuple[int, ...]
) -> None:
    arr = np.random.default_rng(6).standard_normal(shape).astype(np.float32)
    m, n = shape[-2:]
    k = min(m, n)
    grads = []
    for full in (True, False):
        a = lucid.tensor(arr, device=device, requires_grad=True)
        u, _, vh = LA.svd(a, full_matrices=full)
        # Sign-invariant: the projectors onto the leading singular subspaces.
        loss = (u[..., :k] @ u[..., :k].mT).pow(2).sum() * 0.5
        loss = loss + (vh[..., :k, :].mT @ vh[..., :k, :]).sum()
        loss.backward()
        assert a.grad is not None
        grads.append(a.grad.numpy())
    np.testing.assert_allclose(grads[0], grads[1], rtol=1e-5, atol=1e-5)
