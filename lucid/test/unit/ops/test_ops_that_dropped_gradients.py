"""Ops that returned the right values detached from their inputs.

Each of these handed back an output with ``requires_grad=False`` for an
input that required grad, so a loss through it trained nothing — and,
until ``backward()`` started refusing untracked roots, said nothing.  The
audit's gradient axis filed them under "unsupported" among the genuinely
non-differentiable ops; it now fails any float output that comes back
detached unless the op is on a list with a reason.

Several were wrong in the forward too: ``masked_select`` ignored
broadcasting, ``F.ctc_loss`` read padded targets as concatenated ones and
averaged without dividing by target length, ``lstsq`` answered wide and
rank-deficient systems with something other than the least-squares
solution, and ``pinv`` sent a singular square matrix through ``inv``.

Gradients are checked against central differences in float64.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.autograd import grad
from lucid.test._fixtures.devices import metal_available

DEVICES = ["cpu"] + (["metal"] if metal_available() else [])
H = 1e-6


def _directional(f, x0: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    """(analytic, numeric) derivative of ``f`` along a random direction."""
    x = lucid.tensor(x0, requires_grad=True)
    (g,) = grad(f(x), [x])
    e = rng.standard_normal(x0.shape)
    num = (
        float(f(lucid.tensor(x0 + H * e)).item()) - float(f(lucid.tensor(x0 - H * e)).item())
    ) / (2 * H)
    return float((g.numpy() * e).sum()), num


# ── masked_select ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("device", DEVICES)
def test_masked_select_broadcasts_and_differentiates(device: str) -> None:
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    row = lucid.tensor(np.array([True, False, True]), device=device)
    x = lucid.tensor(a, device=device, requires_grad=True)
    y = lucid.masked_select(x, row)
    assert y.tolist() == [0.0, 2.0, 3.0, 5.0]
    (y * lucid.tensor([1.0, 2.0, 3.0, 4.0], device=device)).sum().backward()
    assert x.grad.tolist() == [[1.0, 0.0, 2.0], [3.0, 0.0, 4.0]]
    col = lucid.tensor(np.array([[True, False, True], [False, True, True]]), device=device)
    assert lucid.masked_select(
        lucid.tensor(np.arange(3.0, dtype=np.float32), device=device), col
    ).tolist() == [0.0, 2.0, 1.0, 2.0]


# ── ctc_loss ──────────────────────────────────────────────────────────────────


def _ctc_case() -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    rng = np.random.default_rng(0)
    logits = rng.standard_normal((10, 3, 6))
    padded = np.array([[1, 2, 2], [3, 0, 0], [4, 5, 1]], dtype=np.int64)
    return logits, padded, [10, 7, 4], [3, 1, 3]


def test_ctc_padded_and_concatenated_targets_agree() -> None:
    logits, padded, il, tl = _ctc_case()
    lp = F.log_softmax(lucid.tensor(logits), dim=-1)
    concat = np.concatenate([row[:n] for row, n in zip(padded, tl)])
    a = F.ctc_loss(lp, lucid.tensor(padded), il, tl, reduction="none").numpy()
    b = F.ctc_loss(lp, lucid.tensor(concat), il, tl, reduction="none").numpy()
    np.testing.assert_allclose(a, b, rtol=1e-12)
    mean = float(F.ctc_loss(lp, lucid.tensor(padded), il, tl, reduction="mean").item())
    assert mean == pytest.approx(float(np.mean(a / np.array(tl))), rel=1e-12)


def test_ctc_infeasible_alignment_is_infinite_unless_zeroed() -> None:
    logits, padded, il, tl = _ctc_case()
    lp = F.log_softmax(lucid.tensor(logits), dim=-1)
    short = [10, 7, 2]  # three labels cannot fit in two frames
    losses = F.ctc_loss(lp, lucid.tensor(padded), short, tl, reduction="none").numpy()
    assert np.isinf(losses[2]) and np.isfinite(losses[:2]).all()
    zeroed = F.ctc_loss(lp, lucid.tensor(padded), short, tl, reduction="none", zero_infinity=True)
    assert float(zeroed.numpy()[2]) == 0.0


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_ctc_differentiates_through_log_softmax(reduction: str) -> None:
    logits, padded, il, tl = _ctc_case()
    rng = np.random.default_rng(1)

    def f(x):  # type: ignore[no-untyped-def]
        return F.ctc_loss(
            F.log_softmax(x, dim=-1), lucid.tensor(padded), il, tl, reduction=reduction
        )

    analytic, numeric = _directional(f, logits, rng)
    assert analytic == pytest.approx(numeric, rel=1e-6, abs=1e-8)


@pytest.mark.skipif(not metal_available(), reason="metal unavailable")
def test_ctc_on_metal_with_cpu_or_metal_integers() -> None:
    logits, padded, il, tl = _ctc_case()
    cpu = F.ctc_loss(
        F.log_softmax(lucid.tensor(logits.astype(np.float32)), dim=-1),
        lucid.tensor(padded),
        il,
        tl,
    )
    for where in ("cpu", "metal"):
        x = lucid.tensor(logits.astype(np.float32), device="metal", requires_grad=True)
        loss = F.ctc_loss(
            F.log_softmax(x, dim=-1),
            lucid.tensor(padded, device=where),
            lucid.tensor(il, device=where),
            lucid.tensor(tl, device=where),
        )
        loss.backward()
        assert float(loss.item()) == pytest.approx(float(cpu.item()), rel=1e-5)
        assert np.isfinite(x.grad.numpy()).all()


# ── linalg ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (4, 4)])
def test_pinv_matches_numpy_and_differentiates(shape: tuple[int, int]) -> None:
    rng = np.random.default_rng(sum(shape))
    a = rng.standard_normal(shape)
    np.testing.assert_allclose(
        lucid.linalg.pinv(lucid.tensor(a)).numpy(), np.linalg.pinv(a), atol=1e-10
    )
    w = rng.standard_normal(shape[::-1])
    analytic, numeric = _directional(
        lambda x: (lucid.linalg.pinv(x) * lucid.tensor(w)).sum(), a, rng
    )
    assert analytic == pytest.approx(numeric, rel=1e-6, abs=1e-8)


def test_pinv_of_a_singular_square_matrix_is_the_pseudo_inverse() -> None:
    a = np.array([[1.0, 2.0], [2.0, 4.0]])
    np.testing.assert_allclose(
        lucid.linalg.pinv(lucid.tensor(a)).numpy(), np.linalg.pinv(a), atol=1e-12
    )


@pytest.mark.parametrize("shape", [(5, 3), (3, 5), "rank-deficient"])
def test_lstsq_is_the_minimum_norm_solution_and_differentiates(shape) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(3)
    a = (
        rng.standard_normal((6, 2)) @ rng.standard_normal((2, 4))
        if shape == "rank-deficient"
        else rng.standard_normal(shape)
    )
    b = rng.standard_normal((a.shape[0], 2))
    got = lucid.linalg.lstsq(lucid.tensor(a), lucid.tensor(b))[0].numpy()
    np.testing.assert_allclose(got, np.linalg.lstsq(a, b, rcond=None)[0], atol=1e-9)
    if shape != "rank-deficient":  # a rank change has no derivative
        w = rng.standard_normal(got.shape)
        analytic, numeric = _directional(
            lambda x: (lucid.linalg.lstsq(x, lucid.tensor(b))[0] * lucid.tensor(w)).sum(),
            a,
            rng,
        )
        assert analytic == pytest.approx(numeric, rel=1e-6, abs=1e-8)


def test_lu_and_lu_factor_differentiate_twice() -> None:
    rng = np.random.default_rng(5)
    a = rng.standard_normal((4, 4))
    w1, w2, w3 = (rng.standard_normal((4, 4)) for _ in range(3))

    def f(x):  # type: ignore[no-untyped-def]
        packed, _ = lucid.linalg.lu_factor(x)
        _, L, U = lucid.linalg.lu(x)
        return (
            (packed * lucid.tensor(w1)).sum()
            + (L * lucid.tensor(w2)).sum()
            + (U * lucid.tensor(w3)).sum()
        )

    analytic, numeric = _directional(f, a, rng)
    assert analytic == pytest.approx(numeric, rel=1e-6, abs=1e-8)
    x = lucid.tensor(a, requires_grad=True)
    (g,) = grad(f(x), [x], create_graph=True)
    e = rng.standard_normal(a.shape)
    (h,) = grad((g * lucid.tensor(e)).sum(), [x])
    assert np.isfinite(h.numpy()).all() and np.abs(h.numpy()).max() > 0


def test_householder_product_differentiates_in_both_inputs() -> None:
    rng = np.random.default_rng(7)
    h0, t0 = rng.standard_normal((5, 3)), rng.uniform(0.2, 1.8, 3)
    w = rng.standard_normal((5, 3))
    np.testing.assert_allclose(
        lucid.linalg.householder_product(lucid.tensor(h0, requires_grad=True), lucid.tensor(t0))
        .detach()
        .numpy(),
        lucid.linalg.householder_product(lucid.tensor(h0), lucid.tensor(t0)).numpy(),
        atol=1e-12,
    )
    analytic, numeric = _directional(
        lambda x: (lucid.linalg.householder_product(x, lucid.tensor(t0)) * lucid.tensor(w)).sum(),
        h0,
        rng,
    )
    assert analytic == pytest.approx(numeric, rel=1e-6, abs=1e-8)
    analytic, numeric = _directional(
        lambda t: (lucid.linalg.householder_product(lucid.tensor(h0), t) * lucid.tensor(w)).sum(),
        t0,
        rng,
    )
    assert analytic == pytest.approx(numeric, rel=1e-6, abs=1e-8)


def test_eig_differentiates_its_values_and_phase_free_vectors() -> None:
    rng = np.random.default_rng(11)
    a = rng.standard_normal((4, 4))
    c1, c2, wv = (
        rng.standard_normal(4),
        rng.standard_normal(4),
        rng.standard_normal((4, 4)),
    )

    def f(x):  # type: ignore[no-untyped-def]
        w, V = lucid.linalg.eig(x)
        return (
            (lucid.real(w) * lucid.tensor(c1)).sum()
            + (lucid.imag(w) * lucid.tensor(c2)).sum()
            + (V.abs() * lucid.tensor(wv)).sum()
        )

    analytic, numeric = _directional(f, a, rng)
    assert analytic == pytest.approx(numeric, rel=1e-5, abs=1e-7)
    x = lucid.tensor(a, requires_grad=True)
    (g,) = grad(lucid.real(lucid.linalg.eigvals(x)).sum(), [x])
    np.testing.assert_allclose(g.numpy(), np.eye(4), atol=1e-9)  # d trace / dA


def test_parameters_to_vector_stays_in_the_graph() -> None:
    layer = nn.Linear(3, 2)
    v = nn.utils.parameters_to_vector(layer.parameters())
    assert v.requires_grad
    (v * v).sum().backward()
    np.testing.assert_allclose(
        layer.weight.grad.numpy(), 2 * layer.weight.detach().numpy(), rtol=1e-6
    )
