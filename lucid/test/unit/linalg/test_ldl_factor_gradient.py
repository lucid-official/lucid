"""``ldl_factor`` / ``ldl_factor_ex`` carry a gradient (LCD-413).

The packed factor came back with ``requires_grad=False`` from an input
that required grad, so a loss through it trained nothing without a word.
The reference framework defines no derivative here (its backward raises);
Lucid differentiates the factor wherever the Bunch-Kaufman pivots are
constant.  Every check is a central difference in float64 — first and
second order — so no oracle is needed; Metal is held to the CPU's float64
gradient.  The cases cover 1×1 pivots only (SPD), 2×2 pivots, row swaps,
and a batch that mixes them.
"""

import random

import pytest

import lucid
import lucid.linalg as LA
from lucid.autograd import gradcheck, gradgradcheck

_SWAPS_AND_2X2 = [
    [1.0, 2.0, 0.0, 3.0],
    [2.0, 0.5, 3.0, 1.0],
    [0.0, 3.0, -2.0, 4.0],
    [3.0, 1.0, 4.0, 0.1],
]


def _symmetric(n: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    a = [[rng.gauss(0.0, 1.0) for _ in range(n)] for _ in range(n)]
    return [[a[i][j] + a[j][i] for j in range(n)] for i in range(n)]


def _spd(n: int, seed: int) -> list[list[float]]:
    a = _symmetric(n, seed)
    return [
        [sum(a[i][k] * a[j][k] for k in range(n)) + (n if i == j else 0.0) for j in range(n)]
        for i in range(n)
    ]


_CASES = {
    "spd2": [[4.0, 1.0], [1.0, 3.0]],
    "spd5": _spd(5, 1),
    "zero_diagonal_2x2": [[0.0, 1.0], [1.0, 0.0]],
    "swaps_and_2x2": _SWAPS_AND_2X2,
    **{f"indefinite{n}_{s}": _symmetric(n, s) for n in (3, 6) for s in range(3)},
}


def _pivot_kinds(a: list[list[float]]) -> set[str]:
    _, piv = LA.ldl_factor(lucid.tensor(a, dtype=lucid.float64))
    kinds = {"2x2" if p < 0 else "1x1" for p in piv.tolist()}
    if any(abs(p) - 1 != i for i, p in enumerate(piv.tolist()) if p > 0):
        kinds.add("swap")
    return kinds


def test_the_cases_reach_every_pivot_kind() -> None:
    kinds = set().union(*(_pivot_kinds(a) for a in _CASES.values()))
    assert kinds == {"1x1", "2x2", "swap"}


@pytest.mark.parametrize("name", list(_CASES))
@pytest.mark.parametrize("op", ["ldl_factor", "ldl_factor_ex"])
def test_first_and_second_order_match_central_differences(op: str, name: str) -> None:
    A = lucid.tensor(_CASES[name], dtype=lucid.float64, requires_grad=True)
    w = lucid.randn(*A.shape, dtype=lucid.float64)

    def loss(a: lucid.Tensor) -> lucid.Tensor:
        return (getattr(LA, op)(a)[0] * w).sum()

    assert gradcheck(loss, [A])
    assert gradgradcheck(loss, [A])


@pytest.mark.parametrize("op", ["ldl_factor", "ldl_factor_ex"])
def test_an_input_that_requires_grad_never_gets_a_detached_factor(op: str, device: str) -> None:
    A = lucid.tensor(_SWAPS_AND_2X2, device=device, requires_grad=True)
    LD, piv, *_ = getattr(LA, op)(A)
    assert LD.requires_grad and LD.grad_fn is not None
    assert not piv.requires_grad
    with lucid.no_grad():
        assert not getattr(LA, op)(A)[0].requires_grad
    assert not getattr(LA, op)(A.detach())[0].requires_grad


def test_the_factor_values_are_the_kernels(device: str) -> None:
    A = lucid.tensor(_SWAPS_AND_2X2, device=device)
    want = LA.ldl_factor(A)[0].tolist()
    assert LA.ldl_factor(A.requires_grad_(True))[0].tolist() == want


@pytest.mark.parametrize("name", list(_CASES))
def test_metal_agrees_with_the_cpu_gradient(name: str, device: str) -> None:
    w = lucid.tensor(_symmetric(len(_CASES[name]), 7), dtype=lucid.float64)
    A64 = lucid.tensor(_CASES[name], dtype=lucid.float64, requires_grad=True)
    (LA.ldl_factor(A64)[0] * w).sum().backward()
    A = lucid.tensor(_CASES[name], device=device, requires_grad=True)
    (LA.ldl_factor(A)[0] * w.to(lucid.float32).to(device)).sum().backward()
    got = A.grad.to("cpu").to(lucid.float64)
    scale = 1.0 + float(A64.grad.abs().max().item())
    assert float((got - A64.grad).abs().max().item()) <= 1e-4 * scale


def test_a_batch_differentiates_per_matrix_and_skips_the_failed_one(
    device: str,
) -> None:
    good = [_SWAPS_AND_2X2, _spd(4, 3)]
    A = lucid.tensor(good + [[[0.0] * 4] * 4], device=device, requires_grad=True)
    LD, _, info = LA.ldl_factor_ex(A)
    assert info.tolist()[:2] == [0, 0] and info.tolist()[2] != 0
    LD.sum().backward()
    assert A.grad[2].abs().sum().item() == 0.0
    for b, a in enumerate(good):
        single = lucid.tensor(a, device=device, requires_grad=True)
        LA.ldl_factor(single)[0].sum().backward()
        assert lucid.allclose(A.grad[b], single.grad, atol=1e-5)


def test_the_gradient_lands_on_the_lower_triangle() -> None:
    A = lucid.tensor(_SWAPS_AND_2X2, dtype=lucid.float64, requires_grad=True)
    LA.ldl_factor(A)[0].sum().backward()
    assert lucid.triu(A.grad, 1).abs().sum().item() == 0.0
    assert lucid.tril(A.grad, -1).abs().sum().item() > 0.0


def test_an_empty_matrix_backpropagates_without_error() -> None:
    A = lucid.zeros(0, 0, dtype=lucid.float64, requires_grad=True)
    LD = LA.ldl_factor_ex(A)[0]
    assert LD.requires_grad
    LD.sum().backward()
    assert A.grad is None or A.grad.numel() == 0
