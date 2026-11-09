"""``slogdet`` / ``logdet`` stay correct where ``det`` leaves the float range (LCD-939).

The sign and ``log|det|`` come from the LU diagonal, so a determinant that
overflows to ``inf`` or underflows to ``0`` no longer drags them to
``(·, inf)`` / ``(0, -inf)``, and the gradient is ``A⁻ᵀ`` instead of
``cof(A) / det(A)``, whose cofactor overflows exactly where the answer exists.
"""

import math
from types import ModuleType

import pytest

import lucid
from lucid.linalg import slogdet
from lucid.test._fixtures.devices import metal_available

# log(10) * 400 and its negation, in float64 — ln(10) is 2.302585092994046.
_LOG_10_X_400 = 921.0340371976183


def _devices() -> list[object]:
    metal = pytest.mark.skipif(not metal_available(), reason="needs Metal")
    return ["cpu", pytest.param("metal", marks=metal)]


def _spd_cond20(dtype: lucid.dtype = lucid.float32, device: str = "cpu") -> lucid.Tensor:
    lucid.manual_seed(0)
    M = lucid.randn(80, 80, dtype=lucid.float64)
    C = M @ M.mT / 80 * 0.05 + 0.01 * lucid.eye(80, dtype=lucid.float64)
    return C.to(dtype).to(device)


def _chol_logdet(C: lucid.Tensor) -> float:
    """``2 Σ log L_ii`` from a float64 Cholesky: a route that never forms det."""
    L = lucid.linalg.cholesky(C.to("cpu").to(lucid.float64))
    return float(2.0 * lucid.log(lucid.linalg.diagonal(L, dim1=-2, dim2=-1)).sum().item())


@pytest.mark.parametrize("device", _devices())
def test_spd_80x80_has_a_finite_logdet_and_an_inverse_transpose_gradient(device: str) -> None:
    C = _spd_cond20(device=device).requires_grad_()
    sign, logabs = slogdet(C)
    assert sign.item() == 1.0
    assert math.isclose(logabs.item(), _chol_logdet(C), rel_tol=1e-4)
    assert math.isclose(lucid.logdet(C).item(), logabs.item(), rel_tol=1e-6)
    logabs.backward()
    assert C.grad is not None
    want = lucid.linalg.inv(C.detach().to("cpu").to(lucid.float64)).mT
    got = C.grad.to("cpu").to(lucid.float64)
    assert bool(lucid.isfinite(got).all())
    assert float((got - want).abs().max().item()) < 1e-3 * float(want.abs().max().item())


@pytest.mark.parametrize("device", _devices())
def test_logdet_of_the_spd_matrix_has_a_finite_gradient(device: str) -> None:
    C = _spd_cond20(device=device).requires_grad_()
    lucid.logdet(C).backward()
    assert C.grad is not None and bool(lucid.isfinite(C.grad).all())


@pytest.mark.parametrize("device", _devices())
def test_randn_64x64_logabsdet_is_finite_and_agrees_with_float64(device: str) -> None:
    lucid.manual_seed(1)
    M = lucid.randn(64, 64, dtype=lucid.float64)
    s32, l32 = slogdet(M.to(lucid.float32).to(device))
    s64, l64 = slogdet(M)
    assert math.isfinite(l32.item())
    assert s32.item() == s64.item()
    assert math.isclose(l32.item(), l64.item(), rel_tol=1e-4)


_DTYPES = [lucid.float32, lucid.float64]


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_a_diagonal_whose_determinant_overflows(dtype: lucid.dtype, device: str) -> None:
    if device == "metal" and dtype == lucid.float64:
        pytest.skip("no float64 on Metal")
    A = (lucid.eye(400, dtype=dtype, device=device) * 10).requires_grad_()
    sign, logabs = slogdet(A)
    assert sign.item() == 1.0
    assert math.isclose(logabs.item(), _LOG_10_X_400, rel_tol=1e-5)
    assert lucid.logdet(A).item() == pytest.approx(_LOG_10_X_400, rel=1e-5)
    logabs.backward()
    assert A.grad is not None
    assert bool(lucid.isfinite(A.grad).all())
    assert float(A.grad.diagonal().mean().item()) == pytest.approx(0.1, rel=1e-5)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", _DTYPES)
def test_a_diagonal_whose_determinant_underflows(dtype: lucid.dtype, device: str) -> None:
    if device == "metal" and dtype == lucid.float64:
        pytest.skip("no float64 on Metal")
    A = (lucid.eye(40, dtype=dtype, device=device) * 1e-10).requires_grad_()
    sign, logabs = slogdet(A)
    assert sign.item() == 1.0
    assert math.isclose(logabs.item(), -_LOG_10_X_400, rel_tol=1e-5)
    logabs.backward()
    assert A.grad is not None
    got = A.grad.to("cpu").to(lucid.float64)
    assert bool(lucid.isfinite(got).all())
    assert float(got.diagonal().mean().item()) == pytest.approx(1e10, rel=1e-5)
    assert float(got.abs().sum().item()) == pytest.approx(40 * 1e10, rel=1e-5)


def test_the_gradient_matches_a_finite_difference_in_float64() -> None:
    lucid.manual_seed(2)
    A = lucid.randn(4, 4, dtype=lucid.float64) + 3 * lucid.eye(4, dtype=lucid.float64)
    assert lucid.autograd.gradcheck(lambda a: slogdet(a)[1], [A.clone().requires_grad_()])
    assert lucid.autograd.gradcheck(lambda a: lucid.logdet(a), [A.clone().requires_grad_()])


def test_the_gradient_differentiates_twice() -> None:
    lucid.manual_seed(3)
    A = lucid.randn(3, 3, dtype=lucid.float64) + 3 * lucid.eye(3, dtype=lucid.float64)
    assert lucid.autograd.gradgradcheck(lambda a: slogdet(a)[1], [A.clone().requires_grad_()])


def test_the_gradient_is_the_inverse_transpose_on_a_well_conditioned_batch() -> None:
    lucid.manual_seed(4)
    A = lucid.randn(3, 5, 5, dtype=lucid.float64) + 4 * lucid.eye(5, dtype=lucid.float64)
    A.requires_grad_()
    slogdet(A)[1].sum().backward()
    assert A.grad is not None
    want = lucid.linalg.inv(A.detach()).mT
    assert float((A.grad - want).abs().max().item()) < 1e-10


def test_a_batch_of_mixed_sign_determinants() -> None:
    rows = [[[2.0, 0.0], [0.0, 3.0]], [[0.0, 1.0], [1.0, 0.0]], [[-2.0, 0.0], [0.0, -3.0]]]
    A = lucid.tensor(rows, dtype=lucid.float64)
    sign, logabs = slogdet(A)
    assert sign.tolist() == [1.0, -1.0, 1.0]
    assert logabs.tolist() == pytest.approx([math.log(6.0), 0.0, math.log(6.0)])
    # logdet is NaN only for the negative determinant.
    got = lucid.logdet(A).tolist()
    assert got[0] == pytest.approx(math.log(6.0)) and math.isnan(got[1])
    assert got[2] == pytest.approx(math.log(6.0))


@pytest.mark.parametrize("device", _devices())
def test_a_batch_mixing_singular_regular_and_overflowing(device: str) -> None:
    # A regular matrix whose det overflows next to a singular one: neither may
    # poison the other's value or gradient.
    big = lucid.eye(3, dtype=lucid.float32, device=device) * 1e20
    sing = lucid.tensor([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 1.0]], device=device)
    reg = lucid.tensor([[3.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]], device=device)
    A = lucid.stack([big, sing, reg]).requires_grad_()
    sign, logabs = slogdet(A)
    assert sign.to("cpu").tolist() == [1.0, 0.0, 1.0]
    got = logabs.to("cpu").tolist()
    assert got[0] == pytest.approx(3 * 20 * math.log(10), rel=1e-5)
    assert got[1] == -math.inf
    assert got[2] == pytest.approx(math.log(5.0), rel=1e-5)
    logabs.sum().backward()
    assert A.grad is not None
    g = A.grad.to("cpu")
    assert g[0].diagonal().tolist() == pytest.approx([1e-20] * 3, rel=1e-4)
    want_reg = lucid.linalg.inv(reg.to("cpu")).mT
    assert float((g[2] - want_reg).abs().max().item()) < 1e-5
    # The singular matrix keeps its infinite, signed-by-the-zero gradient.
    assert any(math.isinf(v) for row in g[1].tolist() for v in row)


@pytest.mark.parametrize("device", _devices())
def test_a_singular_matrix_keeps_its_contract(device: str) -> None:
    A = lucid.tensor([[1.0, 2.0], [2.0, 4.0]], device=device).requires_grad_()
    sign, logabs = slogdet(A)
    assert sign.item() == 0.0 and math.copysign(1.0, sign.item()) == -1.0
    assert logabs.item() == -math.inf
    assert lucid.logdet(A.detach()).item() == -math.inf
    logabs.backward()
    assert A.grad is not None
    assert A.grad.to("cpu").tolist() == [[-math.inf, math.inf], [math.inf, -math.inf]]


def test_the_empty_matrix_and_the_empty_batch() -> None:
    sign, logabs = slogdet(lucid.zeros(3, 0, 0))
    assert sign.tolist() == [1.0] * 3 and logabs.tolist() == [0.0] * 3
    sign, logabs = slogdet(lucid.zeros(0, 2, 2))
    assert tuple(sign.shape) == (0,) and tuple(logabs.shape) == (0,)


@pytest.mark.parametrize("bad", [lucid.zeros(2, 3), lucid.zeros(3), lucid.zeros(2, 2, dtype=lucid.int64)])
def test_an_invalid_input_is_refused_as_det_refuses_it(bad: lucid.Tensor) -> None:
    with pytest.raises(Exception) as via_det:
        lucid.linalg.det(bad)
    with pytest.raises(type(via_det.value)):
        slogdet(bad)


@pytest.mark.parity
@pytest.mark.parametrize("device", _devices())
def test_the_four_rows_have_the_references_values(device: str, ref: ModuleType) -> None:
    lucid.manual_seed(5)
    cases = {
        "spd": _spd_cond20(device="cpu"),
        "randn": lucid.randn(64, 64),
        "big": lucid.eye(400) * 10,
        "small": lucid.eye(40) * 1e-10,
    }
    for name, M in cases.items():
        got_sign, got_log = slogdet(M.to(device))
        # The reference in float64: the truth the float32 answer is held to.
        want = ref.linalg.slogdet(ref.tensor(M.to(lucid.float64).tolist(), dtype=ref.float64))
        assert got_sign.item() == float(want.sign), name
        assert got_log.item() == pytest.approx(float(want.logabsdet), rel=1e-4), name
