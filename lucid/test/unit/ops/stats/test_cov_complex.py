"""``cov`` and ``corrcoef`` of a complex input are Hermitian, as the reference's are.

The covariance is ``sum_k (x_i - mean_i) * conj(x_j - mean_j) / (M - correction)``.
``cov`` multiplied the centred rows by their plain transpose, leaving out the
conjugate, so a complex input gave a matrix that was neither Hermitian nor the
reference's: its diagonal was complex and ``cov([[1+2j, 2-1j, .5j], [3, 1-1j,
2+2j]])[0, 0]`` was ``-1.25-1.5j`` where the variance is ``3.25`` (LCD-663).
``corrcoef`` divided by the square root of that complex diagonal.

The CPU stream has no complex matmul kernel (LCD-667), so the complex cases run
on Metal, and the CPU one is pinned as a strict expected failure to flip when
that lands.
"""

import cmath
from types import ModuleType

import pytest

import lucid
from lucid.test._fixtures.devices import metal_available

_needs_metal = pytest.mark.skipif(not metal_available(), reason="complex matmul needs Metal")

_X = [[1 + 2j, 2 - 1j, 0.5j], [3 + 0j, 1 - 1j, 2 + 2j]]
_FWEIGHTS = [1, 2, 1]
_AWEIGHTS = [0.5, 1.0, 2.0]
_CASES = {
    "plain": {},
    "population": {"correction": 0},
    "fweights": {"fweights": _FWEIGHTS},
    "aweights": {"aweights": _AWEIGHTS},
    "both-weights": {"correction": 0, "fweights": _FWEIGHTS, "aweights": _AWEIGHTS},
}


def _weights(
    case: dict[str, object], lib: ModuleType, **tensor_kwargs: object
) -> dict[str, object]:
    """``case`` with each weight list made a tensor of ``lib``."""
    return {
        k: lib.tensor(v, **tensor_kwargs) if isinstance(v, list) else v for k, v in case.items()
    }


def _assert_close(
    got: list[list[complex]], want: list[list[complex]], rel_tol: float = 2e-5
) -> None:
    scale = max(abs(w) for row in want for w in row)
    for g_row, w_row in zip(got, want, strict=True):
        for g, w in zip(g_row, w_row, strict=True):
            assert cmath.isclose(g, w, rel_tol=rel_tol, abs_tol=rel_tol * scale), f"{g!r} vs {w!r}"


@_needs_metal
@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
def test_complex_cov_matches_the_reference(ref: ModuleType, case: str) -> None:
    kwargs = _CASES[case]
    got = lucid.cov(
        lucid.tensor(_X, dtype=lucid.complex64, device="metal"),
        **_weights(kwargs, lucid, device="metal"),
    )
    want = ref.cov(ref.tensor(_X, dtype=ref.complex64), **_weights(kwargs, ref))
    assert got.dtype == lucid.complex64
    _assert_close(got.to("cpu").tolist(), want.tolist())


@_needs_metal
def test_complex_corrcoef_matches_the_reference(ref: ModuleType) -> None:
    got = lucid.corrcoef(lucid.tensor(_X, dtype=lucid.complex64, device="metal"))
    want = ref.corrcoef(ref.tensor(_X, dtype=ref.complex64))
    _assert_close(got.to("cpu").tolist(), want.tolist())


@_needs_metal
@pytest.mark.parametrize("op", ["cov", "corrcoef"])
def test_complex_cov_is_hermitian_with_a_real_diagonal(op: str) -> None:
    lucid.manual_seed(0)
    x = lucid.complex(lucid.randn(5, 12, device="metal"), lucid.randn(5, 12, device="metal"))
    c = getattr(lucid, op)(x).to("cpu").tolist()
    for i, row in enumerate(c):
        # Rounding leaves the diagonal's imaginary part at a few ulp of its size.
        assert abs(row[i].imag) <= 1e-5 * abs(row[i].real)
        assert row[i].real > 0.0
        for j in range(i + 1, len(c)):
            assert cmath.isclose(row[j], c[j][i].conjugate(), rel_tol=1e-5, abs_tol=1e-5)
    if op == "corrcoef":
        assert all(abs(c[i][i] - 1.0) <= 1e-5 for i in range(len(c)))
        assert all(abs(z) <= 1.0 + 1e-5 for row in c for z in row)


@pytest.mark.parametrize("dtype", [lucid.float32, lucid.float64])
def test_real_cov_is_the_plain_product(dtype: lucid.dtype) -> None:
    lucid.manual_seed(1)
    x = lucid.randn(4, 9, dtype=dtype)
    xc = x - x.sum(dim=1, keepdim=True) * (1.0 / 9.0)
    want = lucid.matmul(xc, xc.swapaxes(-1, -2)) * (1.0 / 8.0)
    assert lucid.cov(x).tolist() == want.tolist()


@pytest.mark.xfail(strict=True, reason="LCD-667: the CPU stream has no complex matmul kernel")
def test_complex_cov_on_cpu_is_hermitian() -> None:
    c = lucid.cov(lucid.tensor(_X, dtype=lucid.complex64))
    assert cmath.isclose(c.tolist()[0][1], 0.25 + 3j, rel_tol=1e-5)
