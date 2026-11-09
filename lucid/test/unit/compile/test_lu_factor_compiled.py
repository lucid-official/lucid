"""``lu_factor``, ``det`` and ``slogdet`` compile (LCD-942).

``lu_factor`` has a compile emitter: an unrolled partial-pivoting elimination
that returns the packed factor, the one-based pivots and ``info`` the way eager's
``getrf`` does.  ``det`` of a batched or larger-than-3x3 matrix is the product of
its pivots, and ``slogdet`` (which reads the LU diagonal since LCD-939) compiles
on top of both, with the values eager gives where the determinant itself would
leave the float range.

The emitter takes up to ``kMaxLuDim`` (32) rows: one serial chain of small
kernels per step makes it slower than the eager fallback beyond that.  A larger
matrix still answers correctly, by running the call eagerly.
"""

import math
import warnings

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.compile import make_step
from lucid.compile._core.fallback import CompileFallbackWarning
from lucid.linalg import _slogdet

_LA = lucid.linalg
_MAX_LU_DIM = 32


def _compiled(fn: object, *args: lucid.Tensor) -> tuple[object, object]:
    """``fn(*args)`` through ``lucid.compile``, which must not have fallen back."""
    cf = lucid.compile(fn)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        with lucid.no_grad():
            out = cf(*args)
    assert cf.cache_info()["entries"] == 1
    assert not cf.cache_info()["eager_only"]
    return out, cf


def _lu(a: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor]:
    return _LA._lu_factor_packed(a)


def _assert_lu_matches_eager(a: lucid.Tensor, *, tol: float = 5e-6) -> None:
    got, _ = _compiled(_lu, a)
    want = _lu(a)
    lu_g, lu_w = got[0].to("cpu"), want[0].to("cpu")
    scale = max(1.0, float(lu_w.abs().max().item()))
    assert float((lu_g - lu_w).abs().max().item()) <= tol * scale
    assert got[1].to("cpu").tolist() == want[1].to("cpu").tolist()
    assert got[2].to("cpu").tolist() == want[2].to("cpu").tolist()
    assert tuple(got[1].shape) == tuple(want[1].shape) and tuple(got[2].shape) == tuple(
        want[2].shape
    )


@pytest.mark.parametrize(
    "shape",
    [(1, 1), (2, 2), (3, 3), (7, 7), (16, 16), (32, 32), (4, 6), (6, 4), (1, 5), (5, 1)]
    + [(3, 4, 4), (2, 3, 5, 5), (5, 1, 1), (2, 4, 6)],
)
def test_lu_factor_matches_eager(shape: tuple[int, ...], device_gpu_only: str) -> None:
    lucid.manual_seed(sum(shape))
    _assert_lu_matches_eager(lucid.randn(*shape, device=device_gpu_only))


@pytest.mark.parametrize(
    "rows",
    [
        # Singular: a zero pivot is reported in ``info`` and the factor goes on.
        [[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0], [0.0, 0.0]],
        # Rank 1 with multipliers that are exact in float32.  A rank-deficient
        # matrix whose multipliers are not (a pivot of 3) leaves a rounding
        # residue where the exact zero is, and which residue depends on the
        # order the kernel subtracts in: eager's LAPACK and the graph differ
        # there, in the factor and in ``info``, and both are right.
        [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [4.0, 8.0, 12.0]],
        # The largest entry is not on the diagonal, and ties pick the first row.
        [[1.0, 2.0], [-1.0, 3.0]],
        [[2.0, 1.0, 1.0], [-2.0, 3.0, 1.0], [2.0, 5.0, 7.0]],
        [[0.0, 1.0], [1.0, 0.0]],
    ],
)
def test_lu_factor_singular_tied_and_permuted(
    rows: list[list[float]], device_gpu_only: str
) -> None:
    _assert_lu_matches_eager(lucid.tensor(rows, device=device_gpu_only))


def test_lu_factor_reports_the_first_zero_pivot_of_each_batch_member(
    device_gpu_only: str,
) -> None:
    ok = [[2.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]]
    late = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    early = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    a = lucid.tensor([ok, late, early], device=device_gpu_only)
    got, _ = _compiled(_lu, a)
    assert got[2].to("cpu").tolist() == [0, 3, 1]
    _assert_lu_matches_eager(a)


def test_lu_factor_past_the_unroll_bound_runs_eagerly(device_gpu_only: str) -> None:
    n = _MAX_LU_DIM + 1
    lucid.manual_seed(7)
    a = lucid.randn(n, n, device=device_gpu_only)
    cf = lucid.compile(_lu)
    with pytest.warns(CompileFallbackWarning, match="lu_factor"):
        with lucid.no_grad():
            got = cf(a)
    want = _lu(a)
    assert float((got[0].to("cpu") - want[0].to("cpu")).abs().max().item()) < 1e-3
    assert got[1].to("cpu").tolist() == want[1].to("cpu").tolist()


@pytest.mark.parametrize("shape", [(4, 4), (6, 6), (32, 32), (3, 3, 3), (2, 5, 5), (3, 2, 2)])
def test_det_beyond_the_closed_forms_matches_eager(
    shape: tuple[int, ...], device_gpu_only: str
) -> None:
    lucid.manual_seed(sum(shape))
    a = lucid.randn(*shape, device=device_gpu_only) + 2.0 * lucid.eye(
        shape[-1], device=device_gpu_only
    )
    got, _ = _compiled(_LA.det, a)
    want = _LA.det(a.to("cpu"))
    assert tuple(got.shape) == tuple(want.shape)
    assert lucid.allclose(got.to("cpu"), want, rtol=2e-4, atol=1e-5)


def _slogdet_pair(a: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
    return _LA.slogdet(a)


def _assert_slogdet_matches_eager(a: lucid.Tensor) -> None:
    (sign_c, log_c), _ = _compiled(_slogdet_pair, a)
    sign_e, log_e = _LA.slogdet(a.to("cpu"))
    assert sign_c.to("cpu").tolist() == sign_e.tolist()
    for got, want in zip(
        log_c.to("cpu").reshape(-1).tolist(), log_e.reshape(-1).tolist(), strict=True
    ):
        assert got == want if math.isinf(want) else got == pytest.approx(want, rel=1e-4, abs=1e-5)


@pytest.mark.parametrize(
    "scale, n",
    [
        # det overflows float32: 1e20 ** 32 and 10 ** 32 * 1e30.
        (1e20, 32),
        (1e20, 4),
        (1e30, 8),
        # ... and underflows to zero.
        (1e-20, 32),
        (1e-10, 30),
        (1e-30, 8),
        (10.0, 32),
    ],
)
def test_slogdet_where_det_leaves_the_float_range(
    scale: float, n: int, device_gpu_only: str
) -> None:
    a = lucid.eye(n, device=device_gpu_only) * scale
    (sign, logabs), _ = _compiled(_slogdet_pair, a)
    assert sign.item() == 1.0
    assert logabs.item() == pytest.approx(n * math.log(scale), rel=1e-5)
    assert math.isfinite(logabs.item())
    # What eager gives, which is what the compiled call must not replace with
    # ``log(abs(det))`` (inf, or -inf) when it traces.
    _assert_slogdet_matches_eager(a)


def test_slogdet_of_a_random_matrix_and_a_negative_determinant(device_gpu_only: str) -> None:
    lucid.manual_seed(11)
    for n in (2, 5, 16, 32):
        _assert_slogdet_matches_eager(lucid.randn(n, n, device=device_gpu_only))
    flipped = lucid.tensor([[0.0, 1.0], [1.0, 0.0]], device=device_gpu_only)
    (sign, logabs), _ = _compiled(_slogdet_pair, flipped)
    assert sign.item() == -1.0 and logabs.item() == 0.0


def test_slogdet_of_a_singular_matrix_keeps_its_signed_zero(device_gpu_only: str) -> None:
    a = lucid.tensor([[1.0, 2.0], [2.0, 4.0]], device=device_gpu_only)
    (sign, logabs), _ = _compiled(_slogdet_pair, a)
    assert sign.item() == 0.0 and math.copysign(1.0, sign.item()) == -1.0
    assert logabs.item() == -math.inf
    _assert_slogdet_matches_eager(a)


def test_slogdet_of_a_batch_mixing_singular_regular_and_overflowing(
    device_gpu_only: str,
) -> None:
    big = lucid.eye(3, device=device_gpu_only) * 1e20
    sing = lucid.tensor([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 1.0]], device=device_gpu_only)
    reg = lucid.tensor([[3.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]], device=device_gpu_only)
    a = lucid.stack([big, sing, reg])
    (sign, logabs), _ = _compiled(_slogdet_pair, a)
    assert sign.to("cpu").tolist() == [1.0, 0.0, 1.0]
    got = logabs.to("cpu").tolist()
    assert got[0] == pytest.approx(3 * 20 * math.log(10), rel=1e-5)
    assert got[1] == -math.inf
    assert got[2] == pytest.approx(math.log(5.0), rel=1e-5)
    _assert_slogdet_matches_eager(a)


@pytest.mark.parametrize("shape", [(5, 4, 4), (2, 3, 6, 6), (4, 1, 1), (3, 32, 32)])
def test_slogdet_of_a_batch_matches_eager(shape: tuple[int, ...], device_gpu_only: str) -> None:
    lucid.manual_seed(sum(shape))
    _assert_slogdet_matches_eager(lucid.randn(*shape, device=device_gpu_only))


def test_logdet_is_nan_only_for_a_negative_determinant(device_gpu_only: str) -> None:
    rows = [[[2.0, 0.0], [0.0, 3.0]], [[0.0, 1.0], [1.0, 0.0]], [[1.0, 2.0], [2.0, 4.0]]]
    a = lucid.tensor(rows, device=device_gpu_only)
    got, _ = _compiled(lucid.logdet, a)
    want = lucid.logdet(a.to("cpu")).tolist()
    got = got.to("cpu").tolist()
    assert got[0] == pytest.approx(want[0]) and math.isnan(got[1]) and got[2] == -math.inf


def test_slogdet_past_the_unroll_bound_runs_eagerly_and_stays_right(
    device_gpu_only: str,
) -> None:
    a = lucid.eye(_MAX_LU_DIM + 8, device=device_gpu_only) * 10
    cf = lucid.compile(_slogdet_pair)
    with pytest.warns(CompileFallbackWarning):
        with lucid.no_grad():
            sign, logabs = cf(a)
    assert sign.item() == 1.0
    assert logabs.item() == pytest.approx((_MAX_LU_DIM + 8) * math.log(10), rel=1e-5)


class _SlogdetNet(nn.Module):
    def __init__(self, n: int, batch: int | None) -> None:
        super().__init__()
        self.n, self.batch = n, batch
        self.fc = nn.Linear(8, 40 * n * n)
        self.out = nn.Linear(40 * n * n, 8)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = self.fc(x)
        k = h[:, : self.n * self.n]
        mats = k[: self.batch or 1].reshape(-1, self.n, self.n) * 0.1
        mats = mats + lucid.eye(self.n, device=x.device) * 2.0
        sign, logabs = _LA.slogdet(mats if self.batch else mats[0])
        return self.out(h) + (sign * logabs).sum() * 0.1


@pytest.fixture
def slogdet_traced_backward(monkeypatch: pytest.MonkeyPatch) -> None:
    """The marker LCD-939 dropped; the tests below are about what is behind it."""
    monkeypatch.setattr(_slogdet._SlogdetFromLU, "_traced_backward", True)


@pytest.mark.parametrize("n, batch", [(3, None), (4, None), (4, 3), (8, 5)])
def test_a_training_step_through_slogdet_falls_back_with_the_eager_gradient(
    n: int, batch: int | None, slogdet_traced_backward: None, device_gpu_only: str
) -> None:
    """``DetVjp`` compiles only an unbatched matrix up to 3x3, and the singular
    matrix's gradient goes through it, so the step answers eagerly: warned, and
    with the eager value and gradients."""
    lucid.manual_seed(0)
    x = lucid.randn(6, 8, device=device_gpu_only)
    t = lucid.randn(6, 8, device=device_gpu_only)
    lucid.manual_seed(1)
    ref = _SlogdetNet(n, batch).to(device_gpu_only)
    lucid.manual_seed(1)
    net = _SlogdetNet(n, batch).to(device_gpu_only)
    want_loss = F.mse_loss(ref(x), t)
    want_loss.backward()
    step = make_step(net, F.mse_loss)
    with pytest.warns(CompileFallbackWarning):
        loss = step(x, t)
        loss.backward()
    assert loss.item() == pytest.approx(want_loss.item(), rel=1e-5)
    for p, q in zip(net.parameters(), ref.parameters(), strict=True):
        assert p.grad is not None and q.grad is not None
        assert lucid.allclose(p.grad.to("cpu"), q.grad.to("cpu"), rtol=1e-4, atol=1e-5)
