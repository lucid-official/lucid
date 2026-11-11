"""``det`` / ``slogdet`` / ``inv`` train through a compiled step at any size (LCD-946).

The gradient of ``det`` is the cofactor ``cof(A)``.  Eager forms it as
``det(A) A^-T`` while ``A`` is invertible and from the SVD otherwise, because the
first is wrong at a singular matrix.  The compiled backward has one form that is
right at every rank: ``cof(A) = sgn(P) (adj(U) L^-1 P)^T`` from the LU
elimination, with ``adj(U)`` made of products of ``U``'s entries and no division
by a pivot.  ``inv`` compiles through the same elimination with the identity
carried along.  Both decline past ``kMaxLuDim`` (32) rows.
"""

import math
import warnings
from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
from lucid.compile import make_step
from lucid.compile._core.fallback import CompileFallbackWarning

_LA = lucid.linalg
_MAX_COFACTOR_DIM = 24


class _MatrixNet(nn.Module):
    """``f(A)`` with the matrix ``A`` as the one parameter, so its gradient is the answer."""

    def __init__(self, a: lucid.Tensor, kind: str) -> None:
        super().__init__()
        self.A = nn.Parameter(a.clone())
        self.kind = kind

    def forward(self, w: lucid.Tensor) -> lucid.Tensor:
        if self.kind == "det":
            return (_LA.det(self.A) * w).sum()
        if self.kind == "inv":
            return (_LA.inv(self.A) * w).sum()
        sign, logabs = _LA.slogdet(self.A)
        return (logabs * w).sum() + sign.sum() * 0.0


def _weights(kind: str, a: lucid.Tensor) -> lucid.Tensor:
    if kind == "inv":
        lucid.manual_seed(3)
        return lucid.randn(*a.shape, device=a.device)
    return lucid.ones((1,), device=a.device)


def _grads(
    kind: str, a: lucid.Tensor, *, expect_compiled: bool = True
) -> tuple[list[float], list[float]]:
    """``dA`` of ``f(A)``: from a compiled step, and from the eager call."""
    w = _weights(kind, a)
    dummy = lucid.zeros((1,), device=a.device)
    ref = _MatrixNet(a, kind).to(a.device)
    ref(w).backward()
    assert ref.A.grad is not None
    net = _MatrixNet(a, kind).to(a.device)
    step = make_step(net, lambda y, _t: y)
    with warnings.catch_warnings():
        if expect_compiled:
            warnings.simplefilter("error", CompileFallbackWarning)
        # The first call runs eagerly and records; the second is the compiled run.
        step(w, dummy).backward()
        net.A.grad = None
        step(w, dummy).backward()
    assert net.A.grad is not None
    return net.A.grad.to("cpu").reshape(-1).tolist(), ref.A.grad.to("cpu").reshape(-1).tolist()


def _assert_same_gradient(got: list[float], want: list[float], *, rtol: float = 2e-4) -> None:
    """Equal to ``rtol`` of the largest finite entry (at least 1: a cofactor that is zero
    in exact arithmetic is rounding noise in eager's SVD form); the non-finite entries
    equal in kind and sign."""
    scale = max([abs(v) for v in want if math.isfinite(v)] + [1.0])
    for g, e in zip(got, want, strict=True):
        if math.isfinite(e):
            assert math.isfinite(g) and abs(g - e) <= rtol * scale, (g, e, scale)
        elif math.isnan(e):
            assert math.isnan(g)
        else:
            assert math.isinf(g) and (g > 0) == (e > 0)


def _compiled(
    fn: Callable[[lucid.Tensor], lucid.Tensor], a: lucid.Tensor
) -> tuple[lucid.Tensor, object]:
    """``fn(a)`` through ``lucid.compile``, which must not have fallen back."""
    cf = lucid.compile(fn)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        with lucid.no_grad():
            out = cf(a)
    assert not cf.cache_info()["eager_only"]
    return out, cf


def _well_conditioned(*shape: int, device: str, seed: int) -> lucid.Tensor:
    lucid.manual_seed(seed)
    return lucid.randn(*shape, device=device) * 0.3 + 2.0 * lucid.eye(shape[-1], device=device)


@pytest.mark.parametrize(
    "shape",
    [(3, 3), (4, 4), (8, 8), (16, 16), (24, 24), (5, 3, 3), (3, 4, 4), (5, 8, 8), (2, 3, 6, 6)],
)
@pytest.mark.parametrize("kind", ["det", "slogdet"])
def test_a_det_step_compiles_with_the_eager_gradient(
    kind: str, shape: tuple[int, ...], device_gpu_only: str
) -> None:
    a = _well_conditioned(*shape, device=device_gpu_only, seed=sum(shape))
    got, want = _grads(kind, a)
    # ``det`` of a 32x32 matrix is large: the gradient's own scale sets the bar.
    _assert_same_gradient(got, want, rtol=1e-3 if shape[-1] >= 16 else 2e-4)


@pytest.mark.parametrize(
    "shape", [(3, 3), (4, 4), (8, 8), (16, 16), (5, 3, 3), (3, 4, 4), (2, 3, 6, 6)]
)
def test_an_inv_step_compiles_with_the_eager_gradient(
    shape: tuple[int, ...], device_gpu_only: str
) -> None:
    a = _well_conditioned(*shape, device=device_gpu_only, seed=sum(shape) + 1)
    got, want = _grads("inv", a)
    _assert_same_gradient(got, want, rtol=1e-3)


def _matrix(rows: list[list[float]], device: str) -> lucid.Tensor:
    return lucid.tensor(rows, device=device)


def _dup_row(n: int, device: str) -> lucid.Tensor:
    lucid.manual_seed(n)
    a = lucid.randn(n, n, device=device)
    return lucid.cat([a[: n - 1], a[:1]], dim=0)


def _outer(n: int, device: str) -> lucid.Tensor:
    """Rank 1 with every multiplier a power of two, so its zero pivots are exact."""
    rows = [[float(2**r) * (1.0 + c % 2) for c in range(n)] for r in range(n)]
    return _matrix(rows, device)


def _block_singular(n: int, device: str) -> lucid.Tensor:
    """``[[1, 2], [2, 4]]`` beside ``2 I``: singular in the matrix and in its transpose alike."""
    rows = [[0.0] * n for _ in range(n)]
    rows[0][:2], rows[1][:2] = [1.0, 2.0], [2.0, 4.0]
    for k in range(2, n):
        rows[k][k] = 2.0
    return _matrix(rows, device)


def _shift(n: int, device: str) -> lucid.Tensor:
    """Nilpotent: every pivot is zero and the rank is n - 1, so ``cof`` is one entry."""
    rows = [[1.0 if c == r + 1 else 0.0 for c in range(n)] for r in range(n)]
    return _matrix(rows, device)


@pytest.mark.parametrize(
    "build",
    [
        lambda d: _dup_row(3, d),
        lambda d: _dup_row(4, d),
        lambda d: _dup_row(8, d),
        lambda d: _dup_row(16, d),
        lambda d: _outer(4, d),
        lambda d: _outer(8, d),
        lambda d: _block_singular(6, d),
        lambda d: _shift(3, d),
        lambda d: _shift(5, d),
        lambda d: _matrix([[0.0] * 4] * 4, d),
        lambda d: _matrix([[1.0] * 4] * 4, d),
        lambda d: _matrix([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]], d),
        lambda d: _matrix([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 1.0]], d),
    ],
)
def test_the_det_gradient_is_the_cofactor_at_a_singular_matrix(
    build: Callable[[str], lucid.Tensor], device_gpu_only: str
) -> None:
    """``det(A) A^-T`` is 0 * inf there; the cofactor is finite, and eager's SVD form agrees."""
    a = build(device_gpu_only)
    got, want = _grads("det", a)
    _assert_same_gradient(got, want, rtol=1e-3)


def test_a_batch_mixing_singular_regular_and_zero_matrices(device_gpu_only: str) -> None:
    a = _well_conditioned(5, 4, 4, device=device_gpu_only, seed=11)
    a = lucid.cat(
        [
            a[:1],
            _dup_row(4, device_gpu_only).unsqueeze(0),
            a[2:3],
            lucid.zeros((1, 4, 4), device=device_gpu_only),
            _shift(4, device_gpu_only).unsqueeze(0),
        ],
        dim=0,
    )
    got, want = _grads("det", a)
    _assert_same_gradient(got, want, rtol=1e-3)


def _assert_singular_member_gradient(
    got: list[float], want: list[float], n: int, member: int, *, rtol: float = 2e-4
) -> None:
    """The gradient of a slogdet step through a batch with one ``_block_singular`` member.

    ``cof / det`` there is ``+-inf`` along the cofactor, which is nonzero only in the
    leading 2x2 block: those entries are ``+-inf`` of the same sign in both.  Where the
    cofactor is exactly zero the compiled form gives ``0 * inf = nan``, and eager's SVD
    cofactor, rounding noise instead of that zero, gives ``+-inf``.  Every other member is
    an ordinary finite gradient."""
    per = n * n
    scale = max([abs(v) for v in want[: member * per] + want[(member + 1) * per :]] + [1.0])
    for idx, (g, e) in enumerate(zip(got, want, strict=True)):
        if idx // per != member:
            assert math.isfinite(g) and abs(g - e) <= rtol * scale, (idx, g, e)
        elif (idx % per) // n < 2 and idx % n < 2:
            assert math.isinf(g) and math.isinf(e) and (g > 0) == (e > 0), (idx, g, e)
        else:
            assert math.isnan(g), (idx, g, e)


@pytest.mark.parametrize("n", [3, 4, 8, 16])
def test_a_slogdet_step_through_a_singular_matrix_has_the_eager_infinities(
    n: int, device_gpu_only: str
) -> None:
    """A zero pivot's ``log|det|`` gradient is ``cof / det``: ``+-inf`` along the cofactor."""
    a = _block_singular(n, device_gpu_only)
    got, want = _grads("slogdet", a)
    _assert_singular_member_gradient(got, want, n, 0)


def test_a_slogdet_batch_with_one_singular_member(device_gpu_only: str) -> None:
    a = _well_conditioned(4, 4, 4, device=device_gpu_only, seed=5)
    a = lucid.cat([a[:2], _block_singular(4, device_gpu_only).unsqueeze(0), a[3:]], dim=0)
    got, want = _grads("slogdet", a)
    _assert_singular_member_gradient(got, want, 4, 2)


#: Singular matrices whose determinant is a signed zero; ``A^T`` and ``A`` factor differently.
#: Every multiplier is a power of two, so the zero pivots are exact in both factorisations (a
#: rank-deficient matrix with a pivot of 3 leaves a rounding residue instead, which differs
#: between eager's LAPACK and the graph).
_SIGNED_ZERO_ROWS = [
    [[1.0, 2.0], [2.0, 4.0]],
    [[-2.0, 4.0], [1.0, -2.0]],
    [[-1.0, 2.0], [2.0, -4.0]],
    [[4.0, 2.0], [2.0, 1.0]],
    [[2.0, 4.0], [1.0, 2.0]],
    [[0.0, 1.0], [0.0, 2.0]],
    [[0.0, 0.0], [0.0, 0.0]],
    [[1.0, 1.0, 2.0], [2.0, 2.0, 4.0], [4.0, 4.0, 8.0]],
    [[2.0, 4.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
    [[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 1.0]],
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
    [[1.0, 1.0, 2.0, 2.0], [2.0, 2.0, 4.0, 4.0], [4.0, 4.0, 8.0, 8.0], [8.0, 8.0, 16.0, 16.0]],
]


@pytest.mark.parametrize("rows", _SIGNED_ZERO_ROWS)
def test_a_compiled_det_of_a_singular_matrix_is_the_signed_zero_eager_gives(
    rows: list[list[float]], device_gpu_only: str
) -> None:
    """Eager factors the transpose, so a singular matrix's ``det`` ends on one of the two
    zeros by that factorisation; the compiled ``det`` (a closed form at 2x2 and 3x3 once)
    must end on the same one, since ``slogdet``'s gradient takes its infinity's sign from it."""
    a = lucid.tensor(rows, device=device_gpu_only)
    got, _ = _compiled(_LA.det, a)
    want = _LA.det(a).item()
    assert got.item() == want
    assert math.copysign(1.0, got.item()) == math.copysign(1.0, want)


def test_inv_forward_of_a_batch_matches_eager(device_gpu_only: str) -> None:
    a = _well_conditioned(3, 6, 6, device=device_gpu_only, seed=2)
    cf = lucid.compile(_LA.inv)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        with lucid.no_grad():
            got = cf(a)
    assert not cf.cache_info()["eager_only"]
    want = _LA.inv(a.to("cpu"))
    assert tuple(got.shape) == tuple(want.shape)
    assert lucid.allclose(got.to("cpu"), want, rtol=1e-4, atol=1e-5)


def test_past_the_unroll_bound_a_det_step_runs_eagerly_with_the_eager_gradient(
    device_gpu_only: str,
) -> None:
    n = _MAX_COFACTOR_DIM + 1
    a = _well_conditioned(n, n, device=device_gpu_only, seed=9)
    with pytest.warns(CompileFallbackWarning):
        got, want = _grads("det", a, expect_compiled=False)
    _assert_same_gradient(got, want, rtol=1e-3)
