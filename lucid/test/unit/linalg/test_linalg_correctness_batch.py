"""``lucid.linalg`` correctness against the reference — one section per issue.

Each section names the defect it pins and was written to fail first:

* **CHA-141** — ``eigh`` / ``eigvalsh`` ignored ``UPLO`` (the CPU always read
  the lower triangle, Metal the upper), and the ``*_ex`` family failed the
  whole batch when one matrix failed, with an invented ``info = 1``.
* **CHA-142** — ``ldl_solve`` applied the Bunch-Kaufman interchanges as one
  up-front permutation (wrong as soon as a later step swaps rows an earlier
  column of ``L`` reaches), ``ldl_factor`` left the input's upper triangle
  in ``LD``, and both pivot paths read their pivots through numpy (H4).
* **CHA-146** (Python half) — ``qr``'s backward failed on a wide matrix,
  ``matrix_power(·, 0)`` returned a CPU identity with no ``grad_fn`` and
  ``matrix_power(a, 1)`` returned ``a`` itself, and ``cond`` refused a batch,
  a singular matrix and an empty one.  ``det`` / ``slogdet`` at a singular
  matrix are the engine half and not covered here.
* **CHA-147** — ``vecdot`` did not conjugate ``x``; ``vector_norm`` of a
  complex ``x`` squared the entries (a complex "norm" on Metal, an error on
  the CPU); ``vector_norm(dtype=)`` was accepted and ignored.

Values are compared with the reference framework through the ``ref``
fixture on every device; gradients in float64 on the CPU, where the
reference and a finite difference are both exact enough to disagree with.
"""

import subprocess
import sys
import textwrap
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.linalg as LA


def _np(t: lucid.Tensor) -> np.ndarray:
    return np.asarray(t.to("cpu").numpy())


def _rtol(device: str) -> float:
    return 1e-3 if device == "metal" else 1e-5


# ── CHA-141: eigh / eigvalsh read the triangle UPLO names ────────────────────


def _lopsided(seed: int = 0, batch: tuple[int, ...] = ()) -> np.ndarray:
    """SPD matrices with +10 added to their upper triangle only.

    The two triangles describe different symmetric matrices with different
    spectra, so reading the wrong one cannot pass by accident.
    """
    rng = np.random.default_rng(seed)
    m = rng.standard_normal(batch + (4, 4))
    spd = m @ np.swapaxes(m, -1, -2) + 4 * np.eye(4)
    return (spd + np.triu(np.full((4, 4), 10.0), 1)).astype(np.float32)


_BATCHES = [(), (3,), (2, 2)]


@pytest.mark.parity
@pytest.mark.parametrize("batch", _BATCHES, ids=str)
@pytest.mark.parametrize("uplo", ["L", "U"])
def test_eigh_reads_the_triangle_uplo_names(
    uplo: str, batch: tuple[int, ...], device: str, ref: Any
) -> None:
    a = _lopsided(0, batch)
    w, v = LA.eigh(lucid.tensor(a, device=device), UPLO=uplo)
    rw, rv = ref.linalg.eigh(ref.tensor(a), UPLO=uplo)
    np.testing.assert_allclose(_np(w), rw.numpy(), rtol=_rtol(device), atol=1e-3)
    # Eigenvectors are defined up to sign; their squares are not.
    np.testing.assert_allclose(_np(v) ** 2, rv.numpy() ** 2, atol=1e-3)


@pytest.mark.parametrize("uplo", ["L", "U"])
def test_complex_eigh_is_refused_by_eigh_itself(uplo: str, device: str) -> None:
    """Whichever path ``UPLO`` takes, the refusal is the kernel's own dtype
    check — not a helper op failing first on a dtype it lacks."""
    h = lucid.tensor([[2.0, 1 - 1j], [1 + 1j, 3.0]], device=device)
    for fn in (LA.eigh, LA.eigvalsh):
        with pytest.raises(NotImplementedError, match="eigh: only F32/F64"):
            fn(h, UPLO=uplo)


@pytest.mark.parity
@pytest.mark.parametrize("batch", _BATCHES, ids=str)
@pytest.mark.parametrize("uplo", ["L", "U"])
@pytest.mark.parametrize("requires_grad", [False, True])
def test_eigvalsh_reads_the_triangle_uplo_names(
    uplo: str, requires_grad: bool, batch: tuple[int, ...], device: str, ref: Any
) -> None:
    a = _lopsided(1, batch)
    got = LA.eigvalsh(lucid.tensor(a, device=device, requires_grad=requires_grad), UPLO=uplo)
    want = ref.linalg.eigvalsh(ref.tensor(a), UPLO=uplo)
    np.testing.assert_allclose(_np(got.detach()), want.numpy(), rtol=_rtol(device), atol=1e-3)


@pytest.mark.parity
@pytest.mark.parametrize("uplo", ["L", "U"])
def test_eigh_gradient_matches_the_reference(uplo: str, ref: Any) -> None:
    """The reference's gradient is the symmetric one, whichever triangle
    was read — not a gradient confined to that triangle."""
    a = _lopsided(2).astype(np.float64)
    rng = np.random.default_rng(3)
    ww, wv = rng.standard_normal(4), rng.standard_normal((4, 4))

    x = lucid.tensor(a, requires_grad=True)
    w, v = LA.eigh(x, UPLO=uplo)
    ((w * lucid.tensor(ww)).sum() + (v * v * lucid.tensor(wv)).sum()).backward()

    t = ref.tensor(a, requires_grad=True)
    rw, rv = ref.linalg.eigh(t, UPLO=uplo)
    ((rw * ref.tensor(ww)).sum() + (rv * rv * ref.tensor(wv)).sum()).backward()
    np.testing.assert_allclose(_np(x.grad), t.grad.numpy(), atol=1e-8)


def test_eigh_refuses_an_unknown_uplo() -> None:
    with pytest.raises(ValueError, match="UPLO"):
        LA.eigh(lucid.eye(2), UPLO="X")
    with pytest.raises(ValueError, match="UPLO"):
        LA.eigvalsh(lucid.eye(2), UPLO="lower")


# ── CHA-141: *_ex report each matrix of a batch on its own ───────────────────

_SINGULAR = [[1.0, 2.0], [2.0, 4.0]]  # getrf: U[2, 2] == 0  → info 2
_NOT_PD = [[1.0, 2.0], [2.0, 1.0]]  # potrf: order-2 minor < 0 → info 2


def _batch(*mats: Any) -> np.ndarray:
    return np.stack([np.asarray(m, dtype=np.float32) for m in mats])


@pytest.mark.parity
def test_inv_ex_keeps_the_matrices_that_succeeded(device: str, ref: Any) -> None:
    a = _batch(np.eye(2), _SINGULAR, [[4.0, 7.0], [2.0, 6.0]])
    out, info = LA.inv_ex(lucid.tensor(a, device=device))
    rout, rinfo = ref.linalg.inv_ex(ref.tensor(a))
    assert _np(info).tolist() == rinfo.numpy().tolist() == [0, 2, 0]
    assert info.dtype == lucid.int32
    ok = [0, 2]
    np.testing.assert_allclose(_np(out)[ok], rout.numpy()[ok], rtol=_rtol(device), atol=1e-5)
    np.testing.assert_array_equal(_np(out)[1], np.zeros((2, 2)))


@pytest.mark.parity
@pytest.mark.parametrize("rhs", ["matrix", "vector"])
def test_solve_ex_keeps_the_systems_that_succeeded(rhs: str, device: str, ref: Any) -> None:
    a = _batch(np.eye(2), _SINGULAR)
    b = np.ones((2, 2, 1) if rhs == "matrix" else (2, 2), dtype=np.float32)
    out, info = LA.solve_ex(lucid.tensor(a, device=device), lucid.tensor(b, device=device))
    rout, rinfo = ref.linalg.solve_ex(ref.tensor(a), ref.tensor(b))
    assert _np(info).tolist() == rinfo.numpy().tolist() == [0, 2]
    assert tuple(out.shape) == tuple(rout.shape)
    np.testing.assert_allclose(_np(out)[0], rout.numpy()[0], rtol=_rtol(device))
    np.testing.assert_array_equal(_np(out)[1], np.zeros_like(rout.numpy()[1]))


@pytest.mark.parity
def test_cholesky_ex_keeps_the_factors_that_succeeded(device: str, ref: Any) -> None:
    a = _batch([[4.0, 2.0], [2.0, 3.0]], _NOT_PD, [[9.0, 3.0], [3.0, 5.0]])
    out, info = LA.cholesky_ex(lucid.tensor(a, device=device))
    rout, rinfo = ref.linalg.cholesky_ex(ref.tensor(a))
    assert _np(info).tolist() == rinfo.numpy().tolist() == [0, 2, 0]
    ok = [0, 2]
    np.testing.assert_allclose(_np(out)[ok], rout.numpy()[ok], rtol=_rtol(device), atol=1e-6)
    # The failed one holds only the leading factor LAPACK defines (LCD-395).
    np.testing.assert_allclose(_np(out)[1], [[1.0, 0.0], [0.0, 0.0]])


@pytest.mark.parity
@pytest.mark.parametrize(
    "op,matrix",
    [
        ("inv_ex", _SINGULAR),
        ("inv_ex", [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        ("inv_ex", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        ("cholesky_ex", _NOT_PD),
        ("cholesky_ex", np.diag([1.0, 1.0, -1.0])),
        ("cholesky_ex", np.diag([1.0, 0.0, 1.0])),
        ("cholesky_ex", [[float("nan"), 0.0], [0.0, 4.0]]),
    ],
)
def test_ex_info_is_the_lapack_code(op: str, matrix: Any, device: str, ref: Any) -> None:
    """Not a bare non-zero flag: the pivot / leading-minor order."""
    a = np.asarray(matrix, dtype=np.float32)
    _, info = getattr(LA, op)(lucid.tensor(a, device=device))
    _, rinfo = getattr(ref.linalg, op)(ref.tensor(a))
    assert info.shape == ()
    assert int(info.item()) == int(rinfo.item()) != 0


@pytest.mark.parity
@pytest.mark.parametrize("size", [2, 5, 16, 33])
def test_cholesky_ex_finds_scattered_failures_in_a_batch(size: int, device: str, ref: Any) -> None:
    """The failures are located by bisecting the batch; each one's order by
    bisecting its leading blocks.  Both must land where LAPACK's do."""
    rng = np.random.default_rng(size)
    m = rng.standard_normal((size, 5, 5)).astype(np.float32)
    a = m @ np.swapaxes(m, -1, -2) + 0.5 * np.eye(5, dtype=np.float32)
    broken = rng.random(size) < 0.3
    broken[rng.integers(size)] = True
    for i in np.flatnonzero(broken):
        j = int(rng.integers(5))
        a[i, j, j] = -5.0
    _, info = LA.cholesky_ex(lucid.tensor(a, device=device))
    _, rinfo = ref.linalg.cholesky_ex(ref.tensor(a))
    assert _np(info).tolist() == rinfo.numpy().tolist()


@pytest.mark.parametrize(
    "op,matrix,message",
    [
        ("inv_ex", _SINGULAR, "diagonal element 2 is zero"),
        ("cholesky_ex", _NOT_PD, "leading minor of order 2"),
    ],
)
def test_check_errors_names_the_failed_batch_element(
    op: str, matrix: Any, message: str, device: str
) -> None:
    a = _batch(np.eye(2), matrix)
    with pytest.raises(RuntimeError, match=rf"Batch element 1\): .*{message}"):
        getattr(LA, op)(lucid.tensor(a, device=device), check_errors=True)


def test_solve_ex_check_errors_raises(device: str) -> None:
    a = _batch(np.eye(2), _SINGULAR)
    with pytest.raises(RuntimeError, match="Batch element 1"):
        LA.solve_ex(
            lucid.tensor(a, device=device), lucid.ones(2, 2, 1, device=device), check_errors=True
        )


@pytest.mark.parametrize("op", ["inv_ex", "cholesky_ex"])
def test_ex_gradient_reaches_only_the_matrices_that_succeeded(op: str, device: str) -> None:
    good = np.array([[4.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    bad = np.asarray(_SINGULAR if op == "inv_ex" else _NOT_PD, dtype=np.float32)
    x = lucid.tensor(_batch(good, bad), device=device, requires_grad=True)
    out, _ = getattr(LA, op)(x)
    out.sum().backward()

    alone = lucid.tensor(good, device=device, requires_grad=True)
    (LA.inv(alone) if op == "inv_ex" else LA.cholesky(alone)).sum().backward()
    np.testing.assert_allclose(_np(x.grad)[0], _np(alone.grad), rtol=_rtol(device), atol=1e-6)
    np.testing.assert_array_equal(_np(x.grad)[1], np.zeros((2, 2)))


@pytest.mark.parametrize("op", ["inv_ex", "cholesky_ex", "solve_ex"])
@pytest.mark.parametrize("shape", [(2, 3), (3,)])
def test_ex_refuses_a_non_square_input(op: str, shape: tuple[int, ...]) -> None:
    """A shape error is not a singular matrix: it was reported as
    ``info = 1`` with a zero result, because the kernel's shape refusal is
    the same exception type as a failed factorisation."""
    args = (lucid.ones(*shape),) + ((lucid.ones(2, 1),) if op == "solve_ex" else ())
    with pytest.raises(ValueError, match="square"):
        getattr(LA, op)(*args)


# ── CHA-142: ldl_solve replays Bunch-Kaufman interchanges in order ───────────


def _symmetric(seed: int, n: int, dtype: Any = np.float64) -> np.ndarray:
    a = np.random.default_rng(seed).standard_normal((n, n))
    return (a + a.T).astype(dtype)


def _dtype(device: str) -> Any:
    return np.float32 if device == "metal" else np.float64


@pytest.mark.parity
def test_ldl_factor_zeroes_the_upper_triangle(device: str, ref: Any) -> None:
    """``sytrf`` leaves the input's upper triangle in place; the reference
    clears it, and so does Lucid now."""
    s = _symmetric(0, 4, _dtype(device))
    ld, piv = LA.ldl_factor(lucid.tensor(s, device=device))
    rld, rpiv = ref.linalg.ldl_factor(ref.tensor(s))
    assert _np(piv).tolist() == rpiv.numpy().tolist()
    np.testing.assert_array_equal(np.triu(_np(ld), 1), np.zeros((4, 4)))
    np.testing.assert_allclose(_np(ld), rld.numpy(), rtol=_rtol(device), atol=1e-6)


@pytest.mark.parity
def test_ldl_solve_with_row_interchanges(device: str, ref: Any) -> None:
    """The issue's case: pivots ``[4, 4, 3, 4]`` — two 1×1 steps that both
    swap with row 4, which a single up-front permutation of ``B`` cannot
    express."""
    a = np.random.default_rng(0).standard_normal((4, 4)).astype(_dtype(device))
    s = a + a.T
    b = np.random.default_rng(1).standard_normal((4, 2)).astype(_dtype(device))
    ld, piv = LA.ldl_factor(lucid.tensor(s, device=device))
    assert _np(piv).tolist() == [4, 4, 3, 4]
    x = LA.ldl_solve(ld, piv, lucid.tensor(b, device=device))
    rld, rpiv = ref.linalg.ldl_factor(ref.tensor(s))
    want = ref.linalg.ldl_solve(rld, rpiv, ref.tensor(b)).numpy()
    np.testing.assert_allclose(_np(x), want, rtol=_rtol(device), atol=1e-4)
    np.testing.assert_allclose(
        _np(x), np.linalg.solve(s.astype(np.float64), b), rtol=1e-3, atol=1e-4
    )


def test_ldl_solve_over_every_pivot_pattern(device: str) -> None:
    """Random symmetric indefinite systems: 1×1 interchanges, 2×2 blocks
    and none, checked against a dense solve — and the sweep asserts it
    actually met each pattern."""
    dtype = _dtype(device)
    seen = {"interchange": 0, "2x2": 0, "none": 0}
    worst = 0.0
    for seed in range(60):
        n = 3 + seed % 5
        s = _symmetric(seed, n, dtype)
        b = np.random.default_rng(seed + 1000).standard_normal((n, 3)).astype(dtype)
        ld, piv = LA.ldl_factor(lucid.tensor(s, device=device))
        p = _np(piv).tolist()
        seen["2x2"] += any(v < 0 for v in p)
        seen["interchange"] += any(v > 0 and v != i + 1 for i, v in enumerate(p))
        seen["none"] += all(v == i + 1 for i, v in enumerate(p))
        x = _np(LA.ldl_solve(ld, piv, lucid.tensor(b, device=device)))
        want = np.linalg.solve(s.astype(np.float64), b.astype(np.float64))
        worst = max(worst, float(np.max(np.abs(x - want)) / np.max(np.abs(want))))
    assert all(seen.values()), seen
    assert worst < (1e-3 if device == "metal" else 1e-10), worst


def test_ldl_solve_batched_broadcast_and_vector() -> None:
    s = np.stack([_symmetric(seed, 5) for seed in (3, 4, 5)])
    b = np.random.default_rng(9).standard_normal((3, 5, 2))
    ld, piv = LA.ldl_factor(lucid.tensor(s))
    x = LA.ldl_solve(ld, piv, lucid.tensor(b))
    np.testing.assert_allclose(_np(x), np.linalg.solve(s, b), atol=1e-10)
    # One right-hand side shared by the batch.
    x = LA.ldl_solve(ld, piv, lucid.tensor(b[0]))
    np.testing.assert_allclose(
        _np(x), np.linalg.solve(s, np.broadcast_to(b[0], (3, 5, 2))), atol=1e-10
    )
    # A single vector.
    x = LA.ldl_solve(ld[1], piv[1], lucid.tensor(b[1, :, 0]))
    assert tuple(x.shape) == (5,)
    np.testing.assert_allclose(_np(x), np.linalg.solve(s[1], b[1, :, 0]), atol=1e-10)


def test_ldl_solve_is_differentiable_in_b() -> None:
    s = _symmetric(0, 4)
    ld, piv = LA.ldl_factor(lucid.tensor(s))
    b = lucid.tensor(np.ones((4, 1)), requires_grad=True)
    LA.ldl_solve(ld, piv, b).sum().backward()
    # d(1ᵀ A⁻¹ b)/db = A⁻ᵀ 1 = A⁻¹ 1 for symmetric A.
    np.testing.assert_allclose(_np(b.grad), np.linalg.solve(s, np.ones((4, 1))), atol=1e-10)


def test_ldl_solve_differentiates_through_a_2x2_block() -> None:
    """Pivots ``[-4, -4, 4, 4, 5]``: a 2×2 block of ``D``, solved in closed
    form, and interchanges — checked in both ``LD`` and ``B``."""
    rng = np.random.default_rng(4)
    ld, piv = LA.ldl_factor(lucid.tensor(_symmetric(4, 5)))
    assert _np(piv).tolist() == [-4, -4, 4, 4, 5]
    ld = lucid.tensor(_np(ld), requires_grad=True)
    b = lucid.tensor(rng.standard_normal((5, 2)), requires_grad=True)
    w = lucid.tensor(rng.standard_normal((5, 2)))
    assert lucid.autograd.gradcheck(lambda L, B: (LA.ldl_solve(L, piv, B) * w).sum(), (ld, b))


@pytest.mark.parametrize(
    "pivots",
    [
        [1, 2, 0],  # zero is not a LAPACK pivot
        [1, 2, 7],  # out of range
        [-2, 2, 3],  # an unpaired negative pivot
        [1, 2, -3],  # a 2x2 block running past the end
    ],
)
def test_ldl_solve_refuses_malformed_pivots(pivots: list[int]) -> None:
    ld, _ = LA.ldl_factor(lucid.tensor(_symmetric(0, 3)))
    with pytest.raises(ValueError, match="pivot"):
        LA.ldl_solve(ld, lucid.tensor(pivots, dtype=lucid.int32), lucid.ones(3, 1))


def test_ldl_solve_refuses_float_pivots() -> None:
    ld, piv = LA.ldl_factor(lucid.tensor(_symmetric(0, 3)))
    with pytest.raises(TypeError, match="integer"):
        LA.ldl_solve(ld, piv.to(lucid.float32), lucid.ones(3, 1))


def test_lu_and_lu_factor_gradient_take_a_batch(device: str) -> None:
    """The pivot-to-permutation conversion handled one matrix only, so
    ``lu`` and the gradient of ``lu_factor`` refused a batch."""
    a = np.random.default_rng(4).standard_normal((3, 4, 4)).astype(np.float32)
    P, L, U = LA.lu(lucid.tensor(a, device=device))
    np.testing.assert_allclose(_np(P @ L @ U), a, atol=1e-5)

    x = lucid.tensor(a, device=device, requires_grad=True)
    LA.lu_factor(x)[0].sum().backward()
    for i in range(3):
        one = lucid.tensor(a[i], device=device, requires_grad=True)
        LA.lu_factor(one)[0].sum().backward()
        np.testing.assert_allclose(_np(x.grad)[i], _np(one.grad), rtol=_rtol(device), atol=1e-5)


def test_pivot_paths_run_without_numpy() -> None:
    """``ldl_solve`` and the LU permutation read their pivots through
    ``Tensor.numpy`` — numpy on the compute path, outside every H4 bridge.
    ``check_numpy_h4`` only sees ``import`` statements, and the import
    happened inside the sanctioned ``Tensor.numpy``, so it never noticed.
    """
    script = textwrap.dedent("""
        import sys

        for mod in list(sys.modules):
            if mod == "numpy" or mod.startswith("numpy."):
                del sys.modules[mod]

        class _Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == "numpy" or name.startswith("numpy."):
                    raise ImportError("numpy blocked for test")
                return None

        sys.meta_path.insert(0, _Blocker())

        import lucid
        import lucid.linalg as LA

        S = lucid.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 1.0]])
        LD, piv = LA.ldl_factor(S)
        x = LA.ldl_solve(LD, piv, lucid.ones(3, 1))
        assert float((S @ x - 1.0).abs().max().item()) < 1e-4

        A = lucid.tensor([[[0.0, 1.0], [2.0, 3.0]], [[4.0, 1.0], [1.0, 3.0]]])
        P, L, U = LA.lu(A)
        assert float((P @ L @ U - A).abs().max().item()) < 1e-5
        A.requires_grad_(True)
        LA.lu_factor(A)[0].sum().backward()
        assert A.grad is not None
        assert "numpy" not in sys.modules
        print("ok")
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


# ── CHA-146: QR of a wide matrix differentiates ──────────────────────────────


def _qr_loss(q: Any, r: Any, wq: Any, wr: Any, path: str) -> Any:
    loss = 0.0
    if path in ("Q", "both"):
        loss = loss + (q * wq).sum()
    if path in ("R", "both"):
        loss = loss + (r * r * wr).sum()
    return loss


@pytest.mark.parity
@pytest.mark.parametrize("shape", [(3, 5), (2, 2, 4), (1, 3)])
@pytest.mark.parametrize("path", ["Q", "R", "both"])
def test_wide_qr_gradient_matches_the_reference(
    shape: tuple[int, ...], path: str, device: str, ref: Any
) -> None:
    rng = np.random.default_rng(sum(shape))
    a = rng.standard_normal(shape).astype(_dtype(device))
    k = shape[-2]
    wq = rng.standard_normal(shape[:-1] + (k,)).astype(a.dtype)
    wr = rng.standard_normal(shape[:-2] + (k, shape[-1])).astype(a.dtype)

    x = lucid.tensor(a, device=device, requires_grad=True)
    q, r = LA.qr(x)
    weights = (lucid.tensor(wq, device=device), lucid.tensor(wr, device=device))
    _qr_loss(q, r, *weights, path).backward()

    t = ref.tensor(a, requires_grad=True)
    rq, rr = ref.linalg.qr(t)
    _qr_loss(rq, rr, ref.tensor(wq), ref.tensor(wr), path).backward()
    # Same LAPACK factorization, same signs, so Q and R agree as they stand.
    np.testing.assert_allclose(_np(r.detach()), rr.detach().numpy(), atol=1e-4)
    np.testing.assert_allclose(_np(x.grad), t.grad.numpy(), rtol=_rtol(device), atol=1e-4)


@pytest.mark.parametrize("output", [0, 1])
def test_wide_qr_differentiates_twice(output: int) -> None:
    rng = np.random.default_rng(5)
    x = lucid.tensor(rng.standard_normal((3, 5)), requires_grad=True)
    w = lucid.tensor(rng.standard_normal((3, 3) if output == 0 else (3, 5)))
    assert lucid.autograd.gradcheck(lambda t: (LA.qr(t)[output] * w).sum(), (x,))
    assert lucid.autograd.gradgradcheck(lambda t: (LA.qr(t)[output] ** 2 * w).sum(), (x,))


# ── CHA-146: matrix_power at 0 and 1 ─────────────────────────────────────────


def test_matrix_power_zero_is_a_fresh_identity_on_the_input_device(device: str) -> None:
    x = lucid.rand(2, 3, 3, device=device, requires_grad=True)
    y = LA.matrix_power(x, 0)
    assert y.device == x.device
    np.testing.assert_array_equal(_np(y.detach()), np.broadcast_to(np.eye(3), (2, 3, 3)))
    assert y.grad_fn is not None
    (y * 3).sum().backward()
    np.testing.assert_array_equal(_np(x.grad), np.zeros((2, 3, 3)))
    # Downstream ops on the input's device no longer meet a CPU tensor.
    _ = y + x


def test_matrix_power_one_is_a_copy(device: str) -> None:
    a = lucid.rand(2, 2, device=device)
    before = _np(a).copy()
    b = LA.matrix_power(a, 1)
    b.add_(1.0)
    np.testing.assert_array_equal(_np(a), before)
    np.testing.assert_array_equal(_np(b), before + 1.0)


def test_matrix_power_one_keeps_the_gradient() -> None:
    x = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    (LA.matrix_power(x, 1) * 2).sum().backward()
    np.testing.assert_array_equal(_np(x.grad), np.full((2, 2), 2.0))


# ── CHA-146: cond on a batch, a singular matrix and an empty one ─────────────

_ORDERS = [None, 2, -2, 1, -1, float("inf"), float("-inf"), "fro", "nuc"]


@pytest.mark.parity
@pytest.mark.parametrize("p", _ORDERS, ids=str)
def test_cond_takes_a_batch(p: Any, device: str, ref: Any) -> None:
    a = np.random.default_rng(6).standard_normal((2, 3, 4, 4)).astype(np.float32)
    got = LA.cond(lucid.tensor(a, device=device), p)
    want = ref.linalg.cond(ref.tensor(a), p).numpy()
    assert tuple(got.shape) == want.shape == (2, 3)
    np.testing.assert_allclose(_np(got), want, rtol=1e-3 if device == "metal" else 1e-4)


@pytest.mark.parity
@pytest.mark.parametrize("p", _ORDERS, ids=str)
def test_cond_of_a_singular_matrix(p: Any, device: str, ref: Any) -> None:
    """Infinite under the inverse-based orders, rather than an error; the
    singular-value orders are a ratio of what the SVD found."""
    a = np.stack([np.asarray(_SINGULAR, np.float32), np.eye(2, dtype=np.float32)])
    got = _np(LA.cond(lucid.tensor(a, device=device), p))
    want = ref.linalg.cond(ref.tensor(a), p).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-3)
    if p in (1, -1, float("inf"), float("-inf"), "fro"):
        assert got[0] == np.inf and np.isfinite(got[1])


@pytest.mark.parity
@pytest.mark.parametrize("p", [None, 1, "fro"], ids=str)
def test_cond_of_a_zero_matrix(p: Any, device: str, ref: Any) -> None:
    got = _np(LA.cond(lucid.zeros(2, 2, device=device), p))
    want = ref.linalg.cond(ref.zeros(2, 2), p).numpy()
    np.testing.assert_array_equal(got, want)  # nan for the SVD ratio, else inf


@pytest.mark.parity
@pytest.mark.parametrize("shape", [(0, 0), (2, 0, 0), (3, 0)])
def test_cond_of_an_empty_matrix_is_zero(shape: tuple[int, ...], device: str, ref: Any) -> None:
    got = LA.cond(lucid.zeros(*shape, device=device))
    want = ref.linalg.cond(ref.zeros(*shape)).numpy()
    assert tuple(got.shape) == want.shape
    np.testing.assert_array_equal(_np(got), want)


@pytest.mark.parametrize(
    "args,match",
    [
        ((lucid.ones(2, 3), "fro"), "square"),
        ((lucid.ones(2, 3), 1), "square"),
        ((lucid.eye(2), 3), "unsupported order"),
        ((lucid.ones(3), None), "at least 2"),
    ],
)
def test_cond_refuses_what_it_cannot_compute(args: tuple[Any, ...], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        LA.cond(*args)


def test_cond_rectangular_spectral() -> None:
    a = np.random.default_rng(8).standard_normal((3, 5))
    np.testing.assert_allclose(_np(LA.cond(lucid.tensor(a))), np.linalg.cond(a))


# ── CHA-147: complex vecdot / vector_norm, and vector_norm(dtype=) ───────────


def _complex(seed: int, shape: tuple[int, ...]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)


def _np_complex(t: lucid.Tensor) -> np.ndarray:
    t = t.to("cpu")
    if not t.is_complex():
        return _np(t)
    return _np(lucid.real(t)) + 1j * _np(lucid.imag(t))


@pytest.mark.parity
def test_vecdot_conjugates_its_first_operand(device: str, ref: Any) -> None:
    x, y = _complex(0, (3, 4)), _complex(1, (3, 4))
    got = LA.vecdot(lucid.tensor(x, device=device), lucid.tensor(y, device=device))
    want = ref.linalg.vecdot(ref.tensor(x), ref.tensor(y)).numpy()
    np.testing.assert_allclose(_np_complex(got), want, rtol=1e-5, atol=1e-5)
    # The issue's numbers.
    a = lucid.tensor(np.array([1 + 2j, 3 - 1j], np.complex64), device=device)
    b = lucid.tensor(np.array([2 - 1j, 1 + 1j], np.complex64), device=device)
    assert complex(_np_complex(LA.vecdot(a, b))) == 2 - 1j


def test_vecdot_of_a_complex_vector_with_itself_is_real(device: str) -> None:
    x = _complex(2, (5,))
    got = _np_complex(LA.vecdot(lucid.tensor(x, device=device), lucid.tensor(x, device=device)))
    np.testing.assert_allclose(got, np.sum(np.abs(x) ** 2), rtol=1e-5)


@pytest.mark.parity
@pytest.mark.parametrize("ord", [2, 1, 3, 0.5, float("inf"), float("-inf"), 0])
def test_vector_norm_of_a_complex_input_is_real(ord: float, device: str, ref: Any) -> None:
    x = _complex(3, (3, 5))
    got = LA.vector_norm(lucid.tensor(x, device=device), ord=ord, dim=-1)
    want = ref.linalg.vector_norm(ref.tensor(x), ord=ord, dim=-1).numpy()
    assert not got.is_complex()
    assert str(got.dtype).split(".")[-1] == str(want.dtype).split(".")[-1]
    np.testing.assert_allclose(_np(got), want, rtol=1e-5)


@pytest.mark.parity
def test_complex_norm_gradient_matches_the_reference(ref: Any) -> None:
    x = _complex(4, (4,))
    t = lucid.tensor(x, requires_grad=True)
    LA.vector_norm(t).backward()
    r = ref.tensor(x, requires_grad=True)
    ref.linalg.vector_norm(r).backward()
    np.testing.assert_allclose(_np_complex(t.grad), r.grad.numpy(), rtol=1e-5)


@pytest.mark.parity
def test_vector_norm_computes_in_the_requested_dtype(ref: Any) -> None:
    """``1e20`` squared overflows float32; in float64 it does not."""
    x = np.array([[1e20, 1e20], [3.0, 4.0]], dtype=np.float32)
    got = LA.vector_norm(lucid.tensor(x), dim=-1, dtype=lucid.float64)
    want = ref.linalg.vector_norm(ref.tensor(x), dim=-1, dtype=ref.float64).numpy()
    assert got.dtype == lucid.float64
    np.testing.assert_allclose(_np(got), want, rtol=1e-12)
    assert np.isfinite(_np(got)).all()


def test_vector_norm_widens_half_on_every_device(device: str) -> None:
    """float64 is CPU-only; half → float32 is the widening Metal has."""
    x = lucid.tensor([3.0, 4.0], dtype=lucid.float16, device=device)
    got = LA.vector_norm(x, dtype=lucid.float32)
    assert got.dtype == lucid.float32
    assert float(got.item()) == 5.0


def test_vector_norm_complex_dtype_gives_its_real_counterpart() -> None:
    x = lucid.tensor(np.array([3 + 4j, 0j], np.complex64))
    got = LA.vector_norm(x, dtype=lucid.complex128)
    assert got.dtype == lucid.float64
    assert float(got.item()) == 5.0


@pytest.mark.parametrize(
    "values,dtype,match",
    [
        ([1.0, 2.0], lucid.float16, "narrow"),
        ([1.0, 2.0], lucid.complex64, "real"),
        ([1 + 1j], lucid.float64, "complex"),
    ],
)
def test_vector_norm_refuses_a_dtype_the_reference_refuses(
    values: list[Any], dtype: Any, match: str
) -> None:
    with pytest.raises(TypeError, match=match):
        LA.vector_norm(lucid.tensor(values), dtype=dtype)


@pytest.mark.parametrize("driver", ["gelsd", "gelss", "gels"])
def test_lstsq_residuals_with_no_columns_match_the_reference(ref: Any, driver: str) -> None:
    # With n == 0 the solution is empty: the SVD drivers report all of ``b`` as
    # the residual, while the reference's ``gels`` reports zeros.
    b = [[[1.0, 0.5], [2.0, -1.0], [3.0, 2.0]], [[0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]]
    got = lucid.linalg.lstsq(lucid.zeros(2, 3, 0), lucid.tensor(b), driver=driver).residuals
    want = ref.linalg.lstsq(ref.zeros(2, 3, 0), ref.tensor(b), driver=driver).residuals
    assert got.tolist() == want.tolist()
