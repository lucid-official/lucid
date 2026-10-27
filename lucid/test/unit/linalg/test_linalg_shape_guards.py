"""The CPU linalg backend must refuse the shapes it cannot compute.

The backend reads ``m`` and ``n`` off the last two dimensions of ``H``
and loops over neither a batch nor a missing axis, so two malformed
inputs used to be *answered* rather than refused — which is worse than
an error, because the answer has the right dtype and a plausible shape:

    H (3, 4, 3), tau (3, 3)  ->  Q (4, 3)   the other two matrices gone
    H (4,),      tau (4,)    ->  Q (0, 0)   an empty tensor, no complaint

``lstsq`` has the same shape of hole one function over, and there it
is not only a wrong answer: it copies ``m`` rows out of ``B`` without
asking how many ``B`` has, which segfaulted mid-``memmove`` inside the
audit's grad2 axis and returned a plausible answer on the runs where the
over-read happened to land on mapped memory.

``qr`` is unaffected — it loops the batch itself — so these are about the
direct entry points, the ones reached for when the caller already holds
reflectors or an over-determined system.
"""

import numpy as np
import pytest

import lucid
import lucid.linalg as LA


def _matrix(m: int, n: int) -> lucid.Tensor:
    lucid.manual_seed(0)
    return lucid.randn(m, n, dtype=lucid.float64)


def test_a_batched_H_solves_every_matrix_in_the_batch() -> None:
    lucid.manual_seed(0)
    H_np = np.random.default_rng(0).standard_normal((3, 4, 3)) * 0.2
    tau_np = np.random.default_rng(1).standard_normal((3, 3)) * 0.2
    q = LA.householder_product(lucid.tensor(H_np), lucid.tensor(tau_np))
    assert tuple(q.shape) == (3, 4, 3)
    for i in range(3):
        qi = LA.householder_product(lucid.tensor(H_np[i]), lucid.tensor(tau_np[i]))
        np.testing.assert_allclose(q[i].numpy(), qi.numpy(), rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="batch dimensions"):
        LA.householder_product(
            lucid.randn(3, 4, 3, dtype=lucid.float64), lucid.randn(2, 3, dtype=lucid.float64)
        )


def test_a_one_dimensional_H_is_refused_rather_than_returning_empty() -> None:
    H = lucid.randn(4, dtype=lucid.float64)
    tau = lucid.randn(4, dtype=lucid.float64)
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        LA.householder_product(H, tau)


def test_a_non_vector_tau_is_refused() -> None:
    with pytest.raises(ValueError, match="one dimension less than H"):
        LA.householder_product(_matrix(4, 3), lucid.randn(2, 3, dtype=lucid.float64))


def test_too_few_reflectors_are_refused() -> None:
    """``tau`` shorter than ``min(m, n)`` is an over-read waiting to happen."""
    with pytest.raises(ValueError, match="at least min"):
        LA.householder_product(_matrix(4, 3), lucid.randn(2, dtype=lucid.float64))


@pytest.mark.parametrize(("m", "n"), [(4, 3), (3, 3), (5, 2), (2, 5)])
def test_a_well_formed_call_still_works(m: int, n: int) -> None:
    """The guard must not cost the supported case.

    Lucid exposes no raw-reflector factorisation, so the reflectors here
    are synthetic — which is fine, because what is being pinned is that
    a well-formed *shape* reaches the backend and comes back with the
    documented ``(m, min(m, n))`` and finite values.  Whether those
    particular numbers spell an orthogonal matrix is ``qr``'s question,
    and the last test asks it.
    """
    lucid.manual_seed(0)
    k = min(m, n)
    q = LA.householder_product(
        lucid.randn(m, n, dtype=lucid.float64), lucid.randn(k, dtype=lucid.float64) * 0.1
    )
    assert tuple(q.shape) == (m, k)
    assert np.isfinite(q.numpy()).all()


def test_a_longer_tau_is_accepted() -> None:
    """Only *too few* reflectors are an error; extras are ignored by LAPACK."""
    lucid.manual_seed(0)
    q = LA.householder_product(
        lucid.randn(4, 3, dtype=lucid.float64), lucid.randn(3, dtype=lucid.float64) * 0.1
    )
    assert tuple(q.shape) == (4, 3)


# ── lstsq, the same pattern one function over ────────────────────────────────


def test_lstsq_refuses_a_one_dimensional_A() -> None:
    """``shape[-2]`` on a rank-1 shape underflowed into a 1.8e19-byte ask."""
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        LA.lstsq(lucid.randn(4, dtype=lucid.float64), lucid.randn(4, dtype=lucid.float64))


def test_lstsq_solves_a_batched_A() -> None:
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((3, 4, 3))
    b_np = rng.standard_normal((3, 4, 2))
    sol, *_ = LA.lstsq(lucid.tensor(a_np), lucid.tensor(b_np))
    assert tuple(sol.shape) == (3, 3, 2)
    for i in range(3):
        want, *_ = np.linalg.lstsq(a_np[i], b_np[i], rcond=None)
        np.testing.assert_allclose(sol[i].numpy(), want, rtol=1e-10, atol=1e-10)
    sol_vec, *_ = LA.lstsq(lucid.tensor(a_np), lucid.tensor(b_np[:, :, 0]))
    np.testing.assert_allclose(sol_vec.numpy(), sol[:, :, 0].numpy(), rtol=1e-12, atol=1e-12)


def test_lstsq_refuses_a_B_with_too_few_rows() -> None:
    """The over-read itself: ``m`` rows copied out of a shorter buffer.

    This is the one that segfaulted mid-``memmove`` — and it had been
    *returning an answer* on the runs where the read happened to land on
    mapped memory, which is the worse half of the bug.
    """
    with pytest.raises(ValueError, match="one row per row of A"):
        LA.lstsq(lucid.randn(6, 3, dtype=lucid.float64), lucid.randn(2, 2, dtype=lucid.float64))


@pytest.mark.parametrize("b_shape", [(3,), (3, 1), (3, 4)])
def test_lstsq_still_solves_what_it_should(b_shape: tuple[int, ...]) -> None:
    lucid.manual_seed(0)
    a = lucid.randn(3, 2, dtype=lucid.float64)
    sol, *_ = LA.lstsq(a, lucid.randn(*b_shape, dtype=lucid.float64))
    assert tuple(sol.shape)[0] == 2
    assert np.isfinite(sol.numpy()).all()


def test_lstsq_matches_its_documented_example() -> None:
    a = lucid.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    b = lucid.tensor([[6.0], [9.0], [12.0]])
    sol, *_ = LA.lstsq(a, b)
    np.testing.assert_allclose(sol.numpy().ravel(), [3.0, 3.0], rtol=1e-6, atol=1e-6)


def test_qr_still_handles_a_batch() -> None:
    """The batch path users actually have, and the reason the guard points there."""
    lucid.manual_seed(0)
    a = lucid.randn(3, 4, 3, dtype=lucid.float64)
    q, r = LA.qr(a)
    assert tuple(q.shape) == (3, 4, 3)
    recon = (q @ r).numpy()
    np.testing.assert_allclose(recon, a.numpy(), rtol=1e-10, atol=1e-10)


def test_tensorinv_inverts_tensordot_and_differentiates() -> None:
    rng = np.random.default_rng(42)
    m = 4 * 6
    a_np = (rng.standard_normal((m, m)) * 0.2 + np.eye(m)).reshape(4, 6, 8, 3)
    a = lucid.tensor(a_np, requires_grad=True)
    inv_a = LA.tensorinv(a, ind=2)
    assert tuple(inv_a.shape) == (8, 3, 4, 6)
    np.testing.assert_allclose(
        inv_a.detach().numpy(), np.linalg.tensorinv(a_np, ind=2), rtol=1e-10, atol=1e-10
    )
    prod = lucid.tensordot(a, inv_a, dims=2).reshape(m, m)
    np.testing.assert_allclose(prod.detach().numpy(), np.eye(m), rtol=1e-10, atol=1e-10)
    inv_a.sum().backward()
    assert a.grad is not None and np.isfinite(a.grad.numpy()).all()
    for bad_ind in (0, -1, 5):
        with pytest.raises(RuntimeError):
            LA.tensorinv(a, ind=bad_ind)
    with pytest.raises(RuntimeError):
        LA.tensorinv(lucid.ones(2, 3, 5, dtype=lucid.float64), ind=1)


def test_tensorsolve_solves_tensordot_with_and_without_dims() -> None:
    rng = np.random.default_rng(7)
    m = 2 * 3
    a_np = (rng.standard_normal((m, m)) * 0.2 + np.eye(m)).reshape(2, 3, 3, 2)
    b_np = rng.standard_normal((2, 3))
    a = lucid.tensor(a_np, requires_grad=True)
    b = lucid.tensor(b_np, requires_grad=True)
    x = LA.tensorsolve(a, b)
    assert tuple(x.shape) == (3, 2)
    np.testing.assert_allclose(
        x.detach().numpy(), np.linalg.tensorsolve(a_np, b_np), rtol=1e-10, atol=1e-10
    )
    recon = lucid.tensordot(a, x, dims=2)
    np.testing.assert_allclose(recon.detach().numpy(), b_np, rtol=1e-10, atol=1e-10)
    x.sum().backward()
    assert a.grad is not None and np.isfinite(a.grad.numpy()).all()
    assert b.grad is not None and np.isfinite(b.grad.numpy()).all()

    a_perm = lucid.movedim(a.detach(), (-2, -1), (0, 2))
    x_dims = LA.tensorsolve(a_perm, b.detach(), dims=(0, 2))
    np.testing.assert_allclose(x_dims.numpy(), x.detach().numpy(), rtol=1e-10, atol=1e-10)
    with pytest.raises(RuntimeError, match="prod"):
        LA.tensorsolve(lucid.ones(2, 3, 2, 2, dtype=lucid.float64), b.detach())
    with pytest.raises(RuntimeError, match="other.shape"):
        LA.tensorsolve(a.detach(), lucid.ones(3, 2, dtype=lucid.float64))


def test_vander_supports_batched_inputs() -> None:
    x_np = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    x = lucid.tensor(x_np)
    v_def = LA.vander(x)
    v_inc = LA.vander(x, N=4, increasing=True)
    assert tuple(v_def.shape) == (2, 3, 3)
    assert tuple(v_inc.shape) == (2, 3, 4)
    for i in range(2):
        np.testing.assert_allclose(
            v_def[i].numpy(), np.vander(x_np[i], increasing=True), rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(
            v_inc[i].numpy(), np.vander(x_np[i], N=4, increasing=True), rtol=1e-12, atol=1e-12
        )
    assert tuple(LA.vander(x, N=0).shape) == (2, 3, 0)
    # A 0-d input is a vector of one, as the reference reads it.
    assert LA.vander(lucid.tensor(2.0), N=3).tolist() == [[1.0, 2.0, 4.0]]
    with pytest.raises(RuntimeError):
        LA.vander(x, N=-1)


@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("adjoint", [False, True])
def test_lu_solve_supports_left_and_adjoint_with_gradients(left: bool, adjoint: bool) -> None:
    rng = np.random.default_rng(11)
    a_np = rng.standard_normal((2, 3, 3)) * 0.25 + np.eye(3) * 2.0
    b_np = rng.standard_normal((2, 3, 2)) if left else rng.standard_normal((2, 2, 3))
    a = lucid.tensor(a_np, requires_grad=True)
    b = lucid.tensor(b_np, requires_grad=True)
    lu_res = LA.lu_factor(a)
    x = LA.lu_solve(lu_res.LU, lu_res.pivots, b, left=left, adjoint=adjoint)
    mat = a.mT if adjoint else a
    want = LA.solve(mat, b, left=left)
    np.testing.assert_allclose(x.detach().numpy(), want.detach().numpy(), rtol=1e-10, atol=1e-10)
    (x * 1.5).sum().backward()
    assert a.grad is not None and np.isfinite(a.grad.numpy()).all()
    assert b.grad is not None and np.isfinite(b.grad.numpy()).all()


def test_linalg_decompositions_return_named_tuples() -> None:
    a = lucid.tensor([[4.0, 1.0], [1.0, 3.0]])
    b = lucid.tensor([[1.0], [2.0]])
    sd = LA.slogdet(a)
    assert (sd.sign, sd.logabsdet) == (sd[0], sd[1])
    eh = LA.eigh(a)
    assert (eh.eigenvalues, eh.eigenvectors) == (eh[0], eh[1])
    eg = LA.eig(a)
    assert (eg.eigenvalues, eg.eigenvectors) == (eg[0], eg[1])
    sv = LA.svd(a)
    assert (sv.U, sv.S, sv.Vh) == (sv[0], sv[1], sv[2])
    qr = LA.qr(a)
    assert (qr.Q, qr.R) == (qr[0], qr[1])
    luf = LA.lu_factor(a)
    assert (luf.LU, luf.pivots) == (luf[0], luf[1])
    lufe = LA.lu_factor_ex(a)
    assert (lufe.LU, lufe.pivots, lufe.info) == (lufe[0], lufe[1], lufe[2])
    lu = LA.lu(a)
    assert (lu.P, lu.L, lu.U) == (lu[0], lu[1], lu[2])
    inve = LA.inv_ex(a)
    assert (inve.inverse, inve.info) == (inve[0], inve[1])
    che = LA.cholesky_ex(a)
    assert (che.L, che.info) == (che[0], che[1])
    sole = LA.solve_ex(a, b)
    assert (sole.result, sole.info) == (sole[0], sole[1])
    lsq = LA.lstsq(a, b)
    assert (lsq.solution, lsq.residuals, lsq.rank, lsq.singular_values) == tuple(lsq)
    ldlf = LA.ldl_factor(a)
    assert (ldlf.LD, ldlf.pivots) == (ldlf[0], ldlf[1])
    ldlfe = LA.ldl_factor_ex(a)
    assert (ldlfe.LD, ldlfe.pivots, ldlfe.info) == (ldlfe[0], ldlfe[1], ldlfe[2])


@pytest.mark.parity
@pytest.mark.parametrize("shape", [(4, 2), (2, 4), (3, 3), (2, 4, 2)])
def test_qr_complete_and_r_modes_match_reference(
    ref: object, device: str, shape: tuple[int, ...]
) -> None:
    rng = np.random.default_rng(19)
    a_np = rng.standard_normal(shape).astype(np.float32)
    a = lucid.tensor(a_np, device=device, requires_grad=True)
    ra = ref.tensor(a_np)  # type: ignore[attr-defined]

    q_c, r_c = LA.qr(a, mode="complete")
    rq_c, rr_c = ref.linalg.qr(ra, mode="complete")  # type: ignore[attr-defined]
    assert q_c.shape == tuple(rq_c.shape)
    assert r_c.shape == tuple(rr_c.shape)
    m = shape[-2]
    eye = np.broadcast_to(np.eye(m, dtype=np.float32), (*shape[:-2], m, m))
    np.testing.assert_allclose((q_c.mT @ q_c).detach().numpy(), eye, atol=1e-5)
    np.testing.assert_allclose((q_c @ r_c).detach().numpy(), a_np, atol=1e-5)
    (q_c.sum() + r_c.sum()).backward()
    assert a.grad is not None and np.isfinite(a.grad.numpy()).all()

    q_r, r_r = LA.qr(a.detach(), mode="r")
    rq_r, rr_r = ref.linalg.qr(ra, mode="r")  # type: ignore[attr-defined]
    assert q_r.shape == tuple(rq_r.shape) == (0,)
    assert r_r.shape == tuple(rr_r.shape)
    np.testing.assert_allclose(
        np.abs(r_r.diagonal(dim1=-2, dim2=-1).numpy()),
        np.abs(rr_r.diagonal(dim1=-2, dim2=-1).numpy()),
        atol=1e-5,
    )
    with pytest.raises(ValueError, match="mode"):
        LA.qr(a.detach(), mode="raw")


@pytest.mark.parity
@pytest.mark.parametrize("driver", [None, "gels", "gelsy", "gelsd", "gelss"])
def test_lstsq_drivers_rcond_residuals_rank_and_sv_match_reference(
    ref: object, driver: str | None
) -> None:
    rng = np.random.default_rng(23)
    a_np = rng.standard_normal((5, 3))
    b_np = rng.standard_normal((5, 2))
    a, b = lucid.tensor(a_np), lucid.tensor(b_np)
    ra, rb = ref.tensor(a_np), ref.tensor(b_np)  # type: ignore[attr-defined]
    got = LA.lstsq(a, b, rcond=1e-6, driver=driver)
    want = ref.linalg.lstsq(ra, rb, rcond=1e-6, driver=driver)  # type: ignore[attr-defined]
    for g, w in zip(got, want, strict=True):
        assert g.shape == tuple(w.shape)
        assert g.dtype == getattr(lucid, str(w.dtype).split(".")[-1])
        if g.numel() > 0:
            np.testing.assert_allclose(g.numpy(), w.numpy(), rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="driver"):
        LA.lstsq(a, b, driver="bogus")
