"""LCD-940: five linalg gaps against the reference.

* ``vector_norm`` / ``matrix_norm('fro')`` answered 0 on Metal for float32
  magnitudes below ~1e-19, where ``norm`` did not: the rescale lived in
  ``norm``'s dispatch alone.
* ``lu`` refused a rectangular input.
* ``eigh`` / ``eigvalsh`` refused a lowercase ``UPLO``.
* ``householder_product`` needed ``tau`` of length ``min(m, n)``.
* ``qr(mode='complete')`` of a tall matrix handed back a meaningless gradient
  through ``Q``.
"""

from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.linalg as LA


def _t(a: np.ndarray, device: str = "cpu", **kw: object) -> lucid.Tensor:
    return lucid.tensor(a, device=device, **kw)


# ── vector_norm / matrix_norm('fro') at tiny and huge magnitudes (CPU) ───────
#
# The CPU reduction rescales a slice whose powers would leave the float range.
# Metal's own range-safe kernel is covered by ``test_p_norm_scaled_gpu.py``.


@pytest.mark.parity
@pytest.mark.parametrize("magnitude", [1e-20, 1e15])
def test_entrywise_norms_agree_with_reference_at_extreme_magnitudes(
    ref: ModuleType, device_cpu_only: str, magnitude: float
) -> None:
    arr = np.full((2, 3), magnitude, dtype=np.float32)
    want = float(ref.linalg.vector_norm(ref.tensor(arr)))
    x = _t(arr)
    for got in (LA.vector_norm(x), LA.matrix_norm(x, "fro"), LA.norm(x), LA.norm(x.reshape(6))):
        assert got.item() == pytest.approx(want, rel=1e-5, abs=0)


@pytest.mark.parametrize("magnitude", [1e-30, 1e-35, 1e25, 1e30])
def test_entrywise_norms_stay_in_range_on_cpu(device_cpu_only: str, magnitude: float) -> None:
    """The reference's squares leave the range here; the true norm is representable."""
    x = _t(np.full((2, 3), magnitude, dtype=np.float32))
    want = float(np.sqrt(6.0) * magnitude)
    for got in (LA.vector_norm(x), LA.matrix_norm(x, "fro"), LA.norm(x)):
        assert got.item() == pytest.approx(want, rel=1e-5, abs=0)


@pytest.mark.parametrize("magnitude", [1e-20, 1e-30, 1e25])
def test_norm_still_rescales_on_every_device(device: str, magnitude: float) -> None:
    """``norm`` rescales by the largest magnitude whatever the device, as before."""
    x = _t(np.full((2, 3), magnitude, dtype=np.float32), device)
    want = float(np.sqrt(6.0) * magnitude)
    assert LA.norm(x).item() == pytest.approx(want, rel=1e-5, abs=0)
    assert LA.norm(x.reshape(6)).item() == pytest.approx(want, rel=1e-5, abs=0)


@pytest.mark.parametrize("ord_", [1, 2, 3.5, 0.5])
def test_vector_norm_orders_survive_tiny_magnitudes(device_cpu_only: str, ord_: float) -> None:
    base = np.array([3.0, -4.0, 1.0, 2.0])
    want = float((np.abs(base) ** ord_).sum() ** (1.0 / ord_)) * 1e-20
    arr = (base * 1e-20).astype(np.float32)
    assert LA.vector_norm(_t(arr), ord_).item() == pytest.approx(want, rel=1e-4, abs=0)


@pytest.mark.parity
def test_rescale_is_per_slice(ref: ModuleType, device_cpu_only: str) -> None:
    """A tiny row beside a normal one is not rescaled by the normal row's scale."""
    arr = np.array([[1e-20, 2e-20, 2e-20], [1.0, 2.0, 2.0]], dtype=np.float32)
    want = ref.linalg.vector_norm(ref.tensor(arr), 2, 1).tolist()
    got = LA.vector_norm(_t(arr), 2, 1).tolist()
    assert got == pytest.approx(want, rel=1e-5, abs=0)


def test_normal_magnitudes_match_the_plain_reduction(device: str) -> None:
    """Inside the window the CPU reduction is the plain one, bit for bit.

    Metal's kernel accumulates relative to the running largest magnitude (LCD-944),
    so its rounding differs from ``sqrt(sum(x * x))``'s by an ulp or two: it is
    faithful, not identical, and the test allows 4 ulp there.
    """
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((64, 33)).astype(np.float32)
    x = _t(arr, device)
    plain = lucid.sqrt((x * x).sum(dim=1))
    got = LA.vector_norm(x, 2, 1)
    if device == "cpu":
        assert got.tolist() == plain.tolist()
    else:
        assert got.tolist() == pytest.approx(plain.tolist(), rel=4 * 1.1920929e-07, abs=0)


def test_zero_nan_inf_slices_are_not_rescaled_into_nan(device_cpu_only: str) -> None:
    arr = np.array([[0.0, 0.0], [np.inf, 1.0], [np.nan, 1.0], [1e-30, 0.0]], dtype=np.float32)
    got = LA.vector_norm(_t(arr), 2, 1).tolist()
    assert got[0] == 0.0
    assert got[1] == float("inf")
    assert np.isnan(got[2])
    assert got[3] == pytest.approx(1e-30, rel=1e-5, abs=0)


@pytest.mark.parametrize("magnitude", [1e-20, 1e20])
def test_gradient_through_rescaled_norm_is_the_plain_norms(
    device_cpu_only: str, magnitude: float
) -> None:
    arr = np.array([3.0, 4.0], dtype=np.float32) * np.float32(magnitude)
    x = _t(arr, requires_grad=True)
    LA.vector_norm(x).backward()
    assert x.grad is not None
    assert x.grad.tolist() == pytest.approx([0.6, 0.8], rel=1e-5)


def test_tiny_float64_and_half_inputs(device_cpu_only: str) -> None:
    x64 = _t(np.full(4, 1e-200), dtype=lucid.float64)
    assert LA.vector_norm(x64).item() == pytest.approx(2e-200, rel=1e-12, abs=0)
    x16 = _t(np.full(4, 1.0, dtype=np.float16), dtype=lucid.float16)
    assert LA.vector_norm(x16).dtype is lucid.float16


@pytest.mark.parametrize(
    "scale",
    [
        1.0,
        pytest.param(
            1e-25,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "LCD-944: eager Metal runs the range-safe p-norm kernel, which "
                    "lucid.compile has no emitter for; a trace records the plain form"
                ),
            ),
        ),
    ],
)
def test_compiled_metal_vector_norm_records_the_plain_form(
    device_gpu_only: str, scale: float
) -> None:
    """A trace runs the plain reduction: it agrees with eager Metal at normal magnitudes."""
    fn = lucid.compile(lambda t: LA.vector_norm(t, 2, -1))
    fn(lucid.randn(4, 5, device="metal"))
    x = lucid.randn(4, 5, device="metal") * scale
    assert fn(x).tolist() == pytest.approx(LA.vector_norm(x, 2, -1).tolist(), rel=1e-6, abs=0)


# ── eigh / eigvalsh: UPLO is case-insensitive ────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("uplo", ["U", "L", "u", "l"])
def test_eigh_and_eigvalsh_accept_either_case_of_uplo(
    ref: ModuleType, device: str, uplo: str
) -> None:
    # Not symmetric: the answer depends on which triangle is read.
    arr = np.random.default_rng(3).standard_normal((2, 4, 4)).astype(np.float32)
    x = _t(arr, device)
    want_w, want_v = ref.linalg.eigh(ref.tensor(arr), uplo)
    w, v = LA.eigh(x, uplo)
    want = np.asarray(want_w.tolist())
    np.testing.assert_allclose(w.numpy(), want, atol=1e-4)
    np.testing.assert_allclose(LA.eigvalsh(x, uplo).numpy(), want, atol=1e-4)
    # Eigenvectors are sign-ambiguous: compare their squares.
    np.testing.assert_allclose(v.numpy() ** 2, np.asarray(want_v.tolist()) ** 2, atol=1e-3)


def test_uplo_lowercase_equals_uppercase_under_grad(device: str) -> None:
    arr = np.random.default_rng(4).standard_normal((3, 3)).astype(np.float32)
    grads = []
    for uplo in ("u", "U"):
        x = _t(arr, device, requires_grad=True)
        LA.eigvalsh(x, uplo).sum().backward()
        assert x.grad is not None
        grads.append(x.grad.tolist())
    assert grads[0] == grads[1]


def test_unknown_uplo_is_still_refused(device: str) -> None:
    with pytest.raises((ValueError, RuntimeError)):
        LA.eigvalsh(_t(np.eye(3, dtype=np.float32), device), "x")


# ── householder_product: any number of reflectors up to the column count ─────


def _reflectors(m: int, n: int, k: int, *, batch: tuple[int, ...] = ()) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(10 * m + n + k)
    return (
        rng.standard_normal((*batch, m, n)).astype(np.float32) * 0.5,
        rng.standard_normal((*batch, k)).astype(np.float32) * 0.5,
    )


@pytest.mark.parity
@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("batch", [(), (2,)])
def test_householder_product_accepts_fewer_reflectors_than_columns(
    ref: ModuleType, device: str, k: int, batch: tuple[int, ...]
) -> None:
    h, tau = _reflectors(5, 3, k, batch=batch)
    got = LA.householder_product(_t(h, device), _t(tau, device))
    want = ref.linalg.householder_product(ref.tensor(h), ref.tensor(tau))
    assert tuple(got.shape) == tuple(want.shape) == (*batch, 5, 3)
    np.testing.assert_allclose(got.numpy(), np.asarray(want.tolist()), atol=1e-5)


def test_householder_product_without_reflectors_is_the_identity_columns(device: str) -> None:
    h, tau = _reflectors(5, 3, 0)
    got = LA.householder_product(_t(h, device), _t(tau, device))
    np.testing.assert_array_equal(got.numpy(), np.eye(5, 3, dtype=np.float32))


@pytest.mark.parity
def test_householder_product_square_with_fewer_reflectors(ref: ModuleType, device: str) -> None:
    h, tau = _reflectors(4, 4, 2)
    got = LA.householder_product(_t(h, device), _t(tau, device))
    want = ref.linalg.householder_product(ref.tensor(h), ref.tensor(tau))
    np.testing.assert_allclose(got.numpy(), np.asarray(want.tolist()), atol=1e-5)


@pytest.mark.parity
def test_householder_product_refuses_more_reflectors_than_columns(
    ref: ModuleType, device: str
) -> None:
    h, tau = _reflectors(5, 3, 4)
    with pytest.raises(RuntimeError):
        ref.linalg.householder_product(ref.tensor(h), ref.tensor(tau))
    with pytest.raises(ValueError, match="tau.shape"):
        LA.householder_product(_t(h, device), _t(tau, device))


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_householder_product_gradcheck_float64(k: int) -> None:
    h, tau = _reflectors(5, 3, k)
    hh = _t(h.astype(np.float64), dtype=lucid.float64, requires_grad=True)
    tt = _t(tau.astype(np.float64), dtype=lucid.float64, requires_grad=True)
    w = lucid.tensor(np.random.default_rng(1).standard_normal((5, 3)), dtype=lucid.float64)
    inputs = [hh, tt] if k else [hh]
    assert lucid.autograd.gradcheck(
        lambda a, *rest: (LA.householder_product(a, rest[0] if rest else tt.detach()) * w).sum(),
        inputs,
    )


# ── qr(mode='complete'): Q's extra columns are not differentiable ────────────


@pytest.mark.parity
@pytest.mark.parametrize("shape", [(5, 3), (4, 1), (2, 5, 3)])
def test_complete_qr_backward_through_q_is_refused_when_tall(
    ref: ModuleType, device: str, shape: tuple[int, ...]
) -> None:
    arr = np.random.default_rng(5).standard_normal(shape).astype(np.float32)
    with pytest.raises(RuntimeError, match="not differentiable"):
        rq, _ = ref.linalg.qr(ref.tensor(arr, requires_grad=True), mode="complete")
        rq.sum().backward()
    x = _t(arr, device, requires_grad=True)
    q, _ = LA.qr(x, mode="complete")
    with pytest.raises(RuntimeError, match="not differentiable when mode='complete'"):
        q.sum().backward()
    # Reading only the thin columns still leads through the refused node.
    x = _t(arr, device, requires_grad=True)
    q, _ = LA.qr(x, mode="complete")
    with pytest.raises(RuntimeError, match="not differentiable"):
        q[..., :, : shape[-1]].sum().backward()


@pytest.mark.parametrize("shape", [(3, 3), (3, 5), (2, 3, 5)])
def test_complete_qr_is_differentiable_when_q_has_no_extra_columns(
    device: str, shape: tuple[int, ...]
) -> None:
    arr = np.random.default_rng(6).standard_normal(shape).astype(np.float32)
    x = _t(arr, device, requires_grad=True)
    q, r = LA.qr(x, mode="complete")
    (q.sum() + r.sum()).backward()
    assert x.grad is not None and np.isfinite(x.grad.numpy()).all()


def test_complete_qr_r_alone_stays_differentiable_float64() -> None:
    """The reference refuses this too; the gradient is right, so it is kept."""
    arr = np.random.default_rng(7).standard_normal((5, 3))
    x = _t(arr, dtype=lucid.float64, requires_grad=True)
    w = _t(np.random.default_rng(8).standard_normal((5, 3)), dtype=lucid.float64)
    assert lucid.autograd.gradcheck(lambda a: (LA.qr(a, mode="complete")[1] * w).sum(), [x])


def test_complete_qr_forward_unchanged_without_grad(device: str) -> None:
    arr = np.random.default_rng(9).standard_normal((5, 3)).astype(np.float32)
    q, r = LA.qr(_t(arr, device), mode="complete")
    assert tuple(q.shape) == (5, 5) and tuple(r.shape) == (5, 3)
    np.testing.assert_allclose((q.mT @ q).numpy(), np.eye(5), atol=1e-5)
    np.testing.assert_allclose((q @ r).numpy(), arr, atol=1e-5)


# ── lu: any shape, differentiable in any shape ───────────────────────────────

_LU_SHAPES = [(5, 3), (3, 5), (4, 4), (1, 4), (4, 1), (2, 5, 3), (2, 3, 5)]


@pytest.mark.parity
@pytest.mark.parametrize("shape", _LU_SHAPES)
def test_lu_factors_match_reference(ref: ModuleType, device: str, shape: tuple[int, ...]) -> None:
    arr = np.random.default_rng(sum(shape)).standard_normal(shape).astype(np.float32)
    got = LA.lu(_t(arr, device))
    want = ref.linalg.lu(ref.tensor(arr))
    for g, w in zip(got, want, strict=True):
        assert tuple(g.shape) == tuple(w.shape)
        np.testing.assert_allclose(g.numpy(), np.asarray(w.tolist()), atol=1e-5)
    m, n = shape[-2:]
    k = min(m, n)
    assert [tuple(t.shape) for t in got] == [
        (*shape[:-2], m, m),
        (*shape[:-2], m, k),
        (*shape[:-2], k, n),
    ]
    P, L, U = got
    np.testing.assert_allclose((P @ L @ U).numpy(), arr, atol=1e-5)


def test_lu_documented_shapes_for_tall_and_wide(device: str) -> None:
    tall = LA.lu(lucid.randn(5, 3, device=device))
    wide = LA.lu(lucid.randn(3, 5, device=device))
    assert [tuple(t.shape) for t in tall] == [(5, 5), (5, 3), (3, 3)]
    assert [tuple(t.shape) for t in wide] == [(3, 3), (3, 3), (3, 5)]


def test_lu_without_pivoting_is_still_refused() -> None:
    with pytest.raises(NotImplementedError):
        LA.lu(lucid.randn(3, 3), pivot=False)


def test_lu_refuses_a_vector() -> None:
    with pytest.raises(ValueError, match="matrix"):
        LA.lu(lucid.randn(4))


@pytest.mark.parity
@pytest.mark.parametrize("shape", _LU_SHAPES)
def test_lu_gradient_matches_reference(ref: ModuleType, shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(100 + sum(shape))
    arr = rng.standard_normal(shape)
    x = _t(arr, dtype=lucid.float64, requires_grad=True)
    _, L, U = LA.lu(x)
    wl = rng.standard_normal(tuple(L.shape))
    wu = rng.standard_normal(tuple(U.shape))
    ((L * _t(wl, dtype=lucid.float64)).sum() + (U * _t(wu, dtype=lucid.float64)).sum()).backward()
    rx = ref.tensor(arr, requires_grad=True)
    _, rL, rU = ref.linalg.lu(rx)
    ((rL * ref.tensor(wl)).sum() + (rU * ref.tensor(wu)).sum()).backward()
    assert x.grad is not None
    np.testing.assert_allclose(x.grad.numpy(), np.asarray(rx.grad.tolist()), atol=1e-10)


@pytest.mark.parametrize("shape", [(5, 3), (3, 5), (4, 4), (2, 4, 3)])
def test_lu_gradcheck_float64(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(200 + sum(shape))
    x = _t(rng.standard_normal(shape), dtype=lucid.float64, requires_grad=True)
    _, L0, U0 = LA.lu(x)
    wl = _t(rng.standard_normal(tuple(L0.shape)), dtype=lucid.float64)
    wu = _t(rng.standard_normal(tuple(U0.shape)), dtype=lucid.float64)

    def loss(a: lucid.Tensor) -> lucid.Tensor:
        _, L, U = LA.lu(a)
        return (L * wl).sum() + (U * wu).sum()

    assert lucid.autograd.gradcheck(loss, [x])


@pytest.mark.parametrize("shape", [(5, 3), (3, 5)])
def test_lu_factor_rectangular_is_differentiable(shape: tuple[int, ...]) -> None:
    """The packed factor's backward no longer refuses a non-square input."""
    x = _t(np.random.default_rng(1).standard_normal(shape), dtype=lucid.float64, requires_grad=True)
    LU, _ = LA.lu_factor(x)
    LU.sum().backward()
    assert x.grad is not None and np.isfinite(x.grad.numpy()).all()


def test_rectangular_lu_second_derivative_runs() -> None:
    x = _t(np.random.default_rng(2).standard_normal((4, 3)), dtype=lucid.float64)
    w = _t(np.random.default_rng(3).standard_normal((4, 3)), dtype=lucid.float64)
    assert lucid.autograd.gradgradcheck(lambda a: (LA.lu(a)[1] * w).sum(), [x.requires_grad_()])


# ── mixed dtypes: solve / tensorsolve / householder_product refuse ───────────
#    solve_triangular reads B in A's dtype and answers in it
# Metal has no float64, so the mixed pair there is (float32, float16).

_MIXED_PAIRS_CPU = [(lucid.float32, lucid.float64), (lucid.float64, lucid.float32)]
_MIXED_PAIRS_GPU = [(lucid.float32, lucid.float16), (lucid.float16, lucid.float32)]


def _mixed(device: str) -> list[tuple[lucid.dtype, lucid.dtype]]:
    return _MIXED_PAIRS_CPU if device == "cpu" else _MIXED_PAIRS_GPU


def _spd(n: int, dtype: lucid.dtype, device: str) -> lucid.Tensor:
    return (lucid.eye(n, dtype=dtype, device=device) * 3.0).triu() + lucid.ones(
        n, n, dtype=dtype, device=device
    ).triu(1)


@pytest.mark.parity
def test_solve_and_tensorsolve_refuse_mixed_dtypes(ref: ModuleType, device: str) -> None:
    ref_dtypes = {
        lucid.float32: ref.float32,
        lucid.float64: ref.float64,
        lucid.float16: ref.float16,
    }
    for da, db in _mixed(device):
        with pytest.raises(RuntimeError):
            ref.linalg.solve(ref.eye(3, dtype=ref_dtypes[da]), ref.ones(3, 2, dtype=ref_dtypes[db]))
        a = lucid.eye(3, dtype=da, device=device) * 2
        b = lucid.ones(3, 2, dtype=db, device=device)
        with pytest.raises(lucid._C.engine.DtypeMismatch, match="same dtype"):
            LA.solve(a, b)
        with pytest.raises(TypeError):  # DtypeMismatch is a TypeError
            LA.solve_ex(a, b)
        ta = lucid.eye(6, dtype=da, device=device).reshape(2, 3, 6)
        tb = lucid.ones(2, 3, dtype=db, device=device)
        with pytest.raises(lucid._C.engine.DtypeMismatch):
            LA.tensorsolve(ta, tb)


@pytest.mark.parity
def test_householder_product_refuses_mixed_dtypes(ref: ModuleType, device_cpu_only: str) -> None:
    for da, db in _MIXED_PAIRS_CPU:
        h = lucid.randn(4, 3, dtype=da)
        tau = lucid.randn(3, dtype=db)
        rd = {lucid.float32: ref.float32, lucid.float64: ref.float64}
        with pytest.raises(RuntimeError):
            ref.linalg.householder_product(
                ref.randn(4, 3, dtype=rd[da]), ref.randn(3, dtype=rd[db])
            )
        with pytest.raises(lucid._C.engine.DtypeMismatch, match="same dtype"):
            LA.householder_product(h, tau)


@pytest.mark.parity
def test_solve_triangular_answers_in_the_matrix_dtype(
    ref: ModuleType, device_cpu_only: str
) -> None:
    for da, db in _MIXED_PAIRS_CPU:
        rd = {lucid.float32: ref.float32, lucid.float64: ref.float64}
        a = _spd(3, da, "cpu")
        b = lucid.ones(3, 2, dtype=db) / 3
        got = LA.solve_triangular(a, b, upper=True)
        want = ref.linalg.solve_triangular(
            (ref.eye(3, dtype=rd[da]) * 3).triu() + ref.ones(3, 3, dtype=rd[da]).triu(1),
            ref.ones(3, 2, dtype=rd[db]) / 3,
            upper=True,
        )
        assert got.dtype is da and want.dtype == rd[da]
        np.testing.assert_allclose(got.numpy(), np.asarray(want.tolist()), rtol=1e-6)


def test_solve_triangular_mixed_dtype_runs_on_metal_and_keeps_gradients(
    device_gpu_only: str,
) -> None:
    a = _spd(3, lucid.float32, device_gpu_only).requires_grad_()
    b = lucid.ones(3, 2, dtype=lucid.float16, device=device_gpu_only)
    out = LA.solve_triangular(a, b, upper=True)
    assert out.dtype is lucid.float32
    out.sum().backward()
    assert a.grad is not None


def test_solve_triangular_mixed_dtype_gradient_reaches_b_in_its_own_dtype() -> None:
    a = _spd(3, lucid.float32, "cpu")
    b = (lucid.ones(3, 2, dtype=lucid.float64)).requires_grad_()
    LA.solve_triangular(a, b, upper=True).sum().backward()
    assert b.grad is not None and b.grad.dtype is lucid.float64


def test_same_dtype_calls_are_unchanged(device: str) -> None:
    dt = lucid.float32
    a = lucid.eye(3, dtype=dt, device=device) * 2
    b = lucid.ones(3, 2, dtype=dt, device=device)
    assert LA.solve(a, b).tolist() == [[0.5, 0.5]] * 3
    assert LA.solve_triangular(a, b, upper=True).dtype is dt


# ── householder_product refuses a wide H ─────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("batch", [(), (2,)])
def test_householder_product_refuses_a_wide_input(
    ref: ModuleType, device: str, batch: tuple[int, ...]
) -> None:
    h, tau = _reflectors(3, 5, 3, batch=batch)
    with pytest.raises(RuntimeError, match="must be greater than or equal"):
        ref.linalg.householder_product(ref.tensor(h), ref.tensor(tau))
    with pytest.raises(ValueError, match=r"input\.shape\[-2\] must be >= input\.shape\[-1\]"):
        LA.householder_product(_t(h, device), _t(tau, device))
