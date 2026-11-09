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


# ── vector_norm / matrix_norm('fro') at tiny and huge magnitudes ─────────────


@pytest.mark.parametrize("magnitude", [1e-20, 1e-30, 1e-35, 1e15])
def test_entrywise_norms_agree_with_reference_at_extreme_magnitudes(
    ref: ModuleType, device: str, magnitude: float
) -> None:
    arr = np.full((2, 3), magnitude, dtype=np.float32)
    want = float(ref.linalg.vector_norm(ref.tensor(arr)))
    x = _t(arr, device)
    for got in (LA.vector_norm(x), LA.matrix_norm(x, "fro"), LA.norm(x), LA.norm(x.reshape(6))):
        assert got.item() == pytest.approx(want, rel=1e-5)


@pytest.mark.parametrize("magnitude", [1e25, 1e30])
def test_entrywise_norms_do_not_overflow(device: str, magnitude: float) -> None:
    """The reference's squares overflow to ``inf`` here; the true norm is representable."""
    x = _t(np.full((2, 3), magnitude, dtype=np.float32), device)
    want = float(np.sqrt(6.0) * magnitude)
    for got in (LA.vector_norm(x), LA.matrix_norm(x, "fro"), LA.norm(x)):
        assert got.item() == pytest.approx(want, rel=1e-5)


@pytest.mark.parametrize("ord_", [1, 2, 3.5, 0.5])
def test_vector_norm_orders_survive_tiny_magnitudes(
    ref: ModuleType, device: str, ord_: float
) -> None:
    arr = np.array([3.0, -4.0, 1.0, 2.0], dtype=np.float32) * np.float32(1e-20)
    want = float(ref.linalg.vector_norm(ref.tensor(arr), ord_))
    assert LA.vector_norm(_t(arr, device), ord_).item() == pytest.approx(want, rel=1e-4)


def test_rescale_is_per_slice(ref: ModuleType, device: str) -> None:
    """A tiny row beside a normal one is not rescaled by the normal row's scale."""
    arr = np.array([[1e-20, 2e-20, 2e-20], [1.0, 2.0, 2.0]], dtype=np.float32)
    want = ref.linalg.vector_norm(ref.tensor(arr), 2, 1).tolist()
    got = LA.vector_norm(_t(arr, device), 2, 1).tolist()
    assert got == pytest.approx(want, rel=1e-5)


def test_normal_magnitudes_are_bit_identical(device: str) -> None:
    """Inside the window the rescale is a no-op, so prune & friends see no change."""
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((64, 33)).astype(np.float32)
    x = _t(arr, device)
    plain = lucid.sqrt((x * x).sum(dim=1))
    assert LA.vector_norm(x, 2, 1).tolist() == plain.tolist()


def test_zero_nan_inf_slices_are_not_rescaled_into_nan(device: str) -> None:
    arr = np.array([[0.0, 0.0], [np.inf, 1.0], [np.nan, 1.0], [1e-30, 0.0]], dtype=np.float32)
    got = LA.vector_norm(_t(arr, device), 2, 1).tolist()
    assert got[0] == 0.0
    assert got[1] == float("inf")
    assert np.isnan(got[2])
    assert got[3] == pytest.approx(1e-30, rel=1e-5)


@pytest.mark.parametrize("magnitude", [1e-20, 1e20])
def test_gradient_through_rescaled_norm_is_the_plain_norms(device: str, magnitude: float) -> None:
    arr = np.array([3.0, 4.0], dtype=np.float32) * np.float32(magnitude)
    x = _t(arr, device, requires_grad=True)
    LA.vector_norm(x).backward()
    assert x.grad is not None
    assert x.grad.tolist() == pytest.approx([0.6, 0.8], rel=1e-5)


def test_tiny_float64_and_half_inputs(device_cpu_only: str) -> None:
    x64 = _t(np.full(4, 1e-200), dtype=lucid.float64)
    assert LA.vector_norm(x64).item() == pytest.approx(2e-200, rel=1e-12)
    x16 = _t(np.full(4, 1.0, dtype=np.float16), dtype=lucid.float16)
    assert LA.vector_norm(x16).dtype is lucid.float16


def test_compiled_vector_norm_still_rescales_after_tracing_normal_input() -> None:
    """A trace records the rescaled form, not a branch decided on the traced values."""
    if not lucid.metal.is_available():
        pytest.skip("compile traces Metal only")
    fn = lucid.compile(lambda t: LA.vector_norm(t, 2, -1))
    fn(lucid.randn(4, 5, device="metal"))
    tiny = lucid.full((4, 5), 1e-25, device="metal")
    assert fn(tiny).tolist() == pytest.approx([np.sqrt(5) * 1e-25] * 4, rel=1e-4)


# ── eigh / eigvalsh: UPLO is case-insensitive ────────────────────────────────


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


def test_householder_product_square_with_fewer_reflectors(ref: ModuleType, device: str) -> None:
    h, tau = _reflectors(4, 4, 2)
    got = LA.householder_product(_t(h, device), _t(tau, device))
    want = ref.linalg.householder_product(ref.tensor(h), ref.tensor(tau))
    np.testing.assert_allclose(got.numpy(), np.asarray(want.tolist()), atol=1e-5)


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
