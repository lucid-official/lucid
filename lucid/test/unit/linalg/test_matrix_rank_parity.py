"""Parity and contract tests for ``lucid.linalg.matrix_rank`` (LCD-51).

Covers:
1. Per-matrix reduction over ``dim=-1`` for batched inputs (instead of global
   ``max`` and global ``sum`` across all batch dimensions).
2. Dtype-aware default relative tolerance ``max(m, n) * finfo(S.dtype).eps``
   (instead of a hardcoded ``float32`` epsilon on ``float64`` inputs).
3. Reference signature and semantics for ``tol``, ``atol``, ``rtol`` (scalars
   and broadcastable tensors), and ``hermitian=True``.
"""

import inspect
from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.linalg as LA


def _to_ref_dtype(ref: ModuleType, dt: lucid.dtype) -> object:
    return {
        lucid.float32: ref.float32,
        lucid.float64: ref.float64,
    }[dt]


def test_matrix_rank_signature_exposes_tol_hermitian_atol_rtol() -> None:
    sig = inspect.signature(LA.matrix_rank)
    params = list(sig.parameters.values())
    assert [p.name for p in params] == ["input", "tol", "hermitian", "atol", "rtol"]
    assert params[0].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params[1].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params[1].default is None
    assert params[2].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params[2].default is False
    assert params[3].kind is inspect.Parameter.KEYWORD_ONLY
    assert params[3].default is None
    assert params[4].kind is inspect.Parameter.KEYWORD_ONLY
    assert params[4].default is None


def test_batched_matrix_rank_reduces_per_matrix_not_globally(ref: ModuleType, device: str) -> None:
    """LCD-51 repro 1: a large singular value in one batch entry must not
    inflate the threshold of another batch entry, and the output must have
    shape ``A.shape[:-2]`` rather than ``()``."""
    arr = np.array([[[1.0, 0.0], [0.0, 1e-3]], [[1e4, 0.0], [0.0, 1e4]]], dtype=np.float32)
    got = LA.matrix_rank(lucid.tensor(arr, device=device))
    want = ref.linalg.matrix_rank(ref.tensor(arr))
    assert got.dtype is lucid.int64
    assert device in str(got.device)
    assert tuple(got.shape) == (2,)
    assert got.tolist() == want.tolist() == [2, 2]


@pytest.mark.parametrize("batch_shape", [(3,), (2, 4), (2, 1, 3)])
@pytest.mark.parametrize("mat_shape", [(3, 3), (4, 2), (2, 5)])
def test_higher_rank_batches_match_reference(
    ref: ModuleType,
    device: str,
    batch_shape: tuple[int, ...],
    mat_shape: tuple[int, ...],
) -> None:
    rng = np.random.default_rng(42)
    m, n = mat_shape
    k = min(m, n)
    # Construct matrices with varying ranks and scales across the batch.
    u, _ = np.linalg.qr(rng.standard_normal((*batch_shape, m, m)))
    v, _ = np.linalg.qr(rng.standard_normal((*batch_shape, n, n)))
    s = np.zeros((*batch_shape, k), dtype=np.float32)
    flat_s = s.reshape(-1, k)
    for idx in range(flat_s.shape[0]):
        target_rank = idx % (k + 1)
        scale = 10.0 ** (idx - flat_s.shape[0] // 2)
        flat_s[idx, :target_rank] = scale * np.arange(target_rank, 0, -1, dtype=np.float32)
    mat = (u[..., :, :k] * s[..., None, :]) @ v[..., :k, :]
    got = LA.matrix_rank(lucid.tensor(mat, dtype=lucid.float32, device=device))
    want = ref.linalg.matrix_rank(ref.tensor(mat, dtype=ref.float32))
    assert tuple(got.shape) == batch_shape
    assert got.dtype is lucid.int64
    assert got.tolist() == want.tolist()


def test_float64_default_tolerance_uses_float64_eps(ref: ModuleType) -> None:
    """LCD-51 repro 2: ``diag(1, 1e-9)`` has rank 2 in float64 (eps ≈ 2.22e-16)
    and rank 1 in float32 (eps ≈ 1.19e-7)."""
    a64 = lucid.diag_embed(lucid.tensor([1.0, 1e-9], dtype=lucid.float64))
    r64 = ref.diag(ref.tensor([1.0, 1e-9], dtype=ref.float64))
    assert LA.matrix_rank(a64).tolist() == ref.linalg.matrix_rank(r64).tolist() == 2

    a32 = lucid.diag_embed(lucid.tensor([1.0, 1e-9], dtype=lucid.float32))
    r32 = ref.diag(ref.tensor([1.0, 1e-9], dtype=ref.float32))
    assert LA.matrix_rank(a32).tolist() == ref.linalg.matrix_rank(r32).tolist() == 1


@pytest.mark.parametrize("dtype", [lucid.float32, lucid.float64])
@pytest.mark.parametrize(
    "atol", [None, -1.0, 0.0, 1e-10, 1e-5, 0.5, 2.0, float("inf"), float("-inf")]
)
@pytest.mark.parametrize(
    "rtol", [None, -1.0, 0.0, 1e-10, 1e-5, 0.05, 0.5, float("inf"), float("-inf")]
)
def test_scalar_atol_rtol_grid_matches_reference(
    ref: ModuleType,
    dtype: lucid.dtype,
    atol: float | None,
    rtol: float | None,
) -> None:
    arr = np.array(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1e-8]],
            [[100.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.float64 if dtype is lucid.float64 else np.float32,
    )
    a_l = lucid.tensor(arr, dtype=dtype)
    a_r = ref.tensor(arr, dtype=_to_ref_dtype(ref, dtype))
    got = LA.matrix_rank(a_l, atol=atol, rtol=rtol)
    want = ref.linalg.matrix_rank(a_r, atol=atol, rtol=rtol)
    assert got.tolist() == want.tolist()


@pytest.mark.parametrize("tol", [-1.0, 0.0, 1e-10, 1e-5, 0.5, 2.0])
def test_tol_positional_and_keyword_matches_reference(
    ref: ModuleType, device: str, tol: float
) -> None:
    arr = np.array(
        [
            [[1.0, 0.0], [0.0, 1e-8]],
            [[1e10, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    a_l = lucid.tensor(arr, dtype=lucid.float32, device=device)
    a_r = ref.tensor(arr, dtype=ref.float32)
    got_pos = LA.matrix_rank(a_l, tol)
    got_kw = LA.matrix_rank(a_l, tol=tol)
    want = ref.linalg.matrix_rank(a_r, tol=tol)
    assert got_pos.tolist() == want.tolist()
    assert got_kw.tolist() == want.tolist()


def test_tensor_tolerances_broadcast_across_batches(ref: ModuleType, device: str) -> None:
    arr = np.array(
        [
            [[1e10, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1e-8]],
            [[1e10, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1e-8]],
        ],
        dtype=np.float32,
    )
    a_l = lucid.tensor(arr, device=device)
    a_r = ref.tensor(arr)

    # Element 0 has atol=0.0 (so rtol defaults to max(m,n)*eps -> rank 1);
    # element 1 has atol=0.5 > 0 (so rtol defaults to 0.0 -> rank 2).
    atol_l = lucid.tensor([0.0, 0.5], device=device)
    atol_r = ref.tensor([0.0, 0.5])
    assert (
        LA.matrix_rank(a_l, atol=atol_l).tolist()
        == ref.linalg.matrix_rank(a_r, atol=atol_r).tolist()
        == [1, 2]
    )

    # With tol=tensor([0.0, 0.5]), rtol is unconditionally 0.0 -> [3, 2].
    assert (
        LA.matrix_rank(a_l, tol=atol_l).tolist()
        == ref.linalg.matrix_rank(a_r, tol=atol_r).tolist()
        == [3, 2]
    )

    # Higher-rank tolerance tensor (2, 1) broadcasting against batch (2,) -> (2, 2).
    atol_2d_l = lucid.tensor([[0.0], [0.5]], device=device)
    atol_2d_r = ref.tensor([[0.0], [0.5]])
    got_2d = LA.matrix_rank(a_l, atol=atol_2d_l)
    want_2d = ref.linalg.matrix_rank(a_r, atol=atol_2d_r)
    assert tuple(got_2d.shape) == tuple(want_2d.shape) == (2, 2)
    assert got_2d.tolist() == want_2d.tolist()

    # Combined atol and rtol tensors.
    rtol_l = lucid.tensor([1e-12, 0.5], device=device)
    rtol_r = ref.tensor([1e-12, 0.5])
    assert (
        LA.matrix_rank(a_l, atol=atol_l, rtol=rtol_l).tolist()
        == ref.linalg.matrix_rank(a_r, atol=atol_r, rtol=rtol_r).tolist()
        == [2, 1]
    )


def test_hermitian_uses_lower_triangle_eigenvalues(ref: ModuleType, device: str) -> None:
    mats = np.array(
        [
            [[2.0, 1.0], [1.0, 3.0]],
            [[-5.0, 0.0], [0.0, 1.0]],
            [[0.0, 2.0], [0.0, 0.0]],  # upper-only entry ignored when hermitian=True
            [[1.0, 99.0], [2.0, 4.0]],  # lower triangle is [[1, 2], [2, 4]] -> rank 1
        ],
        dtype=np.float32,
    )
    a_l = lucid.tensor(mats, device=device)
    a_r = ref.tensor(mats)
    for herm in (False, True):
        got = LA.matrix_rank(a_l, hermitian=herm)
        want = ref.linalg.matrix_rank(a_r, hermitian=herm)
        assert got.tolist() == want.tolist()


@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (3, 0), (2, 0, 3), (0, 3, 3)])
def test_empty_matrices_return_int64_zeros_of_batch_shape(
    ref: ModuleType, device: str, shape: tuple[int, ...]
) -> None:
    a_l = lucid.zeros(*shape, device=device)
    a_r = ref.zeros(*shape)
    got = LA.matrix_rank(a_l)
    want = ref.linalg.matrix_rank(a_r)
    assert tuple(got.shape) == tuple(want.shape) == shape[:-2]
    assert got.dtype is lucid.int64
    assert device in str(got.device)
    assert got.tolist() == want.tolist()


def test_nonfinite_inputs_and_tolerances_match_reference(ref: ModuleType, device: str) -> None:
    for val in (float("nan"), float("inf"), float("-inf")):
        a_l = lucid.tensor([[1.0, 0.0], [0.0, val]], device=device)
        a_r = ref.tensor([[1.0, 0.0], [0.0, val]])
        assert LA.matrix_rank(a_l).tolist() == ref.linalg.matrix_rank(a_r).tolist()

        eye_l = lucid.eye(2, device=device)
        eye_r = ref.eye(2)
        assert (
            LA.matrix_rank(eye_l, atol=val).tolist()
            == ref.linalg.matrix_rank(eye_r, atol=val).tolist()
        )


def test_requires_grad_input_returns_detached_int64(device: str) -> None:
    a = lucid.eye(3, device=device, requires_grad=True)
    out = LA.matrix_rank(a)
    assert out.dtype is lucid.int64
    assert not out.requires_grad
    assert int(out.item()) == 3


def test_invalid_inputs_and_tolerance_combinations_are_refused() -> None:
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        LA.matrix_rank(lucid.tensor(1.0))
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        LA.matrix_rank(lucid.tensor([1.0, 2.0]))
    with pytest.raises(ValueError, match="tol cannot be specified together"):
        LA.matrix_rank(lucid.eye(2), tol=1e-3, atol=1e-3)
    with pytest.raises(ValueError, match="tol cannot be specified together"):
        LA.matrix_rank(lucid.eye(2), tol=1e-3, rtol=1e-3)
    c_tol = lucid.tensor(1e-3 + 0j, dtype=lucid.complex64)
    with pytest.raises(TypeError, match="complex"):
        LA.matrix_rank(lucid.eye(2), atol=c_tol)
    with pytest.raises(TypeError, match="complex"):
        LA.matrix_rank(lucid.eye(2), rtol=c_tol)
    with pytest.raises(TypeError, match="complex"):
        LA.matrix_rank(lucid.eye(2), tol=c_tol)


def test_solve_and_solve_ex_left_false_and_matrix_norm_dtype(ref: ModuleType, device: str) -> None:
    rng = np.random.default_rng(7)
    a_np = (rng.standard_normal((2, 3, 3)) + 4.0 * np.eye(3)).astype(np.float32)
    b2_np = rng.standard_normal((2, 4, 3)).astype(np.float32)
    b1_np = rng.standard_normal((3,)).astype(np.float32)

    a_l = lucid.tensor(a_np, device=device, requires_grad=True)
    b2_l = lucid.tensor(b2_np, device=device, requires_grad=True)
    a_r = ref.tensor(a_np, requires_grad=True)
    b2_r = ref.tensor(b2_np, requires_grad=True)

    x_l = LA.solve(a_l, b2_l, left=False)
    x_r = ref.linalg.solve(a_r, b2_r, left=False)
    assert np.allclose(x_l.detach().numpy(), x_r.detach().numpy(), atol=1e-5)
    x_l.sum().backward()
    x_r.sum().backward()
    assert a_l.grad is not None and b2_l.grad is not None
    assert np.allclose(a_l.grad.numpy(), a_r.grad.numpy(), atol=1e-5)
    assert np.allclose(b2_l.grad.numpy(), b2_r.grad.numpy(), atol=1e-5)

    x_ex_l, info_l = LA.solve_ex(
        lucid.tensor(a_np, device=device),
        lucid.tensor(b2_np, device=device),
        left=False,
    )
    x_ex_r, info_r = ref.linalg.solve_ex(ref.tensor(a_np), ref.tensor(b2_np), left=False)
    assert np.allclose(x_ex_l.numpy(), x_ex_r.numpy(), atol=1e-5)
    assert info_l.tolist() == info_r.tolist() == [0, 0]

    with pytest.raises(RuntimeError):
        LA.solve(
            lucid.tensor(a_np[0], device=device),
            lucid.tensor(b1_np, device=device),
            left=False,
        )
    with pytest.raises(RuntimeError):
        ref.linalg.solve(ref.tensor(a_np[0]), ref.tensor(b1_np), left=False)

    # Batched vector B (shape == A.shape[:-1]) is also refused under left=False,
    # whereas 2-D B against higher-rank A broadcasts across the batch.
    with pytest.raises(RuntimeError):
        LA.solve(
            lucid.tensor(a_np, device=device),
            lucid.zeros(2, 3, device=device),
            left=False,
        )
    a4_l = lucid.tensor(a_np[:, None], device=device)
    b_unbatched_l = lucid.tensor(b2_np[0], device=device)
    x4_l = LA.solve(a4_l, b_unbatched_l, left=False)
    x4_r = ref.linalg.solve(ref.tensor(a_np[:, None]), ref.tensor(b2_np[0]), left=False)
    assert tuple(x4_l.shape) == (2, 1, 4, 3)
    assert np.allclose(x4_l.numpy(), x4_r.numpy(), atol=1e-5)

    # pinv backward with hermitian=True and atol truncation
    sym_np = np.array([[2.0, 1.0], [1.0, 3.0]], dtype=np.float32)
    cot_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    sym_l = lucid.tensor(sym_np, device=device, requires_grad=True)
    sym_r = ref.tensor(sym_np, requires_grad=True)
    (LA.pinv(sym_l, hermitian=True) * lucid.tensor(cot_np, device=device)).sum().backward()
    (ref.linalg.pinv(sym_r, hermitian=True) * ref.tensor(cot_np)).sum().backward()
    assert sym_l.grad is not None and sym_r.grad is not None
    assert np.allclose(sym_l.grad.numpy(), sym_r.grad.numpy(), atol=1e-5)

    tr_np = np.array([[2.0, 0.0], [0.0, 0.1]], dtype=np.float32)
    tr_l = lucid.tensor(tr_np, device=device, requires_grad=True)
    tr_r = ref.tensor(tr_np, requires_grad=True)
    (LA.pinv(tr_l, atol=0.5) * lucid.tensor(cot_np, device=device)).sum().backward()
    (ref.linalg.pinv(tr_r, atol=0.5) * ref.tensor(cot_np)).sum().backward()
    assert tr_l.grad is not None and tr_r.grad is not None
    assert np.allclose(tr_l.grad.numpy(), tr_r.grad.numpy(), atol=1e-5)

    if device == "cpu":
        mn_l = LA.matrix_norm(lucid.tensor(a_np[0]), ord="fro", dtype=lucid.float64)
        mn_r = ref.linalg.matrix_norm(ref.tensor(a_np[0]), ord="fro", dtype=ref.float64)
        assert mn_l.dtype is lucid.float64
        assert np.allclose(mn_l.numpy(), mn_r.numpy(), atol=1e-12)
