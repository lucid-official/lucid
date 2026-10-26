"""``lucid.einops`` — rearrange / reduce / repeat / einsum."""

import numpy as np
import pytest

import lucid


class TestRearrange:
    def test_basic(self, device: str) -> None:
        x = lucid.arange(0.0, 24.0, 1.0, device=device).reshape(2, 3, 4)
        out = lucid.einops.rearrange(x, "a b c -> a c b")
        assert out.shape == (2, 4, 3)

    def test_merge_dims(self, device: str) -> None:
        x = lucid.zeros(2, 3, 4, device=device)
        out = lucid.einops.rearrange(x, "a b c -> a (b c)")
        assert out.shape == (2, 12)


class TestReduce:
    def test_sum(self, device: str) -> None:
        x = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        out = lucid.einops.reduce(x, "h w -> h", "sum")
        np.testing.assert_array_equal(out.numpy(), [3.0, 7.0])

    def test_mean(self, device: str) -> None:
        x = lucid.tensor([[1.0, 3.0], [5.0, 7.0]], device=device)
        out = lucid.einops.reduce(x, "h w -> h", "mean")
        np.testing.assert_array_equal(out.numpy(), [2.0, 6.0])


class TestRepeat:
    def test_basic(self, device: str) -> None:
        x = lucid.tensor([1.0, 2.0, 3.0], device=device)
        out = lucid.einops.repeat(x, "n -> n k", k=2)
        assert out.shape == (3, 2)


# Two-operand contraction patterns.  The matmul-routed ones (a shared
# contracted label) are listed first; the trailing three exercise the
# mul+sum fallback (no contraction: elementwise / outer product) and the
# single-operand reduction shortcut.
_EINSUM_PAIRS: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = [
    ("ij,jk->ik", (4, 5), (5, 6)),  # plain GEMM
    ("bij,bjk->bik", (3, 4, 5), (3, 5, 6)),  # batched matmul
    ("bhij,bhjk->bhik", (2, 3, 4, 5), (2, 3, 5, 6)),  # 2-batch (attention)
    ("ij,ij->i", (4, 5), (4, 5)),  # batched dot (contract j)
    ("bid,bjd->bij", (2, 4, 8), (2, 5, 8)),  # attention scores
    ("bij,bjd->bid", (2, 4, 5), (2, 5, 8)),  # attention apply
    ("nqhd,nkhd->nqhk", (2, 4, 3, 8), (2, 5, 3, 8)),  # multi-batch contract
    ("ij,ij->ij", (4, 5), (4, 5)),  # elementwise (mul+sum path)
    ("i,j->ij", (4,), (5,)),  # outer product (mul+sum path)
]


class TestEinsum:
    @pytest.mark.parametrize(
        "eq,sa,sb", _EINSUM_PAIRS, ids=[p[0] for p in _EINSUM_PAIRS]
    )
    def test_two_operand_matches_reference(
        self, eq: str, sa: tuple[int, ...], sb: tuple[int, ...], device: str
    ) -> None:
        np.random.seed(0)
        a = np.random.randn(*sa).astype(np.float32)
        b = np.random.randn(*sb).astype(np.float32)
        out = lucid.einops.einsum(
            eq,
            lucid.tensor(a.copy(), device=device),
            lucid.tensor(b.copy(), device=device),
        )
        ref = np.einsum(eq, a, b)
        assert out.shape == ref.shape
        np.testing.assert_allclose(out.numpy(), ref, atol=1e-4)

    def test_single_operand_reduction(self, device: str) -> None:
        np.random.seed(1)
        a = np.random.randn(3, 4, 5).astype(np.float32)
        out = lucid.einops.einsum("ijk->ik", lucid.tensor(a.copy(), device=device))
        np.testing.assert_allclose(out.numpy(), np.einsum("ijk->ik", a), atol=1e-4)

    def test_three_operand_chain(self, device: str) -> None:
        np.random.seed(2)
        x = np.random.randn(3, 4).astype(np.float32)
        y = np.random.randn(4, 5).astype(np.float32)
        z = np.random.randn(5, 6).astype(np.float32)
        out = lucid.einops.einsum(
            "ij,jk,kl->il",
            lucid.tensor(x.copy(), device=device),
            lucid.tensor(y.copy(), device=device),
            lucid.tensor(z.copy(), device=device),
        )
        np.testing.assert_allclose(
            out.numpy(), np.einsum("ij,jk,kl->il", x, y, z), atol=1e-4
        )

    def test_integer_is_exact(self, device: str) -> None:
        # Integer einsum must stay on the exact mul+sum path (GEMM is
        # float-only), so the result is bit-exact, not just close.
        a = np.arange(12).reshape(3, 4).astype(np.int64)
        b = np.arange(20).reshape(4, 5).astype(np.int64)
        out = lucid.einops.einsum(
            "ij,jk->ik",
            lucid.tensor(a.copy(), device=device),
            lucid.tensor(b.copy(), device=device),
        )
        np.testing.assert_array_equal(out.numpy(), np.einsum("ij,jk->ik", a, b))
        assert out.dtype == lucid.int64

    def test_mixed_dtypes_refuse_on_contraction_and_promote_on_outer(
        self, device: str
    ) -> None:
        # Contractions refuse mixed dtypes outside autocast (matching the
        # reference), while non-contracted outer / Hadamard products promote.
        a = np.arange(6).reshape(2, 3).astype(np.int64)
        b = (np.arange(12).reshape(3, 4) / 4).astype(np.float32)
        with pytest.raises(TypeError, match="DtypeMismatch"):
            lucid.einops.einsum(
                "ij,jk->ik",
                lucid.tensor(a.copy(), device=device),
                lucid.tensor(b.copy(), device=device),
            )
        outer = lucid.einops.einsum(
            "i,j->ij",
            lucid.tensor([1, 2], dtype=lucid.int64, device=device),
            lucid.tensor([0.5, 1.5], device=device),
        )
        assert outer.dtype == lucid.float32
        assert outer.tolist() == [[0.5, 1.5], [1.0, 3.0]]

    def test_backward_flows(self, device: str) -> None:
        # The matmul-routed path must remain differentiable end to end.
        np.random.seed(3)
        a = np.random.randn(2, 4, 5).astype(np.float32)
        b = np.random.randn(2, 5, 6).astype(np.float32)
        la = lucid.tensor(a.copy(), device=device, requires_grad=True)
        lb = lucid.tensor(b.copy(), device=device, requires_grad=True)
        lucid.einops.einsum("bij,bjk->bik", la, lb).sum().backward()
        go = np.ones((2, 4, 6), np.float32)
        np.testing.assert_allclose(
            la.grad.numpy(), np.einsum("bik,bjk->bij", go, b), atol=1e-4
        )
        np.testing.assert_allclose(
            lb.grad.numpy(), np.einsum("bik,bij->bjk", go, a), atol=1e-4
        )


class TestIdentityPatternsReturnDistinctTensor:
    """Identity patterns must return a distinct Tensor / TensorImpl view so
    mutating metadata like ``requires_grad_`` on the output does not mutate
    the input tensor."""

    def test_rearrange_identity_isolates_requires_grad(self, device: str) -> None:
        x = lucid.randn(2, 3, device=device)
        y = lucid.einops.rearrange(x, "a b -> a b")
        assert y is not x
        assert y._impl is not x._impl
        y.requires_grad_(True)
        assert not x.requires_grad
        assert y.requires_grad

    def test_reduce_identity_isolates_requires_grad(self, device: str) -> None:
        x = lucid.randn(2, 3, device=device)
        y = lucid.einops.reduce(x, "a b -> a b", "sum")
        assert y is not x
        assert y._impl is not x._impl
        y.requires_grad_(True)
        assert not x.requires_grad
        assert y.requires_grad

    def test_repeat_identity_isolates_requires_grad(self, device: str) -> None:
        x = lucid.randn(2, 3, device=device)
        y = lucid.einops.repeat(x, "a b -> a b")
        assert y is not x
        assert y._impl is not x._impl
        y.requires_grad_(True)
        assert not x.requires_grad
        assert y.requires_grad

    def test_einsum_identity_isolates_requires_grad(self, device: str) -> None:
        x = lucid.randn(2, 3, device=device)
        y = lucid.einops.einsum("ij->ij", x)
        assert y is not x
        assert y._impl is not x._impl
        y.requires_grad_(True)
        assert not x.requires_grad
        assert y.requires_grad
