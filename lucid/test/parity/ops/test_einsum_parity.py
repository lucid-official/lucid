"""Reference parity for ``lucid.einops.einsum``.

The einsum implementation routes matmul-reducible pairwise contractions
through GEMM (a shared contracted label) and falls back to broadcast+mul+sum
for the rest.  Both paths must match the reference framework's ``einsum``,
forward and backward.
"""

from typing import Any

import numpy as np
import pytest

import lucid
from lucid.test._helpers.compare import assert_close

# (equation, shape_a, shape_b) — covers GEMM, batched matmul, attention
# shapes (matmul path) plus elementwise / outer product (mul+sum path).
_EINSUM_PAIRS: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = [
    ("ij,jk->ik", (4, 5), (5, 6)),
    ("bij,bjk->bik", (3, 4, 5), (3, 5, 6)),
    ("bhij,bhjk->bhik", (2, 3, 4, 5), (2, 3, 5, 6)),
    ("ij,ij->i", (4, 5), (4, 5)),
    ("bid,bjd->bij", (2, 4, 8), (2, 5, 8)),
    ("bij,bjd->bid", (2, 4, 5), (2, 5, 8)),
    ("nqhd,nkhd->nqhk", (2, 4, 3, 8), (2, 5, 3, 8)),
    ("ij,ij->ij", (4, 5), (4, 5)),
    ("i,j->ij", (4,), (5,)),
]


@pytest.mark.parity
class TestEinsumParity:
    @pytest.mark.parametrize("eq,sa,sb", _EINSUM_PAIRS, ids=[p[0] for p in _EINSUM_PAIRS])
    def test_forward(self, ref: Any, eq: str, sa: tuple[int, ...], sb: tuple[int, ...]) -> None:
        np.random.seed(0)
        a = np.random.randn(*sa).astype(np.float32)
        b = np.random.randn(*sb).astype(np.float32)
        out = lucid.einops.einsum(eq, lucid.tensor(a.copy()), lucid.tensor(b.copy()))
        expected = ref.einsum(eq, ref.tensor(a.copy()), ref.tensor(b.copy()))
        assert_close(out, expected, atol=1e-4)

    def test_backward(self, ref: Any) -> None:
        np.random.seed(1)
        a = np.random.randn(2, 4, 5).astype(np.float32)
        b = np.random.randn(2, 5, 6).astype(np.float32)

        la = lucid.tensor(a.copy(), requires_grad=True)
        lb = lucid.tensor(b.copy(), requires_grad=True)
        lucid.einops.einsum("bij,bjk->bik", la, lb).sum().backward()

        ra = ref.tensor(a.copy(), requires_grad=True)
        rb = ref.tensor(b.copy(), requires_grad=True)
        ref.einsum("bij,bjk->bik", ra, rb).sum().backward()

        assert_close(la.grad, ra.grad, atol=1e-4)
        assert_close(lb.grad, rb.grad, atol=1e-4)
