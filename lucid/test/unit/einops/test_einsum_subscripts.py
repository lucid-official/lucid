"""``lucid.einops.einsum`` beyond plain contractions.

A subscript repeated within one operand takes that operand's diagonal;
``...`` spans the remaining axes and broadcasts; a size-1 subscript
broadcasts against a longer one; the implicit form (no ``->``) picks the
once-occurring subscripts.  The engine planner assumed none of these, so
``"ii->i"`` returned its input unchanged and ``"ii->"`` summed the whole
matrix (LCD-186).  Values are checked against ``numpy.einsum`` on every
run; under ``-m parity`` values and gradients are compared with the
reference on every device, and malformed equations must be refused
wherever the reference refuses them.
"""

from typing import Any

import numpy as np
import pytest

import lucid
from lucid.test._helpers.compare import assert_close

# (equation, operand shapes).  No 0-d operands: what they test (an empty
# term) is covered by the engine's own path and needs no normalisation.
_CASES: list[tuple[str, list[tuple[int, ...]]]] = [
    # one operand, one repeated subscript
    ("ii->i", [(3, 3)]),
    ("ii->", [(3, 3)]),
    ("ii", [(3, 3)]),
    ("iji->j", [(2, 4, 2)]),
    ("bii->bi", [(2, 3, 3)]),
    ("iij->ij", [(2, 2, 3)]),
    ("iii->i", [(3, 3, 3)]),
    ("ijij->ij", [(2, 3, 2, 3)]),
    ("iI->I", [(3, 3)]),
    # repeated subscripts among several operands
    ("ij,jj->i", [(2, 3), (3, 3)]),
    ("ii,i->i", [(3, 3), (3,)]),
    ("ijj,jk->ik", [(2, 3, 3), (3, 4)]),
    ("bii,bjj->bij", [(2, 3, 3), (2, 4, 4)]),
    ("ba,ii", [(2, 3), (3, 3)]),
    # ellipsis, alone and with a repeat
    ("...ii->...i", [(2, 4, 3, 3)]),
    ("i...i->...", [(3, 2, 3)]),
    ("...ij,...jk->...ik", [(5, 1, 2, 3), (4, 3, 2)]),
    ("...->", [(2, 3)]),
    ("...,...->...", [(2, 3), (3,)]),
    ("Ab...,b->...A", [(2, 3, 4, 5), (3,)]),
    ("ij->i...", [(2, 2)]),
    # size-1 subscripts broadcast
    ("ij,jk->ik", [(2, 1), (3, 4)]),
    ("ij,ij->", [(1, 3), (4, 1)]),
    ("ij,j->ij", [(2, 1), (5,)]),
    # three operands
    ("ij,jk,kl->il", [(2, 3), (3, 4), (4, 5)]),
]

# ``numpy.einsum`` refuses to sum an ellipsis away; the reference sums it.
_NUMPY_CASES = [c for c in _CASES if c[0] != "...->"]

_REFUSED: list[tuple[str, list[tuple[int, ...]]]] = [
    ("ij->ik", [(2, 2)]),  # output subscript in no input
    ("ij->ii", [(2, 2)]),  # repeated output subscript
    ("i->i", [(2, 2)]),  # rank does not match the subscripts
    ("ijk->", [(2, 2)]),
    ("ij,jk->ik", [(2, 2)]),  # too few operands
    ("i,i->i", [(2,), (2,), (2,)]),  # too many operands
    ("i1->i", [(2, 2)]),  # subscript outside [a-zA-Z]
    ("ij->i->j", [(2, 2)]),
    ("ij-->i", [(2, 2)]),
    ("ij..->i", [(2, 2)]),  # '.' outside an ellipsis
    ("i->.", [(2,)]),
    ("...i...->i", [(2, 2)]),  # two ellipses in one term
    ("ii->i", [(2, 3)]),  # repeated subscript over unequal sizes
    ("ii->i", [(1, 3)]),  # ... where 1 does not broadcast
    ("ij,jk->ik", [(2, 3), (4, 5)]),  # sizes that do not broadcast
]


def _arrays(shapes: list[tuple[int, ...]], seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(s).astype(np.float32) for s in shapes]


class TestEinsumSubscripts:
    @pytest.mark.parametrize(
        "eq,shapes", _NUMPY_CASES, ids=[f"{c[0]}|{len(c[1])}" for c in _NUMPY_CASES]
    )
    def test_matches_numpy(self, device: str, eq: str, shapes: list[tuple[int, ...]]) -> None:
        arrays = _arrays(shapes, seed=0)
        out = lucid.einops.einsum(eq, *[lucid.tensor(a, device=device) for a in arrays])
        expected = np.einsum(eq, *arrays)
        assert out.shape == expected.shape
        np.testing.assert_allclose(out.numpy(), expected, atol=1e-4)

    def test_trace_and_diagonal_literals(self, device: str) -> None:
        m = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
        assert lucid.einops.einsum("ii->i", m).tolist() == [1.0, 4.0]
        assert lucid.einops.einsum("ii->", m).item() == 5.0
        assert lucid.einops.einsum("ii", m).item() == 5.0

    def test_empty_diagonal(self, device: str) -> None:
        x = lucid.zeros(0, 0, device=device, requires_grad=True)
        out = lucid.einops.einsum("ii->i", x)
        assert out.shape == (0,)
        out.sum().backward()
        assert x.grad is not None and x.grad.shape == (0, 0)


@pytest.mark.parity
class TestEinsumSubscriptsParity:
    @pytest.mark.parametrize("eq,shapes", _CASES, ids=[f"{c[0]}|{len(c[1])}" for c in _CASES])
    def test_value_and_grad(
        self, ref: Any, device: str, eq: str, shapes: list[tuple[int, ...]]
    ) -> None:
        arrays = _arrays(shapes, seed=0)
        ours = [lucid.tensor(a, device=device, requires_grad=True) for a in arrays]
        theirs = [ref.tensor(a, requires_grad=True) for a in arrays]

        out = lucid.einops.einsum(eq, *ours)
        expected = ref.einsum(eq, *theirs)
        assert out.device.type == device
        assert_close(out, expected, atol=1e-4)

        # A non-uniform upstream gradient: an all-ones one cannot tell the
        # diagonal's scatter apart from a broadcast.
        upstream = _arrays([tuple(expected.shape)], seed=1)[0]
        (out * lucid.tensor(upstream, device=device)).sum().backward()
        (expected * ref.tensor(upstream)).sum().backward()
        for k, (a, b) in enumerate(zip(ours, theirs, strict=True)):
            assert_close(a.grad, b.grad, atol=1e-4, msg=f"grad of operand {k}")

    def test_integer_trace_is_exact(self, ref: Any, device: str) -> None:
        a = np.arange(16, dtype=np.int64).reshape(4, 4)
        out = lucid.einops.einsum("ii->", lucid.tensor(a, device=device))
        expected = ref.einsum("ii->", ref.tensor(a))
        assert out.dtype == lucid.int64
        assert out.item() == expected.item()

    @pytest.mark.parametrize("eq,shapes", _REFUSED, ids=[f"{c[0]}|{c[1]}" for c in _REFUSED])
    def test_refused_like_reference(self, ref: Any, eq: str, shapes: list[tuple[int, ...]]) -> None:
        arrays = _arrays(shapes, seed=0)
        with pytest.raises(RuntimeError):
            ref.einsum(eq, *[ref.tensor(a) for a in arrays])
        # Lucid's refusals are ``InvalidArgument`` / ``ShapeMismatch`` —
        # both a ``ValueError`` and the ``RuntimeError`` they used to be.
        with pytest.raises(ValueError, match=r"^einsum\(\)"):
            lucid.einops.einsum(eq, *[lucid.tensor(a) for a in arrays])
