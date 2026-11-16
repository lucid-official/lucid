"""The rank-5+ index cases of LCD-1014, and the twin each one is held against.

A twin is the same elements in the layout ``RankCanon`` reads the case as, at rank <= 4.  A case
the pass rewrites to that layout therefore computes what its twin computes, and two tests read the
pair:

* ``test_rank_canonical_index.py`` compares the pass's IR for the two -- the same ops at the same
  shapes, none of them wider than rank 4.  It is deterministic, so it gates the phase gate;
* ``lucid/test/perf/compile/test_index_rank_twin.py`` compares the clock -- the case is not slower
  than its twin.  A ~1 ms ratio is not something the hosted runner holds (LCD-1039: 2.04x against
  a 1.5x bound, with the graph unchanged), so it lives in the recorded perf tier.

Every case is built on a device the caller names: the IR needs none, the clock needs Metal.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import lucid
import lucid.nn.functional as F
from lucid.test.unit.compile._helpers import COMPILE_DEVICE

Shape = tuple[int, ...]


def ramp(
    shape: Shape, modulo: int, dtype: lucid.dtype, device: str = COMPILE_DEVICE
) -> lucid.Tensor:
    """``0 .. modulo - 1`` repeating, made on ``device`` (a timed operand has up to 1M elements)."""
    return (lucid.arange(math.prod(shape)) % modulo).reshape(*shape).to(dtype).to(device)


def rotation(shape: Shape, dim: int, extent: int, device: str = COMPILE_DEVICE) -> lucid.Tensor:
    """Positions along ``dim`` that are distinct in every slice, so an overwrite has one answer."""
    dim %= len(shape)
    stride = math.prod(shape[dim + 1 :])
    flat = lucid.arange(math.prod(shape))
    return ((flat // stride + 1) % extent).reshape(*shape).to(device)


@dataclass(frozen=True)
class Side:
    """One compiled call: ``fn`` on ``args``."""

    fn: Callable[..., lucid.Tensor]
    args: tuple[lucid.Tensor, ...]


@dataclass(frozen=True)
class IndexCase:
    #: the recorded op the rank pass rewrites (the node both sides must hold)
    op: str
    wide: Side
    twin: Side


def _embed_sum(w: lucid.Tensor, i: lucid.Tensor) -> lucid.Tensor:
    return F.embedding(i, w).reshape(-1).sum()


def _embedding_case(
    table: Shape, wide: Shape, twin: Shape, dtype: lucid.dtype, dev: str
) -> IndexCase:
    w = ramp(table, 1000, dtype, dev)

    def side(shape: Shape) -> Side:
        return Side(_embed_sum, (w, ramp(shape, table[0], lucid.int64, dev)))

    return IndexCase("embedding", side(wide), side(twin))


_E_TABLE: Shape = (1000, 32)

# rank 5 and the layout the rule merges it to: ``(2048, 4, 4, 4, 8)`` around each axis
_G5: Shape = (2048, 4, 4, 4, 8)
_G_TWINS: dict[str, tuple[int, int, Shape]] = {
    # name: (dim of the rank-5 op, dim of the twin, twin shape)
    "last": (-1, -1, (131072, 8)),
    "mid": (2, 1, (8192, 4, 32)),
    "first": (0, 0, (2048, 512)),
}


def _gather_case(where: str, dev: str) -> IndexCase:
    dim5, dim_twin, twin = _G_TWINS[where]

    def side(shape: Shape, dim: int) -> Side:
        data, index = ramp(shape, 1000, lucid.float32, dev), ramp(
            shape, shape[dim], lucid.int64, dev
        )
        return Side(lambda t, i: lucid.gather(t, dim, i).reshape(-1).sum(), (data, index))

    return IndexCase("gather", side(_G5, dim5), side(twin, dim_twin))


def _scatter_case(where: str, dev: str) -> IndexCase:
    dim5, dim_twin, twin = _G_TWINS[where]

    def side(shape: Shape, dim: int) -> Side:
        base, src = ramp(shape, 1000, lucid.float32, dev), ramp(shape, 7, lucid.float32, dev)
        index = rotation(shape, dim, shape[dim], dev)
        fn: Callable[..., lucid.Tensor] = lambda t, i, s: (
            t.scatter(dim, i, s, reduce="add").reshape(-1).sum()
        )
        return Side(fn, (base, index, src))

    return IndexCase("scatter_add", side(_G5, dim5), side(twin, dim_twin))


def _search_case(dev: str) -> IndexCase:
    sequence = ramp((256,), 256, lucid.float32, dev)

    def side(shape: Shape) -> Side:
        values = ramp(shape, 251, lucid.float32, dev)
        return Side(lambda s, v: lucid.searchsorted(s, v).reshape(-1).sum(), (sequence, values))

    # lead 128: the unfixed binary search is ~100 ms, the flat one under 1
    return IndexCase("searchsorted", side((128, 2, 2, 2, 16)), side((16384,)))


CASES: dict[str, Callable[[str], IndexCase]] = {
    # float32, a rank-5 index (the answer is rank 6, lead 4096): 40 ms against 0.8 unfixed
    "embedding_rank_5_index": lambda dev: _embedding_case(
        _E_TABLE, (2048, 2, 2, 2, 2), (32768,), lucid.float32, dev
    ),
    # int32, a rank-4 index (the answer is rank 5 by itself): the LCD-997 lane merge made it
    # 2.1x slower than the native graph, which is itself 4-5x the flat form.  The twin holds the
    # same 65536 indices: with 16384 (a quarter of them) the ratio read 1.5x to 2.0x on an M1 Pro
    # and 2.04x on the hosted runner for a graph that is the twin's (LCD-1039).
    "embedding_int32_rank_4_index": lambda dev: _embedding_case(
        _E_TABLE, (512, 4, 4, 8), (65536,), lucid.int32, dev
    ),
    "searchsorted_one_sequence": _search_case,
}
for _where in sorted(_G_TWINS):
    CASES[f"gather_{_where}"] = lambda dev, w=_where: _gather_case(w, dev)
    CASES[f"scatter_add_{_where}"] = lambda dev, w=_where: _scatter_case(w, dev)
