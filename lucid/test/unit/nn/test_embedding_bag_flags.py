"""``embedding_bag`` honours ``max_norm``, ``norm_type``, ``scale_grad_by_freq`` and ``sparse`` (LCD-235).

All four were accepted and dropped: ``max_norm=1`` pooled the raw rows, and
``scale_grad_by_freq`` gave every row its unscaled gradient.  ``max_norm``
now renormalises the rows looked up, through the one write ``F.embedding``
makes (``sparse._renorm_rows``); ``scale_grad_by_freq`` routes the bag
through ``F.embedding``, which owns the scaling at its lookup node; and
``sparse=True`` is refused, as ``F.embedding`` refuses it.  The modules pass
every flag on.

The reference side is the definition — ``embedding`` with the same flags,
then the bag reduction — rather than the reference's ``embedding_bag``,
whose CPU kernel divides some rows by the count of a different index: for
indices ``[0, 0, 1]`` it gives row 1 half its gradient though row 1 occurs
once.  Its ``embedding_bag`` is compared directly where no scaling applies.
"""

from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

_TABLE = [[3.0, 4.0], [1.0, 0.0], [0.5, 0.5], [6.0, 8.0], [0.1, 0.2]]
_IDX = [0, 0, 1, 3, 0, 4, 4]
_OFFSETS = [0, 2, 2]  # bags [0, 0], [], [1, 3, 0, 4, 4]
_UPSTREAM = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def _close(a: list[list[float]], b: list[list[float]]) -> bool:
    flat = [(p, q) for ra, rb in zip(a, b, strict=True) for p, q in zip(ra, rb, strict=True)]
    return all(abs(p - q) < 1e-5 for p, q in flat)


def _lucid_bag(
    device: str, mode: str, max_norm: float | None, scale: bool, pad: int | None
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    w = lucid.tensor(_TABLE, device=device, requires_grad=True)
    out = F.embedding_bag(
        lucid.tensor(_IDX, device=device),
        w,
        lucid.tensor(_OFFSETS, device=device),
        max_norm=max_norm,
        scale_grad_by_freq=scale,
        mode=mode,
        padding_idx=pad,
    )
    (out * lucid.tensor(_UPSTREAM, device=device)).sum().backward()
    assert w.is_leaf and w.grad is not None
    return out.tolist(), w.grad.tolist(), w.detach().tolist()


def _ref_bag(
    ref: ModuleType, mode: str, max_norm: float | None, scale: bool, pad: int | None
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    rf = ref.nn.functional
    w = ref.tensor(_TABLE, requires_grad=True)
    idx = ref.tensor(_IDX)
    rows = rf.embedding(idx, w, pad, max_norm, 2.0, scale)
    ends = [*_OFFSETS[1:], len(_IDX)]
    bags = []
    for start, end in zip(_OFFSETS, ends, strict=True):
        kept = [i for i in range(start, end) if _IDX[i] != pad]
        if not kept:
            bags.append(ref.zeros(2))
        elif mode == "sum":
            bags.append(rows[kept].sum(0))
        else:
            bags.append(rows[kept].mean(0))
    out = ref.stack(bags)
    if not scale:
        direct = rf.embedding_bag(idx, w.detach(), ref.tensor(_OFFSETS), mode=mode, padding_idx=pad)
        assert _close(direct.tolist(), out.tolist())
    (out * ref.tensor(_UPSTREAM)).sum().backward()
    return out.tolist(), w.grad.tolist(), w.detach().tolist()


@pytest.mark.parity
@pytest.mark.parametrize("pad", [None, 4])
@pytest.mark.parametrize("scale", [False, True])
@pytest.mark.parametrize("max_norm", [None, 1.0])
@pytest.mark.parametrize("mode", ["sum", "mean"])
def test_matches_the_reference(
    mode: str,
    max_norm: float | None,
    scale: bool,
    pad: int | None,
    device: str,
    ref: ModuleType,
) -> None:
    got = _lucid_bag(device, mode, max_norm, scale, pad)
    want = _ref_bag(ref, mode, max_norm, scale, pad)
    for name, g, w in zip(("output", "grad", "table"), got, want, strict=True):
        assert _close(g, w), f"{name}: {g} != {w}"


def test_max_norm_rescales_only_the_rows_looked_up(device: str) -> None:
    w = lucid.tensor(_TABLE, device=device)
    out = F.embedding_bag(
        lucid.tensor([3, 2], device=device),
        w,
        lucid.tensor([0], device=device),
        max_norm=1.0,
        mode="sum",
    )
    # Row 3 (norm 10) is cut to norm 1; row 2 (norm 0.71) is under the cap.
    assert _close(out.tolist(), [[0.6 + 0.5, 0.8 + 0.5]])
    assert _close(w.tolist(), [_TABLE[0], _TABLE[1], _TABLE[2], [0.6, 0.8], _TABLE[4]])


def test_norm_type_picks_the_norm(device: str) -> None:
    w = lucid.tensor(_TABLE, device=device)
    F.embedding_bag(
        lucid.tensor([0], device=device),
        w,
        lucid.tensor([0], device=device),
        max_norm=1.4,
        norm_type=1.0,
    )
    # L1 norm of [3, 4] is 7.
    assert _close(w[0:1].tolist(), [[0.6, 0.8]])


def test_scaled_gradient_divides_by_each_index_count(device: str) -> None:
    w = lucid.ones(4, 1, device=device, requires_grad=True)
    F.embedding_bag(
        lucid.tensor([0, 0, 1], device=device),
        w,
        lucid.tensor([0], device=device),
        mode="sum",
        scale_grad_by_freq=True,
    ).sum().backward()
    assert w.grad is not None
    assert _close(w.grad.tolist(), [[1.0], [1.0], [0.0], [0.0]])


def test_scaling_with_max_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="scale_grad_by_freq"):
        F.embedding_bag(
            lucid.tensor([0]),
            lucid.ones(2, 2),
            lucid.tensor([0]),
            mode="max",
            scale_grad_by_freq=True,
        )


@pytest.mark.parametrize(
    "call",
    [
        lambda: F.embedding_bag(
            lucid.tensor([0]), lucid.ones(2, 2), lucid.tensor([0]), sparse=True
        ),
        lambda: nn.EmbeddingBag(2, 2, sparse=True)(lucid.tensor([[0]])),
        lambda: nn.Embedding(2, 2, sparse=True)(lucid.tensor([0])),
    ],
    ids=["F.embedding_bag", "nn.EmbeddingBag", "nn.Embedding"],
)
def test_sparse_is_refused(call: Callable[[], object]) -> None:
    with pytest.raises(NotImplementedError, match="sparse"):
        call()


@pytest.mark.parametrize("mode", ["sum", "mean"])
def test_module_passes_every_flag_on(mode: str, device: str) -> None:
    bag = nn.EmbeddingBag(5, 2, max_norm=1.0, scale_grad_by_freq=True, mode=mode, device=device)
    with lucid.no_grad():
        bag.weight.copy_(lucid.tensor(_TABLE, device=device))
    out = bag(lucid.tensor(_IDX, device=device), lucid.tensor(_OFFSETS, device=device))
    (out * lucid.tensor(_UPSTREAM, device=device)).sum().backward()
    want = _lucid_bag(device, mode, 1.0, True, None)
    assert bag.weight.grad is not None
    assert _close(out.tolist(), want[0])
    assert _close(bag.weight.grad.tolist(), want[1])
    assert _close(bag.weight.detach().tolist(), want[2])


def test_module_refuses_an_unknown_mode() -> None:
    # The module used to map an unknown mode to "mean" before the call.
    with pytest.raises(ValueError, match="mode"):
        nn.EmbeddingBag(3, 2, mode="sun")(lucid.tensor([[0, 1]]))
