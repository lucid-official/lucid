"""Every op-matrix case compiled with ``dynamic=True``, called at four batches.

``test_dynamic_batch`` checks a handful of models.  This runs each case of
:mod:`._op_matrix` that has a batch axis through ``lucid.compile(...,
dynamic=True)`` at the traced batch and at three others, against eager at
each — whatever the compile decided: one symbolic executable, or a static one
per shape after the gate or an emitter declined.

Its first run found the symbolic path answering for every batch with the
traced one baked in: ``std(x, dim=0)`` scaled by the traced ``n/(n-1)``;
``permute(1, 0)``, ``diagonal`` and ``batch_norm_eval`` returning the traced
batch's shape; ``split`` along the batch returning the traced number of
pieces; ``roll`` over it, wrong values.  And it aborted the process —
uncatchably — in MPSGraph for ``.mT.contiguous()``, ``conv1d``, the
transposed convolutions, nearest resize, and a gather whose index is
shorter than its input (on the static path too).  The gate now refuses a
graph that moves, reduces, splits, rolls or gathers along the batch axis,
and those emitters decline a symbolic batch they cannot carry.
"""

import pytest

import lucid
import lucid.nn as nn
from lucid.test.unit.compile import _op_matrix as M
from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


# Compiled forwards under grad mode run eagerly (LCD-582); these test them.
pytestmark = [
    pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable"),
    pytest.mark.usefixtures("inference"),
]


class _Apply(nn.Module):
    def __init__(self, fn: object) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> object:
        leaves = M._leaves(self.fn(x))  # type: ignore[operator]
        return leaves[0] if len(leaves) == 1 else tuple(leaves)


def _names() -> list[str]:
    return [
        c.name
        for c in M.CASES
        if c.shape
        and not c.random
        and "f32" in c.dtypes
        and M.refusal(c.name, "f32", M.DEV) is None
    ]


@pytest.mark.parametrize("name", _names())
def test_a_dynamic_compile_answers_every_batch_as_eager(name: str) -> None:
    case = M.CASE_BY_NAME[name]
    model = _Apply(case.fn).eval()
    compiled = lucid.compile(model, dynamic=True)
    first = case.shape[0]
    for i, batch in enumerate((first, first + 2, first + 5, first + 2)):
        x = M.make_input(case.kind, "f32", (batch, *case.shape[1:]), 10 + i)
        try:
            want = model(x)
        except Exception:  # noqa: BLE001 — the recipe itself needs this batch
            return
        why = M._compare(compiled(x), want, False)
        assert not why, f"batch {batch}: {why}"
