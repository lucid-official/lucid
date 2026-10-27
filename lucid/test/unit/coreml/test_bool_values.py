"""A bool tensor moves and converts in a package as it does in eager (LCD-388).

Two defects of one kind — a bool value meeting a MIL operation that does
not take it as given:

* MIL's gathers and scatters take no bool data. ``t[0] = True`` inside a
  forward failed when Core ML compiled the package ("Param 'updates' has
  incorrect type for operator 'ios17.scatter_along_axis'"), and so did a
  ``gather``, ``index_select``, ``unbind``, ``diagonal`` or ``sort`` of a
  mask (MIL's ``argsort`` refuses bool too).
* Core ML folds a ``cast`` to or from bool wrongly when its operand is
  built from constants: ``ones(5, 5).triu(1).bool()`` — a causal mask — was
  all false in a float16 package, and a float32 package lost it once it
  was cast on to int32 for the scatter.  The package loaded and answered.

Each case is exported at both body precisions, run on ``CPU_ONLY`` and
compared with eager.
"""

from collections.abc import Callable

import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)

_IDX = [[0, 2, 1], [1, 1, 0], [2, 0, 2]]


def _causal() -> lucid.Tensor:
    """A mask made from constants alone, as attention builds one."""
    return lucid.ones(5, 5).triu(1).bool()


def _set_row(x: lucid.Tensor) -> lucid.Tensor:
    t = _causal()
    t[0] = True
    return x.masked_fill(t, 0.0)


def _set_row_from_input(x: lucid.Tensor) -> lucid.Tensor:
    t = _causal()
    t[2] = x[2] > 0
    return x.masked_fill(t, 0.0)


def _set_into_input_mask(x: lucid.Tensor) -> lucid.Tensor:
    t = x > 0
    t[1] = True
    return x.masked_fill(t, 0.0)


def _scatter(x: lucid.Tensor) -> lucid.Tensor:
    t = (x > 0).scatter(1, lucid.tensor(_IDX), x[:3, :3] < 0)
    return x.masked_fill(t, 0.0)


def _scatter_add(x: lucid.Tensor) -> lucid.Tensor:
    t = (x > 0).scatter_add(1, lucid.tensor(_IDX), x[:3, :3] < 0)
    return x.masked_fill(t, 0.0)


def _gather(x: lucid.Tensor) -> lucid.Tensor:
    return x[:, :3].masked_fill((x > 0).gather(1, lucid.tensor(_IDX + _IDX[:2])), 0.0)


def _index_select(x: lucid.Tensor) -> lucid.Tensor:
    return x[:2].masked_fill((x > 0).index_select(0, lucid.tensor([2, 0])), 0.0)


def _unbind(x: lucid.Tensor) -> lucid.Tensor:
    rows = lucid.unbind(x > 0, 0)
    return lucid.stack([x[0].masked_fill(r, 0.0) for r in rows], 0)


def _diagonal(x: lucid.Tensor) -> lucid.Tensor:
    return x[0].masked_fill((x > 0).diagonal(0, 0, 1), 0.0)


def _sort(x: lucid.Tensor) -> lucid.Tensor:
    return x.masked_fill(lucid.sort(x > 0, dim=1).values.bool(), 0.0)


def _argsort(x: lucid.Tensor) -> lucid.Tensor:
    # One true and one false per row: no ties, so the order is unique.
    sign = x[:, 0] > 0
    pair = lucid.stack([sign, sign.logical_not()], dim=1)
    return x[:, :2].gather(1, lucid.argsort(pair, dim=1))


def _causal_mask(x: lucid.Tensor) -> lucid.Tensor:
    return x.masked_fill(_causal(), 0.0)


def _mask_as_number(x: lucid.Tensor) -> lucid.Tensor:
    t = _causal()
    t[4] = True
    return x * t.to(lucid.float32) + t.to(lucid.int32).to(lucid.float32)


def _any_of_constant(x: lucid.Tensor) -> lucid.Tensor:
    return x * _causal().any().to(lucid.float32)


CASES: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "set_row": _set_row,
    "set_row_from_input": _set_row_from_input,
    "set_into_input_mask": _set_into_input_mask,
    "scatter": _scatter,
    "scatter_add": _scatter_add,
    "gather": _gather,
    "index_select": _index_select,
    "unbind": _unbind,
    "diagonal": _diagonal,
    "sort": _sort,
    "argsort": _argsort,
    "causal_mask": _causal_mask,
    "mask_as_number": _mask_as_number,
    "any_of_constant": _any_of_constant,
}


class _Fn(nn.Module):
    def __init__(self, fn: Callable[[lucid.Tensor], lucid.Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn(x)


@pytest.mark.parametrize(
    "precision", [cml.Precision.FLOAT32, cml.Precision.FLOAT16], ids=["fp32", "fp16"]
)
@pytest.mark.parametrize("case", sorted(CASES))
def test_a_bool_value_matches_eager(case: str, precision: cml.Precision, tmp_path: object) -> None:
    lucid.manual_seed(0)
    x = lucid.randn(5, 5)
    model = _Fn(CASES[case]).eval()
    exported = cml.export(
        model,
        x,
        str(tmp_path / "bools.mlpackage"),
        compute_units=cml.ComputeUnits.CPU_ONLY,
        precision=precision,
    )
    try:
        got, want = exported.predict(x), model(x)
    finally:
        exported.close()
    assert tuple(got.shape) == tuple(want.shape)
    # Every masked position is exactly 0 in both; the rest differ by the
    # body's rounding only — a wrong mask is off by a whole input value.
    tol = 2e-3 if precision == cml.Precision.FLOAT16 else 1e-6
    assert float((got - want).abs().max().item()) <= tol
