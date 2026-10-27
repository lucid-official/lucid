"""Every training step costs exactly what ``step_cost.json`` says.

The gate the nightly wall-clock job could not be: a timing on a shared
runner measures the runner, so it was recorded and never enforced.  These
counts — storages allocated, freed and at their peak, forward ops by name,
host storages made during a GPU step, a compiled step's executables and
eager fallbacks — depend only on the code, so they are exact on any
machine, and any change is one somebody made.

Strict both ways.  A step that got costlier fails with what grew.  One that
got cheaper fails too, until the budget is rewritten with
``python -m lucid.test.unit.cost --update`` — otherwise the gain could be
lost again without a test noticing.
"""

import json
from pathlib import Path

import pytest

from lucid.test.unit.cost._workloads import WORKLOADS, measure

_BUDGET = json.loads(Path(__file__).with_name("step_cost.json").read_text())

_CASES = [(w, device) for w in WORKLOADS for device in w.devices]


def _describe(key: str, want: object, got: object) -> list[str]:
    if key == "forward_ops":
        assert isinstance(want, dict) and isinstance(got, dict)
        lines = []
        for op in sorted(set(want) | set(got)):
            a, b = want.get(op, 0), got.get(op, 0)
            if a != b:
                lines.append(f"  forward op {op!r}: {a} -> {b} ({b - a:+d})")
        return lines
    assert isinstance(want, int) and isinstance(got, int)
    return [f"  {key}: {want} -> {got} ({got - want:+d})"] if want != got else []


def test_the_budget_lists_every_workload() -> None:
    assert sorted(_BUDGET) == sorted(f"{w.name}/{d}" for w, d in _CASES)


@pytest.mark.parametrize(("workload", "device"), _CASES, ids=[f"{w.name}-{d}" for w, d in _CASES])
def test_a_step_costs_what_the_budget_says(workload: object, device: str) -> None:
    got = measure(workload, device)  # type: ignore[arg-type]
    want = _BUDGET[f"{workload.name}/{device}"]  # type: ignore[attr-defined]
    diff = [
        line
        for key in sorted(set(want) | set(got))
        for line in _describe(key, want.get(key, 0), got.get(key, 0))
    ]
    assert not diff, (
        "the step's cost changed:\n"
        + "\n".join(diff)
        + "\nIf that is the intended change, record it with "
        "`python -m lucid.test.unit.cost --update`."
    )
