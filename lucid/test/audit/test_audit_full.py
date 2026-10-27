"""The whole audit, as collectable pytest cases — one per (symbol, axis).

The sweep is 9,134 atomic verdicts, but until this file existed they only
came into being when someone ran the CLI; ``pytest`` collected 37 tests
from this package and none of them was an actual check of the framework.
A suite that reports only through its own front end is not integrated
with anything.

So every pair is a real test case with a readable id::

    pytest -m audit                              # all 9,134
    pytest -m audit -k 'grad and conv'           # the ones you care about
    pytest -m audit -k 'nonfinite'               # one axis
    pytest lucid/test/audit -m audit --tb=line   # the whole sweep, terse

Opt-in on purpose.  ``addopts`` excludes the ``audit`` marker so a plain
``pytest`` stays the fast suite; the sweep is minutes, not seconds, and
its value is in being run deliberately.

**pytest's own summary is the coverage report.**  A pair the harness
cannot build inputs for calls ``pytest.skip`` with the reason, and an op
that refuses by design skips too — so ``N passed, M skipped`` is exactly
the reach-versus-depth split the CLI prints, without a second mechanism
to keep in agreement.
"""

from pathlib import Path

import pytest

from lucid.test.audit import _axes, _probe, _surface
from lucid.test.audit._result import Baseline, Status

_KNOWN = Path(__file__).with_name("known.json")


def _cases() -> list[tuple[str, object, object]]:
    """Every (axis, symbol) pair an axis can express, with a stable id."""
    symbols = _surface.enumerate_surface()
    out = []
    for axis in _axes.ALL_AXES:
        for symbol in symbols:
            if axis.applies(symbol):
                out.append((f"{axis.name}-{symbol.qualname}", axis, symbol))
    return out


_CASES = _cases()
_IDS = [case[0] for case in _CASES]


@pytest.fixture(scope="session")
def audit_context() -> _axes.Context:
    """One context for the whole session — probing Metal once is enough."""
    return _axes.Context(quick=True, metal=_probe.metal_available())


@pytest.fixture(scope="session")
def audit_baseline() -> Baseline:
    return Baseline.load(_KNOWN)


@pytest.mark.audit
@pytest.mark.parametrize("_id,axis,symbol", _CASES, ids=_IDS)
def test_audit(
    _id: str,
    axis: _axes.Axis,
    symbol: _surface.Symbol,
    audit_context: _axes.Context,
    audit_baseline: Baseline,
) -> None:
    """One question, asked of one symbol.

    ``FAIL`` is the only outcome that fails the test.  Everything else is
    either a pass or a skip carrying the reason, because a survey that
    turns "I could not build inputs for this" into a failure teaches its
    reader to ignore it.
    """
    finding = audit_baseline.apply(axis.run(symbol, audit_context))

    if finding.status is Status.FAIL:
        pytest.fail(f"{axis.name} · {symbol.qualname}\n  {finding.detail}", pytrace=False)
    if finding.status is Status.ERROR:
        pytest.fail(f"the probe itself broke: {finding.detail}", pytrace=False)
    if finding.status in (Status.SKIP, Status.UNSUPPORTED):
        pytest.skip(f"{finding.status.value}: {finding.detail}")
    if finding.status is Status.VACUOUS:
        pytest.skip(f"vacuous: {finding.detail}")


def test_the_parametrisation_is_the_size_it_claims() -> None:
    """Guard the instrument.

    If a refactor quietly narrows an axis, this file would still collect
    and still pass — with far fewer cases and no sign of it.  The floor is
    deliberately loose (the exact count moves as the framework grows) but
    tight enough to catch a subsystem dropping out.
    """
    assert len(_CASES) > 8000, f"only {len(_CASES)} audit cases — an axis has narrowed"
    axes_present = {case[1].name for case in _CASES}
    assert axes_present == {a.name for a in _axes.ALL_AXES}, "an axis matched nothing"
