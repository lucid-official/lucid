"""The fatal smoke mutant runs outside the supervising interpreter."""

import subprocess

import pytest

from lucid.test.audit import _mutants


def test_every_axis_has_a_negative_control() -> None:
    assert _mutants.unproven_axes() == []


@pytest.mark.parametrize(("exit_code", "caught"), [(73, True), (0, False), (1, False)])
def test_only_the_deliberate_child_exit_proves_smoke(
    monkeypatch: pytest.MonkeyPatch, exit_code: int, caught: bool
) -> None:
    mutant = next(m for m in _mutants.MUTANTS if m.isolated_exit is not None)
    monkeypatch.setattr(_mutants, "MUTANTS", (mutant,))
    monkeypatch.setattr(
        _mutants.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess([], exit_code),
    )
    (result,) = _mutants.verify()
    assert result.caught is caught
