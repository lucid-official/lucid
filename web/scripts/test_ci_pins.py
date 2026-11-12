#!/usr/bin/env python3
"""The Griffe and ruff versions have one home each, and installs read it (LCD-970).

``web/public/api-data`` is Griffe's output, committed, and rebuilt by a nightly
runner that installed ``griffe`` unpinned: it got 2.4.1 while the dev venv had
2.1.0, so the committed tree could change with whichever of them ran last.  The
pin is now ``[dependency-groups].apidata`` in ``pyproject.toml``; a workflow
installs it with ``pip install --group apidata``.

A pin only holds while nothing installs the package another way, so this fails
on the two ways it comes undone: the group stops being one exact requirement,
or a workflow / CI script runs a ``pip install`` that names Griffe without the
group.

The Docs workflow also lints ``web/scripts`` (the quality gate scans ``lucid/``
only).  It installs ruff from the ``lint`` group, which must stay the same
version as the ``quality`` group's ruff -- two lines, one version -- and must
stay what the workflow installs.

It installs the ``apidata`` group too (LCD-976): the builder's cost guards in
``test_build_api_data_perf.py`` need Griffe, and without it they printed ``skip``
there and counted as passed.  Two more ways to come undone, both failed here:
the Docs workflow's install step goes, or ``LUCID_REQUIRE_GRIFFE`` does -- the
variable that makes those guards fail, not skip, when Griffe is missing.

Standalone like the other ``web/scripts/test_*.py`` (stdlib only): ``python3
web/scripts/test_ci_pins.py``, also pytest-collectable.
"""

import re
import sys
import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_GROUP = "apidata"
_EXACT = re.compile(r"^griffe==\d+(\.\d+)+$")
_RUFF = re.compile(r"^ruff==\d+(\.\d+)+$")
_INSTALL = re.compile(r"\bpip\s+install\b")


def _install_sources() -> list[Path]:
    return sorted((_ROOT / ".github" / "workflows").glob("*.yml")) + sorted(
        (_ROOT / "scripts").glob("*.sh")
    )


def _code_lines(path: Path) -> list[str]:
    """The stripped lines of ``path`` that are not whole-line comments."""
    stripped = (raw.strip() for raw in path.read_text().splitlines())
    return [line for line in stripped if not line.startswith("#")]


def _install_lines(path: Path) -> list[str]:
    """The non-comment lines of ``path`` that run a ``pip install``."""
    return [line for line in _code_lines(path) if _INSTALL.search(line)]


def _groups() -> dict[str, list[str]]:
    return tomllib.loads((_ROOT / "pyproject.toml").read_text())["dependency-groups"]


def test_the_group_is_one_exact_griffe_requirement() -> None:
    group = _groups()[_GROUP]
    assert len(group) == 1 and _EXACT.match(group[0]), (
        f"[dependency-groups].{_GROUP} must be exactly one `griffe==X.Y.Z`, got {group!r}: "
        "an open range lets the runner and a laptop build the committed api-data "
        "with different Griffe versions"
    )


def test_no_workflow_installs_griffe_outside_the_group() -> None:
    offenders = [
        f"{path.relative_to(_ROOT)}: {line}"
        for path in _install_sources()
        for line in _install_lines(path)
        if "griffe" in line.lower() and f"--group {_GROUP}" not in line
    ]
    assert not offenders, (
        f"install Griffe with `pip install --group {_GROUP}` (the one pin): {offenders}"
    )


def test_the_refresh_workflow_installs_the_group() -> None:
    """Dropping the install step would also pass the test above."""
    lines = _install_lines(_ROOT / ".github" / "workflows" / "docs-api-data.yml")
    assert any(f"--group {_GROUP}" in line for line in lines), lines


def test_the_lint_group_is_the_quality_groups_ruff() -> None:
    groups = _groups()
    ruff = [r for r in groups["quality"] if r.startswith("ruff")]
    assert groups["lint"] == ruff and _RUFF.match(ruff[0]), (
        f"[dependency-groups].lint {groups['lint']!r} must be exactly the `quality` group's "
        f"pinned ruff {ruff!r}: the Docs lint and the ratchet gate must not run different ruffs"
    )


def test_the_docs_workflow_lints_with_the_lint_group() -> None:
    lines = _install_lines(_ROOT / ".github" / "workflows" / "docs.yml")
    assert any("--group lint" in line for line in lines), lines


def test_the_docs_workflow_installs_the_griffe_group() -> None:
    """The builder cost guards need it (LCD-976); a runner without Griffe skips them."""
    lines = _install_lines(_ROOT / ".github" / "workflows" / "docs.yml")
    assert any(f"--group {_GROUP}" in line for line in lines), lines


def test_the_docs_workflow_makes_a_missing_griffe_fail_the_tests() -> None:
    """Installing Griffe is not enough: a later edit that drops or breaks the install
    would put the guards back to skipping, so the step that runs them must also say
    Griffe is required (test_build_api_data_perf.py turns its skip into a failure)."""
    lines = _code_lines(_ROOT / ".github" / "workflows" / "docs.yml")
    runs = [n for n, line in enumerate(lines) if "scripts/run_tests.py" in line]
    assert len(runs) == 1, f"expected one run_tests.py step in docs.yml, found {len(runs)}"
    step_start = max(n for n in range(runs[0]) if lines[n].startswith("- "))
    step = lines[step_start : runs[0] + 1]
    assert 'LUCID_REQUIRE_GRIFFE: "1"' in step, step
    assert not any("--skip" in line for line in step), (
        f"the Docs run must not leave a test file out: {step}"
    )


_TESTS =[v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    failed = 0
    for t in _TESTS:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}\n        {exc!r}", file=sys.stderr)
    print(f"\n{len(_TESTS) - failed}/{len(_TESTS)} ci-pin tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
