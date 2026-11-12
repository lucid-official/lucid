#!/usr/bin/env python3
"""The Griffe version has one home, and every install reads it (LCD-970).

``web/public/api-data`` is Griffe's output, committed, and rebuilt by a nightly
runner that installed ``griffe`` unpinned: it got 2.4.1 while the dev venv had
2.1.0, so the committed tree could change with whichever of them ran last.  The
pin is now ``[dependency-groups].apidata`` in ``pyproject.toml``; a workflow
installs it with ``pip install --group apidata``.

A pin only holds while nothing installs the package another way, so this fails
on the two ways it comes undone: the group stops being one exact requirement,
or a workflow / CI script runs a ``pip install`` that names Griffe without the
group.

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
_INSTALL = re.compile(r"\bpip\s+install\b")


def _install_sources() -> list[Path]:
    return sorted((_ROOT / ".github" / "workflows").glob("*.yml")) + sorted(
        (_ROOT / "scripts").glob("*.sh")
    )


def _install_lines(path: Path) -> list[str]:
    """The non-comment lines of ``path`` that run a ``pip install``."""
    lines = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line.startswith("#") and _INSTALL.search(line):
            lines.append(line)
    return lines


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


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


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
