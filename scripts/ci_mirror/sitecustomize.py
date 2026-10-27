"""Hold this interpreter to the packages CI's gate installs.

``scripts/ci_local.sh`` puts this directory on ``PYTHONPATH``, so every
Python process the gate starts — pytest, the audit's children, the zoo
sweep's — imports it first.  A developer's venv carries far more than the
hosted runner: the reference framework, the model-zoo oracles, scipy.  A
test that finds one of them runs, or takes another path, where on CI it
skips, so a local pass would say little about the run that counts.

Everything installed that CI would not install is made absent the way the
import system itself spells it: ``sys.modules[name] = None``.  ``import``
then raises ``ModuleNotFoundError`` and ``importlib.util.find_spec``
returns ``None`` — both exactly as on a runner that never had it.

What CI installs is read, not listed: ``pyproject.toml``'s dependencies and
``test`` extra, the tools the phase-gate job installs beside them
(``_CI_TOOLS``), and everything those require, per the installed metadata.
"""

import importlib.metadata
import re
import sys
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

_ROOT = Path(__file__).resolve().parents[2]

#: What ``.github/workflows/ci.yml``'s phase-gate job installs besides the
#: package and its ``test`` extra — keep the two in step.
_CI_TOOLS = ("pip", "setuptools", "wheel", "cmake", "ninja", "pybind11", "clang-format")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _wanted(requirement: Requirement, extras: tuple[str, ...]) -> bool:
    marker = requirement.marker
    return marker is None or any(marker.evaluate({"extra": e}) for e in extras)


def _installed_by_ci() -> set[str]:
    """Normalised names of every distribution CI's install would bring in."""
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"]
    roots = [
        *project.get("dependencies", ()),
        *project.get("optional-dependencies", {}).get("test", ()),
        *_CI_TOOLS,
    ]
    todo = [r for r in map(Requirement, roots) if _wanted(r, ("",))]
    found = {_norm(project["name"])}
    while todo:
        requirement = todo.pop()
        name = _norm(requirement.name)
        if name in found:
            continue
        found.add(name)
        try:
            requires = importlib.metadata.distribution(requirement.name).requires
        except importlib.metadata.PackageNotFoundError:
            continue
        extras = ("", *requirement.extras)
        todo.extend(r for r in map(Requirement, requires or ()) if _wanted(r, extras))
    return found


def _hide_the_rest() -> None:
    allowed = _installed_by_ci()
    for module, dists in importlib.metadata.packages_distributions().items():
        if module in sys.modules or module in sys.stdlib_module_names:
            continue
        if not any(_norm(dist) in allowed for dist in dists):
            sys.modules[module] = None  # type: ignore[assignment]


_hide_the_rest()
