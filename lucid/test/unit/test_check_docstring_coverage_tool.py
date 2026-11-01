"""``tools/check_docstring_coverage.py`` reads a class page the way the docs build does.

The docs site lists, on a public class's page, every public member it has,
those it inherits from a private base included, and renders the docstring of
the function that *defines* each.  An override with no docstring renders an
empty row even when the method it overrides is documented, and a sentence at
column 0 inside a ``Parameters`` block is read as a parameter named after its
first word.  Neither shows in a check of the public symbols alone (LCD-807),
so the tool follows each exported class through its bases.

Each case runs the tool on a scratch checkout: a copy of the tool and a small
package under ``lucid/``.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_TOOL = _ROOT / "tools" / "check_docstring_coverage.py"

_LONG = "x" * 700
_BASE = '''
class _Base:
    """A private base the public classes build on, documented well enough to pass."""

    def run(self) -> None:
        """Run the thing this base is for, and say what it did to the caller."""

    @property
    def size(self) -> int:
        """How many items the base holds right now, as the page should say."""
        return 0
'''


def _checkout(tmp_path: Path, files: dict[str, str]) -> Path:
    (tmp_path / "tools").mkdir()
    shutil.copy(_TOOL, tmp_path / "tools" / _TOOL.name)
    package = tmp_path / "lucid" / "pkg"
    package.mkdir(parents=True)
    for name, text in files.items():
        (package / name).write_text(text)
    return tmp_path


def _run(root: Path) -> tuple[int, str]:
    done = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / _TOOL.name),
            "--list",
            "--member-packages",
            "lucid/pkg",
        ],
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )
    return done.returncode, done.stderr


def _package(body: str, exports: str = "Public") -> dict[str, str]:
    """A package whose ``__init__`` docstring is long enough and exports ``exports`` from ``_m``."""
    return {
        "__init__.py": f'"""{_LONG}"""\n\nfrom lucid.pkg._m import {exports}\n\n__all__ = ["{exports}"]\n',
        "_m.py": _BASE + body,
    }


class TestMembers:
    def test_documented_members_pass(self, tmp_path: Path) -> None:
        body = '''
class Public(_Base):
    """The exported class, whose members are all documented, so no rule of the tool flags it."""

    def __init__(self) -> None:
        """Build the public class from nothing at all, which is all it needs."""
'''
        code, err = _run(_checkout(tmp_path, _package(body)))
        assert code == 0, err

    def test_override_without_docstring_is_flagged_though_the_base_has_one(
        self, tmp_path: Path
    ) -> None:
        body = '''
class Public(_Base):
    """The exported class, whose override of ``run`` says nothing at all."""

    def __init__(self) -> None:
        """Build the public class from nothing at all, which is all it needs."""

    def run(self) -> None:
        return None
'''
        code, err = _run(_checkout(tmp_path, _package(body)))
        assert code == 1
        assert "`Public.run` has no docstring" in err

    def test_member_defined_on_a_private_base_is_flagged_at_its_owner(self, tmp_path: Path) -> None:
        base = '''
class _Bare:
    """A private base that leaves a public method undocumented, for the subclass to inherit."""

    def reset(self) -> None:
        return None
'''
        body = base + '''
class Public(_Bare):
    """The exported class, which inherits ``reset`` without a docstring."""

    def __init__(self) -> None:
        """Build the public class from nothing at all, which is all it needs."""
'''
        code, err = _run(_checkout(tmp_path, _package(body)))
        assert code == 1
        assert "`_Bare.reset` has no docstring (reached from public `Public`)" in err

    def test_property_without_docstring_is_flagged(self, tmp_path: Path) -> None:
        body = '''
class Public(_Base):
    """The exported class, whose ``size`` property says nothing at all."""

    def __init__(self) -> None:
        """Build the public class from nothing at all, which is all it needs."""

    @property
    def size(self) -> int:
        return 1
'''
        code, err = _run(_checkout(tmp_path, _package(body)))
        assert code == 1
        assert "`Public.size`" in err

    def test_classes_a_package_does_not_export_are_not_held_to_it(self, tmp_path: Path) -> None:
        body = '''
class Public(_Base):
    """The exported class, whose members are all documented, so no rule of the tool flags it."""

    def __init__(self) -> None:
        """Build the public class from nothing at all, which is all it needs."""


class Helper:
    def undocumented(self) -> None:
        return None
'''
        code, err = _run(_checkout(tmp_path, _package(body)))
        assert code == 0, err

    def test_thin_package_docstring_is_flagged(self, tmp_path: Path) -> None:
        files = _package('''
class Public(_Base):
    """The exported class, whose members are all documented, so no rule of the tool flags it."""

    def __init__(self) -> None:
        """Build the public class from nothing at all, which is all it needs."""
''')
        files["__init__.py"] = files["__init__.py"].replace(_LONG, "Just a line.")
        code, err = _run(_checkout(tmp_path, files))
        assert code == 1
        assert "package `lucid.pkg` has a thin docstring" in err


class TestParametersBlock:
    @pytest.mark.parametrize(
        ("block", "flagged"),
        [
            ("    x : int\n        The thing.\n", False),
            ("    x:\n        The thing.\n", False),
            ("    x : int\n        The thing.\n\n    The remaining ones are x's.\n", True),
        ],
    )
    def test_prose_at_column_zero_is_read_as_a_parameter(
        self, tmp_path: Path, block: str, flagged: bool
    ) -> None:
        body = f'''
class Public(_Base):
    """The exported class, documented well enough to pass the member rule.

    Parameters
    ----------
{block}    """

    def __init__(self) -> None:
        """Build the public class from nothing at all, which is all it needs."""
'''
        code, err = _run(_checkout(tmp_path, _package(body)))
        assert (code == 1) is flagged, err
        if flagged:
            assert "prose inside its Parameters block" in err


def test_the_peft_package_passes_its_own_rule() -> None:
    """``lucid.peft`` is in ``MEMBER_DOC_PACKAGES``; a member added without a docstring fails here."""
    done = subprocess.run(
        [sys.executable, str(_TOOL), "--list", "--member-packages", "lucid/peft"],
        capture_output=True,
        text=True,
        cwd=_ROOT,
        check=False,
    )
    assert done.returncode == 0, done.stderr
