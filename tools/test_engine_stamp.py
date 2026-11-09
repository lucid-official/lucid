"""The engine build stamp (LCD-881): what an engine binary was built from is a recorded digest.

mtimes cannot say it -- ``unison`` keeps them across machines and a ``cp -c`` of an ``.so``
is "now" -- so a freshness check that falls to them passes an engine nobody built from this
tree.  These pin the stamp as the authority, and the mtimes as a fallback only.
"""

import os
import time
from pathlib import Path

import pytest

from tools import agent_ws, engine_stamp
from tools.engine_stamp import EngineTree, Freshness

SO = Path("lucid/_C/engine.cpython-314-darwin.so")
PYPROJECT = '[project]\nname = "thing"\nversion = "1"\n\n[tool.lucid.mlx]\nversion = "0.1"\n'


def _set_times(path: Path, when: float) -> None:
    os.utime(path, (when, when))


def _tree(root: Path, *, built: bool = True) -> EngineTree:
    """A checkout with C++, the native inputs, and (``built``) an engine and its stamp."""
    for name, text in {
        "lucid/_C/version.h": "#define LUCID_ABI_VERSION 7\n",
        "lucid/_C/core/Thing.cpp": "int thing() { return 1; }\n",
        "lucid/_C/CMakeLists.txt": "project(thing)\n",
        "setup.py": "built = True\n",
        "tools/fetch_mlx.py": "pinned = True\n",
        "pyproject.toml": PYPROJECT,
    }.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
        _set_times(root / name, 1000.0)
    tree = EngineTree(root)
    if built:
        (root / SO).write_bytes(b"engine")
        _set_times(root / SO, 2000.0)
        assert tree.write()
    return tree


def _edit(root: Path, name: str, text: str, *, later_than: Path | None = None) -> None:
    (root / name).write_text(text)
    if later_than is not None:  # an mtime-only check would call the binary newer than this edit
        _set_times(root / name, (root / later_than).stat().st_mtime - 500.0)


def test_a_stamp_that_matches_reads_fresh_whatever_the_mtimes_say(tmp_path: Path) -> None:
    tree = _tree(tmp_path)
    assert tree.freshness() is Freshness.MATCH and tree.recorded() == tree.digest() != ""
    assert not agent_ws._engine_is_stale(tmp_path)
    # A rebase touches every C++ file and changes none: no reason to rebuild.
    _set_times(tmp_path / "lucid/_C/core/Thing.cpp", (tmp_path / SO).stat().st_mtime + 500.0)
    assert not agent_ws._engine_is_stale(tmp_path)


def test_a_cpp_edit_stales_the_stamp_even_under_an_older_mtime(tmp_path: Path) -> None:
    tree = _tree(tmp_path)
    _edit(tmp_path, "lucid/_C/core/Thing.cpp", "int thing() { return 2; }\n", later_than=SO)
    assert tree.freshness() is Freshness.MISMATCH
    assert agent_ws._engine_is_stale(tmp_path)


def test_no_stamp_is_missing_and_only_then_do_mtimes_decide(tmp_path: Path) -> None:
    tree = _tree(tmp_path, built=False)
    (tmp_path / SO).write_bytes(b"engine")
    assert tree.freshness() is Freshness.MISSING and tree.recorded() == ""
    _set_times(tmp_path / SO, (tmp_path / "lucid/_C/core/Thing.cpp").stat().st_mtime + 1.0)
    assert not agent_ws._engine_is_stale(tmp_path)  # the fallback: the binary is newer
    _set_times(tmp_path / SO, (tmp_path / "lucid/_C/core/Thing.cpp").stat().st_mtime - 1.0)
    assert agent_ws._engine_is_stale(tmp_path)


def test_a_copied_engine_with_a_fresh_mtime_cannot_outvote_a_stale_stamp(tmp_path: Path) -> None:
    """The LCD-881 case: the C++ moved on (a sync, a land) and an ``.so`` was ``cp -c``-ed in."""
    tree = _tree(tmp_path)
    _edit(tmp_path, "lucid/_C/core/Thing.cpp", "int thing() { return 3; }\n")
    (tmp_path / SO).write_bytes(b"an engine from another tree")
    _set_times(tmp_path / SO, time.time() + 60.0)  # newer than every source
    assert tree.freshness() is Freshness.MISMATCH
    assert agent_ws._engine_is_stale(tmp_path)


@pytest.mark.parametrize(
    ("name", "text", "stays"),
    [
        ("lucid/_C/core/Thing.cpp", "int thing() { return 4; }\n", False),
        ("lucid/_C/CMakeLists.txt", "project(other)\n", False),
        ("lucid/_C/core/New.mm", "// a new file\n", False),
        ("setup.py", "built = False\n", False),
        ("tools/fetch_mlx.py", "pinned = False\n", False),
        ("pyproject.toml", PYPROJECT.replace('version = "0.1"', 'version = "9"'), False),
        # Not what builds the engine: the project's own version, a stub, the binary itself.
        ("pyproject.toml", PYPROJECT.replace('version = "1"', 'version = "2"'), True),
        ("lucid/_C/engine.pyi", "def f() -> None: ...\n", True),
        ("lucid/_C/__init__.py", "x = 1\n", True),
        ("lucid/_C/engine.cpython-314-darwin.so", "rebuilt\n", True),
    ],
)
def test_the_stamp_covers_what_builds_the_engine_and_only_that(
    tmp_path: Path, name: str, text: str, stays: bool
) -> None:
    tree = _tree(tmp_path)
    (tmp_path / name).write_text(text)
    assert (tree.freshness() is Freshness.MATCH) is stays, name


def test_an_engine_built_into_another_directory_is_not_stamped(tmp_path: Path) -> None:
    tree = _tree(tmp_path, built=False)
    staging = tmp_path / "build/lib.macosx/lucid/_C"
    staging.mkdir(parents=True)
    assert not tree.write(staging)  # a wheel's staging directory vouches for nothing here
    assert not tree.stamp_path.exists()
    assert tree.write(tmp_path / "lucid/_C/../_C")  # the same directory, spelled another way


def test_a_tree_with_no_cpp_has_no_digest_and_takes_no_stamp(tmp_path: Path) -> None:
    tree = EngineTree(tmp_path)
    assert tree.digest() == "" and not tree.write() and tree.freshness() is Freshness.MISSING


def test_a_worktree_that_adopts_main_reads_fresh_only_while_its_sources_match(
    tmp_path: Path,
) -> None:
    main, linked = tmp_path / "main", tmp_path / "linked"
    for root in (main, linked):
        root.mkdir()
        _tree(root, built=False)
    assert EngineTree(linked).adopt(EngineTree(main))  # bootstrap copied main's .so
    assert EngineTree(linked).freshness() is Freshness.MATCH
    _edit(linked, "lucid/_C/core/Thing.cpp", "int thing() { return 5; }\n")
    assert EngineTree(linked).freshness() is Freshness.MISMATCH
    assert not EngineTree(tmp_path / "empty").adopt(EngineTree(tmp_path / "none"))


def test_the_command_line_stamps_and_checks_the_tree_it_sits_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tree = _tree(tmp_path, built=False)
    monkeypatch.setattr(engine_stamp, "HERE", tree)
    assert engine_stamp.main(["check"]) == 1 and "missing" in capsys.readouterr().out
    assert engine_stamp.main(["write", "--engine-dir", str(tmp_path / "elsewhere")]) == 0
    assert "not written" in capsys.readouterr().out and not tree.stamp_path.exists()
    assert engine_stamp.main(["write", "--engine-dir", str(tree.engine_dir)]) == 0
    assert tree.digest()[:12] in capsys.readouterr().out
    assert engine_stamp.main(["check"]) == 0 and "match" in capsys.readouterr().out
    _edit(tmp_path, "setup.py", "built = False\n")
    assert engine_stamp.main(["check"]) == 1 and "mismatch" in capsys.readouterr().out
