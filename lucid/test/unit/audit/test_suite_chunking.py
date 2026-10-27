"""The suite stage splits the tree; these are the two ways that went wrong.

Both bugs shipped and both were found by running the gate, not by
reading it, which is what these tests are for.

1. **Chunks must partition.**  Naming a directory *and* something under
   it runs the descendant twice and doubles its counts, so the gate
   reports a suite larger than the one that exists.

2. **A chunk of entirely-skipped tests is not a dead chunk.**  pytest
   exits ``0`` and prints "16 skipped" with no passed, failed or errored
   among them.  Judging liveness on those three words called that a
   chunk that had died — and the whole stage unfinished — on a tree
   where the reference framework simply is not installed.
"""

from pathlib import Path

from lucid.test.audit._console import Console
from lucid.test.audit._suite import SuiteResult, chunks_for, run_suite


def _tree(root: Path) -> None:
    (root / "alpha" / "one").mkdir(parents=True)
    (root / "alpha" / "two").mkdir(parents=True)
    (root / "beta").mkdir(parents=True)
    (root / "alpha" / "one" / "test_a.py").write_text("def test_a(): pass\n")
    (root / "alpha" / "two" / "test_b.py").write_text("def test_b(): pass\n")
    (root / "alpha" / "test_loose.py").write_text("def test_c(): pass\n")
    (root / "beta" / "test_d.py").write_text("def test_d(): pass\n")
    (root / "test_top.py").write_text("def test_e(): pass\n")


def test_chunks_partition_without_overlap(tmp_path: Path) -> None:
    _tree(tmp_path / "suite")
    chunks = chunks_for(tmp_path, "suite")

    # No chunk may be a prefix of another: that is exactly the shape that
    # runs a directory's tests once for itself and again for its parent.
    for outer in chunks:
        for inner in chunks:
            if outer is inner:
                continue
            assert not inner.startswith(
                outer.rstrip("/") + "/"
            ), f"{inner} is inside {outer} — these chunks overlap"

    # And together they still reach every test file.
    covered = set()
    for chunk in chunks:
        target = tmp_path / chunk
        if target.is_dir():
            covered.update(p for p in target.rglob("test_*.py"))
        else:
            covered.add(target)
    assert covered == set((tmp_path / "suite").rglob("test_*.py"))


def test_an_ignored_subtree_is_not_chunked(tmp_path: Path) -> None:
    _tree(tmp_path / "suite")
    chunks = chunks_for(tmp_path, "suite", ("suite/alpha/one",))
    assert "suite/alpha/one" not in chunks


def test_a_wholly_skipped_chunk_is_not_a_dead_chunk(tmp_path: Path) -> None:
    """Exit 0 with only skips is a complete run, not a casualty."""
    folder = tmp_path / "suite"
    folder.mkdir()
    (folder / "test_all_skipped.py").write_text(
        "import pytest\n"
        "@pytest.mark.skip(reason='no reference framework here')\n"
        "def test_one(): pass\n"
    )
    result = run_suite(
        Console(colour=False, quiet=True), path="suite", with_coverage=False, root=tmp_path
    )
    assert result.unfinished == [], result.unfinished
    assert result.counts.get("skipped") == 1
    assert result.broken == 0


def test_a_killed_chunk_is_named(tmp_path: Path) -> None:
    """A chunk that exits badly with nothing to say must be reported."""
    folder = tmp_path / "suite"
    folder.mkdir()
    # ``exit`` during collection: pytest returns non-zero and never
    # reaches a summary line, which is the shape of the -9 death.
    (folder / "test_dies.py").write_text("import os\nos._exit(9)\n")
    result = run_suite(
        Console(colour=False, quiet=True), path="suite", with_coverage=False, root=tmp_path
    )
    assert result.unfinished, "a chunk that died silently was not recorded"
    assert "suite" in result.unfinished[0]
    assert not result.finished
    assert result.broken >= 1


def test_skipped_chunks_make_the_stage_red() -> None:
    """Declining to run part of the tree is not a pass."""
    result = SuiteResult(ran=True, counts={"passed": 100})
    assert result.broken == 0
    result.skipped_chunks.append(("suite/heavy", "300 MB available, floor 1024 MB"))
    assert result.broken == 1
