"""A test in the nightly sanitizer target that reads process memory carries the marker (LCD-1056).

``@pytest.mark.requires_unsanitized_memory`` skips a test under ASan, whose quarantine and
redzones inflate the process footprint (``lucid/test/_fixtures/sanitizer.py``).  A marker that
has to be remembered is forgotten: the next test that reads a child's ``ru_maxrss`` fails the
nightly native job, on the Studio, a day after it landed.  So this module lists the tests the
nightly job runs and fails on any that reaches a process-memory reader without the marker.

What is scanned.  The target is read out of ``scripts/ci_sanitizer.sh`` (its default
``PYTEST_TARGET``), so widening the nightly job widens this guard with it.  A test *reaches* a
reader when the reader's name (``ru_maxrss``, ``getrusage``, ``footprint_gb``, ``footprint_mb``,
``settled_footprint_mb``, ``resident_mb``, ``phys_footprint``, ``task_info``) appears in the test, in a module-level helper, constant or
fixture it names, transitively -- as an identifier, or inside a string, because the repository
measures a child's peak from code held in a string (``python -c``).  Docstrings are prose and
are skipped.

What is not.  An allocator counter -- ``memory_stats(...)``, ``live_bytes``, ``metal_peak_memory``
-- counts what the engine asked for, which a sanitizer does not change, so those tests keep
running under it.  A reader imported from a module other than the test's own is found by its
name where the test uses it; a *new* reader function in a shared helper has to be added to
``_READERS`` here.
"""

import ast
import re
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import pytest

from lucid.test._fixtures import sanitizer
from lucid.test._fixtures.sanitizer import MEMORY_MARKER

_REPO = Path(__file__).resolve().parents[3]
_ROOT_CONFTEST = _REPO / "lucid" / "test" / "conftest.py"
_SCRIPT = _REPO / "scripts" / "ci_sanitizer.sh"
_DEFAULT_TARGET = re.compile(
    r'^PYTEST_TARGET="\$\{PYTEST_TARGET:-(?P<paths>[^}"]+)\}"', re.MULTILINE
)

# Functions and fields that read how much memory the *process* holds or peaked at.
_READERS = re.compile(
    r"\b(?:ru_maxrss|getrusage|RUSAGE_\w+|footprint_gb|footprint_mb|settled_footprint_mb|"
    r"resident_mb|phys_footprint|task_info)\b"
)


class _Unit(NamedTuple):
    """One top-level definition (or one test method): whether it names a reader, and what else."""

    reads: bool
    refs: frozenset[str]


def _docstring_ids(tree: ast.AST) -> set[int]:
    skip: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                skip.add(id(first.value))
    return skip


def _unit(node: ast.AST, skip: set[int]) -> _Unit:
    reads = False
    refs: set[str] = set()
    for sub in ast.walk(node):
        text: str | None = None
        if isinstance(sub, ast.Name):
            text = sub.id
            refs.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            text = sub.attr
            refs.add(sub.attr)
        elif isinstance(sub, ast.arg):  # a fixture is reached by its parameter name
            text = sub.arg
            refs.add(sub.arg)
        elif isinstance(sub, ast.alias):
            text = sub.name.rsplit(".", 1)[-1]
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str) and id(sub) not in skip:
            text = sub.value
        if text is not None and _READERS.search(text) is not None:
            reads = True
    return _Unit(reads, frozenset(refs))


def _defined_names(node: ast.stmt) -> list[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [(a.asname or a.name).split(".")[0] for a in node.names]
    return []


def _reaches_reader(start: _Unit, scope: dict[str, _Unit]) -> bool:
    seen: set[str] = set()
    pending = [start]
    while pending:
        unit = pending.pop()
        if unit.reads:
            return True
        for name in unit.refs - seen:
            seen.add(name)
            if name in scope:
                pending.append(scope[name])
    return False


def _names_marker(node: ast.AST) -> bool:
    return any(
        isinstance(sub, ast.Attribute) and sub.attr == MEMORY_MARKER for sub in ast.walk(node)
    )


def unmarked_memory_tests(source: str) -> list[str]:
    """Tests in ``source`` that reach a process-memory reader and do not carry the marker."""
    tree = ast.parse(source)
    skip = _docstring_ids(tree)
    scope: dict[str, _Unit] = {}
    module_marked = False
    for stmt in tree.body:
        unit = _unit(stmt, skip)
        for name in _defined_names(stmt):
            scope[name] = unit
        if isinstance(stmt, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in stmt.targets
        ):
            module_marked = module_marked or _names_marker(stmt.value)

    found: list[str] = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and stmt.name.startswith(
            "test"
        ):
            marked = module_marked or any(_names_marker(d) for d in stmt.decorator_list)
            if not marked and _reaches_reader(_unit(stmt, skip), scope):
                found.append(stmt.name)
        elif isinstance(stmt, ast.ClassDef) and stmt.name.startswith("Test"):
            class_marked = module_marked or any(_names_marker(d) for d in stmt.decorator_list)
            methods = [
                m for m in stmt.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            inner = {m.name: _unit(m, skip) for m in methods}
            for method in methods:
                if not method.name.startswith("test"):
                    continue
                marked = class_marked or any(_names_marker(d) for d in method.decorator_list)
                if not marked and _reaches_reader(inner[method.name], {**scope, **inner}):
                    found.append(f"{stmt.name}::{method.name}")
    return found


def nightly_target() -> list[Path]:
    """The test files the nightly sanitizer job runs, read from its script."""
    match = _DEFAULT_TARGET.search(_SCRIPT.read_text(encoding="utf-8"))
    assert match is not None, f"no default PYTEST_TARGET found in {_SCRIPT}; update this guard"
    files: list[Path] = []
    for entry in match["paths"].split():
        path = _REPO / entry
        assert path.exists(), f"{entry} is in the sanitizer job's target and does not exist"
        files.extend(sorted(path.rglob("test_*.py")) if path.is_dir() else [path])
    return files


def _iter_unmarked() -> Iterator[str]:
    for path in nightly_target():
        for name in unmarked_memory_tests(path.read_text(encoding="utf-8")):
            yield f"{path.relative_to(_REPO)}::{name}"


def test_the_sanitizer_target_is_found() -> None:
    files = nightly_target()
    assert any(f.name == "test_to_device_host_buffers.py" for f in files)
    assert any(f.name == "test_audit_smoke.py" for f in files)


def test_every_process_memory_test_in_the_sanitizer_target_carries_the_marker() -> None:
    offenders = sorted(_iter_unmarked())
    assert not offenders, (
        f"these tests read process memory (RSS / footprint) in the nightly sanitizer target "
        f"without @pytest.mark.{MEMORY_MARKER}; ASan's quarantine and redzones inflate that "
        "number, so they fail the job under it.  Mark each, or read an allocator counter "
        "(memory_stats / live_bytes) instead:\n  " + "\n  ".join(offenders)
    )


# ── the detector itself: each case is a way the guard could miss or over-reach ──


def test_the_detector_flags_a_reader_named_in_the_test() -> None:
    src = "import resource\ndef test_a():\n    assert resource.getrusage(0).ru_maxrss < 5\n"
    assert unmarked_memory_tests(src) == ["test_a"]


def test_the_detector_follows_a_helper_and_a_child_script_held_in_a_string() -> None:
    src = (
        "import subprocess\n"
        "_CHILD = 'import resource; print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)'\n"
        "def _rise():\n"
        "    return subprocess.run(['python', '-c', _CHILD])\n"
        "def test_a():\n"
        "    assert _rise()\n"
        "def test_b():\n"
        "    assert 1\n"
    )
    assert unmarked_memory_tests(src) == ["test_a"]


def test_the_detector_follows_a_fixture_and_a_reader_imported_by_name() -> None:
    src = (
        "from lucid.test.audit._zoo_budget import footprint_gb as fp\n"
        "import pytest\n"
        "@pytest.fixture\n"
        "def grown():\n"
        "    return fp(1)\n"
        "def test_a(grown):\n"
        "    assert grown < 1\n"
    )
    assert unmarked_memory_tests(src) == ["test_a"]


def test_the_detector_follows_self_helpers_in_a_test_class() -> None:
    src = (
        "class TestX:\n"
        "    def _peak(self):\n"
        "        return resident_mb()\n"
        "    def test_a(self):\n"
        "        assert self._peak() < 5\n"
        "    def test_b(self):\n"
        "        assert 1\n"
    )
    assert unmarked_memory_tests(src) == ["TestX::test_a"]


def test_the_detector_leaves_an_allocator_counter_alone() -> None:
    src = (
        "def test_a():\n"
        "    before = _C_engine.memory_stats(dev).current_bytes\n"
        "    assert _C_engine.memory_stats(dev).peak_bytes - before < 4\n"
        "    assert live_bytes('cpu') == 0 and _C_engine.metal_peak_memory() < 9\n"
    )
    assert unmarked_memory_tests(src) == []


def test_the_detector_skips_prose() -> None:
    src = 'def test_a():\n    """Reads ru_maxrss in a child."""\n    assert 1\n'
    assert unmarked_memory_tests(src) == []


@pytest.mark.parametrize(
    "marked",
    [
        "@pytest.mark.requires_unsanitized_memory\ndef test_a():\n    getrusage()\n",
        "@pytest.mark.slow\n@pytest.mark.requires_unsanitized_memory\n"
        "@pytest.mark.parametrize('c', [1])\ndef test_a(c):\n    getrusage()\n",
        "pytestmark = pytest.mark.requires_unsanitized_memory\ndef test_a():\n    getrusage()\n",
        "pytestmark = [pytest.mark.slow, pytest.mark.requires_unsanitized_memory]\n"
        "def test_a():\n    getrusage()\n",
        "@pytest.mark.requires_unsanitized_memory\nclass TestX:\n    def test_a(self):\n"
        "        getrusage()\n",
        "class TestX:\n    @pytest.mark.requires_unsanitized_memory\n    def test_a(self):\n"
        "        getrusage()\n",
    ],
)
def test_the_detector_accepts_every_place_the_marker_can_sit(marked: str) -> None:
    assert unmarked_memory_tests(marked) == []


# ── the detection and the skip ──

_ASAN = "/Xcode/usr/lib/clang/17/lib/darwin/libclang_rt.asan_osx_dynamic.dylib"
_UBSAN = "/Xcode/usr/lib/clang/17/lib/darwin/libclang_rt.ubsan_osx_dynamic.dylib"
_TSAN = "/Xcode/usr/lib/clang/17/lib/darwin/libclang_rt.tsan_osx_dynamic.dylib"
_OTHER = "/usr/lib/libSystem.B.dylib"


def test_each_sanitizer_runtime_is_recognised_by_its_image_name() -> None:
    assert sanitizer.sanitizers_in([_OTHER]) == frozenset()
    assert sanitizer.sanitizers_in([_OTHER, _ASAN]) == {"address"}
    assert sanitizer.sanitizers_in([_UBSAN]) == {"undefined"}
    assert sanitizer.sanitizers_in([_TSAN, _UBSAN]) == {"thread", "undefined"}


def test_only_asan_and_tsan_inflate_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    for images, inflates, sanitized in (
        ([_OTHER], False, False),
        ([_OTHER, _UBSAN], False, True),
        ([_OTHER, _ASAN], True, True),
        ([_OTHER, _TSAN], True, True),
    ):
        monkeypatch.setattr(sanitizer, "_loaded_image_paths", lambda images=images: images)
        assert sanitizer.inflates_memory() is inflates, images
        assert sanitizer.is_sanitized() is sanitized, images


def test_the_image_list_is_read_from_the_running_process() -> None:
    """The dyld calls work on this host: an empty list would read as "no sanitizer" forever."""
    images = sanitizer._loaded_image_paths()
    assert any(path.endswith("libSystem.B.dylib") for path in images), images[:5]
    assert sanitizer.loaded_sanitizers() == sanitizer.sanitizers_in(images)


class _Item:
    def __init__(self, marked: bool) -> None:
        self.marked = marked
        self.added: list[pytest.MarkDecorator] = []

    def get_closest_marker(self, name: str) -> object | None:
        return object() if self.marked and name == MEMORY_MARKER else None

    def add_marker(self, marker: pytest.MarkDecorator) -> None:
        self.added.append(marker)


def test_the_skip_lands_on_marked_tests_only_and_only_under_asan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marked, plain = _Item(True), _Item(False)

    monkeypatch.setattr(sanitizer, "_loaded_image_paths", lambda: [_OTHER, _UBSAN])
    sanitizer.skip_where_memory_is_inflated([marked, plain])
    assert marked.added == [] and plain.added == []

    monkeypatch.setattr(sanitizer, "_loaded_image_paths", lambda: [_OTHER, _ASAN])
    sanitizer.skip_where_memory_is_inflated([marked, plain])
    assert plain.added == []
    assert len(marked.added) == 1
    assert marked.added[0].mark.name == "skip"
    assert "ASan quarantine and redzones" in str(marked.added[0].mark.kwargs["reason"])


def test_the_root_conftest_runs_that_skip(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The marker is a no-op unless the collection hook is wired: call the registered one."""
    hook = request.config.pluginmanager.hook.pytest_collection_modifyitems
    ours = [impl for impl in hook.get_hookimpls() if Path(impl.plugin_name) == _ROOT_CONFTEST]
    assert len(ours) == 1, [impl.plugin_name for impl in hook.get_hookimpls()]
    calls: list[object] = []
    monkeypatch.setattr(sanitizer, "skip_where_memory_is_inflated", calls.append)
    ours[0].function([])
    assert calls == [[]]
