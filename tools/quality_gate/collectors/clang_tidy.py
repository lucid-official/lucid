"""clang-tidy on the C++ engine (DS-3 / LCD-262).

Runs ``clang-tidy`` over the engine's ``compile_commands.json`` with the DS-3
check set (``config.CLANG_TIDY_CHECKS``):
- ``bugprone-*``
- ``cppcoreguidelines-pro-bounds-*``
- ``cppcoreguidelines-init-variables``
- ``cppcoreguidelines-narrowing-conversions``
- ``readability-function-size``
- ``performance-*``

Opt-in (``default = False``, invoked via ``--collectors clang-tidy``) because
it requires a configured CMake compile database and ~30 s across all 249
translation units; its baseline is recorded in ``tools/quality_baseline.json``.
"""

from concurrent.futures import ThreadPoolExecutor
import functools
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from typing import override

from tools.quality_gate import config
from tools.quality_gate.collectors import Collector, run_tool
from tools.quality_gate.core import Counts, GateError, Scope, is_cpp_target

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DIAG_RE = re.compile(
    r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+):\s+"
    r"(?P<sev>warning|error):\s+.+?\[(?P<rules>[^\]]+)\]\s*$"
)
_VERSION_RE = re.compile(r"version\s+([0-9]+\.[0-9]+\.[0-9]+)")


@functools.cache
def clang_tidy_bin() -> str:
    """Resolve the ``clang-tidy`` binary from env, the active venv, or PATH."""
    env_bin = os.environ.get("CLANG_TIDY")
    if env_bin and shutil.which(env_bin):
        return str(shutil.which(env_bin))
    venv_bin = Path(sys.executable).parent / "clang-tidy"
    if venv_bin.is_file():
        return str(venv_bin)
    try:
        dist = importlib.metadata.distribution("clang-tidy")
        for f in dist.files or ():
            if f.name == "clang-tidy" and "bin" in f.parts:
                path = Path(str(dist.locate_file(f))).resolve()
                if path.is_file():
                    return str(path)
    except importlib.metadata.PackageNotFoundError:
        pass
    which_bin = shutil.which("clang-tidy")
    if which_bin:
        return which_bin
    brew_bin = Path("/opt/homebrew/opt/llvm/bin/clang-tidy")
    if brew_bin.is_file():
        return str(brew_bin)
    raise GateError("clang-tidy is not installed (pip install clang-tidy)")


def _ensure_compile_commands(scope: Scope) -> Path:
    """Return the directory containing ``compile_commands.json`` for *scope*."""
    root = scope.root.resolve()
    for candidate in (root / "build" / "compile_commands.json", root / "compile_commands.json"):
        if candidate.is_file():
            return candidate
    script = root / "scripts" / "build_compile_commands.sh"
    if script.is_file():
        env = {**os.environ, "PYTHON_BIN": sys.executable}
        run_tool([str(script)], cwd=root, env=env)
        built = root / "build" / "compile_commands.json"
        if built.is_file():
            return built
    repo_db = _REPO_ROOT / "build" / "compile_commands.json"
    if repo_db.is_file() and root != _REPO_ROOT:
        adapted_dir = scope.workdir / "compile_commands"
        adapted_dir.mkdir(parents=True, exist_ok=True)
        adapted = adapted_dir / "compile_commands.json"
        text = repo_db.read_text(encoding="utf-8").replace(str(_REPO_ROOT), str(root))
        adapted.write_text(text, encoding="utf-8")
        return adapted
    raise GateError(
        "clang-tidy: compile_commands.json not found (run ./scripts/build_compile_commands.sh)"
    )


def _translation_units(db_path: Path, scope: Scope) -> list[str]:
    """Translation units to analyze from *db_path*, filtered to *scope*."""
    root = scope.root.resolve()
    scope_set = set(scope.cpp)
    try:
        entries = json.loads(db_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"clang-tidy: unreadable {db_path} ({exc})") from exc
    units: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        raw_file = entry.get("file", "")
        if not raw_file:
            continue
        p = Path(raw_file)
        if not p.is_absolute():
            p = (Path(entry.get("directory", root)) / p).resolve()
        else:
            p = p.resolve()
        try:
            rel = p.relative_to(root).as_posix()
        except ValueError:
            continue
        if rel.startswith("lucid/_C/test/") or not p.is_file():
            continue
        if rel not in seen and (rel in scope_set or len(scope_set) > 50):
            seen.add(rel)
            units.append(str(p))
    return units


def _run_batch(exe: str, db_dir: Path, batch: list[str], cwd: Path) -> str:
    cmd = [
        exe,
        f"-p={db_dir}",
        "--quiet",
        f"--checks={config.CLANG_TIDY_CHECKS}",
        "--warnings-as-errors=-*",
        f"--header-filter={config.CLANG_TIDY_HEADER_FILTER}",
        *batch,
    ]
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise GateError(f"{exe}: not found ({exc})") from exc
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode not in (0, 1):
        tail = out.strip()[-2000:]
        raise GateError(f"clang-tidy exited {proc.returncode}:\n{tail}")
    return out


class ClangTidyCollector(Collector):
    name = "clang-tidy"
    default = False

    @override
    def version(self) -> str:
        out = run_tool([clang_tidy_bin(), "--version"], Path.cwd())
        m = _VERSION_RE.search(out)
        return m.group(1) if m else out.strip().splitlines()[0]

    @override
    def collect(self, scope: Scope) -> Counts:
        cpp_files = [f for f in scope.cpp if not f.startswith("lucid/_C/test/")]
        if not cpp_files:
            return {}
        db_path = _ensure_compile_commands(scope)
        units = _translation_units(db_path, scope)
        if not units:
            return {}
        exe = clang_tidy_bin()
        db_dir = db_path.parent
        root = scope.root.resolve()
        batch_size = 8
        batches = [units[i : i + batch_size] for i in range(0, len(units), batch_size)]
        workers = min(len(batches), os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outputs = list(pool.map(lambda b: _run_batch(exe, db_dir, b, root), batches))

        seen_sites: set[tuple[str, int, int, str]] = set()
        counts: Counts = defaultdict(int)
        for out in outputs:
            for line in out.splitlines():
                m = _DIAG_RE.match(line)
                if not m:
                    continue
                rules = [r.strip() for r in m.group("rules").split(",") if r.strip()]
                if not rules:
                    continue
                if m.group("sev") == "error" and any(
                    r.startswith("clang-diagnostic-") for r in rules
                ):
                    raise GateError(f"clang-tidy compile error: {line}")
                rule = rules[0]
                if rule.startswith("clang-diagnostic-"):
                    continue
                raw_path = Path(m.group("file"))
                full_path = (raw_path if raw_path.is_absolute() else (root / raw_path)).resolve()
                try:
                    rel = full_path.relative_to(root).as_posix()
                except ValueError:
                    continue
                if not is_cpp_target(rel) or rel.startswith("lucid/_C/test/"):
                    continue
                site = (rel, int(m.group("line")), int(m.group("col")), rule)
                if site not in seen_sites:
                    seen_sites.add(site)
                    counts[("clang-tidy", rule, rel)] += 1
        return dict(counts)
