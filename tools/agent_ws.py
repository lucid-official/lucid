#!/usr/bin/env python3
"""
agent_ws.py — the parallel-agent workspace: claims, bootstrap, land, status.

Several Claude sessions work on Lucid at once, each in its own git worktree
under ``.claude/worktrees/``.  The main checkout is the integration zone: the
orchestrator lands finished work there by fast-forward and nobody else writes
to ``main``.  Every moving part lives in this one stdlib-only script so the
Claude hook, the git hooks, the agents and a human all run the same code.

The invariant that keeps merges conflict-free is simple: **two in-flight
worktrees never modify the same file.**  A file is *claimed* by a worktree
when it differs there from the merge-base with ``main`` — committed or not,
tracked or new.  Claims are recomputed from git state on every call, so there
is nothing to release and nothing to leak: landing a branch or removing its
worktree drops its claims.  Three gates enforce it:

  1. ``hook``          Claude Code PreToolUse: an Edit/Write to a file another
                       worktree claims is denied (asked, from the main checkout).
  2. ``check-commit``  git pre-commit: the same test on the staged files, which
                       also catches edits made through Bash.
  3. ``land``          rebases onto main and only fast-forwards, so a merge
                       commit — and with it a merge conflict — never happens.

Generated files (the three stubs, the audit manifests, CHANGELOG.md) are
exempt: they are regenerated, never hand-merged.

Subcommands::

  status [--files] [--json]   board of worktrees, labels, claims and locks
  label TEXT                  tag the current worktree with its task
  bootstrap [PATH]            make a worktree runnable (.so, light .venv, vault)
  doctor                      show where `import lucid` resolves from here
  land WORKTREE [--check] [--cleanup] [--skip-stub-check] [--no-changelog]
                [--allow-slop REASON]   (refuses a quality-gate increase, exit 6,
                an audit baseline whose totals disagree with its body, exit 7,
                changed-line coverage under 90%, exit 8, and --check refuses a
                diff-cover that cannot measure, exit 9)
  heavy -- CMD...             run CMD under the machine-wide heavy-job lock
  build                       build the C++ engine in place in a worktree
  hook                        Claude Code PreToolUse entry point (stdin JSON)
  check-commit                git pre-commit entry point
  setup                       idempotent git config (rerere, merge driver)

Why it is shaped this way: obsidian/architecture/arch-parallel-agent-workspace.md
"""

import argparse
import contextlib
import fcntl
import fnmatch
import functools
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

MAIN_BRANCH = "main"

# The audit's recorded baselines: each holds a header total beside the body it
# sums, and `land` refuses one where they disagree (LCD-358).
AUDIT_BASELINES = (
    "lucid/test/audit/coverage.json",
    "lucid/test/audit/doctest.json",
    "lucid/test/audit/suite.json",
)

# Regenerated rather than merged: a conflict here is resolved by running the
# generator again, so claiming them would only serialise unrelated work.
GENERATED = (
    "lucid/_C/engine.pyi",
    "lucid/_tensor/tensor.pyi",
    "lucid/__init__.pyi",
    *AUDIT_BASELINES,
    "CHANGELOG.md",
    # Every branch that lowers a count writes it; the `lucid-quality-baseline`
    # merge driver (setup) merges it per key, so it must not serialise them.
    "tools/quality_baseline.json",
)

# git merge drivers registered by `setup` (and re-checked by `land`).  Repo-local
# config, so the shared .git/config carries them to every worktree and .sync
# carries them to the other Mac.
MERGE_DRIVERS = {
    "merge.lucid-generated.name": "Lucid generated file: keep one side, regenerate after",
    "merge.lucid-generated.driver": "true",
    "merge.lucid-quality-baseline.name": "Lucid quality baseline: per-key 3-way, lower wins",
    "merge.lucid-quality-baseline.driver": (
        '"$(test -x .venv/bin/python3 && echo .venv/bin/python3 || echo python3)" '
        "-m tools.quality_gate --merge-baseline %O %A %B"
    ),
}
SLOP_EXIT = 6
BASELINE_EXIT = 7
DIFF_COVER_EXIT = 8
DIFF_COVER_UNMEASURED_EXIT = 9
DIFF_COVER_THRESHOLD = 90
# diff-cover's own verdict for --fail-under.  Any other non-zero exit means
# it never measured (module missing, git error, unreadable XML).
DIFF_COVER_BELOW = "Coverage is below"
COVERAGE_OMIT_PREFIXES = ("lucid/test/", "lucid/models/", "lucid/benchmarks/")

# Machine-local and outside the synced project tree: locks must never travel
# to the other Mac through .sync.
STATE_DIR = Path(
    os.environ.get(
        "LUCID_AGENT_STATE", str(Path.home() / "Library/Caches/lucid-agents")
    )
)
HEAVY_LOCK = STATE_DIR / "heavy.lock"
LAND_LOCK = STATE_DIR / "land.lock"
HOOK_LOG = STATE_DIR / "hook.log"
SLOP_LOG = STATE_DIR / "slop-overrides.log"  # every `land --allow-slop`, with its reason
SELF = Path(__file__).resolve()
BASELINE_CHECK = SELF.with_name("check_audit_baselines.py")

USER_FACING_RE = r"^(feat|fix|perf|refactor|revert|remove|deprec|sec)(\([^)]*\))?!?:\s"
EDIT_TOOLS = ("Edit", "Write", "MultiEdit", "NotebookEdit")
XCODE_SDK = Path(
    "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"
)


class WorkspaceError(Exception):
    """A precondition failed; the message is meant for the agent or human."""

    def __init__(self, message: str, code: int = 1) -> None:
        super().__init__(message)
        self.code = code


# ── git plumbing ──────────────────────────────────────────────────────────────


@functools.cache
def repo_env() -> dict[str, str]:
    """This process's environment minus git's repository-local variables.

    git exports some of them to the hooks it runs: pre-commit always gets
    GIT_INDEX_FILE (a relative ``.git/index`` in the main checkout, a temporary
    ``next-index-*.lock`` for ``git commit -- <path>``) and, from a linked
    worktree, GIT_DIR; ``git rebase --exec`` exports GIT_DIR too.  They outrank
    ``cwd``, so a query of another tree that inherits them reads the committing
    tree's index and config instead — every tree then claims the staged files,
    under the committer's label.  Without them each query finds its tree from
    ``cwd``.

    The names are git's own list, ``git rev-parse --local-env-vars`` — what git
    clears before it enters another repository (a submodule) — so they follow
    the installed git.  GIT_CONFIG_KEY_<n> / GIT_CONFIG_VALUE_<n> are read only
    through GIT_CONFIG_COUNT, which is on it.
    """
    bare = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    proc = subprocess.run(  # outside any repository: the list depends on none
        ["git", "rev-parse", "--local-env-vars"],
        cwd="/",
        env=bare,
        capture_output=True,
        text=True,
        check=False,
    )
    names = set(proc.stdout.split())
    if proc.returncode != 0 or "GIT_INDEX_FILE" not in names:
        raise WorkspaceError(
            f"git rev-parse --local-env-vars failed:\n{proc.stderr.strip()}"
        )
    return {k: v for k, v in os.environ.items() if k not in names}


def git(
    *args: str, cwd: Path, check: bool = True, env: dict[str, str] | None = None
) -> str:
    """Run git in ``cwd``.  Without ``env`` it gets ``repo_env()``: the tree is
    found from ``cwd``, never from a GIT_DIR / GIT_INDEX_FILE a hook inherited."""
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=repo_env() if env is None else env,
        check=False,
    )
    if check and proc.returncode != 0:
        raise WorkspaceError(
            f"git {' '.join(args)} failed in {cwd}:\n{proc.stderr.strip()}"
        )
    return proc.stdout


def git_ok(*args: str, cwd: Path) -> bool:
    return (
        subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, env=repo_env()
        ).returncode
        == 0
    )


@dataclass
class Worktree:
    path: Path
    head: str
    branch: str | None
    is_main: bool
    prunable: bool
    _changed: set[str] | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return "main checkout" if self.is_main else self.path.name

    @property
    def ref(self) -> str:
        return self.branch or self.head


def list_worktrees(any_path: Path) -> list[Worktree]:
    out = git("worktree", "list", "--porcelain", cwd=any_path)
    trees: list[Worktree] = []
    for block in out.strip().split("\n\n"):
        info: dict[str, str] = {}
        for line in block.splitlines():
            key, _, value = line.partition(" ")
            info[key] = value
        if "worktree" not in info or "bare" in info:
            continue
        branch = info.get("branch", "")
        trees.append(
            Worktree(
                path=Path(info["worktree"]),
                head=info.get("HEAD", ""),
                branch=branch.removeprefix("refs/heads/") or None,
                is_main=not trees,  # git lists the main worktree first
                prunable="prunable" in info,
            )
        )
    return trees


def existing_dir(path: Path) -> Path:
    """Nearest existing directory at or above ``path`` (a new file has none yet)."""
    probe = path if path.is_dir() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe


def toplevel_of(path: Path) -> Path | None:
    """Repository root containing ``path``, found without spawning git."""
    probe = existing_dir(path.absolute())
    while True:
        if (probe / ".git").exists():
            return probe
        if probe == probe.parent:
            return None
        probe = probe.parent


def is_linked(root: Path) -> bool:
    return (root / ".git").is_file()


def dirty_files(root: Path) -> set[str]:
    """Modified, staged, deleted and untracked paths (untracked dirs end in '/')."""
    out = git("status", "--porcelain=v1", "-z", "--untracked-files=normal", cwd=root)
    files: set[str] = set()
    entries = out.split("\0")
    i = 0
    while i < len(entries):
        entry = entries[i]
        i += 1
        if len(entry) < 4:
            continue
        status, path = entry[:2], entry[3:]
        files.add(path)
        if "R" in status or "C" in status:  # rename/copy: the source path follows
            if i < len(entries) and entries[i]:
                files.add(entries[i])
            i += 1
    return files


def merge_base(root: Path, ref: str) -> str | None:
    out = git("merge-base", MAIN_BRANCH, ref, cwd=root, check=False).strip()
    return out or None


def diff_names(root: Path, a: str, b: str) -> set[str]:
    out = git("diff", "--name-only", "--no-renames", f"{a}..{b}", cwd=root)
    return {line for line in out.splitlines() if line}


def changed_files(wt: Worktree) -> set[str]:
    """Everything this worktree has changed relative to where it left main."""
    if wt._changed is None:
        files = dirty_files(wt.path)
        if not wt.is_main and wt.head:
            base = merge_base(wt.path, wt.head)
            if base:
                files |= diff_names(wt.path, base, wt.head)
        wt._changed = files
    return wt._changed


def is_exempt(rel: str) -> bool:
    return rel in GENERATED or rel.startswith("obsidian/")


def covers(claims: set[str], rel: str) -> bool:
    if rel in claims:
        return True
    return any(c.endswith("/") and rel.startswith(c) for c in claims)


def find_tree(trees: list[Worktree], root: Path) -> Worktree:
    resolved = root.resolve()
    for wt in trees:
        if wt.path.resolve() == resolved:
            return wt
    raise WorkspaceError(f"{root} is not a worktree of this repository")


# ── claims ────────────────────────────────────────────────────────────────────


@dataclass
class Verdict:
    rel: str
    owners: list[Worktree]
    main_moved: bool = False

    @property
    def clear(self) -> bool:
        return not self.owners and not self.main_moved


def judge(rel: str, me: Worktree, trees: list[Worktree]) -> Verdict:
    if is_exempt(rel):
        return Verdict(rel, [])
    owners = [
        wt
        for wt in trees
        if wt.path != me.path
        and not wt.prunable
        and wt.path.exists()
        and covers(changed_files(wt), rel)
    ]
    verdict = Verdict(rel, owners)
    # main moved under us: editing now would conflict on the rebase at land.
    if not owners and not me.is_main and not covers(changed_files(me), rel):
        base = merge_base(me.path, me.head)
        if base and rel in diff_names(me.path, base, MAIN_BRANCH):
            verdict.main_moved = True
    return verdict


def label_of(wt: Worktree) -> str:
    return git("config", "--get", "lucid.task", cwd=wt.path, check=False).strip()


def describe_owner(wt: Worktree) -> str:
    label = label_of(wt)
    where = (
        "the main checkout (uncommitted)" if wt.is_main else f"worktree {wt.path.name}"
    )
    return f"{where} [{wt.ref}]" + (f" — {label}" if label else "")


def explain(verdict: Verdict) -> str:
    if verdict.owners:
        who = "; ".join(describe_owner(wt) for wt in verdict.owners)
        return (
            f"{verdict.rel} is claimed by {who}. Two in-flight branches may not modify the "
            "same file (it would conflict at land). Work around it without touching this file, "
            "or stop and report BLOCKED with this message so the orchestrator can sequence the "
            "work. Do not edit it through Bash either — the pre-commit gate will refuse it."
        )
    return (
        f"{verdict.rel} changed on main after this worktree branched. Bring the branch up to "
        "date first — commit or stash, then `git rebase main` — and edit it after that."
    )


# ── locks ─────────────────────────────────────────────────────────────────────


@contextlib.contextmanager
def held_lock(path: Path, label: str, wait: bool = True) -> Iterator[None]:
    """flock-based: the kernel drops it when the holder dies, so it cannot go stale."""
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = path.with_suffix(".owner")
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            holder = owner.read_text().strip() if owner.exists() else "unknown"
            if not wait:
                raise WorkspaceError(
                    f"{path.name} is held by {holder}", code=2
                ) from None
            print(
                f"[agent_ws] waiting for {path.name} (held by {holder})",
                file=sys.stderr,
            )
            fcntl.flock(fd, fcntl.LOCK_EX)
        stamp = time.strftime("%H:%M:%S")
        owner.write_text(f"pid {os.getpid()} since {stamp}: {label}\n")
        try:
            yield
        finally:
            owner.unlink(missing_ok=True)
    finally:
        os.close(fd)


def lock_holder(path: Path) -> str:
    if not path.exists():
        return "free"
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return "free"
    except BlockingIOError:
        owner = path.with_suffix(".owner")
        return owner.read_text().strip() if owner.exists() else "held"
    finally:
        os.close(fd)


# ── bootstrap ─────────────────────────────────────────────────────────────────


def main_root_of(trees: list[Worktree]) -> Path:
    return trees[0].path


def _newest_cpp_mtime(root: Path) -> float:
    """Newest mtime across ``lucid/_C/**/*.{cpp,h,mm}``, or ``0.0`` if none."""
    c_dir = root / "lucid" / "_C"
    if not c_dir.is_dir():
        return 0.0
    newest = 0.0
    for pat in ("*.cpp", "*.h", "*.mm"):
        for p in c_dir.rglob(pat):
            try:
                mt = p.stat().st_mtime
            except OSError:
                continue
            if mt > newest:
                newest = mt
    return newest


def _cpp_source_hash(root: Path) -> str:
    """SHA-256 over relative path and bytes across ``lucid/_C/**/*.{cpp,h,mm,def}``."""
    c_dir = root / "lucid" / "_C"
    if not c_dir.is_dir():
        return ""
    files = sorted(
        p
        for pat in ("*.cpp", "*.h", "*.mm", "*.def")
        for p in c_dir.rglob(pat)
        if p.is_file()
    )
    if not files:
        return ""
    digest = hashlib.sha256()
    for p in files:
        try:
            rel = p.relative_to(root).as_posix()
            data = p.read_bytes()
        except OSError:
            continue
        digest.update(f"{rel}\0".encode())
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


def _header_abi_version(root: Path) -> int | None:
    """Parse ``LUCID_ABI_VERSION`` from ``lucid/_C/version.h``, if present."""
    vh = root / "lucid" / "_C" / "version.h"
    if not vh.is_file():
        return None
    try:
        text = vh.read_text()
    except OSError:
        return None
    m = re.search(r"^\s*#\s*define\s+LUCID_ABI_VERSION\s+(\d+)\b", text, re.MULTILINE)
    return int(m.group(1)) if m else None


def _so_abi_version(root: Path) -> int | None:
    """Read ``_C_engine.ABI_VERSION`` using ``root/.venv/bin/python3``, if runnable."""
    py = root / ".venv/bin/python3"
    if not py.is_file():
        return None
    proc = subprocess.run(
        [str(py), "-c", "from lucid import _C_engine; print(_C_engine.ABI_VERSION)"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return int(out) if out.isdigit() else None


def _engine_stamp_path(root: Path) -> Path:
    return root / "build" / ".engine_stamp"


def _write_engine_stamp(root: Path, src_hash: str) -> None:
    if not src_hash:
        return
    stamp = _engine_stamp_path(root)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(f"{src_hash}\n")


def _engine_is_stale(root: Path) -> bool:
    """True when ``root``'s ``engine.cpython-*.so`` does not match its ``lucid/_C`` sources.

    Checks ``LUCID_ABI_VERSION`` against ``_C_engine.ABI_VERSION`` and the
    recorded ``build/.engine_stamp`` source digest first so a ``git rebase``
    or checkout that bumps file mtimes without changing C++ sources (or after
    cloning a fresh ``main`` ``.so``) does not false-positive on mtime alone.
    Falls back to mtime comparison when neither stamp nor ABI probe applies.
    """
    sos = list(root.glob("lucid/_C/engine.cpython-*.so"))
    cpp_mtime = _newest_cpp_mtime(root)
    if not sos:
        return cpp_mtime > 0.0
    hdr_abi = _header_abi_version(root)
    so_abi = _so_abi_version(root)
    if hdr_abi is not None and so_abi is not None and hdr_abi != so_abi:
        return True
    stamp = _engine_stamp_path(root)
    if stamp.is_file():
        recorded = stamp.read_text().strip()
        current = _cpp_source_hash(root)
        if recorded and current:
            return recorded != current
    if hdr_abi is not None and so_abi is not None:
        return False
    return any(cpp_mtime > so.stat().st_mtime for so in sos)


def bootstrap(root: Path, main_root: Path) -> list[str]:
    """Idempotent; cheap when already done (one stat)."""
    done: list[str] = []
    if root.resolve() == main_root.resolve():
        return done
    # 1. Engine binary: an APFS clone costs no space and no time.
    cloned_so = False
    for so in main_root.glob("lucid/_C/engine.cpython-*.so"):
        dst = root / so.relative_to(main_root)
        if not dst.exists() or so.stat().st_mtime > dst.stat().st_mtime:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.unlink(missing_ok=True)
            subprocess.run(["cp", "-c", str(so), str(dst)], check=True)
            done.append(f"cloned {dst.relative_to(root)}")
            cloned_so = True
    if cloned_so and not _engine_is_stale(main_root):
        _write_engine_stamp(root, _cpp_source_hash(main_root))
    if _engine_is_stale(root) and list(root.glob("lucid/_C/engine.cpython-*.so")):
        done.append(
            "⚠ engine .so is older than lucid/_C sources — run: python3 tools/agent_ws.py build"
        )
    # 2. A light venv: `lucid` and `tools` from this worktree, then the main
    #    venv's site-packages for every dependency.  `lucid` therefore resolves
    #    to this worktree even for `python tools/x.py`, and the shared editable
    #    install is never touched.  Rewritten when stale, so an older worktree
    #    picks up a change on its next bootstrap.
    venv_py = root / ".venv/bin/python3"
    main_py = main_root / ".venv/bin/python3"
    if not venv_py.exists() and main_py.exists():
        subprocess.run(
            [str(main_py), "-m", "venv", "--without-pip", str(root / ".venv")],
            check=True,
        )
        done.append("created .venv (worktree lucid + main venv packages)")
    if venv_py.exists() and main_py.exists():
        main_site = next((main_root / ".venv/lib").glob("python3.*/site-packages"))
        site = next((root / ".venv/lib").glob("python3.*/site-packages"))
        for name, text in venv_site_files(root, main_site).items():
            if not (site / name).is_file() or (site / name).read_text() != text:
                (site / name).write_text(text)
                done.append(f"wrote .venv site-packages/{name}")
    # 3. The vault is not in git; link its folders so notes written here land
    #    in the one shared vault instead of a copy that dies with the worktree.
    vault = main_root / "obsidian"
    if vault.is_dir():
        local = root / "obsidian"
        local.mkdir(exist_ok=True)
        for entry in vault.iterdir():
            link = local / entry.name
            if entry.name == ".gitignore" or link.exists() or link.is_symlink():
                continue
            link.symlink_to(entry)
            done.append(f"linked obsidian/{entry.name}")
    return done


def venv_site_files(root: Path, main_site: Path) -> dict[str, str]:
    """The worktree venv's site-packages additions, by file name.

    A meta-path finder rather than the worktree root on ``sys.path``: mypy
    reads the interpreter's ``sys.path`` as installed packages and silences
    every error in a module it finds under one of those directories.  A root
    entry therefore hid the errors of the ``tools/`` modules ``lucid/``
    imports in worktrees only, while the main checkout (an editable *finder*)
    and CI reported them (LCD-317).
    """
    finder = (
        '"""Written by tools/agent_ws.py bootstrap: lucid and tools from this worktree."""\n'
        "\n"
        "import sys\n"
        "from importlib.machinery import PathFinder\n"
        "\n"
        f"ROOT = {str(root)!r}\n"
        "\n"
        "\n"
        "class WorktreeFinder:\n"
        "    @classmethod\n"
        "    def find_spec(cls, fullname, path=None, target=None):\n"
        '        if fullname in ("lucid", "tools"):\n'
        "            return PathFinder.find_spec(fullname, [ROOT])\n"
        "        return None\n"
        "\n"
        "\n"
        "sys.meta_path.insert(0, WorktreeFinder)\n"
    )
    pth = (
        "import _lucid_worktree_finder\n"
        f"import site; site.addsitedir({str(main_site)!r})\n"
    )
    return {"_lucid_worktree_finder.py": finder, "_lucid_worktree.pth": pth}


def needs_bootstrap(root: Path) -> bool:
    return is_linked(root) and not (root / ".venv/bin/python3").exists()


# ── subcommands ───────────────────────────────────────────────────────────────


def here() -> tuple[Path, list[Worktree]]:
    root = toplevel_of(Path.cwd())
    if root is None:
        raise WorkspaceError("not inside a git repository")
    return root, list_worktrees(root)


def cmd_status(args: argparse.Namespace) -> int:
    _, trees = here()
    rows = []
    for wt in trees:
        if wt.prunable or not wt.path.exists():
            rows.append({"worktree": str(wt.path), "missing": True})
            continue
        claims = sorted(c for c in changed_files(wt) if not is_exempt(c))
        ahead = behind = 0
        if not wt.is_main and wt.head:
            counts = git(
                "rev-list",
                "--left-right",
                "--count",
                f"{MAIN_BRANCH}...{wt.head}",
                cwd=wt.path,
            ).split()
            behind, ahead = int(counts[0]), int(counts[1])
        rows.append(
            {
                "worktree": str(wt.path),
                "name": wt.name,
                "branch": wt.ref,
                "label": label_of(wt),
                "ahead": ahead,
                "behind": behind,
                "dirty": len(dirty_files(wt.path)),
                "last_commit": git(
                    "log", "-1", "--format=%cr", wt.head, cwd=wt.path
                ).strip(),
                "claims": claims,
            }
        )
    overlaps = []
    live = [r for r in rows if not r.get("missing")]
    for i, a in enumerate(live):
        for b in live[i + 1 :]:
            shared = sorted(set(a["claims"]) & set(b["claims"]))
            if shared:
                overlaps.append({"a": a["name"], "b": b["name"], "files": shared})
    locks = {"heavy": lock_holder(HEAVY_LOCK), "land": lock_holder(LAND_LOCK)}
    if args.json:
        print(
            json.dumps(
                {"worktrees": rows, "overlaps": overlaps, "locks": locks}, indent=2
            )
        )
        return 0
    print(
        f"{'worktree':<26} {'branch':<28} {'+ahead/-behind':<15} {'dirty':>5} {'claims':>6}  task"
    )
    for r in rows:
        if r.get("missing"):
            print(
                f"{Path(r['worktree']).name:<26} (directory missing — git worktree prune)"
            )
            continue
        ab = f"+{r['ahead']}/-{r['behind']}" if r["name"] != "main checkout" else "—"
        print(
            f"{r['name'][:26]:<26} {r['branch'][:28]:<28} {ab:<15} {r['dirty']:>5} "
            f"{len(r['claims']):>6}  {r['label'] or '—'}  ({r['last_commit']})"
        )
        if args.files:
            for c in r["claims"]:
                print(f"{'':<28}· {c}")
    print()
    if overlaps:
        print("⚠️  overlapping claims — these will conflict at land:")
        for o in overlaps:
            print(f"   {o['a']} ↔ {o['b']}: {', '.join(o['files'])}")
    else:
        print("✓ no overlapping claims")
    print(f"locks: heavy={locks['heavy']} | land={locks['land']}")
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    root, _ = here()
    git("config", "--worktree", "lucid.task", args.text, cwd=root)
    print(f"labelled {root.name}: {args.text}")
    return 0


def cmd_bootstrap(args: argparse.Namespace) -> int:
    root = toplevel_of(Path(args.path) if args.path else Path.cwd())
    if root is None:
        raise WorkspaceError("not inside a git repository")
    trees = list_worktrees(root)
    done = bootstrap(root, main_root_of(trees))
    print("\n".join(done) if done else "already bootstrapped")
    return 0


def _check_engine_freshness(root: Path) -> bool:
    """Verify that the worktree's ``engine.cpython-*.so`` matches its ``lucid/_C`` sources."""
    sos = list(root.glob("lucid/_C/engine.cpython-*.so"))
    if _engine_is_stale(root):
        print("engine     ✗ engine .so is older than its sources — run: agent_ws.py build")
        return False
    if sos:
        print(f"engine     {sos[0].relative_to(root)}  ✓")
    return True


def cmd_doctor(args: argparse.Namespace) -> int:
    root, trees = here()
    main_root = main_root_of(trees)
    py = root / ".venv/bin/python3"
    print(f"worktree   {root}{' (main checkout)' if root == main_root else ''}")
    print(f"main       {main_root}")
    print(
        f"branch     {git('branch', '--show-current', cwd=root).strip() or '(detached)'}"
    )
    wt_cfg_ok = (
        git(
            "config",
            "--bool",
            "extensions.worktreeConfig",
            cwd=main_root,
            check=False,
        ).strip()
        == "true"
    )
    print(
        "wtconfig   extensions.worktreeConfig = true  ✓"
        if wt_cfg_ok
        else "wtconfig   ✗ extensions.worktreeConfig is not true — run: agent_ws.py setup"
    )
    engine_ok = _check_engine_freshness(root)
    print(f"python     {py if py.exists() else 'missing — run: agent_ws.py bootstrap'}")
    if py.exists():
        proc = subprocess.run(
            [str(py), "-c", "import lucid; print(lucid.__file__)"],
            cwd=root
            / "tools",  # deliberately not the root: proves the .pth, not the cwd
            capture_output=True,
            text=True,
            check=False,
        )
        where = proc.stdout.strip() or (
            proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
        )
        ok = where.startswith(str(root))
        print(
            f"lucid      {where}  {'✓' if ok else '✗ resolves outside this worktree'}"
        )
        return 0 if (wt_cfg_ok and engine_ok and ok) else 1
    return 1


def cmd_hook(args: argparse.Namespace) -> int:
    """PreToolUse. Fails open: a bug here must never wedge an agent."""
    try:
        payload = json.load(sys.stdin)
        return hook(payload)
    except Exception as exc:  # noqa: BLE001 — fail open, but leave a trace
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with HOOK_LOG.open("a") as log:
            log.write(f"{time.strftime('%F %T')} {type(exc).__name__}: {exc}\n")
        return 0


def hook(payload: dict[str, object]) -> int:
    tool = payload.get("tool_name")
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return 0
    if tool == "Bash":
        root = toplevel_of(Path(str(payload.get("cwd") or os.getcwd())))
        if root is not None and needs_bootstrap(root):
            bootstrap(root, main_root_of(list_worktrees(root)))
        return 0
    if tool not in EDIT_TOOLS:
        return 0
    target = tool_input.get("file_path") or tool_input.get("notebook_path")
    if not target:
        return 0
    path = Path(str(target))
    root = toplevel_of(path)
    if root is None:
        return 0
    trees = list_worktrees(root)
    if needs_bootstrap(root):
        bootstrap(root, main_root_of(trees))
    if len(trees) < 2:
        return 0
    me = find_tree(trees, root)
    try:
        rel = str(path.absolute().relative_to(root))
    except ValueError:
        return 0
    verdict = judge(rel, me, trees)
    if verdict.clear:
        return 0
    decision = "ask" if me.is_main else "deny"
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": decision,
                    "permissionDecisionReason": explain(verdict),
                }
            }
        )
    )
    return 0


def cmd_check_commit(args: argparse.Namespace) -> int:
    if os.environ.get("LUCID_SKIP_CLAIMS") == "1":
        return 0
    root, trees = here()
    if len(trees) < 2:
        return 0
    me = find_tree(trees, root)
    # The one query that must see the hook's environment: what is being
    # committed lives in the index git handed over as GIT_INDEX_FILE — for
    # `git commit -- <path>` a temporary one, while the worktree's own index
    # may stage something else entirely.  Everything else asks about a tree's
    # ordinary state and runs on repo_env().
    staged = git(
        "diff",
        "--cached",
        "--name-only",
        "--no-renames",
        cwd=root,
        env=dict(os.environ),
    ).split()
    blocked = [v for v in (judge(rel, me, trees) for rel in staged) if not v.clear]
    if not blocked:
        return 0
    print("  ❌  Commit blocked by the parallel-agent claim gate:", file=sys.stderr)
    for v in blocked:
        print(f"      · {explain(v)}", file=sys.stderr)
    print("      Board:     python3 tools/agent_ws.py status --files", file=sys.stderr)
    print(
        "      Override:  LUCID_SKIP_CLAIMS=1 git commit ...  (breaks the no-conflict "
        "guarantee for that file)",
        file=sys.stderr,
    )
    return 1


def resolve_target(trees: list[Worktree], spec: str) -> Worktree:
    for wt in trees:
        if spec in (str(wt.path), wt.path.name, wt.branch):
            return wt
    candidate = Path(spec).expanduser()
    if candidate.exists():
        return find_tree(trees, candidate.resolve())
    raise WorkspaceError(f"no worktree matches {spec!r} (try: agent_ws.py status)")


def user_facing(subject: str) -> bool:
    return re.match(USER_FACING_RE, subject) is not None


def cmd_land(args: argparse.Namespace) -> int:
    _, trees = here()
    main = trees[0]
    target = resolve_target(trees, args.worktree)
    if target.is_main:
        raise WorkspaceError("land takes a linked worktree, not the main checkout")
    if main.branch != MAIN_BRANCH:
        raise WorkspaceError(
            f"the main checkout is on {main.ref!r}, not {MAIN_BRANCH!r}"
        )
    wt = target.path
    if dirty_files(wt):
        raise WorkspaceError(
            f"{wt.name} has uncommitted changes — the worker must commit (or discard) first:\n"
            + git("status", "--short", cwd=wt)
        )
    commits = git("rev-list", "--reverse", f"{MAIN_BRANCH}..HEAD", cwd=wt).split()
    if not commits:
        print(f"{wt.name}: nothing to land (no commits ahead of {MAIN_BRANCH})")
        return 0

    venv_py = wt / ".venv/bin/python3"
    py = str(venv_py) if venv_py.exists() else sys.executable

    if args.check:
        proc = subprocess.run(
            ["git", "merge-tree", "--write-tree", "--name-only", MAIN_BRANCH, "HEAD"],
            cwd=wt,
            env=repo_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        print(f"{wt.name}: {len(commits)} commit(s) ahead")
        if proc.returncode != 0:
            # --name-only: tree OID, then one conflicted path per line, then a blank line.
            lines = proc.stdout.splitlines()[1:]
            conflicted = lines[: lines.index("")] if "" in lines else lines
            print("✗ would conflict: " + ", ".join(conflicted or ["(see git merge-tree)"]))
            return 3
        print("✓ merges cleanly onto main")
        # The branch's own delta: against where it forked, so main's later
        # fixes do not read as this branch's slop.
        fork = git("merge-base", MAIN_BRANCH, "HEAD", cwd=wt).strip()
        touched_fork = diff_names(wt, fork, "HEAD")
        inconsistent = audit_baseline_errors(wt, touched_fork)
        if inconsistent:
            print(f"✗ {inconsistent}")
            return BASELINE_EXIT
        slop_rc = judge_slop(wt, py, fork, args.allow_slop)
        cov_rc, cov_out = diff_coverage_report(wt, py, fork, touched_fork)
        if cov_out:
            print(cov_out)
        return slop_rc or cov_rc

    ensure_merge_drivers(main.path)
    with held_lock(LAND_LOCK, f"land {wt.name}", wait=True):
        # 1. Rebase onto the current main.  Claims make this conflict-free except
        #    for generated files, which the merge driver settles (setup).
        rebase = subprocess.run(
            ["git", "rebase", MAIN_BRANCH],
            cwd=wt,
            env=repo_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        if rebase.returncode != 0:
            conflicted = git("diff", "--name-only", "--diff-filter=U", cwd=wt).split()
            git("rebase", "--abort", cwd=wt, check=False)
            raise WorkspaceError(
                f"rebase onto {MAIN_BRANCH} conflicted in {', '.join(conflicted) or '?'} — "
                "aborted, nothing changed. Resume the worker to rebase and resolve.",
                code=3,
            )
        if needs_bootstrap(wt):
            bootstrap(wt, main.path)
            py = str(venv_py) if venv_py.exists() else sys.executable
        touched = diff_names(wt, MAIN_BRANCH, "HEAD")

        # 2. Stubs must match the combined source; a stale stub fails CI.
        if not args.skip_stub_check and any(
            fnmatch.fnmatch(p, "lucid/*") and p.endswith((".py", ".cpp", ".h", ".mm"))
            for p in touched
        ):
            stubs = subprocess.run(
                [py, "tools/check_stubs.py"],
                cwd=wt,
                capture_output=True,
                text=True,
                check=False,
            )
            if stubs.returncode != 0:
                raise WorkspaceError(
                    "stubs are stale after the rebase. In the worktree run "
                    "`.venv/bin/python3 tools/gen_pyi.py` (C++ changes need "
                    "`agent_ws.py build` first), commit, then land again.\n"
                    + (stubs.stdout + stubs.stderr)[-1500:],
                    code=4,
                )

        # 2a. Audit baselines: the merge driver keeps one side whole, so only a
        #     hand edit or a commit with hooks off leaves a total disagreeing
        #     with its body — and a unit test red on main (LCD-358).
        inconsistent = audit_baseline_errors(wt, touched)
        if inconsistent:
            raise WorkspaceError(inconsistent, code=BASELINE_EXIT)

        # 2b. Quality ratchet: the rebased branch against main (now its fork
        #     point) — no measured count may rise, and the actual counts may
        #     not exceed the baseline the rebase merged.  tools/README.md.
        if judge_slop(wt, py, MAIN_BRANCH, args.allow_slop):
            raise WorkspaceError(
                "quality gate refused the land (SLOP DELTA above)", code=SLOP_EXIT
            )

        # 3. CHANGELOG: on branches the post-commit hook stands down, so the
        #    entries are folded in here — replaying the branch with an exec
        #    after every commit gives the same history post-commit would have
        #    (each user-facing commit carries its own entry).  The branch already
        #    sits on main and the lock keeps main still, so this cannot conflict.
        if not args.no_changelog:
            fold = f"{shlex.quote(sys.executable)} {shlex.quote(str(SELF))} _fold-changelog"
            replay = subprocess.run(
                ["git", "rebase", "--force-rebase", "--exec", fold, MAIN_BRANCH],
                cwd=wt,
                env=repo_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            if replay.returncode != 0:
                git("rebase", "--abort", cwd=wt, check=False)
                raise WorkspaceError(
                    f"CHANGELOG fold failed:\n{replay.stderr.strip()[-1500:]}", code=3
                )

        # 4. Fast-forward main.  Never a merge commit, so never a merge conflict.
        tip = git("rev-parse", "HEAD", cwd=wt).strip()
        if not git_ok("merge-base", "--is-ancestor", MAIN_BRANCH, tip, cwd=wt):
            raise WorkspaceError("main moved during land; run land again", code=5)
        before = git("rev-parse", MAIN_BRANCH, cwd=main.path).strip()
        ff = subprocess.run(
            ["git", "merge", "--ff-only", tip],
            cwd=main.path,
            env=repo_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        if ff.returncode != 0:
            raise WorkspaceError(
                "fast-forward of main failed (uncommitted edits in the main checkout touch the "
                f"same files?):\n{ff.stderr.strip()}",
                code=5,
            )
        landed = git("log", "--oneline", f"{before}..{tip}", cwd=main.path)

    print(f"✓ landed {wt.name} onto {MAIN_BRANCH} (not pushed):")
    print(landed.rstrip())
    if any(p.startswith("lucid/_C/") for p in touched):
        print(
            "⚠️  C++ changed: the main checkout's engine .so is now older than its source."
        )
        print(
            "    Rebuild main before relying on it (uv pip install -e ., see CLAUDE.md)."
        )
    if args.cleanup:
        removed = subprocess.run(
            ["git", "worktree", "remove", str(wt)],
            cwd=main.path,
            env=repo_env(),
            capture_output=True,
            text=True,
        )
        if removed.returncode == 0 and target.branch:
            git("branch", "-d", target.branch, cwd=main.path, check=False)
            print(f"removed worktree {wt.name} and branch {target.branch}")
        else:
            print(f"kept worktree {wt.name}: {removed.stderr.strip()}")
    return 0


def cmd_fold_changelog(args: argparse.Namespace) -> int:
    """Run by `land` after each replayed commit: post-commit's job, on a branch."""
    try:
        root = toplevel_of(Path.cwd())
        tool = root / "tools/changelog.py" if root else None
        if root is None or tool is None or not tool.exists():
            return 0
        if not user_facing(git("log", "-1", "--format=%s", cwd=root).strip()):
            return 0
        if (
            "CHANGELOG.md"
            in git("show", "--name-only", "--format=", "HEAD", cwd=root).split()
        ):
            return 0
        venv_py = root / ".venv/bin/python3"
        py = str(venv_py) if venv_py.exists() else sys.executable
        with tempfile.NamedTemporaryFile("w", suffix=".msg", delete=False) as msg:
            msg.write(git("log", "-1", "--format=%B", cwd=root))
        subprocess.run(
            [py, str(tool), "propose", "--auto", "--message-file", msg.name],
            cwd=root,
            capture_output=True,
            check=False,
        )
        Path(msg.name).unlink(missing_ok=True)
        if "CHANGELOG.md" in dirty_files(root):
            env = {**os.environ, "LUCID_CHANGELOG_AMEND_IN_PROGRESS": "1"}
            git("add", "CHANGELOG.md", cwd=root)
            git("commit", "--amend", "--no-edit", "--no-verify", cwd=root, env=env)
    except Exception as exc:  # noqa: BLE001 — never stop the replay over an entry
        print(f"[agent_ws] changelog fold skipped: {exc}", file=sys.stderr)
    return 0


def cmd_heavy(args: argparse.Namespace) -> int:
    command = args.cmd[1:] if args.cmd[:1] == ["--"] else list(args.cmd)
    if not command:
        raise WorkspaceError("usage: agent_ws.py heavy -- CMD...")
    with held_lock(HEAVY_LOCK, " ".join(command)[:120], wait=not args.no_wait):
        return subprocess.call(command)


def mlx_macos_floor(root: Path) -> str | None:
    """The deployment floor of the tree being built: its pinned MLX's minos.

    Read from that tree's own pyproject, so a branch that predates the pin
    keeps its setup.py default.
    """
    try:
        with (root / "pyproject.toml").open("rb") as fh:
            floor = tomllib.load(fh).get("tool", {}).get("lucid", {}).get("mlx", {}).get("macos")
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return floor if isinstance(floor, str) else None


def cmd_build(args: argparse.Namespace) -> int:
    root, trees = here()
    main_root = main_root_of(trees)
    if root == main_root:
        raise WorkspaceError(
            "the main checkout builds through `uv pip install -e .` (CLAUDE.md); "
            "`build` is for worktrees, where an editable install would hijack the shared venv"
        )
    env = dict(os.environ)
    floor = mlx_macos_floor(root)
    if floor is not None:
        env.setdefault("MACOSX_DEPLOYMENT_TARGET", floor)
    if XCODE_SDK.exists():  # MacBook: the CLT 27 SDK breaks Xcode 26's ld
        env.setdefault("SDKROOT", str(XCODE_SDK))
    env.setdefault("CCACHE_BASEDIR", str(main_root))
    env["PATH"] = f"{main_root / '.venv/bin'}:{env.get('PATH', '')}"
    command = [
        str(main_root / ".venv/bin/python3"),
        "setup.py",
        "build_ext",
        "--inplace",
        "--build-temp",
        "build/agent",
    ]
    with held_lock(HEAVY_LOCK, f"build {root.name}", wait=True):
        rc = subprocess.call(command, cwd=root, env=env)
    if rc == 0:
        _write_engine_stamp(root, _cpp_source_hash(root))
    return rc


def ensure_merge_drivers(root: Path) -> list[str]:
    """Register the merge drivers in repo-local config; idempotent.  Returns what changed."""
    changed = []
    for key, value in MERGE_DRIVERS.items():
        if git("config", "--local", "--get", key, cwd=root, check=False).strip() != value:
            git("config", "--local", key, value, cwd=root)
            changed.append(key)
    return changed


def cmd_setup(args: argparse.Namespace) -> int:
    root, _ = here()
    settings = {
        "extensions.worktreeConfig": "true",
        "rerere.enabled": "true",
        "rerere.autoupdate": "true",
        **MERGE_DRIVERS,
    }
    for key, value in settings.items():
        git("config", "--local", key, value, cwd=root)
        print(f"git config {key} = {value}")
    return 0


def _measurable_lucid_py(touched: set[str]) -> list[str]:
    """Changed Python source files under ``lucid/`` subject to coverage."""
    return sorted(
        p
        for p in touched
        if p.startswith("lucid/")
        and p.endswith(".py")
        and not p.startswith(COVERAGE_OMIT_PREFIXES)
    )


def _coverage(py: str, wt: Path, *argv: str, env: dict[str, str] | None = None) -> str:
    """Run ``coverage <argv>`` in *wt*: empty on success, else why, in one line."""
    proc = subprocess.run(
        [py, "-m", "coverage", *argv],
        cwd=wt,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return ""
    lines = (proc.stderr or proc.stdout).strip().splitlines()
    return f"coverage {argv[0]} exited {proc.returncode}" + (
        f": {lines[-1]}" if lines else ""
    )


_COVERAGE_XMLS = ("build/coverage/coverage.xml", "coverage.xml")


def _has_coverage_data(wt: Path) -> bool:
    return (wt / ".coverage").is_file() or any(wt.glob(".coverage.*"))


def _touched_tests(wt: Path, touched: set[str]) -> list[str]:
    """The changed test files a coverage run of *wt* would execute."""
    if not (wt / "pyproject.toml").is_file():
        return []
    return sorted(
        p
        for p in touched
        if p.startswith("lucid/test/") and p.endswith(".py") and (wt / p).is_file()
    )


def _nothing_to_measure_from(wt: Path, touched: set[str]) -> bool:
    """No coverage.xml, no coverage data, and no touched test to produce one."""
    return (
        not any((wt / rel).is_file() for rel in _COVERAGE_XMLS)
        and not _has_coverage_data(wt)
        and not _touched_tests(wt, touched)
    )


def _resolve_coverage_xml(
    wt: Path, py: str, touched: set[str]
) -> tuple[Path | None, str]:
    """Locate or generate a Cobertura ``coverage.xml`` for *wt*.

    ``(xml, "")``, or ``(None, why)`` naming the step that could not
    measure.  Called once :func:`_nothing_to_measure_from` said there is a
    source to measure from.
    """
    for rel in _COVERAGE_XMLS:
        cand = wt / rel
        if cand.is_file():
            return cand, ""
    out_xml = STATE_DIR / f"coverage-{wt.name}.xml"
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    why = "no coverage data"
    if _has_coverage_data(wt):
        if any(wt.glob(".coverage.*")):
            _coverage(py, wt, "combine")  # a failure shows in the xml step
        why = _coverage(py, wt, "xml", "-o", str(out_xml))
        if not why and out_xml.is_file():
            return out_xml, ""
    tests = _touched_tests(wt, touched)
    if not tests:
        return None, why
    data_file = STATE_DIR / f".coverage-{wt.name}"
    data_file.unlink(missing_ok=True)
    env = {**repo_env(), "COVERAGE_FILE": str(data_file)}
    try:
        why = _coverage(py, wt, "run", "--source=lucid", "-m", "pytest", "-q", *tests, env=env)
        if not why:
            why = _coverage(py, wt, "xml", "-o", str(out_xml), env=env)
    finally:
        data_file.unlink(missing_ok=True)
    if not why and out_xml.is_file():
        return out_xml, ""
    return None, why or "the touched tests left no coverage.xml"


def _diff_cover_unimportable(py: str) -> str:
    """Why *py* cannot import diff-cover; empty when it can."""
    proc = subprocess.run(
        [py, "-c", "import diff_cover.diff_cover_tool"],
        env=repo_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return ""
    lines = proc.stderr.strip().splitlines()
    return lines[-1] if lines else f"exit {proc.returncode}"


def diff_coverage_report(
    wt: Path, py: str, base: str, touched: set[str]
) -> tuple[int, str]:
    """Run ``diff-cover`` on *wt* against *base* when ``lucid/*.py`` changed (LCD-269).

    Exit 0 when there is nothing to measure: no changed ``lucid/*.py``, or no
    coverage data and no touched test to produce it — a property of the
    branch, reported as a note.  An attempt that could not measure (a tool
    missing, the coverage run or diff-cover failing) is
    DIFF_COVER_UNMEASURED_EXIT, never a pass and never "coverage below the
    threshold"; only diff-cover's own verdict is DIFF_COVER_EXIT.
    """
    measured = _measurable_lucid_py(touched)
    if not measured:
        return 0, "(no covered lucid/*.py changes — diff-cover skipped)"
    if _nothing_to_measure_from(wt, touched):
        return (
            0,
            f"(diff-cover: {len(measured)} lucid/*.py file(s) changed; no coverage.xml or touched tests to measure)",
        )
    # Checked before the XML, which may mean a coverage run of the touched tests.
    why = _diff_cover_unimportable(py)
    if why:
        return (
            DIFF_COVER_UNMEASURED_EXIT,
            f"✗ diff-cover cannot measure: {py} cannot import it ({why}). It is in "
            "the `quality` dependency group; this is not a coverage verdict.",
        )
    xml_path, failed = _resolve_coverage_xml(wt, py, touched)
    if xml_path is None:
        return (
            DIFF_COVER_UNMEASURED_EXIT,
            f"✗ diff-cover cannot measure: {failed}; this is not a coverage verdict.",
        )
    proc = subprocess.run(
        [
            py,
            "-m",
            "diff_cover.diff_cover_tool",
            str(xml_path),
            f"--compare-branch={base}",
            f"--fail-under={DIFF_COVER_THRESHOLD}",
            "--show-uncovered",
            "--include",
            "lucid/**",
            "--exclude",
            "lucid/test/**",
            "lucid/models/**",
            "lucid/benchmarks/**",
            "**/*.pyi",
        ],
        cwd=wt,
        env=repo_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    out = (proc.stdout + proc.stderr).rstrip()
    if proc.returncode == 0:
        return 0, out
    if DIFF_COVER_BELOW in out:
        return (
            DIFF_COVER_EXIT,
            f"{out}\n✗ diff-cover: changed-line coverage is below {DIFF_COVER_THRESHOLD}%",
        )
    return (
        DIFF_COVER_UNMEASURED_EXIT,
        f"{out}\n✗ diff-cover cannot measure (exit {proc.returncode}); this is not a "
        "coverage verdict.",
    )


def quality_delta(wt: Path, py: str, base: str) -> tuple[int, str]:
    """Run the ratchet gate's ``--diff`` in *wt* against *base*: (exit code, output).

    Exit 0 = no count rose, 1 = slop added (or over the baseline), 2 = the
    gate could not measure.  A branch from before the gate existed has
    nothing to run and passes with a note.
    """
    if not (wt / "tools/quality_gate/__main__.py").exists():
        return 0, "(no tools/quality_gate on this branch — skipped)"
    proc = subprocess.run(
        [py, "-m", "tools.quality_gate", "--diff", base],
        cwd=wt,
        env=repo_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    # stderr is progress on success; on failure it carries the reason.
    detail = "" if proc.returncode in (0, 1) else "\n" + proc.stderr[-3000:]
    return proc.returncode, (proc.stdout + detail).rstrip()


def audit_baseline_errors(wt: Path, touched: set[str]) -> str:
    """What is wrong with the audit baselines *wt* changed; empty when they agree.

    The checker beside this script (the main checkout's, stdlib only) reads
    *wt*'s files, so a branch cut before it existed is judged too.  *wt* is
    clean whenever `land` asks, so its files are its HEAD.
    """
    if not touched.intersection(AUDIT_BASELINES) or not BASELINE_CHECK.exists():
        return ""
    proc = subprocess.run(
        [sys.executable, str(BASELINE_CHECK), "--root", str(wt)],
        capture_output=True,
        text=True,
        check=False,
    )
    return "" if proc.returncode == 0 else (proc.stdout + proc.stderr).strip()


def judge_slop(wt: Path, py: str, base: str, allow: str | None) -> int:
    """Print the SLOP DELTA; 0 to go on, SLOP_EXIT to refuse.  ``allow`` overrides, logged."""
    code, output = quality_delta(wt, py, base)
    print(output)
    if code == 0:
        return 0
    if allow:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with SLOP_LOG.open("a") as log:
            log.write(f"{stamp}\t{wt.name}\texit {code}\t{allow}\n")
        print(f"⚠️  quality gate overridden (--allow-slop): {allow!r} — logged to {SLOP_LOG}")
        return 0
    why = "added slop" if code == 1 else "could not measure"
    print(
        f"✗ the quality gate {why} (exit {code}). The worker fixes it at the owner, or the "
        "orchestrator lands anyway with --allow-slop \"<reason>\"."
    )
    return SLOP_EXIT


def main(argv: list[str] | None = None) -> int:
    summary = (__doc__ or "").strip().split("\n\n")[0]
    parser = argparse.ArgumentParser(prog="agent_ws.py", description=summary)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("status", help="board of worktrees, claims and locks")
    p.add_argument(
        "--files", action="store_true", help="list each worktree's claimed files"
    )
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)
    p = sub.add_parser("label", help="tag the current worktree with its task")
    p.add_argument("text")
    p.set_defaults(func=cmd_label)
    p = sub.add_parser("bootstrap", help="make a worktree runnable")
    p.add_argument("path", nargs="?")
    p.set_defaults(func=cmd_bootstrap)
    sub.add_parser("doctor", help="where does `import lucid` resolve").set_defaults(
        func=cmd_doctor
    )
    p = sub.add_parser("land", help="rebase a worktree onto main and fast-forward main")
    p.add_argument("worktree", help="worktree path, directory name, or branch")
    p.add_argument("--check", action="store_true", help="report only; change nothing")
    p.add_argument(
        "--cleanup", action="store_true", help="remove the worktree and branch after"
    )
    p.add_argument("--skip-stub-check", action="store_true")
    p.add_argument("--no-changelog", action="store_true")
    p.add_argument(
        "--allow-slop",
        metavar="REASON",
        help="land despite a quality-gate increase (orchestrator only; logged)",
    )
    p.set_defaults(func=cmd_land)
    p = sub.add_parser("heavy", help="run a command under the heavy-job lock")
    p.add_argument("--no-wait", action="store_true", help="fail instead of queueing")
    p.add_argument("cmd", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_heavy)
    sub.add_parser("build", help="build the engine in place (worktrees)").set_defaults(
        func=cmd_build
    )
    sub.add_parser("hook", help="Claude Code PreToolUse entry").set_defaults(
        func=cmd_hook
    )
    sub.add_parser("check-commit", help="git pre-commit entry").set_defaults(
        func=cmd_check_commit
    )
    sub.add_parser("setup", help="idempotent git config").set_defaults(func=cmd_setup)
    sub.add_parser("_fold-changelog").set_defaults(
        func=cmd_fold_changelog
    )  # land-internal
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except WorkspaceError as exc:
        print(f"[agent_ws] {exc}", file=sys.stderr)
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
