"""Counts, the baseline, and the comparisons the ratchet is made of.

A finding is reduced to a key ``(tool, rule, file)`` and the gate keeps only
how many there are.  Counting instead of fingerprinting lines means an edit
that shifts a finding down the file is not new debt; the price is that
swapping one finding for another of the same rule in the same file is
invisible — the count, not the site, is what may not rise.
"""

import datetime
import functools
import json
import os
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from tools.quality_gate import config

Key = tuple[str, str, str]  # (tool, rule, file)
Counts = dict[Key, int]


class GateError(Exception):
    """The gate could not measure (a tool failed); never a pass."""


# ── git ───────────────────────────────────────────────────────────────────


@functools.cache
def git_env() -> dict[str, str]:
    """This environment minus git's repository-local variables.

    A git hook exports GIT_INDEX_FILE / GIT_DIR (agent_ws.repo_env has the
    story); they outrank ``cwd`` and would aim a query of another tree — the
    ``--root`` of a test repository — at the committing one.  git's own list.
    """
    bare = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    proc = subprocess.run(
        ["git", "rev-parse", "--local-env-vars"],
        cwd="/",
        env=bare,
        capture_output=True,
        text=True,
        check=False,
    )
    local = set(proc.stdout.split()) | {"GIT_INDEX_FILE", "GIT_DIR"}
    return {k: v for k, v in os.environ.items() if k not in local}


def git(root: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=root, env=git_env(), capture_output=True, text=True, check=False
    )
    if check and proc.returncode != 0:
        raise GateError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def is_git_tree(root: Path) -> bool:
    return (root / ".git").exists()


def tracked_files(root: Path) -> list[str]:
    """Tracked plus untracked-but-not-ignored files; a plain walk off git."""
    if is_git_tree(root):
        out = git(root, "ls-files", "-co", "--exclude-standard", "-z")
        return sorted({p for p in out.split("\0") if p and (root / p).is_file()})
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", ".venv")]
        found += [(Path(dirpath) / name).relative_to(root).as_posix() for name in filenames]
    return sorted(found)


def changed_files(root: Path, ref: str) -> list[str]:
    """Paths that differ between *ref* and the working tree (both sides of a rename)."""
    out = git(root, "diff", "--name-only", "--no-renames", "-z", ref)
    names = {p for p in out.split("\0") if p}
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "-z")
    names |= {p for p in untracked.split("\0") if p}
    return sorted(names)


def renames(root: Path, ref: str) -> dict[str, str]:
    """old path -> new path for files git sees as moved since *ref*."""
    out = git(root, "diff", "-M", "--name-status", "-z", ref, check=False)
    parts = out.split("\0")
    moved: dict[str, str] = {}
    i = 0
    while i < len(parts) and parts[i]:
        status = parts[i]
        if status.startswith(("R", "C")):
            if status.startswith("R"):
                moved[parts[i + 1]] = parts[i + 2]
            i += 3
        else:
            i += 2
    return moved


def baseline_commit(root: Path) -> str | None:
    """The last commit that wrote the baseline — renames are judged from there."""
    if not is_git_tree(root):
        return None
    sha = git(root, "log", "-1", "--format=%H", "--", config.BASELINE_PATH, check=False).strip()
    return sha or None


# ── file scope ────────────────────────────────────────────────────────────


def is_py_target(path: str) -> bool:
    return (
        path.startswith(config.PY_ROOT)
        and path.endswith(".py")
        and not path.startswith(config.PY_EXCLUDE_PREFIXES)
    )


def is_cpp_target(path: str) -> bool:
    return path.startswith(config.CPP_ROOT) and path.endswith(config.CPP_SUFFIXES)


@dataclass
class Scope:
    """What one collector run reads: a tree, its target files, a scratch directory."""

    root: Path
    py: list[str]
    cpp: list[str]
    workdir: Path

    @property
    def files(self) -> list[str]:
        return self.py + self.cpp


def scope_of(root: Path, files: Iterable[str], workdir: Path) -> Scope:
    present = [f for f in files if (root / f).is_file()]
    return Scope(
        root=root,
        py=sorted(f for f in present if is_py_target(f)),
        cpp=sorted(f for f in present if is_cpp_target(f)),
        workdir=workdir,
    )


# ── baseline ──────────────────────────────────────────────────────────────


@dataclass
class Baseline:
    counts: Counts
    tools: dict[str, str] = field(default_factory=dict)
    thresholds: dict[str, object] = field(default_factory=dict)
    rebaselines: list[dict[str, str]] = field(default_factory=list)


def counts_from_nested(nested: Mapping[str, Mapping[str, Mapping[str, int]]]) -> Counts:
    return {
        (tool, rule, path): int(n)
        for tool, rules in nested.items()
        for rule, files in rules.items()
        for path, n in files.items()
        if int(n) > 0
    }


def nested_from_counts(counts: Counts) -> dict[str, dict[str, dict[str, int]]]:
    nested: dict[str, dict[str, dict[str, int]]] = {}
    for (tool, rule, path), n in sorted(counts.items()):
        if n > 0:
            nested.setdefault(tool, {}).setdefault(rule, {})[path] = n
    return nested


def baseline_from_json(data: Mapping[str, object]) -> Baseline:
    version = data.get("version")
    if version != config.BASELINE_VERSION:
        raise GateError(f"baseline version {version!r}, expected {config.BASELINE_VERSION}")
    counts = data.get("counts", {})
    tools = data.get("tools", {})
    thresholds = data.get("thresholds", {})
    history = data.get("rebaselines", [])
    if not (isinstance(counts, dict) and isinstance(tools, dict)):
        raise GateError("baseline: malformed counts/tools")
    if not (isinstance(thresholds, dict) and isinstance(history, list)):
        raise GateError("baseline: malformed thresholds/rebaselines")
    return Baseline(counts_from_nested(counts), dict(tools), dict(thresholds), list(history))


def load_baseline(path: Path) -> Baseline:
    if not path.exists():
        return Baseline({})
    try:
        return baseline_from_json(json.loads(path.read_text()))
    except json.JSONDecodeError as exc:
        raise GateError(f"{path}: not JSON ({exc})") from exc


def baseline_to_json(baseline: Baseline) -> str:
    data = {
        "version": config.BASELINE_VERSION,
        "_about": (
            "Ratchet baseline for tools/quality_gate (tools/README.md). Counts may only "
            "go down: lower them with `python -m tools.quality_gate --update`; raising "
            "one is `--rebaseline --reason` (orchestrator only)."
        ),
        "tools": dict(sorted(baseline.tools.items())),
        "thresholds": dict(sorted(baseline.thresholds.items())),
        "rebaselines": baseline.rebaselines,
        "counts": nested_from_counts(baseline.counts),
    }
    return json.dumps(data, indent=1, sort_keys=False) + "\n"


def save_baseline(path: Path, baseline: Baseline) -> None:
    path.write_text(baseline_to_json(baseline))


def rebaseline_entry(reason: str) -> dict[str, str]:
    return {
        "date": datetime.date.today().isoformat(),
        "reason": reason,
        "by": os.environ.get("USER", "?"),
    }


# ── comparison ────────────────────────────────────────────────────────────


def apply_renames(counts: Counts, moved: Mapping[str, str]) -> Counts:
    out: Counts = defaultdict(int)
    for (tool, rule, path), n in counts.items():
        out[(tool, rule, moved.get(path, path))] += n
    return dict(out)


def restrict(counts: Counts, tools: Iterable[str], files: set[str] | None) -> Counts:
    """The part of *counts* a run measured: its tools, and its files unless complete."""
    wanted = set(tools)
    return {k: n for k, n in counts.items() if k[0] in wanted and (files is None or k[2] in files)}


@dataclass
class Delta:
    before: Counts
    after: Counts

    @property
    def increases(self) -> dict[Key, tuple[int, int]]:
        return {
            k: (self.before.get(k, 0), n)
            for k, n in sorted(self.after.items())
            if n > self.before.get(k, 0)
        }

    @property
    def decreases(self) -> dict[Key, tuple[int, int]]:
        keys = sorted(set(self.before) | set(self.after))
        return {
            k: (self.before.get(k, 0), self.after.get(k, 0))
            for k in keys
            if self.after.get(k, 0) < self.before.get(k, 0)
        }


def class_totals(counts: Counts) -> dict[str, int]:
    """Findings per class; a per-file measure (lines over the cap) counts as one finding."""
    totals = dict.fromkeys(config.CLASSES, 0)
    for (tool, rule, _), n in counts.items():
        totals[config.class_of(tool, rule)] += 1 if rule in config.PER_FILE_RULES else n
    return totals


def rule_totals(counts: Counts) -> dict[tuple[str, str], int]:
    totals: dict[tuple[str, str], int] = defaultdict(int)
    for (tool, rule, _), n in counts.items():
        totals[(tool, rule)] += n
    return dict(totals)


# ── 3-way merge of two baselines (git merge driver) ───────────────────────


class MergeRefused(Exception):
    """A merge that would raise a count without a recorded rebaseline."""


def merge_baselines(base: Baseline, ours: Baseline, theirs: Baseline) -> Baseline:
    """Per key: a side that did not change it defers to the other; both changed -> min,
    unless one side carries a rebaseline, where deltas combine (``o + t - b``).

    A missing key is a count of 0, so a key deleted on either side (file gone,
    count reached zero) stays deleted unless the other side rebaselined it.
    A side may raise a key above *base* only if it also carries a new
    ``--rebaseline`` record; otherwise the merge is refused rather than
    resolved, because a raise the ratchet did not authorise must not slip in
    through a merge.
    """
    base_hist = {json.dumps(e, sort_keys=True) for e in base.rebaselines}
    ours_rebased = any(json.dumps(e, sort_keys=True) not in base_hist for e in ours.rebaselines)
    theirs_rebased = any(json.dumps(e, sort_keys=True) not in base_hist for e in theirs.rebaselines)
    merged: Counts = {}
    for k in sorted(set(base.counts) | set(ours.counts) | set(theirs.counts)):
        b, o, t = base.counts.get(k, 0), ours.counts.get(k, 0), theirs.counts.get(k, 0)
        if o > b and not ours_rebased:
            raise MergeRefused(f"ours raises {k} {b} -> {o} without a rebaseline")
        if t > b and not theirs_rebased:
            raise MergeRefused(f"theirs raises {k} {b} -> {t} without a rebaseline")
        if o == b:
            value = t
        elif t == b:
            value = o
        elif ours_rebased != theirs_rebased:
            value = max(0, o + t - b)
        else:
            value = min(o, t)
        if value > 0:
            merged[k] = value
    history: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in [*ours.rebaselines, *theirs.rebaselines]:
        tag = json.dumps(entry, sort_keys=True)
        if tag not in seen:
            seen.add(tag)
            history.append(entry)
    newer = theirs if theirs_rebased and not ours_rebased else ours
    return Baseline(merged, dict(newer.tools), dict(newer.thresholds), history)


def baseline_raises(root: Path, current: Baseline, ref: str = "HEAD") -> dict[Key, tuple[int, int]]:
    """Keys raised in *current* above *ref*'s baseline without a new ``rebaselines`` entry."""
    if not is_git_tree(root):
        return {}
    spec = f"{ref}:{config.BASELINE_PATH}"
    if git(root, "cat-file", "-t", spec, check=False).strip() != "blob":
        return {}
    raw = git(root, "show", spec)
    try:
        prior = baseline_from_json(json.loads(raw))
    except json.JSONDecodeError as exc:
        raise GateError(f"{spec}: not JSON ({exc})") from exc
    except GateError as exc:
        raise GateError(f"{spec}: {exc}") from exc
    if current.counts == prior.counts:
        return {}
    prior_hist = {json.dumps(e, sort_keys=True) for e in prior.rebaselines}
    if any(json.dumps(e, sort_keys=True) not in prior_hist for e in current.rebaselines):
        return {}
    since = git(root, "rev-parse", "--verify", f"{ref}^{{commit}}", check=False).strip()
    moved = renames(root, since) if since else {}
    prior_counts = prior.counts | apply_renames(prior.counts, moved)
    return {
        k: (prior_counts.get(k, 0), n)
        for k, n in sorted(current.counts.items())
        if n > prior_counts.get(k, 0)
    }

