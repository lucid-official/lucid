"""What the zoo stage remembers: results per commit, and the notes written from them.

Two records, deliberately separate:

* **state** — one JSON per family under
  ``~/Library/Caches/lucid/audit-zoo/<sha>/``.  It exists so an interrupted run
  resumes: a family recorded for this commit is not run again.  It lives in the
  cache directory, not the checkout, because the stage must leave the tree as it
  found it.  A tree with uncommitted changes never reuses a record — the commit
  it would be filed under no longer describes what is on disk.
* **notes** — ``zoo-verify-<family>.md`` plus ``zoo-verify-ledger.md``, written
  only where the operator points (``--zoo-record``) or the repository has its
  vault.  They are for people: the ledger is read *before* a run to avoid
  repeating one.
"""

import dataclasses
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from lucid.test.audit._zoo_stage import ZOO_WIDE, StageResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lucid.test.audit._zoo_budget import MachineBudget

STATE_ROOT = Path.home() / "Library" / "Caches" / "lucid" / "audit-zoo"


def overall(results: Sequence[StageResult]) -> str:
    """The worst status among ``results``, or ``pass``."""
    statuses = {result.status for result in results}
    for worst in ("fail", "timeout", "mem-guard"):
        if worst in statuses:
            return worst
    return "pass"


def repo_state(root: Path) -> tuple[str, bool]:
    """``(HEAD sha, whether tracked files differ from it)``."""
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return sha, bool(status.strip())


class ZooState:
    """Per-commit results, so an interrupted run resumes where it stopped."""

    def __init__(self, base: Path, sha: str, *, dirty: bool, resume: bool) -> None:
        self.sha = sha
        self.dirty = dirty
        # A dirty tree writes under its own name so it cannot be mistaken for
        # the clean commit later, and never reads anything back.
        self.root = base / (f"{sha}-dirty" if dirty else sha)
        self.resume = resume and not dirty
        self.logs = self.root / "logs"
        self.temp = self.root / "tmp"
        self.logs.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def recorded(self, name: str) -> list[StageResult] | None:
        """What is on disk for ``name``, whether or not a run may reuse it."""
        path = self._path(name)
        if not path.exists():
            return None
        try:
            return [StageResult(**item) for item in json.loads(path.read_text())]
        except (OSError, ValueError, TypeError):
            return None  # a record this version cannot read is a record to redo

    def load(self, name: str) -> list[StageResult] | None:
        """The recorded results of ``name`` for this commit, if a run may reuse them."""
        return self.recorded(name) if self.resume else None

    def save(self, name: str, results: Sequence[StageResult]) -> None:
        self._path(name).write_text(json.dumps([dataclasses.asdict(r) for r in results]))


@dataclass(frozen=True)
class RunFacts:
    """What a note says about the run that produced it."""

    sha: str
    command: str
    budget: MachineBudget
    dirty: bool = False


def render_note(name: str, domain: str, results: Sequence[StageResult], facts: RunFacts) -> str:
    """The Markdown note for one family, in the format the vault has always had."""
    seconds = sum(result.seconds for result in results)
    peak = max((result.peak_gb for result in results), default=0.0)
    commit = f"{facts.sha[:9]}{' (dirty)' if facts.dirty else ''}"
    lines = [
        f"# zoo-verify: {name}",
        "",
        f"> **{overall(results)}** · `{commit}` · {datetime.now():%Y-%m-%d %H:%M} · "
        f"{domain} · {seconds / 60:.1f} 분 · 최대 {peak:.1f} GB",
        f"> 명령: `{facts.command}` · 머신: {facts.budget.describe()}",
        "",
        "| 단계 | 결과 | 시간 (s) | 최대 (GB) | 요약 |",
        "|---|---|---|---|---|",
    ]
    for result in results:
        summary = result.summary.replace("|", "/")
        lines.append(
            f"| {result.name} | {result.status} | {result.seconds:.0f} "
            f"| {result.peak_gb:.1f} | {summary} |"
        )
    for title, field_name in (
        ("실패", "failures"),
        ("확인할 것", "checks"),
        ("건너뛴 것", "skips"),
        ("이 머신에서 정한 것", "notes"),
    ):
        items = [f"- `{r.name}`: {item}" for r in results for item in getattr(r, field_name)]
        if items:
            lines += ["", f"## {title}", "", *items]
    lines += ["", "장부: [[zoo-verify-ledger]] · 계약: [[arch-models-family-contract]]", ""]
    return "\n".join(lines)


_LEDGER_HEAD = """# model zoo 전수 검증 장부

**다시 돌리기 전에 이 표를 먼저 본다.** `lucid-audit --models` 가 family 하나가 끝날 때마다 자동으로 고쳐 쓴다. 손으로 고치지 않는다.

**돌린 것**: 계약 검증 (`validate_model_zoo --runtime`), 모든 factory 의 실제 크기 build, compiled training 행렬, family 전용 테스트와 parity 파일 (`heavy` 포함), zoo 공통 테스트, 마지막으로 zoo 공통 파일 전부 (`control` 벤치 포함) 를 파일당 한 프로세스로.

**안 돌린 것**: 네트워크를 껐으므로 가중치를 내려받는 테스트 (`LUCID_TEST_NETWORK`). 실제 크기 build 는 forward 를 돌리지 않는다. 학습 단계는 테스트가 정한 축소 config 로 돈다. 메모리 상한을 넘어 중단된 단계는 `mem-guard` 로 표시한다. 돌린 머신의 메모리에 맞춰 건너뛴 것 (예산을 넘는 factory 의 build, `heavy` 테스트) 은 family 노트의 "이 머신에서 정한 것" 에 적는다.
"""


def render_ledger(facts: RunFacts, state: ZooState, families: dict[str, str]) -> str:
    """The table of every family's latest result for this commit."""
    lines = [_LEDGER_HEAD, f"**대상 커밋**: `{facts.sha}`{' (dirty)' if facts.dirty else ''}", ""]
    lines += ["| family | 분야 | 결과 | 분 | 최대 GB | 기록 |", "|---|---|---|---|---|---|"]
    done = 0
    for name, domain in [*families.items(), (ZOO_WIDE, "공통")]:
        record = state.recorded(name)
        if record is None:
            lines.append(f"| {name} | {domain} | 대기 | | | |")
            continue
        done += 1
        minutes = sum(item.seconds for item in record) / 60
        peak = max((item.peak_gb for item in record), default=0.0)
        lines.append(
            f"| {name} | {domain} | {overall(record)} | {minutes:.1f} | {peak:.1f} "
            f"| [[zoo-verify-{name}]] |"
        )
    lines += ["", f"진행: {done} / {len(families) + 1}", ""]
    return "\n".join(lines)


class Notebook:
    """Writes a family's note and the ledger after the family finishes."""

    def __init__(
        self, directory: Path, state: ZooState, families: dict[str, str], facts: RunFacts
    ) -> None:
        self._directory = directory
        self._state = state
        self._families = families
        self._facts = facts
        directory.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, results: Sequence[StageResult]) -> None:
        domain = self._families.get(name, "공통")
        (self._directory / f"zoo-verify-{name}.md").write_text(
            render_note(name, domain, results, self._facts)
        )
        (self._directory / "zoo-verify-ledger.md").write_text(
            render_ledger(self._facts, self._state, self._families)
        )


__all__ = [
    "STATE_ROOT",
    "Notebook",
    "RunFacts",
    "ZooState",
    "overall",
    "render_ledger",
    "render_note",
    "repo_state",
]
