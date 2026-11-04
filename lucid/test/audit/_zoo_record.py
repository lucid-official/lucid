"""What the zoo stage remembers: results per commit, and the notes written from them.

Two records, deliberately separate:

* **state** — one JSON per family under
  ``~/Library/Caches/lucid/audit-zoo/<sha>/``.  It exists so an interrupted run
  resumes: a family recorded for this commit is not run again — *by a machine at
  least as large as the one that recorded it*, because a record that left things
  out for a small machine is not an answer for a big one.  It lives in the cache
  directory, not the checkout, because the stage must leave the tree as it found
  it.  A tree with uncommitted changes never reuses a record — the commit it
  would be filed under no longer describes what is on disk — and a tree that
  changes *during* a run is an error, not a result filed under the old commit.
* **notes** — ``zoo-verify-<family>.md`` plus ``zoo-verify-ledger.md``, written
  only where the operator points (``--zoo-record``) or the repository has its
  vault.  They are for people: the ledger is read *before* a run to avoid
  repeating one.
"""

import contextlib
import dataclasses
import fcntl
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from lucid.test.audit._zoo_stage import ZOO_WIDE, StageResult

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from lucid.test.audit._zoo_budget import MachineBudget

STATE_ROOT = Path.home() / "Library" / "Caches" / "lucid" / "audit-zoo"


class ZooBusy(RuntimeError):
    """Another run is using this commit's state directory."""


def overall(results: Sequence[StageResult]) -> str:
    """The worst status among ``results``: a failure, else ``reduced``, else ``pass``.

    ``reduced`` is a family that passed everything it was asked and was asked
    for less than the zoo's full verification — a machine left heavy tests or a
    build out, or a stage ran nothing and gave no reason.  It is not a failure,
    and it is not ``pass``: reading it as one is how a 6 GB machine's ``lenet``
    came to look like a 28 GB machine's.
    """
    statuses = {result.status for result in results}
    for worst in ("fail", "timeout", "mem-guard", "machine-short"):
        if worst in statuses:
            return worst
    return "reduced" if any(result.reduced for result in results) else "pass"


def repo_state(root: Path) -> tuple[str, bool]:
    """``(HEAD sha, whether the tree differs from it)``.

    Untracked files count: the catalog reads the filesystem, so a test file that
    is not yet committed is part of what a run verifies.  Ignored files (build
    products, the engine) do not.
    """
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True
    ).stdout
    return sha, bool(status.strip())


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` so a reader — or ``sync`` — never sees half of it."""
    scratch = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    scratch.write_text(text)
    os.replace(scratch, path)


@dataclass
class FamilyRecord:
    """A family's results and the machine that produced them."""

    results: list[StageResult]
    limit_gb: float
    #: The device the stages ran on.  A zoo verified on CPU says nothing about
    #: Metal (different kernels, different dtype rules, MLX's laziness), so a
    #: record is the answer only for a run on the same device.
    device: str

    def reusable_under(self, budget: MachineBudget, device: str) -> bool:
        """Whether a run with ``budget`` on ``device`` may take these results as its own.

        What a machine's limit changed — heavy tests dropped, a build skipped,
        a stage killed at the limit — was a decision about *that* machine.  A
        larger one would have run it, so only a machine no larger than the
        recorder's may rely on the record.  A stage the machine itself ran out
        of memory under proved nothing about the stage, and is never reused.
        """
        if self.device != device or any(r.status == "machine-short" for r in self.results):
            return False
        limited = any(r.notes or r.status == "mem-guard" for r in self.results)
        return not limited or self.limit_gb >= budget.limit_gb


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

    @contextlib.contextmanager
    def lock(self) -> Iterator[None]:
        """Hold this commit's state directory; two runs would share ``tmp/`` and each other's logs."""
        fd = os.open(self.root / "lock", os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise ZooBusy(
                f"another zoo run is using {self.root}; wait for it, or run from another clone"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def recorded(self, name: str) -> FamilyRecord | None:
        """What is on disk for ``name``, whether or not a run may reuse it."""
        path = self._path(name)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
            results = [StageResult(**item) for item in payload["results"]]
            return FamilyRecord(results, float(payload["limit_gb"]), str(payload["device"]))
        except (OSError, ValueError, TypeError, KeyError):
            return None  # a record this version cannot read is a record to redo

    def load(self, name: str, budget: MachineBudget, device: str) -> list[StageResult] | None:
        """The recorded results of ``name``, if a run with ``budget`` on ``device`` may reuse them."""
        record = self.recorded(name) if self.resume else None
        return record.results if record and record.reusable_under(budget, device) else None

    def save(
        self, name: str, results: Sequence[StageResult], budget: MachineBudget, device: str
    ) -> None:
        payload = {
            "limit_gb": budget.limit_gb,
            "device": device,
            "results": [dataclasses.asdict(r) for r in results],
        }
        write_atomic(self._path(name), json.dumps(payload))


@dataclass(frozen=True)
class RunFacts:
    """What a note says about the run that produced it."""

    sha: str
    command: str
    budget: MachineBudget
    dirty: bool = False
    device: str = "cpu"


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
        f"> 명령: `{facts.command}` · 장치: {facts.device} · 머신: {facts.budget.describe()}",
        f"> {_RESULT_RULE}",
        "",
        "| 단계 | 결과 | 시간 (s) | 최대 (GB) | 요약 |",
        "|---|---|---|---|---|",
    ]
    for result in results:
        summary = result.summary.replace("|", "/")
        if result.device and result.device != facts.device:
            summary += f" · 장치: {result.device} (--zoo-device 와 무관한 단계)"
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
        items = [
            f"- `{r.name}`: {item}"
            for r in results
            for item in (r.notes + r.events if field_name == "notes" else getattr(r, field_name))
        ]
        if items:
            lines += ["", f"## {title}", "", *items]
    lines += ["", "장부: [[zoo-verify-ledger]] · 계약: [[arch-models-family-contract]]", ""]
    return "\n".join(lines)


_RESULT_RULE = (
    "결과: pass = 모든 단계가 요청한 만큼 돌았다 · reduced = 통과했지만 이 머신이 일부를 뺐거나 "
    "(heavy 제외, 예산 초과 build, 줄여서 재시도) 이유 없이 아무것도 돌지 않은 단계가 있다 "
    "— 이유가 기록된 설계된 skip 은 reduced 가 아니다 · fail/timeout/mem-guard/machine-short = 실패"
)

_LEDGER_HEAD = """# model zoo 전수 검증 장부

**다시 돌리기 전에 이 표를 먼저 본다.** `lucid-audit --models` 가 family 하나가 끝날 때마다 자동으로 고쳐 쓴다. 손으로 고치지 않는다.

**돌린 것**: 계약 검증 (`validate_model_zoo --runtime`), 모든 factory 의 실제 크기 build, compiled training 행렬, family 전용 테스트와 parity 파일 (`heavy` 포함), zoo 공통 테스트, 마지막으로 zoo 공통 파일 전부 (`control` 벤치 포함) 를 파일당 한 프로세스로.

**안 돌린 것**: 네트워크를 껐으므로 가중치를 내려받는 테스트 (`LUCID_TEST_NETWORK`). 실제 크기 build 는 forward 를 돌리지 않는다. 학습 단계는 테스트가 정한 축소 config 로 돈다. 메모리 상한을 넘어 중단된 단계는 `mem-guard` 로 표시한다. 돌린 머신의 메모리에 맞춰 건너뛴 것 (예산을 넘는 factory 의 build, `heavy` 테스트) 은 family 노트의 "이 머신에서 정한 것" 에 적는다.
"""


def render_ledger(facts: RunFacts, state: ZooState, families: dict[str, str]) -> str:
    """The table of every family's latest result for this commit."""
    lines = [
        _LEDGER_HEAD,
        f"**대상 커밋**: `{facts.sha}`{' (dirty)' if facts.dirty else ''} · **이번 실행 장치**: {facts.device}",
        "",
    ]
    lines += [
        "| family | 분야 | 결과 | 분 | 최대 GB | 장치 | 기록 |",
        "|---|---|---|---|---|---|---|",
    ]
    done = 0
    for name, domain in [*families.items(), (ZOO_WIDE, "공통")]:
        recorded = state.recorded(name)
        if recorded is None:
            lines.append(f"| {name} | {domain} | 대기 | | | | |")
            continue
        done += 1
        record = recorded.results
        minutes = sum(item.seconds for item in record) / 60
        peak = max((item.peak_gb for item in record), default=0.0)
        lines.append(
            f"| {name} | {domain} | {overall(record)} | {minutes:.1f} | {peak:.1f} "
            f"| {recorded.device} | [[zoo-verify-{name}]] |"
        )
    lines += ["", f"진행: {done} / {len(families) + 1}", ""]
    return "\n".join(lines)


class Notebook:
    """Writes a family's note, and the ledger over all of them."""

    def __init__(
        self, directory: Path, state: ZooState, families: dict[str, str], facts: RunFacts
    ) -> None:
        self._directory = directory
        self._state = state
        self._families = families
        self._facts = facts
        directory.mkdir(parents=True, exist_ok=True)

    def note_path(self, name: str) -> Path:
        return self._directory / f"zoo-verify-{name}.md"

    def write_note(self, name: str, results: Sequence[StageResult]) -> None:
        domain = self._families.get(name, "공통")
        write_atomic(self.note_path(name), render_note(name, domain, results, self._facts))

    def write_ledger(self) -> None:
        write_atomic(
            self._directory / "zoo-verify-ledger.md",
            render_ledger(self._facts, self._state, self._families),
        )

    def write(self, name: str, results: Sequence[StageResult]) -> None:
        self.write_note(name, results)
        self.write_ledger()


__all__ = [
    "STATE_ROOT",
    "FamilyRecord",
    "Notebook",
    "RunFacts",
    "ZooBusy",
    "ZooState",
    "overall",
    "render_ledger",
    "render_note",
    "repo_state",
    "write_atomic",
]
