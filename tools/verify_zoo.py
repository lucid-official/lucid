"""Verify the whole model zoo on one machine, one family at a time, and write it down.

CI runs the zoo's tests in pieces, on shrunken configs, and a cancelled nightly
leaves no record of how far it got.  This runs everything the zoo has — the
contract validator, a real-size build of every factory, each family's own test
files, every zoo-wide file and the compiled-training matrix — strictly one
process at a time, with the markers cleared and the network off, so nothing is
downloaded and nothing is kept.

After each family it writes ``<vault>/zoo-verify-<family>.md`` and regenerates
``<vault>/zoo-verify-ledger.md``.  A family already recorded for this commit is
skipped, so an interrupted run resumes and a finished one is not repeated.

Memory is guarded from outside the test process: a stage whose physical
footprint passes ``--limit-gb``, or that leaves the machine short, is killed and
recorded as ``mem-guard``, and the run moves on.  A stage waits for the machine
to have room before it starts, since an engine build may be sharing it.

usage: python tools/verify_zoo.py --vault DIR --state DIR [--family NAME ...]
"""

import argparse
import ctypes
import ctypes.util
import dataclasses
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_MODELS = _ROOT / "lucid" / "models"
_UNIT = _ROOT / "lucid" / "test" / "unit"
_PARITY = _ROOT / "lucid" / "test" / "parity" / "models"
_DOMAINS = ("vision", "text", "generative", "multimodal")
_ZOO_WIDE = "zoo-wide"

_GIB = 1024**3
_STAGE_TIMEOUT_S = 2 * 3600
_POLL_S = 2.0
_ROOM_TO_START_GB = 14.0
_ROOM_WAIT_S = 30 * 60
_MACHINE_FLOOR_GB = 2.5
_BYTES_PER_PARAM = 4
_BUILD_SHARE = 0.8

_SUMMARY = re.compile(r"^(?:=+ )?(\d+ .*?) in [0-9.]+s")
_FAILED = re.compile(r"^(?:FAILED|ERROR) (.+)$")
_SKIPPED = re.compile(r"^SKIPPED \[(\d+)\] (.+)$")
_CHECK = re.compile(r"^CHECK (.+)$")
_MEMBER = re.compile(r"create_model\(")


@dataclasses.dataclass(frozen=True)
class Stage:
    """One process: what to run and how a run of it should read."""

    name: str
    argv: tuple[str, ...]


@dataclasses.dataclass
class StageResult:
    """What one stage did, in the form the notes and the resume check read."""

    name: str
    status: str
    seconds: float
    peak_gb: float
    summary: str = ""
    failures: list[str] = dataclasses.field(default_factory=list)
    skips: list[str] = dataclasses.field(default_factory=list)
    checks: list[str] = dataclasses.field(default_factory=list)


class _RusageInfo(ctypes.Structure):
    """The head of ``rusage_info_v2``, up to the footprint this reads."""

    _fields_ = [
        ("uuid", ctypes.c_uint8 * 16),
        ("user_time", ctypes.c_uint64),
        ("system_time", ctypes.c_uint64),
        ("pkg_idle_wkups", ctypes.c_uint64),
        ("interrupt_wkups", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("wired_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("phys_footprint", ctypes.c_uint64),
        ("padding", ctypes.c_uint64 * 32),
    ]


_LIBPROC = ctypes.CDLL(ctypes.util.find_library("proc") or "libproc.dylib", use_errno=True)


def _footprint_gb(pid: int) -> float:
    """Physical footprint, which counts Metal buffers where resident size does not."""
    info = _RusageInfo()
    if _LIBPROC.proc_pid_rusage(pid, 2, ctypes.byref(info)) != 0:
        return 0.0
    return info.phys_footprint / _GIB


def _available_gb() -> float:
    """Memory the machine could hand out now: free, inactive, speculative, purgeable pages."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))
    pages = 0
    for label in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        found = re.search(rf"{label}:\s+(\d+)\.", out)
        pages += int(found.group(1)) if found else 0
    return pages * page / _GIB


def _env() -> dict[str, str]:
    """The children's environment: this tree, no network, bytecode out of the tree."""
    env = dict(os.environ)
    env.pop("LUCID_TEST_NETWORK", None)
    env.update(
        PYTHONPATH=str(_ROOT),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONPYCACHEPREFIX=str(Path.home() / "Library" / "Caches" / "lucid-pycache"),
    )
    return env


def _wait_for_room(log: Path) -> None:
    deadline = time.monotonic() + _ROOM_WAIT_S
    while _available_gb() < _ROOM_TO_START_GB and time.monotonic() < deadline:
        time.sleep(30)
    with log.open("a") as out:
        out.write(f"[verify_zoo] starting with {_available_gb():.1f} GB available\n")


def _run_process(stage: Stage, log: Path, limit_gb: float) -> tuple[str, float, float]:
    """Run ``stage`` under the guard.  Returns its raw status, seconds and peak footprint."""
    _wait_for_room(log)
    started = time.monotonic()
    peak = 0.0
    verdict = ""
    with log.open("a") as out:
        proc = subprocess.Popen(
            stage.argv,
            cwd=_ROOT,
            env=_env(),
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while proc.poll() is None:
            time.sleep(_POLL_S)
            peak = max(peak, _footprint_gb(proc.pid))
            if peak > limit_gb or _available_gb() < _MACHINE_FLOOR_GB:
                verdict = "mem-guard"
            elif time.monotonic() - started > _STAGE_TIMEOUT_S:
                verdict = "timeout"
            if verdict:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                break
    seconds = time.monotonic() - started
    if verdict:
        return verdict, seconds, peak
    return {0: "pass", 5: "none"}.get(proc.returncode, "fail"), seconds, peak


def _read_result(stage: Stage, log: Path, raw: tuple[str, float, float]) -> StageResult:
    status, seconds, peak = raw
    result = StageResult(stage.name, status, round(seconds, 1), round(peak, 2))
    for line in log.read_text(errors="replace").splitlines():
        if failed := _FAILED.match(line):
            result.failures.append(failed.group(1)[:300])
        elif skipped := _SKIPPED.match(line):
            result.skips.append(f"{skipped.group(1)} × {skipped.group(2)[:200]}")
        elif check := _CHECK.match(line):
            result.checks.append(check.group(1)[:300])
        elif summary := _SUMMARY.match(line):
            result.summary = summary.group(1)
    if status in ("fail", "timeout", "mem-guard") and not result.failures:
        tail = log.read_text(errors="replace").splitlines()[-3:]
        result.failures.append(" / ".join(tail)[:300])
    return result


def _run_stage(stage: Stage, logs: Path, limit_gb: float) -> StageResult:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", stage.name)
    log = logs / f"{slug}.log"
    log.write_text("")
    print(f"  {stage.name} ...", flush=True)
    result = _read_result(stage, log, _run_process(stage, log, limit_gb))
    print(f"  {stage.name}: {result.status} {result.seconds:.0f}s {result.summary}", flush=True)
    return result


def _pytest(temp: Path, *selection: str) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        "-rsfE",
        "-p",
        "no:cacheprovider",
        f"--basetemp={temp}",
        "-o",
        "faulthandler_timeout=900",
        *selection,
    )


def _families() -> dict[str, str]:
    """Every family directory, ``name -> domain``, in a stable order."""
    found: dict[str, str] = {}
    for domain in _DOMAINS:
        for path in sorted((_MODELS / domain).iterdir()):
            if path.is_dir() and not path.name.startswith("_"):
                found[path.name] = domain
    return found


def _own_files(family: str) -> list[Path]:
    """The test files named for ``family`` alone."""
    own = sorted((_UNIT / "models").glob(f"test_models_{family}.py"))
    own += sorted(_PARITY.glob(f"*/test_{family}.py"))
    return own


def _zoo_files(families: dict[str, str]) -> list[Path]:
    """Files reaching several families: every model test no family owns, then the rest."""
    owned = {path for family in families for path in _own_files(family)}
    wide = [path for path in sorted((_UNIT / "models").glob("test_*.py")) if path not in owned]
    wide += [path for path in sorted(_PARITY.rglob("test_*.py")) if path not in owned]
    for path in sorted(_UNIT.rglob("test_*.py")):
        if path.parent != _UNIT / "models" and _MEMBER.search(path.read_text(errors="replace")):
            wide.append(path)
    return wide


def _family_stages(family: str, temp: Path, wide: list[Path]) -> list[Stage]:
    py = sys.executable
    tool = str(Path(__file__).resolve())
    stages = [
        Stage("contract", (py, "-m", "tools.validate_model_zoo", "--family", family, "--runtime")),
        Stage("build", (py, tool, "--build-family", family)),
        Stage("compiled-training", (py, tool, "--matrix-family", family)),
    ]
    own = [str(path) for path in _own_files(family)]
    if own:
        stages.append(Stage("own-tests", _pytest(temp, "-m", "", *own)))
    # Control benchmarks take tens of minutes and run once, in the zoo-wide pass.
    models = [str(path) for path in wide if path.parent == _UNIT / "models"]
    stages.append(Stage("zoo-tests", _pytest(temp, "-m", "not control", "-k", family, *models)))
    return stages


def _labels_by_family() -> dict[str, dict[str, object]]:
    """Registered factories grouped by family directory: their names and registry labels."""
    import lucid.models  # noqa: F401 — registers every factory
    from lucid.models._registry import _REGISTRY

    grouped: dict[str, dict[str, object]] = {}
    for name, entry in _REGISTRY.items():
        parts = getattr(entry.factory, "__module__", "").split(".")
        if len(parts) < 4 or parts[1] != "models":
            continue
        slot = grouped.setdefault(parts[3], {"names": [], "labels": set()})
        slot["names"].append(name)
        slot["labels"].add(entry.family)
    return grouped


def _build_family(family: str, limit_gb: float) -> int:
    """Build every factory of ``family`` at its real size; report what the build allocated."""
    import gc

    import lucid
    import lucid.metal
    from lucid.models import create_model
    from lucid.models._registry import _REGISTRY

    budget = limit_gb * _BUILD_SHARE * _GIB
    built_count = past_budget = failed = 0
    for name in _labels_by_family().get(family, {}).get("names", []):
        declared = _REGISTRY[name].params
        if declared and declared * _BYTES_PER_PARAM > budget:
            past_budget += 1
            print(f"SKIPPED [1] {name}: {declared:,} params is past the build budget", flush=True)
            continue
        began = time.monotonic()
        try:
            model = create_model(name)
        except Exception as exc:  # noqa: BLE001 — reported per factory; the rest still build
            failed += 1
            print(f"FAILED {name} - {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            continue
        built_count += 1
        built = sum(int(param.numel()) for param in model.parameters())
        seconds = time.monotonic() - began
        print(f"built {name}: {built:,} params, declared {declared}, {seconds:.1f}s", flush=True)
        if declared and built != declared:
            print(f"CHECK {name}: built {built:,} params, declared {declared:,}", flush=True)
        del model
        gc.collect()
        lucid.metal.empty_cache()
    print(f"{built_count} built, {past_budget} past budget, {failed} failed in 0.0s")
    return 1 if failed else 0


def _matrix_family(family: str) -> int:
    """The compiled-training pairs of ``family``, held to the matrix's own expectations."""
    from lucid.test.unit.compile import _zoo_matrix as matrix

    labels = _labels_by_family().get(family, {}).get("labels", set()) | {family}
    ran = bad = 0
    for task in matrix.TASKS:
        for label in matrix.families(task):
            if label in labels:
                record = matrix.run_isolated(label, task)
                why = matrix.verdict(record)
                ran += 1
                bad += why is not None
                print(f"FAILED {task} {label} - {why}" if why else f"ok {task} {label}", flush=True)
    print(f"{ran} pairs, {bad} failing in 0.0s")
    return 1 if bad else (0 if ran else 5)


def _overall(results: list[StageResult]) -> str:
    statuses = {result.status for result in results}
    for worst in ("fail", "timeout", "mem-guard"):
        if worst in statuses:
            return worst
    return "pass"


def _note(name: str, domain: str, sha: str, results: list[StageResult]) -> str:
    seconds = sum(result.seconds for result in results)
    peak = max((result.peak_gb for result in results), default=0.0)
    lines = [
        f"# zoo-verify: {name}",
        "",
        f"> **{_overall(results)}** · `{sha[:9]}` · {datetime.now():%Y-%m-%d %H:%M} · "
        f"{domain} · {seconds / 60:.1f} 분 · 최대 {peak:.1f} GB",
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
    failures = [f"- `{r.name}`: {item}" for r in results for item in r.failures]
    if failures:
        lines += ["", "## 실패", "", *failures]
    checks = [f"- `{r.name}`: {item}" for r in results for item in r.checks]
    if checks:
        lines += ["", "## 확인할 것", "", *checks]
    skips = [f"- `{r.name}`: {item}" for r in results for item in r.skips]
    if skips:
        lines += ["", "## 건너뛴 것", "", *skips]
    lines += ["", "장부: [[zoo-verify-ledger]] · 계약: [[arch-models-family-contract]]", ""]
    return "\n".join(lines)


_LEDGER_HEAD = """# model zoo 전수 검증 장부

**다시 돌리기 전에 이 표를 먼저 본다.** `tools/verify_zoo.py` 가 family 하나가 끝날 때마다 자동으로 고쳐 쓴다. 손으로 고치지 않는다.

**돌린 것**: 계약 검증 (`validate_model_zoo --runtime`), 모든 factory 의 실제 크기 build, compiled training 행렬, family 전용 테스트와 parity 파일 (`heavy` 포함), zoo 공통 테스트, 마지막으로 zoo 공통 파일 전부 (`control` 벤치 포함) 를 파일당 한 프로세스로.

**안 돌린 것**: 네트워크를 껐으므로 가중치를 내려받는 테스트 (`LUCID_TEST_NETWORK`). 실제 크기 build 는 forward 를 돌리지 않는다. 학습 단계는 테스트가 정한 축소 config 로 돈다. 메모리 상한을 넘어 중단된 단계는 `mem-guard` 로 표시한다.
"""


def _ledger(sha: str, state: Path, families: dict[str, str]) -> str:
    lines = [_LEDGER_HEAD, f"**대상 커밋**: `{sha}`", ""]
    lines += ["| family | 분야 | 결과 | 분 | 최대 GB | 기록 |", "|---|---|---|---|---|---|"]
    done = 0
    for name, domain in [*families.items(), (_ZOO_WIDE, "공통")]:
        path = state / f"{name}.json"
        if not path.exists():
            lines.append(f"| {name} | {domain} | 대기 | | | |")
            continue
        done += 1
        record = [StageResult(**item) for item in json.loads(path.read_text())]
        minutes = sum(item.seconds for item in record) / 60
        peak = max((item.peak_gb for item in record), default=0.0)
        lines.append(
            f"| {name} | {domain} | {_overall(record)} | {minutes:.1f} | {peak:.1f} "
            f"| [[zoo-verify-{name}]] |"
        )
    lines += ["", f"진행: {done} / {len(families) + 1}", ""]
    return "\n".join(lines)


def _record(
    name: str,
    domain: str,
    sha: str,
    results: list[StageResult],
    state: Path,
    vault: Path,
    families: dict[str, str],
) -> None:
    (state / f"{name}.json").write_text(json.dumps([dataclasses.asdict(r) for r in results]))
    (vault / f"zoo-verify-{name}.md").write_text(_note(name, domain, sha, results))
    (vault / "zoo-verify-ledger.md").write_text(_ledger(sha, state, families))


def _run_group(stages: list[Stage], logs: Path, temp: Path, limit_gb: float) -> list[StageResult]:
    results = []
    for stage in stages:
        results.append(_run_stage(stage, logs, limit_gb))
        shutil.rmtree(temp, ignore_errors=True)
    return results


def _verify(args: argparse.Namespace) -> int:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    state = args.state / sha
    logs = state / "logs"
    temp = state / "tmp"
    logs.mkdir(parents=True, exist_ok=True)
    args.vault.mkdir(parents=True, exist_ok=True)
    families = _families()
    wide = _zoo_files(families)
    chosen = args.family or [*families, _ZOO_WIDE]
    for name in chosen:
        if (state / f"{name}.json").exists():
            print(f"{name}: already recorded for {sha[:9]}", flush=True)
            continue
        print(f"{name}:", flush=True)
        if name == _ZOO_WIDE:
            stages = [
                Stage(str(path.relative_to(_UNIT.parent)), _pytest(temp, "-m", "", str(path)))
                for path in wide
            ]
        else:
            stages = _family_stages(name, temp, wide)
        results = _run_group(stages, logs, temp, args.limit_gb)
        _record(name, families.get(name, "공통"), sha, results, state, args.vault, families)
    if not args.family:
        (state / "DONE").write_text(sha + "\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.partition("\n")[0])
    parser.add_argument("--vault", type=Path, help="directory the notes are written to")
    parser.add_argument("--state", type=Path, help="directory results and logs are kept in")
    parser.add_argument("--family", action="append", help="only this family (repeatable)")
    parser.add_argument("--limit-gb", type=float, default=28.0, help="footprint that kills a stage")
    parser.add_argument("--build-family", help=argparse.SUPPRESS)
    parser.add_argument("--matrix-family", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.build_family:
        return _build_family(args.build_family, args.limit_gb)
    if args.matrix_family:
        return _matrix_family(args.matrix_family)
    if not (args.vault and args.state):
        parser.error("--vault and --state are required")
    return _verify(args)


if __name__ == "__main__":
    sys.exit(main())
