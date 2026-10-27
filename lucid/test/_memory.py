"""Keep the suite inside the memory the machine actually has.

Why this exists
---------------
The full suite was being killed with ``SIGKILL`` partway through the
model zoo — ``pytest exited -9`` after two to seven minutes, with no
traceback and no failing test, because the killer is macOS jetsam rather
than anything in the process.  Measured on a 16 GB M1 Pro, per test:

===============================  ==========  ============  ==========
                                   RSS peak    MLX cache      total
===============================  ==========  ============  ==========
model zoo, as it was                2057 MB       1582 MB     3639 MB
same tests, reclaiming as it goes   2385 MB          0 MB     2385 MB
===============================  ==========  ============  ==========

Two things that table says.  First, the single largest consumer is not
any model — it is MLX's buffer cache, which grows without bound and is
*entirely* reclaimable, so a third of the footprint can be handed back
for free.  Second, what remains is still 2.4 GB, and on a machine whose
swap was already 6.7 GB into its 8 GB the suite is the fattest target in
the room whatever it does.  So reclaiming is necessary and not
sufficient, and the governor does both jobs in that order:

1. hand memory back when the cache has grown past ``reclaim_at``;
2. only if the machine is *still* short, skip the next test rather than
   let jetsam pick the victim.

Order matters.  Skipping first would drop coverage to save memory that
was free for the asking.

Nothing here is silent.  A run that skipped tests says so in the summary
and names them: a suite that quietly shrinks under load reports "all
passed" for a run that checked less than the last one, which is the one
failure mode worse than being killed.

Knobs, all via the environment:

``LUCID_TEST_MEM_GOVERNOR=0``     turn the whole thing off.
``LUCID_TEST_MEM_FLOOR_MB``       skip below this much available memory
                                  (default 1024).  ``0`` disables
                                  skipping but keeps reclaiming.
``LUCID_TEST_MEM_RECLAIM_MB``     reclaim once the cache passes this
                                  (default 256).
"""

import ctypes
import ctypes.util
import gc
import os
import subprocess
import time

_MIB = 1024 * 1024


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


ENABLED: bool = os.environ.get("LUCID_TEST_MEM_GOVERNOR", "1") != "0"
FLOOR_MB: int = _env_int("LUCID_TEST_MEM_FLOOR_MB", 1024)
RECLAIM_MB: int = _env_int("LUCID_TEST_MEM_RECLAIM_MB", 256)

#: Below this resident size the machine cannot plausibly be short because
#: of us, so the availability probe — which shells out — is not run at
#: all.  This keeps the governor free for the thousands of small tests.
_WATCH_FROM_MB: int = 512

#: Availability is sampled at most this often; the probe costs two
#: subprocesses and the number does not move meaningfully faster.
_SAMPLE_EVERY_S: float = 2.0


# ── reading the machine ──────────────────────────────────────────────────────


class _MachTaskBasicInfo(ctypes.Structure):
    _fields_ = [
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_max", ctypes.c_uint64),
        ("user_time", ctypes.c_uint64),
        ("system_time", ctypes.c_uint64),
        ("policy", ctypes.c_int),
        ("suspend_count", ctypes.c_int),
    ]


_MACH_TASK_BASIC_INFO = 20
_INFO_COUNT = ctypes.sizeof(_MachTaskBasicInfo) // ctypes.sizeof(ctypes.c_uint)

try:
    _libc: ctypes.CDLL | None = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    _task_self = _libc.mach_task_self()  # type: ignore[union-attr]
except Exception:  # noqa: BLE001 - a governor that cannot read is a no-op
    _libc = None


def resident_mb() -> float:
    """This process's live resident size, in MiB.

    ``resource.getrusage`` reports the *peak*, which is monotonic and so
    cannot say whether memory came back — the whole question here.
    """
    if _libc is not None:
        info = _MachTaskBasicInfo()
        count = ctypes.c_uint(_INFO_COUNT)
        rc = _libc.task_info(
            _task_self,
            ctypes.c_int(_MACH_TASK_BASIC_INFO),
            ctypes.byref(info),
            ctypes.byref(count),
        )
        if rc == 0:
            return info.resident_size / _MIB
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(out.stdout.strip() or 0) / 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def cache_mb() -> float:
    """MLX's buffer cache, in MiB — reclaimable, and the biggest lever."""
    try:
        import lucid

        return lucid.metal.get_cache_memory() / _MIB
    except Exception:  # noqa: BLE001
        return 0.0


_last_sample: tuple[float, float] = (0.0, -1.0)


def available_mb(force: bool = False) -> float:
    """Roughly what the machine could still hand out, in MiB.

    Free pages alone are the wrong number on Apple Silicon: the unified
    accounting reports a comfortable "64% free" on a machine whose swap
    is one page from full, which is exactly the state a run dies in.  So
    inactive and speculative pages — reclaimable without swapping — are
    counted in, and the remaining swap is added, because that is the
    headroom jetsam is actually watching.

    Returns ``-1.0`` when the machine cannot be read, which every caller
    treats as "do not intervene".
    """
    global _last_sample
    now = time.monotonic()
    if not force and now - _last_sample[0] < _SAMPLE_EVERY_S:
        return _last_sample[1]

    total = -1.0
    try:
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
        page = 4096
        first = vm.stdout.splitlines()[0] if vm.stdout else ""
        if "page size of" in first:
            page = int(first.split("page size of")[1].split()[0])
        counts: dict[str, int] = {}
        for line in vm.stdout.splitlines()[1:]:
            if ":" not in line:
                continue
            key, _, value = line.partition(":")
            digits = value.strip().rstrip(".")
            if digits.isdigit():
                counts[key.strip()] = int(digits)
        reclaimable = (
            counts.get("Pages free", 0)
            + counts.get("Pages inactive", 0)
            + counts.get("Pages speculative", 0)
            + counts.get("Pages purgeable", 0)
        )
        total = reclaimable * page / _MIB

        sw = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5
        )
        # "total = 8192.00M  used = 6673.69M  free = 1518.31M  (encrypted)"
        parts = sw.stdout.replace("=", " ").split()
        for i, tok in enumerate(parts):
            if tok == "free" and i + 1 < len(parts):
                total += float(parts[i + 1].rstrip("M"))
                break
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        total = -1.0

    _last_sample = (now, total)
    return total


# ── acting on it ─────────────────────────────────────────────────────────────


def reclaim() -> float:
    """Hand back what can be handed back.  Returns the MiB recovered."""
    before = resident_mb() + cache_mb()
    gc.collect()
    try:
        import lucid

        lucid.metal.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return max(0.0, before - (resident_mb() + cache_mb()))


class Governor:
    """Session-scoped state: what was reclaimed, and what was skipped."""

    def __init__(self) -> None:
        self.reclaimed_mb = 0.0
        self.reclaims = 0
        self.skipped: list[tuple[str, float]] = []
        self.peak_mb = 0.0

    def before_test(self, nodeid: str) -> str | None:
        """Return a skip reason, or ``None`` to let the test run."""
        if not ENABLED or FLOOR_MB <= 0:
            return None
        if resident_mb() < _WATCH_FROM_MB:
            return None
        free = available_mb()
        if free < 0 or free >= FLOOR_MB:
            return None
        # Short — but the cache may be holding most of it.  Ask for it
        # back and look again before giving up on the test.
        self.reclaimed_mb += reclaim()
        self.reclaims += 1
        free = available_mb(force=True)
        if free < 0 or free >= FLOOR_MB:
            return None
        self.skipped.append((nodeid, free))
        return (
            f"not enough memory: {free:.0f} MB available after reclaiming, "
            f"floor is {FLOOR_MB} MB (LUCID_TEST_MEM_FLOOR_MB)"
        )

    def after_test(self) -> None:
        if not ENABLED:
            return
        self.peak_mb = max(self.peak_mb, resident_mb() + cache_mb())
        if cache_mb() >= RECLAIM_MB:
            self.reclaimed_mb += reclaim()
            self.reclaims += 1

    def summary(self) -> list[str]:
        """Lines for the terminal summary, or empty when nothing happened."""
        if not ENABLED:
            return []
        lines: list[str] = []
        if self.reclaims:
            lines.append(
                f"memory governor: reclaimed {self.reclaimed_mb / 1024:.1f} GB "
                f"over {self.reclaims} collection(s); peak footprint "
                f"{self.peak_mb:.0f} MB"
            )
        if self.skipped:
            lines.append(
                f"memory governor: SKIPPED {len(self.skipped)} test(s) — the "
                f"machine was short of memory, so this run checked less than a "
                f"full one:"
            )
            lines.extend(f"    {nodeid}  ({free:.0f} MB free)" for nodeid, free in self.skipped)
        return lines
