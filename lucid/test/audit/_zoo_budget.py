"""What the machine can afford, and what the zoo stage does about it.

The zoo stage used to carry four fixed numbers tuned on a 36 GB Mac Studio.
On a 16 GB MacBook the first of them (a stage may grow to 28 GB) is a number
the machine does not have, so the stage would be killed by jetsam instead of
by its own guard, with no record of why.  :class:`MachineBudget` derives the
same four numbers from ``hw.memsize`` in the proportions they had where they
were tuned, so the Studio keeps exactly its old values and every other
machine gets the same *margins*.

Nothing adaptive is silent.  Whatever a budget causes the stage to skip or to
retry is recorded where the result is — see ``StageResult.notes`` — because a
run that quietly checked less than the last one is the one failure worse than
a run that was killed.
"""

import ctypes
import ctypes.util
import functools
import math
import re
import subprocess
from dataclasses import dataclass

_GIB = 1024**3

# The proportions of the machine the numbers were tuned on: a 36 GB Mac Studio,
# where a stage is killed at 28 GB of physical footprint, waits for 14 GB to be
# free before it starts, and is killed if the machine itself falls under 2.5 GB.
_LIMIT_SHARE = 28.0 / 36.0
_START_SHARE_OF_LIMIT = 0.5
_FLOOR_SHARE = 2.5 / 36.0

#: How much of the footprint limit a single model build may claim.
_BUILD_SHARE = 0.8
_BYTES_PER_PARAM = 4

#: Numbers are reported and compared in half-gigabyte steps, which keeps the
#: derived values readable ("12.5 GB", not "12.444444 GB").
_STEP_GB = 0.5
_MIN_FLOOR_GB = 1.0

#: The peak physical footprint of the largest ``heavy``-marked test.  Measured,
#: not estimated: each of the 24 heavy tests was run in its own process on the
#: 36 GB Mac Studio under ``/usr/bin/time -l`` (the "peak memory footprint"
#: line, interpreter and ``import lucid`` included), 2026-11-05.  They peaked
#: between 1.1 and 5.5 GB; the largest was ``HEAVY_PEAK_TEST``, which builds the
#: 632M-parameter ViT-H/14 (2.5 GB of weights) and runs it.  The VGG-16 trunk
#: (138M parameters) that gave the marker its name peaks at 1.7-3.1 GB.
HEAVY_PEAK_GB = 5.5
HEAVY_PEAK_TEST = "lucid/test/parity/models/vision/test_vit.py::test_self_consistency[vit_huge_14]"

#: How much room over the measured peak a machine needs to be offered the
#: ``heavy`` tests.  The guard kills a stage at the footprint limit, and a
#: peak measured on an otherwise idle machine is not the peak next to a build
#: or an editor; half again is the room that leaves.  With this a 16 GB machine
#: (limit 12.5 GB) runs them and an 8 GB machine (limit 6 GB) deselects them,
#: which are the two sizes the choice matters for.  A wrong guess costs one
#: retry, not a wrong answer: a stage the guard kills is rerun without ``heavy``.
HEAVY_HEADROOM = 1.5


def _round_step(value: float) -> float:
    """Round half up to the nearest step."""
    return math.floor(value / _STEP_GB + 0.5) * _STEP_GB


@dataclass(frozen=True)
class MachineBudget:
    """The memory limits one run of the zoo stage is held to.

    Attributes
    ----------
    total_gb : float
        Physical memory of the machine.
    limit_gb : float
        Footprint at which a stage is killed.
    start_room_gb : float
        Free memory a stage waits for before it starts, since an engine build
        may be sharing the machine.
    floor_gb : float
        Free memory of the machine below which a stage is killed, whatever its
        own footprint is.
    """

    total_gb: float
    limit_gb: float
    start_room_gb: float
    floor_gb: float

    @classmethod
    def for_memory(cls, total_gb: float, limit_gb: float | None = None) -> MachineBudget:
        """The budget of a machine with ``total_gb`` of memory.

        ``limit_gb`` overrides the footprint limit; the room to start follows it
        (half of it, as it always has), and the floor follows the machine.
        """
        if total_gb <= 0 or (limit_gb is not None and limit_gb <= 0):
            raise ValueError(f"memory sizes must be positive, got {total_gb=} {limit_gb=}")
        limit = limit_gb if limit_gb is not None else _round_step(total_gb * _LIMIT_SHARE)
        return cls(
            total_gb=total_gb,
            limit_gb=limit,
            start_room_gb=_round_step(limit * _START_SHARE_OF_LIMIT),
            floor_gb=max(_round_step(total_gb * _FLOOR_SHARE), _MIN_FLOOR_GB),
        )

    @classmethod
    def detect(cls, limit_gb: float | None = None) -> MachineBudget:
        """The budget of the machine this runs on, from ``hw.memsize``."""
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
        ).stdout
        return cls.for_memory(int(out.strip()) / _GIB, limit_gb)

    @property
    def build_budget_gb(self) -> float:
        """The most a single real-size model build may allocate."""
        return self.limit_gb * _BUILD_SHARE

    def fits_build(self, params: int) -> bool:
        """Whether ``params`` float32 weights fit the build budget."""
        return params * _BYTES_PER_PARAM <= self.build_budget_gb * _GIB

    @property
    def admits_heavy(self) -> bool:
        """Whether the largest ``heavy`` test fits under the footprint limit, with headroom."""
        return self.limit_gb >= HEAVY_PEAK_GB * HEAVY_HEADROOM

    def describe(self) -> str:
        return (
            f"{self.total_gb:.0f} GB machine · footprint limit {self.limit_gb:g} GB · "
            f"starts with {self.start_room_gb:g} GB free · floor {self.floor_gb:g} GB · "
            f"builds up to {self.build_budget_gb:.1f} GB · "
            f"heavy tests {'on' if self.admits_heavy else 'deselected'}"
        )


# ── reading the machine ──────────────────────────────────────────────────────


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


@functools.cache
def _libproc() -> ctypes.CDLL:
    return ctypes.CDLL(ctypes.util.find_library("proc") or "libproc.dylib", use_errno=True)


def footprint_gb(pid: int) -> float:
    """Physical footprint of ``pid``, which counts Metal buffers where resident size does not."""
    info = _RusageInfo()
    if _libproc().proc_pid_rusage(pid, 2, ctypes.byref(info)) != 0:
        return 0.0
    return info.phys_footprint / _GIB


def available_gb() -> float:
    """Memory the machine could hand out now: free, inactive, speculative, purgeable pages."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    size = re.search(r"page size of (\d+) bytes", out)
    page = int(size.group(1)) if size else 16384
    pages = 0
    for label in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        found = re.search(rf"{label}:\s+(\d+)\.", out)
        pages += int(found.group(1)) if found else 0
    return pages * page / _GIB


__all__ = [
    "HEAVY_HEADROOM",
    "HEAVY_PEAK_GB",
    "HEAVY_PEAK_TEST",
    "MachineBudget",
    "available_gb",
    "footprint_gb",
]
