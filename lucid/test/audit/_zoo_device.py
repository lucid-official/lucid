"""Which devices the zoo is verified on, and which of them each stage is for.

The zoo is a set of deep-learning models, so verifying it means verifying it on
both devices a model runs on: Metal, where it trains, and the CPU.  A model that
works on one and not the other is a defect, and the two fail for different
reasons (MLX's laziness and dtype rules on Metal, Accelerate kernels on CPU).

Two things live here and nowhere else:

* **what ``--zoo-device`` means** — the devices a run takes and the order it
  takes them in (Metal first: it is the quick one, the device models train on,
  and its answer should not wait on the slow pass);
* **what a stage declares about devices** — a :class:`DeviceScope`.  A stage
  says whether it is for each device, for one pass, or for one device only; the
  run asks the scope and never branches on a stage's name.

A result is always stamped with the device it ran on (``StageResult.device``),
so the report can say *where* something failed, which is what triage needs:
:func:`where_failed` turns the results of a family into ``metal only`` /
``cpu only`` / ``both devices`` and is the one place that wording is made.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lucid.test.audit._zoo_stage import StageResult

#: The devices the zoo is verified on, in the order a run takes them.
ORDER = ("metal", "cpu")

#: What ``--zoo-device`` accepts: both, or one on purpose.
CHOICES = ("both", *ORDER)
DEFAULT_CHOICE = "both"

#: The device compiled training runs on (``COMPILE_DEVICE`` in the compile
#: tests): its matrix moves every model to Metal itself, so it has nothing to
#: say about the CPU.  A test pins that the two agree.
COMPILED_TRAINING_DEVICE = "metal"


def devices_for(choice: str) -> tuple[str, ...]:
    """The devices ``--zoo-device choice`` verifies, in the order they are run."""
    if choice == "both":
        return ORDER
    if choice in ORDER:
        return (choice,)
    raise ValueError(f"unknown zoo device {choice!r}; expected one of {CHOICES}")


class DeviceScope(ABC):
    """Which of a run's devices a stage is for."""

    @abstractmethod
    def covers(self, device: str, asked: Sequence[str]) -> bool:
        """Whether the stage runs in the pass for ``device``, when the run takes ``asked``."""

    @abstractmethod
    def describe(self) -> str:
        """The scope in a few words, for the legend of a note."""


@dataclass(frozen=True)
class EachDevice(DeviceScope):
    """The stage's answer depends on the device: it runs once per device the run takes."""

    def covers(self, device: str, asked: Sequence[str]) -> bool:
        del device, asked  # every pass of a run is one the stage is for
        return True

    def describe(self) -> str:
        return "각 장치에서"


@dataclass(frozen=True)
class OnceOnly(DeviceScope):
    """The stage's answer does not depend on the device: one pass of the run is enough.

    The first pass, which is Metal when Metal is asked for.
    """

    def covers(self, device: str, asked: Sequence[str]) -> bool:
        return device == asked[0]

    def describe(self) -> str:
        return "한 번 (장치와 무관)"


@dataclass(frozen=True)
class OnlyOn(DeviceScope):
    """The stage is about one device and has nothing to say about the others."""

    device: str

    def covers(self, device: str, asked: Sequence[str]) -> bool:
        del asked  # a run that does not take this device does not run the stage
        return device == self.device

    def describe(self) -> str:
        return f"{self.device} 에서만"


EACH_DEVICE = EachDevice()
ONCE_ONLY = OnceOnly()


def devices_run(results: Sequence[StageResult], stage: str | None = None) -> tuple[str, ...]:
    """The devices ``results`` were produced on (for the stage called ``stage``, if given)."""
    seen = {r.device for r in results if r.device and (stage is None or r.name == stage)}
    return tuple(d for d in ORDER if d in seen)


def where_failed(results: Sequence[StageResult], stage: str | None = None) -> str:
    """Where a failure happened, when the run could tell devices apart.

    ``results`` are a family's, from every device it ran on; ``stage`` narrows
    the question to the stage of that name.  ``"cpu only"`` is a claim that the
    other device ran the same and passed, so it is made only when both were run:
    a narrowed run, or a stage that is for one device, has nothing to compare
    and says nothing.
    """
    ran = devices_run(results, stage)
    bad = {r.device for r in results if r.failed and (stage is None or r.name == stage)}
    failing = tuple(d for d in ran if d in bad)
    if not failing or len(ran) < 2:
        return ""
    return "both devices" if failing == ran else f"{' and '.join(failing)} only"


__all__ = [
    "CHOICES",
    "COMPILED_TRAINING_DEVICE",
    "DEFAULT_CHOICE",
    "EACH_DEVICE",
    "ONCE_ONLY",
    "ORDER",
    "DeviceScope",
    "EachDevice",
    "OnceOnly",
    "OnlyOn",
    "devices_for",
    "devices_run",
    "where_failed",
]
