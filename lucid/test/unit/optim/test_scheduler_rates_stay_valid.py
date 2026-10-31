"""A schedule run to its end, and one step past it, never writes a rate the optimizer refuses.

A linear ``OneCycleLR`` extrapolates on its last step -- the step after the
final batch, which a loop that steps the scheduler once per batch always
takes -- to a negative rate, and a cosine from a base below ``eta_min`` lands
a few ulps under zero at the bottom of a cycle.  The reference writes both
and its optimizer takes them; Lucid's optimizer refuses a negative rate, so
the ``InvalidArgument`` came out of ``scheduler.step()`` at the end of an
otherwise finished run (LCD-787).  The write path now holds the schedule at
the nearest rate the optimizer takes, so every other step is the reference's.
"""

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import lucid

type Build = Callable[[Any, Any], Any]

_LRS = (0.1, 0.05, 0.0)


def _optimizer(lib: ModuleType, kind: str) -> Any:
    params = [lib.nn.Parameter(lib.zeros(2)) for _ in _LRS]
    groups = [{"params": [p], "lr": lr} for p, lr in zip(params, _LRS, strict=True)]
    if kind == "sgd":
        return lib.optim.SGD(groups, lr=0.1, momentum=0.9)
    return lib.optim.Adam(groups, lr=0.1)


def _trajectory(lib: ModuleType, build: Build, kind: str, steps: int) -> list[list[float]]:
    opt = _optimizer(lib, kind)
    sched = build(lib.optim.lr_scheduler, opt)
    rows = []
    for _ in range(steps):
        opt.step()
        sched.step()
        rows.append([float(g["lr"]) for g in opt.param_groups])
    return rows


def _one_cycle(strategy: str, three_phase: bool) -> Build:
    return lambda S, o: S.OneCycleLR(
        o,
        [0.2, 0.1, 0.05],
        total_steps=100,
        anneal_strategy=strategy,
        three_phase=three_phase,
        pct_start=0.25,
    )


# (schedule, the steps before the last one it takes).  A one-cycle schedule
# takes ``total_steps`` steps and refuses the next, so its last is the 100th.
_SCHEDULES: dict[str, tuple[Build, int]] = {
    "OneCycleLR-linear": (_one_cycle("linear", False), 99),
    "OneCycleLR-linear-3phase": (_one_cycle("linear", True), 99),
    "OneCycleLR-cos": (_one_cycle("cos", False), 99),
    "OneCycleLR-cos-3phase": (_one_cycle("cos", True), 99),
    "CosineAnnealingLR-eta-above-base": (
        lambda S, o: S.CosineAnnealingLR(o, T_max=10, eta_min=0.01),
        40,
    ),
    "CosineAnnealingLR": (lambda S, o: S.CosineAnnealingLR(o, T_max=17), 68),
    "CosineAnnealingWarmRestarts": (
        lambda S, o: S.CosineAnnealingWarmRestarts(o, T_0=5, T_mult=2, eta_min=0.01),
        60,
    ),
    "StepLR": (lambda S, o: S.StepLR(o, 7, 0.5), 30),
    "MultiStepLR": (lambda S, o: S.MultiStepLR(o, [5, 20], 0.3), 30),
    "ExponentialLR": (lambda S, o: S.ExponentialLR(o, 0.95), 30),
    "LinearLR": (lambda S, o: S.LinearLR(o, 0.1, 1.0, 10), 30),
    "ConstantLR": (lambda S, o: S.ConstantLR(o, 0.25, 12), 30),
    "PolynomialLR": (lambda S, o: S.PolynomialLR(o, 30, 2.0), 40),
    "CyclicLR": (lambda S, o: S.CyclicLR(o, 0.01, 0.2, 5, 8, mode="triangular2"), 60),
    "CyclicLR-exp-range": (
        lambda S, o: S.CyclicLR(o, 0.01, 0.2, 5, 8, mode="exp_range", gamma=0.95),
        60,
    ),
    "LambdaLR": (lambda S, o: S.LambdaLR(o, lambda e: 0.9**e), 30),
    "MultiplicativeLR": (lambda S, o: S.MultiplicativeLR(o, lambda e: 0.95), 30),
    "SequentialLR": (
        lambda S, o: S.SequentialLR(o, [S.LinearLR(o, 0.1, 1, 5), S.CosineAnnealingLR(o, 20)], [5]),
        40,
    ),
    "ChainedScheduler": (
        lambda S, o: S.ChainedScheduler([S.ConstantLR(o, 0.1, 3), S.ExponentialLR(o, 0.9)]),
        30,
    ),
}


@pytest.mark.parametrize("kind", ["sgd", "adam"])
@pytest.mark.parametrize("name", list(_SCHEDULES))
def test_a_schedule_runs_one_step_past_its_length_without_raising(name: str, kind: str) -> None:
    build, length = _SCHEDULES[name]
    for row in _trajectory(lucid, build, kind, length + 1):
        assert all(lr >= 0.0 for lr in row)


@pytest.mark.parity
@pytest.mark.parametrize("kind", ["sgd", "adam"])
@pytest.mark.parametrize("name", list(_SCHEDULES))
def test_the_trajectory_is_the_references_but_for_what_the_optimizer_refuses(
    name: str, kind: str, ref: Any
) -> None:
    build, length = _SCHEDULES[name]
    got = _trajectory(lucid, build, kind, length + 1)
    want = _trajectory(ref, build, kind, length + 1)
    for step, (g, w) in enumerate(zip(got, want, strict=True)):
        # The reference's negative rates are the ones held at zero.
        assert g == pytest.approx([max(r, 0.0) for r in w], rel=1e-9, abs=1e-12), step


def test_a_rate_a_callable_returns_is_not_held_for_the_caller() -> None:
    # LambdaLR and MultiplicativeLR apply the caller's function: a negative
    # result is theirs, and the optimizer says so rather than the schedule
    # hiding it.
    opt = _optimizer(lucid, "sgd")
    sched = lucid.optim.lr_scheduler.LambdaLR(opt, lambda e: 1.0 if e == 0 else -1.0)
    opt.step()
    with pytest.raises(ValueError, match="lr must be >= 0"):
        sched.step()
