"""Run one schedule in Lucid and in the reference, and compare every rate.

A builder ``build(S, optimizer)`` gets the framework's ``lr_scheduler``
module, so one definition builds the schedule in both.
"""

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import lucid

LR, LATE_LR, STEPS = 0.1, 0.2, 10

type Build = Callable[..., Any]


def run(
    lib: ModuleType,
    build: Build,
    add_at: int | None,
    refused: tuple[type[Exception], ...],
    late_keys: dict[str, float],
) -> tuple[list[tuple[float, ...]], type[Exception] | None]:
    """Every group's rate before each step, and the error that ended the run."""
    opt = lib.optim.SGD([lib.nn.Parameter(lib.zeros(1))], lr=LR, momentum=0.9)
    sched = build(lib.optim.lr_scheduler, opt)
    rates: list[tuple[float, ...]] = []
    for k in range(STEPS + 1):
        if k == add_at:
            late = lib.nn.Parameter(lib.zeros(1))
            opt.add_param_group({"params": [late], "lr": LATE_LR, **late_keys})
        rates.append(tuple(float(g["lr"]) for g in opt.param_groups))
        if k == STEPS:
            break
        opt.step()
        try:
            sched.step()
        except refused as exc:
            return rates, type(exc)
    return rates, None


def assert_as_the_reference(
    build: Build, add_at: int | None, ref: Any, late_keys: dict[str, float] | None = None
) -> None:
    """Lucid's rates equal the reference's, and a refusal comes at the same step."""
    keys = late_keys or {}
    # The reference refuses with a strict zip's ValueError, a KeyError on a
    # group (OneCycleLR) or a ZeroDivisionError (ConstantLR at factor 0);
    # Lucid with one typed ValueError.
    want, want_err = run(ref, build, add_at, (ValueError, KeyError, ZeroDivisionError), keys)
    got, got_err = run(lucid, build, add_at, (ValueError,), keys)
    assert (got_err is None) == (want_err is None), (got_err, want_err)
    assert len(got) == len(want)
    for step, (g, w) in enumerate(zip(got, want, strict=True)):
        assert g == pytest.approx(w, rel=1e-9, abs=1e-12), f"before step {step}"
