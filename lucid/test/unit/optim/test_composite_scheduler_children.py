"""What ``SequentialLR`` and ``ChainedScheduler`` accept as children, and how they run one.

``SequentialLR`` put every rate back to its base and took back its
children's step 0 before it looked at them, so a child it could not drive
— a ``ReduceLROnPlateau``, a ``ChainedScheduler`` — failed with an untyped
``AttributeError`` and left the optimizer altered.  The composites now check
their children first, as the reference does, and a ``ChainedScheduler``
runs inside a ``SequentialLR``.
"""

from collections.abc import Callable
from typing import Any

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.test.unit.optim._scheduler_runs import LR, assert_as_the_reference

type _Kids = Callable[[Any, Any, Any], list[Any]]
type _Make = Callable[[Any, Any, list[Any]], Any]

# (children on the optimizer — and one on another —, the composite over them)
_REFUSED: dict[str, tuple[type[Exception], _Kids, _Make]] = {
    "sequential-empty": (
        ValueError,
        lambda S, o, other: [],
        lambda S, o, kids: S.SequentialLR(o, kids, []),
    ),
    "sequential-other-optimizer": (
        ValueError,
        lambda S, o, other: [S.LinearLR(o), S.ExponentialLR(other, gamma=0.9)],
        lambda S, o, kids: S.SequentialLR(o, kids, [3]),
    ),
    "sequential-milestone-count": (
        ValueError,
        lambda S, o, other: [S.LinearLR(o), S.ExponentialLR(o, gamma=0.9)],
        lambda S, o, kids: S.SequentialLR(o, kids, [2, 3]),
    ),
    "sequential-plateau": (
        ValueError,
        lambda S, o, other: [S.LinearLR(o), S.ReduceLROnPlateau(o)],
        lambda S, o, kids: S.SequentialLR(o, kids, [3]),
    ),
    "sequential-not-a-scheduler": (
        TypeError,
        lambda S, o, other: [S.LinearLR(o), object()],
        lambda S, o, kids: S.SequentialLR(o, kids, [3]),
    ),
    "chained-empty": (
        ValueError,
        lambda S, o, other: [],
        lambda S, o, kids: S.ChainedScheduler(kids, o),
    ),
    "chained-other-optimizer": (
        ValueError,
        lambda S, o, other: [S.LinearLR(o), S.ExponentialLR(other, gamma=0.9)],
        lambda S, o, kids: S.ChainedScheduler(kids),
    ),
    "chained-plateau": (
        ValueError,
        lambda S, o, other: [S.ExponentialLR(o, gamma=0.9), S.ReduceLROnPlateau(o)],
        lambda S, o, kids: S.ChainedScheduler(kids),
    ),
    "chained-not-a-scheduler": (
        TypeError,
        lambda S, o, other: [S.ExponentialLR(o, gamma=0.9), object()],
        lambda S, o, kids: S.ChainedScheduler(kids),
    ),
}


def _refusal(lib: Any, case: str) -> tuple[type[Exception], list[Any], list[Any]]:
    """The error building ``case`` raised, and the rates and epochs before and after."""
    _, kids_of, make = _REFUSED[case]
    S = lib.optim.lr_scheduler
    opt = lib.optim.SGD([lib.nn.Parameter(lib.zeros(1))], lr=LR)
    other = lib.optim.SGD([lib.nn.Parameter(lib.zeros(1))], lr=LR)
    kids = kids_of(S, opt, other)

    def snapshot() -> list[Any]:
        rates = [float(g["lr"]) for g in opt.param_groups]
        return rates + [getattr(k, "last_epoch", None) for k in kids]

    before = snapshot()
    with pytest.raises((ValueError, TypeError)) as info:
        make(S, opt, kids)
    return info.type, before, snapshot()


@pytest.mark.parity
@pytest.mark.parametrize("case", list(_REFUSED))
def test_a_composite_refuses_a_child_it_cannot_drive_before_changing_anything(
    case: str, ref: Any
) -> None:
    want, _, _ = _refusal(ref, case)
    got, before, after = _refusal(lucid, case)
    assert got is want is _REFUSED[case][0]
    assert after == before


@pytest.mark.parametrize("case", [c for c, (err, _, _) in _REFUSED.items() if err is TypeError])
def test_a_child_that_is_not_a_scheduler_is_named_by_index(case: str) -> None:
    _, kids_of, make = _REFUSED[case]
    S = optim.lr_scheduler
    opt = optim.SGD([nn.Parameter(lucid.zeros(1))], lr=LR)
    with pytest.raises(TypeError, match="at index 1: expected .* got object"):
        make(S, opt, kids_of(S, opt, opt))


def _chain(S: Any, o: Any) -> Any:
    return S.ChainedScheduler([S.ExponentialLR(o, gamma=0.9), S.StepLR(o, step_size=2, gamma=0.5)])


@pytest.mark.parity
@pytest.mark.parametrize("add_at", [None, 0, 3, 6])
def test_sequential_runs_a_chain_first_as_the_reference_does(add_at: int | None, ref: Any) -> None:
    assert_as_the_reference(
        lambda S, o: S.SequentialLR(
            o, [_chain(S, o), S.LinearLR(o, start_factor=0.5)], milestones=[4]
        ),
        add_at,
        ref,
    )


@pytest.mark.parity
def test_a_chain_taking_over_at_a_milestone_starts_as_a_chain_run_first(
    ref: Any,
) -> None:
    """The reference has no closed form for a chain and fails at its milestone."""

    def rates(lib: Any, first: bool, steps: int) -> list[float]:
        S = lib.optim.lr_scheduler
        opt = lib.optim.SGD([lib.nn.Parameter(lib.zeros(1))], lr=LR)
        warmup = S.LinearLR(opt, start_factor=0.25, total_iters=3)
        kids = [_chain(S, opt), warmup] if first else [warmup, _chain(S, opt)]
        sched = S.SequentialLR(opt, kids, milestones=[100 if first else 3])
        out = [float(opt.param_groups[0]["lr"])]
        for _ in range(steps):
            opt.step()
            sched.step()
            out.append(float(opt.param_groups[0]["lr"]))
        return out

    with pytest.raises(NotImplementedError):
        rates(ref, first=False, steps=3)
    late = rates(lucid, first=False, steps=9)
    assert late[:3] == pytest.approx([0.025, 0.05, 0.075])
    assert late[3:] == pytest.approx(rates(lucid, first=True, steps=6))


@pytest.mark.parity
def test_an_older_lucids_sequential_checkpoint_resumes_on_schedule(ref: Any) -> None:
    """It counted ``last_epoch`` from -1 and kept the active child as ``_idx``."""

    def build(lib: Any) -> tuple[Any, Any]:
        S = lib.optim.lr_scheduler
        opt = lib.optim.SGD([lib.nn.Parameter(lib.zeros(1))], lr=LR)
        kids = [S.ConstantLR(opt, factor=0.5, total_iters=5), S.ExponentialLR(opt, 0.9)]
        return opt, S.SequentialLR(opt, kids, milestones=[3])

    def advance(opt: Any, sched: Any, steps: int) -> list[float]:
        out = []
        for _ in range(steps):
            opt.step()
            sched.step()
            out.append(float(opt.param_groups[0]["lr"]))
        return out

    ref_opt, ref_sched = build(ref)
    want = advance(ref_opt, ref_sched, 8)[2:]

    opt, sched = build(lucid)
    advance(opt, sched, 2)
    saved = sched.state_dict()
    # The shape the older Lucid wrote: no step 0 taken back on the children.
    older = {k: v for k, v in saved.items() if k != "schedulers"}
    older["last_epoch"] = saved["last_epoch"] - 1
    older["_idx"] = 0
    first, second = saved["schedulers"]
    older["schedulers"] = [first, {**second, "last_epoch": 0}]

    resumed_opt, resumed = build(lucid)
    resumed_opt.load_state_dict(opt.state_dict())
    resumed.load_state_dict(older)
    assert advance(resumed_opt, resumed, 6) == pytest.approx(want)


def _everything(opt: Any, scheds: list[Any]) -> list[Any]:
    """The rates and momenta, and every scheduler's own state."""
    return [dict(g, params=None) for g in opt.param_groups] + [dict(vars(s)) for s in scheds]


def _chain_step(S: Any, o: Any) -> tuple[Any, list[Any]]:
    kids = [S.ExponentialLR(o, gamma=0.9), S.LambdaLR(o, lambda e: 0.5**e)]
    chain = S.ChainedScheduler(kids)
    return chain, [chain, *kids]


def _chain_restart(S: Any, o: Any) -> tuple[Any, list[Any]]:
    kids = [S.ExponentialLR(o, gamma=0.9), S.LambdaLR(o, lambda e: 0.5**e)]
    chain = S.ChainedScheduler(kids)
    warmup = S.LinearLR(o)
    seq = S.SequentialLR(o, [warmup, chain], milestones=[3])
    return seq, [seq, warmup, chain, *kids]


# The chain's second child refuses a group added after it, once the first
# has already stepped: in a step of the chain's own, and in its restart at
# a SequentialLR milestone.
_HALF_STEPPED = {"chain-step": _chain_step, "chain-restart": _chain_restart}


@pytest.mark.parametrize("case", list(_HALF_STEPPED))
def test_a_refused_composite_step_changes_nothing(case: str) -> None:
    """A chain stepped its first child before the second refused."""
    opt = optim.SGD([nn.Parameter(lucid.zeros(1))], lr=LR, momentum=0.9)
    composite, scheds = _HALF_STEPPED[case](optim.lr_scheduler, opt)
    for _ in range(2):
        opt.step()
        composite.step()
    opt.add_param_group({"params": [nn.Parameter(lucid.zeros(1))], "lr": 0.2})
    before = _everything(opt, scheds)
    with pytest.raises(ValueError, match="param group 1"):
        composite.step()
    assert _everything(opt, scheds) == before
