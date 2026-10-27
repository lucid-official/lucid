"""A param group added after the scheduler gets the schedule the reference gives it.

Most schedules scale each group's *current* rate, so the reference carries
a group added with ``add_param_group`` through them from the rate it was
added with, and a ``ChainedScheduler`` multiplies every child's factor into
it.  ``ConstantLR``, ``LinearLR``, ``CosineAnnealingLR`` and ``PolynomialLR``
computed from the base rates they had captured instead, so a late group
kept its rate; the ``ChainedScheduler`` built per-child tracks on top of
that and gave a late group the last child's factor only — StepLR with
ExponentialLR left it at 0.0405 where the reference has 0.02025 (LCD-492).
The schedules that do compute from their base rates alone refuse a late
group, as the reference's do.  ``SequentialLR`` switched children one step
late and carried the outgoing child's rate into the incoming one.
"""

import itertools
from typing import Any

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.test.unit.optim._scheduler_runs import (
    LATE_LR,
    LR,
    Build,
    assert_as_the_reference,
)

_MILESTONE = 4
# Never, before the first step, before the milestone, after it.
_ADD_AT = [None, 0, 3, 6]

# Each step scales the rate the group has: a late group is carried along.
_CHAINABLE: dict[str, Build] = {
    "StepLR": lambda S, o, **kw: S.StepLR(o, step_size=2, gamma=0.5, **kw),
    "MultiStepLR": lambda S, o, **kw: S.MultiStepLR(o, milestones=[2, 5, 5], gamma=0.5, **kw),
    "ExponentialLR": lambda S, o, **kw: S.ExponentialLR(o, gamma=0.9, **kw),
    "ConstantLR": lambda S, o, **kw: S.ConstantLR(o, factor=0.5, total_iters=4, **kw),
    "LinearLR": lambda S, o, **kw: S.LinearLR(o, start_factor=0.25, total_iters=4, **kw),
    "CosineAnnealingLR": lambda S, o, **kw: S.CosineAnnealingLR(o, T_max=10, eta_min=0.01, **kw),
    "PolynomialLR": lambda S, o, **kw: S.PolynomialLR(o, total_iters=6, power=2.0, **kw),
}

# Each rate comes from what was captured at construction: a late group has
# none, and the step is refused.
_BASE_RATED: dict[str, Build] = {
    "LambdaLR": lambda S, o: S.LambdaLR(o, lambda e: 0.5**e),
    "MultiplicativeLR": lambda S, o: S.MultiplicativeLR(o, lambda e: 0.9),
    "CyclicLR": lambda S, o: S.CyclicLR(o, base_lr=0.01, max_lr=0.2, step_size_up=2),
    "OneCycleLR": lambda S, o: S.OneCycleLR(o, max_lr=0.1, total_steps=12),
    "CosineAnnealingWarmRestarts": lambda S, o: S.CosineAnnealingWarmRestarts(o, T_0=3),
    # Off the trough the rise is a share of the base rate.
    "CosineAnnealingLR-restart": lambda S, o: S.CosineAnnealingLR(o, T_max=4),
}

_ALL = {**_CHAINABLE, **_BASE_RATED}


def _chained(first: str, second: str) -> Build:
    return lambda S, o: S.ChainedScheduler([_ALL[first](S, o), _ALL[second](S, o)])


def _sequential(first: str, second: str) -> Build:
    return lambda S, o: S.SequentialLR(
        o, [_ALL[first](S, o), _ALL[second](S, o)], milestones=[_MILESTONE]
    )


@pytest.mark.parity
@pytest.mark.parametrize("add_at", _ADD_AT)
@pytest.mark.parametrize("name", list(_CHAINABLE))
def test_a_chainable_schedule_carries_a_late_group_from_its_own_rate(
    name: str, add_at: int | None, ref: Any
) -> None:
    assert_as_the_reference(_CHAINABLE[name], add_at, ref)


@pytest.mark.parity
@pytest.mark.parametrize("add_at", _ADD_AT)
@pytest.mark.parametrize("name", list(_BASE_RATED))
def test_a_base_rated_schedule_refuses_a_late_group(
    name: str, add_at: int | None, ref: Any
) -> None:
    assert_as_the_reference(_BASE_RATED[name], add_at, ref)


_CHAINABLE_PAIRS = list(itertools.product(_CHAINABLE, repeat=2))
# A base-rated child writes its rate over the rate it is handed.
_MIXED_PAIRS = [
    ("LambdaLR", "ExponentialLR"),
    ("ExponentialLR", "LambdaLR"),
    ("CyclicLR", "StepLR"),
    ("LinearLR", "MultiplicativeLR"),
    ("MultiplicativeLR", "ConstantLR"),
]
_PAIRS = _CHAINABLE_PAIRS + _MIXED_PAIRS


@pytest.mark.parity
@pytest.mark.parametrize("add_at", _ADD_AT)
@pytest.mark.parametrize(("first", "second"), _PAIRS, ids=[f"{a}+{b}" for a, b in _PAIRS])
def test_a_chain_composes_its_children_on_every_group(
    first: str, second: str, add_at: int | None, ref: Any
) -> None:
    assert_as_the_reference(_chained(first, second), add_at, ref)


@pytest.mark.parity
@pytest.mark.parametrize("add_at", _ADD_AT)
@pytest.mark.parametrize(("first", "second"), _PAIRS, ids=[f"{a}|{b}" for a, b in _PAIRS])
def test_sequential_hands_over_at_its_milestone_from_the_base(
    first: str, second: str, add_at: int | None, ref: Any
) -> None:
    assert_as_the_reference(_sequential(first, second), add_at, ref)


def test_a_late_group_gets_the_whole_chains_factor() -> None:
    """The issue's numbers: each child's factor, not the last one's."""
    p1, p2 = nn.Parameter(lucid.zeros(1)), nn.Parameter(lucid.zeros(1))
    opt = optim.SGD([p1], lr=0.1)
    S = optim.lr_scheduler
    chain = S.ChainedScheduler(
        [S.StepLR(opt, step_size=1, gamma=0.5), S.ExponentialLR(opt, gamma=0.9)]
    )
    opt.add_param_group({"params": [p2], "lr": 0.1})
    for _ in range(2):
        opt.step()
        chain.step()
    assert opt.param_groups[1]["lr"] == pytest.approx(0.1 * (0.5 * 0.9) ** 2)


@pytest.mark.parametrize("name", list(_BASE_RATED))
def test_a_refused_step_changes_nothing(name: str) -> None:
    opt = optim.SGD([nn.Parameter(lucid.zeros(1))], lr=LR, momentum=0.9)
    sched = _BASE_RATED[name](optim.lr_scheduler, opt)
    for _ in range(4):
        opt.step()
        sched.step()
    opt.add_param_group({"params": [nn.Parameter(lucid.zeros(1))], "lr": LATE_LR})
    groups = [dict(g, params=None) for g in opt.param_groups]
    state = {k: v for k, v in vars(sched).items() if k != "optimizer"}
    with pytest.raises(ValueError, match="param group 1"):
        sched.step()
    assert [dict(g, params=None) for g in opt.param_groups] == groups
    assert {k: v for k, v in vars(sched).items() if k != "optimizer"} == state


@pytest.mark.parity
def test_one_cycle_schedules_a_late_group_that_brings_its_bounds(ref: Any) -> None:
    """The phases read their bounds off each group, so a group can bring its own."""
    bounds = {
        "initial_lr": 0.02,
        "max_lr": 0.4,
        "min_lr": 1e-4,
        "max_momentum": 0.95,
        "base_momentum": 0.85,
    }
    assert_as_the_reference(_BASE_RATED["OneCycleLR"], 3, ref, late_keys=bounds)


@pytest.mark.parity
@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("LinearLR", {"start_factor": 0.0}),
        ("LinearLR", {"start_factor": 1.5}),
        ("LinearLR", {"end_factor": -0.1}),
        ("LinearLR", {"end_factor": 1.5}),
        ("ConstantLR", {"factor": -0.1}),
        ("ConstantLR", {"factor": 1.5}),
    ],
)
def test_a_factor_out_of_range_is_refused_at_construction(
    name: str, kwargs: dict[str, float], ref: Any
) -> None:
    with pytest.raises(ValueError):
        getattr(ref.optim.lr_scheduler, name)(
            ref.optim.SGD([ref.nn.Parameter(ref.zeros(1))], lr=LR), **kwargs
        )
    with pytest.raises(ValueError, match=name):
        getattr(optim.lr_scheduler, name)(
            optim.SGD([nn.Parameter(lucid.zeros(1))], lr=LR), **kwargs
        )


# The reference divides by a zero factor at total_iters; never reaching
# it — total_iters 0, or a milestone before it — is fine in both.
_ZERO_FACTOR: dict[str, Build] = {
    "plain": lambda S, o: S.ConstantLR(o, factor=0.0, total_iters=3),
    "total_iters-0": lambda S, o: S.ConstantLR(o, factor=0.0, total_iters=0),
    "chained": lambda S, o: S.ChainedScheduler(
        [S.ConstantLR(o, factor=0.0, total_iters=3), S.ExponentialLR(o, gamma=0.9)]
    ),
    "milestone-first": lambda S, o: S.SequentialLR(
        o,
        [S.ConstantLR(o, factor=0.0, total_iters=3), S.ExponentialLR(o, gamma=0.9)],
        milestones=[2],
    ),
    "milestone-after": lambda S, o: S.SequentialLR(
        o,
        [S.ConstantLR(o, factor=0.0, total_iters=3), S.ExponentialLR(o, gamma=0.9)],
        milestones=[5],
    ),
}


@pytest.mark.parity
@pytest.mark.parametrize("case", list(_ZERO_FACTOR))
def test_a_zero_constant_factor_is_refused_only_where_it_is_undone(case: str, ref: Any) -> None:
    assert_as_the_reference(_ZERO_FACTOR[case], None, ref)


def test_a_zero_constant_factor_names_the_way_out() -> None:
    opt = optim.SGD([nn.Parameter(lucid.zeros(1))], lr=LR)
    sched = optim.lr_scheduler.ConstantLR(opt, factor=0.0, total_iters=2)
    sched.step()
    with pytest.raises(ValueError, match="factor=0 cannot be restored at total_iters"):
        sched.step()
    assert sched.last_epoch == 1


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_CHAINABLE))
def test_a_resumed_chainable_schedule_does_not_step_on_construction(name: str, ref: Any) -> None:
    """Built with ``last_epoch``, it leaves the rate a checkpoint restored."""

    def resumed(S: Any, o: Any) -> Any:
        o.param_groups[0]["initial_lr"] = LR
        o.param_groups[0]["lr"] = 0.05
        return _CHAINABLE[name](S, o, last_epoch=2)

    assert_as_the_reference(resumed, None, ref)


@pytest.mark.parity
@pytest.mark.parametrize("last_epoch", [0, 1, 2, 4, 7])
def test_resumed_warm_restarts_continue_their_cycle(last_epoch: int, ref: Any) -> None:
    """``last_epoch`` puts the cycle that far in; it was always back at its start."""

    def resumed(S: Any, o: Any) -> Any:
        o.param_groups[0]["initial_lr"] = LR
        return S.CosineAnnealingWarmRestarts(o, T_0=3, T_mult=2, last_epoch=last_epoch)

    assert_as_the_reference(resumed, None, ref)
