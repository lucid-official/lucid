"""A SequentialLR can wrap a SequentialLR; a warm-restarts cycle is public state.

* Both composites admit every kind of scheduler a composite can step —
  an LR scheduler, a ``SequentialLR``, a ``ChainedScheduler`` — by one rule.
* ``CosineAnnealingWarmRestarts`` keeps its position in the cycle as
  ``T_cur`` / ``T_i``, as the reference does, and a checkpoint written
  under the older private names still loads.
"""

from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import lucid
import lucid.optim as optim

LR = 1.0

type Build = Callable[[Any, Any], Any]


def _inner(S: Any, o: Any) -> Any:
    return S.SequentialLR(o, [S.StepLR(o, 2, gamma=0.5), S.ExponentialLR(o, 0.9)], [3])


def _inner_first(S: Any, o: Any) -> Any:
    return S.SequentialLR(o, [_inner(S, o), S.StepLR(o, 2, gamma=0.5)], [5])


def _inner_second(S: Any, o: Any) -> Any:
    warmup = S.ConstantLR(o, factor=0.5, total_iters=2)
    return S.SequentialLR(o, [warmup, _inner(S, o)], [4])


def _rates(lib: ModuleType, build: Build, steps: int) -> list[float]:
    opt = lib.optim.SGD([lib.nn.Parameter(lib.zeros(1))], lr=LR)
    sched = build(lib.optim.lr_scheduler, opt)
    out = [float(opt.param_groups[0]["lr"])]
    for _ in range(steps):
        opt.step()
        sched.step()
        out.append(float(opt.param_groups[0]["lr"]))
    return out


def _children_of_every_kind(S: Any, o: Any) -> dict[str, Any]:
    return {
        "lr": S.StepLR(o, 2),
        "sequential": S.SequentialLR(o, [S.StepLR(o, 2), S.ExponentialLR(o, 0.9)], [3]),
        "chained": S.ChainedScheduler([S.StepLR(o, 2), S.ExponentialLR(o, 0.9)]),
    }


@pytest.mark.parametrize("kind", ["lr", "sequential", "chained"])
@pytest.mark.parametrize("composite", ["sequential", "chained"])
def test_a_composite_admits_every_kind_of_composable_child(composite: str, kind: str) -> None:
    S = optim.lr_scheduler
    opt = optim.SGD([lucid.nn.Parameter(lucid.zeros(1))], lr=LR)
    child = _children_of_every_kind(S, opt)[kind]
    if composite == "sequential":
        sched = S.SequentialLR(opt, [child, S.StepLR(opt, 2)], [5])
    else:
        sched = S.ChainedScheduler([child, S.StepLR(opt, 2)])
    for _ in range(8):
        opt.step()
        sched.step()
    assert all(lr > 0.0 for lr in sched.get_last_lr())


@pytest.mark.parity
def test_a_nested_sequential_runs_its_phase_then_the_outer_next(ref: Any) -> None:
    """Held to the reference after the outer milestone, and to a standalone run before it.

    The reference double-steps a SequentialLR that runs first in another, so
    its inner phase runs ahead of the same SequentialLR on its own.
    """
    got = _rates(lucid, _inner_first, 14)
    assert got[5:] == pytest.approx(_rates(ref, _inner_first, 14)[5:])
    assert got[:5] == pytest.approx(_rates(ref, _inner, 4))


@pytest.mark.parity
def test_a_sequential_taking_over_at_a_milestone_starts_over(ref: Any) -> None:
    """The reference has no closed form for it and fails at the milestone."""
    with pytest.raises(NotImplementedError):
        _rates(ref, _inner_second, 5)
    got = _rates(lucid, _inner_second, 14)
    assert got[:4] == pytest.approx([0.5, 0.5, 1.0, 1.0])
    assert got[4:] == pytest.approx(_rates(ref, _inner, 10))


def test_a_nested_sequential_resumes_from_its_checkpoint() -> None:
    S = optim.lr_scheduler

    def build() -> tuple[Any, Any]:
        opt = optim.SGD([lucid.nn.Parameter(lucid.zeros(1))], lr=LR)
        return opt, _inner_second(S, opt)

    def advance(opt: Any, sched: Any, steps: int) -> list[float]:
        out = []
        for _ in range(steps):
            opt.step()
            sched.step()
            out.append(float(opt.param_groups[0]["lr"]))
        return out

    opt, sched = build()
    advance(opt, sched, 7)
    resumed_opt, resumed = build()
    resumed_opt.load_state_dict(opt.state_dict())
    resumed.load_state_dict(sched.state_dict())
    assert advance(resumed_opt, resumed, 6) == pytest.approx(advance(opt, sched, 6))


def _warm_restarts(lib: Any, **kwargs: Any) -> tuple[Any, Any]:
    opt = lib.optim.SGD([lib.nn.Parameter(lib.zeros(1))], lr=0.1)
    return opt, lib.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=3, T_mult=2, **kwargs)


@pytest.mark.parity
def test_the_cycle_position_is_public_and_follows_the_reference(ref: Any) -> None:
    ref_opt, want = _warm_restarts(ref)
    opt, got = _warm_restarts(lucid)
    for step in range(20):
        assert (got.T_cur, got.T_i) == pytest.approx((want.T_cur, want.T_i)), step
        ref_opt.step()
        want.step()
        opt.step()
        got.step()
    # The fractional-epoch form places the cycle too.
    want.step(31.5)
    got.step(31.5)
    assert (got.T_cur, got.T_i) == pytest.approx((want.T_cur, want.T_i))


@pytest.mark.parity
def test_the_cycle_is_checkpointed_under_the_reference_names(ref: Any) -> None:
    _, want = _warm_restarts(ref)
    _, got = _warm_restarts(lucid)
    for _ in range(5):
        want.step()
        got.step()
    cycle = ("T_0", "T_mult", "eta_min", "T_cur", "T_i")
    assert {k: got.state_dict()[k] for k in cycle} == pytest.approx(
        {k: want.state_dict()[k] for k in cycle}
    )
    assert not {"_T_cur", "_T_i"} & set(got.state_dict())


def test_an_older_lucids_cycle_checkpoint_resumes_on_schedule() -> None:
    opt, sched = _warm_restarts(lucid)
    for _ in range(5):
        opt.step()
        sched.step()
    saved = sched.state_dict()
    older = {k: v for k, v in saved.items() if k not in ("T_cur", "T_i")}
    older["_T_cur"], older["_T_i"] = saved["T_cur"], saved["T_i"]

    resumed_opt, resumed = _warm_restarts(lucid)
    resumed_opt.load_state_dict(opt.state_dict())
    resumed.load_state_dict(older)
    assert (resumed.T_cur, resumed.T_i) == (sched.T_cur, sched.T_i)
    assert "_T_cur" not in vars(resumed)
    for _ in range(9):
        opt.step()
        sched.step()
        resumed_opt.step()
        resumed.step()
        assert resumed.get_last_lr() == pytest.approx(sched.get_last_lr())
