"""A scheduler stepped before its optimizer warns, as the reference's does (LCD-567).

Stepping the scheduler first skips the schedule's first value.  This is the
classic training loop with the two calls the wrong way round.  The
reference's scheduler puts a counter in place of ``optimizer.step``.  It
warns on its own first step when the optimizer has not stepped, or when
``optimizer.step`` was replaced after the counter.  Lucid kept no count and
never warned.

The check lives in the scheduler base, so every schedule built on it
warns, and the composites warn through their children.  The two schedules
whose step skips the check, ``ReduceLROnPlateau`` and
``CosineAnnealingWarmRestarts``, stay silent, as in the reference.  A
compiled step drives the optimizer without ``optimizer.step`` and reports
the step itself, so a compiled loop is silent too.
"""

import copy
import gc
import inspect
import pickle
import warnings
import weakref
from collections.abc import Callable
from types import ModuleType
from typing import Any

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.test.unit.optim._scheduler_runs import LR

type Build = Callable[[Any, Any], Any]
type Loop = Callable[[Any, Any], None]

_STEPS = 5

_SCHEDULES: dict[str, Build] = {
    "StepLR": lambda S, o: S.StepLR(o, step_size=2),
    "MultiStepLR": lambda S, o: S.MultiStepLR(o, milestones=[2, 4]),
    "ExponentialLR": lambda S, o: S.ExponentialLR(o, gamma=0.9),
    "ConstantLR": lambda S, o: S.ConstantLR(o, factor=0.5, total_iters=2),
    "LinearLR": lambda S, o: S.LinearLR(o, start_factor=0.5, total_iters=2),
    "CosineAnnealingLR": lambda S, o: S.CosineAnnealingLR(o, T_max=4),
    "PolynomialLR": lambda S, o: S.PolynomialLR(o, total_iters=4),
    "LambdaLR": lambda S, o: S.LambdaLR(o, lambda e: 0.5**e),
    "MultiplicativeLR": lambda S, o: S.MultiplicativeLR(o, lambda e: 0.9),
    "CyclicLR": lambda S, o: S.CyclicLR(o, base_lr=0.01, max_lr=0.2, step_size_up=2),
    "OneCycleLR": lambda S, o: S.OneCycleLR(o, max_lr=0.1, total_steps=12),
    "CosineAnnealingWarmRestarts": lambda S, o: S.CosineAnnealingWarmRestarts(o, T_0=3),
    "ReduceLROnPlateau": lambda S, o: S.ReduceLROnPlateau(o),
    "ChainedScheduler": lambda S, o: S.ChainedScheduler(
        [S.StepLR(o, step_size=2), S.ExponentialLR(o, gamma=0.9)]
    ),
    # The child after the milestone checks again on its own first step.
    "SequentialLR": lambda S, o: S.SequentialLR(
        o,
        [S.ConstantLR(o, factor=0.5, total_iters=2), S.ExponentialLR(o, gamma=0.9)],
        milestones=[2],
    ),
    # At the milestone the child restarts without a step; its first step is the next.
    "SequentialLR-milestone-1": lambda S, o: S.SequentialLR(
        o,
        [S.ConstantLR(o, factor=0.5, total_iters=2), S.ExponentialLR(o, gamma=0.9)],
        milestones=[1],
    ),
    # A chain that runs first steps its children while the SequentialLR is built.
    "SequentialLR-of-chain": lambda S, o: S.SequentialLR(
        o,
        [
            S.ChainedScheduler(
                [S.ConstantLR(o, factor=0.5, total_iters=2), S.ExponentialLR(o, 0.9)]
            ),
            S.ExponentialLR(o, gamma=0.8),
        ],
        milestones=[3],
    ),
    "ChainedScheduler-of-sequential": lambda S, o: S.ChainedScheduler(
        [
            S.SequentialLR(
                o,
                [S.ConstantLR(o, factor=0.5, total_iters=2), S.ExponentialLR(o, 0.9)],
                milestones=[2],
            ),
            S.ExponentialLR(o, gamma=0.95),
        ]
    ),
    # Two schedulers built back to back: each checks its own first step.
    "two-on-one-optimizer": lambda S, o: _Both(
        S.StepLR(o, step_size=2), S.ExponentialLR(o, gamma=0.9)
    ),
}


class _Both:
    """Two schedulers on one optimizer, stepped one after the other."""

    def __init__(self, *scheds: Any) -> None:
        self.scheds = scheds

    def step(self) -> None:
        for sched in self.scheds:
            sched.step()


def _step(sched: Any) -> None:
    if type(sched).__name__ == "ReduceLROnPlateau":
        sched.step(1.0)
    else:
        sched.step()


def _scheduler_first(opt: Any, sched: Any) -> None:
    for _ in range(_STEPS):
        _step(sched)
        opt.step()


def _optimizer_first(opt: Any, sched: Any) -> None:
    for _ in range(_STEPS):
        opt.step()
        _step(sched)


def _scheduler_only(opt: Any, sched: Any) -> None:
    del opt  # the loop that never steps the optimizer
    for _ in range(_STEPS):
        _step(sched)


_LOOPS: dict[str, Loop] = {
    "scheduler-first": _scheduler_first,
    "optimizer-first": _optimizer_first,
    "scheduler-only": _scheduler_only,
}


def _order_warnings(
    lib: ModuleType, build: Build, loop: Loop, *, step_before_build: bool = False
) -> list[str]:
    """The first sentence of each step-order warning a run gives, in order.

    The sentences after the first name the reference framework.
    """
    opt = lib.optim.SGD([lib.nn.Parameter(lib.zeros(2))], lr=LR, momentum=0.9)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if step_before_build:
            opt.step()
        loop(opt, build(lib.optim.lr_scheduler, opt))
    return [
        str(w.message).split(". ")[0]
        for w in caught
        if w.category is UserWarning and "lr_scheduler.step()" in str(w.message)
    ]


def _assert_as_the_reference(ref: Any, build: Build, loop: Loop, **kw: bool) -> None:
    want = _order_warnings(ref, build, loop, **kw)
    got = _order_warnings(lucid, build, loop, **kw)
    assert got == want


@pytest.mark.parity
@pytest.mark.parametrize("loop", list(_LOOPS))
@pytest.mark.parametrize("name", list(_SCHEDULES))
def test_step_order_warnings_are_the_references(name: str, loop: str, ref: Any) -> None:
    _assert_as_the_reference(ref, _SCHEDULES[name], _LOOPS[loop])


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_SCHEDULES))
def test_an_optimizer_step_before_the_scheduler_does_not_count(name: str, ref: Any) -> None:
    _assert_as_the_reference(ref, _SCHEDULES[name], _scheduler_first, step_before_build=True)


def _second_scheduler_after_a_step(S: Any, o: Any) -> Any:
    first = S.StepLR(o, step_size=2)
    o.step()
    first.step()
    return S.ExponentialLR(o, gamma=0.9)


def _replaced_after(S: Any, o: Any) -> Any:
    sched = S.StepLR(o, step_size=2)
    o.step = lambda *args, **kwargs: None
    return sched


def _replaced_before(S: Any, o: Any) -> Any:
    o.step = o.step  # the instance's own step, put on the instance
    return S.StepLR(o, step_size=2)


def _late_group(S: Any, o: Any) -> Any:
    sched = S.LambdaLR(o, lambda e: 0.5**e)
    first = o.param_groups[0]["params"][0]
    o.add_param_group({"params": [type(first)(first.detach().clone())]})
    return sched


def _refused_once(opt: Any, sched: Any) -> None:
    del opt  # the refusal comes before any optimizer step
    with pytest.raises(ValueError):
        sched.step()


@pytest.mark.parity
@pytest.mark.parametrize(
    ("build", "loop"),
    [
        pytest.param(
            _second_scheduler_after_a_step,
            _scheduler_first,
            id="second-built-after-a-step",
        ),
        pytest.param(_replaced_after, _optimizer_first, id="step-replaced-after"),
        pytest.param(_replaced_before, _optimizer_first, id="step-replaced-before"),
        pytest.param(_late_group, _refused_once, id="refused-first-step"),
    ],
)
def test_the_references_edge_cases(build: Build, loop: Loop, ref: Any) -> None:
    _assert_as_the_reference(ref, build, loop)


@pytest.mark.parity
def test_a_resumed_scheduler_does_not_check_again(ref: Any) -> None:
    """The checkpoint carries the count of steps taken."""

    def resumed(S: Any, o: Any) -> Any:
        sched = S.StepLR(o, step_size=2)
        o.step()
        sched.step()
        state = sched.state_dict()
        other = type(o)(o.param_groups[0]["params"], lr=LR)
        sched = S.StepLR(other, step_size=2)
        sched.load_state_dict(state)
        return sched

    _assert_as_the_reference(ref, resumed, _scheduler_only)


def test_the_warning_points_at_the_scheduler_step_call() -> None:
    opt = optim.SGD([nn.Parameter(lucid.zeros(2))], lr=LR)
    sched = optim.StepLR(opt, step_size=2)
    with pytest.warns(UserWarning, match="before `optimizer.step\\(\\)`") as caught:
        sched.step()
    assert [w.filename for w in caught] == [__file__]


def test_the_step_counter_forwards_the_call() -> None:
    """``optimizer.step`` still takes the step, closure and result included."""
    p = nn.Parameter(lucid.ones(2))
    opt = optim.SGD([p], lr=0.5)
    optim.StepLR(opt, step_size=2)

    def closure() -> lucid.Tensor:
        opt.zero_grad()
        loss = (p * p).sum()
        loss.backward()
        return loss

    loss = opt.step(closure)
    assert loss is not None and float(loss.item()) == 2.0
    assert p.tolist() == [0.0, 0.0]


@pytest.mark.parametrize(
    "clone",
    [copy.copy, copy.deepcopy, lambda o: pickle.loads(pickle.dumps(o))],
    ids=["copy", "deepcopy", "pickle"],
)
def test_a_watched_optimizer_copies_with_its_counter(
    clone: Callable[[Any], Any],
) -> None:
    """The copy's counter steps and reports the copy, not the one it came from."""
    opt = optim.SGD([nn.Parameter(lucid.zeros(2))], lr=LR)
    optim.StepLR(opt, step_size=2)
    twin = clone(opt)
    sched = optim.StepLR(twin, step_size=2)
    twin.step()
    assert twin._opt_called and not opt._opt_called
    assert twin._engines_built and not opt._engines_built
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sched.step()


def test_the_counter_does_not_keep_the_optimizer_alive() -> None:
    """Without the cyclic collector, a dropped optimizer is freed at once.

    Its parameters, gradients and state — on Metal, too — go with it.
    """
    gc.disable()
    try:
        opt = optim.Adam([nn.Parameter(lucid.zeros(2))], lr=LR)
        sched = optim.StepLR(opt, step_size=2)
        opt.step()
        sched.step()
        alive = weakref.ref(opt)
        del opt, sched
        assert alive() is None
    finally:
        gc.enable()


@pytest.mark.parity
def test_the_counter_reads_as_the_step_it_counts(ref: Any) -> None:
    """Name, docstring and signature are the step's, as on the reference's counter."""

    def looks(lib: ModuleType) -> tuple[str, list[str], bool]:
        opt = lib.optim.SGD([lib.nn.Parameter(lib.zeros(2))], lr=LR)
        lib.optim.lr_scheduler.StepLR(opt, step_size=2)
        step = opt.step
        same_doc = step.__doc__ == type(opt).step.__doc__
        return step.__name__, list(inspect.signature(step).parameters), same_doc

    assert looks(lucid) == looks(ref)


# A compiled step drives the optimizer's state without ``optimizer.step``.
# Each one reports itself, so a compiled loop that steps in the right order
# is silent, whichever was built first and whichever handle the scheduler is
# on, and one that steps the scheduler first still warns.


def _compiled_loop(entry: str, on: str, built: str, scheduler_first: bool) -> list[str]:
    from lucid.compile import compile_optimizer, fused_step, make_step

    lucid.manual_seed(0)
    model = nn.Linear(4, 2).to("metal")
    x = lucid.randn(8, 4, device="metal")
    t = lucid.randn(8, 2, device="metal")
    opt = optim.SGD(model.parameters(), lr=LR)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sched = optim.StepLR(opt, step_size=2) if built == "before" else None
        handle: Any = opt
        if entry == "fused_step":
            fused = fused_step(model, F.mse_loss, opt)
            handle = fused.optimizer

            def train() -> None:
                fused(x, t)

        elif entry == "make_step":
            traced = make_step(model, F.mse_loss)

            def train() -> None:
                opt.zero_grad()
                traced(x, t).backward()
                opt.step()

        else:
            copt = handle = compile_optimizer(opt)
            stepper = copt if entry == "compile_optimizer" else opt

            def train() -> None:
                copt.zero_grad()
                F.mse_loss(model(x), t).backward()
                stepper.step()

        if sched is None:
            sched = optim.StepLR(handle if on == "handle" else opt, step_size=2)
        for _ in range(3):
            if scheduler_first:
                sched.step()
            train()
            if not scheduler_first:
                sched.step()
    return [
        str(w.message).split(". ")[0] for w in caught if "lr_scheduler.step()" in str(w.message)
    ]


_CASES = [
    pytest.param(entry, "optimizer", built, id=f"{entry}-built-{built}")
    for entry in ["compile_optimizer", "compile_optimizer-via-opt", "fused_step"] + ["make_step"]
    for built in ["before", "after"]
] + [
    # The documented handles of a compiled optimizer: the wrapper itself, and
    # ``fused.optimizer``.  A scheduler can only be built on one after it.
    pytest.param("compile_optimizer", "handle", "after", id="on-the-compiled-wrapper"),
    pytest.param("fused_step", "handle", "after", id="on-fused.optimizer"),
]


@pytest.mark.gpu
@pytest.mark.parametrize(("entry", "on", "built"), _CASES)
def test_a_compiled_loop_in_order_is_silent(entry: str, on: str, built: str) -> None:
    assert _compiled_loop(entry, on, built, scheduler_first=False) == []


@pytest.mark.gpu
@pytest.mark.parametrize(("entry", "on", "built"), _CASES)
def test_a_compiled_loop_stepping_the_scheduler_first_warns(
    entry: str, on: str, built: str
) -> None:
    assert _compiled_loop(entry, on, built, scheduler_first=True) == [
        "Detected call of `lr_scheduler.step()` before `optimizer.step()`"
    ]
