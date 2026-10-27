"""A compiled optimizer step is an eager step, over a whole training run.

One-step parity (``test_optimizer.py``) cannot see what goes wrong only
across steps or around the step: a learning-rate schedule, a second
parameter group, a frozen backbone, a checkpoint resumed half way, a
parameter the loss never reaches.  Each test here drives the eager
optimizer and its compiled counterpart — ``compile_optimizer`` and
``fused_step`` — through the same run and compares the parameters after
every step.

Tolerance: ``compile_optimizer`` uses the eager gradients and the eager
kernel's arithmetic order, so it is exact; ``fused_step`` derives its
gradients inside MPSGraph, which rounds a few ulp differently — 1e-6 over
a handful of steps.
"""

import subprocess
import sys
import textwrap
from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import compile_optimizer, fused_step
from lucid.optim.lr_scheduler import CosineAnnealingLR, StepLR

from lucid._C import engine as _C_engine
from lucid.test.unit.compile._helpers import COMPILE_DEVICE

TOL = 1e-6
STEPS = 5

OptFactory = Callable[[object], optim.Optimizer]


class _Net(nn.Module):
    """Two linear layers — two groups' worth of parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(6, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc2(self.fc1(x).tanh())


def _net() -> _Net:
    lucid.manual_seed(0)
    return _Net().to(COMPILE_DEVICE)


def _data() -> tuple[lucid.Tensor, lucid.Tensor]:
    lucid.manual_seed(1)
    return (lucid.randn(10, 6).to(COMPILE_DEVICE), lucid.randn(10, 3).to(COMPILE_DEVICE))


def _snapshot(model: nn.Module) -> list[lucid.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def _drift(a: list[lucid.Tensor], b: list[lucid.Tensor]) -> float:
    return max(float((x - y).abs().max().item()) for x, y in zip(a, b))


class _Runner:
    """Drives one model + optimizer as eager, compiled or fused steps."""

    def __init__(self, mode: str, model: nn.Module, opt: optim.Optimizer) -> None:
        self.mode = mode
        self.model = model
        self.opt = opt
        self.stepper: object = opt
        if mode == "compiled":
            self.stepper = compile_optimizer(opt)
        elif mode == "fused":
            self.stepper = fused_step(model, F.mse_loss, opt)

    def step(self, x: lucid.Tensor, t: lucid.Tensor) -> None:
        if self.mode == "fused":
            self.stepper(x, t)  # type: ignore[operator]
            return
        self.stepper.zero_grad()  # type: ignore[attr-defined]
        F.mse_loss(self.model(x), t).backward()
        self.stepper.step()  # type: ignore[attr-defined]

    def executables(self) -> int:
        return len(self.stepper._plans)  # type: ignore[attr-defined]


def _run(mode: str, make_opt: OptFactory, model: nn.Module | None = None) -> _Runner:
    """A ``mode`` runner over ``model`` (a fresh seeded net by default)."""
    model = _net() if model is None else model
    return _Runner(mode, model, make_opt(model))


def _both(mode: str, make_opt: OptFactory) -> tuple[_Runner, _Runner]:
    """An eager runner and a ``mode`` runner over identical models."""
    return _run("eager", make_opt), _run(mode, make_opt)


def _copy_of(model: nn.Module) -> nn.Module:
    """A fresh net holding ``model``'s current parameters."""
    out = _net()
    with lucid.no_grad():
        for p, q in zip(out.parameters(), model.parameters()):
            p.copy_(q.detach())
    return out


COMPILED_MODES = ["compiled", "fused"]


# ── CHA-175: hyper-parameters are read every step ──────────────────


SCHEDULED = [
    pytest.param(lambda m: optim.SGD(m.parameters(), lr=0.3, momentum=0.9), id="SGD"),
    pytest.param(lambda m: optim.Adam(m.parameters(), lr=0.05), id="Adam"),
    pytest.param(lambda m: optim.AdamW(m.parameters(), lr=0.05, weight_decay=0.1), id="AdamW"),
]
SCHEDULERS = [
    pytest.param(lambda o: StepLR(o, step_size=1, gamma=0.3), id="StepLR"),
    pytest.param(lambda o: CosineAnnealingLR(o, T_max=4), id="Cosine"),
]


@pytest.mark.parametrize("mode", COMPILED_MODES)
@pytest.mark.parametrize("make_sched", SCHEDULERS)
@pytest.mark.parametrize("make_opt", SCHEDULED)
def test_an_lr_schedule_reaches_the_compiled_step(
    make_opt: OptFactory, make_sched: Callable[[object], object], mode: str
) -> None:
    """The schedule's rate is the one each compiled step uses — no retrace."""
    x, t = _data()
    eager, comp = _both(mode, make_opt)
    eager_sched = make_sched(eager.opt)
    comp_sched = make_sched(comp.opt)
    for k in range(STEPS):
        eager.step(x, t)
        comp.step(x, t)
        eager_sched.step()  # type: ignore[attr-defined]
        comp_sched.step()  # type: ignore[attr-defined]
        drift = _drift(_snapshot(eager.model), _snapshot(comp.model))
        assert drift < TOL, f"step {k + 1}: drift {drift:.3e}"
    # Changing the rate changes a number fed to the executable, not the trace.
    assert comp.executables() == 1


def test_a_scheduler_built_on_the_compiled_optimizer_works() -> None:
    """``StepLR(compile_optimizer(opt))`` — the wrapper takes the scheduler's calls."""
    x, t = _data()
    eager, comp = _both("compiled", lambda m: optim.SGD(m.parameters(), lr=0.3))
    eager_sched = StepLR(eager.opt, step_size=1, gamma=0.5)
    comp_sched = StepLR(comp.stepper, step_size=1, gamma=0.5)  # type: ignore[arg-type]
    for _ in range(3):
        eager.step(x, t)
        comp.step(x, t)
        eager_sched.step()
        comp_sched.step()
    assert _drift(_snapshot(eager.model), _snapshot(comp.model)) < TOL


@pytest.mark.parametrize("mode", COMPILED_MODES)
def test_other_hyper_parameters_are_read_every_step(mode: str) -> None:
    """``beta1``, ``eps`` and ``weight_decay`` edited mid-run take effect.

    Both sides take the same hand edit of the group: the eager step reads
    it too (CHA-210).
    """
    x, t = _data()

    def make(m: nn.Module) -> optim.Optimizer:
        return optim.Adam(m.parameters(), lr=0.05, weight_decay=0.01)

    eager, comp = _both(mode, make)
    new = {"beta1": 0.5, "eps": 1e-3, "weight_decay": 0.2, "lr": 0.02}
    for k in range(STEPS):
        if k == 2:
            eager.opt.param_groups[0].update(new)
            comp.opt.param_groups[0].update(new)
        eager.step(x, t)
        comp.step(x, t)
        drift = _drift(_snapshot(eager.model), _snapshot(comp.model))
        assert drift < TOL, f"step {k + 1}: drift {drift:.3e}"
    assert comp.executables() == 1


# Edits that change what SGD's momentum buffer starts from: the first
# momentum step sets ``buf = g``, undamped, whether it is the run's first
# step or momentum was switched on part-way (CHA-210).
SGD_MOMENTUM_EDITS = [
    pytest.param({"momentum": 0.9, "dampening": 0.5}, {}, id="damped-from-the-start"),
    pytest.param({}, {"momentum": 0.9, "dampening": 0.5}, id="switched-on"),
    pytest.param({"momentum": 0.9, "dampening": 0.3}, {"momentum": 0.0}, id="switched-off"),
    pytest.param(
        {"momentum": 0.9, "nesterov": True}, {"lr": 0.05, "weight_decay": 0.1}, id="nesterov"
    ),
]


@pytest.mark.parametrize("mode", COMPILED_MODES)
@pytest.mark.parametrize("built, edit", SGD_MOMENTUM_EDITS)
def test_sgd_momentum_starts_as_in_eager(
    built: dict[str, object], edit: dict[str, object], mode: str
) -> None:
    """The compiled SGD starts and keeps its momentum buffer as eager does."""
    x, t = _data()

    def make(m: nn.Module) -> optim.Optimizer:
        return optim.SGD(m.parameters(), lr=0.2, **built)  # type: ignore[arg-type]

    eager, comp = _both(mode, make)
    for k in range(STEPS):
        if k == 2:
            eager.opt.param_groups[0].update(edit)
            comp.opt.param_groups[0].update(edit)
        eager.step(x, t)
        comp.step(x, t)
        drift = _drift(_snapshot(eager.model), _snapshot(comp.model))
        assert drift < TOL, f"step {k + 1}: drift {drift:.3e}"
    _assert_same_state(comp.opt.state_dict(), eager.opt.state_dict())


# A feature switched off and on again resumes the buffer it kept; a
# checkpoint taken while it is off carries that buffer, eager and compiled.
KEPT_BUFFERS = [
    pytest.param(
        lambda m: optim.SGD(m.parameters(), lr=0.2, momentum=0.9),
        "momentum",
        0.0,
        0.9,
        "momentum_buffer",
        id="SGD-momentum",
    ),
    pytest.param(
        lambda m: optim.Adam(m.parameters(), lr=0.05, amsgrad=True),
        "amsgrad",
        False,
        True,
        "max_exp_avg_sq",
        id="Adam-amsgrad",
    ),
    pytest.param(
        lambda m: optim.RMSprop(m.parameters(), lr=0.01, momentum=0.9),
        "momentum",
        0.0,
        0.5,
        "momentum_buffer",
        id="RMSprop-momentum",
    ),
    pytest.param(
        lambda m: optim.RMSprop(m.parameters(), lr=0.01, centered=True),
        "centered",
        False,
        True,
        "grad_avg",
        id="RMSprop-centered",
    ),
]


@pytest.mark.parametrize("make_opt, key, off, on, kept", KEPT_BUFFERS)
def test_a_kept_buffer_is_checkpointed_and_resumed_as_in_eager(
    make_opt: OptFactory, key: str, off: object, on: object, kept: str
) -> None:
    x, t = _data()
    eager, comp = _both("compiled", make_opt)
    for k in range(STEPS + 1):
        if k == 2:
            eager.opt.param_groups[0][key] = off
            comp.opt.param_groups[0][key] = off
        if k == 4:
            want = eager.opt.state_dict()
            assert all(kept in entry for entry in want["state"].values())  # type: ignore[attr-defined]
            _assert_same_state(comp.opt.state_dict(), want)
            eager.opt.param_groups[0][key] = on
            comp.opt.param_groups[0][key] = on
        eager.step(x, t)
        comp.step(x, t)
        drift = _drift(_snapshot(eager.model), _snapshot(comp.model))
        assert drift < TOL, f"step {k + 1}: drift {drift:.3e}"


@pytest.mark.parametrize("mode", COMPILED_MODES)
def test_a_rejected_edit_stops_the_compiled_step(mode: str) -> None:
    """A value the eager step refuses, the compiled one refuses too."""
    x, t = _data()
    runner = _run(mode, lambda m: optim.SGD(m.parameters(), lr=0.2, momentum=0.9))
    runner.step(x, t)
    before = _snapshot(runner.model)
    runner.opt.param_groups[0]["lr"] = -1.0
    for _ in range(2):
        with pytest.raises(_C_engine.InvalidArgument, match="SGD: lr must be >= 0"):
            runner.step(x, t)
    assert _drift(before, _snapshot(runner.model)) == 0.0
    with pytest.raises(_C_engine.InvalidArgument):
        runner.opt.state_dict()
    runner.opt.param_groups[0]["lr"] = 0.2
    runner.step(x, t)
    assert _drift(before, _snapshot(runner.model)) > 0.0


@pytest.mark.parametrize("mode", COMPILED_MODES)
def test_an_edit_is_checked_once_not_every_step(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    x, t = _data()
    runner = _run(mode, lambda m: optim.Adam(m.parameters(), lr=0.05))
    opt = runner.opt
    checked: list[int] = []
    hand_over = opt._apply_hyperparams

    def counting(idx: int, values: dict[str, object]) -> None:
        checked.append(idx)
        hand_over(idx, values)

    monkeypatch.setattr(opt, "_apply_hyperparams", counting)
    for _ in range(3):
        runner.step(x, t)
    assert checked == []
    opt.param_groups[0]["eps"] = 1e-4
    for _ in range(3):
        runner.step(x, t)
    assert checked == [0]


def _no_decay_groups(model: nn.Module, cls: type, **kw: object) -> optim.Optimizer:
    """The transformer recipe: decay the weights, not the biases."""
    weights = [p for n, p in model.named_parameters() if n.endswith("weight")]
    biases = [p for n, p in model.named_parameters() if n.endswith("bias")]
    return cls(  # type: ignore[no-any-return]
        [{"params": weights}, {"params": biases, "weight_decay": 0.0, "lr": 0.01}], **kw
    )


MULTI_GROUP = [
    pytest.param(lambda m: _no_decay_groups(m, optim.AdamW, lr=0.05, weight_decay=0.1), id="AdamW"),
    pytest.param(lambda m: _no_decay_groups(m, optim.Adam, lr=0.05, weight_decay=0.1), id="Adam"),
    pytest.param(
        lambda m: _no_decay_groups(m, optim.Adamax, lr=0.05, weight_decay=0.1), id="Adamax"
    ),
    pytest.param(lambda m: _no_decay_groups(m, optim.NAdam, lr=0.05, weight_decay=0.1), id="NAdam"),
    pytest.param(lambda m: _no_decay_groups(m, optim.RAdam, lr=0.05, weight_decay=0.1), id="RAdam"),
    pytest.param(
        lambda m: optim.Rprop(
            [
                {"params": list(m.fc1.parameters())},
                {"params": list(m.fc2.parameters()), "lr": 0.002},
            ],
            lr=0.01,
            etas=(0.4, 1.3),
        ),
        id="Rprop",
    ),
    pytest.param(
        lambda m: _no_decay_groups(m, optim.SGD, lr=0.2, momentum=0.9, weight_decay=0.05), id="SGD"
    ),
]


@pytest.mark.parametrize("mode", COMPILED_MODES)
@pytest.mark.parametrize("make_opt", MULTI_GROUP)
def test_parameter_groups_compile_together(make_opt: OptFactory, mode: str) -> None:
    """Every group keeps its own hyper-parameters, in one executable."""
    x, t = _data()
    eager, comp = _both(mode, make_opt)
    for k in range(STEPS):
        eager.step(x, t)
        comp.step(x, t)
        drift = _drift(_snapshot(eager.model), _snapshot(comp.model))
        assert drift < TOL, f"step {k + 1}: drift {drift:.3e}"
    assert comp.executables() == 1


def test_group_counts_do_not_share_an_executable() -> None:
    """One group and two groups over the same shapes are different executables.

    The per-group hyper-parameters are runtime inputs now, so a cache hit
    between the two would feed one group's numbers to the other's layout.
    """
    x, t = _data()

    def one_group(m: nn.Module) -> optim.Optimizer:
        return optim.SGD(m.parameters(), lr=0.1)

    def two_groups(m: nn.Module) -> optim.Optimizer:
        return optim.SGD(
            [
                {"params": list(m.fc1.parameters())},  # type: ignore[union-attr]
                {"params": list(m.fc2.parameters()), "lr": 0.01},  # type: ignore[union-attr]
            ],
            lr=0.1,
        )

    eager_one, one = _both("compiled", one_group)
    eager_two, two = _both("compiled", two_groups)
    for _ in range(3):
        for r in (one, two, eager_one, eager_two):
            r.step(x, t)
    assert _drift(_snapshot(one.model), _snapshot(eager_one.model)) < TOL
    assert _drift(_snapshot(two.model), _snapshot(eager_two.model)) < TOL
    key_one = next(iter(one.stepper._plans))  # type: ignore[attr-defined]
    key_two = next(iter(two.stepper._plans))  # type: ignore[attr-defined]
    assert key_one != key_two


# ── CHA-176: the state is checkpointed in the eager format ─────────


STATEFUL = [
    pytest.param(lambda m: optim.SGD(m.parameters(), lr=0.2, momentum=0.9), id="SGD"),
    pytest.param(lambda m: optim.Adam(m.parameters(), lr=0.05), id="Adam"),
    pytest.param(lambda m: optim.Adam(m.parameters(), lr=0.05, amsgrad=True), id="Adam_amsgrad"),
    pytest.param(lambda m: optim.AdamW(m.parameters(), lr=0.05), id="AdamW"),
    pytest.param(
        lambda m: optim.RMSprop(m.parameters(), lr=0.01, momentum=0.9, centered=True), id="RMSprop"
    ),
    pytest.param(lambda m: optim.Adagrad(m.parameters(), lr=0.1), id="Adagrad"),
    pytest.param(lambda m: optim.Adadelta(m.parameters()), id="Adadelta"),
    pytest.param(lambda m: optim.Adamax(m.parameters(), lr=0.05), id="Adamax"),
    pytest.param(lambda m: optim.NAdam(m.parameters(), lr=0.05), id="NAdam"),
    pytest.param(lambda m: optim.RAdam(m.parameters(), lr=0.05), id="RAdam"),
    pytest.param(lambda m: optim.ASGD(m.parameters(), lr=0.05, lambd=0.1, t0=1.0), id="ASGD"),
    pytest.param(lambda m: optim.Rprop(m.parameters(), lr=0.01), id="Rprop"),
]


def _assert_same_state(got: dict[str, object], want: dict[str, object]) -> None:
    gs = got["state"]
    ws = want["state"]
    assert sorted(gs) == sorted(ws)  # type: ignore[call-overload]
    for idx in ws:  # type: ignore[attr-defined]
        g, w = gs[idx], ws[idx]  # type: ignore[index]
        assert sorted(g) == sorted(w), idx
        for key in w:
            a, b = g[key], w[key]
            assert a.dtype == b.dtype and a.shape == b.shape, (idx, key)
            # Relative past 1: an RMSprop momentum buffer grows to ~20.
            assert abs(a - b).max() <= TOL * max(1.0, abs(b).max()), (idx, key)


@pytest.mark.parametrize("make_opt", STATEFUL)
def test_state_dict_matches_the_eager_one(make_opt: OptFactory) -> None:
    """Same keys, dtypes, shapes and values as the eager optimizer's state."""
    x, t = _data()
    eager, comp = _both("compiled", make_opt)
    for _ in range(3):
        eager.step(x, t)
        comp.step(x, t)
    _assert_same_state(comp.stepper.state_dict(), eager.opt.state_dict())  # type: ignore[attr-defined]


@pytest.mark.parametrize("make_opt", STATEFUL)
def test_fused_step_state_reaches_the_optimizer_state_dict(make_opt: OptFactory) -> None:
    """After ``fused_step`` the optimizer the user holds checkpoints its state."""
    x, t = _data()
    eager, comp = _both("fused", make_opt)
    for _ in range(3):
        eager.step(x, t)
        comp.step(x, t)
    got = comp.opt.state_dict()
    assert got["state"], "fused_step left the optimizer's state_dict empty"
    _assert_same_state(got, eager.opt.state_dict())


@pytest.mark.parametrize("source", ["eager", "compiled"])
@pytest.mark.parametrize("mode", COMPILED_MODES)
@pytest.mark.parametrize("make_opt", STATEFUL)
def test_resuming_from_a_checkpoint_matches_an_uninterrupted_run(
    make_opt: OptFactory, mode: str, source: str
) -> None:
    """Save after 3 steps, resume in a fresh compiled run, compare after 3 more.

    ``source="eager"`` loads an eager checkpoint into the compiled
    optimizer; ``"compiled"`` round-trips the compiled one.
    """
    x, t = _data()
    reference, first = _both(mode if source == "compiled" else "eager", make_opt)
    for _ in range(3):
        reference.step(x, t)
        first.step(x, t)
    checkpoint = first.opt.state_dict()

    resumed = _run(mode, make_opt, _copy_of(first.model))
    resumed.opt.load_state_dict(checkpoint)
    for k in range(3):
        reference.step(x, t)
        resumed.step(x, t)
        drift = _drift(_snapshot(reference.model), _snapshot(resumed.model))
        assert drift < TOL, f"step {k + 4}: drift {drift:.3e}"


def test_a_compiled_checkpoint_loads_into_an_eager_optimizer() -> None:
    """The format is the eager one: the eager optimizer can carry on from it."""
    x, t = _data()

    def make(m: nn.Module) -> optim.Optimizer:
        return optim.NAdam(m.parameters(), lr=0.05)

    reference, comp = _both("compiled", make)
    for _ in range(3):
        reference.step(x, t)
        comp.step(x, t)
    eager = _run("eager", make, _copy_of(comp.model))
    eager.opt.load_state_dict(comp.opt.state_dict())
    for _ in range(3):
        reference.step(x, t)
        eager.step(x, t)
    assert _drift(_snapshot(reference.model), _snapshot(eager.model)) < TOL


def test_stepping_the_eager_handle_after_compiling_steps_the_wrapper() -> None:
    """One owner of the state: ``opt.step()`` / ``state_dict`` / ``load_state_dict``
    on the eager handle all reach the compiled state once it is compiled.

    Before, eager steps after ``compile_optimizer`` advanced the eager
    engines while ``opt.state_dict()`` reported the wrapper's empty state —
    a checkpoint that silently lost everything.
    """
    x, t = _data()

    def make(m: nn.Module) -> optim.Optimizer:
        return optim.Adam(m.parameters(), lr=0.05, weight_decay=0.01)

    reference = _run("eager", make)
    model = _net()
    opt = make(model)
    compile_optimizer(opt)
    handle = _Runner("eager", model, opt)  # steps through ``opt`` itself
    for _ in range(3):
        reference.step(x, t)
        handle.step(x, t)
    assert _drift(_snapshot(reference.model), _snapshot(model)) < TOL
    checkpoint = opt.state_dict()
    _assert_same_state(checkpoint, reference.opt.state_dict())

    # Save → load round trip on a fresh eager run equals the uninterrupted one.
    resumed = _run("eager", make, _copy_of(model))
    resumed.opt.load_state_dict(checkpoint)
    for _ in range(2):
        reference.step(x, t)
        resumed.step(x, t)
    assert _drift(_snapshot(reference.model), _snapshot(resumed.model)) < TOL

    # And ``opt.load_state_dict`` reaches the state ``opt.step()`` uses: load
    # the reference's step-5 checkpoint, take one step, land on its step 6.
    with lucid.no_grad():
        for p, q in zip(model.parameters(), reference.model.parameters()):
            p.copy_(q.detach())
    opt.load_state_dict(reference.opt.state_dict())
    reference.step(x, t)
    handle.step(x, t)
    assert _drift(_snapshot(reference.model), _snapshot(model)) < TOL


@pytest.mark.parametrize("mode", COMPILED_MODES)
@pytest.mark.parametrize("make_opt", STATEFUL)
def test_compiling_after_eager_steps_carries_their_state_on(
    make_opt: OptFactory, mode: str
) -> None:
    """Three eager steps, then compile and take three more: the same as six
    eager steps — the warm-up state is adopted, not restarted."""
    x, t = _data()
    reference, warm = _both("eager", make_opt)
    for _ in range(3):
        reference.step(x, t)
        warm.step(x, t)
    compiled = _Runner(mode, warm.model, warm.opt)
    for k in range(3):
        reference.step(x, t)
        compiled.step(x, t)
        drift = _drift(_snapshot(reference.model), _snapshot(warm.model))
        assert drift < TOL, f"step {k + 4}: drift {drift:.3e}"
    _assert_same_state(warm.opt.state_dict(), reference.opt.state_dict())


EVERY_OPTIMIZER = [
    pytest.param(lambda m: optim.SGD(m.parameters(), lr=0.1), id="SGD"),
    pytest.param(lambda m: optim.Adam(m.parameters()), id="Adam"),
    pytest.param(lambda m: optim.AdamW(m.parameters()), id="AdamW"),
    pytest.param(lambda m: optim.RMSprop(m.parameters()), id="RMSprop"),
    pytest.param(lambda m: optim.Adagrad(m.parameters()), id="Adagrad"),
    pytest.param(lambda m: optim.Adadelta(m.parameters()), id="Adadelta"),
    pytest.param(lambda m: optim.Adamax(m.parameters()), id="Adamax"),
    pytest.param(lambda m: optim.NAdam(m.parameters()), id="NAdam"),
    pytest.param(lambda m: optim.RAdam(m.parameters()), id="RAdam"),
    pytest.param(lambda m: optim.ASGD(m.parameters()), id="ASGD"),
    pytest.param(lambda m: optim.Rprop(m.parameters()), id="Rprop"),
    pytest.param(lambda m: optim.SparseAdam(m.parameters()), id="SparseAdam"),
    pytest.param(lambda m: optim.LBFGS(m.parameters()), id="LBFGS"),
]


@pytest.mark.parametrize("make_opt", EVERY_OPTIMIZER)
def test_every_state_entry_point_has_one_owner(make_opt: OptFactory) -> None:
    """Class-wide guard for the one-owner rule, over every compilable class.

    Each method of the eager optimizer that reads or advances its state is
    the wrapper's after compiling — or, for LBFGS (a different compiled
    algorithm), still the eager optimizer's own.  A new state entry point
    on ``Optimizer`` that is not in the handed-over list shows up here.
    """
    from lucid.compile._optim.compiler import _STATE_ENTRY_POINTS
    from lucid.optim.optimizer import Optimizer

    # Every public method of an optimizer either touches its state — and is
    # handed over — or is known not to.  A new public method has to be put
    # on one side or the other.
    public = {
        name
        for name in dir(Optimizer)
        if not name.startswith("_") and callable(getattr(Optimizer, name))
    }
    stateless = {"zero_grad", "add_param_group"}
    assert public - stateless == set(_STATE_ENTRY_POINTS)

    opt = make_opt(_net())
    wrapper = compile_optimizer(opt)
    assert compile_optimizer(opt) is wrapper
    for name in _STATE_ENTRY_POINTS:
        bound = getattr(opt, name)
        if isinstance(opt, optim.LBFGS):
            assert getattr(bound, "__self__", None) is opt, name
        else:
            assert getattr(bound, "__self__", None) is wrapper, name
            assert getattr(bound, "__func__", None) is getattr(type(wrapper), name), name


def test_lbfgs_keeps_its_eager_state_and_step() -> None:
    """The compiled LBFGS is a different algorithm, so ``opt`` stays its own:
    its ``step`` (closure line search), state and checkpoint are untouched."""
    lucid.manual_seed(1)
    x = lucid.randn(10, 6).to(COMPILE_DEVICE)
    t = lucid.randn(10, 3).to(COMPILE_DEVICE)

    def run(compile_first: bool) -> tuple[list[lucid.Tensor], dict[str, object]]:
        model = _net()
        opt = optim.LBFGS(model.parameters(), lr=0.5, max_iter=4)
        if compile_first:
            compile_optimizer(opt)
            for name in ("step", "state_dict", "load_state_dict"):
                assert name not in opt.__dict__, name

        def closure() -> lucid.Tensor:
            opt.zero_grad()
            loss = F.mse_loss(model(x), t)
            loss.backward()
            return loss

        for _ in range(2):
            opt.step(closure)
        return _snapshot(model), opt.state_dict()

    plain_params, plain_state = run(False)
    params, state = run(True)
    assert _drift(plain_params, params) == 0.0
    assert sorted(state["state"][0]) == sorted(plain_state["state"][0])  # type: ignore[index]


# ── CHA-177: parameters without a gradient do not move ─────────────


FREEZABLE = [
    pytest.param(
        lambda m: optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=0.1), id="SGD"
    ),
    pytest.param(lambda m: optim.AdamW(m.parameters(), lr=0.05, weight_decay=0.1), id="AdamW"),
    pytest.param(lambda m: optim.Adam(m.parameters(), lr=0.05, weight_decay=0.1), id="Adam"),
    pytest.param(lambda m: optim.NAdam(m.parameters(), lr=0.05), id="NAdam"),
]


@pytest.mark.parametrize("mode", COMPILED_MODES)
@pytest.mark.parametrize("make_opt", FREEZABLE)
def test_a_frozen_parameter_stays_put(make_opt: OptFactory, mode: str) -> None:
    """``requires_grad=False`` with every parameter in the optimizer: no update."""
    x, t = _data()
    eager, comp = _both(mode, make_opt)
    for r in (eager, comp):
        r.model.fc1.weight.requires_grad_(False)
    frozen = comp.model.fc1.weight.detach().clone()
    for _ in range(3):
        eager.step(x, t)
        comp.step(x, t)
    assert float((comp.model.fc1.weight - frozen).abs().max().item()) == 0.0
    assert _drift(_snapshot(eager.model), _snapshot(comp.model)) < TOL
    # Never stepped, so no state for it (flat index 0), as in eager.
    assert 0 not in comp.opt.state_dict()["state"]


@pytest.mark.parametrize("mode", COMPILED_MODES)
@pytest.mark.parametrize("make_opt", FREEZABLE)
def test_freezing_and_unfreezing_mid_run_follows_eager(make_opt: OptFactory, mode: str) -> None:
    """Train, freeze the first layer, unfreeze it: each parameter keeps its own
    step count, so bias corrections after the unfreeze match eager too."""
    x, t = _data()
    eager, comp = _both(mode, make_opt)
    plan = [True, True, False, False, True, True]
    for k, trainable in enumerate(plan):
        for r in (eager, comp):
            r.model.fc1.weight.requires_grad_(trainable)
            r.model.fc1.bias.requires_grad_(trainable)
        eager.step(x, t)
        comp.step(x, t)
        drift = _drift(_snapshot(eager.model), _snapshot(comp.model))
        assert drift < TOL, f"step {k + 1}: drift {drift:.3e}"
    _assert_same_state(comp.opt.state_dict(), eager.opt.state_dict())
    # One executable per participant set, plus one once the unfrozen layer's
    # step count trails the rest (its bias corrections are its own).
    assert comp.executables() <= 3


def test_an_active_set_that_keeps_changing_uses_one_masked_executable() -> None:
    """Gradients present for a different subset every step (a routed model):
    after a few exact executables one masked executable serves every set."""
    lucid.manual_seed(0)
    experts_e = [nn.Linear(4, 4).to(COMPILE_DEVICE) for _ in range(4)]
    lucid.manual_seed(0)
    experts_c = [nn.Linear(4, 4).to(COMPILE_DEVICE) for _ in range(4)]
    params_e = [p for e in experts_e for p in e.parameters()]
    params_c = [p for e in experts_c for p in e.parameters()]
    eager = optim.Adam(params_e, lr=0.05, weight_decay=0.01)
    comp = compile_optimizer(optim.Adam(params_c, lr=0.05, weight_decay=0.01))
    lucid.manual_seed(2)
    x = lucid.randn(5, 4).to(COMPILE_DEVICE)
    routes = [(0,), (1, 2), (3,), (0, 3), (1,), (2, 3), (0, 1, 2), (2,), (0, 1)]
    for route in routes:
        for experts, opt in ((experts_e, eager), (experts_c, comp)):
            opt.zero_grad()
            sum(experts[i](x).square().mean() for i in route).backward()  # type: ignore[union-attr]
            opt.step()
        drift = _drift(params_e, params_c)  # type: ignore[arg-type]
        assert drift < TOL, f"route {route}: drift {drift:.3e}"
    _assert_same_state(comp.state_dict(), eager.state_dict())
    assert len(comp._plans) <= comp._EXACT_PLAN_LIMIT + 1


# ── CHA-174: a parameter the loss never reaches ────────────────────


def _run_child(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter — an abort there is an exit code here."""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=300
    )


def test_fused_step_skips_a_parameter_the_loss_never_reaches() -> None:
    """Used to abort the process inside MPSGraph's autodiff ("Not a predecessor
    of primaryTensor").  Now the unused head is left alone, as in eager."""
    proc = _run_child("""
        import lucid, lucid.nn as nn, lucid.nn.functional as F, lucid.optim as optim
        from lucid.compile import fused_step

        class N(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 4)
                self.aux = nn.Linear(8, 4)

            def forward(self, x):
                return self.fc(x)

        def make():
            lucid.manual_seed(0)
            m = N().to("metal")
            return m, optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=0.1)

        lucid.manual_seed(1)
        x = lucid.randn(4, 8).to("metal")
        t = lucid.randn(4, 4).to("metal")
        em, eo = make()
        cm, co = make()
        aux0 = cm.aux.weight.detach().clone()
        step = fused_step(cm, F.mse_loss, co)
        for _ in range(3):
            eo.zero_grad()
            F.mse_loss(em(x), t).backward()
            eo.step()
            step(x, t)
        assert float((cm.aux.weight - aux0).abs().max().item()) == 0.0
        drift = max(
            float((a - b).abs().max().item())
            for a, b in zip(em.parameters(), cm.parameters())
        )
        assert drift < 1e-6, drift
        assert sorted(co.state_dict()["state"]) == sorted(eo.state_dict()["state"])
        print("ok")
        """)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().endswith("ok")


def test_the_builder_refuses_a_gradient_off_the_loss_path() -> None:
    """Asked directly for such a gradient, the builder fails instead of aborting."""
    proc = _run_child("""
        import lucid
        from lucid._C import engine as _C_engine
        from lucid._dispatch import _unwrap
        from lucid.compile import _tracing

        w = lucid.randn(4, 4).to("metal")
        unused = lucid.randn(4, 4).to("metal")
        ghost_w = lucid.zeros(4, 4, device="metal")
        ghost_u = lucid.zeros(4, 4, device="metal")
        x = lucid.randn(2, 4).to("metal")
        with lucid.no_grad():
            with _tracing() as tr:
                loss = (x @ w).square().mean()
                new_w = w - ghost_w
                new_u = unused - ghost_u
        ids = lambda *ts: [int(tr.lookup_id(_unwrap(t))) for t in ts]
        try:
            exe = _C_engine.compile.compile_generic_fused_step(
                tr.graph,
                dict(tr.external_feeds),
                ids(loss)[0],
                ids(w, unused),
                ids(ghost_w, ghost_u),
                ids(new_w, new_u),
            )
            print("compiled" if exe is not None else "returned None")
        except RuntimeError as e:
            print("refused:", e)
        """)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = proc.stdout.strip()
    # Refused by the loss-path check, naming the parameter — not some
    # other failure, and not a silent ``None``.
    assert out.startswith("refused:"), out
    assert "does not reach the loss" in out, out
    assert "param id" in out, out


# ── LCD-213: a parameter seen only through a non-differentiable op ──

#: Ways a parameter can reach the loss with no gradient flowing back: eager
#: leaves its ``grad`` None and the optimizer skips it.  ``mixed`` also
#: uses it directly, so it does get a gradient — the stop is per edge.
GATES: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "detach": lambda g: g.detach(),
    "comparison": lambda g: (g > 0).to(lucid.float32),
    "detach_then_comparison": lambda g: (g.detach() > 0).to(lucid.float32),
    "integer_cast": lambda g: g.to(lucid.int32).to(lucid.float32),
    "mixed": lambda g: g.detach() + g,
}


class _Gated(nn.Module):
    def __init__(self, gate: str) -> None:
        super().__init__()
        self.fc = nn.Linear(6, 3)
        self.gate = nn.Parameter(lucid.full((1,), 1.5))
        self._through = GATES[gate]

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc(x) * self._through(self.gate)


GATED_OPTS = [
    pytest.param(
        lambda m: optim.SGD(m.parameters(), lr=0.1, momentum=0.9, weight_decay=0.1), id="SGD"
    ),
    pytest.param(lambda m: optim.AdamW(m.parameters(), lr=0.05, weight_decay=0.1), id="AdamW"),
]


@pytest.mark.parametrize("make_opt", GATED_OPTS)
@pytest.mark.parametrize("gate", sorted(GATES))
def test_fused_step_skips_a_parameter_with_no_gradient(gate: str, make_opt: OptFactory) -> None:
    """Used to raise "manual_vjp: no gradient produced for param id …": the
    reach test followed data, through ``detach`` and comparisons alike.
    Weight decay and momentum move a parameter fed a zero gradient, so only
    skipping it, as eager does, keeps the two runs together."""

    def make() -> tuple[_Gated, optim.Optimizer]:
        lucid.manual_seed(0)
        m = _Gated(gate).to(COMPILE_DEVICE)
        return m, make_opt(m)

    x, t = _data()
    em, eo = make()
    cm, co = make()
    step = fused_step(cm, F.mse_loss, co)
    for _ in range(STEPS):
        eo.zero_grad()
        F.mse_loss(em(x), t).backward()
        eo.step()
        step(x, t)
    assert (em.gate.grad is None) == (gate != "mixed")
    assert _drift(_snapshot(em), _snapshot(cm)) < TOL
    assert sorted(co.state_dict()["state"]) == sorted(eo.state_dict()["state"])


@pytest.mark.parametrize("gate", sorted(GATES))
def test_make_step_leaves_a_gradient_none_where_eager_does(gate: str) -> None:
    """The backward walker under ``make_step``: an integer cast used to pass a
    truncated gradient back to the float it came from."""
    lucid.manual_seed(0)
    model = _Gated(gate).to(COMPILE_DEVICE)
    x, t = _data()
    F.mse_loss(model(x), t).backward()
    eager = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    model.zero_grad()
    for p in model.parameters():
        p.grad = None
    lucid.compile.make_step(model, F.mse_loss)(x, t).backward()
    for want, p in zip(eager, model.parameters(), strict=True):
        assert (p.grad is None) == (want is None)
        if want is not None:
            assert float((p.grad - want).abs().max().item()) < TOL
