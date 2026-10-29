"""A compiled optimizer honours every option its eager twin keeps in ``param_groups``.

``compile_optimizer`` / ``fused_step`` once ignored ``maximize`` (every
optimizer) and ``decoupled_weight_decay`` (Adam, RAdam): the compiled step
trained the wrong way and nothing said so (LCD-687).  A spot fix per option
leaves the next option added to an eager optimizer unhandled, so this file
holds the account of all of them.

``OPTIONS`` names, for each optimizer, every ``param_groups`` key and a
non-default value for it.  The tests then

* require the table to match the keys the optimizer really keeps — a new
  option fails ``test_every_option_is_in_the_table`` until it is added and
  so handled;
* run the eager optimizer and its compiled twin with each option flipped
  and compare parameters after every step and the checkpointed state at the
  end (``test_a_flipped_option_follows_eager``), checking first that the
  flip changes the run at all, so a value that does nothing cannot pass for
  a handled option.

What the compiled path cannot do is listed in ``ALGORITHM_DIFFERS`` and
nowhere else.
"""

import functools
import inspect
from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.amp import GradScaler
from lucid.compile import fused_step
from lucid.optim.optimizer import Optimizer
from lucid.test.unit.compile.test_compiled_optimizer_eager_equivalence import (
    TOL,
    _both,
    _data,
    _drift,
    _net,
    _run,
    _snapshot,
)

# ``foreach`` / ``fused`` pick an eager kernel strategy, ``capturable`` /
# ``differentiable`` are accepted for signature parity and change no result
# (``differentiable=True`` is refused by the eager optimizer itself).
IGNORED_KEYS = frozenset({"params", "foreach", "fused", "capturable", "differentiable"})

# The rule RAdam leaves for step ~6 at these betas, so the run reaches it.
STEPS = 7

# Optimizer -> constructor kwargs of the unflipped run, then option -> the
# value it is flipped to.  An option the base already sets is flipped to
# a different value.
BASE: dict[str, dict[str, object]] = {
    "SGD": dict(lr=0.1, momentum=0.9, weight_decay=0.01),
    "Adam": dict(lr=0.05, weight_decay=0.1),
    "AdamW": dict(lr=0.05, weight_decay=0.1),
    "RMSprop": dict(lr=0.01, weight_decay=0.01, momentum=0.5),
    "Adagrad": dict(lr=0.1, weight_decay=0.01),
    "Adadelta": dict(weight_decay=0.01),
    "Adamax": dict(lr=0.05, weight_decay=0.01),
    "NAdam": dict(lr=0.05, weight_decay=0.01),
    "SparseAdam": dict(lr=0.05),
    "Rprop": dict(lr=0.01),
    "ASGD": dict(lr=0.05, lambd=0.2, weight_decay=0.01),
    "RAdam": dict(lr=0.05, betas=(0.9, 0.9), weight_decay=0.01),
}

OPTIONS: dict[str, dict[str, object]] = {
    "SGD": dict(
        lr=0.05, momentum=0.5, dampening=0.3, weight_decay=0.2, nesterov=True, maximize=True
    ),
    "Adam": dict(
        lr=0.02,
        betas=(0.8, 0.95),
        eps=1e-2,
        weight_decay=0.3,
        amsgrad=True,
        maximize=True,
        decoupled_weight_decay=True,
    ),
    "AdamW": dict(
        lr=0.02, betas=(0.8, 0.95), eps=1e-2, weight_decay=0.3, amsgrad=True, maximize=True
    ),
    "RMSprop": dict(
        lr=0.02, alpha=0.9, eps=1e-2, weight_decay=0.2, momentum=0.9, centered=True, maximize=True
    ),
    "Adagrad": dict(
        lr=0.05,
        lr_decay=0.2,
        weight_decay=0.2,
        initial_accumulator_value=0.5,
        eps=1e-2,
        maximize=True,
    ),
    "Adadelta": dict(lr=0.5, rho=0.8, eps=1e-3, weight_decay=0.2, maximize=True),
    "Adamax": dict(lr=0.02, betas=(0.8, 0.95), eps=1e-2, weight_decay=0.2, maximize=True),
    "NAdam": dict(
        lr=0.02,
        betas=(0.8, 0.95),
        eps=1e-2,
        weight_decay=0.2,
        momentum_decay=0.04,
        decoupled_weight_decay=True,
        maximize=True,
    ),
    "SparseAdam": dict(lr=0.02, betas=(0.8, 0.95), eps=1e-2, maximize=True),
    "Rprop": dict(lr=0.05, etas=(0.4, 1.3), step_sizes=(0.015, 0.02), maximize=True),
    "ASGD": dict(lr=0.02, lambd=0.5, alpha=0.5, t0=1.0, weight_decay=0.2, maximize=True),
    "RAdam": dict(
        lr=0.02,
        betas=(0.8, 0.95),
        eps=1e-2,
        weight_decay=0.2,
        decoupled_weight_decay=True,
        maximize=True,
    ),
}

# A key an optimizer keeps that is no choice of the caller's: AdamW's engine
# is the decoupled one, so the group's copy of the flag changes nothing in
# eager — and nothing in the compiled twin.
FIXED: dict[str, dict[str, object]] = {"AdamW": {"decoupled_weight_decay": False}}

# What the compiled path does not do.  LBFGS's compiled step is a different
# algorithm — per-element Barzilai-Borwein, no closure, no line search —
# documented on ``_CompiledLBFGS`` and in ``compile_optimizer``; only ``lr``
# reaches it.
ALGORITHM_DIFFERS: dict[str, frozenset[str]] = {
    "LBFGS": frozenset(
        {
            "lr",
            "max_iter",
            "max_eval",
            "tolerance_grad",
            "tolerance_change",
            "history_size",
            "line_search_fn",
        }
    )
}

COMPILED_OPTIMIZERS = sorted(OPTIONS)


def _optimizer_class(name: str) -> type[Optimizer]:
    cls: type[Optimizer] = getattr(optim, name)
    return cls


def _group_options(name: str) -> set[str]:
    """The options ``name`` keeps in ``param_groups``, as the caller sees them.

    The split halves of a pair (``beta1`` / ``beta2`` of ``betas``) are the
    optimizer's own bookkeeping, not options of their own.
    """
    cls = _optimizer_class(name)
    opt = cls([nn.Parameter(lucid.zeros(2, 2))])
    halves = {k for pair in cls._PAIR_KEYS.values() for k in pair}
    return set(opt.param_groups[0]) - IGNORED_KEYS - halves


def _maker(name: str, **overrides: object) -> Callable[[nn.Module], Optimizer]:
    kwargs = {**BASE[name], **overrides}
    cls = _optimizer_class(name)
    return lambda m: cls(m.parameters(), **kwargs)  # type: ignore[arg-type]


# ── The account ─────────────────────────────────────────────────────


def test_every_optimizer_class_is_in_the_table() -> None:
    """A new optimizer class has to be put in the table (or the exceptions)."""
    classes = {
        n
        for n, c in vars(optim).items()
        if inspect.isclass(c) and issubclass(c, Optimizer) and c is not Optimizer
    }
    assert classes == set(OPTIONS) | set(ALGORITHM_DIFFERS)


@pytest.mark.parametrize("name", sorted(OPTIONS) + sorted(ALGORITHM_DIFFERS))
def test_every_option_is_in_the_table(name: str) -> None:
    """The table names exactly the options the optimizer keeps.

    A key missing from it is an option the compiled twin may not read; a key
    only the table has is one the optimizer no longer keeps.
    """
    kept = _group_options(name)
    accounted = set(OPTIONS.get(name, ())) | set(FIXED.get(name, ()))
    accounted |= ALGORITHM_DIFFERS.get(name, frozenset())
    assert kept - accounted == set(), f"{name}: options with no entry: {sorted(kept - accounted)}"
    assert (
        accounted - kept == set()
    ), f"{name}: entries for options it lacks: {sorted(accounted - kept)}"


def test_a_table_key_is_a_constructor_argument() -> None:
    for name, options in OPTIONS.items():
        params = inspect.signature(_optimizer_class(name)).parameters
        assert set(options) <= set(params), name


# ── The trajectories ────────────────────────────────────────────────


def _state_gap(a: dict[str, object], b: dict[str, object]) -> float:
    """Largest gap between two checkpoints' state, relative past 1 (``inf`` if they differ in shape).

    An RMSprop momentum buffer grows to ~20, so the gap is read against its
    size; ``step`` is a 0-d array or, for SparseAdam, a plain int.
    """
    sa = a["state"]
    sb = b["state"]
    if sorted(sa) != sorted(sb):  # type: ignore[call-overload]
        return float("inf")
    gap = 0.0
    for idx in sa:  # type: ignore[attr-defined]
        ea, eb = sa[idx], sb[idx]  # type: ignore[index]
        if sorted(ea) != sorted(eb):
            return float("inf")
        for key, va in ea.items():
            vb = eb[key]
            if isinstance(va, int) or isinstance(vb, int):
                gap = max(gap, abs(float(va) - float(vb)))
                continue
            if va.shape != vb.shape or va.dtype != vb.dtype:
                return float("inf")
            gap = max(gap, float(abs(va - vb).max()) / max(1.0, float(abs(vb).max())))
    return gap


def _train(
    mode: str, name: str, **overrides: object
) -> tuple[list[lucid.Tensor], dict[str, object]]:
    runner = _run(mode, _maker(name, **overrides))
    x, t = _data()
    for _ in range(STEPS):
        runner.step(x, t)
    return _snapshot(runner.model), runner.opt.state_dict()


@functools.cache
def _unflipped(name: str) -> tuple[list[lucid.Tensor], dict[str, object]]:
    return _train("eager", name)


def _flip_cases() -> list[pytest.ParameterSet]:
    return [
        pytest.param(name, key, id=f"{name}-{key}")
        for name in COMPILED_OPTIMIZERS
        for key in OPTIONS[name]
    ]


@pytest.mark.parametrize("name, key", _flip_cases())
def test_a_flipped_option_follows_eager(name: str, key: str) -> None:
    """With ``key`` flipped the compiled run is the eager run, step by step."""
    flipped = {key: OPTIONS[name][key]}
    x, t = _data()
    eager, comp = _both("compiled", _maker(name, **flipped))
    for k in range(STEPS):
        eager.step(x, t)
        comp.step(x, t)
        drift = _drift(_snapshot(eager.model), _snapshot(comp.model))
        assert drift < TOL, f"{name} {key}: step {k + 1}: drift {drift:.3e}"
    gap = _state_gap(comp.opt.state_dict(), eager.opt.state_dict())
    assert gap <= TOL, f"{name} {key}: state gap {gap:.3e}"

    # The flip must show in the run, or agreeing proves nothing.
    base_params, base_state = _unflipped(name)
    moved = max(
        _drift(base_params, _snapshot(eager.model)), _state_gap(base_state, eager.opt.state_dict())
    )
    assert moved > 1e-5, f"{name}: flipping {key} changes nothing in {STEPS} steps"


@pytest.mark.parametrize("name", sorted(FIXED))
def test_a_fixed_key_changes_nothing_in_either_path(name: str) -> None:
    """AdamW decays decoupled whatever its group's copy of the flag says."""
    x, t = _data()
    eager, comp = _both("compiled", _maker(name))
    for runner in (eager, comp):
        runner.opt.param_groups[0].update(FIXED[name])
    for _ in range(STEPS):
        eager.step(x, t)
        comp.step(x, t)
    assert _drift(_snapshot(eager.model), _snapshot(comp.model)) < TOL
    assert _state_gap(comp.opt.state_dict(), eager.opt.state_dict()) <= TOL


# ── maximize, the option every optimizer has ────────────────────────


@pytest.mark.parametrize("name", COMPILED_OPTIMIZERS)
def test_maximize_flipped_mid_run_follows_eager(name: str) -> None:
    """The eager step reads ``maximize`` from the group every step; so does the compiled one."""
    x, t = _data()
    eager, comp = _both("compiled", _maker(name))
    for k in range(STEPS):
        if k in (2, 5):
            for runner in (eager, comp):
                group = runner.opt.param_groups[0]
                group["maximize"] = not group["maximize"]
        eager.step(x, t)
        comp.step(x, t)
        drift = _drift(_snapshot(eager.model), _snapshot(comp.model))
        assert drift < TOL, f"{name}: step {k + 1}: drift {drift:.3e}"
    gap = _state_gap(comp.opt.state_dict(), eager.opt.state_dict())
    assert gap <= TOL, f"{name}: state gap {gap:.3e}"
    # Both senses (and the way back) are executables of their own, kept.
    assert comp.executables() == 2


@pytest.mark.parametrize("name", ["SGD", "Adam", "RAdam"])
def test_maximize_is_a_choice_of_each_group(name: str) -> None:
    """One group maximizes, the other minimizes — a group's flag is its own."""

    def make(m: nn.Module) -> Optimizer:
        cls = _optimizer_class(name)
        return cls(  # type: ignore[call-arg]
            [
                {"params": list(m.fc1.parameters()), "maximize": True},  # type: ignore[union-attr]
                {"params": list(m.fc2.parameters())},  # type: ignore[union-attr]
            ],
            lr=0.05,
        )

    x, t = _data()
    eager, comp = _both("compiled", make)
    for _ in range(STEPS):
        eager.step(x, t)
        comp.step(x, t)
    assert _drift(_snapshot(eager.model), _snapshot(comp.model)) < TOL
    plain = _run("eager", lambda m: _optimizer_class(name)(m.parameters(), lr=0.05))
    for _ in range(STEPS):
        plain.step(x, t)
    assert _drift(_snapshot(plain.model), _snapshot(eager.model)) > 1e-3


# ── fused_step reads the options too ────────────────────────────────

# ``fused_step`` derives its gradients in MPSGraph, a few ulp from the eager
# ones and by a different amount on each chip (RMSprop, 4 steps: 1.1e-6 on an
# M1 Pro).  A direction the step ignores is off by 1e-2 and more.
FUSED_TOL = 5e-6
# The options the fused step shares with ``compile_optimizer`` (both build
# their update with the same ``_emit``), checked once per class that has them.
FUSED_OPTIONS = [
    pytest.param(name, key, id=f"{name}-{key}")
    for name in COMPILED_OPTIMIZERS
    for key in ("maximize", "decoupled_weight_decay")
    if key in OPTIONS[name]
]


@pytest.mark.parametrize("name, key", FUSED_OPTIONS)
def test_fused_step_reads_the_option_too(name: str, key: str) -> None:
    flipped = {key: OPTIONS[name][key]}
    x, t = _data()
    eager, fused = _both("fused", _maker(name, **flipped))
    for k in range(4):
        eager.step(x, t)
        fused.step(x, t)
        drift = _drift(_snapshot(eager.model), _snapshot(fused.model))
        assert drift < FUSED_TOL, f"{name} {key}: step {k + 1}: drift {drift:.3e}"


def test_fused_step_with_a_grad_scaler_maximizes_too() -> None:
    """The scaled path unscales the gradient, and only then does the update negate it."""
    x, t = _data()
    steps = {}
    for mode in ("eager", "fused"):
        model = _net()
        opt = optim.SGD(model.parameters(), lr=0.1, maximize=True)
        scaler = GradScaler(init_scale=2.0**8, growth_interval=1000)
        if mode == "fused":
            step = fused_step(model, F.mse_loss, opt, grad_scaler=scaler)
        else:

            def step(x: lucid.Tensor, t: lucid.Tensor) -> None:
                opt.zero_grad()
                scaler.scale(F.mse_loss(model(x), t)).backward()  # type: ignore[union-attr]
                scaler.step(opt)
                scaler.update()

        for _ in range(3):
            step(x, t)
        steps[mode] = _snapshot(model)
    assert _drift(steps["eager"], steps["fused"]) < FUSED_TOL
