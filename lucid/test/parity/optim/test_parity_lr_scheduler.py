"""Parity tests: LR schedulers vs reference framework.

For every scheduler, we:
  1. Build identical optimizers (same initial LR, same dummy param)
  2. Build schedulers with identical hyperparameters
  3. Step both N times, recording the LR after each step
  4. Assert the LR sequences match to < 1e-6 relative tolerance

Covers:
  StepLR, ExponentialLR, MultiStepLR, CosineAnnealingLR,
  CosineAnnealingWarmRestarts, OneCycleLR, LinearLR, ConstantLR,
  PolynomialLR, LambdaLR, MultiplicativeLR, CyclicLR,
  ReduceLROnPlateau, SequentialLR, ChainedScheduler
"""

from typing import Any

import pytest

import lucid.nn as nn
import lucid.optim as optim

# ── helpers ───────────────────────────────────────────────────────────────────


def _dummy_lucid_optimizer(lr: float) -> optim.SGD:
    p = nn.Linear(2, 2)
    return optim.SGD(p.parameters(), lr=lr)


def _dummy_ref_optimizer(lr: float, ref: Any) -> Any:
    p = ref.nn.Linear(2, 2)
    return ref.optim.SGD(p.parameters(), lr=lr)


def _collect_lrs(scheduler: Any, steps: int) -> list[float]:
    # The rate in force before the first step() is the first optimizer
    # step's.  It was left out, and five schedulers ran that step at the
    # base rate — OneCycleLR at 25x its own — with every test here green.
    lrs: list[float] = [scheduler.get_last_lr()[0]]
    for _ in range(steps):
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    return lrs


def _assert_lr_sequence(lucid_lrs: list[float], ref_lrs: list[float], atol: float = 1e-6) -> None:
    assert len(lucid_lrs) == len(ref_lrs)
    for i, (l, r) in enumerate(zip(lucid_lrs, ref_lrs)):
        assert abs(l - r) < atol, f"step {i}: lucid={l:.8f} ref={r:.8f} diff={abs(l - r):.2e}"


# ── StepLR ────────────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestStepLRParity:
    def test_basic(self, ref: Any) -> None:
        lr0 = 0.1
        lucid_sched = optim.lr_scheduler.StepLR(_dummy_lucid_optimizer(lr0), step_size=3, gamma=0.5)
        ref_sched = ref.optim.lr_scheduler.StepLR(
            _dummy_ref_optimizer(lr0, ref), step_size=3, gamma=0.5
        )

        _assert_lr_sequence(_collect_lrs(lucid_sched, 12), _collect_lrs(ref_sched, 12))

    def test_gamma_0_1(self, ref: Any) -> None:
        lr0 = 1.0
        lucid_sched = optim.lr_scheduler.StepLR(_dummy_lucid_optimizer(lr0), step_size=5)
        ref_sched = ref.optim.lr_scheduler.StepLR(_dummy_ref_optimizer(lr0, ref), step_size=5)

        _assert_lr_sequence(_collect_lrs(lucid_sched, 15), _collect_lrs(ref_sched, 15))


# ── ExponentialLR ─────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestExponentialLRParity:
    def test_basic(self, ref: Any) -> None:
        lr0 = 0.1
        lucid_sched = optim.lr_scheduler.ExponentialLR(_dummy_lucid_optimizer(lr0), gamma=0.9)
        ref_sched = ref.optim.lr_scheduler.ExponentialLR(_dummy_ref_optimizer(lr0, ref), gamma=0.9)

        _assert_lr_sequence(_collect_lrs(lucid_sched, 20), _collect_lrs(ref_sched, 20))


# ── MultiStepLR ───────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestMultiStepLRParity:
    def test_basic(self, ref: Any) -> None:
        lr0 = 0.1
        milestones = [3, 6, 9]
        lucid_sched = optim.lr_scheduler.MultiStepLR(
            _dummy_lucid_optimizer(lr0), milestones=milestones, gamma=0.5
        )
        ref_sched = ref.optim.lr_scheduler.MultiStepLR(
            _dummy_ref_optimizer(lr0, ref), milestones=milestones, gamma=0.5
        )

        _assert_lr_sequence(_collect_lrs(lucid_sched, 12), _collect_lrs(ref_sched, 12))


# ── CosineAnnealingLR ─────────────────────────────────────────────────────────


@pytest.mark.parity
class TestCosineAnnealingLRParity:
    def test_basic(self, ref: Any) -> None:
        lr0 = 0.1
        T_max = 10
        lucid_sched = optim.lr_scheduler.CosineAnnealingLR(
            _dummy_lucid_optimizer(lr0), T_max=T_max, eta_min=1e-4
        )
        ref_sched = ref.optim.lr_scheduler.CosineAnnealingLR(
            _dummy_ref_optimizer(lr0, ref), T_max=T_max, eta_min=1e-4
        )

        _assert_lr_sequence(_collect_lrs(lucid_sched, 20), _collect_lrs(ref_sched, 20))

    def test_no_eta_min(self, ref: Any) -> None:
        lr0 = 0.5
        lucid_sched = optim.lr_scheduler.CosineAnnealingLR(_dummy_lucid_optimizer(lr0), T_max=8)
        ref_sched = ref.optim.lr_scheduler.CosineAnnealingLR(
            _dummy_ref_optimizer(lr0, ref), T_max=8
        )

        _assert_lr_sequence(_collect_lrs(lucid_sched, 16), _collect_lrs(ref_sched, 16))


# ── CosineAnnealingWarmRestarts ────────────────────────────────────────────────


@pytest.mark.parity
class TestCosineAnnealingWarmRestartsParity:
    def test_t_mult_1(self, ref: Any) -> None:
        lr0 = 0.1
        lucid_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            _dummy_lucid_optimizer(lr0), T_0=5, T_mult=1, eta_min=0.0
        )
        ref_sched = ref.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            _dummy_ref_optimizer(lr0, ref), T_0=5, T_mult=1, eta_min=0.0
        )

        _assert_lr_sequence(_collect_lrs(lucid_sched, 15), _collect_lrs(ref_sched, 15), atol=1e-6)

    def test_t_mult_2(self, ref: Any) -> None:
        lr0 = 0.2
        lucid_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            _dummy_lucid_optimizer(lr0), T_0=4, T_mult=2, eta_min=1e-4
        )
        ref_sched = ref.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            _dummy_ref_optimizer(lr0, ref), T_0=4, T_mult=2, eta_min=1e-4
        )

        _assert_lr_sequence(_collect_lrs(lucid_sched, 16), _collect_lrs(ref_sched, 16), atol=1e-6)


# ── LinearLR ──────────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestLinearLRParity:
    def test_basic(self, ref: Any) -> None:
        lr0 = 0.1
        lucid_sched = optim.lr_scheduler.LinearLR(
            _dummy_lucid_optimizer(lr0), start_factor=0.2, end_factor=1.0, total_iters=5
        )
        ref_sched = ref.optim.lr_scheduler.LinearLR(
            _dummy_ref_optimizer(lr0, ref),
            start_factor=0.2,
            end_factor=1.0,
            total_iters=5,
        )

        _assert_lr_sequence(_collect_lrs(lucid_sched, 8), _collect_lrs(ref_sched, 8))


# ── ConstantLR ────────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestConstantLRParity:
    def test_basic(self, ref: Any) -> None:
        lr0 = 0.1
        lucid_sched = optim.lr_scheduler.ConstantLR(
            _dummy_lucid_optimizer(lr0), factor=0.5, total_iters=4
        )
        ref_sched = ref.optim.lr_scheduler.ConstantLR(
            _dummy_ref_optimizer(lr0, ref), factor=0.5, total_iters=4
        )

        _assert_lr_sequence(_collect_lrs(lucid_sched, 7), _collect_lrs(ref_sched, 7))


# ── PolynomialLR ──────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestPolynomialLRParity:
    def test_linear_decay(self, ref: Any) -> None:
        lr0 = 1.0
        lucid_sched = optim.lr_scheduler.PolynomialLR(
            _dummy_lucid_optimizer(lr0), total_iters=10, power=1.0
        )
        ref_sched = ref.optim.lr_scheduler.PolynomialLR(
            _dummy_ref_optimizer(lr0, ref), total_iters=10, power=1.0
        )

        _assert_lr_sequence(_collect_lrs(lucid_sched, 12), _collect_lrs(ref_sched, 12))

    def test_quadratic_decay(self, ref: Any) -> None:
        lr0 = 0.5
        lucid_sched = optim.lr_scheduler.PolynomialLR(
            _dummy_lucid_optimizer(lr0), total_iters=8, power=2.0
        )
        ref_sched = ref.optim.lr_scheduler.PolynomialLR(
            _dummy_ref_optimizer(lr0, ref), total_iters=8, power=2.0
        )

        _assert_lr_sequence(_collect_lrs(lucid_sched, 10), _collect_lrs(ref_sched, 10))


# ── LambdaLR ──────────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestLambdaLRParity:
    def test_linear_warmup(self, ref: Any) -> None:
        lr0 = 0.1
        fn = lambda epoch: epoch / 10 if epoch < 10 else 1.0  # noqa: E731

        lucid_sched = optim.lr_scheduler.LambdaLR(_dummy_lucid_optimizer(lr0), lr_lambda=fn)
        ref_sched = ref.optim.lr_scheduler.LambdaLR(_dummy_ref_optimizer(lr0, ref), lr_lambda=fn)

        _assert_lr_sequence(_collect_lrs(lucid_sched, 15), _collect_lrs(ref_sched, 15))


# ── MultiplicativeLR ──────────────────────────────────────────────────────────


@pytest.mark.parity
class TestMultiplicativeLRParity:
    def test_constant_factor(self, ref: Any) -> None:
        lr0 = 1.0
        fn = lambda epoch: 0.95  # noqa: E731

        lucid_sched = optim.lr_scheduler.MultiplicativeLR(_dummy_lucid_optimizer(lr0), lr_lambda=fn)
        ref_sched = ref.optim.lr_scheduler.MultiplicativeLR(
            _dummy_ref_optimizer(lr0, ref), lr_lambda=fn
        )

        _assert_lr_sequence(_collect_lrs(lucid_sched, 10), _collect_lrs(ref_sched, 10))


# ── CyclicLR ──────────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestCyclicLRParity:
    def test_triangular(self, ref: Any) -> None:
        lr0 = 0.01
        lucid_sched = optim.lr_scheduler.CyclicLR(
            _dummy_lucid_optimizer(lr0),
            base_lr=0.01,
            max_lr=0.1,
            step_size_up=5,
            mode="triangular",
        )
        ref_sched = ref.optim.lr_scheduler.CyclicLR(
            _dummy_ref_optimizer(lr0, ref),
            base_lr=0.01,
            max_lr=0.1,
            step_size_up=5,
            mode="triangular",
        )

        _assert_lr_sequence(_collect_lrs(lucid_sched, 20), _collect_lrs(ref_sched, 20))

    def test_triangular2(self, ref: Any) -> None:
        lr0 = 0.01
        lucid_sched = optim.lr_scheduler.CyclicLR(
            _dummy_lucid_optimizer(lr0),
            base_lr=0.01,
            max_lr=0.1,
            step_size_up=4,
            mode="triangular2",
        )
        ref_sched = ref.optim.lr_scheduler.CyclicLR(
            _dummy_ref_optimizer(lr0, ref),
            base_lr=0.01,
            max_lr=0.1,
            step_size_up=4,
            mode="triangular2",
        )

        _assert_lr_sequence(_collect_lrs(lucid_sched, 16), _collect_lrs(ref_sched, 16))


# ── OneCycleLR ────────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestOneCycleLRParity:
    def test_cosine_anneal(self, ref: Any) -> None:
        lr0 = 0.1
        total_steps = 20
        max_lr = 0.5
        lucid_sched = optim.lr_scheduler.OneCycleLR(
            _dummy_lucid_optimizer(lr0),
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )
        ref_sched = ref.optim.lr_scheduler.OneCycleLR(
            _dummy_ref_optimizer(lr0, ref),
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )

        _assert_lr_sequence(
            _collect_lrs(lucid_sched, total_steps),
            _collect_lrs(ref_sched, total_steps),
            atol=1e-6,
        )

    def test_linear_anneal(self, ref: Any) -> None:
        lr0 = 0.1
        total_steps = 15
        lucid_sched = optim.lr_scheduler.OneCycleLR(
            _dummy_lucid_optimizer(lr0),
            max_lr=0.3,
            total_steps=total_steps,
            pct_start=0.4,
            anneal_strategy="linear",
        )
        ref_sched = ref.optim.lr_scheduler.OneCycleLR(
            _dummy_ref_optimizer(lr0, ref),
            max_lr=0.3,
            total_steps=total_steps,
            pct_start=0.4,
            anneal_strategy="linear",
        )

        _assert_lr_sequence(
            _collect_lrs(lucid_sched, total_steps),
            _collect_lrs(ref_sched, total_steps),
            atol=1e-6,
        )


# ── ReduceLROnPlateau ─────────────────────────────────────────────────────────


@pytest.mark.parity
class TestReduceLROnPlateauParity:
    def test_min_mode(self, ref: Any) -> None:
        lr0 = 0.1
        lucid_opt = _dummy_lucid_optimizer(lr0)
        ref_opt = _dummy_ref_optimizer(lr0, ref)

        lucid_sched = optim.lr_scheduler.ReduceLROnPlateau(
            lucid_opt, mode="min", factor=0.5, patience=3, threshold=1e-4
        )
        ref_sched = ref.optim.lr_scheduler.ReduceLROnPlateau(
            ref_opt, mode="min", factor=0.5, patience=3, threshold=1e-4
        )

        metrics = [1.0, 0.9, 0.8, 0.8, 0.8, 0.8, 0.5, 0.5, 0.5, 0.5]
        lucid_lrs, ref_lrs = [], []
        for m in metrics:
            lucid_sched.step(m)
            ref_sched.step(m)
            lucid_lrs.append(float(lucid_opt.param_groups[0]["lr"]))
            ref_lrs.append(float(ref_opt.param_groups[0]["lr"]))

        _assert_lr_sequence(lucid_lrs, ref_lrs)


# ── SequentialLR ──────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestSequentialLRParity:
    def test_two_schedulers(self, ref: Any) -> None:
        lr0 = 0.1
        lucid_opt = _dummy_lucid_optimizer(lr0)
        ref_opt = _dummy_ref_optimizer(lr0, ref)

        lucid_sched = optim.lr_scheduler.SequentialLR(
            lucid_opt,
            schedulers=[
                optim.lr_scheduler.ConstantLR(lucid_opt, factor=0.5, total_iters=3),
                optim.lr_scheduler.ExponentialLR(lucid_opt, gamma=0.9),
            ],
            milestones=[3],
        )
        ref_sched = ref.optim.lr_scheduler.SequentialLR(
            ref_opt,
            schedulers=[
                ref.optim.lr_scheduler.ConstantLR(ref_opt, factor=0.5, total_iters=3),
                ref.optim.lr_scheduler.ExponentialLR(ref_opt, gamma=0.9),
            ],
            milestones=[3],
        )

        _assert_lr_sequence(_collect_lrs(lucid_sched, 8), _collect_lrs(ref_sched, 8))


# ── ChainedScheduler ──────────────────────────────────────────────────────────


@pytest.mark.parity
class TestChainedSchedulerParity:
    def test_two_schedulers(self, ref: Any) -> None:
        lr0 = 0.1
        lucid_opt = _dummy_lucid_optimizer(lr0)
        ref_opt = _dummy_ref_optimizer(lr0, ref)

        lucid_sched = optim.lr_scheduler.ChainedScheduler(
            [
                optim.lr_scheduler.ExponentialLR(lucid_opt, gamma=0.95),
                optim.lr_scheduler.StepLR(lucid_opt, step_size=5, gamma=0.5),
            ]
        )
        ref_sched = ref.optim.lr_scheduler.ChainedScheduler(
            [
                ref.optim.lr_scheduler.ExponentialLR(ref_opt, gamma=0.95),
                ref.optim.lr_scheduler.StepLR(ref_opt, step_size=5, gamma=0.5),
            ]
        )

        _assert_lr_sequence(_collect_lrs(lucid_sched, 15), _collect_lrs(ref_sched, 15))

    @pytest.mark.parametrize(
        "make",
        [
            lambda S, o: [
                S.ConstantLR(o, factor=0.5, total_iters=3),
                S.ExponentialLR(o, gamma=0.9),
            ],
            lambda S, o: [
                S.LinearLR(o, start_factor=0.25, total_iters=4),
                S.StepLR(o, step_size=3, gamma=0.5),
            ],
        ],
        ids=["warmup-and-decay", "linear-and-step"],
    )
    def test_the_factors_multiply(self, make: Any, ref: Any) -> None:
        """Each child computed its rate from its own base and wrote it over the
        last one's, so a warmup chained with a decay lost the decay."""
        lucid_opt = _dummy_lucid_optimizer(0.1)
        ref_opt = _dummy_ref_optimizer(0.1, ref)
        lucid_sched = optim.lr_scheduler.ChainedScheduler(make(optim.lr_scheduler, lucid_opt))
        ref_sched = ref.optim.lr_scheduler.ChainedScheduler(make(ref.optim.lr_scheduler, ref_opt))
        _assert_lr_sequence(_collect_lrs(lucid_sched, 12), _collect_lrs(ref_sched, 12))
