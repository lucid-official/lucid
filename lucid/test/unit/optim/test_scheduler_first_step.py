"""A scheduler's step-0 rate is in force for the very first optimizer step.

Only ``step()`` used to write a rate, so until the first ``step()`` the
optimizer ran at its base rate whatever the schedule said: a warmup's first
update at full rate, ``OneCycleLR``'s at ``max_lr`` — 25x the
``max_lr / div_factor`` it starts from.  The scheduler parity tests read the
rate only after each ``step()`` and never saw it; a 3,000-step GPT run
against the reference did, as a 1% gap in its first loss window.

``ChainedScheduler`` multiplied nothing: each child wrote its own rate over
the last, so a warmup chained with a decay lost the decay once the warmup
ended.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim

S = optim.lr_scheduler


def _opt(lr: float = 0.1) -> optim.SGD:
    return optim.SGD([nn.Parameter(lucid.zeros(1))], lr=lr, momentum=0.9)


def _rate(opt: optim.SGD) -> float:
    return float(opt.param_groups[0]["lr"])


@pytest.mark.parametrize(
    ("make", "first"),
    [
        (lambda o: S.LambdaLR(o, lr_lambda=lambda s: (s + 1) / 5), 0.02),
        (lambda o: S.LinearLR(o, start_factor=0.2, total_iters=4), 0.02),
        (lambda o: S.ConstantLR(o, factor=0.25, total_iters=3), 0.025),
        (lambda o: S.CyclicLR(o, base_lr=0.01, max_lr=0.1, step_size_up=2), 0.01),
        (lambda o: S.OneCycleLR(o, max_lr=0.1, total_steps=10), 0.004),
    ],
    ids=["LambdaLR", "LinearLR", "ConstantLR", "CyclicLR", "OneCycleLR"],
)
def test_the_first_step_runs_at_the_schedules_first_rate(make: object, first: float) -> None:
    opt = _opt()
    sched = make(opt)  # type: ignore[operator]
    assert _rate(opt) == pytest.approx(first)
    assert sched.get_last_lr()[0] == pytest.approx(first)


def test_a_second_scheduler_still_starts_from_the_true_base() -> None:
    opt = _opt()
    S.LinearLR(opt, start_factor=0.2, total_iters=3)
    cosine = S.CosineAnnealingLR(opt, T_max=4)
    assert cosine.base_lrs == [pytest.approx(0.1)]


def test_sequential_starts_at_its_first_childs_rate() -> None:
    opt = _opt()
    S.SequentialLR(
        opt,
        [S.LinearLR(opt, start_factor=0.2, total_iters=3), S.CosineAnnealingLR(opt, 4)],
        milestones=[3],
    )
    assert _rate(opt) == pytest.approx(0.02)


def test_chained_factors_multiply() -> None:
    opt = _opt()
    sched = S.ChainedScheduler(
        [S.ConstantLR(opt, factor=0.5, total_iters=2), S.ExponentialLR(opt, gamma=0.9)]
    )
    rates = [_rate(opt)]
    for _ in range(4):
        opt.step()
        sched.step()
        rates.append(_rate(opt))
    # 0.1 x constant (0.5 for two steps, then 1) x 0.9 ** t
    want = [0.1 * (0.5 if t < 2 else 1.0) * 0.9**t for t in range(5)]
    assert rates == pytest.approx(want)


def test_one_cycle_refuses_to_step_past_its_end() -> None:
    opt = _opt()
    sched = S.OneCycleLR(opt, max_lr=0.1, total_steps=3)
    for _ in range(3):
        opt.step()
        sched.step()
    with pytest.raises(ValueError, match="total steps is 3"):
        sched.step()
