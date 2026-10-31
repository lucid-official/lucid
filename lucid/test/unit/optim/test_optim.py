"""``lucid.optim`` — optimizers + LR schedulers."""

import pathlib
import subprocess
import sys

import numpy as np
import pytest

import lucid
import lucid.optim as optim
from lucid.test._fixtures.devices import metal_available


def _quadratic_problem() -> tuple[lucid.Tensor, callable]:
    """Return ``(x, fn)`` where ``fn(x) = (x − target)²`` minimised at
    ``target = 1.0`` with ``x`` initialised at the origin."""
    x = lucid.tensor([0.0, 0.0], requires_grad=True)
    target = lucid.tensor([1.0, 1.0])

    def loss() -> lucid.Tensor:
        return ((x - target) ** 2).sum()

    return x, loss


class TestSGD:
    def test_converges(self) -> None:
        x, loss_fn = _quadratic_problem()
        opt = optim.SGD([x], lr=0.1)
        for _ in range(200):
            opt.zero_grad()
            loss_fn().backward()
            opt.step()
        np.testing.assert_allclose(x.numpy(), [1.0, 1.0], atol=1e-3)


class TestAdam:
    def test_converges(self) -> None:
        x, loss_fn = _quadratic_problem()
        opt = optim.Adam([x], lr=0.1)
        for _ in range(200):
            opt.zero_grad()
            loss_fn().backward()
            opt.step()
        np.testing.assert_allclose(x.numpy(), [1.0, 1.0], atol=1e-3)


class TestAdamW:
    def test_step_runs(self) -> None:
        x, loss_fn = _quadratic_problem()
        opt = optim.AdamW([x], lr=0.1, weight_decay=0.01)
        for _ in range(50):
            opt.zero_grad()
            loss_fn().backward()
            opt.step()
        # After 50 steps the value should have moved meaningfully.
        assert (x.numpy() > 0.5).all()


class TestRMSprop:
    def test_converges_loose(self) -> None:
        x, loss_fn = _quadratic_problem()
        opt = optim.RMSprop([x], lr=0.05)
        for _ in range(500):
            opt.zero_grad()
            loss_fn().backward()
            opt.step()
        np.testing.assert_allclose(x.numpy(), [1.0, 1.0], atol=5e-3)


class TestZeroGrad:
    def test_zero_grad_resets(self) -> None:
        x, loss_fn = _quadratic_problem()
        opt = optim.SGD([x], lr=0.1)
        loss_fn().backward()
        assert x.grad is not None
        opt.zero_grad()
        # ``zero_grad`` may leave ``grad`` as zeros or None — both
        # satisfy "no leftover gradient".
        if x.grad is not None:
            np.testing.assert_array_equal(x.grad.numpy(), [0.0, 0.0])


class TestLRScheduler:
    def test_step_lr_decays(self) -> None:
        x, _ = _quadratic_problem()
        opt = optim.SGD([x], lr=1.0)
        if not hasattr(optim, "lr_scheduler"):
            pytest.skip("lr_scheduler module not exposed")
        sched = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
        sched.step()
        # After one step the LR should be halved.
        for group in opt.param_groups:
            assert abs(group["lr"] - 0.5) < 1e-6

    def test_lambda_lr_wants_one_lambda_per_group(self) -> None:
        a, b = lucid.zeros(1, requires_grad=True), lucid.zeros(1, requires_grad=True)
        opt = optim.SGD([{"params": [a]}, {"params": [b]}], lr=0.1)
        with pytest.raises(ValueError, match="Expected 2 lr_lambdas, but got 1"):
            optim.lr_scheduler.LambdaLR(opt, [lambda epoch: 1.0])


# What a scheduler does with a param group added after it was built is
# checked against the reference in test_scheduler_late_param_group.py.


_LEAK_PROBE = """
import gc, sys, lucid, lucid.nn as nn, lucid.optim as optim
from lucid._C import engine as _C_engine

retain = sys.argv[1] == "1"
model = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64)).to("metal")
opt = optim.Adam(model.parameters(), lr=1e-3)
x = lucid.randn(16, 64).to("metal")
t = lucid.randn(16, 64).to("metal")
held = []


def step():
    opt.zero_grad()
    out = model(x)
    ((out - t) ** 2).mean().backward()
    opt.step()          # NO per-step .item() — the leak-exposing pattern
    if retain:
        held.append(out)


for _ in range(20):     # warm up — lazy alloc / first-step state settles
    step()
_C_engine.synchronize_gpu()
base = _C_engine.metal_active_memory()

marks = []
for _ in range(12):
    for _ in range(100):
        step()
    _C_engine.synchronize_gpu()
    marks.append(_C_engine.metal_active_memory() - base)

half = len(marks) // 2
print("DRIFT", min(marks[half:]) - min(marks[:half]))
"""


@pytest.mark.skipif(not metal_available(), reason="the probe measures Metal memory")
class TestOptimizerGpuMemory:
    """Regression: the GPU optimizer step must not leak active MLX memory.

    ``Optimizer::step`` writes each param back as an UNEVALUATED MLX array;
    without a per-step eval the lazy graph can pin every prior step's
    compute → unbounded active-memory / RSS growth on the GPU path.

    **This runs in its own process, and that is the whole point.** Two
    earlier versions of this test were wrong in ways worth recording,
    because both looked right and both produced a red CI for days.

    The first took ``get_active_memory()`` once before a 300-step loop and
    once after. That quantity *oscillates*: sampled over four consecutive
    windows on a loaded process it reads ``+16384, -16384, +16384, -16384``.
    A leak cannot go negative, so a single difference cannot tell monotone
    growth from an allocator holding a block at the moment you look.

    The second measured the trough across twelve windows, which is the
    right statistic — and it still failed, reporting a genuinely linear
    20480 B/step, exactly one step's forward activations, unbounded to
    61 MB over 3000 steps. That is what a real leak looks like. It was
    not one: it is absent in a fresh interpreter, absent from a standalone
    script under heavy allocation load, unaffected by an explicit
    ``_metal_eval_params()`` flush or a per-step ``.item()``, and not
    attributable to any single test directory. It appears only inside a
    large pytest session, which holds references this measurement cannot
    tell apart from the optimizer's own.

    Since the statistic is process-global, the measurement has to own the
    process. The subprocess measures Lucid; running it in-session measured
    pytest.
    """

    _TIMEOUT = 600

    @staticmethod
    def _trough_drift(retain: bool) -> int:
        """Run the probe in a fresh interpreter; return the floor's drift.

        Parameters
        ----------
        retain : bool
            Hold one activation per step — a leak by construction, used to
            prove this measurement is able to fail.

        Returns
        -------
        int
            ``min(second half) - min(first half)`` in bytes, over twelve
            windows of a hundred steps.
        """
        completed = subprocess.run(
            [sys.executable, "-c", _LEAK_PROBE, "1" if retain else "0"],
            capture_output=True,
            text=True,
            timeout=TestOptimizerGpuMemory._TIMEOUT,
            cwd=str(pathlib.Path(__file__).resolve().parents[4]),
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        line = next(ln for ln in completed.stdout.splitlines() if ln.startswith("DRIFT "))
        return int(line.split()[1])

    def test_no_active_memory_leak(self) -> None:
        drift = self._trough_drift(retain=False)
        assert drift < 8192, (
            f"the optimizer's active-memory floor rose {drift} B over 1200 "
            f"steps — Optimizer::step eval flush regressed"
        )

    def test_the_measurement_can_detect_a_leak(self) -> None:
        """Guards the test above.

        A measurement that always read zero would pass forever. Retaining
        one activation per step is a leak by construction and must
        register — it reads about 2.4 MB against the 8192 B allowance.
        """
        assert self._trough_drift(retain=True) > 8192
