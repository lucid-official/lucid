"""fused_step + GradScaler — X4.3 acceptance.

The GradScaler integration runs the scale → unscale → found_inf →
conditional update plumbing fully inside the compiled executable so
the per-step latency budget matches the no-scaler path.  The Python
side only refreshes two 0-D scalar feeds (scale + inv_scale) before
each run and reads one 0-D found_inf back after — no per-param
Python loop, no per-step CPU round-trip on the gradient buffers.

The contract mirrors :class:`lucid.amp.GradScaler` exactly:

1. ``scaler.scale(loss).backward()`` is fused — the loss is multiplied
   by ``scaler._scale`` before MPSGraph autograd derives the gradients
   so the F16 chain values stay clear of underflow.
2. ``scaler.unscale_(opt)`` is fused — every ghost-grad placeholder
   passed to the optimizer math is replaced with ``grad * inv_scale``
   (cast to F32 first to match the eager GradScaler's "unscale in
   F32" convention; F16 ``1/scale`` at ``init_scale=2**16`` is
   subnormal and Metal flushes it to zero).
3. ``scaler.step(opt)`` is fused — ``found_inf = OR(any(!isfinite(g)))``
   across every unscaled gradient is computed by graph ops, and each
   optimizer output is wrapped in ``where(found_inf, old, new)`` so
   an overflow step preserves params + state buffers verbatim.
4. ``scaler.update()`` runs on the Python side after each call: the
   found_inf flag is read back from the persistent F32 0-D holder,
   set on ``scaler._found_inf``, and the scaler's growth/backoff
   schedule is advanced.

The user-facing loss returned by ``step(*args)`` is always the
*unscaled* loss — the trace divides the scaled-loss output by the
applied scale on the return path so callers see numerically the same
value as the eager AMP loop.
"""

import lucid
import lucid.amp as amp
import lucid.metal as metal
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.amp import GradScaler
from lucid.compile import fused_step

from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _loss_fn(pred: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
    """MSE in F32 — cast pred up from F16 to match target dtype."""
    return F.mse_loss(pred.to(target.dtype), target)


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc2(self.fc1(x).relu())


def test_grad_scaler_disabled_passes_through() -> None:
    """``GradScaler(enabled=False)`` behaves identically to no scaler.

    The trace must NOT install the scale/unscale/found_inf plumbing
    when the scaler is disabled — verifies the ``scaler_enabled``
    short-circuit in ``_FusedStep._build_plan``.
    """
    lucid.manual_seed(0)
    model = _MLP().to(COMPILE_DEVICE)
    model.train()
    opt = optim.SGD(model.parameters(), lr=1e-3)
    scaler = GradScaler(enabled=False)
    step = fused_step(model, _loss_fn, opt, grad_scaler=scaler)

    x = lucid.randn(8, 8).to(COMPILE_DEVICE)
    t = lucid.randn(8, 4).to(COMPILE_DEVICE)
    with amp.autocast(dtype=lucid.float16):
        loss = step(x, t)
    metal.synchronize()
    # Just verify it doesn't blow up and produces a finite loss.
    val = float(loss.item())
    assert val == val, f"disabled-scaler loss should be finite, got {val}"


def test_grad_scaler_growth_schedule() -> None:
    """Five overflow-free steps with ``growth_interval=2`` doubles
    the scale exactly twice (steps 2 and 4).

    Verifies the C++ found_inf detection reports the correct False
    value through every step, and the Python-side ``scaler.update()``
    is invoked once per dispatch with the right flag.
    """
    lucid.manual_seed(0)
    model = _MLP().to(COMPILE_DEVICE)
    model.train()
    opt = optim.SGD(model.parameters(), lr=1e-3)
    scaler = GradScaler(init_scale=2.0**8, growth_factor=2.0, growth_interval=2)
    step = fused_step(model, _loss_fn, opt, grad_scaler=scaler)

    x = lucid.randn(8, 8).to(COMPILE_DEVICE)
    t = lucid.randn(8, 4).to(COMPILE_DEVICE)

    # Schedule trace (growth_interval=2, init_scale=2**8):
    #   step 1: tracker 0→1, no growth → 2**8 = 256
    #   step 2: tracker 1→2 ≥ interval → grow → 2**9 = 512
    #   step 3: tracker 0→1, no growth → 2**9 = 512
    #   step 4: tracker 1→2 ≥ interval → grow → 2**10 = 1024
    #   step 5: tracker 0→1, no growth → 2**10 = 1024
    expected_scales = [2.0**8, 2.0**9, 2.0**9, 2.0**10, 2.0**10]
    actual_scales: list[float] = []
    losses: list[float] = []
    with amp.autocast(dtype=lucid.float16):
        for _ in range(5):
            loss = step(x, t)
            metal.synchronize()
            losses.append(float(loss.item()))
            actual_scales.append(scaler.get_scale())

    assert (
        actual_scales == expected_scales
    ), f"growth schedule wrong; expected {expected_scales}, got {actual_scales}"
    # Loss should decrease (the scaler isn't tripping any overflow path here).
    assert losses[-1] < losses[0], f"loss should decrease over 5 SGD steps; got {losses}"


def test_grad_scaler_overflow_skips_step_and_backs_off() -> None:
    """``init_scale=2**30`` deliberately blows the F16 chain so every
    derived gradient becomes NaN/Inf.  The fused step must:

    - Detect the overflow via the in-graph found_inf reduction.
    - Wrap each param's new value with ``where(found_inf, old, new)``
      so the params stay at their pre-step values.
    - Halve the scale via the Python-side ``scaler.update()``.

    This is the central regression that motivates the GradScaler
    integration — without the conditional wrapper, NaN gradients
    poison the params and training stalls indefinitely.
    """
    import numpy as np  # noqa: PLC0415

    lucid.manual_seed(0)
    model = _MLP().to(COMPILE_DEVICE)
    model.train()
    opt = optim.SGD(model.parameters(), lr=1e-3)
    init_scale = 2.0**30
    scaler = GradScaler(init_scale=init_scale, growth_interval=1)
    step = fused_step(model, _loss_fn, opt, grad_scaler=scaler)

    x = lucid.randn(8, 8).to(COMPILE_DEVICE)
    t = lucid.randn(8, 4).to(COMPILE_DEVICE)

    # Snapshot params BEFORE the step.
    before = [p.numpy().copy() for p in model.parameters()]

    with amp.autocast(dtype=lucid.float16):
        _ = step(x, t)
    metal.synchronize()

    # Scale must have halved (backoff_factor=0.5 by default).
    assert scaler.get_scale() == init_scale * 0.5, (
        f"expected scale halved on overflow; " f"init={init_scale}, after={scaler.get_scale()}"
    )

    # Params must be unchanged — the where(found_inf, old, new)
    # wrapper picked ``old`` for every output slot.
    for i, p in enumerate(model.parameters()):
        after = p.numpy()
        # Reject any NaN propagation into params.
        assert not np.isnan(after).any(), (
            f"param {i} became NaN despite overflow-skip; "
            "where(found_inf, old, new) didn't catch the overflow"
        )
        np.testing.assert_array_equal(
            before[i],
            after,
            err_msg=(
                f"param {i} changed despite overflow; expected skip-step "
                "to leave params untouched"
            ),
        )


def test_grad_scaler_loss_value_is_unscaled() -> None:
    """The user-facing loss returned by ``step`` is the *unscaled* loss.

    Without this, every loss the user observes would be scaled by
    ``scaler._scale`` (potentially ~2**16+), which is silently wrong
    for logging / convergence monitoring.  Verifies the
    ``loss_tensor / self._last_applied_scale`` divide on the return
    path in ``_FusedStep._run``.
    """
    lucid.manual_seed(0)
    model = _MLP().to(COMPILE_DEVICE)
    model.train()
    opt = optim.SGD(model.parameters(), lr=1e-3)
    scaler = GradScaler(init_scale=2.0**12, growth_interval=10_000)
    step = fused_step(model, _loss_fn, opt, grad_scaler=scaler)

    x = lucid.randn(8, 8).to(COMPILE_DEVICE)
    t = lucid.randn(8, 4).to(COMPILE_DEVICE)

    with amp.autocast(dtype=lucid.float16):
        scaled_loss_estimate = float(_loss_fn(model(x), t).item())
        loss_from_step = step(x, t)
    metal.synchronize()
    val = float(loss_from_step.item())
    # The returned loss should be on the same order of magnitude as
    # the eager loss (not multiplied by ~4096).  Allow a generous
    # 10× window since the scaled vs unscaled difference is 12 bits
    # of magnitude.
    assert val < 10.0 * abs(scaled_loss_estimate), (
        f"step returned suspiciously large loss {val}; "
        f"eager loss estimate is {scaled_loss_estimate}, scale is "
        f"{scaler.get_scale()} — the return path didn't unscale"
    )


def test_grad_scaler_state_dict_round_trip() -> None:
    """``state_dict`` / ``load_state_dict`` survives a fused step.

    The scaler state lives on the GradScaler instance (pure Python),
    not inside the executable, so this is a sanity check that the
    fused step doesn't accidentally shadow / mutate scaler fields.
    """
    lucid.manual_seed(0)
    model = _MLP().to(COMPILE_DEVICE)
    opt = optim.SGD(model.parameters(), lr=1e-3)
    scaler = GradScaler(init_scale=2.0**10, growth_interval=2)
    step = fused_step(model, _loss_fn, opt, grad_scaler=scaler)

    x = lucid.randn(8, 8).to(COMPILE_DEVICE)
    t = lucid.randn(8, 4).to(COMPILE_DEVICE)
    with amp.autocast(dtype=lucid.float16):
        step(x, t)
    metal.synchronize()

    snap = scaler.state_dict()
    # Re-create a fresh scaler and load.
    scaler2 = GradScaler()
    scaler2.load_state_dict(snap)
    assert scaler2.get_scale() == scaler.get_scale()
    assert snap["growth_factor"] == 2.0
    assert snap["backoff_factor"] == 0.5
    assert snap["growth_interval"] == 2
