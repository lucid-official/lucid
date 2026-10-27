"""Regression tests for LCD-593.

``GradScaler``'s skip decision (eager ``scaler.step()`` vs.
:func:`lucid.compile.fused_step`) must agree across a training run: both
read "did any gradient overflow" from the same scaled loss, so they must
halve (or grow) the scale on the same step, including right at the
``float16`` overflow boundary (`65504.0`).

Eager's backward runs each op's gradient in the op's forward compute dtype
(e.g. ``float16`` for ``nn.Linear`` under ``autocast(float16)``) and casts
the resulting gradient back to each input's dtype at accumulation time.
Compiled VJP emitters (:mod:`lucid._C.compile.VjpEmitters`) reconcile the
incoming gradient and saved forward operands to the op's forward compute
dtype via ``vjp_compute_dtype`` and commit half-precision intermediates
via ``commit_half_before_widen`` before widening to ``float32`` so
``MPSGraph`` does not fuse ``float16 -> float32`` casts into ``float32``
kernels and elide ``float16`` overflow or rounding.
"""

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
import pytest
from lucid.amp import GradScaler, autocast
from lucid.compile import fused_step

from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _scale_sequence(
    seed: int, init_scale: float, n_steps: int = 4
) -> tuple[list[float], list[float]]:
    """Run eager and ``fused_step`` side by side; return their scale
    trajectories (one entry per step, read after that step's ``update()``).
    """
    lucid.manual_seed(seed)
    eager_model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).to(COMPILE_DEVICE)
    comp_model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).to(COMPILE_DEVICE)
    with lucid.no_grad():
        for pe, pc in zip(eager_model.parameters(), comp_model.parameters(), strict=True):
            pc.copy_(pe.detach().clone())

    opt_eager = optim.AdamW(list(eager_model.parameters()), lr=1e-2)
    opt_comp = optim.AdamW(list(comp_model.parameters()), lr=1e-2)
    scaler_eager = GradScaler(init_scale=init_scale)
    scaler_comp = GradScaler(init_scale=init_scale)
    step = fused_step(comp_model, F.mse_loss, opt_comp, grad_scaler=scaler_comp)

    eager_scales: list[float] = []
    comp_scales: list[float] = []
    for bsz in [6] * n_steps:
        xb = lucid.randn(bsz, 8, device=COMPILE_DEVICE)
        yb = lucid.randn(bsz, 4, device=COMPILE_DEVICE)
        with autocast("metal", dtype=lucid.float16):
            opt_eager.zero_grad()
            loss_e = F.mse_loss(eager_model(xb), yb)
            scaler_eager.scale(loss_e).backward()
            scaler_eager.step(opt_eager)
            scaler_eager.update()
            step(xb, yb)
        eager_scales.append(scaler_eager.get_scale())
        comp_scales.append(scaler_comp.get_scale())
    return eager_scales, comp_scales


@pytest.mark.parametrize("init_scale_exp", list(range(15, 23)))
def test_skip_sequence_matches_away_from_known_boundary(init_scale_exp: int) -> None:
    """Away from the LCD-593 boundary band, eager and ``fused_step`` must
    halve/grow the scale on the same steps.  Seed 12 never enters that band
    for ``init_scale`` in ``2**15 .. 2**22`` (verified empirically) — a
    regression anywhere in this sweep is a *new* defect, not LCD-593."""
    eager_scales, comp_scales = _scale_sequence(seed=12, init_scale=2.0**init_scale_exp)
    assert eager_scales == comp_scales


@pytest.mark.parametrize(
    "seed,init_scale_exp",
    [(1, 18), (1, 19), (36, 17), (36, 18), (36, 19)],
)
def test_skip_sequence_matches_at_overflow_boundary(seed: int, init_scale_exp: int) -> None:
    eager_scales, comp_scales = _scale_sequence(seed=seed, init_scale=2.0**init_scale_exp)
    assert eager_scales == comp_scales


def test_compiled_vjp_and_half_widening_preserve_float16_overflow_boundary() -> None:
    """Directly pin the float16 overflow boundary (65504.0) regardless of GPU:
    when a float16 intermediate or Linear VJP exceeds 65504.0 before widening
    to float32, both eager and compiled produce ``inf`` so ``GradScaler``
    skips the step and halves the scale identically."""
    eager_layer = nn.Linear(4, 4, bias=False).to(COMPILE_DEVICE)
    comp_layer = nn.Linear(4, 4, bias=False).to(COMPILE_DEVICE)
    with lucid.no_grad():
        eager_layer.weight.fill_(16.0)
        comp_layer.weight.fill_(16.0)

    opt_eager = optim.SGD(list(eager_layer.parameters()), lr=0.1)
    opt_comp = optim.SGD(list(comp_layer.parameters()), lr=0.1)
    scaler_eager = GradScaler(init_scale=4096.0)
    scaler_comp = GradScaler(init_scale=4096.0)
    step = fused_step(comp_layer, F.mse_loss, opt_comp, grad_scaler=scaler_comp)

    xb = lucid.ones(2, 4, device=COMPILE_DEVICE)
    yb = lucid.zeros(2, 4, device=COMPILE_DEVICE)
    with autocast("metal", dtype=lucid.float16):
        opt_eager.zero_grad()
        loss_e = F.mse_loss(eager_layer(xb), yb)
        scaler_eager.scale(loss_e).backward()
        scaler_eager.step(opt_eager)
        scaler_eager.update()
        step(xb, yb)

    assert scaler_eager.get_scale() == 2048.0
    assert scaler_comp.get_scale() == 2048.0
