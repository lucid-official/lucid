"""Regression tests for LCD-593.

``GradScaler``'s skip decision (eager ``scaler.step()`` vs.
:func:`lucid.compile.fused_step`) must agree across a training run: both
read "did any gradient overflow" from the same scaled loss, so they must
halve (or grow) the scale on the same step.

Near the float16 overflow boundary they currently do not (LCD-593): eager's
``nn.Linear`` backward operates on the saved *forward* tensors, which
``autocast(float16)`` already downcast to float16, so the scaled gradient
overflows to ``inf`` in float16 precision once ``scale`` is large enough.
The compiled backward (``lucid/_C/compile/VjpEmitters/linalg/Linear.mm``)
instead casts those saved tensors *up* to the incoming gradient's dtype
(``chain_dt = grad.dataType``, which is float32 here) before the matmul, so
the same matmul runs in float32 and never overflows at that scale. The two
paths only disagree in the narrow scale band where eager's float16 matmul
has already overflowed but naive float32 arithmetic has not — outside that
band (or once the scale is large enough that even the float32-side value
overflows too) they agree again.

Fixing the root cause means changing how every VJP emitter under
``lucid/_C/compile/VjpEmitters/`` reconciles mixed forward/gradient dtypes
(13 files use the same ``cast_if_needed(..., chain_dt)`` convention), which
needs a C++ rebuild and a full autocast+compile parity pass — out of scope
for this guard. See the LCD-593 report for the evidence (per-seed,
per-``init_scale`` scale sequences) and the escalation to the orchestrator.
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
    eager_model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).to(
        COMPILE_DEVICE
    )
    comp_model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).to(
        COMPILE_DEVICE
    )
    with lucid.no_grad():
        for pe, pc in zip(
            eager_model.parameters(), comp_model.parameters(), strict=True
        ):
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


# Not strict: which (seed, init_scale) points fall in the band depends on
# the GPU.  These five diverge on an M1 Pro, while the CI runner's GPU puts
# the same gradients on the finite side of the float16 max, so they XPASSed
# there (main 590a4b5a4).  The defect is the dtype direction, not these
# points.  The C++ fix's own test must pin the overflow directly.
@pytest.mark.xfail(
    strict=False,
    reason=(
        "LCD-593: compiled Linear VJP (lucid/_C/compile/VjpEmitters/"
        "linalg/Linear.mm) casts the saved forward x/W up to the incoming "
        "gradient's dtype (chain_dt = grad.dataType) instead of running "
        "the matmul in the op's own forward (autocast-lowered float16) "
        "precision the way eager's backward does, so the compiled matmul "
        "misses the float16 overflow eager hits by design in this scale "
        "band on some GPUs. Needs a C++ fix + rebuild."
    ),
)
@pytest.mark.parametrize(
    "seed,init_scale_exp",
    [(1, 18), (1, 19), (36, 17), (36, 18), (36, 19)],
)
def test_skip_sequence_matches_at_overflow_boundary(
    seed: int, init_scale_exp: int
) -> None:
    eager_scales, comp_scales = _scale_sequence(
        seed=seed, init_scale=2.0**init_scale_exp
    )
    assert eager_scales == comp_scales
