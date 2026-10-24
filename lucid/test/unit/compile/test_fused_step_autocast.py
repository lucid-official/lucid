"""fused_step + autocast — X4.4 acceptance.

The autocast → fused_step integration works as a side-effect of:

1. C++ ``AutocastGuard`` already dispatches per-op forward
   (so the trace records ``astype`` casts at op boundaries when
   the trace runs inside ``with autocast()``)
2. The ``astype`` VJP from Manual VJP P1 differentiates those
   casts cleanly (gradient flows back as ``astype`` to the
   source dtype)
3. The mixed-dtype trace acceptance lifted in X4.1
4. Elementwise optimizer math runs under ``KeepInput`` (LCD-500),
   so F32 master weights stay F32 under autocast without needing an
   explicit ``AutocastGuard`` opt-out around optimizer emission.

Known limitations (documented for follow-up):
- BatchNorm under autocast in fused_step fails (`emitter
  'batch_norm' returned false`).  The BN op's AmpPolicy is
  ``KeepInput`` (vs ``ForceFP32`` in the reference framework)
  which casts BN inputs to F16 — the compile-path BN emitter
  doesn't support F16 inputs.  Worked around in production by
  using LayerNorm (transformer pattern) or by manually disabling
  autocast around BN modules.  Tracked as a separate AMP
  coverage gap.
- GradScaler (X4.3) — not integrated; the basic mixed-dtype
  fused_step is numerically stable for many models without it,
  but production AMP training typically needs it for F16
  gradient underflow protection.  Tracked as P2.2.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
import lucid.amp as amp
import lucid.metal as metal
from lucid.compile import fused_step

from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _loss_fn(pred: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
    """MSE in F32 — cast pred up from F16 to match target dtype."""
    return F.mse_loss(pred.to(target.dtype), target)


# ── Simple MLP — base case ──────────────────────────────────────────


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc2(self.fc1(x).relu())


def test_fused_step_autocast_mlp_trains() -> None:
    """3-layer MLP + fused_step + autocast(F16) — 5 SGD steps converge."""
    lucid.manual_seed(0)
    model = _MLP().to(COMPILE_DEVICE)
    model.train()
    opt = optim.SGD(model.parameters(), lr=1e-3)
    step = fused_step(model, _loss_fn, opt)

    x = lucid.randn(8, 8).to(COMPILE_DEVICE)
    t = lucid.randn(8, 4).to(COMPILE_DEVICE)

    losses: list[float] = []
    for _ in range(5):
        with amp.autocast(dtype=lucid.float16):
            loss = step(x, t)
        _ = float(loss.item())
        metal.synchronize()
        losses.append(float(loss.item()))

    assert (
        losses[0] > losses[-1]
    ), f"expected loss to decrease over 5 SGD steps; got {losses}"


def test_fused_step_autocast_preserves_f32_master_weights() -> None:
    """Critical invariant: model parameters stay F32 after autocast steps.

    The split-scope fix in `_FusedStep._build_plan` is what
    enforces this — without it the optimizer would read the params
    through the active autocast guard, cast them to F16, and the
    new_param output would be F16 (causing a dtype-mismatch error
    in `run_executable_inplace`'s buffer-binding step).
    """
    lucid.manual_seed(0)
    model = _MLP().to(COMPILE_DEVICE)
    model.train()
    opt = optim.SGD(model.parameters(), lr=1e-3)
    step = fused_step(model, _loss_fn, opt)

    x = lucid.randn(8, 8).to(COMPILE_DEVICE)
    t = lucid.randn(8, 4).to(COMPILE_DEVICE)

    with amp.autocast(dtype=lucid.float16):
        step(x, t)
    metal.synchronize()

    for name, p in model.named_parameters():
        assert p.dtype == lucid.float32, (
            f"param {name} got cast to {p.dtype} under autocast; "
            "master weights must stay F32 — the split-scope fix in "
            "_FusedStep._build_plan failed."
        )


# ── LayerNorm-MLP — transformer-shape ───────────────────────────────


class _LNMLP(nn.Module):
    """LayerNorm-based MLP — exercises the LN path under autocast.

    LayerNorm's compile-path emitter handles F16 inputs cleanly
    (unlike the BatchNorm emitter which currently rejects F16 — see
    module docstring §"Known limitations").  This shape is the
    standard "pre-norm transformer block without attention" pattern.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 32)
        self.ln1 = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, 32)
        self.ln2 = nn.LayerNorm(32)
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc3(self.ln2(self.fc2(self.ln1(self.fc1(x))).relu()))


def test_fused_step_autocast_layernorm_mlp_trains() -> None:
    """LN-based MLP + autocast(F16) + fused_step — 5 SGD steps converge."""
    lucid.manual_seed(0)
    model = _LNMLP().to(COMPILE_DEVICE)
    model.train()
    opt = optim.SGD(model.parameters(), lr=1e-3)
    step = fused_step(model, _loss_fn, opt)

    x = lucid.randn(8, 8).to(COMPILE_DEVICE)
    t = lucid.randn(8, 4).to(COMPILE_DEVICE)

    losses: list[float] = []
    for _ in range(5):
        with amp.autocast(dtype=lucid.float16):
            loss = step(x, t)
        _ = float(loss.item())
        metal.synchronize()
        losses.append(float(loss.item()))

    assert (
        losses[0] > losses[-1]
    ), f"expected loss to decrease over 5 SGD steps; got {losses}"


# ── Forward-only ── `lucid.compile(model)` + autocast ──────────────


class _BN1DMLP(nn.Module):
    """BatchNorm1d-based MLP — exercises BN1d under autocast.

    Tests the BN F16 fix: previously the BN emitter checked
    ``x_t.shape.count`` which can be 0 for MPSGraph tensors
    produced by ops like ``conv2d`` that don't populate the shape
    attribute eagerly.  Switched to the trace's recorded
    ``node.outputs[0].shape`` which is always accurate.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.bn = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc2(self.bn(self.fc1(x)))


def test_fused_step_autocast_batchnorm1d_mlp_trains() -> None:
    """BN1d-based MLP + fused_step + autocast(F16) — 3 SGD steps converge.

    Previously broken: BN emit returned false on the trace produced
    under autocast because the MPSGraph tensor's ``.shape`` attribute
    was empty.  Fix in BN emitter (Norm.mm) reads
    ``node.outputs[0].shape`` from trace IR instead.
    """
    lucid.manual_seed(0)
    model = _BN1DMLP().to(COMPILE_DEVICE)
    model.train()
    opt = optim.SGD(model.parameters(), lr=1e-3)
    step = fused_step(model, _loss_fn, opt)

    x = lucid.randn(8, 8).to(COMPILE_DEVICE)
    t = lucid.randn(8, 4).to(COMPILE_DEVICE)

    losses: list[float] = []
    for _ in range(3):
        with amp.autocast(dtype=lucid.float16):
            loss = step(x, t)
        _ = float(loss.item())
        metal.synchronize()
        losses.append(float(loss.item()))

    assert losses[0] > losses[-1], (
        f"expected loss to decrease over 3 SGD steps with BN1d under "
        f"autocast; got {losses}"
    )


class _BN2DConvMLP(nn.Module):
    """Conv2d → BN2d → Linear — exercises the BN-after-Conv path
    where MPSGraph's conv output may have empty shape attribute.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16 * 8 * 8, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc(self.bn(self.conv(x)).reshape(x.shape[0], -1))


def test_fused_step_autocast_batchnorm2d_conv_trains() -> None:
    """Conv→BN2d→FC + fused_step + autocast(F16) — 3 SGD steps converge.

    The BN F16 emit fix specifically targets this case (BN after Conv,
    where the MPSGraph tensor produced by ``convolution2DWithSourceTensor:``
    has empty ``.shape``).
    """
    lucid.manual_seed(0)
    model = _BN2DConvMLP().to(COMPILE_DEVICE)
    model.train()
    opt = optim.SGD(model.parameters(), lr=1e-3)
    step = fused_step(model, _loss_fn, opt)

    x = lucid.randn(4, 3, 8, 8).to(COMPILE_DEVICE)
    t = lucid.randn(4, 4).to(COMPILE_DEVICE)

    losses: list[float] = []
    for _ in range(3):
        with amp.autocast(dtype=lucid.float16):
            loss = step(x, t)
        _ = float(loss.item())
        metal.synchronize()
        losses.append(float(loss.item()))

    assert losses[0] > losses[-1], (
        f"expected loss to decrease over 3 SGD steps with BN2d-after-Conv "
        f"under autocast; got {losses}"
    )


@pytest.mark.usefixtures("inference")
def test_lucid_compile_forward_autocast_matches_eager() -> None:
    """lucid.compile(model) + autocast(F16) — output parity vs eager."""
    lucid.manual_seed(0)
    model = _MLP().to(COMPILE_DEVICE)
    model.eval()
    x = lucid.randn(2, 8).to(COMPILE_DEVICE)

    with amp.autocast(dtype=lucid.float16):
        out_eager = model(x)
    _ = float(out_eager.sum().item())
    metal.synchronize()

    cm = lucid.compile(model)
    with amp.autocast(dtype=lucid.float16):
        out_compile = cm(x)
    _ = float(out_compile.sum().item())
    metal.synchronize()

    assert out_eager.dtype == lucid.float16
    assert out_compile.dtype == lucid.float16

    abs_diff = float((out_eager - out_compile).abs().max().item())
    scale = float(out_eager.abs().max().item())
    rel_diff = abs_diff / max(scale, 1e-9)
    # F16 tolerance — 1e-2 is the standard "within ~3 decimal digits" gate.
    assert rel_diff < 1e-2, (
        f"compile output diverges from eager under autocast; "
        f"rel_diff={rel_diff:.4e}"
    )


class _ResBlock(nn.Module):
    """ResNet-style residual block: conv-bn-relu-conv-bn + residual + relu.

    Exercises the systemic VJP mixed-dtype reconciliation path:
    - The residual ``h + x`` creates two-path grad accumulation, which
      under autocast can route F32 (eager-downcast forward) and F16
      (chain) grads to the same accumulator.
    - The conv VJP needs matching dtypes on its grad + weight inputs.
    - The ReLU VJP must derive its mask dtype from grad, not from x.
    """

    def __init__(self, c: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + x)


class _ResNetMini(nn.Module):
    """conv stem → maxpool → ResBlock → adaptive avg pool → linear.

    Mirrors the canonical ResNet topology in miniature; the ResBlock
    plus the stem chain together exposed the systemic VJP mixed-dtype
    bug that landed with P5.3 (Activation / Arith / Norm / Conv /
    Linear / Loss / Shape VJPs all needing chain-dtype reconciliation).
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2)
        self.block = _ResBlock(16)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = F.relu(self.bn(self.conv(x)))
        h = self.pool(h)
        h = self.block(h)
        h = self.gap(h).flatten(1)
        return self.fc(h)


def test_fused_step_autocast_resnet_mini_trains() -> None:
    """ResNet-mini + fused_step + autocast(F16) with manual VJP REQUIRED.

    The ``LUCID_MANUAL_VJP_REQUIRE=1`` env gate forces every VJP in the
    backward walker to land via a manual emitter — no fallback to
    MPSGraph autograd permitted.  This is the regression gate for the
    P5 systemic mixed-dtype reconciliation: if any VJP body multiplies
    a F16 chain operand with a F32 forward activation, MPSGraph's
    same-dtype check trips and ``compile_generic_fused_step`` raises.
    """
    import os

    lucid.manual_seed(0)
    model = _ResNetMini().to(COMPILE_DEVICE)
    model.train()
    opt = optim.SGD(model.parameters(), lr=1e-2)

    def _ce_loss(pred: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
        return F.cross_entropy(pred, target)

    prev_require = os.environ.get("LUCID_MANUAL_VJP_REQUIRE")
    os.environ["LUCID_MANUAL_VJP_REQUIRE"] = "1"
    try:
        step = fused_step(model, _ce_loss, opt)
        x = lucid.randn(4, 3, 8, 8).to(COMPILE_DEVICE)
        t = lucid.randint(0, 4, (4,)).to(COMPILE_DEVICE)

        losses: list[float] = []
        for _ in range(3):
            with amp.autocast(dtype=lucid.float16):
                loss = step(x, t)
            metal.synchronize()
            losses.append(float(loss.item()))
    finally:
        if prev_require is None:
            os.environ.pop("LUCID_MANUAL_VJP_REQUIRE", None)
        else:
            os.environ["LUCID_MANUAL_VJP_REQUIRE"] = prev_require

    assert losses[0] > losses[-1], (
        f"expected ResNet-mini loss to decrease over 3 SGD steps under "
        f"autocast + LUCID_MANUAL_VJP_REQUIRE=1; got {losses}"
    )
