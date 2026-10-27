"""Manual VJP — Conv-transpose + Conv3d acceptance (X1.3).

Two small tests:
  * ConvTranspose2d as a tiny generator block.
  * Conv3d on a (B, C, D, H, W) volume.

Both verify forward + backward + bias gradient match eager under
``LUCID_MANUAL_VJP_REQUIRE=1`` (manual-or-die — proves the new
VJPs are actually executing).
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.compile import fused_step

from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor


@pytest.fixture(autouse=True)
def _manual_vjp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUCID_MANUAL_VJP", "1")
    monkeypatch.setenv("LUCID_MANUAL_VJP_REQUIRE", "1")


# ── ConvTranspose2d ─────────────────────────────────────────────────


class _ConvTransposeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # in=4, out=2, kernel=2, stride=2 → doubles spatial size.
        self.up = nn.ConvTranspose2d(4, 2, kernel_size=2, stride=2)
        self.fc = nn.Linear(2, 1)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = self.up(x).relu()
        h = h.mean(dim=(2, 3), keepdim=False)  # (B, 2)
        return self.fc(h)


def _loss(p: lucid.Tensor, t: lucid.Tensor) -> lucid.Tensor:
    return F.mse_loss(p, t)


def test_manual_vjp_conv_transpose2d_parity() -> None:
    """ConvTranspose2d + ReLU + mean + Linear + MSE: ≤ 2e-3 over 3 SGD steps."""
    lucid.manual_seed(0)
    a = _ConvTransposeNet().to(COMPILE_DEVICE)
    b = _ConvTransposeNet().to(COMPILE_DEVICE)
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        with lucid.no_grad():
            pb.copy_(pa.detach().clone())

    x = metal_tensor(2, 4, 3, 3)
    t = metal_tensor(2, 1)

    opt_eager = optim.SGD(list(a.parameters()), lr=1e-2)
    eager: list[float] = []
    for _ in range(3):
        opt_eager.zero_grad()
        loss = _loss(a(x), t)
        loss.backward()
        opt_eager.step()
        eager.append(float(loss.item()))

    opt_comp = optim.SGD(list(b.parameters()), lr=1e-2)
    step = fused_step(b, _loss, opt_comp)
    comp: list[float] = []
    for _ in range(3):
        out = step(x, t)
        comp.append(float(out.item() if hasattr(out, "item") else out[0].item()))

    for k in range(3):
        diff = abs(eager[k] - comp[k])
        assert diff < 2e-3, (
            f"ConvTranspose2d VJP drift at step {k}: "
            f"eager={eager[k]:.6f}, compile={comp[k]:.6f}, diff={diff:.6f}"
        )


# ── Conv3d ──────────────────────────────────────────────────────────


class _Conv3dNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(2, 3, kernel_size=2, padding=0)
        self.fc = nn.Linear(3, 1)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = self.conv(x).relu()
        h = h.mean(dim=(2, 3, 4), keepdim=False)  # (B, 3)
        return self.fc(h)


def test_manual_vjp_conv3d_parity() -> None:
    """Conv3d + ReLU + mean + Linear + MSE: ≤ 2e-3 over 3 SGD steps."""
    lucid.manual_seed(0)
    a = _Conv3dNet().to(COMPILE_DEVICE)
    b = _Conv3dNet().to(COMPILE_DEVICE)
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        with lucid.no_grad():
            pb.copy_(pa.detach().clone())

    x = metal_tensor(2, 2, 3, 3, 3)
    t = metal_tensor(2, 1)

    opt_eager = optim.SGD(list(a.parameters()), lr=1e-2)
    eager: list[float] = []
    for _ in range(3):
        opt_eager.zero_grad()
        loss = _loss(a(x), t)
        loss.backward()
        opt_eager.step()
        eager.append(float(loss.item()))

    opt_comp = optim.SGD(list(b.parameters()), lr=1e-2)
    step = fused_step(b, _loss, opt_comp)
    comp: list[float] = []
    for _ in range(3):
        out = step(x, t)
        comp.append(float(out.item() if hasattr(out, "item") else out[0].item()))

    for k in range(3):
        diff = abs(eager[k] - comp[k])
        assert diff < 2e-3, (
            f"Conv3d VJP drift at step {k}: "
            f"eager={eager[k]:.6f}, compile={comp[k]:.6f}, diff={diff:.6f}"
        )


# ── ConvTranspose2d with grouping / dilation ────────────────────────
#
# The VJP builds a descriptor for the *forward* convolution this op is
# the data gradient of, so every geometry field has to come off the
# trace.  ``dilation`` was pinned to 1 there, which produced a
# well-formed gradient of a different convolution — the forward stayed
# right, so only a training-loop comparison shows it.


class _OptionNet(nn.Module):
    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(4, 4, kernel_size=2, stride=2, **kwargs)  # type: ignore[arg-type]
        self.fc = nn.Linear(4, 1)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = self.up(x).relu()
        h = h.mean(dim=(2, 3), keepdim=False)
        return self.fc(h)


def _step_and_compare(kwargs: dict[str, object], steps: int = 3) -> float:
    """Largest parameter disagreement after each SGD step.

    The loss is a poor probe for a wrong gradient — step 0 is identical
    whatever the backward does, and later steps only feel the error
    through a scalar.  Comparing the *parameters* reads the gradient
    almost directly, which is what makes this fail when the VJP builds a
    descriptor for a different convolution than the forward used.
    """
    lucid.manual_seed(0)
    a = _OptionNet(**kwargs).to(COMPILE_DEVICE)
    b = _OptionNet(**kwargs).to(COMPILE_DEVICE)
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        with lucid.no_grad():
            pb.copy_(pa.detach().clone())

    x = metal_tensor(2, 4, 3, 3)
    t = metal_tensor(2, 1)

    opt_eager = optim.SGD(list(a.parameters()), lr=1e-1)
    opt_comp = optim.SGD(list(b.parameters()), lr=1e-1)
    step = fused_step(b, _loss, opt_comp)

    worst = 0.0
    for _ in range(steps):
        opt_eager.zero_grad()
        _loss(a(x), t).backward()
        opt_eager.step()
        step(x, t)
        for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
            scale = max(float(pa.abs().max().item()), 1e-3)
            worst = max(worst, float((pa - pb).abs().max().item()) / scale)
    return worst


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("grouped", {"groups": 2}),
        ("depthwise", {"groups": 4}),
        ("dilated", {"dilation": 2}),
        ("grouped and dilated", {"groups": 2, "dilation": 2}),
    ],
    ids=["grouped", "depthwise", "dilated", "grouped-dilated"],
)
def test_manual_vjp_conv_transpose2d_options_parity(name: str, kwargs: dict[str, object]) -> None:
    worst = _step_and_compare(kwargs)
    assert worst < 1e-4, (
        f"ConvTranspose2d ({name}) VJP disagrees with eager: "
        f"parameters diverge by {worst:.2e} relative after an SGD step"
    )
