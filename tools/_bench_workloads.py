"""Workload registry for ``tools/bench_compile_vs_eager.py``.

Each :class:`Workload` exposes a uniform ``(mk_model, mk_input,
mk_target, loss_fn)`` interface so the runner can sweep
eager / compile / fused_step paths without per-workload branching.

The 5 workloads picked here cover the axes that matter for the
compile-vs-eager perf question:

  * **mlp** — simplest, validates baseline cache + dispatch overhead.
  * **cifar_resnet_block** — Conv + BN + ReLU + skip; isolates the
    largest known Lucid eager-vs-ref gap (Conv backward 1.53×).
  * **resnet18_train** — canonical compile acceptance gate (already
    used in retros); 1 step = forward + CE + SGD.
  * **gpt2_block** — LN + Linear + attention + Linear + LN; transformer
    ops are already at ref-parity in eager, so this isolates the
    *compile* delta cleanly.
  * **wide_mlp_dropout** — exercises the Option-A Phase 1 stateful
    Philox dropout path on a workload large enough that the per-step
    state-rotation isn't lost in dispatch noise.

The registry is intentionally **small** — 5 workloads × 2 precisions
× 3 batch sizes × 3 paths is already ~90 measurements per trial.
"""

import dataclasses
from typing import Callable

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

# ── Workload record ─────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Workload:
    """Single benchmark workload — factories exposed to the runner.

    All factories return *fresh* objects on every call so eager /
    compile / fused_step rows start from an identical fixture state.

    Attributes
    ----------
    name : str
        Identifier used in the CSV / markdown output (lowercase, no
        spaces).
    mk_model : Callable[[], nn.Module]
        Construct the model.  Caller moves it to ``"metal"`` and
        switches to eval / train as needed.
    mk_input : Callable[[int], tuple[lucid.Tensor, ...]]
        Build the positional input tuple for ``model(*inputs)``.
        Parameterised by batch size so the runner can sweep ``BS``.
    mk_target : Callable[[int], lucid.Tensor] | None
        Build the target tensor for ``loss_fn(out, target)``.  ``None``
        for forward-only workloads (which then skip the fused_step
        path).
    loss_fn : Callable[..., lucid.Tensor] | None
        Scalar loss for fused_step path.  ``None`` for forward-only.
    supports_amp : bool
        When ``False`` the runner skips the AMP F16 row for this
        workload — flag exists so the LSTM-style workloads (where
        F16 is known fragile) can opt out without aborting the sweep.
    """

    name: str
    mk_model: Callable[[], nn.Module]
    mk_input: Callable[[int], tuple[lucid.Tensor, ...]]
    mk_target: Callable[[int], lucid.Tensor] | None
    loss_fn: Callable[..., lucid.Tensor] | None
    supports_amp: bool


# ── Workload 1: MLP ─────────────────────────────────────────────────


class _MLP(nn.Module):
    """3-layer MLP: 64 → 128 → 64 → 10.  Forward + loss only."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc3(self.fc2(self.fc1(x).relu()).relu())


def _mlp_input(bs: int) -> tuple[lucid.Tensor, ...]:
    return (lucid.randn(bs, 64),)


def _mlp_target(bs: int) -> lucid.Tensor:
    return lucid.randn(bs, 10)


# ── Workload 2: CIFAR-style ResNet block ────────────────────────────


class _CifarResNetBlock(nn.Module):
    """Conv → BN → ReLU → Conv → BN + skip, 64 channels at 32×32.

    Mimics one residual block from a CIFAR-style ResNet (no
    downsampling, no projection).  Isolates Conv2d + BN train mode
    + skip-connection sums — the configuration where Lucid eager
    sits 1.53× behind reference framework eager.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        identity = x
        out = self.bn1(self.conv1(x)).relu()
        out = self.bn2(self.conv2(out))
        return (out + identity).relu()


def _cifar_block_input(bs: int) -> tuple[lucid.Tensor, ...]:
    return (lucid.randn(bs, 64, 32, 32),)


def _cifar_block_target(bs: int) -> lucid.Tensor:
    # Match output shape for MSE.
    return lucid.randn(bs, 64, 32, 32)


# ── Workload 3: ResNet-18 + CE + SGD train step ─────────────────────


class _UnwrappingResNet18(nn.Module):
    """ResNet-18 classifier with the ``ImageClassificationOutput``
    dataclass unwrapped to a raw ``[BS, num_classes]`` logits Tensor.

    The model-zoo task wrappers return dataclasses for API stability,
    but ``fused_step`` and the benchmark loss application both expect
    raw Tensors.  This adapter unwraps in ``forward`` so both code
    paths can consume the model uniformly.
    """

    def __init__(self) -> None:
        super().__init__()
        import lucid.models as M  # local to avoid heavy import at module load

        self.inner = M.resnet_18_cls(num_classes=10)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        out = self.inner(x)
        # ImageClassificationOutput exposes ``.logits`` by zoo convention.
        return out.logits


def _resnet18_factory() -> nn.Module:
    """Real ResNet-18 image classifier from the model zoo, with the
    output dataclass unwrapped to a raw logits Tensor so the benchmark
    runner doesn't need per-workload unwrap branches.
    """
    return _UnwrappingResNet18()


def _resnet18_input(bs: int) -> tuple[lucid.Tensor, ...]:
    # Standard CIFAR/ImageNet-style RGB tensor; bs varies, channels +
    # spatial fixed.  224×224 is the canonical ResNet input size used
    # in every prior compile-path acceptance gate.
    return (lucid.randn(bs, 3, 224, 224),)


def _resnet18_target(bs: int) -> lucid.Tensor:
    # CE expects integer class indices in [0, num_classes).
    return lucid.randint(0, 10, (bs,), dtype=lucid.int64)


# ── Workload 4: GPT-2-base transformer block ────────────────────────


class _GPT2Block(nn.Module):
    """Single transformer block: LN → MHA → residual → LN → MLP → residual.

    Mirrors GPT-2-base's hidden size (768) and 12 attention heads.
    Sequence length is fixed at 128 — long enough to exercise the
    softmax + matmul path but short enough that one block runs in
    sub-millisecond on M-series GPUs.
    """

    def __init__(self) -> None:
        super().__init__()
        d_model = 768
        n_heads = 12
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        # Pre-norm transformer block.  MHA returns (out, weights) — we
        # need only the activations.
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + a
        return x + self.mlp(self.ln2(x))


def _gpt2_block_input(bs: int) -> tuple[lucid.Tensor, ...]:
    # (BS, seq=128, d_model=768)
    return (lucid.randn(bs, 128, 768),)


def _gpt2_block_target(bs: int) -> lucid.Tensor:
    return lucid.randn(bs, 128, 768)


# ── Workload 5: Wide MLP + Dropout (stateful Philox path) ──────────


class _WideMLPDropout(nn.Module):
    """Wide MLP with Dropout(p=0.5) in the middle.

    Exists specifically to exercise the Phase 1 stateful Philox
    plumbing.  Hidden size 1024 is large enough that the per-call
    state advancement isn't lost in dispatch noise but small enough
    that 100 calls finish in a few seconds.
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(256, 1024)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = self.fc1(x).relu()
        h = self.drop(h)
        h = self.fc2(h).relu()
        return self.fc3(h)


def _wide_mlp_input(bs: int) -> tuple[lucid.Tensor, ...]:
    return (lucid.randn(bs, 256),)


def _wide_mlp_target(bs: int) -> lucid.Tensor:
    return lucid.randn(bs, 10)


# ── Registry ────────────────────────────────────────────────────────


WORKLOADS: list[Workload] = [
    Workload(
        name="mlp",
        mk_model=_MLP,
        mk_input=_mlp_input,
        mk_target=_mlp_target,
        loss_fn=F.mse_loss,
        supports_amp=True,
    ),
    Workload(
        name="cifar_resnet_block",
        mk_model=_CifarResNetBlock,
        mk_input=_cifar_block_input,
        mk_target=_cifar_block_target,
        loss_fn=F.mse_loss,
        supports_amp=True,
    ),
    Workload(
        name="resnet18_train",
        mk_model=_resnet18_factory,
        mk_input=_resnet18_input,
        mk_target=_resnet18_target,
        loss_fn=F.cross_entropy,
        supports_amp=True,
    ),
    Workload(
        name="gpt2_block",
        mk_model=_GPT2Block,
        mk_input=_gpt2_block_input,
        mk_target=_gpt2_block_target,
        loss_fn=F.mse_loss,
        supports_amp=True,
    ),
    Workload(
        name="wide_mlp_dropout",
        mk_model=_WideMLPDropout,
        mk_input=_wide_mlp_input,
        mk_target=_wide_mlp_target,
        loss_fn=F.mse_loss,
        supports_amp=True,
    ),
]


def get(name: str) -> Workload:
    """Look up a workload by name; raises ``KeyError`` if missing."""
    for w in WORKLOADS:
        if w.name == name:
            return w
    available = ", ".join(w.name for w in WORKLOADS)
    raise KeyError(f"unknown workload {name!r}; available: {available}")
