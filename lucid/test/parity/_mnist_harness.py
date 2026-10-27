"""Shared machinery for the MNIST parity modules.

Holds everything the training-parity and AMP/quantization tests both need:
the dataset, the paired model builders, state transfer, and the two training
loops.  Kept out of the test modules so the two do not drift apart — the whole
value of these tests rests on both frameworks receiving *identical* setup, and
that guarantee is easiest to keep in one place.

Each model here exists in two forms that must stay layer-for-layer identical,
because :func:`copy_state` matches parameters positionally.  Extending a model
means extending both halves in the same order.
"""

import math
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.models as M
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim

# Subset sizes: large enough to learn a real decision boundary, small enough
# that two full training runs stay inside a test-suite time budget.
N_TRAIN = 6_000
N_TEST = 2_000
BATCH = 64
EPOCHS = 5
SEED = 0

# Per-optimizer settings, identical on both sides.  Adam and AdamW run at
# their own natural rate rather than SGD's — 0.05 would diverge.
SGD_LR = 0.05
MOMENTUM = 0.9
ADAM_LR = 1e-3

# Steps over which the two runs are still provably on the same trajectory.
# Systematic defects show up here; chaotic drift has not accumulated yet.
EARLY_STEPS = 30

# MNIST's own normalisation constants.
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

# Transformer case dimensions — small enough to train in seconds, wide enough
# that four attention heads are meaningful.
EMBED_DIM = 32
NUM_HEADS = 4
PATCH = 8  # 32x32 → a 4x4 grid of patches → 16 tokens


# ── data ────────────────────────────────────────────────────────────────────


def load_mnist(ref_vision: ModuleType, root: Path) -> tuple[np.ndarray, ...]:
    """Fetch MNIST and return it as 32x32 float32 arrays.

    LeNet-5 expects 32x32; MNIST is 28x28, so the canonical treatment (and
    the paper's) is to pad by 2 on every side rather than resample.
    """
    try:
        train = ref_vision.datasets.MNIST(str(root), train=True, download=True)
        test = ref_vision.datasets.MNIST(str(root), train=False, download=True)
    except Exception as exc:  # offline, or the mirror is down
        pytest.skip(f"MNIST unavailable ({type(exc).__name__}: {exc})")

    def _prep(ds: Any, n: int) -> tuple[np.ndarray, np.ndarray]:
        x = ds.data.numpy()[:n].astype(np.float32) / 255.0
        x = (x - MNIST_MEAN) / MNIST_STD
        x = np.pad(x, ((0, 0), (2, 2), (2, 2)))  # 28x28 → 32x32
        return x[:, None, :, :], ds.targets.numpy()[:n].astype(np.int64)

    x_tr, y_tr = _prep(train, N_TRAIN)
    x_te, y_te = _prep(test, N_TEST)
    return x_tr, y_tr, x_te, y_te


def batches(n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """One epoch of shuffled index batches.  Both frameworks consume the
    identical list, so batch composition and order are never a variable."""
    order = rng.permutation(n)
    return [order[i : i + BATCH] for i in range(0, n, BATCH)]


# ── models: LeNet-5 ─────────────────────────────────────────────────────────


def build_ref_lenet(ref: ModuleType) -> Any:
    """Mirror ``lucid.models.lenet_5_cls`` layer for layer.

    Written out rather than imported because the reference vision package
    ships no LeNet.  The ordering here must match Lucid's parameter order
    (features.0/3/6, f6, classifier) — ``copy_state`` zips the two.
    """
    nn_ref = ref.nn

    class RefLeNet(nn_ref.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.features = nn_ref.Sequential(
                nn_ref.Conv2d(1, 6, 5),
                nn_ref.Tanh(),
                nn_ref.AvgPool2d(2, stride=2),
                nn_ref.Conv2d(6, 16, 5),
                nn_ref.Tanh(),
                nn_ref.AvgPool2d(2, stride=2),
                nn_ref.Conv2d(16, 120, 5),
                nn_ref.Tanh(),
            )
            self.f6 = nn_ref.Linear(120, 84)
            self.act_f6 = nn_ref.Tanh()
            self.classifier = nn_ref.Linear(84, 10)

        def forward(self, x: Any) -> Any:
            h = self.features(x).flatten(1)
            return self.classifier(self.act_f6(self.f6(h)))

    return RefLeNet()


# ── models: BatchNorm / ReLU / MaxPool ──────────────────────────────────────


def build_lucid_bn_cnn() -> nn.Module:
    """A conv / BatchNorm / ReLU / MaxPool stack — the block LeNet lacks.

    Deliberately small (~11k parameters); the point is covering BatchNorm's
    running statistics and the ReLU/MaxPool pair, not a headline accuracy.
    """
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 32 → 16
        nn.Conv2d(8, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2),  # 16 → 8
        nn.Flatten(),
        nn.Linear(16 * 8 * 8, 10),
    )


def build_ref_bn_cnn(ref: ModuleType) -> Any:
    nn_ref = ref.nn
    return nn_ref.Sequential(
        nn_ref.Conv2d(1, 8, 3, padding=1),
        nn_ref.BatchNorm2d(8),
        nn_ref.ReLU(),
        nn_ref.MaxPool2d(2),
        nn_ref.Conv2d(8, 16, 3, padding=1),
        nn_ref.BatchNorm2d(16),
        nn_ref.ReLU(),
        nn_ref.MaxPool2d(2),
        nn_ref.Flatten(),
        nn_ref.Linear(16 * 8 * 8, 10),
    )


# ── models: residual ────────────────────────────────────────────────────────


class _LucidResBlock(nn.Module):
    """Post-activation residual block, the ResNet-v1 arrangement.

    The skip is the point: its backward adds the upstream gradient to the
    branch gradient, so a wrong accumulation here shows up as a model that
    trains but underperforms — the residual path silently contributing
    nothing, which is indistinguishable from a plain stack.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + x)


class _LucidResNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1, 8, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8)
        self.block1 = _LucidResBlock(8)
        self.block2 = _LucidResBlock(8)
        self.head = nn.Linear(8 * 8 * 8, 10)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = F.relu(self.bn(self.stem(x)))
        h = F.avg_pool2d(self.block1(h), 2)  # 32 → 16
        h = F.avg_pool2d(self.block2(h), 2)  # 16 → 8
        return self.head(h.reshape(h.shape[0], -1))


def build_lucid_resnet() -> nn.Module:
    return _LucidResNet()


def build_ref_resnet(ref: ModuleType) -> Any:
    nn_ref = ref.nn
    f_ref = ref.nn.functional

    class RefResBlock(nn_ref.Module):  # type: ignore[misc, name-defined]
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.conv1 = nn_ref.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn1 = nn_ref.BatchNorm2d(channels)
            self.conv2 = nn_ref.Conv2d(channels, channels, 3, padding=1, bias=False)
            self.bn2 = nn_ref.BatchNorm2d(channels)

        def forward(self, x: Any) -> Any:
            h = f_ref.relu(self.bn1(self.conv1(x)))
            h = self.bn2(self.conv2(h))
            return f_ref.relu(h + x)

    class RefResNet(nn_ref.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn_ref.Conv2d(1, 8, 3, padding=1, bias=False)
            self.bn = nn_ref.BatchNorm2d(8)
            self.block1 = RefResBlock(8)
            self.block2 = RefResBlock(8)
            self.head = nn_ref.Linear(8 * 8 * 8, 10)

        def forward(self, x: Any) -> Any:
            h = f_ref.relu(self.bn(self.stem(x)))
            h = f_ref.avg_pool2d(self.block1(h), 2)
            h = f_ref.avg_pool2d(self.block2(h), 2)
            return self.head(h.reshape(h.shape[0], -1))

    return RefResNet()


# ── models: attention + LayerNorm ───────────────────────────────────────────


class _LucidViT(nn.Module):
    """A single pre-norm transformer block over 16 image patches.

    Covers three things at once that the convolutional cases do not touch:
    multi-head attention (softmax over a score matrix, and its backward),
    LayerNorm (per-token statistics computed inside the graph, unlike
    BatchNorm's running buffers), and the residual around each sub-layer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.patch = nn.Conv2d(1, EMBED_DIM, PATCH, stride=PATCH)
        self.pos = nn.Parameter(lucid.zeros((1, (32 // PATCH) ** 2, EMBED_DIM)))
        self.ln1 = nn.LayerNorm(EMBED_DIM)
        self.attn = nn.MultiheadAttention(EMBED_DIM, NUM_HEADS, batch_first=True)
        self.ln2 = nn.LayerNorm(EMBED_DIM)
        self.fc1 = nn.Linear(EMBED_DIM, 2 * EMBED_DIM)
        self.fc2 = nn.Linear(2 * EMBED_DIM, EMBED_DIM)
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, 10)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        h = self.patch(x)  # (B, D, 4, 4)
        h = h.reshape(h.shape[0], EMBED_DIM, -1).permute(0, 2, 1)
        h = h + self.pos
        a = self.ln1(h)
        h = h + self.attn(a, a, a)[0]
        m = self.ln2(h)
        h = h + self.fc2(F.gelu(self.fc1(m)))
        return self.head(self.ln_f(h).mean(dim=1))


def build_lucid_vit() -> nn.Module:
    return _LucidViT()


def build_ref_vit(ref: ModuleType) -> Any:
    nn_ref = ref.nn
    f_ref = ref.nn.functional

    class RefViT(nn_ref.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.patch = nn_ref.Conv2d(1, EMBED_DIM, PATCH, stride=PATCH)
            self.pos = nn_ref.Parameter(ref.zeros(1, (32 // PATCH) ** 2, EMBED_DIM))
            self.ln1 = nn_ref.LayerNorm(EMBED_DIM)
            self.attn = nn_ref.MultiheadAttention(EMBED_DIM, NUM_HEADS, batch_first=True)
            self.ln2 = nn_ref.LayerNorm(EMBED_DIM)
            self.fc1 = nn_ref.Linear(EMBED_DIM, 2 * EMBED_DIM)
            self.fc2 = nn_ref.Linear(2 * EMBED_DIM, EMBED_DIM)
            self.ln_f = nn_ref.LayerNorm(EMBED_DIM)
            self.head = nn_ref.Linear(EMBED_DIM, 10)

        def forward(self, x: Any) -> Any:
            h = self.patch(x)
            h = h.reshape(h.shape[0], EMBED_DIM, -1).transpose(1, 2)
            h = h + self.pos
            a = self.ln1(h)
            h = h + self.attn(a, a, a, need_weights=False)[0]
            m = self.ln2(h)
            h = h + self.fc2(f_ref.gelu(self.fc1(m)))
            return self.head(self.ln_f(h).mean(dim=1))

    return RefViT()


# ── shared plumbing ─────────────────────────────────────────────────────────


def logits(out: Any) -> Any:
    """Zoo models return an output dataclass; the hand-built ones return the
    tensor itself."""
    return out.logits if hasattr(out, "logits") else out


def copy_state(src: Any, dst: Any, ref: ModuleType) -> None:
    """Copy Lucid's initialised parameters *and* buffers into the reference.

    This is what makes the comparison meaningful: without it the two models
    start from different points and any later difference is unattributable.
    Layouts match exactly — conv is (out, in, kH, kW), linear is (out, in) and
    packed attention projections are (3*embed, embed) on both sides — so no
    transpose is involved, and both frameworks list direct parameters before
    submodule ones, so the positional zip is stable.
    """
    for kind, s_items, d_items in (
        ("parameter", list(src.named_parameters()), list(dst.named_parameters())),
        ("buffer", list(src.named_buffers()), list(dst.named_buffers())),
    ):
        assert len(s_items) == len(d_items), (
            f"{kind} count differs: lucid {len(s_items)} vs reference "
            f"{len(d_items)} — the architectures have drifted apart"
        )
        with ref.no_grad():
            for (s_name, s_t), (d_name, d_t) in zip(s_items, d_items):
                s_arr = s_t.numpy()
                # ``ascontiguousarray`` returns at least 1-d, so it silently
                # turns BatchNorm's 0-d ``num_batches_tracked`` into shape
                # (1,).  Reshape back to what Lucid actually reported.
                s_arr = np.ascontiguousarray(s_arr).reshape(s_arr.shape)
                assert s_arr.shape == tuple(d_t.shape), (
                    f"shape mismatch at {kind} {s_name} / {d_name}: "
                    f"{s_arr.shape} vs {tuple(d_t.shape)}"
                )
                d_t.copy_(ref.from_numpy(s_arr))


def lucid_opt(name: str, params: Any) -> Any:
    if name == "sgd":
        return optim.SGD(params, lr=SGD_LR, momentum=MOMENTUM)
    if name == "adam":
        return optim.Adam(params, lr=ADAM_LR)
    return optim.AdamW(params, lr=ADAM_LR)


def ref_opt(name: str, params: Any, ref: ModuleType) -> Any:
    if name == "sgd":
        return ref.optim.SGD(params, lr=SGD_LR, momentum=MOMENTUM)
    if name == "adam":
        return ref.optim.Adam(params, lr=ADAM_LR)
    return ref.optim.AdamW(params, lr=ADAM_LR)


_LUCID_BUILDERS = {
    "lenet": lambda: M.lenet_5_cls(),
    "bn_cnn": build_lucid_bn_cnn,
    "resnet": build_lucid_resnet,
    "vit": build_lucid_vit,
}


def build_pair(kind: str, device: str, ref: ModuleType) -> tuple[Any, Any]:
    """Build the Lucid model and its reference twin, sharing Lucid's init."""
    lucid.manual_seed(SEED)
    lucid_model = _LUCID_BUILDERS[kind]().to(device)
    ref_model = {
        "lenet": lambda: build_ref_lenet(ref),
        "bn_cnn": lambda: build_ref_bn_cnn(ref),
        "resnet": lambda: build_ref_resnet(ref),
        "vit": lambda: build_ref_vit(ref),
    }[kind]()
    copy_state(lucid_model, ref_model, ref)
    return lucid_model, ref_model


# ── training loops ──────────────────────────────────────────────────────────


def train_lucid(
    model: Any,
    data: tuple[np.ndarray, ...],
    schedule: list[list[np.ndarray]],
    device: str,
    opt_name: str,
    make_sched: Any = None,
) -> tuple[list[float], list[float], float, list[float]]:
    """Train and report (epoch means, per-step losses, accuracy, lr trace)."""
    x_tr, y_tr, x_te, y_te = data
    opt = lucid_opt(opt_name, model.parameters())
    sched = make_sched(opt) if make_sched is not None else None

    epoch_losses: list[float] = []
    step_losses: list[float] = []
    lrs: list[float] = []
    for epoch_batches in schedule:
        losses = []
        for idx in epoch_batches:
            xb = lucid.tensor(x_tr[idx], device=device)
            yb = lucid.tensor(y_tr[idx], dtype=lucid.int64, device=device)

            lrs.append(float(opt.param_groups[0]["lr"]))
            opt.zero_grad()
            loss = F.cross_entropy(logits(model(xb)), yb)
            loss.backward()
            opt.step()

            value = float(loss.item())
            assert math.isfinite(
                value
            ), f"lucid loss became {value} — training is numerically unstable"
            losses.append(value)
            step_losses.append(value)
        if sched is not None:
            sched.step()
        epoch_losses.append(float(np.mean(losses)))

    # Accuracy in eval mode, batched so a 2,000-image forward does not spike
    # memory on either device.  Eval mode is also what puts BatchNorm on its
    # running statistics rather than the batch's own.
    model.eval()
    correct = 0
    with lucid.no_grad():
        for i in range(0, len(y_te), 256):
            xb = lucid.tensor(x_te[i : i + 256], device=device)
            pred = logits(model(xb)).numpy().argmax(axis=1)
            correct += int((pred == y_te[i : i + 256]).sum())
    model.train()
    return epoch_losses, step_losses, correct / len(y_te), lrs


def train_ref(
    model: Any,
    data: tuple[np.ndarray, ...],
    schedule: list[list[np.ndarray]],
    ref: ModuleType,
    opt_name: str,
    make_sched: Any = None,
) -> tuple[list[float], list[float], float, list[float]]:
    x_tr, y_tr, x_te, y_te = data
    opt = ref_opt(opt_name, model.parameters(), ref)
    sched = make_sched(opt) if make_sched is not None else None

    epoch_losses: list[float] = []
    step_losses: list[float] = []
    lrs: list[float] = []
    for epoch_batches in schedule:
        losses = []
        for idx in epoch_batches:
            xb = ref.from_numpy(x_tr[idx])
            yb = ref.from_numpy(y_tr[idx])

            lrs.append(float(opt.param_groups[0]["lr"]))
            opt.zero_grad()
            loss = ref.nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()

            value = float(loss.item())
            losses.append(value)
            step_losses.append(value)
        if sched is not None:
            sched.step()
        epoch_losses.append(float(np.mean(losses)))

    model.eval()
    correct = 0
    with ref.no_grad():
        for i in range(0, len(y_te), 256):
            pred = model(ref.from_numpy(x_te[i : i + 256])).numpy().argmax(axis=1)
            correct += int((pred == y_te[i : i + 256]).sum())
    model.train()
    return epoch_losses, step_losses, correct / len(y_te), lrs
