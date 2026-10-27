"""Thousands of training steps in Lucid and in the reference, side by side.

The MNIST parity runs about 470 steps.  A defect that only shows after a few
thousand — momentum or a running statistic that drifts, a schedule that is
off by one step, weight decay applied twice — passes it.  These runs are
long enough for that, on models built from the parts real training uses:
residual blocks with BatchNorm under SGD with momentum, weight decay and a
cosine schedule; and a small GPT with causal attention under AdamW with
warmup, cosine decay and gradient clipping.

Each model is written once and built by each framework, so the two are
layer-for-layer identical by construction; :func:`copy_state` then gives the
reference Lucid's initial weights.  The data is generated from a seed rather
than downloaded — a missing mirror would make the nightly skip, and a skip
there is a failure — and both frameworks consume the same batches in the
same order.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.test.parity._mnist_harness import copy_state

SEED = 0

# ── data ──────────────────────────────────────────────────────────────────────

N_TEST_IMAGES = 2_048
IMAGE_BATCH = 64
# Heavy enough that the classes overlap: the loss settles well above zero,
# where a difference between two runs is a difference in the numbers rather
# than in which near-zero spike landed where.
IMAGE_NOISE = 1.2

VOCAB = 32
CONTEXT = 64
TEXT_BATCH = 32
N_TRAIN_TOKENS = 400_000
N_EVAL_WINDOWS = 256


def shapes_dataset(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``n`` 3x32x32 images of ten classes: five shapes, each small or large.

    Position, size within the class, colour and background noise vary per
    image, so the task needs convolution to generalise and never saturates
    at the first epoch.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:32, 0:32].astype(np.float32)
    labels = rng.integers(0, 10, n)
    kind, large = labels % 5, labels // 5
    cx = rng.uniform(10.0, 22.0, n).astype(np.float32)[:, None, None]
    cy = rng.uniform(10.0, 22.0, n).astype(np.float32)[:, None, None]
    radius = np.where(large == 1, rng.uniform(7.0, 9.0, n), rng.uniform(3.5, 5.5, n))
    r = radius.astype(np.float32)[:, None, None]
    dx, dy = xx[None] - cx, yy[None] - cy
    d2 = dx**2 + dy**2
    shapes = [
        d2 <= r**2,  # disc
        (np.abs(dx) <= r) & (np.abs(dy) <= r),  # square
        (np.abs(dy) <= r) & (np.abs(dx) <= (dy + r) / 2),  # triangle
        ((np.abs(dx) <= r / 3) & (np.abs(dy) <= r))
        | ((np.abs(dy) <= r / 3) & (np.abs(dx) <= r)),  # cross
        (d2 <= r**2) & (d2 >= (0.6 * r) ** 2),  # ring
    ]
    mask = np.zeros((n, 32, 32), dtype=bool)
    for k, shape in enumerate(shapes):
        mask[kind == k] = shape[kind == k]
    colour = rng.uniform(0.5, 1.0, (n, 3, 1, 1)).astype(np.float32)
    noise = rng.normal(0.0, IMAGE_NOISE, (n, 3, 32, 32)).astype(np.float32)
    images = noise + mask[:, None].astype(np.float32) * colour
    # Fixed constants, not the batch's own statistics: every batch is drawn
    # fresh, and each must be scaled the same way.
    images = (images - 0.08) / IMAGE_NOISE
    return images.astype(np.float32), labels.astype(np.int64)


def image_batch(step: int) -> tuple[np.ndarray, np.ndarray]:
    """The training batch for ``step``: drawn fresh, never repeated.

    A fixed training set is memorised within a few thousand steps and its
    loss sinks to zero; fresh images keep the loss at the task's own floor
    for the whole run.  Seeded by the step, so both frameworks see it.
    """
    return shapes_dataset(IMAGE_BATCH, 1_000_000 + step)


@dataclass(frozen=True)
class Corpus:
    train: np.ndarray
    eval_windows: np.ndarray
    entropy: float  # the best achievable loss, in nats per token


def markov_corpus(seed: int) -> Corpus:
    """A stream from a random order-2 Markov chain over :data:`VOCAB` tokens.

    Each two-token context favours a few successors (a sparse Dirichlet), so
    a model has to learn the table from context — and the table's own
    conditional entropy is the floor its loss can reach, which says whether
    both runs actually learned rather than merely agreed.
    """
    rng = np.random.default_rng(seed)
    table = rng.dirichlet(np.full(VOCAB, 0.08), size=(VOCAB, VOCAB))
    cumulative = np.cumsum(table, axis=-1)

    def stream(n: int, stream_rng: np.random.Generator) -> np.ndarray:
        out = np.empty(n, dtype=np.int64)
        out[:2] = stream_rng.integers(0, VOCAB, 2)
        u = stream_rng.random(n)
        for i in range(2, n):
            row = cumulative[out[i - 2], out[i - 1]]
            out[i] = min(int(np.searchsorted(row, u[i])), VOCAB - 1)
        return out

    train = stream(N_TRAIN_TOKENS, np.random.default_rng(seed + 1))
    held = stream(N_EVAL_WINDOWS * (CONTEXT + 1), np.random.default_rng(seed + 2))
    windows = held.reshape(N_EVAL_WINDOWS, CONTEXT + 1)
    p = table.reshape(-1, VOCAB)
    entropy = float(-(p * np.log(np.clip(p, 1e-12, None))).sum(axis=-1).mean())
    return Corpus(train, windows, entropy)


def text_batches(corpus: Corpus, steps: int, seed: int) -> np.ndarray:
    """Start offsets for every step, drawn once and shared by both runs."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, len(corpus.train) - CONTEXT - 1, (steps, TEXT_BATCH))


def window(corpus: Corpus, starts: np.ndarray) -> np.ndarray:
    return np.stack([corpus.train[s : s + CONTEXT + 1] for s in starts])


# ── models, written once for both frameworks ──────────────────────────────────


def make_resnet(lib: ModuleType, nn_: ModuleType, F_: ModuleType) -> Any:
    """Three stages of basic residual blocks (16/32/64 channels), ~0.1M params."""

    class Block(nn_.Module):  # type: ignore[misc, name-defined]
        def __init__(self, cin: int, cout: int, stride: int) -> None:
            super().__init__()
            self.conv1 = nn_.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
            self.bn1 = nn_.BatchNorm2d(cout)
            self.conv2 = nn_.Conv2d(cout, cout, 3, padding=1, bias=False)
            self.bn2 = nn_.BatchNorm2d(cout)
            if stride == 1 and cin == cout:
                self.shortcut = nn_.Identity()
            else:
                self.shortcut = nn_.Sequential(
                    nn_.Conv2d(cin, cout, 1, stride=stride, bias=False),
                    nn_.BatchNorm2d(cout),
                )

        def forward(self, x: Any) -> Any:
            h = F_.relu(self.bn1(self.conv1(x)))
            h = self.bn2(self.conv2(h))
            return F_.relu(h + self.shortcut(x))

    class ResNet(nn_.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn_.Conv2d(3, 16, 3, padding=1, bias=False)
            self.bn = nn_.BatchNorm2d(16)
            self.stage1 = Block(16, 16, 1)
            self.stage2 = Block(16, 32, 2)
            self.stage3 = Block(32, 64, 2)
            self.head = nn_.Linear(64, 10)

        def forward(self, x: Any) -> Any:
            h = F_.relu(self.bn(self.stem(x)))
            h = self.stage3(self.stage2(self.stage1(h)))
            return self.head(h.mean(dim=(2, 3)))

    return ResNet()


def make_gpt(lib: ModuleType, nn_: ModuleType, F_: ModuleType) -> Any:
    """Two pre-norm blocks, width 64, four heads, causal attention."""
    width, heads, depth = 64, 4, 2

    class Block(nn_.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.ln1 = nn_.LayerNorm(width)
            self.qkv = nn_.Linear(width, 3 * width)
            self.proj = nn_.Linear(width, width)
            self.ln2 = nn_.LayerNorm(width)
            self.fc1 = nn_.Linear(width, 4 * width)
            self.fc2 = nn_.Linear(4 * width, width)

        def forward(self, x: Any) -> Any:
            b, t, c = x.shape
            qkv = self.qkv(self.ln1(x)).reshape(b, t, 3, heads, c // heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            att = F_.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)
            x = x + self.proj(att.permute(0, 2, 1, 3).reshape(b, t, c))
            return x + self.fc2(F_.gelu(self.fc1(self.ln2(x))))

    class GPT(nn_.Module):  # type: ignore[misc, name-defined]
        def __init__(self) -> None:
            super().__init__()
            self.pos = nn_.Parameter(lib.randn(1, CONTEXT, width) * 0.02)
            self.tok = nn_.Embedding(VOCAB, width)
            self.blocks = nn_.Sequential(*[Block() for _ in range(depth)])
            self.ln_f = nn_.LayerNorm(width)
            self.head = nn_.Linear(width, VOCAB, bias=False)

        def forward(self, idx: Any) -> Any:
            h = self.tok(idx) + self.pos[:, : idx.shape[1]]
            return self.head(self.ln_f(self.blocks(h)))

    return GPT()


# ── the runs ──────────────────────────────────────────────────────────────────


@dataclass
class Run:
    losses: list[float]
    metric: float  # test accuracy for images, held-out loss for text


@dataclass(frozen=True)
class Recipe:
    """One long run: the model, its optimiser and schedule, and its data."""

    name: str
    steps: int
    make_model: Callable[[ModuleType, ModuleType, ModuleType], Any]
    make_optimizer: Callable[[ModuleType, Any], Any]
    make_schedule: Callable[[ModuleType, Any], Any]
    clip: float | None


def _warmup_cosine(steps: int, warmup: int) -> Callable[[int], float]:
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


RESNET = Recipe(
    name="resnet_sgd_cosine",
    steps=3_072,
    make_model=make_resnet,
    make_optimizer=lambda o, p: o.SGD(p, lr=0.05, momentum=0.9, weight_decay=5e-4),
    make_schedule=lambda o, opt: o.lr_scheduler.CosineAnnealingLR(opt, T_max=3_072),
    clip=None,
)

GPT = Recipe(
    name="gpt_adamw_warmup",
    steps=3_000,
    make_model=make_gpt,
    make_optimizer=lambda o, p: o.AdamW(p, lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1),
    make_schedule=lambda o, opt: o.lr_scheduler.LambdaLR(opt, _warmup_cosine(3_000, 100)),
    clip=1.0,
)


def nudge(model: Any, seed: int, scale: float = 1e-6) -> None:
    """Move every weight by a relative ``scale`` — float rounding, not a change.

    A run started from here is what "the same training" looks like once
    rounding has been given room to compound.  Two such runs measure how far
    chaos alone carries a chaotic model; Lucid against the reference is held
    to that, not to a number picked by hand.
    """
    rng = np.random.default_rng(seed)
    with lucid.no_grad():
        for p in model.parameters():
            values = p.numpy()
            noise = rng.standard_normal(values.shape).astype(np.float32)
            p.copy_(lucid.tensor(values * (1.0 + scale * noise), device=p.device))


def build_pair(recipe: Recipe, device: str, ref: ModuleType) -> tuple[Any, Any]:
    """Lucid's model on ``device`` and the reference's, holding Lucid's init."""
    lucid.manual_seed(SEED)
    mine = recipe.make_model(lucid, nn, F)
    theirs = recipe.make_model(ref, ref.nn, ref.nn.functional)
    copy_state(mine, theirs, ref)
    return mine.to(device), theirs


def run_images(recipe: Recipe, framework: str, model: Any, device: str, ref: ModuleType) -> Run:
    x_te, y_te = shapes_dataset(N_TEST_IMAGES, SEED + 100)
    lib, o, fn = (lucid, optim, F) if framework == "lucid" else (ref, ref.optim, ref.nn.functional)
    put = (lambda a: lucid.tensor(a, device=device)) if framework == "lucid" else ref.from_numpy
    opt = recipe.make_optimizer(o, model.parameters())
    sched = recipe.make_schedule(o, opt)
    losses: list[float] = []
    model.train()
    for step in range(recipe.steps):
        x, y = image_batch(step)
        opt.zero_grad()
        loss = fn.cross_entropy(model(put(x)), put(y))
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
    model.eval()
    correct = 0
    with lib.no_grad():
        for i in range(0, N_TEST_IMAGES, 256):
            pred = model(put(x_te[i : i + 256])).argmax(dim=1)
            correct += int((pred == put(y_te[i : i + 256])).sum().item())
    return Run(losses, correct / N_TEST_IMAGES)


def run_text(recipe: Recipe, framework: str, model: Any, device: str, ref: ModuleType) -> Run:
    corpus = markov_corpus(SEED)
    starts = text_batches(corpus, recipe.steps, SEED)
    lib, o, fn = (lucid, optim, F) if framework == "lucid" else (ref, ref.optim, ref.nn.functional)
    put = (lambda a: lucid.tensor(a, device=device)) if framework == "lucid" else ref.from_numpy
    clip = lucid.nn.utils.clip_grad_norm_ if framework == "lucid" else ref.nn.utils.clip_grad_norm_
    opt = recipe.make_optimizer(o, model.parameters())
    sched = recipe.make_schedule(o, opt)
    losses: list[float] = []
    model.train()
    for step in range(recipe.steps):
        batch = window(corpus, starts[step])
        inputs, targets = put(batch[:, :-1]), put(batch[:, 1:])
        opt.zero_grad()
        out = model(inputs)
        loss = fn.cross_entropy(out.reshape(-1, VOCAB), targets.reshape(-1))
        loss.backward()
        if recipe.clip is not None:
            clip(model.parameters(), recipe.clip)
        opt.step()
        sched.step()
        losses.append(float(loss.item()))
    model.eval()
    with lib.no_grad():
        w = corpus.eval_windows
        out = model(put(w[:, :-1]))
        held = fn.cross_entropy(out.reshape(-1, VOCAB), put(w[:, 1:]).reshape(-1))
    return Run(losses, float(held.item()))


def windowed_means(losses: list[float], n: int = 10) -> np.ndarray:
    return np.array([chunk.mean() for chunk in np.array_split(np.asarray(losses), n)])
