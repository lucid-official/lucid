"""Compiled training (``make_step``) against eager, one model per zoo family.

The op matrices prove each op alone; a model is the first place ops meet —
a view feeding a scatter, a mask feeding a softmax — and the first place a
missing VJP or an autodiff gap shows up as a model that will not train
compiled.  For each family the smallest registered factory is built through
its public name with a shrunken config (``create_model(name, **overrides)``,
never an invented factory — H11), and checked twice:

* **eval**: one compiled step is traced on one batch and replayed on
  another; the loss and every parameter's gradient must match eager
  backward on the second;
* **train**: from the same parameters, buffers and seed, one compiled step
  and one eager step must agree on the loss, every gradient and every
  buffer the step updates (batch-norm running statistics).  A step that
  runs eager — dropout does, by design — is reported with its reason.

The model's own random draws (diffusion noise, timesteps) are seeded the
same way before the eager and the compiled pass; a compiled call draws
exactly what eager draws (``compile/test_rng.py``).

The objective is the model's own ``loss`` when it returns one, else a
fixed random probe dotted with its main output, so every output element
carries gradient.

Agreement means nothing between two NaNs, or between two steps that move
nothing, so every status asks first that the loss be finite and the
gradients finite and not all zero — on the eager step, the compiled one and
the float64 tie-breaker alike.  A pair that fails it is ``nonfinite``, never
``ok`` (realnvp was: its loss NaN on inputs outside its data domain,
LCD-563).  A model that is a density over a bounded domain is fed inputs
inside it (``Spec.domain``).

Run one family per process (``python -m …_zoo_matrix <family> <task>``):
an MPSGraph abort kills the interpreter, and it must not take the other
families with it.  ``--sweep`` runs every family of every task that way and
holds the result to :data:`EXPECTED` — the nightly CI stage.
"""

import dataclasses
import inspect
import io
import json
import math
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

import lucid
from lucid.models._registry import _REGISTRY

from lucid.test.unit.compile._helpers import COMPILE_DEVICE

DEV = COMPILE_DEVICE
BATCH = 2
SEQ = 16

# ── config shrinking ──────────────────────────────────────────────────────
#
# Field-name rules first; a family whose defaults defeat them gets an entry
# in ``SPECS`` below, with the reason.  ``None`` = leave the field alone.


def _shrink_value(name: str, value: Any) -> Any:
    if isinstance(value, bool):
        return None
    if name in ("num_classes", "num_labels"):
        return 10 if name == "num_classes" else 3
    if name == "vocab_size":
        return 128
    if name in ("max_position_embeddings", "max_seq_len", "max_len"):
        return 64 if isinstance(value, int) else None
    if name in (
        "hidden_size",
        "dim",
        "embed_dim",
        "d_model",
        "text_width",
        "vision_width",
        "predictor_dim",
    ):
        return 32 if isinstance(value, int) else None
    if name in ("intermediate_size", "mlp_dim", "ffn_dim", "dim_feedforward"):
        return 64 if isinstance(value, int) else None
    if name in (
        "num_hidden_layers",
        "num_layers",
        "depth",
        "num_decoder_layers",
        "num_encoder_layers",
        "predictor_depth",
        "vision_layers",
        "text_layers",
        "n_layer",
    ):
        return 1 if isinstance(value, int) else None
    if name in (
        "num_attention_heads",
        "num_heads",
        "attn_heads",
        "text_heads",
        "vision_heads",
        "predictor_heads",
        "n_head",
    ):
        if isinstance(value, int):
            return 2
        if isinstance(value, (tuple, list)):
            return type(value)(2 for _ in value)
        return None
    if name in ("depths", "layers", "blocks_per_stage", "blocks"):
        if isinstance(value, (tuple, list)) and all(isinstance(v, int) for v in value):
            return type(value)(1 for _ in value)
        return None
    if name in ("base_channels", "stem_width", "stem_channels"):
        return 16 if isinstance(value, int) else None
    return None


def shrink(cfg: Any) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for f in dataclasses.fields(cfg):
        new = _shrink_value(f.name, getattr(cfg, f.name))
        if new is not None and new != getattr(cfg, f.name):
            overrides[f.name] = new
    return overrides


@dataclasses.dataclass(frozen=True)
class Spec:
    """What the name rules cannot guess about one family."""

    overrides: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    #: Image side length; ``None`` = read the config, else 64.
    size: int | None = None
    #: Builds ``(args, kwargs)`` for one batch from the config and a seed.
    inputs: Callable[[Any, int], tuple[tuple[Any, ...], dict[str, Any]]] | None = None
    #: Why the family is not a training-step check at all, when it is not.
    skip: str | None = None
    #: ``[low, high)`` of the data a density model is defined on; image
    #: inputs are drawn uniformly inside it.  ``None`` = standard normal.
    domain: tuple[float, float] | None = None


def _video(cfg: Any, seed: int) -> tuple[tuple[Any, ...], dict[str, Any]]:
    lucid.manual_seed(seed)
    c = int(getattr(cfg, "in_channels", 3))
    x = lucid.randn(BATCH, cfg.num_frames, c, cfg.image_size, cfg.image_size)
    return (x.to(DEV),), {}


def _volume(cfg: Any, seed: int) -> tuple[tuple[Any, ...], dict[str, Any]]:
    lucid.manual_seed(seed)
    x = lucid.randn(BATCH, int(getattr(cfg, "in_channels", 1)), 16, 16, 16)
    return (x.to(DEV),), {}


def _clip(cfg: Any, seed: int) -> tuple[tuple[Any, ...], dict[str, Any]]:
    lucid.manual_seed(seed)
    x = lucid.randn(BATCH, 3, cfg.image_size, cfg.image_size)
    prompts = lucid.randint(0, cfg.vocab_size, (3, cfg.context_length))
    return (x.to(DEV), prompts.to(DEV)), {}


def _flat(cfg: Any, seed: int) -> tuple[tuple[Any, ...], dict[str, Any]]:
    lucid.manual_seed(seed)
    return (lucid.randn(BATCH, int(cfg.input_dim)).to(DEV),), {}


def _regions(cfg: Any, seed: int) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """R-CNN: images, their proposals, and the boxes the proposals are scored on.

    R-CNN runs its CNN on proposal crops only, so without proposals the
    forward touches no parameter and there is nothing to train.  Of the four
    proposals per image one is the object itself, one overlaps it past the
    regression threshold, and two are background.
    """
    lucid.manual_seed(seed)
    x = lucid.randn(BATCH, int(cfg.in_channels), 64, 64)
    box = [8.0, 8.0, 40.0, 40.0]
    proposals = [
        lucid.tensor(
            [
                box,
                [10.0, 10.0, 42.0, 42.0],
                [44.0, 44.0, 63.0, 63.0],
                [0.0, 40.0, 20.0, 63.0],
            ]
        ).to(DEV)
        for _ in range(BATCH)
    ]
    targets = [
        {
            "boxes": lucid.tensor([box]).to(DEV),
            "labels": lucid.tensor([1 + (seed + b) % int(cfg.num_classes)]).to(DEV),
        }
        for b in range(BATCH)
    ]
    return (x.to(DEV), proposals, targets), {}


_T = 4  # sequence length for the world models


def _episodes(cfg: Any, seed: int) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Dreamer / PlaNet: frames, actions into each step, rewards, discounts."""
    lucid.manual_seed(seed)
    c, hw = int(cfg.in_channels), int(cfg.sample_size)
    obs = lucid.rand(BATCH, _T, c, hw, hw)
    act = lucid.randn(BATCH, _T, int(cfg.action_dim))
    rew = lucid.randn(BATCH, _T)
    args = [obs.to(DEV), act.to(DEV), rew.to(DEV)]
    if getattr(cfg, "pcont", False):
        args.append(lucid.ones(BATCH, _T).to(DEV))
    return tuple(args), {}


def _action_video(cfg: Any, seed: int) -> tuple[tuple[Any, ...], dict[str, Any]]:
    lucid.manual_seed(seed)
    video = lucid.randn(
        BATCH, cfg.num_frames, cfg.in_channels, cfg.image_size, cfg.image_size
    )
    steps = cfg.num_frames - 1  # one conditioning row per transition
    actions = lucid.randn(BATCH, steps, cfg.action_dim)
    states = lucid.randn(BATCH, steps, cfg.state_dim)
    return (video.to(DEV), actions.to(DEV), states.to(DEV)), {}


_ROLLOUT = "forward is a rollout (sampling), not a training step"

#: family → what the rules miss, and why.
SPECS: dict[str, Spec] = {
    # Zero-shot scoring takes the candidate prompts as a second input.
    "clip": Spec(
        {"image_size": 32, "patch_size": 16, "context_length": 8}, inputs=_clip
    ),
    # The attention stage is sized for 7×7 tokens: a 224 input.
    "efficientformer": Spec(size=224),
    # The native 299: at 96 the last stages are 1×1, and a train-mode batch
    # norm over two values is (a − b)/2 scaled by up to 1/√eps — rounding
    # alone moves the forward.
    "inception": Spec(size=299),
    "inception_resnet": Spec(size=299),
    "inception_v4": Spec(size=299),
    "xception": Spec(size=299),
    # The smallest factory is the 3-D one: volumes.
    "attention_unet": Spec(inputs=_volume),
    # One 32×32 channel — the paper's MNIST input.
    "lenet": Spec(size=32),
    # Window and grid attention both use 7×7 partitions of a /32 map.
    "maxvit": Spec(size=224),
    # Clips, (B, T, C, H, W).
    "vjepa": Spec({"image_size": 32, "num_frames": 4}, inputs=_video),
    "vjepa2": Spec({"image_size": 32, "num_frames": 4}, inputs=_video),
    # Group norms of 32 groups: every stage width a multiple of 32.
    "ddpm": Spec({"base_channels": 32}),
    "ncsn": Spec({"base_channels": 32}),
    "score_sde": Spec({"base_channels": 32}),
    "flow_matching": Spec({"base_channels": 32}),
    # Flat samples, (B, 784).
    "nice": Spec(inputs=_flat),
    # A density on [0, 1]: the flow opens with logit(alpha + (1 - 2 alpha) x),
    # NaN wherever that leaves (0, 1) — for 62% of a standard-normal draw.
    "realnvp": Spec(domain=(0.0, 1.0)),
    # Proposals and their targets: the CNN sees nothing else.
    "rcnn": Spec(inputs=_regions),
    "dreamer": Spec(inputs=_episodes),
    "dreamer_v2": Spec(inputs=_episodes),
    "dreamer_v3": Spec(inputs=_episodes),
    "planet": Spec(inputs=_episodes),
    "vjepa2_ac": Spec(
        {
            "encoder_dim": 32,
            "encoder_depth": 1,
            "encoder_heads": 2,
            "image_size": 32,
            "num_frames": 4,
        },
        inputs=_action_video,
    ),
    "stable_diffusion": Spec(
        skip=_ROLLOUT + " — 50 denoising steps from a text context"
    ),
    "diamond": Spec(skip=_ROLLOUT + " — imagines a horizon from real frames"),
    "genie": Spec(skip=_ROLLOUT + " — generates frames from a prompt"),
}


# ── picking the factory ───────────────────────────────────────────────────


def representative(family: str, task: str) -> str | None:
    """The smallest registered factory of ``family`` for ``task``."""
    names = [n for n, e in _REGISTRY.items() if e.family == family and e.task == task]
    if not names:
        return None
    return min(names, key=lambda n: (_REGISTRY[n].params or 10**12, n))


def families(task: str) -> list[str]:
    return sorted({e.family for e in _REGISTRY.values() if e.task == task})


# ── inputs ────────────────────────────────────────────────────────────────


def _cfg_int(cfg: Any, names: tuple[str, ...], default: int) -> int:
    for key in names:
        v = getattr(cfg, key, None)
        if isinstance(v, bool):
            continue
        if isinstance(v, int):
            return v
        if isinstance(v, (tuple, list)) and v and isinstance(v[-1], int):
            return int(v[-1])
    return default


_IMAGE_ARGS = ("x", "pixel_values", "images", "image", "sample")
_TOKEN_ARGS = ("input_ids", "decoder_input_ids")


def default_inputs(
    model: Any, cfg: Any, family: str, seed: int
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """One batch for every required parameter of ``model.forward``, by name."""
    spec = SPECS.get(family, Spec())
    if spec.inputs is not None:
        return spec.inputs(cfg, seed)
    lucid.manual_seed(seed)
    size = spec.size or _cfg_int(
        cfg, ("image_size", "img_size", "input_size", "sample_size", "resolution"), 64
    )
    chans = _cfg_int(
        cfg, ("in_channels", "num_channels", "image_channels", "channels"), 3
    )
    vocab = _cfg_int(cfg, ("vocab_size",), 128)
    args: list[Any] = []
    for p in list(inspect.signature(type(model).forward).parameters.values())[1:]:
        if p.default is not inspect.Parameter.empty or p.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue
        if p.name in _IMAGE_ARGS and spec.domain is not None:
            low, high = spec.domain
            image = lucid.rand(BATCH, chans, size, size) * (high - low) + low
            args.append(image.to(DEV))
        elif p.name in _IMAGE_ARGS:
            args.append(lucid.randn(BATCH, chans, size, size).to(DEV))
        elif p.name in _TOKEN_ARGS:
            args.append(lucid.randint(0, vocab, (BATCH, SEQ)).to(DEV))
        elif p.name in ("timestep", "timesteps", "t"):
            steps = _cfg_int(
                cfg, ("num_timesteps", "timesteps", "num_train_timesteps"), 1000
            )
            args.append(lucid.randint(0, steps, (BATCH,)).to(DEV))
        else:
            raise LookupError(f"no input rule for parameter {p.name!r}")
    return tuple(args), {}


# ── objective ─────────────────────────────────────────────────────────────


def _main_output(out: object) -> lucid.Tensor:
    if isinstance(out, lucid.Tensor):
        return out
    for attr in (
        "logits",
        "last_hidden_state",
        "pooler_output",
        "sample",
        "pred",
        "prediction",
        "reconstruction",
    ):
        t = getattr(out, attr, None)
        if isinstance(t, lucid.Tensor):
            return t
    if dataclasses.is_dataclass(out):
        for f in dataclasses.fields(out):
            t = getattr(out, f.name)
            if isinstance(t, lucid.Tensor) and t.is_floating_point():
                return t
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], lucid.Tensor):
        return out[0]
    raise TypeError(f"no tensor in model output {type(out).__name__}")


_PROBES: dict[tuple[tuple[int, ...], str, str], lucid.Tensor] = {}


def _probe(like: lucid.Tensor) -> lucid.Tensor:
    key = (tuple(like.shape), str(like.device), str(like.dtype))
    if key not in _PROBES:
        state = lucid.get_rng_state()
        lucid.manual_seed(11)
        _PROBES[key] = (lucid.rand(*like.shape) + 0.5).to(like.dtype).to(like.device)
        lucid.set_rng_state(state)
    return _PROBES[key]


def objective(out: object) -> lucid.Tensor:
    loss = getattr(out, "loss", None)
    if isinstance(loss, lucid.Tensor) and loss.ndim == 0:
        return loss
    t = _main_output(out)
    return (t * _probe(t)).sum()


# ── one family ────────────────────────────────────────────────────────────


def build(family: str, task: str) -> tuple[str, Any, Any, dict[str, Any]]:
    name = representative(family, task)
    if name is None:
        raise LookupError(f"no {task} factory in {family}")
    cfg = _REGISTRY[name].default_config
    overrides = shrink(cfg)
    overrides.update(SPECS.get(family, Spec()).overrides)
    lucid.manual_seed(0)
    model = lucid.models.create_model(name, **overrides)
    cfg = dataclasses.replace(cfg, **overrides)
    return name, model.to(DEV), cfg, overrides


def _grads(model: Any) -> dict[str, np.ndarray]:
    return {
        n: p.grad.numpy().copy()
        for n, p in model.named_parameters()
        if p.grad is not None
    }


def _state(model: Any) -> dict[str, np.ndarray]:
    return {k: v.numpy().copy() for k, v in model.state_dict().items()}


def _restore(
    model: Any, state: dict[str, np.ndarray], device: str = DEV, wide: bool = False
) -> None:
    def cast(v: np.ndarray) -> lucid.Tensor:
        t = lucid.tensor(v)
        if wide and t.is_floating_point():
            t = t.to(lucid.float64)
        return t.to(device)

    model.load_state_dict({k: cast(v) for k, v in state.items()})


def _zero(model: Any) -> None:
    for p in model.parameters():
        p.grad = None


def nonfinite(loss: float, grads: Mapping[str, np.ndarray]) -> str | None:
    """Why one step's numbers cannot be compared with another's, or ``None``.

    The comparisons below all go quiet on NaN — every ``NaN < tol`` is false,
    every ``NaN > worst`` too — and two steps that train nothing agree
    trivially, so neither may reach them.
    """
    if not math.isfinite(loss):
        return f"loss is {loss}"
    bad = sorted(n for n, g in grads.items() if not np.isfinite(g).all())
    if bad:
        return f"{len(bad)} gradient(s) not finite, first {bad[0]!r}"
    if not grads:
        return "no parameter received a gradient"
    if not any(np.any(g != 0) for g in grads.values()):
        return f"every gradient is zero ({len(grads)} parameters)"
    return None


def _worst(
    want: dict[str, np.ndarray], got: dict[str, np.ndarray]
) -> tuple[float, str, list[str]]:
    """Largest error over the entries, each against its own scale.

    The scale is floored at a small fraction of the largest entry in the
    set: an entry whose true value is ~0 (a key-projection bias, which the
    softmax cancels) otherwise turns float noise into a large relative
    error.
    """
    missing = sorted(set(want) - set(got))
    top = max((float(np.abs(w).max()) for w in want.values() if w.size), default=1.0)
    worst, worst_name = 0.0, ""
    for n, w in want.items():
        g = got.get(n)
        if g is None or g.shape != w.shape or not w.size:
            continue
        scale = max(float(np.abs(w).max()), 1e-3 * top, 1e-12)
        d = float(np.abs(g.astype(np.float64) - w.astype(np.float64)).max()) / scale
        if d > worst:
            worst, worst_name = d, n
    return worst, worst_name, missing


class _Through(lucid.nn.Module):
    """Lets a model that takes several inputs train through ``make_step``.

    ``make_step`` hands the model one input and the rest to the loss.  The
    model runs inside the objective instead, so every input is one of the
    step's own arguments — bound afresh on each call, never pinned.
    """

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return x


def _compiled_step(
    model: Any,
    batches: list[tuple[tuple[Any, ...], dict[str, Any]]],
    seed: int,
    start: dict[str, np.ndarray],
) -> tuple[float, str]:
    """Trace on ``batches[0]``, restore ``start``, replay on ``batches[1]``.

    Returns ``(replay loss, eager-fallback reason)``.
    """
    if len(batches[0][0]) > 1 or batches[0][1]:
        wrapper = _Through(model)
        wrapper.train(model.training)

        def run(x: lucid.Tensor, *rest: lucid.Tensor) -> lucid.Tensor:
            return objective(wrapper.model(x, *rest))

        step = lucid.compile.make_step(wrapper, run)
    else:
        step = lucid.compile.make_step(model, objective)
    err, old = io.StringIO(), sys.stderr
    sys.stderr = err
    try:
        _zero(model)
        a, _ = batches[0]
        step(*a).backward()
        _restore(model, start)
        _zero(model)
        a, _ = batches[1]
        lucid.manual_seed(seed)
        loss = step(*a)
        loss.backward()
    finally:
        sys.stderr = old
    fallback = step.eager_only  # type: ignore[attr-defined]
    fb = len(fallback.snapshot()) if hasattr(fallback, "snapshot") else len(fallback)
    why = ""
    if fb:
        lines = [ln for ln in err.getvalue().splitlines() if "eager fallback" in ln]
        why = (lines[-1] if lines else "eager fallback")[:300]
    return float(loss.item()), why


def _eager_step(
    model: Any, batch: tuple[tuple[Any, ...], dict[str, Any]], seed: int
) -> float:
    _zero(model)
    a, k = batch
    lucid.manual_seed(seed)
    loss = objective(model(*a, **k))
    loss.backward()
    return float(loss.item())


def _silence_dropout(model: Any) -> list[tuple[Any, str, float]]:
    """Set every dropout-like rate to 0; return what to put back."""
    saved: list[tuple[Any, str, float]] = []
    for m in model.modules():
        if "Drop" not in type(m).__name__:
            continue
        for attr in ("p", "drop_prob", "drop_rate"):
            v = getattr(m, attr, None)
            if isinstance(v, float) and v > 0.0:
                saved.append((m, attr, v))
                setattr(m, attr, 0.0)
    return saved


def _global_rel(got: dict[str, np.ndarray], want: dict[str, np.ndarray]) -> float:
    """‖got − want‖ / ‖want‖ over every gradient at once."""
    num = sum(
        float(((got[n].astype(np.float64) - w.astype(np.float64)) ** 2).sum())
        for n, w in want.items()
        if n in got
    )
    den = sum(float((w.astype(np.float64) ** 2).sum()) for w in want.values())
    return (num / den) ** 0.5 if den > 0 else 0.0


def _float64_verdict(
    family: str,
    task: str,
    model: Any,
    mode: str,
    batches: list[tuple[tuple[Any, ...], dict[str, Any]]],
    start: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Judge compiled against float64 when it disagrees with eager.

    Per parameter, float32 is not reliable enough to judge by in a deep
    train-mode network: a ReLU fed a value within rounding of zero flips,
    and in Inception-v3 eager and compiled were both 3–4% off float64 over
    the whole gradient — compiled the closer.  So the question asked here is
    whether compiled is any further from float64 than eager is, measured
    over the whole gradient.  A wrong VJP is, by far; conditioning is not.
    """
    saved = _silence_dropout(model)
    try:
        want_loss = _eager_step(model, batches[1], 100)
        want = _grads(model)
        _restore(model, start)
        got_loss, why = _compiled_step(model, batches, 100, start)
        got = _grads(model)
        _restore(model, start)
        if why:
            return {"verdict": "fallback without dropout", "why": why[:200]}
        ref_loss, ref = _float64_reference(family, task, mode, start, batches[1], 100)
    finally:
        for m, attr, v in saved:
            setattr(m, attr, v)
    for side, loss, grads in (
        ("eager", want_loss, want),
        ("compiled", got_loss, got),
        ("float64", ref_loss, ref),
    ):
        if (bad := nonfinite(loss, grads)) is not None:
            return {"verdict": f"{side} without dropout: {bad}", "ok": False}
    e = _global_rel(want, ref)
    c = _global_rel(got, ref)
    return {"eager": e, "compiled": c, "ok": c <= max(1.25 * e, 1e-3)}


def _float64_reference(
    family: str,
    task: str,
    mode: str,
    start: dict[str, np.ndarray],
    batch: tuple[tuple[Any, ...], dict[str, Any]],
    seed: int,
) -> tuple[float, dict[str, np.ndarray]]:
    """The same step in float64 on the CPU — the tie-breaker.

    A float32 step can land on the other side of a kink (a ReLU at a value
    within rounding of zero) from its float64 counterpart, and then eager
    and compiled disagree without either being wrong: in DenseNet-121's
    last dense layer eager was 37% off float64 and compiled 3%.  When the
    two float32 paths disagree, the compiled one is judged against this —
    with dropout off, since the CPU draws a different stream from Metal.
    """
    _, model, _, _ = build(family, task)
    model = model.to("cpu").to(lucid.float64)
    model.train(mode == "train")
    _silence_dropout(model)
    _restore(model, start, device="cpu", wide=True)

    def widen(t: Any) -> Any:
        if not isinstance(t, lucid.Tensor):
            return t
        t = t.to("cpu")
        return t.to(lucid.float64) if t.is_floating_point() else t

    a, k = batch
    loss = _eager_step(
        model, (tuple(widen(t) for t in a), {n: widen(t) for n, t in k.items()}), seed
    )
    return loss, _grads(model)


class _StepFailed(Exception):
    """The eager or the compiled step raised; ``status`` says which."""

    def __init__(self, status: str, err: Exception) -> None:
        super().__init__(f"{type(err).__name__}: {str(err)[:200]}")
        self.status = status


def _check_mode(
    family: str,
    task: str,
    model: Any,
    mode: str,
    batches: list[tuple[tuple[Any, ...], dict[str, Any]]],
) -> tuple[str, Any]:
    """One mode's comparison: ``("ok" | "wrong" | "fallback" | "nonfinite", detail)``."""
    model.train(mode == "train")
    start = _state(model)
    try:
        want_loss = _eager_step(model, batches[1], seed=100)
    except Exception as e:  # noqa: BLE001
        raise _StepFailed("eager", e) from e
    want, want_state = _grads(model), _state(model)
    _restore(model, start)
    if (bad := nonfinite(want_loss, want)) is not None:
        return "nonfinite", f"eager: {bad}"
    try:
        got_loss, why = _compiled_step(model, batches, 100, start)
    except Exception as e:  # noqa: BLE001
        raise _StepFailed("error", e) from e
    got, got_state = _grads(model), _state(model)
    _restore(model, start)
    if (bad := nonfinite(got_loss, got)) is not None:
        return "nonfinite", f"compiled: {bad}"
    if why:
        return "fallback", why
    worst, worst_name, missing = _worst(want, got)
    buf_worst, buf_name = 0.0, ""
    if mode == "train":
        buf_worst, buf_name, _ = _worst(
            {k: v for k, v in want_state.items() if k not in want},
            {k: v for k, v in got_state.items() if k not in want},
        )
    loss_rel = abs(got_loss - want_loss) / max(abs(want_loss), 1e-6)
    ok = loss_rel < 1e-3 and worst < 1e-2 and buf_worst < 1e-2 and not missing
    detail: dict[str, Any] = {
        "loss_rel": loss_rel,
        "grad": worst,
        "grad_param": worst_name,
        "buffer": buf_worst,
        "buffer_name": buf_name,
        "missing": missing[:5],
    }
    if not ok and buf_worst < 1e-2 and not missing:
        try:
            verdict = _float64_verdict(family, task, model, mode, batches, start)
        except Exception as e:  # noqa: BLE001
            detail["float64"] = f"{type(e).__name__}: {str(e)[:120]}"
        else:
            detail["float64"] = verdict
            ok = bool(verdict.get("ok"))
    return ("ok" if ok else "wrong"), detail


def run_family(family: str, task: str) -> dict[str, Any]:
    """Both checks for one family.  Returns a JSON-able record.

    A mode that disagrees on its first pair of batches is run again on a
    second.  In float32 a train-mode batch norm over a handful of values,
    feeding a ReLU, is discontinuous: a value within rounding of zero goes
    either way, and any path — eager, compiled, float64 — can land on a
    different side.  A real defect shows on every input; a kink on one.
    What passes on the second pair is reported ``ok`` with the first
    pair's numbers kept under ``unstable``.
    """
    rec: dict[str, Any] = {"family": family, "task": task}
    skip = SPECS.get(family, Spec()).skip
    if skip is not None:
        rec.update(status="skip", detail=skip)
        return rec
    try:
        name, model, cfg, overrides = build(family, task)
    except Exception as e:  # noqa: BLE001
        rec.update(status="build", detail=f"{type(e).__name__}: {str(e)[:200]}")
        return rec
    rec.update(factory=name, overrides=overrides)
    try:
        batches = [default_inputs(model, cfg, family, s) for s in (1, 2)]
    except Exception as e:  # noqa: BLE001
        rec.update(status="input", detail=f"{type(e).__name__}: {str(e)[:200]}")
        return rec

    for mode in ("eval", "train"):
        try:
            status, detail = _check_mode(family, task, model, mode, batches)
            if status == "wrong":
                retry = [default_inputs(model, cfg, family, s) for s in (3, 4)]
                status2, detail2 = _check_mode(family, task, model, mode, retry)
                if status2 == "ok":
                    status, detail = "ok", {"unstable": detail, "retry": detail2}
        except _StepFailed as e:
            rec.update(status=e.status, mode=mode, detail=str(e))
            return rec
        if status == "nonfinite":
            # Nothing the other mode says can make this pair a pass.
            rec.update(status=status, mode=mode, detail=f"{mode} {detail}")
            return rec
        rec[f"{mode}_status"] = status
        rec[f"{mode}_detail"] = detail
    modes = (rec.get("eval_status"), rec.get("train_status"))
    if "wrong" in modes:
        rec["status"] = "wrong"
    elif modes[0] == "fallback":
        rec["status"] = "fallback"
    else:
        rec["status"] = "ok"
    return rec


# ── the sweep ─────────────────────────────────────────────────────────────

#: Every task with families to train; together they cover the whole zoo.
TASKS: tuple[str, ...] = (
    "image-classification",
    "image-generation",
    "language-modeling",
    "object-detection",
    "semantic-segmentation",
    "sequence-classification",
    "token-classification",
    "world-modeling",
)

#: (family, task) → (eval status, train status, a substring of the fallback
#: reason) for every pair that does not train compiled.  Every other pair must
#: be ``ok`` in both modes.  Strict both ways, like the op matrices'
#: ``EXPECTED_EAGER``: a new fallback, or one for a different reason, fails,
#: and so does a listed pair that now compiles — its entry is stale.
EXPECTED: dict[tuple[str, str], tuple[str, str, str]] = {
    # Proposal sampling and NMS read scores back to the host.
    ("faster_rcnn", "object-detection"): ("fallback", "fallback", "read on the host"),
    ("mask_rcnn", "object-detection"): ("fallback", "fallback", "read on the host"),
    # Stage 3 picks the proposals it regresses by their overlap, on the host.
    ("rcnn", "object-detection"): ("fallback", "fallback", "read on the host"),
    # An image without proposals leaves a zero-length box tensor.
    ("fast_rcnn", "object-detection"): ("fallback", "fallback", "zero-size tensor"),
    # The loss is built from a JVP, which is a backward pass.
    ("mean_flow", "image-generation"): (
        "fallback",
        "fallback",
        "backward pass ran inside",
    ),
    # So is TVM's, and its gradient then runs back through that JVP.
    ("tvm", "image-generation"): (
        "fallback",
        "fallback",
        "backward pass ran inside",
    ),
    # The adaptive solver differentiates the dynamics inside the forward.
    ("neural_ode", "image-generation"): (
        "fallback",
        "fallback",
        "backward pass ran inside",
    ),
}

_MODULE = "lucid.test.unit.compile._zoo_matrix"


def run_isolated(family: str, task: str, timeout: float = 900.0) -> dict[str, Any]:
    """:func:`run_family` in a child interpreter, so an abort is a record."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", _MODULE, family, task],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"family": family, "task": task, "status": "timeout"}
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")]
    if lines:
        return dict(json.loads(lines[-1][len("RESULT ") :]))
    tail = " | ".join((proc.stderr or "").strip().splitlines()[-3:])
    return {
        "family": family,
        "task": task,
        "status": "abort",
        "detail": f"exit {proc.returncode}: {tail}"[:400],
    }


def verdict(rec: Mapping[str, Any]) -> str | None:
    """``None`` when a record is what :data:`EXPECTED` says, else why not."""
    if rec["status"] == "skip":
        return None
    if rec["status"] not in ("ok", "fallback", "wrong"):
        return f"{rec['status']}: {rec.get('detail')}"
    got = (rec.get("eval_status"), rec.get("train_status"))
    want = EXPECTED.get((rec["family"], rec["task"]))
    if want is None:
        if got == ("ok", "ok"):
            return None
        return f"expected ok/ok, got {got[0]}/{got[1]}: {_details(rec)}"
    if got == ("ok", "ok"):
        return "now trains compiled — delete its EXPECTED entry"
    if got != want[:2]:
        return f"expected {want[0]}/{want[1]}, got {got[0]}/{got[1]}: {_details(rec)}"
    for mode, status in zip(("eval", "train"), got):
        reason = str(rec.get(f"{mode}_detail"))
        if status == "fallback" and want[2] not in reason:
            return f"{mode} falls back for another reason: {reason[:200]}"
    return None


def _details(rec: Mapping[str, Any]) -> str:
    return "; ".join(
        f"{m}={str(rec.get(f'{m}_detail'))[:160]}" for m in ("eval", "train")
    )


def sweep(tasks: tuple[str, ...] = TASKS, shard: tuple[int, int] = (1, 1)) -> int:
    """Every family of every task, one process each.  Returns the failures.

    ``shard=(k, n)`` runs every n-th pair from the k-th, so n CI jobs split
    the sweep between them; only shard 1 checks that every :data:`EXPECTED`
    entry still names a factory, so that failure is counted once.
    """
    k, n = shard
    pairs = [(family, task) for task in tasks for family in families(task)]
    mine = pairs[k - 1 :: n]
    failures = 0
    for family, task in mine:
        rec = run_isolated(family, task)
        why = verdict(rec)
        failures += why is not None
        mark = "FAIL" if why else rec["status"]
        print(f"{mark:8s} {task:24s} {family:18s} {why or ''}", flush=True)
    if k == 1:
        for key in sorted(set(EXPECTED) - set(pairs)):
            if key[1] in tasks:
                failures += 1
                print(f"FAIL     {key[1]:24s} {key[0]:18s} EXPECTED names no factory")
    of = f" (shard {k}/{n} of {len(pairs)})" if n > 1 else ""
    print(f"{len(mine)} pairs{of}, {failures} failing")
    return failures


def _shard(text: str) -> tuple[int, int]:
    k, _, n = text.partition("/")
    if not (k.isdigit() and n.isdigit() and 1 <= int(k) <= int(n)):
        raise SystemExit(f"--shard wants K/N with 1 <= K <= N, got {text!r}")
    return int(k), int(n)


def main() -> None:  # pragma: no cover — triage driver
    args = sys.argv[1:]
    if args[:1] == ["--sweep"]:
        args = args[1:]
        shard = (1, 1)
        if args[:1] == ["--shard"]:
            shard, args = _shard(args[1]), args[2:]
        sys.exit(1 if sweep(tuple(args) or TASKS, shard) else 0)
    os.environ["LUCID_COMPILE_VERBOSE"] = "1"
    family, task = sys.argv[1], sys.argv[2]
    rec = run_family(family, task)
    print("RESULT " + json.dumps(rec, default=str))


if __name__ == "__main__":  # pragma: no cover
    main()
