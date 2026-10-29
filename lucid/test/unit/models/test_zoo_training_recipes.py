"""The workflow a user runs on a zoo model, start to finish, on cpu and metal.

The per-family tests each prove a piece — a shape, a gradient, a cache.
None proves that the pieces compose into what a user does with a model:
build it from a factory, train it a few steps, run it, save it, load it
into a fresh instance, resume training, generate with it, compile it.
This module runs that whole recipe on one representative of each kind of
family (a classifier with batch-norm state, a ConvNeXt, a causal LM, an
encoder-decoder LM, a diffusion denoiser) and asserts, per recipe:

* **trains** — a few AdamW steps on one fixed seeded batch leave every loss
  finite and the last below the first;
* **infers** — in eval mode the output keeps the family's ``ModelOutput``
  contract (fields, shapes, dtype, no loss without labels) and is the same
  on a second call;
* **checkpoints** — ``lucid.save`` / ``lucid.load`` of the state dict into a
  freshly built model reproduces the outputs exactly, and saving the
  optimizer state as well resumes the loss trajectory unchanged;
* **generates** (the two LMs) — greedy tokens are the same through the
  uncached, dynamic-cache and static-cache paths, and on metal through the
  compiled decode step;
* **compiles** (metal) — ``lucid.compile(model)`` answers what eager does.

Every model is a real factory with config overrides (H11), dropout off so
the same batch gives the same loss.  The training run is shared by the
checks of a (recipe, device) pair, so each model trains twice in all.
"""

import math
import os
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import pytest

import lucid
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.models import create_model
from lucid.models.generative import DDPMScheduler
from lucid.test._fixtures.devices import metal_available
from lucid.test._helpers.compare import assert_close

_STEPS = 3  # per leg; the continuous run takes 2 * _STEPS
_LR = 1e-2

Batch = dict[str, lucid.Tensor]

_TEXT: dict[str, Any] = {
    "vocab_size": 64,
    "hidden_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 32,
    "max_position_embeddings": 32,
    "hidden_dropout": 0.0,
    "attention_dropout": 0.0,
}


@dataclass(frozen=True)
class Recipe:
    """One family's way through the workflow."""

    id: str
    factory: str
    overrides: dict[str, Any]
    make_batch: Callable[[], Batch]
    #: ``(model, batch) -> scalar loss`` — what a training step minimises.
    loss: Callable[[Any, Batch], lucid.Tensor]
    #: ``(model, batch) -> ModelOutput`` — a plain forward, no labels.
    infer: Callable[[Any, Batch], Any]
    #: Output fields a plain forward must carry: name -> shape.
    contract: dict[str, tuple[int, ...]]
    lr: float = _LR
    #: Greedy-generation prompt key in the batch, for the two LMs.
    prompt: str | None = None
    #: Output field ``lucid.compile`` is compared on.
    compared: str = "logits"
    #: Compile-parity tolerance (absolute, relative) — see ``test_compiled_...``.
    tol: tuple[float, float] = (1e-4, 1e-4)


def _images(side: int, classes: int = 4) -> Callable[[], Batch]:
    def make() -> Batch:
        lucid.manual_seed(1234)
        return {"x": lucid.randn(2, 3, side, side), "labels": lucid.randint(0, classes, (2,))}

    return make


def _classifier_loss(model: Any, b: Batch) -> lucid.Tensor:
    return model(b["x"], labels=b["labels"]).loss


def _classifier_infer(model: Any, b: Batch) -> Any:
    return model(b["x"])


def _tokens() -> Batch:
    lucid.manual_seed(1234)
    return {"ids": lucid.randint(0, 64, (2, 8))}


def _seq2seq() -> Batch:
    lucid.manual_seed(1234)
    return {"src": lucid.randint(0, 64, (2, 8)), "tgt": lucid.randint(0, 64, (2, 8))}


def _denoising() -> Batch:
    lucid.manual_seed(1234)
    return {
        "x_t": lucid.randn(2, 3, 16, 16),
        "t": lucid.tensor([3, 11]).long(),
        "noise": lucid.randn(2, 3, 16, 16),
    }


def _denoiser_loss(model: Any, b: Batch) -> lucid.Tensor:
    return F.mse_loss(model(b["x_t"], b["t"]).sample, b["noise"])


_DDPM: dict[str, Any] = {
    "sample_size": 16,
    "base_channels": 16,
    "channel_mult": (1, 2),
    "num_res_blocks": 1,
    "attention_resolutions": (8,),
    "num_heads": 2,
    "resnet_groups": 8,
    "num_train_timesteps": 20,
    "dropout": 0.0,
}

RECIPES: list[Recipe] = [
    Recipe(
        id="resnet",
        factory="resnet_18_cls",
        overrides={
            "layers": (1, 1, 1, 1),
            "stem_channels": 16,
            "hidden_sizes": (16, 32, 64, 128),
            "num_classes": 4,
        },
        make_batch=_images(32),
        loss=_classifier_loss,
        infer=_classifier_infer,
        contract={"logits": (2, 4)},
    ),
    Recipe(
        id="convnext",
        factory="convnext_tiny_cls",
        overrides={"depths": (1, 1, 1, 1), "dims": (16, 32, 64, 128), "num_classes": 4},
        make_batch=_images(64),
        loss=_classifier_loss,
        infer=_classifier_infer,
        contract={"logits": (2, 4)},
    ),
    Recipe(
        id="gpt2",
        factory="gpt2_small_lm",
        overrides=_TEXT,
        make_batch=_tokens,
        loss=lambda m, b: m(b["ids"], labels=b["ids"]).loss,
        infer=lambda m, b: m(b["ids"]),
        contract={"logits": (2, 8, 64)},
        prompt="ids",
    ),
    Recipe(
        id="t5",
        factory="t5_small_seq2seq",
        overrides={**_TEXT, "num_decoder_layers": 1, "head_dim": 8},
        make_batch=_seq2seq,
        loss=lambda m, b: m(b["src"], labels=b["tgt"]).loss,
        infer=lambda m, b: m(b["src"], decoder_input_ids=b["tgt"]),
        contract={"logits": (2, 8, 64), "encoder_last_hidden_state": (2, 8, 16)},
        prompt="src",
    ),
    Recipe(
        id="ddpm",
        factory="ddpm_cifar",
        overrides=_DDPM,
        make_batch=_denoising,
        loss=_denoiser_loss,
        infer=lambda m, b: m(b["x_t"], b["t"]),
        contract={"sample": (2, 3, 16, 16)},
        compared="sample",
    ),
]


# --------------------------------------------------------------------------
# One shared run per (recipe, device)
# --------------------------------------------------------------------------


@dataclass
class Run:
    """A recipe trained for ``2 * _STEPS`` steps, with a checkpoint at the half."""

    recipe: Recipe
    device: str
    batch: Batch
    model: Any
    losses: list[float]
    half_model: str
    half_optim: str
    final_model: str


def _build(recipe: Recipe, device: str, seed: int) -> Any:
    lucid.manual_seed(seed)
    return create_model(recipe.factory, **recipe.overrides).to(device)


def _on(batch: Batch, device: str) -> Batch:
    return {k: v.to(device) for k, v in batch.items()}


def _train(model: Any, recipe: Recipe, batch: Batch, opt: Any, steps: int) -> list[float]:
    model.train()
    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        loss = recipe.loss(model, batch)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


_RUNS: dict[tuple[str, str], Run] = {}


@pytest.fixture(scope="module")
def workdir(tmp_path_factory: pytest.TempPathFactory) -> str:
    return str(tmp_path_factory.mktemp("zoo_recipes"))


def _run(recipe: Recipe, device: str, workdir: str) -> Run:
    key = (recipe.id, device)
    if key in _RUNS:
        return _RUNS[key]
    batch = _on(recipe.make_batch(), device)
    model = _build(recipe, device, seed=0)
    opt = optim.AdamW(model.parameters(), lr=recipe.lr)
    stem = os.path.join(workdir, f"{recipe.id}-{device}")
    losses = _train(model, recipe, batch, opt, _STEPS)
    lucid.save(model.state_dict(), f"{stem}-half-model.lct")
    lucid.save(opt.state_dict(), f"{stem}-half-optim.lct")
    losses += _train(model, recipe, batch, opt, _STEPS)
    lucid.save(model.state_dict(), f"{stem}-final-model.lct")
    _RUNS[key] = Run(
        recipe,
        device,
        batch,
        model,
        losses,
        f"{stem}-half-model.lct",
        f"{stem}-half-optim.lct",
        f"{stem}-final-model.lct",
    )
    return _RUNS[key]


def _fields_present(out: Any) -> dict[str, lucid.Tensor]:
    """The tensor fields of an output — a cache object is not one."""
    named = {f.name: getattr(out, f.name) for f in fields(out)}
    return {k: v for k, v in named.items() if isinstance(v, lucid.Tensor)}


def _infer(run: Run, model: Any | None = None) -> Any:
    m = (model if model is not None else run.model).eval()
    with lucid.no_grad():
        return run.recipe.infer(m, run.batch)


def _identical(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


@pytest.fixture(params=RECIPES, ids=lambda r: r.id)
def recipe(request: pytest.FixtureRequest) -> Recipe:
    return request.param


@pytest.fixture
def run(recipe: Recipe, device: str, workdir: str) -> Run:
    return _run(recipe, device, workdir)


@pytest.fixture
def metal_run(recipe: Recipe, device_gpu_only: str, workdir: str) -> Run:
    return _run(recipe, device_gpu_only, workdir)


# --------------------------------------------------------------------------
# 1. train
# --------------------------------------------------------------------------


def test_a_few_adamw_steps_lower_the_loss(run: Run) -> None:
    assert all(math.isfinite(loss) for loss in run.losses), run.losses
    assert run.losses[-1] < run.losses[0], f"loss did not fall: {run.losses}"


# --------------------------------------------------------------------------
# 2. infer
# --------------------------------------------------------------------------


def test_inference_keeps_the_output_contract_and_is_deterministic(run: Run) -> None:
    first, second = _fields_present(_infer(run)), _fields_present(_infer(run))
    assert "loss" not in first, "a forward without labels reported a loss"
    for name, shape in run.recipe.contract.items():
        assert name in first, f"the output lost its {name!r} field"
        t = first[name]
        assert tuple(t.shape) == shape, f"{name}: {tuple(t.shape)} != {shape}"
        assert t.dtype == lucid.float32, f"{name}: dtype {t.dtype}"
        assert bool(lucid.isfinite(t).all().item()), f"{name} has non-finite values"
        assert _identical(t, second[name]), f"{name} differs between two calls"


# --------------------------------------------------------------------------
# 3. checkpoint
# --------------------------------------------------------------------------


def test_a_saved_state_dict_rebuilds_the_same_model(run: Run) -> None:
    fresh = _build(run.recipe, run.device, seed=99)
    fresh.load_state_dict(lucid.load(run.final_model))
    want = _fields_present(_infer(run))
    got = _fields_present(_infer(run, fresh))
    assert set(got) == set(want)
    for name, t in want.items():
        assert _identical(got[name], t), f"{name} differs after a checkpoint round trip"


def test_resuming_from_a_checkpoint_continues_the_same_trajectory(run: Run) -> None:
    fresh = _build(run.recipe, run.device, seed=99)
    fresh.load_state_dict(lucid.load(run.half_model))
    opt = optim.AdamW(fresh.parameters(), lr=run.recipe.lr)
    opt.load_state_dict(lucid.load(run.half_optim))
    resumed = _train(fresh, run.recipe, run.batch, opt, _STEPS)
    assert_close(resumed, run.losses[_STEPS:], atol=0.0, rtol=1e-6, msg="resumed trajectory")


def test_resuming_on_weights_alone_would_have_been_caught(run: Run) -> None:
    """The control: a cold optimizer does not continue the run, so the test above bites."""
    fresh = _build(run.recipe, run.device, seed=99)
    fresh.load_state_dict(lucid.load(run.half_model))
    opt = optim.AdamW(fresh.parameters(), lr=run.recipe.lr)
    cold = _train(fresh, run.recipe, run.batch, opt, _STEPS)
    assert cold != run.losses[_STEPS:], "a cold optimizer reproduced the warm trajectory"


def test_a_checkpoint_loads_on_the_other_device(run: Run) -> None:
    other = "cpu" if run.device == "metal" else "metal"
    if other == "metal" and not metal_available():
        pytest.skip("Metal device not available on this host")
    fresh = _build(run.recipe, other, seed=99)
    fresh.load_state_dict(lucid.load(run.final_model))
    want = _fields_present(_infer(run))
    with lucid.no_grad():
        got = _fields_present(run.recipe.infer(fresh.eval(), _on(run.batch, other)))
    for name, t in want.items():
        assert_close(got[name].to("cpu"), t.to("cpu"), atol=1e-4, rtol=1e-4, msg=name)


# --------------------------------------------------------------------------
# 4. generate
# --------------------------------------------------------------------------

_LMS = [r for r in RECIPES if r.prompt is not None]


def _generate(run: Run, **kw: Any) -> lucid.Tensor:
    prompt = run.batch[run.recipe.prompt][:, :4]
    model = run.model.eval()
    if run.recipe.id == "t5":
        return model.generate(prompt, max_length=8, **kw)
    return model.generate(prompt, max_new_tokens=6, do_sample=False, **kw)


@pytest.fixture(params=_LMS, ids=lambda r: r.id)
def lm_run(request: pytest.FixtureRequest, device: str, workdir: str) -> Run:
    return _run(request.param, device, workdir)


@pytest.fixture(params=_LMS, ids=lambda r: r.id)
def lm_metal_run(request: pytest.FixtureRequest, device_gpu_only: str, workdir: str) -> Run:
    return _run(request.param, device_gpu_only, workdir)


def test_greedy_tokens_agree_across_the_cache_implementations(lm_run: Run) -> None:
    reference = _generate(lm_run, use_cache=False)
    dynamic = _generate(lm_run, use_cache=True)
    static = _generate(lm_run, use_cache=True, cache_implementation="static", max_cache_len=16)
    assert _identical(dynamic, reference), "dynamic cache decoded other tokens"
    assert _identical(static, reference), "static cache decoded other tokens"


def test_greedy_tokens_agree_through_the_compiled_decode_step(lm_metal_run: Run) -> None:
    reference = _generate(lm_metal_run, use_cache=False)
    compiled = _generate(
        lm_metal_run, use_cache=True, cache_implementation="compiled", max_cache_len=16
    )
    assert _identical(compiled, reference), "the compiled decode step decoded other tokens"


# --------------------------------------------------------------------------
# 5. compile
# --------------------------------------------------------------------------


def test_compiled_forward_matches_eager(metal_run: Run) -> None:
    """Within fp32 reordering, not bitwise: CI's GPU is not this machine's.

    Eager runs MLX kernels, the compiled graph MPSGraph's, so reductions are
    ordered differently and the answers agree to a few ulp, not exactly.
    """
    name = metal_run.recipe.compared
    want = _fields_present(_infer(metal_run))[name]
    compiled = lucid.compile(metal_run.model.eval())
    with lucid.no_grad():
        got = _fields_present(metal_run.recipe.infer(compiled, metal_run.batch))[name]
    atol, rtol = metal_run.recipe.tol
    assert_close(got, want, atol=atol, rtol=rtol, msg=f"{metal_run.recipe.id}.{name}")


def test_compiling_a_model_leaves_its_checkpoint_format_alone(metal_run: Run) -> None:
    plain = metal_run.model.state_dict()
    compiled = lucid.compile(metal_run.model).state_dict()
    assert list(compiled) == list(plain), "compile changed the state_dict keys"


# --------------------------------------------------------------------------
# 6. sample (the denoiser's generation loop)
# --------------------------------------------------------------------------


def test_a_seeded_ddpm_sampling_loop_returns_the_same_finite_images(device: str) -> None:
    recipe = next(r for r in RECIPES if r.id == "ddpm")
    wrapper = create_model("ddpm_cifar_gen", **recipe.overrides).to(device).eval()
    scheduler = DDPMScheduler(num_train_timesteps=recipe.overrides["num_train_timesteps"])
    draws = []
    for _ in range(2):
        lucid.manual_seed(5)
        draws.append(wrapper.generate(scheduler, n_samples=2, num_inference_steps=4).samples)
    first, second = draws
    assert tuple(first.shape) == (2, 3, 16, 16) and first.dtype == lucid.float32
    assert bool(lucid.isfinite(first).all().item())
    assert _identical(first, second), "the same seed sampled different images"
