"""A model built under a Metal default is whole on Metal (LCD-849).

``lucid.set_default_device("metal"); create_model(name)`` is how a model meant
for training gets built, and it failed in two ways, both a constructor assuming
the CPU:

* a table computed in float64 on the default device.  Metal has no float64, so
  the build raised (I-JEPA, V-JEPA, Stable Diffusion's scheduler);
* a helper whose ``device="cpu"`` default opted out of the default device.  The
  build left one table behind and the first forward raised ``DeviceMismatch``
  (``positional_encoding.pe`` in Transformer, ``sigmas`` in NCSN, the schedules
  of DiT and of learned-variance DDPM).

No single kind of build sees everything, so there are two sweeps.  A
``shadow_alloc`` build covers every factory at its real size, reports placement
and refuses float64 on Metal as a real build does, but it carries metadata only:
no kernel runs and no host buffer is filled or uploaded.  A real build of each
family's smallest factory, shrunk through its config, runs them the way a user's
does.  Below them, the families that failed run for real, a table holds the same
numbers whichever default it was built under, and no helper defaults its
``device`` to the CPU, which also covers the ones no constructor reaches.

Every test leaves the default device as it found it.
"""

import ast
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

import lucid
import lucid.models as models
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._registry import _REGISTRY, list_models
from lucid.models._utils._generative import (
    beta_schedule_floats,
    diffusion_posterior_constants,
    flow_prior_sample,
    make_beta_schedule,
    make_sigma_schedule,
    resolve_generation_device,
)
from lucid.models._utils._segmentation import sample_point_coords
from lucid.models.generative._common._schedulers import DDPMScheduler
from lucid.models.generative._common._transformers import sincos_position_embedding
from lucid.models.generative.stable_diffusion import (
    DDIMScheduler,
    PNDMScheduler,
    StableDiffusionConfig,
)
from lucid.models.vision._common._transformers import sincos_embedding
from lucid.models.vision.ijepa._model import _sincos_2d
from lucid.models.vision.vjepa._model import _sincos_3d
from lucid.nn._shadow import shadow_alloc
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile import _zoo_matrix as zoo

pytestmark = pytest.mark.skipif(not metal_available(), reason="needs Metal")


@pytest.fixture
def metal_default() -> Iterator[None]:
    previous = lucid.get_default_device()
    lucid.set_default_device("metal")
    try:
        yield
    finally:
        lucid.set_default_device(previous)


def _off_metal(model: nn.Module) -> list[str]:
    held = [*model.named_parameters(), *model.named_buffers()]
    return [name for name, tensor in held if tensor.device.type != "metal"]


# ── every factory, at real size ──────────────────────────────────────────────


@pytest.mark.usefixtures("metal_default")
@pytest.mark.parametrize("name", list_models())
def test_a_factory_builds_whole_on_metal(name: str) -> None:
    with shadow_alloc():
        model = models.create_model(name)
    assert not _off_metal(model)


# ── every family, for real ───────────────────────────────────────────────────

#: Families the name rules leave at full size: 943M and 20B parameters.  Metal
#: is lazy, so a build costs little either way, but one read of a parameter
#: would materialize it.
_FURTHER_SHRUNK: dict[str, dict[str, Any]] = {
    "stable_diffusion": {
        "sample_size": 32,
        "downsample_factor": 4,
        "vae_block_out_channels": (32, 64, 64),
        "vae_layers_per_block": 1,
        "unet_block_out_channels": (32, 64),
        "unet_layers_per_block": 1,
        "attention_head_dim": 4,
        "cross_attention_dim": 16,
        "context_length": 4,
        "norm_num_groups": 32,
    },
    "genie": {
        "sample_size": (8, 8),
        "num_frames": 4,
        "num_codes": 16,
        "code_dim": 4,
        "tokenizer_encoder_layers": 1,
        "tokenizer_encoder_dim": 16,
        "tokenizer_encoder_heads": 2,
        "tokenizer_encoder_head_dim": 8,
        "tokenizer_decoder_layers": 1,
        "tokenizer_decoder_dim": 16,
        "tokenizer_decoder_heads": 2,
        "tokenizer_decoder_head_dim": 8,
        "action_patch_size": 4,
        "action_dim": 4,
        "action_encoder_layers": 1,
        "action_encoder_dim": 16,
        "action_encoder_heads": 2,
        "action_decoder_layers": 1,
        "action_decoder_dim": 16,
        "action_decoder_heads": 2,
        "dynamics_layers": 1,
        "dynamics_dim": 16,
        "dynamics_heads": 2,
        "dynamics_head_dim": 8,
        "maskgit_steps": 4,
    },
}


def _representatives() -> list[str]:
    """One factory per (family, task): a task wrapper can hold tables of its own."""
    pairs = {(entry.family, entry.task) for entry in _REGISTRY.values()}
    return sorted({str(zoo.representative(family, task)) for family, task in pairs})


def _shrunk(name: str) -> dict[str, Any]:
    entry = _REGISTRY[name]
    return {
        **zoo.shrink(entry.default_config),
        **zoo.SPECS.get(str(entry.family), zoo.Spec()).overrides,
        **_FURTHER_SHRUNK.get(str(entry.family), {}),
    }


@pytest.mark.usefixtures("metal_default")
@pytest.mark.parametrize("name", _representatives())
def test_a_family_builds_for_real_and_whole_on_metal(name: str) -> None:
    assert not _off_metal(models.create_model(name, **_shrunk(name)))


# ── the families that failed, trained and sampled ────────────────────────────


_TEXT: dict[str, Any] = {
    "vocab_size": 64,
    "hidden_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 32,
    "max_position_embeddings": 32,
}
_NCSN: dict[str, Any] = {
    "sample_size": 16,
    "base_channels": 16,
    "channel_mult": (1, 2),
    "num_res_blocks": 1,
    "attention_resolutions": (8,),
    "num_heads": 2,
    "resnet_groups": 8,
    "num_noise_levels": 5,
    "sigma_max": 10.0,
    "sigma_min": 0.1,
    "langevin_steps": 1,
}
_DDPM: dict[str, Any] = {
    "sample_size": 16,
    "base_channels": 16,
    "channel_mult": (1, 2),
    "num_res_blocks": 1,
    "attention_resolutions": (8,),
    "num_heads": 2,
    "resnet_groups": 8,
    "num_train_timesteps": 20,
}
_DIT: dict[str, Any] = {
    "sample_size": 8,
    "hidden_size": 32,
    "depth": 1,
    "num_heads": 4,
    "num_classes": 4,
}
_IJEPA: dict[str, Any] = {
    "image_size": 32,
    "patch_size": 8,
    "dim": 16,
    "depth": 1,
    "num_heads": 2,
    "predictor_dim": 8,
    "predictor_depth": 1,
    "min_keep": 1,
    "target_scale": (0.1, 0.2),
    "context_scale": (0.8, 1.0),
}
_VJEPA: dict[str, Any] = {
    "image_size": 32,
    "patch_size": 8,
    "tubelet_size": 2,
    "num_frames": 4,
    "dim": 24,
    "depth": 1,
    "num_heads": 2,
    "predictor_dim": 12,
    "predictor_depth": 1,
    "short_range_blocks": 2,
    "long_range_blocks": 1,
}

Inputs = tuple[tuple[Any, ...], dict[str, Any]]


def _source_and_target() -> Inputs:
    source = lucid.randint(0, 64, (2, 8))
    target = lucid.randint(0, 64, (2, 8))
    return (source, target), {"labels": target}


def _token_class() -> Inputs:
    return (lucid.randint(0, 64, (2, 8)),), {"labels": lucid.randint(0, 2, (2,))}


def _images(*shape: int) -> Callable[[], Inputs]:
    return lambda: ((lucid.randn(2, *shape),), {})


def _noised_images() -> Inputs:
    x = lucid.randn(2, 3, 16, 16)
    return (x, lucid.randint(0, 20, (2,))), {"target": lucid.randn(2, 3, 16, 16)}


def _latents_and_labels() -> Inputs:
    return (lucid.randn(2, 4, 8, 8), lucid.randint(0, 4, (2,))), {}


def _images_and_context() -> Inputs:
    return (lucid.randn(2, 3, 32, 32), lucid.randn(2, 4, 16)), {}


_TRAINS = [
    ("transformer_base_seq2seq", {**_TEXT, "num_decoder_layers": 1}, _source_and_target),
    ("transformer_base_cls", {**_TEXT, "num_labels": 2}, _token_class),
    ("ncsn_cifar_gen", _NCSN, _images(3, 16, 16)),
    ("ddpm_imagenet64_gen", _DDPM, _noised_images),
    ("dit_small_2_gen", _DIT, _latents_and_labels),
    ("ijepa_base_16", _IJEPA, _images(3, 32, 32)),
    ("vjepa_large_16", _VJEPA, _images(4, 3, 32, 32)),
    ("stable_diffusion_gen", _FURTHER_SHRUNK["stable_diffusion"], _images_and_context),
]


@pytest.mark.usefixtures("metal_default")
@pytest.mark.parametrize(
    ("name", "overrides", "make_inputs"), _TRAINS, ids=[case[0] for case in _TRAINS]
)
def test_a_family_that_failed_trains_a_step_on_metal(
    name: str, overrides: dict[str, Any], make_inputs: Callable[[], Inputs]
) -> None:
    lucid.manual_seed(0)
    model = models.create_model(name, **overrides).train()
    assert not _off_metal(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = lucid.optim.SGD(trainable, lr=1e-3)
    args, kwargs = make_inputs()
    loss = model(*args, **kwargs).loss
    assert loss.device.type == "metal"
    assert bool(loss.isfinite().item()), name
    loss.backward()
    optimizer.step()
    grads = [p.grad for p in trainable if p.grad is not None]
    assert grads, name
    assert all(g.device.type == "metal" for g in grads), name


@pytest.mark.usefixtures("metal_default")
@pytest.mark.parametrize(
    ("name", "overrides", "make_inputs"),
    [
        ("transformer_base_seq2seq", {**_TEXT, "num_decoder_layers": 1}, _source_and_target),
        ("ijepa_base_16_cls", _IJEPA, _images(3, 32, 32)),
    ],
    ids=["transformer", "ijepa"],
)
def test_a_family_that_failed_answers_on_metal(
    name: str, overrides: dict[str, Any], make_inputs: Callable[[], Inputs]
) -> None:
    model = models.create_model(name, **overrides).eval()
    args, _ = make_inputs()
    with lucid.no_grad():
        out = model(*args)
    assert out.logits.device.type == "metal"


@pytest.mark.usefixtures("metal_default")
@pytest.mark.parametrize(
    ("name", "overrides", "sample"),
    [
        ("ncsn_cifar_gen", _NCSN, lambda m: m.generate(1)),
        ("dit_small_2_gen", _DIT, lambda m: m.generate(1, steps=2)),
        (
            "ddpm_cifar_gen",
            _DDPM,
            lambda m: m.generate(m.make_scheduler(), n_samples=1, num_inference_steps=2),
        ),
        (
            "stable_diffusion_gen",
            _FURTHER_SHRUNK["stable_diffusion"],
            lambda m: m.generate(lucid.randn(1, 4, 16), num_inference_steps=2),
        ),
    ],
    ids=["ncsn", "dit", "ddpm", "stable_diffusion"],
)
def test_a_family_that_failed_samples_on_metal(
    name: str, overrides: dict[str, Any], sample: Callable[[Any], Any]
) -> None:
    lucid.manual_seed(0)
    model = models.create_model(name, **overrides).eval()
    with lucid.no_grad():
        samples = sample(model).samples
    assert samples.device.type == "metal"
    assert bool(samples.isfinite().all().item()), name


# ── the numbers do not depend on the default ─────────────────────────────────


def _learned_variance_ddpm() -> list[Tensor]:
    model = models.create_model("ddpm_imagenet64_gen", **_DDPM)
    return [
        model._log_betas,
        model._sqrt_recip_ab,
        model._sqrt_recipm1_ab,
        model._posterior_mean_coef_start,
        model._posterior_mean_coef_t,
        model._posterior_log_variance,
    ]


def _dit() -> list[Tensor]:
    model = models.create_model("dit_small_2_gen", **_DIT)
    return [model.betas, model.alphas_cumprod, model.dit.pos_embed]


def _ddpm_scheduler(schedule: str) -> Callable[[], list[Tensor]]:
    def tables() -> list[Tensor]:
        s = DDPMScheduler(50, beta_schedule=schedule)
        return [s.betas, s.alphas, s.alphas_cumprod, s._signal_scale, s._noise_scale, s.timesteps]

    return tables


def _sd_noising_tables(schedule: str) -> Callable[[], list[Tensor]]:
    def tables() -> list[Tensor]:
        s = DDIMScheduler(StableDiffusionConfig(beta_schedule=schedule))
        return [s._signal_scale, s._noise_scale]

    return tables


_TABLES: dict[str, Callable[[], list[Tensor]]] = {
    "sinusoidal_embedding": lambda: [F.sinusoidal_embedding(64, 32)],
    "sinusoidal_embedding_2d": lambda: [F.sinusoidal_embedding_2d(8, 6, 32)],
    "SinusoidalEmbedding": lambda: [nn.SinusoidalEmbedding(64, 32)()],
    "SinusoidalEmbedding2D": lambda: [nn.SinusoidalEmbedding2D(8, 6, 32)()],
    "RotaryEmbedding": lambda: list(nn.RotaryEmbedding(32, 64)()),
    "ijepa": lambda: [_sincos_2d(64, 14)],
    "vjepa_uniform": lambda: [_sincos_3d(64, (4, 7, 7), True)],
    "vjepa_weighted": lambda: [_sincos_3d(64, (4, 7, 7), False)],
    "generative_sincos": lambda: [sincos_position_embedding(64, 14)],
    "make_beta_schedule_linear": lambda: [make_beta_schedule(200, "linear")],
    "make_beta_schedule_scaled": lambda: [make_beta_schedule(200, "scaled_linear")],
    "make_beta_schedule_cosine": lambda: [make_beta_schedule(200, "cosine")],
    "make_sigma_schedule": lambda: [make_sigma_schedule(32, sigma_max=50.0, sigma_min=0.01)],
    "posterior_from_floats": lambda: list(
        diffusion_posterior_constants(beta_schedule_floats(200, "linear"))
    ),
    "ddpm_scheduler_linear": _ddpm_scheduler("linear"),
    "ddpm_scheduler_cosine": _ddpm_scheduler("cosine"),
    "stable_diffusion_scaled": _sd_noising_tables("scaled_linear"),
    "stable_diffusion_linear": _sd_noising_tables("linear"),
    "learned_variance_ddpm": _learned_variance_ddpm,
    "dit": _dit,
}


@pytest.mark.parametrize("build", _TABLES.values(), ids=_TABLES.keys())
def test_a_table_holds_the_same_numbers_under_either_default(
    build: Callable[[], list[Tensor]], metal_default: None
) -> None:
    on_metal = build()
    lucid.set_default_device("cpu")
    on_cpu = build()
    assert len(on_metal) == len(on_cpu)
    for metal, cpu in zip(on_metal, on_cpu, strict=True):
        assert (metal.device.type, cpu.device.type) == ("metal", "cpu")
        assert metal.dtype == cpu.dtype
        assert metal.to("cpu").numpy().tobytes() == cpu.numpy().tobytes()


def test_stable_diffusion_keeps_its_step_table_on_the_host(metal_default: None) -> None:
    """``step`` reads ᾱ at Python-int timesteps, which a device would stall."""
    scheduler = PNDMScheduler(StableDiffusionConfig())
    assert scheduler.betas.device.type == "cpu"
    assert scheduler.alphas_cumprod.device.type == "cpu"
    assert scheduler._ddim._signal_scale.device.type == "metal"
    assert scheduler._ddim._noise_scale.device.type == "metal"


def test_sincos_embedding_answers_on_the_device_of_its_positions(metal_default: None) -> None:
    on_metal = sincos_embedding(lucid.arange(6), 8)
    on_cpu = sincos_embedding(lucid.arange(6, device="cpu"), 8)
    assert (on_metal.device.type, on_cpu.device.type) == ("metal", "cpu")
    assert on_metal.to("cpu").numpy().tobytes() == on_cpu.numpy().tobytes()


# ── the helpers themselves ───────────────────────────────────────────────────

_PLACED: dict[str, Callable[..., Tensor]] = {
    "sinusoidal_embedding": lambda **kw: F.sinusoidal_embedding(8, 4, **kw),
    "sinusoidal_embedding_2d": lambda **kw: F.sinusoidal_embedding_2d(2, 2, 4, **kw),
    "make_beta_schedule": lambda **kw: make_beta_schedule(5, **kw),
    "make_sigma_schedule": lambda **kw: make_sigma_schedule(4, sigma_max=1.0, sigma_min=0.1, **kw),
    "posterior_constants": lambda **kw: (
        diffusion_posterior_constants([0.1, 0.2], **kw).log_variance
    ),
    "flow_prior_sample_gaussian": lambda **kw: flow_prior_sample("gaussian", (2, 2), **kw),
    "flow_prior_sample_logistic": lambda **kw: flow_prior_sample("logistic", (2, 2), **kw),
    "sample_point_coords": lambda **kw: sample_point_coords(3, **kw),
    "ddpm_scheduler": lambda **kw: DDPMScheduler(5, **kw).alphas_cumprod,
}


@pytest.mark.parametrize("make", _PLACED.values(), ids=_PLACED.keys())
def test_a_helper_follows_the_default_and_an_explicit_device_wins(
    make: Callable[..., Tensor], metal_default: None
) -> None:
    assert make().device.type == "metal"
    assert make(device="cpu").device.type == "cpu"
    lucid.set_default_device("cpu")
    assert make().device.type == "cpu"
    assert make(device="metal").device.type == "metal"


def test_a_model_with_no_parameters_samples_on_the_default_device(metal_default: None) -> None:
    assert resolve_generation_device(nn.Identity(), None) == "metal"
    assert resolve_generation_device(nn.Identity(), "cpu") == "cpu"


def test_no_helper_defaults_its_device_to_the_cpu() -> None:
    """A ``device`` left out means the default device, in every signature."""
    root = Path(lucid.__file__).parent
    offenders: list[str] = []
    for package in ("models", "nn"):
        for path in sorted((root / package).rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                spec = node.args
                positional = [*spec.posonlyargs, *spec.args]
                defaults: list[ast.expr | None] = [
                    *[None] * (len(positional) - len(spec.defaults)),
                    *spec.defaults,
                ]
                pairs = [
                    *zip(positional, defaults, strict=True),
                    *zip(spec.kwonlyargs, spec.kw_defaults, strict=True),
                ]
                for arg, default in pairs:
                    if (
                        arg.arg == "device"
                        and isinstance(default, ast.Constant)
                        and default.value == "cpu"
                    ):
                        offenders.append(f"{path.relative_to(root)}:{node.lineno} {node.name}")
    assert not offenders, offenders
