"""Every zoo family computes in the dtype it is handed (LCD-562).

DETR raised ``DtypeMismatch`` in float16, bfloat16 and float64: the first
input of its decoder was ``lucid.zeros(..., device=)`` — float32 whatever
the model was — and so was its sine position table.  The same defect ran
through two dozen families without raising.  Lucid widens a mixed-dtype
product rather than refusing it, so a float16 model that met one float32
tensor ran the rest of its forward in float32 and answered in float32:
MaskFormer, the R-CNNs, YOLO, every diffusion and flow family, the world
models.  A float64 one answered with float32 values where a float32 draw
or table met it first.

The rule is that a tensor a forward builds for itself — a start state, a
grid, a weight table, a noise draw, a zero fallback — takes the activations'
dtype as well as their device.  Position and time tables are computed at
float32 at least and cast once, the reference's convention.

One narrow and one wide dtype per device, since float16 detects every
float32 constant a float64 run would and more:

* **eval** — every family of the compiled-training matrix (its smallest
  factory, shrunk by the matrix's rules and further where that is cheap)
  runs one forward in float64 on the CPU and in float16 on Metal (on the
  CPU without one); every floating tensor it answers with has the
  input's dtype.
* **rollouts** — Stable Diffusion, DIAMOND and Genie, whose task forward
  is a rollout the matrix skips, are driven through that forward, their
  sampler and their parts in float64 on the CPU and float16 / bfloat16
  on Metal.
* **sampling** (LCD-597) — every image-generation family's ``generate``
  at two steps in float64 on the CPU and float16 / bfloat16 on Metal
  answers finite, in the model's dtype; likewise Score-SDE's three
  samplers and its likelihood, a DiT schedule whose float16 copy rounds
  ᾱ_1 to 1, DiT's three non-default conditionings, TVM's latent
  whitening, PlaNet's planner, and both schedulers' ``add_noise``,
  whose float32 tables are cast at use.
* **train** — every family of the training-step table, and the detection
  and segmentation losses that table leaves to other files, take their
  step in float16 on the CPU (the losses in float64 too); the loss and
  every gradient keep the model's dtype.
* DETR and MaskFormer, the reported pair, are pinned through their public
  factories against the float32 model's numbers; Score-SDE and Fast
  R-CNN, whose float64 answers carried float32 values, likewise.
"""

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.models as models
from lucid.models._registry import _REGISTRY
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.compile import _zoo_matrix as zoo
from lucid.test.unit.models.test_models_train_step import CASES, _family_of

_F16, _BF16, _F64 = lucid.float16, lucid.bfloat16, lucid.float64


def _name(dtype: lucid.dtype) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def _need(device: str) -> None:
    if device == "metal" and not metal_available():
        pytest.skip("Metal unavailable")


def _floating(value: Any, path: str = "output") -> list[tuple[str, lucid.Tensor]]:
    """Every floating tensor in a forward's answer, named for the message."""
    if isinstance(value, lucid.Tensor):
        return [(path, value)] if value.is_floating_point() else []
    if isinstance(value, (tuple, list)):
        return [t for i, v in enumerate(value) for t in _floating(v, f"{path}[{i}]")]
    if isinstance(value, dict):
        return [t for k, v in value.items() for t in _floating(v, f"{path}.{k}")]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [
            t
            for f in dataclasses.fields(value)
            for t in _floating(getattr(value, f.name, None), f"{path}.{f.name}")
        ]
    return []


def _cast(value: Any, device: str, dtype: lucid.dtype) -> Any:
    """Inputs on ``device``; floating ones in ``dtype``, integer ones untouched."""
    if isinstance(value, lucid.Tensor):
        value = value.to(device)
        return value.to(dtype) if value.is_floating_point() else value
    if isinstance(value, (tuple, list)):
        return type(value)(_cast(v, device, dtype) for v in value)
    if isinstance(value, dict):
        return {k: _cast(v, device, dtype) for k, v in value.items()}
    return value


def _wrong(answer: Any, dtype: lucid.dtype) -> list[str]:
    return [f"{n}: {_name(t.dtype)}" for n, t in _floating(answer) if t.dtype != dtype]


def _step_keeps(model: Any, args: Any, kwargs: Any, dtype: lucid.dtype) -> None:
    """Loss and every gradient of one training step are in ``dtype``."""
    loss = model(*_cast(args, "cpu", dtype), **_cast(kwargs, "cpu", dtype)).loss
    assert loss.dtype == dtype, f"a {_name(dtype)} model has a {loss.dtype} loss"
    loss.backward()
    grads = [(n, p.grad) for n, p in model.named_parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    wrong = [n for n, g in grads if g.dtype != dtype]
    assert not wrong, f"gradients not in {_name(dtype)}: {wrong[:6]}"


# -- every family of the matrix, one forward -----------------------------------

#: Families whose task forward is a rollout; the matrix skips them and the
#: component checks below cover them instead.
_ROLLOUTS = {f for f, spec in zoo.SPECS.items() if spec.skip is not None}

_PAIRS = [
    (family, task)
    for task in zoo.TASKS
    for family in zoo.families(task)
    if family not in _ROLLOUTS
]

_DETECTOR = {"backbone_layers": (1, 1, 1, 1), "fpn_out_channels": 32}

#: Cheaper than the matrix's own shrink where a dtype check loses nothing:
#: the matrix sizes some families for a numerically stable train-mode step.
_SMALLER: dict[str, dict[str, object]] = {
    "faster_rcnn": {**_DETECTOR, "rpn_post_nms_top_n": 10},
    "mask_rcnn": {
        **_DETECTOR,
        "rpn_post_nms_top_n": 10,
        "roi_representation": 64,
        "mask_hidden_channels": 16,
        "mask_num_convs": 1,
        "mask_predictor_hidden": 16,
    },
    "mask2former": {
        "swin_depths": (1, 1, 1, 1),
        "swin_embed_dim": 24,
        "swin_num_heads": (1, 1, 1, 1),
        "mask_feature_size": 32,
        "encoder_feedforward_dim": 32,
        "num_queries": 4,
    },
}
#: Image side where the matrix's is the network's native one.
_SIDE = {"inception": 96, "inception_resnet": 96, "inception_v4": 96, "xception": 96}

#: Half precision runs on Metal, or on the CPU where there is no Metal,
#: so a runner without a GPU still sweeps a narrow dtype.
_HALF_DEVICE = "metal" if metal_available() else "cpu"
_EVAL = [("cpu", _F64), (_HALF_DEVICE, _F16)]


@pytest.mark.parametrize(
    ("device", "dtype"), _EVAL, ids=[f"{d}-{_name(t)}" for d, t in _EVAL]
)
@pytest.mark.parametrize(("family", "task"), _PAIRS, ids=[f for f, _ in _PAIRS])
def test_a_forward_answers_in_the_dtype_it_is_given(
    family: str,
    task: str,
    device: str,
    dtype: lucid.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _need(device)
    monkeypatch.setattr(zoo, "DEV", device)
    if family in _SIDE:
        spec = dataclasses.replace(zoo.SPECS[family], size=_SIDE[family])
        monkeypatch.setitem(zoo.SPECS, family, spec)
    name = zoo.representative(family, task)
    assert name is not None
    cfg = _REGISTRY[name].default_config
    overrides = {
        **zoo.shrink(cfg),
        **zoo.SPECS.get(family, zoo.Spec()).overrides,
        **_SMALLER.get(family, {}),
    }
    lucid.manual_seed(0)
    model = models.create_model(name, **overrides).to(device).to(dtype).eval()
    cfg = dataclasses.replace(cfg, **overrides)
    args, kwargs = zoo.default_inputs(model, cfg, family, 1)
    with lucid.no_grad():
        answer = model(*_cast(args, device, dtype), **_cast(kwargs, device, dtype))
    assert _floating(answer), "the forward answered with no floating tensor"
    wrong = _wrong(answer, dtype)
    assert not wrong, f"a {_name(dtype)} {family} answered with {wrong[:6]}"


# -- the rollout families, component by component -------------------------------

_SD = {
    "sample_size": 32,
    "downsample_factor": 4,
    "vae_block_out_channels": (32, 64, 64),
    "unet_block_out_channels": (32, 64),
    "attention_head_dim": 32,
    "cross_attention_dim": 16,
    "context_length": 4,
}
_DIAMOND = {
    "sample_size": 16,
    "unet_channels": (8, 8),
    "unet_layers": (1, 1),
    "reward_channels": (8, 8),
    "reward_layers": (1, 1),
    "actor_channels": (8, 8),
    "actor_layers": (1, 1),
    "cond_dim": 16,
    "reward_cond_dim": 8,
    "reward_lstm_dim": 16,
    "actor_lstm_dim": 16,
    "num_actions": 4,
}
_GENIE = {
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
}

Component = Callable[[str, lucid.dtype], Callable[[], Any]]


def _sd_task(device: str, dtype: lucid.dtype) -> Any:
    lucid.manual_seed(0)
    model = models.create_model("stable_diffusion_gen", **_SD)
    return model.to(device).to(dtype).eval()


def _diamond_task(device: str, dtype: lucid.dtype) -> Any:
    lucid.manual_seed(0)
    model = models.create_model("diamond_world_model", **_DIAMOND)
    return model.to(device).to(dtype).eval()


def _like(device: str, dtype: lucid.dtype, *shape: int) -> lucid.Tensor:
    return lucid.randn(*shape).to(device).to(dtype)


def _history(m: Any, device: str, dtype: lucid.dtype) -> tuple[Any, Any]:
    """Two DIAMOND conditioning histories and the actions between them."""
    count = m.config.conditioning_frames
    actions = lucid.ones(2, count, dtype=lucid.int64).to(device)
    return _like(device, dtype, 2, count, 3, 16, 16), actions


def _sd_training(device: str, dtype: lucid.dtype) -> Callable[[], Any]:
    m = _sd_task(device, dtype)
    images, context = _like(device, dtype, 1, 3, 32, 32), _like(device, dtype, 1, 4, 16)
    return lambda: m(images, context)


def _sd_generate_ddim(device: str, dtype: lucid.dtype) -> Callable[[], Any]:
    m = _sd_task(device, dtype)
    context = _like(device, dtype, 1, 4, 16)
    return lambda: m.generate(context, num_inference_steps=2, sampler="ddim")


def _sd_generate_pndm_guided(device: str, dtype: lucid.dtype) -> Callable[[], Any]:
    """The released sampler, with classifier-free guidance's doubled batch."""
    m = _sd_task(device, dtype)
    context, empty = _like(device, dtype, 1, 4, 16), _like(device, dtype, 1, 4, 16)
    return lambda: m.generate(context, empty, num_inference_steps=2, sampler="pndm")


def _sd_unet_integer_step(device: str, dtype: lucid.dtype) -> Callable[[], Any]:
    """The step a sampler hands the U-Net: an integer tensor."""
    m = _sd_task(device, dtype).stable_diffusion
    latent, context = _like(device, dtype, 1, 4, 8, 8), _like(device, dtype, 1, 4, 16)
    step = lucid.tensor([10]).to(device)
    return lambda: m.unet(latent, step, context)


def _sd_vae(device: str, dtype: lucid.dtype) -> Callable[[], Any]:
    m = _sd_task(device, dtype).stable_diffusion
    images = _like(device, dtype, 1, 3, 32, 32)
    return lambda: m.decode_latent(m.encode_image(images))


def _diamond_world_rollout(device: str, dtype: lucid.dtype) -> Callable[[], Any]:
    """The task forward: imagine two steps, score the policy on them."""
    m = _diamond_task(device, dtype)
    frames, actions = _history(m.diamond, device, dtype)
    return lambda: m(frames, actions, horizon=2)


def _diamond_imagine_frame(device: str, dtype: lucid.dtype) -> Callable[[], Any]:
    m = _diamond_task(device, dtype).diamond
    frames, actions = _history(m, device, dtype)
    return lambda: m.imagine_frame(frames, actions, steps=2)


def _diamond_denoiser(device: str, dtype: lucid.dtype) -> Callable[[], Any]:
    m = _diamond_task(device, dtype).diamond
    frames, actions = _history(m, device, dtype)
    next_frame = _like(device, dtype, 2, 3, 16, 16)
    return lambda: m(frames, actions, next_frame)


def _diamond_reward_end(device: str, dtype: lucid.dtype) -> Callable[[], Any]:
    m = _diamond_task(device, dtype)
    frames, actions = _history(m.diamond, device, dtype)
    rewards = _like(device, dtype, 2, int(frames.shape[1]))
    ends = lucid.zeros(2, int(frames.shape[1])).to(device).to(dtype)
    return lambda: m.reward_end_loss(frames, actions, rewards, ends)


def _diamond_actor_critic(device: str, dtype: lucid.dtype) -> Callable[[], Any]:
    m = _diamond_task(device, dtype).diamond
    frame = _like(device, dtype, 2, 3, 16, 16)
    return lambda: m.step_actor_critic(frame)


def _genie_task(device: str, dtype: lucid.dtype) -> Any:
    lucid.manual_seed(0)
    model = models.create_model("genie_world_model", **_GENIE, maskgit_steps=2)
    return model.to(device).to(dtype).eval()


def _genie_rollout(device: str, dtype: lucid.dtype) -> Callable[[], Any]:
    """The task forward: play two frames from a one-frame prompt."""
    m = _genie_task(device, dtype)
    prompt = lucid.rand(1, 1, 3, 8, 8).to(device).to(dtype)
    actions = lucid.zeros(1, 2, dtype=lucid.int64).to(device)
    return lambda: m(prompt, actions)


def _genie_losses(device: str, dtype: lucid.dtype) -> Callable[[], Any]:
    m = _genie_task(device, dtype).genie
    video = lucid.rand(2, 4, 3, 8, 8).to(device).to(dtype)
    return lambda: m(video)


#: name -> (family, kind, builder).  ``task`` is the task model's own
#: forward, ``sampler`` a public sampling method, ``part`` a sub-module or
#: training term.  Every rollout family needs one of the first two.
_COMPONENTS: dict[str, tuple[str, str, Component]] = {
    "sd-training": ("stable_diffusion", "task", _sd_training),
    "sd-generate-ddim": ("stable_diffusion", "sampler", _sd_generate_ddim),
    "sd-generate-pndm-guided": (
        "stable_diffusion",
        "sampler",
        _sd_generate_pndm_guided,
    ),
    "sd-unet-integer-step": ("stable_diffusion", "part", _sd_unet_integer_step),
    "sd-vae": ("stable_diffusion", "part", _sd_vae),
    "diamond-world-rollout": ("diamond", "task", _diamond_world_rollout),
    "diamond-imagine-frame": ("diamond", "sampler", _diamond_imagine_frame),
    "diamond-denoiser": ("diamond", "part", _diamond_denoiser),
    "diamond-reward-end-loss": ("diamond", "part", _diamond_reward_end),
    "diamond-actor-critic": ("diamond", "part", _diamond_actor_critic),
    "genie-rollout": ("genie", "task", _genie_rollout),
    "genie-all-three-losses": ("genie", "part", _genie_losses),
}

_ROLLOUT_DTYPES = [("cpu", _F64), (_HALF_DEVICE, _F16), ("metal", _BF16)]


@pytest.mark.parametrize(
    ("device", "dtype"),
    _ROLLOUT_DTYPES,
    ids=[f"{d}-{_name(t)}" for d, t in _ROLLOUT_DTYPES],
)
@pytest.mark.parametrize("component", sorted(_COMPONENTS))
def test_a_rollout_family_component_answers_in_its_dtype(
    component: str, device: str, dtype: lucid.dtype
) -> None:
    _need(device)
    call = _COMPONENTS[component][2](device, dtype)
    with lucid.no_grad():
        answer = call()
    assert _floating(answer), "the component answered with no floating tensor"
    wrong = _wrong(answer, dtype)
    assert not wrong, f"a {_name(dtype)} {component} answered with {wrong[:6]}"


def test_every_family_is_swept_or_driven_through_its_own_entry_point() -> None:
    """A rollout family counts only when its task forward or sampler is run.

    Sub-module checks alone let Stable Diffusion's sampler and DIAMOND's
    imagination stay float32 while the family looked covered.
    """
    families = {e.family for e in _REGISTRY.values()}
    swept = {f for f, _ in _PAIRS}
    assert families - swept == _ROLLOUTS, sorted(families - swept - _ROLLOUTS)
    entry = {f for f, kind, _ in _COMPONENTS.values() if kind in ("task", "sampler")}
    assert entry == _ROLLOUTS, sorted(entry ^ _ROLLOUTS)


# -- sampling -----------------------------------------------------------------

#: On top of the matrix's shrink: a sampler runs the network once per step.
_GENERATE_SMALLER: dict[str, dict[str, object]] = {
    "ddpm": {
        "sample_size": 16,
        "channel_mult": (1, 2),
        "num_res_blocks": 1,
        "attention_resolutions": (),
        "num_train_timesteps": 20,
    },
    "ncsn": {
        "sample_size": 16,
        "channel_mult": (1, 2),
        "num_res_blocks": 1,
        "attention_resolutions": (),
        "num_noise_levels": 10,
        "langevin_steps": 1,
    },
    "neural_ode": {
        "sample_size": 4,
        "hidden_dim": 8,
        "num_blocks": 1,
        "solver": "euler",
    },
    "nice": {"hidden_dim": 32},
    "realnvp": {"sample_size": 8, "num_scales": 1, "residual_blocks": 1, "base_dim": 8},
    "score_sde": {
        "sample_size": 16,
        "channel_mult": (1, 2),
        "num_res_blocks": 1,
        "attention_resolutions": (),
    },
    "stable_diffusion": _SD,
    "vae": {"latent_dim": (8, 16, 32), "down_block_channels": (8, 16, 32)},
    "vqvae": {
        "num_embeddings": 16,
        "embedding_dim": 8,
        "hidden_channels": 16,
        "residual_hidden_channels": 8,
    },
}

_STEPPED = {"dit", "flow_matching", "mean_flow", "rectified_flow", "score_sde", "tvm"}


def _generate(family: str, model: Any, device: str, dtype: lucid.dtype) -> Any:
    """Two sampling steps through each family's own ``generate`` signature."""
    if family == "ddpm":
        return model.generate(model.make_scheduler(), num_inference_steps=2)
    if family == "stable_diffusion":
        context = lucid.randn(1, 4, 16).to(device).to(dtype)
        return model.generate(context, num_inference_steps=2, sampler="ddim")
    if family == "ncsn":
        return model.generate(1, langevin_steps=1)
    if family in _STEPPED:
        return model.generate(1, steps=2)
    return model.generate(1)


def _generator(family: str, device: str, dtype: lucid.dtype, **extra: Any) -> Any:
    """The family's sampler, shrunk, moved and cast."""
    name = zoo.representative(family, "image-generation")
    assert name is not None
    overrides = _GENERATE_SMALLER.get(family, {})
    if family != "stable_diffusion":
        cfg = _REGISTRY[name].default_config
        overrides = {
            **zoo.shrink(cfg),
            **zoo.SPECS.get(family, zoo.Spec()).overrides,
            **overrides,
        }
    lucid.manual_seed(0)
    model = models.create_model(name, **{**overrides, **extra})
    return model.to(device).to(dtype).eval()


def _sampled_in(answer: Any, dtype: lucid.dtype, what: str) -> None:
    """Every floating tensor ``answer`` holds is in ``dtype`` and finite."""
    assert _floating(answer), f"{what} answered with no floating tensor"
    wrong = _wrong(answer, dtype)
    assert not wrong, f"a {_name(dtype)} {what}: {wrong}"
    for path, t in _floating(answer):
        finite = np.isfinite(t.to("cpu").to(_F64).numpy()).all()
        assert finite, f"a {_name(dtype)} {what}: {path} is not finite"


@pytest.mark.parametrize(
    ("device", "dtype"),
    _ROLLOUT_DTYPES,
    ids=[f"{d}-{_name(t)}" for d, t in _ROLLOUT_DTYPES],
)
@pytest.mark.parametrize("family", zoo.families("image-generation"))
def test_generate_answers_in_the_model_dtype(
    family: str, device: str, dtype: lucid.dtype
) -> None:
    """The start noise, the step's own draws and its coefficients follow it.

    A float32 start widened the whole reverse chain of a half model, and
    two schedules fed half precision ``1 - e^{-z}`` and ``1 - ᾱ`` near 0:
    Score-SDE's sub-VP standard deviation came out 0 at ``t_min`` and its
    samples NaN, so the answer must be finite as well as narrow.
    """
    _need(device)
    model = _generator(family, device, dtype)
    with lucid.no_grad():
        answer = _generate(family, model, device, dtype)
    _sampled_in(answer, dtype, f"{family} generate")


@pytest.mark.parametrize("method", ["euler", "pc", "ode"])
def test_every_score_sde_sampler_answers_in_half(method: str) -> None:
    model = _generator("score_sde", _HALF_DEVICE, _F16)
    with lucid.no_grad():
        answer = model.generate(1, steps=2, method=method)
    _sampled_in(answer, _F16, f"score_sde {method}")


@pytest.mark.parametrize(
    ("device", "dtype"),
    _ROLLOUT_DTYPES,
    ids=[f"{d}-{_name(t)}" for d, t in _ROLLOUT_DTYPES],
)
def test_score_sde_log_likelihood_answers_in_the_data_dtype(
    device: str, dtype: lucid.dtype
) -> None:
    """Its divergence accumulator and times were float32 ``zeros`` / ``ones``."""
    _need(device)
    model = _generator("score_sde", device, dtype)
    side, channels = model.config.sample_size, model.config.in_channels
    x = lucid.randn(1, channels, side, side).to(device).to(dtype)
    assert model.log_likelihood(x, steps=2).dtype == dtype


def test_a_half_dit_samples_where_half_rounds_alpha_bar_to_one() -> None:
    """ᾱ_1 = 0.99977 is 1 in float16; the last DDIM step divided 0 by 0.

    The sampler now works its coefficients out at float32 from the config's
    schedule, not from the float16 buffer the model's cast left behind.
    """
    model = _generator("dit", _HALF_DEVICE, _F16, num_train_timesteps=4, beta_end=2e-4)
    assert float(model.alphas_cumprod[1].item()) == 1.0, "the premise moved"
    with lucid.no_grad():
        answer = model.generate(1, steps=4)
    _sampled_in(answer, _F16, "dit over its whole schedule")


@pytest.mark.parametrize(
    ("device", "dtype"),
    _ROLLOUT_DTYPES,
    ids=[f"{d}-{_name(t)}" for d, t in _ROLLOUT_DTYPES],
)
def test_tvm_latent_whitening_keeps_the_latent_dtype(
    device: str, dtype: lucid.dtype
) -> None:
    """Its per-channel mean and std were built at the default dtype."""
    _need(device)
    model = _generator("tvm", device, dtype)
    latent = lucid.randn(1, model.config.in_channels, 4, 4).to(device).to(dtype)
    whitened = model.from_vae_latent(latent)
    assert whitened.dtype == dtype
    assert model.to_vae_latent(whitened).dtype == dtype


def test_a_half_dit_training_step_noises_where_half_rounds_alpha_bar_to_one() -> None:
    """At t ∈ {0, 1} the float16 buffer's ``1 - ᾱ`` is 0, so x_t was x_0.

    A two-step schedule draws only those steps.  The float16 forward's
    x_t must carry the noise the float32 forward adds, to half precision.
    """
    extra = {"num_train_timesteps": 2, "beta_start": 1e-4, "beta_end": 1.1e-4}
    wide = _generator("dit", "cpu", lucid.float32, **extra)
    narrow = _generator("dit", "cpu", _F16, **extra)
    assert narrow.alphas_cumprod.tolist() == [1.0, 1.0], "the premise moved"
    side = wide.config.sample_size
    lucid.manual_seed(1)
    x0 = lucid.randn(2, wide.config.in_channels, side, side) * 0.1
    added = []
    for model, x in ((wide, x0), (narrow, x0.to(_F16))):
        seen: list[lucid.Tensor] = []
        handle = model.dit.register_forward_pre_hook(
            lambda _module, args: seen.append(args[0])
        )
        lucid.manual_seed(2)  # the same steps and the same noise, rounded
        with lucid.no_grad():
            model(x)
        handle.remove()
        added.append((seen[0].to(_F64) - x.to(_F64)).numpy())
    want, got = added
    scale = float(np.abs(want).max())
    assert scale > 0.0
    assert float(np.abs(got - want).max()) < _TOL[_F16] * scale


@pytest.mark.parametrize(
    "conditioning", ["adaln_zero", "adaln", "cross_attention", "in_context"]
)
def test_every_dit_conditioning_answers_in_half(conditioning: str) -> None:
    """The ungated adaLN block multiplied by a float32 ``ones`` gate.

    The matrix builds only the default adaLN-Zero, so the paper's other
    three designs are driven here.
    """
    model = _generator("dit", _HALF_DEVICE, _F16, conditioning=conditioning)
    side = model.config.sample_size
    latent = lucid.randn(1, model.config.in_channels, side, side)
    with lucid.no_grad():
        _sampled_in(model(latent.to(_HALF_DEVICE).to(_F16)), _F16, conditioning)
        _sampled_in(model.generate(1, steps=2), _F16, conditioning)


@pytest.mark.parametrize(
    ("device", "dtype"),
    _ROLLOUT_DTYPES,
    ids=[f"{d}-{_name(t)}" for d, t in _ROLLOUT_DTYPES],
)
def test_planet_plans_in_the_model_dtype(device: str, dtype: lucid.dtype) -> None:
    """The CEM search distribution was float32 ``zeros`` / ``ones``."""
    _need(device)
    from lucid.models.generative.planet import PlaNetConfig, PlaNetForWorldModeling

    lucid.manual_seed(0)
    config = PlaNetConfig(
        action_dim=2,
        stoch_size=4,
        deter_size=8,
        hidden_size=8,
        cnn_depth=4,
        reward_hidden=8,
    )
    model = PlaNetForWorldModeling(config).to(device).to(dtype).eval()
    start = model.planet.rssm.initial(1)
    with lucid.no_grad():
        action = model.plan(start, horizon=3, iterations=2, candidates=8, elites=2)
    _sampled_in(action, dtype, "planet plan")


def _ddpm_scheduler() -> Any:
    from lucid.models.generative import DDPMScheduler

    return DDPMScheduler(num_train_timesteps=50)


def _ddim_scheduler() -> Any:
    from lucid.models.generative.stable_diffusion import (
        DDIMScheduler,
        StableDiffusionConfig,
    )

    return DDIMScheduler(StableDiffusionConfig(num_train_timesteps=50))


_SCHEDULERS = {"ddpm": _ddpm_scheduler, "ddim": _ddim_scheduler}


@pytest.mark.parametrize(
    ("device", "dtype"),
    _ROLLOUT_DTYPES,
    ids=[f"{d}-{_name(t)}" for d, t in _ROLLOUT_DTYPES],
)
@pytest.mark.parametrize("scheduler", sorted(_SCHEDULERS))
def test_a_scheduler_noises_a_sample_in_its_dtype(
    scheduler: str, device: str, dtype: lucid.dtype
) -> None:
    """``add_noise(x16, …)`` answered float32: the tables are float32.

    The coefficients are cast at use, so the float32 answer is unchanged
    and a narrow one is the float32 one, rounded.
    """
    _need(device)
    lucid.manual_seed(0)
    clean, noise = lucid.randn(2, 3, 4, 4), lucid.randn(2, 3, 4, 4)
    steps = lucid.tensor([3, 41])
    wide = _SCHEDULERS[scheduler]().add_noise(clean, noise, steps)
    narrow = _SCHEDULERS[scheduler]().add_noise(
        clean.to(device).to(dtype), noise.to(device).to(dtype), steps.to(device)
    )
    assert narrow.dtype == dtype
    want = wide.to(_F64).numpy()
    got = narrow.to("cpu").to(_F64).numpy()
    assert float(np.abs(got - want).max()) < _TOL[dtype] * float(np.abs(want).max())


# -- a training step --------------------------------------------------------------


@pytest.mark.parametrize(
    ("family", "factory", "overrides", "make_inputs"),
    CASES,
    ids=[_family_of(c) for c in CASES],
)
def test_a_training_step_keeps_the_model_dtype(
    family: str,
    factory: str,
    overrides: dict[str, Any],
    make_inputs: Any,
) -> None:
    lucid.manual_seed(0)
    model = models.create_model(factory, **overrides).to(_F16)
    model.train()
    args, kwargs = make_inputs()
    _step_keeps(model, args, kwargs, _F16)


def _square(side: int, lo: int, hi: int) -> lucid.Tensor:
    """A ``(1, side, side)`` binary mask holding one square."""
    grid = [
        [1.0 if lo <= r < hi and lo <= c < hi else 0.0 for c in range(side)]
        for r in range(side)
    ]
    return lucid.tensor([grid])


def _boxes(side: int, *, masks: bool = False) -> list[dict[str, lucid.Tensor]]:
    """One pixel-space box per image, and its mask when the head predicts one."""
    lo, hi = side // 4, 3 * side // 4
    target = {
        "boxes": lucid.tensor([[float(lo), float(lo), float(hi), float(hi)]]),
        "labels": lucid.tensor([1]),
    }
    if masks:
        target["masks"] = _square(side, lo, hi)
    return [target]


def _segments(side: int) -> list[dict[str, lucid.Tensor]]:
    return [{"labels": lucid.tensor([1]), "masks": _square(side, side // 8, side // 2)}]


#: The losses the training-step table leaves to their own files: the RPN /
#: box / mask heads of the R-CNNs, Mask2Former's matched point loss, and the
#: YOLO losses other than v3's.
_LOSSES: list[tuple[str, dict[str, object], int, Callable[[], Any]]] = [
    (
        "faster_rcnn_resnet50_fpn",
        {
            "num_classes": 3,
            **_DETECTOR,
            "rpn_post_nms_top_n": 10,
            "roi_representation_size": 64,
        },
        64,
        lambda: _boxes(64),
    ),
    (
        "mask_rcnn_resnet50_fpn",
        {"num_classes": 3, **_SMALLER["mask_rcnn"]},
        64,
        lambda: _boxes(64, masks=True),
    ),
    (
        "mask2former_swin_tiny",
        {
            "num_classes": 3,
            **_SMALLER["mask2former"],
            "d_model": 32,
            "n_head": 2,
            "num_encoder_layers": 1,
            "num_decoder_layers": 3,
            "dim_feedforward": 32,
            "train_num_points": 64,
        },
        64,
        lambda: _segments(64),
    ),
    ("yolo_tiny", {"num_classes": 4}, 128, lambda: _boxes(128)),
    ("yolo_v2", {"num_classes": 4}, 64, lambda: _boxes(64)),
    ("yolo_v4", {"num_classes": 4}, 64, lambda: _boxes(64)),
]


@pytest.mark.parametrize("dtype", [_F64, _F16], ids=_name)
@pytest.mark.parametrize(
    ("factory", "overrides", "side", "targets"),
    _LOSSES,
    ids=[factory for factory, *_ in _LOSSES],
)
def test_a_detection_loss_keeps_the_model_dtype(
    factory: str,
    overrides: dict[str, object],
    side: int,
    targets: Callable[[], Any],
    dtype: lucid.dtype,
) -> None:
    lucid.manual_seed(0)
    model = models.create_model(factory, **overrides).to(dtype).train()
    _step_keeps(model, (lucid.randn(1, 3, side, side),), {"targets": targets()}, dtype)


# -- the reported pair, through the public factories --------------------------

_SMALL = {
    "detr_resnet50": {
        "num_classes": 4,
        "backbone_layers": (1, 1, 1, 1),
        "n_head": 2,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "dim_feedforward": 32,
        "num_queries": 4,
    },
    "maskformer_resnet50": {
        "num_classes": 4,
        "backbone_layers": (1, 1, 1, 1),
        "n_head": 2,
        "num_decoder_layers": 1,
        "dim_feedforward": 32,
        "num_queries": 4,
        "fpn_out_channels": 32,
    },
}

_INFER = [
    ("cpu", _F16),
    ("cpu", _BF16),
    ("cpu", _F64),
    ("metal", _F16),
    ("metal", _BF16),
]

#: Largest error against the float32 model, relative to its largest output.
_TOL = {_F64: 1e-4, _F16: 5e-2, _BF16: 2e-1}


def _paired(factory: str, device: str, dtype: lucid.dtype) -> tuple[Any, Any]:
    lucid.manual_seed(0)
    wide = models.create_model(factory, **_SMALL[factory]).eval()
    narrow = models.create_model(factory, **_SMALL[factory]).eval()
    narrow.load_state_dict(wide.state_dict())
    return wide.to(device), narrow.to(device).to(dtype)


@pytest.mark.parametrize(
    ("device", "dtype"), _INFER, ids=[f"{d}-{_name(t)}" for d, t in _INFER]
)
@pytest.mark.parametrize("factory", sorted(_SMALL))
def test_detr_and_maskformer_infer_in_reduced_and_double_precision(
    factory: str, device: str, dtype: lucid.dtype
) -> None:
    _need(device)
    wide, narrow = _paired(factory, device, dtype)
    lucid.manual_seed(1)
    x = lucid.randn(1, 3, 64, 64).to(device)
    with lucid.no_grad():
        want = wide(x)
        got = narrow(x.to(dtype))
    assert not _wrong(got, dtype), _wrong(got, dtype)
    for (name, g), (_, w) in zip(_floating(got), _floating(want), strict=True):
        a = g.to("cpu").to(lucid.float64).numpy()
        b = w.to("cpu").to(lucid.float64).numpy()
        assert np.isfinite(a).all(), name
        scale = max(float(np.abs(b).max()), 1e-6)
        err = float(np.abs(a - b).max()) / scale
        assert err < _TOL[dtype], f"{name}: {err:.3g} off the float32 model"


# -- float64 answers that carried float32 values --------------------------------


def _shrunk(family: str, task: str) -> tuple[str, dict[str, Any]]:
    name = zoo.representative(family, task)
    assert name is not None
    overrides = zoo.shrink(_REGISTRY[name].default_config)
    overrides.update(zoo.SPECS.get(family, zoo.Spec()).overrides)
    return name, overrides


def _float64_twin(name: str, overrides: dict[str, Any]) -> tuple[Any, Any]:
    lucid.manual_seed(0)
    narrow = models.create_model(name, **overrides).eval()
    wide = models.create_model(name, **overrides).eval()
    wide.load_state_dict(narrow.state_dict())
    return narrow, wide.to(_F64)


def _close(got: lucid.Tensor, want: lucid.Tensor, tol: float, what: str) -> None:
    assert got.dtype == _F64, f"{what} is {got.dtype}"
    a, b = got.numpy(), want.to(_F64).numpy()
    assert a.shape == b.shape, what
    scale = max(float(np.abs(b).max()), 1e-12)
    assert float(np.abs(a - b).max()) / scale < tol, what


def test_score_sde_float64_draws_its_target_and_noise_scale_in_float64() -> None:
    """``target`` was a float32 draw and ``std`` a float32 schedule value."""
    narrow, wide = _float64_twin(*_shrunk("score_sde", "image-generation"))
    lucid.manual_seed(1)
    x = lucid.randn(2, 3, 32, 32)
    lucid.manual_seed(2)
    want = narrow(x)
    lucid.manual_seed(2)
    got = wide(x.to(_F64))
    # The same seed draws the same numbers in either dtype, up to float32's
    # own error in the normal's tails, so the float64 answer is the float32
    # one, widened — not float32 values handed over as float64.
    _close(got.target, want.target, 1e-4, "target")
    _close(got.std, want.std, 1e-5, "std")
    _close(got.score, want.score, 1e-3, "score")
    _close(got.loss, want.loss, 1e-3, "loss")


def test_fast_rcnn_float64_keeps_its_dtype_with_and_without_proposals() -> None:
    """With no proposals the empty stand-in was a float32 ``zeros``."""
    narrow, wide = _float64_twin(*_shrunk("fast_rcnn", "object-detection"))
    lucid.manual_seed(1)
    x = lucid.randn(1, 3, 64, 64)
    with lucid.no_grad():
        empty = wide(x.to(_F64))
        assert not _wrong(empty, _F64), _wrong(empty, _F64)
        boxes = [lucid.tensor([[4.0, 4.0, 40.0, 40.0], [10.0, 12.0, 60.0, 50.0]])]
        want = narrow(x, proposals=boxes)
        got = wide(x.to(_F64), proposals=[b.to(_F64) for b in boxes])
    _close(got.logits, want.logits, 1e-4, "logits")
    _close(got.pred_boxes, want.pred_boxes, 1e-4, "pred_boxes")
