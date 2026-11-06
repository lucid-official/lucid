"""A rectangular ``sample_size`` is kept or refused, never squared (LCD-893).

``sample_size`` is declared ``int | tuple[int, int]`` on the base every
image-generative family shares.  A family either builds the rectangle it was
given or refuses it where the config is validated; reading ``sample_size[0]``
and carrying on built a square model with no word said.

The table below is the one place that records each family's stance.  A test
fails if a config class on the substrate is missing from it, so a new family
has to choose.
"""

import pytest

import lucid
from lucid.models import create_model
from lucid.models.generative._common._config import (
    GenerativeModelConfig,
    patch_grid_side,
    square_side,
)
from lucid.models.generative.diamond import DIAMONDConfig, DIAMONDModel
from lucid.models.generative.drifting import DriftingConfig
from lucid.models.generative.dit import DiTConfig
from lucid.models.generative.dreamer import DreamerConfig
from lucid.models.generative.dreamer_v2 import DreamerV2Config
from lucid.models.generative.dreamer_v3 import DreamerV3Config
from lucid.models.generative.genie import GenieConfig
from lucid.models.generative.mean_flow import MeanFlowConfig
from lucid.models.generative.planet import PlaNetConfig
from lucid.models.generative.tvm import TVMConfig

_RECT = (16, 32)

# Families whose network is built over a square grid: the config refuses.
_SQUARE_ONLY: dict[str, type[GenerativeModelConfig]] = {
    "dit": DiTConfig,
    "mean_flow": MeanFlowConfig,
    "tvm": TVMConfig,
    "drifting": DriftingConfig,
    "dreamer": DreamerConfig,
    "dreamer_v2": DreamerV2Config,
    "dreamer_v3": DreamerV3Config,
    "planet": PlaNetConfig,
}

# Families that carry the rectangle through to what they generate:
# (factory, overrides that make it small, ``generate`` keywords).
_RECTANGULAR: dict[str, tuple[str, dict[str, object], dict[str, object]]] = {
    "vae": ("vae_gen", dict(latent_dim=8, down_block_channels=(8, 16)), dict(n_samples=1)),
    "vqvae": (
        "vqvae_gen",
        dict(hidden_channels=16, residual_hidden_channels=16, embedding_dim=8, num_embeddings=16),
        dict(n_samples=1),
    ),
    "ncsn": (
        "ncsn_cifar_gen",
        dict(
            base_channels=16,
            channel_mult=(1, 2),
            num_res_blocks=1,
            resnet_groups=4,
            num_noise_levels=3,
            langevin_steps=1,
        ),
        dict(n_samples=1),
    ),
    "score_sde": (
        "score_sde_vp_gen",
        dict(
            base_channels=8,
            channel_mult=(1,),
            num_res_blocks=1,
            resnet_groups=4,
            attention_resolutions=(),
        ),
        dict(n_samples=1, steps=2),
    ),
    "rectified_flow": (
        "rectified_flow_cifar_gen",
        dict(base_channels=16, channel_mult=(1, 2), num_res_blocks=1, attention_resolutions=()),
        dict(n_samples=1, steps=2),
    ),
    "flow_matching": (
        "flow_matching_cifar_gen",
        dict(
            base_channels=16,
            resnet_groups=4,
            channel_mult=(1, 2),
            num_res_blocks=1,
            attention_resolutions=(),
        ),
        dict(n_samples=1, steps=2),
    ),
    "neural_ode": ("neural_ode_gen", dict(hidden_dim=8, num_blocks=1), dict(n_samples=1)),
    "realnvp": (
        "realnvp_cifar_gen",
        dict(num_scales=2, residual_blocks=1, base_dim=8),
        dict(n_samples=1),
    ),
}

# Families with no ``generate`` that a rectangle reaches; the other tests in
# this file cover what they build.
_CONFIG_ONLY = {"ddpm", "diamond", "genie", "stable_diffusion", "nice"}


def _config_classes() -> list[type[GenerativeModelConfig]]:
    found: list[type[GenerativeModelConfig]] = []
    stack = list(GenerativeModelConfig.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if cls.__module__.startswith("lucid.models.generative.") and "._common." not in (
            cls.__module__
        ):
            found.append(cls)
    return found


def test_every_config_on_the_substrate_declares_a_stance() -> None:
    known = {c for c in _SQUARE_ONLY.values()}
    family_of = {c: c.__module__.split(".")[3] for c in _config_classes()}
    unplaced = {
        family
        for cls, family in family_of.items()
        if cls not in known and family not in _RECTANGULAR and family not in _CONFIG_ONLY
    }
    assert not unplaced, f"add a stance for {sorted(unplaced)}"


@pytest.mark.parametrize("family", sorted(_SQUARE_ONLY))
def test_a_square_only_family_refuses_a_rectangle_by_name(family: str) -> None:
    config_class = _SQUARE_ONLY[family]
    with pytest.raises(ValueError, match=r"sample_size.*\(16, 32\)"):
        config_class(sample_size=_RECT)


@pytest.mark.parametrize("family", ["dit", "mean_flow", "tvm", "drifting"])
def test_a_square_only_family_still_takes_a_tuple_with_equal_sides(family: str) -> None:
    config = _SQUARE_ONLY[family](sample_size=(16, 16))
    assert config.sample_size == (16, 16)
    assert patch_grid_side(config.sample_size) == 16
    assert config.num_patches == _SQUARE_ONLY[family](sample_size=16).num_patches


@pytest.mark.parametrize("family", ["dit", "mean_flow", "tvm"])
def test_a_backbone_refuses_a_depth_that_is_not_positive(family: str) -> None:
    with pytest.raises(ValueError, match="depth must be positive"):
        _SQUARE_ONLY[family](depth=0)


def test_dit_builds_and_generates_over_its_square_latent() -> None:
    model = create_model(
        "dit_small_2_gen", sample_size=(16, 16), hidden_size=32, depth=1, num_heads=2, num_classes=3
    ).eval()
    with lucid.no_grad():
        samples = model.generate(n_samples=1, steps=2).samples
    assert tuple(samples.shape[-2:]) == (16, 16)


def test_genie_reads_its_frame_shape_from_the_base_pair() -> None:
    assert GenieConfig(sample_size=_RECT).frame_shape == _RECT
    assert GenieConfig(sample_size=32).frame_shape == (32, 32)


@pytest.mark.parametrize("family", sorted(_RECTANGULAR))
def test_a_rectangular_family_generates_the_shape_it_was_given(family: str) -> None:
    name, overrides, keywords = _RECTANGULAR[family]
    model = create_model(name, sample_size=_RECT, **overrides).eval()
    with lucid.no_grad():
        samples = model.generate(**keywords).samples
    assert tuple(samples.shape[-2:]) == _RECT


def test_ddpm_denoises_a_rectangle_without_squaring_it() -> None:
    model = create_model(
        "ddpm_cifar",
        sample_size=_RECT,
        base_channels=16,
        channel_mult=(1, 2),
        num_res_blocks=1,
        resnet_groups=4,
        attention_resolutions=(),
    ).eval()
    x = lucid.randn((1, 3, *_RECT))
    with lucid.no_grad():
        out = model(x, lucid.tensor([1]))
    assert tuple(out.sample.shape) == (1, 3, *_RECT)


def test_a_rectangular_frame_reaches_every_part_of_diamond_with_its_agent() -> None:
    """The reward model and actor flatten a feature map: both sides count."""
    config = DIAMONDConfig(
        sample_size=(30, 56),
        unet_channels=(8, 8),
        unet_layers=(1, 1),
        reward_channels=(8, 8, 8),
        reward_layers=(1, 1, 1),
        actor_channels=(8, 8, 8),
        actor_layers=(1, 1, 1),
        cond_dim=16,
        reward_cond_dim=8,
        reward_lstm_dim=16,
        actor_lstm_dim=16,
        num_actions=4,
        horizon=3,
    )
    assert config.reward_feature_hw == (8, 14)
    assert config.actor_feature_hw == (3, 7)
    model = DIAMONDModel(config).eval()
    assert model.reward_end is not None and model.actor_critic is not None
    frames = lucid.randn((2, config.reward_frames * 3, 30, 56))
    with lucid.no_grad():
        reward, _, _ = model.reward_end(frames, lucid.zeros((2,), dtype=lucid.int64))
        logits, _, _ = model.actor_critic(lucid.randn((2, 3, 30, 56)))
    assert tuple(reward.shape) == (2, 3)
    assert tuple(logits.shape) == (2, 4)


def test_a_square_diamond_frame_off_the_stride_still_fits_its_reward_lstm() -> None:
    """A stride-2 convolution rounds up: 30 halves to 15, then 8, not 7."""
    config = DIAMONDConfig(
        sample_size=30,
        unet_channels=(8, 8),
        unet_layers=(1, 1),
        reward_channels=(8, 8, 8),
        reward_layers=(1, 1, 1),
        actor_channels=(8, 8),
        actor_layers=(1, 1),
        cond_dim=16,
        reward_cond_dim=8,
        reward_lstm_dim=16,
        actor_lstm_dim=16,
        num_actions=4,
        horizon=3,
    )
    assert config.reward_feature_hw == (8, 8)
    model = DIAMONDModel(config).eval()
    assert model.reward_end is not None
    with lucid.no_grad():
        model.reward_end(
            lucid.randn((1, config.reward_frames * 3, 30, 30)), lucid.zeros((1,), dtype=lucid.int64)
        )


def test_square_side_names_the_field_the_value_and_the_reason() -> None:
    assert square_side(8, "x") == 8
    assert square_side((8, 8), "x") == 8
    with pytest.raises(ValueError, match=r"sample_size must be square — because, got \(8, 16\)"):
        square_side((8, 16), "because")
    with pytest.raises(ValueError, match="sample_size"):
        patch_grid_side((8, 16))


def test_sample_hw_is_the_pair_whichever_way_it_was_written() -> None:
    assert GenerativeModelConfig(sample_size=8).sample_hw == (8, 8)
    assert GenerativeModelConfig(sample_size=(8, 16)).sample_hw == (8, 16)
