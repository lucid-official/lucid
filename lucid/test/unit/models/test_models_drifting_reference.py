"""Drifting Models — fixed-input reference values for the generator and the feature encoders.

The expected values below are **transcribed from the official release, not produced by running
it**: its formulas (``models/generator.py``, ``models/mae_model.py``, ``models/convnext.py``) were
written out by hand in the reference framework, in float64, over the same fixed parameters and
inputs these tests build.  Antialiased resizing, which the release takes from ``jax.image.resize``,
is the reference framework's in that transcription.

Parameters and inputs come from Python's ``math``, not a tensor op or the RNG, so the CPU, Metal and
the transcription all start from the same float32 bits.  A value is a :func:`digest`, so a map
transposed or a pair swapped moves it where a mean would not.
"""

import math
from collections.abc import Sequence

import pytest

import lucid
import lucid.nn as nn
from lucid.models.generative.drifting import DriftingConfig, DriftingModel
from lucid.models.generative.drifting._feature_encoder import (
    _ConvNeXtV2,
    _DriftingFeatureEncoder,
)


def fixed_values(count: int, frequency: float, phase: float) -> list[float]:
    """``sin(frequency * i + phase)`` for ``i`` in ``range(count)``."""
    return [math.sin(frequency * i + phase) for i in range(count)]


def fixed_tensor(shape: tuple[int, ...], frequency: float, phase: float) -> lucid.Tensor:
    """:func:`fixed_values` as a float32 tensor of ``shape``."""
    values = fixed_values(math.prod(shape), frequency, phase)
    return lucid.tensor(values, dtype=lucid.float32).reshape(shape)


def fill_parameters(module: nn.Module, amplitude: float = 0.3) -> None:
    """Give every parameter a fixed value set by its place in ``named_parameters``."""
    with lucid.no_grad():
        for index, (_, parameter) in enumerate(module.named_parameters()):
            values = fixed_tensor(tuple(parameter.shape), 0.37 + 0.011 * index, 0.5 * index)
            parameter.copy_(values.to(parameter.device.type) * amplitude)


def digest(values: Sequence[float], parts: int = 1) -> list[float]:
    """``parts`` order-sensitive sums over ``values``, split evenly and weighted by position.

    Summed in Python, in double precision, so no device's rounding enters the sum.
    """
    size = len(values) // parts
    weights = fixed_values(size, 0.73, 0.3)
    return [
        math.fsum(v * w for v, w in zip(values[p * size : (p + 1) * size], weights, strict=True))
        for p in range(parts)
    ]


def flat_values(tensor: lucid.Tensor) -> list[float]:
    """The tensor's elements in row-major order, as Python floats."""
    values: list[float] = tensor.detach().to(lucid.float32).reshape(-1).tolist()
    return values


# ── The generator ────────────────────────────────────────────────────────────
GENERATOR_CONFIGS: dict[str, dict[str, object]] = {
    # The released architecture, at test size.
    "lightning": dict(
        sample_size=8,
        hidden_size=32,
        cond_dim=32,
        depth=2,
        num_heads=4,
        num_classes=10,
        noise_classes=4,
        noise_coords=2,
        n_cls_tokens=2,
        use_mae=False,
    ),
    # Every modernisation off: LayerNorm, the GELU MLP, no RoPE, no tokens, no styles.
    "plain": dict(
        sample_size=8,
        hidden_size=32,
        cond_dim=32,
        depth=2,
        num_heads=4,
        num_classes=10,
        noise_classes=0,
        noise_coords=2,
        n_cls_tokens=0,
        use_qknorm=False,
        use_swiglu=False,
        use_rope=False,
        use_rmsnorm=False,
        use_mae=False,
    ),
    # QK-norm as an affine LayerNorm, with RoPE.
    "qk_layernorm": dict(
        sample_size=8,
        hidden_size=32,
        cond_dim=32,
        depth=1,
        num_heads=4,
        num_classes=10,
        noise_classes=4,
        noise_coords=2,
        n_cls_tokens=2,
        use_swiglu=False,
        use_rmsnorm=False,
        use_mae=False,
    ),
    # A pixel-style patch, with a conditioning width of its own.
    "pixel_patch": dict(
        sample_size=64,
        patch_size=16,
        in_channels=3,
        out_channels=3,
        hidden_size=32,
        cond_dim=48,
        depth=1,
        num_heads=4,
        num_classes=10,
        noise_classes=4,
        noise_coords=2,
        n_cls_tokens=2,
        use_mae=False,
    ),
}

GENERATOR_LABELS = [1, 9]
GENERATOR_GUIDANCE = [1.0, 2.5]
GENERATOR_STYLES = [[0, 3], [2, 1]]

# name -> (conditioning digest per row, sample digest per row and channel).
GENERATOR_REFERENCE: dict[str, tuple[list[float], list[float]]] = {
    "lightning": (
        [0.5778608964, 1.749669516],
        [
            -0.4044339678,
            -0.2809417873,
            0.6089722045,
            -0.2193447076,
            -0.08896905337,
            -0.4171968258,
            0.4806777091,
            0.09314554102,
        ],
    ),
    "plain": (
        [-0.3167496862, 0.2649702719],
        [
            -0.6210542575,
            0.1489787875,
            0.2756152579,
            -0.3860274512,
            0.0344208577,
            -0.4848144176,
            0.5381882603,
            -0.1325708337,
        ],
    ),
    "qk_layernorm": (
        [0.5778608964, 1.749669516],
        [
            -0.05110994538,
            0.07312849168,
            -0.5106808704,
            0.7324002125,
            -0.1937534171,
            0.02525995025,
            -0.2801907351,
            0.3572837248,
        ],
    ),
    "pixel_patch": (
        [0.8918994634, 1.107934932],
        [-6.481504685, 5.504460267, 3.610570805, -6.209533962, 5.225294344, 3.397642899],
    ),
}

# ── The feature encoders ─────────────────────────────────────────────────────
MAE_CONFIGS: dict[str, dict[str, object]] = {
    "latent": dict(
        sample_size=16,
        hidden_size=32,
        cond_dim=32,
        depth=1,
        num_heads=4,
        mae_base_channels=8,
        mae_layers=(1, 2, 1, 1),
    ),
    # A pooling side that fits only some maps, and every block's output.
    "every_block": dict(
        sample_size=16,
        hidden_size=32,
        cond_dim=32,
        depth=1,
        num_heads=4,
        mae_base_channels=8,
        mae_layers=(2, 1, 1, 1),
        every_k_block=1,
        patch_mean_size=(2,),
        patch_std_size=(4,),
        use_std=False,
    ),
    # Space-to-depth over a pixel sample.
    "pixel": dict(
        sample_size=64,
        patch_size=4,
        in_channels=3,
        out_channels=3,
        hidden_size=32,
        cond_dim=32,
        depth=1,
        num_heads=4,
        mae_base_channels=8,
        mae_layers=(1, 1, 1, 1),
        mae_input_patch_size=8,
    ),
}

# name -> {feature set -> digest}, in the release's order.
MAE_REFERENCE: dict[str, dict[str, float]] = {
    "latent": {
        "norm_x": 0.122055178,
        "conv1": 4.679976026,
        "conv1_mean": 1.381509127,
        "conv1_std": 1.088956524,
        "conv1_mean_2": -0.7891482378,
        "conv1_mean_4": 2.478042365,
        "conv1_std_2": 0.006283764213,
        "conv1_std_4": 2.084317265,
        "layer1": -41.86733405,
        "layer1_mean": -2.228053039,
        "layer1_std": 0.02358802081,
        "layer1_mean_2": -2.136315147,
        "layer1_mean_4": -3.182354285,
        "layer1_std_2": -1.931033476,
        "layer1_std_4": 0.6534169137,
        "layer2": -5.358224483,
        "layer2_mean": 0.4025284025,
        "layer2_std": 0.5536076694,
        "layer2_mean_2": 5.057649131,
        "layer2_mean_4": -0.8112981414,
        "layer2_std_2": 1.849650796,
        "layer2_std_4": 0.3246272297,
        "layer3": -0.3843138172,
        "layer3_mean": 0.3762763049,
        "layer3_std": 0.2920240923,
        "layer3_mean_2": 0.7238323226,
        "layer3_mean_4": 0.3762763049,
        "layer3_std_2": -0.6634564507,
        "layer3_std_4": 0.2920240923,
        "layer4": 1.237609062,
        "layer4_mean": 0.6154962085,
        "layer4_std": -0.288103553,
        "layer4_mean_2": 0.6154962085,
        "layer4_std_2": -0.288103553,
        "layer2_blk2": 1.872977555,
        "layer2_blk2_mean": 1.108253823,
        "layer2_blk2_std": 1.052885579,
        "layer2_blk2_mean_2": 4.547062729,
        "layer2_blk2_mean_4": -0.7696782781,
        "layer2_blk2_std_2": 0.5754083281,
        "layer2_blk2_std_4": 0.645181217,
    },
    "every_block": {
        "norm_x": 0.122055178,
        "conv1": 4.679976026,
        "conv1_mean": 1.381509127,
        "conv1_mean_2": -0.7891482378,
        "conv1_std_4": 2.084317265,
        "layer1": -10.48223785,
        "layer1_mean": -2.228053039,
        "layer1_mean_2": 1.015304449,
        "layer1_std_4": -1.035846908,
        "layer2": -1.577577672,
        "layer2_mean": 0.4025284025,
        "layer2_mean_2": 3.19116646,
        "layer2_std_4": 1.876565661,
        "layer3": -1.479837999,
        "layer3_mean": 0.3762763049,
        "layer3_mean_2": 1.418997576,
        "layer3_std_4": 0.1984511858,
        "layer4": 0.9167954846,
        "layer4_mean": 0.6228191519,
        "layer4_mean_2": 0.6228191519,
        "layer1_blk1": 33.7719724,
        "layer1_blk1_mean": 0.2044025041,
        "layer1_blk1_mean_2": 2.899572982,
        "layer1_blk1_std_4": 3.087761216,
        "layer1_blk2": 36.57256138,
        "layer1_blk2_mean": 2.284367325,
        "layer1_blk2_mean_2": 5.287836518,
        "layer1_blk2_std_4": 3.634033278,
        "layer2_blk1": -16.00867647,
        "layer2_blk1_mean": -1.765603067,
        "layer2_blk1_mean_2": -1.150607041,
        "layer2_blk1_std_4": 0.05140873711,
        "layer3_blk1": -0.6292277125,
        "layer3_blk1_mean": -2.505912756,
        "layer3_blk1_mean_2": -0.5203468037,
        "layer3_blk1_std_4": -1.630181811,
        "layer4_blk1": -1.221095646,
        "layer4_blk1_mean": 0.8554960165,
        "layer4_blk1_mean_2": 0.8554960165,
    },
    "pixel": {
        "norm_x": 1.771973499,
        "conv1": -0.4858776171,
        "conv1_mean": 1.451933704,
        "conv1_std": 0.9047089147,
        "conv1_mean_2": 2.024572958,
        "conv1_mean_4": 0.7192612739,
        "conv1_std_2": 0.6495657514,
        "conv1_std_4": 1.364274033,
        "layer1": -2.879569381,
        "layer1_mean": 2.278416575,
        "layer1_std": 0.06679860403,
        "layer1_mean_2": 2.750446659,
        "layer1_mean_4": 2.208025236,
        "layer1_std_2": -1.523223692,
        "layer1_std_4": 0.3991869703,
        "layer2": -1.638161663,
        "layer2_mean": 1.164555318,
        "layer2_std": 0.5132032285,
        "layer2_mean_2": 0.6297251692,
        "layer2_mean_4": 1.164555318,
        "layer2_std_2": 0.1022496476,
        "layer2_std_4": 0.5132032285,
        "layer3": -2.169506396,
        "layer3_mean": -3.539636001,
        "layer3_std": 0.7237301064,
        "layer3_mean_2": -3.539636001,
        "layer3_std_2": 0.7237301064,
        "layer4": 0.6937313568,
        "layer4_mean": 0.6937313568,
        "layer4_std": 0.0004975338582,
    },
}

CONVNEXT_DEPTHS = (1, 1, 2, 1)
CONVNEXT_DIMS = (8, 16, 32, 64)
# Pixel-model configuration, read at 256: the resize to 224 has to shrink for antialiasing to count.
CONVNEXT_CONFIG: dict[str, object] = dict(
    sample_size=256,
    patch_size=16,
    in_channels=3,
    out_channels=3,
    hidden_size=32,
    cond_dim=32,
    depth=1,
    num_heads=4,
    use_mae=False,
    use_convnext=True,
)

# {feature set -> digest}, in the release's order, "global" first.
CONVNEXT_REFERENCE: dict[str, float] = {
    "global": 1.924526338,
    "convenxt_stage_0_mean": 1.946997959,
    "convenxt_stage_0_std": 0.2596931126,
    "convenxt_stage_1": -0.5841841433,
    "convenxt_stage_1_mean": -0.560075296,
    "convenxt_stage_1_std": 0.007132238128,
    "convenxt_stage_2": 1.458192631,
    "convenxt_stage_2_mean": 1.314264164,
    "convenxt_stage_2_std": 0.02378336989,
    "convenxt_stage_3": -0.1488527842,
    "convenxt_stage_3_mean": -0.1403659589,
    "convenxt_stage_3_std": 0.01530570822,
    "global_mean": 0.06716688732,
    "global_std": 0.01530570822,
}


def generator_outputs(name: str) -> tuple[lucid.Tensor, lucid.Tensor]:
    """The conditioning and the sample of the fixed generator ``name``."""
    config = DriftingConfig(**GENERATOR_CONFIGS[name])  # type: ignore[arg-type]
    model = DriftingModel(config).eval()
    fill_parameters(model)
    side = config.grid_size * config.patch_size
    x = fixed_tensor((2, config.in_channels, side, side), 0.19, 0.5)
    labels = lucid.tensor(GENERATOR_LABELS, dtype=lucid.int64)
    guidance = lucid.tensor(GENERATOR_GUIDANCE, dtype=lucid.float32)
    styles = lucid.tensor(GENERATOR_STYLES, dtype=lucid.int64)
    cond = model.conditioning(labels, guidance, styles, x.dtype)
    return cond, model(x, labels, guidance, styles).sample


def mae_features(name: str) -> dict[str, lucid.Tensor]:
    """Every feature set of the fixed MAE encoder ``name``."""
    config = DriftingConfig(**MAE_CONFIGS[name])  # type: ignore[arg-type]
    encoder = _DriftingFeatureEncoder(config)
    fill_parameters(encoder)
    side = config.grid_size * config.patch_size
    return encoder(fixed_tensor((2, config.in_channels, side, side), 0.23, 0.1))


def convnext_features() -> dict[str, lucid.Tensor]:
    """Every feature set of a small fixed ConvNeXt-V2, through the encoder's RGB path."""
    config = DriftingConfig(**CONVNEXT_CONFIG)  # type: ignore[arg-type]
    encoder = _DriftingFeatureEncoder(config, build=False)
    encoder.convnext = _ConvNeXtV2(depths=CONVNEXT_DEPTHS, dims=CONVNEXT_DIMS)
    encoder.built = True
    fill_parameters(encoder)
    return encoder(fixed_tensor((1, 3, 256, 256), 0.031, 0.7))


class TestGenerator:
    @pytest.mark.parametrize("name", sorted(GENERATOR_CONFIGS))
    def test_conditioning_and_sample_match_the_transcription(self, name: str) -> None:
        # Measured worst errors: 9.1e-8 (conditioning) and 6.0e-6 (sample), CPU and Metal.
        cond, sample = generator_outputs(name)
        want_cond, want_sample = GENERATOR_REFERENCE[name]
        assert digest(flat_values(cond), 2) == pytest.approx(want_cond, rel=None, abs=1e-6)
        rows = int(sample.shape[0]) * int(sample.shape[1])
        assert digest(flat_values(sample), rows) == pytest.approx(want_sample, rel=None, abs=5e-5)


class TestMAEEncoder:
    @pytest.mark.parametrize("name", sorted(MAE_CONFIGS))
    def test_feature_sets_match_the_transcription(self, name: str) -> None:
        # Measured worst errors: 2.0e-5 on the latent configs, 4.1e-4 on the pixel one, whose
        # space-to-depth input is 192 wide, CPU and Metal.
        feats = mae_features(name)
        want = MAE_REFERENCE[name]
        assert list(feats) == ["global", *want]
        got = {k: digest(flat_values(v))[0] for k, v in feats.items() if k != "global"}
        assert got == pytest.approx(want, rel=None, abs=2e-3)


class TestConvNeXt:
    def test_feature_sets_match_the_transcription(self) -> None:
        # Measured worst error: 2.0e-5, CPU and Metal.
        feats = convnext_features()
        assert list(feats) == list(CONVNEXT_REFERENCE)
        got = {k: digest(flat_values(v))[0] for k, v in feats.items()}
        assert got == pytest.approx(CONVNEXT_REFERENCE, rel=None, abs=2e-4)
