"""The rows of the LoRA coverage matrix: a tiny, real factory per zoo ``model_type``.

Most rows are the families the zoo's other table tests already build at
unit-test size — ``test_models_train_step.CASES`` and
``test_zoo_dtype_follows_input._LOSSES`` — taken as they are, so a family
added there is a row here.  The rest are the families those files leave to
their own tests, with the config and inputs those tests use.

What distinguishes a row from the common case is declared apart from it, by
``model_type``: ``_GAPS`` for a family LoRA cannot adapt yet, ``_UNREACHED``
for adapters the row's forward leaves without a gradient on purpose.
"""

import dataclasses
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, Final

import lucid
from lucid.test.unit.models._peft_matrix import Call, Gap, ZooRow, collect
from lucid.test.unit.models.test_models_train_step import CASES
from lucid.test.unit.models.test_zoo_dtype_follows_input import _LOSSES, _boxes

# ── inputs ───────────────────────────────────────────────────────────────────


def _call(*args: Any, **kwargs: Any) -> Callable[[], Call]:
    """Inputs built by calling each argument that is a factory; plain values pass through."""

    def make() -> Call:
        built = tuple(a() if callable(a) else a for a in args)
        return built, {k: v() if callable(v) else v for k, v in kwargs.items()}

    return make


def _randn(*shape: int) -> Callable[[], lucid.Tensor]:
    return lambda: lucid.randn(*shape)


def _rand(*shape: int) -> Callable[[], lucid.Tensor]:
    return lambda: lucid.rand(*shape)


def _zeros_int(*shape: int) -> Callable[[], lucid.Tensor]:
    return lambda: lucid.zeros(shape, dtype=lucid.int64)


def _labels(*values: int) -> Callable[[], lucid.Tensor]:
    return lambda: lucid.tensor(list(values), dtype=lucid.int64)


def _classes(*shape: int, high: int) -> Callable[[], lucid.Tensor]:
    return lambda: lucid.randint(0, high, shape)


def _indices(batch: int, start: int, stop: int) -> Callable[[], lucid.Tensor]:
    """The token positions ``start`` to ``stop`` for each of ``batch`` clips."""
    return lambda: lucid.arange(start, stop, dtype=lucid.int32).repeat(batch, 1)


def _proposals() -> list[lucid.Tensor]:
    return [lucid.tensor([[5.0, 5.0, 40.0, 40.0], [10.0, 10.0, 50.0, 50.0]])]


def _row(
    factory: str, overrides: Mapping[str, Any], inputs: Callable[[], Call], **row: Any
) -> ZooRow:
    return ZooRow(factory=factory, overrides=MappingProxyType(overrides), inputs=inputs, **row)


# ── configs the other tests use ──────────────────────────────────────────────

_UNET: Final = dict(sample_size=8, base_channels=8, num_res_blocks=1, resnet_groups=2)
_LATENT: Final = dict(sample_size=8, hidden_size=32, depth=1, num_heads=4, num_classes=4)
_PIXEL_UNET: Final = dict(
    sample_size=16,
    base_channels=16,
    channel_mult=(1, 2),
    num_res_blocks=1,
    attention_resolutions=(8,),
    num_heads=2,
    resnet_groups=8,
)
_DREAMER: Final = dict(
    stoch_size=4,
    deter_size=16,
    hidden_size=16,
    cnn_depth=4,
    actor_hidden=16,
    value_hidden=16,
    reward_hidden=16,
    actor_layers=1,
    value_layers=1,
    horizon=2,
)
_DIAMOND: Final = dict(
    sample_size=16,
    unet_channels=(8, 8),
    unet_layers=(1, 1),
    reward_channels=(8, 8),
    reward_layers=(1, 1),
    actor_channels=(8, 8),
    actor_layers=(1, 1),
    cond_dim=16,
    reward_cond_dim=8,
    reward_lstm_dim=16,
    actor_lstm_dim=16,
    num_actions=4,
    horizon=3,
)
_GENIE: Final = dict(
    sample_size=(8, 8),
    num_frames=4,
    num_codes=16,
    code_dim=4,
    tokenizer_encoder_layers=1,
    tokenizer_encoder_dim=16,
    tokenizer_encoder_heads=2,
    tokenizer_encoder_head_dim=8,
    tokenizer_decoder_layers=1,
    tokenizer_decoder_dim=16,
    tokenizer_decoder_heads=2,
    tokenizer_decoder_head_dim=8,
    action_patch_size=4,
    action_dim=4,
    action_encoder_layers=1,
    action_encoder_dim=16,
    action_encoder_heads=2,
    action_decoder_layers=1,
    action_decoder_dim=16,
    action_decoder_heads=2,
    dynamics_layers=1,
    dynamics_dim=16,
    dynamics_heads=2,
    dynamics_head_dim=8,
    maskgit_steps=4,
)
_JEPA_IMAGE: Final = dict(
    image_size=32,
    patch_size=8,
    dim=16,
    depth=1,
    num_heads=2,
    predictor_dim=8,
    predictor_depth=1,
    min_keep=1,
    target_scale=(0.1, 0.2),
    context_scale=(0.8, 1.0),
)
_JEPA_VIDEO: Final = dict(
    image_size=32,
    patch_size=8,
    tubelet_size=2,
    num_frames=4,
    dim=24,
    depth=1,
    num_heads=2,
    predictor_dim=12,
    predictor_depth=1,
    short_range_blocks=2,
    long_range_blocks=1,
)
_VJEPA2: Final = dict(
    image_size=16,
    patch_size=8,
    num_frames=4,
    dim=24,
    depth=1,
    num_heads=2,
    predictor_dim=12,
    predictor_depth=1,
    predictor_heads=2,
    num_classes=5,
    num_pooler_layers=2,
)
_VJEPA2_AC: Final = dict(
    image_size=16,
    patch_size=8,
    tubelet_size=2,
    num_frames=4,
    encoder_dim=24,
    encoder_depth=1,
    encoder_heads=2,
    predictor_dim=24,
    predictor_depth=1,
    predictor_heads=2,
    action_dim=3,
    state_dim=3,
    extrinsics_dim=2,
    use_extrinsics=True,
)
_CLIP: Final = dict(
    embed_dim=16,
    image_size=16,
    patch_size=8,
    vision_layers=1,
    vision_width=16,
    vision_heads=2,
    context_length=8,
    vocab_size=32,
    text_width=16,
    text_heads=2,
    text_layers=1,
)
_STABLE_DIFFUSION: Final = dict(
    sample_size=32,
    vae_block_out_channels=(8, 8),
    vae_layers_per_block=1,
    unet_block_out_channels=(8, 16),
    unet_layers_per_block=1,
    norm_num_groups=2,
    cross_attention_dim=16,
    attention_head_dim=4,
    context_length=8,
    downsample_factor=2,
)
_DREAMER_SEQUENCE: Final = _call(
    _randn(2, 4, 3, 64, 64), _randn(2, 4, 1), _randn(2, 4), lambda: lucid.ones((2, 4))
)

# ── the families the shared tables leave out ─────────────────────────────────

_GENERATIVE: Final[list[ZooRow]] = [
    _row(
        "nice_mnist",
        dict(
            sample_size=8,
            in_channels=1,
            input_dim=64,
            num_coupling_layers=2,
            num_hidden_layers=2,
            hidden_dim=16,
        ),
        _call(_randn(2, 64)),
    ),
    _row(
        "realnvp_cifar",
        dict(sample_size=8, num_scales=2, residual_blocks=1, base_dim=8),
        _call(_rand(2, 3, 8, 8)),
    ),
    _row(
        "flow_matching_cifar",
        {**_UNET, "num_head_channels": 8, "solver": "euler"},
        _call(_randn(2, 3, 8, 8), _rand(2)),
    ),
    _row("rectified_flow_cifar", {**_UNET, "solver": "euler"}, _call(_randn(2, 3, 8, 8), _rand(2))),
    _row(
        "score_sde_vp",
        dict(
            sample_size=8,
            base_channels=8,
            channel_mult=(1, 2),
            num_res_blocks=1,
            resnet_groups=2,
            num_heads=1,
        ),
        _call(_randn(2, 3, 8, 8), _rand(2)),
    ),
    _row(
        "ddpm_cifar_gen",
        {**_PIXEL_UNET, "num_train_timesteps": 20},
        _call(
            _randn(2, 3, 16, 16), lambda: lucid.tensor([4, 17]).long(), target=_randn(2, 3, 16, 16)
        ),
    ),
    _row(
        "ncsn_cifar_gen",
        {**_PIXEL_UNET, "num_noise_levels": 5, "sigma_max": 10.0, "sigma_min": 0.1},
        _call(_randn(2, 3, 16, 16)),
    ),
    _row("mean_flow_base_2_gen", _LATENT, _call(_randn(2, 4, 8, 8), _labels(0, 1))),
    _row("dit_small_2_gen", _LATENT, _call(_randn(2, 4, 8, 8), _labels(0, 1))),
    _row(
        "vqvae_gen",
        dict(
            sample_size=8,
            num_embeddings=16,
            embedding_dim=8,
            hidden_channels=8,
            residual_hidden_channels=8,
            num_downsample_layers=1,
            num_residual_layers=1,
        ),
        _call(_randn(2, 3, 8, 8)),
    ),
    _row(
        "stable_diffusion",
        _STABLE_DIFFUSION,
        _call(_randn(1, 3, 32, 32), _randn(1, 8, 16), return_loss=True),
    ),
    _row("dreamer_world_model", {**_DREAMER, "reward_layers": 1}, _DREAMER_SEQUENCE),
    _row(
        "dreamer_v2_world_model",
        {**_DREAMER, "discrete": 4, "reward_layers": 1, "pcont_layers": 1},
        _DREAMER_SEQUENCE,
    ),
    _row(
        "dreamer_v3_12m_world_model",
        {**_DREAMER, "discrete": 4, "blocks": 2, "num_bins": 15},
        _DREAMER_SEQUENCE,
    ),
    _row(
        "planet",
        dict(
            action_dim=2, stoch_size=4, deter_size=8, hidden_size=8, cnn_depth=4, reward_hidden=8
        ),
        _call(_randn(2, 3, 3, 64, 64), _randn(2, 3, 2)),
    ),
    _row("diamond", _DIAMOND, _call(_randn(2, 4, 3, 16, 16), _zeros_int(2, 4), _randn(2, 3, 16, 16))),
    _row("genie", _GENIE, _call(_rand(2, 4, 3, 8, 8))),
    _row("clip_vit_base_32", _CLIP, _call(_randn(2, 3, 16, 16), _zeros_int(2, 8))),
]

_VISION: Final[list[ZooRow]] = [
    _row("ijepa_base_16", _JEPA_IMAGE, _call(_rand(2, 3, 32, 32))),
    _row("vjepa_large_16", _JEPA_VIDEO, _call(_rand(2, 4, 3, 32, 32))),
    _row(
        "vjepa2_vit_large",
        _VJEPA2,
        _call(_randn(2, 4, 3, 16, 16), _indices(2, 0, 6), _indices(2, 6, 8)),
    ),
    _row(
        "vjepa2_ac_vit_giant",
        _VJEPA2_AC,
        _call(_randn(1, 4, 3, 16, 16), _randn(1, 3, 3), _randn(1, 3, 3), _randn(1, 3, 2)),
    ),
    _row("rcnn", dict(num_classes=3), _call(_randn(1, 3, 64, 64), _proposals)),
    _row("fast_rcnn", dict(num_classes=3), _call(_randn(1, 3, 64, 64), _proposals), heavy=True),
    _row("yolo_v3", dict(num_classes=4), _call(_randn(1, 3, 64, 64), targets=lambda: _boxes(64))),
    _row(
        "attention_unet",
        dict(num_classes=3, base_channels=8, depth=2),
        _call(_randn(2, 1, 32, 32), targets=_classes(2, 32, 32, high=3)),
    ),
]


def _losses() -> list[ZooRow]:
    """The detectors and segmenters whose losses ``test_zoo_dtype_follows_input`` builds."""
    return [
        _row(
            factory,
            overrides,
            lambda side=side, targets=targets: (
                (lucid.randn(1, 3, side, side),),
                {"targets": targets()},
            ),
        )
        for factory, overrides, side, targets in _LOSSES
    ]


# ── what sets a row apart, by ``model_type`` ─────────────────────────────────

#: Families LoRA cannot adapt yet, and the exact piece that is missing.
_GAPS: Final[Mapping[str, Gap]] = {
    "tvm": Gap(
        "forward: TVM syncs its EMA copy by load_state_dict from the live network, and an "
        "injected network's base_layer/lora_* keys are not the copy's",
        (RuntimeError,),
    ),
}

#: Adapters the row's forward leaves without a gradient on purpose.
_EMA_TARGET: Final = (".target_encoder.",)
_UNREACHED: Final[Mapping[str, tuple[str, ...]]] = {
    "ijepa": _EMA_TARGET,
    "vjepa": _EMA_TARGET,
    "vjepa2": _EMA_TARGET,
    "diamond": (".reward_end.", ".actor_critic."),
}


def _from_cases() -> list[ZooRow]:
    return [ZooRow.from_case(case) for case in CASES]


def _distinguished(row: ZooRow) -> ZooRow:
    return dataclasses.replace(
        row, gap=_GAPS.get(row.model_type), unreached=_UNREACHED.get(row.model_type, ())
    )


ROWS: Final[Mapping[str, ZooRow]] = collect(
    [_distinguished(r) for r in (*_from_cases(), *_losses(), *_GENERATIVE, *_VISION)]
)
