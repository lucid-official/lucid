"""Configuration for the V-JEPA 2 action-conditioned world model."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta


@model_family_meta(
    canonical_name="V-JEPA 2-AC",
    citation=(
        "Assran, Mahmoud, et al. "
        '"V-JEPA 2: Self-Supervised Video Models Enable Understanding, '
        'Prediction and Planning." arXiv:2506.09985, 2025.'
    ),
    theory=r"""
    V-JEPA 2-AC post-trains a V-JEPA 2 visual encoder as a latent,
    action-conditioned world model.  Each frame is encoded into its own
    spatial token grid — the three-dimensional patch embedding wants
    :math:`2` frames, so a frame is repeated to fill one tubelet.  The
    predictor interleaves an action token and a state token before every
    grid, applies a frame-causal transformer, and emits the next latent
    grid.  The visual target is kept in representation space, so
    the model predicts the consequences of an action without rendering pixels.

    The released action-conditioned checkpoint contains one ViT-g/16 variant
    with a 1024-wide, 24-layer predictor and seven-dimensional robot action
    and state vectors.  Its encoder uses the ViT-g wide MLP ratio, while the
    predictor retains the released four-times expansion.

    Attention is *frame-causal* rather than token-causal: every token of a
    step attends to every token of that step and of the ones before it, and
    to nothing after.  A step is therefore a unit of time rather than a
    position in a sequence, and the conditioning tokens sit inside it —
    action first, then state, then the spatial grid.  The action and state
    tokens are rotated only along the time axis, since they have no place in
    the spatial grid, which is what separates this attention from the
    encoder's.

    The training signal is next-frame prediction in representation space:
    the predictor is given the latents of frames up to :math:`t` and the
    action taken at :math:`t`, and is scored against the latents of frame
    :math:`t + 1`, both sides layer-normalised.  Scoring against the same
    frame instead would be satisfied by copying the input, which the causal
    mask already permits.  Planning and controller search are consumers of
    this latent transition model, not additional network layers — the
    forward is one transition, and a rollout is that forward applied to its
    own output.
    """,
)
@dataclass(frozen=True)
class VJEPA2ACConfig(ModelConfig):
    r"""Frozen architecture configuration for V-JEPA 2-AC.

    Inputs use Lucid's video convention ``(B, T, C, H, W)``.  The encoder
    is applied independently to each frame, producing ``T`` action-model
    steps — one per frame, which is the rate the released checkpoint was
    post-trained at.

    ``num_frames`` records the released clip length and sizes
    :attr:`token_grid`; it does not constrain a call.  One action-model
    step is one frame — the released loop repeats each frame to fill the
    encoder's tubelet — so a rollout is as long as the actions handed to
    it and any frame count of two or more is valid.

    ``normalize_targets`` layer-normalises the latents on both sides of
    the transition loss, which the released post-training does.
    """

    model_type: ClassVar[str] = "vjepa2_ac"

    image_size: int = 256
    patch_size: int = 16
    tubelet_size: int = 2
    num_frames: int = 64
    in_channels: int = 3

    encoder_dim: int = 1408
    encoder_depth: int = 40
    encoder_heads: int = 22
    mlp_ratio: float = 48.0 / 11.0
    predictor_mlp_ratio: float = 4.0
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True
    init_std: float = 0.02
    use_silu: bool = False
    wide_silu: bool = True
    use_rope: bool = True
    rope_base: float = 10_000.0

    predictor_dim: int = 1024
    predictor_depth: int = 24
    predictor_heads: int = 16
    action_dim: int = 7
    state_dim: int = 7
    extrinsics_dim: int = 6
    use_extrinsics: bool = False
    frame_causal: bool = True
    normalize_targets: bool = True

    @property
    def token_grid(self) -> tuple[int, int, int]:
        """Configured step, row and column counts.

        One step is one frame, so the temporal entry is the clip length
        rather than a tubelet count.
        """
        return (
            self.num_frames,
            self.image_size // self.patch_size,
            self.image_size // self.patch_size,
        )

    @property
    def tokens_per_step(self) -> int:
        """Spatial tokens emitted for one encoded frame."""
        return self.token_grid[1] * self.token_grid[2]

    def __post_init__(self) -> None:
        positive = (
            "image_size",
            "patch_size",
            "tubelet_size",
            "num_frames",
            "in_channels",
            "encoder_dim",
            "encoder_depth",
            "encoder_heads",
            "predictor_dim",
            "predictor_depth",
            "predictor_heads",
            "action_dim",
            "state_dim",
            "extrinsics_dim",
        )
        for name in positive:
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                "image_size must be divisible by patch_size, got "
                f"{self.image_size} and {self.patch_size}"
            )
        if self.encoder_dim % self.encoder_heads != 0:
            raise ValueError(
                "encoder_dim must be divisible by encoder_heads, got "
                f"{self.encoder_dim} and {self.encoder_heads}"
            )
        if self.predictor_dim % self.predictor_heads != 0:
            raise ValueError(
                "predictor_dim must be divisible by predictor_heads, got "
                f"{self.predictor_dim} and {self.predictor_heads}"
            )
        if self.mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {self.mlp_ratio}")
        if self.predictor_mlp_ratio <= 0.0:
            raise ValueError(
                "predictor_mlp_ratio must be positive, got " f"{self.predictor_mlp_ratio}"
            )
        if self.layer_norm_eps <= 0.0 or self.init_std <= 0.0:
            raise ValueError("layer_norm_eps and init_std must be positive")
        if self.rope_base <= 0.0:
            raise ValueError(f"rope_base must be positive, got {self.rope_base}")
        if self.use_extrinsics and self.extrinsics_dim <= 0:
            raise ValueError("extrinsics_dim must be positive when enabled")


__all__ = ["VJEPA2ACConfig"]
