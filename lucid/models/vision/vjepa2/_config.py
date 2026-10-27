"""Configuration for the V-JEPA 2 video representation family."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta


@model_family_meta(
    canonical_name="V-JEPA 2",
    citation=(
        "Assran, Mahmoud, et al. "
        '"V-JEPA 2: Self-Supervised Video Models Enable Understanding, '
        'Prediction and Planning." arXiv:2506.09985, 2025.'
    ),
    theory=r"""
    V-JEPA 2 learns a video representation by predicting the target encoder's
    latent features at hidden tubelet positions.  A 3-D patch embedding turns
    a clip into a sequence of :math:`2\times16\times16` tubelets.  The context
    encoder sees only the unmasked indices, while the predictor inserts learned
    mask tokens and reconstructs the target encoder's representation at the
    held-out indices.  The target encoder is an exponential moving average of
    the context encoder and is not differentiated through.

    The released backbone uses a pre-normalised transformer with three-axis
    rotary position encoding.  Each attention head gives one third of its
    width to time, row and column — rounded down to an even number, so a
    head of 64 channels rotates 20 per axis and leaves the last four
    untouched.  Because position enters through that rotation rather than a
    learned table, a clip of a different length or resolution needs no
    interpolation and no new parameters: the 384-pixel variant is the
    256-pixel network read over a wider grid.

    Two details of the released rotation are reproduced rather than
    corrected.  The frequency half is repeated as a block before adjacent
    feature pairs are rotated, which pairs each rotation with a frequency
    that a straightforward implementation would not choose; the upstream
    source marks it as a bug and keeps it, because the published weights
    were trained through it.  Fixing it yields a different model, not a
    better-implemented one.

    The predictor is narrower than the video encoder — 384 channels and 12
    heads in every released checkpoint, whatever the encoder's width — and
    owns ten mask-token slots, one per block the mask generator samples.  It
    sees the context tokens and the mask tokens together, sorted into token
    order so that a position's identity reaches attention through the same
    rotary coordinates the encoder used, then answers only at the held-out
    positions.  Targets come from the momentum encoder and are
    layer-normalised, so the objective compares directions in
    representation space rather than magnitudes.
    """,
)
@dataclass(frozen=True)
class VJEPA2Config(ModelConfig):
    r"""Frozen architecture configuration for V-JEPA 2.

    The defaults are the released ViT-L/16 checkpoint geometry.  The public
    factories select the other paper variants by overriding ``dim``, ``depth``
    and ``num_heads``; callers may still use a small explicit override for
    tests or experiments.

    Parameters
    ----------
    image_size : int, default=256
        Height and width of each video frame.
    patch_size : int, default=16
        Spatial side of one patch.
    tubelet_size : int, default=2
        Number of adjacent frames in one token.
    num_frames : int, default=64
        Clip length used by the released pretraining configuration.
    dim, depth, num_heads : int, default=1024, 24, 16
        Width, block count and head count of the video encoder.
    predictor_dim : int, default=384
        Width of the latent predictor.
    predictor_depth : int, default=12
        Number of predictor blocks.
    predictor_heads : int, default=12
        Predictor attention heads.
    predictor_mlp_ratio : float, default=4.0
        Predictor feed-forward expansion.  The ViT-g encoder ratio is wider
        than its predictor ratio in the released checkpoint.
    predictor_num_mask_tokens : int, default=10
        Learned mask-token slots used by the official mask generator.
    use_rope : bool, default=True
        Use the released three-axis rotary attention geometry.
    num_pooler_layers : int, default=3
        Self-attention blocks the attentive probe runs before its query
        attends.  Every released classifier checkpoint carries three.
    ema : tuple of float, default=(0.99925, 0.99925)
        Target-encoder momentum at the start and end of the schedule.
        The released pretraining configuration holds it constant, which
        is where V-JEPA 2 departs from V-JEPA 1's ramp.
    ema_schedule_scale : float, default=1.25
        Stretches the momentum horizon past the run's own length, as the
        released ``ipe_scale`` does.
    """

    model_type: ClassVar[str] = "vjepa2"

    image_size: int = 256
    patch_size: int = 16
    tubelet_size: int = 2
    num_frames: int = 64
    in_channels: int = 3
    num_classes: int = 400

    dim: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True
    init_std: float = 0.02
    use_silu: bool = False
    wide_silu: bool = True
    use_rope: bool = True
    rope_base: float = 10_000.0
    uniform_power: bool = False
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0

    predictor_dim: int = 384
    predictor_out_dim: int | None = None
    predictor_depth: int = 12
    predictor_heads: int = 12
    predictor_mlp_ratio: float = 4.0
    predictor_num_mask_tokens: int = 10
    zero_init_mask_tokens: bool = True
    predictor_return_all_tokens: bool = False

    num_pooler_layers: int = 3

    normalize_targets: bool = True
    objective: str = "l1"
    ema: tuple[float, float] = (0.99925, 0.99925)
    ema_schedule_scale: float = 1.25

    @property
    def token_grid(self) -> tuple[int, int, int]:
        """Number of tubelet positions along time, height and width."""
        return (
            self.num_frames // self.tubelet_size,
            self.image_size // self.patch_size,
            self.image_size // self.patch_size,
        )

    @property
    def num_tokens(self) -> int:
        """Number of tokens in a full configured clip."""
        time, rows, cols = self.token_grid
        return time * rows * cols

    @property
    def predictor_output_dim(self) -> int:
        """Target width emitted by the predictor."""
        return self.dim if self.predictor_out_dim is None else self.predictor_out_dim

    def __post_init__(self) -> None:
        # A config read back from JSON carries a list where the field
        # declares a pair, and a frozen dataclass will keep it.
        object.__setattr__(self, "ema", (float(self.ema[0]), float(self.ema[1])))
        positive = (
            "image_size",
            "patch_size",
            "tubelet_size",
            "num_frames",
            "in_channels",
            "num_classes",
            "dim",
            "depth",
            "num_heads",
            "predictor_dim",
            "predictor_depth",
            "predictor_heads",
            "predictor_num_mask_tokens",
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
        if self.num_frames % self.tubelet_size != 0:
            raise ValueError(
                "num_frames must be divisible by tubelet_size, got "
                f"{self.num_frames} and {self.tubelet_size}"
            )
        if self.dim % self.num_heads != 0:
            raise ValueError(
                f"dim must be divisible by num_heads, got {self.dim} and " f"{self.num_heads}"
            )
        if self.predictor_dim % self.predictor_heads != 0:
            raise ValueError(
                "predictor_dim must be divisible by predictor_heads, got "
                f"{self.predictor_dim} and {self.predictor_heads}"
            )
        if self.predictor_mlp_ratio <= 0.0:
            raise ValueError(
                "predictor_mlp_ratio must be positive, got " f"{self.predictor_mlp_ratio}"
            )
        if self.predictor_output_dim <= 0:
            raise ValueError(
                f"predictor output width must be positive, got {self.predictor_output_dim}"
            )
        if self.mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {self.mlp_ratio}")
        if self.layer_norm_eps <= 0.0 or self.init_std <= 0.0:
            raise ValueError("layer_norm_eps and init_std must be positive")
        if self.rope_base <= 0.0:
            raise ValueError(f"rope_base must be positive, got {self.rope_base}")
        if not 0.0 <= self.drop_rate < 1.0:
            raise ValueError(f"drop_rate must lie in [0, 1), got {self.drop_rate}")
        if not 0.0 <= self.attn_drop_rate < 1.0:
            raise ValueError(f"attn_drop_rate must lie in [0, 1), got {self.attn_drop_rate}")
        if not 0.0 <= self.drop_path_rate < 1.0:
            raise ValueError(f"drop_path_rate must lie in [0, 1), got {self.drop_path_rate}")
        if self.objective not in {"l1", "l2", "smooth_l1"}:
            raise ValueError(
                "objective must be one of {'l1', 'l2', 'smooth_l1'}, got " f"{self.objective!r}"
            )
        if self.num_pooler_layers < 0:
            raise ValueError(
                f"num_pooler_layers must not be negative, got {self.num_pooler_layers}"
            )
        if len(self.ema) != 2 or not all(0.0 <= value <= 1.0 for value in self.ema):
            raise ValueError(f"ema must be two values in [0, 1], got {self.ema}")
        if self.ema_schedule_scale <= 0.0:
            raise ValueError(f"ema_schedule_scale must be positive, got {self.ema_schedule_scale}")


__all__ = ["VJEPA2Config"]
