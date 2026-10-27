r"""V-JEPA — predicting features of a clip's hidden tubelets.

The same three networks as I-JEPA, reading video.  A clip becomes
tubelets two frames deep; two collections of blocks hide most of them —
eight small ones and two large ones, each running the whole depth of the
clip in time — and the predictor must say what an averaged copy of the
encoder reports at the hidden positions.  The averaged encoder reads the
clip whole; the trained one reads only what is left.

The transformer itself is shared with I-JEPA and lives in
``vision/_common``.  What is here is what video changes: the tubelet
embedding, a position table with three axes, masks that are tubes, and
the attentive probe the paper evaluates with.
"""

import math
import random
from dataclasses import dataclass
from typing import ClassVar, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._utils._common import host_random, update_averages
from lucid.models._output import ImageClassificationOutput, ModelOutput
from lucid.models._tasks import ImageClassificationModel
from lucid.models.vision._common._transformers import (
    Block as _Block,
    MLP as _MLP,
    gather_tokens as _gather_tokens,
    init_transformer_weights as _init_transformer_weights,
    scale_residual_projection as _scale_residual_projection,
    sincos_embedding as _sincos_embedding,
)
from lucid.models.vision.vjepa._config import VJEPAConfig

__all__ = [
    "VJEPAModel",
    "VJEPAForVideoClassification",
    "VJEPAOutput",
]


# ── positions ────────────────────────────────────────────────────────────────


def _sincos_3d(dim: int, grid: tuple[int, int, int], uniform_power: bool) -> Tensor:
    """Fixed three-dimensional sine/cosine positions, ``(1, T*H*W, dim)``.

    Time, row and column each get their own table, concatenated in that
    order and cut to ``dim``.  ``uniform_power`` splits the width evenly
    between the three, which is what every released configuration asks
    for; the alternative gives time half the width and the spatial axes a
    quarter each, and loading released weights under the wrong split puts
    them on positions they never saw.
    """
    duration, rows, cols = grid
    if uniform_power:
        each = int(math.ceil(dim / 6) * 2)
        widths = (each, each, each)
    else:
        widths = (dim // 2, dim // 4, dim // 4)

    time = lucid.arange(duration).to(lucid.float64)
    down = lucid.arange(rows).to(lucid.float64)
    across = lucid.arange(cols).to(lucid.float64)
    zeros = lucid.zeros(duration, rows, cols, dtype=lucid.float64)
    coords = (
        time.reshape(duration, 1, 1) + zeros,
        down.reshape(1, rows, 1) + zeros,
        across.reshape(1, 1, cols) + zeros,
    )
    table = lucid.cat(
        [
            _sincos_embedding(axis.reshape(-1), width)
            for axis, width in zip(coords, widths, strict=True)
        ],
        dim=1,
    )
    return table[:, :dim].unsqueeze(dim=0)


# ── tubelets ─────────────────────────────────────────────────────────────────


class _TubeletEmbed(nn.Module):
    """Cut a clip into tubelets and project each with one convolution.

    Clips arrive as ``(B, T, C, H, W)``, the shape every family in this
    zoo uses, and a 3-D convolution wants the channel axis second — hence
    the one permutation.  Section 3.3: a 16x16 patch spanning two frames.
    """

    def __init__(self, in_channels: int, tubelet_size: int, patch_size: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels,
            dim,
            (tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        clip = x.permute(0, 2, 1, 3, 4)
        tokens = self.proj(clip)
        b, c = int(tokens.shape[0]), int(tokens.shape[1])
        return tokens.reshape(b, c, -1).permute(0, 2, 1)


class _Encoder(nn.Module):
    """A ViT over tubelets, with no class token."""

    pos_embed: Tensor

    def __init__(self, config: VJEPAConfig) -> None:
        super().__init__()
        hidden = int(config.dim * config.mlp_ratio)
        self.patch_embed = _TubeletEmbed(
            config.in_channels, config.tubelet_size, config.patch_size, config.dim
        )
        self.register_buffer(
            "pos_embed",
            _sincos_3d(config.dim, config.token_grid, config.uniform_power),
        )
        blocks = [
            _Block(config.dim, config.num_heads, hidden, config.layer_norm_eps)
            for _ in range(config.depth)
        ]
        self.blocks: nn.ModuleList[_Block] = nn.ModuleList([*blocks])
        self.norm = nn.LayerNorm(config.dim, eps=config.layer_norm_eps)

        # Same initialisation as I-JEPA's tower, down to the tubelet
        # convolution: the released code treats Conv3d exactly as the image
        # side treats Conv2d.
        _init_transformer_weights(self)
        for depth_index, block in enumerate(blocks, start=1):
            _scale_residual_projection(block.attn.proj, block.mlp.fc2, depth_index)

    @override
    def forward(self, x: Tensor, indices: Tensor | None = None) -> Tensor:
        """Encode a clip, optionally keeping only the given tubelets.

        Masking drops tokens rather than attending around them, so the
        context encoder's sequence is genuinely shorter than the clip.
        """
        tokens = self.patch_embed(x) + self.pos_embed
        if indices is not None:
            tokens = _gather_tokens(tokens, indices)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


class _Predictor(nn.Module):
    """Answers for hidden tubelets, one mask token per mask collection.

    The released code carries a learned mask token for each collection
    and picks by index, so the short-range and long-range questions are
    not asked in quite the same voice.  They start at zero there, which
    is the one place V-JEPA differs from I-JEPA's initialisation.
    """

    predictor_pos_embed: Tensor

    def __init__(self, config: VJEPAConfig, num_mask_tokens: int) -> None:
        super().__init__()
        width = config.predictor_dim
        hidden = int(width * config.mlp_ratio)
        heads = config.resolved_predictor_heads
        self.num_mask_tokens = num_mask_tokens
        self.predictor_embed = nn.Linear(config.dim, width)
        self.mask_tokens = nn.ParameterList(
            [nn.Parameter(lucid.zeros(1, 1, width)) for _ in range(num_mask_tokens)]
        )
        self.register_buffer(
            "predictor_pos_embed",
            _sincos_3d(width, config.token_grid, config.uniform_power),
        )
        blocks = [
            _Block(width, heads, hidden, config.layer_norm_eps)
            for _ in range(config.predictor_depth)
        ]
        self.predictor_blocks: nn.ModuleList[_Block] = nn.ModuleList([*blocks])
        self.predictor_norm = nn.LayerNorm(width, eps=config.layer_norm_eps)
        self.predictor_proj = nn.Linear(width, config.dim)

        # The mask tokens above stay at zero: only linear, convolutional
        # and norm layers are visited, which is what keeps this family's
        # one initialisation difference from I-JEPA intact.
        _init_transformer_weights(self)
        for depth_index, block in enumerate(blocks, start=1):
            _scale_residual_projection(block.attn.proj, block.mlp.fc2, depth_index)

    @override
    def forward(
        self,
        context: Tensor,
        context_indices: Tensor,
        target_indices: Tensor,
        mask_index: int = 0,
    ) -> Tensor:
        """Predict the target encoder's output at ``target_indices``."""
        if not 0 <= mask_index < self.num_mask_tokens:
            raise ValueError(
                f"mask_index must name one of the {self.num_mask_tokens} mask "
                f"tokens, got {mask_index}"
            )
        positions = self.predictor_pos_embed
        batch = int(context.shape[0])
        tokens = self.predictor_embed(context)
        tokens = tokens + _gather_tokens(
            positions + lucid.zeros(batch, 1, 1, dtype=positions.dtype, device=positions.device),
            context_indices,
        )

        count = int(target_indices.shape[1])
        width = int(tokens.shape[2])
        queries = self.mask_tokens[mask_index] + lucid.zeros(
            batch, count, width, dtype=tokens.dtype, device=tokens.device
        )
        queries = queries + _gather_tokens(
            positions + lucid.zeros(batch, 1, 1, dtype=positions.dtype, device=positions.device),
            target_indices,
        )

        hidden = lucid.cat([tokens, queries], dim=1)
        for block in self.predictor_blocks:
            hidden = block(hidden)
        hidden = self.predictor_norm(hidden)
        return self.predictor_proj(hidden[:, -count:])


# ── the attentive probe ──────────────────────────────────────────────────────


class _AttentivePooler(nn.Module):
    """One learned query, cross-attending a frozen feature map.

    Section 4.3 evaluates V-JEPA this way rather than by averaging tokens
    and fitting a linear map, and the difference is not decorative: the
    paper measures it at 17 points on Kinetics-400.  A frozen encoder's
    tokens carry what a clip contains *somewhere*; average-pooling throws
    away which token said what, and this asks a question instead.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, eps: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query_token = nn.Parameter(lucid.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query_token, std=0.02)
        self.norm_keys = nn.LayerNorm(dim, eps=eps)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.norm_out = nn.LayerNorm(dim, eps=eps)
        self.mlp = _MLP(dim, int(dim * mlp_ratio))

        # The probe is a one-block tower and is initialised as one, so a
        # frozen encoder is evaluated by the same probe the paper's numbers
        # came from rather than by a differently-scaled one.
        _init_transformer_weights(self)
        _scale_residual_projection(self.proj, self.mlp.fc2, 1)

    @override
    def forward(self, tokens: Tensor) -> Tensor:
        """Pool ``(B, N, D)`` tokens into one ``(B, D)`` vector."""
        batch, count, width = (int(s) for s in tokens.shape)
        keys_in = self.norm_keys(tokens)

        def heads(t: Tensor, length: int) -> Tensor:
            return t.reshape(batch, length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        query_in = self.query_token + lucid.zeros(
            batch, 1, width, dtype=tokens.dtype, device=tokens.device
        )
        q = heads(self.query(query_in), 1)
        k = heads(self.key(keys_in), count)
        v = heads(self.value(keys_in), count)
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.permute(0, 2, 1, 3).reshape(batch, 1, width)

        pooled = query_in + self.proj(attended)
        pooled = pooled + self.mlp(self.norm_out(pooled))
        return pooled.reshape(batch, width)


# ── masking ──────────────────────────────────────────────────────────────────


def _tube_shape(
    scale: tuple[float, float],
    aspect: tuple[float, float],
    rows: int,
    cols: int,
    rng: random.Random,
) -> tuple[int, int]:
    """Height and width, in tokens, of one block of a mask collection.

    A block's size is drawn once per batch, as the released collator does
    so that every clip's mask is the same length.  The paper's scales are
    points rather than ranges — 0.15 and 0.7 — so only the aspect ratio
    varies in practice.
    """
    patches = rows * cols
    for _ in range(20):
        area = patches * (rng.random() * (scale[1] - scale[0]) + scale[0])
        ratio = rng.random() * (aspect[1] - aspect[0]) + aspect[0]
        height = max(1, min(rows, int(round(math.sqrt(area * ratio)))))
        width = max(1, min(cols, int(round(math.sqrt(area / ratio)))))
        if height * width < patches:
            return height, width
    return max(1, min(rows, height)), max(1, min(cols, width))


def _tube_indices(
    grid: tuple[int, int, int], height: int, width: int, rng: random.Random
) -> set[int]:
    """Token indices of one block, running the whole clip in time.

    Section 3.2 extends every block through the temporal axis: a mask
    that leaves a tubelet visible in a neighbouring frame asks nothing,
    because the answer is next door.
    """
    duration, rows, cols = grid
    top = rng.randrange(rows - height + 1)
    left = rng.randrange(cols - width + 1)
    return {
        (t * rows + top + r) * cols + left + c
        for t in range(duration)
        for r in range(height)
        for c in range(width)
    }


def _sample_collection(
    config: VJEPAConfig,
    batch: int,
    blocks: int,
    scale: tuple[float, float],
    device: DeviceLike,
) -> tuple[Tensor, Tensor]:
    """Context and target indices for one mask collection.

    The blocks are unioned, the union is what the predictor answers for,
    and what is left over is what the encoder reads.  Both are truncated
    to the batch's shortest so they collate — the released code does the
    same, and it does mean a few targets are dropped.  Every draw comes
    from one host stream, seeded from Lucid's generator.
    """
    rng = host_random()
    duration, rows, cols = config.token_grid
    total = duration * rows * cols
    height, width = _tube_shape(scale, config.aspect_ratio, rows, cols, rng)

    targets: list[list[int]] = []
    contexts: list[list[int]] = []
    for _ in range(batch):
        hidden: set[int] = set()
        for _ in range(blocks):
            hidden |= _tube_indices(config.token_grid, height, width, rng)
        visible = [index for index in range(total) if index not in hidden]
        if not visible:
            # Every token was hidden.  Hand one back rather than run the
            # encoder on an empty sequence.
            chosen = sorted(hidden)[0]
            hidden.discard(chosen)
            visible = [chosen]
        targets.append(sorted(hidden))
        contexts.append(visible)

    keep_target = min(len(t) for t in targets)
    keep_context = min(len(c) for c in contexts)
    return (
        lucid.tensor([c[:keep_context] for c in contexts], dtype=lucid.int64, device=device),
        lucid.tensor([t[:keep_target] for t in targets], dtype=lucid.int64, device=device),
    )


# ── outputs ──────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class VJEPAOutput(ModelOutput):
    r"""What one pretraining step produced, for both mask collections.

    Attributes
    ----------
    loss : Tensor
        The two collections' discrepancies, averaged.  Scalar.
    short_prediction, short_target : Tensor
        What the predictor said and what the target encoder said for the
        eight-block collection, ``(B, Ns, D)``.
    long_prediction, long_target : Tensor
        The same for the two-block collection, ``(B, Nl, D)``.
    short_context, short_targets : Tensor
        The tubelets the encoder read and the ones it answered for,
        ``(B, Cs)`` and ``(B, Ns)``.
    long_context, long_targets : Tensor
        The same for the second collection.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vjepa import VJEPAOutput
    >>> zero = lucid.zeros(1, 4, 8)
    >>> index = lucid.zeros(1, 4).to(lucid.int64)
    >>> out = VJEPAOutput(
    ...     loss=lucid.zeros(()), short_prediction=zero, short_target=zero,
    ...     long_prediction=zero, long_target=zero, short_context=index,
    ...     short_targets=index, long_context=index, long_targets=index)
    >>> out.short_prediction.shape
    (1, 4, 8)
    """

    loss: Tensor
    short_prediction: Tensor
    short_target: Tensor
    long_prediction: Tensor
    long_target: Tensor
    short_context: Tensor
    short_targets: Tensor
    long_context: Tensor
    long_targets: Tensor


# ── the model ────────────────────────────────────────────────────────────────


class VJEPAModel(PretrainedModel, BackboneMixin):
    r"""V-JEPA: a context encoder, an averaged target encoder, a predictor.

    Parameters
    ----------
    config : VJEPAConfig
        Frozen configuration.

    Attributes
    ----------
    encoder : Module
        The context encoder — the one that is trained.
    target_encoder : Module
        Its exponential moving average.  Frozen; produces the targets and,
        after pretraining, the representation everything downstream reads.
    predictor : Module
        The narrow transformer, with one mask token per mask collection.

    Notes
    -----
    Reference: Bardes, Adrien, et al., *"Revisiting Feature Prediction for
    Learning Visual Representations from Video"*, arXiv:2404.08471, 2024,
    Sections 3.1–3.3.

    Clips are ``(B, T, C, H, W)`` with ``T = config.num_frames``.  Call
    :meth:`update_target` after each optimiser step, with the value
    :meth:`momentum` gives.

    The target features are layer-normalised without affine terms before
    any block is taken from them — that is in the released code and not
    in the paper, and it is the usual way a reimplementation fails.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vjepa import VJEPAConfig, VJEPAModel
    >>> config = VJEPAConfig(
    ...     image_size=32, patch_size=8, tubelet_size=2, num_frames=4,
    ...     dim=24, depth=1, num_heads=2, predictor_dim=12,
    ...     predictor_depth=1, short_range_blocks=2, long_range_blocks=1)
    >>> model = VJEPAModel(config)
    >>> out = model(lucid.rand(2, 4, 3, 32, 32))
    >>> out.short_prediction.shape == out.short_target.shape
    True
    >>> float(out.loss.item()) >= 0.0
    True

    The representation a downstream task reads is the target encoder's:

    >>> model.encode(lucid.rand(2, 4, 3, 32, 32)).shape
    (2, 24)
    """

    config_class: ClassVar[type[VJEPAConfig]] = VJEPAConfig

    def __init__(self, config: VJEPAConfig) -> None:
        """Build the three networks. See the class docstring for parameters."""
        super().__init__(config)
        self.config: VJEPAConfig = config
        self.encoder = _Encoder(config)
        self.target_encoder = _Encoder(config)
        self.predictor = _Predictor(config, num_mask_tokens=2)

        nn.utils.copy_parameters_and_buffers(self.encoder, self.target_encoder)
        self.target_encoder.requires_grad_(False)

        self._feature_info = [
            FeatureInfo(stage=1, num_channels=config.dim, reduction=config.patch_size)
        ]

    # ── backbone ─────────────────────────────────────────────────────────

    @override
    @property
    def feature_info(self) -> list[FeatureInfo]:
        """One stage: tubelet tokens at the encoder's width."""
        return self._feature_info

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        """The average-pooled target encoder, which is :meth:`encode`."""
        return self.encode(x)

    # ── parameter groups ─────────────────────────────────────────────────

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Everything an optimiser should be given.

        Returns
        -------
        list of Parameter
            The context encoder and the predictor.  The target encoder
            follows by moving average and must not be optimised.
        """
        return list(self.encoder.parameters()) + list(self.predictor.parameters())

    # ── the moving average ───────────────────────────────────────────────

    def momentum(self, step: int, total_steps: int) -> float:
        """The target encoder's momentum at a training step.

        Parameters
        ----------
        step : int
            Steps already taken.
        total_steps : int
            Steps the run will take in total.

        Returns
        -------
        float
            Linear from ``config.ema[0]`` toward ``config.ema[1]`` over
            ``total_steps * config.ema_schedule_scale`` steps.  The
            released runs stretch the schedule by a quarter and stop
            early, so the momentum is still short of 1 when training
            ends — reproducing it with a schedule the length of the run
            is a different experiment.
        """
        if total_steps < 1:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        start, end = self.config.ema
        horizon = total_steps * self.config.ema_schedule_scale
        fraction = min(max(step / horizon, 0.0), 1.0)
        return start + (end - start) * fraction

    def update_target(self, momentum: float) -> None:
        """Move the target encoder toward the context encoder.

        Call it after the optimiser step: this writes parameters in place.

        Parameters
        ----------
        momentum : float
            Weight kept on the target encoder, from :meth:`momentum`.
        """
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must lie in [0, 1], got {momentum}")
        update_averages(self.encoder, [(self.target_encoder, momentum)])

    # ── representations ──────────────────────────────────────────────────

    def encode(self, x: Tensor) -> Tensor:
        """The representation downstream tasks read.

        Parameters
        ----------
        x : Tensor
            Clips ``(B, T, C, H, W)``.

        Returns
        -------
        Tensor
            ``(B, dim)``, the target encoder's tokens averaged.  The
            paper's own probe does better than this average — see
            :class:`VJEPAForVideoClassification` — but a single vector is
            what a backbone owes its caller.
        """
        return self.tokens(x).mean(dim=1)

    def tokens(self, x: Tensor) -> Tensor:
        """The target encoder's whole feature map, ``(B, N, dim)``.

        What the attentive probe reads, and what average-pooling in
        :meth:`encode` throws away.
        """
        self._check_clip(x)
        return self.target_encoder(x)

    def _check_clip(self, x: Tensor) -> None:
        config = self.config
        expected = (
            config.num_frames,
            config.in_channels,
            config.image_size,
            config.image_size,
        )
        if x.ndim != 5 or tuple(int(s) for s in x.shape[1:]) != expected:
            raise ValueError(
                f"clips must be (B, {expected[0]}, {expected[1]}, {expected[2]}, "
                f"{expected[3]}), got shape {tuple(x.shape)}"
            )

    def _discrepancy(self, prediction: Tensor, target: Tensor) -> Tensor:
        if self.config.objective == "l2":
            return F.mse_loss(prediction, target)
        if self.config.objective == "smooth_l1":
            return F.smooth_l1_loss(prediction, target, beta=self.config.smooth_l1_beta)
        return F.l1_loss(prediction, target)

    @override
    def forward(self, x: Tensor) -> VJEPAOutput:
        """Run one pretraining step's worth of computation.

        Parameters
        ----------
        x : Tensor
            Clips ``(B, T, C, H, W)`` at ``config.image_size``.

        Returns
        -------
        VJEPAOutput
            The averaged loss, and what each collection was asked and
            answered.
        """
        config = self.config
        self._check_clip(x)
        batch = int(x.shape[0])
        device = x.device

        short = _sample_collection(
            config, batch, config.short_range_blocks, config.short_range_scale, device
        )
        long = _sample_collection(
            config, batch, config.long_range_blocks, config.long_range_scale, device
        )

        with lucid.no_grad():
            features = self.target_encoder(x)
            features = F.layer_norm(features, (int(features.shape[-1]),))
            short_target = _gather_tokens(features, short[1])
            long_target = _gather_tokens(features, long[1])

        answers: list[Tensor] = []
        for index, (context_index, target_index) in enumerate((short, long)):
            context = self.encoder(x, context_index)
            answers.append(self.predictor(context, context_index, target_index, mask_index=index))

        loss = (
            self._discrepancy(answers[0], short_target) + self._discrepancy(answers[1], long_target)
        ) / 2.0
        return VJEPAOutput(
            loss=loss,
            short_prediction=answers[0],
            short_target=short_target,
            long_prediction=answers[1],
            long_target=long_target,
            short_context=short[0],
            short_targets=short[1],
            long_context=long[0],
            long_targets=long[1],
        )


class VJEPAForVideoClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""V-JEPA under the paper's attentive probe.

    Parameters
    ----------
    config : VJEPAConfig
        Frozen configuration; ``num_classes`` sizes the classifier.

    Attributes
    ----------
    vjepa : VJEPAModel
        The pretrained networks.  Its target encoder is frozen.
    pooler : Module
        The attentive pooler: one learned query over the feature map.
    head : nn.Linear
        The classifier on the pooled vector.

    Notes
    -----
    Reference: Bardes et al., arXiv:2404.08471, Section 4.3.  The paper
    evaluates a frozen backbone with this probe rather than with a linear
    map on averaged tokens, and reports the gap at 17 points on
    Kinetics-400 — so a linear probe compared against its tables would be
    measuring something else.

    Registered under ``image-classification`` because that is the task
    this zoo has; the input is a clip rather than an image, which is what
    the class name says.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vjepa import (
    ...     VJEPAConfig, VJEPAForVideoClassification)
    >>> config = VJEPAConfig(
    ...     image_size=32, patch_size=8, tubelet_size=2, num_frames=4,
    ...     dim=24, depth=1, num_heads=2, predictor_dim=12,
    ...     predictor_depth=1, num_classes=10)
    >>> model = VJEPAForVideoClassification(config).eval()
    >>> model(lucid.rand(2, 4, 3, 32, 32)).logits.shape
    (2, 10)
    """

    config_class: ClassVar[type[VJEPAConfig]] = VJEPAConfig

    def __init__(self, config: VJEPAConfig) -> None:
        """Build the probe. See the class docstring for parameters."""
        super().__init__(config)
        self.config: VJEPAConfig = config
        self.vjepa = VJEPAModel(config)
        self.pooler = _AttentivePooler(
            config.dim, config.num_heads, config.mlp_ratio, config.layer_norm_eps
        )
        self.head = nn.Linear(config.dim, config.num_classes)

    @override
    def forward(self, x: Tensor, labels: Tensor | None = None) -> ImageClassificationOutput:
        """Classify clips from the frozen feature map.

        Parameters
        ----------
        x : Tensor
            Clips ``(B, T, C, H, W)``.
        labels : Tensor or None, optional
            Class indices ``(B,)``; when given, the cross-entropy comes
            back with the logits.

        Returns
        -------
        ImageClassificationOutput
            Logits ``(B, num_classes)``, and the loss when labels came.
        """
        pooled = self.pooler(self.vjepa.tokens(x))
        logits = self.head(pooled)
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return ImageClassificationOutput(logits=logits, loss=loss)
