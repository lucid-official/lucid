r"""I-JEPA — predicting representations of blocks the encoder never saw.

Three networks and one idea.  The **context encoder** sees one large block
of an image.  The **target encoder** sees the whole image, and is never
trained: its weights are an exponential moving average of the context
encoder's, so what it says is the model's own opinion from a little while
ago.  The **predictor** is handed the context's tokens and the *positions*
of four blocks it was not given, and must say what the target encoder said
about them.

Nothing is reconstructed.  The loss lives in representation space, which is
what lets the model discard detail a pixel objective would have to keep.

Parameter names follow the ViT convention this zoo already uses
(``patch_embed.proj``, ``blocks.N.attn.qkv``, ``mlp.fc1``), which is also
the released implementation's, so a released checkpoint maps onto this
model by prefix alone.
"""

import math
import random
from dataclasses import dataclass
from typing import ClassVar, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._utils._common import host_random, update_averages
from lucid.models._output import ImageClassificationOutput, ModelOutput
from lucid.models._tasks import ImageClassificationModel
from lucid.models.vision._common._transformers import (
    Block as _Block,
    gather_tokens as _gather_tokens,
    init_transformer_weights as _init_transformer_weights,
    scale_residual_projection as _scale_residual_projection,
    sincos_embedding as _sincos_embedding,
)
from lucid.models.vision.ijepa._config import IJEPAConfig

__all__ = [
    "IJEPAModel",
    "IJEPAForImageClassification",
    "IJEPAOutput",
]


# ── positions ────────────────────────────────────────────────────────────────


def _sincos_2d(dim: int, grid: int) -> Tensor:
    """Fixed two-dimensional sine/cosine positions, ``(1, grid**2, dim)``.

    Half the width encodes the column and half the row, in that order —
    the convention the released implementation inherited from MAE.  These
    are *not* learned: the released code builds the table once and marks it
    as requiring no gradient, and making it learnable is a re-implementation
    error rather than a variation.
    """
    if dim % 4 != 0:
        raise ValueError(f"a two-dimensional sin-cos table needs dim % 4 == 0, got {dim}")
    coords = lucid.arange(grid).to(lucid.float64)
    zeros = lucid.zeros(grid, grid, dtype=lucid.float64)
    rows = coords.reshape(grid, 1) + zeros
    cols = coords.reshape(1, grid) + zeros
    table = lucid.cat(
        [
            _sincos_embedding(cols.reshape(-1), dim // 2),
            _sincos_embedding(rows.reshape(-1), dim // 2),
        ],
        dim=1,
    )
    return table.unsqueeze(dim=0)


# ── patches ──────────────────────────────────────────────────────────────────


class _PatchEmbed(nn.Module):
    """Non-overlapping patches, projected by one convolution."""

    def __init__(self, in_channels: int, patch_size: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        b, c, h, w = (int(s) for s in x.shape)
        return x.reshape(b, c, h * w).permute(0, 2, 1)


class _Encoder(nn.Module):
    """A ViT over patches, with no class token.

    Appendix A.1: *"I-JEPA is pretrained without a ``[cls]`` token. We use
    the target-encoder for evaluation and average pool its output"*.
    """

    pos_embed: Tensor

    def __init__(self, config: IJEPAConfig) -> None:
        super().__init__()
        hidden = int(config.dim * config.mlp_ratio)
        self.patch_embed = _PatchEmbed(config.in_channels, config.patch_size, config.dim)
        self.register_buffer("pos_embed", _sincos_2d(config.dim, config.grid_size))
        blocks = [
            _Block(config.dim, config.num_heads, hidden, config.layer_norm_eps)
            for _ in range(config.depth)
        ]
        self.blocks: nn.ModuleList[_Block] = nn.ModuleList([*blocks])
        self.norm = nn.LayerNorm(config.dim, eps=config.layer_norm_eps)

        # The released code — not the paper — initialises the tower at 0.02
        # and narrows each block's two residual writes by its depth.  A
        # fan-in default starts it several times wider, which nothing but a
        # from-scratch run would notice.
        _init_transformer_weights(self)
        for depth_index, block in enumerate(blocks, start=1):
            _scale_residual_projection(block.attn.proj, block.mlp.fc2, depth_index)

    @override
    def forward(self, x: Tensor, indices: Tensor | None = None) -> Tensor:
        """Encode an image, optionally keeping only the given patches.

        The positions are added *before* the tokens are selected, so a
        kept patch carries where it came from into the blocks.
        """
        tokens = self.patch_embed(x) + self.pos_embed
        if indices is not None:
            tokens = _gather_tokens(tokens, indices)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)


class _Predictor(nn.Module):
    """Predicts target representations from context tokens and positions.

    One learned mask token stands for "a patch you were not given", made
    distinct only by the fixed positional embedding added to it — so the
    predictor's whole input about a target is *where* it is.
    """

    predictor_pos_embed: Tensor

    def __init__(self, config: IJEPAConfig) -> None:
        super().__init__()
        width = config.predictor_dim
        hidden = int(width * config.mlp_ratio)
        heads = config.resolved_predictor_heads
        self.predictor_embed = nn.Linear(config.dim, width)
        self.mask_token = nn.Parameter(lucid.zeros(1, 1, width))
        init.trunc_normal_(self.mask_token, std=0.02)
        self.register_buffer("predictor_pos_embed", _sincos_2d(width, config.grid_size))
        blocks = [
            _Block(width, heads, hidden, config.layer_norm_eps)
            for _ in range(config.predictor_depth)
        ]
        self.predictor_blocks: nn.ModuleList[_Block] = nn.ModuleList([*blocks])
        self.predictor_norm = nn.LayerNorm(width, eps=config.layer_norm_eps)
        self.predictor_proj = nn.Linear(width, config.dim)

        # The predictor is initialised on its own terms, as its own tower;
        # the mask token above keeps the draw it was given, since only
        # linear, convolutional and norm layers are visited here.
        _init_transformer_weights(self)
        for depth_index, block in enumerate(blocks, start=1):
            _scale_residual_projection(block.attn.proj, block.mlp.fc2, depth_index)

    @override
    def forward(self, context: Tensor, context_indices: Tensor, target_indices: Tensor) -> Tensor:
        """Predict the target encoder's output at ``target_indices``."""
        positions = self.predictor_pos_embed
        tokens = self.predictor_embed(context)
        tokens = tokens + _gather_tokens(
            positions
            + lucid.zeros(
                int(context.shape[0]),
                1,
                1,
                dtype=positions.dtype,
                device=positions.device,
            ),
            context_indices,
        )

        batch, count = int(target_indices.shape[0]), int(target_indices.shape[1])
        width = int(tokens.shape[2])
        queries = self.mask_token + lucid.zeros(
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


# ── masking ──────────────────────────────────────────────────────────────────


def _block_shape(
    scale: tuple[float, float],
    aspect: tuple[float, float],
    grid: int,
    min_keep: int,
    rng: random.Random,
) -> tuple[int, int]:
    """Height and width, in patches, of one sampled block.

    The size is drawn once per batch rather than per image — the released
    code does this to keep every image's block the same length, and says so
    in its own issue tracker.  Scale and aspect are drawn *independently*
    here; the released code reuses one draw for both, which correlates them
    perfectly and reads as an oversight rather than a design.
    """
    patches = grid * grid
    for _ in range(20):
        area = patches * (rng.random() * (scale[1] - scale[0]) + scale[0])
        ratio = rng.random() * (aspect[1] - aspect[0]) + aspect[0]
        height = max(1, min(grid, int(round(math.sqrt(area * ratio)))))
        width = max(1, min(grid, int(round(math.sqrt(area / ratio)))))
        if height * width > min_keep:
            return height, width
    return max(1, min(grid, height)), max(1, min(grid, width))


def _block_indices(grid: int, height: int, width: int, rng: random.Random) -> list[int]:
    """Patch indices of one block placed uniformly at random.

    The last row and column are reachable.  The released code's ``randint``
    excludes them, which two of its open pull requests set out to fix.
    """
    top = rng.randrange(grid - height + 1)
    left = rng.randrange(grid - width + 1)
    return [(top + r) * grid + (left + c) for r in range(height) for c in range(width)]


def _sample_masks(config: IJEPAConfig, batch: int, device: DeviceLike) -> tuple[Tensor, Tensor]:
    """Context and target patch indices for one batch.

    Returns ``(B, Nc)`` context indices and ``(B, M, Nt)`` target indices.
    Every image in the batch gets blocks of the same *size* but its own
    *position*, and the context keeps a common number of patches: the
    shortest after target removal, truncated as the released code does.
    Every draw comes from one host stream, seeded from Lucid's generator.
    """
    rng = host_random()
    grid = config.grid_size
    target_h, target_w = _block_shape(
        config.target_scale, config.target_aspect, grid, config.min_keep, rng
    )
    context_h, context_w = _block_shape(
        config.context_scale, (1.0, 1.0), grid, config.min_keep, rng
    )

    targets: list[list[list[int]]] = []
    contexts: list[list[int]] = []
    for _ in range(batch):
        blocks = [
            _block_indices(grid, target_h, target_w, rng) for _ in range(config.num_target_blocks)
        ]
        targets.append(blocks)
        context = _block_indices(grid, context_h, context_w, rng)
        if not config.allow_overlap:
            taken = {index for block in blocks for index in block}
            context = [index for index in context if index not in taken]
        if not context:
            # Everything the context block held was also a target.  Keep the
            # patch the targets left free, or the first patch, rather than
            # handing the encoder an empty sequence.
            free = sorted(set(range(grid * grid)) - {i for b in blocks for i in b})
            context = free[:1] or [0]
        contexts.append(context)

    keep = min(len(context) for context in contexts)
    context_index = lucid.tensor(
        [context[:keep] for context in contexts], dtype=lucid.int64, device=device
    )
    target_index = lucid.tensor(targets, dtype=lucid.int64, device=device)
    return context_index, target_index


# ── outputs ──────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class IJEPAOutput(ModelOutput):
    r"""What one pretraining step produced.

    Attributes
    ----------
    loss : Tensor
        The discrepancy between predicted and target representations,
        averaged over target blocks.  Scalar.
    prediction : Tensor
        What the predictor said, ``(B, M, Nt, D)``.
    target : Tensor
        What the target encoder said, ``(B, M, Nt, D)``.  Carries no
        gradient.
    context_indices : Tensor
        Patches the context encoder was given, ``(B, Nc)``.
    target_indices : Tensor
        Patches each target block covers, ``(B, M, Nt)``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.ijepa import IJEPAOutput
    >>> out = IJEPAOutput(
    ...     loss=lucid.zeros(()),
    ...     prediction=lucid.zeros(1, 4, 9, 8),
    ...     target=lucid.zeros(1, 4, 9, 8),
    ...     context_indices=lucid.zeros(1, 20).to(lucid.int64),
    ...     target_indices=lucid.zeros(1, 4, 9).to(lucid.int64))
    >>> out.prediction.shape
    (1, 4, 9, 8)
    """

    loss: Tensor
    prediction: Tensor
    target: Tensor
    context_indices: Tensor
    target_indices: Tensor


# ── the model ────────────────────────────────────────────────────────────────


class IJEPAModel(PretrainedModel, BackboneMixin):
    r"""I-JEPA: a context encoder, an averaged target encoder, a predictor.

    Parameters
    ----------
    config : IJEPAConfig
        Frozen configuration.

    Attributes
    ----------
    encoder : Module
        The context encoder — the one that is trained.
    target_encoder : Module
        Its exponential moving average.  Frozen; produces the targets and,
        after pretraining, the representation everything downstream uses.
    predictor : Module
        The narrow transformer that answers for blocks it was not given.

    Notes
    -----
    Reference: Assran, Mahmoud, et al., *"Self-Supervised Learning from
    Images with a Joint-Embedding Predictive Architecture"*, CVPR 2023
    (arXiv:2301.08243), Section 3 and Appendix A.1.

    The target encoder is not updated by the optimiser.  Call
    :meth:`update_target` after each step, with
    :meth:`momentum` for the value the schedule asks for.

    Two details come from the released code rather than the paper: the
    target representations are layer-normalised (without affine terms)
    before the loss reads them, and that loss is smooth L1 where the paper
    writes :math:`\ell_2`.  ``config.objective`` selects between them.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.ijepa import IJEPAConfig, IJEPAModel
    >>> config = IJEPAConfig(
    ...     image_size=32, patch_size=8, dim=16, depth=1, num_heads=2,
    ...     predictor_dim=8, predictor_depth=1, min_keep=1,
    ...     target_scale=(0.1, 0.2), context_scale=(0.8, 1.0))
    >>> model = IJEPAModel(config)
    >>> out = model(lucid.rand(2, 3, 32, 32))
    >>> out.prediction.shape == out.target.shape
    True
    >>> float(out.loss.item()) >= 0.0
    True

    The representation a downstream task uses is the target encoder's,
    average-pooled:

    >>> model.encode(lucid.rand(2, 3, 32, 32)).shape
    (2, 16)
    """

    config_class: ClassVar[type[IJEPAConfig]] = IJEPAConfig

    def __init__(self, config: IJEPAConfig) -> None:
        """Build the three networks. See the class docstring for parameters."""
        super().__init__(config)
        self.config: IJEPAConfig = config
        self.encoder = _Encoder(config)
        self.target_encoder = _Encoder(config)
        self.predictor = _Predictor(config)

        # The target encoder starts as a copy and is never trained: the
        # paper's stop-gradient is structural, not a call site that might
        # be forgotten.
        nn.utils.copy_parameters_and_buffers(self.encoder, self.target_encoder)
        self.target_encoder.requires_grad_(False)

        self._feature_info = [
            FeatureInfo(stage=1, num_channels=config.dim, reduction=config.patch_size)
        ]

    # ── backbone ─────────────────────────────────────────────────────────

    @override
    @property
    def feature_info(self) -> list[FeatureInfo]:
        """One stage: patch tokens at the encoder's width."""
        return self._feature_info

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        """The average-pooled target encoder, which is :meth:`encode`.

        A backbone's features here are the representation the paper
        evaluates — the target encoder's, not the context encoder's.
        """
        return self.encode(x)

    # ── parameter groups ─────────────────────────────────────────────────

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Everything an optimiser should be given.

        Returns
        -------
        list of Parameter
            The context encoder and the predictor.  The target encoder is
            excluded: it follows by moving average, and handing it to an
            optimiser would train the thing that defines the target.
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
            Linear between ``config.ema[0]`` and ``config.ema[1]`` — 0.996
            to 1 at the paper's values, so the target stops moving exactly
            as training ends.
        """
        if total_steps < 1:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        start, end = self.config.ema
        fraction = min(max(step / float(total_steps), 0.0), 1.0)
        return start + (end - start) * fraction

    def update_target(self, momentum: float) -> None:
        """Move the target encoder toward the context encoder.

        Call it after the optimiser step: this writes parameters in place,
        and doing that while a graph that read them is alive would sever
        the two.

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
        """The representation downstream tasks use.

        Appendix A.1 evaluates the *target* encoder, average-pooled over
        patches — not the context encoder, which is a silent accuracy loss.

        Parameters
        ----------
        x : Tensor
            Images ``(B, C, H, W)``.

        Returns
        -------
        Tensor
            ``(B, dim)``.

        Notes
        -----
        Not wrapped in ``no_grad``: the target encoder's parameters are
        frozen already, so nothing accumulates into them, and a method a
        probe reads through should not sever the gradient a probe needs.
        The stop-gradient that matters is in :meth:`forward`, where the
        targets are built.
        """
        return self.target_encoder(x).mean(dim=1)

    def _discrepancy(self, prediction: Tensor, target: Tensor) -> Tensor:
        if self.config.objective == "l2":
            return F.mse_loss(prediction, target)
        return F.smooth_l1_loss(prediction, target, beta=self.config.smooth_l1_beta)

    @override
    def forward(self, x: Tensor) -> IJEPAOutput:
        """Run one pretraining step's worth of computation.

        Parameters
        ----------
        x : Tensor
            Images ``(B, C, H, W)`` at ``config.image_size``.

        Returns
        -------
        IJEPAOutput
            The loss and what it was computed from.
        """
        config = self.config
        expected = (config.in_channels, config.image_size, config.image_size)
        if x.ndim != 4 or tuple(int(s) for s in x.shape[1:]) != expected:
            raise ValueError(
                f"images must be (B, {expected[0]}, {expected[1]}, {expected[2]}), "
                f"got shape {tuple(x.shape)}"
            )

        batch = int(x.shape[0])
        context_index, target_index = _sample_masks(config, batch, x.device)

        with lucid.no_grad():
            tokens = self.target_encoder(x)
            # Not in the paper: the released code normalises the target
            # encoder's output over the feature axis, without affine terms,
            # before the blocks are taken from it.  Leaving it out is a
            # known way to fail to reproduce the results.
            tokens = F.layer_norm(tokens, (int(tokens.shape[-1]),))
            targets = lucid.stack(
                [
                    _gather_tokens(tokens, target_index[:, m])
                    for m in range(config.num_target_blocks)
                ],
                dim=1,
            )

        context = self.encoder(x, context_index)
        predictions = lucid.stack(
            [
                self.predictor(context, context_index, target_index[:, m])
                for m in range(config.num_target_blocks)
            ],
            dim=1,
        )
        return IJEPAOutput(
            loss=self._discrepancy(predictions, targets),
            prediction=predictions,
            target=targets,
            context_indices=context_index,
            target_indices=target_index,
        )


class IJEPAForImageClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""A linear probe on I-JEPA's representation.

    Parameters
    ----------
    config : IJEPAConfig
        Frozen configuration; ``num_classes`` sizes the head.

    Attributes
    ----------
    ijepa : IJEPAModel
        The pretrained networks.
    head : nn.Linear
        The probe, over the average-pooled target encoder.

    Notes
    -----
    Reference: Assran et al., arXiv:2301.08243, Appendix A.2 — the paper
    evaluates by linear probing the frozen target encoder, which is what
    this is.  The target encoder cannot be trained through this class: its
    parameters are frozen where they are built, so an optimiser handed
    everything still only moves the head.

    It carries the whole pretraining model — both encoders and the
    predictor — because that is what a checkpoint holds and what
    continuing to pretrain would need.  Inference uses one of the three,
    so an exported package is about three times the size it has to be.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.ijepa import (
    ...     IJEPAConfig, IJEPAForImageClassification)
    >>> config = IJEPAConfig(
    ...     image_size=32, patch_size=8, dim=16, depth=1, num_heads=2,
    ...     predictor_dim=8, predictor_depth=1, num_classes=10, min_keep=1)
    >>> model = IJEPAForImageClassification(config).eval()
    >>> model(lucid.rand(2, 3, 32, 32)).logits.shape
    (2, 10)
    """

    config_class: ClassVar[type[IJEPAConfig]] = IJEPAConfig

    def __init__(self, config: IJEPAConfig) -> None:
        """Build the probe. See the class docstring for parameters."""
        super().__init__(config)
        self.config: IJEPAConfig = config
        self.ijepa = IJEPAModel(config)
        self.head = nn.Linear(config.dim, config.num_classes)

    @override
    def forward(self, x: Tensor, labels: Tensor | None = None) -> ImageClassificationOutput:
        """Classify images from the frozen representation.

        Parameters
        ----------
        x : Tensor
            Images ``(B, C, H, W)``.
        labels : Tensor or None, optional
            Class indices ``(B,)``; when given, the cross-entropy is
            returned with the logits.

        Returns
        -------
        ImageClassificationOutput
            Logits ``(B, num_classes)``, and the loss when labels came.
        """
        logits = self.head(self.ijepa.encode(x))
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return ImageClassificationOutput(logits=logits, loss=loss)
