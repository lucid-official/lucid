r"""Genie — a world model learned from videos with no actions in them.

Three networks, each a spatiotemporal transformer.

The **video tokenizer** is a VQ-VAE over patches: it turns each frame into
a grid of codes from a 1024-entry codebook, and back.  Its temporal
attention is causal, so a frame's codes carry what the frames before it
showed.

The **latent action model** is what makes Genie possible.  Its encoder
reads the video and names each transition with one of eight codes; its
decoder has to rebuild the next frame from the frames before it and that
code alone.  Eight codes cannot carry a frame, so they learn to carry the
one thing the past does not predict — what the player did.

The **dynamics model** is a MaskGIT transformer over the tokenizer's
codes.  It fills in a masked frame from the frames before it plus the
embedding of a latent action, and at play time it reveals a frame over
several steps, most confident tokens first.

Each is trained on its own objective.  The dynamics model reads the
tokenizer's codes, which are indices and carry no gradient, and the latent
actions under a stop-gradient, so one ``backward`` on the summed loss
still reaches each network only through its own term.
"""

import math
from dataclasses import dataclass
from typing import ClassVar, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.nn.init as init
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._output import ModelOutput
from lucid.models._tasks import WorldModelingModel
from lucid.models._utils._generative import model_dtype
from lucid.models.generative.genie._config import GenieConfig

__all__ = ["GenieModel", "GenieForWorldModeling", "GenieOutput", "GenieRolloutOutput"]


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "elu": nn.ELU,
}


# ── patches ──────────────────────────────────────────────────────────────────


def _patchify(video: Tensor, patch: int) -> Tensor:
    """Cut ``(B, T, C, H, W)`` into ``(B, T, N, C * patch**2)``.

    A side that is not a multiple of ``patch`` is padded with zeros at the
    bottom or right.  The frames are folded into the batch first so no
    intermediate passes rank 6.
    """
    b, t, c, h, w = (int(s) for s in video.shape)
    frames = video.reshape(b * t, c, h, w)
    pad_h, pad_w = (-h) % patch, (-w) % patch
    if pad_h or pad_w:
        frames = F.pad(frames, (0, pad_w, 0, pad_h))
    rows, cols = (h + pad_h) // patch, (w + pad_w) // patch
    patches = frames.reshape(b * t, c, rows, patch, cols, patch)
    patches = patches.permute(0, 2, 4, 1, 3, 5)
    return patches.reshape(b, t, rows * cols, c * patch * patch)


def _unpatchify(
    x: Tensor, patch: int, grid: tuple[int, int], channels: int, frame: tuple[int, int]
) -> Tensor:
    """Invert :func:`_patchify`, cropping the padding away."""
    b, t = int(x.shape[0]), int(x.shape[1])
    rows, cols = grid
    y = x.reshape(b * t, rows, cols, channels, patch, patch).permute(0, 3, 1, 4, 2, 5)
    y = y.reshape(b * t, channels, rows * patch, cols * patch)
    h, w = frame
    if rows * patch != h or cols * patch != w:
        y = y[:, :, :h, :w]
    return y.reshape(b, t, channels, h, w)


def _check_video(config: GenieConfig, video: Tensor, name: str, min_frames: int) -> None:
    """Reject a video the configured networks cannot read."""
    if video.ndim != 5:
        raise ValueError(f"{name} must be (B, T, C, H, W), got shape {tuple(video.shape)}")
    t, c = int(video.shape[1]), int(video.shape[2])
    frame = (int(video.shape[3]), int(video.shape[4]))
    if c != config.in_channels or frame != config.frame_shape:
        raise ValueError(
            f"{name} frames must be {config.in_channels}x{config.frame_shape[0]}"
            f"x{config.frame_shape[1]}, got {c}x{frame[0]}x{frame[1]}"
        )
    if not min_frames <= t <= config.num_frames:
        raise ValueError(
            f"{name} must hold between {min_frames} and num_frames="
            f"{config.num_frames} frames, got {t}"
        )


# ── the spatiotemporal transformer ───────────────────────────────────────────


class _Attention(nn.Module):
    """Multi-head self-attention whose inner width is ``heads * head_dim``.

    The paper tabulates the query/key size apart from the model width —
    the dynamics model's 36 heads of 128 are 4608 wide inside a 5120-wide
    model — so the two are not tied here either.  The value size is not
    stated and is taken to be the query/key size.
    """

    def __init__(self, dim: int, heads: int, head_dim: int, qk_norm: bool) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        inner = heads * head_dim
        self.query = nn.Linear(dim, inner)
        self.key = nn.Linear(dim, inner)
        self.value = nn.Linear(dim, inner)
        self.out = nn.Linear(inner, dim)
        self.query_norm: nn.LayerNorm | None = nn.LayerNorm(head_dim) if qk_norm else None
        self.key_norm: nn.LayerNorm | None = nn.LayerNorm(head_dim) if qk_norm else None

    @override
    def forward(self, x: Tensor, causal: bool) -> Tensor:
        batch, length = int(x.shape[0]), int(x.shape[1])

        def heads(t: Tensor) -> Tensor:
            return t.reshape(batch, length, self.heads, self.head_dim).permute(0, 2, 1, 3)

        query = heads(self.query(x))
        key = heads(self.key(x))
        value = heads(self.value(x))
        if self.query_norm is not None and self.key_norm is not None:
            query = self.query_norm(query)
            key = self.key_norm(key)
        y = F.scaled_dot_product_attention(query, key, value, is_causal=causal)
        y = y.permute(0, 2, 1, 3).reshape(batch, length, self.heads * self.head_dim)
        return self.out(y)


class _STBlock(nn.Module):
    """Spatial attention, causal temporal attention, then one feed-forward.

    Section 2: *"we include only one FFW after both spatial and temporal
    components, omitting the post-spatial FFW"*.  Layer norms sit before
    each sub-layer — *not stated*; the pre-norm arrangement every
    transformer at this scale trains with.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        head_dim: int,
        hidden: int,
        activation: type[nn.Module],
        qk_norm: bool,
    ) -> None:
        super().__init__()
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial = _Attention(dim, heads, head_dim, qk_norm)
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal = _Attention(dim, heads, head_dim, qk_norm)
        self.ffw_norm = nn.LayerNorm(dim)
        self.ffw = nn.Sequential(nn.Linear(dim, hidden), activation(), nn.Linear(hidden, dim))

    @override
    def forward(self, x: Tensor) -> Tensor:
        b, t, n, d = (int(s) for s in x.shape)
        # Space: the N tokens of one frame attend to each other.
        spatial_in = self.spatial_norm(x).reshape(b * t, n, d)
        spatial = self.spatial(spatial_in, causal=False)
        x = x + spatial.reshape(b, t, n, d)
        # Time: the T tokens at one position, each seeing only the past.
        temporal_in = self.temporal_norm(x).permute(0, 2, 1, 3)
        temporal = self.temporal(temporal_in.reshape(b * n, t, d), causal=True)
        x = x + temporal.reshape(b, n, t, d).permute(0, 2, 1, 3)
        return x + self.ffw(self.ffw_norm(x))


class _STTransformer(nn.Module):
    """A stack of :class:`_STBlock` over ``(B, T, N, D)``.

    Positions are learned, one table per axis, added before the first
    block — *not stated*; factorised like the attention itself, so a frame
    and a patch position each cost one row.
    """

    def __init__(
        self, config: GenieConfig, stack: str, num_patches: int, qk_norm: bool = False
    ) -> None:
        super().__init__()
        dim: int = getattr(config, f"{stack}_dim")
        layers: int = getattr(config, f"{stack}_layers")
        heads: int = getattr(config, f"{stack}_heads")
        head_dim = config.attention_head_dim(stack)
        hidden = max(1, int(round(dim * config.mlp_ratio)))
        activation = _ACTIVATIONS[config.act_fn]
        self.num_frames = config.num_frames
        self.spatial_position = nn.Parameter(lucid.zeros(1, 1, num_patches, dim))
        self.temporal_position = nn.Parameter(lucid.zeros(1, config.num_frames, 1, dim))
        init.trunc_normal_(self.spatial_position, std=0.02)
        init.trunc_normal_(self.temporal_position, std=0.02)
        self.blocks = nn.ModuleList(
            [_STBlock(dim, heads, head_dim, hidden, activation, qk_norm) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        frames = int(x.shape[1])
        if frames > self.num_frames:
            raise ValueError(
                f"the transformer holds {self.num_frames} frames of positions, got {frames}"
            )
        x = x + self.spatial_position + self.temporal_position[:, :frames]
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# ── the codebook ─────────────────────────────────────────────────────────────

#: How fast a code's share of assignments is averaged over batches.  The
#: window is about a hundred batches, which has to be long enough that a
#: code the batch simply did not reach does not look dead: a batch holds
#: far fewer tokens than the tokenizer has codes, so a healthy code is
#: absent from most of them.
_USAGE_DECAY = 0.99


class _Codebook(nn.Module):
    """A vector quantiser whose unused codes can be moved back onto the data.

    Genie's action codebook holds eight entries, six in the CoinRun case
    study, and a codebook that small collapses readily: a few codes win
    every assignment, the rest never appear in a batch, so they receive no
    gradient and stay wherever initialisation left them.  The model then
    has three buttons rather than eight, which is invisible in the loss —
    the surviving codes simply take up the slack.  A fresh model really
    does start there: the first batch through an untrained latent action
    model picks one code out of six.

    :meth:`forward` only *watches*, averaging each code's share of the
    assignments over batches.  Moving a code is :meth:`revive`, which the
    training loop calls after the optimiser step —
    :meth:`GenieModel.revive_codes` does it for both codebooks.  It is
    separate for a reason that cost a silent defect to find: writing a
    parameter in place while a graph that read it is still alive severs
    the two, and the codebook's gradient stops arriving with no error
    raised.  A forward that revived on its own would do exactly that to
    anyone accumulating gradients over several batches.

    The paper says nothing about any of this; Jafar resets dead codes on
    a schedule.  ``reset_threshold=0`` switches it off.
    """

    usage: Tensor
    candidates: Tensor

    def __init__(
        self, num_codes: int, dim: int, commitment_cost: float, reset_threshold: float
    ) -> None:
        super().__init__()
        self.num_codes = num_codes
        self.reset_threshold = reset_threshold
        self.quantizer = nn.VectorQuantizer(num_codes, dim, commitment_cost=commitment_cost)
        self.register_buffer("usage", lucid.ones(num_codes) / float(num_codes))
        # Where a revived code would land: rows drawn from the last batch
        # this saw.  Kept so that reviving needs no data of its own, and
        # not persistent — a checkpoint has no use for one old batch.
        self.register_buffer("candidates", lucid.zeros(num_codes, dim), persistent=False)

    @property
    def weight(self) -> nn.Parameter:
        """The codebook itself, ``(num_codes, dim)``."""
        return self.quantizer.weight

    def lookup(self, indices: Tensor) -> Tensor:
        """Map an index field to its codes."""
        return self.quantizer.lookup(indices)

    def loss(self, out: nn.VectorQuantizerOutput) -> Tensor:
        """The codebook and commitment terms, weighted as the quantiser is."""
        return self.quantizer.loss(out)

    @override
    def forward(self, x: Tensor) -> nn.VectorQuantizerOutput:
        """Quantise ``(*, dim)``, noting which codes were used.

        Nothing moves here.  On a training step each code's share of the
        assignments is folded into a running average and a few rows of
        ``x`` are kept for :meth:`revive` to draw on; both are buffers, so
        no graph depends on them.

        A pass with gradients off is not a training step and is not
        counted: a rollout run to look at the model would otherwise age
        the averages, and so would the frozen re-tokenisation inside
        :meth:`GenieModel.dynamics_loss`, which would count every batch
        twice in the staged schedule the paper trains under.
        """
        out = self.quantizer(x)
        if self.training and self.reset_threshold > 0.0 and lucid.is_grad_enabled():
            self._observe(x, out.indices)
        return out

    def revive(self) -> None:
        """Move codes nothing is using onto points from the last batch.

        Call it after the optimiser step, when no graph is alive: this
        writes the codebook in place.

        A code whose share of the assignments, averaged over batches, has
        fallen below ``reset_threshold`` of its fair share is moved onto
        an encoder output, where there is something for it to explain.  At
        the default threshold that takes roughly two hundred batches of
        disuse, which is deliberately slow: reviving a code that was
        merely unlucky costs the codebook an entry that was working, and a
        batch holds far fewer tokens than the tokenizer has codes.

        There is no branch on how many codes are dead — the blend below is
        the identity when there are none, and counting them would mean a
        device synchronisation on every step.
        """
        if self.reset_threshold <= 0.0:
            return
        fair = 1.0 / float(self.num_codes)
        with lucid.no_grad():
            usage = self.usage
            dead = (usage < self.reset_threshold * fair).to(usage.dtype)
            weight = self.quantizer.weight
            # Writing a parameter in place gives it a new buffer and drops
            # whatever gradient it was holding.  After the optimiser step
            # that is what one wants, but reviving before the step would
            # then leave the step with nothing to apply and no error to
            # show for it, so the gradient is put back.
            pending = weight.grad
            revived = dead.unsqueeze(-1)
            candidates = self.candidates
            weight[:] = weight * (1.0 - revived) + candidates * revived
            if pending is not None:
                weight.grad = pending
            # A revived code starts from a fair share, so the batches after
            # it judge where it landed rather than where it came from.
            usage[:] = usage * (1.0 - dead) + fair * dead

    def _observe(self, x: Tensor, indices: Tensor) -> None:
        """Fold this batch into the usage average, and keep somewhere to land.

        The shares come from a histogram rather than a one-hot: the
        tokenizer assigns 920 tokens a frame over a codebook of 1024, so
        a dense ``(tokens, codes)`` indicator of one training batch at the
        paper's sizes would be tens of gigabytes to produce 1024 numbers.
        """
        flat = x.reshape(-1, self.quantizer.embedding_dim).detach()
        flat_indices = indices.reshape(-1)
        counts = lucid.bincount(flat_indices, minlength=self.num_codes)
        share = counts.to(flat.dtype) / float(int(flat_indices.shape[0]))
        picks = lucid.randint(0, int(flat.shape[0]), (self.num_codes,), device=flat.device)
        with lucid.no_grad():
            usage = self.usage
            usage[:] = _USAGE_DECAY * usage + (1.0 - _USAGE_DECAY) * share
            self.candidates[:] = flat[picks]


# ── the three networks ───────────────────────────────────────────────────────


class _VideoTokenizer(nn.Module):
    """ST-ViViT VQ-VAE: frames to a grid of codes and back (Section 2.1)."""

    def __init__(self, config: GenieConfig) -> None:
        super().__init__()
        self.config = config
        patch = config.tokenizer_patch_size
        rows, cols = config.token_grid
        self.encoder_in = nn.Linear(
            config.in_channels * patch * patch, config.tokenizer_encoder_dim
        )
        self.encoder = _STTransformer(config, "tokenizer_encoder", rows * cols)
        self.to_code = nn.Linear(config.tokenizer_encoder_dim, config.code_dim)
        self.quantizer = _Codebook(
            config.num_codes, config.code_dim, config.commitment_cost, config.code_reset_threshold
        )
        self.decoder_in = nn.Linear(config.code_dim, config.tokenizer_decoder_dim)
        self.decoder = _STTransformer(config, "tokenizer_decoder", rows * cols)
        self.to_pixels = nn.Linear(
            config.tokenizer_decoder_dim, config.out_channels * patch * patch
        )

    def quantize(self, video: Tensor) -> nn.VectorQuantizerOutput:
        """Encode and quantise ``(B, T, C, H, W)`` to ``(B, T, N)`` codes."""
        patches = _patchify(video, self.config.tokenizer_patch_size)
        hidden = self.encoder(self.encoder_in(patches))
        codes = self.to_code(hidden)
        return self.quantizer(codes)

    def decode(self, codes: Tensor) -> Tensor:
        """Decode code vectors ``(B, T, N, code_dim)`` to frames in ``[0, 1]``.

        The output is squashed by a sigmoid — *not stated* for the
        tokenizer; Appendix C.1 states it for the latent action model's
        decoder, whose targets are the same normalised frames.
        """
        config = self.config
        hidden = self.decoder(self.decoder_in(codes))
        patches = self.to_pixels(hidden)
        frames = _unpatchify(
            patches,
            config.tokenizer_patch_size,
            config.token_grid,
            config.out_channels,
            config.frame_shape,
        )
        return lucid.sigmoid(frames)


class _LatentActionModel(nn.Module):
    """Names each transition with one of ``num_latent_actions`` codes.

    Section 2.1 and Appendix C.1: pixels in, normalised to ``[0, 1]``,
    and a sigmoid on the decoder's output.
    """

    def __init__(self, config: GenieConfig) -> None:
        super().__init__()
        self.config = config
        patch = config.action_patch_size
        rows, cols = config.action_grid
        pixels = config.in_channels * patch * patch
        self.encoder_in = nn.Linear(pixels, config.action_encoder_dim)
        self.encoder = _STTransformer(config, "action_encoder", rows * cols)
        self.to_action = nn.Linear(config.action_encoder_dim, config.action_dim)
        self.quantizer = _Codebook(
            config.num_latent_actions,
            config.action_dim,
            config.commitment_cost,
            config.code_reset_threshold,
        )
        self.decoder_in = nn.Linear(pixels, config.action_decoder_dim)
        self.action_in = nn.Linear(config.action_dim, config.action_decoder_dim)
        self.decoder = _STTransformer(config, "action_decoder", rows * cols)
        self.to_pixels = nn.Linear(config.action_decoder_dim, config.out_channels * patch * patch)

    def quantize(self, video: Tensor) -> nn.VectorQuantizerOutput:
        """Latent actions ``(B, T - 1)`` for the transitions of ``(B, T, C, H, W)``.

        The encoder's output at frame ``t + 1`` is the first to have seen
        the transition into it — the temporal mask hides that frame from
        every earlier output — and its patches are averaged into one
        vector.  Which output becomes the action is *not stated*.
        """
        patches = _patchify(video, self.config.action_patch_size)
        hidden = self.encoder(self.encoder_in(patches))
        vectors = self.to_action(hidden[:, 1:].mean(dim=2))
        return self.quantizer(vectors)

    def decode(self, video: Tensor, actions: Tensor) -> Tensor:
        """Predict frames ``2..T`` from the frames before each and its action.

        The action is added to every patch of the frame it leaves —
        Section 2.1 found additive embeddings better than concatenation.
        """
        config = self.config
        past = _patchify(video[:, :-1], config.action_patch_size)
        steps = self.action_in(actions).unsqueeze(2)
        hidden = self.decoder_in(past) + steps
        patches = self.to_pixels(self.decoder(hidden))
        frames = _unpatchify(
            patches,
            config.action_patch_size,
            config.action_grid,
            config.out_channels,
            config.frame_shape,
        )
        return lucid.sigmoid(frames)


class _DynamicsModel(nn.Module):
    """Decoder-only MaskGIT transformer over tokenizer codes (Section 2.1)."""

    def __init__(self, config: GenieConfig) -> None:
        super().__init__()
        rows, cols = config.token_grid
        self.mask_token = config.num_codes
        self.token_embedding = nn.Embedding(config.num_codes + 1, config.dynamics_dim)
        self.action_embedding = nn.Linear(config.action_dim, config.dynamics_dim)
        self.transformer = _STTransformer(
            config, "dynamics", rows * cols, qk_norm=config.dynamics_qk_norm
        )
        self.to_logits = nn.Linear(config.dynamics_dim, config.num_codes)

    @override
    def forward(self, tokens: Tensor, actions: Tensor) -> Tensor:
        """Logits ``(B, T, N, num_codes)`` for tokens ``(B, T, N)``.

        ``actions`` is ``(B, T - 1, action_dim)``: action ``t`` is added to
        frame ``t + 1``, the frame it leads to, and the first frame gets
        none.
        """
        hidden = self.token_embedding(tokens)
        frames = int(tokens.shape[1])
        if frames > 1:
            steps = self.action_embedding(actions)
            first = lucid.zeros(
                int(steps.shape[0]), 1, int(steps.shape[2]), dtype=steps.dtype, device=steps.device
            )
            hidden = hidden + lucid.cat([first, steps], dim=1).unsqueeze(2)
        return self.to_logits(self.transformer(hidden))


# ── outputs ──────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class GenieOutput(ModelOutput):
    r"""The three training objectives, and what each was computed from.

    Attributes
    ----------
    loss : Tensor
        Sum of the three objectives.  Scalar.  Each reaches only its own
        network, so one ``backward`` trains all three independently.
    tokenizer_loss : Tensor
        Reconstruction error of the video tokenizer plus its codebook and
        commitment terms.
    latent_action_loss : Tensor
        Error predicting frames ``2..T`` from the frames before each and a
        latent action, plus the action codebook terms.
    dynamics_loss : Tensor
        Cross-entropy over the masked tokens of frames ``2..T``.
    reconstruction : Tensor
        The tokenizer's reconstruction, ``(B, T, C, H, W)``.
    prediction : Tensor
        The latent action model's predicted frames, ``(B, T - 1, C, H, W)``.
    tokens : Tensor
        The tokenizer's codes, ``(B, T, N)``.
    actions : Tensor
        The latent action of each transition, ``(B, T - 1)``.
    logits : Tensor
        The dynamics model's logits, ``(B, T, N, num_codes)``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.genie import GenieOutput
    >>> zero = lucid.zeros(())
    >>> out = GenieOutput(
    ...     loss=zero, tokenizer_loss=zero, latent_action_loss=zero,
    ...     dynamics_loss=zero, reconstruction=lucid.zeros(1, 4, 3, 8, 8),
    ...     prediction=lucid.zeros(1, 3, 3, 8, 8),
    ...     tokens=lucid.zeros(1, 4, 4).to(lucid.int64),
    ...     actions=lucid.zeros(1, 3).to(lucid.int64),
    ...     logits=lucid.zeros(1, 4, 4, 16))
    >>> out.actions.shape
    (1, 3)
    """

    loss: Tensor
    tokenizer_loss: Tensor
    latent_action_loss: Tensor
    dynamics_loss: Tensor
    reconstruction: Tensor
    prediction: Tensor
    tokens: Tensor
    actions: Tensor
    logits: Tensor


@dataclass(slots=True)
class GenieRolloutOutput(ModelOutput):
    r"""Frames generated by playing the learned environment.

    Attributes
    ----------
    frames : Tensor
        One generated frame per action, ``(B, S, C, H, W)``, in ``[0, 1]``.
    tokens : Tensor
        The codes of the prompt and every generated frame,
        ``(B, P + S, N)``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.genie import GenieRolloutOutput
    >>> out = GenieRolloutOutput(
    ...     frames=lucid.zeros(1, 2, 3, 8, 8),
    ...     tokens=lucid.zeros(1, 3, 4).to(lucid.int64))
    >>> out.frames.shape
    (1, 2, 3, 8, 8)
    """

    frames: Tensor
    tokens: Tensor


# ── the model ────────────────────────────────────────────────────────────────


class GenieModel(PretrainedModel):
    r"""Genie's three networks and their objectives.

    Parameters
    ----------
    config : GenieConfig
        Frozen configuration.

    Attributes
    ----------
    tokenizer : Module
        The video tokenizer.
    latent_action_model : Module
        The latent action model.
    dynamics : Module
        The MaskGIT dynamics model.

    Notes
    -----
    Reference: Bruce et al., ICML 2024, Section 2.

    The paper trains the tokenizer first and the other two after it, on
    its codes.  ``forward`` computes all three objectives in one pass;
    staging is a matter of which parameter group an optimiser is given —
    :meth:`tokenizer_parameters`, :meth:`latent_action_parameters` and
    :meth:`dynamics_parameters`.

    Videos are ``(B, T, C, H, W)`` in ``[0, 1]`` with
    ``2 <= T <= num_frames``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.genie import GenieConfig, GenieModel
    >>> config = GenieConfig(
    ...     sample_size=(8, 8), num_frames=4, num_codes=16, code_dim=4,
    ...     tokenizer_encoder_layers=1, tokenizer_encoder_dim=16,
    ...     tokenizer_encoder_heads=2, tokenizer_encoder_head_dim=8,
    ...     tokenizer_decoder_layers=1, tokenizer_decoder_dim=16,
    ...     tokenizer_decoder_heads=2, tokenizer_decoder_head_dim=8,
    ...     action_patch_size=4, action_dim=4,
    ...     action_encoder_layers=1, action_encoder_dim=16, action_encoder_heads=2,
    ...     action_decoder_layers=1, action_decoder_dim=16, action_decoder_heads=2,
    ...     dynamics_layers=1, dynamics_dim=16, dynamics_heads=2,
    ...     dynamics_head_dim=8, maskgit_steps=2)
    >>> model = GenieModel(config)
    >>> out = model(lucid.rand(2, 4, 3, 8, 8))
    >>> out.tokens.shape, out.actions.shape, out.logits.shape
    ((2, 4, 4), (2, 3), (2, 4, 4, 16))
    """

    config_class: ClassVar[type[GenieConfig]] = GenieConfig

    def __init__(self, config: GenieConfig) -> None:
        """Initialise the three networks. See the class docstring for parameters."""
        super().__init__(config)
        self.config: GenieConfig = config
        self.tokenizer = _VideoTokenizer(config)
        self.latent_action_model = _LatentActionModel(config)
        self.dynamics = _DynamicsModel(config)

    # ── parameter groups ─────────────────────────────────────────────────

    def tokenizer_parameters(self) -> list[nn.Parameter]:
        """The video tokenizer's parameters, trained by ``tokenizer_loss``.

        Returns
        -------
        list of Parameter
            Encoder, codebook and decoder.
        """
        return list(self.tokenizer.parameters())

    def latent_action_parameters(self) -> list[nn.Parameter]:
        """The latent action model's parameters, trained by ``latent_action_loss``.

        Returns
        -------
        list of Parameter
            Encoder, action codebook and decoder.
        """
        return list(self.latent_action_model.parameters())

    def dynamics_parameters(self) -> list[nn.Parameter]:
        """The dynamics model's parameters, trained by ``dynamics_loss``.

        Returns
        -------
        list of Parameter
            Embeddings, transformer and output head.
        """
        return list(self.dynamics.parameters())

    def revive_codes(self) -> None:
        """Move unused codes of both codebooks back onto the data.

        Call it after the optimiser step.  Genie's action codebook holds
        eight entries and collapses to a handful without this; the loss
        does not show it, because the codes that survive take up the
        slack.  It writes the codebooks in place, which is why it is a
        call of its own rather than part of ``forward``: doing it while a
        graph that read them is alive severs the two, and their gradients
        stop arriving silently.

        Does nothing when ``config.code_reset_threshold`` is 0.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models import genie_coinrun
        >>> model = genie_coinrun(
        ...     sample_size=(8, 8), num_frames=2, num_codes=8, code_dim=4,
        ...     tokenizer_encoder_layers=1, tokenizer_encoder_dim=8,
        ...     tokenizer_encoder_heads=1, tokenizer_decoder_layers=1,
        ...     tokenizer_decoder_dim=8, tokenizer_decoder_heads=1,
        ...     action_patch_size=4, action_dim=4, action_encoder_layers=1,
        ...     action_encoder_dim=8, action_encoder_heads=1,
        ...     action_decoder_layers=1, action_decoder_dim=8,
        ...     action_decoder_heads=1, dynamics_layers=1, dynamics_dim=8,
        ...     dynamics_heads=1)
        >>> optimizer = lucid.optim.Adam(model.parameters(), lr=1e-3)
        >>> model(lucid.rand(1, 2, 3, 8, 8)).loss.backward()
        >>> optimizer.step()
        >>> model.revive_codes()
        """
        self.tokenizer.quantizer.revive()
        self.latent_action_model.quantizer.revive()

    # ── codes ────────────────────────────────────────────────────────────

    def tokenize(self, video: Tensor) -> Tensor:
        """Encode a video to tokenizer codes.

        Parameters
        ----------
        video : Tensor
            ``(B, T, C, H, W)`` in ``[0, 1]``, with ``1 <= T <= num_frames``.

        Returns
        -------
        Tensor
            Codes ``(B, T, N)``.  The encoder is causal in time, so frame
            ``t``'s codes depend on frames ``1..t`` only.
        """
        _check_video(self.config, video, "video", 1)
        return self.tokenizer.quantize(video).indices

    def detokenize(self, tokens: Tensor) -> Tensor:
        """Decode tokenizer codes to frames.

        Parameters
        ----------
        tokens : Tensor
            Codes ``(B, T, N)`` with ``T <= num_frames``.

        Returns
        -------
        Tensor
            Frames ``(B, T, C, H, W)`` in ``[0, 1]``.
        """
        return self.tokenizer.decode(self.tokenizer.quantizer.lookup(tokens))

    def infer_actions(self, video: Tensor) -> Tensor:
        """Name each transition of a video with a latent action.

        This is how a real video is replayed through the model: its first
        frame and the actions inferred here reproduce it (Section 2.2).

        Parameters
        ----------
        video : Tensor
            ``(B, T, C, H, W)`` in ``[0, 1]``, with ``2 <= T <= num_frames``.

        Returns
        -------
        Tensor
            Latent actions ``(B, T - 1)``, each in ``[0, num_latent_actions)``.
        """
        _check_video(self.config, video, "video", 2)
        return self.latent_action_model.quantize(video).indices

    # ── objectives ───────────────────────────────────────────────────────

    def tokenizer_loss(self, video: Tensor) -> Tensor:
        r"""The video tokenizer's VQ-VAE objective.

        Parameters
        ----------
        video : Tensor
            ``(B, T, C, H, W)`` in ``[0, 1]``.

        Returns
        -------
        Tensor
            Scalar: mean squared reconstruction error plus the codebook
            term and :math:`\beta` times the commitment term.
        """
        _check_video(self.config, video, "video", 1)
        loss, _, _ = self._tokenizer_objective(video)
        return loss

    def latent_action_loss(self, video: Tensor) -> Tensor:
        """The latent action model's objective.

        Parameters
        ----------
        video : Tensor
            ``(B, T, C, H, W)`` in ``[0, 1]``, with ``T >= 2``.

        Returns
        -------
        Tensor
            Scalar: mean squared error of the predicted frames ``2..T``
            plus the action codebook's terms.
        """
        _check_video(self.config, video, "video", 2)
        loss, _, _ = self._latent_action_objective(video)
        return loss

    def dynamics_loss(self, video: Tensor) -> Tensor:
        """The dynamics model's MaskGIT objective, on frozen codes and actions.

        The video is tokenized and its actions inferred without gradient —
        the stop-gradient of Section 2.1 — so this trains the dynamics
        model and nothing else.

        Parameters
        ----------
        video : Tensor
            ``(B, T, C, H, W)`` in ``[0, 1]``, with ``T >= 2``.

        Returns
        -------
        Tensor
            Scalar cross-entropy over the masked tokens.
        """
        _check_video(self.config, video, "video", 2)
        with lucid.no_grad():
            tokens = self.tokenizer.quantize(video).indices
            actions = self.latent_action_model.quantize(video).quantized
        loss, _, _ = self._dynamics_objective(tokens, actions)
        return loss

    def _tokenizer_objective(self, video: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        vq = self.tokenizer.quantize(video)
        reconstruction = self.tokenizer.decode(vq.quantized)
        loss = F.mse_loss(reconstruction, video) + self.tokenizer.quantizer.loss(vq)
        return loss, reconstruction, vq.indices

    def _latent_action_objective(
        self, video: Tensor
    ) -> tuple[Tensor, Tensor, nn.VectorQuantizerOutput]:
        vq = self.latent_action_model.quantize(video)
        prediction = self.latent_action_model.decode(video, vq.quantized)
        loss = F.mse_loss(prediction, video[:, 1:]) + self.latent_action_model.quantizer.loss(vq)
        return loss, prediction, vq

    def _dynamics_objective(self, tokens: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Mask frames ``2..T`` at a random rate and predict what was masked.

        Section 2.1 draws the rate from :math:`U[0.5, 1]` and masks the
        input tokens with a Bernoulli per token.  Two readings of the rest
        of that sentence differ, and this is the second:

        * the paper's words — take ``z_{1:T-1}``, predict ``ẑ_{2:T}``,
          cross-entropy against all of ``z_{2:T}``;
        * in place — frame ``t + 1``'s masked tokens, plus action ``t``,
          predict frame ``t + 1``, cross-entropy over what was masked.

        The first cannot be refined over 25 MaskGIT steps, because a
        frame's own partially revealed tokens never re-enter the model;
        the open reimplementations all take the second.  The first frame
        is never masked: at play time it is the prompt.
        """
        config = self.config
        b, t, n = (int(s) for s in tokens.shape)
        device = tokens.device
        span = config.mask_ratio_max - config.mask_ratio_min
        # Section 2.1 samples "uniformly between 0.5 and 1" — an interval
        # closed at the top, while ``rand`` draws [0, 1).  The draw is
        # mirrored so the reachable end is the upper one: a rate of 1
        # masks a whole frame, which is where every generated frame starts
        # at play time, and a rate of exactly 0.5 is the one lost instead.
        rate = config.mask_ratio_max - lucid.rand(b, 1, 1, device=device) * span
        dt = model_dtype(self)  # the loss weights meet the logits
        later = (lucid.arange(t, device=device) > 0).to(dt).reshape(1, t, 1)
        mask = (lucid.rand(b, t, n, device=device) < rate).to(dt) * later
        hide = mask.to(tokens.dtype)
        inputs = tokens + (self.dynamics.mask_token - tokens) * hide
        logits = self.dynamics(inputs, actions)
        per_token = F.cross_entropy(
            logits.reshape(b * t * n, config.num_codes), tokens.reshape(b * t * n), reduction="none"
        )
        flat_mask = mask.reshape(b * t * n)
        loss = (per_token * flat_mask).sum() / flat_mask.sum().clamp(min=1.0)
        return loss, logits, mask

    @override
    def forward(self, video: Tensor) -> GenieOutput:
        """Compute all three objectives on one video batch.

        Parameters
        ----------
        video : Tensor
            ``(B, T, C, H, W)`` in ``[0, 1]``, with ``2 <= T <= num_frames``.

        Returns
        -------
        GenieOutput
            The summed loss, each objective, and what they were computed
            from.
        """
        _check_video(self.config, video, "video", 2)
        tokenizer_loss, reconstruction, tokens = self._tokenizer_objective(video)
        action_loss, prediction, actions = self._latent_action_objective(video)
        dynamics_loss, logits, _ = self._dynamics_objective(tokens, actions.quantized.detach())
        return GenieOutput(
            loss=tokenizer_loss + action_loss + dynamics_loss,
            tokenizer_loss=tokenizer_loss,
            latent_action_loss=action_loss,
            dynamics_loss=dynamics_loss,
            reconstruction=reconstruction,
            prediction=prediction,
            tokens=tokens,
            actions=actions.indices,
            logits=logits,
        )


class GenieForWorldModeling(WorldModelingModel):
    r"""Genie played as an environment: a prompt frame, then latent actions.

    Parameters
    ----------
    config : GenieConfig
        Frozen configuration.

    Attributes
    ----------
    genie : GenieModel
        The three networks and their objectives.

    Notes
    -----
    Reference: Bruce et al., ICML 2024, Section 2.2.

    A user's action is an index into the latent action codebook.  The rest
    of the latent action model is needed only to fill in the actions
    *between* prompt frames, which a prompt of one frame does not have —
    Section 2.2 replays a real video that way, from its first frame and
    the actions the model infers from it.  Each frame is revealed over
    ``maskgit_steps`` steps at ``temperature``: at every step each still
    masked token is sampled, and the most confident samples are kept, so
    many remain masked under a cosine schedule until the last step
    reveals them all.  The schedule and the confidence rule are MaskGIT's
    — *not stated* in the paper.

    The model remembers ``num_frames`` frames, as the paper's does.  A
    longer rollout keeps the most recent ones — *not stated*; Section 5
    names the 16-frame memory as a limitation and says nothing more.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.genie import GenieConfig, GenieForWorldModeling
    >>> config = GenieConfig(
    ...     sample_size=(8, 8), num_frames=4, num_codes=16, code_dim=4,
    ...     tokenizer_encoder_layers=1, tokenizer_encoder_dim=16,
    ...     tokenizer_encoder_heads=2, tokenizer_encoder_head_dim=8,
    ...     tokenizer_decoder_layers=1, tokenizer_decoder_dim=16,
    ...     tokenizer_decoder_heads=2, tokenizer_decoder_head_dim=8,
    ...     action_patch_size=4, action_dim=4,
    ...     action_encoder_layers=1, action_encoder_dim=16, action_encoder_heads=2,
    ...     action_decoder_layers=1, action_decoder_dim=16, action_decoder_heads=2,
    ...     dynamics_layers=1, dynamics_dim=16, dynamics_heads=2,
    ...     dynamics_head_dim=8, maskgit_steps=2)
    >>> model = GenieForWorldModeling(config).eval()
    >>> prompt = lucid.rand(1, 1, 3, 8, 8)
    >>> actions = lucid.tensor([[0, 5, 7, 2, 1]], dtype=lucid.int64)
    >>> out = model(prompt, actions)
    >>> out.frames.shape, out.tokens.shape
    ((1, 5, 3, 8, 8), (1, 6, 4))
    """

    config_class: ClassVar[type[GenieConfig]] = GenieConfig

    def __init__(self, config: GenieConfig) -> None:
        """Initialise the wrapper. See the class docstring for parameters."""
        super().__init__(config)
        self.config: GenieConfig = config
        self.genie = GenieModel(config)

    def predict_frame(self, tokens: Tensor, actions: Tensor) -> Tensor:
        """Generate the next frame's codes by MaskGIT decoding.

        Parameters
        ----------
        tokens : Tensor
            Codes of the frames so far, ``(B, t, N)``.  Only the last
            ``num_frames - 1`` are read.
        actions : Tensor
            Latent action vectors ``(B, t, action_dim)``: entry ``i`` is
            the action taken from frame ``i``, so the last is the one that
            leads to the frame being generated.

        Returns
        -------
        Tensor
            Codes of the next frame, ``(B, N)``.
        """
        config = self.config
        dynamics = self.genie.dynamics
        if int(actions.shape[1]) != int(tokens.shape[1]):
            # One action per frame here, unlike every other action-taking
            # method, which takes one per transition.  Passing that shape
            # raises early in a rollout and silently conditions on the
            # previous frame's action once the window is full, so it is
            # refused rather than truncated into agreement.
            raise ValueError(
                f"predict_frame takes the action leaving each frame so far, so "
                f"one per frame: got {int(actions.shape[1])} actions for "
                f"{int(tokens.shape[1])} frames"
            )
        context = config.num_frames - 1
        tokens, actions = tokens[:, -context:], actions[:, -context:]
        b, n = int(tokens.shape[0]), int(tokens.shape[2])
        device = tokens.device
        frame = lucid.full((b, n), float(dynamics.mask_token), dtype=tokens.dtype, device=device)
        masked = lucid.ones(b, n, device=device)
        hidden = n
        for step in range(config.maskgit_steps):
            ratio = math.cos(math.pi / 2.0 * (step + 1) / config.maskgit_steps)
            remaining = min(hidden, math.floor(n * ratio))
            reveal = hidden - remaining
            if reveal <= 0:
                continue
            sequence = lucid.cat([tokens, frame.unsqueeze(1)], dim=1)
            logits = dynamics(sequence, actions)[:, -1]
            drawn = F.softmax(logits / config.temperature, dim=-1)
            sampled = lucid.multinomial(drawn.reshape(b * n, config.num_codes), 1)
            sampled = sampled.reshape(b, n).to(tokens.dtype)
            # The temperature shapes the draw, which is what Section 3
            # gives it for.  The confidence that orders the reveal is the
            # model's own probability of the token it drew, as MaskGIT
            # defines it: a tempered probability compresses a peaked
            # distribution more than a flat one, so ranking by it compares
            # positions on a scale that depends on how sure they are.
            belief = F.softmax(logits, dim=-1)
            confidence = lucid.gather(belief, -1, sampled.unsqueeze(-1)).squeeze(-1)
            # A revealed token scores below any probability, so it is never
            # chosen again; the threshold is the reveal-th best score.
            score = confidence * masked - (1.0 - masked)
            best, _ = lucid.topk(score, reveal, dim=-1)
            threshold = best.min(dim=-1, keepdim=True).values
            chosen = (score >= threshold).to(lucid.float32) * masked
            frame = frame + (sampled - frame) * chosen.to(tokens.dtype)
            masked = masked * (1.0 - chosen)
            hidden = remaining
        return frame

    def _decode(self, tokens: Tensor, start: int) -> Tensor:
        """Frames for codes ``tokens[:, start:]``.

        The decoder is causal in time and holds ``num_frames`` frames of
        positions, so a frame within the first window decodes the same
        whether it is taken from one pass over that window or from a pass
        of its own: its position is the same and it attends only to what
        precedes it.  Those frames are therefore decoded together, and
        only frames past the window need one pass each, at the end of the
        most recent ``num_frames`` codes.
        """
        window = self.config.num_frames
        total = int(tokens.shape[1])
        if total <= window:
            return self.genie.detokenize(tokens)[:, start:]
        frames: list[Tensor] = []
        if start < window:
            frames.append(self.genie.detokenize(tokens[:, :window])[:, start:])
        frames.extend(
            self.genie.detokenize(tokens[:, j - window + 1 : j + 1])[:, -1:]
            for j in range(max(start, window), total)
        )
        return lucid.cat(frames, dim=1) if len(frames) > 1 else frames[0]

    @override
    def forward(self, prompt: Tensor, actions: Tensor) -> GenieRolloutOutput:
        """Play the environment from a prompt, one latent action per frame.

        Parameters
        ----------
        prompt : Tensor
            Prompt frames ``(B, P, C, H, W)`` in ``[0, 1]``, with
            ``1 <= P <= num_frames``.  The actions between prompt frames
            are inferred by the latent action model.
        actions : Tensor
            Integer latent actions ``(B, S)`` in ``[0, num_latent_actions)``,
            one per frame to generate.

        Returns
        -------
        GenieRolloutOutput
            ``S`` generated frames and the codes of the whole sequence.
            Computed without gradient, and with gradients off nothing is
            counted either: a rollout leaves the codebooks and their usage
            exactly as it found them, in training mode as in eval.
        """
        config = self.config
        _check_video(config, prompt, "prompt", 1)
        if actions.ndim != 2 or int(actions.shape[0]) != int(prompt.shape[0]):
            raise ValueError(
                f"actions must be (B, S) with the prompt's batch, got "
                f"{tuple(actions.shape)} for a prompt of {tuple(prompt.shape)}"
            )
        steps = int(actions.shape[1])
        if steps < 1:
            raise ValueError("actions must name at least one frame to generate")
        low, high = int(actions.min().item()), int(actions.max().item())
        if low < 0 or high >= config.num_latent_actions:
            raise ValueError(
                f"latent actions index a codebook of {config.num_latent_actions}, "
                f"got values in [{low}, {high}]"
            )

        genie = self.genie
        with lucid.no_grad():
            tokens = genie.tokenizer.quantize(prompt).indices
            chosen = genie.latent_action_model.quantizer.lookup(actions)
            if int(prompt.shape[1]) > 1:
                inferred = genie.latent_action_model.quantize(prompt).quantized
                chosen = lucid.cat([inferred, chosen], dim=1)
            known = int(tokens.shape[1])
            for step in range(steps):
                frame = self.predict_frame(tokens, chosen[:, : known + step])
                tokens = lucid.cat([tokens, frame.unsqueeze(1)], dim=1)
            frames = self._decode(tokens, known)
        return GenieRolloutOutput(frames=frames, tokens=tokens)
