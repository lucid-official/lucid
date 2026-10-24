r"""Terminal Velocity Matching — the network, the objective and the sampler.

**The network is DiT-XL/2 with its Lipschitz holes closed.**  The paper
keeps DiT's shape — patchify, adaLN-Zero blocks over a summed
conditioning vector, unpatchify — and changes only what Theorem 1 needs:
every LayerNorm becomes a parameter-free RMSNorm, queries and keys are
RMS-normalised, every adaLN shift, scale and gate is RMS-normalised
before it touches the residual stream, and linear layers start at unit
spectral norm.  Without the modulation norm the activations of the
conditioning path jump by orders of magnitude partway through training
(Figure 4).

**Four things condition the field.**  The start time :math:`t`, the
interval :math:`t - s` (not :math:`s` itself — the convention inherited
from IMM and MeanFlow), the reciprocal guidance weight :math:`1/w`, and
the class.  Each scalar is a sinusoid through its own two-layer MLP; the
four are summed.

**The objective back-propagates through its own derivative.**  The
terminal velocity :math:`\frac{d}{ds} f_\theta = F_\theta + (s - t)
\partial_s F_\theta` is one Jacobian-vector product along :math:`s`, and
unlike sCT and MeanFlow, whose JVP is a stop-gradiented target, TVM's
JVP *is* the prediction — the gradient goes through it.  What sits under
a stop-gradient is the other side: the velocity at the landing point,
read off an exponential moving average of the weights.
"""

from dataclasses import dataclass
from typing import ClassVar, final, override

import lucid
import lucid.func
import lucid.linalg
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.special
from lucid import Tensor
from lucid._C import engine as _C_engine
from lucid._engine_scope import untraced
from lucid.models._base import PretrainedModel
from lucid.models._output import DiffusionModelOutput, GenerationOutput, ModelOutput
from lucid.models._tasks import ImageGenerationModel
from lucid.models._utils._generative import resolve_generation_device, start_noise
from lucid.models.generative._common._config import patch_grid_side
from lucid.models.generative._common._transformers import (
    check_latent_size,
    sincos_position_embedding,
    timestep_embedding,
    unpatchify,
)
from lucid.models.generative.tvm._config import TVMConfig
from lucid.nn.utils import skip_init

__all__ = [
    "TVMModel",
    "TVMForImageGeneration",
    "TVMOutput",
]

# Every norm in the released network uses this epsilon — the token norms,
# the query/key norms and the modulation norm alike.
_NORM_EPS = 1e-6


def _spectral_(weight: Tensor) -> None:
    r"""Initialise ``weight`` to a Gaussian matrix of unit spectral norm.

    The Lipschitz initialisation of LipsFormer that the paper adopts: a
    Xavier-normal draw divided by its largest singular value.  Xavier's
    variance cancels in the division, so a unit Gaussian is drawn
    directly.  A convolution kernel is treated as the matrix of its
    output channels against everything else, which is how the patch
    embedding is a linear layer over flattened patches.
    """
    nn.init.normal_(weight)
    sigma = lucid.linalg.svdvals(weight.reshape(int(weight.shape[0]), -1))[0]
    weight.div_(sigma)


def _zero_projection(head: nn.Module) -> None:
    """Zero the linear layer of an adaLN head — the "Zero" of adaLN-Zero."""
    for layer in head.modules():
        if isinstance(layer, nn.Linear):
            layer.weight.zero_()


def _modulation(
    params: Tensor, count: int, width: int, normalize: bool
) -> list[Tensor]:
    r"""Split an adaLN projection into ``count`` vectors of ``width``.

    With ``normalize`` each vector is RMS-normalised on its own — Eq. 16
    and Appendix B.2.  An adaLN layer's Lipschitz constant scales with its
    modulation's magnitude, which nothing else bounds; dividing every
    shift, scale and gate by its own RMS fixes that magnitude at
    :math:`\sqrt{d}` while leaving the direction to the conditioning.

    Returns each vector as ``(B, 1, width)``, ready to broadcast over the
    token axis.
    """
    params = params.reshape(-1, count, width)
    if normalize:
        params = F.rms_norm(params, (width,), eps=_NORM_EPS)
    return [params[:, index : index + 1, :] for index in range(count)]


class _ScalarEmbedder(nn.Module):
    """One scalar condition's embedding: sinusoid, then a two-layer MLP.

    Parameters
    ----------
    hidden_size : int
        Width of the conditioning vector.
    freq_size : int, default=256
        Width of the sinusoid.
    """

    def __init__(self, hidden_size: int, freq_size: int = 256) -> None:
        """Initialise the embedder. See the class docstring for parameters."""
        super().__init__()
        self.freq_size = freq_size
        self.fc1 = nn.Linear(freq_size, hidden_size)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    @override
    def forward(self, value: Tensor) -> Tensor:
        """Embed ``(B,)`` scalars into ``(B, hidden_size)``.

        The scalars enter the sinusoid as they are — times in
        :math:`[0, 1]` and :math:`1/w` alike, with no rescaling to a
        thousand-step grid.
        """
        emb = timestep_embedding(value, self.freq_size, dtype=self.fc1.weight.dtype)
        return self.fc2(self.act(self.fc1(emb)))


class _Attention(nn.Module):
    """Multi-head self-attention with RMS-normalised queries and keys.

    Parameters
    ----------
    hidden_size : int
        Residual stream width.
    num_heads : int
        Attention heads.
    qk_norm : bool
        Whether to RMS-normalise each head's queries and keys, with a
        learnable per-channel scale — the :math:`L_2` QK-norm of
        LipsFormer, which bounds the attention logits.
    """

    def __init__(self, hidden_size: int, num_heads: int, qk_norm: bool) -> None:
        """Initialise the layer. See the class docstring for parameters."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.q_norm: nn.Module = (
            nn.RMSNorm(self.head_dim, eps=_NORM_EPS) if qk_norm else nn.Identity()
        )
        self.k_norm: nn.Module = (
            nn.RMSNorm(self.head_dim, eps=_NORM_EPS) if qk_norm else nn.Identity()
        )
        self.proj = nn.Linear(hidden_size, hidden_size)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Attend over ``(B, N, D)`` tokens."""
        batch, tokens, width = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query = self.q_norm(qkv[0])
        key = self.k_norm(qkv[1])
        attended = F.scaled_dot_product_attention(query, key, qkv[2])
        return self.proj(attended.permute(0, 2, 1, 3).reshape(batch, tokens, width))


class _FeedForward(nn.Module):
    """The block's MLP — tanh-approximated GELU, as DiT ships it.

    Parameters
    ----------
    hidden_size : int
        Residual stream width.
    inner : int
        Expanded width.
    """

    def __init__(self, hidden_size: int, inner: int) -> None:
        """Initialise the MLP. See the class docstring for parameters."""
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, inner)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(inner, hidden_size)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Expand, activate, project back."""
        return self.fc2(self.act(self.fc1(x)))


class _TVMBlock(nn.Module):
    r"""A DiT block with every LayerNorm and modulation RMS-normalised.

    Parameters
    ----------
    hidden_size : int
        Residual stream width.
    num_heads : int
        Attention heads.
    mlp_ratio : float
        Feed-forward expansion.
    qk_norm : bool
        See :attr:`TVMConfig.qk_norm`.
    modulation_norm : bool
        See :attr:`TVMConfig.modulation_norm`.

    Notes
    -----
    Appendix B.2 writes the block as

    .. math::

        x \leftarrow x + c_1^- \cdot \mathrm{ATTN}\bigl(
            \mathrm{RMSNorm}^-(x)\,a_1^- + b_1^-\bigr), \qquad
        x \leftarrow x + c_2^- \cdot \mathrm{MLP}\bigl(
            \mathrm{RMSNorm}^-(x)\,a_2^- + b_2^-\bigr),

    with every modulation vector RMS-normalised.  The released network
    keeps DiT's ``1 +`` on the scale, :math:`\mathrm{RMSNorm}^-(x)
    (1 + a^-) + b^-`, and so does this block — it is what the checkpoints
    were trained with.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        qk_norm: bool,
        modulation_norm: bool,
    ) -> None:
        """Initialise the block. See the class docstring for parameters."""
        super().__init__()
        self.modulation_norm = modulation_norm
        self.norm1 = nn.RMSNorm(hidden_size, eps=_NORM_EPS, elementwise_affine=False)
        self.attn = _Attention(hidden_size, num_heads, qk_norm)
        self.norm2 = nn.RMSNorm(hidden_size, eps=_NORM_EPS, elementwise_affine=False)
        self.mlp = _FeedForward(hidden_size, int(hidden_size * mlp_ratio))
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    @override
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Attention and MLP, each modulated and gated by ``cond``."""
        width = int(x.shape[2])
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = _modulation(
            self.ada_ln(cond), 6, width, self.modulation_norm
        )
        x = x + gate_a * self.attn(self.norm1(x) * (1.0 + scale_a) + shift_a)
        return x + gate_m * self.mlp(self.norm2(x) * (1.0 + scale_m) + shift_m)


class _FinalLayer(nn.Module):
    """Modulated RMSNorm and the projection back to patch space.

    Parameters
    ----------
    hidden_size : int
        Residual stream width.
    patch_size : int
        Side of a patch.
    out_channels : int
        Channels per latent pixel.
    modulation_norm : bool
        See :attr:`TVMConfig.modulation_norm` — the final shift and scale
        are normalised the same way the blocks' are.
    """

    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        modulation_norm: bool,
    ) -> None:
        """Initialise the head. See the class docstring for parameters."""
        super().__init__()
        self.modulation_norm = modulation_norm
        self.norm = nn.RMSNorm(hidden_size, eps=_NORM_EPS, elementwise_affine=False)
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.proj = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    @override
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Project ``(B, N, D)`` tokens to ``(B, N, patch^2 * out_channels)``."""
        shift, scale = _modulation(
            self.ada_ln(cond), 2, int(x.shape[2]), self.modulation_norm
        )
        return self.proj(self.norm(x) * (1.0 + scale) + shift)


@dataclass(slots=True)
class TVMOutput(ModelOutput):
    r"""What a TVM training step reports.

    Attributes
    ----------
    loss : Tensor
        The objective of Eq. 15, a scalar — the two terms below, summed
        per sample and averaged over the batch.
    terminal_loss : Tensor
        The batch mean of the terminal-velocity term, :math:`\lVert
        \frac{d}{ds} f_\theta - u_{\theta^*}(\hat x_s, s)\rVert^2 / w^2`.
        Zero for samples drawn with :math:`s = t`.
    flow_matching_loss : Tensor
        The batch mean of the guided Flow Matching term.  Reported apart
        because the two move differently: Flow Matching settles early
        while the terminal term keeps falling, and a terminal term that
        stalls while Flow Matching is fine is the first sign of a bad
        target EMA.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.tvm._model import TVMOutput
    >>> out = TVMOutput(
    ...     loss=lucid.zeros(()),
    ...     terminal_loss=lucid.zeros(()),
    ...     flow_matching_loss=lucid.zeros(()),
    ... )
    >>> out.loss.shape
    ()
    """

    loss: Tensor
    terminal_loss: Tensor
    flow_matching_loss: Tensor


class TVMModel(PretrainedModel):
    r"""The two-time velocity network :math:`w\,F_\theta(x_t, t, s, c, w)`.

    Parameters
    ----------
    config : TVMConfig
        The variant to build.

    Attributes
    ----------
    patch_embed : lucid.nn.Conv2d
        Patchifying projection — stride equals kernel.
    pos_embed : Tensor
        Frozen two-dimensional sine-cosine table, added to the tokens.
    time_embed, interval_embed, guidance_embed : _ScalarEmbedder
        Embeddings of :math:`t`, :math:`t - s` and :math:`1/w`.
    label_embed : lucid.nn.Embedding
        Class embedding, with the null label as its last row.
    blocks : lucid.nn.ModuleList
        The semi-Lipschitz DiT blocks.
    final : _FinalLayer
        Modulated norm and projection back to patch space.

    Notes
    -----
    Reference: Zhou, Parger, Haque, and Song, *"Terminal Velocity
    Matching"*, ICLR 2026 (arXiv:2511.19797), Sections 3–4 and
    Appendix F.1.

    The forward returns the **average velocity over** :math:`[s, t]` —
    the displacement divided by its length — so that one step of the
    sampler is :math:`x_s = x_t + (s - t)\cdot\mathrm{sample}`.  Under the
    scaled parameterisation that is :math:`w\,F_\theta`, the network's
    output times the guidance weight; otherwise it is :math:`F_\theta`
    itself.  At :math:`s = t` it is the instantaneous velocity
    :math:`u_\theta(x_t, t)`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.tvm import TVMConfig, TVMModel
    >>> config = TVMConfig(sample_size=8, patch_size=2, hidden_size=32,
    ...                    depth=2, num_heads=4, num_classes=10)
    >>> model = TVMModel(config).eval()
    >>> x = lucid.randn((1, 4, 8, 8))
    >>> t, s = lucid.tensor([1.0]), lucid.tensor([0.0])
    >>> model(x, t, s, lucid.tensor([3])).sample.shape
    (1, 4, 8, 8)
    """

    config_class: ClassVar[type[TVMConfig]] = TVMConfig
    base_model_prefix = "tvm"

    def __init__(self, config: TVMConfig) -> None:
        """Initialise the network. See the class docstring for parameters."""
        super().__init__(config)
        self.config: TVMConfig = config
        side = patch_grid_side(config.sample_size)
        self.grid = side // config.patch_size
        width = config.hidden_size

        self.patch_embed = nn.Conv2d(
            config.in_channels,
            width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        # Derived from the config and frozen in the reference too, so it is
        # rebuilt here rather than carried in a checkpoint.
        self.register_buffer(
            "pos_embed",
            sincos_position_embedding(width, self.grid),
            persistent=False,
        )

        freq = config.frequency_embedding_size
        self.time_embed = _ScalarEmbedder(width, freq)
        self.interval_embed = _ScalarEmbedder(width, freq)
        self.guidance_embed = _ScalarEmbedder(width, freq)
        self.label_embed = nn.Embedding(config.num_classes + 1, width)

        self.blocks = nn.ModuleList(
            [
                _TVMBlock(
                    width,
                    config.num_heads,
                    config.mlp_ratio,
                    config.qk_norm,
                    config.modulation_norm,
                )
                for _ in range(config.depth)
            ]
        )
        self.final = _FinalLayer(
            width, config.patch_size, config.out_channels, config.modulation_norm
        )
        self._init_weights()

    def _init_weights(self) -> None:
        r"""Lipschitz initialisation, then adaLN-Zero (Table 6).

        Every linear layer that is not zeroed starts at unit spectral
        norm.  The scalar embedders follow :attr:`TVMConfig.
        time_embedding_init` instead — DiT's :math:`\mathcal{N}(0, 0.02)`
        or the spectral one — and the label table is always DiT's
        :math:`\mathcal{N}(0, 0.02)`.  The modulation projections and the
        decoder start at zero, so every block begins as the identity and
        the untrained field is zero.  Biases start at zero throughout.
        """
        spectral_time = self.config.time_embedding_init == "spectral"
        with lucid.no_grad():
            _spectral_(self.patch_embed.weight)
            for block in self.blocks:
                for linear in (
                    block.attn.qkv,
                    block.attn.proj,
                    block.mlp.fc1,
                    block.mlp.fc2,
                ):
                    _spectral_(linear.weight)
                _zero_projection(block.ada_ln)
            for embedder in (self.time_embed, self.interval_embed, self.guidance_embed):
                for linear in (embedder.fc1, embedder.fc2):
                    if spectral_time:
                        _spectral_(linear.weight)
                    else:
                        nn.init.normal_(linear.weight, std=0.02)
            nn.init.normal_(self.label_embed.weight, std=0.02)
            _zero_projection(self.final.ada_ln)
            self.final.proj.weight.zero_()
            for module in self.modules():
                bias = getattr(module, "bias", None)
                if isinstance(module, (nn.Linear, nn.Conv2d)) and bias is not None:
                    bias.zero_()

    def resolve_guidance(
        self, guidance_scale: Tensor | float | None, labels: Tensor | None
    ) -> float | Tensor:
        r"""The guidance weight a call runs at.

        Parameters
        ----------
        guidance_scale : Tensor, float or None
            What the caller asked for.  ``None`` picks the weight the
            model was trained at for a labelled call and :math:`w = 1`
            for an unlabelled one — the only weight the null label is
            ever trained with.
        labels : Tensor or None
            The call's labels.

        Returns
        -------
        float or Tensor
            ``guidance_scale``, or the default it resolves to.
        """
        if guidance_scale is not None:
            return guidance_scale
        return 1.0 if labels is None else self.config.guidance_scale

    @override
    def forward(
        self,
        x: Tensor,
        t: Tensor,
        s: Tensor,
        labels: Tensor | None = None,
        guidance_scale: Tensor | float | None = None,
    ) -> DiffusionModelOutput:
        r"""Evaluate the average velocity from time ``t`` to time ``s``.

        Parameters
        ----------
        x : Tensor
            ``(B, in_channels, H, W)`` — the point on the path at ``t``.
        t : Tensor
            ``(B,)`` start times.
        s : Tensor
            ``(B,)`` end times, ``s <= t``.  ``s = t`` gives the
            instantaneous velocity.
        labels : Tensor or None, optional
            ``(B,)`` class indices.  ``None`` uses the null label, index
            ``num_classes``.
        guidance_scale : Tensor, float or None, optional
            The guidance weight :math:`w`, per sample or shared.  The
            network is conditioned on :math:`1/w`.  ``None`` resolves as
            :meth:`resolve_guidance` describes.

        Returns
        -------
        DiffusionModelOutput
            ``sample`` of shape ``(B, out_channels, H, W)`` — the
            average velocity, :math:`w\,F_\theta` under the scaled
            parameterisation and :math:`F_\theta` otherwise.
        """
        check_latent_size(
            int(x.shape[-2]),
            int(x.shape[-1]),
            self.grid,
            self.config.patch_size,
            "TVM",
        )
        batch = int(x.shape[0])
        device = x.device.type
        guidance = self.resolve_guidance(guidance_scale, labels)
        if not isinstance(guidance, Tensor):
            guidance = lucid.full(
                (batch,), float(guidance), device=device, dtype=x.dtype
            )
        if labels is None:
            labels = lucid.full(
                (batch,),
                float(self.config.num_classes),
                dtype=lucid.int64,
                device=device,
            )

        tokens = self.patch_embed(x)
        tokens = tokens.reshape(batch, int(tokens.shape[1]), -1).permute(0, 2, 1)
        tokens = tokens + self.pos_embed

        cond = (
            self.time_embed(t)
            + self.interval_embed(t - s)
            + self.guidance_embed(1.0 / guidance)
            + self.label_embed(labels)
        )
        for block in self.blocks:
            tokens = block(tokens, cond)
        out = unpatchify(
            self.final(tokens, cond),
            self.grid,
            self.config.patch_size,
            self.config.out_channels,
        )
        if self.config.scaled_parameterization:
            out = out * guidance.reshape(-1, 1, 1, 1)
        return DiffusionModelOutput(sample=out)


@final
class _AveragedTVM(TVMModel):
    """The network's structure for one of Algorithm 1's averages.

    An average holds the live network's weights from the moment it exists,
    so the Lipschitz initialisation — an SVD per linear layer — would be
    computed only to be overwritten.  Built under ``skip_init`` with that
    step left out, it costs the allocation and draws no random numbers.
    """

    @override
    def _init_weights(self) -> None:
        """Leave the allocation as it is: the live weights are copied in next."""


class TVMForImageGeneration(ImageGenerationModel):
    r"""TVM's objective and its one- and few-step sampler.

    Parameters
    ----------
    config : TVMConfig
        The variant to build.

    Attributes
    ----------
    tvm : TVMModel
        The network being trained.

    Notes
    -----
    Reference: Zhou, Parger, Haque, and Song, *"Terminal Velocity
    Matching"*, ICLR 2026 (arXiv:2511.19797).  Training is Algorithm 1
    with the objective of Eq. 15; sampling is Figure 3.

    **Call** :meth:`update_ema` **once per optimiser step.**  Algorithm 1
    keeps two exponential moving averages of the weights and updates both
    after every step: :math:`\theta^*` at
    :attr:`TVMConfig.target_ema_decay`, which the terminal target and the
    unconditional half of the guided velocity are read off, and
    :math:`\theta^{**}` at :attr:`TVMConfig.eval_ema_decay`, the model
    that is evaluated and that the released checkpoints are.  Only the
    caller knows when an optimiser step has happened.

    Both copies are created from the live weights the first time either
    is needed — the first training step reads :math:`\theta^*`, so that
    is :math:`\theta^* \leftarrow \theta,\ \theta^{**} \leftarrow
    \theta` at initialisation, as Algorithm 1 has it — and both are kept
    outside the module tree: not parameters, not in :meth:`state_dict`,
    not counted.  They are training state, like an optimiser's moments;
    :attr:`target_network` and :attr:`ema_network` expose them for
    checkpointing.  Loading weights into the network — through
    :meth:`load_state_dict` here or on :attr:`tvm` — restarts both from the
    loaded weights, so to resume a run load the network first and the
    averages' own checkpoints after it.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.tvm import (
    ...     TVMConfig, TVMForImageGeneration)
    >>> config = TVMConfig(sample_size=8, patch_size=2, hidden_size=32,
    ...                    depth=2, num_heads=4, num_classes=10)
    >>> model = TVMForImageGeneration(config).eval()
    >>> model.generate(2, labels=lucid.tensor([1, 2])).samples.shape
    (2, 4, 8, 8)
    """

    config_class: ClassVar[type[TVMConfig]] = TVMConfig
    base_model_prefix = "tvm"

    def __init__(self, config: TVMConfig) -> None:
        """Initialise the model. See the class docstring for parameters."""
        super().__init__(config)
        self.config: TVMConfig = config
        self.tvm = TVMModel(config)
        # A dict rather than attributes: a module assigned directly would be
        # registered as a child, adding the parameter count and the
        # checkpoint of a network no optimiser may touch — twice over.
        self._averages: dict[str, TVMModel] = {}
        # On the network rather than here: ``model.tvm.load_state_dict``
        # never passes through this module, and ``model.load_state_dict``
        # reaches the network's hooks too.
        self.tvm.register_load_state_dict_post_hook(self._restart_averages)

    # ── the two EMAs of Algorithm 1 ──────────────────────────────────────

    def _restart_averages(self, network: nn.Module, incompatible: object) -> None:
        """Drop both averages once new weights are loaded into the network.

        Loading weights is initialising them, so the averages restart from
        them the next time either is read, as at Algorithm 1's start.  Kept,
        they averaged toward the loaded weights from whatever was there
        before — a fine-tune from a checkpoint read its first targets off
        the random initialisation.
        """
        del network, incompatible  # the hook's signature; nothing to inspect
        self._averages.clear()

    def _decays(self) -> dict[str, float]:
        """The averages this configuration keeps, by name, with their rates."""
        decays: dict[str, float] = {}
        if self.config.target_ema_decay > 0.0:
            decays["target"] = self.config.target_ema_decay
        if self.config.eval_ema_decay is not None:
            decays["eval"] = self.config.eval_ema_decay
        return decays

    def _average(self, name: str) -> TVMModel:
        r"""One average, every average created together on first use.

        Creating them together is what makes both start from the same
        weights — the initial ones, since the first training step is what
        first asks for :math:`\theta^*`.  Each is frozen, kept on the live
        network's device, and in evaluation mode.

        Built and moved outside any active trace: the first forward to read
        :math:`\theta^*` is what builds them, so a trace taken then recorded
        their construction too and differed from one taken a call later.
        """
        # The averages follow the live network's device *and* dtype: built
        # lazily in float32, they ran a half model's targets in float32.
        live = next(iter(self.tvm.parameters()))
        with untraced():
            if not self._averages:
                weights = self.tvm.state_dict()
                for key in self._decays():
                    copy = skip_init(_AveragedTVM, self.config)
                    assert isinstance(copy, _AveragedTVM), "skip_init builds it"
                    copy.load_state_dict(weights)
                    for param in copy.parameters():
                        param.requires_grad_(False)
                    self._averages[key] = copy.eval()
            average = self._averages[name]
            slow = next(iter(average.parameters()))
            if slow.device != live.device or slow.dtype != live.dtype:
                average.to(device=live.device, dtype=live.dtype)
        return average

    @property
    def target_network(self) -> TVMModel:
        r"""The EMA network :math:`\theta^*` the targets are read from.

        With :attr:`TVMConfig.target_ema_decay` at zero this is the live
        network itself, used under a stop-gradient — Table 4b's
        :math:`\gamma = 0`.

        Returns
        -------
        TVMModel
            The target network.
        """
        if "target" not in self._decays():
            return self.tvm
        return self._average("target")

    @property
    def ema_network(self) -> TVMModel:
        r"""The evaluation EMA :math:`\theta^{**}` — what Algorithm 1 outputs.

        Returns
        -------
        TVMModel
            The averaged network.

        Raises
        ------
        ValueError
            If :attr:`TVMConfig.eval_ema_decay` is ``None``, so no
            evaluation average is kept.
        """
        if "eval" not in self._decays():
            raise ValueError(
                "eval_ema_decay is None, so no evaluation EMA is kept; "
                "generate from the live network instead"
            )
        return self._average("eval")

    def update_ema(self) -> None:
        r"""Advance both of Algorithm 1's averages by one step.

        :math:`\theta^* \leftarrow \gamma^*\theta^* + (1 - \gamma^*)\theta`
        and :math:`\theta^{**} \leftarrow \gamma^{**}\theta^{**} +
        (1 - \gamma^{**})\theta`, over parameters only — the released
        training loop averages nothing else, and this network has no
        persistent buffers.  Call it after each optimiser step.

        The averages are evaluated once they are written.  They are
        outside :meth:`parameters`, so neither the optimiser's flush nor a
        loop's reaches them on Metal.
        """
        decays = self._decays()
        if not decays:
            return
        # Paired by name, not by position: the averages are separate
        # networks, and a name is the one key both sides are sure to share —
        # a position holds only while no edit to either tree has moved it,
        # and deleting a child or a parameter and assigning it back moves
        # it to the end of its registry.
        live = dict(self.tvm.named_parameters())
        written: list[_C_engine.TensorImpl] = []
        with lucid.no_grad():
            for key, decay in decays.items():
                average = self._average(key)
                for name, slow in average.named_parameters():
                    # `slow[:] = ...`, not `slow.data[:] = ...`: the latter
                    # writes to a copy and silently does nothing.
                    slow[:] = decay * slow + (1.0 - decay) * live[name]
                    written.append(slow._impl)
        # The writes are lazy on Metal, and θ** is not read until sampling:
        # unevaluated, each step would chain onto the last and keep every
        # past copy of the live weights.  One batched evaluation, as the
        # optimiser flushes its parameters: one per parameter cost a Metal
        # sync per tensor, several times slower.
        _C_engine.eval_tensors(written)

    # ── sampling the training variables ──────────────────────────────────

    def _logit_normal(self, mean: float, std: float, batch: int, device: str) -> Tensor:
        """``(B,)`` draws of ``sigmoid(N(mean, std))``."""
        return F.sigmoid(lucid.randn((batch,), device=device) * std + mean)

    def _truncated_logit_normal(
        self, mean: float, std: float, upper: Tensor, device: str
    ) -> Tensor:
        r"""Logit-normal draws conditioned on lying below ``upper``.

        Inverse-CDF sampling: a uniform draw scaled into
        :math:`[0, \Phi((\mathrm{logit}(\mathrm{upper}) - \mu)/\sigma)]`,
        mapped back through :math:`\Phi^{-1}` and the logistic.  Exact, so
        no draw is rejected and none lands above the bound.
        """
        logit_upper = lucid.log(upper / (1.0 - upper))
        mass = lucid.special.ndtr((logit_upper - mean) / std)
        uniform = lucid.rand(tuple(upper.shape), device=device) * mass
        return F.sigmoid(mean + std * lucid.special.ndtri(uniform))

    def _sample_times(self, batch: int, device: str) -> tuple[Tensor, Tensor, Tensor]:
        r"""Draw ``(t, s)`` for the terminal term and the Flow Matching time.

        Returns
        -------
        tuple of Tensor
            ``t``, ``s`` with :math:`s \le t`, and the time the Flow
            Matching term is evaluated at — an independent logit-normal
            draw under ``gap*``, ``s`` otherwise.
        """
        config = self.config
        if config.time_sampler == "gap":
            gap = self._logit_normal(config.gap_mean, config.gap_std, batch, device)
            s = self._truncated_logit_normal(
                config.s_mean, config.s_std, 1.0 - gap, device
            )
            t = lucid.clamp(s + gap, max=1.0)
        elif config.time_sampler == "trunc":
            t = self._logit_normal(config.t_mean, config.t_std, batch, device)
            s = self._truncated_logit_normal(config.s_mean, config.s_std, t, device)
        else:
            t = self._logit_normal(config.t_mean, config.t_std, batch, device)
            s = lucid.minimum(
                self._logit_normal(config.s_mean, config.s_std, batch, device), t
            )

        if config.ratio_t_equals_s > 0.0:
            same = lucid.rand((batch,), device=device) < config.ratio_t_equals_s
            s = lucid.where(same, t, s)

        if config.separate_flow_matching_time:
            s_fm = self._logit_normal(config.s_mean, config.s_std, batch, device)
        else:
            s_fm = s
        return t, s, s_fm

    def _sample_guidance(self, batch: int, device: str) -> Tensor:
        r"""``(B,)`` guidance weights — constant, or :math:`1/U(1/w_{\max},
        1/w_{\min})` under :attr:`TVMConfig.guidance_range`."""
        config = self.config
        if config.guidance_range is None:
            return lucid.full((batch,), config.guidance_scale, device=device)
        low, high = config.guidance_range
        beta = 1.0 / high + lucid.rand((batch,), device=device) * (
            1.0 / low - 1.0 / high
        )
        return 1.0 / beta

    # ── objective ────────────────────────────────────────────────────────

    def _terminal_velocity(
        self, x_t: Tensor, t: Tensor, s: Tensor, labels: Tensor, guidance: Tensor
    ) -> tuple[Tensor, Tensor]:
        r"""The model's average velocity and its terminal velocity at ``s``.

        Returns
        -------
        tuple of Tensor
            :math:`w F_\theta(x_t, t, s)` and :math:`\frac{d}{ds}
            f_\theta = w\bigl(F_\theta + (s - t)\,\partial_s F_\theta\bigr)`.

        Notes
        -----
        The Jacobian-vector product is along :math:`s` alone — a unit
        tangent that does not depend on :math:`x_t` or :math:`t`, which
        is what the paper credits for TVM's stability under random
        guidance (Section 7.2).  ``s`` enters as a leaf that requires
        grad so the product is taken in one evaluation with its graph
        kept: the gradient flows through :math:`\partial_s F_\theta`
        unless :attr:`TVMConfig.detach_jvp` stops it there.
        """

        def field(s_: Tensor) -> Tensor:
            return self.tvm(x_t, t, s_, labels, guidance).sample

        tangent = lucid.ones_like(s)
        if self.config.detach_jvp:
            velocity = field(s)
            with lucid.no_grad():
                _, along_s = lucid.func.jvp(field, (s,), (tangent,))
        else:
            primal, along_s = lucid.func.jvp(
                field, (s.detach().requires_grad_(),), (tangent,)
            )
            assert isinstance(primal, Tensor), "the field returns one tensor"
            velocity = primal
        assert isinstance(along_s, Tensor), "the field returns one tensor"
        interval = (s - t).reshape(-1, 1, 1, 1)
        return velocity, velocity + interval * along_s

    @override
    def forward(self, images: Tensor, labels: Tensor | None = None) -> TVMOutput:
        r"""One training step of the TVM objective (Eq. 15).

        Parameters
        ----------
        images : Tensor
            ``(B, C, H, W)`` — the data, here the VAE latent of it.
        labels : Tensor or None, optional
            ``(B,)`` class indices.  Each is replaced by the null label
            with probability :attr:`TVMConfig.class_dropout`, and that
            sample trains at :math:`w = 1`.  ``None`` trains the
            unconditional field throughout.

        Returns
        -------
        TVMOutput
            The loss and its two terms.

        Notes
        -----
        Per sample, with :math:`\beta = 1/w`,

        .. math::

            \beta^2\Bigl\lVert \tfrac{d}{ds} f_\theta(x_t, t, s)
              - u_{\theta^*}\bigl(\mathrm{sg}[x_t + f_\theta(x_t, t, s)],
              s\bigr)\Bigr\rVert^2 \mathbb{1}_{t \ne s}
            + \beta^2\bigl\lVert u_\theta(x_{s'}, s')
              - \bigl[w\,v + (1 - w)\,u_{\theta^*}(x_{s'}, s', \varnothing,
              1)\bigr]\bigr\rVert^2 .

        Squared norms are averaged over elements, as the released
        training code computes them; the paper's sum differs by the
        constant :math:`CHW`.
        """
        config = self.config
        batch = int(images.shape[0])
        device = images.device.type

        # Drawn in float32 (``ndtri`` wants it), then in the images' dtype.
        t, s, s_fm = (v.to(images.dtype) for v in self._sample_times(batch, device))
        guidance = self._sample_guidance(batch, device).to(images.dtype)
        null = lucid.full(
            (batch,), float(config.num_classes), dtype=lucid.int64, device=device
        )
        if labels is None:
            labels = null
            guidance = lucid.ones_like(guidance)
        elif config.class_dropout > 0.0:
            drop = lucid.rand((batch,), device=device) < config.class_dropout
            labels = lucid.where(drop, null.to(labels.dtype), labels)
            guidance = lucid.where(drop, lucid.ones_like(guidance), guidance)

        noise = lucid.randn(images.shape, dtype=images.dtype, device=device)
        t_b = t.reshape(-1, 1, 1, 1)
        s_b = s_fm.reshape(-1, 1, 1, 1)
        x_t = (1.0 - t_b) * images + t_b * noise
        x_fm = (1.0 - s_b) * images + s_b * noise
        w_b = guidance.reshape(-1, 1, 1, 1)
        beta_b = 1.0 / w_b

        velocity, terminal = self._terminal_velocity(x_t, t, s, labels, guidance)
        landing = (x_t + (s - t).reshape(-1, 1, 1, 1) * velocity).detach()

        # Both targets come from θ* in one batch: the velocity where the jump
        # lands (class-conditional, at the sample's w) and the unconditional
        # velocity the guided Flow Matching target mixes in (w = 1).
        with lucid.no_grad():
            reads = self.target_network(
                lucid.cat([landing, x_fm], dim=0),
                lucid.cat([s, s_fm], dim=0),
                lucid.cat([s, s_fm], dim=0),
                lucid.cat([labels, null.to(labels.dtype)], dim=0),
                lucid.cat([guidance, lucid.ones_like(guidance)], dim=0),
            ).sample
        terminal_target = reads[:batch]
        unconditional = reads[batch:]

        velocity_fm = self.tvm(x_fm, s_fm, s_fm, labels, guidance).sample
        guided = w_b * (noise - images) + (1.0 - w_b) * unconditional

        # The released loss zeroes the terminal term wherever ``isclose(s,
        # t)`` — its default tolerances, not exact equality.
        differs = 1.0 - lucid.isclose(s, t).to(images.dtype)
        terminal_term = (((terminal - terminal_target) * beta_b) ** 2).mean(
            dim=(1, 2, 3)
        ) * differs
        fm_term = (((velocity_fm - guided) * beta_b) ** 2).mean(dim=(1, 2, 3))
        return TVMOutput(
            loss=(terminal_term + fm_term).mean(),
            terminal_loss=terminal_term.mean().detach(),
            flow_matching_loss=fm_term.mean().detach(),
        )

    # ── sampling ─────────────────────────────────────────────────────────

    def generate(
        self,
        n_samples: int = 1,
        *,
        labels: Tensor | None = None,
        steps: int = 1,
        guidance_scale: float | None = None,
        noise: Tensor | None = None,
        device: str | None = None,
        use_ema: bool = False,
    ) -> GenerationOutput:
        r"""Jump from noise to data in ``steps`` evaluations.

        Parameters
        ----------
        n_samples : int, default=1
            How many to draw.
        labels : Tensor or None, optional
            ``(n_samples,)`` class indices.  ``None`` samples the
            unconditional field.
        steps : int, default=1
            Network evaluations.  :math:`[0, 1]` is split evenly and each
            piece is one jump :math:`x_s = x_t + (s - t)\,w F_\theta(x_t,
            t, s)`; the same network serves every step count, from one up
            to the ODE limit, without retraining.
        guidance_scale : float or None, optional
            :math:`w`.  ``None`` uses the weight the model was trained at
            for labelled draws and one for unlabelled ones.
        noise : Tensor or None, optional
            Starting point :math:`x_1`.  Drawn when absent.
        device : str or None, optional
            Where to draw.  Defaults to the model's own device.
        use_ema : bool, default=False
            Sample from the evaluation EMA :math:`\theta^{**}` rather than
            the live weights — what the paper evaluates after training.  A
            loaded checkpoint already *is* that average, so the default
            samples from the live network.

        Returns
        -------
        GenerationOutput
            ``samples`` of shape ``(n_samples, out_channels, H, W)`` — a
            latent, for the Stable Diffusion VAE to decode.

        Raises
        ------
        ValueError
            If ``steps`` is not positive.

        Examples
        --------
        >>> from lucid.models import tvm_imagenet256_gen
        >>> model = tvm_imagenet256_gen(sample_size=8, hidden_size=32, depth=2, num_heads=4)
        >>> model.eval().generate(2, steps=4).samples.shape
        (2, 4, 8, 8)
        """
        if steps < 1:
            raise ValueError(f"steps must be positive, got {steps}")
        device = resolve_generation_device(self, device)
        if noise is None:
            side = patch_grid_side(self.config.sample_size)
            shape = (n_samples, self.config.in_channels, side, side)
            noise = start_noise(self, shape, device)

        network = self.ema_network if use_ema else self.tvm
        x = noise
        with lucid.no_grad():
            for index in range(steps):
                t_value = 1.0 - index / steps
                s_value = 1.0 - (index + 1) / steps
                t = lucid.full((n_samples,), t_value, device=device, dtype=x.dtype)
                s = lucid.full((n_samples,), s_value, device=device, dtype=x.dtype)
                velocity = network(x, t, s, labels, guidance_scale).sample
                x = x + (s_value - t_value) * velocity
        return GenerationOutput(samples=x)

    def from_vae_latent(self, latent: Tensor) -> Tensor:
        r"""Whiten a Stable Diffusion VAE latent the way the model was trained.

        Parameters
        ----------
        latent : Tensor
            ``(B, C, H, W)`` — a sample from the VAE posterior,
            :math:`\ell = \mu_\phi(x) + \sigma_\phi(x)\,\epsilon`, which is
            what the released pipeline encodes images into (not the
            posterior mean).

        Returns
        -------
        Tensor
            :math:`z = (\ell - \mu) / \sigma` per channel — the ``images``
            :meth:`forward` trains on.  :meth:`to_vae_latent` inverts it.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models import tvm_imagenet256_gen
        >>> model = tvm_imagenet256_gen(sample_size=8, hidden_size=32, depth=2, num_heads=4)
        >>> model.from_vae_latent(lucid.zeros((1, 4, 8, 8))).shape
        (1, 4, 8, 8)
        """
        mean, std = self._latent_statistics(latent)
        return (latent - mean) / std

    def _latent_statistics(self, like: Tensor) -> tuple[Tensor, Tensor]:
        """The per-channel whitening ``(mean, std)``, shaped ``(1, C, 1, 1)``.

        In ``like``'s dtype as well as on its device: built at the default
        dtype, they widened a float16 latent to float32.
        """
        device, dtype = like.device.type, like.dtype
        mean = lucid.tensor(list(self.config.latent_mean), dtype=dtype, device=device)
        std = lucid.tensor(list(self.config.latent_std), dtype=dtype, device=device)
        return mean.reshape(1, -1, 1, 1), std.reshape(1, -1, 1, 1)

    def to_vae_latent(self, samples: Tensor) -> Tensor:
        r"""Undo the per-channel whitening the model's latents carry.

        Parameters
        ----------
        samples : Tensor
            ``(B, C, H, W)`` — :meth:`generate`'s output.

        Returns
        -------
        Tensor
            :math:`\ell = z\,\sigma + \mu` with the statistics in
            :attr:`TVMConfig.latent_mean` / :attr:`TVMConfig.latent_std` —
            the raw latent the Stable Diffusion VAE's decoder expects.

        Notes
        -----
        The released models were trained on ``stabilityai/sd-vae-ft-ema``
        posterior samples whitened per channel, not scaled by 0.18215, so
        decoding a sample without this step yields saturated noise.
        :class:`~lucid.models.generative.stable_diffusion.AutoencoderKL`
        is that decoder.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models import tvm_imagenet256_gen
        >>> model = tvm_imagenet256_gen(sample_size=8, hidden_size=32, depth=2, num_heads=4)
        >>> model.to_vae_latent(lucid.zeros((1, 4, 8, 8))).shape
        (1, 4, 8, 8)
        """
        mean, std = self._latent_statistics(samples)
        return samples * std + mean
