r"""MeanFlow — the average velocity, and the identity that makes it trainable.

The network is DiT's, unchanged: patchify a latent, run adaLN-Zero
transformer blocks conditioned on a summed embedding, unpatchify.  The
paper is explicit that it keeps "the DiT architecture blocks untouched"
and that architectural improvements are orthogonal, so what is
MeanFlow-specific lives entirely in the conditioning and the objective.

**Two time variables, not one.**  A diffusion DiT conditions on :math:`t`.
Here the field is :math:`u(z_t, r, t)` and both times are embedded, each
through its own two-layer MLP, and summed.  Which pair is embedded is a
configuration choice — the paper's default is :math:`(t,\; t - r)` and
the interval turns out to carry more of the signal than the endpoint.

**The objective is one JVP away from Flow Matching.**  Everything up to
the target is identical: interpolate, take the conditional velocity
:math:`v_t = \epsilon - x`.  The target then subtracts
:math:`(t-r)\,\frac{d}{dt}u_\theta`, which the chain rule turns into a
Jacobian-vector product along :math:`(v, 0, 1)`.  The tangent is
load-bearing rather than a detail: the paper's destructive ablation puts
the correct one at FID 61.06 and every wrong one between 137 and 329.
"""

from dataclasses import dataclass
from typing import ClassVar, cast, override

import lucid
import lucid.func
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._output import (
    DiffusionModelOutput,
    GenerationOutput,
    ModelOutput,
)
from lucid.models._tasks import ImageGenerationModel
from lucid.models._utils._generative import resolve_generation_device
from lucid.models.generative._common._config import patch_grid_side
from lucid.models.generative._common._transformers import (
    DiTBlock,
    DiTFinalLayer,
    sincos_position_embedding,
    timestep_embedding,
    unpatchify,
)
from lucid.models.generative.mean_flow._config import MeanFlowConfig

__all__ = [
    "MeanFlowModel",
    "MeanFlowForImageGeneration",
    "MeanFlowOutput",
]


class _TimeMLP(nn.Module):
    """One time variable's embedding: sinusoid, then a two-layer MLP.

    Parameters
    ----------
    hidden_size : int
        Width of the conditioning vector the blocks are modulated by.
    """

    def __init__(self, hidden_size: int, freq_size: int = 256) -> None:
        """Initialise the head. See the class docstring for parameters."""
        super().__init__()
        self.hidden_size = hidden_size
        self.freq_size = freq_size
        self.fc1 = nn.Linear(freq_size, hidden_size)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    @override
    def forward(self, t: Tensor) -> Tensor:
        """Embed ``(B,)`` times into ``(B, hidden_size)``."""
        emb = timestep_embedding(t, self.freq_size, dtype=self.fc1.weight.dtype)
        return self.fc2(self.act(self.fc1(emb)))


@dataclass(slots=True)
class MeanFlowOutput(ModelOutput):
    r"""What a MeanFlow training step reports.

    Attributes
    ----------
    loss : Tensor
        The adaptively weighted regression error, a scalar.
    prediction : Tensor
        ``(B, C, H, W)`` — the model's average velocity
        :math:`u_\theta(z_t, r, t)` at the sampled times.
    target : Tensor
        The stop-gradiented target the prediction is regressed onto.
        Reported because the gap between the two *is* the objective, and
        because a target that has drifted from the prediction's scale is
        the first symptom of a mis-specified tangent.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.mean_flow._model import MeanFlowOutput
    >>> out = MeanFlowOutput(
    ...     loss=lucid.zeros(()),
    ...     prediction=lucid.zeros(1, 3, 32, 32),
    ...     target=lucid.zeros(1, 3, 32, 32),
    ... )
    >>> out.prediction.shape, out.target.shape
    ((1, 3, 32, 32), (1, 3, 32, 32))

    The target is returned alongside because it is computed, not given —
    it depends on the model's own output at another time, so a caller
    cannot reconstruct it afterwards.
    """

    loss: Tensor
    prediction: Tensor
    target: Tensor


class MeanFlowModel(PretrainedModel):
    r"""The average-velocity field :math:`u_\theta(z, r, t)`.

    Parameters
    ----------
    config : MeanFlowConfig
        The variant to build.

    Attributes
    ----------
    patch_embed : lucid.nn.Conv2d
        Patchifying projection — stride equals kernel, so patches do not
        overlap.
    pos_embed : Tensor
        Frozen two-dimensional sine-cosine table, added to the tokens.
    blocks : lucid.nn.ModuleList
        The DiT blocks.
    final : _FinalLayer
        Modulated norm and projection back to patch space.

    Notes
    -----
    Reference: Geng, Deng, Bai, Kolter, and He, *"Mean Flows for One-step
    Generative Modeling"*, arXiv:2505.13447, 2025.  Backbone
    configurations are Table 4.

    The forward takes both times.  Which of them reach the embedding is
    :attr:`MeanFlowConfig.time_conditioning`; the signature does not
    change with it, because the Jacobian-vector product the objective
    needs is always taken with respect to :math:`(z, r, t)` no matter how
    the network chooses to encode them.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.mean_flow import (
    ...     MeanFlowConfig, MeanFlowModel)
    >>> config = MeanFlowConfig(sample_size=8, patch_size=2, hidden_size=32,
    ...                         depth=2, num_heads=4, num_classes=10)
    >>> model = MeanFlowModel(config).eval()
    >>> z = lucid.randn((1, 4, 8, 8))
    >>> t = lucid.tensor([1.0])
    >>> r = lucid.tensor([0.0])
    >>> model(z, r, t).sample.shape
    (1, 4, 8, 8)
    """

    config_class: ClassVar[type[MeanFlowConfig]] = MeanFlowConfig
    base_model_prefix = "mean_flow"

    def __init__(self, config: MeanFlowConfig) -> None:
        """Initialise the network. See the class docstring for parameters."""
        super().__init__(config)
        self.config: MeanFlowConfig = config
        side = patch_grid_side(config.sample_size)
        self.grid = side // config.patch_size

        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        # A buffer, not a bare attribute: it moves with ``.metal()``,
        # appears where a reader looks for it, and stops being copied
        # host-to-device on every forward.  ``persistent=False`` because
        # it is derived from the config and carries nothing a checkpoint
        # needs — the same choice Swin makes for its position index.
        self.register_buffer(
            "pos_embed",
            sincos_position_embedding(config.hidden_size, self.grid),
            persistent=False,
        )

        # One MLP per embedded time variable.  ``t_r_interval`` embeds
        # three, the rest two or one; a spare module would show up in the
        # parameter count and in every checkpoint, so only what the
        # configuration uses is built.
        self.time_mlps = nn.ModuleList(
            [
                _TimeMLP(config.hidden_size, config.frequency_embedding_size)
                for _ in range(self._num_time_inputs())
            ]
        )
        # The unconditional token is the extra row: guidance needs a label
        # embedding to fall back to when the class is dropped.
        self.label_embed = nn.Embedding(config.num_classes + 1, config.hidden_size)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(config.hidden_size, config.num_heads, config.mlp_ratio)
                for _ in range(config.depth)
            ]
        )
        self.final = DiTFinalLayer(
            config.hidden_size, config.patch_size, config.out_channels
        )
        self._zero_init_conditioning()

    def _num_time_inputs(self) -> int:
        """How many scalars the configured conditioning embeds."""
        return {
            "t_interval": 2,
            "t_r": 2,
            "t_r_interval": 3,
            "interval": 1,
        }[self.config.time_conditioning]

    def _time_inputs(self, r: Tensor, t: Tensor) -> list[Tensor]:
        """The scalars to embed, per :attr:`MeanFlowConfig.time_conditioning`."""
        mode = self.config.time_conditioning
        if mode == "t_interval":
            return [t, t - r]
        if mode == "t_r":
            return [t, r]
        if mode == "t_r_interval":
            return [t, r, t - r]
        return [t - r]

    def _zero_init_conditioning(self) -> None:
        """Zero every adaLN projection, which is what "Zero" names.

        Delegated to the blocks — they own those projections, and a model
        that reached into them would go stale the moment the shared block
        grew another one.
        """
        for block in self.blocks:
            block.zero_conditioning()
        self.final.zero_conditioning()

    @override
    def forward(
        self,
        z: Tensor,
        r: Tensor,
        t: Tensor,
        labels: Tensor | None = None,
    ) -> DiffusionModelOutput:
        r"""Evaluate the average-velocity field.

        Parameters
        ----------
        z : Tensor
            ``(B, in_channels, H, W)`` — the point on the path.
        r : Tensor
            ``(B,)`` — the interval's start.
        t : Tensor
            ``(B,)`` — the interval's end, and where ``z`` lives.
        labels : Tensor or None, optional
            ``(B,)`` class indices.  ``None`` uses the unconditional
            token, which is index ``num_classes``.

        Returns
        -------
        DiffusionModelOutput
            ``sample`` of shape ``(B, out_channels, H, W)`` —
            :math:`u_\theta(z, r, t)`.  ``loss`` stays ``None``: the
            objective lives in :class:`MeanFlowForImageGeneration`.
        """
        tokens = self.patch_embed(z)
        tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], -1).permute(0, 2, 1)
        tokens = tokens + self.pos_embed

        cond = self.time_mlps[0](self._time_inputs(r, t)[0])
        for mlp, scalar in zip(
            list(self.time_mlps)[1:], self._time_inputs(r, t)[1:], strict=True
        ):
            cond = cond + mlp(scalar)

        if labels is None:
            labels = lucid.full(
                (z.shape[0],),
                float(self.config.num_classes),
                dtype=lucid.int64,
                device=z.device.type,
            )
        cond = cond + self.label_embed(labels)

        for block in self.blocks:
            tokens = block(tokens, cond)
        image = unpatchify(
            self.final(tokens, cond),
            self.grid,
            self.config.patch_size,
            self.config.out_channels,
        )
        return DiffusionModelOutput(sample=image)


class MeanFlowForImageGeneration(ImageGenerationModel):
    r"""MeanFlow's objective and its one-step sampler.

    Parameters
    ----------
    config : MeanFlowConfig
        The variant to build.

    Attributes
    ----------
    mean_flow : MeanFlowModel
        The average-velocity network.

    Notes
    -----
    Reference: Geng, Deng, Bai, Kolter, and He, *"Mean Flows for One-step
    Generative Modeling"*, arXiv:2505.13447, 2025.  Training is
    Algorithm 1, sampling is Algorithm 2 and Eq. 12.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.mean_flow import (
    ...     MeanFlowConfig, MeanFlowForImageGeneration)
    >>> config = MeanFlowConfig(sample_size=8, patch_size=2, hidden_size=32,
    ...                         depth=2, num_heads=4, num_classes=10)
    >>> model = MeanFlowForImageGeneration(config).eval()
    >>> model.generate(2).samples.shape
    (2, 4, 8, 8)
    """

    config_class: ClassVar[type[MeanFlowConfig]] = MeanFlowConfig
    base_model_prefix = "mean_flow"

    def __init__(self, config: MeanFlowConfig) -> None:
        """Initialise the model. See the class docstring for parameters."""
        super().__init__(config)
        self.config: MeanFlowConfig = config
        self.mean_flow = MeanFlowModel(config)

    def _sample_times(self, batch: int, device: str) -> tuple[Tensor, Tensor]:
        r"""Draw ``(r, t)`` with ``t`` the larger, then flatten a share to ``r = t``.

        The two are drawn independently and sorted rather than drawn as an
        ordered pair, which is what the paper describes; the ratio that
        keeps :math:`r \neq t` is then imposed on top.  At a ratio of zero
        every sample has :math:`r = t`, the second term of the target
        vanishes, and the objective is exactly Flow Matching.
        """
        config = self.config
        if config.time_sampler == "uniform":
            a = lucid.rand((batch,), device=device)
            b = lucid.rand((batch,), device=device)
        else:
            a = F.sigmoid(
                lucid.randn((batch,), device=device) * config.lognorm_std
                + config.lognorm_mean
            )
            b = F.sigmoid(
                lucid.randn((batch,), device=device) * config.lognorm_std
                + config.lognorm_mean
            )
        t = lucid.maximum(a, b)
        r = lucid.minimum(a, b)

        keep = (lucid.rand((batch,), device=device) < config.ratio_r_not_t).float()
        return r * keep + t * (1.0 - keep), t

    def _velocity_target(
        self, z: Tensor, t: Tensor, noise: Tensor, images: Tensor, labels: Tensor | None
    ) -> Tensor:
        r"""The instantaneous velocity the target is built from.

        Without guidance this is the conditional velocity
        :math:`v_t = \epsilon - x`.  With it, the paper replaces that by a
        mixture of the sample velocity and the model's own average
        velocity at :math:`r = t` — guidance becomes a property of the
        field being learned rather than something applied at sampling
        time, which is what keeps generation at one evaluation.
        """
        config = self.config
        v = noise - images
        omega, kappa = config.guidance_scale, config.guidance_mix
        if omega == 1.0 and kappa == 0.0:
            return v

        low, high = config.guidance_interval
        inside = ((t >= low) & (t <= high)).to(z.dtype).reshape(-1, 1, 1, 1)

        with lucid.no_grad():
            uncond = self.mean_flow(z, t, t, None).sample
            cond = (
                self.mean_flow(z, t, t, labels).sample if labels is not None else uncond
            )
        guided = omega * v + kappa * cond + (1.0 - omega - kappa) * uncond
        return guided * inside + v * (1.0 - inside)

    @override
    def forward(self, images: Tensor, labels: Tensor | None = None) -> MeanFlowOutput:
        r"""One training step of the MeanFlow objective.

        Parameters
        ----------
        images : Tensor
            ``(B, C, H, W)`` — data, or the VAE latent of it.
        labels : Tensor or None, optional
            ``(B,)`` class indices.  Dropped to the unconditional token
            with probability :attr:`MeanFlowConfig.class_dropout`, which
            is what leaves the model able to produce both fields.

        Returns
        -------
        MeanFlowOutput
            The weighted loss, the prediction and the target.

        Notes
        -----
        The Jacobian-vector product is taken along :math:`(v, 0, 1)` —
        the tangent that the chain rule gives for
        :math:`\frac{d}{dt}u(z_t, r, t)` once :math:`dz_t/dt = v` and
        :math:`dr/dt = 0` are substituted.  Any other tangent trains a
        different quantity: the paper's destructive ablation reports FID
        61.06 for this one and 137–329 for the alternatives.
        """
        config = self.config
        batch = int(images.shape[0])
        device = images.device.type

        # Drawn in float32, as at every dtype, then in the images' dtype.
        r, t = (v.to(images.dtype) for v in self._sample_times(batch, device))
        noise = lucid.randn(images.shape, dtype=images.dtype, device=device)
        t_b = t.reshape(-1, 1, 1, 1)
        z = (1.0 - t_b) * images + t_b * noise

        if labels is not None and config.class_dropout > 0.0:
            drop = lucid.rand((batch,), device=device) < config.class_dropout
            labels = lucid.where(
                drop,
                lucid.full(
                    (batch,),
                    float(config.num_classes),
                    dtype=labels.dtype,
                    device=device,
                ),
                labels,
            )

        v = self._velocity_target(z, t, noise, images, labels)

        # ``jvp`` differentiates a function of tensors to a tensor, so the
        # field hands it the prediction, not the output record around it.
        def field(z_: Tensor, r_: Tensor, t_: Tensor) -> Tensor:
            return self.mean_flow(z_, r_, t_, labels).sample

        u, dudt = lucid.func.jvp(
            field,
            (z, r, t),
            (v, lucid.zeros_like(r), lucid.ones_like(t)),
        )
        u = cast(Tensor, u)
        dudt = cast(Tensor, dudt)

        interval = (t - r).reshape(-1, 1, 1, 1)
        # The stop-gradient of Eq. 9.  Redundant against this backend —
        # ``lucid.func.jvp`` computes its tangent through a backward pass
        # and returns it already detached, so removing this changes no
        # gradient (measured: identical to ten decimals).  Kept because
        # the paper makes it part of the objective and because a JVP that
        # someday returns a live graph would otherwise silently start
        # back-propagating through the Jacobian — the double
        # backpropagation this whole formulation exists to avoid.
        target = (v - interval * dudt).detach()

        error = u - target
        squared = (error**2).sum(dim=(1, 2, 3))
        # The adaptive weight is the loss's own magnitude fed back as a
        # stop-gradiented scale.  ``p = 0`` is the plain squared error and
        # underperforms; ``p = 1`` is the paper's default and its best.
        weight = 1.0 / (squared + config.adaptive_weight_eps) ** (
            config.adaptive_weight_power
        )
        loss = (weight.detach() * squared).mean()
        return MeanFlowOutput(loss=loss, prediction=u, target=target)

    def generate(
        self,
        n_samples: int = 1,
        *,
        labels: Tensor | None = None,
        steps: int = 1,
        noise: Tensor | None = None,
        device: str | None = None,
    ) -> GenerationOutput:
        r"""Sample by walking the average velocity backwards from noise.

        Parameters
        ----------
        n_samples : int, default=1
            How many to draw.
        labels : Tensor or None, optional
            ``(n_samples,)`` class indices.  ``None`` samples the
            unconditional field.
        steps : int, default=1
            Network evaluations.  One is the point of the method:
            :math:`z_0 = z_1 - u(z_1, 0, 1)` covers the whole path.  More
            subdivide :math:`[0, 1]` and apply Eq. 12 on each piece, which
            the paper notes is straightforward and reports at two.
        noise : Tensor or None, optional
            Starting point :math:`z_1`.  Drawn when absent.
        device : str or None, optional
            Where to draw.  Defaults to the model's own device.

        Returns
        -------
        GenerationOutput
            ``samples`` of shape ``(n_samples, out_channels, H, W)``.

        Raises
        ------
        ValueError
            If ``steps`` is not positive.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.mean_flow import (
        ...     MeanFlowConfig, MeanFlowForImageGeneration)
        >>> config = MeanFlowConfig(sample_size=8, patch_size=2,
        ...                         hidden_size=32, depth=2, num_heads=4,
        ...                         num_classes=10)
        >>> model = MeanFlowForImageGeneration(config).eval()
        >>> model.generate(2).samples.shape
        (2, 4, 8, 8)
        """
        if steps < 1:
            raise ValueError(f"steps must be positive, got {steps}")
        config = self.config
        side = patch_grid_side(config.sample_size)
        device = resolve_generation_device(self, device)
        if noise is None:
            noise = lucid.randn(
                (n_samples, config.in_channels, side, side), device=device
            )

        z = noise
        with lucid.no_grad():
            for index in range(steps):
                t_value = 1.0 - index / steps
                r_value = 1.0 - (index + 1) / steps
                t = lucid.full((n_samples,), t_value, device=device, dtype=z.dtype)
                r = lucid.full((n_samples,), r_value, device=device, dtype=z.dtype)
                u = self.mean_flow(z, r, t, labels).sample
                z = z - (t_value - r_value) * u
        return GenerationOutput(samples=z)
