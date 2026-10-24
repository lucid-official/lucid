r"""Score-SDE — Song et al., 2021.

A time-dependent score network, the three SDEs it can be trained under,
and the three ways of sampling from it the continuous view makes
available: a plain solver, Predictor-Corrector, and the probability-flow
ODE.

The last is the one worth reading.  Because the deterministic process of
equation 13 shares the reverse SDE's marginals exactly, sampling becomes
an ordinary initial-value problem — and this library already ships a
tested adaptive solver suite, so it is handed to
:func:`lucid.diffeq.odeint` rather than stepped by hand.
"""

import math

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, cast, override

import lucid
from lucid._dtype import _float32_or_wider
from lucid._tensor.tensor import Tensor
from lucid.diffeq import odeint
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageGenerationModel
from lucid.models._output import GenerationOutput, ModelOutput
from lucid.models._utils._generative import model_dtype, resolve_generation_device
from lucid.models.generative.ddpm._config import DDPMConfig
from lucid.models.generative.ddpm._model import _DDPMUNet
from lucid.models.generative.score_sde._config import ScoreSDEConfig
from lucid.models.generative.score_sde._sde import SDE, VESDE, make_sde
from lucid.models._utils._common import host_floats

__all__ = ["ScoreSDEModel", "ScoreSDEForImageGeneration", "ScoreSDEOutput"]

_Guidance = Callable[[Tensor, Tensor], Tensor]


@dataclass(slots=True)
class ScoreSDEOutput(ModelOutput):
    """What the model returns for a batch.

    Attributes
    ----------
    score : Tensor
        :math:`\\nabla_x \\log p_t(x)` at the perturbed sample, same shape
        as the input.
    noise : Tensor
        What the network predicted, before it was divided by the standard
        deviation.
    target : Tensor
        The noise actually added by the perturbation kernel.  Returned
        because the objective is the error against it and recomputing it
        would mean redrawing.
    std : Tensor
        The perturbation kernel's scale at the sampled times, ``(N,)``.
    loss : Tensor or None
        The denoising score-matching objective, set only by
        :class:`ScoreSDEForImageGeneration`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import score_sde_vp
    >>> model = score_sde_vp(sample_size=8, base_channels=8,
    ...     channel_mult=(1,), num_res_blocks=1, resnet_groups=4,
    ...     attention_resolutions=()).eval()
    >>> out = model(lucid.randn((1, 3, 8, 8)))
    >>> out.score.shape
    (1, 3, 8, 8)
    """

    score: Tensor
    noise: Tensor
    target: Tensor
    std: Tensor
    loss: Tensor | None = None


class ScoreSDEModel(PretrainedModel):
    r"""A time-dependent score network over one of the three SDEs.

    Parameters
    ----------
    config : ScoreSDEConfig
        Frozen configuration.

    Notes
    -----
    Reference: Song, Sohl-Dickstein, Kingma, Kumar, Ermon, and Poole,
    *"Score-Based Generative Modeling through Stochastic Differential
    Equations"*, ICLR, 2021 (arXiv:2011.13456).

    The network predicts the noise that was added; the score follows from
    the perturbation kernel, since for
    :math:`x(t) = \mu + \sigma \epsilon`

    .. math::

        \nabla_x \log p_{0t}(x(t) \mid x(0))
            = -\frac{x(t) - \mu}{\sigma^2}
            = -\frac{\epsilon}{\sigma}.

    Dividing by :math:`\sigma` rather than folding it into the network is
    what lets one trained model serve every sampler here — the samplers
    ask for a score and get one at whatever :math:`t` they are at.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import score_sde_ve
    >>> model = score_sde_ve(sample_size=8, base_channels=8,
    ...     channel_mult=(1,), num_res_blocks=1, resnet_groups=4,
    ...     attention_resolutions=()).eval()
    >>> model.sde.__class__.__name__
    'VESDE'
    """

    config_class: ClassVar[type[ScoreSDEConfig]] = ScoreSDEConfig
    base_model_prefix: ClassVar[str] = "score_sde"

    def __init__(self, config: ScoreSDEConfig) -> None:
        super().__init__(config)
        self.sde: SDE = make_sde(
            config.sde_type,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            beta_min=config.beta_min,
            beta_max=config.beta_max,
        )
        self.unet = _DDPMUNet(
            DDPMConfig(
                sample_size=config.sample_size,
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                act_fn=config.act_fn,
                num_train_timesteps=config.num_scales,
                base_channels=config.base_channels,
                channel_mult=config.channel_mult,
                num_res_blocks=config.num_res_blocks,
                attention_resolutions=config.attention_resolutions,
                num_heads=config.num_heads,
                dropout=config.dropout,
                resnet_groups=config.resnet_groups,
            )
        )
        self._num_scales = config.num_scales

    def _conditioning(self, t: Tensor) -> Tensor:
        """Map continuous ``t`` onto the U-Net's discrete embedding.

        Notes
        -----
        ``_DDPMUNet`` conditions on an integer step, so ``t`` is scaled by
        ``num_scales - 1``.  The reference implementation does the same
        for its continuous VP models; for VE it conditions on
        :math:`\\sigma(t)` instead, and that difference is preserved here
        because the two ranges are nothing alike — VE's sigma reaches 50
        while VP's time never leaves the unit interval.
        """
        if isinstance(self.sde, VESDE):
            return self.sde.sigma(t)
        return t * float(self._num_scales - 1)

    def predict_noise(self, x: Tensor, t: Tensor) -> Tensor:
        """What the network thinks was added — same shape as ``x``."""
        return self.unet(x, self._conditioning(t))

    def score(self, x: Tensor, t: Tensor) -> Tensor:
        r""":math:`\nabla_x \log p_t(x)`, the quantity every sampler wants.

        Parameters
        ----------
        x : Tensor
            Perturbed samples, ``(N, C, H, W)``.
        t : Tensor
            Times, ``(N,)``.

        Returns
        -------
        Tensor
            The score, same shape as ``x``.

        Examples
        --------
        >>> import lucid
        >>> import lucid.nn as nn
        >>> from lucid.models import score_sde_vp
        >>> model = score_sde_vp(sample_size=8, base_channels=8,
        ...     channel_mult=(1,), num_res_blocks=1, resnet_groups=4,
        ...     attention_resolutions=()).eval()
        >>> # The output layer starts at zero; give it weights so the check bites.
        >>> _ = nn.init.normal_(model.unet.conv_out.weight, std=0.05)
        >>> x, t = lucid.randn((2, 3, 8, 8)), lucid.tensor([0.05, 0.9])
        >>> score = model.score(x, t)
        >>> score.shape
        (2, 3, 8, 8)
        >>> _, std = model.sde.marginal_prob(x, t)
        >>> noise = model.predict_noise(x, t)
        >>> bool(lucid.allclose(score * std.reshape(2, 1, 1, 1), -noise, atol=1e-5))
        True
        """
        _, std = self.sde.marginal_prob(x, t)
        noise = self.predict_noise(x, t)
        return -noise / std.reshape(int(std.shape[0]), *([1] * (x.ndim - 1)))

    @override
    def forward(self, x: Tensor, t: Tensor | None = None) -> ScoreSDEOutput:
        """Perturb, then score.

        Parameters
        ----------
        x : Tensor
            Clean samples, ``(N, C, H, W)``.
        t : Tensor or None, optional
            Times to evaluate at.  ``None`` draws them uniformly from
            ``[t_min, 1]``, which is what training does.

        Returns
        -------
        ScoreSDEOutput
            Without ``loss`` — that belongs to the task wrapper.
        """
        count = int(x.shape[0])
        if t is None:
            floor = self.sde.t_min
            draw = lucid.rand((count,), dtype=x.dtype, device=x.device)
            t = draw * (1.0 - floor) + floor
        mean, std = self.sde.marginal_prob(x, t)
        shaped = std.reshape(count, *([1] * (x.ndim - 1)))
        target = lucid.randn(x.shape, dtype=x.dtype, device=x.device)
        perturbed = mean + shaped * target
        noise = self.predict_noise(perturbed, t)
        return ScoreSDEOutput(
            score=-noise / shaped, noise=noise, target=target, std=std
        )


class ScoreSDEForImageGeneration(ImageGenerationModel):
    r"""Score-SDE with its objective and the three samplers.

    Parameters
    ----------
    config : ScoreSDEConfig
        Frozen configuration.

    Notes
    -----
    Reference: Song et al., ICLR 2021 (arXiv:2011.13456).

    Trained by denoising score matching, equation 7, in the form that
    weights each time by :math:`\sigma(t)^2` — which makes the loss the
    plain mean squared error between the predicted and the actual noise,
    since the :math:`\sigma` in the score and the :math:`\sigma^2` in the
    weight cancel to one.  That is the same objective DDPM optimises,
    which is the paper's point.

    Three ways to sample, all from one trained network:

    ``"euler"``
        Integrate the reverse-time SDE, equation 6.
    ``"pc"``
        The same, with Langevin corrector steps at fixed :math:`t`.  The
        paper's Predictor-Corrector.
    ``"ode"``
        The probability-flow ODE, equation 13, handed to
        :func:`lucid.diffeq.odeint`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import score_sde_vp_gen
    >>> model = score_sde_vp_gen(sample_size=8, base_channels=8,
    ...     channel_mult=(1,), num_res_blocks=1, resnet_groups=4,
    ...     attention_resolutions=()).eval()
    >>> out = model(lucid.randn((2, 3, 8, 8)))
    >>> bool(out.loss.ndim == 0)
    True
    """

    config_class: ClassVar[type[ScoreSDEConfig]] = ScoreSDEConfig
    base_model_prefix: ClassVar[str] = "score_sde"

    def __init__(self, config: ScoreSDEConfig) -> None:
        super().__init__(config)
        self.score_sde = ScoreSDEModel(config)
        self._snr = config.snr
        self._corrector_steps = config.corrector_steps
        self._num_scales = config.num_scales

    @property
    def sde(self) -> SDE:
        """The process this model was trained under."""
        return self.score_sde.sde

    @override
    def forward(self, x: Tensor, t: Tensor | None = None) -> ScoreSDEOutput:
        """Denoising score matching.

        Parameters
        ----------
        x : Tensor
            Clean samples, ``(N, C, H, W)``.
        t : Tensor or None, optional
            As :meth:`ScoreSDEModel.forward`.

        Returns
        -------
        ScoreSDEOutput
            With ``loss`` set.
        """
        out = self.score_sde(x, t)
        count = int(x.shape[0])
        # Equation 7 weighted by sigma(t)^2. The score is -eps_hat/sigma and
        # its target is -eps/sigma, so the weight cancels both sigmas and
        # what is left is the plain error between the two noises — which is
        # exactly DDPM's objective, and the paper's point.
        loss = ((out.noise - out.target) ** 2).reshape(count, -1).sum(dim=-1).mean()
        return ScoreSDEOutput(
            score=out.score,
            noise=out.noise,
            target=out.target,
            std=out.std,
            loss=loss,
        )

    def _timesteps(self, steps: int, device: str) -> Tensor:
        """Descending times from 1 to the SDE's floor."""
        return lucid.linspace(1.0, self.sde.t_min, steps, device=device)

    def _guided_score(self, x: Tensor, t: Tensor, guidance: _Guidance | None) -> Tensor:
        r"""The score, plus a conditioning term if one was supplied.

        Notes
        -----
        The paper's *controllable generation*: "the conditional
        reverse-time SDE can be efficiently estimated from unconditional
        scores", because

        .. math::

            \nabla_x \log p_t(x \mid y)
                = \nabla_x \log p_t(x) + \nabla_x \log p_t(y \mid x).

        So class-conditioning, inpainting and colourisation are the same
        sampler with a second term added — "all achievable using a single
        unconditional score-based model without re-training".
        """
        score = self.score_sde.score(x, t)
        return score if guidance is None else score + guidance(x, t)

    def _predictor(
        self, x: Tensor, t: Tensor, dt: float, guidance: _Guidance | None = None
    ) -> Tensor:
        """One reverse-time Euler-Maruyama step — equation 6."""
        count = int(x.shape[0])
        g = self.sde.diffusion(t).reshape(count, *([1] * (x.ndim - 1)))
        drift = self.sde.drift(x, t) - g**2 * self._guided_score(x, t, guidance)
        noise = lucid.randn(x.shape, dtype=x.dtype, device=x.device)
        return cast(Tensor, x + drift * (-dt) + g * (dt**0.5) * noise)

    def _corrector(
        self, x: Tensor, t: Tensor, guidance: _Guidance | None = None
    ) -> Tensor:
        """Langevin steps at fixed ``t`` — the 'corrector' half."""
        # The step size is read from two norms, taken at float32 or wider:
        # a float16 score's squares overflow once summed over an image.
        work = _float32_or_wider(x.dtype)
        for _ in range(self._corrector_steps):
            score = self._guided_score(x, t, guidance)
            noise = lucid.randn(x.shape, dtype=x.dtype, device=x.device)
            count = int(x.shape[0])
            grad_norm = float(
                ((score.reshape(count, -1).to(work) ** 2).sum(dim=-1) ** 0.5)
                .mean()
                .item()
            )
            noise_norm = float(
                ((noise.reshape(count, -1).to(work) ** 2).sum(dim=-1) ** 0.5)
                .mean()
                .item()
            )
            if grad_norm == 0.0:
                break
            step = 2.0 * (self._snr * noise_norm / grad_norm) ** 2
            x = x + step * score + (2.0 * step) ** 0.5 * noise
        return x

    def generate(
        self,
        n_samples: int = 1,
        *,
        method: str = "pc",
        steps: int | None = None,
        device: str | None = None,
        guidance: _Guidance | None = None,
    ) -> GenerationOutput:
        r"""Draw samples by integrating backwards from the prior.

        Parameters
        ----------
        n_samples : int, default=1
            How many.
        method : {"pc", "euler", "ode"}, default="pc"
            ``"euler"`` integrates the reverse SDE, ``"pc"`` adds Langevin
            correction, ``"ode"`` solves the probability-flow ODE with
            :func:`lucid.diffeq.odeint`.
        steps : int or None, optional
            Solver steps; ``None`` uses ``num_scales``.  That this is a
            sampling argument rather than a property of the trained model
            is the framework's point.
        device : str or None, optional
            Where to sample; ``None`` follows the parameters.
        guidance : callable or None, optional
            ``(x, t) -> Tensor``, returning
            :math:`\nabla_x \\log p_t(y \\mid x)`.  Added to the score, which
            is all conditional sampling is — the paper's controllable
            generation needs no retraining and no conditional model.

        Returns
        -------
        GenerationOutput
            ``samples`` of shape ``(n_samples, C, H, W)``.

        Raises
        ------
        ValueError
            If ``method`` is not one of the three.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models import score_sde_vp_gen
        >>> model = score_sde_vp_gen(sample_size=8, base_channels=8,
        ...     channel_mult=(1,), num_res_blocks=1, resnet_groups=4,
        ...     attention_resolutions=()).eval()
        >>> [model.generate(n_samples=2, method=m, steps=2).samples.shape
        ...  for m in ("euler", "pc", "ode")]
        [(2, 3, 8, 8), (2, 3, 8, 8), (2, 3, 8, 8)]
        >>> model.generate(method="ddim")
        Traceback (most recent call last):
            ...
        ValueError: method must be 'pc', 'euler' or 'ode', got 'ddim'
        """
        if method not in ("pc", "euler", "ode"):
            raise ValueError(f"method must be 'pc', 'euler' or 'ode', got {method!r}")
        config = cast(ScoreSDEConfig, self.config)
        resolved = resolve_generation_device(self, device)
        dtype = model_dtype(self)
        size = config.sample_size
        assert isinstance(size, int)
        shape: tuple[int, ...] = (n_samples, config.in_channels, size, size)
        count = steps or self._num_scales

        with lucid.no_grad():
            if method == "ode":
                return GenerationOutput(
                    samples=self._solve_ode(shape, resolved, dtype, count, guidance)
                )
            x = self.sde.prior_sampling(shape, resolved, dtype=dtype)
            times = host_floats(self._timesteps(count, resolved))
            dt = (1.0 - self.sde.t_min) / max(count - 1, 1)
            for time in times:
                # In the sample's dtype, as training draws its times.
                t = lucid.ones((n_samples,), dtype=dtype, device=resolved) * time
                if method == "pc":
                    x = self._corrector(x, t, guidance)
                x = self._predictor(x, t, dt, guidance)
        return GenerationOutput(samples=x)

    def _solve_ode(
        self,
        shape: tuple[int, ...],
        device: str,
        dtype: lucid.dtype,
        steps: int,
        guidance: _Guidance | None = None,
    ) -> Tensor:
        r"""The probability-flow ODE, equation 13, through ``lucid.diffeq``.

        Notes
        -----
        This is why the continuous view is worth the trouble.  The
        deterministic process

        .. math::

            dx = \Big[f(x, t)
                 - \tfrac{1}{2} g(t)^2 \nabla_x \log p_t(x)\Big] dt

        has the same marginals as the reverse SDE at every :math:`t`, so
        integrating it draws from the same distribution — and it is an
        ordinary initial-value problem, which this library already has a
        tested solver for.  No sampler is written here beyond the
        right-hand side.
        """
        count = shape[0]

        def rhs(t_scalar: Tensor, flat: Tensor) -> Tensor:
            x = flat.reshape(*shape)
            t = lucid.ones((count,), dtype=flat.dtype, device=device) * float(
                t_scalar.item()
            )
            g = self.sde.diffusion(t).reshape(count, *([1] * (x.ndim - 1)))
            drift = self.sde.drift(x, t) - 0.5 * g**2 * self._guided_score(
                x, t, guidance
            )
            return drift.reshape(-1)

        start = self.sde.prior_sampling(shape, device, dtype=dtype).reshape(-1)
        path = odeint(
            rhs,
            start,
            lucid.linspace(1.0, self.sde.t_min, steps, device=device),
            method="rk4",
        )
        return cast(Tensor, path[-1]).reshape(*shape)

    def log_likelihood(
        self, x: Tensor, *, steps: int = 100, eps: Tensor | None = None
    ) -> Tensor:
        r"""Exact log-likelihood of the data, in nats.

        Parameters
        ----------
        x : Tensor
            Data, ``(N, C, H, W)``.
        steps : int, default=100
            Solver steps for the forward integration.
        eps : Tensor or None, optional
            The probe vector of the trace estimator.  ``None`` draws
            Rademacher noise.  Exposed so a test can hold it fixed, since
            the estimate is stochastic.

        Returns
        -------
        Tensor
            ``(N,)`` log-densities, reported at float32 or wider because it
            sums over every dimension: a float16 sum of a VE prior's
            squares overflows past a few dozen pixels.

        Notes
        -----
        Song et al. (2021), Appendix D.2.  Because the probability-flow
        ODE is a neural ODE, the instantaneous change of variables gives

        .. math::

            \log p_0(x(0)) = \log p_T(x(T))
                + \int_0^T \nabla \cdot \tilde{f}_\theta(x(t), t)\, dt,

        equation 39, where :math:`x(t)` solves that ODE.  The divergence
        is expensive, so it is estimated as the paper does — equation 40,
        the Skilling-Hutchinson trace estimator

        .. math::

            \nabla \cdot \tilde{f}_\theta(x, t)
                = \mathbb{E}_{p(\epsilon)}
                  \big[\epsilon^\top \nabla \tilde{f}_\theta(x, t)\,
                  \epsilon\big],

        with :math:`\epsilon` of zero mean and identity covariance.  The
        vector-Jacobian product is one backward pass, so the whole
        estimate costs one extra gradient per solver step.

        This is what the deterministic sampler buys that a stochastic one
        cannot: an exact density, not a bound.  It is also **stochastic in
        the estimator** — two calls differ, and the variance falls with
        the number of probes, not with ``steps``.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models import score_sde_vp_gen
        >>> model = score_sde_vp_gen(sample_size=8, base_channels=8,
        ...     channel_mult=(1,), num_res_blocks=1, resnet_groups=4,
        ...     attention_resolutions=()).eval()
        >>> model.log_likelihood(lucid.randn((1, 3, 8, 8)), steps=3).shape
        (1,)
        """
        count = int(x.shape[0])
        shape = tuple(int(v) for v in x.shape)
        probe = eps
        if probe is None:
            # Rademacher: mean zero, identity covariance, as equation 40 asks.
            draw = lucid.rand(shape, device=x.device)
            probe = (draw > 0.5).to(x.dtype) * 2.0 - 1.0

        # The times in the data's dtype, as the samplers keep theirs; the sums
        # over every dimension at float32 or wider.
        work = _float32_or_wider(x.dtype)
        divergence = lucid.zeros((count,), dtype=work, device=x.device)
        current = x
        times = lucid.linspace(self.sde.t_min, 1.0, steps + 1, device=x.device)
        bounds = host_floats(times)

        for i in range(steps):
            start = bounds[i]
            step = bounds[i + 1] - start
            t = lucid.ones((count,), dtype=x.dtype, device=x.device) * start
            state = current.detach()
            state.requires_grad = True
            g = self.sde.diffusion(t).reshape(count, *([1] * (x.ndim - 1)))
            drift = self.sde.drift(state, t) - 0.5 * g**2 * self.score_sde.score(
                state, t
            )
            # eps^T (d drift / d x) eps, by one vector-Jacobian product.
            (drift * probe).sum().backward()
            assert state.grad is not None
            trace = (state.grad * probe).reshape(count, -1).to(work).sum(dim=-1)
            divergence = divergence + trace.detach() * step
            current = (state + drift * step).detach()

        prior = self._prior_log_prob(current)
        return prior + divergence

    def _prior_log_prob(self, x: Tensor) -> Tensor:
        r"""log :math:`p_T(x(T))`, the density the ODE integrates up to.

        Notes
        -----
        A zero-mean Gaussian in both cases; only the width differs, which
        is exactly the VE/VP distinction and the reason this is asked of
        the SDE rather than assumed by the caller.  Summed at float32 or
        wider, as :meth:`log_likelihood` reports.
        """
        count = int(x.shape[0])
        dims = 1
        for size in tuple(int(v) for v in x.shape)[1:]:
            dims *= size
        variance = self.sde.prior_variance
        flat = x.reshape(count, -1).to(_float32_or_wider(x.dtype))
        return -0.5 * (
            dims * math.log(2.0 * math.pi * variance) + (flat**2).sum(dim=-1) / variance
        )
