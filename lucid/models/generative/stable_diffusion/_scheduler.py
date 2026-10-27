r"""The samplers — DDIM and PNDM — over the released noise schedule.

**Which one the release actually uses.**  The published
``scheduler_config.json`` names ``PNDMScheduler``, and ``model_index.json``
wires it into the pipeline; ``skip_prk_steps`` in that file is a PNDM
field and nothing else reads it.  The LDM paper, on the other hand,
samples with DDIM and reports its FID under it.

Both are here for that reason.  :class:`DDIMScheduler` is the paper's
and the simpler contract — one parameter :math:`\eta` spanning DDPM at
1 and deterministic DDIM at 0.  :class:`PNDMScheduler` is the release's
default and is what reproduces its images.  They share the schedule,
the :math:`\bar\alpha` table and the epsilon prediction; only the step
rule differs, which is why PNDM is written on top of DDIM rather than
beside it.

DDPM's ancestral sampler needs one network evaluation per training
timestep, so a thousand-step model costs a thousand forwards. DDIM
(Song et al., ICLR 2021) reinterprets the same trained network as
defining a *non-Markovian* process whose reverse can skip: the update

.. math::

    z_{t-1} = \sqrt{\bar\alpha_{t-1}}\,
        \underbrace{\frac{z_t - \sqrt{1-\bar\alpha_t}\,
        \epsilon_\theta}{\sqrt{\bar\alpha_t}}}_{\hat z_0}
        + \sqrt{1 - \bar\alpha_{t-1} - \sigma_t^2}\; \epsilon_\theta
        + \sigma_t \epsilon

is valid for any decreasing subsequence of timesteps, and at
:math:`\eta = 0` the :math:`\sigma_t \epsilon` term vanishes and the
whole trajectory becomes deterministic. That is what makes fifty steps
enough, and what makes the same seed reproduce the same image.

**``scaled_linear`` is not linear.** The released schedule interpolates
linearly in :math:`\sqrt{\beta}` and squares the result:

.. math::

    \beta_t = \Big((1 - \tfrac{t}{T})\sqrt{\beta_0}
        + \tfrac{t}{T}\sqrt{\beta_T}\Big)^2 .

Using a plain linear ramp between the same endpoints gives a schedule
that looks nearly identical plotted, differs most where the signal is
still strong, and denoises to mush. It has its own name here for that
reason, and a test pins the two apart.
"""

from collections.abc import Callable
from typing import cast, final

import lucid
from lucid._engine_scope import untraced
from lucid._tensor.tensor import Tensor
from lucid.models._utils._generative import extract_into_tensor
from lucid.models.generative.stable_diffusion._config import StableDiffusionConfig

__all__ = ["DDIMScheduler", "PNDMScheduler"]


@final
class DDIMScheduler:
    r"""Deterministic (or partly stochastic) sampling over a step subset.

    Parameters
    ----------
    config : StableDiffusionConfig
        Read for ``num_train_timesteps``, the betas and the schedule.

    Attributes
    ----------
    alphas_cumprod : Tensor
        :math:`\bar\alpha_t`, ``(num_train_timesteps,)``.

    Notes
    -----
    Reference: Song, Meng and Ermon, *"Denoising Diffusion Implicit
    Models"*, ICLR, 2021 (`arXiv:2010.02502
    <https://arxiv.org/abs/2010.02502>`_); the schedule constants are
    the released Stable Diffusion configuration.

    Kept as a plain object rather than a :class:`lucid.nn.Module`: it
    holds no parameters, and making it a Module would put its buffers in
    the checkpoint of every model that uses one.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.stable_diffusion import (
    ...     DDIMScheduler, StableDiffusionConfig)
    >>> scheduler = DDIMScheduler(StableDiffusionConfig())
    >>> len(scheduler.timesteps(50))
    50
    >>> scheduler.timesteps(50)[0] > scheduler.timesteps(50)[-1]
    True
    >>> scheduler.timesteps(10)[:4]
    [901, 801, 701, 601]
    """

    def __init__(self, config: StableDiffusionConfig) -> None:
        """Build the schedule. See the class docstring for parameters."""
        self.config = config
        steps = config.num_train_timesteps
        ramp = lucid.linspace(0.0, 1.0, steps)
        if config.beta_schedule == "scaled_linear":
            root = (1.0 - ramp) * (config.beta_start**0.5) + ramp * (config.beta_end**0.5)
            betas = root**2
        else:
            betas = (1.0 - ramp) * config.beta_start + ramp * config.beta_end
        self.betas = betas
        self.alphas_cumprod = lucid.cumprod(1.0 - betas, dim=0)
        # add_noise's two coefficients per timestep, tabulated so that a
        # batch of timesteps is a gather on the device rather than a read
        # of each one.  Taken in double and rounded once: 1 - ᾱ_t is not
        # exact in float32 once ᾱ_t < 0.5, and double is what the scalar
        # formula these replace computed in.
        wide = self.alphas_cumprod.to(lucid.float64)
        self._signal_scale = wide.sqrt().to(lucid.float32)
        self._noise_scale = (1.0 - wide).sqrt().to(lucid.float32)

    def _apply(self, fn: Callable[[Tensor], Tensor]) -> None:
        """Map the noising tables through a model's device / dtype move.

        Only these: :meth:`step` reads ``alphas_cumprod`` at Python-int
        timesteps, which is cheapest left on the host.
        """
        self._signal_scale = fn(self._signal_scale)
        self._noise_scale = fn(self._noise_scale)

    def _follow(self, latent: Tensor) -> None:
        """Move the noising tables to ``latent``'s device, once, if elsewhere.

        A model moves them through :meth:`_apply`; a scheduler used on its
        own was handed Metal latents with CPU tables and copied both across
        on every :meth:`add_noise`.  Moved outside any active trace, as
        ``DiffusionScheduler._follow`` does, so a recording never holds the
        copy (LCD-627, LCD-639).
        """
        device = latent.device
        if self._signal_scale.device != device:
            with untraced():
                self._apply(lambda table: table.to(device))

    def timesteps(self, num_inference_steps: int) -> list[int]:
        """The decreasing subsequence to sample over.

        Parameters
        ----------
        num_inference_steps : int
            How many network evaluations to spend.

        Returns
        -------
        list of int
            Descending indices into the training schedule.

        Raises
        ------
        ValueError
            If more steps are requested than the schedule has.

        Examples
        --------
        >>> from lucid.models.generative.stable_diffusion import (
        ...     DDIMScheduler, StableDiffusionConfig)
        >>> scheduler = DDIMScheduler(StableDiffusionConfig())
        >>> scheduler.timesteps(4)
        [751, 501, 251, 1]

        Evenly strided by ``1000 // 4`` and shifted by ``steps_offset``, so
        the last step is 1 rather than 0.  Asking for more steps than the
        schedule was trained with is an error, not a clamp.

        >>> scheduler.timesteps(1001)
        Traceback (most recent call last):
            ...
        ValueError: num_inference_steps must be in [1, 1000], got 1001
        """
        total = self.config.num_train_timesteps
        if not 1 <= num_inference_steps <= total:
            raise ValueError(
                f"num_inference_steps must be in [1, {total}], got " f"{num_inference_steps}"
            )
        stride = total // num_inference_steps
        # The released schedule is ``arange(n) * stride`` reversed, plus
        # ``steps_offset`` — 901, 801, … rather than 999, 899, ….  The
        # per-step arithmetic is identical either way, so a trajectory
        # built on the wrong times takes correct steps to a different
        # image and nothing reports it.
        ascending = [i * stride + self.config.steps_offset for i in range(num_inference_steps)]
        return ascending[::-1]

    def add_noise(self, latent: Tensor, noise: Tensor, timestep: int | Tensor) -> Tensor:
        r"""The forward process — :math:`z_t` from :math:`z_0`.

        Parameters
        ----------
        latent : Tensor
            ``(B, C, h, w)`` clean latent.
        noise : Tensor
            The same shape.
        timestep : int or Tensor
            Index into the training schedule — one for the whole batch,
            or a ``(B,)`` integer tensor with one per sample, as training
            draws them.

        Returns
        -------
        Tensor
            :math:`\sqrt{\bar\alpha_t} z_0 + \sqrt{1-\bar\alpha_t}\epsilon`.

        Raises
        ------
        IndexError
            If an ``int`` timestep is outside the schedule.  A tensor of
            timesteps raises on CPU and reads NaN on Metal, as
            :func:`~lucid.models._utils._generative.extract_into_tensor`
            describes.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.stable_diffusion import (
        ...     DDIMScheduler, StableDiffusionConfig)
        >>> scheduler = DDIMScheduler(StableDiffusionConfig())
        >>> ones, zeros = lucid.ones((1, 4, 8, 8)), lucid.zeros((1, 4, 8, 8))
        >>> scheduler.add_noise(ones, zeros, 500).shape
        (1, 4, 8, 8)

        A clean latent of ones with no noise reads off the signal
        coefficient :math:`\sqrt{\bar\alpha_t}`, and swapping the two reads
        off the noise one.  At the first step the latent is almost
        untouched; by the last it is almost all noise.

        >>> def coefficients(t):
        ...     signal = scheduler.add_noise(ones, zeros, t)[0, 0, 0, 0].item()
        ...     noise = scheduler.add_noise(zeros, ones, t)[0, 0, 0, 0].item()
        ...     return signal, noise
        >>> [round(c, 4) for c in coefficients(0) + coefficients(999)]
        [0.9996, 0.0292, 0.0683, 0.9977]

        The squares sum to one at every step, which is what keeps a
        unit-variance latent at unit variance as the noise replaces it.

        >>> signal, noise = coefficients(500)
        >>> round(signal**2 + noise**2, 6)
        1.0

        A tensor of timesteps noises each sample at its own step, the same
        as noising it alone.

        >>> batch = lucid.randn((2, 4, 8, 8))
        >>> mixed = scheduler.add_noise(batch, lucid.zeros((2, 4, 8, 8)),
        ...                             lucid.tensor([0, 999]))
        >>> bool((mixed[1:] == scheduler.add_noise(batch[1:], zeros, 999)).all())
        True
        """
        if isinstance(timestep, int):
            # A Python int is checked here, where it costs nothing; a
            # tensor is left to the gather (see ``extract_into_tensor``).
            total = self.config.num_train_timesteps
            if not -total <= timestep < total:
                raise IndexError(f"timestep must lie in [{-total}, {total}), got {timestep}")
            timestep = lucid.tensor([timestep], device=latent.device.type)
        self._follow(latent)
        shape = tuple(int(s) for s in latent.shape)
        # Cast at use: a scheduler outside a model keeps float32 tables,
        # which would widen a float16 latent.
        dtype = latent.dtype
        signal = extract_into_tensor(self._signal_scale, timestep, shape, dtype=dtype)
        noise_scale = extract_into_tensor(self._noise_scale, timestep, shape, dtype=dtype)
        return signal * latent + noise_scale * noise

    def step(
        self,
        model_output: Tensor,
        timestep: int,
        previous_timestep: int,
        latent: Tensor,
        eta: float = 0.0,
    ) -> Tensor:
        r"""One reverse step.

        Parameters
        ----------
        model_output : Tensor
            :math:`\epsilon_\theta(z_t, t, \tau_\theta(y))`.
        timestep : int
            Current :math:`t`.
        previous_timestep : int
            The next index in the subsequence, or ``-1`` for the last
            step, where :math:`\bar\alpha` is taken as 1.
        latent : Tensor
            :math:`z_t`.
        eta : float, default=0.0
            Interpolates DDIM (0) to DDPM (1). At 0 the trajectory is
            deterministic, which is what makes a seed reproducible.

        Returns
        -------
        Tensor
            :math:`z_{t-1}`.

        Raises
        ------
        ValueError
            If ``eta`` is outside ``[0, 1]``.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.stable_diffusion import (
        ...     DDIMScheduler, StableDiffusionConfig)
        >>> scheduler = DDIMScheduler(StableDiffusionConfig())
        >>> lucid.manual_seed(0)
        >>> clean = lucid.randn((1, 4, 8, 8))
        >>> noise = lucid.randn((1, 4, 8, 8))
        >>> noised = scheduler.add_noise(clean, noise, 501)

        Handed the noise that was actually added — a perfect
        :math:`\epsilon_\theta` — the step recovers :math:`\hat z_0 = z_0`,
        so going from 501 to 251 lands on the forward process at 251.

        >>> stepped = scheduler.step(noise, 501, 251, noised)
        >>> target = scheduler.add_noise(clean, noise, 251)
        >>> bool(lucid.allclose(stepped, target, atol=1e-5))
        True

        At ``eta=0`` that is the whole update, so it repeats exactly; any
        positive ``eta`` adds fresh noise on top.

        >>> bool((scheduler.step(noise, 501, 251, noised) == stepped).all())
        True
        >>> bool((scheduler.step(noise, 501, 251, noised, eta=1.0) == stepped).all())
        False
        """
        if not 0.0 <= eta <= 1.0:
            raise ValueError(f"eta must lie in [0, 1], got {eta}")
        alpha_t = float(self.alphas_cumprod[timestep].item())
        # Past the end the reference bootstraps from ``alphas_cumprod[0]``
        # unless ``set_alpha_to_one``; the released config leaves it off.
        final = 1.0 if self.config.set_alpha_to_one else float(self.alphas_cumprod[0].item())
        alpha_prev = (
            float(self.alphas_cumprod[previous_timestep].item())
            if previous_timestep >= 0
            else final
        )

        predicted_x0 = (latent - (1.0 - alpha_t) ** 0.5 * model_output) / alpha_t**0.5
        sigma = (
            eta
            * ((1.0 - alpha_prev) / (1.0 - alpha_t)) ** 0.5
            * (1.0 - alpha_t / alpha_prev) ** 0.5
            if eta > 0.0 and previous_timestep >= 0
            else 0.0
        )
        direction = max(1.0 - alpha_prev - sigma**2, 0.0) ** 0.5 * model_output
        previous = alpha_prev**0.5 * predicted_x0 + direction
        if sigma > 0.0:
            previous = previous + sigma * lucid.randn(
                tuple(int(s) for s in latent.shape),
                device=latent.device.type,
                dtype=latent.dtype,
            )
        return cast(Tensor, previous)


@final
class PNDMScheduler:
    r"""The release's default sampler — linear multistep over DDIM's step.

    Parameters
    ----------
    config : StableDiffusionConfig
        Read for the same fields :class:`DDIMScheduler` uses.

    Notes
    -----
    Reference: Liu et al., *"Pseudo Numerical Methods for Diffusion
    Models on Manifolds"*, ICLR, 2022 (`arXiv:2202.09778
    <https://arxiv.org/abs/2202.09778>`_); the constants are the released
    Stable Diffusion configuration, whose ``_class_name`` is this.

    The update is DDIM's at :math:`\eta = 0` applied not to the current
    epsilon but to an Adams-Bashforth combination of the last four:

    .. math::

        \bar\epsilon_t = \tfrac{1}{24}\big(55\epsilon_t - 59\epsilon_{t-1}
            + 37\epsilon_{t-2} - 9\epsilon_{t-3}\big),

    with lower-order openers while the history fills — the plain
    prediction first, then a trapezoid, then a two-term rule, then a
    three-term one.  Four evaluations of information per step is what
    buys the same image in fifty steps that DDIM needs more of.

    **It carries state**, unlike :class:`DDIMScheduler`.  The history,
    the step counter and the sample the opener began from live on the
    object, so one instance cannot sample two trajectories at once.
    Sequential reuse is safe: :meth:`timesteps` means "start a sample"
    and clears the history, and :meth:`step` checks that the timestep it
    is handed is the one at the counter's position.  That check is the
    one that matters — the order of the correction is chosen by position,
    so stepping out of order applies the wrong rule to right-looking
    numbers.  :meth:`reset` is there for the caller who wants to say so
    without asking for a trajectory.

    The second step repeats the first timestep rather than advancing —
    ``counter == 1`` in the reference — which looks like an off-by-one
    and is the trapezoidal opener needing both endpoints.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.stable_diffusion import (
    ...     PNDMScheduler, StableDiffusionConfig)
    >>> scheduler = PNDMScheduler(StableDiffusionConfig())
    >>> scheduler.timesteps(10)[:4]
    [901, 801, 801, 701]
    >>> scheduler.counter
    0
    """

    def __init__(self, config: StableDiffusionConfig) -> None:
        """Build the schedule. See the class docstring for parameters."""
        self._ddim = DDIMScheduler(config)
        self.config = config
        self.betas = self._ddim.betas
        self.alphas_cumprod = self._ddim.alphas_cumprod
        self.reset()

    def reset(self) -> None:
        """Clear the history so a new trajectory starts clean."""
        self.ets: list[Tensor] = []
        self.counter = 0
        self.cur_sample: Tensor | None = None
        self._stride: int | None = None
        self._trajectory: list[int] = []

    def timesteps(self, num_inference_steps: int) -> list[int]:
        """The decreasing subsequence to sample over.

        Parameters
        ----------
        num_inference_steps : int
            Steps to divide the schedule into.  One more evaluation than
            this actually runs, because the opener visits its interval
            twice.

        Returns
        -------
        list of int
            Descending indices, **one longer than requested**: the
            *second* time is visited twice.

        Notes
        -----
        Calling this starts a sample — it clears the history, the counter
        and the saved opener sample.

        The repeat is the trapezoidal opener, and the reference puts it
        in the list rather than in the step.  Its slices —
        ``[t[:-1], t[-2:-1], t[-1:]]`` — are taken on the *ascending*
        array before reversal, so the duplicated entry lands **second**
        in the descending order: 901, 801, 801, 701, and not
        …, 101, 101, 1.  Reading those slices as descending puts the
        opener at the end, where it is the wrong rule at the wrong time.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.stable_diffusion import (
        ...     PNDMScheduler, StableDiffusionConfig)
        >>> scheduler = PNDMScheduler(StableDiffusionConfig())
        >>> scheduler.timesteps(4)
        [751, 501, 501, 251, 1]

        Four intervals, five evaluations — the second time repeats for the
        opener.  Asking again starts a new sample, so the counter a previous
        trajectory advanced is cleared.

        >>> latent = lucid.randn((1, 4, 8, 8))
        >>> _ = scheduler.step(lucid.zeros((1, 4, 8, 8)), 751, latent)
        >>> scheduler.counter
        1
        >>> _ = scheduler.timesteps(4)
        >>> scheduler.counter
        0
        """
        # Asking for a trajectory is the start of a sample, so the
        # history goes with it.  Carrying it over is the silent failure
        # this class is most exposed to: the derivatives of the previous
        # sequence are the right shape and the wrong numbers.
        self.reset()
        self._stride = self.config.num_train_timesteps // num_inference_steps
        ascending = self._ddim.timesteps(num_inference_steps)[::-1]
        extended = ascending[:-1] + ascending[-2:-1] + ascending[-1:]
        self._trajectory = extended[::-1]
        return list(self._trajectory)

    def step(self, model_output: Tensor, timestep: int, latent: Tensor) -> Tensor:
        r"""One reverse step, using the history.

        Parameters
        ----------
        model_output : Tensor
            :math:`\epsilon_\theta` at ``timestep``.
        timestep : int
            Current :math:`t`.
        latent : Tensor
            :math:`z_t`.

        Returns
        -------
        Tensor
            :math:`z_{t-1}`.

        Notes
        -----
        The previous timestep is derived from the stride rather than
        passed, because the multistep rule is only valid on the uniform
        grid :meth:`timesteps` produces.  Call that first; this raises
        otherwise rather than assuming a stride.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models.generative.stable_diffusion import (
        ...     DDIMScheduler, PNDMScheduler, StableDiffusionConfig)
        >>> config = StableDiffusionConfig()
        >>> scheduler, ddim = PNDMScheduler(config), DDIMScheduler(config)
        >>> lucid.manual_seed(0)
        >>> clean = lucid.randn((1, 4, 8, 8))
        >>> noise = lucid.randn((1, 4, 8, 8))

        Every rule's weights sum to one, so a prediction that is exactly
        right at every step combines to itself and each step lands back on
        the forward process.  A whole trajectory run that way ends where
        :class:`DDIMScheduler` noises the clean latent to timestep 0.

        >>> steps = scheduler.timesteps(4)
        >>> latent = ddim.add_noise(clean, noise, steps[0])
        >>> for t in steps:
        ...     latent = scheduler.step(noise, t, latent)
        >>> bool(lucid.allclose(latent, ddim.add_noise(clean, noise, 0), atol=1e-5))
        True

        The position is checked on every call: a fresh trajectory has to
        start at 751, and anything else raises rather than apply the wrong
        order of correction.

        >>> _ = scheduler.timesteps(4)
        >>> scheduler.step(noise, 501, latent)
        Traceback (most recent call last):
            ...
        RuntimeError: expected timestep 751 at position 0 of the trajectory ...
        """
        if not hasattr(self, "_stride"):
            raise RuntimeError(
                "call timesteps() before step() — PNDM's multistep rule "
                "needs the uniform grid that call establishes"
            )
        if self._stride is None:
            raise RuntimeError(
                "call timesteps() before step() — PNDM's stride comes from "
                "the step count, and its trajectory repeats a timestep, so "
                "it cannot be inferred from the timestep alone"
            )
        # The order of the rule is read off the counter, so the counter
        # has to agree with the caller about where in the trajectory we
        # are.  Checking the timestep is the cheapest way to say so.
        if self.counter >= len(self._trajectory):
            raise RuntimeError(
                f"this trajectory has {len(self._trajectory)} steps and all "
                "of them have been taken; call timesteps() again to start a "
                "new sample"
            )
        expected = self._trajectory[self.counter]
        if timestep != expected:
            raise RuntimeError(
                f"expected timestep {expected} at position {self.counter} of "
                f"the trajectory but got {timestep}; PNDM's multistep rule "
                "is chosen by position, so stepping out of order applies the "
                "wrong order of correction"
            )
        previous = timestep - self._stride

        # The repeated time is already in the list, so the only thing
        # the counter decides here is whether the history advances.
        if self.counter == 1:
            previous, timestep = timestep, timestep + self._stride
        else:
            self.ets = self.ets[-3:]
            self.ets.append(model_output)

        if len(self.ets) == 1 and self.counter == 0:
            combined = model_output
            # The opener needs the sample it started from, not the one
            # the repeated step produced — the second evaluation is of
            # the *same* interval, so it must begin where the first did.
            self.cur_sample = latent
        elif len(self.ets) == 1 and self.counter == 1:
            combined = (model_output + self.ets[-1]) / 2.0
            opener, self.cur_sample = self.cur_sample, None
            if opener is None:
                raise RuntimeError(
                    "the second PNDM step needs the sample the first one "
                    "began from, and none was recorded; call reset() before "
                    "reusing a scheduler on a new sequence"
                )
            latent = opener
        elif len(self.ets) == 2:
            combined = (3.0 * self.ets[-1] - self.ets[-2]) / 2.0
        elif len(self.ets) == 3:
            combined = (23.0 * self.ets[-1] - 16.0 * self.ets[-2] + 5.0 * self.ets[-3]) / 12.0
        else:
            combined = (1.0 / 24.0) * (
                55.0 * self.ets[-1] - 59.0 * self.ets[-2] + 37.0 * self.ets[-3] - 9.0 * self.ets[-4]
            )

        self.counter += 1
        return self._ddim.step(combined, timestep, previous, latent, eta=0.0)
