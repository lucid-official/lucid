"""Noise schedulers for diffusion-family generative models.

Mirrors the diffusers-style separation: the *scheduler* owns the forward
process (``add_noise``) and the reverse-process step (``step``), while the
*model* owns the parametrised denoiser.  This keeps sampler swaps cheap —
DDPM vs DDIM vs Karras only requires re-instantiating the scheduler.

Today this module ships:

    * :class:`DiffusionScheduler` — abstract base (contract every scheduler honours)
    * :class:`DDPMScheduler` — Ho et al. (2020) ancestral sampler
"""

from abc import ABC, abstractmethod
from array import array
from collections.abc import Callable
from typing import Self, override

import lucid
from lucid._engine_scope import untraced
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike
from lucid.models._utils._generative import extract_into_tensor, make_beta_schedule
from lucid.models._utils._common import host_floats

__all__ = ["DiffusionScheduler", "DDPMScheduler"]


class DiffusionScheduler(ABC):
    """Abstract base — every diffusion scheduler honours this contract.

    Attributes
    ----------
    num_train_timesteps : int
        Length of the training-time forward process ``T``.
    timesteps : Tensor
        Currently-active inference schedule (descending integer indices into
        the training schedule).  Set by :meth:`set_timesteps`.

    Examples
    --------
    >>> from lucid.models.generative._common._schedulers import (
    ...     DDPMScheduler,
    ...     DiffusionScheduler,
    ... )
    >>> isinstance(DDPMScheduler(num_train_timesteps=10), DiffusionScheduler)
    True

    The interface a sampler is written against — ``add_noise`` for
    training, ``set_timesteps`` and ``step`` for inference — so a model
    does not know which schedule it is being run with.

    The schedule's tables live on one device, :attr:`device`.  They start
    where the scheduler was built and move with :meth:`to`; a scheduler is
    not a :class:`lucid.nn.Module` (it holds no weights, and its tables do
    not belong in a checkpoint), so a model that keeps one forwards its own
    device moves through :meth:`_apply`.
    """

    num_train_timesteps: int
    timesteps: Tensor

    @property
    def device(self) -> DeviceLike:
        """Where the schedule's tables live."""
        return self.timesteps.device

    def to(self, device: DeviceLike) -> Self:
        """Move every schedule table to ``device``; returns ``self``.

        Examples
        --------
        >>> from lucid.models.generative._common._schedulers import DDPMScheduler
        >>> scheduler = DDPMScheduler(num_train_timesteps=10).to("cpu")
        >>> scheduler.alphas_cumprod.device.type
        'cpu'
        """
        self._apply(lambda table: table.to(device))
        return self

    def _apply(self, fn: Callable[[Tensor], Tensor]) -> None:
        """Map every tensor table through ``fn`` — a model's ``_apply`` hook."""
        for name, value in list(vars(self).items()):
            if isinstance(value, Tensor):
                setattr(self, name, fn(value))

    def _follow(self, sample: Tensor) -> None:
        """Move the tables to ``sample``'s device, once, if they are elsewhere.

        A scheduler built on the CPU and handed Metal samples used to copy
        its whole table across on every call; now the first such call moves
        it and the rest find it in place.  The move runs outside any active
        trace: inside, the first traced call recorded a copy of every table
        and a later one fed the moved tables, so the recording depended on
        what the scheduler had served before (LCD-627).
        """
        if self.timesteps.device != sample.device:
            with untraced():
                self.to(sample.device)

    @abstractmethod
    def set_timesteps(self, num_inference_steps: int, *, device: DeviceLike = None) -> None:
        """Populate :attr:`timesteps` for the requested inference schedule.

        Most schedulers sub-sample the training schedule uniformly when
        ``num_inference_steps < num_train_timesteps`` (DDPM canonical)
        or construct a different grid entirely (Karras / Heun).

        Parameters
        ----------
        num_inference_steps : int
            How many denoising steps the sampler will perform.  Must be
            positive and (for sub-sampling schedulers) ``<=
            num_train_timesteps``.
        device : DeviceLike, optional
            Moves the whole schedule — :attr:`timesteps` and every table —
            there, as :meth:`to` does.  ``None`` (the default) leaves it where
            it is.

        Examples
        --------
        >>> from lucid.models.generative._common._schedulers import DDPMScheduler
        >>> scheduler = DDPMScheduler(num_train_timesteps=10)
        >>> scheduler.set_timesteps(4)
        >>> scheduler.timesteps.tolist()
        [6, 4, 2, 0]
        >>> scheduler.set_timesteps(3)  # 3 does not divide 10; still ends at 0
        >>> scheduler.timesteps.tolist()
        [6, 3, 0]
        """

    @abstractmethod
    def add_noise(self, original: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """Forward process: ``q(x_t | x_0) = N(√ᾱ_t · x_0, (1 - ᾱ_t) · I)``."""

    @abstractmethod
    def step(self, model_output: Tensor, timestep: int, sample: Tensor) -> Tensor:
        """One reverse-process step: ``x_t → x_{t-1}`` given the network output."""


class DDPMScheduler(DiffusionScheduler):
    """Ho et al. (2020) ancestral DDPM sampler.

    Reverse process:
        ``x_{t-1} = (1/√α_t) · (x_t - (β_t / √(1-ᾱ_t)) · ε̂_θ(x_t, t)) + σ_t · z``
        where ``σ_t² = β_t`` (Algorithm 2).  Section 3.2 offers two choices;
    this is the *upper* bound — ``fixedlarge`` in the official code, and
    what ``run_cifar.py`` sets.  The lower bound
    ``σ_t² = β̃_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) · β_t`` is the one other
    libraries call "fixed_small".

    Args:
        num_train_timesteps: Length of forward process ``T``.
        beta_start / beta_end / beta_schedule: see
            :func:`make_beta_schedule`.
        prediction_type: ``"epsilon"`` (the noise, default) or ``"sample"``
            (predict x_0 directly).  Both parameterisations are supported.
        device: Where the schedule's tables are built; :meth:`to` moves
            them later, and ``add_noise`` / ``step`` move them to their
            sample's device on first use.

    Notes:
        The scheduler is *stateless w.r.t. weights* — it only owns the noise
        schedule.  Multiple schedulers (DDPM / DDIM / …) can sample from
        the same trained network without re-loading anything.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative._common._schedulers import DDPMScheduler
    >>> scheduler = DDPMScheduler(num_train_timesteps=1000)
    >>> scheduler.betas.shape
    (1000,)

    Sampling need not visit every step it was trained on:
    ``set_timesteps`` picks a subsequence, which is the whole reason a
    thousand-step model can be sampled in ten.

    >>> scheduler.set_timesteps(10)
    >>> scheduler.timesteps.shape
    (10,)

    >>> _ = lucid.manual_seed(0)
    >>> clean = lucid.zeros(1, 3, 8, 8)
    >>> noise = lucid.ones(1, 3, 8, 8)
    >>> scheduler.add_noise(clean, noise, lucid.tensor([500])).shape
    (1, 3, 8, 8)
    """

    betas: Tensor
    alphas: Tensor
    alphas_cumprod: Tensor

    def __init__(
        self,
        num_train_timesteps: int = 1_000,
        *,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
        clip_denoised: bool = True,
        device: DeviceLike = "cpu",
    ) -> None:
        if prediction_type not in ("epsilon", "sample"):
            raise ValueError(
                f"DDPMScheduler.prediction_type must be 'epsilon' or 'sample', "
                f"got {prediction_type!r}"
            )
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_denoised = clip_denoised
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = make_beta_schedule(
            num_train_timesteps, beta_schedule, beta_start=beta_start, beta_end=beta_end
        )
        self.alphas = 1.0 - self.betas
        # ᾱ_t = ∏_{s≤t} α_s — implemented as a Python prefix product because
        # cumulative-product over a 1-D 1000-entry tensor is one-shot cheap.
        a_list = host_floats(self.alphas)
        cum: list[float] = []
        acc = 1.0
        for a in a_list:
            acc *= a
            cum.append(acc)
        self.alphas_cumprod = lucid.tensor(cum)
        # ``step`` takes a Python-int timestep, so its coefficients are read
        # from host floats rather than from a table that may sit on Metal —
        # a device read per step would stall the sampling loop.
        self._alphas_cumprod_host = array("f", cum).tolist()  # float32, as stored
        # add_noise's two coefficients per timestep, computed once rather
        # than on every training step.
        self._signal_scale = self.alphas_cumprod**0.5
        self._noise_scale = (1.0 - self.alphas_cumprod) ** 0.5

        # Inference schedule defaults to the full training one.
        self.timesteps = lucid.arange(num_train_timesteps - 1, -1, -1).long()
        # Spacing between consecutive entries of ``self.timesteps``.  ``step()``
        # needs it: with a sub-sampled schedule the previous timestep is
        # ``t - step_ratio``, not ``t - 1``, and the per-step α/β must be the
        # ratio across that whole interval.
        self._step_ratio = 1
        if device is not None:
            self.to(device)

    @override
    def set_timesteps(self, num_inference_steps: int, *, device: DeviceLike = None) -> None:
        """Uniform-subsample the training schedule into ``num_inference_steps``."""
        if num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {num_inference_steps}")
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_steps ({num_inference_steps}) exceeds "
                f"num_train_timesteps ({self.num_train_timesteps})"
            )
        step = self.num_train_timesteps // num_inference_steps
        # Anchor the grid at 0 and count *up* by ``step``, then reverse:
        # [(n-1)·step, …, step, 0].  Anchoring at T-1 instead leaves the last
        # entry at ``T-1 - (n-1)·step``, which is only 0 when ``n`` divides
        # ``T``; otherwise the chain stops at some t > 0 and hands back a
        # sample that is still noisy — and, because ``prev_t`` is then >= 0,
        # adds one more σ·z on the way out.  Ending at exactly 0 makes the
        # final step noiseless for every ``n``, which is Algorithm 2 line 3
        # ("z = 0 if t == 1").
        idx = [i * step for i in range(num_inference_steps)][::-1]
        self.timesteps = lucid.tensor(idx, device=self.device).long()
        self._step_ratio = step
        if device is not None:
            # The whole schedule moves, not just ``timesteps``: :attr:`device`
            # reads ``timesteps``, so moving it alone reported the new device
            # while ``add_noise`` kept copying CPU tables on every call.
            self.to(device)

    @override
    def add_noise(self, original: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """Forward process at arbitrary timesteps.

        Args:
            original:  ``(B, C, H, W)`` clean images ``x_0``.
            noise:     ``(B, C, H, W)`` Gaussian noise drawn from ``N(0, I)``.
            timesteps: ``(B,)`` int diffusion steps in ``[0, T)``.

        Returns:
            Noisy images ``x_t`` of the same shape and dtype as
            ``original``: the coefficients are cast at use, so the float32
            tables do not widen a float16 batch.
        """
        self._follow(original)
        shape, dtype = original.shape, original.dtype
        sqrt_ab = extract_into_tensor(self._signal_scale, timesteps, shape, dtype=dtype)
        sqrt_one_minus_ab = extract_into_tensor(self._noise_scale, timesteps, shape, dtype=dtype)
        return sqrt_ab * original + sqrt_one_minus_ab * noise

    @override
    def step(self, model_output: Tensor, timestep: int, sample: Tensor) -> Tensor:
        """Single reverse-process step at integer ``timestep``.

        Args:
            model_output: ``(B, C, H, W)`` denoiser prediction.  Interpreted
                per ``self.prediction_type``.
            timestep:     Current integer ``t``.
            sample:       Current noisy sample ``x_t``.

        Returns:
            ``x_{t-1}`` (still noisy except at ``t == 0``, where mean only).
        """
        t = timestep
        # The previous entry of the *inference* schedule, which is ``t - 1``
        # only when the schedule is the untouched training one.  Under
        # sub-sampling the reverse step spans ``step_ratio`` training steps, so
        # α and β for the step are the cumulative-product ratio across the whole
        # interval — using ``self.alphas[t]`` / ``self.betas[t]`` there would
        # under-denoise by the stride factor and return near-pure noise.
        prev_t = t - self._step_ratio
        ab_t = self._alphas_cumprod_host[t]
        ab_prev = self._alphas_cumprod_host[prev_t] if prev_t >= 0 else 1.0

        alpha_t = ab_t / ab_prev
        beta_t = 1.0 - alpha_t

        if self.prediction_type == "epsilon":
            # x_0 reconstruction from predicted noise.
            x0_pred = (sample - (1.0 - ab_t) ** 0.5 * model_output) / (ab_t**0.5)
        else:  # "sample"
            x0_pred = model_output

        if self.clip_denoised:
            # DDPM samples live in [-1, 1]; without this the 1/sqrt(ᾱ_t) factor
            # (≈157 at t=999) lets an early x̂₀ blow the chain up.
            x0_pred = lucid.clip(x0_pred, -1.0, 1.0)

        # Posterior mean q(x_{t-1} | x_t, x_0).
        coef_x0 = (ab_prev**0.5) * beta_t / (1.0 - ab_t)
        coef_xt = (alpha_t**0.5) * (1.0 - ab_prev) / (1.0 - ab_t)
        mean: Tensor = coef_x0 * x0_pred + coef_xt * sample

        if prev_t < 0:
            return mean
        # Add Gaussian noise with variance σ_t² = β_t — the upper-bound
        # ("fixed-large") choice the class docstring describes.
        sigma = beta_t**0.5
        z = lucid.randn(sample.shape, dtype=sample.dtype, device=sample.device.type)
        out: Tensor = mean + sigma * z
        return out
