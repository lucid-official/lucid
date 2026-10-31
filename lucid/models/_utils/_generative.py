"""Generative-model helpers shared across families.

    * Beta-schedule construction (``make_beta_schedule``)        — diffusion
    * Per-batch schedule indexing (``extract_into_tensor``)      — diffusion
    * Variational bound terms (``normal_kl`` /
      ``discretized_gaussian_log_likelihood`` /
      ``diffusion_vlb_term``)                                    — diffusion
    * Diagonal Gaussian KL divergence (``gaussian_kl_divergence``) — VAE
    * Reparameterisation sample (``reparameterize``)              — VAE
    * Flow prior density / sampling (``flow_prior_log_prob`` /
      ``flow_prior_sample``)                                      — flows
    * Activation dispatch (``generative_activation``)             — all
    * Vector-field divergence (``trace_probe`` /
      ``hutchinson_divergence`` / ``exact_divergence``)      — continuous flows

All operations are stateless and side-effect free.
"""

import math
import struct
from collections.abc import Callable, Sequence
from typing import NamedTuple, Protocol, final, overload

import lucid
import lucid.autograd
import lucid.diffeq
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._dtype import _float32_or_wider
from lucid._tensor.tensor import Tensor
from lucid._types import Reduction
from lucid.models._utils._common import host_floats

__all__ = [
    "make_beta_schedule",
    "beta_schedule_floats",
    "make_sigma_schedule",
    "extract_into_tensor",
    "normal_kl",
    "discretized_gaussian_log_likelihood",
    "DiffusionPosterior",
    "diffusion_posterior_constants",
    "diffusion_posterior",
    "diffusion_vlb_term",
    "gaussian_kl_divergence",
    "reparameterize",
    "flow_prior_log_prob",
    "flow_prior_sample",
    "generative_activation",
    "trace_probe",
    "hutchinson_divergence",
    "exact_divergence",
    "check_image_shape",
    "odeint_endpoint",
    "LikelihoodDynamics",
    "cnf_gaussian_log_prob",
]


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion: β / α / cumulative-α schedule
# ─────────────────────────────────────────────────────────────────────────────


def _beta_values(num_steps: int, schedule: str, beta_start: float, beta_end: float) -> list[float]:
    """The schedule's ``β_t`` as Python doubles, before any tensor stores them."""
    if schedule == "linear":
        step = (beta_end - beta_start) / max(num_steps - 1, 1)
        return [beta_start + i * step for i in range(num_steps)]

    if schedule == "scaled_linear":
        # Stable Diffusion's schedule: linear in sqrt(beta), then squared.
        # Between the same endpoints this differs from "linear" only in
        # the middle, which is why it needs its own name — a plot of the
        # two is nearly indistinguishable and the samples are not.
        root_start, root_end = beta_start**0.5, beta_end**0.5
        step = (root_end - root_start) / max(num_steps - 1, 1)
        return [(root_start + i * step) ** 2 for i in range(num_steps)]

    if schedule == "cosine":
        s = 0.008
        # ᾱ schedule in [0, 1] then β_t = clip(1 - ᾱ_t / ᾱ_{t-1}, 0, 0.999)
        f_values: list[float] = [
            math.cos(((i / num_steps) + s) / (1.0 + s) * math.pi / 2.0) ** 2
            for i in range(num_steps + 1)
        ]
        alpha_bar = [v / f_values[0] for v in f_values]
        betas: list[float] = []
        for i in range(1, num_steps + 1):
            b = 1.0 - alpha_bar[i] / max(alpha_bar[i - 1], 1e-8)
            betas.append(min(max(b, 0.0), 0.999))
        return betas

    raise ValueError(f"Unsupported beta schedule {schedule!r}")


def make_beta_schedule(
    num_steps: int,
    schedule: str = "linear",
    *,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    device: str = "cpu",
) -> Tensor:
    """Build the forward-process noise schedule ``β_1, …, β_T``.

    Args:
        num_steps:  Number of diffusion timesteps ``T``.
        schedule:   ``"linear"`` (Ho 2020), ``"scaled_linear"`` (Rombach
                    2022) or ``"cosine"`` (Nichol-Dhariwal 2021).  Cosine uses the formula from Improved DDPM §3:
                    ``ᾱ_t = cos((t/T + s) / (1 + s) · π/2)² with s = 0.008``.
        beta_start: Start of the linear schedule (ignored for cosine).
        beta_end:   End of the linear schedule (ignored for cosine).
        device:     Where to allocate the resulting buffer.

    Returns:
        ``(num_steps,)`` float tensor.

    Examples
    --------
    >>> from lucid.models._utils._generative import make_beta_schedule
    >>> [round(v, 5) for v in make_beta_schedule(5).tolist()]
    [0.0001, 0.00508, 0.01005, 0.01503, 0.02]

    The noise added per step, rising across the schedule: almost nothing
    at the start, where the signal is still intact and worth protecting,
    and most of it at the end.
    """
    return lucid.tensor(_beta_values(num_steps, schedule, beta_start, beta_end), device=device)


def _stored_precision(values: list[float]) -> list[float]:
    """``values`` as a tensor of the default dtype holds them, widened back to doubles."""
    if lucid.get_default_dtype() == lucid.float64:
        return values
    return [struct.unpack("f", struct.pack("f", v))[0] for v in values]


def beta_schedule_floats(
    num_steps: int,
    schedule: str = "linear",
    *,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> list[float]:
    """The values :func:`make_beta_schedule` stores, as host Python floats.

    A model that derives constants from the schedule at construction reads
    them here rather than from the tensor: reading a tensor's elements on
    the host is impossible while the model is being built without storage
    (:func:`~lucid.nn._shadow.shadow_alloc`), and it is a device round trip
    otherwise.  Each value is rounded the way the tensor rounds it, so
    anything computed from the list matches what it would have been computed
    from the tensor, bit for bit.

    Args:
        num_steps:  Number of diffusion timesteps ``T``.
        schedule:   Schedule name, as for :func:`make_beta_schedule`.
        beta_start: Start of the linear schedule (ignored for cosine).
        beta_end:   End of the linear schedule (ignored for cosine).

    Returns:
        ``num_steps`` Python floats.

    Examples
    --------
    >>> from lucid.models._utils._generative import (
    ...     beta_schedule_floats,
    ...     make_beta_schedule,
    ... )
    >>> beta_schedule_floats(5) == make_beta_schedule(5).tolist()
    True

    Equal to the tensor's own elements, which a double-precision list of
    the same formula is not: ``0.0001`` is not a float32.
    """
    return _stored_precision(_beta_values(num_steps, schedule, beta_start, beta_end))


def extract_into_tensor(
    arr: Tensor,
    timesteps: Tensor,
    broadcast_shape: tuple[int, ...],
    *,
    dtype: lucid.dtype | None = None,
) -> Tensor:
    """Gather entries of a 1-D schedule at given timesteps and reshape for
    broadcasting against an image batch.

    Standard diffusion idiom — used by every scheduler ``add_noise`` /
    ``step`` to lift the per-step scalar ``α_t``, ``β_t`` … into the same
    rank as the noisy image so they multiply elementwise.

    Args:
        arr:             ``(T,)`` schedule tensor.
        timesteps:       ``(B,)`` int timesteps to gather.
        broadcast_shape: Target rank.  Usually ``(B, C, H, W)``.
        dtype:           Dtype of the gathered coefficients; ``None``
                         keeps ``arr``'s.  A scheduler passes its sample's:
                         its tables are kept in float32 and cast at use,
                         since a float32 coefficient widens the float16
                         batch it multiplies.

    Returns:
        ``(B, 1, 1, …)`` (rank == ``len(broadcast_shape)``) so it broadcasts
        elementwise against an ``(B, C, H, W)`` image.  It is on
        ``timesteps``' device, which is the batch's, whichever device
        ``arr`` was built on.

    Raises:
        IndexError: On CPU, for a timestep outside ``[-T, T)``.  On Metal
            the gather reads NaN there instead, so the loss turns NaN —
            the index is not read back to check it, which would stall
            every step.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._generative import extract_into_tensor
    >>> alphas = lucid.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> steps = lucid.tensor([0, 2])
    >>> picked = extract_into_tensor(alphas, steps, (2, 1, 1))
    >>> picked.shape
    (2, 1, 1)
    >>> [round(v, 4) for v in picked.reshape(-1).tolist()]
    [0.1, 0.3]

    One coefficient per item of the batch, shaped so it broadcasts
    against images — each sample in a diffusion batch is usually at a
    different timestep, so a scalar per step is not enough.  A gather
    rather than a read of each timestep on the host: a compiled or
    exported training step then takes the timesteps from each call's
    inputs instead of replaying the ones it was traced with.
    """
    index = timesteps.long().reshape(-1)
    device = index.device.type
    # Schedules are built once on the CPU and are not module state, so a
    # model moved to Metal still holds them there.
    table = arr if arr.device.type == device else arr.to(device)
    picked = lucid.take_along_dim(table, index, dim=0)
    if dtype is not None:
        # Cast the B gathered entries, not the T-entry table.
        picked = picked.to(dtype)
    trailing = [1] * max(len(broadcast_shape) - 1, 0)
    return picked.reshape(int(index.shape[0]), *trailing)


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion: variational bound terms
# ─────────────────────────────────────────────────────────────────────────────


def normal_kl(mean1: Tensor, logvar1: Tensor, mean2: Tensor, logvar2: Tensor) -> Tensor:
    """Elementwise KL divergence between two diagonal Gaussians, in nats.

    Unlike :func:`gaussian_kl_divergence` — which fixes the second
    distribution at ``N(0, I)`` and sums over the feature axis — this keeps
    both distributions free and returns one value *per element*, which is
    what the diffusion variational bound needs before it reduces.

    Args:
        mean1:   Mean of the first Gaussian.
        logvar1: Log-variance of the first Gaussian.
        mean2:   Mean of the second.  Broadcast against the first.
        logvar2: Log-variance of the second.  Broadcast against the first.

    Returns:
        Elementwise ``KL(N(mean1, exp(logvar1)) ‖ N(mean2, exp(logvar2)))``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._generative import normal_kl
    >>> zeros = lucid.zeros(2)
    >>> normal_kl(zeros, zeros, zeros, zeros).tolist()
    [0.0, 0.0]

    Zero between a distribution and itself, elementwise rather than
    summed, so the caller chooses what to reduce over.
    """
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + (logvar1 - logvar2).exp()
        + ((mean1 - mean2) ** 2) * (-logvar2).exp()
    )


def _approx_standard_normal_cdf(x: Tensor) -> Tensor:
    """Tanh approximation of the standard normal CDF.

    The reference implementation uses this rather than ``erf``; keeping the
    same approximation keeps our bound numerically comparable to published
    bits/dim figures.
    """
    return 0.5 * (1.0 + (math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)).tanh())


def discretized_gaussian_log_likelihood(x: Tensor, *, means: Tensor, log_scales: Tensor) -> Tensor:
    """Log-likelihood of 8-bit data under a Gaussian, discretised to bins.

    The final reverse step ``p(x_0 | x_1)`` is a *discrete* likelihood: the
    image lives on a 256-value grid, so the density is integrated over each
    pixel's bin rather than evaluated at a point.  Evaluating the continuous
    density instead makes the last term of the bound meaningless (it can go
    arbitrarily positive as the variance shrinks).

    Args:
        x:          Target, assumed rescaled to ``[-1, 1]``.
        means:      Predicted bin centres.
        log_scales: Predicted log standard deviations.

    Returns:
        Elementwise log probability, in nats.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._generative import (
    ...     discretized_gaussian_log_likelihood,
    ... )
    >>> x = lucid.zeros(2, 3)
    >>> discretized_gaussian_log_likelihood(
    ...     x, means=x, log_scales=x
    ... ).shape
    (2, 3)

    Discretized because the last step of a diffusion model lands on
    8-bit pixels: the likelihood integrates over the bin a value falls
    in rather than reading a density at a point.
    """
    centered = x - means
    inv_stdv = (-log_scales).exp()
    # Half a bin either side: 1/255 in [-1, 1] coordinates.
    plus_in = inv_stdv * (centered + 1.0 / 255.0)
    min_in = inv_stdv * (centered - 1.0 / 255.0)
    cdf_plus = _approx_standard_normal_cdf(plus_in)
    cdf_min = _approx_standard_normal_cdf(min_in)

    log_cdf_plus = lucid.log(cdf_plus.clip(1e-12, None))
    log_one_minus_cdf_min = lucid.log((1.0 - cdf_min).clip(1e-12, None))
    log_cdf_delta = lucid.log((cdf_plus - cdf_min).clip(1e-12, None))

    # The two edge bins are open-ended — everything below -0.999 falls in the
    # first bin, everything above 0.999 in the last.
    return lucid.where(
        x < -0.999, log_cdf_plus, lucid.where(x > 0.999, log_one_minus_cdf_min, log_cdf_delta)
    )


def _mean_flat(x: Tensor) -> Tensor:
    """Mean over every axis except the batch axis."""
    out = x
    while out.ndim > 1:
        out = out.mean(dim=-1)
    return out


class DiffusionPosterior(NamedTuple):
    r"""Per-timestep constants of :math:`q(x_{t-1} \mid x_t, x_0)`.

    Built once per noise schedule by :func:`diffusion_posterior_constants`
    and reused every step — the constants depend only on ``betas``.

    Attributes:
        mean_coef_start: ``(T,)`` weight on :math:`x_0` in the mean.
        mean_coef_t:     ``(T,)`` weight on :math:`x_t` in the mean.
        log_variance:    ``(T,)`` :math:`\log \tilde\beta_t`, with the ``t=0``
            entry substituted (see :func:`diffusion_posterior_constants`).

    Examples
    --------
    >>> from lucid.models._utils._generative import (
    ...     diffusion_posterior_constants,
    ...     make_beta_schedule,
    ... )
    >>> posterior = diffusion_posterior_constants(make_beta_schedule(5))
    >>> posterior._fields
    ('mean_coef_start', 'mean_coef_t', 'log_variance')
    >>> posterior.mean_coef_start.shape
    (5,)

    One entry per timestep, computed once for a schedule: a sampler
    reads these thousands of times and none of them depends on the data.
    """

    mean_coef_start: Tensor
    mean_coef_t: Tensor
    log_variance: Tensor


def diffusion_posterior_constants(
    betas: Tensor | Sequence[float], *, device: str = "cpu"
) -> DiffusionPosterior:
    r"""Precompute the closed-form posterior constants (Ho 2020 Eq. 6–7):

    .. math::

        \tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t}\beta_t,
        \qquad
        \tilde\mu_t = \frac{\beta_t\sqrt{\bar\alpha_{t-1}}}{1 - \bar\alpha_t}x_0
                    + \frac{(1 - \bar\alpha_{t-1})\sqrt{\alpha_t}}
                           {1 - \bar\alpha_t}x_t.

    Every quotient here divides by :math:`1 - \bar\alpha_t`, which at small
    ``t`` is a difference of two nearly equal numbers.  Taking
    :math:`\bar\alpha` from a float32 tensor loses most of that difference's
    significant digits — at ``t=0`` with ``beta_start=1e-4`` the resulting
    mean is off by ~5e-4 instead of matching :math:`x_0` exactly.  So the
    cumulative product is accumulated here in Python floats (doubles) from
    ``betas`` alone, and only the finished coefficients are cast down.

    Args:
        betas:  ``(T,)`` forward-process variances, as a tensor or as host
            floats (see :func:`beta_schedule_floats`, the form a model can
            use while it is built without storage).
        device: Where to allocate the resulting ``(T,)`` buffers.

    Returns:
        A :class:`DiffusionPosterior` of three ``(T,)`` tensors.

    Examples
    --------
    >>> from lucid.models._utils._generative import (
    ...     diffusion_posterior_constants,
    ...     make_beta_schedule,
    ... )
    >>> constants = diffusion_posterior_constants(make_beta_schedule(5))
    >>> type(constants).__name__
    'DiffusionPosterior'

    Precomputed once for a schedule rather than per step: every one is a
    function of the betas alone, and a sampler reads them thousands of
    times.
    """
    beta_l = host_floats(betas) if isinstance(betas, Tensor) else [float(b) for b in betas]
    T = len(beta_l)

    ab_l: list[float] = []
    acc = 1.0
    for b in beta_l:
        acc *= 1.0 - b
        ab_l.append(acc)
    ab_prev_l = [1.0] + ab_l[:-1]

    coef1_l: list[float] = []
    coef2_l: list[float] = []
    post_var_l: list[float] = []
    for i in range(T):
        one_minus_ab = 1.0 - ab_l[i]
        post_var_l.append(beta_l[i] * (1.0 - ab_prev_l[i]) / one_minus_ab)
        coef1_l.append(beta_l[i] * math.sqrt(ab_prev_l[i]) / one_minus_ab)
        coef2_l.append((1.0 - ab_prev_l[i]) * math.sqrt(1.0 - beta_l[i]) / one_minus_ab)

    # β̃_0 is exactly 0 — the posterior at t=0 is a point mass, so its log is
    # -inf.  Substitute β̃_1 for that one entry, as the reference does; a
    # t=0 sample takes the decoder-NLL branch anyway, where the value only
    # sets the likelihood's scale.
    if T > 1:
        post_var_l[0] = post_var_l[1]

    return DiffusionPosterior(
        mean_coef_start=lucid.tensor(coef1_l, device=device),
        mean_coef_t=lucid.tensor(coef2_l, device=device),
        log_variance=lucid.tensor([math.log(v) for v in post_var_l], device=device),
    )


def diffusion_posterior(
    *, x_start: Tensor, x_t: Tensor, t: Tensor, posterior: DiffusionPosterior
) -> tuple[Tensor, Tensor]:
    r"""Mean and log-variance of :math:`q(x_{t-1} \mid x_t, x_0)` at ``t``.

    Args:
        x_start:   ``(B, C, H, W)`` clean data :math:`x_0`.
        x_t:       ``(B, C, H, W)`` noised sample at step ``t``.
        t:         ``(B,)`` integer timesteps.
        posterior: Schedule constants from
            :func:`diffusion_posterior_constants`.

    Returns:
        ``(mean, log_variance)``, both broadcast against ``x_start``'s rank.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._generative import (
    ...     diffusion_posterior,
    ...     diffusion_posterior_constants,
    ...     make_beta_schedule,
    ... )
    >>> posterior = diffusion_posterior_constants(make_beta_schedule(5))
    >>> x = lucid.zeros(2, 3, 8, 8)
    >>> mean, log_variance = diffusion_posterior(
    ...     x_start=x, x_t=x, t=lucid.tensor([0, 2]), posterior=posterior
    ... )
    >>> mean.shape, log_variance.shape
    ((2, 3, 8, 8), (2, 1, 1, 1))

    The mean follows the image and the variance does not: it depends on
    the timestep alone, which is why it comes back shaped to broadcast
    rather than filled in.
    """
    shape = tuple(int(s) for s in x_start.shape)
    c1 = extract_into_tensor(posterior.mean_coef_start, t, shape)
    c2 = extract_into_tensor(posterior.mean_coef_t, t, shape)
    lv = extract_into_tensor(posterior.log_variance, t, shape)
    return c1 * x_start + c2 * x_t, lv


def diffusion_vlb_term(
    *,
    x_start: Tensor,
    x_t: Tensor,
    t: Tensor,
    model_mean: Tensor,
    model_log_variance: Tensor,
    posterior: DiffusionPosterior,
) -> Tensor:
    r"""One term :math:`L_{t-1}` of the diffusion variational bound, in bits.

    For :math:`t > 0` this is
    :math:`\mathrm{KL}(q(x_{t-1} \mid x_t, x_0) \,\|\, p_\theta(x_{t-1} \mid x_t))`;
    at :math:`t = 0` the KL is replaced by the discretised decoder
    negative log-likelihood :math:`-\log p_\theta(x_0 \mid x_1)`, because
    there is no ``x_{-1}`` to compare against.

    Args:
        x_start:            ``(B, C, H, W)`` clean data :math:`x_0`.
        x_t:                ``(B, C, H, W)`` noised sample at step ``t``.
        t:                  ``(B,)`` integer timesteps.
        model_mean:         Predicted posterior mean of ``p_θ``.
        model_log_variance: Predicted posterior log-variance of ``p_θ``.
        posterior:          Schedule constants from
            :func:`diffusion_posterior_constants`.

    Returns:
        ``(B,)`` per-sample bound term, divided by ``log 2`` so the units are
        bits rather than nats.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._generative import (
    ...     diffusion_posterior_constants,
    ...     diffusion_vlb_term,
    ...     make_beta_schedule,
    ... )
    >>> posterior = diffusion_posterior_constants(make_beta_schedule(5))
    >>> x = lucid.zeros(2, 3, 8, 8)
    >>> diffusion_vlb_term(
    ...     x_start=x,
    ...     x_t=x,
    ...     t=lucid.tensor([0, 2]),
    ...     model_mean=x,
    ...     model_log_variance=lucid.zeros(2, 1, 1, 1),
    ...     posterior=posterior,
    ... ).shape
    (2,)

    One number per sample in nats. This is the term that makes a learned
    variance worth anything — the simple noise-prediction loss ignores
    the variance entirely, so a model that learns one needs this to
    supervise it.
    """
    true_mean, true_log_var = diffusion_posterior(
        x_start=x_start, x_t=x_t, t=t, posterior=posterior
    )

    kl = normal_kl(true_mean, true_log_var, model_mean, model_log_variance)
    kl = _mean_flat(kl) / math.log(2.0)

    decoder_nll = -discretized_gaussian_log_likelihood(
        x_start, means=model_mean, log_scales=0.5 * model_log_variance
    )
    decoder_nll = _mean_flat(decoder_nll) / math.log(2.0)

    # Both branches are already (B,); ``t`` selects between them per sample.
    return lucid.where(t == 0, decoder_nll, kl)


# ─────────────────────────────────────────────────────────────────────────────
# VAE: Gaussian KL + reparameterisation trick
# ─────────────────────────────────────────────────────────────────────────────


def gaussian_kl_divergence(mu: Tensor, logvar: Tensor, *, reduction: Reduction = "mean") -> Tensor:
    """KL divergence ``KL(N(μ, σ²) ‖ N(0, I))`` of a diagonal Gaussian.

    Closed form: ``½ Σ (μ² + σ² - log σ² - 1)``.

    Args:
        mu:        ``(B, D)`` posterior means.
        logvar:    ``(B, D)`` posterior log-variances.
        reduction: ``"none"`` (per-sample), ``"sum"`` (over batch), or
                   ``"mean"`` (default — averaged over batch dim).

    Returns:
        Scalar (or ``(B,)`` if ``reduction="none"``).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._generative import gaussian_kl_divergence
    >>> mu, logvar = lucid.zeros(2, 3), lucid.zeros(2, 3)
    >>> float(gaussian_kl_divergence(mu, logvar).item())
    0.0

    Against the standard normal, so zero mean and unit variance — which
    ``logvar=0`` is — costs nothing. This is the term that pulls a VAE's
    posterior toward the prior.
    """
    var = logvar.exp()
    per_dim = 0.5 * (mu * mu + var - logvar - 1.0)  # (B, D)
    per_sample = per_dim.sum(dim=-1)  # (B,)
    if reduction == "none":
        return per_sample
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"Unsupported reduction {reduction!r}")


def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    r"""VAE reparameterisation trick — sample :math:`z = \mu + \sigma \cdot \epsilon`.

    Standard reparameterisation (Kingma & Welling, 2014) that pushes
    the stochasticity into an auxiliary noise term :math:`\epsilon`
    so gradients flow through :math:`\mu` and :math:`\log\sigma^2`
    deterministically.  ``logvar`` rather than ``sigma`` is taken as
    input so the network can output an unconstrained real (the
    ``exp`` enforces positivity).

    Parameters
    ----------
    mu : Tensor
        Posterior mean :math:`\mu`, any shape.
    logvar : Tensor
        Per-element log-variance :math:`\log \sigma^2`; same shape as
        ``mu``.

    Returns
    -------
    Tensor
        A sample :math:`z` with the same shape and dtype as ``mu``.
        Backward propagates through ``mu`` / ``logvar``; the
        :math:`\epsilon` noise term is treated as a constant.

    References
    ----------
    .. [1] Kingma & Welling, *Auto-Encoding Variational Bayes*, ICLR 2014.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._generative import reparameterize
    >>> _ = lucid.manual_seed(0)
    >>> mu, logvar = lucid.zeros(2, 3), lucid.zeros(2, 3)
    >>> reparameterize(mu, logvar).shape
    (2, 3)

    ``mu + sigma * eps``: the sampling happens in ``eps``, which carries
    no gradient, so the path back to ``mu`` and ``logvar`` stays
    differentiable. Sampling the distribution directly would not.
    """
    std = (0.5 * logvar).exp()
    eps = lucid.randn(mu.shape, dtype=mu.dtype, device=mu.device)
    return mu + std * eps


# ─────────────────────────────────────────────────────────────────────────────
# Normalizing flows: factorised base distribution
# ─────────────────────────────────────────────────────────────────────────────


def flow_prior_log_prob(name: str, h: Tensor) -> Tensor:
    r"""Elementwise log-density of a flow's factorised prior :math:`p_H`.

    A normalizing flow scores data through
    :math:`\log p_X(x) = \log p_H(f(x)) + \log|\det \partial f / \partial x|`,
    so the prior only ever enters as a per-dimension log-density that the
    caller sums over the latent axis.

    Parameters
    ----------
    name : str
        ``"logistic"`` — standard logistic, the Dinh et al. (2014) default:
        :math:`\log p(h) = -\log(1 + e^{h}) - \log(1 + e^{-h})`, written
        with softplus for numerical stability.  ``"gaussian"`` — standard
        normal: :math:`\log p(h) = -\tfrac{1}{2}(h^2 + \log 2\pi)`.
    h : Tensor
        Latent values, any shape.

    Returns
    -------
    Tensor
        Same shape as ``h``.

    Raises
    ------
    ValueError
        If ``name`` is not a supported prior.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._generative import flow_prior_log_prob
    >>> flow_prior_log_prob("gaussian", lucid.zeros(2, 3)).shape
    (2, 3)

    Elementwise, not summed — a flow adds the log-determinant to this
    before reducing, and the two have to line up first.
    """
    if name == "logistic":
        return -(F.softplus(h) + F.softplus(-h))
    if name == "gaussian":
        return -0.5 * (h * h + math.log(2.0 * math.pi))
    raise ValueError(f"Unsupported flow prior {name!r}")


def flow_prior_sample(
    name: str, shape: tuple[int, ...], *, device: str = "cpu", dtype: lucid.dtype | None = None
) -> Tensor:
    r"""Draw i.i.d. samples from a flow's factorised prior :math:`p_H`.

    Flows generate by ancestral sampling — :math:`h \sim p_H`, then
    :math:`x = f^{-1}(h)`.

    Parameters
    ----------
    name : str
        ``"logistic"`` (inverse-CDF transform of a uniform sample,
        :math:`h = \log u - \log(1 - u)`) or ``"gaussian"``.
    shape : tuple[int, ...]
        Shape of the requested sample block.
    device : str, optional
        Placement of the result.  Default ``"cpu"``.
    dtype : lucid.dtype or None, optional
        Dtype of the result — the flow's own, so a float16 flow decodes a
        float16 draw.  ``None`` is the default dtype.

    Returns
    -------
    Tensor
        ``shape``-shaped sample.

    Raises
    ------
    ValueError
        If ``name`` is not a supported prior.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._generative import flow_prior_sample
    >>> flow_prior_sample("gaussian", (2, 3)).shape
    (2, 3)
    >>> flow_prior_sample("logistic", (2, 3), dtype=lucid.float16).dtype
    lucid.float16

    ``"gaussian"`` and ``"logistic"`` are the two a flow can be given;
    anything else is refused rather than silently defaulted.
    """
    out = lucid.get_default_dtype() if dtype is None else dtype
    if name == "logistic":
        # Clip away the open-interval endpoints — the inverse CDF diverges
        # at 0 and 1 and ``rand`` samples the half-open [0, 1).  Drawn and
        # transformed in float32 at least: float16 rounds 1 - 1e-7 up to 1,
        # where ``log(1 - u)`` is -inf.
        eps = 1e-7
        u = lucid.rand(shape, dtype=_float32_or_wider(out), device=device)
        u = u.clip(eps, 1.0 - eps)
        return (u.log() - (1.0 - u).log()).to(out)
    if name == "gaussian":
        return lucid.randn(shape, dtype=out, device=device)
    raise ValueError(f"Unsupported flow prior {name!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Activation dispatch (mirrors text_activation pattern)
# ─────────────────────────────────────────────────────────────────────────────


def generative_activation(name: str, x: Tensor) -> Tensor:
    """Apply the activation referenced by a :data:`GenerativeActivation` literal.

    Args:
        name: One of ``"silu"`` / ``"swish"`` / ``"relu"`` / ``"gelu"`` /
            ``"elu"``.
        x:    Input tensor (elementwise).

    Returns:
        Same shape as ``x``.

    Raises:
        ValueError: If ``name`` is not a supported alias.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._generative import generative_activation
    >>> generative_activation("relu", lucid.tensor([-1.0, 2.0])).tolist()
    [0.0, 2.0]
    """
    if name in ("silu", "swish"):
        return F.silu(x)
    if name == "relu":
        return F.relu(x)
    if name == "gelu":
        return F.gelu(x, approximate="none")
    if name == "elu":
        return F.elu(x)
    raise ValueError(f"Unsupported generative activation {name!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Score-based: geometric sigma schedule
# ─────────────────────────────────────────────────────────────────────────────


def make_sigma_schedule(
    num_noise_levels: int, *, sigma_max: float, sigma_min: float, device: str = "cpu"
) -> Tensor:
    """Geometric noise schedule for score-based models (Song & Ermon, 2019).

    Returns ``(L,)`` tensor ``σ_1 > σ_2 > … > σ_L`` with

        σ_i = σ_max · (σ_min / σ_max) ** (i / (L - 1))    for i = 0..L-1

    so ``σ_0 == σ_max`` and ``σ_{L-1} == σ_min``.  Geometric (log-uniform)
    spacing — the choice every NCSN / NCSNv2 / NCSN++ variant uses.

    Args:
        num_noise_levels: Number of levels ``L`` (paper L=10 originally;
            NCSNv2 found large L ∈ [200, 500] works better).
        sigma_max: Largest noise σ_1.
        sigma_min: Smallest noise σ_L.
        device:    Where to allocate the buffer.

    Returns:
        ``(num_noise_levels,)`` float tensor in *descending* order.

    Examples
    --------
    >>> from lucid.models._utils._generative import make_sigma_schedule
    >>> sigmas = make_sigma_schedule(4, sigma_max=1.0, sigma_min=0.1)
    >>> [round(v, 4) for v in sigmas.tolist()]
    [1.0, 0.4642, 0.2154, 0.1]

    Geometric rather than linear, so each level is the same factor from
    the next — score matching wants the noise levels evenly spaced in
    log, not in magnitude.
    """
    if num_noise_levels <= 0:
        raise ValueError(f"num_noise_levels must be positive, got {num_noise_levels}")
    if not 0.0 < sigma_min < sigma_max:
        raise ValueError(f"require 0 < sigma_min < sigma_max, got min={sigma_min}, max={sigma_max}")
    if num_noise_levels == 1:
        return lucid.tensor([sigma_max], device=device)

    ratio = sigma_min / sigma_max
    sigmas = [sigma_max * (ratio ** (i / (num_noise_levels - 1))) for i in range(num_noise_levels)]
    return lucid.tensor(sigmas, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Continuous flows: divergence of a vector field
# ─────────────────────────────────────────────────────────────────────────────


def trace_probe(like: Tensor, noise: str) -> Tensor:
    r"""Draw a Hutchinson probe vector shaped like ``like``.

    The estimator :math:`\operatorname{tr}(A) =
    \mathbb{E}[\boldsymbol{\epsilon}^{\mathsf{T}} A \boldsymbol{\epsilon}]`
    is unbiased for any :math:`\boldsymbol{\epsilon}` with zero mean and
    identity covariance.  Both choices here qualify; Rademacher has the
    lower variance of the two, measurably so.

    Parameters
    ----------
    like : Tensor
        Shape and device to match.
    noise : {"rademacher", "gaussian"}
        Which distribution to draw from.

    Returns
    -------
    Tensor
        Probe vector of the same shape as ``like``.

    Raises
    ------
    ValueError
        If ``noise`` names neither distribution.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._generative import trace_probe
    >>> trace_probe(lucid.zeros(2, 3), "gaussian").shape
    (2, 3)

    The random vector Hutchinson's estimator contracts the Jacobian
    against. Its shape follows the state, which is why it takes a tensor
    to imitate rather than a shape.
    """
    if noise == "rademacher":
        # sign() of a centred uniform: ±1 with equal probability.
        draw = lucid.rand(like.shape, dtype=like.dtype, device=like.device)
        return lucid.sign(draw - 0.5)
    if noise == "gaussian":
        return lucid.randn(like.shape, dtype=like.dtype, device=like.device)
    raise ValueError(f"trace_probe: noise must be 'rademacher' or 'gaussian', got {noise!r}")


def hutchinson_divergence(
    velocity: Tensor, state: Tensor, probe: Tensor, *, create_graph: bool = True
) -> Tensor:
    r"""Unbiased estimate of :math:`\operatorname{tr}(\partial v / \partial x)`.

    One vector-Jacobian product regardless of the dimension, which is what
    makes a continuous flow's log-density affordable at image scale.

    Parameters
    ----------
    velocity : Tensor
        ``(B, D)`` field value, differentiably connected to ``state``.
    state : Tensor
        ``(B, D)`` point the field was evaluated at; must require grad.
    probe : Tensor
        ``(B, D)`` probe held **fixed for the whole solve** — see
        :func:`trace_probe`.  Redrawing it per step makes the right-hand
        side stochastic, and an adaptive solver cannot converge on a
        function that changes under it.
    create_graph : bool, default=True
        Keep the estimate differentiable.  Required when the divergence
        is inside a training objective — a flow trained by maximum
        likelihood differentiates through it, so the field's own ops must
        supply second derivatives.  Set ``False`` when the density is only
        being *reported*: the estimate then costs one ordinary backward
        pass and places no second-derivative demand on the field, which is
        what lets a convolutional field be scored at all.

    Returns
    -------
    Tensor
        ``(B,)`` estimate; differentiable only when ``create_graph``.

    Notes
    -----
    Reference: Grathwohl et al., *"FFJORD: Free-Form Continuous Dynamics
    for Scalable Reversible Generative Models"*, ICLR, 2019
    (arXiv:1810.01367), §3.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._generative import (
    ...     hutchinson_divergence,
    ...     trace_probe,
    ... )
    >>> state = lucid.randn(2, 3, requires_grad=True)
    >>> velocity = state * 2.0
    >>> probe = trace_probe(state, "gaussian")
    >>> hutchinson_divergence(velocity, state, probe).shape
    (2,)

    ``state`` has to carry a live graph — the divergence is read out of
    the Jacobian by differentiating, so a detached tensor has nothing to
    differentiate through.

    One probe gives an unbiased estimate of the trace and costs a single
    backward pass, whatever the dimension. That is the trade this exists
    to make: the exact answer needs one pass per dimension.
    """
    (row,) = lucid.autograd.grad(
        velocity, [state], grad_outputs=[probe], create_graph=create_graph, retain_graph=True
    )
    if row is None:
        raise RuntimeError(
            "hutchinson_divergence: the field does not depend on the state; "
            "the divergence is undefined"
        )
    return (row * probe).sum(dim=-1)


def exact_divergence(
    velocity: Tensor, state: Tensor, seeds: list[Tensor], *, create_graph: bool = True
) -> Tensor:
    r"""Exact :math:`\operatorname{tr}(\partial v / \partial x)`, one VJP per dimension.

    Seeding the backward pass with :math:`\mathbf{e}_i` returns row
    :math:`i` of the Jacobian, whose :math:`i`-th entry is the diagonal
    element wanted.  Linear in the dimension, which is why it is only
    affordable at the scale the original continuous-flow experiments run
    at.

    Parameters
    ----------
    velocity : Tensor
        ``(B, D)`` field value, differentiably connected to ``state``.
    state : Tensor
        ``(B, D)`` point the field was evaluated at; must require grad.
    seeds : list of Tensor
        ``D`` tensors of shape ``(B, D)``, the identity rows broadcast over
        the batch.  ``grad_outputs`` has to match the output shape exactly,
        so the rows cannot simply be broadcast in place.
    create_graph : bool, default=True
        Keep the trace differentiable — see :func:`hutchinson_divergence`.

    Returns
    -------
    Tensor
        ``(B,)`` exact trace; differentiable only when ``create_graph``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._generative import exact_divergence
    >>> state = lucid.randn(2, 3, requires_grad=True)
    >>> velocity = state * 2.0
    >>> seeds = [lucid.zeros(2, 3) for _ in range(3)]
    >>> exact_divergence(velocity, state, seeds).shape
    (2,)

    One seed per dimension, and that is the cost: three here, but a
    backward pass per channel of an image is not something a sampler can
    afford. Use it to check the estimator, not in the loop.
    """
    total: Tensor | None = None
    for index, seed in enumerate(seeds):
        (row,) = lucid.autograd.grad(
            velocity, [state], grad_outputs=[seed], create_graph=create_graph, retain_graph=True
        )
        if row is None:
            raise RuntimeError(
                "exact_divergence: the field does not depend on the state; "
                "the divergence is undefined"
            )
        piece = row[:, index]
        total = piece if total is None else total + piece
    if total is None:
        raise ValueError("exact_divergence: seeds must not be empty")
    return total


def resolve_generation_device(model: nn.Module, device: str | None) -> str:
    """Pick the device a sampler should draw its latents on.

    ``generate()`` used to hard-default this to ``"cpu"``, so moving a model
    to an accelerator and sampling raised ``DeviceMismatch`` — the weights
    were on one device and the freshly drawn noise on another.  Defaulting to
    the model's own device makes ``model.to("metal").generate(...)`` work,
    while an explicit argument still wins.

    Parameters
    ----------
    model : nn.Module
        Any module; its first parameter decides the device.
    device : str or None
        Caller's explicit choice, or ``None`` to follow the model.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.models._utils._generative import resolve_generation_device
    >>> resolve_generation_device(nn.Linear(2, 2), None)
    'cpu'

    ``None`` means "wherever the model already is", which is what keeps
    sampling from quietly moving a model between devices mid-run.
    """
    if device is not None:
        return device
    for param in model.parameters():
        return str(param.device.type)
    return "cpu"


def model_dtype(model: nn.Module) -> lucid.dtype:
    """The floating dtype a model computes in: its first floating parameter's.

    The dtype twin of :func:`resolve_generation_device`.  A tensor a model
    builds for itself — a recurrent start state, a latent — has to be as
    wide as the weights it meets: a float32 start state ran a float16 world
    model's whole unroll in float32.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.models._utils._generative import model_dtype
    >>> model_dtype(nn.Linear(2, 2).to(lucid.float16))
    lucid.float16
    """
    for param in model.parameters():
        if param.is_floating_point():
            return param.dtype
    return lucid.get_default_dtype()


def start_noise(
    model: nn.Module, shape: tuple[int, ...], device: str | None, *, prior: str = "gaussian"
) -> Tensor:
    """The draw a sampler starts from, on ``model``'s device and in its dtype.

    A float32 start ran a float16 model's whole reverse chain in float32,
    as a CPU one once put a Metal model's on the wrong device.

    Parameters
    ----------
    model : nn.Module
        The sampling model; its first parameter decides the device and its
        first floating parameter the dtype.
    shape : tuple[int, ...]
        Shape of the draw.
    device : str or None
        Explicit device, which wins; ``None`` follows ``model``.
    prior : str, default="gaussian"
        A :func:`flow_prior_sample` name.

    Returns
    -------
    Tensor
        ``shape``-shaped draw from ``prior``.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.models._utils._generative import start_noise
    >>> start_noise(nn.Linear(2, 2).to(lucid.float16), (3, 2), None).dtype
    lucid.float16
    """
    device = resolve_generation_device(model, device)
    return flow_prior_sample(prior, shape, device=device, dtype=model_dtype(model))


class _CNFConfigLike(Protocol):
    @property
    def data_dim(self) -> int: ...
    @property
    def resolved_trace_method(self) -> str: ...
    @property
    def trace_noise(self) -> str: ...
    @property
    def sample_size(self) -> int | tuple[int, int]: ...
    @property
    def in_channels(self) -> int: ...


def check_image_shape(x: Tensor, image_shape: tuple[int, int, int], what: str) -> None:
    """Verify that ``x`` is a 4-D ``(B, C, H, W)`` image batch matching ``image_shape``."""
    channels, height, width = image_shape
    if x.ndim != 4 or tuple(int(s) for s in x.shape[1:]) != (channels, height, width):
        raise ValueError(
            f"{what} expects (B, {channels}, {height}, {width}) samples, "
            f"got shape {tuple(x.shape)}"
        )


@overload
def odeint_endpoint(
    func: Callable[..., object],
    state: Tensor,
    span: Sequence[float],
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    method: str | None = None,
) -> Tensor: ...


@overload
def odeint_endpoint(
    func: Callable[..., object],
    state: tuple[Tensor, Tensor],
    span: Sequence[float],
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    method: str | None = None,
) -> tuple[Tensor, Tensor]: ...


def odeint_endpoint(
    func: Callable[..., object],
    state: Tensor | tuple[Tensor, Tensor],
    span: Sequence[float],
    *,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    method: str | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Integrate ``func`` over ``span`` and return the endpoint with narrowed state type."""
    res = lucid.diffeq.odeint(
        func, state, span, rtol=rtol, atol=atol, method=method, return_trajectory=False
    )
    if isinstance(state, Tensor):
        assert isinstance(res, Tensor)
        return res
    assert isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], Tensor)
    return res[0], res[1]


@final
class LikelihoodDynamics:
    """Augmented right-hand side ``(x, Delta)`` for continuous-flow density evaluation."""

    def __init__(self, field: Callable[[Tensor, Tensor], Tensor], config: _CNFConfigLike) -> None:
        self.field = field
        self._dim = config.data_dim
        self._trace_method = config.resolved_trace_method
        self._trace_noise = config.trace_noise
        size = config.sample_size
        height, width = size if isinstance(size, tuple) else (size, size)
        self._image_shape = (config.in_channels, height, width)
        self.nfe = 0
        self._eps: Tensor | None = None
        self._seeds: list[Tensor] | None = None

    def reset(self, eps: Tensor | None = None) -> None:
        """Zero the evaluation counter and fix the Hutchinson probe for one solve."""
        self.nfe = 0
        self._eps = eps
        self._seeds = None

    def sample_noise(self, like: Tensor) -> Tensor:
        """Draw a Hutchinson probe shaped like ``like``."""
        return trace_probe(like, self._trace_noise)

    def velocity(self, t: Tensor, x: Tensor) -> Tensor:
        """Evaluate the flat-state velocity field at ``(t, x)``."""
        self.nfe += 1
        batch = int(x.shape[0])
        image = x.reshape(batch, *self._image_shape)
        return self.field(image, t).reshape(batch, -1)

    def __call__(self, t: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        x, _ = state
        self.nfe += 1
        with lucid.enable_grad():
            source = x if x.requires_grad else x.detach().requires_grad_(True)
            batch = int(source.shape[0])
            image = source.reshape(batch, *self._image_shape)
            velocity = self.field(image, t).reshape(batch, -1)
            divergence = self._divergence(velocity, source)
        return velocity, divergence

    def _divergence(self, velocity: Tensor, x: Tensor) -> Tensor:
        if self._trace_method == "exact":
            return exact_divergence(velocity, x, self._exact_seeds(velocity), create_graph=False)
        eps = self._eps
        if eps is None:
            eps = self.sample_noise(x)
            self._eps = eps
        return hutchinson_divergence(velocity, x, eps, create_graph=False)

    def _exact_seeds(self, velocity: Tensor) -> list[Tensor]:
        """Identity rows broadcast over the batch, built once per solve."""
        seeds = self._seeds
        if seeds is None or seeds[0].shape != velocity.shape:
            basis = lucid.eye(self._dim, dtype=velocity.dtype, device=velocity.device)
            ones = lucid.ones_like(velocity[:, :1])
            seeds = [basis[index].reshape(1, self._dim) * ones for index in range(self._dim)]
            self._seeds = seeds
        return seeds


def cnf_gaussian_log_prob(
    dynamics: LikelihoodDynamics,
    x: Tensor,
    *,
    integrate: Callable[
        [Callable[..., object], tuple[Tensor, Tensor], float, float], tuple[Tensor, Tensor]
    ],
    trace_method: str,
    input_dim: int,
    t0: float = 0.0,
) -> Tensor:
    """Integrate ``dynamics`` from ``1.0`` down to ``t0`` and score under a standard normal."""
    batch = int(x.shape[0])
    flat = x.reshape(batch, -1)
    delta = lucid.zeros((batch,), dtype=flat.dtype, device=flat.device)
    eps = None if trace_method == "exact" else dynamics.sample_noise(flat)
    dynamics.reset(eps)
    x0, delta0 = integrate(dynamics, (flat, delta), 1.0, t0)
    prior = -0.5 * (x0 * x0).sum(dim=-1) - 0.5 * input_dim * math.log(2.0 * math.pi)
    return prior + delta0
