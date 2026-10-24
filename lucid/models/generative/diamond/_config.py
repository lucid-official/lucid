"""DIAMOND configuration — Alonso et al., NeurIPS 2024.

The world model that stopped compressing.  DreamerV3, IRIS, TWM and
STORM all model dynamics as a sequence of *discrete latents*, and that
compression is the point — it keeps long rollouts from drifting.  It is
also lossy, and DIAMOND's claim is that what it discards matters: an
enemy sprite a few pixels wide, a reward pickup, a ball.  So DIAMOND
predicts the next frame directly, in pixel space, with a diffusion model.

The design question that decides whether this works is not "diffusion or
not" but *which* diffusion.  A world model is called once per imagined
step over a horizon of fifteen, so it can afford a handful of network
evaluations, not a thousand — and at that budget the usual DDPM
parameterisation drifts into colour-shifted garbage within a hundred
steps.  EDM's does not.  That is the paper's most load-bearing choice,
and the reason it is a configuration field here rather than an
implementation detail.
"""

from dataclasses import dataclass
from typing import ClassVar, override

from lucid.models._meta import model_family_meta
from lucid.models.generative._common._config import (
    GenerativeModelConfig,
    check_sample_space_channels,
)


@model_family_meta(
    canonical_name="DIAMOND",
    citation=(
        "Alonso, Eloi, et al. "
        '"Diffusion for World Modeling: Visual Details Matter in Atari." '
        "Advances in Neural Information Processing Systems, vol. 37, 2024."
    ),
    theory=r"""
    A world model has to answer :math:`p(x_{t+1} \mid x_{\leq t},
    a_{\leq t})`.  DIAMOND answers it with a conditional diffusion model
    over the *image*, dropping the discrete latent bottleneck its
    predecessors relied on.

    **The conditioning is the architecture.**  The last :math:`L` clean
    frames are concatenated to the noised next frame along the channel
    axis --- plain frame stacking, which keeps the network a standard
    U-Net 2D --- while the actions and the diffusion time enter through
    adaptive group normalisation inside the residual blocks.  Nothing
    about the image path knows it is a world model.

    **EDM, not DDPM, and the difference is not cosmetic.**  The network
    :math:`F_\theta` is wrapped in Karras et al.'s preconditioners:

    .. math::

        D_\theta(x^\tau_{t+1}, y^\tau_t) = c_{\text{skip}}^\tau\,
        x^\tau_{t+1} + c_{\text{out}}^\tau\, F_\theta\bigl(
        c_{\text{in}}^\tau x^\tau_{t+1},\, y^\tau_t \bigr),
        \qquad
        c_{\text{skip}}^\tau = \frac{\sigma_{\text{data}}^2}
        {\sigma_{\text{data}}^2 + \sigma^2(\tau)}

    so the training target adaptively mixes signal and noise with the
    degradation level.  When :math:`\sigma \gg \sigma_{\text{data}}` the
    skip vanishes and the network is asked for the *clean frame*; when
    :math:`\sigma \to 0` it is asked for the added noise.  DDPM asks for
    the noise at every level, which makes the high-noise regime an
    identity map and leaves the score badly estimated exactly where
    sampling begins.  Autoregressed over a thousand imagined steps that
    error compounds --- the paper shows DDPM drifting out of
    distribution while the EDM model stays stable even at a **single**
    denoising step.

    **Three steps, and the reason it is not one.**  A denoiser trained
    under an :math:`L_2` loss predicts the *expectation* over possible
    reconstructions, so when the next frame is genuinely multi-modal ---
    an opponent whose move the agent cannot predict --- one step returns
    a blur between the outcomes.  Iterating drives the sample onto a
    single mode.  Hence :math:`n = 3`, everywhere in the paper.

    **What diffusion does not model.**  Rewards and episode ends are
    scalar predictions, not images, so they get their own CNN-LSTM; the
    actor-critic gets another.  The agent is trained entirely inside the
    imagined environment, reaching a mean human-normalised score of
    1.46 on Atari 100k with 13M parameters --- fewer than IRIS's 30M or
    DreamerV3's 18M.
    """,
)
@dataclass(frozen=True)
class DIAMONDConfig(GenerativeModelConfig):
    r"""Frozen configuration for the DIAMOND family.

    Parameters
    ----------
    sample_size : int or tuple of int, default=64
        Frame resolution the world model works at.  Atari observations
        are resized to ``64x64`` before anything else touches them.
    in_channels : int, default=3
        Channels per frame.
    out_channels : int, default=3
        Channels the denoiser emits — one frame, so the same as
        ``in_channels``.
    num_actions : int, default=18
        Size of the discrete action set.  Atari's full set is 18; a game
        exposing fewer simply leaves the tail unused.
    conditioning_frames : int, default=4
        :math:`L`, how many past frames and actions the denoiser sees.
        The 3D-environment experiments in the paper's appendix use 6.
    unet_channels : tuple of int, default=(64, 64, 64, 64)
        Residual-block channels per U-Net resolution.
    unet_layers : tuple of int, default=(2, 2, 2, 2)
        Residual blocks per resolution.
    cond_dim : int, default=256
        Width of the conditioning vector the adaptive group norms are
        regressed from.
    attn_depths : tuple of int or None, default=None
        Whether each U-Net resolution attends.  ``None`` means none of
        them do, which is Atari's setting and the common case; it is
        expanded to match ``unet_channels`` rather than written out, so
        a narrower model does not have to restate it.  All zero for Atari, where
        only the middle blocks do; the CS:GO world model turns it on at
        its two deepest resolutions, which is where a 3D scene needs to
        relate parts of the frame that convolutions cannot reach.
    with_agent : bool, default=True
        Whether to build the reward/termination model and the
        actor-critic.  The released CS:GO configuration sets both to
        ``null``: that experiment trains a world model on static data
        with no reinforcement learning at all, so an agent would be
        parameters nothing ever updates.
    noise_previous_obs : bool, default=False
        Whether the conditioning frames are noised too.  ⚠️ Off for
        Atari, on for CS:GO — the released configs differ here and the
        paper mentions neither.  Noising the history is a form of
        augmentation against the compounding error a long autoregressive
        rollout accumulates.
    upsampler_channels : tuple of int or None, default=None
        When set, a second diffusion model is built that upsamples the
        denoiser's output.  CS:GO generates at 30x56 and upsamples five
        times to 150x280, because diffusing at the full resolution
        directly costs far more for the same detail.
    upsampler_layers : tuple of int or None, default=None
        Blocks per resolution in that model.
    upsampler_attn_depths : tuple of int or None, default=None
        Which of its resolutions attend.
    upsampling_factor : int, default=1
        How much the upsampler magnifies.
    sigma_data : float, default=0.5
        :math:`\sigma_{\text{data}}`, the data distribution's standard
        deviation, which sets where the preconditioners hand over
        between predicting noise and predicting the frame.
    p_mean : float, default=-0.4
        Mean of the log-normal the training noise level is drawn from.
    p_std : float, default=1.2
        Standard deviation of that log-normal.
    sigma_offset_noise : float, default=0.3
        Standard deviation of a per-channel offset added to the target
        alongside the usual noise.  ⚠️ **Not in the paper** — it appears
        only in the released ``config/agent/default.yaml``.  Offset noise
        lets the model shift a frame's overall level, which plain
        isotropic noise cannot express at any single :math:`\sigma`.
    attention_head_dim : int, default=8
        Channels per self-attention head in the blocks that have
        attention.  ⚠️ Also absent from the paper; taken from the
        released implementation.
    denoise_steps : int, default=3
        Denoising steps per imagined frame.  The paper's ablation is
        explicit that 1 is cheaper and worse on multi-modal games.
    reward_channels : tuple of int, default=(32, 32, 32, 32)
        Residual-block channels in the reward/termination model.
    reward_layers : tuple of int, default=(2, 2, 2, 2)
        Residual blocks per resolution there.
    reward_cond_dim : int, default=128
        Conditioning width for that model's adaptive group norms.
    reward_frames : int, default=2
        Frames the reward/termination model reads at once.  ⚠️ The paper
        says "a sequence of frames and actions"; the released encoder
        takes **two** — the observation and the one it led to — because
        a reward is a property of the transition, not of a frame.
    reward_lstm_dim : int, default=512
        Hidden width of its LSTM cell.
    actor_channels : tuple of int, default=(32, 32, 64, 64)
        Residual-block channels in the shared actor-critic trunk.
    actor_layers : tuple of int, default=(1, 1, 1, 1)
        Residual blocks per resolution there.
    actor_lstm_dim : int, default=512
        Hidden width of the actor-critic's LSTM cell.
    horizon : int, default=15
        :math:`H`, the imagination horizon the agent is trained over.
    gamma : float, default=0.985
        Discount factor.
    lambda_ : float, default=0.95
        :math:`\lambda` for the returns the value network regresses to.
    entropy_weight : float, default=0.001
        :math:`\eta`, the weight on the policy's entropy bonus.
    burn_in : int, default=4
        Steps of real experience replayed to initialise the LSTM states
        before imagination starts.  The paper sets this to :math:`L`.

    Notes
    -----
    Reference: Alonso et al., *"Diffusion for World Modeling: Visual
    Details Matter in Atari"*, NeurIPS, 2024 (arXiv:2405.12399).
    Architecture values are Table 2, training values Table 3, and the
    preconditioner constants Appendix C.

    The base is :class:`GenerativeModelConfig` rather than
    :class:`WorldModelConfig`, even though this is a world model.  That
    base describes a *latent* one — ``stoch_size``, ``deter_size``,
    ``free_nats``, ``kl_weight`` — and DIAMOND has no posterior and no
    KL to free-bit.  Inheriting them would put six fields on the
    documentation page that nothing in this family reads.

    Examples
    --------
    >>> from lucid.models.generative.diamond import DIAMONDConfig
    >>> config = DIAMONDConfig()
    >>> config.conditioning_frames, config.denoise_steps
    (4, 3)

    The denoiser's first convolution has to take the noised frame *and*
    the stack of past frames, which is what makes frame stacking cheap:

    >>> config.denoiser_in_channels
    15

    Three fields here have no counterpart in the paper — they come from
    the released configuration, which is more specific than the text:

    >>> config.sigma_offset_noise, config.reward_frames
    (0.3, 2)
    """

    model_type: ClassVar[str] = "diamond"

    sample_size: int | tuple[int, int] = 64
    in_channels: int = 3
    out_channels: int = 3
    num_actions: int = 18

    conditioning_frames: int = 4
    unet_channels: tuple[int, ...] = (64, 64, 64, 64)
    unet_layers: tuple[int, ...] = (2, 2, 2, 2)
    cond_dim: int = 256
    attn_depths: tuple[int, ...] | None = None
    with_agent: bool = True
    noise_previous_obs: bool = False
    upsampler_channels: tuple[int, ...] | None = None
    upsampler_layers: tuple[int, ...] | None = None
    upsampler_attn_depths: tuple[int, ...] | None = None
    upsampling_factor: int = 1

    sigma_data: float = 0.5
    p_mean: float = -0.4
    p_std: float = 1.2
    sigma_offset_noise: float = 0.3
    attention_head_dim: int = 8
    denoise_steps: int = 3

    reward_channels: tuple[int, ...] = (32, 32, 32, 32)
    reward_layers: tuple[int, ...] = (2, 2, 2, 2)
    reward_cond_dim: int = 128
    reward_frames: int = 2
    reward_lstm_dim: int = 512

    actor_channels: tuple[int, ...] = (32, 32, 64, 64)
    actor_layers: tuple[int, ...] = (1, 1, 1, 1)
    actor_lstm_dim: int = 512

    horizon: int = 15
    gamma: float = 0.985
    lambda_: float = 0.95
    entropy_weight: float = 0.001
    burn_in: int = 4

    @property
    def frame_shape(self) -> tuple[int, int]:
        """Frame height and width, whichever way ``sample_size`` was written.

        Atari's frames are square and CS:GO's are not — 150 by 280 —
        so nothing downstream may assume one number.
        """
        if isinstance(self.sample_size, int):
            return (self.sample_size, self.sample_size)
        return (self.sample_size[0], self.sample_size[1])

    @property
    def denoiser_in_channels(self) -> int:
        """Channels the denoiser's first convolution reads.

        The noised next frame plus :math:`L` clean past frames, stacked.
        """
        return self.in_channels * (self.conditioning_frames + 1)

    @override
    def __post_init__(self) -> None:
        super().__post_init__()
        # The denoiser's output is mixed with its noisy input frame by the
        # EDM preconditioning, so it is a frame of the same channels.
        check_sample_space_channels(
            self.in_channels, self.out_channels, "the denoised frame"
        )
        # JSON round-trips turn tuples into lists; the frozen dataclass
        # has to put them back or equality and hashing break.
        # ``None`` means "none of them attend", and so does a stale length
        # that asks for none.  Narrowing ``unet_channels`` through
        # ``dataclasses.replace`` carries the old flags along, and a caller
        # who shrank the U-Net to two stages did not thereby request
        # attention anywhere — only a list that names some is worth
        # failing over.
        if self.attn_depths is None or (
            len(self.attn_depths) != len(self.unet_channels)
            and not any(self.attn_depths)
        ):
            object.__setattr__(self, "attn_depths", (0,) * len(self.unet_channels))
        for name in (
            "unet_channels",
            "unet_layers",
            "attn_depths",
            "reward_channels",
            "reward_layers",
            "actor_channels",
            "actor_layers",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))

        for optional in (
            "upsampler_channels",
            "upsampler_layers",
            "upsampler_attn_depths",
        ):
            value = getattr(self, optional)
            if value is not None:
                object.__setattr__(self, optional, tuple(value))

        assert self.attn_depths is not None
        if len(self.attn_depths) != len(self.unet_channels):
            raise ValueError(
                f"attn_depths names one flag per U-Net resolution, got "
                f"{len(self.attn_depths)} for {len(self.unet_channels)} stages"
            )
        if (self.upsampler_channels is None) != (self.upsampler_layers is None):
            raise ValueError(
                "upsampler_channels and upsampler_layers describe the same "
                "model — give both or neither"
            )
        if self.upsampler_channels is not None and self.upsampling_factor < 2:
            raise ValueError(
                f"an upsampler that does not magnify is a second denoiser, "
                f"got upsampling_factor={self.upsampling_factor}"
            )

        stacks: tuple[str, ...] = ("unet", "reward", "actor")
        if self.upsampler_channels is not None:
            stacks += ("upsampler",)
        for name in stacks:
            channels = getattr(self, f"{name}_channels")
            layers = getattr(self, f"{name}_layers")
            if len(channels) != len(layers):
                raise ValueError(
                    f"{name}_channels and {name}_layers describe the same "
                    f"resolutions, got {len(channels)} and {len(layers)}"
                )
            if not channels:
                raise ValueError(f"{name}_channels must name at least one stage")
        # The U-Net reads one flag per stage; a short tuple failed with an
        # IndexError mid-build and a long one was silently cut.
        if (
            self.upsampler_channels is not None
            and self.upsampler_attn_depths is not None
            and len(self.upsampler_attn_depths) != len(self.upsampler_channels)
        ):
            raise ValueError(
                f"upsampler_attn_depths names one flag per upsampler resolution, "
                f"got {len(self.upsampler_attn_depths)} for "
                f"{len(self.upsampler_channels)} stages"
            )

        self._check_world_model_scalars()
        self._check_agent_scalars()

        # Not a divisibility check.  A stride-2 convolution rounds *up*,
        # so 30 -> 15 -> 8 -> 4 is fine and the decoder resizes to each
        # skip rather than assuming a clean doubling — which is what the
        # CS:GO model at 30x56 needs.  What does have to hold is that
        # there is something left to downsample.
        stages = len(self.unet_channels) - 1
        for axis in self.frame_shape:
            if axis < 2**stages:
                raise ValueError(
                    f"every side of sample_size must survive {stages} "
                    f"halvings, so at least {2**stages} — got "
                    f"{self.frame_shape}"
                )

    def _check_world_model_scalars(self) -> None:
        """Refuse a denoiser or reward-model knob outside its domain."""
        if self.conditioning_frames < 1:
            raise ValueError(
                f"conditioning_frames is the history the denoiser sees and "
                f"must be positive, got {self.conditioning_frames}"
            )
        if self.denoise_steps < 1:
            raise ValueError(
                f"denoise_steps must be positive, got {self.denoise_steps}"
            )
        if self.sigma_offset_noise < 0.0:
            raise ValueError(
                f"sigma_offset_noise is a standard deviation, got "
                f"{self.sigma_offset_noise}"
            )
        if self.reward_frames < 1:
            raise ValueError(
                f"reward_frames must be positive, got {self.reward_frames}"
            )
        if self.sigma_data <= 0.0:
            raise ValueError(
                f"sigma_data scales the preconditioners and must be "
                f"positive, got {self.sigma_data}"
            )

    def _check_agent_scalars(self) -> None:
        """Refuse an actor-critic knob outside its domain."""
        if self.num_actions < 1:
            raise ValueError(f"num_actions must be positive, got {self.num_actions}")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma is a discount factor, got {self.gamma}")
        if not 0.0 <= self.lambda_ <= 1.0:
            raise ValueError(f"lambda_ is a mixing weight, got {self.lambda_}")
        if self.horizon < 2:
            raise ValueError(
                f"lambda-returns need at least two imagined states, got "
                f"horizon={self.horizon}"
            )
        if self.burn_in < 0:
            raise ValueError(f"burn_in cannot be negative, got {self.burn_in}")
