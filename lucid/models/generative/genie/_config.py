"""Genie configuration — Bruce et al., ICML 2024.

A world model with no actions in its training data.  Every other world
model in this zoo learns ``p(x_{t+1} | x_{<=t}, a_{<=t})`` from logged
actions; Genie learns from 30,000 hours of platformer gameplay videos
off the internet, where nobody recorded which button was pressed.  It
invents the actions: a small vector-quantised model is asked what
changed between two frames, and forced to answer with one of eight
codes.  Those codes are the controller.

The paper tabulates three components and leaves several of their details
unstated — the feed-forward width, where the layer norms sit, how
positions are encoded, the commitment weight.  Every such field below
says so and names the value chosen in its place.

Nothing was released to check those choices against: no code, no weights,
no data, by the authors' own statement.  The open reimplementations agree
with what is built here on the mask rate, the in-place masked prediction,
the commitment weight and the padding, and they disagree with each other
about the rest — the feed-forward is four times the width here, as in
Jasmine and 1X's transformer, where Jafar uses one.  Two of their
departures from the paper are deliberate and worth knowing about: they
read the latent action from a dedicated token rather than by averaging
patches, and Jasmine reports that *prepending* the action instead of
adding it, which is what the paper states and what this does, leaves
training loss unchanged while making rollouts markedly better.
"""

import math
from dataclasses import dataclass
from typing import ClassVar, override

from lucid.models._meta import model_family_meta
from lucid.models.generative._common._config import (
    GenerativeActivation,
    GenerativeModelConfig,
    check_sample_space_channels,
)


@model_family_meta(
    canonical_name="Genie",
    citation=(
        "Bruce, Jake, et al. "
        '"Genie: Generative Interactive Environments." '
        "Proceedings of the 41st International Conference on Machine "
        "Learning, PMLR, vol. 235, 2024, pp. 4603–4623."
    ),
    theory=r"""
    Genie turns unlabelled video into a playable environment.  A user
    gives it one image and presses one of :math:`|A| = 8` buttons; it
    answers with the next frame, and keeps answering.  Nothing in its
    training data says what the buttons are --- the meaning of each is
    learned, and emerges as *left*, *right*, *jump* and *no-op* on the
    platformer games it was trained on.

    **Three models, one block.**  Each component is a spatiotemporal
    (ST) transformer over a video of :math:`T` frames, each cut into
    :math:`N` patches.  A block attends *spatially* over the :math:`N`
    tokens of one frame, then *temporally* over the :math:`T` tokens at
    one position under a causal mask, and only then applies a single
    feed-forward layer:

    .. math::

        h \leftarrow h + \mathrm{Attn}_{\text{space}}(h), \qquad
        h \leftarrow h + \mathrm{Attn}_{\text{time}}^{\text{causal}}(h),
        \qquad h \leftarrow h + \mathrm{FFW}(h).

    The cost is :math:`O(TN^2 + NT^2)` rather than :math:`O(T^2N^2)`,
    which is what makes 16 frames of 920 tokens trainable at all.

    **The latent action model** reads frames :math:`x_{1:t+1}` and emits
    a continuous vector for the transition :math:`t \to t+1`, which is
    quantised against a codebook of eight entries.  A decoder must then
    rebuild :math:`x_{t+1}` from the *past* frames and that code alone,
    so the code has to carry exactly what the history does not already
    predict --- the choice.  Eight codes are too few to smuggle the next
    frame through, which is the whole point.

    **The dynamics model** is a MaskGIT transformer over the tokens of a
    separately trained VQ video tokenizer.  The paper writes its
    objective over the whole next frame,

    .. math::

        \mathcal{L}_{\text{dyn}} = -\sum_{t \geq 2}
        \log p_\theta\bigl(z_t \mid z_{<t},\, \tilde a_{<t}\bigr),

    with the input tokens masked at a rate drawn from :math:`U[0.5, 1]`
    and the embedding of latent action :math:`\tilde a_t` *added* rather
    than concatenated.  Read literally that predicts frame :math:`t` from
    the frames before it alone, which cannot be refined: the 25 MaskGIT
    steps a frame is generated over need the tokens revealed so far to
    re-enter the model.  So the frame is filled in *in place* here ---
    frame :math:`t`'s masked tokens, conditioned on
    :math:`z_{<t}`, on what is already revealed of :math:`z_t`, and on
    :math:`\tilde a_{t-1}` --- and the loss covers the masked positions,
    as MaskGIT's does.

    At play time a frame starts fully masked and is revealed over those
    steps, keeping the most confident samples each time.  Only the latent
    action model's codebook is needed then: a user's button is an index
    into it.  The paper reports 10.7B parameters --- a 200M tokenizer, a
    300M latent action model and a 10.1B dynamics model.
    """,
)
@dataclass(frozen=True)
class GenieConfig(GenerativeModelConfig):
    r"""Frozen configuration for the Genie family.

    Defaults are the paper's Platformers model (Tables 5, 7 and 12).
    Fields marked *not stated* are absent from the paper; the value and
    the reason for it are given in each entry.

    Parameters
    ----------
    sample_size : int or tuple of int, default=(90, 160)
        Frame height and width.  Platformers videos are 160x90.
    in_channels : int, default=3
        Channels per frame.
    out_channels : int, default=3
        Channels each decoder emits — one frame, so ``in_channels``.
    act_fn : {"gelu", "relu", "silu", "swish", "elu"}, default="gelu"
        Feed-forward activation.  *Not stated*; GELU, as in the ViT
        encoders the tokenizer is built from.
    num_frames : int, default=16
        Frames every component sees at once, and so the dynamics model's
        memory.  Section 3: sequence length 16 at 10 FPS.
    mlp_ratio : float, default=4.0
        Feed-forward width as a multiple of the model width.  *Not
        stated*; 4, the transformer convention.
    commitment_cost : float, default=0.25
        :math:`\beta` of both vector quantisers.  *Not stated*; 0.25,
        the VQ-VAE paper's value.
    code_reset_threshold : float, default=0.1
        A code used less than this fraction of its fair share, averaged
        over batches, is moved onto an encoder output from the current
        batch while training.  ``0`` switches that off.  *Not stated*; a
        codebook of eight entries collapses readily, and a collapsed one
        is invisible in the loss.
    tokenizer_patch_size : int, default=4
        Tokenizer patch side (Table 7).  A frame whose side is not a
        multiple is padded at the bottom and right — *not stated*; 90
        becomes 92, which is the reading under which the paper's token
        counts are exact (942B tokens is 40x23 tokens per frame).
    num_codes : int, default=1024
        Tokenizer codebook size (Table 7).
    code_dim : int, default=32
        Tokenizer code width (Table 7).
    tokenizer_encoder_layers, tokenizer_encoder_dim, tokenizer_encoder_heads : int
        12, 512 and 8 (Table 7).
    tokenizer_encoder_head_dim : int or None, default=64
        Query/key size per head (Table 7).  ``None`` means
        ``dim // heads``.
    tokenizer_decoder_layers, tokenizer_decoder_dim, tokenizer_decoder_heads : int
        20, 1024 and 16 (Table 7) — the paper found scaling the decoder
        worth more than scaling the encoder.
    tokenizer_decoder_head_dim : int or None, default=64
        Query/key size per head (Table 7).
    action_patch_size : int, default=16
        Latent action model patch side (Table 5).
    num_latent_actions : int, default=8
        :math:`|A|`, the latent action codebook (Table 5).
    action_dim : int, default=32
        Latent action width (Table 5).
    action_encoder_layers, action_encoder_dim, action_encoder_heads : int
        20, 1024 and 16 (Table 5).
    action_encoder_head_dim : int or None, default=None
        *Not stated* — Table 5 gives no query/key size, so
        ``dim // heads``.
    action_decoder_layers, action_decoder_dim, action_decoder_heads : int
        20, 1024 and 16 (Table 5).
    action_decoder_head_dim : int or None, default=None
        *Not stated*, as for the encoder.
    dynamics_layers, dynamics_dim, dynamics_heads : int
        48, 5120 and 36 (Table 12).
    dynamics_head_dim : int or None, default=128
        Query/key size per head (Table 12).  ``36 x 128`` is 4608, not
        5120: the attention's inner width is decoupled from the model's.
    dynamics_qk_norm : bool, default=True
        Normalise queries and keys per head.  Section 3 uses it for the
        dynamics model, for stability in bfloat16.
    mask_ratio_min, mask_ratio_max : float, default=0.5, 1.0
        The training mask rate is drawn uniformly from this range
        (Section 2.1).
    maskgit_steps : int, default=25
        Refinement steps per generated frame (Section 3).
    temperature : float, default=2.0
        Sampling temperature at play time (Section 3).

    Notes
    -----
    Reference: Bruce, Jake, et al., *"Genie: Generative Interactive
    Environments"*, ICML 2024 (arXiv:2402.15391).  The CoinRun case
    study of Appendix F is :func:`lucid.models.genie_coinrun`.

    Examples
    --------
    >>> from lucid.models.generative.genie import GenieConfig
    >>> config = GenieConfig()
    >>> config.frame_shape, config.token_grid
    ((90, 160), (23, 40))
    >>> config.num_latent_actions, config.maskgit_steps, config.temperature
    (8, 25, 2.0)

    Every component's attention may be narrower than its model width:

    >>> config.dynamics_heads * config.attention_head_dim("dynamics")
    4608
    """

    model_type: ClassVar[str] = "genie"

    sample_size: int | tuple[int, int] = (90, 160)
    in_channels: int = 3
    out_channels: int = 3
    act_fn: GenerativeActivation = "gelu"

    num_frames: int = 16
    mlp_ratio: float = 4.0
    commitment_cost: float = 0.25
    code_reset_threshold: float = 0.1

    tokenizer_patch_size: int = 4
    num_codes: int = 1024
    code_dim: int = 32
    tokenizer_encoder_layers: int = 12
    tokenizer_encoder_dim: int = 512
    tokenizer_encoder_heads: int = 8
    tokenizer_encoder_head_dim: int | None = 64
    tokenizer_decoder_layers: int = 20
    tokenizer_decoder_dim: int = 1024
    tokenizer_decoder_heads: int = 16
    tokenizer_decoder_head_dim: int | None = 64

    action_patch_size: int = 16
    num_latent_actions: int = 8
    action_dim: int = 32
    action_encoder_layers: int = 20
    action_encoder_dim: int = 1024
    action_encoder_heads: int = 16
    action_encoder_head_dim: int | None = None
    action_decoder_layers: int = 20
    action_decoder_dim: int = 1024
    action_decoder_heads: int = 16
    action_decoder_head_dim: int | None = None

    dynamics_layers: int = 48
    dynamics_dim: int = 5120
    dynamics_heads: int = 36
    dynamics_head_dim: int | None = 128
    dynamics_qk_norm: bool = True
    mask_ratio_min: float = 0.5
    mask_ratio_max: float = 1.0
    maskgit_steps: int = 25
    temperature: float = 2.0

    # The transformers the config describes, by the prefix their fields
    # share.  ``attention_head_dim`` and the validation both walk it.
    _STACKS: ClassVar[tuple[str, ...]] = (
        "tokenizer_encoder",
        "tokenizer_decoder",
        "action_encoder",
        "action_decoder",
        "dynamics",
    )

    @property
    def frame_shape(self) -> tuple[int, int]:
        """Frame height and width, whichever way ``sample_size`` was written."""
        if isinstance(self.sample_size, int):
            return (self.sample_size, self.sample_size)
        return (self.sample_size[0], self.sample_size[1])

    @property
    def token_grid(self) -> tuple[int, int]:
        """Rows and columns of tokenizer patches, after padding."""
        return _grid(self.frame_shape, self.tokenizer_patch_size)

    @property
    def action_grid(self) -> tuple[int, int]:
        """Rows and columns of latent-action-model patches, after padding."""
        return _grid(self.frame_shape, self.action_patch_size)

    def attention_head_dim(self, stack: str) -> int:
        """Query/key size per head of one transformer.

        Parameters
        ----------
        stack : str
            One of ``"tokenizer_encoder"``, ``"tokenizer_decoder"``,
            ``"action_encoder"``, ``"action_decoder"`` or ``"dynamics"``.

        Returns
        -------
        int
            The tabulated size, or ``dim // heads`` where the paper gives
            none.
        """
        if stack not in self._STACKS:
            raise ValueError(f"unknown transformer {stack!r}; one of {self._STACKS}")
        head_dim: int | None = getattr(self, f"{stack}_head_dim")
        if head_dim is not None:
            return head_dim
        dim: int = getattr(self, f"{stack}_dim")
        heads: int = getattr(self, f"{stack}_heads")
        return dim // heads

    @override
    def __post_init__(self) -> None:
        # JSON round-trips turn tuples into lists; put the frame shape
        # back before the base class checks it.
        if isinstance(self.sample_size, list):
            object.__setattr__(self, "sample_size", tuple(self.sample_size))
        super().__post_init__()

        positive = [
            "num_frames",
            "tokenizer_patch_size",
            "num_codes",
            "code_dim",
            "action_patch_size",
            "num_latent_actions",
            "action_dim",
            "maskgit_steps",
        ]
        for stack in self._STACKS:
            positive += [f"{stack}_layers", f"{stack}_dim", f"{stack}_heads"]
        for name in positive:
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}")

        for stack in self._STACKS:
            head_dim = getattr(self, f"{stack}_head_dim")
            dim = getattr(self, f"{stack}_dim")
            heads = getattr(self, f"{stack}_heads")
            if head_dim is None and dim % heads != 0:
                raise ValueError(
                    f"{stack}_dim={dim} is not divisible by {stack}_heads={heads}; "
                    f"set {stack}_head_dim to size the heads explicitly"
                )
            if head_dim is not None and head_dim < 1:
                raise ValueError(f"{stack}_head_dim must be positive, got {head_dim}")

        if self.num_frames < 2:
            raise ValueError(
                f"a world model needs a transition, so num_frames must be at "
                f"least 2, got {self.num_frames}"
            )
        if self.num_latent_actions < 2:
            raise ValueError(
                f"one latent action is no choice at all; num_latent_actions "
                f"must be at least 2, got {self.num_latent_actions}"
            )
        if self.mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be positive, got {self.mlp_ratio}")
        check_sample_space_channels(
            self.in_channels, self.out_channels, "every decoder's frame"
        )
        if self.commitment_cost < 0.0:
            raise ValueError(
                f"commitment_cost must be non-negative, got {self.commitment_cost}"
            )
        if not 0.0 <= self.code_reset_threshold <= 1.0:
            raise ValueError(
                f"code_reset_threshold is a fraction of a code's fair share, "
                f"got {self.code_reset_threshold}"
            )
        if not 0.0 < self.mask_ratio_min <= self.mask_ratio_max <= 1.0:
            raise ValueError(
                f"mask rates must satisfy 0 < mask_ratio_min <= mask_ratio_max "
                f"<= 1, got {self.mask_ratio_min} and {self.mask_ratio_max}"
            )
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")


def _grid(frame: tuple[int, int], patch: int) -> tuple[int, int]:
    """Patches per side of a frame padded up to a multiple of ``patch``."""
    return (math.ceil(frame[0] / patch), math.ceil(frame[1] / patch))
