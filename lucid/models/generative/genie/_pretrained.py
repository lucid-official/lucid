"""Registry factories for Genie.

Two configurations, both the paper's own.

``genie`` is the Platformers model of Section 3 — the one the paper is
about, with its components in Tables 5, 7 and 12.  ``genie_coinrun`` is
the *Reproducible Case Study* of Appendix F, sized to train on a single
accelerator, with its components in Tables 15, 16 and 17.

The scaling study's dynamics models (Table 10, 41M to 2.7B) share the
Platformers tokenizer and latent action model and differ only in three
dynamics fields, so they are overrides of ``genie`` rather than factories:
``genie(dynamics_layers=18, dynamics_dim=512, dynamics_heads=8,
dynamics_head_dim=64)`` is the row the paper reports as 41M.

The Robotics model is not here.  The paper gives it no architecture and
three different sizes — 2.5B, 1.3B and 1B — so there is nothing to build
that would not be a guess.

No weights were released for either configuration.
"""

from lucid.models._registry import register_model
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained
from lucid.models.generative.genie._config import GenieConfig
from lucid.models.generative.genie._model import GenieForWorldModeling, GenieModel

__all__ = ["genie", "genie_world_model", "genie_coinrun", "genie_coinrun_world_model"]


_CFG_PLATFORMERS = GenieConfig()

# Appendix F, Tables 15–17.  Three things the appendix leaves out are
# filled in: the frame size (64x64, Procgen's native observation), the
# latent action patch size (16, as Table 5 gives for Platformers), and the
# dynamics model's head count — Table 17 prints ``num_layers`` twice, 12
# and 8, and 8 heads of 64 is the only reading that fits a width of 512
# the way every other component in the appendix does.
_CFG_COINRUN = GenieConfig(
    sample_size=64,
    tokenizer_encoder_layers=8,
    tokenizer_encoder_dim=512,
    tokenizer_encoder_heads=8,
    tokenizer_encoder_head_dim=None,
    tokenizer_decoder_layers=8,
    tokenizer_decoder_dim=512,
    tokenizer_decoder_heads=8,
    tokenizer_decoder_head_dim=None,
    num_latent_actions=6,
    action_encoder_layers=8,
    action_encoder_dim=512,
    action_encoder_heads=8,
    action_decoder_layers=8,
    action_decoder_dim=512,
    action_decoder_heads=8,
    dynamics_layers=12,
    dynamics_dim=512,
    dynamics_heads=8,
    dynamics_head_dim=None,
    temperature=1.0,
)


@register_model(
    task="base",
    family="genie",
    model_type="genie",
    model_class=GenieModel,
    default_config=_CFG_PLATFORMERS,
    summary="auto",
)
def genie(pretrained: bool = False, **overrides: object) -> GenieModel:
    r"""Construct the Platformers Genie — tokenizer, latent actions, dynamics.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights were released; ``True`` raises.
    **overrides : object
        Optional :class:`GenieConfig` field overrides.

    Returns
    -------
    GenieModel
        The three networks, untrained.

    Notes
    -----
    Reference: Bruce, Jake, et al., *"Genie: Generative Interactive
    Environments"*, ICML 2024 (arXiv:2402.15391) — Table 7 for the
    tokenizer, Table 5 for the latent action model, Table 12 for the
    dynamics model.

    The paper reports 10.7B parameters: a 200M tokenizer, a 300M latent
    action model and a 10.1B dynamics model.  This build has 20.2B —
    388M, 675M and 19.1B — and no choice of the values the paper leaves
    out reconciles its tables with those counts.  The latent action
    model's two attention layers per block come to 339M at the tabulated
    widths, over its 300M before any feed-forward layer; the tokenizer
    would need a feed-forward a tenth of its width and the dynamics model
    two-fifths of its, which disagree with each other and with every
    transformer at this scale.  The tables are what is built.  At float32
    that is about 81 GB; the scaling study's smaller dynamics models
    (Table 10) are overrides of the ``dynamics_*`` sizes.

    Examples
    --------
    The configuration is read without building the model:

    >>> from lucid.models import AutoConfig
    >>> config = AutoConfig.from_pretrained("genie")
    >>> config.dynamics_layers, config.dynamics_dim, config.dynamics_heads
    (48, 5120, 36)
    >>> config.frame_shape, config.num_latent_actions
    ((90, 160), 8)

    The 41M dynamics model of Table 10 is three fields away:

    >>> small = AutoConfig.from_pretrained("genie")
    >>> from dataclasses import replace
    >>> replace(small, dynamics_layers=18, dynamics_dim=512,
    ...         dynamics_heads=8, dynamics_head_dim=64).dynamics_dim
    512
    """
    if pretrained:
        reject_unavailable_pretrained("genie")
    return GenieModel(apply_overrides(_CFG_PLATFORMERS, overrides))


@register_model(
    task="world-modeling",
    family="genie",
    model_type="genie",
    model_class=GenieForWorldModeling,
    default_config=_CFG_PLATFORMERS,
    summary="auto",
)
def genie_world_model(pretrained: bool = False, **overrides: object) -> GenieForWorldModeling:
    r"""Construct the Platformers Genie as a playable environment.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights were released; ``True`` raises.
    **overrides : object
        Optional :class:`GenieConfig` field overrides.

    Returns
    -------
    GenieForWorldModeling
        The networks, plus MaskGIT decoding and the rollout that plays
        them from a prompt frame and latent actions.

    Notes
    -----
    Reference: Bruce et al., arXiv:2402.15391, Section 2.2 for playing
    the model and Section 3 for its sampling: 25 MaskGIT steps per frame
    at temperature 2.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import genie_world_model
    >>> model = genie_world_model(
    ...     sample_size=(8, 8), num_frames=4, num_codes=16, code_dim=4,
    ...     tokenizer_encoder_layers=1, tokenizer_encoder_dim=16,
    ...     tokenizer_encoder_heads=2, tokenizer_encoder_head_dim=8,
    ...     tokenizer_decoder_layers=1, tokenizer_decoder_dim=16,
    ...     tokenizer_decoder_heads=2, tokenizer_decoder_head_dim=8,
    ...     action_patch_size=4, action_dim=4,
    ...     action_encoder_layers=1, action_encoder_dim=16, action_encoder_heads=2,
    ...     action_decoder_layers=1, action_decoder_dim=16, action_decoder_heads=2,
    ...     dynamics_layers=1, dynamics_dim=16, dynamics_heads=2,
    ...     dynamics_head_dim=8, maskgit_steps=2).eval()
    >>> out = model(lucid.rand(1, 1, 3, 8, 8), lucid.tensor([[3, 1]], dtype=lucid.int64))
    >>> out.frames.shape
    (1, 2, 3, 8, 8)
    """
    if pretrained:
        reject_unavailable_pretrained("genie_world_model")
    return GenieForWorldModeling(apply_overrides(_CFG_PLATFORMERS, overrides))


@register_model(
    task="base",
    family="genie",
    model_type="genie",
    model_class=GenieModel,
    default_config=_CFG_COINRUN,
    summary="auto",
)
def genie_coinrun(pretrained: bool = False, **overrides: object) -> GenieModel:
    r"""Construct the CoinRun case study — Genie at single-accelerator scale.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights were released; ``True`` raises.
    **overrides : object
        Optional :class:`GenieConfig` field overrides.

    Returns
    -------
    GenieModel
        The three networks, untrained.

    Notes
    -----
    Reference: Bruce et al., arXiv:2402.15391, Appendix F — Table 15 for
    the tokenizer, Table 16 for the latent action model (six actions),
    Table 17 for the dynamics model (sampling at temperature 1).  The
    appendix trains the tokenizer on batches of 48 sixteen-frame clips
    and the other two, together, on batches of 36.

    Three values it does not give are filled in: 64x64 frames, a latent
    action patch of 16, and 8 dynamics heads where Table 17 prints
    ``num_layers`` twice.  The appendix reports no parameter count; this
    build has 188M.

    Examples
    --------
    >>> from lucid.models import genie_coinrun
    >>> model = genie_coinrun()
    >>> config = model.config
    >>> config.frame_shape, config.num_latent_actions, config.temperature
    ((64, 64), 6, 1.0)
    >>> config.token_grid
    (16, 16)
    """
    if pretrained:
        reject_unavailable_pretrained("genie_coinrun")
    return GenieModel(apply_overrides(_CFG_COINRUN, overrides))


@register_model(
    task="world-modeling",
    family="genie",
    model_type="genie",
    model_class=GenieForWorldModeling,
    default_config=_CFG_COINRUN,
    summary="auto",
)
def genie_coinrun_world_model(
    pretrained: bool = False, **overrides: object
) -> GenieForWorldModeling:
    r"""Construct the CoinRun case study as a playable environment.

    Parameters
    ----------
    pretrained : bool, default=False
        No weights were released; ``True`` raises.
    **overrides : object
        Optional :class:`GenieConfig` field overrides.

    Returns
    -------
    GenieForWorldModeling
        The networks, plus MaskGIT decoding and the rollout that plays
        them.

    Notes
    -----
    Reference: Bruce et al., arXiv:2402.15391, Appendix F, Table 17:
    25 MaskGIT steps at temperature 1.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import genie_coinrun_world_model
    >>> model = genie_coinrun_world_model(
    ...     sample_size=16, num_frames=3, num_codes=16, code_dim=4,
    ...     tokenizer_encoder_layers=1, tokenizer_encoder_dim=16,
    ...     tokenizer_encoder_heads=2, tokenizer_decoder_layers=1,
    ...     tokenizer_decoder_dim=16, tokenizer_decoder_heads=2,
    ...     action_patch_size=8, action_dim=4,
    ...     action_encoder_layers=1, action_encoder_dim=16, action_encoder_heads=2,
    ...     action_decoder_layers=1, action_decoder_dim=16, action_decoder_heads=2,
    ...     dynamics_layers=1, dynamics_dim=16, dynamics_heads=2,
    ...     maskgit_steps=3).eval()
    >>> model.config.num_latent_actions
    6
    >>> out = model(lucid.rand(2, 2, 3, 16, 16), lucid.tensor([[5], [0]], dtype=lucid.int64))
    >>> out.frames.shape, out.tokens.shape
    ((2, 1, 3, 16, 16), (2, 3, 16))
    """
    if pretrained:
        reject_unavailable_pretrained("genie_coinrun_world_model")
    return GenieForWorldModeling(apply_overrides(_CFG_COINRUN, overrides))
