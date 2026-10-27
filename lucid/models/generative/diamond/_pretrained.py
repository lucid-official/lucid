"""Registry factories for DIAMOND.

One architecture, so one nominal name — the paper trains the same
configuration across all 26 Atari games, and H11 allows a size suffix
only where a paper puts a variant table.  The *weights* are per game,
which is a tag rather than a factory.

The appendix scales the world model up for *Counter-Strike* and motorway
driving and reports 122M parameters, but gives no architecture for it:
no channel widths, no depths, nothing that would let the number be
reproduced rather than guessed.  What it does state is the conditioning
length, :math:`L = 6`, which is a configuration override —
``diamond(conditioning_frames=6)``.

``params`` is registered at the released count.  The architecture here
is the released one rather than the paper's prose, which under-specifies
it in four places, so the number a factory builds is exactly what the
published checkpoints hold.
"""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.generative.diamond._config import DIAMONDConfig
from lucid.models.generative.diamond._model import (
    DIAMONDForWorldModeling,
    DIAMONDModel,
)
from lucid.models.generative.diamond._weights import DIAMONDWeights
from lucid.weights import WeightsEnum
from lucid.models._utils._common import apply_overrides

__all__ = ["diamond", "diamond_world_model", "diamond_csgo"]


_CFG_ATARI = DIAMONDConfig()

# The released ``config/agent/csgo.yaml``, verbatim.  A different model
# rather than a bigger one: no reward head, no actor-critic, non-square
# frames, attention at the two deepest resolutions, and a second
# diffusion model that magnifies five times.
_CFG_CSGO = DIAMONDConfig(
    sample_size=(30, 56),
    unet_channels=(128, 256, 512, 1024),
    unet_layers=(2, 2, 2, 2),
    attn_depths=(0, 0, 1, 1),
    cond_dim=2048,
    num_actions=51,
    with_agent=False,
    noise_previous_obs=True,
    sigma_offset_noise=0.1,
    upsampler_channels=(64, 64, 128, 256),
    upsampler_layers=(2, 2, 2, 2),
    upsampler_attn_depths=(0, 0, 0, 1),
    upsampling_factor=5,
)


def _tag(pretrained: bool | str) -> bool | str:
    """Accept a game's own spelling as a tag.

    The enum's members are upper-case, as Python enums are, but the tag a
    caller has in hand is the game's name — ``"Breakout"``, ``"MsPacman"``
    — because that is what the paper's tables and the released files use.
    Making them type ``"MSPACMAN"`` would be an implementation detail
    leaking into the API.

    Parameters
    ----------
    pretrained : bool or str
        What the caller passed.

    Returns
    -------
    bool or str
        The same value, upper-cased when it is a name.
    """
    if isinstance(pretrained, str):
        tag = pretrained.upper()
        if tag == "CSGO":
            # The enum holds every checkpoint, and list_pretrained("diamond")
            # therefore offers this one — but CS:GO is a 382M world model
            # against this factory's 13M Atari agent, so loading it here
            # fails on several hundred shapes.  Say which factory instead.
            raise ValueError(
                "the CSGO checkpoint belongs to diamond_csgo, not diamond: "
                "it is a world model of a different size, and loading it "
                "here would fail on hundreds of shapes. "
                "Use diamond_csgo(pretrained=True)."
            )
        return tag
    return pretrained


def _csgo_tag(pretrained: bool | str) -> bool | str:
    """Resolve ``pretrained`` for the CS:GO factory.

    ``True`` means *this* factory's checkpoint.  The enum's ``DEFAULT``
    is an Atari agent — correct for :func:`diamond`, wrong here, where
    it would try to load 13M of agent into a 382M world model and fail
    on a shape mismatch.

    Parameters
    ----------
    pretrained : bool or str
        What the caller passed.

    Returns
    -------
    bool or str
        ``"CSGO"`` for ``True``, otherwise the tag upper-cased.
    """
    if pretrained is True:
        return "CSGO"
    if isinstance(pretrained, str):
        if pretrained.upper() == "CSGO":
            # Not via ``_tag``: that one refuses CSGO, which is right for the
            # Atari factories and exactly wrong here.
            return "CSGO"
        # The mirror of the refusal in ``_tag``: an Atari agent named
        # explicitly here would be 13M of weights against a 382M model.
        raise ValueError(
            f"{pretrained!r} is an Atari checkpoint and belongs to diamond, "
            f"not diamond_csgo. Use diamond(pretrained={pretrained!r})."
        )
    return _tag(pretrained)


def _actions_for(entry: WeightsEnum | None, overrides: dict[str, object]) -> dict[str, object]:
    """Add the checkpoint's action count to ``overrides`` when one applies.

    Atari exposes a different minimal action set per game — 3 for
    *Freeway*, 4 for *Breakout*, 6 for *Pong*, up to 18 — and the policy
    head and both action embeddings are built to it.  Loading an agent
    into a model sized for the wrong game fails on a shape mismatch, so
    the tag carries the number and it is applied here rather than left
    for the caller to look up.

    An explicit ``num_actions`` override wins: someone who states a size
    means it, and the strict load will say if it disagrees.

    Parameters
    ----------
    entry : WeightsEnum or None
        The resolved weight tag, or ``None`` when building untrained.
    overrides : dict
        Config overrides the caller passed.

    Returns
    -------
    dict
        ``overrides``, plus ``num_actions`` when a tag supplied one.
    """
    if entry is None or "num_actions" in overrides:
        return overrides
    actions = entry.value.meta.get("num_actions")
    return overrides if actions is None else {**overrides, "num_actions": actions}


@register_model(
    task="base",
    family="diamond",
    model_type="diamond",
    model_class=DIAMONDModel,
    default_config=_CFG_ATARI,
    summary="auto",
)
def diamond(
    pretrained: bool | str = False,
    *,
    weights: DIAMONDWeights | None = None,
    **overrides: object,
) -> DIAMONDModel:
    """Construct the Atari agent — denoiser, reward/end model, actor-critic.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Load a released agent.  ``True`` takes *Breakout*; pass a game
        name — ``"Pong"``, ``"Freeway"``, any of the benchmark's 26 — for
        that one.  The tag also sets ``num_actions``, since Atari's
        minimal action set differs per game.
    weights : DIAMONDWeights or None, optional, keyword-only
        An explicit tag, taking precedence over ``pretrained``.
    **overrides : object
        Optional :class:`DIAMONDConfig` field overrides.

    Returns
    -------
    DIAMONDModel
        The three networks, untrained.

    Notes
    -----
    Reference: Alonso, Eloi, et al., *"Diffusion for World Modeling:
    Visual Details Matter in Atari"*, NeurIPS, 2024 (arXiv:2405.12399),
    Table 2 for the architecture and Table 3 for everything else.

    Examples
    --------
    >>> from lucid.models import diamond
    >>> config = diamond().config
    >>> config.conditioning_frames, config.denoise_steps, config.horizon
    (4, 3, 15)

    The appendix's 3D-environment experiments condition on six frames
    instead of four, which is a field rather than a second factory:

    >>> diamond(conditioning_frames=6).config.denoiser_in_channels
    21
    """
    entry = weights_mod.resolve_weights(DIAMONDWeights, _tag(pretrained), weights)
    model = DIAMONDModel(apply_overrides(_CFG_ATARI, _actions_for(entry, overrides)))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="diamond")
    return model


@register_model(
    task="world-modeling",
    family="diamond",
    model_type="diamond",
    model_class=DIAMONDForWorldModeling,
    default_config=_CFG_ATARI,
    summary="auto",
)
def diamond_world_model(
    pretrained: bool | str = False,
    *,
    weights: DIAMONDWeights | None = None,
    **overrides: object,
) -> DIAMONDForWorldModeling:
    r"""Construct DIAMOND with its objectives and imagination rollout.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Load a released agent.  ``True`` takes *Breakout*; pass a game
        name — ``"Pong"``, ``"Freeway"``, any of the benchmark's 26 — for
        that one.  The tag also sets ``num_actions``, since Atari's
        minimal action set differs per game.
    weights : DIAMONDWeights or None, optional, keyword-only
        An explicit tag, taking precedence over ``pretrained``.
    **overrides : object
        Optional :class:`DIAMONDConfig` field overrides.

    Returns
    -------
    DIAMONDForWorldModeling
        The agent, plus the losses that train it and the rollout that
        generates its experience.

    Notes
    -----
    Reference: Alonso et al., arXiv:2405.12399, Appendix F for the two
    RL objectives and Algorithm 1 for the training loop they sit in.

    The three losses are trained separately, in the order Algorithm 1
    gives: the denoiser on real transitions, the reward/termination model
    on real sequences, and the actor-critic entirely on imagined ones.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models import diamond_world_model
    >>> model = diamond_world_model(
    ...     sample_size=16, unet_channels=(8, 8), unet_layers=(1, 1),
    ...     reward_channels=(8, 8), reward_layers=(1, 1),
    ...     actor_channels=(8, 8), actor_layers=(1, 1), cond_dim=16,
    ...     reward_cond_dim=8, reward_lstm_dim=16, actor_lstm_dim=16,
    ...     num_actions=4, horizon=3).eval()
    >>> frames = lucid.randn((1, 4, 3, 16, 16))
    >>> actions = lucid.tensor([[0, 1, 2, 3]], dtype=lucid.int64)
    >>> with lucid.no_grad():
    ...     out = model(frames, actions)
    >>> out.frames.shape, out.returns.shape
    ((1, 3, 3, 16, 16), (1, 3))
    """
    entry = weights_mod.resolve_weights(DIAMONDWeights, _tag(pretrained), weights)
    model = DIAMONDForWorldModeling(apply_overrides(_CFG_ATARI, _actions_for(entry, overrides)))
    if entry is not None:
        # The checkpoint is the agent's; this wrapper keeps it under
        # ``diamond``, so the load targets that rather than the wrapper.
        weights_mod.load_weight_entry(model.diamond, entry, name="diamond")
    return model


@register_model(
    # "base", not "world-modeling", even though a world model is what it
    # is.  The task tag names the *class* a factory returns, and this one
    # returns the direct model: CS:GO has no agent, so the task wrapper's
    # imagination and actor-critic would be machinery with nothing behind
    # it.  Registering a `PretrainedModel` under a task breaks the zoo's
    # invariant that every task-registered class inherits a task base.
    task="base",
    family="diamond",
    model_type="diamond",
    model_class=DIAMONDModel,
    default_config=_CFG_CSGO,
    summary="auto",
)
def diamond_csgo(
    pretrained: bool | str = False,
    *,
    weights: DIAMONDWeights | None = None,
    **overrides: object,
) -> DIAMONDModel:
    r"""Construct the *Counter-Strike* world model.

    Parameters
    ----------
    pretrained : bool or str, default=False
        Load the released CS:GO world model.  ``True`` and ``"CSGO"``
        both select it; there is only the one.
    weights : DIAMONDWeights or None, optional, keyword-only
        An explicit tag, taking precedence over ``pretrained``.
    **overrides : object
        Optional :class:`DIAMONDConfig` field overrides.

    Returns
    -------
    DIAMONDModel
        The world model and its upsampler.  ``reward_end`` and
        ``actor_critic`` are ``None``: this experiment has no
        reinforcement learning, so there is no agent to build.

    Notes
    -----
    Reference: Alonso et al., arXiv:2405.12399, Section 6 and Appendix M,
    with the architecture taken from the released
    ``config/agent/csgo.yaml`` — the paper reports parameter counts for
    this model but no channel widths or depths.

    Frames are ``30x56`` here and ``150x280`` after the upsampler, which
    is what makes a 3D scene affordable: diffusing at the full resolution
    would cost far more for detail a cheaper second network can add.

    Examples
    --------
    >>> from lucid.models import diamond_csgo
    >>> config = diamond_csgo().config
    >>> config.frame_shape, config.num_actions, config.upsampling_factor
    ((30, 56), 51, 5)

    >>> config.with_agent
    False
    """
    entry = weights_mod.resolve_weights(DIAMONDWeights, _csgo_tag(pretrained), weights)
    model = DIAMONDModel(apply_overrides(_CFG_CSGO, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="diamond_csgo")
    return model
