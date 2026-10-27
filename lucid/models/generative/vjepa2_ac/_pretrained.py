"""Registry factories for the official V-JEPA 2-AC variant.

``params`` is left unregistered: the release's 1B counts the ViT-g
encoder, while a factory here builds that encoder and the 24-block
action-conditioned predictor beside it.
"""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.generative.vjepa2_ac._config import VJEPA2ACConfig
from lucid.models.generative.vjepa2_ac._model import VJEPA2ACForWorldModeling, VJEPA2ACModel
from lucid.models.generative.vjepa2_ac._weights import VJEPA2ACWeights
from lucid.models._utils._common import apply_overrides

__all__ = ["vjepa2_ac_vit_giant", "vjepa2_ac_vit_giant_world_model"]


_CFG_GIANT = VJEPA2ACConfig()


@register_model(
    task="base",
    family="vjepa2_ac",
    model_type="vjepa2_ac",
    model_class=VJEPA2ACModel,
    default_config=_CFG_GIANT,
    summary="auto",
)
def vjepa2_ac_vit_giant(
    pretrained: bool | str = False, *, weights: VJEPA2ACWeights | None = None, **overrides: object
) -> VJEPA2ACModel:
    r"""Construct the official V-JEPA 2-AC ViT-g action model.

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the Lucid-hosted official action checkpoint; a string
        selects a declared weight tag explicitly.
    weights : VJEPA2ACWeights, optional, keyword-only
        Explicit weight enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`VJEPA2ACConfig` field overrides.

    Returns
    -------
    VJEPA2ACModel
        The ViT-g/16 latent encoder and action-conditioned predictor.

    Notes
    -----
    Reference: Assran et al., arXiv:2506.09985, 2025.  The release publishes
    one action-conditioned ViT-g checkpoint; smaller variants are
    intentionally not registered.  One action-model step is one frame — the
    released loop repeats each frame to fill the encoder's tubelet — so a
    clip of ``T`` frames takes ``T - 1`` conditioning rows.  Checked against
    the official implementation with that checkpoint: encoder ``8.5e-5``,
    predictor ``4.5e-5`` relative.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> config = AutoConfig.from_pretrained("vjepa2_ac_vit_giant")
    >>> config.encoder_dim, config.encoder_depth, config.encoder_heads
    (1408, 40, 22)
    >>> config.predictor_dim, config.predictor_depth, config.predictor_heads
    (1024, 24, 16)
    >>> config.tokens_per_step
    256
    """
    entry = weights_mod.resolve_weights(VJEPA2ACWeights, pretrained, weights)
    model = VJEPA2ACModel(apply_overrides(_CFG_GIANT, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="vjepa2_ac_vit_giant")
    return model


@register_model(
    task="world-modeling",
    family="vjepa2_ac",
    model_type="vjepa2_ac",
    model_class=VJEPA2ACForWorldModeling,
    default_config=_CFG_GIANT,
    summary="auto",
)
def vjepa2_ac_vit_giant_world_model(
    pretrained: bool | str = False, *, weights: VJEPA2ACWeights | None = None, **overrides: object
) -> VJEPA2ACForWorldModeling:
    r"""V-JEPA 2-AC under Lucid's world-model task wrapper.

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the Lucid-hosted official action checkpoint into the
        wrapped model; a string selects a declared weight tag explicitly.
    weights : VJEPA2ACWeights, optional, keyword-only
        Explicit weight enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`VJEPA2ACConfig` field overrides.

    Returns
    -------
    VJEPA2ACForWorldModeling
        The same encoder and predictor as :func:`vjepa2_ac_vit_giant`,
        registered under the ``world-modeling`` task.

    Notes
    -----
    Reference: Assran et al., arXiv:2506.09985, 2025.  The wrapper adds no
    parameters; it exists so ``AutoModelForWorldModeling`` can reach the
    family.  Planning and controller search stay outside the model — the
    forward is one teacher-forced latent transition.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> config = AutoConfig.from_pretrained("vjepa2_ac_vit_giant_world_model")
    >>> config.action_dim, config.state_dim
    (7, 7)
    >>> config.frame_causal, config.use_extrinsics
    (True, False)
    """
    entry = weights_mod.resolve_weights(VJEPA2ACWeights, pretrained, weights)
    model = VJEPA2ACForWorldModeling(apply_overrides(_CFG_GIANT, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(
            model.vjepa2_ac, entry, name="vjepa2_ac_vit_giant_world_model"
        )
    return model
