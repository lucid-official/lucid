"""Registry factories for the released V-JEPA 2 variants.

The paper's four checkpoints, each also offered under the attentive probe
it is evaluated through.  ``params`` is left unregistered for the same
reason as I-JEPA and V-JEPA: the release's 300M, 600M and 1B describe one
encoder, while a factory here builds two of them and a predictor, so the
pill would disagree with the size the docs measure.  The probe's own
weights are not published, only the backbones'.
"""

from dataclasses import replace

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models._utils._common import apply_overrides, reject_unavailable_pretrained
from lucid.models.vision.vjepa2._config import VJEPA2Config
from lucid.models.vision.vjepa2._model import (
    VJEPA2ForVideoClassification,
    VJEPA2Model,
)
from lucid.models.vision.vjepa2._weights import (
    VJEPA2ViTGiant384Weights,
    VJEPA2ViTGiantWeights,
    VJEPA2ViTHugeWeights,
    VJEPA2ViTLargeWeights,
)

__all__ = [
    "vjepa2_vit_large",
    "vjepa2_vit_large_cls",
    "vjepa2_vit_huge",
    "vjepa2_vit_huge_cls",
    "vjepa2_vit_giant",
    "vjepa2_vit_giant_cls",
    "vjepa2_vit_giant_384",
    "vjepa2_vit_giant_384_cls",
]


_CFG_LARGE = VJEPA2Config(dim=1024, depth=24, num_heads=16, image_size=256)
_CFG_HUGE = VJEPA2Config(dim=1280, depth=32, num_heads=16, image_size=256)
_CFG_GIANT = VJEPA2Config(
    dim=1408,
    depth=40,
    num_heads=22,
    mlp_ratio=48.0 / 11.0,
    image_size=256,
)
_CFG_GIANT_384 = replace(_CFG_GIANT, image_size=384)


@register_model(
    task="base",
    family="vjepa2",
    model_type="vjepa2",
    model_class=VJEPA2Model,
    default_config=_CFG_LARGE,
    summary="auto",
)
def vjepa2_vit_large(
    pretrained: bool | str = False,
    *,
    weights: VJEPA2ViTLargeWeights | None = None,
    **overrides: object,
) -> VJEPA2Model:
    r"""Construct the released V-JEPA 2 ViT-L/16 backbone at 256 pixels.

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the Lucid FPC64-256 tag; a string selects a declared
        weight tag explicitly.
    weights : VJEPA2ViTLargeWeights, optional, keyword-only
        Explicit weight enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`VJEPA2Config` field overrides.

    Returns
    -------
    VJEPA2Model
    Context encoder, EMA target encoder and predictor.

    Notes
    -----
    Reference: Assran et al., arXiv:2506.09985, 2025.  The release lists
    this variant as ViT-L/16, 300M parameters, resolution 256 — that figure
    counts one encoder, while this factory builds two of them and the
    predictor.  The FPC64-256 tag reproduces its source to ``1.7e-5``
    relative, the closest of the four.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> config = AutoConfig.from_pretrained("vjepa2_vit_large")
    >>> config.dim, config.depth, config.num_heads
    (1024, 24, 16)
    >>> config.token_grid, config.num_tokens
    ((32, 16, 16), 8192)
    """
    entry = weights_mod.resolve_weights(VJEPA2ViTLargeWeights, pretrained, weights)
    model = VJEPA2Model(apply_overrides(_CFG_LARGE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="vjepa2_vit_large")
    return model


@register_model(
    task="base",
    family="vjepa2",
    model_type="vjepa2",
    model_class=VJEPA2Model,
    default_config=_CFG_HUGE,
    summary="auto",
)
def vjepa2_vit_huge(
    pretrained: bool | str = False,
    *,
    weights: VJEPA2ViTHugeWeights | None = None,
    **overrides: object,
) -> VJEPA2Model:
    r"""The released V-JEPA 2 ViT-H/16 backbone at 256 pixels.

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the Lucid FPC64-256 tag; a string selects a declared
        weight tag explicitly.
    weights : VJEPA2ViTHugeWeights, optional, keyword-only
        Explicit weight enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`VJEPA2Config` field overrides.

    Returns
    -------
    VJEPA2Model
        Context encoder, EMA target encoder and predictor.

    Notes
    -----
    Reference: Assran et al., arXiv:2506.09985, 2025.  The release lists
    ViT-H/16 at 600M parameters and resolution 256 — that figure counts one
    encoder, while this factory builds two of them and the predictor.  The
    FPC64-256 SafeTensors tag is hosted by Lucid and reproduces its source
    to ``5.3e-5`` relative.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> config = AutoConfig.from_pretrained("vjepa2_vit_huge")
    >>> config.dim, config.depth, config.num_heads
    (1280, 32, 16)
    >>> config.token_grid, config.num_tokens
    ((32, 16, 16), 8192)
    """
    entry = weights_mod.resolve_weights(VJEPA2ViTHugeWeights, pretrained, weights)
    model = VJEPA2Model(apply_overrides(_CFG_HUGE, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="vjepa2_vit_huge")
    return model


@register_model(
    task="base",
    family="vjepa2",
    model_type="vjepa2",
    model_class=VJEPA2Model,
    default_config=_CFG_GIANT,
    summary="auto",
)
def vjepa2_vit_giant(
    pretrained: bool | str = False,
    *,
    weights: VJEPA2ViTGiantWeights | None = None,
    **overrides: object,
) -> VJEPA2Model:
    r"""The released V-JEPA 2 ViT-g/16 backbone at 256 pixels.

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the Lucid FPC64-256 tag; a string selects a declared
        weight tag explicitly.
    weights : VJEPA2ViTGiantWeights, optional, keyword-only
        Explicit weight enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`VJEPA2Config` field overrides.

    Returns
    -------
    VJEPA2Model
        Context encoder, EMA target encoder and predictor.

    Notes
    -----
    Reference: Assran et al., arXiv:2506.09985, 2025.  The release lists
    ViT-g/16 at 1B parameters and resolution 256.  This is the variant the
    action-conditioned model is post-trained from, and the only one whose
    encoder widens its feed-forward to ``48 / 11`` while keeping 22
    attention heads.  The FPC64-256 tag reproduces its source to ``1.0e-4``
    relative.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> config = AutoConfig.from_pretrained("vjepa2_vit_giant")
    >>> config.dim, config.depth, config.num_heads
    (1408, 40, 22)
    >>> round(config.mlp_ratio, 4), config.predictor_mlp_ratio
    (4.3636, 4.0)
    """
    entry = weights_mod.resolve_weights(VJEPA2ViTGiantWeights, pretrained, weights)
    model = VJEPA2Model(apply_overrides(_CFG_GIANT, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="vjepa2_vit_giant")
    return model


@register_model(
    task="base",
    family="vjepa2",
    model_type="vjepa2",
    model_class=VJEPA2Model,
    default_config=_CFG_GIANT_384,
    summary="auto",
)
def vjepa2_vit_giant_384(
    pretrained: bool | str = False,
    *,
    weights: VJEPA2ViTGiant384Weights | None = None,
    **overrides: object,
) -> VJEPA2Model:
    r"""The released V-JEPA 2 ViT-g/16 backbone at 384 pixels.

    Parameters
    ----------
    pretrained : bool or str, default=False
        ``True`` loads the Lucid FPC64-384 tag; a string selects a declared
        weight tag explicitly.
    weights : VJEPA2ViTGiant384Weights, optional, keyword-only
        Explicit weight enum member; takes precedence over ``pretrained``.
    **overrides : object
        Optional :class:`VJEPA2Config` field overrides.

    Returns
    -------
    VJEPA2Model
        Context encoder, EMA target encoder and predictor.

    Notes
    -----
    Reference: Assran et al., arXiv:2506.09985, 2025.  The same network as
    :func:`vjepa2_vit_giant` evaluated at a higher resolution: the rotary
    geometry is computed from token positions rather than read from a
    learned table, so the wider grid needs no interpolation and no new
    parameters — only more tokens, 18432 against 8192.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> config = AutoConfig.from_pretrained("vjepa2_vit_giant_384")
    >>> config.image_size, config.token_grid
    (384, (32, 24, 24))
    >>> config.num_tokens
    18432
    """
    entry = weights_mod.resolve_weights(VJEPA2ViTGiant384Weights, pretrained, weights)
    model = VJEPA2Model(apply_overrides(_CFG_GIANT_384, overrides))
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="vjepa2_vit_giant_384")
    return model


@register_model(
    task="image-classification",
    family="vjepa2",
    model_type="vjepa2",
    model_class=VJEPA2ForVideoClassification,
    default_config=_CFG_LARGE,
    summary="auto",
)
def vjepa2_vit_large_cls(
    pretrained: bool = False, **overrides: object
) -> VJEPA2ForVideoClassification:
    r"""V-JEPA 2 ViT-L/16 under the paper's attentive probe.

    Parameters
    ----------
    pretrained : bool, default=False
        The probe's own weights are not published; ``True`` raises.  The
        backbone tags are reachable through :func:`vjepa2_vit_large`.
    **overrides : object
        Optional :class:`VJEPA2Config` field overrides.

    Returns
    -------
    VJEPA2ForVideoClassification
        The pretraining networks, the attentive pooler and a classifier.

    Notes
    -----
    Reference: Assran et al., arXiv:2506.09985, 2025.  The released
    classifiers read a frozen backbone through three self-attention
    blocks and one learned query, which is what ``num_pooler_layers``
    defaults to; ``num_classes`` defaults to 400, Kinetics-400's count.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> AutoConfig.from_pretrained("vjepa2_vit_large_cls").num_classes
    400
    """
    if pretrained:
        reject_unavailable_pretrained("vjepa2_vit_large_cls", alternative="vjepa2_vit_large")
    return VJEPA2ForVideoClassification(apply_overrides(_CFG_LARGE, overrides))


@register_model(
    task="image-classification",
    family="vjepa2",
    model_type="vjepa2",
    model_class=VJEPA2ForVideoClassification,
    default_config=_CFG_HUGE,
    summary="auto",
)
def vjepa2_vit_huge_cls(
    pretrained: bool = False, **overrides: object
) -> VJEPA2ForVideoClassification:
    r"""V-JEPA 2 ViT-H/16 under the paper's attentive probe.

    Parameters
    ----------
    pretrained : bool, default=False
        The probe's own weights are not published; ``True`` raises.  The
        backbone tags are reachable through :func:`vjepa2_vit_huge`.
    **overrides : object
        Optional :class:`VJEPA2Config` field overrides.

    Returns
    -------
    VJEPA2ForVideoClassification
        The pretraining networks, the attentive pooler and a classifier.

    Notes
    -----
    Reference: Assran et al., arXiv:2506.09985, 2025.  The released
    classifiers read a frozen backbone through three self-attention blocks
    and one learned query whose cross-attention carries no output
    projection — the layout ``num_pooler_layers`` defaults to, checked
    against the published ``ssv2`` classifier to ``4.3e-6``.
    ``num_classes`` defaults to 400, Kinetics-400's count.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> config = AutoConfig.from_pretrained("vjepa2_vit_huge_cls")
    >>> config.dim, config.depth, config.num_heads
    (1280, 32, 16)
    >>> config.num_pooler_layers, config.num_classes
    (3, 400)
    """
    if pretrained:
        reject_unavailable_pretrained("vjepa2_vit_huge_cls", alternative="vjepa2_vit_huge")
    return VJEPA2ForVideoClassification(apply_overrides(_CFG_HUGE, overrides))


@register_model(
    task="image-classification",
    family="vjepa2",
    model_type="vjepa2",
    model_class=VJEPA2ForVideoClassification,
    default_config=_CFG_GIANT,
    summary="auto",
)
def vjepa2_vit_giant_cls(
    pretrained: bool = False, **overrides: object
) -> VJEPA2ForVideoClassification:
    r"""V-JEPA 2 ViT-g/16 at 256 pixels under the paper's attentive probe.

    Parameters
    ----------
    pretrained : bool, default=False
        The probe's own weights are not published; ``True`` raises.  The
        backbone tags are reachable through :func:`vjepa2_vit_giant`.
    **overrides : object
        Optional :class:`VJEPA2Config` field overrides.

    Returns
    -------
    VJEPA2ForVideoClassification
        The pretraining networks, the attentive pooler and a classifier.

    Notes
    -----
    Reference: Assran et al., arXiv:2506.09985, 2025.  The released
    classifiers read a frozen backbone through three self-attention blocks
    and one learned query whose cross-attention carries no output
    projection — the layout ``num_pooler_layers`` defaults to, checked
    against the published ``ssv2`` classifier to ``4.3e-6``.
    ``num_classes`` defaults to 400, Kinetics-400's count.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> config = AutoConfig.from_pretrained("vjepa2_vit_giant_cls")
    >>> config.dim, config.depth, config.num_heads
    (1408, 40, 22)
    >>> config.num_pooler_layers, config.num_classes
    (3, 400)
    """
    if pretrained:
        reject_unavailable_pretrained("vjepa2_vit_giant_cls", alternative="vjepa2_vit_giant")
    return VJEPA2ForVideoClassification(apply_overrides(_CFG_GIANT, overrides))


@register_model(
    task="image-classification",
    family="vjepa2",
    model_type="vjepa2",
    model_class=VJEPA2ForVideoClassification,
    default_config=_CFG_GIANT_384,
    summary="auto",
)
def vjepa2_vit_giant_384_cls(
    pretrained: bool = False, **overrides: object
) -> VJEPA2ForVideoClassification:
    r"""V-JEPA 2 ViT-g/16 at 384 pixels under the paper's attentive probe.

    Parameters
    ----------
    pretrained : bool, default=False
        The probe's own weights are not published; ``True`` raises.  The
        backbone tags are reachable through :func:`vjepa2_vit_giant_384`.
    **overrides : object
        Optional :class:`VJEPA2Config` field overrides.

    Returns
    -------
    VJEPA2ForVideoClassification
        The pretraining networks, the attentive pooler and a classifier.

    Notes
    -----
    Reference: Assran et al., arXiv:2506.09985, 2025.  The released
    classifiers read a frozen backbone through three self-attention blocks
    and one learned query whose cross-attention carries no output
    projection — the layout ``num_pooler_layers`` defaults to, checked
    against the published ``ssv2`` classifier to ``4.3e-6``.
    ``num_classes`` defaults to 400, Kinetics-400's count.

    Examples
    --------
    >>> from lucid.models import AutoConfig
    >>> config = AutoConfig.from_pretrained("vjepa2_vit_giant_384_cls")
    >>> config.dim, config.depth, config.num_heads
    (1408, 40, 22)
    >>> config.num_pooler_layers, config.num_classes
    (3, 400)
    """
    if pretrained:
        reject_unavailable_pretrained(
            "vjepa2_vit_giant_384_cls", alternative="vjepa2_vit_giant_384"
        )
    return VJEPA2ForVideoClassification(apply_overrides(_CFG_GIANT_384, overrides))
