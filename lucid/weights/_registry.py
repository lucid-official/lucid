"""Discovery registry for pretrained-weight enums.

Maps model names → their :class:`WeightsEnum` so weights can be
discovered by string (mirroring reference_vision's ``get_weight`` /
``list_models`` helpers).  Each per-architecture weights module
registers its enum via :func:`register_weights`.
"""

from typing import Callable

from lucid.weights._base import WeightsEnum

# model factory name (e.g. "resnet_18") → its weights enum class
_WEIGHTS_BY_MODEL: dict[str, type[WeightsEnum]] = {}
# enum class name (e.g. "ResNet18Weights") → its class, for get_weight()
_WEIGHTS_BY_ENUM: dict[str, type[WeightsEnum]] = {}


def register_weights(
    model_name: str,
) -> Callable[[type[WeightsEnum]], type[WeightsEnum]]:
    r"""Class decorator registering a :class:`WeightsEnum` for discovery.

    Parameters
    ----------
    model_name : str
        The factory name the enum belongs to (e.g. ``"resnet_18"``).

    Returns
    -------
    Callable[[type[WeightsEnum]], type[WeightsEnum]]
        A decorator that records the enum under both ``model_name``
        (for :func:`list_pretrained`) and the enum's class name (for
        :func:`get_weight`), then returns the class unchanged.

    Examples
    --------
    >>> from lucid.utils.transforms import ImageClassification
    >>> from lucid.weights import (
    ...     WeightEntry,
    ...     WeightsEnum,
    ...     register_weights,
    ...     weights_for,
    ... )
    >>> @register_weights("my_net_cls")
    ... class MyNetWeights(WeightsEnum):
    ...     IMAGENET1K_V1 = WeightEntry(
    ...         url="https://example.com/my-net/IMAGENET1K_V1/model.safetensors",
    ...         sha256="0" * 64,
    ...         num_classes=1000,
    ...         transforms=ImageClassification(crop_size=224, resize_size=256),
    ...         meta={"tag": "IMAGENET1K_V1"},
    ...     )
    ...     DEFAULT = IMAGENET1K_V1
    >>> weights_for("my_net_cls") is MyNetWeights
    True
    """

    def _decorator(cls: type[WeightsEnum]) -> type[WeightsEnum]:
        _WEIGHTS_BY_MODEL[model_name] = cls
        _WEIGHTS_BY_ENUM[cls.__name__] = cls
        return cls

    return _decorator


def weights_for(model_name: str) -> type[WeightsEnum] | None:
    r"""Return the :class:`WeightsEnum` registered for ``model_name``.

    Parameters
    ----------
    model_name : str
        Factory name, e.g. ``"resnet_18"`` — the same string passed
        to :func:`register_weights` at module import time.

    Returns
    -------
    type[WeightsEnum] or None
        The registered enum class, or ``None`` when the model has
        no pretrained weights declared yet.

    Examples
    --------
    >>> import lucid.models              # registers the zoo's weights enums
    >>> import lucid.weights as W
    >>> W.weights_for("resnet_18_cls")
    <enum 'ResNet18Weights'>
    """
    return _WEIGHTS_BY_MODEL.get(model_name)


def list_pretrained(model_name: str) -> list[str]:
    r"""List the available weight tags for a model.

    Parameters
    ----------
    model_name : str
        Factory name, e.g. ``"resnet_18"``.

    Returns
    -------
    list of str
        Tag names (e.g. ``["IMAGENET1K_V1"]``).  Empty if the model has
        no registered pretrained weights.  The ``DEFAULT`` alias is not
        listed separately — enum iteration yields only canonical
        members.

    Examples
    --------
    >>> import lucid.models              # registers the zoo's weights enums
    >>> import lucid.weights as W
    >>> W.list_pretrained("resnet_18_cls")
    ['IMAGENET1K_V1']
    """
    cls = _WEIGHTS_BY_MODEL.get(model_name)
    if cls is None:
        return []
    return [member.name for member in cls]


def get_weight(name: str) -> WeightsEnum:
    r"""Resolve a dotted ``"EnumName.TAG"`` string to a weights member.

    Parameters
    ----------
    name : str
        Dotted reference, e.g. ``"ResNet18Weights.IMAGENET1K_V1"`` or
        ``"ResNet18Weights.DEFAULT"``.

    Returns
    -------
    WeightsEnum
        The matching enum member.

    Raises
    ------
    ValueError
        If the string is malformed, the enum is unknown, or the tag is
        not a member of that enum.

    Examples
    --------
    >>> import lucid.models              # registers the zoo's weights enums
    >>> import lucid.weights as W
    >>> w = W.get_weight("ResNet18Weights.IMAGENET1K_V1")
    >>> w.num_classes
    1000
    """
    enum_name, sep, tag = name.partition(".")
    if not sep or not tag:
        raise ValueError(f"get_weight: expected 'EnumName.TAG', got {name!r}")
    cls = _WEIGHTS_BY_ENUM.get(enum_name)
    if cls is None:
        raise ValueError(
            f"get_weight: unknown weights enum {enum_name!r}. " f"Known: {sorted(_WEIGHTS_BY_ENUM)}"
        )
    try:
        return cls[tag]
    except KeyError:
        raise ValueError(
            f"get_weight: {enum_name!r} has no tag {tag!r}. " f"Available: {[m.name for m in cls]}"
        ) from None
