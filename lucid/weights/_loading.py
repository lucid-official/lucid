"""Glue between a :class:`WeightsEnum` / :class:`WeightEntry` and a model.

:func:`load_weight_entry` is what a factory's ``pretrained=`` code path
calls: it downloads + verifies the checkpoint, reads it as a SafeTensors
state dict, and loads it into the model.
"""

from typing import TYPE_CHECKING

from lucid.weights._base import WeightEntry, WeightsEnum
from lucid.weights._hub import download

if TYPE_CHECKING:
    from lucid._tensor import Tensor
    from lucid.nn import Module


def resolve_weights(
    enum_cls: type[WeightsEnum],
    pretrained: bool | str,
    weights: WeightsEnum | None,
) -> WeightsEnum | None:
    r"""Resolve a factory's ``pretrained`` / ``weights`` args to a member.

    Implements the shared selection logic every pretrained factory uses,
    so the three call styles all funnel to one enum member (or ``None``
    for random init).

    Parameters
    ----------
    enum_cls : type[WeightsEnum]
        The architecture's weights enum (e.g. ``ResNet18Weights``).
    pretrained : bool or str
        ``False`` → no weights; ``True`` → the ``DEFAULT`` tag; a string
        → that specific tag (e.g. ``"IMAGENET1K_V1"``).
    weights : WeightsEnum or None
        An explicit enum member.  Takes precedence over ``pretrained``
        when given.

    Returns
    -------
    WeightsEnum or None
        The selected member, or ``None`` for random initialisation.

    Raises
    ------
    TypeError
        If ``weights`` is from a different enum than ``enum_cls``, or
        ``pretrained`` is neither bool nor str.
    ValueError
        If a string tag is not a member of ``enum_cls``.
    """
    if weights is not None:
        if not isinstance(weights, enum_cls):
            raise TypeError(
                f"resolve_weights: weights {weights!r} is not a member of " f"{enum_cls.__name__}"
            )
        return weights
    if pretrained is False:
        return None
    if pretrained is True:
        return enum_cls["DEFAULT"]
    if isinstance(pretrained, str):
        try:
            return enum_cls[pretrained]
        except KeyError:
            raise ValueError(
                f"resolve_weights: {enum_cls.__name__} has no tag "
                f"{pretrained!r}. Available: {[m.name for m in enum_cls]}"
            ) from None
    raise TypeError(
        f"resolve_weights: pretrained must be bool or str, got " f"{type(pretrained).__name__}"
    )


def load_weight_entry(
    model: Module,
    weights: WeightsEnum | WeightEntry,
    *,
    name: str,
    strict: bool = True,
) -> object:
    r"""Download the checkpoint described by ``weights`` and load it.

    Parameters
    ----------
    model : Module
        Destination model.  Its ``state_dict`` keys must match the
        checkpoint's (the SafeTensors file is produced by the
        conversion tool against this exact key layout).
    weights : WeightsEnum or WeightEntry
        The checkpoint to load — either an enum member
        (``ResNet18Weights.IMAGENET1K_V1``) or a bare
        :class:`WeightEntry`.
    name : str, keyword-only
        Cache identifier, conventionally ``"<model>/<tag>"`` so each
        checkpoint caches to its own directory.
    strict : bool, optional, keyword-only, default=True
        Forwarded to :meth:`Module.load_state_dict`.  ``False`` tolerates
        missing / unexpected keys (e.g. when swapping the head).

    Returns
    -------
    object
        The ``_IncompatibleKeys(missing_keys, unexpected_keys)`` result
        from :meth:`Module.load_state_dict`.

    Raises
    ------
    RuntimeError
        On SHA-256 verification failure, or on a key mismatch when
        ``strict=True``.

    Notes
    -----
    SafeTensors loading goes through
    :func:`lucid.serialization.load_safetensors` (the sanctioned
    external-world bridge); this module never touches numpy directly.
    """
    import lucid.serialization as _serial

    entry = weights.entry if isinstance(weights, WeightsEnum) else weights
    _assert_config_matches(model, entry, name=name)
    path = download(entry.url, entry.sha256, name=name)
    state_dict: dict[str, Tensor] = _serial.load_safetensors(str(path))  # type: ignore[assignment]
    if entry.key_map:
        missing = sorted(k for k in entry.key_map if k not in state_dict)
        if missing:
            raise RuntimeError(
                f"{name}: the entry renames {', '.join(missing)}, which the "
                f"checkpoint does not contain. The map describes a layout "
                f"the checkpoint no longer has, so loading would be a guess."
            )
        state_dict = {entry.key_map.get(k, k): v for k, v in state_dict.items()}
    return model.load_state_dict(state_dict, strict=strict)


def _assert_config_matches(model: Module, entry: WeightEntry, *, name: str) -> None:
    """Refuse a checkpoint whose architecture the model was not built for.

    Checked before the download rather than after it: a mismatch is
    settled by two integers, and finding out afterwards costs a few
    hundred megabytes to learn the same thing.

    A field the config does not have is an error too.  It means the
    entry describes a knob that has been renamed or removed, and a
    declaration that quietly matches nothing is worse than none — it
    reads as a check while checking nothing.
    """
    if not entry.requires_config:
        return

    config = getattr(model, "config", None)
    if config is None:
        raise RuntimeError(
            f"{name}: the entry declares the config it needs "
            f"({', '.join(sorted(entry.requires_config))}), and this model "
            f"carries no config to check it against."
        )

    def _comparable(value: object) -> object:
        # A tuple field round-trips through JSON and a config as a list;
        # they describe the same architecture.
        return tuple(value) if isinstance(value, (list, tuple)) else value

    missing: list[str] = []
    wrong: list[str] = []
    for key, wanted in entry.requires_config.items():
        if not hasattr(config, key):
            missing.append(key)
            continue
        got = getattr(config, key)
        if _comparable(got) != _comparable(wanted):
            wrong.append(f"{key}={got!r}, checkpoint needs {wanted!r}")

    if missing:
        raise RuntimeError(
            f"{name}: the entry requires {', '.join(sorted(missing))}, which "
            f"{type(config).__name__} does not have. The declaration is "
            f"describing a config that has since changed."
        )
    if wrong:
        raise RuntimeError(
            f"{name}: this model is built as {'; '.join(wrong)}. The "
            f"checkpoint was trained against the other one, and loading it "
            f"here would either fail on a shape or — where the field touches "
            f"no parameter — succeed and compute something else."
        )
