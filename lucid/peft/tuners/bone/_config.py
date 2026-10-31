"""``BoneConfig``: the config of Bone, the method MiSS replaced."""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal, override

from lucid.peft._config import PeftType, _Check, _check_rank, _choice, _TargetedConfig

_BIASES: Final[tuple[Literal["none", "all", "bone_only"], ...]] = ("none", "all", "bone_only")


def _check_init(name: str, value: object) -> bool | Literal["bat"]:
    """``True`` / ``False`` (the plain variant, zero / random start) or ``"bat"``."""
    if isinstance(value, bool):
        return value
    if value == "bat":
        return "bat"
    raise ValueError(f"`{name}` must be True, False or 'bat', got {value!r}")


@dataclass(kw_only=True)
class BoneConfig(_TargetedConfig, peft_type=PeftType.BONE):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with Bone.

    Bone (Kang, 2024 — Block Affine) is the method MiSS (:class:`~lucid.peft.MissConfig`)
    renamed and extended; the reference library no longer ships it.  It is kept here
    because adapter files written under the name still exist: a Bone adapter is a MiSS
    adapter of the balance (``init_weights=True``) or ``"bat"`` variant with the block
    stored as ``bone_block``.  New work should use MiSS.

    Parameters
    ----------
    r : int, default=64
        Shard width, as MiSS's ``r``.
    target_modules : str or list of str, optional
        The ``nn.Linear`` modules to adapt, matched as ``LoraConfig.target_modules``
        is.  ``None`` takes the LoRA default for the base model's ``model_type``.
    exclude_modules : str or list of str, optional
        Modules never adapted, matched like ``target_modules``.
    init_weights : bool or {"bat"}, default=True
        ``True``: zero blocks, the additive update.  ``False``: the same with a random
        start (for tests).  ``"bat"``: the weight-space variant.
    layers_to_transform, layers_pattern, modules_to_save
        As in ``LoraConfig``.
    bias : {"none"}, default="none"
        The reference also offered ``"all"`` and ``"bone_only"``; those raise
        ``NotImplementedError``.

    Raises
    ------
    TypeError
        A field has the wrong type.
    ValueError
        ``r`` is not positive, or ``init_weights`` is none of True, False, ``"bat"``.
    NotImplementedError
        A trained-bias mode is asked for.

    Examples
    --------
    >>> from lucid.peft.tuners.bone import BoneConfig
    >>> config = BoneConfig(r=8, target_modules=["q_proj", "v_proj"])
    >>> config.peft_type
    <PeftType.BONE: 'BONE'>
    >>> BoneConfig.from_dict(config.to_dict()) == config
    True
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "init_weights": _check_init,
        "bias": _choice(_BIASES),
    }

    r: int = 64
    init_weights: bool | Literal["bat"] = True
    bias: Literal["none", "all", "bone_only"] = "none"

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if self.bias != "none":
            raise NotImplementedError(
                f"bias={self.bias!r}: a Bone adapter in lucid.peft trains its own blocks only "
                "(bias='none')"
            )
