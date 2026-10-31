"""``MissConfig``: the MiSS hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal, override

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_dropout,
    _check_rank,
    _choice,
    _TargetedConfig,
)

_BIASES: Final[tuple[Literal["none", "all", "MiSS_only"], ...]] = ("none", "all", "MiSS_only")


def _check_init(name: str, value: object) -> bool | Literal["bat", "mini"]:
    """``True`` / ``False`` (the balance variant, zero / random start), ``"bat"`` or ``"mini"``."""
    if isinstance(value, bool):
        return value
    if value == "bat":
        return "bat"
    if value == "mini":
        return "mini"
    raise ValueError(f"`{name}` must be True, False, 'bat' or 'mini', got {value!r}")


@dataclass(kw_only=True)
class MissConfig(_TargetedConfig, peft_type=PeftType.MISS):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with MiSS.

    MiSS (Kang et al., 2024, arXiv:2409.15371 — Matrix Shard Sharing, the
    successor of Bone) learns a single ``(r, d_out)`` matrix :math:`S` per layer and
    adds it to the weight in **shards**: the input axis is cut into runs of
    :math:`r` columns, and every run of every output row receives the same
    :math:`r`-vector,

    .. math:: \Delta W_{o,c} = S_{c \bmod r,\, o}, \qquad y = W x + b + \Delta W\,x .

    So a layer costs :math:`r\,d_{out}` parameters whatever its input width, and the
    update is zero at the start (:math:`S = 0`).  Two variants change the shape of
    :math:`S`: ``"mini"`` learns an :math:`r \times m` block and tiles it along the
    output axis (:math:`m` = ``mini_r``), and ``"bat"`` (Bone's rule) learns one
    :math:`r \times r` block per run of :math:`r` output rows, applied *to the weight*
    — :math:`W' = W (I + S) + S` blockwise — so it is non-linear in the shards.

    Field names, defaults and ``adapter_config.json`` are the reference
    adapter library's.  Every field is keyword-only.

    Parameters
    ----------
    r : int, default=64
        Shard width: the rank along the input axis.  Must not exceed a layer's input
        size; the ``"bat"`` variant needs it to divide both of a layer's sizes.
    miss_dropout : float, default=0.0
        Dropout on the update's input (not used by the ``"bat"`` variant), in ``[0, 1)``.
    mini_r : int, default=1
        The ``"mini"`` variant's rank along the output axis; it must divide a layer's
        output size.
    target_modules : str or list of str, optional
        The ``nn.Linear`` modules to adapt, matched as ``LoraConfig.target_modules``
        is.  ``None`` takes the LoRA default for the base model's ``model_type``.
    exclude_modules : str or list of str, optional
        Modules never adapted, matched like ``target_modules``.
    init_weights : bool or {"bat", "mini"}, default=True
        ``True``: the balance variant, :math:`S = 0`.  ``False``: the balance variant with a
        random start (for tests).  ``"bat"`` or ``"mini"``: that variant, :math:`S = 0`.
    layers_to_transform, layers_pattern, modules_to_save
        As in ``LoraConfig``.
    bias : {"none"}, default="none"
        The reference also offers ``"all"`` and ``"MiSS_only"``; here the adapter trains
        its own blocks only, and those raise ``NotImplementedError``.

    Raises
    ------
    TypeError
        A field is not of its type.
    ValueError
        ``r``, ``mini_r`` or ``miss_dropout`` is out of range, or ``init_weights`` is not
        one of the four accepted values.
    NotImplementedError
        ``bias`` asks for trained biases.

    Examples
    --------
    >>> from lucid.peft import MissConfig
    >>> config = MissConfig(r=8, init_weights="bat", target_modules=["q_proj", "v_proj"])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.MISS: 'MISS'>)
    >>> MissConfig.from_dict(config.to_dict()) == config
    True
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "miss_dropout": _check_dropout,
        "mini_r": _check_rank,
        "init_weights": _check_init,
        "bias": _choice(_BIASES),
    }

    r: int = 64
    miss_dropout: float = 0.0
    mini_r: int = 1
    init_weights: bool | Literal["bat", "mini"] = True
    bias: Literal["none", "all", "MiSS_only"] = "none"

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if self.bias != "none":
            raise NotImplementedError(
                f"bias={self.bias!r}: a MiSS adapter in lucid.peft trains its own blocks only "
                "(bias='none')"
            )
