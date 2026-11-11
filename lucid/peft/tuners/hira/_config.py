"""``HiraConfig``: the HiRA hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass, field
from typing import ClassVar, Literal

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_dropout,
    _check_rank,
    _keyed,
    _nullable,
    _TargetedConfig,
)


def _check_init(name: str, value: object) -> bool | Literal["gaussian"] | None:
    """``True`` / ``False``, ``"gaussian"`` or ``None``."""
    if value is None or isinstance(value, bool):
        return value
    if value == "gaussian":
        return "gaussian"
    raise ValueError(f"`{name}` must be True, False, 'gaussian' or None, got {value!r}")


@dataclass
class HiraConfig(_TargetedConfig, peft_type=PeftType.HIRA):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with HiRA.

    HiRA (Huang et al., 2025, ICLR — Hadamard High-Rank Adaptation) multiplies the frozen weight
    elementwise by a low-rank matrix,

    .. math:: \Delta W = W \odot (B A), \qquad A \in \mathbb{R}^{r \times d_{in}},\ B \in \mathbb{R}^{d_{out} \times r},

    so the update can have a rank far above ``r`` while costing :math:`r\,(d_{in} + d_{out})` parameters.
    :math:`B` starts at zero, and so does the update.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    r : int, default=32
        Rank of the modulating matrix.
    hira_dropout : float, default=0.0
        Dropout on the update's input, in ``[0, 1)``.
    init_weights : {True, False, "gaussian"}, optional
        ``True``: :math:`B` starts at zero, so the update does.  ``False`` (or ``None``): both
        factors start random (for tests).  ``"gaussian"``: :math:`A` starts Gaussian with standard
        deviation ``1 / r``, :math:`B` at zero.  A lookup table starts :math:`A` at zero and
        :math:`B` Gaussian for any value but ``False``.
    rank_pattern : dict of str to int, optional
        Per-module rank overrides, keyed by name or regex suffix.
    target_modules : str or list of str, optional
        The modules HiRA adapts, matched as ``LoraConfig.target_modules`` is.
    exclude_modules : str or list of str, optional
        Modules HiRA never adapts, matched like ``target_modules``.
    fan_in_fan_out : bool, default=False
        For a base that stores its weight as ``(in, out)``.  The modules adapted here store
        ``(out, in)``, so ``True`` warns and is set back to ``False`` when the adapter is made.
    modules_to_save, layers_to_transform, layers_pattern
        As in :class:`~lucid.peft.LoraConfig`.

    Examples
    --------
    >>> from lucid.peft import HiraConfig
    >>> config = HiraConfig(r=4, target_modules=['q_proj', 'v_proj'])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.HIRA: 'HIRA'>)
    >>> HiraConfig.from_dict(config.to_dict()) == config
    True

    Notes
    -----
    Every field of the reference library's config is here, in its order and with its defaults, read
    and written as ``adapter_config.json``.  A field of the wrong type raises ``TypeError``.
    ``ValueError`` is raised when ``r`` is not positive, or ``init_weights`` is not one of its
    values.  The tuner is :class:`~lucid.peft.HiraModel`; it adapts ``nn.Linear``, ``nn.Embedding``
    and ``nn.Conv1d`` / ``Conv2d`` / ``Conv3d`` modules.
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "hira_dropout": _check_dropout,
        "fan_in_fan_out": _check_bool,
        "init_weights": _check_init,
        "rank_pattern": _nullable(_keyed(_check_rank)),
    }

    r: int = 32
    target_modules: str | list[str] | set[str] | None = None
    exclude_modules: str | list[str] | set[str] | None = None
    hira_dropout: float = 0.0
    fan_in_fan_out: bool = False
    modules_to_save: list[str] | None = None
    init_weights: bool | Literal["gaussian"] | None = True
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None
    rank_pattern: dict[str, int] | None = field(default_factory=dict)
