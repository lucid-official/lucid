"""``TinyLoraConfig``: the TinyLoRA hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass
from typing import ClassVar, Literal, override

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_dropout,
    _check_int,
    _check_number,
    _check_rank,
    _choice,
    _TargetedConfig,
)


def _check_init(name: str, value: object) -> bool | Literal["uniform"]:
    """``True`` / ``False`` or ``"uniform"``."""
    if isinstance(value, bool):
        return value
    if value == "uniform":
        return "uniform"
    raise ValueError(f"`{name}` must be True, False or 'uniform', got {value!r}")


@dataclass
class TinyLoraConfig(_TargetedConfig, peft_type=PeftType.TINYLORA):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with TinyLoRA.

    TinyLoRA (Morris et al., 2026, arXiv:2602.04118 — Learning to Reason in 13 Parameters) freezes the
    rank-``r`` SVD of each adapted weight, :math:`W \approx U \Sigma V^\top`, and trains only a vector :math:`v`
    of ``u`` numbers.  Fixed random tensors :math:`P_i` project it into an :math:`r \times r` matrix
    :math:`R = \sum_i v_i P_i`, and the update is :math:`\Delta W = U \Sigma R V^\top`.  ``weight_tying`` between
    ``0`` and ``1`` sets how much of :math:`v` the modules share; at ``1`` the whole model learns ``u`` numbers.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    r : int, default=2
        Rank of the frozen SVD.
    u : int, default=64
        Length of the trainable vector per group.
    weight_tying : float, default=0.0
        Degree of sharing of the vector across modules, in ``[0, 1]``.
    projection_seed : int, default=42
        Seed of the fixed projections.
    save_projection : bool, default=True
        Save the projections with the adapter instead of regenerating them from the seed.
    init_v_bound : float, default=0.02
        Uniform bound the vector starts from when ``init_weights`` is ``"uniform"``.
    tinylora_dropout : float, default=0.0
        Dropout on the update's input, in ``[0, 1)``.
    bias : {"none", "all", "tinylora_only"}, default="none"
        Which biases also train with TinyLoRA: none, all of them, or only those of the adapted
        layers.
    init_weights : {True, False, "uniform"}, default=True
        ``True``: :math:`v` starts at zero, so the update does.  ``"uniform"``: uniform in
        ``[-init_v_bound, init_v_bound]``.  ``False``: left uninitialised.
    target_modules : str or list of str, optional
        The modules TinyLoRA adapts, matched as ``LoraConfig.target_modules`` is.
    fan_in_fan_out : bool, default=False
        Set for a base that stores its weight as ``(in, out)``; a TinyLoRA layer follows that
        layout.
    modules_to_save, layers_to_transform, layers_pattern
        As in :class:`~lucid.peft.LoraConfig`.

    Examples
    --------
    >>> from lucid.peft import TinyLoraConfig
    >>> config = TinyLoraConfig(r=2, u=16, target_modules=['q_proj', 'v_proj'])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.TINYLORA: 'TINYLORA'>)
    >>> TinyLoraConfig.from_dict(config.to_dict()) == config
    True

    Notes
    -----
    TinyLoRA's config is complete: every field of the reference library's, in its order and with its
    defaults, read and written as ``adapter_config.json``.  A field of the wrong type raises
    ``TypeError``.  ``ValueError`` is raised when ``r`` or ``u`` is not positive, or
    ``weight_tying`` is outside ``[0, 1]``.  No tuner applies it yet: ``get_peft_model`` raises
    ``NotImplementedError`` naming LCD-824, which tracks it.
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "u": _check_rank,
        "weight_tying": _check_number,
        "projection_seed": _check_int,
        "save_projection": _check_bool,
        "init_v_bound": _check_number,
        "tinylora_dropout": _check_dropout,
        "fan_in_fan_out": _check_bool,
        "bias": _choice(("none", "all", "tinylora_only")),
        "init_weights": _check_init,
    }

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if not 0.0 <= self.weight_tying <= 1.0:
            raise ValueError(
                f"`weight_tying` should be a float between 0.0 and 1.0 but the value passed is {self.weight_tying}"
            )

    r: int = 2
    u: int = 64
    weight_tying: float = 0.0
    projection_seed: int = 42
    save_projection: bool = True
    init_v_bound: float = 0.02
    target_modules: str | list[str] | set[str] | None = None
    tinylora_dropout: float = 0.0
    fan_in_fan_out: bool = False
    bias: Literal["none", "all", "tinylora_only"] = "none"
    modules_to_save: list[str] | None = None
    init_weights: bool | str = True
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None
