"""``FrodConfig``: the FRoD hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass
from typing import ClassVar, Literal, override

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_dropout,
    _check_int,
    _check_number,
    _choice,
    _TargetedConfig,
)


@dataclass
class FrodConfig(_TargetedConfig, peft_type=PeftType.FROD):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with FRoD.

    FRoD (Wan et al., 2025, arXiv:2512.23485 — Full-Rank Efficient Fine-Tuning with Rotational Degrees)
    reaches a full-rank update with few parameters.  A basis shared by the layers of one kind is built from the
    base weights, and a sparse trainable rotation matrix, with a fraction ``sparse_rate`` of its off-diagonal
    entries free, turns that basis into the weight update.  The sparsity pattern is drawn from
    ``projection_prng_key``.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    projection_prng_key : int, default=0
        Seed of the sparsity pattern.
    save_projection : bool, default=True
        Save the projection tensors with the adapter instead of rebuilding them on load.
    frod_dropout : float, default=0.0
        Dropout in the adapter layers, in ``[0, 1)``.
    bias : {"none", "all", "frod_only"}, default="none"
        Which biases also train with FRoD: none, all of them, or only those of the adapted
        layers.
    init_weights : bool, default=True
        Use the default initialisation (``True``) or leave the rotation random.
    sparse_rate : float, default=0.01
        Fraction of the rotation's off-diagonal entries that are trainable, in ``[0, 1]``.
    regularization_alpha : float, default=0.001
        Small positive value that stabilises the matrix inverses when the shared basis is built.
    progressbar : bool, default=True
        Show a progress bar while the projections are built (slow on a large model).
    runtime_offload_base_weight : bool, default=False
        Keep target base weights off the accelerator while the adapter does not need them.
    target_modules : str or list of str, optional
        The modules FRoD adapts, matched as ``LoraConfig.target_modules`` is.
    fan_in_fan_out : bool, default=False
        Set for a base that stores its weight as ``(in, out)``; a FRoD layer follows that
        layout.
    modules_to_save, layers_to_transform, layers_pattern
        As in :class:`~lucid.peft.LoraConfig`.

    Examples
    --------
    >>> from lucid.peft import FrodConfig
    >>> config = FrodConfig(sparse_rate=0.05, target_modules=['q_proj', 'v_proj'])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.FROD: 'FROD'>)
    >>> FrodConfig.from_dict(config.to_dict()) == config
    True

    Notes
    -----
    FRoD's config is complete: every field of the reference library's, in its order and with its
    defaults, read and written as ``adapter_config.json``.  A field of the wrong type raises
    ``TypeError``.  ``ValueError`` is raised when ``sparse_rate`` is outside ``[0, 1]``.  No tuner
    applies it yet: ``get_peft_model`` raises ``NotImplementedError`` naming LCD-825, which tracks
    it.
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "projection_prng_key": _check_int,
        "save_projection": _check_bool,
        "frod_dropout": _check_dropout,
        "fan_in_fan_out": _check_bool,
        "bias": _choice(("none", "all", "frod_only")),
        "init_weights": _check_bool,
        "sparse_rate": _check_number,
        "regularization_alpha": _check_number,
        "progressbar": _check_bool,
        "runtime_offload_base_weight": _check_bool,
    }

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if not 0 <= self.sparse_rate <= 1:
            raise ValueError(f"`sparse_rate` should be between 0 and 1, got {self.sparse_rate}.")

    target_modules: str | list[str] | set[str] | None = None
    projection_prng_key: int = 0
    save_projection: bool = True
    frod_dropout: float = 0.0
    fan_in_fan_out: bool = False
    bias: Literal["none", "all", "frod_only"] = "none"
    modules_to_save: list[str] | None = None
    init_weights: bool = True
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None
    sparse_rate: float = 0.01
    regularization_alpha: float = 0.001
    progressbar: bool = True
    runtime_offload_base_weight: bool = False
