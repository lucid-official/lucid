"""``UniLoraConfig``: the Uni-LoRA hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass
from typing import ClassVar, Literal

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


def _check_positive(name: str, value: object) -> float:
    number = _check_number(name, value)
    if number <= 0:
        raise ValueError(f"`{name}` should be a positive float, got {value!r}.")
    return number


@dataclass
class UniLoraConfig(_TargetedConfig, peft_type=PeftType.UNILORA):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with Uni-LoRA.

    Uni-LoRA (Li et al., 2025, arXiv:2506.00799 — One Vector Is All You Need) reads every LoRA factor of
    every adapted layer out of a single trainable vector :math:`\theta_d` of ``theta_d_length`` entries.  A
    fixed projection, drawn once from ``proj_seed``, assigns each entry of the flattened factors one scaled
    entry of :math:`\theta_d`, so the adapter of the whole model is ``theta_d_length`` numbers.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    r : int, default=4
        Rank of the low-rank factors read out of the vector.
    proj_seed : int, default=42
        Seed of the fixed index assignment.
    theta_d_length : int, default=256
        Length of the shared trainable vector.
    unilora_dropout : float, default=0.0
        Dropout on the update's input, in ``[0, 1)``.
    bias : {"none", "all", "unilora_only"}, default="none"
        Which biases also train with Uni-LoRA: none, all of them, or only those of the adapted
        layers.
    init_theta_d_bound : float, default=0.02
        The vector starts uniform in ``[-bound, bound]``; positive.
    init_weights : bool, default=True
        Use the default initialisation of the vector (``True``) or leave it random.
    save_indices : bool, default=False
        Also save the index and scale buffers of each layer, so an adapter does not depend on how
        they are drawn.  Each adapter of a model chooses for itself, as in the reference: the file of
        one that does not save them holds the vector alone, whatever the others do.
    target_modules : str or list of str, optional
        The modules Uni-LoRA adapts, matched as ``LoraConfig.target_modules`` is.
    fan_in_fan_out : bool, default=False
        Set for a base that stores its weight as ``(in, out)``; a Uni-LoRA layer follows that
        layout.
    modules_to_save, layers_to_transform, layers_pattern
        As in :class:`~lucid.peft.LoraConfig`.

    Examples
    --------
    >>> from lucid.peft import UniLoraConfig
    >>> config = UniLoraConfig(r=4, theta_d_length=128, target_modules=['q_proj', 'v_proj'])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.UNILORA: 'UNILORA'>)
    >>> UniLoraConfig.from_dict(config.to_dict()) == config
    True

    Notes
    -----
    The config reads and writes ``adapter_config.json`` as the reference library's does: every field,
    in its order and with its defaults.  A field of the wrong type raises ``TypeError``; ``ValueError``
    is raised when ``r``, ``theta_d_length`` or ``init_theta_d_bound`` is not positive.  The fixed
    projection is drawn from ``proj_seed`` by this library's generator, so an adapter file written
    with ``save_indices=False`` by another library reads its indices back as this one draws them;
    ``save_indices=True`` carries them (and their scales) in the file.  ``bias`` other than ``"none"``
    raises ``NotImplementedError``.  The tuner is :class:`~lucid.peft.UniLoraModel`; it adapts
    ``nn.Linear`` modules.
    """

    _unsupported: ClassVar[dict[str, str]] = {
        "bias": "a Uni-LoRA adapter trains its own tensors only (bias='none')"
    }

    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "proj_seed": _check_int,
        "theta_d_length": _check_rank,
        "unilora_dropout": _check_dropout,
        "fan_in_fan_out": _check_bool,
        "bias": _choice(("none", "all", "unilora_only")),
        "init_theta_d_bound": _check_positive,
        "init_weights": _check_bool,
        "save_indices": _check_bool,
    }

    r: int = 4
    proj_seed: int = 42
    theta_d_length: int = 256
    target_modules: str | list[str] | set[str] | None = None
    unilora_dropout: float = 0.0
    fan_in_fan_out: bool = False
    bias: Literal["none", "all", "unilora_only"] = "none"
    modules_to_save: list[str] | None = None
    init_theta_d_bound: float = 0.02
    init_weights: bool = True
    save_indices: bool = False
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None
