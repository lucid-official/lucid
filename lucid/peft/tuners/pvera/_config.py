"""``PveraConfig``: the PVeRA hyper-parameters on top of the shared targeting fields."""

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
    _nullable,
    _TargetedConfig,
)


@dataclass
class PveraConfig(_TargetedConfig, peft_type=PeftType.PVERA):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with PVeRA.

    PVeRA (Fillioux et al., 2025, arXiv:2512.07703 — Probabilistic Vector-Based Random Matrix
    Adaptation) makes VeRA probabilistic.  As in VeRA, two frozen random projections shared by all layers and two
    learned vectors per layer give the update; but the rank-``r`` latent between them is the mean and
    log-variance of a Gaussian, sampled in training in the manner of a variational encoder.  At inference the
    learned mean is used unless ``sample_at_inference`` is set.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    r : int, default=256
        Rank of the shared projections; larger than a LoRA rank, since they are shared.
    projection_prng_key : int, default=0
        Seed of the shared projections.
    save_projection : bool, default=True
        Save the projections with the adapter, so it reloads on any machine.
    pvera_dropout : float, default=0.0
        Dropout on the update's input, in ``[0, 1)``.
    d_initial : float, default=0.1
        Initial value of the ``d`` vector.
    bias : {"none", "all", "pvera_only"}, default="none"
        Which biases also train with PVeRA: none, all of them, or only those of the adapted
        layers.
    init_weights : bool, default=True
        Use the default initialisation (``True``) or leave the vectors random.
    sample_at_inference : bool, default=False
        Sample the latent at inference too; ``False`` uses its mean.
    generator_seed : int, optional
        Seed of the sampler; ``None`` leaves it random.
    target_modules : str or list of str, optional
        The modules PVeRA adapts, matched as ``LoraConfig.target_modules`` is.
    fan_in_fan_out : bool, default=False
        Set for a base that stores its weight as ``(in, out)``; a PVeRA layer follows that
        layout.
    modules_to_save, layers_to_transform, layers_pattern
        As in :class:`~lucid.peft.LoraConfig`.

    Examples
    --------
    >>> from lucid.peft import PveraConfig
    >>> config = PveraConfig(r=16, target_modules=['q_proj', 'v_proj'])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.PVERA: 'PVERA'>)
    >>> PveraConfig.from_dict(config.to_dict()) == config
    True

    Notes
    -----
    PVeRA's config is complete: every field of the reference library's, in its order and with its
    defaults, read and written as ``adapter_config.json``.  A field of the wrong type raises
    ``TypeError``.  ``ValueError`` is raised when ``r`` is not positive, or ``layers_pattern`` is
    given without ``layers_to_transform``.  No tuner applies it yet: ``get_peft_model`` raises
    ``NotImplementedError`` naming LCD-824, which tracks it.
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "projection_prng_key": _check_int,
        "save_projection": _check_bool,
        "pvera_dropout": _check_dropout,
        "d_initial": _check_number,
        "fan_in_fan_out": _check_bool,
        "bias": _choice(("none", "all", "pvera_only")),
        "init_weights": _check_bool,
        "sample_at_inference": _check_bool,
        "generator_seed": _nullable(_check_int),
    }

    r: int = 256
    target_modules: str | list[str] | set[str] | None = None
    projection_prng_key: int = 0
    save_projection: bool = True
    pvera_dropout: float = 0.0
    d_initial: float = 0.1
    fan_in_fan_out: bool = False
    bias: Literal["none", "all", "pvera_only"] = "none"
    modules_to_save: list[str] | None = None
    init_weights: bool = True
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None
    sample_at_inference: bool = False
    generator_seed: int | None = None
