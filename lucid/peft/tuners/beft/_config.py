"""``BeftConfig``: the BEFT hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.peft._config import PeftType, _Check, _check_bool, _TargetedConfig


@dataclass
class BeftConfig(_TargetedConfig, peft_type=PeftType.BEFT):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with BEFT.

    BEFT (Huang et al., 2025, arXiv:2509.15974 — Bias-Efficient Fine-Tuning of Language Models in
    Low-Data Regimes) trains nothing but a bias added to chosen linear layers, the value projections by default:
    :math:`y = W x + b + \beta` with one learned vector :math:`\beta` of the layer's output width, zero at the
    start.  It adds the fewest parameters of any method here.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    init_weights : bool, default=True
        Start the added bias at zero (``True``) or at random.
    target_modules : str or list of str, optional
        The modules BEFT adapts, matched as ``LoraConfig.target_modules`` is.
    modules_to_save
        As in :class:`~lucid.peft.LoraConfig`.

    Examples
    --------
    >>> from lucid.peft import BeftConfig
    >>> config = BeftConfig(target_modules=['q_proj', 'v_proj'])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.BEFT: 'BEFT'>)
    >>> BeftConfig.from_dict(config.to_dict()) == config
    True

    Notes
    -----
    BEFT's config is complete: every field of the reference library's, in its order and with its
    defaults, read and written as ``adapter_config.json``.  ``TypeError`` is raised when a field
    holds a value of the wrong type.  No tuner applies it yet: ``get_peft_model`` raises
    ``NotImplementedError`` naming LCD-826, which tracks it.
    """

    _field_checks: ClassVar[dict[str, _Check]] = {"init_weights": _check_bool}

    target_modules: str | list[str] | set[str] | None = None
    modules_to_save: list[str] | None = None
    init_weights: bool = True
