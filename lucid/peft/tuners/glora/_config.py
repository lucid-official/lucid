"""``GloraConfig``: the GLoRA hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass
from typing import ClassVar, Literal

from lucid.peft._config import PeftType, _Check, _check_bool, _check_rank, _choice, _TargetedConfig


@dataclass
class GloraConfig(_TargetedConfig, peft_type=PeftType.GLORA):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with GLoRA.

    GLoRA (Chavan et al., 2023, arXiv:2306.07967 — Generalized LoRA) corrects the weight and the bias
    of a layer with learned terms,

    .. math:: W' = W + W A + B, \qquad b' = b + b \odot D + E + W C .

    Each of :math:`A, B, C, D, E` is a low-rank product, a vector, a scalar or switched off
    (``config_A_B``, ``config_C``, ``config_D_E``), so one family covers LoRA-like, scale-and-shift and
    prompt-like adapters.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    r : int, default=8
        Rank of the parts written as low-rank products.
    bias : {"none", "all", "glora_only"}, default="none"
        The reference trains none, all, or only the adapted layers' biases.  A GLoRA adapter
        here trains its own tensors only, so ``"all"`` and ``"glora_only"`` raise
        ``NotImplementedError``.
    config_A_B : {"lora", "vector", "constant", "none"}, default="lora"
        Parameterisation of :math:`A` and :math:`B`: a low-rank product, one value per output
        channel, one scalar, or off.
    config_C : {"lora", "vector", "none"}, default="lora"
        Parameterisation of :math:`C`, which couples the bias to the weight: a low-rank column,
        a vector, or off.
    config_D_E : {"vector", "constant", "none"}, default="constant"
        Parameterisation of :math:`D` and :math:`E`, the bias's own corrections: a vector, one
        scalar, or off (never low-rank).
    init_weights : bool, default=True
        ``True``: the adapter is the identity at the start.  ``False``: a Kaiming start.
    target_modules : str or list of str, optional
        The modules GLoRA adapts, matched as ``LoraConfig.target_modules`` is.
    modules_to_save
        As in :class:`~lucid.peft.LoraConfig`.

    Examples
    --------
    >>> from lucid.peft import GloraConfig
    >>> config = GloraConfig(r=4, config_C='vector', target_modules=['q_proj', 'v_proj'])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.GLORA: 'GLORA'>)
    >>> GloraConfig.from_dict(config.to_dict()) == config
    True

    Notes
    -----
    Every field of the reference library's config is here, in its order and with its defaults, read
    and written as ``adapter_config.json``.  A field of the wrong type raises ``TypeError``.
    ``ValueError`` is raised when ``r`` is not positive, or a ``config_*`` field names an unknown
    parameterisation.  ``bias`` other than ``"none"`` raises ``NotImplementedError``.  The tuner is
    :class:`~lucid.peft.GloraModel`; it adapts ``nn.Linear`` modules, and a module without a bias
    has no :math:`C`, :math:`D` or :math:`E`.
    """

    _unsupported: ClassVar[dict[str, str]] = {
        "bias": "a GLoRA adapter trains its own tensors only (bias='none')"
    }

    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "bias": _choice(("none", "all", "glora_only")),
        "config_A_B": _choice(("lora", "vector", "constant", "none")),
        "config_C": _choice(("lora", "vector", "none")),
        "config_D_E": _choice(("vector", "constant", "none")),
        "init_weights": _check_bool,
    }

    r: int = 8
    target_modules: str | list[str] | set[str] | None = None
    bias: Literal["none", "all", "glora_only"] = "none"
    modules_to_save: list[str] | None = None
    config_A_B: Literal["lora", "vector", "constant", "none"] = "lora"
    config_C: Literal["lora", "vector", "none"] = "lora"
    config_D_E: Literal["vector", "constant", "none"] = "constant"
    init_weights: bool = True
