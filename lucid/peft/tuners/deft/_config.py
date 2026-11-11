"""``DeftConfig``: the DEFT hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass
from typing import ClassVar, Literal

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_dropout,
    _check_number,
    _check_rank,
    _choice,
    _nullable,
    _TargetedConfig,
)


@dataclass
class DeftConfig(_TargetedConfig, peft_type=PeftType.DEFT):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with DEFT.

    DEFT (Kumar et al., 2025, arXiv:2509.22793 — Decompositional Efficient Fine-Tuning) injects
    knowledge through a residual projection.  For a frozen weight :math:`W` it learns a direction matrix
    :math:`P \in \mathbb{R}^{d_{out} \times r}` and an injection matrix :math:`R \in \mathbb{R}^{r \times d_{in}}`
    and uses

    .. math:: W' = (I - P_{proj})\,W + Q_P R ,

    removing the subspace :math:`P_{proj}` from the weight and writing :math:`R` into the span of
    :math:`Q_P`.  The projector is :math:`P\,\mathrm{relu}(P)^\top` (``"relu"``) or :math:`Q_P Q_P^\top` from a QR
    decomposition of :math:`P` (``"qr"``).  With ``para=True`` it is pure subspace removal (PaRa).

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    r : int, default=8
        Rank of the projection and the injection.
    decomposition_method : {"relu", "qr"}, default="relu"
        How the projector is derived from :math:`P`: ``"relu"`` (not orthogonal) or ``"qr"``
        (orthogonal).
    init_scale : float, default=1.0
        Scale of the standard deviation :math:`R` starts from.
    alpha : float, optional
        Scales the injection by ``alpha / r``; ``None`` leaves it unscaled.
    para : bool, default=False
        Pure subspace removal (PaRa): no injection matrix, and the adapter cannot start as the
        identity.
    deft_dropout : float, default=0.0
        Dropout on the layer's input, in ``[0, 1)``.
    init_weights : bool, default=True
        Start the adapter as the identity (``True``) or at random.
    bias : {"none", "all", "deft_only"}, default="none"
        Which biases also train with DEFT: none, all of them, or only those of the adapted
        layers.
    target_modules : str or list of str, optional
        The modules DEFT adapts, matched as ``LoraConfig.target_modules`` is.
    exclude_modules : str or list of str, optional
        Modules DEFT never adapts, matched like ``target_modules``.
    fan_in_fan_out : bool, default=False
        Set for a base that stores its weight as ``(in, out)``; a DEFT layer follows that
        layout.
    layers_to_transform, layers_pattern, modules_to_save
        As in :class:`~lucid.peft.LoraConfig`.

    Examples
    --------
    >>> from lucid.peft import DeftConfig
    >>> config = DeftConfig(r=4, decomposition_method='qr', target_modules=['q_proj', 'v_proj'])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.DEFT: 'DEFT'>)
    >>> DeftConfig.from_dict(config.to_dict()) == config
    True

    Notes
    -----
    DEFT's config is complete: every field of the reference library's, in its order and with its
    defaults, read and written as ``adapter_config.json``.  A field of the wrong type raises
    ``TypeError``.  ``ValueError`` is raised when ``r`` is not positive, or ``decomposition_method``
    is not one of its two values.  No tuner applies it yet: ``get_peft_model`` raises
    ``NotImplementedError`` naming LCD-823, which tracks it.
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "decomposition_method": _choice(("relu", "qr")),
        "init_scale": _check_number,
        "alpha": _nullable(_check_number),
        "para": _check_bool,
        "fan_in_fan_out": _check_bool,
        "deft_dropout": _check_dropout,
        "init_weights": _check_bool,
        "bias": _choice(("none", "all", "deft_only")),
    }

    r: int = 8
    target_modules: str | list[str] | set[str] | None = None
    exclude_modules: str | list[str] | set[str] | None = None
    decomposition_method: Literal["relu", "qr"] = "relu"
    init_scale: float = 1.0
    alpha: float | None = None
    para: bool = False
    fan_in_fan_out: bool = False
    deft_dropout: float = 0.0
    init_weights: bool = True
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None
    bias: Literal["none", "all", "deft_only"] = "none"
    modules_to_save: list[str] | None = None
