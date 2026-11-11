"""``AdamssConfig``: the AdaMSS hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass
from typing import ClassVar, Literal, override

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_int,
    _check_number,
    _check_rank,
    _TargetedConfig,
)


def _check_init(name: str, value: object) -> Literal["orthogonal"] | None:
    """``"orthogonal"``, or ``None``; ``False`` reads as ``None``."""
    if value is None or value is False:
        return None
    if value == "orthogonal":
        return "orthogonal"
    raise ValueError(f"`{name}` must be 'orthogonal' or None, got {value!r}")


@dataclass
class AdamssConfig(_TargetedConfig, peft_type=PeftType.ADAMSS):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with AdaMSS.

    AdaMSS (Zheng et al., 2025, NeurIPS — Adaptive Multi-Subspace Approach for Parameter-Efficient
    Fine-Tuning) takes the SVD of each weight (joined to its bias as one more column), clusters the rows of its
    top-``r`` left singular directions into ``num_subspaces`` subspaces, and learns an independent
    rank-``subspace_rank`` update inside each one while the weight stays frozen.  With ``use_asa``, an adaptive
    allocation freezes the less important subspaces during training, down to about ``asa_target_subspaces``
    trainable ones in all.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    r : int, default=100
        Total rank of the SVD (``R`` in the paper), clamped to the smaller dimension of the weight joined to its bias.
    num_subspaces : int, default=5
        Subspaces :math:`K` the SVD space is clustered into; with ASA, the number before
        pruning.
    subspace_rank : int, default=1
        Rank of the update in each subspace, capped by the subspace's rows and by ``r``.
    init_weights : {"orthogonal"}, optional
        ``"orthogonal"``: each subspace's :math:`A` starts with orthonormal rows and its :math:`B` at zero,
        so the adapter starts as the identity.  ``None`` (or ``False``): :math:`B` starts as a small random
        draw (standard deviation 0.01, from ``random_seed``), so the adapter already changes the output.
    use_asa : bool, default=False
        Enable Adaptive Subspace Allocation, which freezes subspaces during training.
    asa_target_subspaces : int, default=50
        Trainable subspaces across all layers that ASA freezes down to (it keeps one fewer, the reference's
        rule: a subspace stays only if its score is strictly above the ``asa_target_subspaces``-th highest);
        positive with ``use_asa``.
    init_warmup : int, default=50
        Steps before pruning starts; below ``final_warmup`` with ``use_asa``.
    final_warmup : int, default=1000
        Step at which pruning reaches ``asa_target_subspaces``.
    mask_interval : int, default=100
        Steps between mask updates; positive with ``use_asa``.
    asa_importance_beta : float, default=0.85
        EMA coefficient of the importance scores.
    asa_uncertainty_beta : float, default=0.85
        EMA coefficient of the importance uncertainty.
    asa_schedule_exponent : float, default=3.0
        Exponent of the pruning schedule; positive with ``use_asa``.
    use_dynamic_rank : bool, default=False
        Set each subspace's rank by counting singular values above ``svd_threshold``.
    svd_threshold : float, default=0.1
        Fraction of the largest singular value a singular value must exceed to count (dynamic
        rank).
    random_seed : int, default=0
        Seed that rebuilds the adapter deterministically.
    target_modules : str or list of str, optional
        The modules AdaMSS adapts, matched as ``LoraConfig.target_modules`` is.
    modules_to_save, layers_to_transform, layers_pattern
        As in :class:`~lucid.peft.LoraConfig`.

    Examples
    --------
    >>> from lucid.peft import AdamssConfig
    >>> config = AdamssConfig(r=50, num_subspaces=4, target_modules=['q_proj', 'v_proj'])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.ADAMSS: 'ADAMSS'>)
    >>> AdamssConfig.from_dict(config.to_dict()) == config
    True

    Notes
    -----
    Every field of the reference library's config, in its order and with its defaults, is read and
    written as ``adapter_config.json``.  A field of the wrong type raises ``TypeError``.
    ``ValueError`` is raised when a rank is not positive, ``init_weights`` is unknown, or the ASA
    schedule is inconsistent.  The tuner that applies it is :class:`~lucid.peft.AdamssModel`.
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "num_subspaces": _check_rank,
        "subspace_rank": _check_rank,
        "init_weights": _check_init,
        "use_asa": _check_bool,
        "asa_target_subspaces": _check_int,
        "init_warmup": _check_int,
        "final_warmup": _check_int,
        "mask_interval": _check_int,
        "asa_importance_beta": _check_number,
        "asa_uncertainty_beta": _check_number,
        "asa_schedule_exponent": _check_number,
        "use_dynamic_rank": _check_bool,
        "svd_threshold": _check_number,
        "random_seed": _check_int,
    }

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if not self.use_asa:
            return
        if self.mask_interval <= 0:
            raise ValueError(
                f"`mask_interval` must be a positive integer when `use_asa=True`, got {self.mask_interval}."
            )
        if self.init_warmup >= self.final_warmup:
            raise ValueError(
                "`init_warmup` must be smaller than `final_warmup` when `use_asa=True`, got "
                f"init_warmup={self.init_warmup} and final_warmup={self.final_warmup}."
            )
        if self.asa_target_subspaces <= 0:
            raise ValueError(
                "`asa_target_subspaces` must be a positive integer when `use_asa=True`, "
                f"got {self.asa_target_subspaces}."
            )
        if self.asa_schedule_exponent <= 0:
            raise ValueError(
                "`asa_schedule_exponent` must be a positive number when `use_asa=True`, "
                f"got {self.asa_schedule_exponent}."
            )

    r: int = 100
    num_subspaces: int = 5
    subspace_rank: int = 1
    target_modules: str | list[str] | set[str] | None = None
    init_weights: Literal["orthogonal"] | None = "orthogonal"
    modules_to_save: list[str] | None = None
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None
    use_asa: bool = False
    asa_target_subspaces: int = 50
    init_warmup: int = 50
    final_warmup: int = 1000
    mask_interval: int = 100
    asa_importance_beta: float = 0.85
    asa_uncertainty_beta: float = 0.85
    asa_schedule_exponent: float = 3.0
    use_dynamic_rank: bool = False
    svd_threshold: float = 0.1
    random_seed: int = 0
