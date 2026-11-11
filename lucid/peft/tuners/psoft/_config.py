"""``PsoftConfig``: the PSOFT hyper-parameters on top of the shared targeting fields."""

import warnings
from dataclasses import dataclass
from typing import ClassVar, Literal, override

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_count,
    _check_dropout,
    _check_int,
    _check_number,
    _check_rank,
    _choice,
    _nullable,
    _TargetedConfig,
)


@dataclass
class PsoftConfig(_TargetedConfig, peft_type=PeftType.PSOFT):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with PSOFT.

    PSOFT (Wu et al., 2025, arXiv:2505.11235 — Efficient Orthogonal Fine-Tuning with Principal Subspace
    Adaptation, section 4) rotates a weight inside its principal subspace only.  An SVD of the weight gives its top-``r``
    directions as frozen factors :math:`A` and :math:`B`, and only an :math:`r \times r` matrix :math:`R` between them
    is trained:

    .. math:: W' = W_{res} + A\,R\,B ,

    with :math:`W_{res}` the part of the weight outside the subspace.  :math:`R` is kept orthogonal by a Cayley
    transform (``psoft_orth``) and may carry diagonal magnitude vectors on either side (``psoft_mag_a``,
    ``psoft_mag_b``).  It starts at the identity, so the adapted weight starts as the original.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    r : int, default=32
        Rank of the principal subspace, and size of the trained matrix :math:`R`.
    psoft_alpha : float, default=32
        Scaling factor, with the meaning of LoRA's ``lora_alpha``.
    psoft_dropout : float, default=0.0
        Dropout on the update path, in ``[0, 1)``.
    ab_svd_init : {"psoft_init", "pissa_init"}, default="psoft_init"
        How :math:`A` and :math:`B` are built from the SVD: ``"psoft_init"`` with row-orthogonal
        :math:`A` (strictly orthogonal), or ``"pissa_init"`` with symmetric factors (as PiSSA).
    psoft_svd : {"full", "lowrank"}, default="full"
        SVD backend of the initialisation: the full decomposition or a randomised low-rank one.
    psoft_svd_lowrank_niter : int, default=10
        Power iterations of the low-rank SVD; used with ``psoft_svd="lowrank"`` only.
    random_seed : int, default=0
        Seed that rebuilds the adapter deterministically when the SVD is low-rank.
    psoft_orth : bool, default=True
        Constrain :math:`R` to be orthogonal by a Cayley parameterisation; ``False`` leaves it
        free.
    psoft_mag_b : bool, default=True
        Learn a diagonal scaling on the output side of :math:`R`.
    psoft_mag_a : bool, default=True
        Learn a diagonal scaling on the input side of :math:`R`.
    use_cayley_neumann : bool, default=False
        Approximate the Cayley transform by a Neumann series: faster, with an orthogonality
        error.
    num_cayley_neumann_terms : int, default=5
        Terms of that series; positive when ``use_cayley_neumann`` is set.
    cayley_neumann_eps : float, optional
        Frobenius-norm bound in ``(0, 1)`` on the generator, which is rescaled when it exceeds
        it; ``None`` applies none.  Used with ``use_cayley_neumann`` only.
    init_weights : bool, default=True
        Use the default initialisation (``R`` is the identity).
    target_modules : str or list of str, optional
        The modules PSOFT adapts, matched as ``LoraConfig.target_modules`` is.
    exclude_modules : str or list of str, optional
        Modules PSOFT never adapts, matched like ``target_modules``.
    fan_in_fan_out : bool, default=False
        Set for a base that stores its weight as ``(in, out)``; a PSOFT layer follows that
        layout.
    modules_to_save, layers_to_transform, layers_pattern
        As in :class:`~lucid.peft.LoraConfig`.

    Examples
    --------
    >>> from lucid.peft import PsoftConfig
    >>> config = PsoftConfig(r=8, psoft_mag_a=False, target_modules=['q_proj', 'v_proj'])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.PSOFT: 'PSOFT'>)
    >>> PsoftConfig.from_dict(config.to_dict()) == config
    True

    Notes
    -----
    Field names, their order, defaults and ``adapter_config.json`` are the reference library's.  A
    field of the wrong type raises ``TypeError``.  ``ValueError`` is raised when ``r`` is not
    positive, a choice field is unknown, or the Cayley-Neumann options are out of range; an option
    that the chosen SVD backend or transform does not read (``psoft_svd_lowrank_niter`` without
    ``psoft_svd="lowrank"``, the Neumann options without ``use_cayley_neumann``) warns that it is
    ignored.  ``PsoftModel`` also needs ``r`` to be at most the smaller side of each adapted weight.
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "psoft_alpha": _check_number,
        "psoft_dropout": _check_dropout,
        "fan_in_fan_out": _check_bool,
        "ab_svd_init": _choice(("psoft_init", "pissa_init")),
        "psoft_svd": _choice(("full", "lowrank")),
        "psoft_svd_lowrank_niter": _check_count,
        "random_seed": _check_int,
        "psoft_orth": _check_bool,
        "psoft_mag_b": _check_bool,
        "psoft_mag_a": _check_bool,
        "use_cayley_neumann": _check_bool,
        "num_cayley_neumann_terms": _check_int,
        "cayley_neumann_eps": _nullable(_check_number),
        "init_weights": _check_bool,
    }

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        fields = type(self).__dataclass_fields__
        if (
            self.psoft_svd != "lowrank"
            and self.psoft_svd_lowrank_niter != fields["psoft_svd_lowrank_niter"].default
        ):
            warnings.warn(
                "`psoft_svd_lowrank_niter` is only used when `psoft_svd='lowrank'`. "
                f"Got psoft_svd={self.psoft_svd!r}, so psoft_svd_lowrank_niter="
                f"{self.psoft_svd_lowrank_niter} will be ignored.",
                UserWarning,
                stacklevel=3,
            )
        if self.use_cayley_neumann:
            if self.num_cayley_neumann_terms <= 0:
                raise ValueError(
                    f"`num_cayley_neumann_terms` must be a positive integer; got {self.num_cayley_neumann_terms}."
                )
            eps = self.cayley_neumann_eps
            if eps is not None and not 0.0 < eps < 1.0:
                raise ValueError(f"`cayley_neumann_eps` must be in (0, 1) when set; got {eps}.")
            return
        if self.num_cayley_neumann_terms != fields["num_cayley_neumann_terms"].default:
            warnings.warn(
                "`num_cayley_neumann_terms` is only used when `use_cayley_neumann=True`. Since "
                f"`use_cayley_neumann=False`, `num_cayley_neumann_terms={self.num_cayley_neumann_terms}` "
                "will be ignored.",
                UserWarning,
                stacklevel=3,
            )
        if self.cayley_neumann_eps is not None:
            warnings.warn(
                "`cayley_neumann_eps` is only used when `use_cayley_neumann=True`. Since "
                f"`use_cayley_neumann=False`, `cayley_neumann_eps={self.cayley_neumann_eps}` will be ignored.",
                UserWarning,
                stacklevel=3,
            )

    r: int = 32
    target_modules: str | list[str] | set[str] | None = None
    exclude_modules: str | list[str] | set[str] | None = None
    psoft_alpha: float = 32
    psoft_dropout: float = 0.0
    fan_in_fan_out: bool = False
    ab_svd_init: Literal["psoft_init", "pissa_init"] = "psoft_init"
    psoft_svd: Literal["full", "lowrank"] = "full"
    psoft_svd_lowrank_niter: int = 10
    random_seed: int = 0
    psoft_orth: bool = True
    psoft_mag_b: bool = True
    psoft_mag_a: bool = True
    use_cayley_neumann: bool = False
    num_cayley_neumann_terms: int = 5
    cayley_neumann_eps: float | None = None
    modules_to_save: list[str] | None = None
    init_weights: bool = True
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None
