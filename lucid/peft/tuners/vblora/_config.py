"""``VBLoRAConfig``: the VB-LoRA hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass
from typing import ClassVar, Final, Literal, override

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_dropout,
    _check_number,
    _check_rank,
    _choice,
    _TargetedConfig,
)

_BIASES: Final[tuple[Literal["none", "all", "vblora_only"], ...]] = ("none", "all", "vblora_only")


@dataclass
class VBLoRAConfig(_TargetedConfig, peft_type=PeftType.VBLORA):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with VB-LoRA.

    VB-LoRA (Li et al., 2024, arXiv:2405.15179 — Vector Bank LoRA) builds every
    low-rank factor of every layer out of one shared **vector bank**
    :math:`V \in \mathbb{R}^{N \times b}` of :math:`N` vectors of length :math:`b`.
    The factors :math:`A \in \mathbb{R}^{r \times d_{in}}` and
    :math:`B \in \mathbb{R}^{d_{out} \times r}` are cut into sub-vectors of length
    :math:`b`; each sub-vector is a softmax-weighted mix of the top-:math:`k`
    bank vectors picked by learned logits,

    .. math:: \mathrm{sub}(A)_{ij} = \sum_{t \in \mathrm{top}_k(\ell_{ij})}
              \mathrm{softmax}(\ell_{ij})_t\, V_t,
              \qquad y = W x + b + B A\,x ,

    so the trainable parameters are the logits :math:`\ell` and the bank, shared by
    the whole model.  The adapter does **not** start as the base: both factors are
    non-zero from the first step (as in the paper).

    Field names, their order, defaults and ``adapter_config.json`` are the reference
    adapter library's.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    r : int, default=4
        Rank of the factors.
    num_vectors : int, default=256
        Vectors in the bank (:math:`N`); use more for bigger models.
    vector_length : int, default=256
        Length :math:`b` of a bank vector; every targeted layer's input and output size
        must be a multiple of it.
    topk : int, default=2
        How many bank vectors mix into a sub-vector (:math:`k`).  ``2`` is the paper's
        recommendation; larger values make the saved model bigger.
    target_modules : str or list of str, optional
        The ``nn.Linear`` modules to adapt, matched as ``LoraConfig.target_modules``
        is.  ``None`` takes the default for the base model's ``model_type``
        (``VBLORA_DEFAULT_TARGET_MODULES`` in ``lucid.peft.tuners.vblora._defaults``);
        another model type raises ``ValueError``.
    exclude_modules : str or list of str, optional
        Modules never adapted, matched like ``target_modules``.
    save_only_topk_weights : bool, default=False
        Write only each sub-vector's top-:math:`k` indices and weights instead of the
        logits.  The file is much smaller and good for inference or merging, not for
        resuming training.
    vblora_dropout : float, default=0.0
        Dropout on the update's input, in ``[0, 1)``.
    fan_in_fan_out : bool, default=False
        For bases that store their weight ``(in, out)``.  ``nn.Linear`` does not,
        so ``True`` warns and is reset to ``False``.
    bias : {"none"}, default="none"
        The reference also offers ``"all"`` and ``"vblora_only"``; here the adapter
        trains its own tensors only, and those raise ``NotImplementedError``.
    modules_to_save : list of str, optional
        Modules trained in full beside the adapter (a classifier head).
    init_vector_bank_bound : float, default=0.02
        The bank starts uniform in ``[-bound, bound]`` (not zero, or the gradients vanish).
    init_logits_std : float, default=0.1
        The logits start normal with this standard deviation.
    layers_to_transform : int or list of int, optional
        As in :class:`~lucid.peft.LoraConfig`.
    layers_pattern : str or list of str, optional
        As in :class:`~lucid.peft.LoraConfig`.

    Raises
    ------
    TypeError
        A field holds a value of the wrong type.
    ValueError
        A field is out of range, or ``topk`` exceeds ``num_vectors``.
    NotImplementedError
        ``bias`` is not ``"none"``.

    Examples
    --------
    >>> from lucid.peft.tuners.vblora import VBLoRAConfig
    >>> config = VBLoRAConfig(r=4, num_vectors=60, vector_length=64, target_modules=["q", "v"])
    >>> sorted(config.target_modules), config.peft_type
    (['q', 'v'], <PeftType.VBLORA: 'VBLORA'>)
    >>> VBLoRAConfig.from_dict(config.to_dict()) == config
    True
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "num_vectors": _check_rank,
        "vector_length": _check_rank,
        "topk": _check_rank,
        "save_only_topk_weights": _check_bool,
        "vblora_dropout": _check_dropout,
        "fan_in_fan_out": _check_bool,
        "bias": _choice(_BIASES),
        "init_vector_bank_bound": _check_number,
        "init_logits_std": _check_number,
    }

    r: int = 4
    num_vectors: int = 256
    vector_length: int = 256
    topk: int = 2
    target_modules: str | list[str] | set[str] | None = None
    exclude_modules: str | list[str] | set[str] | None = None
    save_only_topk_weights: bool = False
    vblora_dropout: float = 0.0
    fan_in_fan_out: bool = False
    bias: Literal["none", "all", "vblora_only"] = "none"
    modules_to_save: list[str] | None = None
    init_vector_bank_bound: float = 0.02
    init_logits_std: float = 0.1
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if self.topk > self.num_vectors:
            raise ValueError(
                f"`topk` ({self.topk}) cannot exceed `num_vectors` ({self.num_vectors})"
            )
        if self.bias != "none":
            raise NotImplementedError(
                f"bias={self.bias!r}: a VB-LoRA adapter in lucid.peft trains its own tensors "
                "only (bias='none')"
            )
