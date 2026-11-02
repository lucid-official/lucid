"""``C3AConfig``: the C3A hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass, field
from typing import ClassVar, Final, Literal, override

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_rank,
    _check_regex,
    _choice,
    _keyed,
    _pattern_key_regex,
    _TargetedConfig,
)

_BIASES: Final[tuple[Literal["none", "all", "c3a_only"], ...]] = ("none", "all", "c3a_only")
_INITS: Final[tuple[str, ...]] = ("gaussian", "kaiming_uniform", "xavier_uniform")


def _check_init(name: str, value: object) -> bool | str:
    """``True`` (zeros), ``False`` (as ``"xavier_uniform"``) or one of the named samplers."""
    if isinstance(value, bool):
        return value
    return _choice(_INITS)(name, value)


@dataclass
class C3AConfig(_TargetedConfig, peft_type=PeftType.C3A):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with C3A.

    C3A (Chen et al., 2024, arXiv:2407.19342 — Circular Convolution Adaptation) makes the
    weight update a **block-circulant matrix**: the :math:`d_{out} \times d_{in}` update is
    cut into :math:`b \times b` blocks and each block is a circulant, fixed by its first row
    — a length-:math:`b` kernel.  The layer trains :math:`d_{out} d_{in} / b` numbers
    (:math:`b` kernels per block row and column), which is high rank, and applies them by
    FFT without building the matrix,

    .. math:: y = W x + b + \tfrac{1}{d_{in}}\,\mathcal C(w)\,x .

    With ``init_weights=True`` the kernels start at zero and the adapted model is the base;
    the default is a Xavier-uniform sample (non-zero).  The update folds into the weight.

    Field names, their order, defaults and ``adapter_config.json`` are the reference adapter
    library's.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    block_size : int, default=256
        The block size :math:`b`; it must divide both of a layer's sizes.  Larger blocks
        mean fewer parameters.  The greatest common divisor of all the targeted layers' sizes
        is the largest that fits them all.
    target_modules : str or list of str, optional
        The ``nn.Linear`` modules to adapt, matched as ``LoraConfig.target_modules``
        is.  ``None`` takes the LoRA default for the base model's ``model_type``, with ``gpt2``
        narrowed to ``mlp.c_proj`` as the reference does.
    bias : {"none"}, default="none"
        The reference also offers ``"all"`` and ``"c3a_only"``; here the adapter trains its
        kernels only, and those raise ``NotImplementedError``.
    modules_to_save : list of str, optional
        Modules trained in full beside the adapter (a classifier head).
    layers_to_transform : int or list of int, optional
        As in :class:`~lucid.peft.LoraConfig`.
    layers_pattern : str or list of str, optional
        As in :class:`~lucid.peft.LoraConfig`.
    block_size_pattern : dict of str to int, optional
        Per-module ``block_size`` overrides, keyed by name or regex suffix.
    init_weights : bool or {"gaussian", "kaiming_uniform", "xavier_uniform"}, default="xavier_uniform"
        ``True``: zeros.  ``False``: as ``"xavier_uniform"``.  A name picks that sampler.
    exclude_modules : str or list of str, optional
        Modules never adapted, matched like ``target_modules``.

    Raises
    ------
    TypeError
        A field is not of its type.
    ValueError
        ``block_size`` is not positive, or ``init_weights`` is not an accepted value.
    NotImplementedError
        ``bias`` asks for trained biases.

    Examples
    --------
    >>> from lucid.peft import C3AConfig
    >>> config = C3AConfig(block_size=16, target_modules=["q_proj", "v_proj"])
    >>> config.block_size, config.peft_type
    (16, <PeftType.C3A: 'C3A'>)
    >>> C3AConfig.from_dict(config.to_dict()) == config
    True
    """

    _extension_fields: ClassVar[frozenset[str]] = frozenset({"exclude_modules"})
    _field_checks: ClassVar[dict[str, _Check]] = {
        "block_size": _check_rank,
        "bias": _choice(_BIASES),
        "block_size_pattern": _keyed(_check_rank),
        "init_weights": _check_init,
    }

    block_size: int = 256
    target_modules: str | list[str] | set[str] | None = None
    bias: Literal["none", "all", "c3a_only"] = "none"
    modules_to_save: list[str] | None = None
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None
    block_size_pattern: dict[str, int] = field(default_factory=dict)
    init_weights: bool | Literal["gaussian", "kaiming_uniform", "xavier_uniform"] = "xavier_uniform"
    exclude_modules: str | list[str] | set[str] | None = field(default=None, kw_only=True)

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        for key in self.block_size_pattern:
            _check_regex(f"block_size_pattern[{key!r}]", _pattern_key_regex(key))
        if self.bias != "none":
            raise NotImplementedError(
                f"bias={self.bias!r}: a C3A adapter in lucid.peft trains its kernels only "
                "(bias='none')"
            )
