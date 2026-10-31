"""``BOFTConfig``: the BOFT hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass
from typing import ClassVar, Literal, override

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_count,
    _check_dropout,
    _check_rank,
    _choice,
    _TargetedConfig,
)

_BIASES = ("none", "all", "boft_only")


@dataclass(kw_only=True)
class BOFTConfig(_TargetedConfig, peft_type=PeftType.BOFT):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with BOFT.

    BOFT (Liu et al., 2023, arXiv:2311.06243 — Butterfly Orthogonal Finetuning)
    factors the orthogonal matrix OFT builds block-diagonally into a product of
    :math:`m` sparse orthogonal factors in butterfly order, so a small number of
    parameters spans a dense rotation of the input space:

    .. math:: y = \bigl(W\,M^\top\bigr)\operatorname{diag}(s)\,x + b,
              \qquad M = M_{m-1}\cdots M_1 M_0,\quad M_i = P_i\,B_i\,P_i^\top,

    each :math:`B_i` block-diagonal with Cayley-orthogonal blocks and
    :math:`P_i` a fixed butterfly permutation, :math:`s` a learned scale of the
    output rows.  The blocks start at the identity and :math:`s` at one, so the adapted
    model starts as the base; the transform folds into the weight.  With
    ``boft_n_butterfly_factor=1`` the method is OFT.

    Field names, defaults and ``adapter_config.json`` are the reference
    adapter library's.  Every field is keyword-only.

    Parameters
    ----------
    boft_block_size : int, default=4
        Size of each block.  Give exactly one of this and ``boft_block_num``
        (their product is the layer's input size); set this to ``0`` when you give the number.
    boft_block_num : int, default=0
        Number of blocks per factor.
    boft_n_butterfly_factor : int, default=1
        Number of butterfly factors :math:`m`.  Each extra factor doubles the
        effective block size of the rotation (and needs a block count divisible by two).
    target_modules : str or list of str, optional
        The ``nn.Linear`` / ``nn.Conv2d`` modules to adapt, matched as
        ``LoraConfig.target_modules`` is.  ``None`` takes the default for the base
        model's ``model_type`` (the LoRA table, as the reference does).
    exclude_modules : str or list of str, optional
        Modules never adapted, matched like ``target_modules``.
    boft_dropout : float, default=0.0
        During training, the probability of replacing a block of one randomly chosen factor by the identity.
    fan_in_fan_out : bool, default=False
        For bases that store their weight ``(in, out)``.  ``nn.Linear`` does not,
        so ``True`` warns and is reset to ``False``.
    bias : {"none"}, default="none"
        The reference also offers ``"all"`` and ``"boft_only"``; here the adapter
        trains its own tensors only, and those raise ``NotImplementedError``.
    modules_to_save : list of str, optional
        Modules trained in full beside the adapter (a classifier head).
    init_weights : bool, default=True
        ``True``: the blocks start at zero (the identity rotation) and :math:`s` at one.
        ``False``: both random (for tests).
    layers_to_transform, layers_pattern
        As in ``LoraConfig``.

    Raises
    ------
    TypeError
        A field holds a value of the wrong type.
    ValueError
        A field is out of range, or the block size and the block count are both set or both zero.
    NotImplementedError
        ``bias`` is not ``"none"``.

    Examples
    --------
    >>> from lucid.peft import BOFTConfig
    >>> config = BOFTConfig(boft_block_size=4, boft_n_butterfly_factor=2, target_modules=["q"])
    >>> config.boft_n_butterfly_factor, config.peft_type
    (2, <PeftType.BOFT: 'BOFT'>)
    >>> BOFTConfig.from_dict(config.to_dict()) == config
    True
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "boft_block_size": _check_count,
        "boft_block_num": _check_count,
        "boft_n_butterfly_factor": _check_rank,
        "boft_dropout": _check_dropout,
        "fan_in_fan_out": _check_bool,
        "bias": _choice(_BIASES),
        "init_weights": _check_bool,
    }

    boft_block_size: int = 4
    boft_block_num: int = 0
    boft_n_butterfly_factor: int = 1
    boft_dropout: float = 0.0
    fan_in_fan_out: bool = False
    bias: Literal["none", "all", "boft_only"] = "none"
    init_weights: bool = True

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        size, num = self.boft_block_size, self.boft_block_num
        if size == 0 and num == 0:
            raise ValueError(
                "Either `boft_block_size` or `boft_block_num` must be non-zero. "
                f"Currently, boft_block_size = {size} and boft_block_num = {num}."
            )
        if size != 0 and num != 0:
            raise ValueError(
                f"You can only specify either boft_block_size ({size}) or boft_block_num ({num}), "
                "but not both simultaneously, because boft_block_size x boft_block_num == in_features."
            )
        if self.bias != "none":
            raise NotImplementedError(
                f"bias={self.bias!r}: a BOFT adapter in lucid.peft trains its own rotations only "
                "(bias='none')"
            )
