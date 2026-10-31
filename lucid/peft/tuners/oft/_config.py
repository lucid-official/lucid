"""``OFTConfig``: the OFT hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass
from typing import ClassVar, Literal, override

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_count,
    _check_dropout,
    _check_number,
    _check_rank,
    _choice,
    _TargetedConfig,
)

_BIASES = ("none", "all", "oft_only")


@dataclass(kw_only=True)
class OFTConfig(_TargetedConfig, peft_type=PeftType.OFT):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with OFT.

    OFT (Qiu et al., 2023, arXiv:2306.07280 — Orthogonal Finetuning) rotates the
    input space of each targeted layer instead of adding to its weight:
    :math:`y = (W R^\top) x + b` with :math:`R` block-diagonal, each block an
    orthogonal matrix :math:`(I - Q)(I + Q)^{-1}` of a learned skew-symmetric
    :math:`Q`.  A rotation keeps the pairwise angles between the weight's
    neurons, which is what OFT tunes while the weight's spectrum stays.  The
    :math:`Q`\ s start at zero, so the adapted model starts as the base; the
    rotation folds into the weight, so a merged model costs what the base does.

    Field names, defaults and ``adapter_config.json`` are the reference
    adapter library's.  Every field is keyword-only.

    Parameters
    ----------
    r : int, default=0
        Number of blocks per layer.  Give exactly one of ``r`` and
        ``oft_block_size`` (their product is the layer's input size): the default
        ``r=0`` leaves the choice to ``oft_block_size``.
    oft_block_size : int, default=32
        Size of each block.  Pass ``oft_block_size=0`` when you set ``r``.  A size
        that does not divide a layer's input size is moved to the nearest one that
        does, with a warning.
    module_dropout : float, default=0.0
        During training, the probability of replacing a block by the identity.
    target_modules : str or list of str, optional
        The ``nn.Linear`` / ``nn.Conv2d`` modules to adapt, matched as
        ``LoraConfig.target_modules`` is.  ``None`` takes the default for the base
        model's ``model_type`` (the LoRA table, as the reference does).
    exclude_modules : str or list of str, optional
        Modules never adapted, matched like ``target_modules``.
    fan_in_fan_out : bool, default=False
        For bases that store their weight ``(in, out)``.  ``nn.Linear`` does not,
        so ``True`` warns and is reset to ``False``.
    bias : {"none"}, default="none"
        The reference also offers ``"all"`` and ``"oft_only"``; here the adapter
        trains its own tensors only, and those raise ``NotImplementedError``.
    init_weights : bool, default=True
        ``True``: every :math:`Q` starts at zero (the update is the identity).
        ``False``: :math:`Q \sim \mathcal N(0, 0.1^2)` (for tests).
    layers_to_transform, layers_pattern, modules_to_save
        As in ``LoraConfig``.
    coft : bool, default=False
        The constrained variant: each block's :math:`Q` is projected into a ball
        of radius ``eps`` around zero every time the rotation is computed.
    eps : float, default=6e-5
        The radius of that ball (scaled by one over the square root of the block
        count).  Only read when ``coft`` is set.
    block_share : bool, default=False
        Share one block among all positions: fewer parameters, one rotation.
    use_cayley_neumann : bool, default=True
        Approximate the Cayley transform by a truncated Neumann series
        (matrix products only, orthogonal up to the truncation) instead of an exact solve.
    num_cayley_neumann_terms : int, default=5
        Terms of that series.

    Raises
    ------
    TypeError
        A field holds a value of the wrong type.
    ValueError
        A field is out of range, or ``r`` and ``oft_block_size`` are both set or both zero.
    NotImplementedError
        ``bias`` is not ``"none"``.

    Examples
    --------
    >>> from lucid.peft import OFTConfig
    >>> config = OFTConfig(oft_block_size=16, target_modules=["q_proj", "v_proj"])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.OFT: 'OFT'>)
    >>> OFTConfig.from_dict(config.to_dict()) == config
    True
    >>> OFTConfig(r=4, oft_block_size=0, target_modules=["x"]).r
    4
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_count,
        "oft_block_size": _check_count,
        "module_dropout": _check_dropout,
        "fan_in_fan_out": _check_bool,
        "bias": _choice(_BIASES),
        "init_weights": _check_bool,
        "coft": _check_bool,
        "eps": _check_number,
        "block_share": _check_bool,
        "use_cayley_neumann": _check_bool,
        "num_cayley_neumann_terms": _check_rank,
    }

    r: int = 0
    oft_block_size: int = 32
    module_dropout: float = 0.0
    fan_in_fan_out: bool = False
    bias: Literal["none", "all", "oft_only"] = "none"
    init_weights: bool = True
    coft: bool = False
    eps: float = 6e-5
    block_share: bool = False
    use_cayley_neumann: bool = True
    num_cayley_neumann_terms: int = 5

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if self.r == 0 and self.oft_block_size == 0:
            raise ValueError(
                "Either `r` or `oft_block_size` must be non-zero. "
                f"Currently, r = {self.r} and oft_block_size = {self.oft_block_size}."
            )
        if self.r != 0 and self.oft_block_size != 0:
            raise ValueError(
                f"You can only specify either r ({self.r}) or oft_block_size "
                f"({self.oft_block_size}), but not both simultaneously, because "
                "r x oft_block_size == in_features."
            )
        if self.bias != "none":
            raise NotImplementedError(
                f"bias={self.bias!r}: an OFT adapter in lucid.peft trains its own rotations only "
                "(bias='none')"
            )
