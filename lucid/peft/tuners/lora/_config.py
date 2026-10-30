"""``LoraConfig``: the LoRA hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass, field
from typing import ClassVar, Final, Literal, override

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_dropout,
    _check_init,
    _check_number,
    _check_rank,
    _check_regex,
    _choice,
    _keyed,
    _pattern_key_regex,
    _TargetedConfig,
)

_BIASES: Final[tuple[Literal["none", "all", "lora_only"], ...]] = ("none", "all", "lora_only")


@dataclass(kw_only=True)
class LoraConfig(_TargetedConfig, peft_type=PeftType.LORA):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with LoRA.

    LoRA (Hu et al., 2021, arXiv:2106.09685) freezes a pretrained weight
    :math:`W` and learns a low-rank update beside it, so a targeted
    ``nn.Linear`` computes :math:`W x + s \, B A x` with
    :math:`A \in \mathbb{R}^{r \times d_{in}}`,
    :math:`B \in \mathbb{R}^{d_{out} \times r}` and scale
    :math:`s = \alpha / r` (or :math:`\alpha / \sqrt{r}` with rsLoRA).

    Field names, defaults and ``adapter_config.json`` are the reference
    adapter library's.  Every field is keyword-only.

    Parameters
    ----------
    task_type : TaskType or str, optional
        The task the adapter is for (metadata; see ``TaskType``).
    peft_type : PeftType or str, optional
        Always ``PeftType.LORA``; filled in when omitted.
    base_model_name_or_path : str, optional
        Name of the model the adapter was trained on (metadata).
    revision : str, optional
        Revision of that model (metadata).
    inference_mode : bool, default=False
        Freeze the adapter too, for inference.
    target_modules : str or list of str, optional
        The ``nn.Linear`` modules to adapt.  A list matches a module whose
        dotted name equals an entry or ends with ``"." + entry``; a string
        is a regular expression the whole name must match; ``"all-linear"``
        selects every ``nn.Linear`` except the output head and any layer
        whose weight is shared with another module.  Required: there is
        no per-architecture default.
    exclude_modules : str or list of str, optional
        Modules never adapted, matched like ``target_modules``.
    modules_to_save : list of str, optional
        Modules trained in full beside the adapter (a classifier head);
        a trainable copy is kept and saved with it, the original stays.
        An entry matches a module whose dotted name equals it or ends
        with ``"." + entry`` — on a dot boundary, so ``classifier`` does
        not also catch ``pre_classifier`` as the reference's bare suffix
        test does.  Nothing inside these modules is adapted.
    layers_to_transform : int or list of int, optional
        Adapt only these layer indices (the number after the layer list's
        name, ``transformer.h.3.attn`` → 3).  Applies to modules matched
        by suffix; one named in full is adapted regardless.  The index is
        the reference's current rule: the first (outer) ``<list>.<n>.`` in
        the name, so ``encoder.layers.0.blocks.1.qkv`` is layer 0.  A
        layer list at the model's root (``h.3.attn``) is indexed too,
        where the reference finds no index without a ``layers_pattern``.
    layers_pattern : str or list of str, optional
        Name of the layer list (``"h"``, ``"layers"``), a regex fragment;
        the first match in the name gives the index.  Needs
        ``layers_to_transform``.
    r : int, default=8
        Rank of the update.
    lora_alpha : float, default=8
        :math:`\alpha` in the scale.
    lora_dropout : float, default=0.0
        Dropout on the update's input, in ``[0, 1)``.
    fan_in_fan_out : bool, default=False
        For bases that store their weight ``(in, out)``.  ``nn.Linear``
        does not, so ``True`` warns and is reset to ``False``.
    bias : {"none", "all", "lora_only"}, default="none"
        Which biases also train: none, every parameter named ``bias``, or
        only the biases of the adapted layers.
    use_rslora : bool, default=False
        Scale by :math:`\alpha / \sqrt{r}` (Kalajdzievski, 2023).
    init_lora_weights : bool or "gaussian", default=True
        ``True``: Kaiming-uniform :math:`A` (``a = \sqrt 5``) and zero
        :math:`B`, so the adapted model starts as the base.
        ``"gaussian"``: :math:`A \sim \mathcal N(0, 1/r^2)`, zero
        :math:`B`.  ``False``: both keep ``nn.Linear``'s initialisation
        (the update is non-zero from the start — for tests).
    rank_pattern : dict of str to int, optional
        Per-module rank overrides, keyed by name or regex suffix.
    alpha_pattern : dict of str to float, optional
        Per-module :math:`\alpha` overrides, keyed the same way.
    use_dora : bool, default=False
        DoRA; not implemented in v1, ``True`` raises.

    Raises
    ------
    TypeError
        A field holds a value of the wrong type.
    ValueError
        A field is out of range or two fields conflict.
    NotImplementedError
        ``use_dora=True`` or an initialisation v1 lacks.

    Examples
    --------
    >>> from lucid.peft import LoraConfig
    >>> config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.LORA: 'LORA'>)
    >>> LoraConfig.from_dict(config.to_dict()) == config
    True
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "lora_alpha": _check_number,
        "lora_dropout": _check_dropout,
        "fan_in_fan_out": _check_bool,
        "bias": _choice(_BIASES),
        "use_rslora": _check_bool,
        "init_lora_weights": _check_init,
        "rank_pattern": _keyed(_check_rank),
        "alpha_pattern": _keyed(_check_number),
        "use_dora": _check_bool,
    }

    r: int = 8
    lora_alpha: float = 8
    lora_dropout: float = 0.0
    fan_in_fan_out: bool = False
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = False
    init_lora_weights: bool | str = True
    rank_pattern: dict[str, int] = field(default_factory=dict)
    alpha_pattern: dict[str, float] = field(default_factory=dict)
    use_dora: bool = False

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if self.use_dora:
            raise NotImplementedError("use_dora=True: DoRA is not implemented in lucid.peft v1")
        for name, keys in (
            ("rank_pattern", self.rank_pattern),
            ("alpha_pattern", self.alpha_pattern),
        ):
            for key in keys:
                _check_regex(f"{name}[{key!r}]", _pattern_key_regex(key))
