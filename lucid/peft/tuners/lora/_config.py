"""``LoraConfig``: the LoRA hyper-parameters on top of the shared targeting fields."""

import warnings
from dataclasses import dataclass, field
from typing import ClassVar, Final, Literal, override

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_dropout,
    _check_number,
    _check_rank,
    _check_regex,
    _choice,
    _keyed,
    _optional_strings,
    _pattern_key_regex,
    _TargetedConfig,
)
from lucid.peft.tuners.lora._inits import _check_init, _Init

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
        The task the adapter is for (picks the ``PeftModelFor*`` class
        ``get_peft_model`` returns; see ``TaskType``).
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
        whose weight is shared with another module.  ``None`` takes the
        default for the base model's ``model_type``
        (``DEFAULT_LORA_TARGET_MODULES`` in ``lucid.peft._target_defaults``);
        a model type with no row raises ``ValueError``.
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
    init_lora_weights : bool or str, default=True
        ``True``: Kaiming-uniform :math:`A` (``a = \sqrt 5``) and zero
        :math:`B`, so the adapted model starts as the base.
        ``"gaussian"``: :math:`A \sim \mathcal N(0, 1/r^2)`, zero
        :math:`B`.  ``False``: both keep ``nn.Linear``'s initialisation
        (the update is non-zero from the start — for tests).
        ``"pissa"`` (Meng et al., 2024) puts the weight's top-:math:`r`
        singular components into :math:`BA` and the residual into the base
        weight; ``"pissa_niter_<k>"`` does it with a randomised SVD of ``k``
        power iterations.  ``"olora"`` (Büyükakyüz, 2024) puts the first
        :math:`r` columns of a QR decomposition's :math:`Q` and rows of
        :math:`R` there instead.  Both **rewrite the base weight in place**
        (the adapted layer still starts as the original), so an adapter
        trained from them is relative to that rewritten base — save one
        with ``save_pretrained(..., path_initial_model_for_weight_conversion=)``
        to get a plain LoRA of twice the rank that fits the original base;
        they apply to ``nn.Linear`` targets only.  ``"eva"``, ``"corda"``, ``"loftq"`` and
        the reference's other initialisations raise ``NotImplementedError``.
    rank_pattern : dict of str to int, optional
        Per-module rank overrides, keyed by name or regex suffix.
    alpha_pattern : dict of str to float, optional
        Per-module :math:`\alpha` overrides, keyed the same way.
    use_dora : bool, default=False
        Weight-decomposed low-rank adaptation (Liu et al., 2024): a trainable
        magnitude per output unit beside :math:`BA`, so the adapted weight
        is :math:`m\,(W + sBA)/\lVert W + sBA\rVert` row by row.  Supported
        on ``nn.Linear``, ``nn.Embedding`` and the convolutions (not grouped
        ones); ``nn.MultiheadAttention`` and ``target_parameters`` refuse it,
        as the reference does.  Costs a norm of the full weight per forward
        pass; merge before inference.
    lora_bias : bool, default=False
        Give :math:`B` a bias, folded into the base bias on merge.  For
        adapters extracted from a fully fine-tuned layer; needs
        ``init_lora_weights`` of ``True`` or ``False`` and not ``use_dora``.
        ``nn.Linear`` and convolutions only.
    target_parameters : list of str, optional
        Raw ``nn.Parameter``\ s to adapt, for a parameter that no layer
        reads through a module of its own (an attention's fused
        ``in_proj_weight``, a mixture-of-experts weight stack).  An entry
        matches a parameter whose dotted name equals it or ends with
        ``"." + entry``.  The parameter is ``(out, in)``, or a stack of
        them, ``(experts, out, in)``; ``lora_dropout`` must be 0 for it.
        The holding module cannot also be in ``target_modules``.  With no
        ``target_modules``, no module is adapted (the model-type default is not taken).

    Raises
    ------
    TypeError
        A field holds a value of the wrong type.
    ValueError
        A field is out of range or two fields conflict.
    NotImplementedError
        An ``init_lora_weights`` the library lacks (``"eva"``, ``"loftq"`` …).

    Examples
    --------
    >>> from lucid.peft import LoraConfig
    >>> config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.LORA: 'LORA'>)
    >>> LoraConfig.from_dict(config.to_dict()) == config
    True
    >>> dora = LoraConfig(use_dora=True, init_lora_weights="PiSSA_niter_4")
    >>> dora.use_dora, dora.init_lora_weights
    (True, 'pissa_niter_4')
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
        "target_parameters": _optional_strings,
        "lora_bias": _check_bool,
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
    target_parameters: list[str] | None = None
    lora_bias: bool = False

    @override
    def _load_fields(self) -> dict[str, object]:
        return {"init_lora_weights": _Init.of(self.init_lora_weights).on_load().value}

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if self.lora_bias:
            if self.init_lora_weights not in (True, False):
                raise ValueError(
                    "The argument lora_bias=True is only supported with init_lora_weights=True "
                    f"or False, got init_lora_weights={self.init_lora_weights} instead."
                )
            if self.use_dora:
                raise ValueError(
                    "The argument lora_bias=True is not supported for DoRA, "
                    "please pass use_dora=False"
                )
        if (
            self.use_rslora
            and (self.rank_pattern or self.alpha_pattern)
            and _Init.of(self.init_lora_weights).factorises
        ):
            warnings.warn(
                "Using Rank-Stabilized LoRA with rank_pattern/alpha_pattern and an "
                "initialisation that rewrites the base weights (PiSSA / OLoRA) means the "
                "initial base weights cannot be restored exactly afterwards; if you intend "
                "to do that, do not use rslora with rank_pattern/alpha_pattern.",
                stacklevel=2,
            )
        for name, keys in (
            ("rank_pattern", self.rank_pattern),
            ("alpha_pattern", self.alpha_pattern),
        ):
            for key in keys:
                _check_regex(f"{name}[{key!r}]", _pattern_key_regex(key))
