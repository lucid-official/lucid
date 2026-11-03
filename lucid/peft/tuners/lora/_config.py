"""``LoraConfig``: the LoRA hyper-parameters on top of the shared targeting fields."""

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Final, Literal, override

import lucid.nn as nn

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


#: The reference ``LoraConfig`` fields this package accepts at their default and refuses otherwise.
_LORA_UNSUPPORTED: Final = {
    "megatron_config": "Megatron tensor-parallel layers do not exist here",
    "megatron_core": "Megatron tensor-parallel layers do not exist here",
    "trainable_token_indices": "use TrainableTokensConfig for trainable tokens",
    "loftq_config": "LoftQ quantisation-aware initialisation is not implemented",
    "corda_config": "CorDA initialisation is not implemented",
    "velora_config": "VeLoRA is not implemented",
    "alora_invocation_tokens": "activated LoRA is not implemented",
    "use_qalora": "quantisation-aware LoRA needs a quantised base layer",
    "qalora_group_size": "quantisation-aware LoRA needs a quantised base layer",
    "monteclora_config": "MonteCLoRA is not implemented",
    "layer_replication": "layer replication is not implemented",
    "use_bdlora": "block-diagonal LoRA is not implemented",
    "arrow_config": "Arrow routing is not implemented",
    "kasa_config": "KaSA is not implemented",
    "ensure_weight_tying": "weight-tying bookkeeping is not implemented",
}


@dataclass
class LoraRuntimeConfig:
    """Settings that are valid at runtime only and never written to an adapter file.

    Parameters
    ----------
    ephemeral_gpu_offload : bool, default=False
        The reference keeps a layer's weights on the CPU and moves them to the accelerator
        for one forward pass.  Not implemented: ``True`` raises ``NotImplementedError``.
    """

    ephemeral_gpu_offload: bool = False

    def __post_init__(self) -> None:
        if self.ephemeral_gpu_offload:
            raise NotImplementedError(
                "LoraRuntimeConfig: `ephemeral_gpu_offload=True` is not supported by lucid.peft: "
                "adapter layers are never offloaded"
            )


def _check_runtime(name: str, value: object) -> LoraRuntimeConfig:
    """A ``LoraRuntimeConfig``, or the dict of its fields."""
    if isinstance(value, LoraRuntimeConfig):
        return value
    if isinstance(value, Mapping):
        return LoraRuntimeConfig(**value)
    raise TypeError(f"`{name}` must be a LoraRuntimeConfig, got {value!r}")


@dataclass
class EvaConfig:
    r"""Configuration for Explained Variance Adaptation (EVA) data-driven LoRA initialisation.

    EVA (Paischer et al., 2024, arXiv:2410.07170) runs an incremental SVD over
    calibration activations to redistribute the LoRA rank budget across layers
    according to their explained variance ratios and initialise :math:`A` from
    the top right-singular vectors.

    Parameters
    ----------
    rho : float, default=2.0
        Maximum rank redistribution multiplier (``>= 1.0``); a layer of rank ``r``
        may receive up to ``round(r * rho)`` components.
    tau : float, default=0.99
        Cosine-similarity convergence threshold in ``[0, 1]`` between consecutive
        SVD steps.
    use_label_mask : bool, default=True
        Ignore token positions where ``labels == label_mask_value`` during the SVD.
    label_mask_value : int, default=-100
        Label value marking ignored positions when ``use_label_mask=True``.
    whiten : bool, default=False
        Divide each singular vector by the square root of its singular value.
    adjust_scaling_factors : bool, default=True
        Scale ``lora_alpha`` proportionally when a layer's rank changes during
        redistribution.

    Raises
    ------
    ValueError
        ``rho < 1.0`` or ``tau`` is outside ``[0.0, 1.0]``.

    Examples
    --------
    >>> from lucid.peft import EvaConfig
    >>> cfg = EvaConfig(rho=1.5, tau=0.95, whiten=True)
    >>> cfg.rho, cfg.whiten
    (1.5, True)
    """

    rho: float = 2.0
    tau: float = 0.99
    use_label_mask: bool = True
    label_mask_value: int = -100
    whiten: bool = False
    adjust_scaling_factors: bool = True

    def __post_init__(self) -> None:
        if self.rho < 1.0:
            raise ValueError("`rho` must be >= 1.0")
        if self.tau < 0.0 or self.tau > 1.0:
            raise ValueError("`tau` must be between 0.0 and 1.0.")


def _check_eva(name: str, value: object) -> EvaConfig | None:
    if value is None or isinstance(value, EvaConfig):
        return value
    if isinstance(value, Mapping):
        return EvaConfig(**value)
    raise TypeError(f"`{name}` must be an EvaConfig, a dict, or None, got {value!r}")


_LORAGA_DIRECTIONS: Final = ("ArBr", "A2rBr", "ArB2r", "random")
_LORAGA_SCALES: Final = ("stable", "weight_svd", "gd_scale", "unit")


@dataclass
class LoraGAConfig:
    r"""Configuration for LoRA-GA (Gradient Approximation) initialisation.

    LoRA-GA (Wang et al., 2024, arXiv:2407.05000) estimates full-model weight
    gradients over a calibration step (:func:`~lucid.peft.preprocess_loraga`),
    aligned via low-rank SVD into :math:`A` and :math:`B`, and subtracts
    :math:`s\,BA` from the base weight so the adapted layer starts at the
    pretrained function while its first gradient step matches full fine-tuning.

    Parameters
    ----------
    direction : {"ArBr", "A2rBr", "ArB2r", "random"}, default="ArB2r"
        How the top-:math:`2r` singular vectors of the gradient are partitioned
        between :math:`A` and :math:`B`.
    scale : {"stable", "weight_svd", "gd_scale", "unit"}, default="stable"
        How the initialised factors are scaled.
    stable_gamma : int, default=16
        Denominator hyperparameter :math:`\gamma` for ``scale="stable"``.

    Raises
    ------
    ValueError
        ``direction`` or ``scale`` is unknown, or ``stable_gamma <= 0``.

    Examples
    --------
    >>> from lucid.peft import LoraGAConfig
    >>> cfg = LoraGAConfig(direction="ArB2r", scale="stable", stable_gamma=16)
    >>> cfg.direction, cfg.scale
    ('ArB2r', 'stable')
    """

    direction: Literal["ArBr", "A2rBr", "ArB2r", "random"] = "ArB2r"
    scale: Literal["stable", "weight_svd", "gd_scale", "unit"] = "stable"
    stable_gamma: int = 16

    def __post_init__(self) -> None:
        if self.direction not in _LORAGA_DIRECTIONS:
            raise ValueError(
                f"Invalid `direction` {self.direction!r}; must be one of {_LORAGA_DIRECTIONS}."
            )
        if self.scale not in _LORAGA_SCALES:
            raise ValueError(f"Invalid `scale` {self.scale!r}; must be one of {_LORAGA_SCALES}.")
        gamma: object = self.stable_gamma
        if isinstance(gamma, bool) or not isinstance(gamma, int) or gamma <= 0:
            raise ValueError(f"`stable_gamma` must be a positive int, got {self.stable_gamma!r}.")


def _check_lora_ga(name: str, value: object) -> LoraGAConfig | None:
    if value is None or isinstance(value, LoraGAConfig):
        return value
    if isinstance(value, Mapping):
        return LoraGAConfig(**value)
    raise TypeError(f"`{name}` must be a LoraGAConfig, a dict, or None, got {value!r}")


@dataclass
class LoftQConfig:
    r"""Configuration stub for LoftQ quantisation-aware LoRA initialisation.

    Kept for signature parity with the reference adapter library; passing a
    non-empty ``loftq_config`` or ``init_lora_weights="loftq"`` to
    :class:`LoraConfig` raises ``NotImplementedError`` until 4-bit quantised
    LoRA layers land (LCD-717).

    Parameters
    ----------
    loftq_bits : int, default=4
        Quantisation bit-width for LoftQ.
    loftq_iter : int, default=1
        Number of alternating quantisation / SVD iterations.

    Examples
    --------
    >>> from lucid.peft import LoftQConfig
    >>> cfg = LoftQConfig(loftq_bits=4, loftq_iter=1)
    >>> cfg.loftq_bits, cfg.loftq_iter
    (4, 1)
    """

    loftq_bits: int = 4
    loftq_iter: int = 1


def replace_lora_weights_loftq(
    peft_model: nn.Module,
    model_path: str | None = None,
    adapter_name: str = "default",
    callback: Callable[[nn.Module, str], bool] | None = None,
) -> None:
    r"""Refused with ``NotImplementedError``: 4-bit quantised LoRA layers are not implemented yet.

    Provided for API parity with the reference adapter library.

    Parameters
    ----------
    peft_model : nn.Module
        Quantised PEFT model whose LoRA weights would be replaced.
    model_path : str, optional
        Path to the unquantised safetensors weights.
    adapter_name : str, default="default"
        Adapter name to update.
    callback : callable, optional
        Per-module acceptance callback ``(model, module_name) -> bool``.

    Raises
    ------
    NotImplementedError
        Always raised; 4-bit quantised LoRA layers are not implemented yet.
    """
    del peft_model, model_path, adapter_name, callback
    raise NotImplementedError(
        "replace_lora_weights_loftq requires 4-bit quantized LoRA layers, "
        "which are not implemented in lucid.peft yet."
    )


@dataclass
class _LoraFields(_TargetedConfig):
    """The fields of the reference's ``LoraConfig``, in its order, for the configs that extend it.

    The reference's ``AdaLoraConfig`` is a ``LoraConfig`` with more fields; here the two share
    this base instead, since an AdaLoRA config is not a LoRA one (it does not take its
    initialisations, its patterns or its adapters).  The fields the library lacks are accepted at
    their defaults only (``_LORA_UNSUPPORTED``).
    """

    _unsupported: ClassVar[dict[str, str]] = _LORA_UNSUPPORTED
    #: ``runtime_config`` is valid at runtime only: a file never holds it.
    _extension_fields: ClassVar[frozenset[str]] = frozenset({"runtime_config"})
    _field_checks: ClassVar[dict[str, _Check]] = {"runtime_config": _check_runtime}

    r: int = 8
    target_modules: str | list[str] | set[str] | None = None
    exclude_modules: str | list[str] | set[str] | None = None
    lora_alpha: float = 8
    lora_dropout: float = 0.0
    fan_in_fan_out: bool = False
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = False
    modules_to_save: list[str] | None = None
    init_lora_weights: bool | str = True
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None
    rank_pattern: dict[str, int] = field(default_factory=dict)
    alpha_pattern: dict[str, float] = field(default_factory=dict)
    megatron_config: dict[str, object] | None = None
    megatron_core: str | None = "megatron.core"
    trainable_token_indices: list[int] | dict[str, list[int]] | None = None
    loftq_config: dict[str, object] = field(default_factory=dict)
    eva_config: EvaConfig | None = None
    corda_config: object | None = None
    lora_ga_config: LoraGAConfig | None = None
    use_dora: bool = False
    velora_config: object | None = None
    alora_invocation_tokens: list[int] | None = None
    use_qalora: bool = False
    qalora_group_size: int = 16
    monteclora_config: object | None = None
    layer_replication: list[tuple[int, int]] | None = None
    runtime_config: LoraRuntimeConfig = field(default_factory=LoraRuntimeConfig)
    lora_bias: bool = False
    target_parameters: list[str] | None = None
    use_bdlora: object | None = None
    arrow_config: object | None = None
    kasa_config: object | None = None
    ensure_weight_tying: bool = False


@dataclass
class LoraConfig(_LoraFields, peft_type=PeftType.LORA):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with LoRA.

    LoRA (Hu et al., 2021, arXiv:2106.09685) freezes a pretrained weight
    :math:`W` and learns a low-rank update beside it, so a targeted
    ``nn.Linear`` computes :math:`W x + s \, B A x` with
    :math:`A \in \mathbb{R}^{r \times d_{in}}`,
    :math:`B \in \mathbb{R}^{d_{out} \times r}` and scale
    :math:`s = \alpha / r` (or :math:`\alpha / \sqrt{r}` with rsLoRA).

    Field names, their order, defaults and ``adapter_config.json`` are the
    reference adapter library's.  The reference's options for features this
    package lacks (``eva_config``, ``loftq_config``, ``use_qalora`` …) are
    fields too, at the reference's defaults; any other value raises
    ``NotImplementedError`` naming the field.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    r : int, default=8
        Rank of the update.
    target_modules : str or list of str, optional
        The modules to adapt: ``nn.Linear``, ``nn.Embedding``,
        ``nn.Conv1d`` / ``nn.Conv2d`` / ``nn.Conv3d``, their
        ``nn.ConvTranspose`` counterparts and ``nn.MultiheadAttention``;
        the module's type picks the LoRA layer that wraps it
        (``lucid.peft.tuners.lora``).  A list matches a module whose
        dotted name equals an entry or ends with ``"." + entry``; a string
        is a regular expression the whole name must match; ``"all-linear"``
        selects every ``nn.Linear`` except the output head and any layer
        whose weight is shared with another module.  ``None`` takes the
        default for the base model's ``model_type``
        (``DEFAULT_LORA_TARGET_MODULES`` in ``lucid.peft._target_defaults``);
        a model type with no row raises ``ValueError``.
    exclude_modules : str or list of str, optional
        Modules never adapted, matched like ``target_modules``.
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
    modules_to_save : list of str, optional
        Modules trained in full beside the adapter (a classifier head);
        a trainable copy is kept and saved with it, the original stays.
        An entry matches a module whose dotted name equals it or ends
        with ``"." + entry`` — on a dot boundary, so ``classifier`` does
        not also catch ``pre_classifier`` as the reference's bare suffix
        test does.  Nothing inside these modules is adapted.
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
    rank_pattern : dict of str to int, optional
        Per-module rank overrides, keyed by name or regex suffix.
    alpha_pattern : dict of str to float, optional
        Per-module :math:`\alpha` overrides, keyed the same way.
    megatron_config : dict, optional
        Megatron tensor-parallel layers do not exist here.  Accepted at the reference's default only:
        any other value raises ``NotImplementedError``.
    megatron_core : str, default='megatron.core'
        Megatron tensor-parallel layers do not exist here.  Accepted at the reference's default only:
        any other value raises ``NotImplementedError``.
    trainable_token_indices : list of int or dict of str to list of int, optional
        Use TrainableTokensConfig for trainable tokens.  Accepted at the reference's default only:
        any other value raises ``NotImplementedError``.
    loftq_config : dict, optional
        LoftQ quantisation-aware initialisation is not implemented.  Accepted at the reference's default only:
        any other value raises ``NotImplementedError``.
    eva_config : object, optional
        EVA initialisation is not implemented.  Accepted at the reference's default only:
        any other value raises ``NotImplementedError``.
    corda_config : object, optional
        CorDA initialisation is not implemented.  Accepted at the reference's default only:
        any other value raises ``NotImplementedError``.
    lora_ga_config : object, optional
        LoRA-GA initialisation is not implemented.  Accepted at the reference's default only:
        any other value raises ``NotImplementedError``.
    use_dora : bool, default=False
        Weight-decomposed low-rank adaptation (Liu et al., 2024): a trainable
        magnitude per output unit beside :math:`BA`, so the adapted weight
        is :math:`m\,(W + sBA)/\lVert W + sBA\rVert` row by row.  Supported
        on ``nn.Linear``, ``nn.Embedding`` and the convolutions (not grouped
        ones); ``nn.MultiheadAttention`` and ``target_parameters`` refuse it,
        as the reference does.  Costs a norm of the full weight per forward
        pass; merge before inference.
    velora_config : object, optional
        VeLoRA is not implemented.  Accepted at the reference's default only:
        any other value raises ``NotImplementedError``.
    alora_invocation_tokens : list of int, optional
        Activated LoRA is not implemented.  Accepted at the reference's default only:
        any other value raises ``NotImplementedError``.
    use_qalora : bool, default=False
        Quantisation-aware LoRA needs a quantised base layer.  Accepted at the reference's default only:
        any other value raises ``NotImplementedError``.
    qalora_group_size : int, default=16
        Quantisation-aware LoRA needs a quantised base layer.  Accepted at the reference's default only:
        any other value raises ``NotImplementedError``.
    monteclora_config : object, optional
        MonteCLoRA is not implemented.  Accepted at the reference's default only:
        any other value raises ``NotImplementedError``.
    layer_replication : list of tuple of int, optional
        Layer replication is not implemented.  Accepted at the reference's default only:
        any other value raises ``NotImplementedError``.
    runtime_config : LoraRuntimeConfig
        Settings valid at runtime only, never written to a file.  Only the default
        (``ephemeral_gpu_offload=False``) is supported.
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
    use_bdlora : object, optional
        Block-diagonal LoRA is not implemented.  Accepted at the reference's default only:
        any other value raises ``NotImplementedError``.
    arrow_config : object, optional
        Arrow routing is not implemented.  Accepted at the reference's default only:
        any other value raises ``NotImplementedError``.
    kasa_config : object, optional
        KaSA is not implemented.  Accepted at the reference's default only:
        any other value raises ``NotImplementedError``.
    ensure_weight_tying : bool, default=False
        Weight-tying bookkeeping is not implemented.  Accepted at the reference's default only:
        any other value raises ``NotImplementedError``.

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
        "eva_config": _check_eva,
        "lora_ga_config": _check_lora_ga,
    }

    @override
    def _load_fields(self) -> dict[str, object]:
        return {"init_lora_weights": _Init.of(self.init_lora_weights).on_load().value}

    def _check_init_subconfigs(self) -> None:
        if self.init_lora_weights == "eva" and self.eva_config is None:
            warnings.warn(
                "`init_lora_weights` is 'eva' but `eva_config` is not specified. Using default EVA config.",
                stacklevel=3,
            )
            self.eva_config = EvaConfig()
        elif self.init_lora_weights != "eva" and self.eva_config is not None:
            warnings.warn(
                "`eva_config` specified but will be ignored when `init_lora_weights` is not 'eva'.",
                stacklevel=3,
            )
        if self.init_lora_weights != "lora_ga" and self.lora_ga_config is not None:
            warnings.warn(
                "`lora_ga_config` specified but will be ignored when `init_lora_weights` is not 'lora_ga'.",
                stacklevel=3,
            )

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        self._check_init_subconfigs()
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
