"""``IA3Config``: the (IA)^3 hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass
from typing import ClassVar, override

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_regex,
    _name_set,
    _TargetedConfig,
)


@dataclass(kw_only=True)
class IA3Config(_TargetedConfig, peft_type=PeftType.IA3):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with (IA)^3.

    (IA)^3 (Liu et al., 2022, arXiv:2205.05638 — "Infused Adapter by
    Inhibiting and Amplifying Inner Activations") freezes the model and learns
    one vector :math:`\ell` per targeted layer that rescales its **output**
    channels, :math:`y = \ell \odot (W x + b)`; for the feed-forward layers of
    a block it rescales the **input** instead, :math:`y = W (\ell \odot x) + b`.
    The vector starts at ones, so the adapted model starts as the base, and it
    folds into the weight (and bias), so a merged model costs what the base does.

    Field names, defaults and ``adapter_config.json`` are the reference
    adapter library's.  Every field is keyword-only.  Unlike the reference's,
    this config also takes ``layers_to_transform`` / ``layers_pattern``, which
    select layers the way ``LoraConfig``'s do.

    Parameters
    ----------
    target_modules : str or list of str, optional
        The ``nn.Linear`` / ``nn.Conv2d`` / ``nn.Conv3d`` modules to adapt,
        matched as ``LoraConfig.target_modules`` is.  ``None`` takes the
        default for the base model's ``model_type`` (``gpt2``, ``bert`` and
        ``t5``; ``IA3_DEFAULT_TARGET_MODULES`` in
        ``lucid.peft.tuners.ia3._defaults``); another model type raises ``ValueError``.
    exclude_modules : str or list of str, optional
        Modules never adapted, matched like ``target_modules``.
    feedforward_modules : str or list of str, optional
        The modules treated as feed-forward (their *input* is scaled): a
        regex the whole name must match, or names matched as
        ``target_modules`` is.  A list must be a subset of ``target_modules``
        (when that is a list too).  ``None`` takes the default for the model type.
    fan_in_fan_out : bool, default=False
        For bases that store their weight ``(in, out)``.  ``nn.Linear`` does
        not, so ``True`` warns and is reset to ``False``.
    modules_to_save : list of str, optional
        Modules trained in full beside the adapter (a classifier head).
    init_ia3_weights : bool, default=True
        Start every vector at ones (the adapted model equals the base).
        ``False`` leaves them random — for tests.

    Raises
    ------
    TypeError
        A field holds a value of the wrong type.
    ValueError
        A field is out of range, or ``feedforward_modules`` is not a subset of ``target_modules``.

    Examples
    --------
    >>> from lucid.peft import IA3Config
    >>> config = IA3Config(target_modules=["k", "v", "wo"], feedforward_modules=["wo"])
    >>> sorted(config.feedforward_modules), config.peft_type
    (['wo'], <PeftType.IA3: 'IA3'>)
    >>> IA3Config.from_dict(config.to_dict()) == config
    True
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "feedforward_modules": _name_set,
        "fan_in_fan_out": _check_bool,
        "init_ia3_weights": _check_bool,
    }

    feedforward_modules: str | list[str] | set[str] | None = None
    fan_in_fan_out: bool = False
    init_ia3_weights: bool = True

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        feedforward, targets = self.feedforward_modules, self.target_modules
        if isinstance(feedforward, str):
            _check_regex("feedforward_modules", feedforward)
        if isinstance(feedforward, set) and isinstance(targets, set) and not feedforward <= targets:
            raise ValueError("`feedforward_modules` should be a subset of `target_modules`")
