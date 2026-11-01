"""``XLoraConfig``: the X-LoRA classifier's shape and the LoRA adapters it mixes."""

import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, override

from lucid.peft._config import (
    PeftConfig,
    PeftType,
    _Check,
    _check_bool,
    _check_dropout,
    _check_number,
    _check_rank,
    _optional_count,
)


def _adapters(name: str, value: object) -> dict[str, str] | None:
    """``None``, or ``{adapter name: directory}`` (a path is kept as its string)."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(
            f"`{name}` must be a dict of adapter name to directory or None, got {value!r}"
        )
    adapters: dict[str, str] = {}
    for key, path in value.items():
        if not isinstance(key, str) or not isinstance(path, (str, os.PathLike)):
            raise TypeError(f"`{name}` must map str to a directory, got {key!r}: {path!r}")
        adapters[key] = os.fspath(path)
    return adapters


def _positive_number(name: str, value: object) -> float:
    number = _check_number(name, value)
    if number <= 0:
        raise ValueError(f"`{name}` must be positive, got {number}")
    return number


@dataclass(kw_only=True)
class XLoraConfig(PeftConfig, peft_type=PeftType.XLORA):
    r"""How :func:`lucid.peft.get_peft_model` mixes LoRA adapters with X-LoRA.

    X-LoRA (Buehler & Buehler, 2024, arXiv:2402.07148) keeps several trained LoRA adapters (the
    experts) and a small classifier on the base model's final hidden states that scales each
    expert per token: a scalings pass computes the hidden states with the experts held at a
    fixed value, the classifier turns them into one weight per expert (per adapted layer if
    ``layerwise_scalings``), and the real pass adds each expert's update with its weight.  Only the
    classifier trains unless ``use_trainable_adapters``.

    Field names, defaults and ``adapter_config.json`` are the reference adapter library's.  Every
    field is keyword-only.  Unlike the reference's, ``hidden_size`` left unset is read off the model
    rather than guessed, and an empty ``adapters`` is refused when the model is built.

    Parameters
    ----------
    hidden_size : int, optional
        Hidden size of the base model, the classifier's input width; the model's when ``None``.
    adapters : dict of str to str, optional
        The experts: each name maps to the directory of a saved LoRA adapter.  They are loaded
        under the names ``"0"``, ``"1"``, ... in this order, which is also the order of the
        classifier's outputs.  Needed when the model is built.
    enable_softmax : bool, default=True
        Normalise each token's expert weights with a softmax.
    enable_softmax_topk : bool, default=False
        Softmax over the ``top_k_lora`` kept experts instead; set ``enable_softmax`` off with it.
    layerwise_scalings : bool, default=False
        One set of weights per adapted layer (the classifier outputs ``n_experts * n_layers``)
        instead of one set shared by every layer.
    xlora_depth : int, default=1
        Linear layers of the classifier.
    xlora_size : int, default=2048
        Width of its hidden layers; unused at depth one.
    xlora_dropout_p : float, default=0.2
        Dropout between its hidden layers; unused at depth one.
    use_trainable_adapters : bool, default=False
        Train the experts too.
    softmax_temperature : float, default=1.0
        Divides the logits before the softmax; lower is sharper.
    top_k_lora : int, optional
        Keep only this many experts per token (the rest weigh zero).
    scaling_pass_value : float, default=0.0
        The weight every expert has in the scalings pass; ``0`` makes that pass the base model.
    global_scaling_weight : float, default=1.0
        A factor on every expert's update.

    Raises
    ------
    TypeError
        A field holds a value of the wrong type.
    ValueError
        A count is not positive, or ``top_k_lora`` is below one.

    Examples
    --------
    >>> from lucid.peft import XLoraConfig
    >>> config = XLoraConfig(hidden_size=64, adapters={"a": "/tmp/a", "b": "/tmp/b"}, xlora_depth=2)
    >>> config.peft_type, sorted(config.adapters)
    (<PeftType.XLORA: 'XLORA'>, ['a', 'b'])
    >>> XLoraConfig.from_dict(config.to_dict()) == config
    True
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "hidden_size": _optional_count,
        "adapters": _adapters,
        "enable_softmax": _check_bool,
        "enable_softmax_topk": _check_bool,
        "layerwise_scalings": _check_bool,
        "xlora_depth": _check_rank,
        "xlora_size": _check_rank,
        "xlora_dropout_p": _check_dropout,
        "use_trainable_adapters": _check_bool,
        "softmax_temperature": _positive_number,
        "top_k_lora": _optional_count,
        "scaling_pass_value": _check_number,
        "global_scaling_weight": _check_number,
    }

    hidden_size: int | None = None
    adapters: dict[str, str] | None = None
    enable_softmax: bool = True
    enable_softmax_topk: bool = False
    layerwise_scalings: bool = False
    xlora_depth: int = 1
    xlora_size: int = 2048
    xlora_dropout_p: float = 0.2
    use_trainable_adapters: bool = False
    softmax_temperature: float = 1.0
    top_k_lora: int | None = None
    scaling_pass_value: float = 0.0
    global_scaling_weight: float = 1.0

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if self.enable_softmax_topk and self.top_k_lora is None:
            warnings.warn("`enable_softmax_topk` enabled `top_k_lora` is not set", stacklevel=2)
        if self.enable_softmax_topk and self.enable_softmax:
            warnings.warn(
                "`enable_softmax_topk` and `enable_softmax` are both enabled. "
                "This will result in worse performance.",
                stacklevel=2,
            )
