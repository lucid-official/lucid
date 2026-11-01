"""``CPTConfig``: context-aware prompt tuning — the prompt's tokens, their types, and the loss and projection settings."""

from dataclasses import dataclass
from typing import ClassVar, Literal, Self, override

from lucid.peft._config import (
    PeftType,
    PromptLearningConfig,
    TaskType,
    _check_number,
    _choice,
    _Check,
    _optional_str,
)

_LOSS_WEIGHTINGS = ("none", "decay")


def _ints(name: str, value: object) -> list[int] | None:
    """``None``, or a list of ``int`` (``bool`` excluded)."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"`{name}` must be a list of int or None, got {value!r}")
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"`{name}` holds {item!r}, which is not an int")
        out.append(item)
    return out


def _non_negative(name: str, value: object) -> float:
    number = _check_number(name, value)
    if number < 0:
        raise ValueError(f"`{name}` must not be negative, got {number}")
    return number


@dataclass(kw_only=True)
class CPTConfig(PromptLearningConfig, peft_type=PeftType.CPT):
    r"""How :func:`lucid.peft.get_peft_model` adapts a causal language model with context-aware prompt tuning.

    CPT (Blau et al., 2024, arXiv:2410.17222) tunes a prompt made of in-context examples: the
    prompt's tokens start as the embeddings of ``cpt_token_ids`` and a trainable *delta* is
    added to each.  Each token has a type (``cpt_tokens_type_mask``, read modulo 4: ``1`` an input
    template, ``2`` an input, ``3`` an output template, ``0`` or ``4`` an output or padding) that
    says which deltas train (types 1-3 only) and how far a delta may move: after every step its
    norm is projected back onto a ball of radius ``opt_projection_epsilon`` (inputs) or
    ``opt_projection_format_epsilon`` (templates), scaled by :math:`\sqrt{d/2048}`.  The loss
    covers the output tokens of the prompt and of the batch, optionally down-weighted for earlier
    examples (``opt_weighted_loss_type="decay"``).

    Field names, defaults and ``adapter_config.json`` are the reference adapter library's.  Every
    field is keyword-only; ``num_virtual_tokens`` is ``len(cpt_token_ids)`` whatever is passed.

    Parameters
    ----------
    cpt_token_ids : list of int, optional
        The vocabulary ids whose embeddings the prompt starts from; ``[0]`` by default.
    cpt_mask : list of int, optional
        One entry per token (kept for the file's sake); ones by default.
    cpt_tokens_type_mask : list of int, optional
        One type per token, as above; ones by default.
    opt_weighted_loss_type : {"none", "decay"}, default="none"
        ``"decay"`` weights the loss of each earlier output segment of an example by a further
        factor of ``opt_loss_decay_factor``.
    opt_loss_decay_factor : float, default=1.0
        The factor of ``"decay"``.
    opt_projection_epsilon : float, default=0.1
        Radius of the ball a delta of an input token stays in.
    opt_projection_format_epsilon : float, default=0.1
        Radius for the deltas of template tokens.
    tokenizer_name_or_path : str, optional
        Kept for the file's sake; the prompt's tokens are given as ids.

    Raises
    ------
    TypeError
        A field holds a value of the wrong type.
    ValueError
        ``task_type`` is not ``"CAUSAL_LM"``, or the three per-token lists differ in length.

    Examples
    --------
    >>> from lucid.peft import CPTConfig
    >>> config = CPTConfig(cpt_token_ids=[5, 6, 7], cpt_tokens_type_mask=[1, 2, 4], task_type="CAUSAL_LM")
    >>> config.num_virtual_tokens, config.cpt_mask
    (3, [1, 1, 1])
    >>> CPTConfig.from_dict(config.to_dict()) == config
    True
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "cpt_token_ids": _ints,
        "cpt_mask": _ints,
        "cpt_tokens_type_mask": _ints,
        "opt_weighted_loss_type": _choice(_LOSS_WEIGHTINGS),
        "opt_loss_decay_factor": _non_negative,
        "opt_projection_epsilon": _non_negative,
        "opt_projection_format_epsilon": _non_negative,
        "tokenizer_name_or_path": _optional_str,
    }

    cpt_token_ids: list[int] | None = None
    cpt_mask: list[int] | None = None
    cpt_tokens_type_mask: list[int] | None = None
    opt_weighted_loss_type: Literal["none", "decay"] = "none"
    opt_loss_decay_factor: float = 1.0
    opt_projection_epsilon: float = 0.1
    opt_projection_format_epsilon: float = 0.1
    tokenizer_name_or_path: str | None = None

    @classmethod
    @override
    def _blank(cls) -> Self:
        # The per-token lists are derived from one another; ``from_dict`` fills in only what a
        # file names, so none of the defaults ``cls()`` derived may stay behind.
        config = cls(task_type=TaskType.CAUSAL_LM)
        config.cpt_token_ids = config.cpt_mask = config.cpt_tokens_type_mask = None
        return config

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if self.task_type != TaskType.CAUSAL_LM:
            raise ValueError(
                f"{type(self).__name__} only supports task_type = {TaskType.CAUSAL_LM.value}."
            )
        if self.cpt_token_ids is None:
            self.cpt_token_ids = [0]
        ids = self.cpt_token_ids
        if self.cpt_mask is None:
            self.cpt_mask = [1] * len(ids)
        if self.cpt_tokens_type_mask is None:
            self.cpt_tokens_type_mask = [1] * len(ids)
        if not len(ids) == len(self.cpt_mask) == len(self.cpt_tokens_type_mask):
            raise ValueError(
                "cpt_token_ids, cpt_mask and cpt_tokens_type_mask must have the same length."
            )
        self.num_virtual_tokens = len(ids)
        self.num_transformer_submodules = 1
