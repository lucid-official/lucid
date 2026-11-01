"""``PolyConfig``: the Poly / MHR hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass
from typing import ClassVar, Literal

from lucid.peft._config import PeftType, _Check, _check_bool, _check_rank, _choice, _TargetedConfig

_POLY_TYPES = ("poly",)


@dataclass(kw_only=True)
class PolyConfig(_TargetedConfig, peft_type=PeftType.POLY):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with Poly (multi-skill LoRA with a task router).

    Polytropon (Ponti et al., 2022, arXiv:2202.13914) gives each targeted layer ``n_skills``
    low-rank adapters (skills) and a router that holds one learned logit vector per task; the
    layer's update for a task is the router-weighted mixture of the skills,
    :math:`y = W x + \tfrac{1}{r}\,x\,(\sum_s m_s A_s)(\sum_s m_s B_s)`.  Multi-Head Routing
    (Caccia et al., 2022, arXiv:2211.03831) splits every skill into ``n_splits`` blocks along the
    feature axis, each routed on its own: ``n_splits > 1``.  The weights :math:`m` are the
    sigmoid of the task's logits, normalised to sum to one — in training a relaxed Bernoulli
    sample of them instead, which makes the routing stochastic.

    Field names, defaults and ``adapter_config.json`` are the reference adapter library's.  Every
    field is keyword-only.  Unlike the reference's, this config also takes
    ``layers_to_transform`` / ``layers_pattern``, which select layers the way ``LoraConfig``'s do;
    ``adapter_config.json`` holds them only when they are set.

    Parameters
    ----------
    r : int, default=8
        Rank of every skill.
    target_modules : str or list of str, optional
        The ``nn.Linear`` modules to adapt, matched as ``LoraConfig.target_modules`` is.  ``None``
        takes the default for the base model's ``model_type``.
    exclude_modules : str or list of str, optional
        Modules never adapted, matched like ``target_modules``.
    modules_to_save : list of str, optional
        Modules trained in full beside the adapter (a classifier head).
    init_weights : bool, default=True
        Start ``B`` at zero (the adapted model equals the base).  ``False`` initialises it like a
        linear layer — for tests.
    poly_type : {"poly"}, default="poly"
        The router; Poly's is the only one.
    n_tasks : int, default=1
        How many tasks the router has a logit vector for.
    n_skills : int, default=4
        Skills (low-rank adapters) per layer.
    n_splits : int, default=1
        Blocks each skill is split into along the features; ``> 1`` is Multi-Head Routing.  Must
        divide the input and output size of every adapted layer.

    Raises
    ------
    TypeError
        A field holds a value of the wrong type.
    ValueError
        A count is not positive.

    Examples
    --------
    >>> from lucid.peft import PolyConfig
    >>> config = PolyConfig(r=4, n_tasks=3, n_skills=2, target_modules=["c_attn"])
    >>> sorted(config.target_modules), config.peft_type
    (['c_attn'], <PeftType.POLY: 'POLY'>)
    >>> PolyConfig.from_dict(config.to_dict()) == config
    True
    """

    _extension_fields: ClassVar[frozenset[str]] = frozenset(
        {"layers_to_transform", "layers_pattern"}
    )
    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "init_weights": _check_bool,
        "poly_type": _choice(_POLY_TYPES),
        "n_tasks": _check_rank,
        "n_skills": _check_rank,
        "n_splits": _check_rank,
    }

    r: int = 8
    init_weights: bool = True
    poly_type: Literal["poly"] = "poly"
    n_tasks: int = 1
    n_skills: int = 4
    n_splits: int = 1
