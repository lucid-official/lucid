"""``LoHaConfig``: the LoHa hyper-parameters on top of the LyCORIS fields."""

from dataclasses import dataclass, field
from typing import ClassVar

from lucid.peft._config import PeftType, _Check, _check_bool
from lucid.peft.tuners._lycoris import _LycorisConfig


@dataclass
class LoHaConfig(_LycorisConfig, peft_type=PeftType.LOHA):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with LoHa.

    LoHa (Hyeon-Woo et al., 2021, arXiv:2108.06098 — the FedPara
    parameterisation, in the LyCORIS form) learns a weight-shaped update as the
    Hadamard product of two low-rank products,

    .. math:: \Delta W = \frac{\alpha}{r}\,(W_{1a} W_{1b}) \odot (W_{2a} W_{2b}),

    whose rank can reach :math:`r^2` for :math:`2r(d_{out} + d_{in})`
    parameters.  :math:`W_{2b}` starts at zero, so the adapted model starts as
    the base.  A targeted ``nn.Linear``, ``nn.Conv1d`` or ``nn.Conv2d`` computes
    :math:`f(x) + \Delta W\!\cdot x` with :math:`\Delta W` applied like the
    base weight.

    Field names, their order, defaults and ``adapter_config.json`` are the reference
    adapter library's.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    rank_pattern : dict of str to int, optional
        Per-module rank overrides, keyed by name or regex suffix.
    alpha_pattern : dict of str to float, optional
        Per-module :math:`\alpha` overrides, keyed the same way.
    r : int, default=8
        Rank of the two low-rank products.
    alpha : float, default=8
        :math:`\alpha` in the scale :math:`\alpha / r`.
    rank_dropout : float, default=0.0
        During training, the probability of zeroing an output unit's row of
        :math:`\Delta W` (the kept rows are scaled up to keep the mean).
    module_dropout : float, default=0.0
        During training, the probability of skipping the adapter for a forward pass.
    use_effective_conv2d : bool, default=False
        Factor a convolution's kernel with a spatial core (``hada_t1`` /
        ``hada_t2``) instead of flattening it ("Proposition 3" of FedPara).
        Ignored for ``nn.Linear`` and for kernels of size 1.
    target_modules : str or list of str, optional
        The ``nn.Linear`` / ``nn.Conv1d`` / ``nn.Conv2d`` modules to adapt,
        matched as ``LoraConfig.target_modules`` is.  ``None`` takes the
        default for the base model's ``model_type`` — the LoRA table, as
        the reference does.
    exclude_modules : str or list of str, optional
        Modules never adapted, matched like ``target_modules``.
    init_weights : bool, default=True
        ``True``: :math:`W_{2b} = 0` and the rest Kaiming-uniform, so the update starts at zero.
        ``False``: every factor Kaiming-uniform (for tests).
    layers_to_transform : int or list of int, optional
        As in :class:`~lucid.peft.LoraConfig`.
    layers_pattern : str or list of str, optional
        As in :class:`~lucid.peft.LoraConfig`.
    modules_to_save : list of str, optional
        As in :class:`~lucid.peft.LoraConfig`.

    Raises
    ------
    TypeError
        A field holds a value of the wrong type.
    ValueError
        A field is out of range or two fields conflict.

    Examples
    --------
    >>> from lucid.peft import LoHaConfig
    >>> config = LoHaConfig(r=4, alpha=8, target_modules=["q_proj", "v_proj"])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.LOHA: 'LOHA'>)
    >>> LoHaConfig.from_dict(config.to_dict()) == config
    True
    """

    _field_checks: ClassVar[dict[str, _Check]] = {"init_weights": _check_bool}

    rank_pattern: dict[str, int] = field(default_factory=dict)
    alpha_pattern: dict[str, float] = field(default_factory=dict)
    r: int = 8
    alpha: float = 8
    rank_dropout: float = 0.0
    module_dropout: float = 0.0
    use_effective_conv2d: bool = False
    target_modules: str | list[str] | set[str] | None = None
    exclude_modules: str | list[str] | set[str] | None = None
    init_weights: bool = True
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None
    modules_to_save: list[str] | None = None
