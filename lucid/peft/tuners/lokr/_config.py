"""``LoKrConfig``: the LoKr hyper-parameters on top of the LyCORIS fields."""

from dataclasses import dataclass
from typing import ClassVar, Literal

from lucid.peft._config import PeftType, _Check, _check_bool
from lucid.peft.tuners._lycoris import _LycorisConfig

_LYCORIS: Literal["lycoris"] = "lycoris"


def _check_init_weights(name: str, value: object) -> bool | Literal["lycoris"]:
    """``True``, ``False`` or ``"lycoris"``."""
    if value == _LYCORIS:
        return _LYCORIS
    if isinstance(value, bool):
        return value
    raise ValueError(f"`{name}` must be True, False or 'lycoris', got {value!r}")


def _check_factor(name: str, value: object) -> int:
    """``-1`` (search near the square root) or a positive factor."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an int, got {value!r}")
    if value == 0 or value < -1:
        raise ValueError(f"`{name}` must be -1 or a positive integer, got {value}")
    return value


@dataclass(kw_only=True)
class LoKrConfig(_LycorisConfig, peft_type=PeftType.LOKR):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with LoKr.

    LoKr (Yeh et al., 2023, arXiv:2309.14859 — LyCORIS) learns a
    weight-shaped update as a Kronecker product,

    .. math:: \Delta W = \frac{\alpha}{r}\,W_1 \otimes W_2,

    with :math:`W_1 \in \mathbb{R}^{a \times c}` and
    :math:`W_2 \in \mathbb{R}^{b \times d}` where :math:`d_{out} = ab` and
    :math:`d_{in} = cd` (each dimension is factored into two near-equal
    factors).  A factor is a full matrix when the rank is as large as half its
    larger side and a low-rank product otherwise.  The first factor starts at
    zero, so the adapted model starts as the base.  A targeted ``nn.Linear``,
    ``nn.Conv1d`` or ``nn.Conv2d`` computes :math:`f(x) + \Delta W\!\cdot x`
    with :math:`\Delta W` applied like the base weight.

    The fields, their defaults and ``adapter_config.json`` are the reference
    adapter library's; all are keyword-only.

    Parameters
    ----------
    decompose_both : bool, default=False
        Let the left factor :math:`W_1` be low-rank too (otherwise it is always full).
    decompose_factor : int, default=-1
        The factor to split each dimension by: the first factor is as close to
        it as divides the dimension.  ``-1`` searches near the square root.
    r : int, default=8
        Rank of the low-rank factors (when a factor is low-rank).
    alpha : float, default=8
        The numerator of the scale :math:`\alpha / r`.
    rank_dropout : float, default=0.0
        Training only: the chance that an output unit's row of :math:`\Delta W` is zeroed.
    rank_dropout_scale : bool, default=False
        Scale the rows kept under rank dropout to keep the mean.
    module_dropout : float, default=0.0
        Training only: the chance that a forward pass leaves the adapter out.
    use_effective_conv2d : bool, default=False
        Factor a convolution's :math:`W_2` with a spatial core (``lokr_t2``)
        instead of flattening it; ignored for ``nn.Linear``, kernels of size 1
        and a full-matrix :math:`W_2`.
    init_weights : bool or "lycoris", default=True
        ``True``: :math:`W_1` zero, :math:`W_2` Kaiming-uniform (the update
        starts at zero).  ``"lycoris"``: the LyCORIS repository's way, :math:`W_1`
        Kaiming-uniform and :math:`W_2` zero (also starts at zero).  ``False``:
        every factor Kaiming-uniform (for tests).
    target_modules, exclude_modules, layers_to_transform, layers_pattern, modules_to_save
        As in :class:`~lucid.peft.LoHaConfig`: ``nn.Linear`` / ``nn.Conv1d`` /
        ``nn.Conv2d`` modules matched as in ``LoraConfig``, the LoRA table of
        defaults when ``target_modules`` is ``None``.
    rank_pattern : dict of str to int, optional
        Rank overrides per module name or regex suffix.
    alpha_pattern : dict of str to float, optional
        :math:`\alpha` overrides per module name or regex suffix.

    Raises
    ------
    TypeError
        A field holds a value of the wrong type.
    ValueError
        A field is out of range or two fields conflict.

    Examples
    --------
    >>> from lucid.peft import LoKrConfig
    >>> config = LoKrConfig(r=4, alpha=8, decompose_both=True, target_modules=["q_proj"])
    >>> config.decompose_both, config.peft_type
    (True, <PeftType.LOKR: 'LOKR'>)
    >>> LoKrConfig.from_dict(config.to_dict()) == config
    True
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "decompose_both": _check_bool,
        "decompose_factor": _check_factor,
        "rank_dropout_scale": _check_bool,
        "init_weights": _check_init_weights,
    }

    decompose_both: bool = False
    decompose_factor: int = -1
    rank_dropout_scale: bool = False
    init_weights: bool | Literal["lycoris"] = True
