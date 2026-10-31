"""``FourierFTConfig``: the FourierFT hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass, field
from typing import ClassVar, Literal, override

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_number,
    _check_rank,
    _check_regex,
    _choice,
    _keyed,
    _pattern_key_regex,
    _TargetedConfig,
)

_BIASES = ("none", "all", "fourier_only")


def _check_seed(name: str, value: object) -> int:
    """A location seed: a non-negative ``int`` (``bool`` excluded)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an int, got {value!r}")
    if value < 0:
        raise ValueError(f"`{name}` must be non-negative, got {value}")
    return value


@dataclass(kw_only=True)
class FourierFTConfig(_TargetedConfig, peft_type=PeftType.FOURIERFT):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with FourierFT.

    FourierFT (Gao et al., 2024, arXiv:2405.03003) learns the weight update in
    the frequency domain: a handful of coefficients :math:`c_1 \dots c_n` of a
    :math:`d_{out} \times d_{in}` spectrum, at fixed pseudo-random locations,
    and the inverse 2-D Fourier transform of that sparse spectrum is the update,

    .. math:: \Delta W = \alpha\,\operatorname{Re}\,\mathcal F^{-1}(S), \qquad
              S_{\ell_j} = c_j,

    with :math:`\alpha` the ``scaling``.  The locations come from
    ``random_loc_seed`` alone, so an adapter file holds only the :math:`c_j`
    (``n_frequency`` numbers per layer) and any copy of the model rebuilds the
    same :math:`\Delta W`.  A targeted ``nn.Linear`` computes
    :math:`f(x) + \Delta W\,x`; the update folds into the weight.

    Field names, defaults and ``adapter_config.json`` are the reference
    adapter library's.  Every field is keyword-only.

    Parameters
    ----------
    n_frequency : int, default=1000
        Number of learned coefficients per layer; at most ``d_out * d_in``.
    scaling : float, default=150.0
        The scale :math:`\alpha` of the update.
    random_loc_seed : int, default=777
        Seed of the generator that picks the locations.  The locations are
        those the reference framework's CPU generator picks for the same seed.
    target_modules : str or list of str, optional
        The ``nn.Linear`` modules to adapt, matched as ``LoraConfig.target_modules``
        is.  ``None`` takes the default for the base model's ``model_type`` (the
        LoRA table with ``gpt2`` narrowed to ``mlp.c_proj``, as the reference does).
    exclude_modules : str or list of str, optional
        Modules never adapted, matched like ``target_modules``.
    fan_in_fan_out : bool, default=False
        For bases that store their weight ``(in, out)``.  ``nn.Linear`` does not,
        so ``True`` warns and is reset to ``False``.
    bias : {"none"}, default="none"
        The reference also offers ``"all"`` and ``"fourier_only"``; here the adapter
        trains its own coefficients only, and those raise ``NotImplementedError``.
    modules_to_save : list of str, optional
        Modules trained in full beside the adapter (a classifier head).
    layers_to_transform, layers_pattern
        As in ``LoraConfig``.
    n_frequency_pattern : dict of str to int, optional
        Per-module ``n_frequency`` overrides, keyed by name or regex suffix.
    init_weights : bool, default=False
        ``False``: the coefficients start as a standard normal sample (the update is
        non-zero from the start, as in the paper).  ``True``: zeros.

    Raises
    ------
    TypeError
        A field holds a value of the wrong type.
    ValueError
        A field is out of range.
    NotImplementedError
        ``bias`` is not ``"none"``.

    Examples
    --------
    >>> from lucid.peft import FourierFTConfig
    >>> config = FourierFTConfig(n_frequency=200, scaling=300.0, target_modules=["q", "v"])
    >>> config.n_frequency, config.peft_type
    (200, <PeftType.FOURIERFT: 'FOURIERFT'>)
    >>> FourierFTConfig.from_dict(config.to_dict()) == config
    True
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "n_frequency": _check_rank,
        "scaling": _check_number,
        "random_loc_seed": _check_seed,
        "fan_in_fan_out": _check_bool,
        "bias": _choice(_BIASES),
        "n_frequency_pattern": _keyed(_check_rank),
        "init_weights": _check_bool,
    }

    n_frequency: int = 1000
    scaling: float = 150.0
    random_loc_seed: int = 777
    fan_in_fan_out: bool = False
    bias: Literal["none", "all", "fourier_only"] = "none"
    n_frequency_pattern: dict[str, int] = field(default_factory=dict)
    init_weights: bool = False

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        for key in self.n_frequency_pattern:
            _check_regex(f"n_frequency_pattern[{key!r}]", _pattern_key_regex(key))
        if self.bias != "none":
            raise NotImplementedError(
                f"bias={self.bias!r}: a FourierFT adapter in lucid.peft trains its own "
                "coefficients only (bias='none')"
            )
