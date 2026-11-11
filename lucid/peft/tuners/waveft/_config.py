"""``WaveFTConfig``: the WaveFT hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass, field
from typing import ClassVar, Final, Literal, override

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_int,
    _check_number,
    _check_rank,
    _check_str,
    _choice,
    _keyed,
    _nullable,
    _TargetedConfig,
)


#: The wavelet names a config accepts, the reference's: Daubechies ``db1`` to ``db38``, symlets ``sym2`` to
#: ``sym20`` and coiflets ``coif1`` to ``coif17``.  Filters exist for fewer (``_filters.WAVELET_FAMILIES``).
KNOWN_WAVELETS: Final[tuple[str, ...]] = (
    *(f"db{n}" for n in range(1, 39)),
    *(f"sym{n}" for n in range(2, 21)),
    *(f"coif{n}" for n in range(1, 18)),
)


@dataclass
class WaveFTConfig(_TargetedConfig, peft_type=PeftType.WAVEFT):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with WaveFT.

    WaveFT (Bilican et al., 2025, arXiv:2505.12532 — Exploring Sparsity for Parameter Efficient Fine
    Tuning Using Wavelets) learns ``n_frequency`` coefficients of the weight update in a discrete wavelet basis,
    at random positions of the coefficient matrix, and rebuilds the update by an inverse 2-D wavelet transform of
    the chosen ``wavelet_family``.  The update is multiplied by ``scaling`` and starts at zero.  The number of
    coefficients, not a rank, sets the parameter count of a layer.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    n_frequency : int, default=2592
        Learnable wavelet coefficients per layer, at most the number of entries of the weight.
    scaling : float, default=25.0
        Scale of the reconstructed update, as LoRA's ``lora_alpha``.
    wavelet_family : str, default="db1"
        Wavelet family of the transform, such as ``"db1"`` (Haar), ``"sym2"`` or ``"coif1"``.
    use_idwt : bool, default=True
        Rebuild the update by the inverse transform (``True``); ``False`` writes the
        coefficients into a sparse update directly, which is faster and fits worse.
    random_loc_seed : int, default=777
        Seed of the random positions of the learnable coefficients.
    bias : {"none", "all", "waveft_only"}, default="none"
        Which biases also train with WaveFT: none, all of them, or only those of the adapted
        layers.
    n_frequency_pattern : dict of str to int, optional
        Per-module overrides of ``n_frequency``, keyed by name or regex.
    proportional_parameters : bool, default=False
        Allocate ``n_frequency`` in proportion to each layer's weight size (experimental).
    init_weights : bool, default=True
        ``True``: the coefficients start at zero.  ``False``: small normal values.
    fan_in_fan_out : bool, default=False
        Set for a base that stores its weight as ``(in, out)``; a WaveFT layer follows that
        layout.
    target_modules : str or list of str, optional
        The modules WaveFT adapts, matched as ``LoraConfig.target_modules`` is.
    exclude_modules : str or list of str, optional
        Modules WaveFT never adapts, matched like ``target_modules``.
    modules_to_save, layers_to_transform, layers_pattern
        As in :class:`~lucid.peft.LoraConfig`.

    Examples
    --------
    >>> from lucid.peft import WaveFTConfig
    >>> config = WaveFTConfig(n_frequency=64, wavelet_family='db1', target_modules=['q_proj', 'v_proj'])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.WAVEFT: 'WAVEFT'>)
    >>> WaveFTConfig.from_dict(config.to_dict()) == config
    True

    Notes
    -----
    Every field of the reference library's config is here, in its order and with its defaults, and the
    config is read and written as ``adapter_config.json``.  A field of the wrong type raises
    ``TypeError``.  ``ValueError`` is raised when ``n_frequency`` is not positive, or ``wavelet_family``
    is not a name the reference accepts (``db1`` to ``db38``, ``sym2`` to ``sym20``, ``coif1`` to
    ``coif17``).  The filters of ``db1`` to ``db10``, ``sym2`` to ``sym10`` and ``coif1`` to ``coif5``
    are built in; a layer with ``use_idwt=True`` and another family raises ``ValueError`` when it is
    built, and one with ``use_idwt=False`` never reads the family.
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "n_frequency": _check_rank,
        "scaling": _check_number,
        "wavelet_family": _check_str,
        "use_idwt": _check_bool,
        "random_loc_seed": _check_int,
        "fan_in_fan_out": _check_bool,
        "bias": _choice(("none", "all", "waveft_only")),
        "n_frequency_pattern": _nullable(_keyed(_check_rank)),
        "proportional_parameters": _check_bool,
        "init_weights": _check_bool,
    }

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if self.wavelet_family not in KNOWN_WAVELETS:
            raise ValueError(
                f"Wavelet family {self.wavelet_family} not supported. "
                f"Supported wavelet families are: {list(KNOWN_WAVELETS)}"
            )

    n_frequency: int = 2592
    scaling: float = 25.0
    wavelet_family: str = "db1"
    use_idwt: bool = True
    random_loc_seed: int = 777
    fan_in_fan_out: bool = False
    target_modules: str | list[str] | set[str] | None = None
    exclude_modules: str | list[str] | set[str] | None = None
    bias: Literal["none", "all", "waveft_only"] = "none"
    modules_to_save: list[str] | None = None
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None
    n_frequency_pattern: dict[str, int] | None = field(default_factory=dict)
    proportional_parameters: bool = False
    init_weights: bool = True
