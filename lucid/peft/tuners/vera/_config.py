"""``VeraConfig``: the VeRA hyper-parameters on top of the shared targeting fields."""

import warnings
from dataclasses import dataclass
from typing import ClassVar, Final, Literal, override

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_dropout,
    _check_number,
    _check_rank,
    _choice,
    _TargetedConfig,
)

_BIASES: Final[tuple[Literal["none", "all", "vera_only"], ...]] = ("none", "all", "vera_only")


def _check_key(name: str, value: object) -> int:
    """The PRNG key: any ``int`` (``bool`` excluded) the generator takes as a seed."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an int, got {value!r}")
    return value


@dataclass(kw_only=True)
class VeraConfig(_TargetedConfig, peft_type=PeftType.VERA):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with VeRA.

    VeRA (Kopiczko et al., 2023, arXiv:2310.11454 — Vector-based Random Matrix
    Adaptation) freezes the model **and** the low-rank pair: one random matrix
    :math:`A \in \mathbb{R}^{r \times d_{in}}` and one
    :math:`B \in \mathbb{R}^{d_{out} \times r}` are drawn once and shared by every
    targeted layer (sliced to its size), and a layer learns only two vectors,

    .. math:: y = W x + b + \lambda_b \odot B\,(\lambda_d \odot A x),
              \qquad \lambda_d \in \mathbb{R}^{r},\; \lambda_b \in \mathbb{R}^{d_{out}}.

    :math:`\lambda_b` starts at zero, so the adapted model starts as the base.
    The update folds into the weight exactly, so a merged model costs what the
    base does.

    Field names, defaults and ``adapter_config.json`` are the reference
    adapter library's.  Every field is keyword-only.  Unlike the reference's,
    this config also takes ``exclude_modules``, which selects modules the way
    ``LoraConfig``'s does.

    Parameters
    ----------
    r : int, default=256
        The shared projections' rank.  VeRA trains far fewer parameters than LoRA
        at a given rank, so it takes ranks far above LoRA's.
    target_modules : str or list of str, optional
        The ``nn.Linear`` modules to adapt, matched as ``LoraConfig.target_modules``
        is.  ``None`` takes the default for the base model's ``model_type``
        (``VERA_DEFAULT_TARGET_MODULES`` in ``lucid.peft.tuners.vera._defaults``);
        another model type raises ``ValueError``.
    projection_prng_key : int, default=0
        Seed of the generator the shared ``vera_A`` / ``vera_B`` are drawn from
        (Kaiming-uniform, as the reference does).  The same key gives the same
        projections on every device, so an adapter saved without them
        (``save_projection=False``) reloads exactly.
    save_projection : bool, default=True
        Write ``vera_A`` / ``vera_B`` into the adapter file beside the per-layer
        vectors.  ``False`` makes the file small and reloads the projections from
        ``projection_prng_key`` (warns: it relies on the generator staying the same).
    vera_dropout : float, default=0.0
        Dropout on the update's input, in ``[0, 1)``.
    d_initial : float, default=0.1
        The value :math:`\lambda_d` starts at; small values (0.1 or less) are
        recommended (Table 6c of the paper).
    fan_in_fan_out : bool, default=False
        For bases that store their weight ``(in, out)``.  ``nn.Linear`` does not,
        so ``True`` warns and is reset to ``False``.
    bias : {"none"}, default="none"
        The reference also offers ``"all"`` and ``"vera_only"``; here the adapter
        trains its own vectors only, and those raise ``NotImplementedError``.
    modules_to_save : list of str, optional
        Modules trained in full beside the adapter (a classifier head).
    init_weights : bool, default=True
        ``True``: :math:`\lambda_d = d_{initial}` and :math:`\lambda_b = 0`.
        ``False``: :math:`\lambda_d \sim \mathcal N(0, 1)` and :math:`\lambda_b = 1`
        (the update is non-zero from the start — for tests).
    exclude_modules, layers_to_transform, layers_pattern
        As in ``LoraConfig``.  The reference's config has no ``exclude_modules``,
        so ``adapter_config.json`` holds it only when it is set.

    Raises
    ------
    TypeError
        A field holds a value of the wrong type.
    ValueError
        A field is out of range or two fields conflict.
    NotImplementedError
        ``bias`` is not ``"none"``.

    Examples
    --------
    >>> from lucid.peft.tuners.vera import VeraConfig
    >>> config = VeraConfig(r=128, target_modules=["q_proj", "v_proj"], projection_prng_key=7)
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.VERA: 'VERA'>)
    >>> VeraConfig.from_dict(config.to_dict()) == config
    True
    """

    _extension_fields: ClassVar[frozenset[str]] = frozenset({"exclude_modules"})
    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "projection_prng_key": _check_key,
        "save_projection": _check_bool,
        "vera_dropout": _check_dropout,
        "d_initial": _check_number,
        "fan_in_fan_out": _check_bool,
        "bias": _choice(_BIASES),
        "init_weights": _check_bool,
    }

    r: int = 256
    projection_prng_key: int = 0
    save_projection: bool = True
    vera_dropout: float = 0.0
    d_initial: float = 0.1
    fan_in_fan_out: bool = False
    bias: Literal["none", "all", "vera_only"] = "none"
    init_weights: bool = True

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if self.bias != "none":
            raise NotImplementedError(
                f"bias={self.bias!r}: a VeRA adapter in lucid.peft trains its own vectors only "
                "(bias='none')"
            )
        if not self.save_projection:
            warnings.warn(
                "Specified to not save vera_A and vera_B within the state dictionary, instead "
                "they will be restored using the PRNG key store in `config.projection_prng_key`. "
                "Consider setting `config.save_projection` to `True` to guarantee restoring the "
                "checkpoint correctly on all system configurations.",
                stacklevel=2,
            )
