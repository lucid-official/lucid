"""``RandLoraConfig``: the RandLoRA hyper-parameters on top of the shared targeting fields."""

from dataclasses import dataclass, field
from typing import ClassVar, Final, Literal, override

from lucid.peft._config import (
    PeftType,
    _Check,
    _choice,
    _check_rank,
    _check_number,
    _check_dropout,
    _check_bool,
)
from lucid.peft.tuners._shared import _ProjectionConfig

_BIASES: Final[tuple[Literal["none", "all", "randlora_only"], ...]] = (
    "none",
    "all",
    "randlora_only",
)


@dataclass
class RandLoraConfig(_ProjectionConfig, peft_type=PeftType.RANDLORA):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with RandLoRA.

    RandLoRA (Albert et al., 2025, arXiv:2502.00987) reaches a *full-rank* update
    with few parameters: instead of one trainable low-rank pair it sums
    :math:`n = \lceil \min(d_{out}, d_{in}) / r \rceil` frozen random rank-:math:`r`
    pairs :math:`(B_i, A_i)`, drawn once and shared by every targeted layer, and
    trains only a diagonal scale on each side,

    .. math:: \Delta W = \frac{\alpha}{r} \sum_{i=1}^{n} B_i\,
              \mathrm{diag}(\lambda_i)\,A_i\,\mathrm{diag}(\gamma_i),

    with :math:`\lambda \in \mathbb{R}^{r \times n}` and
    :math:`\gamma \in \mathbb{R}^{n \times \min(d_{out}, d_{in})}` per layer.  Lowering
    :math:`r` *raises* the number of trained parameters.  :math:`\lambda` starts at
    zero, so the adapted model starts as the base; the update folds into the weight.

    Names, order, defaults and the ``adapter_config.json`` layout follow the reference
    adapter library.  ``exclude_modules`` is an addition
    (it selects modules as ``LoraConfig``'s does).

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    r : int, default=32
        The random bases' rank.  All adapters of a model share one ``r``.
    target_modules : str or list of str, optional
        The ``nn.Linear`` modules to adapt, matched as ``LoraConfig.target_modules``
        is.  ``None`` takes the LoRA default for the base model's ``model_type``.
    projection_prng_key : int, default=0
        Seed of the generator the shared ``randlora_A`` / ``randlora_B`` are drawn
        from; the same key gives the same bases on every device.
    save_projection : bool, default=True
        Write ``randlora_A`` / ``randlora_B`` into the adapter file.  ``False`` makes
        the file small and redraws them from ``projection_prng_key`` on load (warns).
    sparse : bool, default=False
        Use ternary sparse bases (entries -1, 0, 1 with probabilities 1/6, 2/3, 1/6).
        The bases are still dense tensors; the sparsity is not used for speed.
    very_sparse : bool, default=False
        Use very sparse ternary bases (non-zero with probability :math:`1/\sqrt{d_{min}}`).
    randlora_dropout : float, default=0.0
        Dropout on the update's input, in ``[0, 1)``.
    fan_in_fan_out : bool, default=False
        For bases that store their weight ``(in, out)``.  ``nn.Linear`` does not,
        so ``True`` warns and is reset to ``False``.
    randlora_alpha : float, default=640
        The update is scaled by ``randlora_alpha / r`` (typically 20 times the rank).
    bias : {"none"}, default="none"
        The reference also offers ``"all"`` and ``"randlora_only"``; here the adapter
        trains its own scales only, and those raise ``NotImplementedError``.
    modules_to_save : list of str, optional
        Modules trained in full beside the adapter (a classifier head).
    init_weights : bool, default=True
        ``True``: :math:`\lambda = 0`.  ``False``: :math:`\lambda \sim \mathcal N(0, 1)`
        (the update is non-zero from the start — for tests).
    layers_to_transform : int or list of int, optional
        As in :class:`~lucid.peft.LoraConfig`.
    layers_pattern : str or list of str, optional
        As in :class:`~lucid.peft.LoraConfig`.
    exclude_modules : str or list of str, optional
        Modules never adapted, matched like ``target_modules``.

    Raises
    ------
    TypeError
        A field has the wrong type (``projection_prng_key`` is not an int, say).
    ValueError
        ``r`` or another number is out of range.
    NotImplementedError
        ``bias`` asks for trained biases.

    Examples
    --------
    >>> from lucid.peft.tuners.randlora import RandLoraConfig
    >>> config = RandLoraConfig(r=16, target_modules=["q_proj", "v_proj"], randlora_alpha=320)
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.RANDLORA: 'RANDLORA'>)
    >>> RandLoraConfig.from_dict(config.to_dict()) == config
    True
    """

    _projection_names: ClassVar[str] = "randlora_A and randlora_B"
    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "sparse": _check_bool,
        "very_sparse": _check_bool,
        "randlora_dropout": _check_dropout,
        "fan_in_fan_out": _check_bool,
        "randlora_alpha": _check_number,
        "bias": _choice(_BIASES),
        "init_weights": _check_bool,
    }

    r: int = 32
    target_modules: str | list[str] | set[str] | None = None
    projection_prng_key: int = 0
    save_projection: bool = True
    sparse: bool = False
    very_sparse: bool = False
    randlora_dropout: float = 0.0
    fan_in_fan_out: bool = False
    randlora_alpha: float = 640
    bias: Literal["none", "all", "randlora_only"] = "none"
    modules_to_save: list[str] | None = None
    init_weights: bool = True
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None
    exclude_modules: str | list[str] | set[str] | None = field(default=None, kw_only=True)

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if self.bias != "none":
            raise NotImplementedError(
                f"bias={self.bias!r}: a RandLoRA adapter in lucid.peft trains its own scales "
                "only (bias='none')"
            )
