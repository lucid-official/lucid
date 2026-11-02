"""``ShiraConfig``: the SHiRA hyper-parameters on top of the shared targeting fields."""

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import ClassVar, override

from lucid._tensor.tensor import Tensor
from lucid.peft._config import PeftType, _Check, _check_bool, _check_rank, _TargetedConfig
from lucid.peft.tuners.shira._mask import random_mask

#: A mask function: ``mask_fn(base_layer, r, **options)`` → the ``(d_out, d_in)`` 0/1 mask.
MaskFn = Callable[..., Tensor]


def _check_mask_type(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"`{name}` must be a str, got {value!r}")
    return value


def _check_seed(name: str, value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an int or None, got {value!r}")
    return value


@dataclass
class ShiraConfig(_TargetedConfig, peft_type=PeftType.SHIRA):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with SHiRA.

    SHiRA (Bhardwaj et al., 2024, arXiv:2406.13175 — Sparse High Rank Adapters) trains a
    **sparse subset of the weight itself**: a fixed mask picks :math:`r\,(d_{out}+d_{in})`
    entries of each targeted ``nn.Linear`` weight (as many parameters as a rank-:math:`r`
    LoRA pair has, hence the name of ``r``), and the adapter is the values of those
    entries,

    .. math:: y = (W + s\,\Delta)\,x + b, \qquad \Delta_{ij} = \theta_k \text{ if } (i, j)
              \text{ is the } k\text{-th masked entry, else } 0 .

    The update is sparse but not low rank.  :math:`\theta` starts at zero; ``s`` is the
    adapter's scale (1 while training; ``set_scale`` changes it for inference).  The
    mask is not stored in the config: it is a function of the weight's shape and
    ``random_seed`` (or of your own ``mask_fn``), and the adapter file holds the
    indices beside the values.

    Field names, their order, defaults and ``adapter_config.json`` are the reference adapter
    library's.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    r : int, default=32
        Sizes the number of trained entries per layer to ``r (d_out + d_in)``.
    mask_type : str, default="random"
        ``"random"`` picks the mask function; any other value warns and needs your own function in ``mask_fn``.
    random_seed : int, optional
        Seed of the random mask; ``None`` is the reference's default seed.
    target_modules : str or list of str, optional
        The ``nn.Linear`` modules to adapt, matched as ``LoraConfig.target_modules``
        is.  ``None`` takes the LoRA default for the base model's ``model_type``.
    fan_in_fan_out : bool, default=False
        For bases that store their weight ``(in, out)``.  ``nn.Linear`` does not,
        so ``True`` warns and is reset to ``False``.
    init_weights : bool, default=True
        ``True``: zeros.  ``False``: a standard normal sample (the update is non-zero from
        the start — for tests).
    modules_to_save : list of str, optional
        Modules trained in full beside the adapter (a classifier head).
    exclude_modules : str or list of str, optional
        Modules never adapted, matched like ``target_modules``.
    layers_to_transform : int or list of int, optional
        As in :class:`~lucid.peft.LoraConfig`.
    layers_pattern : str or list of str, optional
        As in :class:`~lucid.peft.LoraConfig`.

    Attributes
    ----------
    mask_fn : callable or None
        Not a field and not written to the file: ``random_mask`` for ``mask_type="random"``,
        else ``None`` until you set a function ``(base_layer, r, **options) -> mask``
        returning the ``(d_out, d_in)`` 0/1 tensor with ``r (d_out + d_in)`` ones.

    Raises
    ------
    TypeError
        A field is not of its type.
    ValueError
        ``r`` is not positive.

    Examples
    --------
    >>> from lucid.peft import ShiraConfig
    >>> config = ShiraConfig(r=8, random_seed=1, target_modules=["q_proj", "v_proj"])
    >>> config.peft_type, config.mask_fn is not None
    (<PeftType.SHIRA: 'SHIRA'>, True)
    >>> ShiraConfig.from_dict(config.to_dict()) == config
    True
    """

    _extension_fields: ClassVar[frozenset[str]] = frozenset(
        {"exclude_modules", "layers_to_transform", "layers_pattern"}
    )
    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_rank,
        "mask_type": _check_mask_type,
        "random_seed": _check_seed,
        "fan_in_fan_out": _check_bool,
        "init_weights": _check_bool,
    }

    r: int = 32
    mask_type: str = "random"
    random_seed: int | None = None
    target_modules: str | list[str] | set[str] | None = None
    fan_in_fan_out: bool = False
    init_weights: bool = True
    modules_to_save: list[str] | None = None
    exclude_modules: str | list[str] | set[str] | None = field(default=None, kw_only=True)
    layers_to_transform: int | list[int] | None = field(default=None, kw_only=True)
    layers_pattern: str | list[str] | None = field(default=None, kw_only=True)

    #: Not a field (so not in the file): the instance's own ``mask_fn``, set below, shadows this.
    mask_fn: ClassVar[MaskFn | None] = None

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if "mask_fn" not in self.__dict__:
            self.__dict__["mask_fn"] = random_mask if self.mask_type == "random" else None
        if self.mask_type != "random" and not self.inference_mode:
            warnings.warn(
                f"Argument {self.mask_type=} is not recognized, please supply your own masking "
                "function by calling `config.mask_fn = my_mask_fn`.",
                stacklevel=2,
            )
