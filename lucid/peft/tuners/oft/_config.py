"""``OFTConfig``: the OFT hyper-parameters on top of the shared targeting fields."""

import re
import warnings
from dataclasses import dataclass
from typing import ClassVar, Final, Literal, override

from lucid.peft._config import (
    PeftType,
    _Check,
    _check_bool,
    _check_count,
    _check_dropout,
    _check_number,
    _check_rank,
    _choice,
    _TargetedConfig,
)

_BIASES = ("none", "all", "oft_only")

#: The reference release that made the Cayley-Neumann form numerically stabler, and the first to
#: write ``peft_version``: a file older than it holds the old parameterisation.
_CAYLEY_NEUMANN_RELEASE: Final = (0, 18, 0)

#: A PEP 440 version, with the optional separators and ``v`` prefix its normal form allows.
_VERSION: Final = re.compile(
    r"v?(?P<release>\d+(?:\.\d+)*)"
    r"(?P<pre>[-_.]?(?:a|b|rc)[-_.]?\d*)?"
    r"(?P<post>[-_.]?post[-_.]?\d*)?"
    r"(?P<dev>[-_.]?dev[-_.]?\d*)?"
    r"(?:\+[a-z0-9]+(?:[-_.][a-z0-9]+)*)?",
    re.IGNORECASE,
)


def _predates(version: str, release: tuple[int, ...]) -> bool:
    """Whether the PEP 440 ``version`` sorts before the final release ``release``.

    A pre-release or development release of ``release`` itself sorts before it; a post-release
    does not.
    """
    found = _VERSION.fullmatch(version.strip())
    if found is None:
        raise ValueError(f"Invalid version: {version!r}")
    numbers = tuple(int(part) for part in found["release"].split("."))
    width = max(len(numbers), len(release))
    padded, bound = (t + (0,) * (width - len(t)) for t in (numbers, release))
    if padded != bound:
        return padded < bound
    return found["post"] is None and (found["pre"] is not None or found["dev"] is not None)


@dataclass
class OFTConfig(_TargetedConfig, peft_type=PeftType.OFT):
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with OFT.

    OFT (Qiu et al., 2023, arXiv:2306.07280 — Orthogonal Finetuning) rotates the
    input space of each targeted layer instead of adding to its weight:
    :math:`y = (W R^\top) x + b` with :math:`R` block-diagonal, each block an
    orthogonal matrix :math:`(I - Q)(I + Q)^{-1}` of a learned skew-symmetric
    :math:`Q`.  A rotation keeps the pairwise angles between the weight's
    neurons, which is what OFT tunes while the weight's spectrum stays.  The
    :math:`Q`\ s start at zero, so the adapted model starts as the base; the
    rotation folds into the weight, so a merged model costs what the base does.

    Field names, their order, defaults and ``adapter_config.json`` are the reference
    adapter library's.

    Parameters
    ----------
    task_type, peft_type, auto_mapping, peft_version, base_model_name_or_path, revision, inference_mode
        As in :class:`~lucid.peft.PeftConfig`.
    r : int, default=0
        Number of blocks per layer.  Give exactly one of ``r`` and
        ``oft_block_size`` (their product is the layer's input size): the default
        ``r=0`` leaves the choice to ``oft_block_size``.
    oft_block_size : int, default=32
        Size of each block.  Pass ``oft_block_size=0`` when you set ``r``.  A size
        that does not divide a layer's input size is moved to the nearest one that
        does, with a warning.
    module_dropout : float, default=0.0
        During training, the probability of replacing a block by the identity.
    target_modules : str or list of str, optional
        The ``nn.Linear`` / ``nn.Conv2d`` modules to adapt, matched as
        ``LoraConfig.target_modules`` is.  ``None`` takes the default for the base
        model's ``model_type`` (the LoRA table, as the reference does).
    fan_in_fan_out : bool, default=False
        For bases that store their weight ``(in, out)``.  ``nn.Linear`` does not,
        so ``True`` warns and is reset to ``False``.
    bias : {"none"}, default="none"
        The reference also offers ``"all"`` and ``"oft_only"``; here the adapter
        trains its own tensors only, and those raise ``NotImplementedError``.
    exclude_modules : str or list of str, optional
        Modules never adapted, matched like ``target_modules``.
    init_weights : bool, default=True
        ``True``: every :math:`Q` starts at zero (the update is the identity).
        ``False``: :math:`Q \sim \mathcal N(0, 0.1^2)` (for tests).
    layers_to_transform : int or list of int, optional
        As in :class:`~lucid.peft.LoraConfig`.
    layers_pattern : str or list of str, optional
        As in :class:`~lucid.peft.LoraConfig`.
    modules_to_save : list of str, optional
        As in :class:`~lucid.peft.LoraConfig`.
    coft : bool, default=False
        The constrained variant: each block's :math:`Q` is projected into a ball
        of radius ``eps`` around zero every time the rotation is computed.
    eps : float, default=6e-5
        The radius of that ball (scaled by one over the square root of the block
        count).  Only read when ``coft`` is set.
    block_share : bool, default=False
        Share one block among all positions: fewer parameters, one rotation.
    use_cayley_neumann : bool, default=True
        Approximate the Cayley transform by a truncated Neumann series
        (matrix products only, orthogonal up to the truncation) instead of an exact solve.
    num_cayley_neumann_terms : int, default=5
        Terms of that series.

    Raises
    ------
    TypeError
        A field holds a value of the wrong type.
    ValueError
        A field is out of range, or ``r`` and ``oft_block_size`` are both set or both zero.
    NotImplementedError
        ``bias`` is not ``"none"``.

    Examples
    --------
    >>> from lucid.peft import OFTConfig
    >>> config = OFTConfig(oft_block_size=16, target_modules=["q_proj", "v_proj"])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.OFT: 'OFT'>)
    >>> OFTConfig.from_dict(config.to_dict()) == config
    True
    >>> OFTConfig(r=4, oft_block_size=0, target_modules=["x"]).r
    4
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "r": _check_count,
        "oft_block_size": _check_count,
        "module_dropout": _check_dropout,
        "fan_in_fan_out": _check_bool,
        "bias": _choice(_BIASES),
        "init_weights": _check_bool,
        "coft": _check_bool,
        "eps": _check_number,
        "block_share": _check_bool,
        "use_cayley_neumann": _check_bool,
        "num_cayley_neumann_terms": _check_rank,
    }

    r: int = 0
    oft_block_size: int = 32
    module_dropout: float = 0.0
    target_modules: str | list[str] | set[str] | None = None
    fan_in_fan_out: bool = False
    bias: Literal["none", "all", "oft_only"] = "none"
    exclude_modules: str | list[str] | set[str] | None = None
    init_weights: bool = True
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None
    modules_to_save: list[str] | None = None
    coft: bool = False
    eps: float = 6e-5
    block_share: bool = False
    use_cayley_neumann: bool = True
    num_cayley_neumann_terms: int = 5

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if self.r == 0 and self.oft_block_size == 0:
            raise ValueError(
                "Either `r` or `oft_block_size` must be non-zero. "
                f"Currently, r = {self.r} and oft_block_size = {self.oft_block_size}."
            )
        if self.r != 0 and self.oft_block_size != 0:
            raise ValueError(
                f"You can only specify either r ({self.r}) or oft_block_size "
                f"({self.oft_block_size}), but not both simultaneously, because "
                "r x oft_block_size == in_features."
            )
        if self.bias != "none":
            raise NotImplementedError(
                f"bias={self.bias!r}: an OFT adapter in lucid.peft trains its own rotations only "
                "(bias='none')"
            )

    @classmethod
    @override
    def check_kwargs(cls, **kwargs: object) -> dict[str, object]:
        r"""Refuse an adapter file written for the OFT the reference had before 0.14.0, and warn about one from before 0.18.0.

        The reference changed OFT in 0.14.0 (``oft_block_size`` replaced the block count as
        the way to size the blocks) and its Cayley-Neumann form in 0.18.0.  A file with no
        ``oft_block_size`` holds weights of the old OFT; one with ``use_cayley_neumann``
        set and a ``peft_version`` older than 0.18.0 holds the old form.
        :meth:`~lucid.peft.PeftConfig.from_pretrained` calls this on an OFT file.

        A file with no ``peft_version`` is not warned about, where the reference reads a missing
        version as 0.0.0.  This library writes none and ``use_cayley_neumann`` defaults to set, so
        the reference's rule would warn on every file this library saved.  The cost is that a
        reference file older than 0.18.0, which holds no version either (the reference added the
        key in 0.18.0), loads without the warning.

        Parameters
        ----------
        **kwargs
            The config's fields, by name.

        Returns
        -------
        dict of str to object
            ``kwargs`` itself, as the base's does.

        Raises
        ------
        ValueError
            ``kwargs`` has no ``oft_block_size``, or, with ``use_cayley_neumann`` set, its
            ``peft_version`` is not a version.
        TypeError
            ``use_cayley_neumann`` is set and ``peft_version`` is neither a string nor ``None``.

        Warns
        -----
        UserWarning
            ``use_cayley_neumann`` is set and the file's ``peft_version`` is older than 0.18.0.

        Examples
        --------
        >>> from lucid.peft import OFTConfig
        >>> OFTConfig.check_kwargs(oft_block_size=32, peft_version="0.18.0")
        {'oft_block_size': 32, 'peft_version': '0.18.0'}
        >>> OFTConfig.check_kwargs(r=8)
        Traceback (most recent call last):
            ...
        ValueError: OFT has been updated since version 0.14.0 of the reference adapter library: ...
        """
        if "oft_block_size" not in kwargs:
            raise ValueError(
                "OFT has been updated since version 0.14.0 of the reference adapter library: "
                "the adapter's weights are incompatible with this OFT. Retrain the adapter "
                "here, or load it with version 0.13.0 of the reference library."
            )
        version = kwargs.get("peft_version")
        if kwargs.get("use_cayley_neumann", False) and version is not None:
            if not isinstance(version, str):
                raise TypeError(f"`peft_version` must be a str, got {version!r}")
            if _predates(version.partition("@")[0], _CAYLEY_NEUMANN_RELEASE):
                warnings.warn(
                    "The Cayley-Neumann parameterization became slightly more numerically stable "
                    "in version 0.18.0 of the reference adapter library. Retrain the adapter here, "
                    "or load it with version 0.17.0 of the reference library to keep the old form.",
                    stacklevel=2,
                )
        return super().check_kwargs(**kwargs)
