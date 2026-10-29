"""``LoraConfig`` and the enumerations an adapter's ``adapter_config.json`` carries.

The field names, defaults and file layout are the reference adapter
library's, so a config written by either side reads on the other.  What
v1 does not implement is refused where it enters — the constructor and
``LoraConfig.from_dict`` — never accepted and then ignored.
"""

import dataclasses
import enum
import json
import math
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Self

_CONFIG_NAME = "adapter_config.json"

_BIASES: tuple[Literal["none", "all", "lora_only"], ...] = ("none", "all", "lora_only")

#: Initialisations the reference offers that need machinery v1 lacks
#: (an SVD, a calibration pass, a quantized base).  Matched by prefix:
#: ``pissa_niter_16`` is PiSSA too.
_UNSUPPORTED_INITS = ("pissa", "olora", "eva", "corda", "loftq", "orthogonal")

#: Keys the reference writes for features v1 does not have, with the value
#: that means "off".  Off is accepted (it changes nothing); anything else
#: would change what the adapter computes, so it is refused.
_INERT_DEFAULTS: dict[str, object] = {
    "lora_bias": False,
    "megatron_config": None,
    "loftq_config": {},
    "eva_config": None,
    "corda_config": None,
    "layer_replication": None,
    "trainable_token_indices": None,
    "target_parameters": None,
    "use_qalora": False,
    "alora_invocation_tokens": None,
}

#: Keys that describe where an adapter came from, not what it computes.
_METADATA_KEYS = frozenset(
    {"auto_mapping", "peft_version", "runtime_config", "megatron_core", "qalora_group_size"}
)


class PeftType(enum.StrEnum):
    """The parameter-efficient method an adapter file holds.

    The members are the reference library's ``peft_type`` strings, so a
    config file names its method the same way on both sides.
    ``lucid.peft`` implements ``LORA``; reading a file of any other
    type raises ``NotImplementedError``.
    """

    PROMPT_TUNING = "PROMPT_TUNING"
    MULTITASK_PROMPT_TUNING = "MULTITASK_PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"
    ADALORA = "ADALORA"
    BOFT = "BOFT"
    ADAPTION_PROMPT = "ADAPTION_PROMPT"
    IA3 = "IA3"
    LOHA = "LOHA"
    LOKR = "LOKR"
    OFT = "OFT"
    POLY = "POLY"
    LN_TUNING = "LN_TUNING"
    VERA = "VERA"
    FOURIERFT = "FOURIERFT"
    XLORA = "XLORA"
    HRA = "HRA"
    VBLORA = "VBLORA"
    CPT = "CPT"
    BONE = "BONE"


class TaskType(enum.StrEnum):
    """The task an adapter was trained for, as ``adapter_config.json`` records it.

    Metadata in v1, with one effect: :func:`lucid.peft.get_peft_model`
    makes a classification or question-answering head trainable when the
    model has one under the reference's conventional name (``classifier``
    / ``score``, ``qa_outputs``), as the reference's task wrappers do.
    """

    SEQ_CLS = "SEQ_CLS"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    CAUSAL_LM = "CAUSAL_LM"
    TOKEN_CLS = "TOKEN_CLS"
    QUESTION_ANS = "QUESTION_ANS"
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"


def _check_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"LoraConfig.{name} must be a bool, got {value!r}")
    return value


def _check_rank(name: str, value: object) -> int:
    """A LoRA rank: a positive ``int`` (``bool`` excluded)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an int, got {value!r}")
    if value <= 0:
        raise ValueError(
            f"`{name}` should be a positive integer value but the value passed is {value}"
        )
    return value


def _check_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"`{name}` must be a real number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"`{name}` must be finite, got {value!r}")
    return value


def _check_dropout(value: object) -> float:
    p = _check_number("lora_dropout", value)
    if not 0.0 <= p < 1.0:
        raise ValueError(f"`lora_dropout` must lie in [0, 1), got {p}")
    return p


def _check_init(value: object) -> bool | str:
    """``True`` (Kaiming-uniform A), ``False`` (no special init) or ``"gaussian"``."""
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        raise TypeError(f"`init_lora_weights` must be a bool or a str, got {value!r}")
    if value.lower() == "gaussian":
        return "gaussian"
    if value.lower().startswith(_UNSUPPORTED_INITS):
        raise NotImplementedError(
            f"init_lora_weights={value!r} is not supported by lucid.peft v1 "
            "(True, False and 'gaussian' are)"
        )
    raise ValueError(f"Unknown initialization init_lora_weights={value!r}")


def _optional_str(name: str, value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"LoraConfig.{name} must be a str or None, got {value!r}")
    return value


def _strings(name: str, value: object) -> list[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise TypeError(f"LoraConfig.{name} must be a str or a list of str, got {value!r}")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"LoraConfig.{name} holds {item!r}, which is not a str")
        out.append(item)
    return out


def _name_set(name: str, value: object) -> str | set[str] | None:
    """Module names: one regex (``str``) or a set of suffixes, as the reference keeps them."""
    if value is None or isinstance(value, str):
        return value
    return set(_strings(name, value))


def _check_bias(value: object) -> Literal["none", "all", "lora_only"]:
    for choice in _BIASES:
        if value == choice:
            return choice
    raise ValueError(f"`bias` must be one of {', '.join(map(repr, _BIASES))}, got {value!r}")


def _check_layers(value: object) -> int | list[int] | None:
    if value is None or (isinstance(value, int) and not isinstance(value, bool)):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(i, int) and not isinstance(i, bool) for i in value
    ):
        return [int(i) for i in value]
    raise TypeError(f"`layers_to_transform` must be an int or a list of int, got {value!r}")


def _check_patterns(value: object) -> str | list[str] | None:
    if value is None or isinstance(value, str):
        return value
    return _strings("layers_pattern", value)


def _rank_map(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"`rank_pattern` must be a dict of str to int, got {value!r}")
    return {str(k): _check_rank(f"rank_pattern[{k!r}]", v) for k, v in value.items()}


def _alpha_map(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"`alpha_pattern` must be a dict of str to number, got {value!r}")
    return {str(k): _check_number(f"alpha_pattern[{k!r}]", v) for k, v in value.items()}


def _task_type(value: object) -> TaskType | None:
    if value is None:
        return None
    if isinstance(value, str) and value in TaskType.__members__.values():
        return TaskType(value)
    raise ValueError(
        f"Invalid task type: {value!r}. Must be one of the following task types: "
        f"{', '.join(TaskType)}."
    )


def _lora_type(value: object) -> PeftType:
    if value is None or value == PeftType.LORA:
        return PeftType.LORA
    raise ValueError(f"LoraConfig describes a LORA adapter; got peft_type={value!r}")


def _json_value(value: object) -> object:
    """``value`` as JSON writes it: an enum as its string, a set as a sorted list."""
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    return value


def _output_directory(save_directory: str | os.PathLike[str]) -> str:
    """Create ``save_directory`` if needed and return it as a ``str``."""
    path = os.fspath(save_directory)
    if os.path.isfile(path):
        raise NotADirectoryError(f"Provided path ({path}) should be a directory, not a file")
    os.makedirs(path, exist_ok=True)
    return path


@dataclass(kw_only=True)
class LoraConfig:
    r"""How :func:`lucid.peft.get_peft_model` adapts a model with LoRA.

    LoRA (Hu et al., 2021, arXiv:2106.09685) freezes a pretrained weight
    :math:`W` and learns a low-rank update beside it, so a targeted
    ``nn.Linear`` computes :math:`W x + s \, B A x` with
    :math:`A \in \mathbb{R}^{r \times d_{in}}`,
    :math:`B \in \mathbb{R}^{d_{out} \times r}` and scale
    :math:`s = \alpha / r` (or :math:`\alpha / \sqrt{r}` with rsLoRA).

    Field names, defaults and ``adapter_config.json`` are the reference
    adapter library's.  Every field is keyword-only.

    Parameters
    ----------
    task_type : TaskType or str, optional
        The task the adapter is for (metadata; see ``TaskType``).
    peft_type : PeftType or str, optional
        Always ``PeftType.LORA``; filled in when omitted.
    base_model_name_or_path : str, optional
        Name of the model the adapter was trained on (metadata).
    revision : str, optional
        Revision of that model (metadata).
    inference_mode : bool, default=False
        Freeze the adapter too, for inference.
    r : int, default=8
        Rank of the update.
    target_modules : str or list of str, optional
        The ``nn.Linear`` modules to adapt.  A list matches a module whose
        dotted name equals an entry or ends with ``"." + entry``; a string
        is a regular expression the whole name must match; ``"all-linear"``
        selects every ``nn.Linear`` except the output head and any layer
        whose weight is shared with another module.  Required.
    exclude_modules : str or list of str, optional
        Modules never adapted, matched like ``target_modules``.
    lora_alpha : float, default=8
        :math:`\alpha` in the scale.
    lora_dropout : float, default=0.0
        Dropout on the update's input, in ``[0, 1)``.
    fan_in_fan_out : bool, default=False
        For bases that store their weight ``(in, out)``.  ``nn.Linear``
        does not, so ``True`` warns and is reset to ``False``.
    bias : {"none", "all", "lora_only"}, default="none"
        Which biases also train: none, every parameter named ``bias``, or
        only the biases of the adapted layers.
    use_rslora : bool, default=False
        Scale by :math:`\alpha / \sqrt{r}` (Kalajdzievski, 2023).
    modules_to_save : list of str, optional
        Modules trained in full beside the adapter (a classifier head);
        a trainable copy is kept and saved with it, the original stays.
    init_lora_weights : bool or "gaussian", default=True
        ``True``: Kaiming-uniform :math:`A` (``a = \sqrt 5``) and zero
        :math:`B`, so the adapted model starts as the base.
        ``"gaussian"``: :math:`A \sim \mathcal N(0, 1/r^2)`, zero
        :math:`B`.  ``False``: both keep ``nn.Linear``'s initialisation
        (the update is non-zero from the start — for tests).
    layers_to_transform : int or list of int, optional
        Adapt only these layer indices (the number after the layer list's
        name, ``h.3.attn`` → 3).
    layers_pattern : str or list of str, optional
        Name of the layer list (``"h"``, ``"layers"``); needs
        ``layers_to_transform``.
    rank_pattern : dict of str to int, optional
        Per-module rank overrides, keyed by name or regex suffix.
    alpha_pattern : dict of str to float, optional
        Per-module :math:`\alpha` overrides, keyed the same way.
    use_dora : bool, default=False
        DoRA; not implemented in v1, ``True`` raises.

    Raises
    ------
    TypeError
        A field holds a value of the wrong type.
    ValueError
        A field is out of range or two fields conflict.
    NotImplementedError
        ``use_dora=True`` or an initialisation v1 lacks.

    Examples
    --------
    >>> from lucid.peft import LoraConfig
    >>> config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
    >>> sorted(config.target_modules), config.peft_type
    (['q_proj', 'v_proj'], <PeftType.LORA: 'LORA'>)
    >>> LoraConfig.from_dict(config.to_dict()) == config
    True
    """

    task_type: TaskType | str | None = None
    peft_type: PeftType | str | None = None
    base_model_name_or_path: str | None = None
    revision: str | None = None
    inference_mode: bool = False
    r: int = 8
    target_modules: str | list[str] | set[str] | None = None
    exclude_modules: str | list[str] | set[str] | None = None
    lora_alpha: float = 8
    lora_dropout: float = 0.0
    fan_in_fan_out: bool = False
    bias: Literal["none", "all", "lora_only"] = "none"
    use_rslora: bool = False
    modules_to_save: list[str] | None = None
    init_lora_weights: bool | str = True
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None
    rank_pattern: dict[str, int] = field(default_factory=dict)
    alpha_pattern: dict[str, float] = field(default_factory=dict)
    use_dora: bool = False

    def __post_init__(self) -> None:
        """Validate every field and normalise lists of names to sets."""
        self.task_type = _task_type(self.task_type)
        self.peft_type = _lora_type(self.peft_type)
        self.base_model_name_or_path = _optional_str(
            "base_model_name_or_path", self.base_model_name_or_path
        )
        self.revision = _optional_str("revision", self.revision)
        for name in ("inference_mode", "fan_in_fan_out", "use_rslora", "use_dora"):
            _check_bool(name, getattr(self, name))
        self.r = _check_rank("r", self.r)
        self.lora_alpha = _check_number("lora_alpha", self.lora_alpha)
        self.lora_dropout = _check_dropout(self.lora_dropout)
        self.bias = _check_bias(self.bias)
        self.target_modules = _name_set("target_modules", self.target_modules)
        self.exclude_modules = _name_set("exclude_modules", self.exclude_modules)
        if self.modules_to_save is not None:
            self.modules_to_save = _strings("modules_to_save", self.modules_to_save)
        self.init_lora_weights = _check_init(self.init_lora_weights)
        self.layers_to_transform = _check_layers(self.layers_to_transform)
        self.layers_pattern = _check_patterns(self.layers_pattern)
        self.rank_pattern = _rank_map(self.rank_pattern)
        self.alpha_pattern = _alpha_map(self.alpha_pattern)
        if self.use_dora:
            raise NotImplementedError("use_dora=True: DoRA is not implemented in lucid.peft v1")
        self._check_layer_selection()

    def _check_layer_selection(self) -> None:
        if isinstance(self.target_modules, str):
            if self.layers_to_transform is not None:
                raise ValueError(
                    "`layers_to_transform` cannot be used when `target_modules` is a str."
                )
            if self.layers_pattern is not None:
                raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")
        if self.layers_pattern and not self.layers_to_transform:
            raise ValueError(
                "When `layers_pattern` is specified, `layers_to_transform` must also be specified."
            )

    def to_dict(self) -> dict[str, object]:
        r"""Return the config as the JSON-ready dict ``adapter_config.json`` holds.

        Enums become their strings and sets of module names sorted lists,
        so the result goes straight to :func:`json.dumps` and back through
        ``from_dict``.

        Returns
        -------
        dict of str to object
            One entry per field, keyed by the reference's field names.

        Examples
        --------
        >>> from lucid.peft import LoraConfig
        >>> d = LoraConfig(target_modules=["v", "q"], task_type="CAUSAL_LM").to_dict()
        >>> d["target_modules"], d["task_type"], d["peft_type"]
        (['q', 'v'], 'CAUSAL_LM', 'LORA')
        """
        return {f.name: _json_value(getattr(self, f.name)) for f in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, object]) -> Self:
        r"""Build a config from a dict such as a parsed ``adapter_config.json``.

        Keys for features v1 lacks are accepted only at the value that
        turns the feature off (``"use_dora": false``, ``"loftq_config": {}``,
        ...); any other value raises.  Provenance keys (``peft_version``,
        ``auto_mapping``) are dropped, and a key this version does not know
        is dropped with a warning, as the reference does.

        Parameters
        ----------
        config_dict : Mapping of str to object
            The config's fields, by the reference's names.

        Returns
        -------
        LoraConfig
            The validated config.

        Raises
        ------
        NotImplementedError
            The dict describes another adapter type, or turns on a feature
            v1 lacks.

        Examples
        --------
        >>> from lucid.peft import LoraConfig
        >>> config = LoraConfig.from_dict(
        ...     {"peft_type": "LORA", "r": 4, "target_modules": ["c_attn"],
        ...      "use_dora": False, "loftq_config": {}, "peft_version": "0.17.0"})
        >>> config.r, config.target_modules
        (4, {'c_attn'})
        """
        peft_type = config_dict.get("peft_type")
        if peft_type != PeftType.LORA and peft_type in PeftType.__members__.values():
            raise NotImplementedError(
                f"this is a {peft_type} adapter; lucid.peft implements {PeftType.LORA} only"
            )
        names = {f.name for f in dataclasses.fields(cls)}
        values: dict[str, object] = {}
        dropped: list[str] = []
        for key, value in config_dict.items():
            if key in names:
                values[key] = value
            elif key in _INERT_DEFAULTS:
                if value != _INERT_DEFAULTS[key]:
                    raise NotImplementedError(
                        f"{key}={value!r} is not supported by lucid.peft v1 "
                        f"(only {_INERT_DEFAULTS[key]!r} is)"
                    )
            elif key not in _METADATA_KEYS:
                dropped.append(key)
        if dropped:
            warnings.warn(
                f"Unexpected keyword arguments {sorted(dropped)} for class {cls.__name__}, "
                "these are ignored.",
                stacklevel=2,
            )
        config = cls()
        for key, value in values.items():
            setattr(config, key, value)
        config.__post_init__()
        return config

    def save_pretrained(self, save_directory: str | os.PathLike[str]) -> None:
        r"""Write ``adapter_config.json`` into ``save_directory``.

        The file is the reference's: ``to_dict`` with its keys sorted,
        indented by two spaces.  The directory is created if missing.

        Parameters
        ----------
        save_directory : str or os.PathLike
            Directory to write into.

        Raises
        ------
        NotADirectoryError
            ``save_directory`` names an existing file.

        Examples
        --------
        >>> import tempfile
        >>> from lucid.peft import LoraConfig
        >>> with tempfile.TemporaryDirectory() as d:
        ...     LoraConfig(r=2, target_modules=["fc"]).save_pretrained(d)
        ...     LoraConfig.from_pretrained(d).r
        2
        """
        path = _output_directory(save_directory)
        payload = self.to_dict()
        ordered = {key: payload[key] for key in sorted(payload)}
        with open(os.path.join(path, _CONFIG_NAME), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(ordered, indent=2))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str]) -> Self:
        r"""Read ``adapter_config.json`` from a local adapter directory.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            A directory holding ``adapter_config.json`` — one written by
            ``save_pretrained`` or by the reference library.  Hub ids
            are not resolved; download the adapter first.

        Returns
        -------
        LoraConfig
            The config, read through ``from_dict``.

        Raises
        ------
        FileNotFoundError
            The directory has no ``adapter_config.json``.
        ValueError
            The file does not hold a JSON object.

        Examples
        --------
        >>> import tempfile
        >>> from lucid.peft import LoraConfig
        >>> with tempfile.TemporaryDirectory() as d:
        ...     LoraConfig(r=16, lora_alpha=32, target_modules="all-linear").save_pretrained(d)
        ...     loaded = LoraConfig.from_pretrained(d)
        >>> loaded.r, loaded.lora_alpha, loaded.target_modules
        (16, 32, 'all-linear')
        """
        path = os.path.join(os.fspath(pretrained_model_name_or_path), _CONFIG_NAME)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"no {_CONFIG_NAME} in {os.fspath(pretrained_model_name_or_path)!r} "
                "(lucid.peft reads adapters from a local directory)"
            )
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return cls.from_dict(data)
        raise ValueError(f"{path} does not hold a JSON object")
