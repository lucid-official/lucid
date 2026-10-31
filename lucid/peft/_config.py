"""``PeftConfig`` — the base of every adapter config — and what a config file carries.

The field names, defaults and file layout are the reference adapter
library's, so a config written by either side reads on the other.  What
v1 does not implement is refused where it enters — the constructor and
``from_dict``.  A key v1 has no field for is accepted only at a value that
turns its feature off (null, false, empty), which changes nothing the
adapter computes; at any other value it is refused.

Each method's config class registers itself under its ``PeftType`` when
it is defined (``class LoraConfig(..., peft_type=PeftType.LORA)``), and
reading a file dispatches on the type the file names — a new method is a
new subclass, not a new branch here.  Field validation is a table per
class (``_field_checks``), merged down the hierarchy when a subclass is
defined.
"""

import copy
import dataclasses
import enum
import json
import math
import os
import re
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar, Final, Self, override

_CONFIG_NAME: Final = "adapter_config.json"

#: Keys the reference writes, at their "off" value, for features v1 does
#: not have.  Read off, they are dropped without a word; any other key that
#: is off is dropped with a warning; a key that is on is refused.
_INERT_KEYS: Final = frozenset(
    {
        "megatron_config",
        "loftq_config",
        "eva_config",
        "corda_config",
        "layer_replication",
        "trainable_token_indices",
        "use_qalora",
        "alora_invocation_tokens",
        # LoRA fields a method that reads LoRA-shaped files (AdaLoRA) has no use for.
        "alpha_pattern",
        "use_rslora",
        "use_dora",
        "lora_bias",
        "target_parameters",
    }
)

#: Keys that describe where an adapter came from, not what it computes.
_METADATA_KEYS: Final = frozenset(
    {"peft_version", "runtime_config", "megatron_core", "qalora_group_size"}
)


class PeftType(enum.StrEnum):
    """The parameter-efficient method an adapter file holds.

    The members are the reference library's ``peft_type`` strings, so a
    config file names its method the same way on both sides.
    ``lucid.peft`` implements ``LORA``, ``ADALORA``, ``IA3``, ``LOHA``, ``LOKR``, ``VERA`` and
    the prompt-learning methods ``PROMPT_TUNING``, ``PREFIX_TUNING`` and ``P_TUNING``;
    reading a file of any other type raises ``NotImplementedError``.
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


#: A file that names no ``peft_type`` is read as the one method v1 implements.
_DEFAULT_PEFT_TYPE: Final = PeftType.LORA


class TaskType(enum.StrEnum):
    """The task an adapter was trained for, as ``adapter_config.json`` records it.

    It chooses the model class :func:`lucid.peft.get_peft_model` and
    ``PeftModel.from_pretrained`` return: ``PeftModelForCausalLM`` for
    ``CAUSAL_LM`` and so on (``MODEL_TYPE_TO_PEFT_MODEL_MAPPING``), a plain
    ``PeftModel`` for an adapter with no task.  A classification or
    question-answering wrapper makes the model's head trainable when it
    has one under the reference's conventional name (``classifier`` /
    ``score``, ``qa_outputs``).
    """

    SEQ_CLS = "SEQ_CLS"
    SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    CAUSAL_LM = "CAUSAL_LM"
    TOKEN_CLS = "TOKEN_CLS"
    QUESTION_ANS = "QUESTION_ANS"
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"


# ── field checks: each takes the field's name and value, returns it normalised ──

#: A field check: ``check(name, value)`` returns the value to store or raises.
_Check = Callable[[str, object], object]


def _is_off(value: object) -> bool:
    """``None``, ``False`` or an empty container / string: a feature switched off."""
    return value is None or value is False or (isinstance(value, (dict, list, str)) and not value)


def _layer_regex(pattern: str | None) -> str:
    """The regex whose group 1 is a module name's layer index.

    The reference's current rule: the first (outer) ``<list>.<index>.``
    segment, matched non-greedily — ``encoder.layers.0.blocks.1.qkv`` is
    layer 0.  Unlike the reference, a layer list at the model's root
    (``h.0.attn``) is indexed too when no ``layers_pattern`` is given.
    """
    if pattern is None:
        return r"(?:.*?\.)?[^.]*\.(\d+)\."
    return rf"(?:.*?\.)?{pattern}\.(\d+)\."


def _pattern_key_regex(key: str) -> str:
    """The regex a ``rank_pattern`` / ``alpha_pattern`` key matches a module name with."""
    return rf"(.*\.)?({key})$"


def _check_regex(name: str, pattern: str) -> None:
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"`{name}` is not a valid regular expression: {exc}") from exc


def _check_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"`{name}` must be a bool, got {value!r}")
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


def _check_dropout(name: str, value: object) -> float:
    p = _check_number(name, value)
    if not 0.0 <= p < 1.0:
        raise ValueError(f"`{name}` must lie in [0, 1), got {p}")
    return p


def _check_auto_mapping(name: str, value: object) -> dict[str, str] | None:
    """``None``, or the ``{"base_model_class": ..., "parent_library": ...}`` dict of strings."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"`{name}` must be a dict of str to str or None, got {value!r}")
    mapping: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise TypeError(f"`{name}` must map str to str, got {key!r}: {item!r}")
        mapping[key] = item
    return mapping


def _optional_str(name: str, value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise TypeError(f"`{name}` must be a str or None, got {value!r}")
    return value


def _strings(name: str, value: object) -> list[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise TypeError(f"`{name}` must be a str or a list of str, got {value!r}")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError(f"`{name}` holds {item!r}, which is not a str")
        out.append(item)
    return out


def _optional_strings(name: str, value: object) -> list[str] | None:
    return None if value is None else _strings(name, value)


def _name_set(name: str, value: object) -> str | set[str] | None:
    """Module names: one regex (``str``) or a set of suffixes, as the reference keeps them."""
    if value is None or isinstance(value, str):
        return value
    return set(_strings(name, value))


def _choice[T: str](choices: tuple[T, ...]) -> Callable[[str, object], T]:
    """A check that accepts exactly one of ``choices``."""

    def check(name: str, value: object) -> T:
        for choice in choices:
            if value == choice:
                return choice
        raise ValueError(f"`{name}` must be one of {', '.join(map(repr, choices))}, got {value!r}")

    return check


def _check_layers(name: str, value: object) -> int | list[int] | None:
    if value is None or (isinstance(value, int) and not isinstance(value, bool)):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(i, int) and not isinstance(i, bool) for i in value
    ):
        return [int(i) for i in value]
    raise TypeError(f"`{name}` must be an int or a list of int, got {value!r}")


def _check_patterns(name: str, value: object) -> str | list[str] | None:
    if value is None or isinstance(value, str):
        return value
    return _strings(name, value)


def _keyed[V](each: Callable[[str, object], V]) -> Callable[[str, object], dict[str, V]]:
    """A check for a ``{module pattern: value}`` dict, each value checked by ``each``."""

    def check(name: str, value: object) -> dict[str, V]:
        if not isinstance(value, Mapping):
            raise TypeError(f"`{name}` must be a dict keyed by module pattern, got {value!r}")
        return {str(k): each(f"{name}[{k!r}]", v) for k, v in value.items()}

    return check


def _task_type(name: str, value: object) -> TaskType | None:
    if value is None:
        return None
    if isinstance(value, str) and value in TaskType.__members__.values():
        return TaskType(value)
    raise ValueError(
        f"Invalid task type: `{name}`={value!r}. Must be one of the following task types: "
        f"{', '.join(TaskType)}."
    )


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


def _adapter_directory(model_id: str | os.PathLike[str], subfolder: str | None = None) -> str:
    """The directory an adapter is read from: ``model_id``, or its ``subfolder``."""
    path = os.fspath(model_id)
    return path if subfolder is None else os.path.join(path, subfolder)


def _read_json_object(path: str | os.PathLike[str]) -> dict[str, object]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        return data
    raise ValueError(f"{os.fspath(path)} does not hold a JSON object")


#: ``PeftType`` → the config class that reads it, filled as each is defined.
_CONFIG_CLASSES: dict[PeftType, type[PeftConfig]] = {}


@dataclass(kw_only=True)
class PeftConfig:
    r"""What every adapter config holds, and the ``adapter_config.json`` round trip.

    The base of each method's config (``LoraConfig``), with the fields
    the reference's ``PeftConfig`` has.  Called on ``PeftConfig`` itself,
    ``from_pretrained`` / ``from_dict`` read the method the file names and
    return that method's config — the reference's idiom for learning which
    base model an adapter needs before loading either.  It is not built
    directly: a config of no method adapts nothing.

    Parameters
    ----------
    task_type : TaskType or str, optional
        The task the adapter is for (metadata; see ``TaskType``).
    peft_type : PeftType or str, optional
        The method; filled in by the subclass when omitted, refused when
        it names another.
    auto_mapping : dict of str to str, optional
        The class of the base model, ``{"base_model_class": name,
        "parent_library": module}`` — what ``AutoPeftModel*`` rebuilds the
        base with.  ``PeftModel.save_pretrained`` writes it.
    base_model_name_or_path : str, optional
        Name of the model the adapter was trained on (metadata).
    revision : str, optional
        Revision of that model (metadata).
    inference_mode : bool, default=False
        Freeze the adapter too, for inference.

    Raises
    ------
    TypeError
        Built directly, or a field holds a value of the wrong type.

    Examples
    --------
    >>> import tempfile
    >>> from lucid.peft import LoraConfig, PeftConfig
    >>> with tempfile.TemporaryDirectory() as d:
    ...     LoraConfig(target_modules=["q"], base_model_name_or_path="gpt2").save_pretrained(d)
    ...     config = PeftConfig.from_pretrained(d)
    >>> type(config).__name__, config.base_model_name_or_path
    ('LoraConfig', 'gpt2')
    """

    #: The method a subclass reads, set by ``peft_type=`` in its class statement.
    _peft_type: ClassVar[PeftType | None] = None
    #: This class's own field checks; ``_all_checks`` adds its bases'.
    _field_checks: ClassVar[dict[str, _Check]] = {
        "task_type": _task_type,
        "auto_mapping": _check_auto_mapping,
        "base_model_name_or_path": _optional_str,
        "revision": _optional_str,
        "inference_mode": _check_bool,
    }
    _all_checks: ClassVar[dict[str, _Check]] = dict(_field_checks)
    #: Fields this class has that the reference's config of the same method lacks.  ``to_dict`` writes
    #: one only when it holds a value, so an ordinary adapter's file stays one the reference reads.
    _extension_fields: ClassVar[frozenset[str]] = frozenset()

    task_type: TaskType | str | None = None
    peft_type: PeftType | str | None = None
    auto_mapping: dict[str, str] | None = None
    base_model_name_or_path: str | None = None
    revision: str | None = None
    inference_mode: bool = False

    @override
    def __init_subclass__(cls, *, peft_type: PeftType | None = None, **kwargs: object) -> None:
        """Merge the subclass's field checks and register it under ``peft_type``."""
        super().__init_subclass__(**kwargs)
        cls._all_checks = {**cls._all_checks, **cls.__dict__.get("_field_checks", {})}
        if peft_type is not None:
            cls._peft_type = peft_type
            _CONFIG_CLASSES[peft_type] = cls

    def __post_init__(self) -> None:
        """Run every field check, then the checks across fields."""
        own = type(self)._peft_type
        if own is None:
            raise TypeError(
                f"{type(self).__name__} is the base of the adapter configs; build one of "
                f"{', '.join(c.__name__ for c in _CONFIG_CLASSES.values())}"
            )
        for name, check in self._all_checks.items():
            setattr(self, name, check(name, getattr(self, name)))
        if self.peft_type is not None and self.peft_type != own:
            raise ValueError(
                f"{type(self).__name__} describes a {own} adapter; got peft_type={self.peft_type!r}"
            )
        self.peft_type = own
        self._check_fields()

    def _check_fields(self) -> None:
        """Checks that read more than one field; a subclass adds its own."""

    @classmethod
    def _blank(cls) -> Self:
        """A config to ``from_dict`` fills in: every field at its default (a class with a required field overrides this)."""
        return cls()

    @classmethod
    def _class_for(cls, peft_type: object) -> type[Self]:
        """The registered config class, ``cls`` or below it, that reads ``peft_type``."""
        wanted = (cls._peft_type or _DEFAULT_PEFT_TYPE) if peft_type is None else peft_type
        for klass in _CONFIG_CLASSES.values():
            if klass._peft_type == wanted and issubclass(klass, cls):
                return klass
        raise NotImplementedError(
            f"this is a {wanted} adapter; {cls.__name__} reads "
            f"{', '.join(str(k._peft_type) for k in _CONFIG_CLASSES.values() if issubclass(k, cls))}"
        )

    def to_dict(self) -> dict[str, object]:
        r"""Return the config as the JSON-ready dict ``adapter_config.json`` holds.

        Enums become their strings and sets of module names sorted lists,
        so the result goes straight to :func:`json.dumps` and back through
        ``from_dict``.  A field the reference's config of this method lacks
        (``IA3Config.layers_to_transform``) is written only when it is set,
        so a file that does not use it is one the reference reads too.

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
        return {
            f.name: _json_value(getattr(self, f.name))
            for f in dataclasses.fields(self)
            if f.name not in self._extension_fields or getattr(self, f.name) != f.default
        }

    def _load_fields(self) -> dict[str, object]:
        """The fields to build an adapter with while its tensors are being loaded, if they differ from the file's.

        An initialisation that changes the base model (PiSSA) must not run
        for tensors that are about to be overwritten; the config the model
        keeps is still the file's (``_staged`` / ``_unstaged``).
        """
        return {}

    def _staged(self) -> Self:
        """A copy of this config to build a loaded adapter with."""
        staged = copy.deepcopy(self)
        for name, value in self._load_fields().items():
            setattr(staged, name, value)
        staged.__post_init__()
        return staged

    def _unstaged(self, stored: PeftConfig) -> None:
        """Put this (the file's) config's values back into ``stored``, the config a staged copy left in the model."""
        for name in self._load_fields():
            setattr(stored, name, getattr(self, name))

    @classmethod
    def from_dict(cls, config_dict: Mapping[str, object]) -> Self:
        r"""Build a config from a dict such as a parsed ``adapter_config.json``.

        The constructor cannot do this: a file from the reference carries
        keys v1 has no field for.  Such a key is accepted only at a value
        that turns its feature off — null, false or empty
        (``"loftq_config": {}``, ``"megatron_config": null``) — and dropped (with
        a warning if v1 does not know the key); at any other value it
        raises, as it would change what the adapter computes.  The
        ``peft_version`` provenance key is dropped.  Called on
        ``PeftConfig``, the dict's ``peft_type`` picks the class.

        Parameters
        ----------
        config_dict : Mapping of str to object
            The config's fields, by the reference's names.

        Returns
        -------
        PeftConfig
            The validated config, of the class the dict's method names.

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
        target = cls._class_for(config_dict.get("peft_type"))
        names = {f.name for f in dataclasses.fields(target)}
        values: dict[str, object] = {}
        dropped: list[str] = []
        for key, value in config_dict.items():
            if key in names:
                values[key] = value
            elif key in _METADATA_KEYS:
                continue
            elif not _is_off(value):
                raise NotImplementedError(
                    f"{key}={value!r} is not supported by lucid.peft v1: it would change "
                    "what the adapter computes"
                )
            elif key not in _INERT_KEYS:
                dropped.append(key)
        if dropped:
            warnings.warn(
                f"Unexpected keyword arguments {sorted(dropped)} for class {target.__name__}, "
                "these are ignored.",
                stacklevel=2,
            )
        config = target._blank()
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

    @staticmethod
    def from_json_file(path_json_file: str | os.PathLike[str]) -> dict[str, object]:
        r"""Read a config file into the dict it holds, unchanged but for ``runtime_config``.

        What ``from_pretrained`` reads before it chooses a config class: the
        file's content as a dict, as the reference's ``from_json_file`` returns
        it.  ``from_dict`` turns it into a config.

        Parameters
        ----------
        path_json_file : str or os.PathLike
            The path of a JSON file such as ``adapter_config.json``.

        Returns
        -------
        dict of str to object
            The file's object, without a ``runtime_config`` key: the
            reference warns that such a key is ignored, since a runtime
            configuration is valid at runtime only.

        Raises
        ------
        FileNotFoundError
            There is no file at ``path_json_file``.
        ValueError
            The file does not hold a JSON object.

        Examples
        --------
        >>> import os, tempfile
        >>> from lucid.peft import LoraConfig, PeftConfig
        >>> with tempfile.TemporaryDirectory() as d:
        ...     LoraConfig(r=2, target_modules=["fc"]).save_pretrained(d)
        ...     raw = PeftConfig.from_json_file(os.path.join(d, "adapter_config.json"))
        >>> raw["peft_type"], raw["r"]
        ('LORA', 2)
        """
        data = _read_json_object(path_json_file)
        if "runtime_config" in data:
            warnings.warn(
                "The configuration file contains a `runtime_config` key. This is ignored. "
                "Runtime configurations are only valid at runtime.",
                stacklevel=2,
            )
            del data["runtime_config"]
        return data

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | os.PathLike[str], subfolder: str | None = None
    ) -> Self:
        r"""Read ``adapter_config.json`` from a local adapter directory.

        The file's ``peft_type`` picks the config class: called on
        ``PeftConfig`` this returns a ``LoraConfig`` for a LoRA adapter,
        and called on a method's own class it refuses the file of another
        method.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            A directory holding ``adapter_config.json`` — one written by
            ``save_pretrained`` or by the reference library.  Hub ids
            are not resolved; download the adapter first.
        subfolder : str, optional
            A directory under ``pretrained_model_name_or_path`` that holds
            the file, e.g. the sub-directory ``save_pretrained`` gives each
            adapter but the one named ``"default"``.

        Returns
        -------
        PeftConfig
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
        directory = _adapter_directory(pretrained_model_name_or_path, subfolder)
        path = os.path.join(directory, _CONFIG_NAME)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"no {_CONFIG_NAME} in {directory!r} "
                "(lucid.peft reads adapters from a local directory)"
            )
        return cls.from_dict(cls.from_json_file(path))


def _require_config(name: str, value: object) -> PeftConfig:
    """``value`` if it is an adapter config, else the one ``TypeError`` that says so."""
    if isinstance(value, PeftConfig):
        return value
    raise TypeError(
        f"{name} must be an adapter config "
        f"({', '.join(c.__name__ for c in _CONFIG_CLASSES.values())}), got {type(value).__name__}"
    )


@dataclass(kw_only=True)
class _TargetedConfig(PeftConfig):
    """The fields of a method that adapts chosen modules, and their joint checks.

    Every tuner the reference ships beside prompt tuning picks modules
    the same way; they live here once rather than in each method's
    config, and the matcher (``lucid.peft._targeting``) reads them.
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "target_modules": _name_set,
        "exclude_modules": _name_set,
        "modules_to_save": _optional_strings,
        "layers_to_transform": _check_layers,
        "layers_pattern": _check_patterns,
    }

    target_modules: str | list[str] | set[str] | None = None
    exclude_modules: str | list[str] | set[str] | None = None
    modules_to_save: list[str] | None = None
    layers_to_transform: int | list[int] | None = None
    layers_pattern: str | list[str] | None = None

    @override
    def _check_fields(self) -> None:
        super()._check_fields()
        if isinstance(self.target_modules, str):
            if self.layers_to_transform is not None:
                raise ValueError(
                    "`layers_to_transform` cannot be used when `target_modules` is a str."
                )
            if self.layers_pattern is not None:
                raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")
        if self.layers_pattern and self.layers_to_transform in (None, []):
            raise ValueError(
                "When `layers_pattern` is specified, `layers_to_transform` must also be specified."
            )
        # Every regex the matcher will build is compiled now, so a bad one
        # fails here rather than halfway through an injection.
        for name in ("target_modules", "exclude_modules"):
            pattern = getattr(self, name)
            if isinstance(pattern, str):
                _check_regex(name, pattern)
        patterns = self.layers_pattern
        for pattern in [patterns] if isinstance(patterns, str) else patterns or []:
            _check_regex("layers_pattern", _layer_regex(pattern))


def _optional_count(name: str, value: object) -> int | None:
    """``None``, or a positive ``int`` (``bool`` excluded)."""
    return None if value is None else _check_rank(name, value)


@dataclass(kw_only=True)
class PromptLearningConfig(PeftConfig):
    r"""The fields of a method that learns an input rather than a weight.

    Prompt tuning, prefix tuning and p-tuning put learned vectors in front
    of the input — as extra token embeddings, or as extra keys and values
    of every attention layer.  The sizes they need are the base model's, so
    everything but ``num_virtual_tokens`` can be left ``None`` and is read
    off the model's ``config`` when the adapter is built.

    Parameters
    ----------
    num_virtual_tokens : int, optional
        How many learned positions go in front of the input; required by
        the time the adapter is built.
    token_dim : int, optional
        The model's hidden size.
    num_transformer_submodules : int, optional
        ``1`` for an encoder-only or decoder-only model, ``2`` for an
        encoder-decoder one; ``2`` for ``task_type="SEQ_2_SEQ_LM"`` and ``1``
        otherwise when omitted.
    num_attention_heads : int, optional
        The model's head count.
    num_layers : int, optional
        The model's layer count.
    modules_to_save : list of str, optional
        Modules of the model trained in full beside the prompt (a
        classification head).
    """

    _field_checks: ClassVar[dict[str, _Check]] = {
        "num_virtual_tokens": _optional_count,
        "token_dim": _optional_count,
        "num_transformer_submodules": _optional_count,
        "num_attention_heads": _optional_count,
        "num_layers": _optional_count,
        "modules_to_save": _optional_strings,
    }

    num_virtual_tokens: int | None = None
    token_dim: int | None = None
    num_transformer_submodules: int | None = None
    num_attention_heads: int | None = None
    num_layers: int | None = None
    modules_to_save: list[str] | None = None
