"""Base classes for model configuration and pretrained-model contract."""

import json
import os
import warnings
from abc import ABC
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import ClassVar, Self, override

import lucid as _lucid
import lucid.nn as nn


def _list_to_tuple(val: object) -> object:
    """Recursively convert JSON-deserialised lists back into tuples."""
    if isinstance(val, list):
        return tuple(_list_to_tuple(item) for item in val)
    return val


class ModelConfig(ABC):
    r"""Base class for every model's configuration dataclass.

    A ``ModelConfig`` subclass holds the hyper-parameters that fully
    determine a model's architecture (depths, widths, vocab sizes,
    activation choices, …).  Configs are immutable, JSON-serialisable,
    and the single argument every concrete :class:`PretrainedModel`
    constructor accepts.

    Subclasses must:

    - Use ``@dataclass(frozen=True)`` (immutability + ``fields()``
      introspection).
    - Set the ``model_type`` ``ClassVar[str]`` to a unique family
      identifier (e.g. ``"resnet"``, ``"bert"``).
    - Optionally implement ``__post_init__`` for cross-field validation.

    Attributes
    ----------
    model_type : ClassVar[str]
        Persistent family identifier.  Embedded in ``config.json`` on
        disk and used by directory-based loading to find the correct
        registry entry.

    See Also
    --------
    lucid.models._meta.model_family_meta
        Class decorator that attaches paper / theory / display metadata
        (``canonical_name`` / ``citation`` / ``theory``) to a Config
        subclass.  Concrete families wrap their Config with this so the
        docs site can render family-root index pages.

    Notes
    -----
    The base class provides a JSON round-trip via :meth:`to_dict` /
    :meth:`from_dict` / :meth:`save` / :meth:`load`.  Unknown fields are
    tolerated on load with a :class:`UserWarning` — this lets newer
    checkpoints (which may have added fields) load into older code
    without crashing.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> from typing import ClassVar
    >>> from lucid.models import ModelConfig
    >>> @dataclass(frozen=True)
    ... class MyConfig(ModelConfig):
    ...     model_type: ClassVar[str] = "myfamily"
    ...     hidden_size: int = 768
    ...     num_layers: int = 12
    >>> cfg = MyConfig(hidden_size=1024)
    >>> cfg.to_dict()["model_type"]
    'myfamily'

    ``model_type`` is a ClassVar, so it identifies the family rather than
    varying per instance — and it survives the round trip that a saved
    config makes.

    >>> cfg.to_dict()["hidden_size"]
    1024
    """

    model_type: ClassVar[str] = "base"

    @override
    def __init_subclass__(cls, **kwargs: object) -> None:
        r"""Layer-1 contract checks for family Configs (see
        ``arch-models-family-contract``).

        Triggered at class-creation time.  Only validates *concrete*
        family Configs: intermediate abstract bases
        (``LanguageModelConfig``, ``DiffusionModelConfig``,
        ``GenerativeModelConfig``, …) keep ``model_type == "base"`` and
        are silently skipped — they are not user-facing entry points
        themselves, only super-classes.

        Raises
        ------
        TypeError
            When a concrete subclass either keeps ``model_type == "base"``
            (no family identifier set) or has a class name that does not
            end with ``"Config"``.
        """
        super().__init_subclass__(**kwargs)
        # Intermediate abstract bases keep the default and skip validation.
        if cls.model_type == "base":
            return
        if not cls.__name__.endswith("Config"):
            raise TypeError(
                f"{cls.__qualname__}: family Config class name must end with "
                f"'Config' (got {cls.__name__!r}).  See "
                f"arch-models-family-contract.md."
            )

    def to_dict(self) -> dict[str, object]:
        r"""Serialise the config (including ``model_type``) to a plain dict.

        Returns
        -------
        dict[str, object]
            JSON-friendly mapping of every dataclass field plus the
            ``model_type`` ClassVar.

        Raises
        ------
        TypeError
            If ``self`` is not a ``@dataclass``.

        Examples
        --------
        >>> cfg = MyConfig(hidden_size=1024)
        >>> d = cfg.to_dict()
        >>> d["hidden_size"], d["model_type"]
        (1024, 'myfamily')
        """
        if not is_dataclass(self):
            raise TypeError(f"{type(self).__name__} must be decorated with @dataclass")
        d: dict[str, object] = asdict(self)
        d["model_type"] = self.model_type
        return d

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> Self:
        r"""Reconstruct a config from a plain dict (inverse of :meth:`to_dict`).

        Parameters
        ----------
        d : dict[str, object]
            Field-name → value mapping; ``model_type`` is silently
            stripped (it's a ClassVar, not a constructor arg).

        Returns
        -------
        Self
            New instance of ``cls`` populated from ``d``.

        Raises
        ------
        TypeError
            If ``cls`` is not a ``@dataclass``.

        Warns
        -----
        UserWarning
            When ``d`` contains keys not declared on ``cls`` — those keys
            are dropped before invoking the constructor so older code can
            load newer checkpoints with added fields.

        Examples
        --------
        >>> cfg = MyConfig.from_dict({"hidden_size": 512, "model_type": "myfamily"})
        >>> cfg.hidden_size
        512
        """
        copy: dict[str, object] = dict(d)
        # ``model_type`` is a ClassVar — never a constructor arg.
        copy.pop("model_type", None)
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a @dataclass")
        cls_fields = {f.name: f for f in fields(cls)}
        unknown = set(copy) - set(cls_fields)
        if unknown:
            # Warn and skip unknown fields so checkpoints from a newer version
            # of the config (with added fields) still load into older code.
            # Only missing *required* fields (those without defaults) will cause
            # an error — via the TypeError raised by cls(**copy) below.
            warnings.warn(
                f"{cls.__name__}.from_dict: ignoring unrecognised fields "
                f"{sorted(unknown)} — checkpoint may be from a newer version.",
                UserWarning,
                stacklevel=2,
            )
            for k in unknown:
                del copy[k]
        for name, val in list(copy.items()):
            if isinstance(val, list):
                f = cls_fields[name]
                if "tuple" in str(f.type) or isinstance(f.default, tuple):
                    copy[name] = _list_to_tuple(val)
        return cls(**copy)

    def save(self, path: str) -> None:
        r"""Write the config as pretty-printed JSON to ``path``.

        Parameters
        ----------
        path : str
            File path to write.  Existing files are overwritten.

        Examples
        --------
        >>> cfg = MyConfig(hidden_size=1024)
        >>> cfg.save("/tmp/myconfig.json")
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> Self:
        r"""Load a config from a JSON file (inverse of :meth:`save`).

        Parameters
        ----------
        path : str
            Path to a JSON file produced by :meth:`save`.

        Returns
        -------
        Self
            Reconstructed config instance.

        Raises
        ------
        ValueError
            If ``path`` does not contain a JSON object.

        Examples
        --------
        >>> cfg = MyConfig.load("/tmp/myconfig.json")
        """
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            raise TypeError(f"{path} did not contain a JSON object")
        return cls.from_dict(d)


class PretrainedModel(nn.Module):
    r"""Base for every Lucid model that supports the pretrained-checkpoint flow.

    Subclasses inherit :meth:`from_pretrained` / :meth:`save_pretrained`
    plus parameter-counting and embedding-access helpers.  The contract:

    - Set ``config_class = MyConfig`` as a ``ClassVar`` — required;
      enforced at ``__init__`` time.
    - Define ``__init__(self, config: MyConfig) -> None`` taking a single
      config argument.  All architectural variation (depth, width,
      activation choices, …) belongs inside the config — no extra
      constructor parameters.
    - Implement ``forward(...) -> ModelOutput`` returning one of the
      dataclasses in :mod:`lucid.models._output`.

    Attributes
    ----------
    config_class : ClassVar[type[ModelConfig] or None]
        Subclasses MUST override.  ``None`` signals "not set" — the base
        ``__init__`` raises in that case.
    base_model_prefix : ClassVar[str]
        Optional name of the backbone attribute (e.g. ``"bert"`` on
        ``BERTForMaskedLM``).  Used by future state-dict remapping logic
        when loading head-less checkpoints into head-bearing models.
    config : ModelConfig
        Instance attribute populated by ``__init__``.

    Notes
    -----
    The class is a thin lift on :class:`lucid.nn.Module` — all parameter
    management still flows through the standard ``Module`` machinery.
    Two persistence formats are supported when saving / loading: the
    pickle-based ``weights.lucid`` (default) and the cross-framework
    ``model.safetensors``.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from dataclasses import dataclass
    >>> from typing import ClassVar
    >>> from lucid.models import ModelConfig, PretrainedModel
    >>> @dataclass(frozen=True)
    ... class MyConfig(ModelConfig):
    ...     model_type: ClassVar[str] = "myfamily"
    ...     hidden_size: int = 768
    ...     num_classes: int = 10
    >>> class MyModel(PretrainedModel):
    ...     config_class: ClassVar[type[MyConfig]] = MyConfig
    ...     def __init__(self, config):
    ...         super().__init__(config)
    ...         self.linear = nn.Linear(config.hidden_size, config.num_classes)
    ...     def forward(self, x):
    ...         return self.linear(x)
    >>> model = MyModel(MyConfig(hidden_size=128, num_classes=10))
    >>> model.num_parameters()
    1290
    """

    # None signals "not set"; __init__ will raise if a concrete subclass forgets.
    config_class: ClassVar[type[ModelConfig] | None] = None
    base_model_prefix: ClassVar[str] = ""

    config: ModelConfig

    def __init__(self, config: ModelConfig) -> None:
        r"""Initialise the module and validate the supplied config.

        Parameters
        ----------
        config : ModelConfig
            Must be an instance of the subclass's declared
            :attr:`config_class`.

        Raises
        ------
        TypeError
            If ``config_class`` has not been set on the concrete subclass,
            or if ``config`` is not an instance of ``config_class``.
        """
        super().__init__()
        cls = type(self)
        if cls.config_class is None:
            raise TypeError(
                f"{cls.__name__} must set the 'config_class' ClassVar to its "
                f"matching ModelConfig subclass (e.g. "
                f"config_class: ClassVar[type[ResNetConfig]] = ResNetConfig)."
            )
        if not isinstance(config, cls.config_class):
            raise TypeError(
                f"{cls.__name__} expects {cls.config_class.__name__}, got {type(config).__name__}"
            )
        self.config = config

    @classmethod
    def from_pretrained(cls, name_or_path: str, *, strict: bool = True) -> Self:
        r"""Load a model from a registered name or a local directory.

        Two modes are supported:

        1. **Registered name** (``"resnet_50"`` / ``"resnet-50"``) —
           looked up in the global registry; the factory is invoked with
           ``pretrained=True``.  The factory result is validated to be an
           instance of ``cls`` (so subclass calls like
           ``ResNet.from_pretrained("vit_base_16")`` raise rather than
           silently returning the wrong family).
        2. **Local directory** containing ``config.json`` plus either
           ``model.safetensors`` (preferred) or ``weights.lucid``.  The
           config is restored via ``cls.config_class.load`` and the
           weights are loaded with :func:`lucid.load`.

        Parameters
        ----------
        name_or_path : str
            Registered model name or filesystem directory path.
        strict : bool, optional, keyword-only, default=True
            Forwarded to :meth:`load_state_dict`.

        Returns
        -------
        Self
            A fully constructed model instance.

        Raises
        ------
        ValueError
            If the name is unrecognised and is not a valid directory.
        FileNotFoundError
            If a directory was supplied but required files are absent.
        TypeError
            If the registered factory yields a non-subclass of ``cls``,
            or ``config_class`` is unset, or the weights file lacks a
            state-dict.

        Notes
        -----
        For task-aware dispatch that resolves to a different concrete
        subclass per task, use the ``AutoModelFor*`` family instead.

        Examples
        --------
        >>> # Registered name
        >>> model = ResNetForImageClassification.from_pretrained("resnet_50")
        >>>
        >>> # Local directory
        >>> model.save_pretrained("/tmp/my_resnet50")
        >>> reloaded = ResNetForImageClassification.from_pretrained("/tmp/my_resnet50")
        """
        from lucid.models._registry import is_model, model_entrypoint

        if is_model(name_or_path):
            factory = model_entrypoint(name_or_path)
            model = factory(pretrained=True)
            if not isinstance(model, cls):
                raise TypeError(
                    f"Factory {name_or_path!r} produced "
                    f"{type(model).__name__}, not a subclass of {cls.__name__}"
                )
            return model

        path = Path(name_or_path)
        if not path.is_dir():
            raise ValueError(
                f"{name_or_path!r} is neither a registered model name nor an existing directory"
            )
        config_file = path / "config.json"
        weights_st = path / "model.safetensors"
        weights_lucid = path / "weights.lucid"
        if not config_file.exists():
            raise FileNotFoundError(f"config.json not found in {path}")
        # Prefer SafeTensors when present; fall back to .lucid format.
        if weights_st.exists():
            weights_file = weights_st
        elif weights_lucid.exists():
            weights_file = weights_lucid
        else:
            raise FileNotFoundError(
                f"No weights file found in {path}. Expected 'model.safetensors' or 'weights.lucid'."
            )

        if cls.config_class is None:
            raise TypeError(f"{cls.__name__} must set config_class before calling from_pretrained")
        config = cls.config_class.load(str(config_file))
        model = cls(config)
        sd = _lucid.load(str(weights_file), weights_only=True)
        if not isinstance(sd, dict):
            raise TypeError(f"Weights file did not contain a state_dict, got {type(sd).__name__}")
        model.load_state_dict(sd, strict=strict)
        return model

    def save_pretrained(
        self,
        path: str,
        *,
        safe_serialization: bool = False,
    ) -> None:
        r"""Write ``config.json`` and weights to ``path``.

        Parameters
        ----------
        path : str
            Destination directory.  Created if it does not exist.
        safe_serialization : bool, optional, keyword-only, default=False
            If ``True``, save weights as ``model.safetensors`` (requires
            ``pip install safetensors``).  If ``False``, save as
            ``weights.lucid`` using the native pickle-based format.

        Notes
        -----
        Output layout::

            path/
              config.json
              model.safetensors   # when safe_serialization=True
              weights.lucid       # when safe_serialization=False

        The companion :meth:`from_pretrained` (or any ``AutoModelFor*``
        class) reads this directory layout and prefers SafeTensors when
        both files are present.

        Examples
        --------
        >>> model = create_model("resnet_50")
        >>> model.save_pretrained("/tmp/my_resnet50", safe_serialization=True)
        """
        os.makedirs(path, exist_ok=True)
        self.config.save(os.path.join(path, "config.json"))
        if safe_serialization:
            _lucid.save_safetensors(
                self.state_dict(),
                os.path.join(path, "model.safetensors"),
                metadata={"model_type": self.config.model_type},
            )
        else:
            _lucid.save(self.state_dict(), os.path.join(path, "weights.lucid"))

    def num_parameters(self, *, only_trainable: bool = False) -> int:
        r"""Return the total number of elements across all parameters.

        Parameters
        ----------
        only_trainable : bool, optional, keyword-only, default=False
            When ``True``, parameters with ``requires_grad=False`` are
            excluded (useful for reporting trainable model size after
            freezing the backbone).

        Returns
        -------
        int
            Sum of ``prod(p.shape)`` over the selected parameters.

        Examples
        --------
        >>> model = create_model("resnet_50")
        >>> model.num_parameters()
        25557032
        >>> for p in model.backbone.parameters():
        ...     p.requires_grad = False
        >>> model.num_parameters(only_trainable=True) < 25557032
        True
        """
        total: int = 0
        for p in self.parameters():
            if only_trainable and not p.requires_grad:
                continue
            sz: int = 1
            for s in p.shape:
                sz *= int(s)
            total += sz
        return total

    def _embedding_trunk(self) -> PretrainedModel | None:
        r"""Return the wrapped trunk that owns the token table, if any.

        A task wrapper keeps its trunk under :attr:`base_model_prefix`
        (``bert`` on ``BERTForMaskedLM``, ``transformer`` on
        ``GPTLMHeadModel``).  That attribute is the trunk when it is a
        :class:`PretrainedModel` other than ``self``; a trunk itself, and a
        model with no such attribute, has none.
        """
        prefix = type(self).base_model_prefix
        trunk = getattr(self, prefix, None) if prefix else None
        if isinstance(trunk, PretrainedModel) and trunk is not self:
            return trunk
        return None

    def get_input_embeddings(self) -> nn.Module | None:
        r"""Return the input-embedding submodule, or ``None`` if not applicable.

        Returns
        -------
        nn.Module or None
            The embedding layer for text / token-id models (BERT, GPT,
            …); ``None`` for vision and other non-embedding models.

        Notes
        -----
        Trunks with an embedding table override this.  A task wrapper does
        not need to: this implementation hands the call to the trunk stored
        under :attr:`base_model_prefix`, so a wrapper reports the same table
        as the trunk inside it.  Used by tools that need to resize / share /
        introspect token embeddings without coupling to family-specific
        attribute names.

        Examples
        --------
        >>> from lucid.models import create_model
        >>> lm = create_model("gpt_lm", vocab_size=100, hidden_size=16,
        ...                   num_hidden_layers=1, num_attention_heads=2,
        ...                   intermediate_size=32, max_position_embeddings=8)
        >>> lm.get_input_embeddings() is lm.transformer.tokens_embed
        True
        """
        trunk = self._embedding_trunk()
        return trunk.get_input_embeddings() if trunk is not None else None

    def set_input_embeddings(self, value: nn.Module) -> None:
        r"""Replace the input-embedding submodule.

        Parameters
        ----------
        value : nn.Module
            New embedding module.

        Raises
        ------
        NotImplementedError
            When the model has no embedding table to replace — neither an
            override of its own nor a trunk under :attr:`base_model_prefix`.

        Notes
        -----
        Companion to :meth:`get_input_embeddings`.  Trunks with a table
        override both; a task wrapper inherits this implementation, which
        makes the swap on its trunk.

        When the wrapper ties an output head to the table
        (``config.tie_word_embeddings``), the head is re-bound to the new
        table's weight, so the two stay one matrix and the head scores the
        new vocabulary.  A head bias sized to the vocabulary is resized with
        it: existing entries are kept and new ones start at zero.  An untied
        head is left as it is.  Either way ``config`` is frozen and keeps
        its original ``vocab_size``.

        Examples
        --------
        A text trunk hands its table over, so a larger one can be swapped
        in — here to make room for twenty token ids the original 100-row
        table could not look up.

        >>> import lucid
        >>> import lucid.nn as nn
        >>> from lucid.models import create_model
        >>> model = create_model("gpt", vocab_size=100, hidden_size=16,
        ...                      num_hidden_layers=1, num_attention_heads=2,
        ...                      intermediate_size=32,
        ...                      max_position_embeddings=8).eval()
        >>> bigger = nn.Embedding(120, 16)
        >>> model.set_input_embeddings(bigger)
        >>> model.get_input_embeddings() is bigger
        True
        >>> ids = lucid.tensor([[110, 3]], dtype=lucid.int64)
        >>> model(ids).last_hidden_state.shape
        (1, 2, 16)

        A task wrapper makes the same swap on its trunk.  Its LM head is tied
        to the table, so the head follows and the logits widen to the new
        vocabulary, rather than keep scoring the table it replaced.

        >>> lm = create_model("gpt_lm", vocab_size=100, hidden_size=16,
        ...                   num_hidden_layers=1, num_attention_heads=2,
        ...                   intermediate_size=32,
        ...                   max_position_embeddings=8).eval()
        >>> lm.set_input_embeddings(nn.Embedding(120, 16))
        >>> lm.lm_head.weight is lm.get_input_embeddings().weight
        True
        >>> lm(ids).logits.shape
        (1, 2, 120)

        A model with no embedding table keeps this base implementation and
        refuses, rather than attach a module that nothing would read.

        >>> lenet = create_model("lenet_5")
        >>> lenet.get_input_embeddings() is None
        True
        >>> lenet.set_input_embeddings(nn.Embedding(10, 4))
        Traceback (most recent call last):
            ...
        NotImplementedError: LeNet does not support set_input_embeddings
        """
        trunk = self._embedding_trunk()
        if trunk is None or trunk.get_input_embeddings() is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support set_input_embeddings"
            )
        trunk.set_input_embeddings(value)
        if getattr(self.config, "tie_word_embeddings", False):
            self._tie_word_embeddings()

    def _tie_word_embeddings(self) -> None:
        r"""Bind an output head's weight to the input embedding table.

        Nothing to bind here.  A wrapper whose head shares the input table
        overrides this and calls it from ``__init__`` when
        ``config.tie_word_embeddings`` is set; :meth:`set_input_embeddings`
        calls it again after a swap, so the head follows the new table
        rather than keep the old one alive.
        """
