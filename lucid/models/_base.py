"""Base classes for model configuration and pretrained-model contract."""

import json
import os
import warnings
from abc import ABC
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import ClassVar, Final, Self, override

import lucid as _lucid
import lucid.nn as nn
from lucid.models._checkpoint import Checkpoint, load_registered, replaced_config
from lucid.models._embeddings import (
    VocabSized,
    resized_embedding,
    resized_projection,
    resized_vocab_size,
)
from lucid.models._utils._text import init_text_module_weights

#: Keys ``config.json`` holds beside the config's own fields: ``model_type`` names the family,
#: ``architectures`` (as in the reference's ``config.json``) names the model class the weights
#: beside it belong to.  Neither is a constructor argument.
_ENVELOPE_KEYS: Final = ("model_type", "architectures")


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
            Field-name → value mapping; ``model_type`` (a ClassVar) and
            ``architectures`` (the saved model class, which
            :meth:`PretrainedModel.save_pretrained` records beside the
            fields) are silently stripped — neither is a constructor arg.

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
        for key in _ENVELOPE_KEYS:
            copy.pop(key, None)
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

    #: Where this model came from, as the reference's ``name_or_path``: the registered factory
    #: name the factory built it under, or the directory ``from_pretrained`` read it from.  The
    #: empty default stays on the class, so a model built straight from a config has none in its
    #: own ``__dict__`` and ``lucid.peft`` leaves ``base_model_name_or_path`` unset.
    name_or_path: str = ""

    #: Dotted path from this model to the ``nn.Embedding`` that looks token ids up, for a model
    #: that owns one (``"embeddings.word_embeddings"``).  A task wrapper leaves it empty: its
    #: trunk, found through ``base_model_prefix``, has the table.  Everything about the table
    #: -- reading it, swapping it, resizing it, keeping the config in step -- hangs off this one
    #: declaration, so a family does not repeat it.
    _input_embeddings: ClassVar[str] = ""
    #: Dotted path to the ``nn.Linear`` that scores the vocabulary, for a model that has such a
    #: head (``"lm_head"``, ``"cls.predictions.decoder"``); empty for one that does not.
    _output_embeddings: ClassVar[str] = ""
    #: Name of the config field that is the standard deviation the family draws its token
    #: tables from (``"initializer_range"``); empty for one that keeps the framework default.
    _token_init_std: ClassVar[str] = ""

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
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        strict: bool = True,
        ignore_mismatched_sizes: bool = False,
        **config_overrides: object,
    ) -> Self:
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
           config is restored via ``cls.config_class.from_dict`` and the
           weights are loaded with :func:`lucid.load`.

        Parameters
        ----------
        name_or_path : str
            Registered model name or filesystem directory path.
        strict : bool, optional, keyword-only, default=True
            Whether a key the model has and the checkpoint lacks, or the reverse, is an
            error.  A registered name given no ``config_overrides`` loads its own weights.
        ignore_mismatched_sizes : bool, optional, keyword-only, default=False
            ``True`` lets a weight of another shape keep its fresh initialisation (a
            warning names it) instead of raising: how a checkpoint is loaded into a
            model with another head, ``num_classes=10``.
        **config_overrides : object
            Config fields to change before the model is built.  A registered name is
            built at the overridden config and takes every published weight that fits.

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
            state-dict, or ``config_overrides`` names a field the config lacks.
        RuntimeError
            If the checkpoint does not fit: the message counts each kind of mismatch
            and names the first few weights, and the class the checkpoint was saved from.

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
            model = load_registered(
                model_entrypoint(name_or_path),
                name_or_path,
                config_overrides,
                strict=strict,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
            )
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
        return Checkpoint.open(path).load_into(
            cls, config_overrides, strict=strict, ignore_mismatched_sizes=ignore_mismatched_sizes
        )

    def save_pretrained(self, path: str, *, safe_serialization: bool = False) -> None:
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
        ``config.json`` holds the config's fields, ``model_type`` and
        ``architectures`` — a one-element list naming this model's class, so a
        directory saved from a head that shares its family and task with
        others (``BERTForQuestionAnswering`` among the sequence classifiers)
        loads back as the class it was saved from.

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
        document = self.config.to_dict()
        document["architectures"] = [type(self).__name__]
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(document, f, indent=2, sort_keys=True)
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

    def _embedding_owner(self) -> PretrainedModel | None:
        r"""The model that owns the token table: this one if it declares one, else its trunk's owner."""
        if self._input_embeddings:
            return self
        trunk = self._embedding_trunk()
        return trunk._embedding_owner() if trunk is not None else None

    def get_input_embeddings(self) -> nn.Module | None:
        r"""Return the input-embedding submodule, or ``None`` if not applicable.

        Returns
        -------
        nn.Module or None
            The embedding layer for text / token-id models (BERT, GPT,
            ...); ``None`` for vision and other non-embedding models.

        Notes
        -----
        A trunk with a token table declares where it is in ``_input_embeddings``.
        A task wrapper declares nothing: this implementation hands the call to the
        trunk stored under :attr:`base_model_prefix`, so a wrapper reports the same
        table as the trunk inside it.  Used by tools that need to resize / share /
        introspect token embeddings without coupling to family-specific attribute
        names.

        Examples
        --------
        >>> from lucid.models import create_model
        >>> lm = create_model("gpt_lm", vocab_size=100, hidden_size=16,
        ...                   num_hidden_layers=1, num_attention_heads=2,
        ...                   intermediate_size=32, max_position_embeddings=8)
        >>> lm.get_input_embeddings() is lm.transformer.tokens_embed
        True
        """
        owner = self._embedding_owner()
        return None if owner is None else owner.get_submodule(owner._input_embeddings)

    def get_output_embeddings(self) -> nn.Module | None:
        r"""Return the projection that scores the vocabulary, or ``None`` if there is none.

        Returns
        -------
        nn.Module or None
            The ``Linear`` an LM head decodes with (``lm_head``, BERT's
            ``cls.predictions.decoder``); ``None`` for a trunk, a classifier, and any
            model that does not predict tokens.

        Notes
        -----
        With ``config.tie_word_embeddings`` the head's weight *is* the input table's,
        so the two are one matrix.

        Examples
        --------
        >>> from lucid.models import create_model
        >>> small = dict(vocab_size=60, hidden_size=8, num_attention_heads=1,
        ...              num_hidden_layers=1, intermediate_size=16,
        ...              max_position_embeddings=4)
        >>> lm = create_model("gpt_lm", **small)
        >>> lm.get_output_embeddings() is lm.lm_head
        True
        >>> create_model("gpt", **small).get_output_embeddings() is None
        True
        """
        if not self._output_embeddings:
            return None
        return self.get_submodule(self._output_embeddings)

    def _input_vocab_fields(self, num_tokens: int) -> dict[str, int]:
        r"""The config fields the input table's row count is, when it has ``num_tokens`` rows.

        ``vocab_size`` for nearly every model.  A family whose config sizes one table
        with more than one field overrides this, and is asked on the model that owns
        the table, before it changes.
        """
        return {"vocab_size": num_tokens}

    def _init_token_table(self, table: nn.Embedding | nn.Linear) -> None:
        r"""Initialise a vocabulary-sized table just allocated, as this family initialises its own.

        Called by :meth:`resize_token_embeddings` before the kept rows are copied in, so
        a larger vocabulary starts from the family's distribution rather than the
        framework default: ``N(0, std^2)`` for ``std = config.<_token_init_std>``, a
        padding row zero.  A family with no ``_token_init_std`` keeps the default, and one
        with another rule overrides this.
        """
        if self._token_init_std:
            init_text_module_weights(table, float(getattr(self.config, self._token_init_std)))

    def _output_vocab_follows_input(self) -> bool:
        r"""Whether the output head scores the input table's vocabulary, so one resizes with the other.

        ``False`` for a model whose head decodes into a vocabulary of its own
        (a translation model with separate source and target tables), which a change to
        the input table must leave alone.
        """
        return True

    def _install_input_embeddings(self, table: nn.Embedding) -> None:
        r"""Put ``table`` where the token table is.  The model that owns the table is the one asked."""
        *parents, name = self._input_embeddings.split(".")
        setattr(self.get_submodule(".".join(parents)), name, table)

    def _config_chain(self) -> list[PretrainedModel]:
        r"""This model and the trunks inside it whose config is of the same class as its own."""
        chain = [self]
        trunk = self._embedding_trunk()
        while trunk is not None:
            if type(trunk.config) is type(self.config):
                chain.append(trunk)
            trunk = trunk._embedding_trunk()
        return chain

    def _planned_configs(self, **changes: object) -> list[tuple[PretrainedModel, ModelConfig]]:
        r"""What the config of this model and of each trunk becomes under ``changes``, not yet applied.

        A config is frozen, so a change is a new object, and a task wrapper's trunk
        holds one of its own (the wrapper's, or one derived from it) that would go
        on describing the model as it was.  Each is built with its other fields
        kept, which runs its validation: one that refuses raises before anything
        has changed.
        """
        return [(model, replaced_config(model.config, **changes)) for model in self._config_chain()]

    @staticmethod
    def _install_configs(plan: list[tuple[PretrainedModel, ModelConfig]]) -> None:
        r"""Apply what :meth:`_planned_configs` built."""
        for model, config in plan:
            model.config = config

    def set_input_embeddings(self, value: nn.Module) -> None:
        r"""Replace the input-embedding submodule.

        Parameters
        ----------
        value : nn.Module
            New embedding module.

        Raises
        ------
        NotImplementedError
            When the model has no embedding table: no declaration of its own and
            no trunk under :attr:`base_model_prefix`.
        TypeError
            When ``value`` is not an :class:`~lucid.nn.Embedding`.
        ValueError
            When the config refuses the new size; nothing is changed.

        Notes
        -----
        Companion to :meth:`get_input_embeddings`.  A task wrapper makes the swap
        on its trunk.  A head tied to the table (``config.tie_word_embeddings``) is
        re-bound to the new weight, with any vocabulary-sized bias fitted to it:
        entries kept, new ones zero.  An untied head is left as it is.

        ``config`` is frozen, so it is replaced: its ``vocab_size`` becomes the new
        row count, on this model and on the trunk inside it, and ``save_pretrained``
        then ``from_pretrained`` rebuilds the model at the new size.  To grow a
        trained vocabulary and keep its rows, use :meth:`resize_token_embeddings`.

        Examples
        --------
        A text trunk hands its table over, so a larger one can be swapped
        in -- here to make room for twenty token ids the original 100-row
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
        >>> model.config.vocab_size
        120
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
        >>> lm.config.vocab_size, lm.transformer.config.vocab_size
        (120, 120)
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
        owner = self._embedding_owner()
        if owner is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support set_input_embeddings"
            )
        if not isinstance(value, nn.Embedding):
            raise TypeError(
                f"{type(owner).__name__} input embeddings must be nn.Embedding, "
                f"got {type(value).__name__}"
            )
        # Built first: a config that refuses the size leaves the model as it was.
        plan = self._planned_configs(**owner._input_vocab_fields(value.num_embeddings))
        owner._install_input_embeddings(value)
        if getattr(self.config, "tie_word_embeddings", False):
            self._tie_word_embeddings()
        self._install_configs(plan)

    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        r"""Grow or shrink the token table, keeping the rows it has.

        Parameters
        ----------
        new_num_tokens : int or None, optional
            The new vocabulary size; ``None`` keeps the current one.
        pad_to_multiple_of : int or None, optional
            Round the size up to a multiple of this positive integer.  With
            ``new_num_tokens=None`` it rounds the current size up.
        mean_resizing : bool, optional, default=True
            Draw each new row, and each new entry of a head's own bias, from a normal
            fitted to the old ones: their mean, with the covariance scaled by ``1e-9``
            (just the mean when it is not positive definite), as the reference does.
            ``False`` uses the family's own initialiser.  Draws follow
            ``lucid.manual_seed``.

        Returns
        -------
        nn.Embedding
            The new input table; the old one, unchanged, when the size does not change.

        Raises
        ------
        NotImplementedError
            When the model has no token table.
        TypeError
            When ``new_num_tokens`` is not an ``int`` or ``None``.
        ValueError
            When ``new_num_tokens`` is below 1, ``pad_to_multiple_of`` is not a
            positive integer, or the config refuses the size.  Nothing is changed.

        Notes
        -----
        The leading ``min(old, new)`` rows are copied, so trained tokens keep their
        vectors; the new table has the device, dtype, ``padding_idx`` and
        ``requires_grad`` of the old one.  What follows the table:

        * A head tied to it (``config.tie_word_embeddings``) is re-bound to the new
          weight, with any vocabulary-sized bias fitted to it, new entries zero.
        * An untied head that scores the same vocabulary is resized the same way, its
          new rows and bias entries drawn as the table's are.
        * ``config.vocab_size`` is replaced, on this model and on the trunk inside it,
          so ``save_pretrained`` then ``from_pretrained`` round-trips.

        A model whose source and target vocabularies are separate tables (a translation
        model built with ``share_embeddings=False``) resizes the source table only.

        Examples
        --------
        >>> import lucid
        >>> from lucid.models import create_model
        >>> lm = create_model("gpt_lm", vocab_size=60, hidden_size=8,
        ...                   num_attention_heads=1, num_hidden_layers=1,
        ...                   intermediate_size=16, max_position_embeddings=4).eval()
        >>> before = lm.get_input_embeddings().weight.tolist()
        >>> table = lm.resize_token_embeddings(80)
        >>> table.weight.shape, lm.lm_head.weight is table.weight, lm.config.vocab_size
        ((80, 8), True, 80)
        >>> table.weight[:60].tolist() == before
        True
        >>> lm.resize_token_embeddings(pad_to_multiple_of=64).weight.shape
        (128, 8)
        """
        owner = self._embedding_owner()
        if owner is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support resize_token_embeddings"
            )
        old = self.get_input_embeddings()
        if not isinstance(old, nn.Embedding):
            raise TypeError(
                f"{type(owner).__name__}'s input embeddings are "
                f"{type(old).__name__}, not nn.Embedding, and cannot be resized"
            )
        size = resized_vocab_size(old.num_embeddings, new_num_tokens, pad_to_multiple_of)
        if size == old.num_embeddings:
            return old
        head = self._untied_output_head()
        # Both replacements are built before either is installed.
        init = owner._init_token_table
        table = resized_embedding(old, size, init, mean_resizing=mean_resizing)
        new_head = (
            None
            if head is None
            else resized_projection(head, size, init, mean_resizing=mean_resizing)
        )
        self.set_input_embeddings(table)
        if new_head is not None:
            self._install_output_head(new_head, mean_resizing=mean_resizing)
        return table

    def _untied_output_head(self) -> nn.Linear | None:
        r"""The output head, when it is a table of its own that scores the input vocabulary."""
        if (
            not self._output_embeddings
            or getattr(self.config, "tie_word_embeddings", False)
            or not self._output_vocab_follows_input()
        ):
            return None
        head = self.get_output_embeddings()
        return head if isinstance(head, nn.Linear) else None

    def _install_output_head(self, head: nn.Linear, *, mean_resizing: bool) -> None:
        r"""Put ``head`` where the output projection is, and fit any bias its holder keeps beside it."""
        *parents, name = self._output_embeddings.split(".")
        holder = self.get_submodule(".".join(parents))
        setattr(holder, name, head)
        if isinstance(holder, VocabSized):
            holder.fit_vocab(head.out_features, mean_resizing=mean_resizing)

    def _tie_word_embeddings(self) -> None:
        r"""Bind an output head's weight to the input embedding table.

        Nothing to bind here.  A wrapper whose head shares the input table
        overrides this and calls it from ``__init__`` when
        ``config.tie_word_embeddings`` is set; :meth:`set_input_embeddings`
        calls it again after a swap, so the head follows the new table
        rather than keep the old one alive.
        """
