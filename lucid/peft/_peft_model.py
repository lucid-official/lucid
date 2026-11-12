"""``PeftModel``: an adapted model that saves and loads as the adapter alone."""

import dataclasses
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from types import TracebackType
from typing import ClassVar, Final, Literal, Self, override

import lucid
import lucid.nn as nn
from lucid.nn._incompatible_keys import _IncompatibleKeys
from lucid._tensor.tensor import Tensor
from lucid.peft._adapter_file import _AdapterFile
from lucid.peft._config import (
    PeftConfig,
    TaskType,
    _adapter_directory,
    _output_directory,
    _require_config,
)
from lucid.peft._convert import AdapterFile, ConversionContext, ConvertMode, convert_adapter
from lucid.peft._forwarding import _as_module, _ForwardsAttributes
from lucid.peft._model_prep import _enable_input_require_grads, _is_quantized_model
from lucid.peft._refusals import (
    NO_HUB,
    NO_HYBRID_CACHE,
    NO_MISMATCH_SKIPPING,
    NO_MODEL_CARD,
    refuse_build_options,
    refuse_kwargs,
    refuse_method,
    refuse_value,
)
from lucid.peft._status import TunerLayerStatus, TunerModelStatus, _parameter_counts
from lucid.peft._state_dict import get_peft_model_state_dict, set_peft_model_state_dict
from lucid.peft._targeting import _config_for_model, _with_heads
from lucid.peft.tuners._host import _AdapterHost, _tuner_for
from lucid.peft.tuners._prompt._tuner import _PromptLearningTuner
from lucid.utils.cache import Cache
from lucid.serialization import load_safetensors, save_safetensors

_WEIGHTS_NAME: Final = "adapter_model.safetensors"
_PICKLE_WEIGHTS_NAME: Final = "adapter_model.bin"


class _AdaptedModel(_ForwardsAttributes, ABC):
    """A module that wraps the host holding a model's adapters: what ``PeftModel`` and ``PeftMixedModel`` share.

    The flows both wrappers run (reading an adapter from a directory, filling a model from
    one, ``disable_adapter()``) are written against this: the host as ``base_model``, how far
    ``disable_adapter()`` is nested, and the four members each wrapper implements in its own
    way.  The wrappers stay separate classes, as the reference's are.
    """

    base_model: _AdapterHost
    _disable_depth: int

    @override
    def _wrapped(self) -> nn.Module | None:
        modules = self.__dict__.get("_modules")
        return _as_module(modules.get("base_model")) if modules else None

    @property
    @abstractmethod
    def peft_config(self) -> dict[str, PeftConfig]:
        """Each adapter's config by adapter name."""

    @abstractmethod
    def add_adapter(
        self,
        adapter_name: str,
        peft_config: PeftConfig,
        low_cpu_mem_usage: bool = False,
        autocast_adapter_dtype: bool = True,
    ) -> None:
        """Add an adapter beside the ones held."""

    @abstractmethod
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove an adapter."""

    @abstractmethod
    def unload(self) -> nn.Module:
        """Remove every adapter without merging and return the original model."""

    @abstractmethod
    def _host_config(self, peft_config: PeftConfig) -> PeftConfig:
        """``peft_config`` as the host takes a new adapter: checked and fitted to the model."""


class _DisabledAdapter(AbstractContextManager[None]):
    """What ``disable_adapter()`` returns: reentrant, the outermost exit re-enables.

    Nested blocks — a helper that disables inside a caller's disabled
    block — leave the adapter off until the outermost one exits.
    """

    __slots__ = ("_model",)

    def __init__(self, model: _AdaptedModel) -> None:
        self._model = model

    @override
    def __enter__(self) -> None:
        model = self._model
        if model._disable_depth == 0:
            model.base_model.disable_adapter_layers()
        model._disable_depth += 1

    @override
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb  # an exception passes through; the adapter comes back regardless
        model = self._model
        model._disable_depth -= 1
        if model._disable_depth == 0:
            model.base_model.enable_adapter_layers()


def _adapter_weights(directory: str, device: str | None = None) -> dict[str, Tensor]:
    path = os.path.join(directory, _WEIGHTS_NAME)
    if not os.path.isfile(path):
        if os.path.isfile(os.path.join(directory, _PICKLE_WEIGHTS_NAME)):
            raise NotImplementedError(
                f"{directory!r} holds {_PICKLE_WEIGHTS_NAME}, a pickle; lucid.peft reads "
                f"{_WEIGHTS_NAME} only — re-save the adapter with safe_serialization=True"
            )
        raise FileNotFoundError(f"no {_WEIGHTS_NAME} in {directory!r}")
    loaded = load_safetensors(path)
    weights = {key: value for key, value in loaded.items() if isinstance(value, Tensor)}
    if device is not None:
        weights = {key: value.to(device) for key, value in weights.items()}
    return weights


def _remapped(
    weights: dict[str, Tensor], key_mapping: Mapping[str, str] | None
) -> dict[str, Tensor]:
    """``weights`` with each key renamed by the first ``key_mapping`` pattern that matches it.

    As the reference does it: the ``base_model.model.`` (or ``base_model.``) prefix is set aside,
    the patterns are tried in order with ``re.sub`` on the rest of the key, and the prefix goes back.
    """
    if not key_mapping:
        return weights
    renamed: dict[str, Tensor] = {}
    for key, value in weights.items():
        prefix = next((p for p in ("base_model.model.", "base_model.") if key.startswith(p)), None)
        if prefix is None:
            raise ValueError(
                f"key_mapping needs adapter keys under 'base_model.'; got {key!r} (an adapter "
                "file of a method that keeps its tensors elsewhere cannot be remapped)"
            )
        rest = key.removeprefix(prefix)
        for pattern, replacement in key_mapping.items():
            changed, count = re.subn(pattern, replacement, rest)
            if count:
                rest = changed
                break
        renamed[prefix + rest] = value
    return renamed


def _names_or_none(value: object) -> Sequence[str] | None:
    """The ``adapter_names`` a call was given, checked: ``None``, or a sequence of ``str``."""
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, str):
        names = [item for item in value if isinstance(item, str)]
        if len(names) == len(value):
            return names
    raise TypeError(f"adapter_names must be a sequence of str, got {value!r}")


def _tensor_or_none(value: object) -> Tensor | None:
    """The ``task_ids`` a call was given, checked: ``None`` or a ``Tensor``."""
    if value is None or isinstance(value, Tensor):
        return value
    raise TypeError(f"task_ids must be a Tensor, got {type(value).__name__}")


def _converted(model: nn.Module, adapter: AdapterFile, convert: ConvertMode) -> AdapterFile:
    """``adapter`` in the layout of ``model``'s zoo family, if ``convert`` asks for it."""
    return convert_adapter(adapter, ConversionContext.of(model, adapter.config), convert)


def _saved_module_names(configs: Iterable[PeftConfig]) -> set[str] | None:
    """The ``modules_to_save`` the configs name, together; ``None`` if there are none."""
    names: set[str] = set()
    for config in configs:
        names.update(getattr(config, "modules_to_save", None) or ())
    return names or None


def _print_parameter_share(trainable: int, total: int) -> None:
    """Print ``trainable`` of ``total`` parameter elements, and their share."""
    share = 100 * trainable / total if total else 0.0
    print(f"trainable params: {trainable:,d} || all params: {total:,d} || trainable%: {share:.4f}")


@dataclasses.dataclass(frozen=True, slots=True)
class _ReadOptions:
    """How a wrapper reads an adapter from its directory, with the options the reference has and this library refuses.

    ``unsupported`` is what the reference's ``**kwargs`` hold beyond the named ones (its hub
    download); the three build options and ``ignore_mismatched_sizes`` are accepted at the
    default only.
    """

    key_mapping: Mapping[str, str] | None = None
    subfolder: str | None = None
    convert: ConvertMode = "auto"
    device: str | None = None
    autocast_adapter_dtype: bool = True
    ephemeral_gpu_offload: bool = False
    low_cpu_mem_usage: bool = False
    ignore_mismatched_sizes: object = False
    unsupported: Mapping[str, object] = dataclasses.field(default_factory=dict)

    def refuse(self, where: str) -> None:
        """Raise ``NotImplementedError`` for an option this library cannot honour; ``where`` names the call."""
        refuse_kwargs(where, self.unsupported, NO_HUB)
        refuse_build_options(
            where,
            autocast_adapter_dtype=self.autocast_adapter_dtype,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            ephemeral_gpu_offload=self.ephemeral_gpu_offload,
        )
        refuse_value(
            where,
            "ignore_mismatched_sizes",
            self.ignore_mismatched_sizes,
            False,
            NO_MISMATCH_SKIPPING,
        )


def _stored_config(directory: str, config: PeftConfig | None, is_trainable: bool) -> PeftConfig:
    """``config`` (the directory's, without one) as the adapter is loaded: for inference unless ``is_trainable``."""
    given = PeftConfig.from_pretrained(directory) if config is None else config
    return dataclasses.replace(_require_config("config", given), inference_mode=not is_trainable)


def _read_converted(
    model: nn.Module, directory: str, config: PeftConfig, is_trainable: bool, options: _ReadOptions
) -> AdapterFile:
    """The adapter ``directory`` holds, for ``config``: renamed by ``key_mapping``, converted to ``model``'s layout."""
    _tuner_for(config)._check_loadable(config, is_trainable)
    read = _remapped(_adapter_weights(directory, options.device), options.key_mapping)
    return _converted(model, AdapterFile(config, read), options.convert)


def _build_filled[W: _AdaptedModel](
    make: Callable[[nn.Module, PeftConfig, str], W],
    model: nn.Module,
    config: PeftConfig,
    weights: Mapping[str, Tensor],
    adapter_name: str,
) -> W:
    """``model`` adapted with ``config`` by ``make`` and holding ``weights``; undone if they do not fit.

    The file's tensors replace whatever the adapter is built with, so it
    is built from ``config._staged()`` — which leaves the base weight
    alone — and the model keeps the file's ``config``.
    """
    wrapper = make(model, config._staged(), adapter_name)
    try:
        set_peft_model_state_dict(wrapper, weights, adapter_name=adapter_name)
    except BaseException:
        # Whatever stopped the load, ``model`` goes back to what it was, earlier adapters included.
        wrapper.base_model._undo_construction()
        raise
    config._unstaged(wrapper.peft_config[adapter_name])
    return wrapper


def _load_adapter(
    wrapper: _AdaptedModel,
    where: str,
    model_id: str | os.PathLike[str],
    adapter_name: str,
    is_trainable: bool,
    options: _ReadOptions,
) -> _IncompatibleKeys:
    """Load the adapter ``model_id`` holds into ``wrapper`` under ``adapter_name``; ``where`` names the call.

    An adapter the wrapper does not hold is added from the directory's config and filled from
    its weights; one it holds has its values replaced.  Nothing changes if the files are
    missing or their tensors do not fit.
    """
    options.refuse(where)
    directory = _adapter_directory(model_id, options.subfolder)
    config = None
    if adapter_name not in wrapper.peft_config:
        config = dataclasses.replace(
            PeftConfig.from_pretrained(directory), inference_mode=not is_trainable
        )
        _tuner_for(config)._check_loadable(config, is_trainable)
    # Read before anything is changed, so a missing file leaves the model alone.
    weights = _remapped(_adapter_weights(directory, options.device), options.key_mapping)
    if options.convert is not False:
        source = config if config is not None else PeftConfig.from_pretrained(directory)
        converted = _converted(
            wrapper.base_model.model, AdapterFile(source, weights), options.convert
        )
        weights = dict(converted.tensors)
        if config is not None:
            config = dataclasses.replace(converted.config, inference_mode=not is_trainable)
    adding: AbstractContextManager[None] = nullcontext()
    if config is not None:
        adding = wrapper.base_model._adding(adapter_name, wrapper._host_config(config._staged()))
    with adding:
        if config is not None and is_trainable:
            wrapper.base_model.set_requires_grad(adapter_name)
        loaded = set_peft_model_state_dict(wrapper, weights, adapter_name=adapter_name)
    if config is not None:
        config._unstaged(wrapper.peft_config[adapter_name])
    if not is_trainable:
        wrapper.eval()
    return loaded


class PeftModel(_AdaptedModel):
    r"""A model with adapters, saved and loaded as the adapters alone.

    Wraps the model in its method's tuner (``LoraModel`` for a
    ``LoraConfig``), kept as ``base_model`` (the user's model is
    ``base_model.model``), so the
    state dict reads ``base_model.model....lora_A.default.weight`` and the
    adapter file ``base_model.model....lora_A.weight``, the reference
    library's layouts.  Calling it runs the wrapped model, and any public
    name it lacks — ``generate``, ``config``, a head — is read from that
    model, so code written for the bare model keeps working.

    The model holds any number of adapters by name.  ``add_adapter`` and
    ``load_adapter`` bring one in beside the others (frozen and inactive),
    ``set_adapter`` switches which one the model computes with — and
    trains — and ``delete_adapter`` drops one.  ``save_pretrained`` writes
    them all, the ``"default"`` one at the directory itself and each other
    in a sub-directory of its own name.

    Usually made by :func:`~lucid.peft.get_peft_model`, or by
    ``from_pretrained`` for a saved adapter.  Both pick the class from the
    config's ``task_type``: the wrappers (``PeftModelForCausalLM`` and the
    others, ``MODEL_TYPE_TO_PEFT_MODEL_MAPPING``) are subclasses of this
    one, each registered for its task by ``class Wrapper(PeftModel,
    task_type=TaskType.X)``; an adapter with no task, or a task no wrapper
    is registered for, gets this class itself.

    Parameters
    ----------
    model : nn.Module
        The model to adapt, changed in place.
    peft_config : PeftConfig
        What to adapt and how — a ``LoraConfig``.
    adapter_name : str, default="default"
        Name of the adapter.
    autocast_adapter_dtype : bool, default=True
        Cast a half-precision base's adapter weights to float32.  Always the case here;
        ``False`` raises ``NotImplementedError``.
    low_cpu_mem_usage : bool, default=False
        Build the adapter on the meta device.  Not implemented: ``True`` raises
        ``NotImplementedError``.

    Raises
    ------
    TypeError
        ``peft_config`` is not an adapter config.
    NotImplementedError
        ``autocast_adapter_dtype`` or ``low_cpu_mem_usage`` holds a value the library cannot
        honour.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoraConfig, PeftModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> peft = PeftModel(net, LoraConfig(r=2, target_modules=["0"]))
    >>> peft.get_nb_trainable_parameters()
    (32, 122)
    >>> peft(lucid.randn(4, 8)).shape
    (4, 2)
    """

    #: Wrapper class by the task it is for, filled as each wrapper is defined.
    _task_models: ClassVar[dict[TaskType, type[PeftModel]]] = {}
    #: Modules this class's task trains in full when the model has them.
    _head_names: ClassVar[tuple[str, ...]] = ()

    @override
    def __init_subclass__(cls, *, task_type: TaskType | None = None, **kwargs: object) -> None:
        """Register the subclass as the wrapper for ``task_type``, when it names one."""
        super().__init_subclass__(**kwargs)
        if task_type is not None:
            PeftModel._task_models[task_type] = cls

    @classmethod
    def _model_class(cls, task_type: TaskType | str | None) -> type[PeftModel]:
        """The wrapper registered for ``task_type``, or ``cls`` when it has none."""
        if task_type is None:
            return cls
        return PeftModel._task_models.get(TaskType(task_type), cls)

    @classmethod
    def _prepared(cls, model: nn.Module, config: PeftConfig) -> PeftConfig:
        """``config`` as this class stores it for ``model``: the task's heads in ``modules_to_save``."""
        if not cls._head_names or not isinstance(model, nn.Module):
            return config
        return _with_heads(model, config, cls._head_names)

    def __init__(
        self,
        model: nn.Module,
        peft_config: PeftConfig,
        adapter_name: str = "default",
        autocast_adapter_dtype: bool = True,
        low_cpu_mem_usage: bool = False,
    ) -> None:
        """Adapt ``model`` in place with ``peft_config`` and wrap it.

        ``adapter_name`` names the first adapter.
        """
        refuse_build_options(
            "PeftModel",
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        config = self._prepared(model, _require_config("peft_config", peft_config))
        super().__init__()
        self.base_model: _AdapterHost = _tuner_for(config)(model, config, adapter_name)
        self._disable_depth = 0

    @property
    @override
    def peft_config(self) -> dict[str, PeftConfig]:
        """Each adapter's config by adapter name (``"all-linear"`` spelled out)."""
        return self.base_model.peft_config

    @property
    def active_adapter(self) -> str | list[str]:
        """Name of the adapter in use (a list once ``delete_adapter`` leaves several or none)."""
        return self.base_model.active_adapter

    @property
    def active_adapters(self) -> list[str]:
        """The adapters in use, as a list."""
        return self.base_model.active_adapters

    def get_base_model(self) -> nn.Module:
        """Return the adapted model itself — the one passed in, now holding adapter layers."""
        return self.base_model.model

    def _sole_adapter(self) -> str:
        """The name of the one adapter in use; ``ValueError`` if there is none or several."""
        names = self.active_adapters
        if len(names) != 1:
            raise ValueError(f"the model has no single adapter in use (active: {names})")
        return names[0]

    @property
    def active_peft_config(self) -> PeftConfig:
        """The config of the adapter in use.

        Raises
        ------
        ValueError
            No adapter is in use, or several are (``active_adapter`` is a list).
        """
        return self.peft_config[self._sole_adapter()]

    @property
    def has_active_enabled_adapter(self) -> bool:
        """Whether the adapters are enabled — the reference's rule, as it computes it.

        For a prompt-learning adapter: ``True`` unless inside ``disable_adapter()``.  For the other
        methods the reference also answers ``True`` when no adapter is in use, so inside
        ``disable_adapter()`` the answer is ``False`` only while an adapter is in use.
        """
        disabled = self._disable_depth > 0
        if self.active_peft_config.is_prompt_learning:
            return not disabled
        return not disabled or not self.active_adapters

    @property
    def modules_to_save(self) -> set[str] | None:
        """The names of the modules trained in full beside the adapters, over every adapter; ``None`` if there are none."""
        return _saved_module_names(self.peft_config.values())

    @property
    def base_model_dtype(self) -> lucid.dtype | None:
        """The dtype of the base model: its first floating-point parameter's; ``None`` if it has none.

        The reference spells this member with its framework's name; that spelling cannot appear
        in this library's source.
        """
        for parameter in self.get_base_model().parameters():
            if parameter.is_floating_point():
                return parameter.dtype
        return None

    def prepare_model_for_gradient_checkpointing(self, model: nn.Module) -> nn.Module:
        """Make a model's embedding output require a gradient, so a checkpointed stack trains.

        What ``use_reentrant=True`` checkpointing needs when the embeddings are frozen: the segment
        has an input that requires a gradient.  A model with ``enable_input_require_grads`` is asked
        to; otherwise a forward hook on its input embeddings (or, with none, a pre-hook on the model
        itself) marks the first floating-point tensor.  A quantized model is left as it is.  Unlike
        the reference, the constructor does not call this: no Lucid model reports
        ``is_gradient_checkpointing``, and the default ``use_reentrant=False`` needs nothing.

        Parameters
        ----------
        model : nn.Module
            The model to prepare.

        Returns
        -------
        nn.Module
            ``model``, prepared.

        Examples
        --------
        >>> import lucid.nn as nn
        >>> from lucid.peft import LoraConfig, get_peft_model
        >>> net = nn.Sequential(nn.Embedding(10, 4), nn.Linear(4, 4))
        >>> peft = get_peft_model(net, LoraConfig(r=2, target_modules=["1"]))
        >>> peft.prepare_model_for_gradient_checkpointing(net) is net
        True
        """
        if not _is_quantized_model(model):
            _enable_input_require_grads(model)
        return model

    def _prompt_host(self, where: str) -> _PromptLearningTuner:
        """The tuner that holds the prompts; ``TypeError`` unless the model's adapters are prompt learning."""
        host = self.base_model
        if not isinstance(host, _PromptLearningTuner):
            raise TypeError(
                f"PeftModel.{where}: only a prompt-learning adapter has a prompt; this model's "
                f"adapters are {type(host).__name__}'s"
            )
        return host

    def get_prompt_embedding_to_save(self, adapter_name: str) -> Tensor:
        """The adapter's prompt as ``save_pretrained`` writes it: ``(rows, width)``, on the CPU.

        Only for a prompt-learning method.  The rows are what the method's encoder makes of its
        virtual tokens — the finished prompt, whether the adapter trains through a projection or
        holds the rows directly.

        Parameters
        ----------
        adapter_name : str
            An adapter the model holds.

        Returns
        -------
        Tensor
            One row per virtual token.

        Raises
        ------
        TypeError
            The model's adapters are not prompt learning.
        ValueError
            The model has no adapter of that name.

        Examples
        --------
        >>> from lucid.models import create_model
        >>> from lucid.peft import PrefixTuningConfig, get_peft_model
        >>> lm = create_model("gpt2_small_lm", vocab_size=32, hidden_size=8, num_hidden_layers=1,
        ...                   num_attention_heads=2, intermediate_size=16, max_position_embeddings=16)
        >>> prefix = PrefixTuningConfig(num_virtual_tokens=2, task_type="CAUSAL_LM")
        >>> get_peft_model(lm, prefix).get_prompt_embedding_to_save("default").shape
        (2, 16)
        """
        return self._prompt_host("get_prompt_embedding_to_save")._prompt_to_save(adapter_name)

    def get_prompt(
        self, batch_size: int, task_ids: Tensor | None = None, max_cache_len: int | None = None
    ) -> Tensor | Cache:
        """The virtual prompt the model is given, for ``batch_size`` rows.

        Only for a prompt-learning method.  A method that goes in front of the input embeddings
        gives the rows, ``(batch_size, rows, token_dim)``; a method that goes in the attention
        (prefix tuning, Cartridge) gives a key/value cache holding the prefix, in the model's dtype,
        that a decoder starts from (an encoder-decoder's carries the decoder's half).

        Parameters
        ----------
        batch_size : int
            The number of rows.
        task_ids : Tensor, optional
            For multitask prompt tuning: the task of each row, ``(batch_size,)``.
        max_cache_len : int, optional
            The reference sizes a fixed-capacity cache with it; this library's cache grows, so
            any value but ``None`` raises ``NotImplementedError``.

        Returns
        -------
        Tensor or Cache
            The prompt rows, or the cache.

        Raises
        ------
        TypeError
            The model's adapters are not prompt learning.
        NotImplementedError
            ``max_cache_len`` is given.

        Examples
        --------
        >>> from lucid.models import create_model
        >>> from lucid.peft import PromptTuningConfig, get_peft_model
        >>> lm = create_model("gpt2_small_lm", vocab_size=32, hidden_size=16, num_hidden_layers=1,
        ...                   num_attention_heads=2, intermediate_size=32, max_position_embeddings=16)
        >>> peft = get_peft_model(lm, PromptTuningConfig(num_virtual_tokens=3, task_type="CAUSAL_LM"))
        >>> peft.get_prompt(batch_size=2).shape
        (2, 3, 16)
        """
        refuse_value("PeftModel.get_prompt", "max_cache_len", max_cache_len, None, NO_HYBRID_CACHE)
        return self._prompt_host("get_prompt")._prompt(
            self._sole_adapter(), batch_size, task_ids, self.base_model_dtype
        )

    def create_or_update_model_card(self, output_dir: str) -> None:
        """Refused: the reference writes a hub model card, which this library has no use for.

        Raises
        ------
        NotImplementedError
            Always.
        """
        del output_dir
        refuse_method("PeftModel.create_or_update_model_card", NO_MODEL_CARD)

    @override
    def add_adapter(
        self,
        adapter_name: str,
        peft_config: PeftConfig,
        low_cpu_mem_usage: bool = False,
        autocast_adapter_dtype: bool = True,
    ) -> None:
        """Add an adapter beside the ones the model holds.

        The new adapter is initialised, not trained, and starts frozen and inactive: the model
        computes what it computed, and every parameter keeps its ``requires_grad``.  Switch to it
        with ``set_adapter``.  Modules an earlier adapter wraps take the new one beside it; a
        classification or question-answering ``task_type`` adds the model's head to
        ``modules_to_save`` as ``get_peft_model`` does.  Merged layers stay merged, and the adapter
        starts from their weights as they were before the merge.

        Parameters
        ----------
        adapter_name : str
            Name of the new adapter; unique among the model's adapters.
        peft_config : PeftConfig
            What to adapt and how.
        low_cpu_mem_usage : bool, default=False
            Build the adapter on the meta device.  Not implemented: ``True``
            raises ``NotImplementedError``.
        autocast_adapter_dtype : bool, default=True
            Cast a half-precision base's adapter weights to float32.  Always
            the case here; ``False`` raises ``NotImplementedError``.

        Raises
        ------
        ValueError
            The name is taken, the config is another method's, no module matches it, two adapters
            would train biases, or a PiSSA / OLoRA / LoRA-GA start would write the base weights of
            merged layers (``base_model.unmerge_adapter()`` first).  The model is left as it was.
        NotImplementedError
            ``low_cpu_mem_usage`` or ``autocast_adapter_dtype`` holds a value
            the library cannot honour.

        Examples
        --------
        >>> import lucid
        >>> import lucid.nn as nn
        >>> from lucid.peft import LoraConfig, get_peft_model
        >>> net = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        >>> peft = get_peft_model(net, LoraConfig(r=2, target_modules=["0"]))
        >>> peft.add_adapter("second", LoraConfig(r=4, target_modules=["0", "1"]))
        >>> peft.active_adapter, sorted(peft.peft_config)
        ('default', ['default', 'second'])
        >>> peft.set_adapter("second")
        >>> peft.get_base_model()[1].lora_A["second"].weight.shape
        (4, 4)
        """
        refuse_build_options(
            "PeftModel.add_adapter",
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        self.base_model.add_adapter(adapter_name, self._host_config(peft_config))

    @override
    def _host_config(self, peft_config: PeftConfig) -> PeftConfig:
        model = self.get_base_model()
        return self._prepared(
            model, _config_for_model(model, _require_config("peft_config", peft_config))
        )

    def set_adapter(self, adapter_name: str, inference_mode: bool = False) -> None:
        """Make ``adapter_name`` the adapter in use.

        Only the one adapter applies.  It trains, and every other adapter is
        frozen, unless ``inference_mode``.  A merged model is unmerged
        first; layers that hold nothing of the adapter run their base alone.

        Parameters
        ----------
        adapter_name : str
            An adapter the model holds.
        inference_mode : bool, default=False
            Freeze the adapter too.

        Raises
        ------
        ValueError
            The model has no adapter of that name.
        """
        self.base_model.set_adapter(adapter_name, inference_mode)

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove an adapter, its config and its ``modules_to_save`` copies from the model.

        If it was the one in use, the first of the others (by name) takes
        over, with a warning.  Deleting the last adapter leaves a model
        that applies nothing; ``unload()`` takes the layers out.

        Raises
        ------
        ValueError
            The model has no adapter of that name, or it is merged.
        """
        self.base_model.delete_adapter(adapter_name)

    def set_requires_grad(
        self, adapter_names: str | Sequence[str], requires_grad: bool = True
    ) -> None:
        """Enable or disable gradients on the named adapter(s), leaving the rest as they are.

        Parameters
        ----------
        adapter_names : str or sequence of str
            The adapter, or adapters, to change.
        requires_grad : bool, default=True
            Whether they train.

        Raises
        ------
        ValueError
            A name is not one of the model's adapters.
        """
        self.base_model.set_requires_grad(adapter_names, requires_grad)

    def _routed(
        self, adapter_names: Sequence[str] | None, num_beams: int = 1
    ) -> AbstractContextManager[None]:
        """The context a call runs in: each batch row through its named adapter, or no change."""
        if adapter_names is None:
            return nullcontext()
        return self.base_model._mixed_batch(adapter_names, num_beams)

    def _tasked(self, task_ids: Tensor | None) -> AbstractContextManager[None]:
        """The context a call runs in: each batch row with the prompt of its task, or no change."""
        return self.base_model._task_scope(task_ids)

    @override
    def forward(self, *args: object, **kwargs: object) -> object:
        r"""Run the adapted model; every other argument goes to it as given.

        Parameters
        ----------
        *args : object
            Positional arguments of the model's ``forward``.
        **kwargs : object
            Keyword arguments of the model's ``forward``, and two of the adapter's own:

            ``adapter_names`` (sequence of str)
                One adapter per row of the batch (the first dimension of the
                inputs): row ``i`` is computed with ``adapter_names[i]`` alone,
                ``"__base__"`` for the base model, all in one pass, so requests
                for different adapters can share a batch.  The adapters in use
                are ignored.  Inference only, for LoRA on ``nn.Linear``,
                ``nn.Embedding`` and convolution layers (and ``modules_to_save``
                modules).
            ``task_ids`` (Tensor)
                For multitask prompt tuning: the task of each row of the batch, ``(batch,)``
                integers in ``[0, num_tasks)``, whose prompt is put in front of that row.

        Returns
        -------
        object
            What the model's ``forward`` returns.

        Raises
        ------
        ValueError
            ``adapter_names`` is given in training mode, names an adapter
            the model lacks, a row count other than the batch's, or the
            model holds merged adapters or a DoRA adapter among those named;
            ``task_ids`` is given to an adapter whose prompt is not made per task.
        NotImplementedError
            The method is not LoRA, or the model has an adapted layer a
            mixed batch cannot route (attention, a raw parameter).

        Examples
        --------
        >>> import lucid
        >>> import lucid.nn as nn
        >>> from lucid.peft import LoraConfig, get_peft_model
        >>> net = nn.Sequential(nn.Linear(4, 4))
        >>> config = LoraConfig(r=2, target_modules=["0"], init_lora_weights=False)
        >>> peft = get_peft_model(net, config, adapter_name="a")
        >>> peft.add_adapter("b", config)
        >>> _ = peft.eval()
        >>> x = lucid.randn(3, 4)
        >>> mixed = peft(x, adapter_names=["a", "b", "__base__"])
        >>> peft.set_adapter("b")
        >>> bool(lucid.allclose(mixed[1], peft(x)[1], atol=1e-6))
        True
        """
        # Through the tuner, whose forward is the model's own unless the method adds to it
        # (AdaLoRA's orthogonality penalty on the loss).
        adapter_names = _names_or_none(kwargs.pop("adapter_names", None))
        task_ids = _tensor_or_none(kwargs.pop("task_ids", None))
        with self._routed(adapter_names), self._tasked(task_ids):
            return self.base_model(*args, **kwargs)

    def generate(self, *args: object, **kwargs: object) -> object:
        r"""Generate with the adapted model: its own ``generate``, called as given.

        The adapter applies at every step, since the model's layers are
        the adapted ones; ``with model.disable_adapter():`` generates with
        the base model alone.

        Parameters
        ----------
        *args : object
            Positional arguments of the model's ``generate``.
        **kwargs : object
            Keyword arguments of the model's ``generate``, and two of the adapter's own:

            ``adapter_names`` (sequence of str)
                One adapter per prompt, as for :meth:`forward`: every step of prompt ``i``
                uses ``adapter_names[i]`` alone (``"__base__"``: the base model).  With beam
                search (``num_beams``) each name covers that prompt's beams; an encoder, if
                the model has one, sees the prompts once.
            ``task_ids`` (Tensor)
                For multitask prompt tuning: the task of each prompt of the batch, as for
                :meth:`forward`.

        Returns
        -------
        object
            What the model's ``generate`` returns.

        Raises
        ------
        AttributeError
            The adapted model has no ``generate``.
        """
        adapter_names = _names_or_none(kwargs.pop("adapter_names", None))
        task_ids = _tensor_or_none(kwargs.pop("task_ids", None))
        model = self.get_base_model()
        generate = getattr(model, "generate", None)
        if generate is None:
            raise AttributeError(f"{type(model).__name__} has no generate() to call")
        beams = kwargs.get("num_beams")
        routed = self._routed(adapter_names, beams if isinstance(beams, int) and beams > 1 else 1)
        with routed, self._tasked(task_ids):
            return generate(*args, **kwargs)

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        """Return ``(trainable, total)`` parameter counts, each shared parameter once.

        Returns
        -------
        tuple of int
            Elements in parameters that require grad, and in all of them.
        """
        return _parameter_counts(self)

    def get_layer_status(self) -> list[TunerLayerStatus]:
        """The state of each adapter layer: see :func:`~lucid.peft.get_layer_status`."""
        return get_layer_status(self)

    def get_model_status(self) -> TunerModelStatus:
        """The state of the model's adapters as a whole: see :func:`~lucid.peft.get_model_status`."""
        return get_model_status(self)

    def print_trainable_parameters(self) -> None:
        """Print the trainable / total parameter counts and their ratio.

        Examples
        --------
        >>> import lucid.nn as nn
        >>> from lucid.peft import LoraConfig, get_peft_model
        >>> peft = get_peft_model(nn.Sequential(nn.Linear(8, 8)),
        ...                       LoraConfig(r=2, target_modules=["0"]))
        >>> peft.print_trainable_parameters()
        trainable params: 32 || all params: 104 || trainable%: 30.7692
        """
        _print_parameter_share(*self.get_nb_trainable_parameters())

    def disable_adapter(self) -> AbstractContextManager[None]:
        """Run the base model alone inside the ``with`` block.

        LoRA updates are skipped (a merged layer is unmerged on entry, and
        stays unmerged after) and every ``modules_to_save`` module runs as
        the original; the adapter comes back when the block exits, even on
        an exception.  Blocks nest: the adapter stays off until the
        outermost one exits.  Merging inside the block is refused.  A
        compiled model follows the switch: the state it branches on is
        part of its cache key.

        Examples
        --------
        >>> import lucid
        >>> import lucid.nn as nn
        >>> from lucid.peft import LoraConfig, get_peft_model
        >>> net = nn.Sequential(nn.Linear(4, 4))
        >>> x = lucid.randn(2, 4)
        >>> base_out = net(x)
        >>> peft = get_peft_model(net, LoraConfig(r=2, target_modules=["0"],
        ...                                       init_lora_weights=False))
        >>> with peft.disable_adapter():
        ...     bool((peft(x) == base_out).all().item())
        True
        """
        return _DisabledAdapter(self)

    def merge_and_unload(
        self,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: list[str] | None = None,
    ) -> nn.Module:
        r"""Fold the adapter(s) into the base model and hand the base model back.

        For deployment: the result runs at the base model's cost and saves,
        exports and compiles as an ordinary model.  This wrapper is spent
        afterwards, and adapters that were not folded in are gone with it.

        Parameters
        ----------
        progressbar : bool, default=False
            Kept for call compatibility with the reference; ignored.
        safe_merge : bool, default=False
            Refuse, before changing any weight, if a merged weight would
            hold NaN or inf.
        adapter_names : list of str, optional
            Which adapters to fold in; the ones in use by default.

        Returns
        -------
        nn.Module
            ``get_base_model()``, with plain ``nn.Linear`` layers again.

        Raises
        ------
        RuntimeError
            A base weight is shared with another module (a tied head),
            or a ``modules_to_save`` module was trained apart from a tie.

        Examples
        --------
        >>> import lucid
        >>> import lucid.nn as nn
        >>> from lucid.peft import LoraConfig, get_peft_model
        >>> peft = get_peft_model(nn.Sequential(nn.Linear(4, 2)),
        ...                       LoraConfig(r=2, target_modules=["0"], init_lora_weights=False))
        >>> x = lucid.randn(3, 4)
        >>> adapted = peft(x)
        >>> plain = peft.merge_and_unload()
        >>> type(plain[0]) is nn.Linear, bool(lucid.allclose(plain(x), adapted, atol=1e-5))
        (True, True)
        """
        return self.base_model.merge_and_unload(progressbar, safe_merge, adapter_names)

    @override
    def unload(self) -> nn.Module:
        """Remove the adapter without merging and return the original model.

        Returns
        -------
        nn.Module
            The model, each adapter layer replaced by the module it wrapped.
        """
        return self.base_model.unload()

    def _config_to_save(self, adapter_name: str) -> PeftConfig:
        """The adapter's config as its file holds it: for inference, and naming the base model's class.

        The class is what ``AutoPeftModel*`` rebuilds the base with, exactly, where the
        zoo's Auto classes would pick the first model of a ``model_type`` and task.
        """
        model_class = type(self.get_base_model())
        mapping = {
            "base_model_class": model_class.__name__,
            "parent_library": model_class.__module__,
        }
        return dataclasses.replace(
            self.peft_config[adapter_name], inference_mode=True, auto_mapping=mapping
        )

    def save_pretrained(
        self,
        save_directory: str | os.PathLike[str],
        safe_serialization: bool = True,
        selected_adapters: list[str] | None = None,
        save_embedding_layers: bool | Literal["auto"] = "auto",
        is_main_process: bool = True,
        path_initial_model_for_weight_conversion: str | os.PathLike[str] | None = None,
        **kwargs: object,
    ) -> None:
        r"""Write the adapters — not the base model — under ``save_directory``.

        Per adapter, the reference library's two files:
        ``adapter_config.json`` (with ``inference_mode`` set, as the
        reference stores it, and the base model's class as ``auto_mapping``)
        and ``adapter_model.safetensors``, keyed by
        :func:`~lucid.peft.get_peft_model_state_dict`.  The adapter named
        ``"default"`` goes into ``save_directory`` itself, every other into
        a sub-directory of its own name, as the reference lays it out; an
        adapter is read back from the directory that holds its two files
        (``PeftModel.from_pretrained`` / ``load_adapter``).

        Parameters
        ----------
        save_directory : str or os.PathLike
            Directory to write into; created if missing.
        safe_serialization : bool, default=True
            Must be ``True``: the pickle format the reference writes for
            ``False`` is not one Lucid can produce.
        selected_adapters : list of str, optional
            The adapters to write; every adapter of the model by default.
        save_embedding_layers : bool or "auto", default="auto"
            The reference saves the input and output embeddings beside the adapter when the
            vocabulary was resized.  Here an adapter file holds the adapter alone (and its
            ``modules_to_save``), so ``"auto"`` and ``False`` write that; ``True`` raises
            ``NotImplementedError``.
        is_main_process : bool, default=True
            ``False`` writes nothing, for the processes of a distributed run that are not the
            main one.
        path_initial_model_for_weight_conversion : str or os.PathLike, optional
            For an adapter initialised with ``"pissa"``, ``"pissa_niter_<k>"``
            or ``"olora"``, which rewrote the base weight: the directory
            where ``save_pretrained`` wrote that adapter right after
            initialisation, before any training.  The adapter is written
            converted to a plain LoRA of twice the rank
            (:math:`\Delta W = s(BA - B_0A_0)`, ``init_lora_weights=True``),
            which loads onto the original, unmodified base model and sits
            beside other adapters.  It converts one adapter, so pass
            ``selected_adapters`` if the model holds several; the directory
            is only read.
        **kwargs : object
            The reference forwards these to its hub upload.  Any key raises
            ``NotImplementedError``.

        Raises
        ------
        NotImplementedError
            ``safe_serialization=False``, ``save_embedding_layers=True``, or a key in ``kwargs``.
        ValueError
            ``selected_adapters`` names an adapter the model lacks, or
            ``path_initial_model_for_weight_conversion`` is given with
            several adapters to write, for a method other than LoRA, for a
            DoRA adapter, for ``use_rslora`` with a rank or alpha pattern,
            or the directory's adapter does not match the trained one.
        FileNotFoundError
            ``path_initial_model_for_weight_conversion`` holds no adapter.
        RuntimeError
            The adapters were merged or unloaded already — there is nothing
            left to save; nothing is written.
        NotADirectoryError
            ``save_directory`` names an existing file.

        Examples
        --------
        >>> import tempfile, os
        >>> import lucid.nn as nn
        >>> from lucid.peft import LoraConfig, get_peft_model
        >>> peft = get_peft_model(nn.Sequential(nn.Linear(4, 2)),
        ...                       LoraConfig(r=2, target_modules=["0"]))
        >>> peft.add_adapter("style", LoraConfig(r=2, target_modules=["0"]))
        >>> with tempfile.TemporaryDirectory() as d:
        ...     peft.save_pretrained(d)
        ...     sorted(os.listdir(d)), sorted(os.listdir(os.path.join(d, "style")))
        (['adapter_config.json', 'adapter_model.safetensors', 'style'], ['adapter_config.json', 'adapter_model.safetensors'])
        """
        if not safe_serialization:
            raise NotImplementedError(
                "safe_serialization=False writes a pickle; lucid.peft writes safetensors only"
            )
        if save_embedding_layers is True:
            raise NotImplementedError(
                "PeftModel.save_pretrained: `save_embedding_layers=True` is not supported by "
                "lucid.peft: an adapter file holds the adapter and its modules_to_save only"
            )
        refuse_kwargs("PeftModel.save_pretrained", kwargs, "it would configure a hub upload")
        names = list(self.peft_config) if selected_adapters is None else list(selected_adapters)
        unknown = [name for name in names if name not in self.peft_config]
        if unknown:
            raise ValueError(
                f"You passed an invalid `selected_adapters` arguments, current supported adapter "
                f"names are {list(self.peft_config)} - got {names}."
            )
        # Every adapter is read, and converted, before the first byte is written.
        files = {
            name: _AdapterFile(
                self._config_to_save(name), get_peft_model_state_dict(self, adapter_name=name)
            )
            for name in names
        }
        if path_initial_model_for_weight_conversion is not None:
            files = self._converted_to_plain(files, path_initial_model_for_weight_conversion)
        if not is_main_process:
            return
        root = _output_directory(save_directory)
        for name, file in files.items():
            path = root if name == "default" else _output_directory(os.path.join(root, name))
            # The reference's writer tags its files with this format string.
            save_safetensors(
                file.state, os.path.join(path, _WEIGHTS_NAME), metadata={"format": "pt"}
            )
            file.config.save_pretrained(path)

    def _converted_to_plain(
        self, files: dict[str, _AdapterFile[PeftConfig]], initial_directory: str | os.PathLike[str]
    ) -> dict[str, _AdapterFile[PeftConfig]]:
        """``files`` (one adapter) as its tuner converts it, given the adapter saved at initialisation."""
        if len(files) != 1:
            raise ValueError(
                "path_initial_model_for_weight_conversion converts one adapter; pass "
                f"selected_adapters to choose one of {list(files)}"
            )
        ((name, trained),) = files.items()
        directory = _adapter_directory(initial_directory)
        initial = _AdapterFile(PeftConfig.from_pretrained(directory), _adapter_weights(directory))
        return {name: _tuner_for(trained.config)._converted_to_plain(trained, initial)}

    @classmethod
    def _filled(
        cls, model: nn.Module, config: PeftConfig, weights: Mapping[str, Tensor], adapter_name: str
    ) -> Self:
        """``model`` adapted with ``config`` and holding ``weights``; undone if they do not fit."""
        return _build_filled(cls, model, config, weights, adapter_name)

    @classmethod
    def from_pretrained(
        cls,
        model: nn.Module,
        model_id: str | os.PathLike[str],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: PeftConfig | None = None,
        autocast_adapter_dtype: bool = True,
        ephemeral_gpu_offload: bool = False,
        low_cpu_mem_usage: bool = False,
        key_mapping: Mapping[str, str] | None = None,
        *,
        subfolder: str | None = None,
        convert: ConvertMode = "auto",
        **kwargs: object,
    ) -> PeftModel:
        r"""Adapt ``model`` with the adapter saved in ``model_id``.

        Reads ``adapter_config.json`` (unless ``config`` is given) and
        ``adapter_model.safetensors`` — written by ``save_pretrained``
        or by the reference library — injects the adapter into ``model`` in
        place and loads its tensors.  The adapter is built without its
        initialisation's effect on the base, so an adapter saved from a PiSSA
        or OLoRA start loads onto the (rewritten) base it was trained beside
        as it was trained, and a pristine base is left as it is.  If the
        tensors do not fit the model, the injection is undone before the error
        is raised.  The class
        returned follows the config's ``task_type`` as in
        :func:`~lucid.peft.get_peft_model`; only an adapter with no task
        is built as ``cls``.

        Parameters
        ----------
        model : nn.Module
            The base model the adapter was trained on.
        model_id : str or os.PathLike
            A local adapter directory.  Hub ids are not resolved.
        adapter_name : str, default="default"
            Name to give the adapter.
        is_trainable : bool, default=False
            Keep the adapter trainable.  By default it is frozen and the
            model is put in ``eval()`` mode, for inference.
        config : PeftConfig, optional
            Use this config instead of the directory's.
        autocast_adapter_dtype : bool, default=True
            Cast a half-precision base's adapter weights to float32.  Always the case here;
            ``False`` raises ``NotImplementedError``.
        ephemeral_gpu_offload : bool, default=False
            Not implemented: ``True`` raises ``NotImplementedError``.
        low_cpu_mem_usage : bool, default=False
            Build the adapter on the meta device.  Not implemented: ``True`` raises
            ``NotImplementedError``.
        key_mapping : dict of str to str, optional
            Renames the adapter file's keys before they are loaded: for each key, with its
            ``base_model.model.`` prefix set aside, the first pattern (a regular expression)
            that matches is replaced by its value, as the reference does.  Applied before
            ``convert``.
        subfolder : str, optional
            The directory under ``model_id`` that holds the adapter.  (The reference takes
            it among ``**kwargs``; it is named here.)
        convert : bool or "auto", default="auto"
            Convert an adapter written for the reference library's model to the
            zoo model's layout (``lucid.peft`` lists what each converter does):
            ``gpt2`` (the ``score`` head), ``roformer``, ``t5`` (renamed
            modules) and ``vit`` (``query`` / ``key`` / ``value`` fused into
            ``qkv``, LoRA only).  ``"auto"`` converts when the model's type has
            a converter and the file is in a layout it changes; ``True`` always
            converts and refuses a type with none; ``False`` reads it as it is.
        **kwargs : object
            The reference forwards these to its hub download (``revision``, ``token``,
            ``cache_dir`` …).  Any key raises ``NotImplementedError``.

        Returns
        -------
        PeftModel
            ``model``, adapted and loaded — a task wrapper if the adapter
            names a task.

        Raises
        ------
        FileNotFoundError
            The directory lacks ``adapter_config.json`` or the weights.
        NotImplementedError
            The weights are a pickle (``adapter_model.bin``), the config
            describes something lucid.peft does not implement, or an option the
            library cannot honour is set (see the parameters).
        ValueError
            The adapter's tensors do not fit ``model``, or ``convert`` cannot apply.

        Examples
        --------
        >>> import tempfile
        >>> import lucid
        >>> import lucid.nn as nn
        >>> from lucid.peft import LoraConfig, PeftModel, get_peft_model
        >>> net = nn.Sequential(nn.Linear(4, 2))
        >>> base_state = {k: v.clone() for k, v in net.state_dict().items()}
        >>> trained = get_peft_model(net, LoraConfig(r=2, target_modules=["0"],
        ...                                         init_lora_weights=False))
        >>> x = lucid.randn(3, 4)
        >>> with tempfile.TemporaryDirectory() as d:
        ...     trained.save_pretrained(d)
        ...     fresh = nn.Sequential(nn.Linear(4, 2))
        ...     _ = fresh.load_state_dict(base_state)
        ...     loaded = PeftModel.from_pretrained(fresh, d)
        >>> bool(lucid.allclose(loaded(x), trained(x)))
        True
        """
        options = _ReadOptions(
            key_mapping=key_mapping,
            subfolder=subfolder,
            convert=convert,
            autocast_adapter_dtype=autocast_adapter_dtype,
            ephemeral_gpu_offload=ephemeral_gpu_offload,
            low_cpu_mem_usage=low_cpu_mem_usage,
            unsupported=kwargs,
        )
        options.refuse("PeftModel.from_pretrained")
        directory = _adapter_directory(model_id, subfolder)
        # Without a config, the file's ``peft_type`` picks the method.
        stored = _stored_config(directory, config, is_trainable)
        converted = _read_converted(model, directory, stored, is_trainable, options)
        config = converted.config
        peft_model = cls._model_class(config.task_type)._filled(
            model, config, converted.tensors, adapter_name
        )
        if not is_trainable:
            peft_model.eval()
        return peft_model

    def load_adapter(
        self,
        model_id: str | os.PathLike[str],
        adapter_name: str,
        is_trainable: bool = False,
        device: str | None = None,
        autocast_adapter_dtype: bool = True,
        ephemeral_gpu_offload: bool = False,
        low_cpu_mem_usage: bool = False,
        key_mapping: Mapping[str, str] | None = None,
        *,
        subfolder: str | None = None,
        convert: ConvertMode = "auto",
        **kwargs: object,
    ) -> _IncompatibleKeys:
        r"""Load a saved adapter into the model, beside the ones it holds.

        An adapter the model does not hold is added from the directory's
        config and filled from its weights; the weights of one the model
        already holds replace the values it has.  The loaded adapter is not
        made the one in use — ``set_adapter`` does that.  Nothing changes
        if the files are missing or their tensors do not fit.

        Parameters
        ----------
        model_id : str or os.PathLike
            The local directory that holds the adapter's ``adapter_config.json``
            and ``adapter_model.safetensors``; for an adapter saved by
            ``save_pretrained`` beside others, that is its sub-directory.
        adapter_name : str
            Name to give the adapter.
        is_trainable : bool, default=False
            Make the adapter trainable.  Otherwise it stays frozen and the
            model is put in ``eval()`` mode, for inference.
        device : str, optional
            The device the adapter file's tensors are read onto before they are copied into
            the model.  The reference names this parameter after its tensor framework; that
            name is not used in this library's source.
        autocast_adapter_dtype : bool, default=True
            Cast a half-precision base's adapter weights to float32.  Always the case here;
            ``False`` raises ``NotImplementedError``.
        ephemeral_gpu_offload : bool, default=False
            Not implemented: ``True`` raises ``NotImplementedError``.
        low_cpu_mem_usage : bool, default=False
            Build the adapter on the meta device.  Not implemented: ``True`` raises
            ``NotImplementedError``.
        key_mapping : dict of str to str, optional
            As for :meth:`from_pretrained`.
        subfolder : str, optional
            The directory under ``model_id`` that holds the adapter.  (The reference takes
            it among ``**kwargs``; it is named here.)
        convert : bool or "auto", default="auto"
            As for :meth:`from_pretrained`: convert an adapter written for the
            reference library's model to the zoo model's layout.
        **kwargs : object
            The reference forwards these to its hub download and its state-dict loader.
            ``ignore_mismatched_sizes`` is accepted at its default ``False``; any other key
            raises ``NotImplementedError``.

        Returns
        -------
        _IncompatibleKeys
            What ``model.load_state_dict(..., strict=False)`` returned, as
            ``set_peft_model_state_dict`` hands it back.

        Raises
        ------
        FileNotFoundError
            The directory lacks ``adapter_config.json`` or the weights.
        NotImplementedError
            The weights are a pickle, the config describes something
            lucid.peft does not implement, or an option the library cannot
            honour is set (see the parameters).
        ValueError
            The adapter's tensors do not fit the model, or its config cannot
            sit beside the adapters held.

        Examples
        --------
        >>> import tempfile
        >>> import lucid.nn as nn
        >>> from lucid.peft import LoraConfig, PeftModel, get_peft_model
        >>> config = LoraConfig(r=2, target_modules=["0"])
        >>> source = get_peft_model(nn.Sequential(nn.Linear(4, 2)), config)
        >>> mine = get_peft_model(nn.Sequential(nn.Linear(4, 2)), config)
        >>> with tempfile.TemporaryDirectory() as d:
        ...     source.save_pretrained(d)
        ...     _ = mine.load_adapter(d, "loaded")
        >>> sorted(mine.peft_config), mine.active_adapter
        (['default', 'loaded'], 'default')
        """
        options = _ReadOptions(
            key_mapping=key_mapping,
            subfolder=subfolder,
            convert=convert,
            device=device,
            autocast_adapter_dtype=autocast_adapter_dtype,
            ephemeral_gpu_offload=ephemeral_gpu_offload,
            low_cpu_mem_usage=low_cpu_mem_usage,
            ignore_mismatched_sizes=kwargs.pop("ignore_mismatched_sizes", False),
            unsupported=kwargs,
        )
        return _load_adapter(
            self, "PeftModel.load_adapter", model_id, adapter_name, is_trainable, options
        )

    def supports_lora_conversion(self, adapter_name: str = "default") -> bool:
        """Whether the adapter ``adapter_name`` on this model can be converted to LoRA."""
        config = self.peft_config.get(adapter_name) or next(iter(self.peft_config.values()), None)
        if config is not None and config.is_prompt_learning:
            return False
        if not hasattr(self.base_model, "supports_lora_conversion"):
            return False
        return bool(self.base_model.supports_lora_conversion(adapter_name))


def get_layer_status(model: nn.Module) -> list[TunerLayerStatus]:
    r"""The state of each adapter layer of ``model``: enabled, active, merged, trainable, and where.

    One :class:`~lucid.peft.TunerLayerStatus` per adapter layer and per
    ``modules_to_save`` module, in module order, named relative to the
    adapting tuner (``"model.0"`` for a ``PeftModel`` around ``net[0]``).

    Parameters
    ----------
    model : nn.Module
        A :class:`~lucid.peft.PeftModel`, or a model adapted in place
        (``inject_adapter_in_model``).

    Returns
    -------
    list of TunerLayerStatus
        The layers' states.

    Raises
    ------
    ValueError
        ``model`` holds no adapter layer.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoraConfig, get_layer_status, get_peft_model
    >>> peft = get_peft_model(nn.Sequential(nn.Linear(4, 4)),
    ...                       LoraConfig(r=2, target_modules=["0"]))
    >>> (layer,) = get_layer_status(peft)
    >>> layer.name, layer.module_type, layer.enabled, layer.active_adapters
    ('model.0', 'lora.Linear', True, ['default'])
    >>> layer.requires_grad, layer.available_adapters, layer.devices
    ({'default': True}, ['default'], {'default': ['cpu']})
    """
    return TunerLayerStatus.all_of(model.base_model if isinstance(model, PeftModel) else model)


def get_model_status(model: nn.Module) -> TunerModelStatus:
    r"""The state of ``model``'s adapters as a whole, folded from its layers.

    A field the layers disagree on reads ``"irregular"`` (see
    :class:`~lucid.peft.TunerModelStatus`): one layer enabled and another
    not, an adapter merged on one layer and not on another that holds it,
    an adapter that trains in some layers and is frozen in others.

    Parameters
    ----------
    model : nn.Module
        A :class:`~lucid.peft.PeftModel`, or a model adapted in place.

    Returns
    -------
    TunerModelStatus
        The model's state.

    Raises
    ------
    ValueError
        ``model`` holds no adapter layer.

    Examples
    --------
    >>> from lucid.peft import LoraConfig, get_model_status, get_peft_model
    >>> import lucid.nn as nn
    >>> net = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    >>> config = LoraConfig(r=2, target_modules=["0", "1"])
    >>> peft = get_peft_model(net, config)
    >>> peft.add_adapter("other", config)
    >>> status = get_model_status(peft)
    >>> status.adapter_model_type, status.peft_types, status.num_adapter_layers
    ('LoraModel', {'default': 'LORA', 'other': 'LORA'}, 2)
    >>> status.active_adapters, status.requires_grad, status.merged_adapters
    (['default'], {'default': True, 'other': False}, [])
    """
    layers = get_layer_status(model)
    if isinstance(model, PeftModel):
        return TunerModelStatus.of(
            layers,
            base_model_type=type(model.get_base_model()).__name__,
            adapter_model_type=type(model.base_model).__name__,
            peft_types={name: str(config.peft_type) for name, config in model.peft_config.items()},
            parameter_counts=model.get_nb_trainable_parameters(),
        )
    return TunerModelStatus.of(
        layers,
        base_model_type="other",
        adapter_model_type="None",
        peft_types={},
        parameter_counts=_parameter_counts(model),
    )
