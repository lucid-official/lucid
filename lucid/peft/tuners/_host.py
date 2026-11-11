"""What every tuner answers to: the host a ``PeftModel`` drives, and the registry of hosts.

A method is three classes, each registered as it is defined:

* its config (``PeftConfig`` subclass, ``peft_type=``) — in ``lucid.peft._config``;
* its host (``_AdapterHost`` subclass, ``peft_type=``), which holds the
  method's adapters — a ``_BaseTuner`` puts layers into the model (the
  generic half is in ``lucid.peft.tuners._tuner``), a prompt-learning host
  keeps a learned input beside it;
* its layers (``_BaseTunerLayer`` subclasses, ``base_type=``), one per
  kind of module it can wrap, found by the wrapped module's type
  (``lucid.peft.tuners._layer``).

``get_peft_model``, ``PeftModel.from_pretrained`` and the state-dict
functions dispatch through ``_TUNERS``, so a new method is subclasses only.

A method can have its config and no adapter: ``_RefusedTuner`` stands in for the tuner of a
method this library will not build adapters for (``_REFUSED_TUNERS``).  It is not in ``_TUNERS``,
so what loops over the working tuners never meets it; building one raises
``NotImplementedError`` saying why.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import ClassVar, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._adapter_file import _AdapterFile
from lucid.peft._config import _CONFIG_CLASSES, PeftConfig, PeftType
from lucid.peft._forwarding import _as_module, _ForwardsAttributes
from lucid.peft._refusals import NO_META_DEVICE, NO_STATE_DICT_MATCHING, refuse_value
from lucid.peft.tuners._layer import _check_adapter_name

#: ``PeftType`` → the tuner that applies it, filled as each is defined.
_TUNERS: dict[PeftType, type[_AdapterHost]] = {}
#: ``PeftType`` → the stand-in of a method this library will not build adapters for.
_REFUSED_TUNERS: dict[PeftType, type[_RefusedTuner]] = {}


def _registered[T](registry: Mapping[PeftType, T], config: PeftConfig) -> T | None:
    """The entry of ``registry`` for ``config``'s method, if it has one."""
    for peft_type, entry in registry.items():
        if config.peft_type == peft_type:
            return entry
    return None


def _tuner_for(config: PeftConfig) -> type[_AdapterHost]:
    """The tuner registered for ``config``'s method.

    Raises
    ------
    NotImplementedError
        The method has no tuner: it is refused, or is unknown.
    """
    tuner = _registered(_TUNERS, config)
    if tuner is not None:
        return tuner
    refused = _registered(_REFUSED_TUNERS, config)
    if refused is not None:
        raise NotImplementedError(refused._refusal())
    raise NotImplementedError(f"lucid.peft has no tuner for {config.peft_type} adapters")


class _AdapterHost(_ForwardsAttributes, ABC):
    """What ``PeftModel`` asks of the object that holds a model's adapters.

    A method registers its host by ``peft_type`` as it is defined.  Most
    methods put layers into the model (``_BaseTuner``); a method that
    learns an input instead (prompt, prefix and p-tuning) holds its
    adapters beside the model, and answers the same questions.
    """

    #: The model the adapters are for, and its adapters' configs by name.
    model: nn.Module
    peft_config: dict[str, PeftConfig]
    active_adapter: str | list[str]
    #: Where the user's model sits inside this host, in the host's state-dict keys.
    _model_path: ClassVar[str] = "model."
    #: The config class this method takes.
    _config_class: ClassVar[type[PeftConfig]]

    def __init__(
        self,
        model: nn.Module,
        peft_config: PeftConfig,
        adapter_name: str,
        low_cpu_mem_usage: bool = False,
        state_dict: dict[str, Tensor] | None = None,
    ) -> None:
        """Make the host; a method's host builds its adapter from ``model`` and ``peft_config`` after this.

        ``low_cpu_mem_usage`` and ``state_dict`` are the reference's options for building an
        adapter on the meta device and for matching targets against a state dict; neither
        exists here, so anything but their defaults is refused.
        """
        del model, peft_config, adapter_name  # the subclass keeps what it needs
        where = type(self).__name__
        refuse_value(where, "low_cpu_mem_usage", low_cpu_mem_usage, False, NO_META_DEVICE)
        refuse_value(where, "state_dict", state_dict, None, NO_STATE_DICT_MATCHING)
        super().__init__()

    @override
    def __init_subclass__(cls, *, peft_type: PeftType | None = None, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if peft_type is not None:
            _TUNERS[peft_type] = cls

    @property
    @abstractmethod
    def active_adapters(self) -> list[str]:
        """The adapters the forward pass applies, as a list."""

    @override
    def _wrapped(self) -> nn.Module | None:
        modules = self.__dict__.get("_modules")
        return _as_module(modules.get("model")) if modules else None

    def _task_scope(self, task_ids: Tensor | None) -> AbstractContextManager[None]:
        """The context a call runs in when it names a task per batch row.

        An adapter whose prompt is not per task has nothing to pick, so it takes none.

        Raises
        ------
        ValueError
            ``task_ids`` is given.
        """
        if task_ids is not None:
            raise ValueError("task_ids given to an adapter whose prompt is not per-task")
        return nullcontext()

    def _admit(self, adapter_name: str, peft_config: PeftConfig) -> str:
        """``adapter_name`` once it and ``peft_config`` are fit to be added beside the adapters held.

        Raises
        ------
        ValueError
            The name is malformed or taken, or the config is another method's.
        TypeError
            The config is not this method's config class.
        """
        adapter_name = _check_adapter_name(adapter_name)
        if adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name '{adapter_name}' already exists.")
        kind = type(self)._config_class
        if peft_config.peft_type != kind._peft_type:
            raise ValueError(
                "Cannot combine adapters with different peft types. "
                f"Found {kind._peft_type} and {peft_config.peft_type}."
            )
        if not isinstance(peft_config, kind):
            raise TypeError(f"config must be a {kind.__name__}, got {type(peft_config).__name__}")
        return adapter_name

    @classmethod
    @abstractmethod
    def _extra_trainable_names(cls, model: nn.Module, config: PeftConfig) -> list[str]:
        """Parameters beyond the adapters' own that train, by name under ``model``."""

    @classmethod
    def _adapt_in_place(
        cls,
        model: nn.Module,
        config: PeftConfig,
        adapter_name: str,
        low_cpu_mem_usage: bool = False,
        state_dict: dict[str, Tensor] | None = None,
    ) -> None:
        """Adapt a bare ``model`` without a wrapper (``inject_adapter_in_model``)."""
        cls(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage, state_dict=state_dict)

    @classmethod
    def _require_adapter_state(cls, model: nn.Module, adapter_name: str) -> None:
        """Raise ``RuntimeError`` if ``model`` no longer holds what ``adapter_name`` saves."""

    @classmethod
    def _check_loadable(cls, config: PeftConfig, is_trainable: bool) -> None:
        """Raise if a saved adapter of this method cannot be loaded as ``is_trainable``."""

    @classmethod
    def _converted_to_plain(
        cls, trained: _AdapterFile[PeftConfig], initial: _AdapterFile[PeftConfig]
    ) -> _AdapterFile[PeftConfig]:
        """``trained`` as an adapter that fits the pristine base, given the adapter as it was at initialisation.

        Only a method whose initialisation rewrites the base weight has anything to
        convert (LoRA's PiSSA / OLoRA); every other method refuses.

        Raises
        ------
        ValueError
            This method has no such conversion.
        """
        del trained, initial  # the default has no conversion
        raise ValueError(
            "Passing `path_initial_model_for_weight_conversion` to `save_pretrained` is not "
            f"supported for {cls.__name__}."
        )

    @abstractmethod
    def add_adapter(self, adapter_name: str, config: PeftConfig) -> None:
        """Add an adapter beside the ones held."""

    @abstractmethod
    def set_adapter(self, adapter_name: str | list[str], inference_mode: bool = False) -> None:
        """Apply the named adapter, and train it unless ``inference_mode``."""

    @abstractmethod
    def set_requires_grad(
        self, adapter_names: str | Sequence[str], requires_grad: bool = True
    ) -> None:
        """Enable or disable gradients on the named adapters."""

    @abstractmethod
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove an adapter."""

    @abstractmethod
    def merge_and_unload(
        self,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: list[str] | None = None,
    ) -> nn.Module:
        """Fold the adapters into the base model and return it."""

    @abstractmethod
    def unload(self) -> nn.Module:
        """Remove every adapter without merging and return the original model."""

    @abstractmethod
    def enable_adapter_layers(self) -> None:
        """Apply the adapter again after ``disable_adapter_layers``."""

    @abstractmethod
    def disable_adapter_layers(self) -> None:
        """Run the base model alone."""

    @abstractmethod
    def _mixed_batch(
        self, adapter_names: Sequence[str], num_beams: int = 1
    ) -> AbstractContextManager[None]:
        """A context in which each batch row goes through the adapter named for it."""


class _WithheldTuner:
    """The stand-in of a method that has a config and an adapter that will not be built.

    Not a host: nothing about a model is changed, held or saved.  It is registered under
    the method's ``PeftType`` so the method has a tuner class to name (``PEFT_TYPE_TO_TUNER_MAPPING``,
    the class exported from ``lucid.peft``) and so building one — through ``get_peft_model``,
    ``PeftModel.from_pretrained`` or the class itself — raises ``NotImplementedError`` with
    the reason, before any model is touched or any file is read.  The signature is the
    reference's, so a script moved over reaches the refusal and not a ``TypeError``.
    """

    #: The method this class stands in for, and the config class that reads its files.
    _peft_type: ClassVar[PeftType]
    _config_class: ClassVar[type[PeftConfig]]

    @classmethod
    def _enrol(cls, peft_type: PeftType) -> None:
        """Pair ``cls`` with ``peft_type`` and the config class registered for it."""
        cls._peft_type = peft_type
        cls._config_class = _CONFIG_CLASSES[peft_type]

    @classmethod
    def _refusal(cls) -> str:
        """The error text: the method, and why no adapter of it is built."""
        raise NotImplementedError

    def __init__(
        self,
        model: nn.Module,
        peft_config: PeftConfig,
        adapter_name: str,
        low_cpu_mem_usage: bool = False,
        state_dict: dict[str, Tensor] | None = None,
    ) -> None:
        """Refuse: no adapter of this method can be built.

        The arguments are the reference's, so a call written for it gets this error; none is
        read.  Each method's tuner class is documented by what it stands in for; this is the
        constructor they share.

        Parameters
        ----------
        model : nn.Module
            The model to adapt; not read.
        peft_config : PeftConfig
            The adapter's config; not read.
        adapter_name : str
            Name of the adapter; not read.
        low_cpu_mem_usage : bool, default=False
            Build the adapter on the meta device; not read.
        state_dict : dict of str to Tensor, optional
            Match targets against these keys; not read.

        Raises
        ------
        NotImplementedError
            Always, with the reason the method is refused.
        """
        del model, peft_config, adapter_name, low_cpu_mem_usage, state_dict
        raise NotImplementedError(self._refusal())


class _RefusedTuner(_WithheldTuner):
    """The stand-in of a method this library will not build adapters for.

    ``class XModel(_RefusedTuner, peft_type=PeftType.X, reason="...")``: ``reason`` is one
    clause saying why.  Registered in ``_REFUSED_TUNERS``.
    """

    _reason: ClassVar[str]

    @override
    def __init_subclass__(cls, *, peft_type: PeftType, reason: str, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls._enrol(peft_type)
        cls._reason = reason
        _REFUSED_TUNERS[peft_type] = cls

    @override
    @classmethod
    def _refusal(cls) -> str:
        return (
            f"{cls._peft_type} adapters are refused by lucid.peft: {cls._reason}; "
            f"{cls._config_class.__name__} still reads and writes adapter_config.json"
        )


class _TunerTable(Mapping[PeftType, type[_AdapterHost] | type[_WithheldTuner]]):
    """Every method's tuner class, working or refused: a live read-only view of the two registries."""

    @override
    def __getitem__(self, key: PeftType) -> type[_AdapterHost] | type[_WithheldTuner]:
        if key in _TUNERS:
            return _TUNERS[key]
        return _REFUSED_TUNERS[key]

    @override
    def __iter__(self) -> Iterator[PeftType]:
        yield from _TUNERS
        yield from _REFUSED_TUNERS

    @override
    def __len__(self) -> int:
        return len(_TUNERS) + len(_REFUSED_TUNERS)


#: ``PEFT_TYPE_TO_TUNER_MAPPING``'s source.
_ALL_TUNERS: Mapping[PeftType, type[_AdapterHost] | type[_WithheldTuner]] = _TunerTable()
