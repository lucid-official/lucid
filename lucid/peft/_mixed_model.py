"""``PeftMixedModel``: one model holding the adapters of several methods, read from single-method files."""

import os
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from typing import NoReturn, Self, override

import lucid.nn as nn
from lucid.nn._incompatible_keys import _IncompatibleKeys
from lucid.peft._config import PeftConfig, _adapter_directory, _require_config
from lucid.peft._convert import ConvertMode
from lucid.peft._modules_to_save import ModulesToSaveWrapper
from lucid.peft._peft_model import (
    _AdaptedModel,
    _build_filled,
    _DisabledAdapter,
    _load_adapter,
    _names_or_none,
    _print_parameter_share,
    _read_converted,
    _ReadOptions,
    _saved_module_names,
    _stored_config,
)
from lucid.peft._refusals import (
    NO_MIXED_SAVE,
    NO_MODEL_CARD,
    refuse_build_options,
    refuse_method,
    reject_unexpected,
)
from lucid.peft._slots import _as_names
from lucid.peft._status import _parameter_counts
from lucid.peft._targeting import _config_for_model, _module_names_matching
from lucid.peft.tuners._layer import _tuner_layers
from lucid.peft.tuners._mixed import MixedModel
from lucid.peft.tuners._tuner import _MIXED_TUNERS

#: What ``load_adapter`` takes positionally after the adapter's name, in the order
#: ``PeftModel.load_adapter`` lists them: the reference forwards them as ``*args``.
_LOAD_POSITIONAL = (
    "is_trainable",
    "device",
    "autocast_adapter_dtype",
    "ephemeral_gpu_offload",
    "low_cpu_mem_usage",
    "key_mapping",
)
_MERGE_POSITIONAL = ("progressbar", "safe_merge", "adapter_names")


def _bound(
    where: str, names: Sequence[str], args: Sequence[object], kwargs: Mapping[str, object]
) -> dict[str, object]:
    """``args`` and ``kwargs`` as one dict by parameter name, the positional ones taking ``names`` in order.

    The reference hands a call's ``*args, **kwargs`` to a function with named parameters; this
    does the binding, so a value is read under its name and checked.
    """
    if len(args) > len(names):
        raise TypeError(
            f"{where}() takes at most {len(names)} more positional arguments, got {len(args)}"
        )
    bound: dict[str, object] = dict(zip(names, args, strict=False))
    for key, value in kwargs.items():
        if key in bound:
            raise TypeError(f"{where}() got multiple values for argument {key!r}")
        bound[key] = value
    return bound


def _flag(where: str, given: dict[str, object], name: str, default: bool) -> bool:
    value = given.pop(name, default)
    if isinstance(value, bool):
        return value
    raise TypeError(f"{where}: `{name}` must be a bool, got {type(value).__name__}")


def _text_or_none(where: str, given: dict[str, object], name: str) -> str | None:
    value = given.pop(name, None)
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"{where}: `{name}` must be a str, got {type(value).__name__}")


def _renaming_or_none(where: str, given: dict[str, object]) -> Mapping[str, str] | None:
    value = given.pop("key_mapping", None)
    if value is None:
        return None
    if isinstance(value, Mapping):
        pairs = {k: v for k, v in value.items() if isinstance(k, str) and isinstance(v, str)}
        if len(pairs) == len(value):
            return pairs
    raise TypeError(f"{where}: `key_mapping` must map str patterns to str replacements")


def _convert_mode(where: str, given: dict[str, object]) -> ConvertMode:
    value = given.pop("convert", "auto")
    if isinstance(value, bool):
        return value
    if value == "auto":
        return "auto"
    raise TypeError(f"{where}: `convert` must be True, False or 'auto', got {value!r}")


def _read_options(where: str, given: dict[str, object]) -> _ReadOptions:
    """The options of reading an adapter out of the keyword arguments ``given`` (consumed); what is left is not supported."""
    return _ReadOptions(
        key_mapping=_renaming_or_none(where, given),
        subfolder=_text_or_none(where, given, "subfolder"),
        convert=_convert_mode(where, given),
        device=_text_or_none(where, given, "device"),
        autocast_adapter_dtype=_flag(where, given, "autocast_adapter_dtype", True),
        ephemeral_gpu_offload=_flag(where, given, "ephemeral_gpu_offload", False),
        low_cpu_mem_usage=_flag(where, given, "low_cpu_mem_usage", False),
        ignore_mismatched_sizes=given.pop("ignore_mismatched_sizes", False),
        unsupported=given,
    )


class PeftMixedModel(_AdaptedModel):
    r"""A model holding the adapters of several methods at once, loaded from files and not saved.

    The first adapter of a method puts that method's layer on each module it targets; the layers of
    other methods stand on it, so the forward pass runs them lowest first.  The tuner is a
    :class:`~lucid.peft.tuners.MixedModel`, kept as ``base_model`` (the user's model is
    ``base_model.model``), and the state dict reads ``base_model.model....`` as a
    :class:`~lucid.peft.PeftModel`'s does.  Calling it runs the wrapped model, and any public name
    it lacks is read from that model, so code written for the bare model keeps working.

    It is not a ``PeftModel``: there is no task wrapper, no ``adapter_names`` routing of a batch
    and no ``save_pretrained``.  An adapter is written by the ``PeftModel`` of its own method
    and read here with :meth:`load_adapter` or :meth:`from_pretrained`, which fill it from the
    single-method file, before or after the layers of other methods are stacked on it.  The
    reference's mixed model writes nothing either, and says so with the same ``NotImplementedError``.

    The methods that can be mixed are the keys of
    ``PEFT_TYPE_TO_MIXED_MODEL_MAPPING``; any other is refused with ``ValueError`` where it is
    added.  Adding an adapter that cannot sit where it would land raises
    ``ValueError`` and leaves the model as it was.  AdaLoRA takes part for inference only.

    Parameters
    ----------
    model : nn.Module
        The model to adapt, changed in place; it must hold no adapters yet.
    peft_config : PeftConfig
        The first adapter's config: any mixable method's.
    adapter_name : str, default="default"
        Name of the first adapter.

    Raises
    ------
    TypeError
        ``peft_config`` is not an adapter config.
    ValueError
        The method is not mixable, the model already holds adapters, or the adapter cannot be
        applied.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoHaConfig, LoraConfig, PeftMixedModel
    >>> net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> mixed = PeftMixedModel(net, LoraConfig(r=2, target_modules=["0"]), "lora")
    >>> mixed.add_adapter("loha", LoHaConfig(r=2, target_modules=["0"]))
    >>> mixed.set_adapter(["lora", "loha"])
    >>> mixed.active_adapters
    ['lora', 'loha']
    >>> mixed(lucid.randn(4, 8)).shape
    (4, 2)
    """

    def __init__(
        self, model: nn.Module, peft_config: PeftConfig, adapter_name: str = "default"
    ) -> None:
        """Adapt ``model`` in place with ``peft_config``'s method as ``adapter_name`` and wrap it."""
        super().__init__()
        self.base_model: MixedModel = MixedModel(model, peft_config, adapter_name)
        self._disable_depth = 0

    @property
    @override
    def peft_config(self) -> dict[str, PeftConfig]:
        """Each adapter's config by adapter name."""
        return self.base_model.peft_config

    @property
    def active_adapter(self) -> str | list[str]:
        """Name of the adapter in use (a list once several are)."""
        return self.base_model.active_adapter

    @property
    def active_adapters(self) -> list[str]:
        """The adapters in use, as a list."""
        return self.base_model.active_adapters

    @property
    def modules_to_save(self) -> set[str] | None:
        """The names of the modules trained in full beside the adapters, over every adapter; ``None`` if there are none."""
        return _saved_module_names(self.peft_config.values())

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        """Return ``(trainable, total)`` parameter counts, each shared parameter once.

        Returns
        -------
        tuple of int
            Elements in parameters that require grad, and in all of them.
        """
        return _parameter_counts(self)

    def print_trainable_parameters(self) -> None:
        """Print the trainable / total parameter counts and their ratio.

        Examples
        --------
        >>> import lucid.nn as nn
        >>> from lucid.peft import LoraConfig, PeftMixedModel
        >>> mixed = PeftMixedModel(nn.Sequential(nn.Linear(8, 8)),
        ...                        LoraConfig(r=2, target_modules=["0"]))
        >>> mixed.print_trainable_parameters()
        trainable params: 32 || all params: 104 || trainable%: 30.7692
        """
        _print_parameter_share(*self.get_nb_trainable_parameters())

    @override
    def forward(self, *args: object, **kwargs: object) -> object:
        r"""Run the adapted model; every argument goes to it as given.

        Parameters
        ----------
        *args : object
            Positional arguments of the model's ``forward``.
        **kwargs : object
            Keyword arguments of the model's ``forward``.  Unlike
            :meth:`~lucid.peft.PeftModel.forward` there is no ``adapter_names`` here.

        Returns
        -------
        object
            What the model's ``forward`` returns.
        """
        return self.base_model(*args, **kwargs)

    def generate(self, *args: object, **kwargs: object) -> object:
        r"""Generate with the adapted model: its own ``generate``, called as given.

        Parameters
        ----------
        *args : object
            Positional arguments of the model's ``generate``.
        **kwargs : object
            Keyword arguments of the model's ``generate``.

        Returns
        -------
        object
            What the model's ``generate`` returns.

        Raises
        ------
        AttributeError
            The adapted model has no ``generate``.
        """
        model = self.base_model.model
        generate = getattr(model, "generate", None)
        if generate is None:
            raise AttributeError(f"{type(model).__name__} has no generate() to call")
        return generate(*args, **kwargs)

    def disable_adapter(self) -> AbstractContextManager[None]:
        """Run the base model alone inside the ``with`` block, whatever stack of adapters is in use.

        The adapters come back when the block exits, even on an exception.  Blocks nest: they stay
        off until the outermost one exits.

        Examples
        --------
        >>> import lucid
        >>> import lucid.nn as nn
        >>> from lucid.peft import LoraConfig, PeftMixedModel
        >>> net = nn.Sequential(nn.Linear(4, 4))
        >>> x = lucid.randn(2, 4)
        >>> base_out = net(x)
        >>> mixed = PeftMixedModel(net, LoraConfig(r=2, target_modules=["0"],
        ...                                        init_lora_weights=False))
        >>> with mixed.disable_adapter():
        ...     bool((mixed(x) == base_out).all().item())
        True
        """
        return _DisabledAdapter(self)

    @override
    def add_adapter(
        self,
        adapter_name: str,
        peft_config: PeftConfig,
        low_cpu_mem_usage: bool = False,
        autocast_adapter_dtype: bool = True,
    ) -> None:
        """Add an adapter, of any mixable method, beside the ones the model holds.

        The new adapter starts frozen and inactive; ``set_adapter`` switches to it.  A module that
        a method already adapts takes this adapter in that method's layer, and a module another
        method adapted gets this method's layer stacked on top.

        Parameters
        ----------
        adapter_name : str
            The new adapter's name, which no adapter of the model has.
        peft_config : PeftConfig
            The adapter's config: a method of the mixed-model mapping.
        low_cpu_mem_usage : bool, default=False
            The reference's meta-device build; ``True`` raises ``NotImplementedError``.
        autocast_adapter_dtype : bool, default=True
            The reference's float32 cast for a half-precision base, always done here; ``False``
            raises ``NotImplementedError``.

        Raises
        ------
        ValueError
            The name is taken, the method is not mixable, no module matches, or the adapter cannot
            stand where it would land.  The model is left as it was.
        NotImplementedError
            ``low_cpu_mem_usage`` or ``autocast_adapter_dtype`` holds a value the library cannot
            honour.

        Examples
        --------
        >>> import lucid.nn as nn
        >>> from lucid.peft import LoHaConfig, LoraConfig, PeftMixedModel
        >>> net = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        >>> mixed = PeftMixedModel(net, LoraConfig(r=2, target_modules=["0"]), "lora")
        >>> mixed.add_adapter("loha", LoHaConfig(r=2, target_modules=["0", "1"]))
        >>> sorted(mixed.peft_config)
        ['loha', 'lora']
        """
        refuse_build_options(
            "PeftMixedModel.add_adapter",
            autocast_adapter_dtype=autocast_adapter_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        self.base_model.add_adapter(adapter_name, self._host_config(peft_config))

    @override
    def _host_config(self, peft_config: PeftConfig) -> PeftConfig:
        return _config_for_model(self.base_model.model, _require_config("peft_config", peft_config))

    def set_modules_to_save(self, peft_config: PeftConfig, adapter_name: str) -> None:
        """Check that the modules ``peft_config`` trains in full are held by ``adapter_name``.

        The reference wraps those modules after its tuner has built the adapter.  Here the tuner
        wraps them as the adapter goes in, so by the time ``adapter_name`` exists its modules are
        wrapped, and this method confirms it: it changes nothing.  The ``modules_to_save`` of an
        adapter cannot be widened afterwards; add the adapter with a config that names them.

        Parameters
        ----------
        peft_config : PeftConfig
            The config naming the modules, normally ``peft_config[adapter_name]``.
        adapter_name : str
            An adapter the model holds.

        Raises
        ------
        ValueError
            The model has no adapter of that name, or a module the config names is not trained
            for it.

        Examples
        --------
        >>> import lucid.nn as nn
        >>> from lucid.peft import LoraConfig, PeftMixedModel
        >>> net = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
        >>> mixed = PeftMixedModel(net, LoraConfig(r=2, target_modules=["0"], modules_to_save=["1"]))
        >>> mixed.set_modules_to_save(mixed.peft_config["default"], "default")
        >>> mixed.modules_to_save
        {'1'}
        """
        self._require_held([adapter_name])
        names = _saved_module_names([_require_config("peft_config", peft_config)])
        if names is None:
            return
        model = self.base_model.model
        untrained = [
            key
            for key in _module_names_matching(model, names)
            if not _holds_copy(model.get_submodule(key), adapter_name)
        ]
        if untrained:
            raise ValueError(
                f"adapter {adapter_name!r} does not train {untrained} in full: modules_to_save "
                "is applied as an adapter is added, so add it with a config that names them"
            )

    def _require_held(self, names: Sequence[str]) -> None:
        """Raise ``ValueError`` listing the names that are no adapter of the model."""
        held = self.peft_config
        missing = sorted(set(names) - set(held))
        if missing:
            raise ValueError(f"Adapter(s) {missing} not found, available adapters: {sorted(held)}")

    def set_adapter(self, adapter_name: str | list[str], inference_mode: bool = False) -> None:
        """Make the named adapter(s), of any methods, the ones in use.

        Each layer applies those of the names it holds and passes its input through otherwise;
        the named adapters train, and every other adapter is frozen, unless ``inference_mode``.
        A merged model is unmerged first.

        Parameters
        ----------
        adapter_name : str or list of str
            The adapter, or adapters, to apply.
        inference_mode : bool, default=False
            Freeze the named adapters too.

        Raises
        ------
        ValueError
            A name is not one of the model's adapters.
        """
        self._require_held(_as_names(adapter_name))
        self.base_model.set_adapter(adapter_name, inference_mode)

    @override
    def delete_adapter(self, adapter_name: str | list[str]) -> None:
        """Remove the named adapter(s), their configs and their layers' tables.

        Every name is checked before the first is removed, so a refusal leaves the model as it was.
        If an adapter in use is removed and none remains in use, the first of the others (by
        name) takes over, with a warning.

        Parameters
        ----------
        adapter_name : str or list of str
            The adapter, or adapters, to remove.

        Raises
        ------
        ValueError
            A name is not one of the model's adapters, or one of them is merged.
        """
        names = list(dict.fromkeys(_as_names(adapter_name)))
        self._require_held(names)
        merged = {
            n for _, layer in _tuner_layers(self.base_model.model) for n in layer.merged_adapters
        }
        folded = sorted(set(names) & merged)
        if folded:
            raise ValueError(
                f"Cannot delete adapter(s) {folded} while they are merged. "
                "Please unmerge them first."
            )
        for name in names:
            self.base_model.delete_adapter(name)

    def merge_and_unload(self, *args: object, **kwargs: object) -> nn.Module:
        r"""Fold the adapters into the base model and hand the base model back.

        Each module's layers are folded lowest first, and the adapters that were not folded in are
        gone with their layers.  A merge that would fold a layer's update under a layer that
        transforms it differently is refused before any weight is written.

        Parameters
        ----------
        *args : object
            ``progressbar``, ``safe_merge`` and ``adapter_names``, positionally and in that order.
        **kwargs : object
            The same, by name.  ``progressbar`` is kept for call compatibility and ignored;
            ``safe_merge`` refuses, before changing any weight, a merged weight holding NaN or
            inf; ``adapter_names`` (a list of str) picks the adapters to fold in, the ones in use
            by default.

        Returns
        -------
        nn.Module
            The base model, with its plain layers again.

        Raises
        ------
        TypeError
            An argument is not one of those, or is of the wrong type.
        ValueError
            A name is not an adapter of the model, or the merge would fold an update under a
            layer that transforms it.

        Examples
        --------
        >>> import lucid
        >>> import lucid.nn as nn
        >>> from lucid.peft import LoHaConfig, LoraConfig, PeftMixedModel
        >>> net = nn.Sequential(nn.Linear(4, 4))
        >>> mixed = PeftMixedModel(net, LoraConfig(r=2, target_modules=["0"],
        ...                                        init_lora_weights=False), "lora")
        >>> mixed.add_adapter("loha", LoHaConfig(r=2, target_modules=["0"]))
        >>> mixed.set_adapter(["lora", "loha"])
        >>> x = lucid.randn(3, 4)
        >>> stacked = mixed(x)
        >>> plain = mixed.merge_and_unload()
        >>> type(plain[0]) is nn.Linear, bool(lucid.allclose(plain(x), stacked, atol=1e-5))
        (True, True)
        """
        where = "PeftMixedModel.merge_and_unload"
        given = _bound(where, _MERGE_POSITIONAL, args, kwargs)
        progressbar = _flag(where, given, "progressbar", False)
        safe_merge = _flag(where, given, "safe_merge", False)
        names = _names_or_none(given.pop("adapter_names", None))
        reject_unexpected(where, given)
        return self.base_model.merge_and_unload(
            progressbar, safe_merge, None if names is None else list(names)
        )

    @override
    def unload(self, *args: object, **kwargs: object) -> nn.Module:
        """Remove every adapter without merging and return the original model.

        Parameters
        ----------
        *args : object
            None: the reference's tuner ignores them, and here they raise ``TypeError``.
        **kwargs : object
            None, as for ``args``.

        Returns
        -------
        nn.Module
            The model, each adapter layer replaced by the module it stood on.

        Raises
        ------
        TypeError
            An argument was given.
        """
        if args or kwargs:
            raise TypeError(
                f"PeftMixedModel.unload() takes no arguments, got {len(args)} positional "
                f"and {sorted(kwargs)} by name"
            )
        return self.base_model.unload()

    def get_layer_status(self) -> NoReturn:
        """Not supported: the status of a layer is read off one method's tuner.

        Raises
        ------
        TypeError
            Always, as the reference's mixed model does.
        """
        raise TypeError(f"get_layer_status is not supported for {type(self).__name__}.")

    def get_model_status(self) -> NoReturn:
        """Not supported: the status of a model is read off one method's tuner.

        Raises
        ------
        TypeError
            Always, as the reference's mixed model does.
        """
        raise TypeError(f"get_model_status is not supported for {type(self).__name__}.")

    def create_or_update_model_card(self, output_dir: str) -> None:
        """Refused: the reference writes a hub model card, which this library has no use for.

        Raises
        ------
        NotImplementedError
            Always.
        """
        del output_dir
        refuse_method("PeftMixedModel.create_or_update_model_card", NO_MODEL_CARD)

    def save_pretrained(
        self,
        save_directory: str | os.PathLike[str],
        safe_serialization: bool = False,
        selected_adapters: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        """Refused: a mixed model is not saved, as the reference's is not.

        Write each adapter from a :class:`~lucid.peft.PeftModel` of its own method; the file is
        read back here by :meth:`load_adapter`.

        Parameters
        ----------
        save_directory : str or os.PathLike
            Where the adapters would go; not read, and not created.
        safe_serialization : bool, default=False
            The reference's default for a format the reference's mixed model never writes; not read.
        selected_adapters : list of str, optional
            The adapters to write; not read.
        **kwargs : object
            Not read.

        Raises
        ------
        NotImplementedError
            Always; nothing is written.
        """
        del save_directory, safe_serialization, selected_adapters, kwargs
        refuse_method("PeftMixedModel.save_pretrained", NO_MIXED_SAVE)

    def load_adapter(
        self, model_id: str | os.PathLike[str], adapter_name: str, *args: object, **kwargs: object
    ) -> _IncompatibleKeys:
        r"""Load a saved adapter into the model, beside the ones it holds.

        An adapter the model does not hold is added from the directory's config and filled from its
        weights, stacked on the modules it targets; the weights of one the model already holds
        replace the values it has.  The directory holds what a single-method
        :meth:`~lucid.peft.PeftModel.save_pretrained` wrote, and loads before or after other
        methods' layers are stacked on its modules.  The loaded adapter is not made the one in use
        — ``set_adapter`` does that.  Nothing changes if the files are missing or do not fit.

        Parameters
        ----------
        model_id : str or os.PathLike
            The local directory that holds the adapter's ``adapter_config.json`` and
            ``adapter_model.safetensors``.
        adapter_name : str
            Name to give the adapter.
        *args : object
            In order: ``is_trainable``, ``device``, ``autocast_adapter_dtype``,
            ``ephemeral_gpu_offload``, ``low_cpu_mem_usage``, ``key_mapping`` — as for
            :meth:`~lucid.peft.PeftModel.load_adapter`.
        **kwargs : object
            Those by name, and ``subfolder``, ``convert`` and ``ignore_mismatched_sizes``, as for
            :meth:`~lucid.peft.PeftModel.load_adapter`; any other key raises
            ``NotImplementedError`` (the reference's hub options).

        Returns
        -------
        _IncompatibleKeys
            What ``set_peft_model_state_dict`` hands back.

        Raises
        ------
        FileNotFoundError
            The directory lacks ``adapter_config.json`` or the weights.
        NotImplementedError
            An option the library cannot honour is set.
        TypeError
            An argument is of the wrong type.
        ValueError
            The tensors do not fit, or the adapter's method cannot be mixed or cannot stand where
            it would land.

        Examples
        --------
        >>> import tempfile
        >>> import lucid.nn as nn
        >>> from lucid.peft import LoHaConfig, LoraConfig, PeftMixedModel, get_peft_model
        >>> source = get_peft_model(nn.Sequential(nn.Linear(4, 2)), LoHaConfig(r=2, target_modules=["0"]))
        >>> mixed = PeftMixedModel(nn.Sequential(nn.Linear(4, 2)), LoraConfig(r=2, target_modules=["0"]), "lora")
        >>> with tempfile.TemporaryDirectory() as d:
        ...     source.save_pretrained(d)
        ...     _ = mixed.load_adapter(d, "loha")
        >>> sorted(mixed.peft_config), mixed.active_adapter
        (['loha', 'lora'], 'lora')
        """
        where = "PeftMixedModel.load_adapter"
        given = _bound(where, _LOAD_POSITIONAL, args, kwargs)
        is_trainable = _flag(where, given, "is_trainable", False)
        return _load_adapter(
            self, where, model_id, adapter_name, is_trainable, _read_options(where, given)
        )

    @classmethod
    def from_pretrained(
        cls,
        model: nn.Module,
        model_id: str | os.PathLike[str],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: PeftConfig | None = None,
        **kwargs: object,
    ) -> Self:
        r"""Adapt ``model`` with the adapter saved in ``model_id``, ready to take the adapters of other methods.

        Reads ``adapter_config.json`` (unless ``config`` is given) and ``adapter_model.safetensors``
        — what a single-method :meth:`~lucid.peft.PeftModel.save_pretrained` wrote — injects the
        adapter into ``model`` in place and loads its tensors; if they do not fit, the injection is
        undone before the error is raised.  Only a method of ``PEFT_TYPE_TO_MIXED_MODEL_MAPPING`` can
        be loaded.

        Parameters
        ----------
        model : nn.Module
            The model the adapter was trained beside; it must hold no adapters.
        model_id : str or os.PathLike
            The local directory of the adapter's two files; hub ids are not resolved.
        adapter_name : str, default="default"
            The name the adapter gets.
        is_trainable : bool, default=False
            Leave the adapter training.  Otherwise it is frozen and the model goes to ``eval()``
            mode, for inference.
        config : PeftConfig, optional
            A config to use in place of the directory's.
        **kwargs : object
            ``subfolder``, ``convert``, ``key_mapping`` and the other options of
            :meth:`load_adapter`, by name, as for :meth:`~lucid.peft.PeftModel.from_pretrained`.

        Returns
        -------
        PeftMixedModel
            ``model``, adapted and loaded.

        Raises
        ------
        FileNotFoundError
            The directory lacks ``adapter_config.json`` or the weights.
        NotImplementedError
            An option the library cannot honour is set.
        TypeError
            ``config`` is not an adapter config, or an option is of the wrong type.
        ValueError
            The adapter's method is not in the mixed-model mapping, or its tensors do not fit
            ``model``.

        Examples
        --------
        >>> import tempfile
        >>> import lucid
        >>> import lucid.nn as nn
        >>> from lucid.peft import LoraConfig, PeftMixedModel, get_peft_model
        >>> def base():
        ...     lucid.manual_seed(0)
        ...     return nn.Sequential(nn.Linear(4, 2))
        >>> config = LoraConfig(r=2, target_modules=["0"], init_lora_weights=False)
        >>> trained = get_peft_model(base(), config)
        >>> x = lucid.randn(3, 4)
        >>> with tempfile.TemporaryDirectory() as d:
        ...     trained.save_pretrained(d)
        ...     loaded = PeftMixedModel.from_pretrained(base(), d)
        >>> bool(lucid.allclose(loaded(x), trained(x)))
        True
        """
        where = "PeftMixedModel.from_pretrained"
        options = _read_options(where, kwargs)
        options.refuse(where)
        directory = _adapter_directory(model_id, options.subfolder)
        stored = _stored_config(directory, config, is_trainable)
        if stored.peft_type not in _MIXED_TUNERS:
            raise ValueError(
                f"Adapter of type {stored.peft_type} is not supported for mixed models."
            )
        converted = _read_converted(model, directory, stored, is_trainable, options)
        mixed = _build_filled(cls, model, converted.config, converted.tensors, adapter_name)
        if not is_trainable:
            mixed.eval()
        return mixed


def _holds_copy(module: nn.Module, adapter_name: str) -> bool:
    """Whether ``module`` is a ``modules_to_save`` wrapper that trains a copy for ``adapter_name``."""
    return isinstance(module, ModulesToSaveWrapper) and adapter_name in module.modules_to_save
