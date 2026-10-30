"""``PeftModel``: an adapted model that saves and loads as the adapter alone."""

import dataclasses
import os
from collections.abc import Sequence
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Final, Self, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._config import PeftConfig, _output_directory, _require_config
from lucid.peft._forwarding import _as_module, _ForwardsAttributes
from lucid.peft._state_dict import get_peft_model_state_dict, set_peft_model_state_dict
from lucid.peft._targeting import _config_for_model
from lucid.peft.tuners._base import _BaseTuner, _tuner_for
from lucid.serialization import load_safetensors, save_safetensors

_WEIGHTS_NAME: Final = "adapter_model.safetensors"
_PICKLE_WEIGHTS_NAME: Final = "adapter_model.bin"


class _DisabledAdapter(AbstractContextManager[None]):
    """What ``PeftModel.disable_adapter()`` returns: reentrant, the outermost exit re-enables.

    Nested blocks — a helper that disables inside a caller's disabled
    block — leave the adapter off until the outermost one exits.
    """

    __slots__ = ("_model",)

    def __init__(self, model: PeftModel) -> None:
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


def _adapter_weights(directory: str) -> dict[str, Tensor]:
    path = os.path.join(directory, _WEIGHTS_NAME)
    if not os.path.isfile(path):
        if os.path.isfile(os.path.join(directory, _PICKLE_WEIGHTS_NAME)):
            raise NotImplementedError(
                f"{directory!r} holds {_PICKLE_WEIGHTS_NAME}, a pickle; lucid.peft reads "
                f"{_WEIGHTS_NAME} only — re-save the adapter with safe_serialization=True"
            )
        raise FileNotFoundError(f"no {_WEIGHTS_NAME} in {directory!r}")
    loaded = load_safetensors(path)
    return {key: value for key, value in loaded.items() if isinstance(value, Tensor)}


class PeftModel(_ForwardsAttributes):
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
    ``from_pretrained`` for a saved adapter.

    Parameters
    ----------
    model : nn.Module
        The model to adapt, changed in place.
    peft_config : PeftConfig
        What to adapt and how — a ``LoraConfig``.
    adapter_name : str, default="default"
        Name of the adapter.

    Raises
    ------
    TypeError
        ``peft_config`` is not an adapter config.

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

    def __init__(
        self, model: nn.Module, peft_config: PeftConfig, adapter_name: str = "default"
    ) -> None:
        config = _require_config("peft_config", peft_config)
        super().__init__()
        self.base_model: _BaseTuner = _tuner_for(config)(model, config, adapter_name)
        self._disable_depth = 0

    @override
    def _wrapped(self) -> nn.Module | None:
        modules = self.__dict__.get("_modules")
        return _as_module(modules.get("base_model")) if modules else None

    @property
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
        """Return the adapted model itself — the one passed in, now holding LoRA layers."""
        return self.base_model.model

    def add_adapter(
        self, adapter_name: str, peft_config: PeftConfig, low_cpu_mem_usage: bool = False
    ) -> None:
        """Add an adapter beside the ones the model holds.

        The new adapter is initialised, not trained, and starts frozen and
        inactive: the model computes what it computed, and every parameter
        keeps its ``requires_grad``.  Switch to it with ``set_adapter``.
        Modules an earlier adapter wraps take the new one beside it; a
        classification or question-answering ``task_type`` adds the model's
        head to ``modules_to_save`` as ``get_peft_model`` does.

        Parameters
        ----------
        adapter_name : str
            Name of the new adapter; unique among the model's adapters.
        peft_config : PeftConfig
            What to adapt and how.
        low_cpu_mem_usage : bool, default=False
            Accepted for the reference's signature; the adapter is built
            the same way either way.

        Raises
        ------
        ValueError
            The name is taken, the config is another method's, no module
            matches it, or two adapters would train biases.  The model is
            left as it was.

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
        config = _config_for_model(
            self.get_base_model(), _require_config("peft_config", peft_config)
        )
        self.base_model.add_adapter(adapter_name, config, low_cpu_mem_usage)

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

    @override
    def forward(self, *args: object, **kwargs: object) -> object:
        return self.get_base_model()(*args, **kwargs)

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        """Return ``(trainable, total)`` parameter counts, each shared parameter once.

        Returns
        -------
        tuple of int
            Elements in parameters that require grad, and in all of them.
        """
        trainable = total = 0
        for _, param in self.named_parameters():
            count = param.numel()
            total += count
            if param.requires_grad:
                trainable += count
        return trainable, total

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
        trainable, total = self.get_nb_trainable_parameters()
        share = 100 * trainable / total if total else 0.0
        print(
            f"trainable params: {trainable:,d} || all params: {total:,d} || trainable%: {share:.4f}"
        )

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

    def unload(self) -> nn.Module:
        """Remove the adapter without merging and return the original model.

        Returns
        -------
        nn.Module
            The model, each LoRA layer replaced by the ``nn.Linear`` it wrapped.
        """
        return self.base_model.unload()

    def save_pretrained(
        self,
        save_directory: str | os.PathLike[str],
        safe_serialization: bool = True,
        selected_adapters: list[str] | None = None,
    ) -> None:
        r"""Write the adapters — not the base model — under ``save_directory``.

        Per adapter, the reference library's two files:
        ``adapter_config.json`` (with ``inference_mode`` set, as the
        reference stores it) and ``adapter_model.safetensors``, keyed by
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

        Raises
        ------
        NotImplementedError
            ``safe_serialization=False``.
        ValueError
            ``selected_adapters`` names an adapter the model lacks.
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
        names = list(self.peft_config) if selected_adapters is None else list(selected_adapters)
        unknown = [name for name in names if name not in self.peft_config]
        if unknown:
            raise ValueError(
                f"You passed an invalid `selected_adapters` arguments, current supported adapter "
                f"names are {list(self.peft_config)} - got {names}."
            )
        # Every adapter is read before the first byte is written.
        states = {name: get_peft_model_state_dict(self, adapter_name=name) for name in names}
        root = _output_directory(save_directory)
        for name, state in states.items():
            path = root if name == "default" else _output_directory(os.path.join(root, name))
            # The reference's writer tags its files with this format string.
            save_safetensors(state, os.path.join(path, _WEIGHTS_NAME), metadata={"format": "pt"})
            dataclasses.replace(self.peft_config[name], inference_mode=True).save_pretrained(path)

    @classmethod
    def from_pretrained(
        cls,
        model: nn.Module,
        model_id: str | os.PathLike[str],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: PeftConfig | None = None,
    ) -> Self:
        r"""Adapt ``model`` with the adapter saved in ``model_id``.

        Reads ``adapter_config.json`` (unless ``config`` is given) and
        ``adapter_model.safetensors`` — written by ``save_pretrained``
        or by the reference library — injects the adapter into ``model`` in
        place and loads its tensors.  If the tensors do not fit the model,
        the injection is undone before the error is raised.

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

        Returns
        -------
        PeftModel
            ``model``, adapted and loaded.

        Raises
        ------
        FileNotFoundError
            The directory lacks ``adapter_config.json`` or the weights.
        NotImplementedError
            The weights are a pickle (``adapter_model.bin``), or the config
            describes something lucid.peft does not implement.
        ValueError
            The adapter's tensors do not fit ``model``.

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
        directory = os.fspath(model_id)
        # Without a config, the file's ``peft_type`` picks the method.
        given = PeftConfig.from_pretrained(directory) if config is None else config
        config = dataclasses.replace(
            _require_config("config", given), inference_mode=not is_trainable
        )
        weights = _adapter_weights(directory)
        peft_model = cls(model, config, adapter_name)
        try:
            set_peft_model_state_dict(peft_model, weights, adapter_name=adapter_name)
        except BaseException:
            # Whatever stopped the load, ``model`` goes back to what it was.
            peft_model.unload()
            raise
        if not is_trainable:
            peft_model.eval()
        return peft_model

    def load_adapter(
        self,
        model_id: str | os.PathLike[str],
        adapter_name: str,
        is_trainable: bool = False,
        low_cpu_mem_usage: bool = False,
    ) -> object:
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
        low_cpu_mem_usage : bool, default=False
            Accepted for the reference's signature; ignored as in
            ``add_adapter``.

        Returns
        -------
        object
            What ``model.load_state_dict(..., strict=False)`` returned, as
            ``set_peft_model_state_dict`` hands it back.

        Raises
        ------
        FileNotFoundError
            The directory lacks ``adapter_config.json`` or the weights.
        NotImplementedError
            The weights are a pickle, or the config describes something
            lucid.peft does not implement.
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
        directory = os.fspath(model_id)
        added = adapter_name not in self.peft_config
        config = None
        if added:
            config = dataclasses.replace(
                PeftConfig.from_pretrained(directory), inference_mode=not is_trainable
            )
        # Read before anything is changed, so a missing file leaves the model alone.
        weights = _adapter_weights(directory)
        if config is not None:
            self.add_adapter(adapter_name, config, low_cpu_mem_usage)
            if is_trainable:
                self.set_requires_grad(adapter_name)
        try:
            loaded = set_peft_model_state_dict(self, weights, adapter_name=adapter_name)
        except BaseException:
            if added:
                self.delete_adapter(adapter_name)
            raise
        if not is_trainable:
            self.eval()
        return loaded
