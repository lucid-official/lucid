"""``PeftModel``: an adapted model that saves and loads as the adapter alone."""

import contextlib
import dataclasses
import os
from collections.abc import Iterator
from typing import Self, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.peft._config import LoraConfig, _output_directory
from lucid.peft._forwarding import _as_module, _ForwardsAttributes
from lucid.peft._state_dict import get_peft_model_state_dict, set_peft_model_state_dict
from lucid.peft.tuners.lora._model import LoraModel
from lucid.serialization import load_safetensors, save_safetensors

_WEIGHTS_NAME = "adapter_model.safetensors"
_PICKLE_WEIGHTS_NAME = "adapter_model.bin"


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
    r"""A model with an adapter, saved and loaded as the adapter alone.

    Wraps the model in a :class:`~lucid.peft.LoraModel` kept as
    ``base_model`` (the user's model is ``base_model.model``), so the
    state dict reads ``base_model.model....lora_A.default.weight`` and the
    adapter file ``base_model.model....lora_A.weight``, the reference
    library's layouts.  Calling it runs the wrapped model, and any public
    name it lacks — ``generate``, ``config``, a head — is read from that
    model, so code written for the bare model keeps working.

    Usually made by :func:`~lucid.peft.get_peft_model`, or by
    ``from_pretrained`` for a saved adapter.

    Parameters
    ----------
    model : nn.Module
        The model to adapt, changed in place.
    peft_config : LoraConfig
        What to adapt and how.
    adapter_name : str, default="default"
        Name of the adapter.

    Raises
    ------
    TypeError
        ``peft_config`` is not a ``LoraConfig``.

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
        self, model: nn.Module, peft_config: LoraConfig, adapter_name: str = "default"
    ) -> None:
        if not isinstance(peft_config, LoraConfig):
            raise TypeError(
                f"peft_config must be a LoraConfig (lucid.peft implements LoRA), "
                f"got {type(peft_config).__name__}"
            )
        super().__init__()
        self.base_model = LoraModel(model, peft_config, adapter_name)

    @override
    def _wrapped(self) -> nn.Module | None:
        modules = self.__dict__.get("_modules")
        return _as_module(modules.get("base_model")) if modules else None

    @property
    def peft_config(self) -> dict[str, LoraConfig]:
        """The adapter's config by adapter name (``"all-linear"`` spelled out)."""
        return self.base_model.peft_config

    @property
    def active_adapter(self) -> str:
        """Name of the adapter in use."""
        return self.base_model.active_adapter

    def get_base_model(self) -> nn.Module:
        """Return the adapted model itself — the one passed in, now holding LoRA layers."""
        return self.base_model.model

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

    @contextlib.contextmanager
    def disable_adapter(self) -> Iterator[None]:
        """Run the base model alone inside the ``with`` block.

        LoRA updates are skipped (a merged layer is unmerged on entry, and
        stays unmerged after) and every ``modules_to_save`` module runs as
        the original; the adapter comes back when the block exits, even on
        an exception.  Merging inside the block is refused.  A compiled
        model follows the switch: the state it branches on is part of its
        cache key.

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
        self.base_model.disable_adapter_layers()
        try:
            yield
        finally:
            self.base_model.enable_adapter_layers()

    def merge_and_unload(
        self,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: list[str] | None = None,
    ) -> nn.Module:
        r"""Fold the adapter into the base model and hand the base model back.

        For deployment: the result runs at the base model's cost and saves,
        exports and compiles as an ordinary model.  This wrapper is spent
        afterwards.

        Parameters
        ----------
        progressbar : bool, default=False
            Kept for call compatibility with the reference; ignored.
        safe_merge : bool, default=False
            Refuse, before changing any weight, if a merged weight would
            hold NaN or inf.
        adapter_names : list of str, optional
            Which adapters to fold in (v1 has one, the default).

        Returns
        -------
        nn.Module
            ``get_base_model()``, with plain ``nn.Linear`` layers again.

        Raises
        ------
        RuntimeError
            A base weight is shared with another module (a tied head).

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
        self, save_directory: str | os.PathLike[str], safe_serialization: bool = True
    ) -> None:
        r"""Write the adapter — not the base model — to ``save_directory``.

        Two files, the reference library's: ``adapter_config.json`` (with
        ``inference_mode`` set, as the reference stores it) and
        ``adapter_model.safetensors``, keyed by
        :func:`~lucid.peft.get_peft_model_state_dict`.  An adapter not
        named ``"default"`` goes into a sub-directory of its own name, as
        the reference lays it out.

        Parameters
        ----------
        save_directory : str or os.PathLike
            Directory to write into; created if missing.
        safe_serialization : bool, default=True
            Must be ``True``: the pickle format the reference writes for
            ``False`` is not one Lucid can produce.

        Raises
        ------
        NotImplementedError
            ``safe_serialization=False``.
        RuntimeError
            The adapter was merged or unloaded already — there is nothing
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
        >>> with tempfile.TemporaryDirectory() as d:
        ...     peft.save_pretrained(d)
        ...     sorted(os.listdir(d))
        ['adapter_config.json', 'adapter_model.safetensors']
        """
        if not safe_serialization:
            raise NotImplementedError(
                "safe_serialization=False writes a pickle; lucid.peft writes safetensors only"
            )
        adapter = self.active_adapter
        state = get_peft_model_state_dict(self, adapter_name=adapter)
        path = _output_directory(save_directory)
        if adapter != "default":
            path = _output_directory(os.path.join(path, adapter))
        # The reference's writer tags its files with this format string.
        save_safetensors(state, os.path.join(path, _WEIGHTS_NAME), metadata={"format": "pt"})
        dataclasses.replace(self.peft_config[adapter], inference_mode=True).save_pretrained(path)

    @classmethod
    def from_pretrained(
        cls,
        model: nn.Module,
        model_id: str | os.PathLike[str],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: LoraConfig | None = None,
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
        config : LoraConfig, optional
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
            describes something v1 does not implement.
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
        if config is None:
            config = LoraConfig.from_pretrained(directory)
        elif not isinstance(config, LoraConfig):
            raise TypeError(f"config must be a LoraConfig, got {type(config).__name__}")
        config = dataclasses.replace(config, inference_mode=not is_trainable)
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
