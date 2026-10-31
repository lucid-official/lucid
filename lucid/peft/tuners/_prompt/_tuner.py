"""What the three prompt-learning methods share: a tuner that holds prompts beside the model.

Prompt tuning, prefix tuning and p-tuning put nothing into the model's tree.
Each adapter is an encoder (``prompt_encoder[name]``) kept on the tuner, and
the tuner hooks the model's call sites (``lucid.peft.tuners._prompt._sites``)
so the active adapter's prompt is in front of the input — or in the
attention cache — whenever the model runs, through ``forward`` or its own
``generate`` alike.  What the method adds to that — the encoder, where the
prompt enters — is two class attributes of a subclass:

    class PromptTuningModel(_PromptLearningTuner, peft_type=PeftType.PROMPT_TUNING):
        _config_class = PromptTuningConfig
        _encoder_class = PromptEmbedding
"""

import copy
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from typing import ClassVar, Final, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn.hooks import RemovableHandle
from lucid.peft._config import PeftConfig, PromptLearningConfig, TaskType
from lucid.peft._modules_to_save import ModulesToSaveWrapper
from lucid.peft._slots import _adapter_slots, _as_names, _outermost_slots
from lucid.peft.tuners._base import (
    _AdapterHost,
    _check_adapter_name,
    _plan_modules_to_save,
    _set_submodule,
)
from lucid.peft.tuners._files import _FileContext, _FileFormat
from lucid.peft.tuners._prompt._encoder import _PromptEncoder
from lucid.peft.tuners._prompt._layout import _Layout
from lucid.peft.tuners._prompt._sites import _Applied

#: The key an adapter file stores the prompt under, whatever the method.
_PROMPT_KEY: Final = "prompt_embeddings"

#: Where a model's config keeps each size, by the names the families use.
_SIZE_NAMES: Final[Mapping[str, tuple[str, ...]]] = {
    "token_dim": ("hidden_size", "n_embd", "d_model"),
    "num_layers": ("num_hidden_layers", "num_layers", "n_layer"),
    "num_attention_heads": (
        "num_attention_heads",
        "n_head",
        "num_heads",
        "encoder_attention_heads",
    ),
}


def _model_size(model: nn.Module, field: str) -> int:
    config = getattr(model, "config", None)
    for name in _SIZE_NAMES[field]:
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    raise ValueError(f"Please specify `{field}` in `peft_config`")


class _PromptLearningTuner(_FileFormat, _AdapterHost):
    """Holds one prompt encoder per adapter and hooks the model so the active one is applied."""

    #: The config class this tuner takes, and the module that makes its prompt.
    _config_class: ClassVar[type[PromptLearningConfig]]
    _encoder_class: ClassVar[type[_PromptEncoder]]

    def __init__(self, model: nn.Module, config: PeftConfig, adapter_name: str = "default") -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
        kind = type(self)._config_class
        if not isinstance(config, kind):
            raise TypeError(f"config must be a {kind.__name__}, got {type(config).__name__}")
        if "peft_config" in model.__dict__:
            raise ValueError(
                "the model already holds adapters; unload() them before adding a prompt, "
                "or add the prompt to the PeftModel that holds them"
            )
        adapter_name = _check_adapter_name(adapter_name)
        super().__init__(model, config, adapter_name)
        self.model = model
        self.peft_config: dict[str, PeftConfig] = {}
        self.active_adapter = adapter_name
        self.prompt_encoder: nn.ModuleDict[_PromptEncoder] = nn.ModuleDict()
        # Public scalars, as the LoRA layer keeps its own (LCD-703): the sites branch on them, and
        # ``lucid.compile`` keys an executable on a module's public scalar attributes.
        self.disable_adapters = False
        self._task_ids: Tensor | None = None
        self._requires_grad_before = {
            name: param.requires_grad
            for name, param in model.named_parameters(remove_duplicate=False)
        }
        self._handles: list[RemovableHandle] = []
        try:
            self._add(adapter_name, config)
            layout = _Layout.for_task(config.task_type)
            for module, site in layout.sites(self, model, type(self)._encoder_class.injection):
                self._handles.append(module.register_forward_pre_hook(site, with_kwargs=True))
            model.peft_config = self.peft_config
            self._mark_only_adapters_as_trainable()
        except BaseException:
            # A layout that refuses the model (no head to cut, no encoder to hook) comes after
            # ``modules_to_save`` wrapped the heads: take it all out, so the model is as given.
            self.unload()
            raise

    # ── what the sites ask ────────────────────────────────────────────────

    def _applying(self) -> _Applied | None:
        """The active adapter, unless adapters are off or none is left."""
        names = self.active_adapters
        if self.disable_adapters or not names:
            return None
        config = self.peft_config[names[0]]
        assert isinstance(config, PromptLearningConfig)  # add_adapter admits no other
        return _Applied(config, self.prompt_encoder[names[0]], self._task_ids)

    @override
    @contextmanager
    def _task_scope(self, task_ids: Tensor | None) -> Iterator[None]:
        """Within the block, the prompt is the one of the task each batch row belongs to.

        Raises
        ------
        ValueError
            ``task_ids`` is given and the active adapter's prompt is not made per task.
        """
        if task_ids is None:
            yield
            return
        names = self.active_adapters
        if not names or not self.prompt_encoder[names[0]].per_task:
            raise ValueError(
                f"task_ids choose a prompt per task, which {type(self).__name__} does not have"
            )
        self._task_ids = task_ids
        try:
            yield
        finally:
            self._task_ids = None

    def _model_dtype(self) -> lucid.dtype:
        return next(iter(self.model.parameters())).dtype

    @property
    @override
    def active_adapters(self) -> list[str]:
        return _as_names(self.active_adapter)

    # ── building an adapter ───────────────────────────────────────────────

    def _complete(self, config: PromptLearningConfig) -> PromptLearningConfig:
        """A copy of ``config`` with every size the model knows filled in."""
        if config.num_virtual_tokens is None:
            raise ValueError("`num_virtual_tokens` must be set to build a prompt")
        stored = copy.deepcopy(config)
        for field in _SIZE_NAMES:
            if getattr(stored, field) is None:
                setattr(stored, field, _model_size(self.model, field))
        if stored.num_transformer_submodules is None:
            two = stored.task_type == TaskType.SEQ_2_SEQ_LM
            stored.num_transformer_submodules = 2 if two else 1
        # ``encoder_hidden_size`` is the width of a projection some methods put in the encoder.
        if hasattr(stored, "encoder_hidden_size") and stored.encoder_hidden_size is None:
            stored.encoder_hidden_size = stored.token_dim
        stored.modules_to_save = list(stored.modules_to_save or []) or None
        return stored

    def _word_embeddings(self) -> nn.Module | None:
        table = getattr(self.model, "get_input_embeddings", lambda: None)()
        return table if isinstance(table, nn.Module) else None

    def _add(self, adapter_name: str, config: PromptLearningConfig) -> None:
        """Build the adapter's encoder and ``modules_to_save`` copies; the model is as it was if that fails."""
        stored = self._complete(config)
        encoder = type(self)._encoder_class.from_config(stored, self._word_embeddings())
        device = next(iter(self.model.parameters())).device
        encoder.to(device)
        saved = _plan_modules_to_save(self.model, stored)
        held = {key: self.model.get_submodule(key) for key in saved}
        for key, module in held.items():
            if isinstance(module, ModulesToSaveWrapper):
                module.update(adapter_name)
            else:
                _set_submodule(self.model, key, ModulesToSaveWrapper(module, adapter_name))
        self.peft_config[adapter_name] = stored
        self.prompt_encoder[adapter_name] = encoder

    def _mark_only_adapters_as_trainable(self) -> None:
        trainable: set[int] = set()
        for name, encoder in self.prompt_encoder.items():
            on = name in self.active_adapters and not self.peft_config[name].inference_mode
            encoder.requires_grad_(on)
            if on:
                for _, slot in _adapter_slots(self.model):
                    trainable.update(id(p) for p in slot._adapter_parameters(name))
        for param in self.model.parameters():
            param.requires_grad_(id(param) in trainable)

    # ── the host's contract ───────────────────────────────────────────────

    @override
    def add_adapter(
        self, adapter_name: str, peft_config: PeftConfig, low_cpu_mem_usage: bool = False
    ) -> None:
        """Add a prompt beside the ones held; it starts frozen, and ``set_adapter`` switches to it.

        Raises
        ------
        ValueError
            The name is taken, or the config is another method's.
        """
        del low_cpu_mem_usage  # the encoder is allocated the one way
        adapter_name = self._admit(adapter_name, peft_config)
        assert isinstance(peft_config, PromptLearningConfig)  # _admit checked the class
        self._add(adapter_name, peft_config)
        self.set_requires_grad(adapter_name, False)

    def _require_known(self, adapter_names: Sequence[str]) -> None:
        for name in adapter_names:
            if name not in self.peft_config:
                raise ValueError(f"Adapter {name} not found.")

    @override
    def set_adapter(self, adapter_name: str | list[str], inference_mode: bool = False) -> None:
        """Apply the named prompt — one at a time — and train it unless ``inference_mode``."""
        names = _as_names(adapter_name)
        if len(names) != 1:
            raise ValueError(f"prompt learning applies one adapter at a time, got {names}")
        self._require_known(names)
        for _, slot in _adapter_slots(self.model):
            slot.set_adapter(names[0], inference_mode)
        self.active_adapter = names[0]
        for held, encoder in self.prompt_encoder.items():
            encoder.requires_grad_(held == names[0] and not inference_mode)

    @override
    def set_requires_grad(
        self, adapter_names: str | Sequence[str], requires_grad: bool = True
    ) -> None:
        names = _as_names(adapter_names)
        self._require_known(names)
        for name in names:
            self.prompt_encoder[name].requires_grad_(requires_grad)
        for _, slot in _adapter_slots(self.model):
            slot.set_requires_grad(names, requires_grad)

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Remove a prompt and its ``modules_to_save`` copies; another takes over if it was in use."""
        if adapter_name not in self.peft_config:
            raise ValueError(f"Adapter {adapter_name} does not exist")
        del self.peft_config[adapter_name]
        del self.prompt_encoder[adapter_name]
        for _, slot in _adapter_slots(self.model):
            slot.delete_adapter(adapter_name)
        if adapter_name not in self.active_adapters:
            return
        if not self.peft_config:
            self.active_adapter = []
            return
        successor = sorted(self.peft_config)[0]
        warnings.warn(
            f"Adapter {adapter_name} was active which is now deleted. "
            f"Setting active adapter to {successor}.",
            stacklevel=3,
        )
        self.set_adapter(successor)

    @override
    def unload(self) -> nn.Module:
        """Take the hooks and the ``modules_to_save`` copies out and give every parameter its old ``requires_grad``."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        model = self.model
        for key, slot in _outermost_slots(model):
            _set_submodule(model, key, slot._unwrap(False))
        for name, param in model.named_parameters(remove_duplicate=False):
            before = self._requires_grad_before.get(name)
            if before is not None:
                param.requires_grad_(before)
        model.__dict__.pop("peft_config", None)
        return model

    @override
    def merge_and_unload(
        self,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: list[str] | None = None,
    ) -> nn.Module:
        """Refused: a prompt has no weight to fold into the model."""
        del progressbar, safe_merge, adapter_names
        raise NotImplementedError(
            "a prompt-learning adapter changes the input, not a weight, so there is nothing "
            "to merge; unload() removes it"
        )

    @override
    def enable_adapter_layers(self) -> None:
        self.disable_adapters = False
        for _, slot in _adapter_slots(self.model):
            slot.enable_adapters(True)

    @override
    def disable_adapter_layers(self) -> None:
        self.disable_adapters = True
        for _, slot in _adapter_slots(self.model):
            slot.enable_adapters(False)

    @override
    def _mixed_batch(
        self, adapter_names: Sequence[str], num_beams: int = 1
    ) -> AbstractContextManager[None]:
        del adapter_names, num_beams
        raise NotImplementedError(
            f"{type(self).__name__} cannot send the rows of a batch through different prompts"
        )

    @override
    def forward(self, *args: object, **kwargs: object) -> object:
        return self.model(*args, **kwargs)

    @classmethod
    @override
    def _extra_trainable_names(cls, model: nn.Module, config: PeftConfig) -> list[str]:
        return []

    @classmethod
    @override
    def _adapt_in_place(cls, model: nn.Module, config: PeftConfig, adapter_name: str) -> None:
        raise TypeError(
            f"{cls.__name__} keeps its prompt beside the model, which a bare model has no place "
            "for; adapt with get_peft_model"
        )

    @classmethod
    @override
    def _check_loadable(cls, config: PeftConfig, is_trainable: bool) -> None:
        if is_trainable:
            raise ValueError(
                "Cannot set a prompt learning adapter to trainable when loading pretrained adapter."
            )

    # ── the adapter file ──────────────────────────────────────────────────

    @classmethod
    def _holder(cls, context: _FileContext) -> _PromptLearningTuner:
        """The tuner in ``context.model``, whether that is a ``PeftModel`` or the tuner itself."""
        model = context.model
        holder = model if isinstance(model, _PromptLearningTuner) else model.base_model
        assert isinstance(holder, _PromptLearningTuner)
        return holder

    @classmethod
    @override
    def _shared_keys(cls, context: _FileContext) -> dict[str, str]:
        held = f"{context.host}prompt_encoder.{context.adapter}."
        return {held + own: disk for own, disk in cls._encoder_class.stored.items()}

    @classmethod
    @override
    def _to_file(cls, context: _FileContext, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """The prompt as the encoder makes it, and the model's own keys without the tuner's ``model.`` level."""
        encoder = cls._holder(context).prompt_encoder[context.adapter]
        own = set(cls._encoder_class.stored.values())
        out: dict[str, Tensor] = {}
        for key, value in state.items():
            if key == _PROMPT_KEY:
                out[key] = value if encoder.stores_prompt else encoder.prompt_embeddings()
            elif key in own:
                out[key] = value
            else:
                out[context.host + key.removeprefix(context.prefix)] = value
        return out

    @classmethod
    @override
    def _from_file(cls, context: _FileContext, state: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        encoder = cls._holder(context).prompt_encoder[context.adapter]
        if _PROMPT_KEY in state and not encoder.stores_prompt:
            raise ValueError(
                "this adapter is built to train, so its embedding is not the prompt a file holds; "
                "load the file as an inference adapter (PeftModel.from_pretrained)"
            )
        own = set(cls._encoder_class.stored.values())
        return {
            (key if key in own else context.prefix + key.removeprefix(context.host)): value
            for key, value in state.items()
        }
