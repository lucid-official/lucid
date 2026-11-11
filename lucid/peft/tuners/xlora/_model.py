"""``XLoraModel``: the X-LoRA tuner — LoRA experts loaded by name, mixed per token by a classifier."""

import copy
import inspect
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from typing import ClassVar, Final, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid._types import _ModuleOutput
from lucid.nn.hooks import RemovableHandle
from lucid.peft._config import PeftConfig, PeftType
from lucid.peft._peft_model import PeftModel
from lucid.peft._refusals import refuse_build_options, refuse_kwargs, refuse_value
from lucid.peft._slots import _as_names
from lucid.peft.tuners._host import _AdapterHost
from lucid.peft.tuners._layer import _adapter_dtype, _check_adapter_name, _tuner_layers
from lucid.peft.tuners._files import _FileContext, _FileFormat
from lucid.peft.tuners._prompt._tuner import _model_size
from lucid.peft.tuners.lora import Linear, LoraLayer
from lucid.peft.tuners.xlora._classifier import XLoraClassifier
from lucid.peft.tuners.xlora._config import XLoraConfig
from lucid.peft.tuners.xlora._hooks import _ScaledExperts
from lucid.utils.cache import Cache

#: The key prefix of the classifier in the adapter file, as the reference spells it.
_CLASSIFIER: Final = "internal_xlora_classifier"
#: Arguments of the scalings pass that must not reach it: it neither computes a loss nor touches a cache.
_NOT_IN_THE_PASS: Final = ("labels", "past_key_values")


def _named_arguments(
    module: nn.Module, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> dict[str, object]:
    """The call as ``{parameter name: value}``, whatever mix of positional and keyword it was."""
    return dict(inspect.signature(module.forward).bind_partial(*args, **kwargs).arguments)


class XLoraModel(_FileFormat, _AdapterHost, peft_type=PeftType.XLORA):
    r"""A model whose LoRA experts are weighed per token by a classifier on its hidden states.

    Construction loads the LoRA adapters ``config.adapters`` names (as ``"0"``, ``"1"``, ... through
    the multi-adapter API, all active together), switches their layers to the base module and
    hooks each to add the experts' updates, scaled.  A call then runs twice: a *scalings pass*,
    without gradients, on the model with every expert held at ``scaling_pass_value``, whose final
    hidden states the classifier (``internal_xlora_classifier``) turns into one weight per expert
    and token; and the real pass, in which each adapted layer adds
    :math:`g\sum_k u_k(s_k\odot x)`.  The classifier is the only thing that trains (the experts as
    well with ``use_trainable_adapters``), and the only thing the adapter file holds.

    The scalings need the whole sequence at every step: generate with ``use_cache=False``.  Layers
    of the experts must be ``nn.Linear`` LoRA layers, without DoRA.  Nothing merges.  The model
    keeps one X-LoRA adapter.

    Parameters
    ----------
    model : nn.Module
        The base model; its trunk (``base_model_prefix``) should return ``last_hidden_state``.
    config : XLoraConfig
        The experts and the classifier.
    adapter_name : str
        Name of the X-LoRA adapter (the experts keep their own names).
    device : str, optional
        The reference loads the experts onto this device.  They load onto the model's own;
        anything but ``None`` raises ``NotImplementedError``.
    ephemeral_gpu_offload : bool, default=False
        Not implemented: ``True`` raises ``NotImplementedError``.
    autocast_adapter_dtype : bool, default=True
        Always the case here; ``False`` raises ``NotImplementedError``.
    **kwargs : object
        Accepted for the reference's signature (it forwards them to its loader); any key
        raises ``NotImplementedError``.

    Raises
    ------
    ValueError
        The model already holds adapters, ``config.adapters`` is empty or names a directory
        that holds no LoRA adapter, an expert uses DoRA, or the model gives no hidden size.
    NotImplementedError
        An expert adapts a layer that is not an ``nn.Linear``.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> import tempfile
    >>> from lucid.models import create_model
    >>> from lucid.peft import LoraConfig, XLoraConfig, XLoraModel, get_peft_model
    >>> arch = dict(vocab_size=32, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
    ...             intermediate_size=32, max_position_embeddings=16)
    >>> with tempfile.TemporaryDirectory() as d:
    ...     for name in ("a", "b"):
    ...         get_peft_model(create_model("gpt2_small_lm", **arch),
    ...                        LoraConfig(r=2, target_modules=["c_attn"])).save_pretrained(f"{d}/{name}")
    ...     lm = create_model("gpt2_small_lm", **arch)
    ...     xlora = XLoraModel(lm, XLoraConfig(adapters={"a": f"{d}/a", "b": f"{d}/b"}), "default")
    ...     out = xlora(lucid.randint(0, 32, (1, 5)))
    >>> out.logits.shape, xlora.get_latest_scalings().shape   # (batch, tokens, layers, experts)
    ((1, 5, 32), (1, 5, 2, 2))
    """

    _config_class: ClassVar[type[PeftConfig]] = XLoraConfig

    def __init__(
        self,
        model: nn.Module,
        config: PeftConfig,
        adapter_name: str,
        device: str | None = None,
        ephemeral_gpu_offload: bool = False,
        autocast_adapter_dtype: bool = True,
        **kwargs: object,
    ) -> None:
        """Load the experts into ``model``, build the classifier and hook the model; ``model`` is as given if that fails."""
        refuse_value("XLoraModel", "device", device, None, "experts load onto the model's device")
        refuse_build_options(
            "XLoraModel",
            autocast_adapter_dtype=autocast_adapter_dtype,
            ephemeral_gpu_offload=ephemeral_gpu_offload,
        )
        refuse_kwargs(
            "XLoraModel", kwargs, "the experts are LoRA adapters read from local directories"
        )
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
        if not isinstance(config, XLoraConfig):
            raise TypeError(f"config must be a XLoraConfig, got {type(config).__name__}")
        if "peft_config" in model.__dict__:
            raise ValueError(
                "the model already holds adapters; unload() them before mixing experts into it"
            )
        adapter_name = _check_adapter_name(adapter_name)
        super().__init__(model, config, adapter_name)
        self.model = model
        stored = self._complete(config, model)
        self.peft_config: dict[str, PeftConfig] = {adapter_name: stored}
        self.active_adapter = adapter_name
        self.disabled = False
        self.experts = [str(i) for i in range(len(stored.adapters or {}))]
        #: The scalings the call in progress mixes with, and the last ones the classifier made.
        self.scalings: Tensor | None = None
        self.internal_xlora_scalings: Tensor | None = None
        self._features: Tensor | None = None
        self._handles: list[RemovableHandle] = []
        try:
            # Not a child: the experts' holder wraps the very model this host holds.
            self.__dict__["_experts"] = self._load_experts(model, stored)
            layers = self._expert_layers()
            anchor = next(iter(model.parameters()))
            self.internal_xlora_classifier = XLoraClassifier(
                stored,
                len(self.experts),
                len(layers),
                _adapter_dtype(anchor),  # the classifier trains: float32 beside a half base
                anchor.device.type,
            )
            self._hook(layers)
            self._mark_only_adapters_as_trainable()
        except BaseException:
            self.unload()
            raise

    @property
    def xlora_config(self) -> XLoraConfig:
        """The stored config of the X-LoRA adapter, which the settings below change."""
        config = self.peft_config[self.active_adapters[0]]
        assert isinstance(config, XLoraConfig)  # the only kind admitted
        return config

    @property
    def lora_model(self) -> PeftModel:
        """The LoRA experts' holder (a ``PeftModel`` over the same model): the multi-adapter API they were loaded by."""
        held = self.__dict__.get("_experts")
        assert isinstance(held, PeftModel)
        return held

    # ── building ──────────────────────────────────────────────────────────

    @staticmethod
    def _complete(config: XLoraConfig, model: nn.Module) -> XLoraConfig:
        """A copy of ``config`` with ``hidden_size`` read off the model; the experts must be named."""
        if not config.adapters:
            raise ValueError("X-LoRA mixes LoRA adapters: `adapters` must name at least one")
        stored = copy.deepcopy(config)
        if stored.hidden_size is None:
            stored.hidden_size = _model_size(model, "token_dim")
        return stored

    @staticmethod
    def _load_experts(model: nn.Module, config: XLoraConfig) -> PeftModel:
        """Every expert of ``config`` loaded into ``model`` as a LoRA adapter named by its index, all active."""
        assert config.adapters is not None  # ``_complete`` checked it
        trainable = config.use_trainable_adapters
        paths = list(config.adapters.values())
        held = PeftModel.from_pretrained(model, paths[0], adapter_name="0", is_trainable=trainable)
        try:
            XLoraModel._check_expert(held, "0", paths[0])
            for i, path in enumerate(paths[1:], start=1):
                held.load_adapter(path, str(i), is_trainable=trainable)
                XLoraModel._check_expert(held, str(i), path)
            held.base_model.set_adapter([str(i) for i in range(len(paths))])
        except BaseException:
            held.unload()
            raise
        return held

    @staticmethod
    def _check_expert(held: PeftModel, name: str, path: str) -> None:
        """Raise ``ValueError`` unless the adapter loaded as ``name`` from ``path`` is a plain LoRA one."""
        expert = held.peft_config[name]
        if expert.peft_type != PeftType.LORA:
            raise ValueError(f"X-LoRA mixes LoRA adapters; {path!r} holds a {expert.peft_type} one")
        if getattr(expert, "use_dora", False):
            raise ValueError(f"X-LoRA does not support DoRA experts; {path!r} is one")

    def _expert_layers(self) -> list[tuple[str, Linear]]:
        """The ``nn.Linear`` LoRA layers the experts sit in, in module order."""
        layers: list[tuple[str, Linear]] = []
        for key, layer in _tuner_layers(self.model):
            if not isinstance(layer, LoraLayer):
                continue
            if not isinstance(layer, Linear):
                raise NotImplementedError(
                    f"X-LoRA mixes experts of nn.Linear layers; {key!r} is a "
                    f"{type(layer).__name__} (an expert adapts a layer X-LoRA cannot weigh per token)"
                )
            layers.append((key, layer))
        return layers

    def _hook(self, layers: list[tuple[str, Linear]]) -> None:
        """Switch the expert layers to their base modules and hook them and the model."""
        for number, (_, layer) in enumerate(layers):
            layer.disable_adapters = True  # the hook adds the experts, scaled; the layer must not
            self._handles.append(
                layer.register_forward_hook(_ScaledExperts(self, number), with_kwargs=True)
            )
        self._handles.append(
            self.model.register_forward_pre_hook(self._scale_pass, with_kwargs=True)
        )
        self._handles.append(self.model.register_forward_hook(self._clear_scalings))
        prefix = getattr(self.model, "base_model_prefix", "")
        trunk = getattr(self.model, prefix, None) if isinstance(prefix, str) and prefix else None
        if isinstance(trunk, nn.Module):
            self._handles.append(trunk.register_forward_hook(self._tap_features))

    def _mark_only_adapters_as_trainable(self) -> None:
        train = not self.xlora_config.inference_mode
        for param in self.model.parameters():
            param.requires_grad_(False)
        if train and self.xlora_config.use_trainable_adapters:
            self.lora_model.base_model.set_requires_grad(self.experts, True)
        self.internal_xlora_classifier.requires_grad_(train)

    # ── a call ────────────────────────────────────────────────────────────

    def _tap_features(
        self, module: object, args: tuple[Tensor, ...], output: _ModuleOutput
    ) -> None:
        """Keep the trunk's final hidden states: what the classifier reads."""
        del module, args  # the output is all that matters
        hidden = getattr(output, "last_hidden_state", None)
        if isinstance(hidden, Tensor):
            self._features = hidden

    def _final_hidden(self, output: object) -> Tensor:
        """The final hidden states of the scalings pass: the trunk's (tapped), else the output's own."""
        tapped, self._features = self._features, None
        if tapped is not None:
            return tapped
        for name in ("last_hidden_state",):
            hidden = getattr(output, name, None)
            if isinstance(hidden, Tensor):
                return hidden
        states = getattr(output, "hidden_states", None)
        if isinstance(states, (tuple, list)) and states and isinstance(states[-1], Tensor):
            return states[-1]
        raise ValueError(
            "X-LoRA reads the base model's final hidden states, but neither its trunk nor its "
            "output gives `last_hidden_state` or `hidden_states`"
        )

    def _scale_pass(
        self, module: object, args: tuple[object, ...], kwargs: dict[str, object]
    ) -> None:
        """Before the model runs: weigh the experts for this input, so the real pass can mix them.

        Raises
        ------
        ValueError
            The call carries a cache that already holds tokens, or no ``input_ids`` / ``inputs_embeds``.
        """
        self.scalings = None
        if self.disabled:
            return
        assert isinstance(module, nn.Module)  # a hook is called by its module
        call = _named_arguments(module, args, kwargs)
        cache = call.get("past_key_values")
        if isinstance(cache, Cache) and cache.get_seq_length() > 0:
            raise ValueError(
                "X-LoRA weighs the experts from the whole sequence at every step, which a "
                "cached step does not see: generate with use_cache=False"
            )
        sized = [
            v for v in (call.get("input_ids"), call.get("inputs_embeds")) if isinstance(v, Tensor)
        ]
        if not sized:
            raise ValueError("X-LoRA needs input_ids or inputs_embeds to size the scalings")
        like = sized[0]
        classifier = self.internal_xlora_classifier
        batch, tokens = int(like.shape[0]), int(like.shape[1])
        self.scalings = classifier.dummy_scalings(batch, tokens, like.device.type)
        probe = {key: value for key, value in call.items() if key not in _NOT_IN_THE_PASS}
        if "use_cache" in inspect.signature(module.forward).parameters:
            probe["use_cache"] = False
        try:
            with lucid.no_grad():
                output = module.forward(**probe)
            self.scalings = classifier(self._final_hidden(output))
        except BaseException:
            self.scalings = None
            raise
        self.internal_xlora_scalings = self.scalings

    def _clear_scalings(
        self, module: object, args: tuple[Tensor, ...], output: _ModuleOutput
    ) -> None:
        """After the model ran: the layers go back to running their base alone."""
        del module, args, output
        self.scalings = None

    # ── settings of the mixture ───────────────────────────────────────────

    def set_topk_lora(self, value: int | None) -> None:
        """Keep only the ``value`` heaviest experts per token (``None``: all of them)."""
        self.xlora_config.top_k_lora = value

    def set_global_scaling_weight(self, weight: float) -> None:
        """Multiply every expert's update by ``weight``."""
        self.xlora_config.global_scaling_weight = weight

    def get_global_scaling_weight(self) -> float:
        """The factor on every expert's update."""
        return self.xlora_config.global_scaling_weight

    def set_scaling_pass_value(self, value: float | None) -> None:
        """Hold every expert at ``value`` in the scalings pass (``None``: one over the number of experts)."""
        self.xlora_config.scaling_pass_value = 1 / len(self.experts) if value is None else value

    def get_latest_scalings(self) -> Tensor | None:
        """The last scalings the classifier made, ``(batch, tokens, layers, experts)``; ``None`` before the first call."""
        return self.internal_xlora_scalings

    def get_scalings_log(self) -> list[Tensor]:
        """A copy of the list of logged scalings (the tensors are not copied)."""
        return list(self.internal_xlora_classifier.log_scalings)

    def enable_scalings_logging(self) -> None:
        """Record the scalings of every call from now on."""
        self.internal_xlora_classifier.scalings_logging = True

    def disable_scalings_logging(self) -> None:
        """Stop recording; the log is kept."""
        self.internal_xlora_classifier.scalings_logging = False

    def clear_scalings_log(self) -> None:
        """Forget the logged scalings."""
        self.internal_xlora_classifier.log_scalings.clear()

    def get_bucketed_scalings_log(self) -> dict[int, tuple[list[int], list[Tensor]]]:
        """The log grouped by sequence length: ``{tokens: (positions in the log, scalings)}``."""
        buckets: dict[int, tuple[list[int], list[Tensor]]] = {}
        for position, scalings in enumerate(self.internal_xlora_classifier.log_scalings):
            places, held = buckets.setdefault(int(scalings.shape[1]), ([], []))
            places.append(position)
            held.append(scalings)
        return buckets

    # ── the host's contract ───────────────────────────────────────────────

    @property
    @override
    def active_adapters(self) -> list[str]:
        """The X-LoRA adapter, as a list."""
        return _as_names(self.active_adapter)

    @override
    def add_adapter(self, adapter_name: str, config: PeftConfig) -> None:
        """Refused: an X-LoRA model holds one classifier; build another model to mix other experts."""
        del adapter_name, config
        raise NotImplementedError(
            "an X-LoRA model holds one adapter; the experts it mixes are managed through "
            "`lora_model` (the multi-adapter API of the LoRA adapters)"
        )

    @override
    def set_adapter(self, adapter_name: str | list[str], inference_mode: bool = False) -> None:
        """Select the X-LoRA adapter — the only one — and train its classifier unless ``inference_mode``."""
        names = _as_names(adapter_name)
        if names != self.active_adapters:
            raise ValueError(f"Adapter {names} not found; the model holds {self.active_adapters}.")
        self.internal_xlora_classifier.requires_grad_(not inference_mode)

    @override
    def set_requires_grad(
        self, adapter_names: str | Sequence[str], requires_grad: bool = True
    ) -> None:
        """Enable or disable gradients on the classifier."""
        if _as_names(adapter_names) != self.active_adapters:
            raise ValueError(f"Adapter {adapter_names} not found.")
        self.internal_xlora_classifier.requires_grad_(requires_grad)

    @override
    def delete_adapter(self, adapter_name: str) -> None:
        """Refused: the classifier is the model's only adapter; ``unload()`` removes it."""
        del adapter_name
        raise NotImplementedError("an X-LoRA model cannot lose its one adapter; unload() it")

    @override
    def merge_and_unload(
        self,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names: list[str] | None = None,
    ) -> nn.Module:
        """Refused: the experts' weights change with every token, so there is nothing to merge."""
        del progressbar, safe_merge, adapter_names
        raise NotImplementedError(
            "X-LoRA weighs its experts per token, so they cannot be folded into the weights; "
            "unload() removes them"
        )

    @override
    def unload(self) -> nn.Module:
        """Take the hooks and the experts out and give every parameter its old ``requires_grad``."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        held = self.__dict__.pop("_experts", None)
        if isinstance(held, PeftModel):
            held.unload()
        self.scalings = None
        return self.model

    @override
    def enable_adapter_layers(self) -> None:
        """Weigh the experts again after ``disable_adapter_layers``."""
        self.disabled = False

    @override
    def disable_adapter_layers(self) -> None:
        """Run the base model alone."""
        self.disabled = True

    @override
    def _mixed_batch(
        self, adapter_names: Sequence[str], num_beams: int = 1
    ) -> AbstractContextManager[None]:
        del adapter_names, num_beams
        raise NotImplementedError(
            "X-LoRA weighs the experts per token, not per adapter name; there is nothing to route"
        )

    @classmethod
    @override
    def _adapt_in_place(
        cls,
        model: nn.Module,
        config: PeftConfig,
        adapter_name: str,
        low_cpu_mem_usage: bool = False,
        state_dict: dict[str, Tensor] | None = None,
    ) -> None:
        raise TypeError(
            f"{cls.__name__} keeps its classifier beside the model, which a bare model has no "
            "place for; adapt with get_peft_model"
        )

    @classmethod
    @override
    def _extra_trainable_names(cls, model: nn.Module, config: PeftConfig) -> list[str]:
        return []

    @override
    def forward(self, *args: object, **kwargs: object) -> object:
        """Call the model; the hooks weigh and mix the experts."""
        return self.model(*args, **kwargs)

    # ── the adapter file: the classifier ──────────────────────────────────

    @classmethod
    @override
    def _shared_keys(cls, context: _FileContext) -> dict[str, str]:
        model = context.model
        holder = model if isinstance(model, XLoraModel) else model.base_model
        assert isinstance(holder, XLoraModel)
        return {
            f"{context.host}{_CLASSIFIER}.{key}": f"{context.host}{_CLASSIFIER}.{key}"
            for key in holder.internal_xlora_classifier.state_dict()
        }
