"""``CartridgeEncoder``: the trained key-value cache of a Cartridge, and ``CartridgeModel``, its tuner."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import ClassVar, Self, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.nn import init
from lucid.nn.parameter import Parameter
from lucid.peft._config import PeftType, PromptLearningConfig
from lucid.peft.tuners._files import _FileContext
from lucid.peft.tuners._prompt._encoder import _Injection, _PromptEncoder
from lucid.peft.tuners._prompt._tuner import _PROMPT_KEY, _PromptLearningTuner
from lucid.peft.tuners.cartridge._config import CartridgeConfig

#: The file key the frozen tokens travel under between the encoder and the file, never written: the file holds the whole cache under ``prompt_embeddings``.
_FROZEN_KEY = "frozen_embedding"


class CartridgeEncoder(_PromptEncoder):
    r"""A cache of keys and values that is trained, as a matrix of rows (one per virtual token) of every layer's key and value.

    A row holds ``2 * num_layers * token_dim`` numbers laid out as prefix tuning's are, the key and the
    value of each layer one after another, so the cache goes in front of the model's own like a
    prefix's.  The first ``num_frozen_tokens`` rows are stored as a parameter that is never trained
    (the Cartridges paper recommends freezing the first token as an attention sink), the rest as the
    trained one.  Both start as a standard normal draw, which the reference's embedding table also
    starts as; ``load_prompt_embeddings`` replaces them, from the first keys and values of a document
    the model has read (``initialize_kv_prefix_from_text``), say.

    Parameters
    ----------
    config : CartridgeConfig
        ``num_virtual_tokens``, ``num_layers``, ``token_dim`` (all set; the tuner fills them in from the
        model), ``num_frozen_tokens`` and ``inference_mode``.

    Attributes
    ----------
    frozen_embedding : Parameter or None
        ``(num_frozen_tokens, hidden)`` and never trained; ``None`` when no token is frozen.
    trainable_embedding : Parameter
        ``(num_virtual_tokens - num_frozen_tokens, hidden)``, trained unless ``inference_mode``.

    Raises
    ------
    ValueError
        ``num_frozen_tokens`` is outside ``[0, num_virtual_tokens]``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.peft import CartridgeConfig, CartridgeEncoder
    >>> config = CartridgeConfig(num_virtual_tokens=5, num_frozen_tokens=2, num_layers=3, token_dim=4)
    >>> encoder = CartridgeEncoder(config)
    >>> tuple(encoder.frozen_embedding.shape), tuple(encoder.trainable_embedding.shape)
    ((2, 24), (3, 24))
    >>> encoder(lucid.zeros(2, 5).long()).shape
    (2, 5, 24)
    >>> encoder.frozen_embedding.requires_grad, encoder.trainable_embedding.requires_grad
    (False, True)
    """

    injection: ClassVar[_Injection] = _Injection.KEY_VALUES
    #: The encoder's own tensors by the file's key: the trained rows are the file's ``prompt_embeddings`` (the frozen rows join them on the way out).
    stored: ClassVar[Mapping[str, str]] = MappingProxyType(
        {"trainable_embedding": _PROMPT_KEY, "frozen_embedding": _FROZEN_KEY}
    )

    frozen_embedding: Parameter | None
    trainable_embedding: Parameter

    def __init__(self, config: CartridgeConfig) -> None:
        """Build the frozen and the trained rows of a cache ``config`` sizes, and draw them."""
        super().__init__()
        self.config = config
        assert config.num_virtual_tokens is not None  # the tuner resolved it
        assert config.num_layers is not None and config.token_dim is not None  # and the sizes
        width = config.num_layers * 2 * config.token_dim
        frozen = int(config.num_frozen_tokens)
        if frozen < 0 or frozen > config.num_virtual_tokens:
            raise ValueError(
                f"`num_frozen_tokens` must be in [0, num_virtual_tokens], got {frozen} for "
                f"num_virtual_tokens={config.num_virtual_tokens}."
            )
        self.num_frozen_tokens = frozen
        self.num_trainable_tokens = config.num_virtual_tokens - frozen
        self.frozen_embedding = (
            Parameter(lucid.empty(frozen, width), requires_grad=False) if frozen else None
        )
        self.trainable_embedding = Parameter(
            lucid.empty(self.num_trainable_tokens, width), requires_grad=not config.inference_mode
        )
        self.reset_parameters()

    @classmethod
    @override
    def from_config(cls, config: PromptLearningConfig, word_embeddings: nn.Module | None) -> Self:
        """The encoder for ``config``; ``word_embeddings`` is not read."""
        del word_embeddings  # the cache is not made of token embeddings
        assert isinstance(config, CartridgeConfig)
        return cls(config)

    @property
    @override
    def stores_prompt(self) -> bool:
        """``True``: the parameters are the cache itself, nothing maps them."""
        return True

    @property
    @override
    def num_embeddings(self) -> int:
        """How many virtual tokens the cache holds: the frozen rows and the trained ones."""
        return self.num_frozen_tokens + self.num_trainable_tokens

    @override
    def indices(self) -> Tensor:
        """The virtual-token ids ``(1, num_embeddings)`` the encoder is called with."""
        ids = lucid.arange(self.num_embeddings, device=self.trainable_embedding.device.type)
        return ids.long().reshape(1, self.num_embeddings)

    @property
    def weight(self) -> Tensor:
        """The whole cache ``(num_virtual_tokens, hidden)``: the frozen rows, then the trained ones."""
        if self.frozen_embedding is None:
            return self.trainable_embedding
        return lucid.cat([self.frozen_embedding, self.trainable_embedding], dim=0)

    def reset_parameters(self) -> None:
        """Draw every row again from a standard normal."""
        with lucid.no_grad():
            if self.frozen_embedding is not None:
                init.normal_(self.frozen_embedding)
            init.normal_(self.trainable_embedding)

    @override
    def requires_grad_(self, requires_grad: bool = True) -> Self:
        """Train (or freeze) the trained rows; the frozen rows are never trained.

        The tuner switches whole encoders on and off with this call, and a plain
        ``Module.requires_grad_`` would make the frozen tokens trainable with the rest.
        """
        self.trainable_embedding.requires_grad_(requires_grad)
        return self

    def _check_prompt(self, prompt: Tensor) -> None:
        """Raise ``ValueError`` unless ``prompt`` is a whole cache, ``(num_virtual_tokens, hidden)``."""
        expected = (self.num_embeddings, int(self.trainable_embedding.shape[1]))
        if prompt.ndim != 2 or tuple(prompt.shape) != expected:
            raise ValueError(
                f"Invalid `prompt_embeddings` shape. Expected ({expected[0]}, hidden), "
                f"got {tuple(prompt.shape)}."
            )

    def load_prompt_embeddings(self, prompt_embeddings: Tensor) -> None:
        """Copy a whole cache ``(num_virtual_tokens, hidden)`` in, the first rows into the frozen ones.

        Parameters
        ----------
        prompt_embeddings : Tensor
            The flattened cache an adapter file holds.

        Raises
        ------
        ValueError
            The shape is not ``(num_virtual_tokens, hidden)``.
        """
        self._check_prompt(prompt_embeddings)
        with lucid.no_grad():
            if self.frozen_embedding is not None:
                self.frozen_embedding.copy_(prompt_embeddings[: self.num_frozen_tokens])
            self.trainable_embedding.copy_(prompt_embeddings[self.num_frozen_tokens :])

    @override
    def forward(self, prefix_tokens: Tensor) -> Tensor:
        """The cache for each row of a batch: ``(batch, ...)`` ids to ``(batch, num_virtual_tokens, hidden)``.

        The ids are ignored; they are there so a cache is called as every prompt is.
        """
        batch = int(prefix_tokens.shape[0])
        cache: Tensor = self.weight.unsqueeze(0).expand(batch, -1, -1)
        return cache


class CartridgeModel(_PromptLearningTuner, peft_type=PeftType.CARTRIDGE):
    """The tuner of Cartridge: a ``CartridgeEncoder`` per adapter, read as every layer's cached keys and values.

    The adapter file holds the cache whole as ``prompt_embeddings``, frozen rows first; in memory the
    frozen and the trained rows are two parameters, so loading splits the file's tensor and saving
    joins them.
    """

    _config_class: ClassVar[type[PromptLearningConfig]] = CartridgeConfig
    _encoder_class: ClassVar[type[_PromptEncoder]] = CartridgeEncoder

    @classmethod
    @override
    def _shared_keys(cls, context: _FileContext) -> dict[str, str]:
        encoder = cls._holder(context).prompt_encoder[context.adapter]
        assert isinstance(encoder, CartridgeEncoder)
        held = f"{context.host}prompt_encoder.{context.adapter}."
        return {
            held + own: disk
            for own, disk in CartridgeEncoder.stored.items()
            if getattr(encoder, own, None) is not None
        }

    @classmethod
    @override
    def _to_file(cls, context: _FileContext, state: dict[str, Tensor]) -> dict[str, Tensor]:
        """The cache whole under ``prompt_embeddings``: the frozen rows, then the trained ones."""
        frozen = state.pop(_FROZEN_KEY, None)
        if frozen is not None:
            state[_PROMPT_KEY] = lucid.cat([frozen, state[_PROMPT_KEY]], dim=0)
        return super()._to_file(context, state)

    @classmethod
    @override
    def _from_file(cls, context: _FileContext, state: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """The file's cache split into the frozen rows and the trained ones, after its shape is checked."""
        encoder = cls._holder(context).prompt_encoder[context.adapter]
        assert isinstance(encoder, CartridgeEncoder)
        split = dict(state)
        prompt = split.pop(_PROMPT_KEY, None)
        if prompt is not None:
            encoder._check_prompt(prompt)
            if encoder.num_frozen_tokens:
                split[_FROZEN_KEY] = prompt[: encoder.num_frozen_tokens]
            split[_PROMPT_KEY] = prompt[encoder.num_frozen_tokens :]
        return super()._from_file(context, split)
