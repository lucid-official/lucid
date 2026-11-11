"""The X-LoRA classifier: from the base model's final hidden states to one weight per expert (private)."""

from typing import override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid._types import DTypeLike
from lucid.peft.tuners.xlora._config import XLoraConfig


class XLoraClassifier(nn.Module):
    r"""Predicts, for each token, how much each LoRA expert should weigh.

    A stack of ``xlora_depth`` linear layers (ReLU and dropout between them) over the hidden
    states, ``(batch, tokens, hidden_size)`` to ``(batch, tokens, n_classes)`` — or
    ``n_classes * n_layers`` outputs with ``layerwise_scalings``, spread over the adapted layers.
    The result is ``(batch, tokens, n_layers, n_classes)``, softmax-normalised over the experts
    (at ``softmax_temperature``) unless ``enable_softmax`` is off.  It reads its settings from
    the config it is given, so changing the config's ``softmax_temperature`` or
    ``scaling_pass_value`` takes effect at the next call.

    Parameters
    ----------
    config : XLoraConfig
        The classifier's shape (``hidden_size`` set) and behaviour.
    n_classes : int
        How many experts it weighs.
    n_layers : int
        How many adapted layers the weights are for.
    dtype : lucid.dtype
        The dtype of its parameters and of the weights it returns (the adapter's: the base
        model's, float32 for a half-precision one).
    device : str
        The device it lives on (the base model's).

    Attributes
    ----------
    layers : nn.Sequential
        The linear layers; their keys (``layers.0.weight`` ...) are the adapter file's.
    log_scalings : list of Tensor
        The scalings of each call, while ``scalings_logging`` is on.

    Examples
    --------
    >>> import lucid
    >>> from lucid.peft import XLoraConfig
    >>> from lucid.peft.tuners.xlora._classifier import XLoraClassifier
    >>> config = XLoraConfig(hidden_size=8, adapters={"a": "x", "b": "y"})
    >>> classifier = XLoraClassifier(config, n_classes=2, n_layers=3, dtype=lucid.float32, device="cpu")
    >>> classifier(lucid.randn(1, 5, 8)).shape
    (1, 5, 3, 2)
    """

    def __init__(
        self, config: XLoraConfig, n_classes: int, n_layers: int, dtype: DTypeLike, device: str
    ) -> None:
        """Build the layers for ``config`` and move them to ``device`` as ``dtype``."""
        super().__init__()
        assert config.hidden_size is not None  # the tuner read it off the model
        self.config = config
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.log_scalings: list[Tensor] = []
        self.scalings_logging = False
        outputs = n_classes * n_layers if config.layerwise_scalings else n_classes
        layers: list[nn.Module] = []
        width = config.hidden_size
        # Each hidden layer is Linear, ReLU and, when the config asks, Dropout: the file's keys count them.
        for _ in range(config.xlora_depth - 1):
            layers.append(nn.Linear(width, config.xlora_size, bias=True))
            layers.append(nn.ReLU())
            if config.xlora_dropout_p > 0.0:
                layers.append(nn.Dropout(p=config.xlora_dropout_p))
            width = config.xlora_size
        layers.append(nn.Linear(width, outputs, bias=True))
        self.layers = nn.Sequential(*layers).to(device).to(dtype)
        self._dtype = dtype

    def dummy_scalings(self, batch: int, tokens: int, device: str) -> Tensor:
        """The scalings of the scalings pass: ``scaling_pass_value`` for every token, layer and expert."""
        shape = (batch, tokens, self.n_layers, self.n_classes)
        return lucid.full(shape, self.config.scaling_pass_value, dtype=self._dtype, device=device)

    @override
    def forward(self, hidden: Tensor) -> Tensor:
        """The ``(batch, tokens, n_layers, n_classes)`` weights of the experts for ``hidden``."""
        batch, tokens = int(hidden.shape[0]), int(hidden.shape[1])
        logits = self.layers(hidden.to(self._dtype))
        if not self.config.layerwise_scalings:
            # One set of weights, shared by every adapted layer.
            logits = logits.unsqueeze(2).expand(batch, tokens, self.n_layers, self.n_classes)
        scalings = logits.reshape(batch, tokens, self.n_layers, self.n_classes)
        if self.config.enable_softmax:
            scalings = F.softmax(scalings / self.config.softmax_temperature, dim=-1)
        if self.scalings_logging:
            self.log_scalings.append(scalings)
        return scalings


def _layer_scalings(config: XLoraConfig, scalings: Tensor, layer: int) -> Tensor:
    """The weights of one adapted layer, ``(batch, tokens, n_classes)``, with ``top_k_lora`` and its softmax applied."""
    weights = scalings[:, :, layer, :]
    if config.top_k_lora is not None:
        kept = weights.topk(config.top_k_lora, dim=-1).indices
        mask = F.one_hot(kept, int(weights.shape[-1])).sum(dim=-2) > 0
        weights = weights * mask.to(weights.dtype)
    if config.enable_softmax_topk:
        nonzero = weights != 0
        sharpened = F.softmax(weights.masked_fill(~nonzero, float("-inf")), dim=-1)
        weights = sharpened.masked_fill(~nonzero, 0.0)
    return weights
