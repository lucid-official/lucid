r"""The adaption-prompt layer: a frozen attention module beside learned prompts it attends to through a zero-init gate.

An adapter holds ``adaption_prompt`` (``(1, adapter_len, width)``, starting as standard normal
noise) and ``adaption_gate`` (``(1,)``, starting at zero).  The layer runs the wrapped attention,
then lets the same queries attend to the keys and values the wrapped module's projections make of
the prompt, and adds that, scaled by the gate, to the output:
:math:`y = f(x) + g\cdot \mathrm{softmax}(q K_p^\top / \sqrt{d})\,V_p`.  With the gate at zero the
adapted layer is the base.  A state dict reads ``....adaption_prompt.default`` and an adapter file
``....adaption_prompt``, the reference's key.
"""

import math
from typing import ClassVar, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter
from lucid.peft._config import _TargetedConfig
from lucid.peft.tuners._layer import _parameter_at
from lucid.peft.tuners._support import _adapter_dtype, _config_as, _ConfigLayer, _Unmergeable
from lucid.peft.tuners.adaption_prompt._attention import _AttentionKind
from lucid.peft.tuners.adaption_prompt._config import AdaptionPromptConfig


class AdaptedAttention(_ConfigLayer, _Unmergeable, family=True, base_type=nn.Module):
    r"""An attention module that also attends to learned prompts: :math:`y = f(x) + g\,A_p(x)`.

    The layer applies the one active adapter it holds; with none, or when adapters are disabled,
    it runs the wrapped attention alone.  The wrapped module's arguments pass through unchanged,
    so it sits wherever the attention did.

    Parameters
    ----------
    base_layer : nn.Module
        The attention module to adapt; kept as ``base_layer``, its parameters shared.
    adapter_name : str
        Name of the adapter created at once.
    config : AdaptionPromptConfig
        The adapter's config; ``adapter_len`` is this layer's prompt length.

    Raises
    ------
    TypeError
        ``base_layer`` is not an attention module of a supported family.
    ValueError
        The config names no ``adapter_len``, or ``base_layer`` builds parameters on its first input.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.text.gpt2 import GPT2Config, GPT2Model
    >>> from lucid.peft import AdaptionPromptConfig
    >>> from lucid.peft.tuners.adaption_prompt import AdaptedAttention
    >>> gpt = GPT2Model(GPT2Config(num_hidden_layers=1, hidden_size=16, num_attention_heads=2,
    ...                            intermediate_size=32, vocab_size=20)).eval()
    >>> attn = gpt.h[0].attn
    >>> layer = AdaptedAttention(attn, "default", AdaptionPromptConfig(adapter_len=4, adapter_layers=1))
    >>> tuple(layer.adaption_prompt["default"].shape), float(layer.adaption_gate["default"].item())
    ((1, 4, 16), 0.0)
    >>> x = lucid.randn(2, 5, 16)
    >>> bool((layer(x) == attn(x)).all().item())   # the gate starts at zero
    True
    """

    adapter_layer_names: ClassVar[tuple[str, ...]] = ("adaption_prompt", "adaption_gate")
    _repr_prefix = "adaption_prompt."
    _unmergeable_because = "an adaption prompt"
    adaption_prompt: nn.ParameterDict
    adaption_gate: nn.ParameterDict

    @property
    def _kind(self) -> type[_AttentionKind]:
        return _AttentionKind.of(self.base_layer)

    @override
    def _enter(self, adapter_name: str, config: _TargetedConfig) -> None:
        cfg = _config_as(AdaptionPromptConfig, config)
        if cfg.adapter_len is None:
            raise ValueError("`adapter_len` must be set to build an adaption prompt")
        kind = self._kind
        anchor = kind.anchor(self.base_layer)
        dtype = _adapter_dtype(anchor)
        # A standard-normal start, as the reference's (and the LLaMA-Adapter authors') embedding.
        prompt = lucid.randn(
            1, cfg.adapter_len, kind.width(self.base_layer), device=anchor.device, dtype=dtype
        )
        gate = lucid.zeros(1, device=anchor.device, dtype=dtype)
        self.adaption_prompt[adapter_name] = Parameter(prompt)
        self.adaption_gate[adapter_name] = Parameter(gate)

    @override
    def _check_activation(self, adapter_names: list[str]) -> None:
        if len(adapter_names) > 1:
            raise ValueError(
                f"an adaption prompt applies one adapter at a time, got {adapter_names}"
            )

    def _attended(self, adapter: str, hidden: Tensor) -> Tensor:
        """What the layer's queries make of the adapter's prompt: the gated attention over it."""
        kind, attention = self._kind, self.base_layer
        num_heads, head_dim = kind.heads(attention)
        prompt = _parameter_at(self.adaption_prompt, adapter)
        length = int(prompt.shape[1])
        key, value = kind.keys_values(attention, prompt)
        key = key.reshape(1, length, num_heads, head_dim).transpose(1, 2)
        value = value.reshape(1, length, num_heads, head_dim).transpose(1, 2)
        batch, tokens = int(hidden.shape[0]), int(hidden.shape[1])
        query = kind.queries(attention, hidden).reshape(batch, tokens, num_heads, head_dim)
        query = query.transpose(1, 2)
        scores = (query @ key.to(query.dtype).transpose(2, 3)) / math.sqrt(head_dim)
        # The softmax runs in float32, as the reference does, whatever the model's dtype.
        weights = F.softmax(scores.float(), dim=-1).to(query.dtype)
        gated = _parameter_at(self.adaption_gate, adapter) * weights
        return (gated @ value.to(query.dtype)).transpose(1, 2).reshape(batch, tokens, -1)

    @override
    def forward(self, hidden: Tensor, *args: object, **kwargs: object) -> Tensor:
        """The wrapped attention's output plus the gated attention over the active adapter's prompt.

        The wrapped attention alone when adapters are disabled or none applies; ``hidden`` is its
        first argument and the rest go to it as given.
        """
        result: Tensor = self.base_layer(hidden, *args, **kwargs)
        held = [name for name in self.active_adapters if name in self.adaption_prompt]
        if self.disable_adapters or not held:
            return result
        added = self._attended(held[0], hidden)
        return result + added.to(result.dtype)
