"""Modules built by hand, for the classes a signature cannot describe.

The constructor ladder reads a class's own signature and fills each
parameter from its name or annotation.  That is right for a layer taking
sizes and has nothing to say about a class whose arguments are *other
objects with a relationship*: ``LoraModel(model, config)`` needs a model
that holds an ``nn.Linear`` and a config that names it, and
``lora.Linear(base_layer, adapter_name, r=0)`` declares a default
(``r=0``) that the class then refuses.  Guessing a value per parameter
name would put a ``model`` entry in a table every other class reads.

A builder returns a **fresh** instance every call: ``PeftModel`` rewrites
the model it is handed, so a shared one would arrive adapted.  Classes
are keyed by identity because ``Linear`` is also ``nn.Linear`` and four
quantised layers.
"""

import inspect
from collections.abc import Callable, Iterator
from typing import Any

import lucid
import lucid.nn as nn
import lucid.peft as peft
from lucid.models import create_model
from lucid.peft import (
    BOFTConfig,
    FourierFTConfig,
    HRAConfig,
    IA3Config,
    IA3Model,
    LNTuningConfig,
    LoHaConfig,
    LoHaModel,
    LoKrConfig,
    LoKrModel,
    LoraConfig,
    LoraModel,
    OFTConfig,
    PeftConfig,
    PeftModel,
    PeftType,
    PrefixEncoder,
    PromptEmbedding,
    PromptEncoder,
)
from lucid.peft.tuners import lora
from lucid.peft.tuners._base import _TUNERS, _BaseTuner, _BaseTunerLayer
from lucid.peft.tuners.adalora import AdaLoraConfig, AdaLoraLayer, AdaLoraModel
from lucid.peft.tuners.ia3 import IA3Layer
from lucid.peft.tuners.loha import LoHaLayer
from lucid.peft.tuners.lokr import LoKrLayer
from lucid.peft.tuners.p_tuning import PromptEncoderConfig, PromptEncoderModel
from lucid.peft.tuners.prefix_tuning import PrefixTuningConfig, PrefixTuningModel
from lucid.peft.tuners.prompt_tuning import PromptTuningConfig, PromptTuningModel
from lucid.peft.tuners.vera import VeraConfig, VeraLayer, VeraModel


def _base_model() -> nn.Module:
    """Two linear layers around a ReLU, at the audit's usual width of 4."""
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))


def _text_model() -> nn.Module:
    """A one-layer GPT-2 language model, the smallest thing a prompt can be put in front of.

    Untied: a tied head is one ``Parameter`` under two names, which the module axis's
    state-dict count reads as a submodule registered twice.
    """
    return create_model(
        "gpt2_small_lm",
        vocab_size=16,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=8,
        tie_word_embeddings=False,
    )


#: The sizes a prompt encoder reads off the model, spelled out for a table of width 4.
_PROMPT_SIZES: dict[str, int] = {
    "token_dim": 4,
    "num_layers": 1,
    "num_attention_heads": 2,
    "num_transformer_submodules": 1,
}


def _config() -> LoraConfig:
    return LoraConfig(r=2, target_modules=["0", "2"])


#: A small module of each kind a tuner layer can wrap (its ``base_type=``).
_BASES: dict[type[nn.Module], Callable[[], nn.Module]] = {
    nn.Linear: lambda: nn.Linear(4, 3),
    nn.Conv1d: lambda: nn.Conv1d(2, 3, 3),
    nn.Conv2d: lambda: nn.Conv2d(2, 3, 3),
    nn.Conv3d: lambda: nn.Conv3d(2, 3, 3),
    nn.ConvTranspose1d: lambda: nn.ConvTranspose1d(2, 3, 3),
    nn.ConvTranspose2d: lambda: nn.ConvTranspose2d(2, 3, 3),
    nn.ConvTranspose3d: lambda: nn.ConvTranspose3d(2, 3, 3),
    nn.Embedding: lambda: nn.Embedding(5, 4),
    nn.MultiheadAttention: lambda: nn.MultiheadAttention(4, 2),
    # A layer that wraps any module (LN tuning) is built on a normalisation layer.
    nn.Module: lambda: nn.LayerNorm(4),
}

#: A small config of each method that is built from ``(module, adapter, config)`` alone, by its
#: ``PeftType``: it adapts the audit's two-linear model at ``0`` and ``2``, and fits any base ``_BASES`` holds.
#: A new method of that shape adds its row here and nowhere else: its tuner's and its layers'
#: builders follow from the registry below.
_EXAMPLE_CONFIGS: dict[PeftType, Callable[[], PeftConfig]] = {
    PeftType.OFT: lambda: OFTConfig(oft_block_size=2, target_modules=["0", "2"]),
    PeftType.BOFT: lambda: BOFTConfig(boft_block_size=2, target_modules=["0", "2"]),
    PeftType.HRA: lambda: HRAConfig(r=2, target_modules=["0", "2"]),
    PeftType.FOURIERFT: lambda: FourierFTConfig(n_frequency=4, target_modules=["0", "2"]),
    PeftType.LN_TUNING: lambda: LNTuningConfig(target_modules=["0", "2"]),
}


def _example_config(layer: type[_BaseTunerLayer]) -> Callable[[], PeftConfig] | None:
    """The example config of the method whose layer family ``layer`` belongs to, if the table has one."""
    for peft_type, tuner in _TUNERS.items():
        if issubclass(tuner, _BaseTuner) and tuner._layer_types is layer._layer_types:
            return _EXAMPLE_CONFIGS.get(peft_type)
    return None

#: The config a layer family takes as its third argument, where its constructor wants one.
_LAYER_CONFIGS: dict[type[_BaseTunerLayer], Callable[[], PeftConfig]] = {
    IA3Layer: lambda: IA3Config(target_modules=["x"], feedforward_modules=[]),
    LoHaLayer: lambda: LoHaConfig(target_modules=["x"]),
    LoKrLayer: lambda: LoKrConfig(target_modules=["x"]),
    AdaLoraLayer: lambda: AdaLoraConfig(total_step=10, target_modules=["x"]),
    VeraLayer: lambda: VeraConfig(target_modules=["x"]),
}


def _layer_types() -> Iterator[tuple[type[nn.Module], type[_BaseTunerLayer]]]:
    """Every registered ``(base_type, layer class)`` pair, over all tuner families."""
    seen: set[type] = set()
    stack: list[type[_BaseTunerLayer]] = [_BaseTunerLayer]
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        for base, layer in cls.__dict__.get("_layer_types", {}).items():
            if layer not in seen:
                seen.add(layer)
                yield base, layer


def _layer_builder(base: type[nn.Module], layer: type[_BaseTunerLayer]) -> Callable[[], Any]:
    """Wrap a fresh ``base`` module in ``layer`` with a rank-2 adapter, giving it what it takes."""
    parameters = inspect.signature(layer).parameters
    keywords = {name: 2 for name in ("r", "alpha") if name in parameters}
    # A layer taking ``**kwargs`` (lora.MultiheadAttention) forwards the rank to its base class.
    if "r" not in keywords and any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
        keywords["r"] = 2
    make_config = next(
        (m for root, m in _LAYER_CONFIGS.items() if issubclass(layer, root)), _example_config(layer)
    )
    wants_config = "config" in parameters and make_config is not None

    def build() -> Any:
        positional = [make_config()] if wants_config and make_config is not None else []
        return layer(_BASES[base](), "default", *positional, **keywords)

    return build


class _Gate(nn.Module):
    """A module holding a bare 2-D parameter, the kind ``target_parameters`` adapts."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(lucid.randn(3, 4))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return x @ self.w.mT


#: Class -> a zero-argument builder of one instance.
BUILDERS: dict[type, Callable[[], Any]] = {
    LoraModel: lambda: LoraModel(_base_model(), _config()),
    IA3Model: lambda: IA3Model(
        _base_model(), IA3Config(target_modules=["0", "2"], feedforward_modules=["2"])
    ),
    LoHaModel: lambda: LoHaModel(_base_model(), LoHaConfig(r=2, target_modules=["0", "2"])),
    LoKrModel: lambda: LoKrModel(_base_model(), LoKrConfig(r=2, target_modules=["0", "2"])),
    AdaLoraModel: lambda: AdaLoraModel(
        _base_model(), AdaLoraConfig(init_r=2, total_step=10, target_modules=["0", "2"])
    ),
    VeraModel: lambda: VeraModel(_base_model(), VeraConfig(r=2, target_modules=["0", "2"])),
    PeftModel: lambda: PeftModel(_base_model(), _config()),
    PromptTuningModel: lambda: PromptTuningModel(
        _text_model(), PromptTuningConfig(num_virtual_tokens=2, task_type="CAUSAL_LM")
    ),
    PrefixTuningModel: lambda: PrefixTuningModel(
        _text_model(), PrefixTuningConfig(num_virtual_tokens=2, task_type="CAUSAL_LM")
    ),
    PromptEncoderModel: lambda: PromptEncoderModel(
        _text_model(), PromptEncoderConfig(num_virtual_tokens=2, task_type="CAUSAL_LM")
    ),
    PromptEmbedding: lambda: PromptEmbedding(
        PromptTuningConfig(num_virtual_tokens=2, **_PROMPT_SIZES), nn.Embedding(5, 4)
    ),
    PrefixEncoder: lambda: PrefixEncoder(
        PrefixTuningConfig(num_virtual_tokens=2, encoder_hidden_size=4, **_PROMPT_SIZES)
    ),
    PromptEncoder: lambda: PromptEncoder(
        PromptEncoderConfig(num_virtual_tokens=2, encoder_hidden_size=4, **_PROMPT_SIZES)
    ),
    lora.ParamWrapper: lambda: lora.ParamWrapper(_Gate(), "default", "w", r=2),
}

# Every tuner layer registered by ``base_type=`` is built on a small module of
# that type, so a new layer kind or tuner family needs no hand row (LCD-712).
BUILDERS.update({layer: _layer_builder(base, layer) for base, layer in _layer_types()})
# The LyCORIS family roots are concrete (a kind is found from the module they wrap).
BUILDERS.update({root: _layer_builder(nn.Linear, root) for root in (LoHaLayer, LoKrLayer)})

# Every tuner whose method has an example config is built on the audit's model with it.
BUILDERS.update(
    {
        tuner: (lambda tuner=tuner, make=_EXAMPLE_CONFIGS[peft_type]: tuner(_base_model(), make()))
        for peft_type, tuner in _TUNERS.items()
        if issubclass(tuner, _BaseTuner) and peft_type in _EXAMPLE_CONFIGS
    }
)

# Every task wrapper ``get_peft_model`` can return takes the same
# ``(model, config)`` pair as ``PeftModel``, so each public subclass is
# registered from the package rather than by hand (LCD-716).
BUILDERS.update(
    {
        cls: (lambda cls=cls: cls(_base_model(), _config()))
        for name in peft.__all__
        if isinstance(cls := getattr(peft, name), type)
        and issubclass(cls, PeftModel)
        and cls is not PeftModel
    }
)
