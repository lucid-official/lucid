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
from lucid.peft import (
    IA3Config,
    IA3Model,
    LoHaConfig,
    LoHaModel,
    LoKrConfig,
    LoKrModel,
    LoraConfig,
    LoraModel,
    PeftConfig,
    PeftModel,
)
from lucid.peft.tuners import lora
from lucid.peft.tuners._base import _BaseTunerLayer
from lucid.peft.tuners.ia3 import IA3Layer
from lucid.peft.tuners.loha import LoHaLayer
from lucid.peft.tuners.lokr import LoKrLayer


def _base_model() -> nn.Module:
    """Two linear layers around a ReLU, at the audit's usual width of 4."""
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))


def _config() -> LoraConfig:
    return LoraConfig(r=2, target_modules=["0", "2"])


#: A small module of each kind a tuner layer can wrap (its ``base_type=``).
_BASES: dict[type[nn.Module], Callable[[], nn.Module]] = {
    nn.Linear: lambda: nn.Linear(4, 3),
    nn.Conv1d: lambda: nn.Conv1d(2, 3, 3),
    nn.Conv2d: lambda: nn.Conv2d(2, 3, 3),
    nn.Conv3d: lambda: nn.Conv3d(2, 3, 3),
    nn.Embedding: lambda: nn.Embedding(5, 4),
    nn.MultiheadAttention: lambda: nn.MultiheadAttention(4, 2),
}

#: The config a layer family takes as its third argument, where its constructor wants one.
_LAYER_CONFIGS: dict[type[_BaseTunerLayer], Callable[[], PeftConfig]] = {
    IA3Layer: lambda: IA3Config(target_modules=["x"], feedforward_modules=[]),
    LoHaLayer: lambda: LoHaConfig(target_modules=["x"]),
    LoKrLayer: lambda: LoKrConfig(target_modules=["x"]),
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
    make_config = next((m for root, m in _LAYER_CONFIGS.items() if issubclass(layer, root)), None)
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
    PeftModel: lambda: PeftModel(_base_model(), _config()),
    lora.ParamWrapper: lambda: lora.ParamWrapper(_Gate(), "default", "w", r=2),
}

# Every tuner layer registered by ``base_type=`` is built on a small module of
# that type, so a new layer kind or tuner family needs no hand row (LCD-712).
BUILDERS.update({layer: _layer_builder(base, layer) for base, layer in _layer_types()})
# The LyCORIS family roots are concrete (a kind is found from the module they wrap).
BUILDERS.update({root: _layer_builder(nn.Linear, root) for root in (LoHaLayer, LoKrLayer)})

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
