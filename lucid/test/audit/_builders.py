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

from collections.abc import Callable
from typing import Any

import lucid.nn as nn
from lucid.peft import LoraConfig, LoraModel, PeftModel
from lucid.peft.tuners import lora


def _base_model() -> nn.Module:
    """Two linear layers around a ReLU, at the audit's usual width of 4."""
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))


def _config() -> LoraConfig:
    return LoraConfig(r=2, target_modules=["0", "2"])


#: Class -> a zero-argument builder of one instance.
BUILDERS: dict[type, Callable[[], Any]] = {
    lora.Linear: lambda: lora.Linear(nn.Linear(4, 3), "default", r=2),
    LoraModel: lambda: LoraModel(_base_model(), _config()),
    PeftModel: lambda: PeftModel(_base_model(), _config()),
}
