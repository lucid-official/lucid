"""Every registered tuner and tuner layer has an audit builder, so the next method cannot miss one.

The audit builds each class by hand where a signature cannot describe it
(``lucid.test.audit._builders``).  A method added as subclasses only is found
through the registries; this guard fails the moment one is registered that the
builders cannot construct.
"""

import pytest

import lucid.nn as nn
from lucid.peft.tuners._base import _TUNERS, _BaseTuner, _BaseTunerLayer
from lucid.test.audit._builders import BUILDERS, _layer_builder, _layer_types

_TUNER_CLASSES = [t for t in _TUNERS.values() if issubclass(t, _BaseTuner)]
_LAYER_CLASSES = [layer for _, layer in _layer_types()]


@pytest.mark.parametrize("tuner", _TUNER_CLASSES, ids=lambda t: t.__name__)
def test_every_registered_tuner_has_a_builder_that_builds_it(tuner: type[_BaseTuner]) -> None:
    assert tuner in BUILDERS, f"{tuner.__name__} has no audit builder"
    first, second = BUILDERS[tuner](), BUILDERS[tuner]()
    assert type(first) is tuner and first is not second


@pytest.mark.parametrize("layer", _LAYER_CLASSES, ids=lambda t: t.__name__)
def test_every_registered_layer_has_a_builder_that_builds_it(layer: type[_BaseTunerLayer]) -> None:
    assert layer in BUILDERS, f"{layer.__name__} has no audit builder"
    assert isinstance(BUILDERS[layer](), layer)


def test_a_layer_naming_a_type_nothing_binds_still_gets_a_builder() -> None:
    """The builders are made when the audit is imported, from each layer's signature.

    Reading the annotations by value raised there for a layer that named a type
    bound only for type checkers, and took every module importing the audit down
    with it, the signature guard in ``test_public_signatures`` too: the one test
    that says which annotation it was.  Names and kinds need only forward references.
    """
    namespace: dict[str, object] = {"__name__": "lucid.scratch"}
    exec(
        "class Layer:\n"
        "    _layer_types = {}\n"
        "    def __init__(self, base_layer, adapter_name, r: Gone = 2): ...\n",
        namespace,
    )
    assert callable(_layer_builder(nn.Linear, namespace["Layer"]))
