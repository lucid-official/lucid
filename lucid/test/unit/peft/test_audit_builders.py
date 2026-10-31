"""Every registered tuner and tuner layer has an audit builder, so the next method cannot miss one.

The audit builds each class by hand where a signature cannot describe it
(``lucid.test.audit._builders``).  A method added as subclasses only is found
through the registries; this guard fails the moment one is registered that the
builders cannot construct.
"""

import pytest

from lucid.peft.tuners._base import _TUNERS, _BaseTuner, _BaseTunerLayer
from lucid.test.audit._builders import BUILDERS, _layer_types

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
