"""The weights axis registers a probe, and takes it out again.

The sweep can share a process with a test session, where a probe left in the
registry reads as a published model with no factory: it failed
``test_every_published_darknet_checkpoint_has_the_release_it_names`` on CI.
"""

from types import SimpleNamespace

from lucid.test.audit._axes_state import WeightsAxis
from lucid.test.audit._result import Status
from lucid.weights import _registry


def test_the_probe_leaves_the_registry_as_it_found_it() -> None:
    models = dict(_registry._WEIGHTS_BY_MODEL)
    enums = dict(_registry._WEIGHTS_BY_ENUM)
    symbol = SimpleNamespace(qualname="lucid.weights.register_weights", short="register_weights")

    finding = WeightsAxis().run(symbol, None)

    assert finding.status is Status.PASS, finding.detail
    assert _registry._WEIGHTS_BY_MODEL == models
    assert _registry._WEIGHTS_BY_ENUM == enums
