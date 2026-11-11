"""What every tuner on the registry promises (``test_tuner_contract``), held for BEFT, Super-Tuning and WaveFT.

The three are the sparse and bias methods of B4-7.  They are added to the shared contract here, in a
file of their own, by naming the same test classes under a ``method`` fixture of their own: each test
runs on the tiny MLP (CPU and Metal) with the update made non-zero where the test needs one, the
adapter files are written by hand in the reference's layout, and what a loaded file must compute is
worked out from the papers' formulas with plain loops (``_sparse_forms``), never through the layers.
"""

from typing import override

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import BeftConfig, BeftModel, PeftType, get_peft_model
from lucid.peft.tuners.beft import BeftLayer
from lucid.test.unit.peft import test_tuner_contract as contract
from lucid.test.unit.peft._closed_forms import CLOSED_FORMS, _PREFIX, _mlp, _run
from lucid.test.unit.peft.test_orthogonal_spectral_contract import TestForward
from lucid.test.unit.peft.test_tuner_contract import (
    _REFERENCE_CONFIGS,
    TestFiles,
    TestMultiAdapter,
    TestRegistration,
    _Method,
    test_a_reference_written_adapter_loads,
    test_compiled_follows_set_adapter_disable_and_merge,
)

#: The shared contract's tests, collected here under this module's ``method`` fixture.
__all__ = [
    "TestFiles",
    "TestForward",
    "TestMerging",
    "TestMultiAdapter",
    "TestRegistration",
    "test_a_reference_written_adapter_loads",
    "test_compiled_follows_set_adapter_disable_and_merge",
]


class TestMerging(contract.TestMerging):
    """The shared merging tests, with the shared-parameter guard aimed at the parameter a method merges into."""

    @override
    def test_a_shared_weight_is_not_merged_into(self, method: _Method) -> None:
        if method.peft_type is not PeftType.BEFT:
            super().test_a_shared_weight_is_not_merged_into(method)
            return
        # BEFT writes the bias and leaves the weight alone: tie the bias, which is what a merge would change.
        net = nn.Sequential(nn.Linear(6, 6), nn.Linear(6, 6))
        net[1].bias = net[0].bias
        peft = get_peft_model(net, method.make(targets=("0",)))
        with pytest.raises(RuntimeError, match="cannot merge into"):
            peft.base_model.merge_adapter()


def _beft(targets: tuple[str, ...], random: bool, **kw: object) -> BeftConfig:
    return BeftConfig.from_dict({"target_modules": list(targets), "init_weights": not random, **kw})


def _beft_forward(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
    """``y = W x + b + beta`` on both layers, ``beta`` the file's ``(1, out)`` row."""
    net = _mlp()
    weights = [net[0].weight, net[2].weight]
    biases = [net[index].bias + t[f"{_PREFIX}{index}.beft_bias"].reshape(-1) for index in (0, 2)]
    return _run(weights, biases, x)


#: Method rows in the order the methods landed.
_METHODS: list[_Method] = [
    _Method(
        "beft",
        PeftType.BEFT,
        _beft,
        BeftModel,
        BeftLayer,
        {f"{_PREFIX}0.beft_bias": (1, 8), f"{_PREFIX}2.beft_bias": (1, 4)},
    ),
]

CLOSED_FORMS["beft"] = _beft_forward
_REFERENCE_CONFIGS["beft"] = {
    "auto_mapping": None,
    "base_model_name_or_path": "tiny",
    "inference_mode": True,
    "init_weights": True,
    "modules_to_save": None,
    "peft_type": "BEFT",
    "peft_version": "0.21.2",
    "revision": None,
    "target_modules": ["0", "2"],
    "task_type": None,
}


@pytest.fixture(params=_METHODS, ids=lambda m: m.name)
def method(request: pytest.FixtureRequest) -> _Method:
    assert isinstance(request.param, _Method)
    return request.param
