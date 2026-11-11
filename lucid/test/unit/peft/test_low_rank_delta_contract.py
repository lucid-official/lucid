"""What every tuner on the registry promises (``test_tuner_contract``), held for DeLoRA, GraLoRA, HiRA, GLoRA, DEFT and PEANuT.

The six are the low-rank delta methods of B4-4.  They are added to the shared contract here, in a file
of their own, by naming the same test classes under a ``method`` fixture of their own: each
test runs on the tiny MLP (CPU and Metal) with the update made non-zero where the test needs one, the
adapter files are written by hand in the reference's layout, and what a loaded file must compute is
worked out entry by entry from the papers' formulas (``_delta_forms``), never through the layers.
"""

import pytest

import lucid
from lucid.peft import DeloraConfig, DeloraModel, PeftType
from lucid.peft.tuners.delora import DeloraLayer
from lucid.test.unit.peft import _delta_forms as forms
from lucid.test.unit.peft._closed_forms import CLOSED_FORMS, _PREFIX, _mlp, _run
from lucid.test.unit.peft.test_tuner_contract import (
    _REFERENCE_CONFIGS,
    TestFiles,
    TestForward,
    TestMerging,
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


def _delora(targets: tuple[str, ...], random: bool, **kw: object) -> DeloraConfig:
    return DeloraConfig.from_dict(
        {"r": 2, "target_modules": list(targets), "init_weights": not random, **kw}
    )


def _delora_forward(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
    """Rank 2: ``dW = (lambda / r) B Xi A`` with column ``j`` scaled by ``delora_w_norm[j]``."""
    net = _mlp()
    weights = []
    for index in (0, 2):
        p = f"{_PREFIX}{index}."
        delta = forms.delora_delta(
            t[p + "delora_A"].tolist(),
            t[p + "delora_B"].tolist(),
            t[p + "delora_lambda"].item(),
            t[p + "delora_w_norm"].tolist(),
        )
        weights.append(net[index].weight + lucid.tensor(delta))
    return _run(weights, [net[0].bias, net[2].bias], x)


_DELORA_FILE = {
    f"{_PREFIX}0.delora_A": (2, 6),
    f"{_PREFIX}0.delora_B": (8, 2),
    f"{_PREFIX}0.delora_lambda": (1,),
    f"{_PREFIX}0.delora_w_norm": (6,),
    f"{_PREFIX}2.delora_A": (2, 8),
    f"{_PREFIX}2.delora_B": (4, 2),
    f"{_PREFIX}2.delora_lambda": (1,),
    f"{_PREFIX}2.delora_w_norm": (8,),
}

#: Method rows in the order the methods landed.
_METHODS = [
    _Method(
        "delora", PeftType.DELORA, _delora, DeloraModel, DeloraLayer, _DELORA_FILE
    ),
]

CLOSED_FORMS["delora"] = _delora_forward
_REFERENCE_CONFIGS["delora"] = {
    "auto_mapping": None,
    "base_model_name_or_path": "tiny",
    "bias": "none",
    "delora_lambda": 15,
    "exclude_modules": None,
    "inference_mode": True,
    "init_weights": True,
    "lambda_pattern": {},
    "layers_pattern": None,
    "layers_to_transform": None,
    "module_dropout": 0.0,
    "modules_to_save": None,
    "peft_type": "DELORA",
    "peft_version": "0.21.2",
    "r": 2,
    "rank_pattern": {},
    "revision": None,
    "target_modules": ["0", "2"],
    "task_type": None,
}


@pytest.fixture(params=_METHODS, ids=lambda m: m.name)
def method(request: pytest.FixtureRequest) -> _Method:
    assert isinstance(request.param, _Method)
    return request.param
