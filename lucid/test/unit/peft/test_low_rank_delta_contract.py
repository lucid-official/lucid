"""What every tuner on the registry promises (``test_tuner_contract``), held for DeLoRA, GraLoRA, HiRA, GLoRA, DEFT and PEANuT.

The six are the low-rank delta methods of B4-4.  They are added to the shared contract here, in a file
of their own, by naming the same test classes under a ``method`` fixture of their own: each
test runs on the tiny MLP (CPU and Metal) with the update made non-zero where the test needs one, the
adapter files are written by hand in the reference's layout, and what a loaded file must compute is
worked out entry by entry from the papers' formulas (``_delta_forms``), never through the layers.
"""

from collections.abc import Callable
from typing import override

import pytest

import lucid
from lucid.peft import (
    DeloraConfig,
    DeloraModel,
    GraloraConfig,
    GraloraModel,
    HiraConfig,
    HiraModel,
    PeftType,
    get_peft_model,
)
from lucid.peft.tuners.delora import DeloraLayer
from lucid.peft.tuners.gralora import GraloraLayer
from lucid.peft.tuners.hira import HiraLayer
from lucid.test.unit.peft import _delta_forms as forms
from lucid.test.unit.peft._closed_forms import CLOSED_FORMS, _PREFIX, _mlp, _run
from lucid.test.unit.peft import test_tuner_contract as contract
from lucid.test.unit.peft.test_tuner_contract import (
    _REFERENCE_CONFIGS,
    TestFiles,
    TestMerging,
    TestMultiAdapter,
    TestRegistration,
    _layers,
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


class TestForward(contract.TestForward):
    """The shared forward tests, with the one that names trainable parameters by a fixed list of prefixes made exact."""

    @override
    def test_only_the_adapter_trains_and_only_it_gets_a_gradient(
        self, method: _Method, device: str
    ) -> None:
        net = contract._mlp(device)
        peft = get_peft_model(net, method.make())
        own = {id(p) for _, layer in _layers(peft) for p in layer._adapter_parameters("default")}
        trainable = {n: p for n, p in peft.named_parameters() if p.requires_grad}
        assert trainable and {id(p) for p in trainable.values()} == own
        peft(lucid.randn(5, 6, device=device)).sum().backward()
        for name, p in peft.named_parameters():
            if name in trainable:
                assert p.grad is not None and float(p.grad.abs().max().item()) > 0, name
            else:
                assert p.grad is None, name


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


def _gralora(hybrid_r: int) -> Callable[..., GraloraConfig]:
    def config(targets: tuple[str, ...], random: bool, **kw: object) -> GraloraConfig:
        return GraloraConfig.from_dict(
            {
                "r": 2,
                "gralora_k": 2,
                "hybrid_r": hybrid_r,
                "alpha": 4 + 2 * hybrid_r,  # scale alpha / (r + hybrid_r) = 2
                "target_modules": list(targets),
                "init_weights": not random,
                **kw,
            }
        )

    return config


def _gralora_forward(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
    """Grid of 2 x 2 blocks of rank 1 (Eq. 5), plus a rank-1 LoRA when the file holds one; scale 2."""
    net = _mlp()
    weights = []
    for index in (0, 2):
        p = f"{_PREFIX}{index}."
        general = None
        if p + "gralora_A_general.weight" in t:
            general = (
                t[p + "gralora_A_general.weight"].tolist(),
                t[p + "gralora_B_general.weight"].tolist(),
            )
        delta = forms.gralora_delta(
            t[p + "gralora_A"].tolist(), t[p + "gralora_B"].tolist(), 2.0, general
        )
        weights.append(net[index].weight + lucid.tensor(delta))
    return _run(weights, [net[0].bias, net[2].bias], x)


_GRALORA_FILE = {
    f"{_PREFIX}0.gralora_A": (2, 3, 2),
    f"{_PREFIX}0.gralora_B": (2, 2, 4),
    f"{_PREFIX}2.gralora_A": (2, 4, 2),
    f"{_PREFIX}2.gralora_B": (2, 2, 2),
}
_GRALORA_HYBRID_FILE = {
    **_GRALORA_FILE,
    f"{_PREFIX}0.gralora_A_general.weight": (1, 6),
    f"{_PREFIX}0.gralora_B_general.weight": (8, 1),
    f"{_PREFIX}2.gralora_A_general.weight": (1, 8),
    f"{_PREFIX}2.gralora_B_general.weight": (4, 1),
}


def _hira(targets: tuple[str, ...], random: bool, **kw: object) -> HiraConfig:
    return HiraConfig.from_dict(
        {"r": 2, "target_modules": list(targets), "init_weights": not random, **kw}
    )


def _hira_forward(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
    """``W' = W + W * (B A)``, ``*`` the elementwise product."""
    net = _mlp()
    weights = []
    for index in (0, 2):
        p = f"{_PREFIX}{index}."
        weight = net[index].weight
        delta = forms.hira_delta(
            weight.tolist(), t[p + "hira_A"].tolist(), t[p + "hira_B"].tolist()
        )
        weights.append(weight + lucid.tensor(delta))
    return _run(weights, [net[0].bias, net[2].bias], x)


_HIRA_FILE = {
    f"{_PREFIX}0.hira_A": (2, 6),
    f"{_PREFIX}0.hira_B": (8, 2),
    f"{_PREFIX}2.hira_A": (2, 8),
    f"{_PREFIX}2.hira_B": (4, 2),
}

#: Method rows in the order the methods landed.
_METHODS = [
    _Method("delora", PeftType.DELORA, _delora, DeloraModel, DeloraLayer, _DELORA_FILE),
    _Method("gralora", PeftType.GRALORA, _gralora(0), GraloraModel, GraloraLayer, _GRALORA_FILE),
    _Method(
        "gralora_hybrid",
        PeftType.GRALORA,
        _gralora(1),
        GraloraModel,
        GraloraLayer,
        _GRALORA_HYBRID_FILE,
    ),
    _Method("hira", PeftType.HIRA, _hira, HiraModel, HiraLayer, _HIRA_FILE),
]

CLOSED_FORMS["delora"] = _delora_forward
CLOSED_FORMS["gralora"] = CLOSED_FORMS["gralora_hybrid"] = _gralora_forward
CLOSED_FORMS["hira"] = _hira_forward
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
_GRALORA_REFERENCE = {
    "alpha": 4,
    "auto_mapping": None,
    "base_model_name_or_path": "tiny",
    "bias": "none",
    "fan_in_fan_out": False,
    "gralora_dropout": 0.0,
    "gralora_k": 2,
    "hybrid_r": 0,
    "inference_mode": True,
    "init_weights": True,
    "layers_pattern": None,
    "layers_to_transform": None,
    "modules_to_save": None,
    "peft_type": "GRALORA",
    "peft_version": "0.21.2",
    "r": 2,
    "revision": None,
    "target_modules": ["0", "2"],
    "task_type": None,
}
_REFERENCE_CONFIGS["gralora"] = _GRALORA_REFERENCE
_REFERENCE_CONFIGS["gralora_hybrid"] = {**_GRALORA_REFERENCE, "alpha": 6, "hybrid_r": 1}
_REFERENCE_CONFIGS["hira"] = {
    "auto_mapping": None,
    "base_model_name_or_path": "tiny",
    "exclude_modules": None,
    "fan_in_fan_out": False,
    "hira_dropout": 0.0,
    "inference_mode": True,
    "init_weights": True,
    "layers_pattern": None,
    "layers_to_transform": None,
    "modules_to_save": None,
    "peft_type": "HIRA",
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
