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
    DeftConfig,
    DeftModel,
    DeloraConfig,
    DeloraModel,
    GloraConfig,
    GloraModel,
    GraloraConfig,
    GraloraModel,
    HiraConfig,
    HiraModel,
    PeanutConfig,
    PeanutModel,
    PeftType,
    get_peft_model,
)
from lucid.peft.tuners.deft import DeftLayer
from lucid.peft.tuners.delora import DeloraLayer
from lucid.peft.tuners.glora import GloraLayer
from lucid.peft.tuners.gralora import GraloraLayer
from lucid.peft.tuners.hira import HiraLayer
from lucid.peft.tuners.peanut import PeanutLayer
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


def _glora(**forms_: str) -> Callable[..., GloraConfig]:
    def config(targets: tuple[str, ...], random: bool, **kw: object) -> GloraConfig:
        return GloraConfig.from_dict(
            {"r": 2, "target_modules": list(targets), "init_weights": not random, **forms_, **kw}
        )

    return config


def _support(t: dict[str, lucid.Tensor], prefix: str) -> forms.Support:
    """The support tensor the file holds under ``prefix`` (``...glora_A``), told by the names it keeps."""
    if prefix + ".Xd" in t:
        return "lora", [t[prefix + ".Xd"].tolist(), t[prefix + ".Xu"].tolist()]
    if prefix + ".X" in t:
        kind = "constant" if tuple(t[prefix + ".X"].shape) == (1,) else "vector"
        return kind, [t[prefix + ".X"].tolist()]
    return "none", [t[prefix + ".zeros"].tolist()]


def _glora_forward(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
    """``W' = W + W * A + B`` and ``b' = b + b * D + E + W C``, each support as the file holds it."""
    net = _mlp()
    weights, biases = [], []
    for index in (0, 2):
        p = f"{_PREFIX}{index}."
        supports = [_support(t, p + f"glora_{name}") for name in "ABCDE"]
        delta_w, delta_b = forms.glora_changes(
            net[index].weight.tolist(), net[index].bias.tolist(), *supports
        )
        assert delta_b is not None
        weights.append(net[index].weight + lucid.tensor(delta_w))
        biases.append(net[index].bias + lucid.tensor(delta_b))
    return _run(weights, biases, x)


def _glora_file(a_b: str, c: str, d_e: str) -> dict[str, tuple[int, ...]]:
    """The adapter file of the tiny MLP (targets ``0`` and ``2``) for rank 2 and the given forms of the supports."""
    file: dict[str, tuple[int, ...]] = {}
    for index, (out, inn) in {0: (8, 6), 2: (4, 8)}.items():
        p = f"{_PREFIX}{index}.glora_"
        shapes = {
            "A": (a_b, (out, inn), True),
            "B": (a_b, (out, inn), True),
            "C": (c, (inn, 1), True),
            "D": (d_e, (out,), False),
            "E": (d_e, (out,), False),
        }
        for name, (kind, shape, column) in shapes.items():
            if kind == "lora":
                file[f"{p}{name}.Xd"], file[f"{p}{name}.Xu"] = (shape[0], 2), (2, shape[-1])
            elif kind == "vector":
                file[f"{p}{name}.X"] = (shape[0], 1) if column else (shape[0],)
            elif kind == "constant":
                file[f"{p}{name}.X"] = (1,)
            else:
                file[f"{p}{name}.zeros"] = shape
    return file


def _deft(**fixed: object) -> Callable[..., DeftConfig]:
    def config(targets: tuple[str, ...], random: bool, **kw: object) -> DeftConfig:
        return DeftConfig.from_dict(
            {"r": 2, "target_modules": list(targets), "init_weights": not random, **fixed, **kw}
        )

    return config


def _deft_forward(method: str, scaling: float, para: bool) -> Callable[..., lucid.Tensor]:
    def forward(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
        """``W' = (I - P_proj) W + Q_P R``, the projector and basis of ``P`` as the method says."""
        net = _mlp()
        weights = []
        for index in (0, 2):
            p = f"{_PREFIX}{index}."
            injection = None if para else t[p + "deft_R"].tolist()
            delta = forms.deft_delta(
                net[index].weight.tolist(),
                t[p + "deft_P"].tolist(),
                injection,
                method=method,
                scaling=scaling,
            )
            weights.append(net[index].weight + lucid.tensor(delta))
        return _run(weights, [net[0].bias, net[2].bias], x)

    return forward


_DEFT_FILE = {
    f"{_PREFIX}0.deft_P": (8, 2),
    f"{_PREFIX}0.deft_R": (2, 6),
    f"{_PREFIX}2.deft_P": (4, 2),
    f"{_PREFIX}2.deft_R": (2, 8),
}
_DEFT_PARA_FILE = {k: v for k, v in _DEFT_FILE.items() if k.endswith("deft_P")}


def _peanut(**fixed: object) -> Callable[..., PeanutConfig]:
    def config(targets: tuple[str, ...], random: bool, **kw: object) -> PeanutConfig:
        return PeanutConfig.from_dict(
            {"r": 2, "target_modules": list(targets), "init_weights": not random, **fixed, **kw}
        )

    return config


def _peanut_forward(depth: int, act: str, scaling: float) -> Callable[..., lucid.Tensor]:
    def forward(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
        """``W' = W + f(W)``: the network of the paper, from the layer weights the file holds."""
        net = _mlp()
        weights = []
        for index in (0, 2):
            p = f"{_PREFIX}{index}."
            delta = forms.peanut_delta(
                net[index].weight.tolist(),
                t[p + "peanut_A.weight"].tolist(),
                t[p + "peanut_B.weight"].tolist(),
                [t[f"{p}peanut_encoders.{i}.weight"].tolist() for i in range(depth)],
                [t[f"{p}peanut_decoders.{i}.weight"].tolist() for i in range(depth)],
                act=forms.ACTIVATIONS[act],
                scaling=scaling,
            )
            weights.append(net[index].weight + lucid.tensor(delta))
        return _run(weights, [net[0].bias, net[2].bias], x)

    return forward


def _peanut_file(depth: int) -> dict[str, tuple[int, ...]]:
    file: dict[str, tuple[int, ...]] = {}
    for index, out in {0: 8, 2: 4}.items():
        p = f"{_PREFIX}{index}."
        file[p + "peanut_A.weight"], file[p + "peanut_B.weight"] = (2, out), (out, 2)
        for i in range(depth):
            file[f"{p}peanut_encoders.{i}.weight"] = (2, 2)
            file[f"{p}peanut_decoders.{i}.weight"] = (2, 2)
    return file


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
    _Method(
        "glora",
        PeftType.GLORA,
        _glora(),
        GloraModel,
        GloraLayer,
        _glora_file("lora", "lora", "constant"),
    ),
    _Method(
        "glora_vector",
        PeftType.GLORA,
        _glora(config_A_B="vector", config_C="vector", config_D_E="vector"),
        GloraModel,
        GloraLayer,
        _glora_file("vector", "vector", "vector"),
    ),
    _Method(
        "glora_mixed",
        PeftType.GLORA,
        _glora(config_A_B="constant", config_C="none", config_D_E="vector"),
        GloraModel,
        GloraLayer,
        _glora_file("constant", "none", "vector"),
    ),
    # A ReLU projector of the 0.02-scale direction matrix barely moves the output until it is trained.
    _Method(
        "deft",
        PeftType.DEFT,
        _deft(alpha=4.0),
        DeftModel,
        DeftLayer,
        _DEFT_FILE,
        random_start=False,
        exact_start=False,
    ),
    _Method(
        "deft_qr",
        PeftType.DEFT,
        _deft(decomposition_method="qr", alpha=4.0),
        DeftModel,
        DeftLayer,
        _DEFT_FILE,
        exact_start=False,
    ),
    _Method(
        "deft_para",
        PeftType.DEFT,
        _deft(para=True, decomposition_method="qr"),
        DeftModel,
        DeftLayer,
        _DEFT_PARA_FILE,
        starts_at_base=False,
    ),
    _Method(
        "peanut", PeftType.PEANUT, _peanut(scaling=2.0), PeanutModel, PeanutLayer, _peanut_file(0)
    ),
    _Method(
        "peanut_deep",
        PeftType.PEANUT,
        _peanut(depth=2, act_fn="tanh", scaling=0.5),
        PeanutModel,
        PeanutLayer,
        _peanut_file(2),
    ),
]

CLOSED_FORMS["delora"] = _delora_forward
CLOSED_FORMS["gralora"] = CLOSED_FORMS["gralora_hybrid"] = _gralora_forward
CLOSED_FORMS["hira"] = _hira_forward
CLOSED_FORMS["peanut"] = _peanut_forward(0, "relu", 2.0)
CLOSED_FORMS["peanut_deep"] = _peanut_forward(2, "tanh", 0.5)
CLOSED_FORMS["deft"] = _deft_forward("relu", 2.0, False)
CLOSED_FORMS["deft_qr"] = _deft_forward("qr", 2.0, False)
CLOSED_FORMS["deft_para"] = _deft_forward("qr", 1.0, True)
CLOSED_FORMS["glora"] = CLOSED_FORMS["glora_vector"] = CLOSED_FORMS["glora_mixed"] = _glora_forward
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
_GLORA_REFERENCE = {
    "auto_mapping": None,
    "base_model_name_or_path": "tiny",
    "bias": "none",
    "config_A_B": "lora",
    "config_C": "lora",
    "config_D_E": "constant",
    "inference_mode": True,
    "init_weights": True,
    "modules_to_save": None,
    "peft_type": "GLORA",
    "peft_version": "0.21.2",
    "r": 2,
    "revision": None,
    "target_modules": ["0", "2"],
    "task_type": None,
}
_REFERENCE_CONFIGS["glora"] = _GLORA_REFERENCE
_REFERENCE_CONFIGS["glora_vector"] = {
    **_GLORA_REFERENCE,
    "config_A_B": "vector",
    "config_C": "vector",
    "config_D_E": "vector",
}
_REFERENCE_CONFIGS["glora_mixed"] = {
    **_GLORA_REFERENCE,
    "config_A_B": "constant",
    "config_C": "none",
    "config_D_E": "vector",
}
_DEFT_REFERENCE = {
    "alpha": 4.0,
    "auto_mapping": None,
    "base_model_name_or_path": "tiny",
    "bias": "none",
    "decomposition_method": "relu",
    "deft_dropout": 0.0,
    "exclude_modules": None,
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_scale": 1.0,
    "init_weights": True,
    "layers_pattern": None,
    "layers_to_transform": None,
    "modules_to_save": None,
    "para": False,
    "peft_type": "DEFT",
    "peft_version": "0.21.2",
    "r": 2,
    "revision": None,
    "target_modules": ["0", "2"],
    "task_type": None,
}
_REFERENCE_CONFIGS["deft"] = _DEFT_REFERENCE
_REFERENCE_CONFIGS["deft_qr"] = {**_DEFT_REFERENCE, "decomposition_method": "qr"}
_REFERENCE_CONFIGS["deft_para"] = {
    **_DEFT_REFERENCE,
    "alpha": None,
    "decomposition_method": "qr",
    "para": True,
}


_PEANUT_REFERENCE = {
    "act_fn": "relu",
    "auto_mapping": None,
    "base_model_name_or_path": "tiny",
    "depth": 0,
    "exclude_modules": None,
    "inference_mode": True,
    "init_weights": True,
    "layers_pattern": None,
    "layers_to_transform": None,
    "modules_to_save": None,
    "peft_type": "PEANUT",
    "peft_version": "0.21.2",
    "r": 2,
    "revision": None,
    "scaling": 2.0,
    "target_modules": ["0", "2"],
    "task_type": None,
}
_REFERENCE_CONFIGS["peanut"] = _PEANUT_REFERENCE
_REFERENCE_CONFIGS["peanut_deep"] = {
    **_PEANUT_REFERENCE,
    "act_fn": "tanh",
    "depth": 2,
    "scaling": 0.5,
}


@pytest.fixture(params=_METHODS, ids=lambda m: m.name)
def method(request: pytest.FixtureRequest) -> _Method:
    assert isinstance(request.param, _Method)
    return request.param
