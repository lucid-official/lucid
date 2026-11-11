"""What every tuner on the registry promises (``test_tuner_contract``), held for RoAd, PSOFT, OSF and FRoD.

The four are the orthogonal and spectral methods of B4-6.  They are added to the shared contract here,
in a file of their own, by naming the same test classes under a ``method`` fixture of their own: each
test runs on the tiny MLP (CPU and Metal) with the update made non-zero where the test needs one, the
adapter files are written by hand in the reference's layout, and what a loaded file must compute is
worked out entry by entry from the papers' formulas (``_spectral_forms``), never through the layers.
"""

import types
from collections.abc import Callable
from typing import override

import pytest

import lucid
import lucid.linalg as linalg
import lucid.nn as nn
from lucid.peft import (
    FrodConfig,
    FrodModel,
    OSFConfig,
    OSFModel,
    PeftType,
    PsoftConfig,
    PsoftModel,
    RoadConfig,
    RoadModel,
    get_peft_model,
)
from lucid.peft.tuners.frod import FrodLayer
from lucid.peft.tuners.osf import OSFLayer
from lucid.peft.tuners.psoft import PsoftLayer
from lucid.peft.tuners.road import RoadLayer
from lucid.test.unit.peft import _spectral_forms as forms
from lucid.test.unit.peft import test_tuner_contract as contract
from lucid.test.unit.peft._closed_forms import CLOSED_FORMS, _PREFIX, _mlp, _run
from lucid.test.unit.peft.test_tuner_contract import (
    _REFERENCE_CONFIGS,
    TestFiles,
    TestMerging,
    TestMultiAdapter,
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


class TestRegistration(contract.TestRegistration):
    """The shared registration tests, with the default-target one made right for a method that falls back to every linear layer."""

    @override
    def test_a_default_of_raw_parameters_is_not_taken_for_a_module_method(
        self, method: _Method
    ) -> None:
        if method.peft_type is not PeftType.OSF:
            super().test_a_default_of_raw_parameters_is_not_taken_for_a_module_method(method)
            return
        # OSF has no "no default" error: a model type its table lacks takes every linear layer, as the
        # reference does, so a model with one linear layer and nothing else to take finds no target.
        net = nn.Sequential(nn.Linear(4, 4))
        net.config = types.SimpleNamespace(model_type="neural_ode")  # type: ignore[assignment]
        config = method.make(targets=())
        config.target_modules = None  # type: ignore[attr-defined]
        peft = method.tuner(net, config, "default")
        assert peft.targeted_module_names == ["0"]


def _road(variant: str) -> Callable[..., RoadConfig]:
    def config(targets: tuple[str, ...], random: bool, **kw: object) -> RoadConfig:
        return RoadConfig.from_dict(
            {
                "variant": variant,
                "group_size": 2,
                "target_modules": list(targets),
                "init_weights": not random,
                **kw,
            }
        )

    return config


def _road_forward(variant: str) -> Callable[..., lucid.Tensor]:
    def forward(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
        """``W' = R W`` and ``b' = R b`` with the paper's pair blocks, ``R`` from the file's angles and scales."""
        net = _mlp()
        weights, biases = [], []
        for index in (0, 2):
            p = f"{_PREFIX}{index}."
            weight, bias = forms.road_adapted(
                net[index].weight.tolist(),
                net[index].bias.tolist(),
                variant,
                2,
                t[p + "road_theta"].tolist(),
                t[p + "road_alpha"].tolist(),
            )
            weights.append(lucid.tensor(weight))
            biases.append(lucid.tensor(bias))
        return _run(weights, biases, x)

    return forward


def _road_file(count: int) -> dict[str, tuple[int, ...]]:
    """The file of the tiny MLP at group size 2: ``count`` numbers per output feature, halved for ``road_1``."""
    file: dict[str, tuple[int, ...]] = {}
    for index, out in {0: 8, 2: 4}.items():
        for name in ("road_theta", "road_alpha"):
            file[f"{_PREFIX}{index}.{name}"] = (out * count // 2,)
    return file


def _psoft(**fixed: object) -> Callable[..., PsoftConfig]:
    def config(targets: tuple[str, ...], random: bool, **kw: object) -> PsoftConfig:
        return PsoftConfig.from_dict(
            {
                "r": 3,
                "psoft_alpha": fixed.get("psoft_alpha", 6),
                "target_modules": list(targets),
                "init_weights": not random,
                **fixed,
                **kw,
            }
        )

    return config


def _psoft_forward(init: str, orth: bool, mag_a: bool) -> Callable[..., lucid.Tensor]:
    def forward(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
        """``W' = W + U S (C - I) V^T`` with ``C`` the file's matrix: Cayley blocks by Gauss-Jordan, the SVD of the base from the primitive."""
        net = _mlp()
        weights = []
        for index in (0, 2):
            p = f"{_PREFIX}{index}.psoft_R."
            rotation = (
                forms.cayley(t[p + "weight"].tolist(), 3) if orth else t[p + "weight"].tolist()
            )
            inner = forms.psoft_inner(
                rotation,
                t[p + "vector_b"].tolist(),
                t[p + "vector_a"].tolist() if mag_a else None,
            )
            u, s, vt = linalg.svd(net[index].weight.detach(), full_matrices=False)
            delta = forms.psoft_delta(u.tolist(), s.tolist(), vt.tolist(), inner, 3, init)
            weights.append(net[index].weight + lucid.tensor(delta))
        return _run(weights, [net[0].bias, net[2].bias], x)

    return forward


def _psoft_file(orth: bool, mag_a: bool) -> dict[str, tuple[int, ...]]:
    file: dict[str, tuple[int, ...]] = {}
    for index in (0, 2):
        p = f"{_PREFIX}{index}.psoft_R."
        file[p + "weight"] = (3,) if orth else (3, 3)
        file[p + "vector_b"] = (3,)
        if mag_a:
            file[p + "vector_a"] = (3,)
    return file


def _osf(targets: tuple[str, ...], random: bool, **kw: object) -> OSFConfig:
    return OSFConfig.from_dict(
        {"effective_rank": 2, "target_modules": list(targets), "init_weights": not random, **kw}
    )


def _osf_forward(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
    """``W' = U_high S_high V_high^T + U_low S_low V_low^T``: the base's top two triplets, the file's low part."""
    net = _mlp()
    weights = []
    for index in (0, 2):
        p = f"{_PREFIX}{index}.osf_svd_params."
        u, s, vt = linalg.svd(net[index].weight.detach(), full_matrices=False)
        weights.append(
            lucid.tensor(
                forms.osf_adapted(
                    u.tolist(),
                    s.tolist(),
                    vt.tolist(),
                    t[p + "U_low"].tolist(),
                    t[p + "S_low"].tolist(),
                    t[p + "V_low"].tolist(),
                    2,
                )
            )
        )
    return _run(weights, [net[0].bias, net[2].bias], x)


_OSF_FILE = {
    f"{_PREFIX}0.osf_svd_params.U_low": (8, 4),
    f"{_PREFIX}0.osf_svd_params.S_low": (4,),
    f"{_PREFIX}0.osf_svd_params.V_low": (4, 6),
    f"{_PREFIX}2.osf_svd_params.U_low": (4, 2),
    f"{_PREFIX}2.osf_svd_params.S_low": (2,),
    f"{_PREFIX}2.osf_svd_params.V_low": (2, 8),
}


def _frod(targets: tuple[str, ...], random: bool, **kw: object) -> FrodConfig:
    return FrodConfig.from_dict(
        {
            "sparse_rate": 0.5,
            "projection_prng_key": 3,
            "progressbar": False,
            "target_modules": list(targets),
            "init_weights": not random,
            **kw,
        }
    )


_FROD_SIZES = {
    "0": (6, 18),
    "2": (8, 32),
}  # category -> (input width, entries of S) of the tiny MLP at rate 0.5


def _frod_file() -> dict[str, tuple[int, ...]]:
    file: dict[str, tuple[int, ...]] = {}
    for index, (width, count) in zip((0, 2), _FROD_SIZES.values(), strict=True):
        file[f"{_PREFIX}{index}.frod_lambda_l"] = (width,)
        file[f"{_PREFIX}{index}.frod_lambda_s_values"] = (count,)
    for category, (width, count) in _FROD_SIZES.items():
        file[f"base_model.frod_V.{category}"] = (width, width)
        file[f"base_model.frod_s_indices.{category}"] = (2, count)
        file[f"base_model.frod_s_size.{category}"] = (2,)
    return file


def _frod_tensors() -> dict[str, lucid.Tensor]:
    """A file a person could have written: an invertible basis, distinct off-diagonal positions, trained values of any size."""
    lucid.manual_seed(7)
    made: dict[str, lucid.Tensor] = {}
    for category, (width, count) in _FROD_SIZES.items():
        index = int(category)
        made[f"{_PREFIX}{index}.frod_lambda_l"] = (
            lucid.randint(-2, 3, (width,)).to(lucid.float32) * 0.5
        )
        made[f"{_PREFIX}{index}.frod_lambda_s_values"] = (
            lucid.randint(-2, 3, (count,)).to(lucid.float32) * 0.5
        )
        made[f"base_model.frod_V.{category}"] = lucid.eye(width) + 0.2 * lucid.randn(width, width)
        pairs = [(i, j) for i in range(width) for j in range(width) if i != j][::-1][:count]
        made[f"base_model.frod_s_indices.{category}"] = lucid.tensor(
            [[p[0] for p in pairs], [p[1] for p in pairs]], dtype=lucid.int64
        )
        made[f"base_model.frod_s_size.{category}"] = lucid.tensor([width, width], dtype=lucid.int64)
    return made


def _frod_forward(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
    """``W' = U (diag(lambda_l) + S) V^T``: ``U`` from the base weight and the file's basis, the rest as the file holds it."""
    net = _mlp()
    weights = []
    for index in (0, 2):
        p = f"{_PREFIX}{index}."
        category = str(index)
        weights.append(
            lucid.tensor(
                forms.frod_adapted(
                    net[index].weight.tolist(),
                    t[f"base_model.frod_V.{category}"].tolist(),
                    t[f"base_model.frod_s_indices.{category}"].tolist(),
                    t[p + "frod_lambda_l"].tolist(),
                    t[p + "frod_lambda_s_values"].tolist(),
                )
            )
        )
    return _run(weights, [net[0].bias, net[2].bias], x)


#: Method rows in the order the methods landed.
_METHODS = [
    _Method("road", PeftType.ROAD, _road("road_1"), RoadModel, RoadLayer, _road_file(1)),
    _Method("road_2", PeftType.ROAD, _road("road_2"), RoadModel, RoadLayer, _road_file(2)),
    _Method("road_4", PeftType.ROAD, _road("road_4"), RoadModel, RoadLayer, _road_file(4)),
    _Method("psoft", PeftType.PSOFT, _psoft(), PsoftModel, PsoftLayer, _psoft_file(True, True)),
    _Method(
        "psoft_free",
        PeftType.PSOFT,
        _psoft(psoft_alpha=3, psoft_orth=False, psoft_mag_a=False, ab_svd_init="pissa_init"),
        PsoftModel,
        PsoftLayer,
        _psoft_file(False, False),
    ),
    _Method("osf", PeftType.OSF, _osf, OSFModel, OSFLayer, _OSF_FILE),
    # The adapted weight is the base's up to the rounding of the decomposition, not bit for bit.
    _Method(
        "frod",
        PeftType.FROD,
        _frod,
        FrodModel,
        FrodLayer,
        _frod_file(),
        exact_start=False,
        tensors=_frod_tensors,
    ),
]

CLOSED_FORMS["road"] = _road_forward("road_1")
CLOSED_FORMS["road_2"] = _road_forward("road_2")
CLOSED_FORMS["road_4"] = _road_forward("road_4")
_ROAD_REFERENCE = {
    "auto_mapping": None,
    "base_model_name_or_path": "tiny",
    "group_size": 2,
    "inference_mode": True,
    "init_weights": True,
    "modules_to_save": None,
    "peft_type": "ROAD",
    "peft_version": "0.21.2",
    "revision": None,
    "target_modules": ["0", "2"],
    "task_type": None,
    "variant": "road_1",
}
_REFERENCE_CONFIGS["road"] = _ROAD_REFERENCE
_REFERENCE_CONFIGS["road_2"] = {**_ROAD_REFERENCE, "variant": "road_2"}
_REFERENCE_CONFIGS["road_4"] = {**_ROAD_REFERENCE, "variant": "road_4"}
CLOSED_FORMS["osf"] = _osf_forward
CLOSED_FORMS["frod"] = _frod_forward
_REFERENCE_CONFIGS["frod"] = {
    "auto_mapping": None,
    "base_model_name_or_path": "tiny",
    "bias": "none",
    "fan_in_fan_out": False,
    "frod_dropout": 0.0,
    "inference_mode": True,
    "init_weights": True,
    "layers_pattern": None,
    "layers_to_transform": None,
    "modules_to_save": None,
    "peft_type": "FROD",
    "peft_version": "0.21.2",
    "progressbar": False,
    "projection_prng_key": 3,
    "regularization_alpha": 0.001,
    "revision": None,
    "runtime_offload_base_weight": False,
    "save_projection": True,
    "sparse_rate": 0.5,
    "target_modules": ["0", "2"],
    "task_type": None,
}
_REFERENCE_CONFIGS["osf"] = {
    "auto_mapping": None,
    "base_model_name_or_path": "tiny",
    "effective_rank": 2,
    "inference_mode": True,
    "init_weights": True,
    "modules_to_save": None,
    "peft_type": "OSF",
    "peft_version": "0.21.2",
    "rank_pattern": None,
    "revision": None,
    "target_modules": ["0", "2"],
    "target_svd_config": None,
    "task_type": None,
}
CLOSED_FORMS["psoft"] = _psoft_forward("psoft_init", True, True)
CLOSED_FORMS["psoft_free"] = _psoft_forward("pissa_init", False, False)
_PSOFT_REFERENCE = {
    "ab_svd_init": "psoft_init",
    "auto_mapping": None,
    "base_model_name_or_path": "tiny",
    "cayley_neumann_eps": None,
    "exclude_modules": None,
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_weights": True,
    "layers_pattern": None,
    "layers_to_transform": None,
    "modules_to_save": None,
    "num_cayley_neumann_terms": 5,
    "peft_type": "PSOFT",
    "peft_version": "0.21.2",
    "psoft_alpha": 6,
    "psoft_dropout": 0.0,
    "psoft_mag_a": True,
    "psoft_mag_b": True,
    "psoft_orth": True,
    "psoft_svd": "full",
    "psoft_svd_lowrank_niter": 10,
    "r": 3,
    "random_seed": 0,
    "revision": None,
    "target_modules": ["0", "2"],
    "task_type": None,
    "use_cayley_neumann": False,
}
_REFERENCE_CONFIGS["psoft"] = _PSOFT_REFERENCE
_REFERENCE_CONFIGS["psoft_free"] = {
    **_PSOFT_REFERENCE,
    "ab_svd_init": "pissa_init",
    "psoft_alpha": 3,
    "psoft_mag_a": False,
    "psoft_orth": False,
}


@pytest.fixture(params=_METHODS, ids=lambda m: m.name)
def method(request: pytest.FixtureRequest) -> _Method:
    assert isinstance(request.param, _Method)
    return request.param
