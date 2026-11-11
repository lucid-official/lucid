"""What every tuner on the registry promises (``test_tuner_contract``), held for Uni-LoRA, PVeRA and TinyLoRA.

The three read the update of every layer out of tensors the tuner holds once for the whole model: a
vector (Uni-LoRA, TinyLoRA) or a pair of random projections (PVeRA).  They join the shared contract here, in
a file of their own, by naming the same test classes under a ``method`` fixture of their own: each test runs
on the tiny MLP (CPU and Metal), the adapter files are written by hand in the reference's layout, and what a
loaded file must compute is worked out entry by entry from the papers' formulas (``_bank_forms``), never
through the layers.  Lily, whose layers cannot merge, has its own file (``test_lily``).
"""

from collections.abc import Callable

import pytest

import lucid
from lucid.peft import PeftType, UniLoraConfig, UniLoraModel
from lucid.peft.tuners.unilora import UniLoraLayer
from lucid.peft.tuners.unilora._indices import assign
from lucid.test.unit.peft import _bank_forms as forms
from lucid.test.unit.peft._closed_forms import _PREFIX, _SIZES, CLOSED_FORMS, _mlp, _run
from lucid.test.unit.peft.test_low_rank_delta_contract import TestForward
from lucid.test.unit.peft.test_tuner_contract import (
    _REFERENCE_CONFIGS,
    TestFiles,
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

_UNILORA_LENGTH = 8


def _unilora(save_indices: bool) -> Callable[..., UniLoraConfig]:
    def config(targets: tuple[str, ...], random: bool, **kw: object) -> UniLoraConfig:
        return UniLoraConfig.from_dict(
            {
                "r": 2,
                "theta_d_length": _UNILORA_LENGTH,
                "save_indices": save_indices,
                "target_modules": list(targets),
                "init_weights": not random,
                **kw,
            }
        )

    return config


def _unilora_projection() -> dict[int, tuple[list[list[int]], ...]]:
    """The file's indices and scales of each layer of the MLP (targets ``0`` and ``2``), the draw the library makes."""
    shapes = [((2, inn), (out, 2)) for out, inn in _SIZES.values()]
    drawn = assign(shapes, _UNILORA_LENGTH, 42)
    return {
        index: (a.tolist(), b.tolist(), sa.tolist(), sb.tolist())
        for index, (a, b, sa, sb) in zip(_SIZES, drawn, strict=True)
    }


def _unilora_tensors() -> dict[str, lucid.Tensor]:
    """The file of the ``save_indices`` row: a vector of small multiples of a half, and the projection with its scales."""
    lucid.manual_seed(7)
    made = {
        "base_model.unilora_theta_d": lucid.randint(-2, 3, (_UNILORA_LENGTH,)).to(lucid.float32)
        * 0.5
    }
    for index, (a, b, sa, sb) in _unilora_projection().items():
        where = f"{_PREFIX}{index}.unilora_"
        made[where + "indices_A"] = lucid.tensor(a, dtype=lucid.int64)
        made[where + "indices_B"] = lucid.tensor(b, dtype=lucid.int64)
        made[where + "scales_A"] = lucid.tensor(sa)
        made[where + "scales_B"] = lucid.tensor(sb)
    return made


def _unilora_forward(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
    """``W' = W + B A`` with ``A`` and ``B`` read from the vector at the file's indices (or the library's draw when the file holds none)."""
    net = _mlp()
    theta = t["base_model.unilora_theta_d"].tolist()
    drawn = _unilora_projection()
    weights = []
    for index in (0, 2):
        where = f"{_PREFIX}{index}.unilora_"
        if where + "indices_A" in t:
            held = tuple(
                t[where + name].tolist()
                for name in ("indices_A", "indices_B", "scales_A", "scales_B")
            )
        else:
            held = drawn[index]
        delta = forms.unilora_delta(theta, *held)
        weights.append(net[index].weight + lucid.tensor(delta))
    return _run(weights, [net[0].bias, net[2].bias], x)


def _unilora_file(save_indices: bool) -> dict[str, tuple[int, ...]]:
    file: dict[str, tuple[int, ...]] = {"base_model.unilora_theta_d": (_UNILORA_LENGTH,)}
    if save_indices:
        for index, (out, inn) in _SIZES.items():
            where = f"{_PREFIX}{index}.unilora_"
            file |= {where + "indices_A": (2, inn), where + "scales_A": (2, inn)}
            file |= {where + "indices_B": (out, 2), where + "scales_B": (out, 2)}
    return file


#: Method rows in the order the methods landed.
_METHODS = [
    _Method(
        "unilora",
        PeftType.UNILORA,
        _unilora(False),
        UniLoraModel,
        UniLoraLayer,
        _unilora_file(False),
        starts_at_base=False,
    ),
    _Method(
        "unilora_indices",
        PeftType.UNILORA,
        _unilora(True),
        UniLoraModel,
        UniLoraLayer,
        _unilora_file(True),
        starts_at_base=False,
        tensors=_unilora_tensors,
    ),
]

CLOSED_FORMS["unilora"] = CLOSED_FORMS["unilora_indices"] = _unilora_forward
_UNILORA_REFERENCE = {
    "auto_mapping": None,
    "base_model_name_or_path": "tiny",
    "bias": "none",
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_theta_d_bound": 0.02,
    "init_weights": True,
    "layers_pattern": None,
    "layers_to_transform": None,
    "modules_to_save": None,
    "peft_type": "UNILORA",
    "peft_version": "0.21.2",
    "proj_seed": 42,
    "r": 2,
    "revision": None,
    "save_indices": False,
    "target_modules": ["0", "2"],
    "task_type": None,
    "theta_d_length": _UNILORA_LENGTH,
    "unilora_dropout": 0.0,
}
_REFERENCE_CONFIGS["unilora"] = _UNILORA_REFERENCE
_REFERENCE_CONFIGS["unilora_indices"] = {**_UNILORA_REFERENCE, "save_indices": True}


@pytest.fixture(params=_METHODS, ids=lambda m: m.name)
def method(request: pytest.FixtureRequest) -> _Method:
    assert isinstance(request.param, _Method)
    return request.param
