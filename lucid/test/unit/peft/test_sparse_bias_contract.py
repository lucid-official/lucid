"""What every tuner on the registry promises (``test_tuner_contract``), held for BEFT, Super-Tuning and WaveFT.

The three are the sparse and bias methods of B4-7.  They are added to the shared contract here, in a
file of their own, by naming the same test classes under a ``method`` fixture of their own: each test
runs on the tiny MLP (CPU and Metal) with the update made non-zero where the test needs one, the
adapter files are written by hand in the reference's layout, and what a loaded file must compute is
worked out from the papers' formulas with plain loops (``_sparse_forms``), never through the layers.
"""

from collections.abc import Callable
from typing import override

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    BeftConfig,
    BeftModel,
    PeftType,
    SupertuningConfig,
    SupertuningModel,
    WaveFTConfig,
    WaveFTModel,
    get_peft_model,
)
from lucid.peft.tuners.beft import BeftLayer
from lucid.peft.tuners.fourierft._locations import _spectrum_positions
from lucid.peft.tuners.supertuning import SupertuningLayer
from lucid.peft.tuners.waveft import WaveFTLayer
from lucid.peft.tuners.waveft._filters import _filter_bank
from lucid.test.unit.peft import _sparse_forms as forms
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


def _supertuning(targets: tuple[str, ...], random: bool, **kw: object) -> SupertuningConfig:
    return SupertuningConfig.from_dict(
        {"sparsity": 0.5, "target_modules": list(targets), "init_weights": not random, **kw}
    )


#: Trained entries of the tiny MLP at sparsity 0.5: half of the 48 and of the 32 entries of its two weights.
_SUPER_COUNTS = {0: 24, 2: 16}
_WEIGHT_SIZES = {0: (8, 6), 2: (4, 8)}


def _supertuning_tensors(rank: int) -> Callable[[], dict[str, lucid.Tensor]]:
    """A file a person could have written: distinct positions (a stride coprime with the weight's size), any values."""

    def tensors() -> dict[str, lucid.Tensor]:
        lucid.manual_seed(7)
        made: dict[str, lucid.Tensor] = {}
        for index, count in _SUPER_COUNTS.items():
            out, inn = _WEIGHT_SIZES[index]
            p = f"{_PREFIX}{index}."
            made[p + "supertuning_values"] = lucid.randint(-2, 3, (count,)).to(lucid.float32) * 0.5
            made[p + "supertuning_indices"] = lucid.tensor(
                [(5 * k) % (out * inn) for k in range(count)], dtype=lucid.int32
            )
            if rank:
                made[p + "supertuning_lora_A.weight"] = (
                    lucid.randint(-2, 3, (rank, inn)).to(lucid.float32) * 0.5
                )
                made[p + "supertuning_lora_B.weight"] = (
                    lucid.randint(-2, 3, (out, rank)).to(lucid.float32) * 0.5
                )
        return made

    return tensors


def _supertuning_file(rank: int) -> dict[str, tuple[int, ...]]:
    file: dict[str, tuple[int, ...]] = {}
    for index, count in _SUPER_COUNTS.items():
        out, inn = _WEIGHT_SIZES[index]
        p = f"{_PREFIX}{index}."
        file[p + "supertuning_values"] = (count,)
        file[p + "supertuning_indices"] = (count,)
        if rank:
            file[p + "supertuning_lora_A.weight"] = (rank, inn)
            file[p + "supertuning_lora_B.weight"] = (out, rank)
    return file


def _supertuning_forward(rank: int, alpha: float) -> Callable[..., lucid.Tensor]:
    def forward(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
        """``W' = W + Delta + (alpha / r) B A``: the file's values laid at its positions, its LoRA pair added."""
        net = _mlp()
        weights = []
        for index in (0, 2):
            p = f"{_PREFIX}{index}."
            weight = forms.scattered(
                net[index].weight.tolist(),
                [int(v) for v in t[p + "supertuning_indices"].tolist()],
                t[p + "supertuning_values"].tolist(),
            )
            if rank:
                weight = forms.low_rank_added(
                    weight,
                    t[p + "supertuning_lora_A.weight"].tolist(),
                    t[p + "supertuning_lora_B.weight"].tolist(),
                    alpha / rank,
                )
            weights.append(lucid.tensor(weight))
        return _run(weights, [net[0].bias, net[2].bias], x)

    return forward


def _waveft(family: str, idwt: bool) -> Callable[..., WaveFTConfig]:
    def config(targets: tuple[str, ...], random: bool, **kw: object) -> WaveFTConfig:
        return WaveFTConfig.from_dict(
            {
                "n_frequency": 6,
                "wavelet_family": family,
                "use_idwt": idwt,
                "target_modules": list(targets),
                "init_weights": not random,
                **kw,
            }
        )

    return config


def _waveft_forward(family: str, idwt: bool) -> Callable[..., lucid.Tensor]:
    def forward(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
        """``W' = W + s IDWT(C)``: the file's coefficients at the seeded permutation's positions, inverse-transformed by the plain sums."""
        net = _mlp()
        bank = _filter_bank(family)
        weights = []
        for index in (0, 2):
            out, inn = _WEIGHT_SIZES[index]
            flat = _spectrum_positions(out * inn, 6, 777)
            delta = forms.waveft_delta(
                (out, inn),
                [p // inn for p in flat],
                [p % inn for p in flat],
                t[f"{_PREFIX}{index}.waveft_spectrum"].tolist(),
                (bank.rec_lo, bank.rec_hi),
                25.0,
                idwt,
            )
            weights.append(net[index].weight + lucid.tensor(delta))
        return _run(weights, [net[0].bias, net[2].bias], x)

    return forward


_WAVEFT_FILE = {f"{_PREFIX}0.waveft_spectrum": (6,), f"{_PREFIX}2.waveft_spectrum": (6,)}


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
    _Method(
        "supertuning",
        PeftType.SUPERTUNING,
        _supertuning,
        SupertuningModel,
        SupertuningLayer,
        _supertuning_file(0),
        tensors=_supertuning_tensors(0),
    ),
    _Method(
        "supertuning_supra",
        PeftType.SUPERTUNING,
        lambda targets, random, **kw: _supertuning(targets, random, r=2, **kw),
        SupertuningModel,
        SupertuningLayer,
        _supertuning_file(2),
        tensors=_supertuning_tensors(2),
    ),
    _Method(
        "waveft", PeftType.WAVEFT, _waveft("db1", True), WaveFTModel, WaveFTLayer, _WAVEFT_FILE
    ),
    _Method(
        "waveft_sym2",
        PeftType.WAVEFT,
        _waveft("sym2", True),
        WaveFTModel,
        WaveFTLayer,
        _WAVEFT_FILE,
    ),
    _Method(
        "waveft_direct",
        PeftType.WAVEFT,
        _waveft("db1", False),
        WaveFTModel,
        WaveFTLayer,
        _WAVEFT_FILE,
    ),
]

CLOSED_FORMS["beft"] = _beft_forward
CLOSED_FORMS["waveft"] = _waveft_forward("db1", True)
CLOSED_FORMS["waveft_sym2"] = _waveft_forward("sym2", True)
CLOSED_FORMS["waveft_direct"] = _waveft_forward("db1", False)
CLOSED_FORMS["supertuning"] = _supertuning_forward(0, 0.0)
CLOSED_FORMS["supertuning_supra"] = _supertuning_forward(2, 4.0)
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
_SUPERTUNING_REFERENCE = {
    "auto_mapping": None,
    "base_model_name_or_path": "tiny",
    "inference_mode": True,
    "init_weights": True,
    "lora_alpha": None,
    "lora_dropout": 0.0,
    "modules_to_save": None,
    "peft_type": "SUPERTUNING",
    "peft_version": "0.21.2",
    "r": None,
    "revision": None,
    "save_precomputed_indices": True,
    "select_top": True,
    "sparsity": 0.5,
    "target_modules": ["0", "2"],
    "task_type": None,
}
_WAVEFT_REFERENCE = {
    "auto_mapping": None,
    "base_model_name_or_path": "tiny",
    "bias": "none",
    "exclude_modules": None,
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_weights": True,
    "layers_pattern": None,
    "layers_to_transform": None,
    "modules_to_save": None,
    "n_frequency": 6,
    "n_frequency_pattern": {},
    "peft_type": "WAVEFT",
    "peft_version": "0.21.2",
    "proportional_parameters": False,
    "random_loc_seed": 777,
    "revision": None,
    "scaling": 25.0,
    "target_modules": ["0", "2"],
    "task_type": None,
    "use_idwt": True,
    "wavelet_family": "db1",
}
_REFERENCE_CONFIGS["waveft"] = _WAVEFT_REFERENCE
_REFERENCE_CONFIGS["waveft_sym2"] = {**_WAVEFT_REFERENCE, "wavelet_family": "sym2"}
_REFERENCE_CONFIGS["waveft_direct"] = {**_WAVEFT_REFERENCE, "use_idwt": False}
_REFERENCE_CONFIGS["supertuning"] = _SUPERTUNING_REFERENCE
_REFERENCE_CONFIGS["supertuning_supra"] = {**_SUPERTUNING_REFERENCE, "r": 2, "lora_alpha": 4}


@pytest.fixture(params=_METHODS, ids=lambda m: m.name)
def method(request: pytest.FixtureRequest) -> _Method:
    assert isinstance(request.param, _Method)
    return request.param
