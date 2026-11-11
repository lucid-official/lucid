"""What every tuner on the registry promises (``test_tuner_contract``), held for AdaMSS.

AdaMSS is added to the shared contract here, in a file of its own, by naming the same test classes under
a ``method`` fixture of its own: each test runs on the tiny MLP (CPU and Metal) with the update made
non-zero where the test needs one, and the adapter files are written by hand in the reference's layout.
What a loaded file must compute is worked out from the paper's formulas with plain loops
(``_adamss_forms``), never through the layers.

Two rows: one subspace, whose split is the identity whatever the weights are, and two subspaces, whose
file shapes and row order are read off the split the tiny MLP's weights make.  A file does not say which
rows a subspace holds, so what a loaded file computes is worked out with the row order of the model it
is loaded into.
"""

from collections.abc import Callable

import pytest

import lucid
from lucid.peft import AdamssConfig, AdamssModel, PeftType, get_peft_model
from lucid.peft.tuners._layer import _tuner_layers
from lucid.peft.tuners.adamss import AdamssLayer
from lucid.test.unit.peft import _adamss_forms as forms
from lucid.test.unit.peft._closed_forms import CLOSED_FORMS, _PREFIX, _mlp
from lucid.test.unit.peft.test_orthogonal_spectral_contract import TestForward
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

#: (out, in) of the two layers of ``_mlp`` and the rank of the SVD of each (``r = 3`` of its joined weight).
_SIZES = {0: (8, 6), 2: (4, 8)}
_RANK = 3


def _adamss(subspaces: int) -> Callable[..., AdamssConfig]:
    def config(targets: tuple[str, ...], random: bool, **kw: object) -> AdamssConfig:
        return AdamssConfig.from_dict(
            {
                "r": _RANK,
                "num_subspaces": subspaces,
                "subspace_rank": 2,
                "target_modules": list(targets),
                "init_weights": None if random else "orthogonal",
                **kw,
            }
        )

    return config


def _split_of(subspaces: int) -> dict[int, tuple[list[list[int]], list[int]]]:
    """For each layer of ``_mlp``: the rows of each subspace and the rank of each, as the layer makes them."""
    peft = get_peft_model(_mlp(), _adamss(subspaces)(("0", "2"), False))
    found: dict[int, tuple[list[list[int]], list[int]]] = {}
    for path, layer in _tuner_layers(peft):
        assert isinstance(layer, AdamssLayer)
        order = [int(v) for v in layer.scatter_index["default"].tolist()]
        sizes = [int(b.shape[0]) for b in layer.adamss_B["default"].parameters()]
        ranks = [int(b.shape[1]) for b in layer.adamss_B["default"].parameters()]
        cuts, start = [], 0
        for size in sizes:
            cuts.append(order[start : start + size])
            start += size
        found[int(path.removeprefix("base_model.model."))] = (cuts, ranks)
    return found


def _file(subspaces: int) -> dict[str, tuple[int, ...]]:
    file: dict[str, tuple[int, ...]] = {}
    for index, (cuts, ranks) in _split_of(subspaces).items():
        p = f"{_PREFIX}{index}."
        file[p + "adamss_newB"] = (_RANK, _SIZES[index][1] + 1)
        for k, (rows, rank) in enumerate(zip(cuts, ranks, strict=True)):
            file[p + f"adamss_A.{k}"] = (rank, _RANK)
            file[p + f"adamss_B.{k}"] = (len(rows), rank)
    return file


def _tensors(subspaces: int) -> Callable[[], dict[str, lucid.Tensor]]:
    """A file a person could have written: small integer-valued numbers under the file's keys."""

    def tensors() -> dict[str, lucid.Tensor]:
        lucid.manual_seed(7)
        return {
            key: lucid.randint(-2, 3, shape).to(lucid.float32) * 0.5
            for key, shape in _file(subspaces).items()
        }

    return tensors


def _forward(subspaces: int) -> Callable[..., lucid.Tensor]:
    def forward(t: dict[str, lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
        """``y = (W + b) [x; 1] + Pi [B_k A_k] V^T [x; 1]`` on both layers, the rows of each subspace from the model the file loads into."""
        net = _mlp()
        split = _split_of(subspaces)
        hidden = x.tolist()
        for index in (0, 2):
            p = f"{_PREFIX}{index}."
            cuts, ranks = split[index]
            delta = forms.joined_delta(
                cuts,
                [t[p + f"adamss_A.{k}"].tolist() for k in range(len(cuts))],
                [t[p + f"adamss_B.{k}"].tolist() for k in range(len(cuts))],
                t[p + "adamss_newB"].tolist(),
                _SIZES[index][0],
            )
            hidden = forms.layer(
                hidden, net[index].weight.tolist(), net[index].bias.tolist(), delta
            )
            if index == 0:
                hidden = [[max(v, 0.0) for v in row] for row in hidden]
        return lucid.tensor(hidden)

    return forward


_METHODS: list[_Method] = [
    _Method(
        "adamss_one_subspace",
        PeftType.ADAMSS,
        _adamss(1),
        AdamssModel,
        AdamssLayer,
        _file(1),
        tensors=_tensors(1),
        random_start=False,
    ),
    _Method(
        "adamss_two_subspaces",
        PeftType.ADAMSS,
        _adamss(2),
        AdamssModel,
        AdamssLayer,
        _file(2),
        tensors=_tensors(2),
        random_start=False,
    ),
]

CLOSED_FORMS["adamss_one_subspace"] = _forward(1)
CLOSED_FORMS["adamss_two_subspaces"] = _forward(2)
_ADAMSS_REFERENCE = {
    "asa_importance_beta": 0.85,
    "asa_schedule_exponent": 3.0,
    "asa_target_subspaces": 50,
    "asa_uncertainty_beta": 0.85,
    "auto_mapping": None,
    "base_model_name_or_path": "tiny",
    "final_warmup": 1000,
    "inference_mode": True,
    "init_warmup": 50,
    "init_weights": "orthogonal",
    "layers_pattern": None,
    "layers_to_transform": None,
    "mask_interval": 100,
    "modules_to_save": None,
    "num_subspaces": 1,
    "peft_type": "ADAMSS",
    "peft_version": "0.21.2",
    "r": _RANK,
    "random_seed": 0,
    "revision": None,
    "subspace_rank": 2,
    "svd_threshold": 0.1,
    "target_modules": ["0", "2"],
    "task_type": None,
    "use_asa": False,
    "use_dynamic_rank": False,
}
_REFERENCE_CONFIGS["adamss_one_subspace"] = _ADAMSS_REFERENCE
_REFERENCE_CONFIGS["adamss_two_subspaces"] = {**_ADAMSS_REFERENCE, "num_subspaces": 2}


@pytest.fixture(params=_METHODS, ids=lambda m: m.name)
def method(request: pytest.FixtureRequest) -> _Method:
    assert isinstance(request.param, _Method)
    return request.param
