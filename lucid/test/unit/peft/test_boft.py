"""BOFT: the butterfly in closed form, the shapes, the layers, and the reference adapter library's numbers.

With one butterfly factor and 2x2 blocks the rotation is block-diagonal and each
block is the Cayley transform of ``0.5 (X - X^T)``; for a 2x2 block ``X`` that is
``q = (X[0][1] - X[1][0]) / 2`` and the block ``1/(1+q^2) [[1-q^2, -2q], [2q, 1-q^2]]``.
The tests write that out by hand.  The ``parity`` tests build the same layer in the
reference library (for more than one factor with the block-diagonal stand-in the
reference uses when its compiled kernel is absent), copy the weights across and
compare the forward pass, the rotation and the merged weight.
"""

from types import ModuleType
from typing import Any

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import BOFTConfig, BOFTModel, PeftModel, get_peft_model
from lucid.peft.tuners import boft
from lucid.peft.tuners.boft._butterfly import _Butterfly
from lucid.test.unit.peft._parity import adapter_library, gap


def _cayley(q: float) -> list[list[float]]:
    d = 1 + q * q
    return [[(1 - q * q) / d, -2 * q / d], [2 * q / d, (1 - q * q) / d]]


def _layer(
    in_features: int = 8, out_features: int = 6, **kw: object
) -> tuple[boft.Linear, nn.Linear]:
    lucid.manual_seed(1)
    base = nn.Linear(in_features, out_features)
    fields: dict[str, object] = {"boft_block_size": 2, "target_modules": ["x"]} | kw
    return boft.Linear(base, "default", BOFTConfig(**fields)), base  # type: ignore[arg-type]


class TestConfig:
    def test_the_defaults_are_the_references(self) -> None:
        config = BOFTConfig(target_modules=["a"])
        assert (config.boft_block_size, config.boft_block_num) == (4, 0)
        assert (config.boft_n_butterfly_factor, config.boft_dropout) == (1, 0.0)
        assert config.init_weights and config.bias == "none" and not config.fan_in_fan_out

    def test_exactly_one_of_block_size_and_count(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            BOFTConfig(boft_block_num=2, target_modules=["a"])
        with pytest.raises(ValueError, match="must be non-zero"):
            BOFTConfig(boft_block_size=0, target_modules=["a"])
        assert (
            BOFTConfig(boft_block_size=0, boft_block_num=2, target_modules=["a"]).boft_block_num
            == 2
        )

    def test_fields_are_checked(self) -> None:
        with pytest.raises(ValueError, match="boft_n_butterfly_factor"):
            BOFTConfig(boft_n_butterfly_factor=0, target_modules=["a"])
        with pytest.raises(ValueError, match="boft_dropout"):
            BOFTConfig(boft_dropout=1.0, target_modules=["a"])

    def test_trained_biases_are_refused(self) -> None:
        with pytest.raises(NotImplementedError, match="boft_only"):
            BOFTConfig(bias="boft_only", target_modules=["a"])

    def test_a_file_round_trips(self) -> None:
        config = BOFTConfig(boft_block_size=2, boft_n_butterfly_factor=2, target_modules=["q"])
        assert BOFTConfig.from_dict(config.to_dict()) == config
        assert config.to_dict()["peft_type"] == "BOFT"


class TestButterfly:
    def test_the_permutations_interleave_runs_of_half_a_block(self) -> None:
        assert _Butterfly.permutation(8, 2, 2, 1) == list(
            range(8)
        )  # a block of 4 is two runs: kept
        assert _Butterfly.permutation(8, 1, 2, 1) == [0, 1, 4, 5, 2, 3, 6, 7]
        assert _Butterfly.permutation(8, 1, 2, 0) == list(range(8))  # one factor: no permutation

    def test_blocks_that_do_not_fit_are_refused(self) -> None:
        with pytest.raises(ValueError, match="Invalid number of blocks"):
            _Butterfly.permutation(8, 4, 2, 1)

    def test_the_parameter_shapes_follow_the_factors_and_the_layer_type(self) -> None:
        layer, _ = _layer(boft_n_butterfly_factor=2)
        assert tuple(layer.boft_R["default"].shape) == (2, 4, 2, 2)
        assert tuple(layer.boft_s["default"].shape) == (6, 1)
        assert layer.boft_block_size == {"default": 2} and layer.boft_block_num == {"default": 4}
        conv = boft.Conv2d(
            nn.Conv2d(2, 5, 2), "default", BOFTConfig(boft_block_size=2, target_modules=["x"])
        )
        assert tuple(conv.boft_s["default"].shape) == (1, 5)

    @pytest.mark.parametrize(
        ("options", "message"),
        [
            ({"boft_block_size": 3}, "divisible"),
            ({"boft_block_size": 8, "boft_n_butterfly_factor": 2}, "Invalid combination"),
            ({"boft_block_size": 2, "boft_n_butterfly_factor": 4}, "Invalid combination"),
            (
                {"boft_block_size": 0, "boft_block_num": 8, "boft_n_butterfly_factor": 5},
                "Invalid combination",
            ),
        ],
    )
    def test_sizes_that_do_not_fit_the_input_are_refused(
        self, options: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            _layer(**options)

    def test_a_count_of_blocks_is_enough(self) -> None:
        layer, _ = _layer(boft_block_size=0, boft_block_num=4, boft_n_butterfly_factor=2)
        assert layer.boft_block_size == {"default": 2}


class TestRotation:
    def test_one_factor_is_a_block_diagonal_cayley_rotation_and_a_row_scale(
        self, device: str
    ) -> None:
        layer, base = _layer()
        layer.to(device)
        x_entries = [
            [[0.0, 0.7], [0.1, 0.0]],
            [[0.0, -0.4], [0.2, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
            [[0.0, 1.0], [-1.0, 0.0]],
        ]
        scale = [0.5, 1.5, 2.0, 1.0, -1.0, 0.25]
        with lucid.no_grad():
            layer.boft_R["default"].copy_(lucid.tensor([x_entries], device=device))
            layer.boft_s["default"].copy_(lucid.tensor(scale, device=device).reshape(6, 1))
        qs = [0.5 * (m[0][1] - m[1][0]) for m in x_entries]
        dense = [[0.0] * 8 for _ in range(8)]
        for k, q in enumerate(qs):
            for i in range(2):
                for j in range(2):
                    dense[2 * k + i][2 * k + j] = _cayley(q)[i][j]
        rotation = lucid.tensor(dense, device=device)
        assert bool(lucid.allclose(layer.get_delta_weight("default"), rotation, atol=1e-6))
        x = lucid.randn(3, 8, device=device)
        # the bias is not scaled: y = s * (W (x M)) + b
        expected = ((x @ rotation) @ base.weight.mT) * lucid.tensor(
            scale, device=device
        ) + base.bias
        assert bool(lucid.allclose(layer(x), expected, atol=1e-5))

    def test_the_default_start_is_the_identity_with_unit_scale(self) -> None:
        layer, base = _layer(boft_n_butterfly_factor=2)
        x = lucid.randn(4, 8)
        assert bool(lucid.allclose(layer.get_delta_weight("default"), lucid.eye(8), atol=1e-6))
        assert bool(lucid.allclose(layer(x), base(x), atol=1e-6))

    @pytest.mark.parametrize("factors", [1, 2, 3])
    def test_the_matrix_is_orthogonal_and_merging_preserves_the_function(
        self, factors: int
    ) -> None:
        layer, base = _layer(boft_n_butterfly_factor=factors, init_weights=False)
        m = layer.get_delta_weight("default")
        assert bool(lucid.allclose(m @ m.mT, lucid.eye(8), atol=1e-5))
        weight = base.weight.clone()
        x = lucid.randn(3, 8)
        with lucid.no_grad():
            adapted = layer(x)
        layer.merge()
        assert bool(lucid.allclose(layer(x), adapted, atol=1e-5))
        layer.unmerge()
        assert bool(lucid.allclose(base.weight, weight, atol=1e-5))

    def test_block_dropout_acts_only_while_training(self) -> None:
        layer, base = _layer(boft_dropout=0.5, init_weights=False)
        x = lucid.randn(3, 8)
        layer.eval()
        with lucid.no_grad():
            quiet = layer(x)
            layer.train()
            lucid.manual_seed(0)
            noisy = layer(x)
        assert not bool(lucid.allclose(quiet, noisy, atol=1e-3))


class TestConv2d:
    def test_merged_equals_unmerged(self) -> None:
        lucid.manual_seed(2)
        base = nn.Conv2d(4, 5, 3, padding=1)
        layer = boft.Conv2d(
            base, "default", BOFTConfig(boft_block_size=4, init_weights=False, target_modules=["x"])
        )
        assert tuple(layer.boft_R["default"].shape) == (1, 9, 4, 4)
        x = lucid.randn(2, 4, 6, 6)
        with lucid.no_grad():
            adapted = layer(x)
        layer.merge()
        assert bool(lucid.allclose(layer(x), adapted, atol=1e-5))

    def test_a_grouped_convolution_is_refused(self) -> None:
        with pytest.raises(NotImplementedError, match="groups > 1"):
            boft.Conv2d(nn.Conv2d(4, 4, 3, groups=2), "x", BOFTConfig(target_modules=["x"]))


def test_the_tuner_adds_a_second_adapter_beside_the_first() -> None:
    net = nn.Sequential(nn.Linear(8, 8))
    peft = get_peft_model(net, BOFTConfig(boft_block_size=2, target_modules=["0"]), "a")
    assert isinstance(peft, PeftModel) and isinstance(peft.base_model, BOFTModel)
    peft.add_adapter(
        "b", BOFTConfig(boft_block_size=4, boft_n_butterfly_factor=2, target_modules=["0"])
    )
    assert peft.base_model.model[0].boft_block_size == {"a": 2, "b": 4}


# ── the reference adapter library ────────────────────────────────────────────


class _ReferenceBlockDiagonal:
    """What the reference library uses in place of its compiled block-diagonal kernel when that is absent."""

    def __init__(self, ref: ModuleType) -> None:
        self._ref = ref

    def forward(self, blocks: Any) -> list[Any]:
        ref = self._ref
        stacked = ref.stack([ref.block_diag(*blocks[i].unbind(0)) for i in range(blocks.shape[0])])
        return [stacked]


@pytest.mark.parity
@pytest.mark.parametrize(
    ("options", "conv"),
    [
        ({"boft_block_size": 4}, False),
        ({"boft_block_size": 0, "boft_block_num": 2}, False),
        ({"boft_block_size": 4, "boft_n_butterfly_factor": 2}, False),
        ({"boft_block_size": 2, "boft_n_butterfly_factor": 3}, False),
        ({"boft_block_size": 4}, True),
        ({"boft_block_size": 4, "boft_n_butterfly_factor": 2}, True),
    ],
    ids=["size", "count", "two-factors", "three-factors", "conv", "conv-two-factors"],
)
def test_a_layer_matches_the_reference(
    ref: ModuleType, monkeypatch: pytest.MonkeyPatch, options: dict[str, object], conv: bool
) -> None:
    library = adapter_library()
    layer_module = pytest.importorskip("peft.tuners.boft.layer")
    stand_in = _ReferenceBlockDiagonal(ref)
    monkeypatch.setattr(layer_module, "_FBD_CUDA", None)
    monkeypatch.setattr(layer_module, "get_fbd_cuda", lambda: stand_in)
    lucid.manual_seed(6)
    if conv:
        net = nn.Sequential(nn.Conv2d(4, 5, 2, padding=1))
        twin = ref.nn.Sequential(ref.nn.Conv2d(4, 5, 2, padding=1))
        x = lucid.randn(2, 4, 6, 6)
    else:
        net = nn.Sequential(nn.Linear(16, 6))
        twin = ref.nn.Sequential(ref.nn.Linear(16, 6))
        x = lucid.randn(4, 16)
    peft = get_peft_model(net, BOFTConfig(target_modules=["0"], init_weights=False, **options))  # type: ignore[arg-type]
    mine = peft.base_model.model[0]
    with ref.no_grad():
        twin[0].weight.copy_(ref.tensor(mine.base_layer.weight.tolist()))
        twin[0].bias.copy_(ref.tensor(mine.base_layer.bias.tolist()))
    theirs = library.get_peft_model(
        twin, library.BOFTConfig(target_modules=["0"], init_weights=False, **options)
    )
    other = theirs.base_model.model[0]
    with ref.no_grad():
        other.boft_R["default"].copy_(ref.tensor(mine.boft_R["default"].tolist()))
        other.boft_s["default"].copy_(ref.tensor(mine.boft_s["default"].tolist()))
    assert tuple(mine.boft_R["default"].shape) == tuple(other.boft_R["default"].shape)
    assert gap(ref, peft(x), theirs(ref.tensor(x.tolist()))) < 1e-5
    assert gap(ref, mine.get_delta_weight("default"), other.get_delta_weight("default")[0]) < 1e-5
    peft.base_model.merge_adapter()
    theirs.base_model.merge_adapter()
    assert gap(ref, mine.base_layer.weight, other.base_layer.weight) < 1e-5
    peft.base_model.unmerge_adapter()
    theirs.base_model.unmerge_adapter()
    assert gap(ref, mine.base_layer.weight, other.base_layer.weight) < 1e-5


@pytest.mark.parity
def test_the_config_file_has_the_references_keys() -> None:
    library = adapter_library()
    ours = BOFTConfig(boft_block_size=2, target_modules=["a"]).to_dict()
    theirs = library.BOFTConfig(boft_block_size=2, target_modules=["a"]).to_dict()
    assert set(ours) == set(theirs) - {"peft_version"}
    assert all(
        ours[k] == (sorted(v) if isinstance(v, set) else v) for k, v in theirs.items() if k in ours
    )
