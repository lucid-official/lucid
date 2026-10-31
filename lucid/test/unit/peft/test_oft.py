"""OFT: the rotation closed forms, the options, the layers, and the reference adapter library's numbers.

The block rotation of an adapter is the Cayley transform of a skew-symmetric
matrix.  For a 2x2 block with entry ``q`` that transform is
``1/(1+q^2) [[1-q^2, -2q], [2q, 1-q^2]]``, which is what the tests expect, written
out by hand.  The ``parity`` tests build the same layer in the reference
library, copy the weights across and compare the forward pass, the rotation
and the merged weight.
"""

import warnings
from types import ModuleType
from typing import Any

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import OFTConfig, OFTModel, PeftModel, get_peft_model
from lucid.peft.tuners import oft
from lucid.test.unit.peft._parity import adapter_library, gap


def _cayley(q: float) -> list[list[float]]:
    d = 1 + q * q
    return [[(1 - q * q) / d, -2 * q / d], [2 * q / d, (1 - q * q) / d]]


def _layer(
    in_features: int = 8, out_features: int = 6, **kw: object
) -> tuple[oft.Linear, nn.Linear]:
    lucid.manual_seed(1)
    base = nn.Linear(in_features, out_features)
    fields: dict[str, object] = {"oft_block_size": 2, "target_modules": ["x"]} | kw
    return oft.Linear(base, "default", OFTConfig(**fields)), base  # type: ignore[arg-type]


def _set_blocks(layer: oft.Linear, values: list[float]) -> None:
    with lucid.no_grad():
        layer.oft_R["default"].copy_(lucid.tensor(values).reshape(layer.oft_R["default"].shape))


class TestConfig:
    def test_the_defaults_are_the_references(self) -> None:
        config = OFTConfig(target_modules=["a"])
        assert (config.r, config.oft_block_size, config.module_dropout) == (0, 32, 0.0)
        assert (config.coft, config.eps, config.block_share) == (False, 6e-5, False)
        assert (config.use_cayley_neumann, config.num_cayley_neumann_terms) == (True, 5)
        assert config.init_weights and config.bias == "none" and not config.fan_in_fan_out

    def test_exactly_one_of_r_and_block_size(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            OFTConfig(r=4, target_modules=["a"])
        with pytest.raises(ValueError, match="must be non-zero"):
            OFTConfig(r=0, oft_block_size=0, target_modules=["a"])
        assert OFTConfig(r=4, oft_block_size=0, target_modules=["a"]).r == 4

    def test_fields_are_checked(self) -> None:
        with pytest.raises(ValueError, match="module_dropout"):
            OFTConfig(module_dropout=1.0, target_modules=["a"])
        with pytest.raises(TypeError, match="coft"):
            OFTConfig(coft=1, target_modules=["a"])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="num_cayley_neumann_terms"):
            OFTConfig(num_cayley_neumann_terms=0, target_modules=["a"])

    def test_trained_biases_are_refused(self) -> None:
        with pytest.raises(NotImplementedError, match="oft_only"):
            OFTConfig(bias="oft_only", target_modules=["a"])

    def test_a_file_round_trips(self, tmp_path: object) -> None:
        config = OFTConfig(oft_block_size=4, coft=True, target_modules=["q", "v"])
        assert OFTConfig.from_dict(config.to_dict()) == config
        assert config.to_dict()["peft_type"] == "OFT"


class TestRotation:
    def test_the_blocks_are_the_cayley_transform_of_the_skew_entries(self, device: str) -> None:
        layer, base = _layer(use_cayley_neumann=False)
        layer.to(device)
        q = [0.3, -0.5, 0.0, 1.2]
        _set_blocks(layer, q)
        rotation = layer.get_delta_weight("default")
        dense = [[0.0] * 8 for _ in range(8)]
        for k, v in enumerate(q):
            for i in range(2):
                for j in range(2):
                    dense[2 * k + i][2 * k + j] = _cayley(v)[i][j]
        assert bool(lucid.allclose(rotation.to("cpu"), lucid.tensor(dense), atol=1e-6))

    def test_the_rotation_is_orthogonal_and_the_neumann_series_nearly_so(self) -> None:
        exact, _ = _layer(use_cayley_neumann=False)
        series, _ = _layer(num_cayley_neumann_terms=8)
        small = [0.05, -0.04, 0.02, 0.03]
        _set_blocks(exact, small)
        _set_blocks(series, small)
        r = exact.get_delta_weight("default")
        assert bool(lucid.allclose(r @ r.mT, lucid.eye(8), atol=1e-6))
        # The series is (I + Q)(I - Q)^-1, the transpose of the exact transform: the reference's own convention.
        assert bool(lucid.allclose(series.get_delta_weight("default"), r.mT, atol=1e-6))

    def test_a_forward_pass_rotates_the_input_by_the_blocks(self) -> None:
        layer, base = _layer()
        _set_blocks(layer, [0.3, -0.5, 0.7, 0.1])
        x = lucid.randn(3, 8)
        rotation = layer.get_delta_weight("default")
        assert bool(lucid.allclose(layer(x), base(x @ rotation), atol=1e-5))

    def test_merging_gives_the_rotated_weight_and_unmerging_gives_it_back(self) -> None:
        layer, base = _layer()
        _set_blocks(layer, [0.3, -0.5, 0.7, 0.1])
        weight = base.weight.clone()
        x = lucid.randn(3, 8)
        adapted = layer(x)
        layer.merge()
        assert bool(lucid.allclose(base.weight, weight @ layer.get_delta_weight("default").mT))
        assert bool(lucid.allclose(layer(x), adapted, atol=1e-5))
        layer.unmerge()
        assert bool(lucid.allclose(base.weight, weight, atol=1e-5))

    def test_block_sharing_uses_one_block_everywhere(self) -> None:
        layer, base = _layer(block_share=True)
        assert tuple(layer.oft_R["default"].shape) == (1, 1)
        _set_blocks(layer, [0.4])
        rotation = layer.get_delta_weight("default")
        first = rotation[:2, :2]
        for k in range(4):
            assert bool(lucid.allclose(rotation[2 * k : 2 * k + 2, 2 * k : 2 * k + 2], first))

    def test_coft_pulls_each_block_into_its_ball(self) -> None:
        layer, _ = _layer(coft=True, eps=0.01)
        _set_blocks(layer, [3.0, -4.0, 0.001, 0.0])
        layer.get_delta_weight("default")
        radius = 0.01 / 2  # eps over the square root of the 4 blocks
        stored = layer.oft_R["default"].reshape(-1).tolist()
        norms = [abs(v) * 2**0.5 for v in stored]  # |Q|_F of [[0, q], [-q, 0]]
        assert all(n <= radius * (1 + 1e-5) for n in norms)
        assert stored[2] == pytest.approx(0.001) and stored[3] == 0.0  # inside the ball: untouched

    def test_block_dropout_swaps_blocks_for_the_identity_only_while_training(self) -> None:
        layer, base = _layer(module_dropout=0.5)
        _set_blocks(layer, [0.9, 0.9, 0.9, 0.9])
        x = lucid.randn(3, 8)
        layer.eval()
        assert bool(
            lucid.allclose(layer(x), base(x @ layer.get_delta_weight("default")), atol=1e-5)
        )
        layer.train()
        lucid.manual_seed(0)
        out = layer(x)
        rotated = base(x @ layer.get_delta_weight("default"))
        assert not bool(
            lucid.allclose(out, rotated, atol=1e-3)
        )  # 2 of 4 blocks became the identity

    def test_an_unfit_block_size_moves_to_the_nearest_divisor(self) -> None:
        with pytest.warns(UserWarning, match="Adjusted `oft_block_size` to \\(4\\)"):
            layer, _ = _layer(in_features=12, oft_block_size=5)
        assert layer.oft_block_size["default"] == 4 and layer.r["default"] == 3
        with pytest.warns(UserWarning, match="Adjusted `r` to \\(4\\)"):
            layer, _ = _layer(in_features=12, r=5, oft_block_size=0)
        assert layer.r["default"] == 4


class TestConv2d:
    def _conv(self, kernel: int, **kw: object) -> tuple[oft.Conv2d, nn.Conv2d]:
        lucid.manual_seed(2)
        base = nn.Conv2d(4, 5, kernel, padding=kernel // 2)
        fields: dict[str, object] = {"oft_block_size": 4, "target_modules": ["x"]} | kw
        return oft.Conv2d(base, "default", OFTConfig(**fields)), base  # type: ignore[arg-type]

    def test_the_rotation_acts_on_the_flattened_in_by_kernel_axis(self) -> None:
        layer, _ = self._conv(3)
        assert layer.in_features == 36 and tuple(layer.oft_R["default"].shape) == (9, 6)

    @pytest.mark.parametrize("kernel", [1, 3])
    def test_merged_equals_unmerged(self, kernel: int) -> None:
        layer, base = self._conv(kernel)
        with lucid.no_grad():
            layer.oft_R["default"].copy_(0.3 * lucid.randn(*layer.oft_R["default"].shape))
        x = lucid.randn(2, 4, 6, 6)
        with lucid.no_grad():  # a live graph holds the weight, which merging then may not write
            adapted = layer(x)
        layer.merge()
        assert bool(lucid.allclose(layer(x), adapted, atol=1e-5))
        layer.unmerge()
        assert bool(lucid.allclose(layer(x), adapted, atol=1e-5))

    def test_a_grouped_convolution_is_refused(self) -> None:
        with pytest.raises(NotImplementedError, match="groups > 1"):
            oft.Conv2d(nn.Conv2d(4, 4, 3, groups=2), "x", OFTConfig(target_modules=["x"]))


class TestModel:
    def test_one_adapter_per_name_and_weighted_state_stays_apart(self) -> None:
        net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
        peft = get_peft_model(net, OFTConfig(oft_block_size=4, target_modules=["0"]), "a")
        assert isinstance(peft, PeftModel) and isinstance(peft.base_model, OFTModel)
        peft.add_adapter("b", OFTConfig(r=2, oft_block_size=0, target_modules=["0"]))
        layer = peft.base_model.model[0]
        assert layer.r == {"a": 2, "b": 2} and layer.oft_block_size == {"a": 4, "b": 4}

    def test_fan_in_fan_out_is_reset_for_a_linear_layer(self) -> None:
        net = nn.Sequential(nn.Linear(8, 8))
        with pytest.warns(UserWarning, match="fan_in_fan_out"):
            peft = get_peft_model(
                net, OFTConfig(oft_block_size=4, fan_in_fan_out=True, target_modules=["0"])
            )
        assert peft.peft_config["default"].fan_in_fan_out is False


# ── the reference adapter library ────────────────────────────────────────────


def _twin(
    ref: ModuleType,
    library: ModuleType,
    layer: oft.OFTLayer,
    shape: tuple[int, ...] | None,
    **kw: object,
) -> Any:
    """The same single-module model in the reference library, weights copied from ``layer``."""
    base = layer.base_layer
    if isinstance(base, nn.Linear):
        net = ref.nn.Sequential(ref.nn.Linear(base.in_features, base.out_features))
    else:
        assert isinstance(base, nn.Conv2d) and isinstance(base.padding, tuple)
        net = ref.nn.Sequential(
            ref.nn.Conv2d(
                base.in_channels, base.out_channels, base.kernel_size, padding=base.padding
            )
        )
    with ref.no_grad():
        net[0].weight.copy_(ref.tensor(base.weight.tolist()))
        net[0].bias.copy_(ref.tensor(base.bias.tolist()))
    config = library.OFTConfig(target_modules=["0"], **kw)
    wrapped = library.get_peft_model(net, config)
    with ref.no_grad():
        wrapped.base_model.model[0].oft_R["default"].weight.copy_(
            ref.tensor(layer.oft_R["default"].tolist())
        )
    return wrapped


@pytest.mark.parity
@pytest.mark.parametrize(
    "options",
    [
        {"oft_block_size": 4},
        {"oft_block_size": 4, "use_cayley_neumann": False},
        {"r": 2, "oft_block_size": 0},
        {"oft_block_size": 4, "block_share": True},
        {"oft_block_size": 4, "coft": True, "eps": 0.01},
    ],
    ids=["neumann", "exact", "by-count", "shared", "coft"],
)
def test_a_linear_layer_matches_the_reference(ref: ModuleType, options: dict[str, object]) -> None:
    library = adapter_library()
    lucid.manual_seed(3)
    peft = get_peft_model(
        nn.Sequential(nn.Linear(8, 6)),
        OFTConfig(target_modules=["0"], init_weights=False, **options),  # type: ignore[arg-type]
    )
    mine = peft.base_model.model[0]
    theirs = _twin(ref, library, mine, None, init_weights=False, **options)
    twin_layer = theirs.base_model.model[0]
    x = lucid.randn(5, 8)
    assert gap(ref, peft(x), theirs(ref.tensor(x.tolist()))) < 1e-5
    assert gap(ref, mine.get_delta_weight("default"), twin_layer.get_delta_weight("default")) < 1e-5
    peft.base_model.merge_adapter()
    theirs.base_model.merge_adapter()
    assert gap(ref, mine.base_layer.weight, twin_layer.base_layer.weight) < 1e-5


@pytest.mark.parity
def test_a_pointwise_convolution_matches_the_reference_and_a_wide_one_its_merged_weight(
    ref: ModuleType,
) -> None:
    library = adapter_library()
    for kernel in (1, 3):
        lucid.manual_seed(4)
        conv = nn.Sequential(nn.Conv2d(4, 5, kernel, padding=kernel // 2))
        peft = get_peft_model(
            conv, OFTConfig(oft_block_size=4, target_modules=["0"], init_weights=False)
        )
        mine = peft.base_model.model[0]
        theirs = _twin(ref, library, mine, None, oft_block_size=4, init_weights=False)
        twin_layer = theirs.base_model.model[0]
        x = lucid.randn(2, 4, 6, 6)
        if kernel == 1:  # the reference's patch-wise forward is the filter rotation only for 1x1
            assert gap(ref, peft(x), theirs(ref.tensor(x.tolist()))) < 1e-5
        peft.base_model.merge_adapter()
        theirs.base_model.merge_adapter()
        assert gap(ref, mine.base_layer.weight, twin_layer.base_layer.weight) < 1e-5


@pytest.mark.parity
def test_the_config_file_has_the_references_keys() -> None:
    library = adapter_library()
    ours = OFTConfig(oft_block_size=4, target_modules=["a"]).to_dict()
    theirs = library.OFTConfig(oft_block_size=4, target_modules=["a"]).to_dict()
    assert set(ours) == set(theirs) - {"peft_version"}
    assert all(
        ours[k] == (sorted(v) if isinstance(v, set) else v) for k, v in theirs.items() if k in ours
    )


def test_warnings_stay_quiet_on_a_fitting_block_size() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _layer(oft_block_size=4)
