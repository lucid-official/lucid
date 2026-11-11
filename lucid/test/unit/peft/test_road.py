"""RoAd: the rotation in closed form, the starts, the merge algebra, the layers and the reference adapter library's numbers.

The paper's matrix (Eq. 3, Table 1) is written out entry by entry in ``_spectral_forms``; the tests
expect the layer to be that matrix applied to the base layer's output, ``y' = R (W x + b)``, and its
merged weight and bias to be ``R W`` and ``R b``.  The ``parity`` tests build the same layer in the
reference library, copy the parameters across and compare the forward pass, the matrix and the merged
weight and bias.
"""

from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import PeftModel, RoadConfig, RoadModel, get_peft_model
from lucid.peft.tuners import road
from lucid.test.unit.peft import _spectral_forms as forms
from lucid.test.unit.peft._parity import adapter_library, gap
from lucid.test.unit.peft._tuner_recipes import magnitude, span

_VARIANTS = ["road_1", "road_2", "road_4"]
#: Relative to the largest value compared: a float32 product and the inverse of a block round.
_TOLERANCE = 1e-5


def _layer(out_features: int = 8, **kw: object) -> tuple[road.Linear, nn.Linear]:
    lucid.manual_seed(1)
    base = nn.Linear(4, out_features)
    fields: dict[str, object] = {"group_size": 4, "target_modules": ["x"]} | kw
    return road.Linear(base, "default", RoadConfig(**fields)), base  # type: ignore[arg-type]


def _norms(rows: lucid.Tensor) -> lucid.Tensor:
    return (rows * rows).sum(dim=-1).sqrt()


def _move(layer: road.RoadLayer, seed: int = 3) -> None:
    """Move the angles and scales well off their starts, as training would."""
    lucid.manual_seed(seed)
    with lucid.no_grad():
        for p in layer._adapter_parameters("default"):
            p.copy_(p + 0.3 * lucid.randn(*p.shape, device=p.device))


def _closed_matrix(layer: road.RoadLayer, adapter: str = "default") -> lucid.Tensor:
    spec = layer._spec(adapter)
    return lucid.tensor(
        forms.road_matrix(
            spec.variant,
            spec.group_size,
            layer.road_theta[adapter].tolist(),
            layer.road_alpha[adapter].tolist(),
        )
    )


class TestConfig:
    def test_the_defaults_are_the_references(self) -> None:
        config = RoadConfig(target_modules=["a"])
        assert (config.variant, config.group_size, config.init_weights) == ("road_1", 64, True)

    def test_fields_are_checked(self) -> None:
        with pytest.raises(ValueError, match="road_1"):
            RoadConfig(variant="road_3", target_modules=["a"])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="divisible by 2"):
            RoadConfig(group_size=3, target_modules=["a"])
        with pytest.raises(ValueError, match="positive"):
            RoadConfig(group_size=0, target_modules=["a"])
        with pytest.raises(TypeError, match="init_weights"):
            RoadConfig(init_weights="yes", target_modules=["a"])  # type: ignore[arg-type]

    def test_a_file_round_trips(self) -> None:
        config = RoadConfig(variant="road_4", group_size=8, target_modules=["q", "v"])
        assert RoadConfig.from_dict(config.to_dict()) == config
        assert config.to_dict()["peft_type"] == "ROAD"


class TestStarts:
    @pytest.mark.parametrize("variant", _VARIANTS)
    def test_angle_zero_and_scale_one_start_as_the_identity(
        self, variant: str, device: str
    ) -> None:
        layer, base = _layer(variant=variant)
        layer.to(device)
        x = lucid.randn(5, 4, device=device)
        assert bool((layer(x) == base(x)).all().item())
        assert tuple(layer.get_delta_weight("default").shape) == (8, 8)
        eye = lucid.eye(8, device=device)
        assert bool((layer.get_delta_weight("default") == eye).all().item())

    @pytest.mark.parametrize("variant", _VARIANTS)
    def test_the_parameters_have_the_variants_sizes(self, variant: str) -> None:
        layer, _ = _layer(variant=variant)
        size = {"road_1": 4, "road_2": 8, "road_4": 16}[variant]
        assert tuple(layer.road_theta["default"].shape) == (size,)
        assert tuple(layer.road_alpha["default"].shape) == (size,)

    def test_a_random_start_moves_the_output(self) -> None:
        layer, base = _layer(init_weights=False)
        x = lucid.randn(5, 4)
        assert span(layer(x), base(x)) > 1e-3


class TestRotation:
    @pytest.mark.parametrize("variant", _VARIANTS)
    @pytest.mark.parametrize("group_size", [2, 4, 8])
    def test_the_layer_applies_the_papers_matrix_to_the_output(
        self, variant: str, group_size: int, device: str
    ) -> None:
        layer, base = _layer(variant=variant, group_size=group_size)
        layer.to(device)
        _move(layer)
        x = lucid.randn(6, 4, device=device)
        with lucid.no_grad():
            rotation = _closed_matrix(layer).to(device)
            expected = base(x) @ rotation.mT
            assert span(layer(x), expected) <= _TOLERANCE * magnitude(expected)
            assert span(layer.get_delta_weight("default"), rotation) <= _TOLERANCE * magnitude(
                rotation
            )

    def test_a_pair_is_the_first_half_of_a_group_with_the_second(self) -> None:
        # With group 4, the pairs are (0, 2) and (1, 3): only those entries of R are not on the diagonal.
        layer, _ = _layer(variant="road_2", group_size=4)
        _move(layer)
        rotation = layer.get_delta_weight("default").tolist()
        off = {(i, j) for i in range(8) for j in range(8) if i != j and rotation[i][j] != 0.0}
        assert off == {(0, 2), (2, 0), (1, 3), (3, 1), (4, 6), (6, 4), (5, 7), (7, 5)}

    @pytest.mark.parametrize("variant", _VARIANTS)
    def test_a_unit_scale_is_an_orthogonal_rotation_that_keeps_norms_and_angles(
        self, variant: str
    ) -> None:
        layer, _ = _layer(variant=variant)
        lucid.manual_seed(4)
        # One angle per pair, laid out as the variant holds it: a rotation needs the four angles of
        # a block equal, which ``road_2`` and ``road_4`` only have when told to.
        pairs = lucid.randn(4)
        shape, angles = {
            "road_1": ((4,), pairs),
            "road_2": ((2, 2, 2), pairs.reshape(2, 1, 2)),
            "road_4": ((2, 2, 2, 2), pairs.reshape(2, 1, 1, 2)),
        }[variant]
        with lucid.no_grad():
            layer.road_theta["default"].copy_(angles.expand(*shape).reshape(-1))
        rotation = layer.get_delta_weight("default")
        assert span(rotation @ rotation.mT, lucid.eye(8)) <= _TOLERANCE
        y = lucid.randn(6, 8)
        turned = y @ rotation.mT
        assert span(_norms(turned), _norms(y)) <= _TOLERANCE * magnitude(y)
        assert span(turned @ turned.mT, y @ y.mT) <= _TOLERANCE * magnitude(y @ y.mT)

    def test_only_the_angles_and_scales_get_a_gradient(self) -> None:
        net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
        peft = get_peft_model(net, RoadConfig(group_size=2, target_modules=["0", "2"]))
        peft(lucid.randn(3, 4)).sum().backward()
        for name, p in peft.named_parameters():
            if p.requires_grad:
                assert "road_" in name and p.grad is not None, name
            else:
                assert p.grad is None, name


class TestMerging:
    @pytest.mark.parametrize("variant", _VARIANTS)
    def test_merging_writes_the_rotated_weight_and_bias(self, variant: str) -> None:
        layer, base = _layer(variant=variant, init_weights=False)
        weight, bias = base.weight.tolist(), base.bias.tolist()
        expected_weight, expected_bias = forms.road_adapted(
            weight,
            bias,
            variant,
            4,
            layer.road_theta["default"].tolist(),
            layer.road_alpha["default"].tolist(),
        )
        x = lucid.randn(5, 4)
        with lucid.no_grad():
            adapted = layer(x)
        layer.merge()
        scale = _TOLERANCE * max(1.0, magnitude(base.weight))
        assert span(base.weight, lucid.tensor(expected_weight)) <= scale
        assert span(base.bias, lucid.tensor(expected_bias)) <= scale
        assert span(layer(x), adapted) <= _TOLERANCE * magnitude(adapted)
        layer.unmerge()
        assert span(layer(x), adapted) <= _TOLERANCE * magnitude(adapted)

    @pytest.mark.parametrize("variant", _VARIANTS)
    def test_unmerging_restores_the_weight_and_bias_to_a_scale_aware_bound(
        self, variant: str
    ) -> None:
        layer, base = _layer(variant=variant, init_weights=False)
        weight, bias = base.weight.clone(), base.bias.clone()
        layer.merge()
        layer.unmerge()
        bound = _TOLERANCE * max(magnitude(weight), 1.0)
        assert span(base.weight, weight) <= bound and span(base.bias, bias) <= bound

    def test_a_base_without_a_bias_merges_the_weight_only(self) -> None:
        lucid.manual_seed(1)
        base = nn.Linear(4, 8, bias=False)
        layer = road.Linear(
            base, "default", RoadConfig(group_size=4, init_weights=False, target_modules=["x"])
        )
        x = lucid.randn(3, 4)
        with lucid.no_grad():
            adapted = layer(x)
        layer.merge()
        assert span(layer(x), adapted) <= _TOLERANCE * magnitude(adapted)

    def test_a_singular_block_cannot_be_unmerged_and_leaves_the_weight_merged(self) -> None:
        layer, base = _layer()
        with lucid.no_grad():
            layer.road_alpha["default"].copy_(lucid.zeros(4))
        layer.merge()
        with pytest.raises(ValueError, match="singular"):
            layer.unmerge()
        assert layer.merged and span(base.weight, lucid.zeros(8, 4)) == 0.0


class TestSeveralAdapters:
    def _two(self) -> tuple[PeftModel, lucid.Tensor]:
        lucid.manual_seed(0)
        net = nn.Sequential(nn.Linear(4, 8))
        config = RoadConfig(
            variant="road_2", group_size=4, init_weights=False, target_modules=["0"]
        )
        peft = get_peft_model(net, config)
        peft.add_adapter("b", config)
        peft.set_adapter(["default", "b"])
        return peft, lucid.randn(5, 4)

    def test_the_adapters_turn_the_output_in_the_order_listed(self) -> None:
        peft, x = self._two()
        layer = peft.base_model.model[0]
        first, second = (_closed_matrix(layer, name) for name in ("default", "b"))
        y = layer.base_layer(x)
        with lucid.no_grad():
            listed = y @ first.mT @ second.mT
            reverse = y @ second.mT @ first.mT
            assert span(peft(x), listed) <= _TOLERANCE * magnitude(listed)
            # Not vacuous: the two blocks do not commute.
            assert span(listed, reverse) > 1e-3 * magnitude(listed)
            peft.set_adapter(["b", "default"])
            assert span(peft(x), reverse) <= _TOLERANCE * magnitude(reverse)


class TestErrors:
    def test_a_width_that_is_not_a_multiple_of_the_group_is_refused(self) -> None:
        with pytest.raises(ValueError, match="divisible by group_size"):
            _layer(out_features=6)

    def test_only_a_linear_is_adapted(self) -> None:
        net = nn.Sequential(nn.Conv2d(2, 4, 3))
        with pytest.raises(ValueError, match="not supported"):
            get_peft_model(net, RoadConfig(group_size=2, target_modules=["0"]))

    def test_a_refused_adapter_leaves_the_model_as_it_was(self) -> None:
        net = nn.Sequential(nn.Linear(4, 6))
        with pytest.raises(ValueError, match="divisible by group_size"):
            get_peft_model(net, RoadConfig(group_size=4, target_modules=["0"]))
        assert type(net[0]) is nn.Linear and all(p.requires_grad for p in net.parameters())


def test_the_tuner_wraps_the_linear_layers_and_unloads_them() -> None:
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    peft = get_peft_model(net, RoadConfig(group_size=2, target_modules=["0", "2"]))
    assert isinstance(peft.base_model, RoadModel)
    assert type(net[0]) is road.Linear and type(net[2]) is road.Linear
    assert peft.unload() is net and type(net[0]) is nn.Linear


# ── the reference adapter library ────────────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("group_size", [2, 4, 8])
def test_a_layer_matches_the_reference(ref: ModuleType, variant: str, group_size: int) -> None:
    library = adapter_library()
    lucid.manual_seed(5)
    net = nn.Sequential(nn.Linear(6, 16))
    twin = ref.nn.Sequential(ref.nn.Linear(6, 16))
    x = lucid.randn(4, 6)
    options = {"variant": variant, "group_size": group_size, "init_weights": False}
    peft = get_peft_model(net, RoadConfig(target_modules=["0"], **options))  # type: ignore[arg-type]
    mine = peft.base_model.model[0]
    with ref.no_grad():
        twin[0].weight.copy_(ref.tensor(mine.base_layer.weight.tolist()))
        twin[0].bias.copy_(ref.tensor(mine.base_layer.bias.tolist()))
    theirs = library.get_peft_model(twin, library.RoadConfig(target_modules=["0"], **options))
    other = theirs.base_model.model[0]
    with ref.no_grad():
        other.road_theta["default"].copy_(ref.tensor(mine.road_theta["default"].tolist()))
        other.road_alpha["default"].copy_(ref.tensor(mine.road_alpha["default"].tolist()))
    assert gap(ref, peft(x), theirs(ref.tensor(x.tolist()))) < 1e-5
    peft.base_model.merge_adapter()
    theirs.base_model.merge_adapter()
    assert gap(ref, mine.base_layer.weight, other.base_layer.weight) < 1e-5
    assert gap(ref, mine.base_layer.bias, other.base_layer.bias) < 1e-5
    peft.base_model.unmerge_adapter()
    theirs.base_model.unmerge_adapter()
    assert gap(ref, mine.base_layer.weight, other.base_layer.weight) < 1e-5


@pytest.mark.parity
def test_the_config_file_has_the_references_keys() -> None:
    library = adapter_library()
    ours = RoadConfig(variant="road_2", group_size=8, target_modules=["a"]).to_dict()
    theirs = library.RoadConfig(variant="road_2", group_size=8, target_modules=["a"]).to_dict()
    assert set(ours) == set(theirs) - {"peft_version"}
    assert all(
        ours[k] == (sorted(v) if isinstance(v, set) else v) for k, v in theirs.items() if k in ours
    )
