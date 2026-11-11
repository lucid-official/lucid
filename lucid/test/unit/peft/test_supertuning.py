"""Super-Tuning: the support chosen by magnitude, the update in closed form, the Supra LoRA part, the file and the reference library's numbers.

The paper's update is ``W' = W + Delta + (alpha / r) B A`` with ``Delta`` zero off the support of the
``1 - sparsity`` fraction of ``|W|`` that is largest (or smallest).  The tests rank the entries with a
plain sort, lay the trained values at those positions with a loop, and expect the layer, its dense
update and its merged weight to be that sum.  The ``parity`` tests build the same layer in the
reference library and compare the positions it chooses, the forward pass, the merged weight and the
shapes it stores.
"""

import math
import warnings
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    PeftModel,
    SupertuningConfig,
    SupertuningModel,
    get_peft_model,
    get_peft_model_state_dict,
)
from lucid.peft.tuners import supertuning
from lucid.test.unit.peft._parity import adapter_library, gap
from lucid.test.unit.peft._tuner_recipes import magnitude, same, span

#: Relative to the largest value compared: one float32 sum or product.
_TOLERANCE = 1e-5


def _layer(**kw: object) -> tuple[supertuning.Linear, nn.Linear]:
    lucid.manual_seed(1)
    base = nn.Linear(6, 8)
    fields: dict[str, object] = {"sparsity": 0.5, "target_modules": ["x"]} | kw
    return supertuning.Linear(base, "default", SupertuningConfig(**fields)), base  # type: ignore[arg-type]


def _ranked(weight: lucid.Tensor, count: int, top: bool = True) -> list[int]:
    """The flat positions of the ``count`` largest (or smallest) ``|w|``, best first: a plain sort."""
    flat = [abs(v) for row in weight.tolist() for v in row]
    order = sorted(range(len(flat)), key=lambda p: flat[p], reverse=top)
    return order[:count]


def _closed_delta(
    layer: supertuning.SupertuningLayer, adapter: str = "default"
) -> list[list[float]]:
    """``Delta + (alpha / r) B A`` entry by entry from the layer's values, positions and LoRA pair."""
    out, inn = layer.out_features, layer.in_features
    delta = [[0.0] * inn for _ in range(out)]
    positions = layer.supertuning_indices[adapter].tolist()
    for place, value in zip(positions, layer.supertuning_values[adapter].tolist(), strict=True):
        delta[place // inn][place % inn] += value
    rank = layer.supertuning_rank[adapter]
    if rank:
        a = layer.supertuning_lora_A[adapter].weight.tolist()
        b = layer.supertuning_lora_B[adapter].weight.tolist()
        scale = layer.supertuning_lora_alpha[adapter] / rank
        for i in range(out):
            for j in range(inn):
                delta[i][j] += scale * sum(b[i][k] * a[k][j] for k in range(rank))
    return delta


def _move(layer: supertuning.SupertuningLayer, seed: int = 3) -> None:
    """Move the values and the LoRA pair well off their starts, as training would."""
    lucid.manual_seed(seed)
    with lucid.no_grad():
        for p in layer._adapter_parameters("default"):
            p.copy_(p + 0.3 * lucid.randn(*p.shape, device=p.device))


class TestConfig:
    def test_the_defaults_are_the_references(self) -> None:
        config = SupertuningConfig(target_modules=["a"])
        assert (config.sparsity, config.select_top, config.r) == (0.99, True, None)
        assert (config.lora_alpha, config.lora_dropout) == (None, 0.0)
        assert (config.init_weights, config.save_precomputed_indices) == (True, True)

    def test_alpha_defaults_to_twice_the_rank(self) -> None:
        assert SupertuningConfig(r=4, target_modules=["a"]).lora_alpha == 8

    def test_fields_are_checked(self) -> None:
        for bad in (-0.1, 1.0):
            with pytest.raises(ValueError, match="sparsity"):
                SupertuningConfig(sparsity=bad, target_modules=["a"])
        with pytest.raises(ValueError, match="lora_alpha is set but r is None"):
            SupertuningConfig(lora_alpha=4, target_modules=["a"])
        with pytest.raises(ValueError, match="r"):
            SupertuningConfig(r=0, target_modules=["a"])
        with pytest.raises(ValueError, match="dropout"):
            SupertuningConfig(r=2, lora_dropout=1.0, target_modules=["a"])
        with pytest.raises(TypeError, match="select_top"):
            SupertuningConfig(select_top="yes", target_modules=["a"])  # type: ignore[arg-type]

    def test_unsaved_positions_warn(self) -> None:
        with pytest.warns(UserWarning, match="recomputed from base"):
            SupertuningConfig(save_precomputed_indices=False, target_modules=["a"])

    def test_a_file_round_trips(self) -> None:
        config = SupertuningConfig(sparsity=0.9, r=4, target_modules=["q", "v"])
        assert SupertuningConfig.from_dict(config.to_dict()) == config
        assert config.to_dict()["peft_type"] == "SUPERTUNING"


class TestSupport:
    @pytest.mark.parametrize("top", [True, False])
    @pytest.mark.parametrize("sparsity", [0.0, 0.5, 0.75, 0.9])
    def test_the_support_is_the_extreme_entries_by_magnitude_best_first(
        self, top: bool, sparsity: float
    ) -> None:
        layer, base = _layer(sparsity=sparsity, select_top=top)
        count = int(48 * (1 - sparsity))
        expected = _ranked(base.weight, count, top)
        assert layer.supertuning_indices["default"].tolist() == expected
        assert tuple(layer.supertuning_values["default"].shape) == (count,)

    def test_the_positions_are_int32_and_not_trained(self) -> None:
        layer, _ = _layer()
        positions = layer.supertuning_indices["default"]
        assert positions.dtype == lucid.int32 and not positions.requires_grad

    def test_a_sparsity_that_leaves_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no trainable entries"):
            _layer(sparsity=0.99)  # 48 entries * 0.01 floors to 0

    def test_the_count_is_the_floor(self) -> None:
        layer, _ = _layer(sparsity=0.7)  # 48 * 0.3 = 14.4
        assert tuple(layer.supertuning_values["default"].shape) == (14,)


class TestStarts:
    @pytest.mark.parametrize("extra", [{}, {"r": 2}])
    def test_the_default_start_is_the_base_bit_for_bit(
        self, extra: dict[str, object], device: str
    ) -> None:
        layer, base = _layer(**extra)
        layer.to(device)
        x = lucid.randn(5, 6, device=device)
        assert same(layer(x), base(x))

    def test_the_lora_pair_starts_with_a_kaiming_a_and_a_zero_b(self) -> None:
        layer, _ = _layer(r=3)
        a = layer.supertuning_lora_A["default"].weight
        b = layer.supertuning_lora_B["default"].weight
        assert tuple(a.shape) == (3, 6) and tuple(b.shape) == (8, 3)
        assert not bool(b.any().item())
        assert bool(a.any().item()) and float(a.abs().max().item()) <= 1 / math.sqrt(6) + 1e-6

    def test_a_random_start_moves_both_parts(self) -> None:
        layer, base = _layer(r=2, init_weights=False)
        assert bool(layer.supertuning_lora_B["default"].weight.any().item())
        x = lucid.randn(5, 6)
        assert span(layer(x), base(x)) > 1e-3

    def test_the_alpha_over_rank_scale_is_recorded(self) -> None:
        layer, _ = _layer(r=2, lora_alpha=6)
        assert layer.supertuning_rank == {"default": 2}
        assert layer.supertuning_lora_alpha == {"default": 6.0}
        plain, _ = _layer()
        assert plain.supertuning_rank == {"default": 0}


class TestUpdate:
    @pytest.mark.parametrize("extra", [{}, {"r": 2, "lora_alpha": 4}, {"select_top": False}])
    def test_the_layer_applies_the_papers_sum_to_the_base_weight(
        self, extra: dict[str, object], device: str
    ) -> None:
        layer, base = _layer(**extra)
        layer.to(device)
        _move(layer)
        x = lucid.randn(5, 6, device=device)
        with lucid.no_grad():
            delta = lucid.tensor(_closed_delta(layer)).to(device)
            expected = x @ (base.weight + delta).mT + base.bias
            assert span(layer(x), expected) <= _TOLERANCE * magnitude(expected)
            assert span(layer.get_delta_weight("default"), delta) <= _TOLERANCE * max(
                1.0, magnitude(delta)
            )

    def test_the_update_is_zero_off_the_support(self) -> None:
        layer, _ = _layer(sparsity=0.75)
        _move(layer)
        delta = layer.get_delta_weight("default").tolist()
        support = set(layer.supertuning_indices["default"].tolist())
        assert {i * 6 + j for i in range(8) for j in range(6) if delta[i][j] != 0.0} <= support

    def test_only_the_values_and_the_lora_pair_get_a_gradient(self) -> None:
        net = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4))
        peft = get_peft_model(
            net, SupertuningConfig(sparsity=0.5, r=2, target_modules=["0", "2"], init_weights=False)
        )
        peft(lucid.randn(3, 6)).sum().backward()
        for name, p in peft.named_parameters():
            if p.requires_grad:
                assert "supertuning_" in name and p.grad is not None, name
            else:
                assert p.grad is None, name

    def test_a_values_gradient_is_the_weight_gradient_at_its_position(self) -> None:
        layer, _ = _layer()
        x = lucid.randn(7, 6)
        layer(x).sum().backward()
        grad = layer.supertuning_values["default"].grad
        assert grad is not None
        # d(sum y) / dW[i][j] = sum over the batch of x[:, j], the same for every row i.
        column_sums = x.sum(dim=0).tolist()
        positions = layer.supertuning_indices["default"].tolist()
        assert span(grad, lucid.tensor([column_sums[p % 6] for p in positions])) <= 1e-5

    def test_a_lora_dropout_touches_the_lora_part_only(self) -> None:
        layer, base = _layer(r=2, lora_dropout=0.5, init_weights=False)
        layer.eval()
        x = lucid.randn(4, 6)
        with lucid.no_grad():
            quiet = layer(x)
        layer.train()
        lucid.manual_seed(0)
        with lucid.no_grad():
            noisy = layer(x)
        # The sparse part is the same in both; the LoRA part differs by the dropped inputs.
        assert span(quiet, noisy) > 0 and span(quiet, base(x)) > 0


class TestMerging:
    @pytest.mark.parametrize("extra", [{}, {"r": 2}])
    def test_merging_adds_the_update_to_the_weight(self, extra: dict[str, object]) -> None:
        layer, base = _layer(**extra)
        _move(layer)
        weight, bias = base.weight.clone(), base.bias.clone()
        expected = weight + lucid.tensor(_closed_delta(layer))
        layer.merge()
        assert span(base.weight, expected) <= _TOLERANCE * magnitude(expected)
        assert same(base.bias, bias)
        layer.unmerge()
        assert span(base.weight, weight) <= _TOLERANCE * magnitude(weight)

    def test_two_adapters_merge_to_the_sum_of_their_updates(self) -> None:
        net = nn.Sequential(nn.Linear(6, 8))
        config = SupertuningConfig(sparsity=0.5, target_modules=["0"], init_weights=False)
        peft = get_peft_model(net, config)
        peft.add_adapter("b", config)
        peft.set_adapter(["default", "b"])
        layer = peft.base_model.model[0]
        x = lucid.randn(4, 6)
        with lucid.no_grad():
            total = layer.get_delta_weight("default") + layer.get_delta_weight("b")
            expected = x @ (layer.get_base_layer().weight + total).mT + layer.get_base_layer().bias
            assert span(peft(x), expected) <= _TOLERANCE * magnitude(expected)
            peft.base_model.merge_adapter()
            assert span(peft(x), expected) <= _TOLERANCE * magnitude(expected)


class TestFile:
    def _file_keys(self, peft: PeftModel) -> set[str]:
        return {k.removeprefix("base_model.model.0.") for k in get_peft_model_state_dict(peft)}

    def test_the_file_holds_values_positions_and_the_lora_pair(self) -> None:
        net = nn.Sequential(nn.Linear(6, 8))
        peft = get_peft_model(net, SupertuningConfig(sparsity=0.5, r=2, target_modules=["0"]))
        assert self._file_keys(peft) == {
            "supertuning_values",
            "supertuning_indices",
            "supertuning_lora_A.weight",
            "supertuning_lora_B.weight",
        }

    @pytest.mark.filterwarnings("ignore:save_precomputed_indices=False")
    def test_unsaved_positions_are_left_out_and_chosen_again_on_load(
        self, tmp_path: object
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = SupertuningConfig(
                sparsity=0.5,
                target_modules=["0"],
                save_precomputed_indices=False,
                init_weights=False,
            )
        lucid.manual_seed(0)
        net = nn.Sequential(nn.Linear(6, 8))
        peft = get_peft_model(net, config)
        assert self._file_keys(peft) == {"supertuning_values"}
        peft.save_pretrained(str(tmp_path))
        lucid.manual_seed(0)
        loaded = PeftModel.from_pretrained(nn.Sequential(nn.Linear(6, 8)), str(tmp_path))
        mine, theirs = peft.base_model.model[0], loaded.base_model.model[0]
        assert (
            mine.supertuning_indices["default"].tolist()
            == theirs.supertuning_indices["default"].tolist()
        )
        assert same(mine.supertuning_values["default"], theirs.supertuning_values["default"])

    @pytest.mark.filterwarnings("ignore:save_precomputed_indices=False")
    def test_adapters_must_agree_on_saving_the_positions(self) -> None:
        net = nn.Sequential(nn.Linear(6, 8))
        peft = get_peft_model(net, SupertuningConfig(sparsity=0.5, target_modules=["0"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            other = SupertuningConfig(
                sparsity=0.5, target_modules=["0"], save_precomputed_indices=False
            )
        with pytest.raises(ValueError, match="saved for all adapters or none"):
            peft.add_adapter("b", other)
        assert "b" not in peft.peft_config


class TestHalfPrecision:
    def test_a_half_base_keeps_a_float32_update_and_is_restored(self) -> None:
        lucid.manual_seed(1)
        base = nn.Linear(6, 8).to(lucid.float16)
        config = SupertuningConfig(sparsity=0.5, r=2, init_weights=False, target_modules=["x"])
        layer = supertuning.Linear(base, "default", config)
        assert layer.supertuning_values["default"].dtype == lucid.float32
        assert layer.supertuning_lora_A["default"].weight.dtype == lucid.float32
        assert layer(lucid.randn(3, 6).to(lucid.float16)).dtype == lucid.float16


def test_the_tuner_wraps_the_linear_layers_and_unloads_them() -> None:
    net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    peft = get_peft_model(net, SupertuningConfig(sparsity=0.5, target_modules=["0", "2"]))
    assert isinstance(peft.base_model, SupertuningModel)
    assert type(net[0]) is supertuning.Linear and type(net[2]) is supertuning.Linear
    assert peft.unload() is net and type(net[0]) is nn.Linear


def test_only_a_linear_is_adapted() -> None:
    net = nn.Sequential(nn.Conv2d(2, 4, 3))
    with pytest.raises(ValueError, match="not supported"):
        get_peft_model(net, SupertuningConfig(sparsity=0.5, target_modules=["0"]))


# ── the reference adapter library ────────────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("top", [True, False])
@pytest.mark.parametrize("extra", [{}, {"r": 3, "lora_alpha": 5}])
def test_a_layer_matches_the_reference(
    ref: ModuleType, top: bool, extra: dict[str, object]
) -> None:
    library = adapter_library()
    lucid.manual_seed(5)
    net = nn.Sequential(nn.Linear(6, 10))
    twin = ref.nn.Sequential(ref.nn.Linear(6, 10))
    x = lucid.randn(4, 6)
    options = {"sparsity": 0.6, "select_top": top, **extra}
    peft = get_peft_model(
        net,
        SupertuningConfig(target_modules=["0"], init_weights=False, **options),  # type: ignore[arg-type]
    )
    mine = peft.base_model.model[0]
    with ref.no_grad():
        twin[0].weight.copy_(ref.tensor(mine.base_layer.weight.tolist()))
        twin[0].bias.copy_(ref.tensor(mine.base_layer.bias.tolist()))
    theirs = library.get_peft_model(
        twin, library.SupertuningConfig(target_modules=["0"], init_weights=False, **options)
    )
    other = theirs.base_model.model[0]
    # The same positions, in the same order, and the same shapes under the same names.
    assert (
        other.supertuning_indices["default"].tolist()
        == mine.supertuning_indices["default"].tolist()
    )
    assert {k: tuple(v.shape) for k, v in other.state_dict().items() if "supertuning" in k} == {
        k: tuple(v.shape) for k, v in mine.state_dict().items() if "supertuning" in k
    }
    with ref.no_grad():
        for name, tensor in mine.state_dict().items():
            if "supertuning" in name and "indices" not in name:
                other.state_dict()[name].copy_(ref.tensor(tensor.tolist()))
    assert gap(ref, peft(x), theirs(ref.tensor(x.tolist()))) < 1e-5
    peft.base_model.merge_adapter()
    theirs.base_model.merge_adapter()
    assert gap(ref, mine.base_layer.weight, other.base_layer.weight) < 1e-5
    peft.base_model.unmerge_adapter()
    theirs.base_model.unmerge_adapter()
    assert gap(ref, mine.base_layer.weight, other.base_layer.weight) < 1e-5


@pytest.mark.parity
def test_the_config_file_has_the_references_keys() -> None:
    library = adapter_library()
    options: dict[str, object] = {"sparsity": 0.9, "r": 4, "target_modules": ["a"]}
    ours = SupertuningConfig(**options).to_dict()  # type: ignore[arg-type]
    theirs = library.SupertuningConfig(**options).to_dict()
    assert set(ours) == set(theirs) - {"peft_version"}
    assert all(
        ours[k] == (sorted(v) if isinstance(v, set) else v) for k, v in theirs.items() if k in ours
    )
