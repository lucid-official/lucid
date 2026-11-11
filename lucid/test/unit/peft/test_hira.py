"""HiRA: the Hadamard update in closed form on every layer type, the stash that makes a merge exact, the starts and the reference adapter library's numbers.

HiRA (Huang et al., 2025) adds ``W * (B A)`` (the elementwise product) to the frozen weight.  The tests
expect that matrix from plain loops (``_delta_forms``) for a linear layer, a lookup table and a
convolution, and expect the merge the update's dependence on ``W`` asks for: the merged weight is
``W0 + sum_a W0 * (B_a A_a)`` read from the stash of ``W0``, and unmerging gives ``W0`` back bit for
bit.  The ``parity`` tests build the same layers in the reference library and compare.
"""

from types import ModuleType

import pytest

import lucid
import lucid.linalg as linalg
import lucid.nn as nn
from lucid.peft import HiraConfig, HiraModel, get_peft_model, get_peft_model_state_dict
from lucid.peft.tuners import hira
from lucid.test.unit.peft import _delta_forms as forms
from lucid.test.unit.peft._parity import adapter_library, gap


def _config(**kw: object) -> HiraConfig:
    fields: dict[str, object] = {"r": 3, "target_modules": ["x"]} | kw
    return HiraConfig.from_dict(fields)


def _randomise(layer: hira.HiraLayer, seed: int = 2) -> None:
    """Both factors non-zero, so the update is."""
    lucid.manual_seed(seed)
    with lucid.no_grad():
        for parameter in layer._adapter_parameters("default"):
            parameter.copy_(lucid.randn(*parameter.shape))


def _span(a: lucid.Tensor, b: lucid.Tensor) -> float:
    return float((a - b).abs().max().item())


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


class TestConfig:
    def test_the_defaults_are_the_references(self) -> None:
        config = HiraConfig(target_modules=["a"])
        assert (config.r, config.hira_dropout, config.init_weights) == (32, 0.0, True)
        assert config.rank_pattern == {} and config.fan_in_fan_out is False

    def test_fields_are_checked(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            HiraConfig(r=0, target_modules=["a"])
        with pytest.raises(ValueError, match="gaussian"):
            HiraConfig.from_dict({"init_weights": "uniform", "target_modules": ["a"]})
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            HiraConfig(hira_dropout=1.0, target_modules=["a"])

    def test_a_file_round_trips(self) -> None:
        config = HiraConfig(
            r=4, init_weights="gaussian", rank_pattern={"q": 2}, target_modules=["q"]
        )
        assert HiraConfig.from_dict(config.to_dict()) == config
        assert config.to_dict()["peft_type"] == "HIRA"


class TestStart:
    @pytest.mark.parametrize("init", [True, "gaussian"], ids=["kaiming", "gaussian"])
    def test_the_update_starts_at_zero_and_the_output_is_the_base_bit_for_bit(
        self, init: bool | str, device: str
    ) -> None:
        base = nn.Linear(6, 4)
        layer = hira.Linear(base, "default", _config(init_weights=init)).to(device)
        x = lucid.randn(5, 6, device=device)
        assert float(layer.get_delta_weight("default").abs().max().item()) == 0.0
        assert _same(layer(x), base(x))

    def test_the_gaussian_start_has_a_standard_deviation_of_one_over_r(self) -> None:
        lucid.manual_seed(0)
        layer = hira.Linear(nn.Linear(300, 4), "default", _config(r=10, init_weights="gaussian"))
        a = layer.hira_A["default"]
        assert abs(float(a.std().item()) - 0.1) < 0.01  # 3000 draws

    @pytest.mark.parametrize("init", [False, None], ids=["false", "none"])
    def test_a_falsy_start_is_random_for_a_linear_layer(self, init: bool | None) -> None:
        base = nn.Linear(6, 4)
        layer = hira.Linear(
            base,
            "default",
            HiraConfig.from_dict({"r": 3, "init_weights": init, "target_modules": ["x"]}),
        )
        x = lucid.randn(5, 6)
        assert _span(layer(x), base(x)) > 1e-3

    def test_a_lookup_table_starts_with_a_zero_a_unless_the_start_is_false(self) -> None:
        base = nn.Embedding(10, 4)
        zero = hira.Embedding(base, "default", _config(init_weights=None))
        assert float(zero.hira_embedding_A["default"].abs().max().item()) == 0.0
        assert float(zero.hira_embedding_B["default"].abs().max().item()) > 0
        random = hira.Embedding(base, "default", _config(init_weights=False))
        assert float(random.hira_embedding_A["default"].abs().max().item()) > 0

    def test_the_first_step_reaches_b_only(self) -> None:
        layer = hira.Linear(nn.Linear(6, 4), "default", _config())
        layer(lucid.randn(5, 6)).sum().backward()
        grad_a, grad_b = layer.hira_A["default"].grad, layer.hira_B["default"].grad
        assert grad_a is not None and grad_b is not None
        assert float(grad_b.abs().max().item()) > 0
        assert float(grad_a.abs().max().item()) == 0.0


class TestFormula:
    def test_a_linear_update_is_the_hadamard_product(self, device: str) -> None:
        lucid.manual_seed(1)
        base = nn.Linear(6, 4)
        layer = hira.Linear(base, "default", _config(r=2)).to(device)
        _randomise(layer)
        expected = forms.hira_delta(
            base.weight.tolist(), layer.hira_A["default"].tolist(), layer.hira_B["default"].tolist()
        )
        got = layer.get_delta_weight("default")
        assert bool(lucid.allclose(got, lucid.tensor(expected, device=device), atol=1e-5))

    def test_the_forward_pass_is_the_modulated_weight_applied_to_the_input(
        self, device: str
    ) -> None:
        # Dense probe: the identity through the layer, minus the identity through the base.
        lucid.manual_seed(1)
        base = nn.Linear(6, 4)
        layer = hira.Linear(base, "default", _config()).to(device)
        _randomise(layer)
        probe = lucid.eye(6, device=device)
        applied = layer(probe) - base(probe)
        assert bool(lucid.allclose(applied, layer.get_delta_weight("default").mT, atol=1e-5))

    def test_the_update_has_more_than_the_rank_of_its_factors(self) -> None:
        # B A has rank 2, but W * (B A) is the sum of two rank-one-scaled copies of W: rank up to min(size).
        lucid.manual_seed(1)
        layer = hira.Linear(nn.Linear(8, 8), "default", _config(r=2))
        _randomise(layer)
        assert int(linalg.matrix_rank(layer.get_delta_weight("default")).item()) > 2

    def test_a_lookup_table_is_modulated_entry_by_entry(self, device: str) -> None:
        lucid.manual_seed(1)
        base = nn.Embedding(10, 4)
        layer = hira.Embedding(base, "default", _config(r=2)).to(device)
        _randomise(layer)
        a, b = (
            layer.hira_embedding_A["default"].tolist(),
            layer.hira_embedding_B["default"].tolist(),
        )
        table = base.weight.tolist()
        expected = [
            [table[v][d] * sum(b[d][s] * a[s][v] for s in range(2)) for d in range(4)]
            for v in range(10)
        ]
        assert bool(
            lucid.allclose(
                layer.get_delta_weight("default"), lucid.tensor(expected, device=device), atol=1e-5
            )
        )
        ids = lucid.tensor([[1, 2, 3], [4, 9, 9]], device=device)
        looked = layer(ids)
        assert bool(
            lucid.allclose(
                looked, base(ids) + lucid.tensor(expected, device=device)[ids], atol=1e-5
            )
        )

    @pytest.mark.parametrize(
        ("make", "shape"),
        [
            (lambda: nn.Conv1d(3, 5, 3, padding=1), (2, 3, 6)),
            (lambda: nn.Conv2d(3, 5, 3, padding=1), (2, 3, 6, 6)),
            (lambda: nn.Conv3d(3, 5, 3, padding=1), (1, 3, 4, 4, 4)),
        ],
        ids=["conv1d", "conv2d", "conv3d"],
    )
    def test_a_convolution_kernel_is_modulated_entry_by_entry(
        self, make: type[nn.Module], shape: tuple[int, ...], device: str
    ) -> None:
        lucid.manual_seed(1)
        base = make()
        layer = hira.HiraLayer._layer_types[type(base)](base, "default", _config(r=2)).to(device)
        _randomise(layer)
        a, b = layer.hira_A["default"], layer.hira_B["default"]
        expected = forms.hira_conv_delta(
            base.weight.reshape(5, 3, -1).tolist(),
            a.reshape(2, 3, -1).tolist(),
            b.reshape(5, 2).tolist(),
        )
        got = layer.get_delta_weight("default").reshape(5, 3, -1)
        assert bool(lucid.allclose(got, lucid.tensor(expected, device=device), atol=1e-5))
        # The convolution runs with the modulated kernel and the base's geometry.
        x = lucid.randn(*shape, device=device)
        delta_kernel = layer.get_delta_weight("default")
        adapted = layer(x)
        modulated = type(base)(3, 5, 3, padding=1, bias=False).to(device)
        with lucid.no_grad():
            modulated.weight.copy_(delta_kernel)
        assert bool(lucid.allclose(adapted, base(x) + modulated(x), atol=1e-4))

    def test_dropout_acts_on_the_input_in_training_only(self) -> None:
        layer = hira.Linear(nn.Linear(6, 4), "default", _config(hira_dropout=0.5))
        _randomise(layer)
        x = lucid.randn(8, 6)
        layer.eval()
        steady = layer(x)
        assert _same(layer(x), steady)
        layer.train()
        lucid.manual_seed(0)
        assert _span(layer(x), steady) > 1e-3


class TestMerging:
    def test_the_merged_weight_is_w_times_one_plus_the_modulation(self) -> None:
        lucid.manual_seed(1)
        base = nn.Linear(6, 4)
        layer = hira.Linear(base, "default", _config(r=2))
        _randomise(layer)
        weight = base.weight.tolist()
        delta = forms.hira_delta(
            weight, layer.hira_A["default"].tolist(), layer.hira_B["default"].tolist()
        )
        layer.merge()
        assert bool(lucid.allclose(base.weight, lucid.tensor(forms.add(weight, delta)), atol=1e-6))

    def test_unmerging_gives_the_weights_back_bit_for_bit(self, device: str) -> None:
        lucid.manual_seed(1)
        base = nn.Linear(6, 4)
        layer = hira.Linear(base, "default", _config()).to(device)
        _randomise(layer)
        before = base.weight.clone()
        layer.merge()
        assert _span(base.weight, before) > 1e-3
        layer.unmerge()
        assert _same(base.weight, before)

    def test_two_adapters_add_their_updates_each_read_from_the_original_weight(self) -> None:
        # W0 + (W0 * M_a) + (W0 * M_b), not W0 * (1 + M_a) * (1 + M_b): the forward pass adds the
        # two modulated copies of the one frozen weight.
        lucid.manual_seed(1)
        base = nn.Linear(6, 4)
        layer = hira.Linear(base, "a", _config(r=2))
        layer.update_layer("b", _config(r=3))
        for name in ("a", "b"):
            lucid.manual_seed(7 if name == "a" else 8)
            with lucid.no_grad():
                for p in layer._adapter_parameters(name):
                    p.copy_(lucid.randn(*p.shape))
        layer.set_adapter(["a", "b"])
        weight = base.weight.clone()
        x = lucid.randn(5, 6)
        with lucid.no_grad():
            both = layer(x)
            expected = weight + layer.get_delta_weight("a") + layer.get_delta_weight("b")
            merged_weights = []
            for order in (["a", "b"], ["b", "a"]):
                layer.merge(adapter_names=order)
                merged_weights.append(base.weight.clone())
                assert bool(lucid.allclose(layer(x), both, atol=1e-4))
                layer.unmerge()
                assert _same(base.weight, weight)
        for merged in merged_weights:
            assert bool(lucid.allclose(merged, expected, atol=1e-5))

    def test_the_stash_is_not_state(self) -> None:
        base = nn.Linear(6, 4)
        layer = hira.Linear(base, "default", _config())
        _randomise(layer)
        keys = sorted(layer.state_dict())
        layer.merge()
        assert sorted(layer.state_dict()) == keys  # not persistent: a file never holds it
        assert (
            len(list(layer.buffers())) == 1
        )  # but a module that holds it moves and copies with it
        layer.unmerge()
        assert len(list(layer.buffers())) == 0

    def test_the_stash_follows_the_layer_to_another_device(self, device: str) -> None:
        base = nn.Linear(6, 4)
        layer = hira.Linear(base, "default", _config())
        _randomise(layer)
        before = base.weight.clone()
        layer.merge()
        layer.to(device)
        layer.unmerge()
        assert _same(base.weight.to("cpu"), before)

    def test_merge_and_unload_gives_the_modulated_model(self, device: str) -> None:
        lucid.manual_seed(1)
        net = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4)).to(device)
        peft = get_peft_model(net, HiraConfig(r=2, init_weights=False, target_modules=["0", "2"]))
        x = lucid.randn(5, 6, device=device)
        with lucid.no_grad():
            adapted = peft(x)
            merged = peft.merge_and_unload()
            assert merged is net and type(net[0]) is nn.Linear
            assert bool(lucid.allclose(net(x), adapted, atol=1e-4))


class TestLayers:
    def test_each_module_type_has_its_layer(self) -> None:
        table = hira.HiraLayer._layer_types
        assert {nn.Linear, nn.Embedding, nn.Conv1d, nn.Conv2d, nn.Conv3d} <= set(table)

    def test_a_model_of_all_the_types_adapts_and_unloads(self) -> None:
        net = nn.Sequential(nn.Embedding(10, 4), nn.Linear(4, 4), nn.Conv1d(4, 4, 3, padding=1))
        config = HiraConfig(r=2, target_modules=["0", "1", "2"])
        peft = get_peft_model(net, config)
        assert [type(m).__name__ for m in peft.base_model.model] == [
            "Embedding",
            "Linear",
            "Conv1d",
        ]
        state = get_peft_model_state_dict(peft)
        assert sorted(state) == [
            "base_model.model.0.hira_embedding_A",
            "base_model.model.0.hira_embedding_B",
            "base_model.model.1.hira_A",
            "base_model.model.1.hira_B",
            "base_model.model.2.hira_A",
            "base_model.model.2.hira_B",
        ]
        assert peft.unload() is net and type(net[0]) is nn.Embedding

    def test_a_grouped_convolution_is_refused_and_the_model_is_left_alone(self) -> None:
        net = nn.Sequential(nn.Conv2d(4, 4, 3, groups=2))
        with pytest.raises(NotImplementedError, match="groups > 1"):
            HiraModel(net, HiraConfig(target_modules=["0"]), "default")
        assert type(net[0]) is nn.Conv2d

    def test_a_module_of_another_type_is_refused(self) -> None:
        net = nn.Sequential(nn.LayerNorm(4))
        with pytest.raises(ValueError, match="not supported"):
            HiraModel(net, HiraConfig(target_modules=["0"]), "default")

    def test_the_rank_pattern_sets_a_modules_own_rank(self) -> None:
        net = nn.Sequential(nn.Linear(6, 8), nn.Linear(8, 4))
        config = HiraConfig(r=2, rank_pattern={"1": 3}, target_modules=["0", "1"])
        peft = get_peft_model(net, config)
        assert peft.base_model.model[0].r == {"default": 2}
        assert peft.base_model.model[1].r == {"default": 3}

    def test_fan_in_fan_out_warns_and_is_cleared(self) -> None:
        net = nn.Sequential(nn.Linear(6, 8))
        with pytest.warns(UserWarning, match="fan_in_fan_out"):
            peft = get_peft_model(net, HiraConfig(r=2, fan_in_fan_out=True, target_modules=["0"]))
        stored = peft.peft_config["default"]
        assert isinstance(stored, HiraConfig) and stored.fan_in_fan_out is False
        with pytest.raises(ValueError, match="fan_in_fan_out"):
            hira.Linear(nn.Linear(6, 8), "x", _config(fan_in_fan_out=True))

    def test_a_deleted_adapter_leaves_nothing_behind(self) -> None:
        layer = hira.Linear(nn.Linear(6, 4), "default", _config())
        layer.update_layer("other", _config())
        layer.delete_adapter("other")
        assert list(layer.hira_A) == ["default"] and list(layer.hira_B) == ["default"]
        assert list(layer.hira_dropout) == ["default"]


class TestHalfPrecision:
    @pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=["float16", "bfloat16"])
    def test_the_adapter_is_float32_beside_a_half_base(self, dtype: lucid.dtype) -> None:
        lucid.manual_seed(1)
        base = nn.Linear(6, 4).to(dtype)
        layer = hira.Linear(base, "default", _config(init_weights=False))
        assert {p.dtype for p in layer._adapter_parameters("default")} == {lucid.float32}
        out = layer(lucid.randn(3, 6).to(dtype))
        assert out.dtype == dtype and bool(lucid.isfinite(out).all().item())
        before = base.weight.clone()
        layer.merge()
        layer.unmerge()
        assert _same(base.weight, before) and base.weight.dtype == dtype


# ── the reference adapter library ────────────────────────────────────────────


def _twins(ref: ModuleType, kind: str) -> tuple[nn.Module, ModuleType, lucid.Tensor, object]:
    """A Lucid module and the reference's with equal weights, and an input for each."""
    lucid.manual_seed(5)
    if kind == "linear":
        mine, theirs = nn.Linear(8, 6), ref.nn.Linear(8, 6)
        x = lucid.randn(4, 8)
    elif kind == "embedding":
        mine, theirs = nn.Embedding(10, 4), ref.nn.Embedding(10, 4)
        x = lucid.tensor([[1, 2, 3], [4, 5, 9]])
    else:
        mine, theirs = nn.Conv2d(3, 5, 3, padding=1), ref.nn.Conv2d(3, 5, 3, padding=1)
        x = lucid.randn(2, 3, 6, 6)
    with ref.no_grad():
        theirs.weight.copy_(ref.tensor(mine.weight.tolist()))
        if kind != "embedding":
            theirs.bias.copy_(ref.tensor(mine.bias.tolist()))
    return mine, theirs, x, ref.tensor(x.tolist())


@pytest.mark.parity
@pytest.mark.parametrize("kind", ["linear", "embedding", "conv2d"])
def test_a_layer_matches_the_reference(ref: ModuleType, kind: str) -> None:
    library = adapter_library()
    base, twin, x, twin_x = _twins(ref, kind)
    options = {"r": 3, "init_weights": False}
    peft = get_peft_model(
        nn.Sequential(base), HiraConfig.from_dict({"target_modules": ["0"], **options})
    )
    theirs = library.get_peft_model(
        ref.nn.Sequential(twin), library.HiraConfig(target_modules=["0"], **options)
    )
    mine, other = peft.base_model.model[0], theirs.base_model.model[0]
    names = (
        ("hira_embedding_A", "hira_embedding_B") if kind == "embedding" else ("hira_A", "hira_B")
    )
    with ref.no_grad():
        for name in names:
            getattr(other, name)["default"].copy_(
                ref.tensor(getattr(mine, name)["default"].tolist())
            )
    assert gap(ref, peft(x), theirs(twin_x)) < 1e-5
    peft.base_model.merge_adapter()
    theirs.base_model.merge_adapter()
    assert gap(ref, mine.base_layer.weight, other.base_layer.weight) < 1e-5
    peft.base_model.unmerge_adapter()
    theirs.base_model.unmerge_adapter()
    assert gap(ref, mine.base_layer.weight, other.base_layer.weight) < 1e-6


@pytest.mark.parity
def test_the_config_file_has_the_references_keys() -> None:
    library = adapter_library()
    ours = HiraConfig(r=4, target_modules=["a"]).to_dict()
    theirs = library.HiraConfig(r=4, target_modules=["a"]).to_dict()
    assert set(ours) == set(theirs) - {"peft_version"}
    assert all(
        ours[k] == (sorted(v) if isinstance(v, set) else v) for k, v in theirs.items() if k in ours
    )
