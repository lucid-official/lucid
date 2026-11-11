"""WaveFT: the update in closed form, the starts, the coefficient counts, the file and the reference library's numbers.

The paper's update is the inverse 2-D wavelet transform of a coefficient matrix that is zero but at
``n_frequency`` random positions, times a scale.  The tests lay the trained coefficients at their
positions, pad, split and inverse-transform them with the plain loops of ``_wavelet_forms``, and
expect the layer, its dense update and its merged weight to be that matrix.  The ``parity`` tests
build the same layer in the reference library, copy the coefficients across and compare the update
matrix itself, the forward pass, the merged weight and the stored shapes, for several wavelet
families and for sizes that need an odd pad.
"""

from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    PeftModel,
    WaveFTConfig,
    WaveFTModel,
    get_peft_model,
    get_peft_model_state_dict,
)
from lucid.peft.tuners import waveft
from lucid.peft.tuners.fourierft._locations import _spectrum_positions
from lucid.peft.tuners.waveft._filters import _filter_bank
from lucid.test.unit.peft import _sparse_forms as forms
from lucid.test.unit.peft._parity import adapter_library, gap
from lucid.test.unit.peft._tuner_recipes import magnitude, same, span

#: Relative to the largest value compared: one float32 transform.
_TOLERANCE = 1e-5


def _layer(out: int = 8, inn: int = 6, **kw: object) -> tuple[waveft.Linear, nn.Linear]:
    lucid.manual_seed(1)
    base = nn.Linear(inn, out)
    fields: dict[str, object] = {"n_frequency": 7, "target_modules": ["x"]} | kw
    return waveft.Linear(base, "default", WaveFTConfig(**fields)), base  # type: ignore[arg-type]


def _closed_delta(layer: waveft.WaveFTLayer, adapter: str = "default") -> list[list[float]]:
    """``scaling * IDWT(coefficients)`` entry by entry, the coefficients at the seeded permutation's positions."""
    out, inn = layer.out_features, layer.in_features
    count = layer.waveft_n_frequency[adapter]
    flat = _spectrum_positions(out * inn, count, layer.waveft_random_loc_seed[adapter])
    bank = _filter_bank(layer.waveft_wavelet_family[adapter])
    return forms.waveft_delta(
        (out, inn),
        [p // inn for p in flat],
        [p % inn for p in flat],
        layer.waveft_spectrum[adapter].tolist(),
        (bank.rec_lo, bank.rec_hi),
        layer.waveft_scaling[adapter],
        layer.waveft_use_idwt[adapter],
    )


def _move(layer: waveft.WaveFTLayer, seed: int = 3) -> None:
    lucid.manual_seed(seed)
    with lucid.no_grad():
        for p in layer._adapter_parameters("default"):
            p.copy_(p + 0.3 * lucid.randn(*p.shape, device=p.device))


class TestConfig:
    def test_the_defaults_are_the_references(self) -> None:
        config = WaveFTConfig(target_modules=["a"])
        assert (config.n_frequency, config.scaling, config.wavelet_family) == (2592, 25.0, "db1")
        assert (config.use_idwt, config.random_loc_seed, config.bias) == (True, 777, "none")
        assert (config.n_frequency_pattern, config.proportional_parameters) == ({}, False)
        assert (config.init_weights, config.fan_in_fan_out) == (True, False)

    def test_fields_are_checked(self) -> None:
        with pytest.raises(ValueError, match="n_frequency"):
            WaveFTConfig(n_frequency=0, target_modules=["a"])
        with pytest.raises(ValueError, match="bias"):
            WaveFTConfig(bias="some", target_modules=["a"])  # type: ignore[arg-type]
        for kind in ("all", "waveft_only"):
            with pytest.raises(NotImplementedError, match="bias"):
                WaveFTConfig(bias=kind, target_modules=["a"])  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="use_idwt"):
            WaveFTConfig(use_idwt="yes", target_modules=["a"])  # type: ignore[arg-type]

    @pytest.mark.parametrize("name", ["db1", "db38", "sym2", "sym20", "coif1", "coif17"])
    def test_the_wavelet_names_the_reference_knows_are_accepted(self, name: str) -> None:
        assert WaveFTConfig(wavelet_family=name, target_modules=["a"]).wavelet_family == name

    @pytest.mark.parametrize("name", ["haar", "db0", "db39", "sym1", "sym21", "coif18", "bior1.3"])
    def test_any_other_name_is_refused(self, name: str) -> None:
        with pytest.raises(ValueError, match="not supported"):
            WaveFTConfig(wavelet_family=name, target_modules=["a"])

    def test_a_file_round_trips(self) -> None:
        config = WaveFTConfig(
            n_frequency=64,
            wavelet_family="sym4",
            n_frequency_pattern={"q": 8},
            target_modules=["q", "v"],
        )
        assert WaveFTConfig.from_dict(config.to_dict()) == config
        assert config.to_dict()["peft_type"] == "WAVEFT"


class TestStarts:
    @pytest.mark.parametrize("idwt", [True, False])
    def test_the_default_start_is_the_base_bit_for_bit(self, idwt: bool, device: str) -> None:
        layer, base = _layer(use_idwt=idwt)
        layer.to(device)
        x = lucid.randn(5, 6, device=device)
        assert same(layer(x), base(x))
        assert not bool(layer.get_delta_weight("default").any().item())

    def test_a_random_start_is_small_and_moves_the_output(self) -> None:
        layer, base = _layer(n_frequency=40, init_weights=False)
        coefficients = layer.waveft_spectrum["default"]
        assert 0.003 < float(coefficients.std().item()) < 0.03
        x = lucid.randn(5, 6)
        assert span(layer(x), base(x)) > 1e-4

    def test_the_coefficients_are_one_vector_of_the_chosen_count(self) -> None:
        layer, _ = _layer(n_frequency=11)
        assert tuple(layer.waveft_spectrum["default"].shape) == (11,)
        assert layer.waveft_n_frequency == {"default": 11}


class TestUpdate:
    @pytest.mark.parametrize(
        ("family", "idwt", "out", "inn"),
        [
            ("db1", True, 8, 6),
            ("db2", True, 8, 6),
            ("db2", True, 7, 5),  # odd sides: padded one more
            ("sym4", True, 9, 12),
            ("coif1", True, 8, 7),
            ("db3", False, 8, 6),
        ],
    )
    def test_the_layer_applies_the_inverse_transform_of_its_coefficients(
        self, family: str, idwt: bool, out: int, inn: int, device: str
    ) -> None:
        layer, base = _layer(out, inn, wavelet_family=family, use_idwt=idwt, scaling=3.0)
        layer.to(device)
        _move(layer)
        x = lucid.randn(5, inn, device=device)
        with lucid.no_grad():
            delta = lucid.tensor(_closed_delta(layer)).to(device)
            expected = x @ (base.weight + delta).mT + base.bias
            assert tuple(layer.get_delta_weight("default").shape) == (out, inn)
            assert span(layer.get_delta_weight("default"), delta) <= _TOLERANCE * max(
                1.0, magnitude(delta)
            )
            assert span(layer(x), expected) <= _TOLERANCE * magnitude(expected)

    def test_the_positions_are_the_first_entries_of_the_seeded_permutation(self) -> None:
        layer, _ = _layer(n_frequency=4, use_idwt=False, scaling=1.0)
        with lucid.no_grad():
            layer.waveft_spectrum["default"].copy_(lucid.tensor([1.0, 2.0, 3.0, 4.0]))
        delta = layer.get_delta_weight("default").tolist()
        flat = _spectrum_positions(48, 4, 777)
        assert {i * 6 + j: delta[i][j] for i in range(8) for j in range(6) if delta[i][j]} == {
            p: float(k + 1) for k, p in enumerate(flat)
        }

    def test_another_seed_places_the_coefficients_elsewhere(self) -> None:
        first, _ = _layer(use_idwt=False)
        second, _ = _layer(use_idwt=False, random_loc_seed=5)
        for layer in (first, second):
            with lucid.no_grad():
                layer.waveft_spectrum["default"].copy_(lucid.ones(7))
        assert span(first.get_delta_weight("default"), second.get_delta_weight("default")) > 0.5

    def test_the_scale_multiplies_the_update(self) -> None:
        one, _ = _layer(scaling=1.0, wavelet_family="db2")
        four, _ = _layer(scaling=4.0, wavelet_family="db2")
        for layer in (one, four):
            _move(layer)
        assert span(four.get_delta_weight("default"), 4 * one.get_delta_weight("default")) <= 1e-5

    def test_only_the_coefficients_get_a_gradient(self) -> None:
        net = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4))
        peft = get_peft_model(
            net,
            WaveFTConfig(
                n_frequency=5, wavelet_family="db2", target_modules=["0", "2"], init_weights=False
            ),
        )
        peft(lucid.randn(3, 6)).sum().backward()
        for name, p in peft.named_parameters():
            if p.requires_grad:
                assert "waveft_spectrum" in name and p.grad is not None, name
            else:
                assert p.grad is None, name

    def test_a_coefficient_count_above_the_weights_entries_is_refused(self) -> None:
        with pytest.raises(ValueError, match="product of the input and output dimensions"):
            _layer(n_frequency=49)  # the weight has 48

    def test_a_wavelet_without_filters_is_refused_when_the_layer_is_built_unless_unused(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="No filters for wavelet 'db11'"):
            _layer(wavelet_family="db11")
        layer, _ = _layer(wavelet_family="db11", use_idwt=False)
        assert layer.waveft_wavelet_family == {"default": "db11"}

    def test_fan_in_fan_out_is_refused_on_a_layer_and_reset_by_the_tuner(self) -> None:
        with pytest.raises(ValueError, match="fan_in_fan_out"):
            _layer(fan_in_fan_out=True)
        with pytest.warns(UserWarning, match="fan_in_fan_out"):
            peft = get_peft_model(
                nn.Sequential(nn.Linear(6, 8)),
                WaveFTConfig(n_frequency=3, fan_in_fan_out=True, target_modules=["0"]),
            )
        assert peft.peft_config["default"].fan_in_fan_out is False  # type: ignore[attr-defined]


class TestMerging:
    @pytest.mark.parametrize("idwt", [True, False])
    def test_merging_adds_the_update_to_the_weight(self, idwt: bool) -> None:
        layer, base = _layer(wavelet_family="sym2", use_idwt=idwt)
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
        config = WaveFTConfig(
            n_frequency=7, wavelet_family="db2", target_modules=["0"], init_weights=False
        )
        peft = get_peft_model(net, config)
        peft.add_adapter(
            "b",
            WaveFTConfig(
                n_frequency=5, random_loc_seed=3, target_modules=["0"], init_weights=False
            ),
        )
        peft.set_adapter(["default", "b"])
        layer = peft.base_model.model[0]
        x = lucid.randn(4, 6)
        with lucid.no_grad():
            total = layer.get_delta_weight("default") + layer.get_delta_weight("b")
            expected = x @ (layer.get_base_layer().weight + total).mT + layer.get_base_layer().bias
            assert span(peft(x), expected) <= _TOLERANCE * magnitude(expected)
            peft.base_model.merge_adapter()
            assert span(peft(x), expected) <= _TOLERANCE * magnitude(expected)


class TestPrecision:
    def test_a_half_base_keeps_float32_coefficients_and_its_own_output_dtype(self) -> None:
        lucid.manual_seed(1)
        base = nn.Linear(6, 8).to(lucid.float16)
        layer = waveft.Linear(
            base,
            "default",
            WaveFTConfig(
                n_frequency=7, wavelet_family="db2", init_weights=False, target_modules=["x"]
            ),
        )
        assert layer.waveft_spectrum["default"].dtype == lucid.float32
        assert layer(lucid.randn(3, 6).to(lucid.float16)).dtype == lucid.float16

    def test_a_float64_base_keeps_a_float64_update_on_the_cpu(self) -> None:
        lucid.manual_seed(1)
        base = nn.Linear(6, 8).to(lucid.float64)
        layer = waveft.Linear(
            base,
            "default",
            WaveFTConfig(
                n_frequency=7, wavelet_family="sym4", init_weights=False, target_modules=["x"]
            ),
        )
        assert layer.waveft_spectrum["default"].dtype == lucid.float64
        assert layer.get_delta_weight("default").dtype == lucid.float64


class TestCounts:
    def _two_layers(self) -> nn.Sequential:
        lucid.manual_seed(0)
        return nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4))

    def _counts(self, peft: PeftModel) -> dict[str, int]:
        return {
            name: layer.waveft_n_frequency["default"]
            for name, layer in peft.base_model.model.named_modules()
            if isinstance(layer, waveft.WaveFTLayer)
        }

    def test_a_pattern_by_name_or_by_regex_sets_a_layers_count(self) -> None:
        peft = get_peft_model(
            self._two_layers(),
            WaveFTConfig(
                n_frequency=7, n_frequency_pattern={"0": 3, "2$": 5}, target_modules=["0", "2"]
            ),
        )
        assert self._counts(peft) == {"0": 3, "2": 5}

    def test_a_layer_the_pattern_does_not_name_takes_the_default(self) -> None:
        peft = get_peft_model(
            self._two_layers(),
            WaveFTConfig(n_frequency=7, n_frequency_pattern={"0": 3}, target_modules=["0", "2"]),
        )
        assert self._counts(peft) == {"0": 3, "2": 7}

    def test_proportional_parameters_share_the_total_by_weight_size(self) -> None:
        peft = get_peft_model(
            self._two_layers(),
            WaveFTConfig(n_frequency=10, proportional_parameters=True, target_modules=["0", "2"]),
        )
        # Budget 20 over weights of 48 and 32 entries: 20 * 48/80 = 12 and 20 * 32/80 = 8.
        assert self._counts(peft) == {"0": 12, "2": 8}
        assert peft.peft_config["default"].n_frequency_pattern == {}  # type: ignore[attr-defined]

    def test_a_proportional_share_is_at_least_one(self) -> None:
        lucid.manual_seed(0)
        net = nn.Sequential(nn.Linear(40, 40), nn.Linear(2, 1))
        peft = get_peft_model(
            net,
            WaveFTConfig(n_frequency=1, proportional_parameters=True, target_modules=["0", "1"]),
        )
        assert self._counts(peft)["1"] == 1

    def test_the_counts_of_a_second_adapter_follow_its_own_config(self) -> None:
        peft = get_peft_model(
            self._two_layers(), WaveFTConfig(n_frequency=7, target_modules=["0", "2"])
        )
        peft.add_adapter(
            "b",
            WaveFTConfig(n_frequency=2, n_frequency_pattern={"2": 9}, target_modules=["0", "2"]),
        )
        layers = {
            name: layer
            for name, layer in peft.base_model.model.named_modules()
            if isinstance(layer, waveft.WaveFTLayer)
        }
        assert {n: layer.waveft_n_frequency["b"] for n, layer in layers.items()} == {"0": 2, "2": 9}


class TestFile:
    def test_the_file_holds_one_coefficient_vector_a_layer_and_no_positions(self) -> None:
        peft = get_peft_model(
            nn.Sequential(nn.Linear(6, 8)), WaveFTConfig(n_frequency=7, target_modules=["0"])
        )
        state = get_peft_model_state_dict(peft)
        assert {k: tuple(v.shape) for k, v in state.items()} == {
            "base_model.model.0.waveft_spectrum": (7,)
        }

    def test_the_positions_are_recomputed_on_load(self, tmp_path: object) -> None:
        lucid.manual_seed(0)
        config = WaveFTConfig(
            n_frequency=7, wavelet_family="sym4", target_modules=["0"], init_weights=False
        )
        peft = get_peft_model(nn.Sequential(nn.Linear(6, 8)), config)
        peft.save_pretrained(str(tmp_path))
        loaded = PeftModel.from_pretrained(nn.Sequential(nn.Linear(6, 8)), str(tmp_path))
        mine, theirs = peft.base_model.model[0], loaded.base_model.model[0]
        assert same(mine.waveft_spectrum["default"], theirs.waveft_spectrum["default"])
        assert span(mine.get_delta_weight("default"), theirs.get_delta_weight("default")) == 0.0


class TestTargets:
    def test_gpt2_takes_the_feed_forward_output_projection_by_default(self) -> None:
        from lucid.peft.tuners.waveft._defaults import WAVEFT_DEFAULT_TARGET_MODULES

        assert WAVEFT_DEFAULT_TARGET_MODULES["gpt2"] == frozenset({"mlp.c_proj"})
        assert WAVEFT_DEFAULT_TARGET_MODULES["bert"] == frozenset({"query", "value"})
        assert WAVEFT_DEFAULT_TARGET_MODULES["t5"] == frozenset({"q", "v"})

    def test_only_a_linear_is_adapted(self) -> None:
        net = nn.Sequential(nn.Conv2d(2, 4, 3))
        with pytest.raises(ValueError, match="not supported"):
            get_peft_model(net, WaveFTConfig(n_frequency=2, target_modules=["0"]))


def test_the_tuner_wraps_the_linear_layers_and_unloads_them() -> None:
    net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    peft = get_peft_model(net, WaveFTConfig(n_frequency=4, target_modules=["0", "2"]))
    assert isinstance(peft.base_model, WaveFTModel)
    assert type(net[0]) is waveft.Linear and type(net[2]) is waveft.Linear
    assert peft.unload() is net and type(net[0]) is nn.Linear


# ── the reference adapter library ────────────────────────────────────────────


def _pair(ref: ModuleType, out: int, inn: int, **options: object) -> tuple[PeftModel, PeftModel]:
    """The same adapted ``Linear(inn, out)`` in both libraries, base weights and coefficients copied across."""
    library = adapter_library()
    lucid.manual_seed(5)
    net = nn.Sequential(nn.Linear(inn, out))
    twin = ref.nn.Sequential(ref.nn.Linear(inn, out))
    peft = get_peft_model(
        net,
        WaveFTConfig(target_modules=["0"], init_weights=False, **options),  # type: ignore[arg-type]
    )
    mine = peft.base_model.model[0]
    with ref.no_grad():
        twin[0].weight.copy_(ref.tensor(mine.base_layer.weight.tolist()))
        twin[0].bias.copy_(ref.tensor(mine.base_layer.bias.tolist()))
    theirs = library.get_peft_model(
        twin, library.WaveFTConfig(target_modules=["0"], init_weights=False, **options)
    )
    with ref.no_grad():
        theirs.base_model.model[0].waveft_spectrum["default"].copy_(
            ref.tensor(mine.waveft_spectrum["default"].tolist())
        )
    return peft, theirs


@pytest.mark.parity
@pytest.mark.parametrize(
    ("family", "idwt", "out", "inn"),
    [
        ("db1", True, 10, 6),
        ("db2", True, 9, 7),
        ("db4", True, 16, 12),
        ("sym3", True, 11, 8),
        ("sym8", True, 20, 18),
        ("coif1", True, 12, 9),
        ("coif2", True, 24, 20),
        ("db1", False, 10, 6),
    ],
)
def test_a_layer_matches_the_reference(
    ref: ModuleType, family: str, idwt: bool, out: int, inn: int
) -> None:
    peft, theirs = _pair(
        ref, out, inn, n_frequency=9, wavelet_family=family, use_idwt=idwt, scaling=4.0
    )
    mine = peft.base_model.model[0]
    other = theirs.base_model.model[0]
    assert gap(ref, mine.get_delta_weight("default"), other.get_delta_weight("default")) <= 1e-6
    x = lucid.randn(4, inn)
    assert gap(ref, peft(x), theirs(ref.tensor(x.tolist()))) <= 1e-5
    assert {k: tuple(v.shape) for k, v in other.state_dict().items() if "waveft" in k} == {
        k: tuple(v.shape) for k, v in mine.state_dict().items() if "waveft" in k
    }
    peft.base_model.merge_adapter()
    theirs.base_model.merge_adapter()
    assert gap(ref, mine.base_layer.weight, other.base_layer.weight) <= 1e-5
    peft.base_model.unmerge_adapter()
    theirs.base_model.unmerge_adapter()
    assert gap(ref, mine.base_layer.weight, other.base_layer.weight) <= 1e-5


@pytest.mark.parity
@pytest.mark.parametrize(
    "options",
    [
        {"n_frequency": 6, "n_frequency_pattern": {"0": 11}},
        {"n_frequency": 4, "proportional_parameters": True},
    ],
)
def test_the_coefficient_counts_match_the_reference(
    ref: ModuleType, options: dict[str, object]
) -> None:
    library = adapter_library()
    lucid.manual_seed(5)
    net = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4))
    twin = ref.nn.Sequential(ref.nn.Linear(6, 8), ref.nn.ReLU(), ref.nn.Linear(8, 4))
    peft = get_peft_model(net, WaveFTConfig(target_modules=["0", "2"], **options))  # type: ignore[arg-type]
    theirs = library.get_peft_model(
        twin, library.WaveFTConfig(target_modules=["0", "2"], **options)
    )
    ours = {i: peft.base_model.model[i].waveft_n_frequency["default"] for i in (0, 2)}
    other = {i: theirs.base_model.model[i].waveft_n_frequency["default"] for i in (0, 2)}
    assert ours == other


@pytest.mark.parity
@pytest.mark.usefixtures("ref")
def test_the_config_file_has_the_references_keys() -> None:
    library = adapter_library()
    options: dict[str, object] = {
        "n_frequency": 64,
        "wavelet_family": "sym4",
        "target_modules": ["a"],
    }
    ours = WaveFTConfig(**options).to_dict()  # type: ignore[arg-type]
    theirs = library.WaveFTConfig(**options).to_dict()
    assert set(ours) == set(theirs) - {"peft_version"}
    assert all(
        ours[k] == (sorted(v) if isinstance(v, set) else v) for k, v in theirs.items() if k in ours
    )
