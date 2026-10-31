"""FourierFT: the spectrum-to-weight map in closed form, the coefficient locations, and the reference's numbers.

The update of an adapter is the real part of the inverse 2-D Fourier transform of a
sparse spectrum.  For coefficients ``c_j`` at ``(u_j, v_j)`` in a ``M x N`` weight
that is ``scale / (M N) * sum_j c_j cos(2 pi (u_j m / M + v_j n / N))``, written out by hand
here.  The locations must be the reference framework's, or an adapter file would not
move between the two: ``test_the_locations_are_the_reference_frameworks`` compares them
for several sizes and seeds.
"""

import math
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import FourierFTConfig, FourierFTModel, PeftModel, get_peft_model
from lucid.peft.tuners import fourierft
from lucid.peft.tuners.fourierft._locations import _entry_indices, _spectrum_positions
from lucid.test.unit.peft._parity import adapter_library, gap


def _layer(
    in_features: int = 6, out_features: int = 4, **kw: object
) -> tuple[fourierft.Linear, nn.Linear]:
    lucid.manual_seed(1)
    base = nn.Linear(in_features, out_features)
    fields: dict[str, object] = {"n_frequency": 5, "target_modules": ["x"]} | kw
    return fourierft.Linear(base, "default", FourierFTConfig(**fields)), base  # type: ignore[arg-type]


class TestConfig:
    def test_the_defaults_are_the_references(self) -> None:
        config = FourierFTConfig(target_modules=["a"])
        assert (config.n_frequency, config.scaling, config.random_loc_seed) == (1000, 150.0, 777)
        assert config.n_frequency_pattern == {} and config.init_weights is False
        assert config.bias == "none" and not config.fan_in_fan_out

    def test_fields_are_checked(self) -> None:
        with pytest.raises(ValueError, match="n_frequency"):
            FourierFTConfig(n_frequency=0, target_modules=["a"])
        with pytest.raises(ValueError, match="random_loc_seed"):
            FourierFTConfig(random_loc_seed=-1, target_modules=["a"])
        with pytest.raises(ValueError, match="regular expression"):
            FourierFTConfig(n_frequency_pattern={"(": 3}, target_modules=["a"])

    def test_trained_biases_are_refused(self) -> None:
        with pytest.raises(NotImplementedError, match="fourier_only"):
            FourierFTConfig(bias="fourier_only", target_modules=["a"])

    def test_a_file_round_trips(self) -> None:
        config = FourierFTConfig(n_frequency=7, scaling=3.0, target_modules=["q", "v"])
        assert FourierFTConfig.from_dict(config.to_dict()) == config
        assert config.to_dict()["peft_type"] == "FOURIERFT"


class TestLocations:
    def test_the_positions_are_distinct_and_repeatable(self) -> None:
        first = _spectrum_positions(10_000, 500, 777)
        assert len(set(first)) == 500 and all(0 <= p < 10_000 for p in first)
        assert _spectrum_positions(10_000, 500, 777) == first
        assert _spectrum_positions(10_000, 500, 778) != first

    def test_asking_for_every_position_is_a_permutation(self) -> None:
        assert sorted(_spectrum_positions(24, 24, 3)) == list(range(24))

    def test_the_indices_split_a_position_into_row_and_column(self) -> None:
        flat = _spectrum_positions(24, 6, 3)
        where = _entry_indices(4, 6, 6, 3, "cpu").tolist()
        assert [r * 6 + c for r, c in zip(where[0], where[1], strict=True)] == list(flat)

    @pytest.mark.parity
    @pytest.mark.parametrize(
        ("size", "count", "seed"),
        [
            (10, 5, 777),
            (3072, 100, 777),
            (589_824, 1000, 777),
            (5, 4, 0),
            (100, 99, 12345),
            (84, 30, 2**40 + 7),
        ],
    )
    def test_the_locations_are_the_reference_frameworks(
        self, ref: ModuleType, size: int, count: int, seed: int
    ) -> None:
        generator = ref.Generator().manual_seed(seed)
        expected = ref.randperm(size, generator=generator)[:count].tolist()
        assert list(_spectrum_positions(size, count, seed)) == expected

    def test_a_spectrum_too_large_for_the_32_bit_draws_is_refused(self) -> None:
        with pytest.raises(NotImplementedError, match="64-bit"):
            _spectrum_positions(300_000_000, 5, 0)


class TestUpdate:
    def test_the_update_is_the_inverse_transform_of_the_sparse_spectrum(self, device: str) -> None:
        layer, _ = _layer(scaling=2.0)
        layer.to(device)
        coefficients = [1.0, -2.0, 0.5, 3.0, -1.5]
        with lucid.no_grad():
            layer.fourierft_spectrum["default"].copy_(lucid.tensor(coefficients, device=device))
        where = _entry_indices(4, 6, 5, 777, "cpu").tolist()
        expected = [
            [
                2.0
                / 24
                * sum(
                    c * math.cos(2 * math.pi * (u * m / 4 + v * n / 6))
                    for c, u, v in zip(coefficients, where[0], where[1], strict=True)
                )
                for n in range(6)
            ]
            for m in range(4)
        ]
        got = layer.get_delta_weight("default").to("cpu")
        assert bool(lucid.allclose(got, lucid.tensor(expected), atol=1e-5))

    def test_the_forward_pass_adds_the_update_to_the_base_output(self) -> None:
        layer, base = _layer()
        x = lucid.randn(3, 6)
        delta = layer.get_delta_weight("default")
        assert bool(lucid.allclose(layer(x), base(x) + x @ delta.mT, atol=1e-5))

    def test_zero_coefficients_start_as_the_base(self) -> None:
        layer, base = _layer(init_weights=True)
        x = lucid.randn(3, 6)
        assert bool((layer(x) == base(x)).all().item())

    def test_every_coefficient_gets_a_gradient(self) -> None:
        layer, _ = _layer()
        layer(lucid.randn(3, 6)).sum().backward()
        grad = layer.fourierft_spectrum["default"].grad
        assert grad is not None and bool((grad != 0).any().item())

    def test_merging_adds_the_update_and_unmerging_removes_it(self) -> None:
        layer, base = _layer()
        weight = base.weight.clone()
        x = lucid.randn(3, 6)
        adapted = layer(x)
        layer.merge()
        assert bool(
            lucid.allclose(base.weight, weight + layer.get_delta_weight("default"), atol=1e-5)
        )
        assert bool(lucid.allclose(layer(x), adapted, atol=1e-4))
        layer.unmerge()
        assert bool(lucid.allclose(base.weight, weight, atol=1e-4))

    def test_more_coefficients_than_weight_entries_are_refused(self) -> None:
        with pytest.raises(ValueError, match="product of the input and output"):
            _layer(n_frequency=25)


class TestModel:
    def test_the_pattern_gives_a_module_its_own_count(self) -> None:
        net = nn.Sequential(nn.Linear(6, 6), nn.ReLU(), nn.Linear(6, 4))
        config = FourierFTConfig(
            n_frequency=5, n_frequency_pattern={"2": 8}, target_modules=["0", "2"]
        )
        peft = get_peft_model(net, config)
        assert isinstance(peft, PeftModel) and isinstance(peft.base_model, FourierFTModel)
        counts = [peft.base_model.model[i].fourierft_n_frequency["default"] for i in (0, 2)]
        assert counts == [5, 8]

    def test_a_second_adapter_uses_its_own_seed(self) -> None:
        net = nn.Sequential(nn.Linear(6, 6))
        peft = get_peft_model(net, FourierFTConfig(n_frequency=5, target_modules=["0"]), "a")
        peft.add_adapter(
            "b", FourierFTConfig(n_frequency=5, random_loc_seed=1, target_modules=["0"])
        )
        assert peft.base_model.model[0].fourierft_random_loc_seed == {"a": 777, "b": 1}

    def test_the_default_target_for_gpt2_is_the_feed_forward_output(self) -> None:
        from lucid.peft.tuners.fourierft._defaults import FOURIERFT_DEFAULT_TARGET_MODULES

        assert FOURIERFT_DEFAULT_TARGET_MODULES["gpt2"] == {"mlp.c_proj"}
        assert FOURIERFT_DEFAULT_TARGET_MODULES["bert"] == {"query", "value"}


# ── the reference adapter library ────────────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize(
    "options",
    [
        {"n_frequency": 20},
        {"n_frequency": 30, "random_loc_seed": 5, "scaling": 3.0},
        {"n_frequency": 84},
    ],
    ids=["default-seed", "seed-and-scale", "every-entry"],
)
def test_a_layer_matches_the_reference(ref: ModuleType, options: dict[str, int | float]) -> None:
    library = adapter_library()
    lucid.manual_seed(7)
    net = nn.Sequential(nn.Linear(12, 7))
    twin = ref.nn.Sequential(ref.nn.Linear(12, 7))
    peft = get_peft_model(net, FourierFTConfig(target_modules=["0"], **options))  # type: ignore[arg-type]
    mine = peft.base_model.model[0]
    with ref.no_grad():
        twin[0].weight.copy_(ref.tensor(mine.base_layer.weight.tolist()))
        twin[0].bias.copy_(ref.tensor(mine.base_layer.bias.tolist()))
    theirs = library.get_peft_model(twin, library.FourierFTConfig(target_modules=["0"], **options))
    other = theirs.base_model.model[0]
    with ref.no_grad():
        other.fourierft_spectrum["default"].copy_(
            ref.tensor(mine.fourierft_spectrum["default"].tolist())
        )
    where = _entry_indices(
        7, 12, int(options["n_frequency"]), int(options.get("random_loc_seed", 777)), "cpu"
    )
    assert where.tolist() == other.indices["default"].tolist()
    x = lucid.randn(4, 12)
    assert gap(ref, mine.get_delta_weight("default"), other.get_delta_weight("default")) < 1e-4
    assert gap(ref, peft(x), theirs(ref.tensor(x.tolist()))) < 1e-3
    peft.base_model.merge_adapter()
    theirs.base_model.merge_adapter()
    assert gap(ref, mine.base_layer.weight, other.base_layer.weight) < 1e-4


@pytest.mark.parity
def test_the_config_file_has_the_references_keys() -> None:
    library = adapter_library()
    ours = FourierFTConfig(n_frequency=9, target_modules=["a"]).to_dict()
    theirs = library.FourierFTConfig(n_frequency=9, target_modules=["a"]).to_dict()
    assert set(ours) == set(theirs) - {"peft_version"}
    assert all(
        ours[k] == (sorted(v) if isinstance(v, set) else v) for k, v in theirs.items() if k in ours
    )
