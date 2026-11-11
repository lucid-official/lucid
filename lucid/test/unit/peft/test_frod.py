"""FRoD: the shared basis and the sparse middle in closed form, the starts, merging against the pristine weight, the files and the reference adapter library's numbers.

The adapted weight ``U (Sigma + S) V^T`` is expected from ``_spectral_forms`` (``U`` from ``W V^-T`` by Gauss-Jordan
on plain lists, ``S`` placed entry by entry), the paper's Theorem 1 is checked on the singular values of the
trained middle, and the ``parity`` tests load a file the reference library wrote (its basis, positions and trained
values) and compare the forward pass, the update and the merged weight.
"""

from types import ModuleType

import pytest

import lucid
import lucid.linalg as linalg
import lucid.nn as nn
from lucid.peft import (
    FrodConfig,
    FrodModel,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from lucid.peft.tuners import frod
from lucid.peft.tuners.fourierft._locations import _spectrum_positions
from lucid.peft.tuners.frod import _basis
from lucid.test.unit.peft import _spectral_forms as forms
from lucid.test.unit.peft._parity import adapter_library, gap
from lucid.test.unit.peft._tuner_recipes import magnitude, span

#: Relative to the largest value compared: a float32 product and a float64 solve cast back round.
_TOLERANCE = 1e-5


class _Block(nn.Module):
    """A block with a query and a value projection and an output head: two categories, ``q`` and ``v``."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.q = nn.Linear(width, width)
        self.v = nn.Linear(width, width)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.q(nn.functional.relu(x)) + self.v(x)


class _Net(nn.Module):
    def __init__(self, width: int = 6, blocks: int = 3) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(width) for _ in range(blocks)])

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        for block in self.layers:
            x = block(x)
        return x


def _net(width: int = 6, blocks: int = 3) -> _Net:
    lucid.manual_seed(1)
    return _Net(width, blocks)


def _layer(
    in_features: int = 6, out_features: int = 8, **kw: object
) -> tuple[frod.Linear, nn.Linear]:
    lucid.manual_seed(1)
    base = nn.Linear(in_features, out_features)
    fields: dict[str, object] = {"sparse_rate": 0.5, "target_modules": ["x"]} | kw
    return frod.Linear(base, "default", FrodConfig(**fields)), base  # type: ignore[arg-type]


def _move(layer: frod.FrodLayer, seed: int = 3) -> None:
    """Move every trained tensor well off its start, as training would."""
    lucid.manual_seed(seed)
    with lucid.no_grad():
        for p in layer._adapter_parameters("default"):
            p.copy_(p + 0.3 * lucid.randn(*p.shape, device=p.device))


def _closed_weight(layer: frod.FrodLayer, base: nn.Linear) -> lucid.Tensor:
    """:math:`W'` from the layer's basis and trained tensors, by the formulas."""
    v, indices, _ = layer._bound().entry(layer._category, "default")
    return lucid.tensor(
        forms.frod_adapted(
            base.weight.tolist(),
            v.tolist(),
            indices.tolist(),
            layer.frod_lambda_l["default"].tolist(),
            layer.frod_lambda_s_values["default"].tolist(),
        )
    )


class TestConfig:
    def test_the_defaults_are_the_references(self) -> None:
        config = FrodConfig(target_modules=["a"])
        assert (config.sparse_rate, config.regularization_alpha, config.projection_prng_key) == (
            0.01,
            0.001,
            0,
        )
        assert (config.save_projection, config.frod_dropout, config.bias) == (True, 0.0, "none")

    def test_fields_are_checked(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            FrodConfig(sparse_rate=1.5, target_modules=["a"])
        with pytest.raises(TypeError, match="save_projection"):
            FrodConfig(save_projection="yes", target_modules=["a"])  # type: ignore[arg-type]
        with pytest.raises(NotImplementedError, match="frod_only"):
            FrodConfig(bias="frod_only", target_modules=["a"])
        with pytest.raises(NotImplementedError, match="unified memory"):
            FrodConfig(runtime_offload_base_weight=True, target_modules=["a"])

    def test_a_file_round_trips(self) -> None:
        config = FrodConfig(sparse_rate=0.05, projection_prng_key=3, target_modules=["q", "v"])
        assert FrodConfig.from_dict(config.to_dict()) == config
        assert config.to_dict()["peft_type"] == "FROD"


class TestCategories:
    @pytest.mark.parametrize(
        ("key", "category", "block"),
        [
            ("encoder.layer.0.attention.self.query", "self_query", 0),
            ("vision_model.encoder.layers.7.self_attn.q_proj", "self_attn_q_proj", 7),
            ("encoder.layer.11.attention.output.dense", "attention_output", 11),
            ("encoder.layer.11.output.dense", "output_dense", 11),
            ("transformer.h.3.mlp.c_fc", "mlp_c_fc", 3),
            ("model.layers.2.0", "0", 2),
            ("0", "0", 0),
            ("classifier.dense", "classifier_dense", 5),
        ],
    )
    def test_a_path_names_the_role_and_the_block(self, key: str, category: str, block: int) -> None:
        assert _basis._category_from_key(key) == category
        assert _basis._layer_index_from_key(key, 5) == block

    def test_the_number_of_entries_is_the_rates_share_of_the_off_diagonal(self) -> None:
        assert _basis._sparse_count(6, 0.5) == 18
        assert _basis._sparse_count(6, 0.0) == 0
        assert _basis._sparse_count(6, 1.0) == 30
        assert _basis._sparse_count(6, 0.001) == 1  # a positive rate keeps at least one
        assert _basis._sparse_count(1, 0.5) == 0


class TestPattern:
    def test_the_positions_are_distinct_off_diagonal_and_follow_the_key(self) -> None:
        indices, size = _basis._draw_pattern(8, 0.25, 11, "cpu")
        assert tuple(indices.shape) == (2, 16) and size.tolist() == [8, 8]
        pairs = list(zip(indices[0].tolist(), indices[1].tolist(), strict=True))
        assert len(set(pairs)) == 16 and all(row != column for row, column in pairs)
        assert indices.tolist() == _basis._draw_pattern(8, 0.25, 11, "cpu")[0].tolist()
        assert indices.tolist() != _basis._draw_pattern(8, 0.25, 12, "cpu")[0].tolist()

    def test_the_positions_are_the_first_of_a_seeded_permutation_of_the_off_diagonal(self) -> None:
        # The reference numbers the off-diagonal entries row by row and takes the first of a seeded permutation.
        width, count = 7, 9
        indices, _ = _basis._draw_pattern(width, count / (width * width), 5, "cpu")
        flat = _spectrum_positions(width * (width - 1), count, 5)
        off = [(i, j) for i in range(width) for j in range(width) if i != j]
        assert list(zip(indices[0].tolist(), indices[1].tolist(), strict=True)) == [
            off[p] for p in flat
        ]

    def test_every_off_diagonal_entry_when_the_rate_is_one(self) -> None:
        indices, _ = _basis._draw_pattern(5, 1.0, 0, "cpu")
        pairs = set(zip(indices[0].tolist(), indices[1].tolist(), strict=True))
        assert pairs == {(i, j) for i in range(5) for j in range(5) if i != j}


class TestBasis:
    def test_a_stack_with_fewer_rows_than_columns_gets_an_orthogonal_basis(self) -> None:
        lucid.manual_seed(0)
        basis = _basis._projection_from_weights([lucid.randn(3, 6)], 1e-3)
        assert span(basis.mT @ basis, lucid.eye(6)) <= _TOLERANCE

    def test_a_basis_and_a_start_rebuild_the_weight(self) -> None:
        lucid.manual_seed(0)
        weights = [lucid.randn(8, 6), lucid.randn(8, 6)]
        basis = _basis._projection_from_weights(weights, 1e-3)
        for weight in weights:
            start = _basis._Start.of(weight, basis, lucid.float32, "cpu")
            rebuilt = start.u @ (lucid.eye(6) * start.norms) @ basis.mT
            assert span(rebuilt, weight) <= _TOLERANCE * magnitude(weight)
            # U has unit columns (Eq. 6), and the norms are what was divided out.
            columns = (start.u * start.u).sum(dim=0)
            assert span(columns, lucid.ones(6)) <= _TOLERANCE

    def test_a_basis_that_cannot_be_inverted_falls_back_to_the_pseudo_inverse(self) -> None:
        weight = lucid.randn(4, 3)
        singular = lucid.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        start = _basis._Start.of(weight, singular, lucid.float32, "cpu")
        assert bool(lucid.isfinite(start.u).all().item()) and start.norms.tolist()[2] == 0.0
        assert start.u[:, 2].tolist() == [0.0] * 4  # no direction: no column


class TestStarts:
    def test_the_adapter_starts_as_the_base_up_to_rounding(self, device: str) -> None:
        layer, base = _layer()
        layer.to(device)
        x = lucid.randn(5, 6, device=device)
        assert span(layer(x), base(x)) <= _TOLERANCE * magnitude(base(x))
        assert (
            span(layer.get_delta_weight("default"), lucid.zeros(8, 6, device=device)) <= _TOLERANCE
        )

    def test_the_trained_tensors_are_the_papers(self) -> None:
        layer, _ = _layer()
        norms = layer.frod_lambda_l["default"]
        entries = layer.frod_lambda_s_values["default"]
        assert tuple(norms.shape) == (6,) and tuple(entries.shape) == (18,)
        assert entries.tolist() == [0.0] * 18  # S starts at zero
        assert layer.r == {"default": 8}
        assert [n for n, p in layer.named_parameters() if p.requires_grad and "frod" in n] == [
            "frod_lambda_l.default",
            "frod_lambda_s_values.default",
        ]

    def test_a_random_start_moves_the_output(self) -> None:
        layer, base = _layer(init_weights=False)
        x = lucid.randn(5, 6)
        assert span(layer(x), base(x)) > 1e-3


class TestUpdate:
    @pytest.mark.parametrize("shape", [(6, 8), (8, 6), (6, 6)], ids=["wide", "tall", "square"])
    def test_the_weight_is_the_papers_u_sigma_plus_s_vt(
        self, shape: tuple[int, int], device: str
    ) -> None:
        layer, base = _layer(in_features=shape[0], out_features=shape[1])
        _move(layer)
        expected = _closed_weight(layer, base)
        layer.to(device)
        x = lucid.randn(4, shape[0], device=device)
        delta = layer.get_delta_weight("default")
        assert span(delta + base.weight, expected.to(device)) <= _TOLERANCE * magnitude(expected)
        with lucid.no_grad():
            adapted = layer(x)
            reference = x @ expected.to(device).mT + base.bias
            assert span(adapted, reference) <= _TOLERANCE * magnitude(reference)

    def test_a_sparse_entry_couples_two_axes_and_nothing_else(self) -> None:
        layer, base = _layer(sparse_rate=0.0)
        # One entry at (row 1, column 4) of S: the update is U[:, 1] V[:, 4]^T, rank one.
        v, _, _ = layer._bound().entry(layer._category, "default")
        layer._bound().put(
            layer._category,
            "default",
            (
                v,
                lucid.tensor([[1], [4]], dtype=lucid.int64),
                lucid.tensor([6, 6], dtype=lucid.int64),
            ),
            True,
        )
        layer.frod_lambda_s_values["default"] = nn.Parameter(lucid.tensor([2.0]))
        u = layer.frod_U["default"]
        expected = 2.0 * u[:, 1:2] @ v[:, 4:5].mT
        assert span(layer.get_delta_weight("default"), expected) <= _TOLERANCE * magnitude(expected)

    def test_the_singular_values_of_the_middle_move_by_at_most_the_spectral_norm_of_the_entries(
        self,
    ) -> None:
        layer, _ = _layer()
        _move(layer)
        v, indices, _ = layer._bound().entry(layer._category, "default")
        entries = layer.frod_lambda_s_values["default"]
        sparse = _basis._Sparse.of(indices, 6).dense(entries)
        diagonal = lucid.eye(6) * layer.frod_lambda_l["default"]
        shifted = linalg.svdvals(diagonal + sparse)
        plain = linalg.svdvals(diagonal)
        assert magnitude(shifted - plain) <= float(linalg.svdvals(sparse).max().item()) * (
            1 + _TOLERANCE
        )

    def test_dropout_acts_on_the_input_in_training_only(self) -> None:
        layer, _ = _layer(frod_dropout=0.5, init_weights=False)
        x = lucid.randn(64, 6)
        layer.eval()
        first = layer(x)
        assert span(layer(x), first) == 0.0
        layer.train()
        assert span(layer(x), first) > 1e-4


class TestMerging:
    def test_merging_writes_the_adapted_weight_and_unmerging_restores_the_base_bit_for_bit(
        self,
    ) -> None:
        layer, base = _layer()
        _move(layer)
        weight = base.weight.clone()
        expected = _closed_weight(layer, base)
        x = lucid.randn(5, 6)
        with lucid.no_grad():
            adapted = layer(x)
        layer.merge()
        assert span(base.weight, expected) <= _TOLERANCE * magnitude(expected)
        assert span(layer(x), adapted) <= _TOLERANCE * magnitude(adapted)
        layer.unmerge()
        assert bool((base.weight == weight).all().item())

    def test_the_update_reads_the_weight_from_before_the_merge(self) -> None:
        net = nn.Sequential(nn.Linear(6, 6))
        config = FrodConfig(sparse_rate=0.5, init_weights=False, target_modules=["0"])
        peft = get_peft_model(net, config, adapter_name="a")
        peft.add_adapter("b", config)
        peft.set_adapter(["a", "b"])
        layer = peft.base_model.model[0]
        weight = layer.base_layer.weight.clone()
        total = weight + layer.get_delta_weight("a") + layer.get_delta_weight("b")
        peft.base_model.merge_adapter()
        assert span(layer.base_layer.weight, total) <= _TOLERANCE * magnitude(total)
        # The second delta is the one of the original weight: asking again changes nothing.
        assert (
            span(weight + layer.get_delta_weight("a") + layer.get_delta_weight("b"), total) == 0.0
        )


class TestSharing:
    def test_the_layers_of_a_category_share_one_basis_made_from_all_of_their_weights(self) -> None:
        net = _net(width=6, blocks=3)
        weights = {
            name: [getattr(block, name).weight.clone() for block in net.layers] for name in "qv"
        }
        peft = get_peft_model(net, FrodConfig(sparse_rate=0.5, target_modules=["q", "v"]))
        layers = peft.base_model.model.layers
        projections = layers[0].q._bound()
        assert sorted(projections.categories()) == ["q", "v"]
        # One basis per category, the same one for every block, and the one the stack of the blocks gives.
        for name in "qv":
            expected = _basis._projection_from_weights(weights[name], 1e-3)
            held = projections.entry(name, "default")[0]
            assert span(held, expected) <= _TOLERANCE * magnitude(expected)
            assert all(getattr(block, name)._bound() is projections for block in layers)
        assert (
            span(projections.entry("q", "default")[0], projections.entry("v", "default")[0]) > 1e-3
        )
        # Each block still starts as itself, over the shared basis.
        x = lucid.randn(4, 6)
        with peft.disable_adapter():
            expected_out = peft(x)
        assert span(peft(x), expected_out) <= 1e-4 * magnitude(expected_out)

    def test_exactly_one_layer_holds_the_tables_in_its_tree(self) -> None:
        net = _net()
        peft = get_peft_model(net, FrodConfig(sparse_rate=0.5, target_modules=["q", "v"]))
        holders = [
            layer
            for _, layer in peft.base_model.model.named_modules()
            if isinstance(layer, frod.FrodLayer) and layer.holds_tables()
        ]
        assert len(holders) == 1
        # One entry per category and adapter in the state dict, under the holder: ``q`` and ``v``.
        assert len([k for k in peft.state_dict() if ".frod_V." in k]) == 2

    def test_layers_of_a_category_with_different_input_widths_are_refused(self) -> None:
        net = _net(width=4, blocks=1)
        net.layers = nn.ModuleList([_Block(4), _Block(6)])
        with pytest.raises(ValueError, match="widths"):
            get_peft_model(net, FrodConfig(target_modules=["q"]))
        assert type(net.layers[0].q) is nn.Linear and all(p.requires_grad for p in net.parameters())

    def test_only_a_linear_is_adapted(self) -> None:
        net = nn.Sequential(nn.Conv2d(2, 4, 3))
        with pytest.raises(ValueError, match="not supported"):
            get_peft_model(net, FrodConfig(target_modules=["0"]))


class TestAdapters:
    def test_a_second_adapter_has_a_basis_of_its_own_from_the_same_key(self) -> None:
        net = nn.Sequential(nn.Linear(6, 6))
        peft = get_peft_model(net, FrodConfig(sparse_rate=0.5, target_modules=["0"]), "a")
        peft.add_adapter("b", FrodConfig(sparse_rate=0.25, target_modules=["0"]))
        layer = peft.base_model.model[0]
        projections = layer._bound()
        assert projections.has("0", "a") and projections.has("0", "b")
        assert tuple(projections.entry("0", "a")[1].shape) == (2, 18)
        assert tuple(projections.entry("0", "b")[1].shape) == (2, 9)
        peft.delete_adapter("b")
        assert projections.has("0", "a") and not projections.has("0", "b")
        assert not any(k.endswith(".b") for k in peft.state_dict())

    def test_adapters_must_agree_on_the_key_and_on_saving_the_projections(self) -> None:
        net = nn.Sequential(nn.Linear(6, 6))
        peft = get_peft_model(net, FrodConfig(projection_prng_key=1, target_modules=["0"]), "a")
        with pytest.raises(ValueError, match="key must be the same"):
            peft.add_adapter("b", FrodConfig(projection_prng_key=2, target_modules=["0"]))
        with pytest.raises(ValueError, match="all adapters or none"):
            peft.add_adapter(
                "b", FrodConfig(projection_prng_key=1, save_projection=False, target_modules=["0"])
            )
        assert sorted(peft.peft_config) == ["a"]


class TestFiles:
    def test_without_the_projections_in_the_file_the_model_makes_them_again(self) -> None:
        net = _net()
        peft = get_peft_model(
            net, FrodConfig(sparse_rate=0.5, save_projection=False, target_modules=["q", "v"])
        )
        state = get_peft_model_state_dict(peft)
        assert not any("frod_V" in k or "frod_s_" in k for k in state)
        with lucid.no_grad():
            for _, layer in peft.base_model.model.named_modules():
                if isinstance(layer, frod.FrodLayer):
                    layer.frod_lambda_s_values["default"].copy_(
                        0.2 * lucid.randn(*layer.frod_lambda_s_values["default"].shape)
                    )
        state = get_peft_model_state_dict(peft)
        fresh = get_peft_model(
            _net(), FrodConfig(sparse_rate=0.5, save_projection=False, target_modules=["q", "v"])
        )
        set_peft_model_state_dict(fresh, state)
        x = lucid.randn(4, 6)
        with lucid.no_grad():
            assert span(fresh(x), peft(x)) <= _TOLERANCE * magnitude(peft(x))

    def test_a_file_that_should_hold_the_projections_and_does_not_is_refused(self) -> None:
        peft = get_peft_model(_net(), FrodConfig(sparse_rate=0.5, target_modules=["q", "v"]))
        state = {
            k: v
            for k, v in get_peft_model_state_dict(peft).items()
            if "frod_V" not in k and "frod_s_" not in k
        }
        with pytest.raises(ValueError, match="were not present"):
            set_peft_model_state_dict(peft, state)

    def test_projections_in_a_file_that_says_it_has_none_are_ignored_with_a_warning(self) -> None:
        saving = get_peft_model(_net(), FrodConfig(sparse_rate=0.5, target_modules=["q", "v"]))
        state = get_peft_model_state_dict(saving)
        regenerating = get_peft_model(
            _net(), FrodConfig(sparse_rate=0.5, save_projection=False, target_modules=["q", "v"])
        )
        with pytest.warns(UserWarning, match="are present"):
            set_peft_model_state_dict(regenerating, state)

    def test_a_loaded_basis_makes_the_frozen_factor_again(self) -> None:
        # A basis other than the one a model makes (here an orthogonal change of it) needs ``U`` from it.
        source = get_peft_model(_net(), FrodConfig(sparse_rate=0.5, target_modules=["q", "v"]))
        state = get_peft_model_state_dict(source)
        rotation = lucid.tensor(
            [[0.0 if i != (j + 1) % 6 else 1.0 for j in range(6)] for i in range(6)]
        )
        changed = {
            k: (v @ rotation if k.startswith("base_model.frod_V.") else v) for k, v in state.items()
        }
        own = get_peft_model(_net(), FrodConfig(sparse_rate=0.5, target_modules=["q", "v"]))
        before = own.base_model.model.layers[0].q.frod_U["default"].clone()
        set_peft_model_state_dict(own, changed)
        layer = own.base_model.model.layers[0].q
        v, _, _ = layer._bound().entry("q", "default")
        assert span(v, changed["base_model.frod_V.q"]) == 0.0
        # U is derived from the loaded basis, not the one the model made: it is another matrix ...
        start = _basis._Start.of(layer.base_layer.weight, v, lucid.float32, "cpu")
        assert span(layer.frod_U["default"], start.u) <= _TOLERANCE
        assert span(layer.frod_U["default"], before) > 1e-3
        # ... and, with the norms of that basis, it rebuilds the weight.
        rebuilt = start.u @ (lucid.eye(6) * start.norms) @ v.mT
        assert span(rebuilt, layer.base_layer.weight) <= _TOLERANCE * magnitude(rebuilt)


def test_the_tuner_wraps_the_linear_layers_and_unloads_them() -> None:
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    peft = get_peft_model(net, FrodConfig(sparse_rate=0.5, target_modules=["0", "2"]))
    assert isinstance(peft, PeftModel) and isinstance(peft.base_model, FrodModel)
    assert type(net[0]) is frod.Linear
    assert peft.unload() is net and type(net[0]) is nn.Linear


# ── the reference adapter library ────────────────────────────────────────────


def _reference_twin(ref: ModuleType, width: int, blocks: int) -> ModuleType:
    """A reference-framework network with the same structure as ``_Net``, weights copied from it."""

    class Block(ref.nn.Module):  # type: ignore[name-defined, misc]
        def __init__(self) -> None:
            super().__init__()
            self.q = ref.nn.Linear(width, width)
            self.v = ref.nn.Linear(width, width)

        def forward(self, x: ModuleType) -> ModuleType:
            return self.q(ref.nn.functional.relu(x)) + self.v(x)

    class Net(ref.nn.Module):  # type: ignore[name-defined, misc]
        def __init__(self) -> None:
            super().__init__()
            self.layers = ref.nn.ModuleList([Block() for _ in range(blocks)])

        def forward(self, x: ModuleType) -> ModuleType:
            for block in self.layers:
                x = block(x)
            return x

    return Net()


@pytest.mark.parity
@pytest.mark.parametrize(("width", "blocks", "rate"), [(8, 3, 0.2), (6, 1, 0.5), (10, 2, 0.05)])
def test_a_file_of_the_reference_loads_and_computes_what_it_does(
    ref: ModuleType, width: int, blocks: int, rate: float
) -> None:
    library = adapter_library()
    net = _net(width, blocks)
    twin = _reference_twin(ref, width, blocks)
    with ref.no_grad():
        for mine, theirs in zip(net.layers, twin.layers, strict=True):
            for name in "qv":
                getattr(theirs, name).weight.copy_(ref.tensor(getattr(mine, name).weight.tolist()))
                getattr(theirs, name).bias.copy_(ref.tensor(getattr(mine, name).bias.tolist()))
    config = {"sparse_rate": rate, "projection_prng_key": 5, "target_modules": ["q", "v"]}
    theirs = library.get_peft_model(twin, library.FrodConfig(progressbar=False, **config))
    peft = get_peft_model(net, FrodConfig(progressbar=False, **config))  # type: ignore[arg-type]
    expected = {k: tuple(v.shape) for k, v in library.get_peft_model_state_dict(theirs).items()}
    assert {k: tuple(v.shape) for k, v in get_peft_model_state_dict(peft).items()} == expected
    # The positions are the reference's, for the first category its generator draws for.
    held = library.get_peft_model_state_dict(theirs)
    mine_state = get_peft_model_state_dict(peft)
    assert (
        held["base_model.frod_s_indices.q"].tolist()
        == mine_state["base_model.frod_s_indices.q"].tolist()
    )
    # Train the reference's tensors a little, hand its file over, and compare what the two compute.
    ref.manual_seed(7)
    with ref.no_grad():
        for p in theirs.parameters():
            if p.requires_grad:
                p.add_(0.3 * ref.randn_like(p))
    file = {
        k: lucid.tensor(v.tolist(), dtype=lucid.int64 if "indices" in k or "size" in k else None)
        for k, v in library.get_peft_model_state_dict(theirs).items()
    }
    set_peft_model_state_dict(peft, file)
    x = lucid.randn(7, width)
    with lucid.no_grad(), ref.no_grad():
        mine_out, their_out = peft(x), theirs(ref.tensor(x.tolist()))
    assert gap(ref, mine_out, their_out) <= 1e-5 * max(1.0, magnitude(mine_out))
    for index in range(blocks):
        for name in "qv":
            left, right = (
                getattr(peft.base_model.model.layers[index], name),
                getattr(theirs.base_model.model.layers[index], name),
            )
            assert (
                gap(ref, left.get_delta_weight("default"), right.get_delta_weight("default")) < 1e-5
            )
    peft.base_model.merge_adapter()
    theirs.base_model.merge_adapter()
    merged = peft.base_model.model.layers[0].q.base_layer.weight
    assert gap(ref, merged, theirs.base_model.model.layers[0].q.base_layer.weight) < 1e-5
    peft.base_model.unmerge_adapter()
    theirs.base_model.unmerge_adapter()
    restored = peft.base_model.model.layers[0].q.base_layer.weight
    assert gap(ref, restored, theirs.base_model.model.layers[0].q.base_layer.weight) < 1e-5


@pytest.mark.parity
def test_the_config_file_has_the_references_keys() -> None:
    library = adapter_library()
    ours = FrodConfig(sparse_rate=0.05, projection_prng_key=3, target_modules=["a"]).to_dict()
    theirs = library.FrodConfig(
        sparse_rate=0.05, projection_prng_key=3, target_modules=["a"]
    ).to_dict()
    assert set(ours) == set(theirs) - {"peft_version"}
    assert all(
        ours[k] == (sorted(v) if isinstance(v, set) else v) for k, v in theirs.items() if k in ours
    )
