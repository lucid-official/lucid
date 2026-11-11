"""AdaMSS: the split into subspaces, the update in closed form, merging, the file, the allocation and the reference's numbers.

The paper's layer (Zheng et al., 2025, NeurIPS, section 3.3) adds ``Pi [B_1 A_1; ...; B_K A_K] V_r^T``
to the weight joined to its bias, ``Pi`` returning each subspace's rows to where the weight had them.  The
tests state that sum with plain loops (``_adamss_forms``), never through the layer's own code, and hold
the split to what makes it a split: every row in exactly one subspace, orthonormal factors, a k-means that
is Lloyd's algorithm from the reference's seeds.  The ``parity`` tests build the same layer in the
reference library (which needs scikit-learn for its clustering), compare keys, shapes, the split and the
forward pass of its state loaded here, the clustering itself and the masks the allocation produces step by
step.
"""

import types
from pathlib import Path
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.peft import (
    AdamssConfig,
    AdamssModel,
    PeftModel,
    PeftType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from lucid.peft.tuners._layer import _tuner_layers
from lucid.peft.tuners.adamss import AdamssLayer, Linear, SubspaceAllocator
from lucid.peft.tuners.adamss._kmeans import _kmeans_labels, _Stream
from lucid.peft.tuners.adamss._layer import _parameters
from lucid.peft.tuners.adamss._subspaces import _subspace_rank
from lucid.serialization import save_safetensors
from lucid.test.unit.peft import _adamss_forms as forms
from lucid.test.unit.peft._parity import across, adapter_library, gap
from lucid.test.unit.peft._tuner_recipes import magnitude, same, span

#: Relative to the largest value compared: a few float32 sums.
_TOLERANCE = 2e-6


def _layer(
    rows: int = 12, cols: int = 4, bias: bool = True, **kw: object
) -> tuple[Linear, nn.Linear]:
    lucid.manual_seed(1)
    base = nn.Linear(cols, rows, bias=bias)
    fields: dict[str, object] = {
        "r": 8,
        "num_subspaces": 3,
        "subspace_rank": 2,
        "target_modules": ["x"],
    } | kw
    return Linear(base, "default", AdamssConfig(**fields)), base  # type: ignore[arg-type]


def _factors(layer: AdamssLayer, adapter: str = "default") -> tuple[list[list[list[float]]], ...]:
    """``(A_k, B_k)`` of every subspace as nested lists."""
    return (
        [p.tolist() for p in _parameters(layer.adamss_A, adapter)],
        [p.tolist() for p in _parameters(layer.adamss_B, adapter)],
    )


def _members(layer: AdamssLayer, adapter: str = "default") -> list[list[int]]:
    """The weight rows of each subspace: the layer's row order cut at the sizes of the ``B_k``."""
    order = [int(v) for v in layer.scatter_index[adapter].tolist()]
    cuts: list[list[int]] = []
    start = 0
    for update in _parameters(layer.adamss_B, adapter):
        cuts.append(order[start : start + int(update.shape[0])])
        start += int(update.shape[0])
    return cuts


def _train(layer: AdamssLayer, seed: int = 3, adapter: str = "default") -> None:
    """Move every trained tensor of ``adapter`` well off its start, as optimizer steps would."""
    lucid.manual_seed(seed)
    with lucid.no_grad():
        for p in layer._adapter_parameters(adapter):
            p.copy_(p + 0.3 * lucid.randn(*p.shape, device=p.device))


def _expected_delta(layer: AdamssLayer, adapter: str = "default") -> forms.Matrix:
    factors, updates = _factors(layer, adapter)
    return forms.joined_delta(
        _members(layer, adapter),
        factors,
        updates,
        layer.adamss_newB[adapter].tolist(),
        layer.out_features,
    )


class TestSplit:
    def test_every_row_is_in_exactly_one_subspace(self) -> None:
        layer, _ = _layer()
        members = _members(layer)
        assert len(members) == 3
        assert sorted(row for rows in members for row in rows) == list(range(12))
        assert all(rows == sorted(rows) for rows in members)

    def test_the_projection_has_orthonormal_rows(self) -> None:
        layer, _ = _layer(r=3)
        v = layer.adamss_newB["default"]
        assert tuple(v.shape) == (3, 5)  # the weight's 4 columns and the bias's
        assert span(v @ v.mT, lucid.eye(3)) <= 1e-5

    def test_the_projection_spans_the_leading_row_space_of_the_joined_weight(self) -> None:
        layer, base = _layer(r=3)
        joined = lucid.cat([base.weight, base.bias.reshape(-1, 1)], dim=1).detach()
        v = layer.adamss_newB["default"]
        _, s, _ = lucid.linalg.svd(joined, full_matrices=False)
        # a rank-3 subspace holds at most the three largest singular values of the weight, and the
        # randomised SVD with power iterations finds nearly all of it: the energy it keeps is a close
        # share of the leading directions' and never exceeds them
        kept = float(((joined @ v.mT) ** 2).sum().item())
        best = float((s[:3] ** 2).sum().item())
        assert kept <= best * (1 + 1e-5) and kept >= 0.95 * best

    def test_the_starts_have_orthonormal_rows_of_the_requested_rank(self) -> None:
        layer, _ = _layer(subspace_rank=2)
        for a, b in zip(
            _parameters(layer.adamss_A, "default"),
            _parameters(layer.adamss_B, "default"),
            strict=True,
        ):
            rank = min(int(b.shape[0]), 2)
            assert tuple(a.shape) == (rank, layer.adamss_newB["default"].shape[0])
            assert span(a @ a.mT, lucid.eye(rank)) <= 1e-5
            assert b.shape[1] == rank

    def test_the_rank_is_the_rows_when_the_subspace_has_fewer(self) -> None:
        layer, _ = _layer(rows=4, num_subspaces=2, subspace_rank=7)
        for a, b in zip(
            _parameters(layer.adamss_A, "default"),
            _parameters(layer.adamss_B, "default"),
            strict=True,
        ):
            assert a.shape[0] == b.shape[1] == min(int(b.shape[0]), a.shape[1])

    def test_the_sizes_of_the_factors_follow_the_split(self) -> None:
        layer, _ = _layer()
        r = layer.adamss_newB["default"].shape[0]
        sizes = [len(rows) for rows in _members(layer)]
        for k, (a, b) in enumerate(
            zip(
                _parameters(layer.adamss_A, "default"),
                _parameters(layer.adamss_B, "default"),
                strict=True,
            )
        ):
            assert tuple(b.shape) == (sizes[k], a.shape[0]) and a.shape[1] == r

    def test_the_rank_of_the_svd_is_clamped_to_the_joined_weight(self) -> None:
        layer, _ = _layer(rows=12, cols=4, r=100)
        assert layer.r["default"] == 5  # min(100, 12, 4 + 1)
        layer, _ = _layer(rows=12, cols=4, bias=False, r=100)
        assert layer.r["default"] == 4

    def test_more_subspaces_than_rows_make_one_subspace_of_each_row(self) -> None:
        layer, _ = _layer(rows=3, num_subspaces=9, r=3)
        assert layer.num_subspaces["default"] == 3
        assert sorted(len(rows) for rows in _members(layer)) == [1, 1, 1]

    def test_the_split_is_the_same_each_time_and_on_every_device(self) -> None:
        first, _ = _layer()
        second, _ = _layer()
        assert _members(first) == _members(second)
        assert same(first.adamss_newB["default"], second.adamss_newB["default"])

    def test_the_default_generator_is_left_where_it_was_found(self) -> None:
        base = nn.Linear(4, 12)
        lucid.manual_seed(11)
        state = lucid.get_rng_state()
        Linear(
            base,
            "default",
            AdamssConfig(r=8, num_subspaces=3, init_weights=None, target_modules=["x"]),
        )
        assert same(state, lucid.get_rng_state())

    def test_a_default_device_that_is_not_the_weights_changes_nothing(self, device: str) -> None:
        base = nn.Linear(4, 12).to(device)
        layer = Linear(base, "default", AdamssConfig(r=8, num_subspaces=3, target_modules=["x"]))
        assert layer.adamss_newB["default"].device.type == device
        assert layer.adamss_rows["default"].device.type == device
        assert all(p.device.type == device for p in layer._adapter_parameters("default"))

    def test_a_weight_of_equal_rows_has_no_subspaces_to_find(self) -> None:
        base = nn.Linear(4, 6)
        with lucid.no_grad():
            base.weight.copy_(lucid.ones(6, 4))
            base.bias.copy_(lucid.ones(6))
        with pytest.raises(ValueError, match="distinct clusters"):
            Linear(base, "default", AdamssConfig(r=1, num_subspaces=2, target_modules=["x"]))

    def test_the_split_reads_the_weight_as_it_was_before_a_merge(self) -> None:
        net = nn.Sequential(nn.Linear(4, 12))
        config = AdamssConfig(r=5, num_subspaces=3, target_modules=["0"], init_weights=None)
        peft = get_peft_model(net, config)
        layer = peft.base_model.model[0]
        peft.base_model.merge_adapter()
        peft.add_adapter("b", config)
        peft.base_model.unmerge_adapter()
        assert _members(layer, "b") == _members(layer, "default")


class TestSubspaceRank:
    def test_it_is_the_configured_rank_capped_by_the_rows(self) -> None:
        sizes = lucid.tensor([3.0, 2.0, 1.0])
        config = AdamssConfig(subspace_rank=5, target_modules=["x"])
        assert _subspace_rank(sizes, 3, config) == 3
        assert _subspace_rank(sizes, 2, AdamssConfig(subspace_rank=1, target_modules=["x"])) == 1

    def test_dynamic_rank_counts_the_gram_singular_values_above_the_threshold(self) -> None:
        # singular values 4, 2, 1, 0.1 of the rows give Gram values 16, 4, 1, 0.01; 10% of 16 is 1.6.
        sizes = lucid.tensor([4.0, 2.0, 1.0, 0.1])
        config = AdamssConfig(
            subspace_rank=4, use_dynamic_rank=True, svd_threshold=0.1, target_modules=["x"]
        )
        assert _subspace_rank(sizes, 4, config) == 2
        config = AdamssConfig(
            subspace_rank=1, use_dynamic_rank=True, svd_threshold=0.1, target_modules=["x"]
        )
        assert _subspace_rank(sizes, 4, config) == 1

    def test_the_rank_is_at_least_one(self) -> None:
        sizes = lucid.tensor([0.0, 0.0])
        config = AdamssConfig(use_dynamic_rank=True, subspace_rank=2, target_modules=["x"])
        assert _subspace_rank(sizes, 2, config) == 1


class TestKMeans:
    def test_the_generator_is_the_legacy_mersenne_twister(self) -> None:
        # numpy's ``RandomState(0).random_sample()`` and ``RandomState(123456789).random_sample(2)``.
        assert _Stream(0)._double() == 0.5488135039273248
        stream = _Stream(123456789)
        assert [stream._double() for _ in range(2)] == [0.532833024789759, 0.5341366008904166]

    def test_the_seeds_are_distinct_positions_of_the_population(self) -> None:
        for population, count in ((10, 3), (50, 9), (7, 7)):
            seeds = _Stream(123456789).distinct(population, count)
            assert len(set(seeds)) == count and all(0 <= s < population for s in seeds)

    def test_the_seeds_are_what_the_reference_draws(self) -> None:
        # ``RandomState(123456789).choice(n, k, replace=False, p=uniform)`` as numpy draws it; the last
        # asks for every position of a small population, so rounds that land twice are drawn again.
        assert _Stream(123456789).distinct(12, 3) == [6, 8, 2]
        assert _Stream(123456789).distinct(100, 5) == [53, 50, 71, 25, 76]
        assert _Stream(123456789).distinct(7, 7) == [3, 4, 1, 5, 6, 0, 2]

    def test_the_labels_are_lloyds_algorithm_from_those_seeds(self) -> None:
        lucid.manual_seed(4)
        for rows, columns, clusters in ((30, 3, 3), (40, 5, 4), (25, 2, 2), (60, 6, 5)):
            points = lucid.randn(rows, columns)
            data = points.tolist()
            variance = float(points.var(dim=0, unbiased=False).mean().item())
            seeds = _Stream(123456789).distinct(rows, clusters)
            expected = forms.lloyd(data, seeds, 10, 1e-4 * variance)
            assert _kmeans_labels(points, clusters).tolist() == expected

    def test_two_distant_blobs_are_found(self) -> None:
        lucid.manual_seed(5)
        points = lucid.cat([lucid.randn(8, 3), lucid.randn(8, 3) + 50.0])
        labels = _kmeans_labels(points, 2).tolist()
        assert len(set(labels[:8])) == 1 and len(set(labels[8:])) == 1 and labels[0] != labels[8]

    def test_every_row_is_nearest_to_its_cluster_at_convergence(self) -> None:
        lucid.manual_seed(6)
        centers = lucid.tensor([[0.0, 0.0], [20.0, 0.0], [0.0, 20.0]])
        points = lucid.cat([lucid.randn(10, 2) + c for c in centers])
        labels = _kmeans_labels(points, 3)
        means = lucid.stack([points[labels == k].mean(dim=0) for k in range(3)])
        distances = ((points.unsqueeze(1) - means.unsqueeze(0)) ** 2).sum(dim=2)
        assert distances.argmin(dim=1).tolist() == labels.tolist()

    def test_one_cluster_labels_every_row_the_same(self) -> None:
        points = lucid.randn(9, 3)
        assert set(_kmeans_labels(points, 1).tolist()) == {0}

    def test_a_cluster_that_loses_its_rows_is_given_the_farthest_row(self) -> None:
        # Both seeds fall on equal rows, so a cluster is empty after the first round and takes the row
        # farthest from its center.  The labels are scikit-learn's for the same rows.
        cases = (
            ([[3.0], [3.0], [3.0], [0.0]], [0, 0, 0, 1]),
            ([[0.0], [1.0], [1.0], [1.0]], [1, 0, 0, 0]),
        )
        for rows, expected in cases:
            assert _kmeans_labels(lucid.tensor(rows), 2).tolist() == expected


class TestStarts:
    def test_the_default_start_is_the_base_bit_for_bit(self, device: str) -> None:
        layer, base = _layer()
        layer.to(device)
        x = lucid.randn(5, 4, device=device)
        assert same(layer(x), base(x))

    def test_every_update_factor_starts_at_zero(self) -> None:
        layer, _ = _layer()
        assert all(not bool(b.any().item()) for b in _parameters(layer.adamss_B, "default"))
        assert not bool(layer.get_delta_weight("default").any().item())
        assert not bool(layer.get_delta_bias("default").any().item())

    def test_no_initialisation_starts_off_the_base(self) -> None:
        layer, base = _layer(init_weights=None)
        x = lucid.randn(5, 4)
        assert not bool(same(layer(x), base(x)))
        updates = lucid.cat([b.reshape(-1) for b in _parameters(layer.adamss_B, "default")])
        assert 0.003 < float(updates.std().item()) < 0.03

    def test_no_initialisation_is_drawn_from_the_seed(self) -> None:
        first, _ = _layer(init_weights=None, random_seed=4)
        second, _ = _layer(init_weights=None, random_seed=4)
        other, _ = _layer(init_weights=None, random_seed=5)
        assert same(first.adamss_B["default"][0], second.adamss_B["default"][0])
        assert not bool(same(first.adamss_B["default"][0], other.adamss_B["default"][0]))

    def test_a_start_at_a_trainable_subspace_gets_a_gradient_after_the_first_step(self) -> None:
        layer, _ = _layer()
        layer(lucid.randn(5, 4)).sum().backward()
        # B = 0 hides A from the loss; B itself gets the gradient.
        for a, b in zip(
            _parameters(layer.adamss_A, "default"),
            _parameters(layer.adamss_B, "default"),
            strict=True,
        ):
            assert a.grad is not None and not bool(a.grad.any().item())
            assert b.grad is not None and bool(b.grad.any().item())


class TestUpdate:
    def test_the_layer_is_the_papers_sum(self, device: str) -> None:
        layer, base = _layer()
        _train(layer)
        layer.to(device)
        x = lucid.randn(5, 4, device=device)
        expected = forms.layer(
            x.tolist(), base.weight.tolist(), base.bias.tolist(), _expected_delta(layer)
        )
        with lucid.no_grad():
            got = layer(x)
        assert span(got, lucid.tensor(expected, device=device)) <= _TOLERANCE * magnitude(got)

    def test_a_base_without_a_bias_has_no_bias_column(self) -> None:
        layer, base = _layer(bias=False)
        _train(layer)
        assert tuple(layer.adamss_newB["default"].shape) == (4, 4)
        x = lucid.randn(5, 4)
        expected = forms.layer(x.tolist(), base.weight.tolist(), None, _expected_delta(layer))
        with lucid.no_grad():
            assert span(layer(x), lucid.tensor(expected)) <= _TOLERANCE * magnitude(layer(x))
            assert not bool(layer.get_delta_bias("default").any().item())
            assert tuple(layer.get_delta_bias("default").shape) == (12,)

    def test_a_sequence_batch_broadcasts(self) -> None:
        layer, base = _layer()
        _train(layer)
        x = lucid.randn(2, 3, 4)
        with lucid.no_grad():
            flat = layer(x.reshape(6, 4)).reshape(2, 3, 12)
            assert span(layer(x), flat) <= _TOLERANCE * magnitude(flat)

    def test_the_change_of_the_weight_and_the_bias_are_the_columns_of_the_joined_update(
        self,
    ) -> None:
        layer, _ = _layer()
        _train(layer)
        delta = lucid.tensor(_expected_delta(layer))
        with lucid.no_grad():
            assert span(layer.get_delta_weight("default"), delta[:, :4]) <= 1e-5
            assert span(layer.get_delta_bias("default"), delta[:, 4]) <= 1e-5

    def test_one_subspace_is_a_single_low_rank_update(self) -> None:
        layer, base = _layer(num_subspaces=1, subspace_rank=3)
        _train(layer)
        a = _parameters(layer.adamss_A, "default")[0]
        b = _parameters(layer.adamss_B, "default")[0]
        v = layer.adamss_newB["default"]
        with lucid.no_grad():
            joined = b @ a @ v
            assert span(layer.get_delta_weight("default"), joined[:, :4]) <= 1e-5

    def test_only_the_factors_train_and_get_a_gradient(self) -> None:
        net = nn.Sequential(nn.Linear(4, 12), nn.ReLU(), nn.Linear(12, 6))
        peft = get_peft_model(net, AdamssConfig(r=4, num_subspaces=3, target_modules=["0", "2"]))
        for _, layer in _tuner_layers(peft):
            _train(layer)
        peft(lucid.randn(5, 4)).sum().backward()
        for name, p in peft.named_parameters():
            if p.requires_grad:
                assert ("adamss_A" in name or "adamss_B" in name) and p.grad is not None, name
            else:
                assert p.grad is None, name
        buffers = dict(peft.named_buffers())
        assert not any(v.requires_grad for v in buffers.values())

    def test_two_adapters_add(self) -> None:
        net = nn.Sequential(nn.Linear(4, 12))
        config = AdamssConfig(r=4, num_subspaces=3, target_modules=["0"], init_weights=None)
        peft = get_peft_model(net, config)
        peft.add_adapter("b", config.__class__.from_dict({**config.to_dict(), "random_seed": 9}))
        layer = peft.base_model.model[0]
        peft.set_adapter(["default", "b"])
        x = lucid.randn(5, 4)
        with lucid.no_grad():
            both = (
                layer.get_base_layer()(x)
                + x @ layer.get_delta_weight("default").mT
                + layer.get_delta_bias("default")
                + x @ layer.get_delta_weight("b").mT
                + layer.get_delta_bias("b")
            )
            assert span(peft(x), both) <= _TOLERANCE * magnitude(both) * 4


class TestMerging:
    def test_merging_adds_the_weight_and_bias_changes(self) -> None:
        layer, base = _layer()
        _train(layer)
        weight, bias = base.weight.detach().clone(), base.bias.detach().clone()
        dw, db = layer.get_delta_weight("default"), layer.get_delta_bias("default")
        layer.merge()
        assert span(base.weight, weight + dw) <= 1e-6 and span(base.bias, bias + db) <= 1e-6

    def test_unmerging_restores_the_weight_and_bias_bit_for_bit(self) -> None:
        layer, base = _layer()
        _train(layer)
        weight, bias = base.weight.detach().clone(), base.bias.detach().clone()
        layer.merge()
        layer.unmerge()
        assert same(base.weight, weight) and same(base.bias, bias)

    def test_a_merged_layer_computes_what_the_unmerged_one_does(self, device: str) -> None:
        layer, _ = _layer()
        _train(layer)
        layer.to(device)
        x = lucid.randn(5, 4, device=device)
        with lucid.no_grad():
            before = layer(x)
            layer.merge()
            assert span(layer(x), before) <= _TOLERANCE * magnitude(before)

    def test_a_base_without_a_bias_merges_the_weight_alone(self) -> None:
        layer, base = _layer(bias=False)
        _train(layer)
        weight = base.weight.detach().clone()
        dw = layer.get_delta_weight("default")
        layer.merge()
        assert base.bias is None and span(base.weight, weight + dw) <= 1e-6
        layer.unmerge()
        assert same(base.weight, weight)


class TestDtypes:
    @pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16])
    def test_a_half_precision_base_gets_a_float32_adapter(
        self, dtype: lucid.dtype, device: str
    ) -> None:
        lucid.manual_seed(1)
        base = nn.Linear(4, 12).to(device).to(dtype)
        layer = Linear(base, "default", AdamssConfig(r=4, num_subspaces=3, target_modules=["x"]))
        assert all(p.dtype == lucid.float32 for p in layer._adapter_parameters("default"))
        assert layer.adamss_newB["default"].dtype == lucid.float32
        _train(layer)
        x = lucid.randn(5, 4, device=device).to(dtype)
        assert layer(x).dtype == dtype

    def test_a_float64_base_keeps_a_float64_adapter(self) -> None:
        base = nn.Linear(4, 12).to(lucid.float64)
        layer = Linear(base, "default", AdamssConfig(r=4, num_subspaces=3, target_modules=["x"]))
        assert all(p.dtype == lucid.float64 for p in layer._adapter_parameters("default"))
        assert layer.adamss_newB["default"].dtype == lucid.float64
        assert layer.adamss_rows["default"].dtype == lucid.int64


class TestFile:
    def _peft(self, **kw: object) -> PeftModel:
        lucid.manual_seed(0)
        net = nn.Sequential(nn.Linear(4, 12), nn.ReLU(), nn.Linear(12, 6))
        fields: dict[str, object] = {
            "r": 4,
            "num_subspaces": 3,
            "subspace_rank": 2,
            "target_modules": ["0", "2"],
        } | kw
        return get_peft_model(net, AdamssConfig(**fields))  # type: ignore[arg-type]

    def test_the_keys_are_the_reference_s(self) -> None:
        peft = self._peft()
        keys = sorted(get_peft_model_state_dict(peft))
        expected = sorted(
            f"base_model.model.{i}.{t}"
            for i in (0, 2)
            for t in ("adamss_newB", *(f"adamss_{n}.{k}" for n in "AB" for k in range(3)))
        )
        assert keys == expected

    def test_the_row_order_is_not_saved(self) -> None:
        peft = self._peft()
        assert not any("rows" in key for key in get_peft_model_state_dict(peft))
        assert not any("rows" in key for key in peft.state_dict())

    def test_a_saved_adapter_loads_into_a_fresh_model_and_computes_the_same(
        self, tmp_path: Path
    ) -> None:
        peft = self._peft()
        for _, layer in _tuner_layers(peft):
            _train(layer)
        peft.save_pretrained(tmp_path)
        lucid.manual_seed(0)
        fresh = nn.Sequential(nn.Linear(4, 12), nn.ReLU(), nn.Linear(12, 6))
        loaded = PeftModel.from_pretrained(fresh, tmp_path)
        x = lucid.randn(5, 4)
        with lucid.no_grad():
            assert span(loaded(x), peft(x)) <= _TOLERANCE * magnitude(peft(x))
        assert [_members(l) for _, l in _tuner_layers(loaded)] == [
            _members(l) for _, l in _tuner_layers(peft)
        ]

    def test_a_file_that_does_not_fit_the_subspaces_is_refused_naming_the_layer(self) -> None:
        peft = self._peft()
        state = get_peft_model_state_dict(peft)
        state["base_model.model.0.adamss_B.0"] = lucid.zeros(99, 2)
        with pytest.raises(ValueError, match="'0'.*do not fit its subspaces"):
            set_peft_model_state_dict(peft, state)

    def test_deleting_an_adapter_takes_its_buffers_with_it(self) -> None:
        peft = self._peft()
        peft.add_adapter("b", AdamssConfig(r=4, num_subspaces=3, target_modules=["0", "2"]))
        peft.delete_adapter("b")
        layer = peft.base_model.model[0]
        assert "b" not in layer.adamss_newB and "b" not in layer.adamss_rows
        assert "b" not in layer.adamss_A and "b" not in layer._specs


class TestAllocation:
    """``update_and_allocate``: the schedule, the importances, the masks."""

    @staticmethod
    def _model(**kw: object) -> tuple[AdamssModel, nn.Sequential]:
        lucid.manual_seed(0)
        net = nn.Sequential(nn.Linear(4, 12), nn.ReLU(), nn.Linear(12, 6))
        fields: dict[str, object] = {
            "r": 4,
            "num_subspaces": 3,
            "use_asa": True,
            "asa_target_subspaces": 2,
            "init_warmup": 1,
            "final_warmup": 5,
            "mask_interval": 1,
            "target_modules": ["0", "2"],
        } | kw
        return AdamssModel(net, AdamssConfig(**fields), "default"), net  # type: ignore[arg-type]

    @staticmethod
    def _give_gradients(net: nn.Sequential, seed: int) -> None:
        """A gradient of its own for every factor that trains (a frozen one has none)."""
        lucid.manual_seed(seed)
        for _, layer in _tuner_layers(net):
            for p in layer._adapter_parameters("default"):
                if p.requires_grad:
                    p.grad = lucid.randn(*p.shape)

    @staticmethod
    def _trainable(net: nn.Sequential) -> list[bool]:
        return [
            p.requires_grad
            for _, layer in _tuner_layers(net)
            for p in layer._adapter_parameters("default")
        ]

    def test_the_schedule_is_the_closed_form(self) -> None:
        model, net = self._model()
        allocator = model.allocator("default")
        assert allocator.total == 6
        for step in range(0, 9):
            expected = None if step < 1 else forms.target(allocator.total, 2, 1, 5, 3.0, step)
            assert allocator.target(step) == expected

    @pytest.mark.parametrize("exponent", [1.0, 2.0, 3.0, 4.5])
    def test_the_exponent_shapes_the_fall(self, exponent: float) -> None:
        model, _ = self._model(asa_schedule_exponent=exponent, init_warmup=2, final_warmup=12)
        allocator = model.allocator("default")
        for step in range(2, 14):
            assert allocator.target(step) == forms.target(6, 2, 2, 12, exponent, step)

    def test_the_budget_starts_at_the_total_and_ends_at_the_target(self) -> None:
        model, _ = self._model(init_warmup=2, final_warmup=10)
        allocator = model.allocator("default")
        assert allocator.target(2) == allocator.total
        assert allocator.target(10) == 2
        assert allocator.target(100) == 2
        values = [allocator.target(step) for step in range(2, 11)]
        assert values == sorted(values, reverse=True)

    def test_the_importance_is_the_two_averages_of_the_papers_score(self) -> None:
        model, net = self._model(asa_importance_beta=0.5, asa_uncertainty_beta=0.25)
        allocator = model.allocator("default")
        layer = net[0]
        p = _parameters(layer.adamss_B, "default")[0]
        with lucid.no_grad():
            p.copy_(lucid.full(tuple(p.shape), 2.0))
        steps = [1.5, -0.5]
        mean = spread = 0.0
        for grad in steps:
            p.grad = lucid.full(tuple(p.shape), grad)
            sensitivity = abs(2.0 * grad)
            # the uncertainty reads the average as it stood before this step's sensitivity entered it
            spread = 0.25 * spread + 0.75 * abs(sensitivity - mean)
            mean = 0.5 * mean + 0.5 * sensitivity
            allocator.update_importance()
            state = allocator._sensitivity[("0", 0, "B")]
            assert span(state.score, lucid.full(tuple(p.shape), mean * spread)) <= 1e-6

    def test_a_factor_without_a_gradient_is_not_recorded(self) -> None:
        model, net = self._model()
        allocator = model.allocator("default")
        allocator.update_importance()
        assert allocator._sensitivity == {}

    def test_the_mask_keeps_the_subspaces_strictly_above_the_target_th_highest(self) -> None:
        model, net = self._model()
        allocator = model.allocator("default")
        spaces = allocator._subspaces()
        # one score per subspace: give each factor a gradient so that the subspace's score is its index
        lucid.manual_seed(0)
        with lucid.no_grad():
            for space in spaces:
                space.left.copy_(lucid.ones(*space.left.shape))
                space.right.copy_(lucid.ones(*space.right.shape))
        for rank, space in enumerate(spaces):
            space.left.grad = lucid.full(tuple(space.left.shape), float(rank + 1))
            space.right.grad = lucid.full(tuple(space.right.shape), float(rank + 1))
        allocator.update_importance()
        allocator.mask_to_target(3)
        kept = [space.left.requires_grad and space.right.requires_grad for space in spaces]
        # six subspaces scored 1..6 apart: the target is 3, the third highest is the score of the
        # fourth subspace from the top, and only those strictly above it keep training
        assert kept.count(True) == 3 - 1
        assert kept == [False, False, False, False, True, True]

    def test_a_target_at_least_the_number_recorded_freezes_nothing(self) -> None:
        model, net = self._model()
        allocator = model.allocator("default")
        self._give_gradients(net, 1)
        allocator.update_importance()
        allocator.mask_to_target(6)
        assert all(self._trainable(net))
        allocator.mask_to_target(100)
        assert all(self._trainable(net))

    def test_a_frozen_subspace_keeps_its_value_and_its_part_of_the_output(self) -> None:
        model, net = self._model()
        allocator = model.allocator("default")
        for _, layer in _tuner_layers(net):
            _train(layer)
        x = lucid.randn(5, 4)
        with lucid.no_grad():
            before = net(x)
        self._give_gradients(net, 2)
        allocator.update_importance()
        allocator.mask_to_target(2)
        assert not all(self._trainable(net))
        with lucid.no_grad():
            assert same(net(x), before)
        frozen = [
            p
            for _, layer in _tuner_layers(net)
            for p in layer._adapter_parameters("default")
            if not p.requires_grad
        ]
        assert frozen and all(p.grad is None for p in frozen)

    def test_a_frozen_subspace_is_not_ranked_again(self) -> None:
        model, net = self._model()
        allocator = model.allocator("default")
        self._give_gradients(net, 3)
        allocator.update_importance()
        allocator.mask_to_target(2)
        frozen = self._trainable(net)
        allocator.reset_importance()
        self._give_gradients(net, 4)
        allocator.update_importance()
        assert len(allocator._scores()) == sum(frozen) // 2
        allocator.mask_to_target(100)
        assert self._trainable(net) == frozen

    def test_nothing_happens_outside_the_window(self) -> None:
        model, net = self._model(init_warmup=3, final_warmup=6)
        self._give_gradients(net, 5)
        for step in (0, 2, 7, 50):
            model.update_and_allocate(step)
        assert all(self._trainable(net))
        assert model.allocator("default")._sensitivity == {}

    def test_importances_are_gathered_every_step_and_the_masks_come_on_the_interval(self) -> None:
        model, net = self._model(init_warmup=2, final_warmup=9, mask_interval=3)
        allocator = model.allocator("default")
        counts = []
        for step in range(0, 11):
            self._give_gradients(net, 10 + step)
            model.update_and_allocate(step)
            counts.append((step, sum(self._trainable(net)), len(allocator._sensitivity)))
        by_step = dict((s, (t, r)) for s, t, r in counts)
        assert by_step[0] == (12, 0) and by_step[1] == (12, 0)  # before the window: nothing
        assert by_step[2][1] > 0 and by_step[2][0] == 12  # recording, not masking
        assert by_step[3] == (by_step[3][0], 0)  # a mask step restarts the importances
        assert by_step[3][0] < 12
        assert by_step[4][1] > 0 and by_step[4][0] == by_step[3][0]  # between masks: the same set

    def test_a_training_run_ends_at_the_target_or_one_below_it(self) -> None:
        # A mask that cuts to the target keeps one fewer (the reference's strict comparison); one that
        # finds no more subspaces than the target leaves them all.  Which one ends a run depends on the
        # steps before it: here the cut to 4 left 3.
        model, net = self._model(
            init_warmup=1, final_warmup=6, mask_interval=1, asa_target_subspaces=3
        )
        counts = []
        for step in range(0, 9):
            self._give_gradients(net, 20 + step)
            model.update_and_allocate(step)
            spaces = model.allocator("default")._subspaces()
            counts.append(sum(1 for s in spaces if s.left.requires_grad and s.right.requires_grad))
        assert counts == sorted(counts, reverse=True)
        assert counts[-1] in (2, 3)

    def test_a_training_loop_learns_while_the_allocation_freezes_subspaces(
        self, device: str
    ) -> None:
        lucid.manual_seed(0)
        net = nn.Sequential(nn.Linear(6, 24), nn.ReLU(), nn.Linear(24, 3)).to(device)
        config = AdamssConfig(
            r=6,
            num_subspaces=4,
            subspace_rank=2,
            target_modules=["0", "2"],
            use_asa=True,
            asa_target_subspaces=3,
            init_warmup=2,
            final_warmup=20,
            mask_interval=4,
        )
        peft = get_peft_model(net, config)
        optimizer = optim.AdamW([p for p in peft.parameters() if p.requires_grad], lr=1e-2)
        x, target = lucid.randn(32, 6, device=device), lucid.randn(32, 3, device=device)
        losses = []
        for step in range(30):
            loss = ((peft(x) - target) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            peft.base_model.update_and_allocate(step)
            optimizer.step()
            losses.append(float(loss.item()))
        spaces = peft.base_model.allocator("default")._subspaces()
        assert losses[-1] < losses[0]
        assert 0 < sum(1 for s in spaces if s.left.requires_grad) < len(spaces)

    def test_an_adapter_without_asa_is_left_alone(self) -> None:
        model, net = self._model(use_asa=False)
        self._give_gradients(net, 6)
        for step in range(0, 8):
            model.update_and_allocate(step)
        assert all(self._trainable(net))

    def test_an_unknown_adapter_has_no_allocator(self) -> None:
        model, _ = self._model()
        with pytest.raises(ValueError, match="no adapter named 'other'"):
            model.allocator("other")

    def test_the_allocator_of_a_deleted_adapter_is_forgotten(self) -> None:
        model, net = self._model()
        model.add_adapter(
            "b", AdamssConfig(r=4, num_subspaces=3, use_asa=True, target_modules=["0", "2"])
        )
        first = model.allocator("b")
        model.delete_adapter("b")
        model.add_adapter(
            "b", AdamssConfig(r=4, num_subspaces=3, use_asa=True, target_modules=["0", "2"])
        )
        assert model.allocator("b") is not first

    def test_an_allocator_needs_a_layer_that_holds_the_adapter(self) -> None:
        net = nn.Sequential(nn.Linear(4, 4))
        with pytest.raises(ValueError, match="no AdaMSS layer"):
            SubspaceAllocator(net, AdamssConfig(use_asa=True, target_modules=["0"]), "default")


class TestTuner:
    def test_it_is_registered_under_its_method(self) -> None:
        from lucid.peft import PEFT_TYPE_TO_TUNER_MAPPING

        assert PEFT_TYPE_TO_TUNER_MAPPING[PeftType.ADAMSS] is AdamssModel

    def test_the_constructor_takes_config_as_the_reference_does(self) -> None:
        net = nn.Sequential(nn.Linear(4, 12))
        model = AdamssModel(
            net, config=AdamssConfig(r=4, num_subspaces=2, target_modules=["0"]), adapter_name="a"
        )
        assert model.targeted_module_names == ["0"]

    def test_the_default_targets_are_the_reference_s(self) -> None:
        net = nn.Sequential(nn.Linear(4, 4))
        net.config = types.SimpleNamespace(model_type="gpt2")  # type: ignore[assignment]
        net.add_module("c_attn", nn.Linear(4, 4))
        model = AdamssModel(net, AdamssConfig(r=2, num_subspaces=2), "default")
        assert model.targeted_module_names == ["c_attn"]

    def test_a_module_that_is_not_linear_is_refused(self) -> None:
        net = nn.Sequential(nn.Conv1d(2, 3, 3))
        with pytest.raises(ValueError):
            AdamssModel(net, AdamssConfig(target_modules=["0"]), "default")

    def test_the_layer_is_built_on_a_linear_only(self) -> None:
        with pytest.raises(TypeError, match="adapts nn.Linear"):
            Linear(nn.Conv1d(2, 3, 3), "default", AdamssConfig(target_modules=["x"]))  # type: ignore[arg-type]

    def test_the_layer_reprs_with_the_methods_prefix(self) -> None:
        layer, _ = _layer()
        assert repr(layer).startswith("adamss.")

    def test_it_unloads_to_the_original_module(self) -> None:
        net = nn.Sequential(nn.Linear(4, 12))
        model = AdamssModel(
            net, AdamssConfig(r=4, num_subspaces=2, target_modules=["0"]), "default"
        )
        assert model.unload() is net and type(net[0]) is nn.Linear


# ── the reference library ────────────────────────────────────────────────────

#: ``(rows, columns, subspaces, subspace rank, dynamic rank)``: weights taller than wide, so the split
#: has structure (the rows of a square left factor are all the same distance apart).
_CASES = [
    (12, 4, 3, 2, False),
    (20, 5, 4, 1, False),
    (30, 6, 5, 3, False),
    (16, 3, 3, 4, True),
    (24, 5, 2, 2, True),
]


def _library() -> ModuleType:
    library = adapter_library()
    pytest.importorskip("sklearn", reason="the reference's clustering needs scikit-learn")
    return library


def _pair(
    ref: ModuleType,
    library: ModuleType,
    rows: int,
    cols: int,
    clusters: int,
    rank: int,
    dynamic: bool,
    bias: bool = True,
    **extra: object,
) -> tuple[PeftModel, ModuleType]:
    """The same one-layer model adapted here and in the reference library (``init_weights=None``)."""
    lucid.manual_seed(rows * 31 + cols)
    mine = nn.Sequential(nn.Linear(cols, rows, bias=bias))
    theirs = ref.nn.Sequential(ref.nn.Linear(cols, rows, bias=bias))
    with ref.no_grad():
        theirs[0].weight.copy_(across(ref, mine[0].weight))
        if bias:
            theirs[0].bias.copy_(across(ref, mine[0].bias))
    fields: dict[str, object] = {
        "r": 100,
        "num_subspaces": clusters,
        "subspace_rank": rank,
        "use_dynamic_rank": dynamic,
        "target_modules": ["0"],
        "init_weights": None,
    } | extra
    ref.manual_seed(0)
    return (
        get_peft_model(mine, AdamssConfig(**fields)),  # type: ignore[arg-type]
        library.get_peft_model(theirs, library.AdamssConfig(**fields)),
    )


def _carry(ref: ModuleType, library: ModuleType, theirs: ModuleType, mine: PeftModel) -> None:
    """Load the reference's adapter tensors into ``mine``."""
    state = library.get_peft_model_state_dict(theirs)
    set_peft_model_state_dict(
        mine, {k: lucid.tensor(v.detach().tolist()).to(lucid.float32) for k, v in state.items()}
    )


@pytest.mark.parity
class TestParity:
    @pytest.mark.parametrize("case", _CASES, ids=lambda c: "x".join(map(str, c)))
    def test_the_keys_shapes_and_split_are_the_reference_s(
        self, ref: ModuleType, case: tuple[int, int, int, int, bool]
    ) -> None:
        library = _library()
        mine, theirs = _pair(ref, library, *case)
        want = library.get_peft_model_state_dict(theirs)
        got = get_peft_model_state_dict(mine)
        assert {k: tuple(v.shape) for k, v in got.items()} == {
            k: tuple(v.shape) for k, v in want.items()
        }
        layer = mine.base_model.model[0]
        reference = theirs.base_model.model[0]
        assert (
            layer.scatter_index["default"].tolist() == reference.scatter_index["default"].tolist()
        )

    @pytest.mark.parametrize("case", _CASES, ids=lambda c: "x".join(map(str, c)))
    def test_a_reference_state_computes_what_the_reference_does(
        self, ref: ModuleType, case: tuple[int, int, int, int, bool]
    ) -> None:
        library = _library()
        mine, theirs = _pair(ref, library, *case)
        _carry(ref, library, theirs, mine)
        layer, reference = mine.base_model.model[0], theirs.base_model.model[0]
        x = lucid.randn(7, case[1])
        batch = lucid.randn(2, 3, case[1])
        with lucid.no_grad(), ref.no_grad():
            for sample in (x, batch):
                assert gap(ref, mine(sample), theirs(across(ref, sample))) <= 2e-5
            assert (
                gap(ref, layer.get_delta_weight("default"), reference.get_delta_weight("default"))
                <= 2e-5
            )
            assert (
                gap(ref, layer.get_delta_bias("default"), reference.get_delta_bias("default"))
                <= 2e-5
            )

    @pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
    def test_a_merged_layer_is_the_reference_s(self, ref: ModuleType, bias: bool) -> None:
        library = _library()
        mine, theirs = _pair(ref, library, 14, 4, 3, 2, False, bias=bias)
        _carry(ref, library, theirs, mine)
        mine.base_model.merge_adapter()
        theirs.base_model.merge_adapter()
        layer, reference = mine.base_model.model[0], theirs.base_model.model[0]
        assert gap(ref, layer.base_layer.weight, reference.base_layer.weight) <= 2e-5
        if bias:
            assert gap(ref, layer.base_layer.bias, reference.base_layer.bias) <= 2e-5

    def test_a_file_written_by_the_reference_loads(self, ref: ModuleType, tmp_path: Path) -> None:
        library = _library()
        mine, theirs = _pair(ref, library, 14, 4, 3, 2, False)
        theirs.peft_config["default"].save_pretrained(tmp_path)
        state = library.get_peft_model_state_dict(theirs)
        save_safetensors(
            {k: lucid.tensor(v.detach().tolist()).to(lucid.float32) for k, v in state.items()},
            str(tmp_path / "adapter_model.safetensors"),
            metadata={"format": "pt"},
        )
        lucid.manual_seed(14 * 31 + 4)
        fresh = nn.Sequential(nn.Linear(4, 14))
        with lucid.no_grad():
            fresh[0].weight.copy_(mine.base_model.model[0].base_layer.weight)
            fresh[0].bias.copy_(mine.base_model.model[0].base_layer.bias)
        loaded = PeftModel.from_pretrained(fresh, tmp_path)
        x = lucid.randn(6, 4)
        with lucid.no_grad(), ref.no_grad():
            assert gap(ref, loaded(x), theirs(across(ref, x))) <= 2e-5

    def test_the_config_reads_and_writes_the_reference_s_file(
        self, ref: ModuleType, tmp_path: Path
    ) -> None:
        library = _library()
        theirs = library.AdamssConfig(r=7, num_subspaces=3, target_modules=["q", "v"], use_asa=True)
        theirs.save_pretrained(tmp_path)
        mine = AdamssConfig.from_pretrained(tmp_path)
        assert mine.r == 7 and mine.use_asa and mine.num_subspaces == 3
        assert {k: v for k, v in mine.to_dict().items() if k != "target_modules"} == {
            k: v for k, v in theirs.to_dict().items() if k != "target_modules"
        } | {"peft_version": mine.peft_version}

    @pytest.mark.parametrize("seed", range(6))
    def test_the_clustering_is_the_reference_s(self, ref: ModuleType, seed: int) -> None:
        clustering_Z = _library().tuners.adamss.utils.clustering_Z
        lucid.manual_seed(seed)
        rows, columns, clusters = 20 + 15 * seed, 3 + seed, 2 + seed % 5
        points = lucid.randn(rows, columns)
        theirs, _ = clustering_Z(across(ref, points), clusters, 10)
        assert _kmeans_labels(points, clusters).tolist() == theirs.tolist()

    def test_the_masks_follow_the_reference_step_by_step(self, ref: ModuleType) -> None:
        library = _library()
        fields: dict[str, object] = {
            "r": 100,
            "num_subspaces": 4,
            "use_asa": True,
            "asa_target_subspaces": 3,
            "init_warmup": 2,
            "final_warmup": 11,
            "mask_interval": 2,
            "asa_importance_beta": 0.7,
            "asa_uncertainty_beta": 0.6,
            "target_modules": ["0", "2"],
            "init_weights": None,
        }
        lucid.manual_seed(8)
        mine = nn.Sequential(nn.Linear(5, 24), nn.ReLU(), nn.Linear(24, 40))
        theirs = ref.nn.Sequential(ref.nn.Linear(5, 24), ref.nn.ReLU(), ref.nn.Linear(24, 40))
        with ref.no_grad():
            for index in (0, 2):
                theirs[index].weight.copy_(across(ref, mine[index].weight))
                theirs[index].bias.copy_(across(ref, mine[index].bias))
        mine_peft = get_peft_model(mine, AdamssConfig(**fields))  # type: ignore[arg-type]
        their_peft = library.get_peft_model(theirs, library.AdamssConfig(**fields))
        ours = [m for _, m in _tuner_layers(mine_peft)]
        others = [their_peft.base_model.model[index] for index in (0, 2)]

        def pairs() -> list[tuple[lucid.Tensor, ModuleType]]:
            return [
                (p, getattr(theirs_layer, name)["default"][index])
                for layer, theirs_layer in zip(ours, others, strict=True)
                for name in ("adamss_A", "adamss_B")
                for index, p in enumerate(_parameters(getattr(layer, name), "default"))
            ]

        for step in range(0, 14):
            lucid.manual_seed(100 + step)
            for p, q in pairs():
                assert p.requires_grad == q.requires_grad
                if p.requires_grad:
                    grad = lucid.randn(*p.shape)
                    p.grad = grad
                    q.grad = across(ref, grad)
            mine_peft.base_model.update_and_allocate(step)
            their_peft.base_model.update_and_allocate(step)
            flags = [(p.requires_grad, q.requires_grad) for p, q in pairs()]
            assert all(a == b for a, b in flags), (step, flags)
        assert not all(a for a, _ in flags)  # it did freeze something
