"""TIES, DARE and magnitude-prune combinations of ``add_weighted_adapter``, against explicit formulas.

The merge pieces (a pruner, a sign election, the disjoint merge) are checked
on hand-sized tensors with the answer written out; the eight combination
types are checked on a layer against the same formulas applied to the
sources' matrices by plain Python loops — never against the strategy's own
arithmetic.  DARE's random drop is checked structurally (each element is
kept at ``x / density`` or dropped, the share kept is ``density``) and for
determinism under ``lucid.manual_seed``.
"""

import math
import warnings

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import LoraConfig, LoraModel
from lucid.peft.tuners.lora import Linear
from lucid.peft.tuners.lora._merging import _Combination
from lucid.peft.tuners.lora._task_vectors import (
    _FrequencyElection,
    _MagnitudePruner,
    _Pruner,
    _RandomPruner,
    _Recipe,
    _SignElection,
    _TaskMerge,
    _TotalElection,
)

FACTOR_TYPES = ("linear", "ties", "dare_linear", "dare_ties", "magnitude_prune")
SVD_TYPES = ("svd", "ties_svd", "dare_linear_svd", "dare_ties_svd", "magnitude_prune_svd")
ALL_TYPES = FACTOR_TYPES + SVD_TYPES + ("cat",)


def _close(a: lucid.Tensor, b: lucid.Tensor, atol: float = 1e-5) -> bool:
    return bool(lucid.allclose(a, b, atol=atol))


def _t(rows: list[list[float]], device: str) -> lucid.Tensor:
    return lucid.tensor(rows, device=device)


# ── the pieces ───────────────────────────────────────────────────────────────


class TestPruners:
    def test_magnitude_keeps_the_largest_by_absolute_value(self, device: str) -> None:
        x = _t([[0.5, -3.0, 2.0], [1.5, -0.25, 0.1]], device)
        kept = _MagnitudePruner(0.5).prune(x)  # 3 of 6
        assert kept.tolist() == [[0.0, -3.0, 2.0], [1.5, 0.0, 0.0]]

    def test_magnitude_with_nothing_to_keep_is_zero(self, device: str) -> None:
        x = _t([[1.0, -2.0]], device)
        assert _MagnitudePruner(0.0).prune(x).tolist() == [[0.0, 0.0]]
        assert _MagnitudePruner(0.4).prune(x).tolist() == [[0.0, 0.0]]  # int(0.8) == 0

    def test_random_drops_or_rescales_each_element(self, device: str) -> None:
        x = lucid.full((200, 50), 2.0, device=device)
        lucid.manual_seed(7)
        kept = _RandomPruner(0.25).prune(x)
        values = set(kept.flatten().tolist())
        assert values == {0.0, 8.0}  # dropped, or 2 / 0.25
        share = float((kept != 0).to(lucid.float32).mean().item())
        assert abs(share - 0.25) < 0.03

    def test_random_repeats_under_the_same_seed(self, device: str) -> None:
        x = lucid.randn(30, 30, device=device)
        lucid.manual_seed(3)
        first = _RandomPruner(0.5).prune(x)
        lucid.manual_seed(3)
        again = _RandomPruner(0.5).prune(x)
        lucid.manual_seed(4)
        other = _RandomPruner(0.5).prune(x)
        assert first.tolist() == again.tolist() and first.tolist() != other.tolist()

    def test_a_density_of_one_or_more_prunes_nothing_and_warns(self, device: str) -> None:
        x = lucid.randn(4, 4, device=device)
        for name in ("magnitude", "random"):
            with pytest.warns(UserWarning, match="no pruning will be performed"):
                assert _Pruner.named(name, 1.0)(x) is x

    def test_density_bounds(self) -> None:
        with pytest.raises(ValueError, match="Density should be >= 0"):
            _Pruner.named("magnitude", -0.1)
        with pytest.raises(ValueError, match="> 0 for random pruning"):
            _Pruner.named("random", 0.0)
        with pytest.raises(ValueError, match="Unknown pruning method 'sparse'"):
            _Pruner.named("sparse", 0.5)


class TestSignElection:
    # column 0: +3 -1 -1  (total > 0, most signs negative); column 1: -2 -2 +1 (both negative)
    STACK = [[[3.0, -2.0]], [[-1.0, -2.0]], [[-1.0, 1.0]]]

    def test_total_follows_the_sum(self, device: str) -> None:
        mask = _TotalElection().mask(lucid.tensor(self.STACK, device=device))
        assert mask.tolist() == [[[True, True]], [[False, True]], [[False, False]]]

    def test_frequency_follows_the_count(self, device: str) -> None:
        mask = _FrequencyElection().mask(lucid.tensor(self.STACK, device=device))
        assert mask.tolist() == [[[False, True]], [[True, True]], [[True, False]]]

    def test_a_tie_elects_positive_and_zeros_never_agree(self, device: str) -> None:
        stack = lucid.tensor([[[1.0, 0.0]], [[-1.0, 0.0]]], device=device)
        for election in (_TotalElection(), _FrequencyElection()):
            assert election.mask(stack).tolist() == [[[True, False]], [[False, False]]]

    def test_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="Unknown majority_sign_method"):
            _SignElection.named("median")


class TestTaskMerge:
    A = [[1.5, -4.0, 0.5], [2.0, 1.0, -0.25]]
    B = [[3.0, 2.0, -0.25], [-1.0, 1.2, 0.5]]

    def test_plain_is_the_weighted_sum(self, device: str) -> None:
        merge = _TaskMerge.make(_Recipe(), None, "total")
        out = merge([_t(self.A, device), _t(self.B, device)], [0.5, -2.0])
        expected = [[0.5 * a - 2.0 * b for a, b in zip(ra, rb)] for ra, rb in zip(self.A, self.B)]
        assert _close(out, _t(expected, device))

    def test_ties_trims_elects_and_averages_the_agreeing(self, device: str) -> None:
        merge = _TaskMerge.make(_Recipe("magnitude", True), 0.5, "total")
        out = merge([_t(self.A, device), _t(self.B, device)], [1.0, 2.0])
        # Trimmed to 3 of 6: A = [[1.5, -4, 0], [2, 0, 0]], B = [[3, 2, 0], [0, 1.2, 0]].
        # (0,0) both positive: (1.5 + 2*3) / 2; (0,1) the sum is negative, so only A's -4 agrees;
        # (1,0) only A has a value; (1,1) only B; the third column is all zeros.
        expected = [[3.75, -4.0, 0.0], [2.0, 2.4, 0.0]]
        assert _close(out, _t(expected, device))

    def test_the_election_ignores_the_weights(self, device: str) -> None:
        # 2 and -1 elect positive; the weights -1 and 1 would elect negative if they counted.
        merge = _TaskMerge.make(_Recipe(None, True), None, "total")
        out = merge([_t([[2.0]], device), _t([[-1.0]], device)], [-1.0, 1.0])
        assert _close(out, _t([[-2.0]], device))

    def test_density_is_required_by_a_pruning_recipe(self) -> None:
        with pytest.raises(ValueError, match="`density` must be given"):
            _TaskMerge.make(_Recipe("magnitude"), None, "total")


# ── the combination types on a layer ─────────────────────────────────────────


def _pair(device: str, r: int = 2) -> tuple[LoraModel, Linear]:
    lucid.manual_seed(1)
    net = nn.Sequential(nn.Linear(6, 5)).to(device)
    config = LoraConfig(r=r, lora_alpha=2 * r, target_modules=["0"], init_lora_weights=False)
    lora = LoraModel(net, config, adapter_name="a")
    lora.add_adapter("b", config)
    layer = net[0]
    assert isinstance(layer, Linear)
    return lora, layer


def _matrix(rows: lucid.Tensor) -> list[list[float]]:
    return [list(map(float, row)) for row in rows.tolist()]


def _signed_roots(scaled: list[float]) -> tuple[list[float], list[float]]:
    roots = [math.sqrt(abs(s)) for s in scaled]
    return [r if s >= 0 else -r for s, r in zip(scaled, roots)], roots


def _magnitude_trim(m: list[list[float]], density: float) -> list[list[float]]:
    flat = sorted((abs(v) for row in m for v in row), reverse=True)
    keep = int(density * len(flat))
    floor = flat[keep - 1] if keep else math.inf
    return [[v if abs(v) >= floor else 0.0 for v in row] for row in m]


def _reference(
    kind: str, tensors: list[list[list[float]]], weights: list[float], density: float, method: str
) -> list[list[float]]:
    """The merge written as loops: trim (magnitude), elect, weight, average the agreeing."""
    if kind in ("ties", "magnitude_prune"):
        tensors = [_magnitude_trim(t, density) for t in tensors]
    rows, cols = len(tensors[0]), len(tensors[0][0])
    out = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            values = [t[i][j] for t in tensors]
            weighted = [w * v for w, v in zip(weights, values)]
            if kind == "magnitude_prune":
                out[i][j] = sum(weighted)
                continue
            score = sum(values) if method == "total" else sum((v > 0) - (v < 0) for v in values)
            elected = 1 if score >= 0 else -1
            agreeing = [
                wv for v, wv in zip(values, weighted) if v != 0 and (v > 0) == (elected > 0)
            ]
            out[i][j] = sum(agreeing) / max(len(agreeing), 1)
    return out


class TestFactorTypes:
    @pytest.mark.parametrize("kind", ["ties", "magnitude_prune"])
    @pytest.mark.parametrize("method", ["total", "frequency"])
    def test_a_and_b_are_merged_apart_by_the_formula(
        self, device: str, kind: str, method: str
    ) -> None:
        lora, layer = _pair(device)
        weights = [0.7, -0.4]
        lora.add_weighted_adapter(
            ["a", "b"],
            weights,
            "mix",
            combination_type=kind,
            density=0.5,  # type: ignore[arg-type]
            majority_sign_method=method,  # type: ignore[arg-type]
        )
        scaled = [w * layer.scaling[n] for n, w in zip(["a", "b"], weights)]
        signed, roots = _signed_roots(scaled)
        a = [_matrix(layer.lora_A[n].weight) for n in ("a", "b")]
        b = [_matrix(layer.lora_B[n].weight) for n in ("a", "b")]
        expected_a = _reference(kind, a, signed, 0.5, method)
        expected_b = _reference(kind, b, roots, 0.5, method)
        assert _close(layer.lora_A["mix"].weight, _t(expected_a, device))
        assert _close(layer.lora_B["mix"].weight, _t(expected_b, device))
        assert layer.r["mix"] == 2 and layer.scaling["mix"] == 1.0

    @pytest.mark.parametrize("kind", ["dare_linear", "dare_ties"])
    def test_dare_repeats_under_the_same_seed_and_differs_across_seeds(
        self, device: str, kind: str
    ) -> None:
        lora, layer = _pair(device)
        for name, seed in (("one", 5), ("again", 5), ("other", 6)):
            lucid.manual_seed(seed)
            lora.add_weighted_adapter(
                ["a", "b"],
                [1.0, 1.0],
                name,
                combination_type=kind,
                density=0.5,  # type: ignore[arg-type]
            )
        weights = {n: layer.lora_A[n].weight.tolist() for n in ("one", "again", "other")}
        assert weights["one"] == weights["again"] and weights["one"] != weights["other"]

    def test_dare_linear_without_pruning_is_linear(self, device: str) -> None:
        lora, layer = _pair(device)
        lora.add_weighted_adapter(["a", "b"], [0.3, 0.9], "lin", combination_type="linear")
        with pytest.warns(UserWarning, match="no pruning will be performed"):
            lora.add_weighted_adapter(
                ["a", "b"], [0.3, 0.9], "dare", combination_type="dare_linear", density=1.0
            )
        assert _close(layer.lora_A["dare"].weight, layer.lora_A["lin"].weight)
        assert _close(layer.lora_B["dare"].weight, layer.lora_B["lin"].weight)

    def test_dare_drops_some_of_every_source(self, device: str) -> None:
        lora, layer = _pair(device)
        lucid.manual_seed(0)
        lora.add_weighted_adapter(
            ["a", "b"], [1.0, 1.0], "mix", combination_type="dare_linear", density=0.5
        )
        lora.add_weighted_adapter(["a", "b"], [1.0, 1.0], "lin", combination_type="linear")
        assert layer.lora_A["mix"].weight.tolist() != layer.lora_A["lin"].weight.tolist()

    def test_a_pruning_type_needs_a_density(self, device: str) -> None:
        lora, _ = _pair(device)
        for kind in ("ties", "dare_linear", "dare_ties", "magnitude_prune", "ties_svd"):
            with pytest.raises(ValueError, match="`density` must be given"):
                lora.add_weighted_adapter(["a", "b"], [1.0, 1.0], "mix", combination_type=kind)  # type: ignore[arg-type]
            assert "mix" not in lora.peft_config

    def test_the_factor_types_need_one_rank(self, device: str) -> None:
        lora, _ = _pair(device)
        lora.add_adapter("wide", LoraConfig(r=3, target_modules=["0"], init_lora_weights=False))
        for kind in ("ties", "dare_ties", "dare_linear", "magnitude_prune"):
            with pytest.raises(ValueError, match="same r value"):
                lora.add_weighted_adapter(
                    ["a", "wide"],
                    [1.0, 1.0],
                    "mix",
                    combination_type=kind,
                    density=0.5,  # type: ignore[arg-type]
                )
            assert "mix" not in lora.peft_config

    def test_a_bad_majority_sign_method_is_refused(self, device: str) -> None:
        lora, _ = _pair(device)
        with pytest.raises(ValueError, match="majority_sign_method"):
            lora.add_weighted_adapter(
                ["a", "b"],
                [1.0, 1.0],
                "mix",
                combination_type="ties",
                density=0.5,
                majority_sign_method="median",  # type: ignore[arg-type]
            )


class TestSvdTypes:
    @pytest.mark.parametrize("kind", ["ties_svd", "magnitude_prune_svd"])
    @pytest.mark.parametrize("method", ["total", "frequency"])
    def test_a_full_rank_result_is_the_merge_of_the_updates(
        self, device: str, kind: str, method: str
    ) -> None:
        lora, layer = _pair(device)
        weights = [0.8, -0.5]
        lora.add_weighted_adapter(
            ["a", "b"],
            weights,
            "mix",
            combination_type=kind,
            density=0.6,
            svd_rank=5,  # type: ignore[arg-type]
            majority_sign_method=method,  # type: ignore[arg-type]
        )
        deltas = []
        for name in ("a", "b"):
            a, b = layer.lora_A[name].weight, layer.lora_B[name].weight
            deltas.append(_matrix((b @ a) * layer.scaling[name]))
        expected = _reference(kind.removesuffix("_svd"), deltas, weights, 0.6, method)
        with lucid.no_grad():
            assert _close(layer.get_delta_weight("mix"), _t(expected, device), atol=1e-4)

    @pytest.mark.parametrize("kind", ["dare_linear_svd", "dare_ties_svd"])
    def test_dare_svd_repeats_under_the_same_seed(self, device: str, kind: str) -> None:
        lora, layer = _pair(device)
        for name, seed in (("one", 2), ("again", 2), ("other", 3)):
            lucid.manual_seed(seed)
            lora.add_weighted_adapter(
                ["a", "b"],
                [1.0, 1.0],
                name,
                combination_type=kind,
                density=0.5,  # type: ignore[arg-type]
            )
        deltas = {}
        with lucid.no_grad():
            for name in ("one", "again", "other"):
                deltas[name] = layer.get_delta_weight(name).tolist()
        assert deltas["one"] == deltas["again"] and deltas["one"] != deltas["other"]

    def test_the_rank_is_svd_rank_or_the_largest_source(self, device: str) -> None:
        lora, layer = _pair(device, r=3)
        lora.add_weighted_adapter(
            ["a", "b"], [1.0, 1.0], "default_rank", combination_type="ties_svd", density=0.5
        )
        lora.add_weighted_adapter(
            ["a", "b"],
            [1.0, 1.0],
            "two",
            combination_type="magnitude_prune_svd",
            density=0.5,
            svd_rank=2,
        )
        assert layer.r["default_rank"] == 3 and layer.r["two"] == 2

    def test_the_result_has_at_most_the_asked_rank(self, device: str) -> None:
        lora, layer = _pair(device)
        lora.add_weighted_adapter(
            ["a", "b"], [1.0, 1.0], "mix", combination_type="ties_svd", density=0.9, svd_rank=1
        )
        with lucid.no_grad():
            singular = lucid.linalg.svdvals(layer.get_delta_weight("mix")).tolist()
        assert singular[0] > 1e-6 and all(abs(s) < 1e-4 for s in singular[1:])


class TestRegistry:
    def test_every_reference_name_is_a_registered_strategy(self) -> None:
        for name in ALL_TYPES:
            assert issubclass(_Combination.named(name), _Combination)

    def test_a_single_adapter_is_linear_for_every_type(self, device: str) -> None:
        lora, layer = _pair(device)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # one adapter: no density needed, nothing to prune
            for kind in ALL_TYPES:
                lora.add_weighted_adapter(["a"], [2.0], f"x_{kind}", combination_type=kind)  # type: ignore[arg-type]
        with lucid.no_grad():
            for kind in ALL_TYPES:
                assert _close(
                    layer.get_delta_weight(f"x_{kind}"), layer.get_delta_weight("a") * 2.0
                )

    def test_an_unknown_type_is_refused(self, device: str) -> None:
        lora, _ = _pair(device)
        with pytest.raises(ValueError, match="Invalid combination_type: mean"):
            lora.add_weighted_adapter(["a", "b"], [1.0, 1.0], "mix", combination_type="mean")  # type: ignore[arg-type]
