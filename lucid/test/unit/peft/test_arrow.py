"""Arrow: a router adapter that sends each token through the best few of a library of LoRA adapters.

``y = W x + b + sum_e c_e s_e B_e A_e x`` where ``c`` is a softmax, at temperature ``T``, over the
``top_k`` largest ``|x . p_e|`` and ``p_e`` is the top right singular vector of ``B_e A_e``.  The
tests build adapters whose ``p_e`` is known in closed form (a rank-one update along a basis
vector, or a rank-two one with a wide gap) and write that sum out token by token in plain Python.
They then hold what follows from it: temperature and ``top_k``, GenKnowSub, what the roster reads
off the layer, the refusals, and ``create_arrow_model`` with saved adapters.  The ``parity`` tests
compare a model built by ``create_arrow_model`` with the reference library's.
"""

import copy
import math
from pathlib import Path
from types import ModuleType

import pytest

import lucid
import lucid.linalg as linalg
import lucid.nn as nn
from lucid.peft import (
    AdaLoraConfig,
    ArrowConfig,
    LoraConfig,
    PeftModel,
    create_arrow_model,
    get_peft_model,
)
from lucid.peft.tuners.lora import Conv2d, Linear
from lucid.peft.tuners.lora._arrow import _ArrowRouter
from lucid.test.unit.peft._parity import adapter_library

_IN, _OUT = 6, 4
_ROUTER = "arrow_router"


def _close(a: lucid.Tensor, b: lucid.Tensor, atol: float = 1e-5) -> bool:
    return bool(lucid.allclose(a, b, atol=atol, rtol=1e-4))


def _rows(t: lucid.Tensor) -> list[list[float]]:
    return [[float(v) for v in row] for row in t.tolist()]


# ── adapters with a known prototype ───────────────────────────────────────


def _basis(i: int, scale: float = 1.0) -> list[float]:
    return [scale if j == i else 0.0 for j in range(_IN)]


def _fill(layer: Linear, name: str, a: list[list[float]], b: list[list[float]]) -> None:
    """Set the factors of ``name`` to ``a`` (r, in) and ``b`` (out, r)."""
    device = layer.lora_A[name].weight.device
    with lucid.no_grad():
        layer.lora_A[name].weight.copy_(lucid.tensor(a, device=device))
        layer.lora_B[name].weight.copy_(lucid.tensor(b, device=device))


def _column(values: list[float]) -> list[list[float]]:
    return [[v] for v in values]


#: Three rank-one experts: expert ``e`` reads the input along axis ``e``, so its prototype is that
#: axis and ``|x . p_e| = |x_e|``.  Their output columns and alphas differ.
_B = ([1.0, -2.0, 0.5, 1.5], [0.3, 0.7, -1.1, 2.0], [-0.8, 0.2, 1.2, -0.6])
_ALPHAS = (2.0, 3.0, 5.0)


def _library(device: str, *, extra: int = 0) -> Linear:
    """A layer holding ``task_0 .. task_2`` (rank one, along axes 0, 1, 2) and ``extra`` more tasks."""
    lucid.manual_seed(0)
    layer = Linear(nn.Linear(_IN, _OUT).to(device), "task_0", r=1, lora_alpha=_ALPHAS[0])
    for e in (1, 2):
        layer.update_layer(f"task_{e}", 1, _ALPHAS[e], 0.0, False, False)
    for e in range(3):
        _fill(layer, f"task_{e}", [_basis(e, 1.0 + e)], _column(list(_B[e])))
    for e in range(extra):
        layer.update_layer(f"task_{3 + e}", 1, 1.0, 0.0, False, False)
    return layer


def _route(layer: Linear, **options: object) -> Linear:
    """Add the router over what ``layer`` holds, and make it the adapter in use."""
    layer.update_layer(_ROUTER, 1, 1.0, 0.0, True, False, arrow_config=ArrowConfig(**options))  # type: ignore[arg-type]
    layer.set_adapter(_ROUTER)
    return layer


def _softmax(scores: list[float], temperature: float) -> list[float]:
    top = max(scores)
    exps = [math.exp((s - top) / temperature) for s in scores]
    return [v / sum(exps) for v in exps]


def _routed(
    layer: Linear,
    x: lucid.Tensor,
    *,
    experts: dict[str, tuple[list[list[float]], list[list[float]], float]],
    prototypes: dict[str, list[float]],
    top_k: int,
    temperature: float,
) -> list[list[float]]:
    """The base output plus ``sum_e c_e s_e B_e A_e x``, token by token, from the formula alone."""
    base = _rows(layer.base_layer(x))
    names = list(experts)
    out: list[list[float]] = []
    for token, base_row in zip(_rows(x), base, strict=True):
        sims = [abs(sum(t * p for t, p in zip(token, prototypes[n], strict=True))) for n in names]
        chosen = sorted(range(len(names)), key=lambda i: -sims[i])[:top_k]
        weights = _softmax([sims[i] for i in chosen], temperature)
        row = list(base_row)
        for weight, i in zip(weights, chosen, strict=True):
            a, b, scale = experts[names[i]]
            hidden = [sum(a_r[f] * token[f] for f in range(_IN)) for a_r in a]
            for o in range(_OUT):
                row[o] += weight * scale * sum(b[o][r] * hidden[r] for r in range(len(hidden)))
        out.append(row)
    return out


def _axis_experts() -> (
    tuple[dict[str, tuple[list[list[float]], list[list[float]], float]], dict[str, list[float]]]
):
    experts = {
        f"task_{e}": ([_basis(e, 1.0 + e)], _column(list(_B[e])), _ALPHAS[e]) for e in range(3)
    }
    return experts, {f"task_{e}": _basis(e) for e in range(3)}


def _tokens(device: str) -> lucid.Tensor:
    return lucid.tensor(
        [
            [0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.1, 0.0, 0.0, 0.0],
            [0.5, -1.5, 1.0, 0.2, -0.3, 0.4],
            [0.0, 0.0, -2.5, 1.0, 0.0, 0.0],
        ],
        device=device,
    )


class TestRouting:
    @pytest.mark.parametrize("top_k", [1, 2, 3])
    @pytest.mark.parametrize("temperature", [0.25, 1.0, 4.0])
    def test_a_token_is_the_formula(self, device: str, top_k: int, temperature: float) -> None:
        layer = _route(_library(device), top_k=top_k, router_temperature=temperature, rng_seed=0)
        x = _tokens(device)
        experts, prototypes = _axis_experts()
        want = _routed(
            layer, x, experts=experts, prototypes=prototypes, top_k=top_k, temperature=temperature
        )
        assert _close(layer(x), lucid.tensor(want, device=device), atol=1e-4)

    def test_with_top_one_a_token_takes_the_adapter_its_axis_matches(self, device: str) -> None:
        layer = _route(_library(device), top_k=1, rng_seed=0)
        x = lucid.tensor([[0.0, 3.0, 0.0, 0.0, 0.0, 0.0]], device=device)
        a, b, scale = ([_basis(1, 2.0)], _column(list(_B[1])), _ALPHAS[1])
        update = layer(x) - layer.base_layer(x)
        # Expert 1 alone: s * B (A x) with A x = 2 * 3.
        want = [[scale * b[o][0] * 6.0 for o in range(_OUT)]]
        del a
        assert _close(update, lucid.tensor(want, device=device), atol=1e-4)

    def test_a_lower_temperature_sharpens_the_weights(self, device: str) -> None:
        """At the same ``top_k`` the larger similarity's expert takes more of the token as ``T`` falls."""
        x = lucid.tensor([[1.0, 0.5, 0.0, 0.0, 0.0, 0.0]], device=device)
        updates = {}
        for temperature in (0.02, 1.0, 500.0):
            layer = _route(_library(device), top_k=2, router_temperature=temperature, rng_seed=0)
            updates[temperature] = layer(x) - layer.base_layer(x)
        only_first = _routed_single(0, x, device)
        only_second = _routed_single(1, x, device)
        assert _close(updates[0.02], only_first, atol=1e-3)
        # Near-uniform weights at a large temperature: the mean of the two experts' updates.
        assert _close(updates[500.0], (only_first + only_second) / 2, atol=5e-2)
        assert not _close(updates[1.0], only_first, atol=1e-1)

    def test_the_sign_of_a_prototype_does_not_matter(self, device: str) -> None:
        """Negating ``A_e`` and ``B_e`` flips the singular vector and leaves ``B_e A_e`` as it was."""
        x = _tokens(device)
        plain = _route(_library(device), top_k=2, rng_seed=0)
        flipped = _library(device)
        _fill(flipped, "task_1", [_basis(1, -2.0)], _column([-v for v in _B[1]]))
        _route(flipped, top_k=2, rng_seed=0)
        assert _close(plain(x), flipped(x), atol=1e-5)

    def test_the_prototype_is_the_top_right_singular_vector(self, device: str) -> None:
        layer = _route(_library(device), top_k=1, rng_seed=0)
        protos = layer.lora_arrow[_ROUTER].prototypes
        assert tuple(protos.shape) == (3, _IN)
        for e in range(3):
            row = [float(v) for v in protos[e].tolist()]
            assert abs(abs(row[e]) - 1.0) < 1e-5 and all(
                abs(v) < 1e-5 for j, v in enumerate(row) if j != e
            )

    def test_a_wide_gap_finds_the_dominant_direction_of_a_rank_two_update(
        self, device: str
    ) -> None:
        """``B A`` of rank two: the iteration lands on the right singular vector ``svd`` gives."""
        lucid.manual_seed(3)
        layer = Linear(nn.Linear(_IN, _OUT).to(device), "task_0", r=2, lora_alpha=2.0)
        a = lucid.randn(2, _IN, device=device)
        b = lucid.randn(_OUT, 2, device=device)
        a[0] = a[0] * 6.0
        b[:, 0] = b[:, 0] * 6.0
        _fill(layer, "task_0", _rows(a), _rows(b))
        _route(layer, top_k=1, rng_seed=7)
        _, _, vh = linalg.svd(b @ a, full_matrices=False)
        proto = layer.lora_arrow[_ROUTER].prototypes[0]
        assert abs(float((proto * vh[0]).sum().item())) > 1 - 1e-5

    def test_the_seed_repeats_the_prototypes_and_is_the_same_on_every_device(
        self, device: str
    ) -> None:
        first = _route(_library(device), rng_seed=11).lora_arrow[_ROUTER].prototypes
        again = _route(_library(device), rng_seed=11).lora_arrow[_ROUTER].prototypes
        cpu = _route(_library("cpu"), rng_seed=11).lora_arrow[_ROUTER].prototypes
        assert _close(first, again, atol=0.0) and _close(first.to("cpu"), cpu, atol=1e-6)

    def test_leading_dimensions_are_kept(self, device: str) -> None:
        layer = _route(_library(device), top_k=2, rng_seed=0)
        x = lucid.randn(2, 3, _IN, device=device)
        flat = layer(x.reshape(6, _IN))
        assert tuple(layer(x).shape) == (2, 3, _OUT)
        assert _close(layer(x).reshape(6, _OUT), flat)

    def test_the_gradient_reaches_the_task_adapters_and_the_token(self, device: str) -> None:
        layer = _route(_library(device), top_k=2, rng_seed=0)
        for name in ("task_0", "task_1", "task_2"):
            layer.set_requires_grad(name, True)
        x = _tokens(device).requires_grad_()
        layer(x).sum().backward()
        assert x.grad is not None and bool(lucid.isfinite(x.grad).all().item())
        for name in ("task_0", "task_1", "task_2"):
            grad = layer.lora_B[name].weight.grad
            assert grad is not None and float(grad.abs().sum().item()) > 0.0

    def test_dropout_acts_on_the_routed_update(self, device: str) -> None:
        layer = _library(device)
        layer.update_layer(
            _ROUTER, 1, 1.0, 1.0 - 1e-9, True, False, arrow_config=ArrowConfig(top_k=2)
        )
        layer.set_adapter(_ROUTER)
        x = _tokens(device)
        layer.train()
        # Almost every element of the update is dropped: the output is the base.
        assert _close(layer(x), layer.base_layer(x), atol=1e-3)
        layer.eval()
        assert not _close(layer(x), layer.base_layer(x), atol=1e-3)


def _routed_single_axis(
    axis: int, scale: float, e: int, x: lucid.Tensor, device: str
) -> lucid.Tensor:
    """What expert ``e`` adds to ``x`` once its ``A`` is ``scale`` along ``axis``."""
    a = lucid.tensor([_basis(axis, scale)], device=device)
    b = lucid.tensor(_column(list(_B[e])), device=device)
    return (x @ a.mT) @ b.mT * _ALPHAS[e]


def _routed_single(e: int, x: lucid.Tensor, device: str) -> lucid.Tensor:
    """What expert ``e`` alone adds to ``x``: ``s_e B_e A_e x``."""
    a = lucid.tensor([_basis(e, 1.0 + e)], device=device)
    b = lucid.tensor(_column(list(_B[e])), device=device)
    return (x @ a.mT) @ b.mT * _ALPHAS[e]


# ── GenKnowSub ────────────────────────────────────────────────────────────


def _gks_library(
    device: str,
) -> tuple[Linear, dict[str, tuple[list[list[float]], list[list[float]]]]]:
    """Two rank-two task adapters with a wide gap and two small general ones (``gks_0``, ``gks_1``)."""
    lucid.manual_seed(5)
    layer = Linear(nn.Linear(_IN, _OUT).to(device), "task_0", r=2, lora_alpha=2.0)
    for name in ("task_1", "gks_0", "gks_1"):
        layer.update_layer(name, 2, 2.0, 0.0, False, False)
    factors: dict[str, tuple[list[list[float]], list[list[float]]]] = {}
    for name, size, axis in (
        ("task_0", 4.0, 0),
        ("task_1", 4.0, 3),
        ("gks_0", 0.05, 0),
        ("gks_1", 0.05, 1),
    ):
        a = lucid.randn(2, _IN, device=device) * size * 0.1
        b = lucid.randn(_OUT, 2, device=device) * size * 0.1
        if size > 1.0:
            a[0] = lucid.tensor(_basis(axis, size), device=device)
            b[:, 0] = b[:, 0] * 0.0 + lucid.tensor([1.0, 0.5, -0.5, 1.0], device=device) * size
        factors[name] = (_rows(a), _rows(b))
        _fill(layer, name, *factors[name])
    return layer, factors


class TestGenKnowSub:
    def test_the_mean_of_the_general_adapters_is_subtracted_from_each_task_adapter(
        self, device: str
    ) -> None:
        layer, factors = _gks_library(device)
        mean_a = [
            [(p + q) / 2 for p, q in zip(r0, r1, strict=True)]
            for r0, r1 in zip(factors["gks_0"][0], factors["gks_1"][0], strict=True)
        ]
        mean_b = [
            [(p + q) / 2 for p, q in zip(r0, r1, strict=True)]
            for r0, r1 in zip(factors["gks_0"][1], factors["gks_1"][1], strict=True)
        ]
        experts: dict[str, tuple[list[list[float]], list[list[float]], float]] = {}
        prototypes: dict[str, list[float]] = {}
        for name in ("task_0", "task_1"):
            a = [
                [v - m for v, m in zip(r, mr, strict=True)]
                for r, mr in zip(factors[name][0], mean_a, strict=True)
            ]
            b = [
                [v - m for v, m in zip(r, mr, strict=True)]
                for r, mr in zip(factors[name][1], mean_b, strict=True)
            ]
            experts[name] = (a, b, 1.0)
            _, _, vh = linalg.svd(
                lucid.tensor(b, device=device) @ lucid.tensor(a, device=device), full_matrices=False
            )
            prototypes[name] = [float(v) for v in vh[0].tolist()]
        _route(layer, top_k=2, router_temperature=0.6, use_gks=True, rng_seed=0)
        x = lucid.randn(5, _IN, device=device)
        want = _routed(layer, x, experts=experts, prototypes=prototypes, top_k=2, temperature=0.6)
        assert _close(layer(x), lucid.tensor(want, device=device), atol=1e-4)

    def test_the_stored_weights_are_not_touched(self, device: str) -> None:
        """The reference subtracts the mean from the task adapters' weights at the first pass; here they stay."""
        layer, _ = _gks_library(device)
        before = {n: layer.lora_A[n].weight.clone() for n in ("task_0", "task_1", "gks_0")}
        _route(layer, top_k=1, use_gks=True, rng_seed=0)
        layer(lucid.randn(3, _IN, device=device))
        layer(lucid.randn(3, _IN, device=device))
        for name, weight in before.items():
            assert bool((layer.lora_A[name].weight == weight).all().item())

    def test_the_general_adapters_are_not_task_adapters(self, device: str) -> None:
        layer, _ = _gks_library(device)
        _route(layer, top_k=1, use_gks=True, rng_seed=0)
        assert layer.lora_arrow[_ROUTER]._seen.experts == ("task_0", "task_1")
        assert layer.lora_arrow[_ROUTER]._seen.general == ("gks_0", "gks_1")

    def test_names_in_the_config_say_which_are_general(self, device: str) -> None:
        layer, _ = _gks_library(device)
        for old, new in (("gks_0", "general_a"), ("gks_1", "general_b")):
            layer.lora_A[new] = layer.lora_A[old]
            layer.lora_B[new] = layer.lora_B[old]
            layer._specs[new] = layer._specs[old]
            layer.lora_dropout[new] = layer.lora_dropout[old]
            layer.delete_adapter(old)
        config = ArrowConfig(top_k=1, use_gks=True, rng_seed=0)
        config.gks_adapter_names = ["general_a", "general_b"]
        layer.update_layer(_ROUTER, 1, 1.0, 0.0, True, False, arrow_config=config)
        assert layer.lora_arrow[_ROUTER]._seen.general == ("general_a", "general_b")
        assert layer.lora_arrow[_ROUTER]._seen.experts == ("task_0", "task_1")

    def test_gks_without_general_adapters_is_refused(self, device: str) -> None:
        with pytest.raises(ValueError, match="general-knowledge adapters"):
            _route(_library(device), use_gks=True)

    def test_adapters_of_other_shapes_cannot_be_subtracted(self, device: str) -> None:
        layer = _library(device)
        layer.update_layer("gks_0", 2, 2.0, 0.0, False, False)
        with pytest.raises(ValueError, match="same rank"):
            _route(layer, use_gks=True, top_k=1)


# ── what the router reads off the layer ───────────────────────────────────


class TestRoster:
    def test_the_task_adapters_are_the_plain_adapters_in_the_order_held(self, device: str) -> None:
        layer = _route(_library(device), top_k=1)
        assert layer.lora_arrow[_ROUTER]._seen.experts == ("task_0", "task_1", "task_2")

    def test_dora_lora_bias_and_other_routers_are_not_task_adapters(self, device: str) -> None:
        layer = _library(device)
        layer.update_layer("dora", 1, 1.0, 0.0, True, False, True)
        layer.update_layer("biased", 1, 1.0, 0.0, True, False, False, lora_bias=True)
        _route(layer, top_k=1)
        layer.update_layer("second", 1, 1.0, 0.0, True, False, arrow_config=ArrowConfig(top_k=1))
        assert layer.lora_arrow[_ROUTER]._seen.experts == ("task_0", "task_1", "task_2")
        assert layer.lora_arrow["second"]._seen.experts == ("task_0", "task_1", "task_2")

    def test_an_adapter_added_later_is_routed_over_at_the_next_pass(self, device: str) -> None:
        layer = _route(_library(device), top_k=1, rng_seed=0)
        x = _tokens(device)
        before = layer(x)
        layer.update_layer("task_3", 1, 4.0, 0.0, False, False)
        _fill(layer, "task_3", [_basis(3, 2.0)], _column([1.0, 1.0, 1.0, 1.0]))
        layer.set_adapter(_ROUTER)
        after = layer(x)
        assert layer.lora_arrow[_ROUTER]._seen.experts[-1] == "task_3"
        assert tuple(layer.lora_arrow[_ROUTER].prototypes.shape) == (4, _IN)
        # Only the token along axis 3 (row 3 has x_3 = 1.0 but x_2 = -2.5 wins): the others are as before.
        assert _close(before[:3], after[:3], atol=1e-5)

    def test_values_replaced_in_place_rebuild_the_prototypes(self, device: str) -> None:
        """A prototype follows the adapter it came from: a load over a held name, or any in-place write."""
        layer = _route(_library(device), top_k=1, rng_seed=0)
        along_four = lucid.tensor([[0.0, 0.0, 0.0, 0.0, 2.0, 0.0]], device=device)
        stale = layer(along_four)
        _fill(layer, "task_1", [_basis(4, 3.0)], _column(list(_B[1])))
        routed = layer(along_four)
        assert not _close(stale, routed, atol=1e-3)
        want = layer.base_layer(along_four) + _routed_single_axis(4, 3.0, 1, along_four, device)
        assert _close(routed, want, atol=1e-4)
        with lucid.no_grad():
            layer.lora_A["task_1"].weight.copy_(lucid.tensor([_basis(5, 3.0)], device=device))
        assert abs(float(layer.lora_arrow[_ROUTER].prototypes[1][4].item())) > 0.99
        layer(along_four)
        assert abs(float(layer.lora_arrow[_ROUTER].prototypes[1][5].item())) > 0.99

    def test_a_deleted_adapter_is_dropped_at_the_next_pass(self, device: str) -> None:
        layer = _route(_library(device), top_k=1, rng_seed=0)
        x = lucid.tensor([[0.0, 3.0, 0.0, 0.0, 0.0, 0.0]], device=device)
        taken = layer(x)
        layer.delete_adapter("task_1")
        dropped = layer(x)
        assert layer.lora_arrow[_ROUTER]._seen.experts == ("task_0", "task_2")
        assert not _close(taken, dropped, atol=1e-3)

    def test_deleting_the_router_removes_its_state(self, device: str) -> None:
        layer = _route(_library(device), top_k=1)
        layer.delete_adapter(_ROUTER)
        assert len(layer.lora_arrow) == 0 and _ROUTER not in layer.lora_A

    def test_nothing_of_the_router_is_in_the_state_dict_but_its_factors(self, device: str) -> None:
        layer = _route(_library(device), top_k=1)
        keys = sorted(layer.state_dict())
        assert "lora_A.arrow_router.weight" in keys and "lora_B.arrow_router.weight" in keys
        assert not any("prototypes" in key or "lora_arrow" in key for key in keys)

    def test_the_task_adapter_names_in_the_config_must_be_held(self, device: str) -> None:
        config = ArrowConfig(top_k=1)
        config.task_adapter_names = ["task_0", "missing"]
        with pytest.raises(ValueError, match="missing"):
            _library(device).update_layer(_ROUTER, 1, 1.0, 0.0, True, False, arrow_config=config)


# ── refusals ──────────────────────────────────────────────────────────────


class TestRefusals:
    def test_a_router_over_nothing_is_refused_and_leaves_no_adapter(self, device: str) -> None:
        layer = Linear(nn.Linear(_IN, _OUT).to(device), "other", r=1, use_dora=True)
        with pytest.raises(ValueError, match="holds none"):
            layer.update_layer(_ROUTER, 1, 1.0, 0.0, True, False, arrow_config=ArrowConfig(top_k=1))
        assert list(layer.lora_A) == ["other"] and len(layer.lora_arrow) == 0

    def test_top_k_larger_than_the_task_adapters_is_refused(self, device: str) -> None:
        with pytest.raises(ValueError, match=r"top_k=4 exceeds the 3 task adapter"):
            _route(_library(device), top_k=4)

    def test_lowering_the_number_of_task_adapters_below_top_k_is_refused_at_the_pass(
        self, device: str
    ) -> None:
        layer = _route(_library(device), top_k=3, rng_seed=0)
        layer.delete_adapter("task_2")
        with pytest.raises(ValueError, match="top_k=3 exceeds the 2"):
            layer(_tokens(device))

    def test_arrow_and_dora_do_not_combine(self, device: str) -> None:
        layer = _library(device)
        with pytest.raises(ValueError, match="variant combination"):
            layer.update_layer(
                _ROUTER, 1, 1.0, 0.0, True, False, True, arrow_config=ArrowConfig(top_k=1)
            )
        with pytest.raises(ValueError, match="variant combination"):
            LoraConfig(use_dora=True, arrow_config=ArrowConfig())

    def test_a_router_needs_an_initialisation_that_leaves_the_base_alone(self) -> None:
        for init in ("pissa", "olora", "lora_ga"):
            with pytest.raises(ValueError, match="init_lora_weights"):
                LoraConfig(init_lora_weights=init, arrow_config=ArrowConfig())
        with (
            pytest.warns(UserWarning, match="eva_config"),
            pytest.raises(ValueError, match="init_lora_weights"),
        ):
            LoraConfig(init_lora_weights="eva", arrow_config=ArrowConfig())
        assert LoraConfig(init_lora_weights="gaussian", arrow_config=ArrowConfig()).arrow_config

    def test_only_a_linear_layer_hosts_a_router(self, device: str) -> None:
        lucid.manual_seed(0)
        conv = Conv2d(nn.Conv2d(2, 3, 3).to(device), "task_0", r=1)
        with pytest.raises(ValueError, match="does not support Arrow routing"):
            conv.update_layer(_ROUTER, 1, 1.0, 0.0, True, False, arrow_config=ArrowConfig(top_k=1))

    def test_the_router_cannot_be_merged(self, device: str) -> None:
        layer = _route(_library(device), top_k=1)
        weight = layer.base_layer.weight.clone()
        with pytest.raises(RuntimeError, match="Cannot merge"):
            layer.merge()
        assert not layer.merged and bool((layer.base_layer.weight == weight).all().item())

    def test_merging_a_task_adapter_under_a_live_router_is_refused(self, device: str) -> None:
        """It would apply twice: folded into the weight, and routed over."""
        net = nn.Sequential(nn.Linear(_IN, _IN))
        peft = get_peft_model(net, LoraConfig(r=1, target_modules=["0"]), adapter_name="task_0")
        peft.add_adapter("task_1", LoraConfig(r=1, target_modules=["0"]))
        peft.add_adapter(
            _ROUTER, LoraConfig(arrow_config=ArrowConfig(top_k=1), target_modules=["0"])
        )
        peft.set_adapter(_ROUTER)
        with pytest.raises(ValueError, match="Arrow routers"):
            peft.base_model.merge_adapter(adapter_names=["task_0"])
        with pytest.raises(RuntimeError, match="Cannot merge"):
            peft.merge_and_unload()

    def test_weighted_adapters_do_not_take_a_router(self) -> None:
        net = nn.Sequential(nn.Linear(_IN, _IN))
        peft = get_peft_model(net, LoraConfig(r=1, target_modules=["0"]), adapter_name="task_0")
        peft.add_adapter("task_1", LoraConfig(r=1, target_modules=["0"]))
        peft.add_adapter(
            _ROUTER, LoraConfig(arrow_config=ArrowConfig(top_k=1), target_modules=["0"])
        )
        with pytest.raises(NotImplementedError, match="Arrow"):
            peft.add_weighted_adapter(["task_0", _ROUTER], [0.5, 0.5], "mix")

    def test_a_mixed_batch_cannot_name_the_router(self) -> None:
        net = nn.Sequential(nn.Linear(_IN, _IN))
        peft = get_peft_model(net, LoraConfig(r=1, target_modules=["0"]), adapter_name="task_0")
        peft.add_adapter("task_1", LoraConfig(r=1, target_modules=["0"]))
        peft.add_adapter(
            _ROUTER, LoraConfig(arrow_config=ArrowConfig(top_k=1), target_modules=["0"])
        )
        peft.eval()
        with pytest.raises(ValueError, match="Arrow router"):
            peft(lucid.randn(2, _IN), adapter_names=["task_0", _ROUTER])

    def test_adalora_does_not_take_the_field(self) -> None:
        with pytest.raises(NotImplementedError, match="arrow_config"):
            AdaLoraConfig(total_step=10, arrow_config=ArrowConfig())

    def test_a_router_config_needs_a_router_over_adapters_the_model_holds(self) -> None:
        net = nn.Sequential(nn.Linear(_IN, _IN))
        with pytest.raises(ValueError, match="holds none"):
            get_peft_model(net, LoraConfig(arrow_config=ArrowConfig(top_k=1), target_modules=["0"]))
        assert type(net[0]) is nn.Linear


# ── the config ────────────────────────────────────────────────────────────


class TestConfig:
    def test_the_field_is_read_written_and_read_back(self, tmp_path: Path) -> None:
        config = LoraConfig(
            arrow_config=ArrowConfig(top_k=2, router_temperature=0.5, use_gks=False, rng_seed=3),
            target_modules=["q"],
        )
        config.save_pretrained(tmp_path)
        loaded = LoraConfig.from_pretrained(tmp_path)
        assert loaded == config and loaded.arrow_config == ArrowConfig(
            top_k=2, router_temperature=0.5, rng_seed=3
        )

    def test_a_file_that_records_the_names_reads_without_them(self) -> None:
        """The names are filled in by ``create_arrow_model``; a file keeps them, a config does not read them."""
        config = LoraConfig(arrow_config=ArrowConfig(top_k=1))
        data = config.to_dict()
        arrow = data["arrow_config"]
        assert isinstance(arrow, dict)
        arrow["task_adapter_names"] = ["task_0"]
        assert LoraConfig.from_dict(data).arrow_config == ArrowConfig(top_k=1)


# ── create_arrow_model ────────────────────────────────────────────────────


def _base(device: str = "cpu") -> nn.Module:
    lucid.manual_seed(1)
    return nn.Sequential(nn.Linear(_IN, 8), nn.ReLU(), nn.Linear(8, _OUT)).to(device)


def _separated(model: PeftModel, seed: int) -> None:
    """Give every adapter layer factors with a wide singular gap, so a prototype is well defined."""
    lucid.manual_seed(seed)
    for module in model.get_base_model().modules():
        if not isinstance(module, Linear):
            continue
        for name in module._adapter_names():
            a, b = module._factor_pairs(name)[0]
            with lucid.no_grad():
                new_a = lucid.randn(*a.shape, device=a.device) * 0.3
                new_b = lucid.randn(*b.shape, device=b.device) * 0.3
                new_a[0] = new_a[0] * 12.0
                new_b[:, 0] = new_b[:, 0] * 12.0
                a.copy_(new_a)
                b.copy_(new_b)


def _save(
    directory: Path, name: str, seed: int, *, r: int = 3, modules: list[str] | None = None
) -> str:
    peft = get_peft_model(
        _base(),
        LoraConfig(
            r=r, lora_alpha=2 * r, target_modules=modules or ["0", "2"], init_lora_weights=False
        ),
    )
    _separated(peft, seed)
    path = directory / name
    peft.save_pretrained(path)
    return str(path)


@pytest.fixture
def library(tmp_path: Path) -> dict[str, str]:
    """Saved adapters: three tasks and two general ones, all rank three on ``0`` and ``2``."""
    return {
        name: _save(tmp_path, name, seed)
        for seed, name in enumerate(("a", "b", "c", "g0", "g1"), 11)
    }


class TestCreateArrowModel:
    def test_it_loads_the_adapters_and_routes_over_them(
        self, device: str, library: dict[str, str]
    ) -> None:
        config = ArrowConfig(top_k=2, router_temperature=0.8, rng_seed=0)
        model = create_arrow_model(
            _base(device), [library["a"], library["b"], library["c"]], config
        )
        assert isinstance(model, PeftModel)
        assert sorted(model.peft_config) == [_ROUTER, "task_0", "task_1", "task_2"]
        assert model.active_adapter == _ROUTER
        assert config.task_adapter_names == ["task_0", "task_1", "task_2"]
        assert config.gks_adapter_names == []
        layer = model.get_base_model()[0]
        assert isinstance(layer, Linear) and layer.lora_arrow[_ROUTER]._seen.experts == (
            "task_0",
            "task_1",
            "task_2",
        )
        x = lucid.randn(4, _IN, device=device)
        assert tuple(model(x).shape) == (4, _OUT)
        assert model.peft_config[_ROUTER].target_modules == {"0", "2"}

    def test_the_router_is_the_formula_over_the_loaded_adapters(
        self, device: str, library: dict[str, str]
    ) -> None:
        """Built one layer at a time by hand and by the function: the same output."""
        paths = [library["a"], library["b"], library["c"]]
        model = create_arrow_model(
            _base(device), paths, ArrowConfig(top_k=2, router_temperature=0.7, rng_seed=4)
        )
        net = _base(device)
        by_hand = get_peft_model(net, LoraConfig.from_pretrained(paths[0]), adapter_name="task_0")
        by_hand.load_adapter(
            paths[0], "task_0"
        )  # the weights are the file's, not the config's draw
        for i, path in enumerate(paths[1:], start=1):
            by_hand.load_adapter(path, f"task_{i}")
        by_hand.add_adapter(
            _ROUTER,
            LoraConfig(
                arrow_config=ArrowConfig(top_k=2, router_temperature=0.7, rng_seed=4),
                target_modules=["0", "2"],
                r=3,
            ),
        )
        by_hand.set_adapter(_ROUTER)
        x = lucid.randn(5, _IN, device=device)
        assert _close(model(x), by_hand(x), atol=1e-4)

    def test_general_adapters_are_loaded_as_gks(self, library: dict[str, str]) -> None:
        config = ArrowConfig(top_k=2, use_gks=True, rng_seed=0)
        model = create_arrow_model(
            _base(),
            [library["a"], library["b"], library["c"]],
            config,
            general_adapter_paths=[library["g0"], library["g1"]],
        )
        assert sorted(model.peft_config) == [
            _ROUTER,
            "gks_0",
            "gks_1",
            "task_0",
            "task_1",
            "task_2",
        ]
        assert config.gks_adapter_names == ["gks_0", "gks_1"]
        layer = model.get_base_model()[0]
        assert layer.lora_arrow[_ROUTER]._seen.general == ("gks_0", "gks_1")
        assert tuple(model(lucid.randn(3, _IN)).shape) == (3, _OUT)

    def test_general_adapters_given_without_use_gks_are_not_loaded(
        self, library: dict[str, str]
    ) -> None:
        model = create_arrow_model(
            _base(), [library["a"], library["b"]], ArrowConfig(top_k=1), [library["g0"]]
        )
        assert sorted(model.peft_config) == [_ROUTER, "task_0", "task_1"]

    def test_it_takes_the_signature_of_the_reference(self) -> None:
        import inspect

        parameters = inspect.signature(create_arrow_model).parameters
        assert list(parameters) == [
            "base_model",
            "task_specific_adapter_paths",
            "arrow_config",
            "general_adapter_paths",
            "adapter_kwargs",
        ]
        assert parameters["general_adapter_paths"].default is None
        assert parameters["adapter_kwargs"].kind is inspect.Parameter.VAR_KEYWORD

    def test_adapter_kwargs_go_to_the_loaders(
        self, tmp_path: Path, library: dict[str, str]
    ) -> None:
        nested = tmp_path / "bundle"
        nested.mkdir()
        for name in ("a", "b"):
            (nested / name).mkdir()
            for entry in Path(library[name]).iterdir():
                (nested / name / entry.name).write_bytes(entry.read_bytes())
        model = create_arrow_model(
            _base(),
            [str(nested / "a"), str(nested / "b")],
            ArrowConfig(top_k=1),
            is_trainable=False,
        )
        assert sorted(model.peft_config) == [_ROUTER, "task_0", "task_1"]
        with pytest.raises(FileNotFoundError):
            create_arrow_model(_base(), [str(tmp_path)], ArrowConfig(top_k=1), subfolder="nowhere")

    def test_the_task_adapters_are_frozen_and_the_router_is_the_one_in_use(
        self, library: dict[str, str]
    ) -> None:
        model = create_arrow_model(_base(), [library["a"], library["b"]], ArrowConfig(top_k=1))
        layer = model.get_base_model()[0]
        assert not layer.lora_A["task_0"].weight.requires_grad
        assert layer.active_adapters == [_ROUTER]

    def test_disabling_the_adapters_runs_the_base(self, library: dict[str, str]) -> None:
        model = create_arrow_model(_base(), [library["a"], library["b"]], ArrowConfig(top_k=1))
        x = lucid.randn(3, _IN)
        with model.disable_adapter():
            off = model(x)
        assert _close(off, _base()(x))
        assert not _close(model(x), off, atol=1e-5)

    def test_the_model_cannot_be_merged_until_the_router_is_removed(
        self, library: dict[str, str]
    ) -> None:
        model = create_arrow_model(_base(), [library["a"], library["b"]], ArrowConfig(top_k=1))
        with pytest.raises(RuntimeError, match="Cannot merge"):
            model.merge_and_unload()
        with pytest.warns(UserWarning, match="Setting active adapter"):
            model.delete_adapter(_ROUTER)
        model.set_adapter("task_0")
        merged = model.merge_and_unload()
        assert type(merged[0]) is nn.Linear


class TestCreateArrowModelRefusals:
    def test_no_task_adapter(self) -> None:
        with pytest.raises(ValueError, match="at least one adapter path"):
            create_arrow_model(_base(), [], ArrowConfig())

    def test_gks_needs_general_adapters_before_anything_is_loaded(
        self, library: dict[str, str]
    ) -> None:
        net = _base()
        for general in (None, []):
            with pytest.raises(ValueError, match="general LoRA paths"):
                create_arrow_model(net, [library["a"]], ArrowConfig(top_k=1, use_gks=True), general)
        assert type(net[0]) is nn.Linear

    def test_a_directory_without_a_config_and_a_missing_one(
        self, tmp_path: Path, library: dict[str, str]
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        net = _base()
        with pytest.raises(ValueError, match="adapter_config.json"):
            create_arrow_model(net, [library["a"], str(empty)], ArrowConfig(top_k=1))
        with pytest.raises(FileNotFoundError, match="hub ids"):
            create_arrow_model(net, [str(tmp_path / "absent")], ArrowConfig(top_k=1))
        assert type(net[0]) is nn.Linear

    def test_a_failure_undoes_the_adapters_loaded_so_far(
        self, tmp_path: Path, library: dict[str, str]
    ) -> None:
        other_rank = _save(tmp_path, "rank2", 99, r=2)
        net = _base()
        with pytest.raises(ValueError, match=r"\[task_1\] rank mismatch"):
            create_arrow_model(net, [library["a"], other_rank], ArrowConfig(top_k=1))
        assert type(net[0]) is nn.Linear and type(net[2]) is nn.Linear

    def test_adapters_on_other_modules(self, tmp_path: Path, library: dict[str, str]) -> None:
        other = _save(tmp_path, "one_module", 98, modules=["0"])
        net = _base()
        with pytest.raises(ValueError, match=r"\[task_1\] target_modules mismatch"):
            create_arrow_model(net, [library["a"], other], ArrowConfig(top_k=1))
        assert type(net[0]) is nn.Linear

    def test_a_top_k_larger_than_the_task_adapters_undoes_the_load(
        self, library: dict[str, str]
    ) -> None:
        net = _base()
        with pytest.raises(ValueError, match="top_k=3 exceeds"):
            create_arrow_model(net, [library["a"], library["b"]], ArrowConfig(top_k=3))
        assert type(net[0]) is nn.Linear

    def test_a_dora_adapter_is_refused(self, tmp_path: Path, library: dict[str, str]) -> None:
        peft = get_peft_model(_base(), LoraConfig(r=3, target_modules=["0", "2"], use_dora=True))
        peft.save_pretrained(tmp_path / "dora")
        net = _base()
        with pytest.raises(ValueError, match="plain LoRA"):
            create_arrow_model(net, [library["a"], str(tmp_path / "dora")], ArrowConfig(top_k=1))
        assert type(net[0]) is nn.Linear

    def test_an_adapter_on_a_convolution_is_refused(self, tmp_path: Path) -> None:
        def conv() -> nn.Module:
            lucid.manual_seed(2)
            return nn.Sequential(nn.Conv2d(2, 3, 3))

        for name in ("a", "b"):
            get_peft_model(conv(), LoraConfig(r=1, target_modules=["0"])).save_pretrained(
                tmp_path / name
            )
        net = conv()
        with pytest.raises(TypeError, match="nn.Linear"):
            create_arrow_model(
                net, [str(tmp_path / "a"), str(tmp_path / "b")], ArrowConfig(top_k=1)
            )
        assert type(net[0]) is nn.Conv2d


class TestSaveAndLoad:
    def test_a_saved_router_model_is_rebuilt_from_its_adapters(
        self, tmp_path: Path, library: dict[str, str]
    ) -> None:
        """Save the library ``create_arrow_model`` made; build it again from those files."""
        model = create_arrow_model(
            _base(), [library["a"], library["b"], library["c"]], ArrowConfig(top_k=2, rng_seed=0)
        )
        x = lucid.randn(6, _IN)
        want = model(x)
        directory = tmp_path / "saved"
        model.save_pretrained(directory)
        assert (directory / _ROUTER / "adapter_config.json").is_file()
        again = create_arrow_model(
            _base(),
            [str(directory / f"task_{i}") for i in range(3)],
            ArrowConfig(top_k=2, rng_seed=0),
        )
        assert _close(again(x), want, atol=1e-5)

    def test_a_saved_router_loads_beside_the_adapters_it_routes_over(
        self, tmp_path: Path, library: dict[str, str]
    ) -> None:
        model = create_arrow_model(
            _base(), [library["a"], library["b"], library["c"]], ArrowConfig(top_k=2, rng_seed=0)
        )
        x = lucid.randn(6, _IN)
        want = model(x)
        directory = tmp_path / "saved"
        model.save_pretrained(directory)
        saved = json_arrow(directory / _ROUTER)
        assert saved["top_k"] == 2 and saved["task_adapter_names"] == ["task_0", "task_1", "task_2"]
        fresh = PeftModel.from_pretrained(_base(), directory / "task_0", adapter_name="task_0")
        fresh.load_adapter(directory / "task_1", "task_1")
        fresh.load_adapter(directory / "task_2", "task_2")
        fresh.load_adapter(directory / _ROUTER, _ROUTER)
        fresh.set_adapter(_ROUTER)
        assert fresh.peft_config[_ROUTER].arrow_config == ArrowConfig(top_k=2, rng_seed=0)
        assert _close(fresh(x), want, atol=1e-5)

    def test_a_saved_router_without_its_adapters_is_refused_and_leaves_the_model(
        self, tmp_path: Path, library: dict[str, str]
    ) -> None:
        model = create_arrow_model(_base(), [library["a"], library["b"]], ArrowConfig(top_k=1))
        model.save_pretrained(tmp_path / "saved")
        net = _base()
        with pytest.raises(ValueError, match="holds none"):
            PeftModel.from_pretrained(net, tmp_path / "saved" / _ROUTER)
        assert type(net[0]) is nn.Linear

    def test_a_copy_of_the_model_routes_the_same(self, library: dict[str, str]) -> None:
        model = create_arrow_model(
            _base(), [library["a"], library["b"], library["c"]], ArrowConfig(top_k=2, rng_seed=0)
        )
        x = lucid.randn(4, _IN)
        twin = copy.deepcopy(model)
        assert _close(twin(x), model(x), atol=1e-6)


def json_arrow(directory: Path) -> dict[str, object]:
    import json

    saved = json.loads((directory / "adapter_config.json").read_text())["arrow_config"]
    assert isinstance(saved, dict)
    return saved


class TestRouterModule:
    def test_it_holds_no_parameter_and_one_buffer_that_is_not_saved(self, device: str) -> None:
        router = _route(_library(device), top_k=1).lora_arrow[_ROUTER]
        assert isinstance(router, _ArrowRouter)
        assert list(router.parameters()) == [] and list(router.state_dict()) == []
        assert [name for name, _ in router.named_buffers()] == ["prototypes"]

    def test_its_settings_are_plain_attributes(self, device: str) -> None:
        """``lucid.compile`` keys a trace on a module's plain attributes."""
        router = _route(_library(device), top_k=2, router_temperature=0.5, rng_seed=9).lora_arrow[
            _ROUTER
        ]
        assert (router.top_k, router.temperature, router.rng_seed) == (2, 0.5, 9)
        assert router._seen.experts == ("task_0", "task_1", "task_2")

    def test_a_half_precision_base_gets_float32_prototypes(self) -> None:
        lucid.manual_seed(0)
        layer = Linear(nn.Linear(_IN, _OUT).to(lucid.float16), "task_0", r=1, lora_alpha=2.0)
        layer.update_layer("task_1", 1, 2.0, 0.0, False, False)
        _route(layer, top_k=1, rng_seed=0)
        assert layer.lora_arrow[_ROUTER].prototypes.dtype == lucid.float32
        x = lucid.randn(3, _IN).to(lucid.float16)
        assert layer(x).dtype == lucid.float16


# ── against the reference library ─────────────────────────────────────────


@pytest.mark.parity
class TestAgainstTheReferenceLibrary:
    @staticmethod
    def _twin(ref: ModuleType, net: nn.Module) -> ModuleType:
        """The reference framework's copy of ``net``: ``Linear(6, 8)``, ReLU, ``Linear(8, 4)`` with the same weights."""
        twin = ref.nn.Sequential(ref.nn.Linear(_IN, 8), ref.nn.ReLU(), ref.nn.Linear(8, _OUT))
        with ref.no_grad():
            for i in (0, 2):
                twin[i].weight.copy_(ref.tensor(net[i].weight.tolist()))
                twin[i].bias.copy_(ref.tensor(net[i].bias.tolist()))
        return twin

    @pytest.mark.parametrize(
        ("top_k", "temperature", "use_gks"),
        [(1, 1.0, False), (2, 0.5, False), (3, 2.0, False), (2, 0.7, True)],
    )
    def test_the_same_adapters_route_to_the_same_output(
        self,
        ref: ModuleType,
        library: dict[str, str],
        top_k: int,
        temperature: float,
        use_gks: bool,
    ) -> None:
        reference = adapter_library()
        paths = [library["a"], library["b"], library["c"]]
        general = [library["g0"], library["g1"]] if use_gks else None
        theirs = reference.create_arrow_model(
            self._twin(ref, _base()),
            paths,
            reference.ArrowConfig(top_k=top_k, router_temperature=temperature, use_gks=use_gks),
            general,
        )
        mine = create_arrow_model(
            _base(),
            paths,
            ArrowConfig(top_k=top_k, router_temperature=temperature, use_gks=use_gks),
            general,
        )
        lucid.manual_seed(1)
        x = lucid.randn(7, _IN)
        theirs.eval()
        with ref.no_grad():
            want = theirs(ref.tensor(x.tolist())).tolist()
        with lucid.no_grad():
            got = mine(x)
        assert _close(got, lucid.tensor(want), atol=1e-4)

    def test_the_prototypes_are_the_same_vectors_up_to_sign(
        self, ref: ModuleType, library: dict[str, str]
    ) -> None:
        reference = adapter_library()
        paths = [library["a"], library["b"], library["c"]]
        theirs = reference.create_arrow_model(
            self._twin(ref, _base()), paths, reference.ArrowConfig(top_k=2)
        )
        mine = create_arrow_model(_base(), paths, ArrowConfig(top_k=2))
        x = ref.randn(3, _IN)
        theirs.eval()
        with ref.no_grad():
            theirs(x)
        mine(lucid.tensor(x.tolist()))
        for module in ("0", "2"):
            index = int(module)
            their_protos = theirs.base_model.model[index].lora_arrow[_ROUTER].prototypes.tolist()
            my_protos = mine.get_base_model()[index].lora_arrow[_ROUTER].prototypes.tolist()
            for theirs_row, mine_row in zip(their_protos, my_protos, strict=True):
                cosine = sum(p * q for p, q in zip(theirs_row, mine_row, strict=True))
                assert abs(abs(cosine) - 1.0) < 1e-4
