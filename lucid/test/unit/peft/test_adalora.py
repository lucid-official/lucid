"""AdaLoRA: its config, SVD layers, rank allocator, loss penalty and adapter files (CPU and Metal).

The tests pin the formulas the reference gives — the update ``B diag(E) A * alpha / ranknum``,
the cubic budget schedule, the importance score, the orthogonality penalty — on tensors small
enough to compute by hand, and the file format with a pruned adapter written in the reference's
layout.  The AdaLoRA entries of the generic tuner contract (merging, multi-adapter, files,
compile) are in ``test_tuner_contract.py``.
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import PeftConfig, PeftModel, PeftType, get_peft_model, get_peft_model_state_dict
from lucid.peft.tuners._base import _tuner_layers
from lucid.peft.tuners.adalora import (
    AdaLoraConfig,
    AdaLoraLayer,
    AdaLoraModel,
    RankAllocator,
    SVDConv2d,
    SVDLinear,
)
from lucid.serialization import load_safetensors, save_safetensors


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def _close(a: lucid.Tensor, b: lucid.Tensor, atol: float = 1e-5) -> bool:
    return bool(lucid.allclose(a, b, atol=atol))


def _mlp(device: str = "cpu") -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4)).to(device)


def _config(**kw: object) -> AdaLoraConfig:
    fields: dict[str, object] = {
        "init_r": 4,
        "target_r": 2,
        "total_step": 10,
        "target_modules": ["0", "2"],
        **kw,
    }
    return AdaLoraConfig(**fields)  # type: ignore[arg-type]


def _layer(model: nn.Module, name: str) -> AdaLoraLayer:
    found = dict(_tuner_layers(model))[name]
    assert isinstance(found, AdaLoraLayer)
    return found


def _set_grads(model: nn.Module, *, singular: dict[str, list[float]]) -> None:
    """Gradients that make the singular values' importances ``singular`` (A and B get none)."""
    for name, grads in singular.items():
        layer = _layer(model, name)
        e, a, b = layer.lora_E["default"], layer.lora_A["default"], layer.lora_B["default"]
        e.grad = lucid.tensor(grads, device=e.device).reshape(-1, 1)
        a.grad = lucid.zeros_like(a)
        b.grad = lucid.zeros_like(b)


def _unit_adapter(model: nn.Module) -> None:
    """Every adapter tensor 1: importance is then the gradient's magnitude."""
    with lucid.no_grad():
        for _, layer in _tuner_layers(model):
            assert isinstance(layer, AdaLoraLayer)
            for table in (layer.lora_E, layer.lora_A, layer.lora_B):
                table["default"].copy_(lucid.ones_like(table["default"]))


# ── the config ───────────────────────────────────────────────────────────────


class TestConfig:
    def test_the_reference_defaults(self) -> None:
        config = AdaLoraConfig(total_step=100)
        assert (config.target_r, config.init_r, config.tinit, config.tfinal) == (8, 12, 0, 0)
        assert (config.deltaT, config.beta1, config.beta2, config.orth_reg_weight) == (
            1,
            0.85,
            0.85,
            0.5,
        )
        assert config.rank_pattern is None and config.peft_type is PeftType.ADALORA

    def test_total_step_is_required(self) -> None:
        with pytest.raises(ValueError, match="total_step"):
            AdaLoraConfig()
        with pytest.raises(ValueError, match="total_step"):
            AdaLoraConfig(total_step=0)

    @pytest.mark.parametrize(
        ("kw", "match"),
        [
            ({"tinit": 8, "tfinal": 2}, "budgeting phase"),
            ({"tinit": 10}, "budgeting phase"),
            ({"orth_reg_weight": -0.1}, "greater than or equal to 0"),
            ({"beta1": 1.0}, "strictly between"),
            ({"beta2": 0.0}, "strictly between"),
            ({"deltaT": 0}, "positive"),
            ({"tinit": -1}, "non-negative"),
            ({"lora_dropout": 1.0}, "lie in"),
        ],
    )
    def test_out_of_range_fields_are_refused(self, kw: dict[str, object], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _config(**kw)

    def test_wrong_types_are_refused(self) -> None:
        with pytest.raises(TypeError, match="init_r"):
            _config(init_r=2.5)
        with pytest.raises(TypeError, match="tfinal"):
            _config(tfinal=True)

    def test_r_is_ignored_with_a_warning(self) -> None:
        with pytest.warns(UserWarning, match="`r` is not used in AdaLora"):
            _config(r=4)

    def test_bias_other_than_none_is_not_built(self) -> None:
        with pytest.raises(NotImplementedError, match="bias='all'"):
            _config(bias="all")

    def test_a_rank_pattern_drops_the_adapter_name_of_its_keys(self) -> None:
        config = _config(rank_pattern={"a.b.lora_E.default": [True, False], "c.lora_E": [True]})
        assert config.rank_pattern == {"a.b.lora_E": [True, False], "c.lora_E": [True]}

    @pytest.mark.parametrize(
        ("pattern", "error"),
        [
            ({"a.weight": [True]}, ValueError),
            ({"a.lora_E": [1, 0]}, TypeError),
            ({"a.lora_E": True}, TypeError),
            (["a.lora_E"], TypeError),
        ],
    )
    def test_a_malformed_rank_pattern_is_refused(
        self, pattern: object, error: type[Exception]
    ) -> None:
        with pytest.raises(error):
            _config(rank_pattern=pattern)

    def test_the_dict_round_trip_keeps_every_field(self) -> None:
        config = _config(rank_pattern={"0.lora_E": [True, False, True, True]}, tinit=1, tfinal=2)
        assert AdaLoraConfig.from_dict(config.to_dict()) == config
        assert PeftConfig.from_dict(config.to_dict()) == config

    def test_a_reference_file_with_lora_fields_reads_without_a_warning(self) -> None:
        reference = {
            **_config().to_dict(),
            "alpha_pattern": {},
            "use_dora": False,
            "use_rslora": False,
            "lora_bias": False,
            "target_parameters": None,
            "loftq_config": {},
            "megatron_config": None,
            "layer_replication": None,
            "peft_version": "0.21.2",
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert AdaLoraConfig.from_dict(reference) == _config()

    def test_a_file_that_turns_a_lora_feature_on_is_refused(self) -> None:
        with pytest.raises(NotImplementedError, match="use_dora"):
            AdaLoraConfig.from_dict({**_config().to_dict(), "use_dora": True})

    def test_a_lora_file_is_not_an_adalora_file(self) -> None:
        from lucid.peft import LoraConfig

        with pytest.raises(NotImplementedError, match="ADALORA"):
            LoraConfig.from_dict(_config().to_dict())


# ── the layers ───────────────────────────────────────────────────────────────


class TestLayer:
    def test_the_tensors_and_their_shapes(self) -> None:
        base = nn.Linear(8, 5)
        layer = SVDLinear(base, "a", _config(), r=3, lora_alpha=6)
        assert tuple(layer.lora_A["a"].shape) == (3, 8)
        assert tuple(layer.lora_E["a"].shape) == (3, 1)
        assert tuple(layer.lora_B["a"].shape) == (5, 3)
        assert float(layer.ranknum["a"].item()) == 3.0 and not layer.ranknum["a"].requires_grad
        assert dict(layer.r) == {"a": 3} and dict(layer.scaling) == {"a": 6.0}
        assert layer.adapter_layer_names == ("lora_A", "lora_B", "lora_E")

    def test_a_zero_alpha_takes_the_rank_as_the_scale(self) -> None:
        layer = SVDLinear(nn.Linear(4, 4), "a", _config(), r=3, lora_alpha=0)
        assert dict(layer.scaling) == {"a": 3.0} and dict(layer.lora_alpha) == {"a": 0}

    def test_the_default_start_is_the_base_bit_for_bit(self, device: str) -> None:
        base = nn.Linear(8, 4).to(device)
        layer = SVDLinear(base, "a", _config(), r=3, lora_alpha=6)
        assert _same(layer.lora_E["a"], lucid.zeros(3, 1, device=device))
        x = lucid.randn(5, 8, device=device)
        assert _same(layer(x), base(x))

    def test_the_update_is_b_diag_e_a_over_ranknum(self, device: str) -> None:
        base = nn.Linear(4, 3).to(device)
        layer = SVDLinear(base, "a", _config(init_lora_weights=False), r=2, lora_alpha=6)
        x = lucid.randn(5, 4, device=device)
        a, e, b = layer.lora_A["a"], layer.lora_E["a"], layer.lora_B["a"]
        expected = base(x) + (x @ (a * e).mT @ b.mT) * (6 / (2 + 1e-5))
        assert _close(layer(x), expected)
        delta = layer.get_delta_weight("a")
        assert tuple(delta.shape) == (3, 4)
        assert _close(layer(x), base(x) + x @ delta.mT)

    def test_a_layer_pruned_to_nothing_is_the_base(self, device: str) -> None:
        base = nn.Linear(4, 3).to(device)
        layer = SVDLinear(base, "a", _config(init_lora_weights=False), r=2, lora_alpha=4)
        layer.prune("a", [])
        assert dict(layer.r) == {"a": 0}
        x = lucid.randn(5, 4, device=device)
        assert _same(layer(x), base(x))
        assert _same(layer.get_delta_weight("a"), lucid.zeros(3, 4, device=device))
        SVDLinear(nn.Linear(4, 3), "z", _config(), r=0)  # r = 0 is accepted, as in the reference

    def test_a_negative_rank_and_a_taken_name_are_refused(self) -> None:
        with pytest.raises(ValueError, match="positive integer or 0"):
            SVDLinear(nn.Linear(4, 3), "a", _config(), r=-1)
        layer = SVDLinear(nn.Linear(4, 3), "a", _config(), r=2)
        with pytest.raises(ValueError, match="already exists"):
            layer.update_layer("a", 2, 1, _config())

    def test_dropout_acts_on_the_adapter_input_only(self) -> None:
        base = nn.Linear(6, 4)
        layer = SVDLinear(base, "a", _config(lora_dropout=0.5, init_lora_weights=False), r=3)
        x = lucid.randn(4, 6)
        layer.eval()
        assert _same(layer(x), layer(x))
        layer.train()
        lucid.manual_seed(1)
        first = layer(x)
        lucid.manual_seed(2)
        assert not _same(first, layer(x))
        assert _same(layer.base_layer(x), base(x))

    def test_the_orthogonality_penalty_is_the_frobenius_departure_from_identity(self) -> None:
        layer = SVDLinear(nn.Linear(5, 4), "a", _config(init_lora_weights=False), r=3)
        a, b = layer.lora_A["a"], layer.lora_B["a"]
        eye = lucid.eye(3)
        want_a = lucid.linalg.norm(a @ a.mT - eye, ord="fro")
        want_b = lucid.linalg.norm(b.mT @ b - eye, ord="fro")
        got_a, got_b = layer.orthogonality("a")
        assert _close(got_a, want_a) and _close(got_b, want_b)
        del got_a, got_b, want_a, want_b  # their graphs hold the storage the write below needs
        with lucid.no_grad():
            a.copy_(lucid.eye(3, 5))
            b.copy_(lucid.eye(4, 3))
        assert all(float(p.item()) == 0.0 for p in layer.orthogonality("a"))

    def test_the_penalty_carries_gradient_to_the_vectors(self) -> None:
        layer = SVDLinear(nn.Linear(5, 4), "a", _config(init_lora_weights=False), r=3)
        sum(layer.orthogonality("a")).backward()  # type: ignore[union-attr]
        assert layer.lora_A["a"].grad is not None and layer.lora_B["a"].grad is not None
        assert layer.lora_E["a"].grad is None

    def test_prune_keeps_the_scale_of_what_is_kept(self) -> None:
        base = nn.Linear(6, 4)
        layer = SVDLinear(base, "a", _config(init_lora_weights=False), r=4, lora_alpha=8)
        x = lucid.randn(5, 6)
        with lucid.no_grad():
            layer.lora_E["a"].copy_(lucid.tensor([[1.0], [0.0], [2.0], [0.0]]))
        before = layer(x)
        layer.prune("a", [0, 2])
        assert dict(layer.r) == {"a": 2} and tuple(layer.lora_E["a"].shape) == (2, 1)
        assert float(layer.ranknum["a"].item()) == 4.0  # the scale is the one training began with
        assert _close(layer(x), before)

    def test_prune_and_resize_keep_the_trainable_flag(self) -> None:
        layer = SVDLinear(nn.Linear(6, 4), "a", _config(), r=3)
        layer.set_requires_grad("a", False)
        layer.prune("a", [0])
        assert not layer.lora_A["a"].requires_grad
        layer.resize("a", 2)
        assert not any(p.requires_grad for p in layer._adapter_parameters("a"))

    def test_resize_grows_with_zero_singular_values(self) -> None:
        base = nn.Linear(6, 4)
        layer = SVDLinear(base, "a", _config(init_lora_weights=False), r=2)
        x = lucid.randn(3, 6)
        before = layer(x)
        layer.resize("a", 4)
        assert dict(layer.r) == {"a": 4} and tuple(layer.lora_B["a"].shape) == (4, 4)
        assert _same(layer.lora_E["a"][2:], lucid.zeros(2, 1))
        assert _close(layer(x), before)
        layer.resize("a", 1)
        assert dict(layer.r) == {"a": 1}

    def test_prune_refuses_an_index_outside_the_rank_and_an_unknown_adapter(self) -> None:
        layer = SVDLinear(nn.Linear(6, 4), "a", _config(), r=3)
        with pytest.raises(IndexError, match="not all in"):
            layer.prune("a", [0, 3])
        with pytest.raises(ValueError, match="holds no adapter"):
            layer.prune("b", [0])

    def test_deleting_an_adapter_leaves_nothing_behind(self) -> None:
        layer = SVDLinear(nn.Linear(6, 4), "a", _config(lora_dropout=0.1), r=3)
        layer.update_layer("b", 2, 1, _config())
        layer.delete_adapter("a")
        assert layer._adapter_names() == ["b"]
        for table in (layer.lora_A, layer.lora_B, layer.lora_E, layer.ranknum, layer.lora_dropout):
            assert list(table) == ["b"]
        assert layer.active_adapters == []

    def test_a_half_base_gets_a_float32_adapter(self) -> None:
        layer = SVDLinear(nn.Linear(4, 3).to(lucid.float16), "a", _config(), r=2)
        assert layer.lora_A["a"].dtype == lucid.float32
        out = layer(lucid.randn(2, 4).to(lucid.float16))
        assert out.dtype == lucid.float16


class TestConv2d:
    def _pair(self, device: str) -> tuple[nn.Conv2d, SVDConv2d]:
        lucid.manual_seed(0)
        base = nn.Conv2d(2, 3, 3, padding=1).to(device)
        layer = SVDConv2d(base, "a", _config(init_lora_weights=False), r=2, lora_alpha=4)
        return base, layer

    def test_shapes_follow_the_kernel(self, device: str) -> None:
        _, layer = self._pair(device)
        assert tuple(layer.lora_A["a"].shape) == (2, 2 * 3 * 3)
        assert tuple(layer.lora_B["a"].shape) == (3, 2)
        assert tuple(layer.get_delta_weight("a").shape) == (3, 2, 3, 3)

    def test_the_default_start_is_the_base(self, device: str) -> None:
        base = nn.Conv2d(2, 3, 3).to(device)
        layer = SVDConv2d(base, "a", _config(), r=2)
        x = lucid.randn(1, 2, 6, 6, device=device)
        assert _same(layer(x), base(x))

    def test_the_update_is_a_convolution_with_the_delta_kernel(self, device: str) -> None:
        base, layer = self._pair(device)
        x = lucid.randn(2, 2, 6, 6, device=device)
        delta = layer.get_delta_weight("a")
        expected = nn.functional.conv2d(x, base.weight + delta, base.bias, padding=1)
        assert _close(layer(x), expected, atol=1e-4)

    def test_merging_matches(self, device: str) -> None:
        _, layer = self._pair(device)
        x = lucid.randn(2, 2, 6, 6, device=device)
        with lucid.no_grad():
            adapted = layer(x)
            layer.merge()
            assert _close(layer(x), adapted, atol=1e-4)
            layer.unmerge()
            assert _close(layer(x), adapted, atol=1e-4)

    def test_a_grouped_convolution_keeps_its_per_group_kernel(self) -> None:
        base = nn.Conv2d(4, 4, 3, groups=2)
        layer = SVDConv2d(base, "a", _config(init_lora_weights=False), r=2)
        assert tuple(layer.lora_A["a"].shape) == (2, 2 * 9)
        x = lucid.randn(1, 4, 5, 5)
        assert _close(
            layer(x),
            nn.functional.conv2d(x, base.weight + layer.get_delta_weight("a"), base.bias, groups=2),
            atol=1e-4,
        )


# ── the rank allocator ───────────────────────────────────────────────────────


def _trainable(device: str = "cpu", **kw: object) -> AdaLoraModel:
    peft = get_peft_model(_mlp(device), _config(**kw))
    assert isinstance(peft.base_model, AdaLoraModel)
    return peft.base_model


class TestSchedule:
    def test_the_budget_follows_the_cubic_schedule(self) -> None:
        tuner = _trainable(tinit=1, tfinal=2, deltaT=2)
        allocator = tuner.rankallocator
        assert (allocator.init_bgt, allocator.target_bgt) == (8, 4)
        got = {step: tuple(allocator.budget_schedule(step)) for step in (0, 1, 2, 3, 8, 9, 10)}
        assert got == {
            0: (8, False),  # warm-up: the full budget, no masking
            1: (8, False),
            2: (6, True),  # int(4 * (6/7)^3 + 4), on a deltaT step
            3: (5, False),  # int(4 * (5/7)^3 + 4), off a deltaT step
            8: (4, True),  # total_step - tfinal: the target
            9: (4, True),  # the final phase masks every step
            10: (4, True),
        }

    def test_the_budget_never_rises(self) -> None:
        allocator = _trainable(tinit=2, tfinal=3).rankallocator
        budgets = [allocator.budget_schedule(step).triplets for step in range(11)]
        assert budgets == sorted(budgets, reverse=True) and budgets[0] == 8 and budgets[-1] == 4

    def test_the_final_step_is_total_minus_tfinal(self) -> None:
        assert _trainable(tfinal=3).rankallocator.final_step == 7

    def test_set_total_step_moves_the_schedule(self) -> None:
        allocator = _trainable().rankallocator
        allocator.set_total_step(20)
        assert allocator.final_step == 20 and allocator.peft_config.total_step == 20


class TestImportance:
    def test_the_score_of_a_unit_triplet(self) -> None:
        tuner = _trainable()
        _unit_adapter(tuner.model)
        _set_grads(tuner.model, singular={"0": [1.0, 2.0, 3.0, 4.0], "2": [1.0, 1.0, 1.0, 1.0]})
        allocator = tuner.rankallocator
        allocator.update_ipt(tuner.model)
        score = allocator._sensitivity[("0", "lora_E")].score
        for i, grad in enumerate((1.0, 2.0, 3.0, 4.0)):
            mean = 0.15 * grad
            want = mean * (0.15 * abs(grad - mean))
            assert abs(float(score[i, 0].item()) - want) < 1e-6

    def test_the_averages_accumulate_over_steps(self) -> None:
        tuner = _trainable()
        _unit_adapter(tuner.model)
        allocator = tuner.rankallocator
        mean = spread = 0.0
        for grad in (2.0, 1.0):
            _set_grads(tuner.model, singular={"0": [grad] * 4, "2": [grad] * 4})
            allocator.update_ipt(tuner.model)
            mean = 0.85 * mean + 0.15 * grad
            spread = 0.85 * spread + 0.15 * abs(grad - mean)
        score = allocator._sensitivity[("2", "lora_E")].score
        assert abs(float(score[0, 0].item()) - mean * spread) < 1e-6
        allocator.reset_ipt()
        assert not allocator._sensitivity

    def test_allocating_before_anything_was_recorded_is_named(self) -> None:
        tuner = _trainable(tfinal=2)
        with pytest.raises(RuntimeError, match="no importance has been recorded"):
            tuner.update_and_allocate(8)

    def test_a_missing_gradient_is_named(self) -> None:
        tuner = _trainable()
        with pytest.raises(RuntimeError, match="after loss.backward"):
            tuner.update_and_allocate(1)


class TestMasking:
    def _allocated(self, grads: dict[str, list[float]], **kw: object) -> AdaLoraModel:
        tuner = _trainable(**kw)
        _unit_adapter(tuner.model)
        _set_grads(tuner.model, singular=grads)
        return tuner

    def test_the_least_important_triplets_are_masked_down_to_the_budget(self, device: str) -> None:
        tuner = _trainable(device)
        _unit_adapter(tuner.model)
        _set_grads(tuner.model, singular={"0": [1.0, 2.0, 3.0, 4.0], "2": [3.0, 4.0, 5.0, 6.0]})
        allocator = tuner.rankallocator
        allocator.update_ipt(tuner.model)
        pattern = allocator.mask_to_budget(tuner.model, 4)
        # Importances are the squares of the gradients here: 1 4 9 16 | 9 16 25 36; the four
        # lowest (1 4 9 9) go.
        assert pattern == {
            "0.lora_E": [False, False, False, True],
            "2.lora_E": [False, True, True, True],
        }
        e0 = [float(v) for v in _layer(tuner.model, "0").lora_E["default"].flatten().tolist()]
        e2 = [float(v) for v in _layer(tuner.model, "2").lora_E["default"].flatten().tolist()]
        assert e0 == [0.0, 0.0, 0.0, 1.0] and e2 == [0.0, 1.0, 1.0, 1.0]

    def test_a_budget_that_drops_nothing_masks_nothing(self) -> None:
        tuner = self._allocated({"0": [1.0, 2.0, 3.0, 4.0], "2": [1.0, 2.0, 3.0, 4.0]})
        allocator = tuner.rankallocator
        allocator.update_ipt(tuner.model)
        pattern = allocator.mask_to_budget(tuner.model, 8)
        assert all(all(flags) for flags in pattern.values())
        assert allocator.mask_to_budget(tuner.model, 100) == pattern

    def test_a_budget_of_zero_masks_every_triplet(self) -> None:
        tuner = self._allocated({"0": [1.0, 2.0, 3.0, 4.0], "2": [1.0, 2.0, 3.0, 4.0]})
        allocator = tuner.rankallocator
        allocator.update_ipt(tuner.model)
        assert not any(any(flags) for flags in allocator.mask_to_budget(tuner.model, 0).values())

    def test_the_schedule_is_run_by_update_and_allocate(self) -> None:
        tuner = self._allocated(
            {"0": [1.0, 2.0, 3.0, 4.0], "2": [3.0, 4.0, 5.0, 6.0]}, tinit=1, tfinal=2
        )
        config = tuner.peft_config["default"]
        assert isinstance(config, AdaLoraConfig)
        tuner.update_and_allocate(1)  # warm-up: recorded, nothing masked
        assert config.rank_pattern is None
        tuner.update_and_allocate(2)  # budget 6 of 8: two triplets masked
        assert config.rank_pattern is not None
        assert sum(sum(flags) for flags in config.rank_pattern.values()) == 6
        tuner.update_and_allocate(8)  # total_step - tfinal: the target, 4 of 8
        assert sum(sum(flags) for flags in config.rank_pattern.values()) == 4
        assert not tuner.rankallocator._sensitivity

    def test_the_final_phase_keeps_the_dropped_triplets_at_zero(self) -> None:
        tuner = self._allocated(
            {"0": [1.0, 2.0, 3.0, 4.0], "2": [3.0, 4.0, 5.0, 6.0]}, tinit=1, tfinal=2
        )
        tuner.update_and_allocate(1)
        tuner.update_and_allocate(8)
        layer = _layer(tuner.model, "0")
        with lucid.no_grad():  # an optimizer step that revives them
            layer.lora_E["default"].copy_(lucid.ones(4, 1))
        tuner.update_and_allocate(9)
        assert [float(v) for v in layer.lora_E["default"].flatten().tolist()] == [0, 0, 0, 1]

    def test_mask_using_a_pattern_skips_a_layer_already_pruned(self) -> None:
        tuner = self._allocated({"0": [1.0, 2.0, 3.0, 4.0], "2": [3.0, 4.0, 5.0, 6.0]})
        layer = _layer(tuner.model, "0")
        layer.prune("default", [3])
        tuner.rankallocator.mask_using_rank_pattern(
            tuner.model, {"0.lora_E.default": [False, False, False, True]}
        )
        assert float(layer.lora_E["default"].item()) == 1.0

    def test_a_real_training_step_updates_and_allocates(self, device: str) -> None:
        peft = get_peft_model(
            _mlp(device), _config(init_lora_weights=False, total_step=4, tfinal=1)
        )
        tuner = peft.base_model
        assert isinstance(tuner, AdaLoraModel)
        x = lucid.randn(8, 6, device=device)
        for step in range(1, 5):
            tuner(x).sum().backward()
            tuner.update_and_allocate(step)
            for p in peft.parameters():
                p.grad = None
        config = tuner.peft_config["default"]
        assert isinstance(config, AdaLoraConfig) and config.rank_pattern is not None
        kept = sum(sum(flags) for flags in config.rank_pattern.values())
        assert kept == tuner.rankallocator.target_bgt

    def test_updating_without_a_trainable_adapter_is_refused(self) -> None:
        net = _mlp()
        tuner = AdaLoraModel(net, _config(inference_mode=True))
        assert tuner.trainable_adapter_name is None
        with pytest.raises(RuntimeError, match="inference_mode"):
            tuner.update_and_allocate(1)
        with pytest.raises(RuntimeError, match="inference_mode"):
            tuner.rankallocator

    def test_a_standalone_allocator_names_its_adapter(self) -> None:
        net = _mlp()
        tuner = AdaLoraModel(net, _config())
        config = tuner.peft_config["default"]
        assert isinstance(config, AdaLoraConfig)
        with pytest.raises(ValueError, match="no AdaLoRA layer"):
            RankAllocator(net, config, "absent")


# ── the loss penalty ─────────────────────────────────────────────────────────


@dataclass
class _Out:
    logits: lucid.Tensor
    loss: lucid.Tensor | None = None


@dataclass(frozen=True)
class _FrozenOut:
    logits: lucid.Tensor
    loss: lucid.Tensor | None = None


class _Plain:
    def __init__(self, logits: lucid.Tensor, loss: lucid.Tensor) -> None:
        self.logits = logits
        self.loss = loss


class _Net(nn.Module):
    """A two-layer net returning an output object that carries a loss (or not)."""

    def __init__(self, kind: type = _Out, with_loss: bool = True) -> None:
        super().__init__()
        lucid.manual_seed(0)
        self.kind, self.with_loss = kind, with_loss
        self.a = nn.Linear(6, 8)
        self.b = nn.Linear(8, 4)

    def forward(self, x: lucid.Tensor) -> object:
        logits = self.b(nn.functional.relu(self.a(x)))
        if not self.with_loss:
            return self.kind(logits)
        return self.kind(logits, (logits**2).mean())


def _penalty(tuner: AdaLoraModel) -> lucid.Tensor:
    flat = [p for _, layer in _tuner_layers(tuner.model) for p in layer.orthogonality("default")]  # type: ignore[attr-defined]
    return sum(flat[1:], flat[0]) / len(flat)


class TestPenalty:
    @pytest.mark.parametrize("kind", [_Out, _FrozenOut, _Plain])
    def test_the_loss_gains_the_weighted_mean_departure(self, kind: type, device: str) -> None:
        net = _Net(kind).to(device)
        x = lucid.randn(5, 6, device=device)
        bare = net(x).loss
        tuner = AdaLoraModel(
            net, _config(target_modules=["a", "b"], init_lora_weights=False, orth_reg_weight=0.5)
        )
        out = tuner(x)
        assert isinstance(out, kind)
        plain = net(x).loss  # the model's own loss with the adapter in
        assert _close(out.loss, plain + 0.5 * _penalty(tuner), atol=1e-4)
        assert float(out.loss.item()) > float(plain.item())
        assert float(bare.item()) > 0

    def test_the_penalty_is_differentiable_through_the_loss(self) -> None:
        net = _Net()
        tuner = AdaLoraModel(net, _config(target_modules=["a", "b"], init_lora_weights=False))
        tuner(lucid.randn(5, 6)).loss.backward()
        layer = _layer(net, "a")
        assert layer.lora_A["default"].grad is not None
        assert float(layer.lora_A["default"].grad.abs().sum().item()) > 0

    def test_a_zero_weight_adds_nothing(self) -> None:
        net = _Net()
        x = lucid.randn(5, 6)
        tuner = AdaLoraModel(
            net, _config(target_modules=["a", "b"], init_lora_weights=False, orth_reg_weight=0.0)
        )
        assert _same(tuner(x).loss, net(x).loss)

    def test_an_output_without_a_loss_is_left_alone(self) -> None:
        net = _Net(with_loss=False)
        x = lucid.randn(5, 6)
        tuner = AdaLoraModel(net, _config(target_modules=["a", "b"], init_lora_weights=False))
        out = tuner(x)
        assert out.loss is None and _same(out.logits, net(x).logits)

    def test_a_tensor_output_passes_through(self) -> None:
        tuner = AdaLoraModel(_mlp(), _config())
        assert tuner(lucid.randn(3, 6)).shape == (3, 4)

    def test_only_the_trainable_adapter_is_penalised(self) -> None:
        net = _Net()
        tuner = AdaLoraModel(net, _config(target_modules=["a", "b"], init_lora_weights=False))
        tuner.add_adapter("other", _config(target_modules=["a"], inference_mode=True))
        x = lucid.randn(5, 6)
        assert tuner.trainable_adapter_name == "default"
        out = tuner(x)
        assert _close(out.loss, net(x).loss + 0.5 * _penalty(tuner), atol=1e-4)

    @pytest.mark.xfail(
        strict=True,
        raises=AssertionError,
        reason="LCD-714b: PeftModel.forward still calls the inner model; the routing through the "
        "tuner is in the patch held for LCD-755 (remove this marker when it lands)",
    )
    def test_a_peft_model_applies_it_too(self) -> None:
        net = _Net()
        peft = get_peft_model(net, _config(target_modules=["a", "b"], init_lora_weights=False))
        x = lucid.randn(5, 6)
        tuner = peft.base_model
        assert isinstance(tuner, AdaLoraModel)
        assert _close(peft(x).loss, tuner(x).loss)


# ── the adapter file ─────────────────────────────────────────────────────────


def _pruned(device: str = "cpu") -> PeftModel:
    """A trained-looking AdaLoRA model whose allocator kept 4 of 8 triplets: 1 in layer 0, 3 in layer 2."""
    peft = get_peft_model(
        _mlp(device), _config(init_lora_weights=False, tinit=1, tfinal=2, total_step=10)
    )
    tuner = peft.base_model
    assert isinstance(tuner, AdaLoraModel)
    _unit_adapter(tuner.model)
    _set_grads(tuner.model, singular={"0": [1.0, 2.0, 3.0, 4.0], "2": [3.0, 4.0, 5.0, 6.0]})
    tuner.update_and_allocate(1)  # records the importances; the warm-up masks nothing
    tuner.update_and_allocate(8)
    return peft


class TestFile:
    def test_the_file_drops_the_pruned_triplets_and_records_the_pattern(
        self, tmp_path: Path
    ) -> None:
        peft = _pruned()
        peft.save_pretrained(tmp_path)
        saved = load_safetensors(str(tmp_path / "adapter_model.safetensors"))
        shapes = {k: tuple(v.shape) for k, v in saved.items() if isinstance(v, lucid.Tensor)}
        assert shapes == {
            "base_model.model.0.lora_A": (1, 6),
            "base_model.model.0.lora_B": (8, 1),
            "base_model.model.0.lora_E": (1, 1),
            "base_model.model.2.lora_A": (3, 8),
            "base_model.model.2.lora_B": (4, 3),
            "base_model.model.2.lora_E": (3, 1),
        }
        config = json.loads((tmp_path / "adapter_config.json").read_text())
        assert config["rank_pattern"] == {
            "0.lora_E": [False, False, False, True],
            "2.lora_E": [False, True, True, True],
        }
        assert config["peft_type"] == "ADALORA" and config["total_step"] == 10

    def test_the_live_model_is_not_pruned_by_saving(self, tmp_path: Path) -> None:
        peft = _pruned()
        peft.save_pretrained(tmp_path)
        assert dict(_layer(peft.base_model.model, "0").r) == {"default": 4}

    def test_a_saved_adapter_reloads_into_resized_layers_with_the_same_logits(
        self, device: str, tmp_path: Path
    ) -> None:
        trained = _pruned(device)
        x = lucid.randn(5, 6, device=device)
        trained.eval()
        with lucid.no_grad():
            expected = trained(x)
        trained.save_pretrained(tmp_path)
        loaded = PeftModel.from_pretrained(_mlp(device), tmp_path)
        assert dict(_layer(loaded.base_model.model, "0").r) == {"default": 1}
        assert dict(_layer(loaded.base_model.model, "2").r) == {"default": 3}
        with lucid.no_grad():
            assert _close(loaded(x), expected, atol=1e-5)
        # The scale of what is kept is the one training began with.
        assert float(_layer(loaded.base_model.model, "0").ranknum["default"].item()) == 4.0

    def test_saving_a_loaded_pruned_adapter_writes_the_same_file(self, tmp_path: Path) -> None:
        _pruned().save_pretrained(tmp_path / "one")
        loaded = PeftModel.from_pretrained(_mlp(), tmp_path / "one")
        loaded.save_pretrained(tmp_path / "two")
        first = load_safetensors(str(tmp_path / "one" / "adapter_model.safetensors"))
        second = load_safetensors(str(tmp_path / "two" / "adapter_model.safetensors"))
        assert sorted(first) == sorted(second)
        for key in first:
            assert _same(first[key], second[key])  # type: ignore[arg-type]

    def test_an_adapter_pruned_to_nothing_round_trips(self, tmp_path: Path) -> None:
        peft = _pruned()
        _layer(peft.base_model.model, "0").prune("default", [])
        config = peft.peft_config["default"]
        assert isinstance(config, AdaLoraConfig) and config.rank_pattern is not None
        config.rank_pattern["0.lora_E"] = [False] * 4  # the pattern follows the layer
        peft.save_pretrained(tmp_path)
        loaded = PeftModel.from_pretrained(_mlp(), tmp_path)
        assert dict(_layer(loaded.base_model.model, "0").r) == {"default": 0}
        x = lucid.randn(3, 6)
        with lucid.no_grad():
            assert _close(loaded(x), peft(x), atol=1e-5)

    def test_a_pruned_adapter_trains_on_after_loading(self, tmp_path: Path) -> None:
        _pruned().save_pretrained(tmp_path)
        loaded = PeftModel.from_pretrained(_mlp(), tmp_path, is_trainable=True)
        tuner = loaded.base_model
        assert isinstance(tuner, AdaLoraModel)
        loaded(lucid.randn(4, 6)).sum().backward()
        tuner.update_and_allocate(9)  # the final phase: a pattern longer than the layer is skipped
        assert tuner.rankallocator.init_bgt == 4

    def test_a_reference_written_pruned_adapter_loads(self, tmp_path: Path) -> None:
        lucid.manual_seed(5)
        tensors = {
            "base_model.model.0.lora_A": lucid.randn(2, 6),
            "base_model.model.0.lora_B": lucid.randn(8, 2),
            "base_model.model.0.lora_E": lucid.randn(2, 1),
            "base_model.model.2.lora_A": lucid.randn(4, 8),
            "base_model.model.2.lora_B": lucid.randn(4, 4),
            "base_model.model.2.lora_E": lucid.randn(4, 1),
        }
        save_safetensors(
            tensors, str(tmp_path / "adapter_model.safetensors"), metadata={"format": "pt"}
        )
        config = _config(inference_mode=True, lora_alpha=8, init_r=4).to_dict()
        # A pattern as the reference wrote it, the adapter's name still in the keys of one entry.
        config["rank_pattern"] = {
            "0.lora_E": [True, False, True, False],
            "2.lora_E.default": [True, True, True, True],
        }
        (tmp_path / "adapter_config.json").write_text(json.dumps(config))
        loaded = PeftModel.from_pretrained(_mlp(), tmp_path)
        saved_config = loaded.peft_config["default"]
        assert isinstance(saved_config, AdaLoraConfig)
        assert saved_config.rank_pattern == {
            "0.lora_E": [True, False, True, False],
            "2.lora_E": [True, True, True, True],
        }
        x = lucid.randn(3, 6)
        net = _mlp()
        w0 = net[0].weight + (
            tensors["base_model.model.0.lora_B"]
            @ (tensors["base_model.model.0.lora_A"] * tensors["base_model.model.0.lora_E"])
        ) * (8 / (4 + 1e-5))
        w2 = net[2].weight + (
            tensors["base_model.model.2.lora_B"]
            @ (tensors["base_model.model.2.lora_A"] * tensors["base_model.model.2.lora_E"])
        ) * (8 / (4 + 1e-5))
        expected = nn.functional.relu(x @ w0.mT + net[0].bias) @ w2.mT + net[2].bias
        with lucid.no_grad():
            assert _close(loaded(x), expected, atol=1e-4)

    def test_tensors_that_disagree_on_the_rank_are_refused_with_the_model_untouched(
        self, tmp_path: Path
    ) -> None:
        peft = get_peft_model(_mlp(), _config(init_lora_weights=False))
        state = get_peft_model_state_dict(peft)
        bad = {**state, "base_model.model.0.lora_B": lucid.zeros(8, 3)}
        from lucid.peft import set_peft_model_state_dict

        with pytest.raises(ValueError, match="disagree on the rank"):
            set_peft_model_state_dict(peft, bad)
        assert dict(_layer(peft.base_model.model, "0").r) == {"default": 4}

    def test_resize_modules_by_rank_pattern_drops_the_pruned_triplets_for_good(self) -> None:
        peft = _pruned()
        tuner = peft.base_model
        assert isinstance(tuner, AdaLoraModel)
        config = tuner.peft_config["default"]
        assert isinstance(config, AdaLoraConfig) and config.rank_pattern is not None
        x = lucid.randn(4, 6)
        with lucid.no_grad():
            before = peft(x)
        tuner.resize_modules_by_rank_pattern(config.rank_pattern, "default")
        assert dict(_layer(tuner.model, "0").r) == {"default": 1}
        assert dict(_layer(tuner.model, "2").r) == {"default": 3}
        with lucid.no_grad():
            assert _close(peft(x), before, atol=1e-5)
        tuner.resize_modules_by_rank_pattern(config.rank_pattern, "default")  # already as small

    def test_resize_refuses_a_pattern_that_does_not_fit(self) -> None:
        tuner = _trainable()
        with pytest.raises(ValueError, match="names layers the model does not have"):
            tuner.resize_modules_by_rank_pattern({"9.lora_E": [True]}, "default")
        with pytest.raises(ValueError, match="flags"):
            tuner.resize_modules_by_rank_pattern({"0.lora_E": [True, False]}, "default")

    def test_a_pattern_that_does_not_fit_the_layer_is_refused_on_save(self, tmp_path: Path) -> None:
        peft = get_peft_model(_mlp(), _config(rank_pattern={"0.lora_E": [True, False]}))
        with pytest.raises(ValueError, match="flags"):
            peft.save_pretrained(tmp_path)


# ── the tuner ────────────────────────────────────────────────────────────────


class TestTuner:
    def test_the_layer_types_follow_the_module(self) -> None:
        net = nn.Sequential(nn.Linear(4, 4), nn.Conv2d(2, 2, 1))
        AdaLoraModel(net, _config(target_modules=["0", "1"]))
        assert isinstance(net[0], SVDLinear) and isinstance(net[1], SVDConv2d)

    def test_a_module_with_no_layer_is_refused_and_nothing_changes(self) -> None:
        net = nn.Sequential(nn.Linear(4, 4), nn.Embedding(5, 4))
        with pytest.raises(ValueError, match="not supported"):
            AdaLoraModel(net, _config(target_modules=["0", "1"]))
        assert type(net[0]) is nn.Linear

    def test_only_one_adapter_may_train(self) -> None:
        tuner = AdaLoraModel(_mlp(), _config())
        with pytest.raises(ValueError, match="only 1 trainable adapter"):
            tuner.add_adapter("second", _config())
        assert sorted(tuner.peft_config) == ["default"]
        tuner.add_adapter("second", _config(inference_mode=True))
        assert tuner.trainable_adapter_name == "default"

    def test_a_second_trainable_adapter_is_refused_at_construction_of_its_config(self) -> None:
        tuner = AdaLoraModel(_mlp(), _config(inference_mode=True))
        tuner.add_adapter("live", _config())
        assert tuner.trainable_adapter_name == "live"
        assert tuner.rankallocator.adapter_name == "live"

    def test_weighted_adapters_are_not_supported(self) -> None:
        tuner = AdaLoraModel(_mlp(), _config())
        with pytest.raises(TypeError, match="does not support add_weighted_adapter"):
            tuner.add_weighted_adapter(["default"], [1.0], "mix")

    def test_fan_in_fan_out_is_reset_with_a_warning(self) -> None:
        with pytest.warns(UserWarning, match="fan_in_fan_out"):
            tuner = AdaLoraModel(_mlp(), _config(fan_in_fan_out=True))
        config = tuner.peft_config["default"]
        assert isinstance(config, AdaLoraConfig) and not config.fan_in_fan_out

    def test_without_names_the_lora_table_is_the_default(self) -> None:
        class Tiny(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.c_attn = nn.Linear(4, 12)
                self.config = type("C", (), {"model_type": "gpt2"})()

            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return self.c_attn(x)

        net = Tiny()
        tuner = AdaLoraModel(net, AdaLoraConfig(total_step=10))
        assert tuner.targeted_module_names == ["c_attn"]

    def test_unload_gives_the_base_back(self) -> None:
        net = _mlp()
        before = {k: v.clone() for k, v in net.state_dict().items()}
        tuner = AdaLoraModel(net, _config(init_lora_weights=False))
        assert tuner.unload() is net
        assert sorted(net.state_dict()) == sorted(before)
        assert all(_same(v, before[k]) for k, v in net.state_dict().items())

    def test_merge_and_unload_matches_the_adapted_model(self, device: str) -> None:
        net = _mlp(device)
        peft = get_peft_model(net, _config(init_lora_weights=False))
        x = lucid.randn(4, 6, device=device)
        with lucid.no_grad():
            adapted = peft(x)
            assert _close(peft.merge_and_unload()(x), adapted, atol=1e-4)

    def test_the_audit_names_are_the_reference_s(self) -> None:
        net = _mlp()
        AdaLoraModel(net, _config())
        assert sorted(k for k in net.state_dict() if "default" in k and "base" not in k) == [
            "0.lora_A.default",
            "0.lora_B.default",
            "0.lora_E.default",
            "0.ranknum.default",
            "2.lora_A.default",
            "2.lora_B.default",
            "2.lora_E.default",
            "2.ranknum.default",
        ]
