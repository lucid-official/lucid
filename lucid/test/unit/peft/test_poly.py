"""Poly / Multi-Head Routing: low-rank skills mixed per task by a learned router.

What follows from the method: ``B`` starts at zero so the adapted model starts as the base, a
batch row's update is the task-weighted mixture of the skills (checked against a per-row rewrite
in plain ops, with and without splits), routing is stochastic in training and deterministic in
evaluation, every call must name its tasks, and nothing merges.
"""

from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.models import create_model
from lucid.peft import (
    LoraConfig,
    PeftModel,
    PeftType,
    PolyConfig,
    PolyModel,
    get_peft_model,
    get_peft_model_state_dict,
)
from lucid.peft.tuners.poly import Linear, PolyLayer
from lucid.peft.tuners.poly._router import PolyRouter, Router

_LM: dict[str, object] = {
    "vocab_size": 32,
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 32,
    "max_position_embeddings": 24,
    "hidden_dropout": 0.0,
    "attention_dropout": 0.0,
}


def _config(
    r: int = 2,
    n_tasks: int = 3,
    n_skills: int = 3,
    n_splits: int = 1,
    init_weights: bool = True,
    target_modules: list[str] | None = None,
    task_type: str | None = None,
) -> PolyConfig:
    return PolyConfig(
        r=r,
        n_tasks=n_tasks,
        n_skills=n_skills,
        n_splits=n_splits,
        init_weights=init_weights,
        target_modules=target_modules or ["x"],
        task_type=task_type,
    )


def _layer(n_splits: int = 1) -> tuple[Linear, nn.Linear]:
    lucid.manual_seed(1)
    base = nn.Linear(8, 6)
    return Linear(base, "default", _config(init_weights=False, n_splits=n_splits)), base


def _lm(device: str = "cpu") -> nn.Module:
    lucid.manual_seed(0)
    return create_model("gpt2_small_lm", **_LM).to(device)


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


class TestConfig:
    def test_the_fields_are_the_references(self) -> None:
        assert set(_config().to_dict()) == {
            *("auto_mapping", "base_model_name_or_path", "exclude_modules", "inference_mode"),
            *("init_weights", "modules_to_save", "n_skills", "n_splits", "n_tasks", "peft_type"),
            *("poly_type", "r", "revision", "target_modules", "task_type"),
        }

    def test_the_defaults_are_the_references(self) -> None:
        config = PolyConfig(target_modules=["x"])
        assert (config.r, config.n_tasks, config.n_skills, config.n_splits) == (8, 1, 4, 1)
        assert config.init_weights and config.poly_type == "poly"
        assert config.peft_type == PeftType.POLY

    def test_layer_selection_is_an_extension_written_only_when_used(self) -> None:
        assert "layers_to_transform" not in _config().to_dict()
        chosen = PolyConfig(target_modules=["x"], layers_to_transform=[0], layers_pattern="h")
        assert PolyConfig.from_dict(chosen.to_dict()) == chosen

    def test_a_file_round_trips(self) -> None:
        config = _config(n_splits=2, target_modules=["c_attn"], task_type="CAUSAL_LM")
        assert PolyConfig.from_dict(config.to_dict()) == config

    @pytest.mark.parametrize("field", ["r", "n_tasks", "n_skills", "n_splits"])
    def test_counts_are_positive_integers(self, field: str) -> None:
        with pytest.raises(ValueError, match="positive"):
            PolyConfig.from_dict({field: 0, "target_modules": ["x"]})
        with pytest.raises(TypeError, match="int"):
            PolyConfig.from_dict({field: 1.5, "target_modules": ["x"]})

    def test_an_unknown_router_is_refused(self) -> None:
        with pytest.raises(ValueError, match="one of 'poly'"):
            PolyConfig.from_dict({"poly_type": "mhr", "target_modules": ["x"]})


class TestRouter:
    def _router(self, **fields: int) -> PolyRouter:
        router = Router.of(_config(**fields))
        assert isinstance(router, PolyRouter)
        router.reset()
        return router

    def test_the_logits_start_near_zero(self) -> None:
        logits = self._router().module_logits
        assert tuple(logits.shape) == (3, 3) and float(logits.abs().max().item()) <= 1e-3

    def test_evaluation_is_the_normalised_sigmoid_of_the_tasks_logits(self) -> None:
        router = self._router(n_splits=2).eval()
        with lucid.no_grad():
            router.module_logits.copy_(lucid.randn(3, 6))
        weights = router(lucid.tensor([2, 0, 2]))
        assert tuple(weights.shape) == (3, 2, 3)
        row = lucid.sigmoid(router.module_logits[2].reshape(2, 3))
        assert bool(lucid.allclose(weights[0], row / row.sum(dim=-1, keepdim=True), atol=1e-6))
        assert _same(weights[0], weights[2])
        assert bool(lucid.allclose(weights.sum(dim=-1), lucid.ones(3, 2), atol=1e-5))

    def test_training_samples_around_it_and_a_seed_repeats_the_draw(self) -> None:
        router = self._router().train()
        tasks = lucid.tensor([1, 1])
        lucid.manual_seed(5)
        first = router(tasks)
        lucid.manual_seed(5)
        again = router(tasks)
        assert _same(first, again)
        assert not _same(first[0], first[1])  # two rows of one task are two draws
        assert bool(lucid.allclose(first.sum(dim=-1), lucid.ones(2, 1), atol=1e-5))

    def test_the_sample_is_differentiable(self) -> None:
        router = self._router().train()
        router(lucid.tensor([0, 1])).sum().backward()  # the weights sum to one: grad is ~0
        assert router.module_logits.grad is not None

    def test_the_tasks_must_be_named_and_known(self) -> None:
        router = self._router().eval()
        with pytest.raises(ValueError, match="task_ids should not be None"):
            router(None)
        with pytest.raises(ValueError, match="Only 3 tasks available. Found task id = 3"):
            router(lucid.tensor([0, 3]))
        with pytest.raises(ValueError, match="must not be negative"):
            router(lucid.tensor([0, -1]))
        with pytest.raises(TypeError, match="integer"):
            router(lucid.tensor([0.0, 1.0]))


class TestLayer:
    def test_the_factors_have_the_references_shapes_and_keys(self) -> None:
        layer, _ = _layer(n_splits=2)
        assert tuple(layer.poly_lora_A["default"].shape) == (
            2,
            3,
            4,
            2,
        )  # (splits, skills, in/2, r)
        assert tuple(layer.poly_lora_B["default"].shape) == (
            2,
            3,
            2,
            3,
        )  # (splits, skills, r, out/2)
        assert tuple(layer.poly_router["default"].module_logits.shape) == (3, 6)
        assert layer.r["default"] == 2 and layer.n_skills["default"] == 3
        assert layer.n_splits["default"] == 2 and layer.poly_type["default"] == "poly"
        assert sorted(layer.state_dict()) == [
            "base_layer.bias",
            "base_layer.weight",
            "poly_lora_A.default",
            "poly_lora_B.default",
            "poly_router.default.module_logits",
        ]

    def test_b_starts_at_zero_and_the_layer_is_the_base(self) -> None:
        lucid.manual_seed(1)
        base = nn.Linear(8, 6)
        layer = Linear(base, "default", _config())
        x = lucid.randn(3, 5, 8)
        assert float(layer.poly_lora_B["default"].abs().max().item()) == 0.0
        assert _same(layer(x, task_ids=lucid.tensor([0, 1, 2])), base(x))

    def test_a_is_drawn_like_a_linear_weight(self) -> None:
        layer, _ = _layer()
        a = layer.poly_lora_A["default"]
        assert (
            float(a.abs().max().item()) <= 1 / 8**0.5 + 1e-6
        )  # kaiming_uniform(a=sqrt(5)), fan-in 8
        assert float(a.std().item()) > 0.05

    @pytest.mark.parametrize("n_splits", [1, 2])
    def test_a_row_gets_its_tasks_mixture_of_the_skills(self, n_splits: int) -> None:
        layer, base = _layer(n_splits=n_splits)
        layer.eval()
        with lucid.no_grad():
            layer.poly_router["default"].module_logits.copy_(lucid.randn(3, 3 * n_splits))
        x = lucid.randn(3, 4, 8)
        tasks = [2, 0, 2]
        out = layer(x, task_ids=lucid.tensor(tasks))
        a, b = layer.poly_lora_A["default"], layer.poly_lora_B["default"]
        logits = layer.poly_router["default"].module_logits
        for row, task in enumerate(tasks):
            m = lucid.sigmoid(logits[task].reshape(n_splits, 3))
            m = m / (m.sum(dim=-1, keepdim=True) + 1e-12)
            blocks_a, blocks_b = [], []
            for split in range(n_splits):
                mix_a, mix_b = a[split][0] * 0, b[split][0] * 0
                for skill in range(3):
                    mix_a = mix_a + m[split, skill] * a[split][skill]
                    mix_b = mix_b + m[split, skill] * b[split][skill]
                blocks_a.append(mix_a)
                blocks_b.append(mix_b)
            full_a = lucid.cat(blocks_a, dim=0)  # (in, r): splits end to end along the features
            full_b = lucid.cat(blocks_b, dim=1)  # (r, out)
            expected = base(x[row]) + x[row] @ full_a @ full_b / 2
            assert bool(lucid.allclose(out[row], expected, atol=1e-5)), row

    def test_two_tasks_route_differently_and_one_task_is_stable(self) -> None:
        layer, _ = _layer()
        layer.eval()
        with lucid.no_grad():
            layer.poly_router["default"].module_logits.copy_(lucid.randn(3, 3) * 3)
        x = lucid.randn(1, 4, 8).repeat(3, 1, 1)
        out = layer(x, task_ids=lucid.tensor([0, 1, 0]))
        assert bool(lucid.allclose(out[0], out[2], atol=1e-6))
        assert not bool(lucid.allclose(out[0], out[1], atol=1e-4))

    def test_a_two_dimensional_input_is_one_position_per_row(self) -> None:
        layer, base = _layer()
        layer.eval()
        out = layer(lucid.randn(3, 8), task_ids=lucid.tensor([0, 1, 2]))
        assert tuple(out.shape) == (3, 6)

    def test_task_ids_must_match_the_batch(self) -> None:
        layer, _ = _layer()
        with pytest.raises(ValueError, match="names 2 tasks for a batch of 3"):
            layer(lucid.randn(3, 8), task_ids=lucid.tensor([0, 1]))

    def test_n_splits_must_divide_the_layer(self) -> None:
        with pytest.raises(ValueError, match="n_splits=4 must divide the layer's output size 6"):
            Linear(nn.Linear(8, 6), "default", _config(n_splits=4))

    def test_a_disabled_layer_is_the_base_and_needs_no_tasks(self) -> None:
        layer, base = _layer()
        layer.enable_adapters(False)
        x = lucid.randn(3, 8)
        assert _same(layer(x), base(x))

    def test_it_cannot_be_merged(self) -> None:
        layer, _ = _layer()
        with pytest.raises(NotImplementedError, match="cannot be folded"):
            layer.merge()
        assert isinstance(layer, PolyLayer)


class TestTuner:
    def _peft(self, **fields: int) -> PeftModel:
        lucid.manual_seed(0)
        net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
        return get_peft_model(net, _config(target_modules=["0", "2"], **fields))

    def test_the_adapted_model_starts_as_the_base(self, device: str) -> None:
        lucid.manual_seed(0)
        net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
        x = lucid.randn(4, 8, device=device)
        expected = net(x)
        peft = get_peft_model(net, _config(target_modules=["0", "2"])).eval()
        assert _same(peft(x, task_ids=lucid.tensor([0, 1, 2, 0], device=device)), expected)

    def test_only_the_skills_and_routers_train(self) -> None:
        peft = self._peft()
        trained = {n for n, p in peft.named_parameters() if p.requires_grad}
        assert trained == {
            f"base_model.model.{i}.{table}"
            for i in ("0", "2")
            for table in (
                "poly_lora_A.default",
                "poly_lora_B.default",
                "poly_router.default.module_logits",
            )
        }
        assert isinstance(peft.base_model, PolyModel)

    def test_a_call_without_tasks_is_refused(self) -> None:
        with pytest.raises(ValueError, match="task_ids should not be None"):
            self._peft()(lucid.randn(2, 8))

    def test_the_tasks_reach_every_poly_layer_and_only_for_the_call(self) -> None:
        peft = self._peft(init_weights=False).eval()
        with lucid.no_grad():
            for name, p in peft.named_parameters():
                if "module_logits" in name:
                    p.copy_(lucid.randn(*p.shape) * 3)
        x = lucid.randn(1, 8).repeat(3, 1)
        out = peft(x, task_ids=lucid.tensor([0, 1, 0]))
        assert bool(lucid.allclose(out[0], out[2], atol=1e-6))
        assert not bool(lucid.allclose(out[0], out[1], atol=1e-4))
        for _, layer in peft.base_model.model.named_modules():
            if isinstance(layer, PolyLayer):
                assert not layer._forward_pre_hooks
        peft(lucid.randn(2, 8), task_ids=lucid.tensor([0, 1]))
        for _, layer in peft.base_model.model.named_modules():
            if isinstance(layer, PolyLayer):
                assert not layer._forward_pre_hooks

    def test_a_standalone_tuner_takes_task_ids_itself(self) -> None:
        lucid.manual_seed(0)
        net = nn.Sequential(nn.Linear(8, 4))
        poly = PolyModel(net, _config(target_modules=["0"]))
        assert tuple(poly(lucid.randn(2, 8), task_ids=lucid.tensor([0, 2])).shape) == (2, 4)

    def test_training_lowers_the_loss_and_moves_only_the_adapter(self, device: str) -> None:
        peft = get_peft_model(
            _lm(device), _config(target_modules=["c_attn"], task_type="CAUSAL_LM")
        )
        lucid.manual_seed(1)
        ids = lucid.randint(0, 32, (4, 8), device=device)
        tasks = lucid.tensor([0, 1, 2, 0], device=device)
        frozen = {n: p.clone() for n, p in peft.named_parameters() if not p.requires_grad}
        opt = optim.AdamW([p for p in peft.parameters() if p.requires_grad], lr=3e-2)
        peft.train()
        losses = []
        for _ in range(6):
            opt.zero_grad()
            loss = peft(ids, labels=ids, task_ids=tasks).loss
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        assert losses[-1] < losses[0], losses
        assert all(_same(p, frozen[n]) for n, p in peft.named_parameters() if n in frozen)

    def test_generate_carries_the_tasks(self) -> None:
        peft = get_peft_model(
            _lm(), _config(target_modules=["c_attn"], task_type="CAUSAL_LM")
        ).eval()
        ids = lucid.randint(0, 32, (2, 5))
        out = peft.generate(ids, max_new_tokens=3, do_sample=False, task_ids=lucid.tensor([0, 1]))
        assert tuple(out.shape) == (2, 8)

    def test_a_model_type_default_names_the_attention_projection(self) -> None:
        peft = get_peft_model(_lm(), PolyConfig(n_tasks=2, task_type="CAUSAL_LM"))
        assert peft.peft_config["default"].target_modules == {"c_attn"}

    def test_adapters_add_switch_and_delete(self) -> None:
        peft = self._peft()
        peft.add_adapter("other", _config(target_modules=["0"], n_tasks=2))
        peft.set_adapter("other")
        assert peft.active_adapter == "other"
        out = peft(lucid.randn(2, 8), task_ids=lucid.tensor([0, 1]))
        assert tuple(out.shape) == (2, 2)
        with pytest.warns(UserWarning, match="Setting active adapter"):
            peft.delete_adapter("other")
        assert peft.active_adapter == "default"

    def test_merging_is_refused(self) -> None:
        with pytest.raises(NotImplementedError, match="cannot be folded"):
            self._peft().merge_and_unload()

    def test_unload_gives_the_model_back(self) -> None:
        lucid.manual_seed(0)
        net = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 2))
        x = lucid.randn(2, 8)
        expected = net(x)
        peft = get_peft_model(net, _config(target_modules=["0"]))
        assert peft.unload() is net
        assert type(net[0]) is nn.Linear and _same(net(x), expected)

    def test_task_ids_are_refused_by_a_method_without_tasks(self) -> None:
        net = nn.Sequential(nn.Linear(4, 4))
        peft = get_peft_model(net, LoraConfig(r=2, target_modules=["0"]))
        with pytest.raises(ValueError, match="not per-task"):
            peft(lucid.randn(2, 4), task_ids=lucid.tensor([0, 0]))


class TestFile:
    def test_the_file_holds_the_references_keys(self) -> None:
        peft = get_peft_model(_lm(), _config(target_modules=["c_attn"], task_type="CAUSAL_LM"))
        keys = sorted(get_peft_model_state_dict(peft))
        assert keys == sorted(
            f"base_model.model.transformer.h.{i}.attn.c_attn.{table}"
            for i in (0, 1)
            for table in ("poly_lora_A", "poly_lora_B", "poly_router.module_logits")
        )

    def test_a_saved_adapter_reloads_to_the_same_logits(self, tmp_path: Path) -> None:
        config = _config(target_modules=["c_attn"], task_type="CAUSAL_LM", init_weights=False)
        trained = get_peft_model(_lm(), config).eval()
        with lucid.no_grad():
            for name, p in trained.named_parameters():
                if "poly_router" in name:
                    p.copy_(lucid.randn(*p.shape))
        ids = lucid.randint(0, 32, (3, 5))
        tasks = lucid.tensor([0, 1, 2])
        expected = trained(ids, task_ids=tasks).logits
        trained.save_pretrained(tmp_path)
        loaded = PeftModel.from_pretrained(_lm(), tmp_path)
        assert type(loaded).__name__ == "PeftModelForCausalLM"
        assert bool(lucid.allclose(loaded(ids, task_ids=tasks).logits, expected, atol=1e-5))
        assert loaded.peft_config["default"].n_tasks == 3
