"""X-LoRA: LoRA experts loaded by name and weighed per token by a classifier on the hidden states.

What follows from the method: the classifier alone trains and alone is saved, the experts' updates
are mixed by the weights it predicts (checked against a model whose weights carry the same mixture
folded in), a scalings pass at a fixed value precedes every real pass, the weights come from the
whole sequence so a cached step is refused, and nothing merges.
"""

import warnings
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.models import create_model
from lucid.peft import (
    IA3Config,
    LoraConfig,
    PeftModel,
    PeftType,
    XLoraConfig,
    XLoraModel,
    get_peft_model,
    get_peft_model_state_dict,
    inject_adapter_in_model,
    set_peft_model_state_dict,
)
from lucid.peft.tuners._base import _tuner_layers
from lucid.peft.tuners.lora import Linear
from lucid.peft.tuners.xlora import XLoraClassifier

_ARCH: dict[str, object] = {
    "vocab_size": 32,
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 32,
    "max_position_embeddings": 24,
    "hidden_dropout": 0.0,
    "attention_dropout": 0.0,
    # Wide enough weights that the logits, and so the loss, respond to the mixture.
    "initializer_range": 0.3,
}
_EXPERTS = ("a", "b", "c")


def _lm(device: str = "cpu") -> nn.Module:
    lucid.manual_seed(0)
    return create_model("gpt2_small_lm", **_ARCH).to(device)


def _ids(device: str = "cpu") -> lucid.Tensor:
    lucid.manual_seed(1)
    return lucid.randint(0, 32, (2, 6), device=device)


def _experts(root: Path, names: tuple[str, ...] = _EXPERTS) -> dict[str, str]:
    """Save one LoRA adapter per name (each its own random draw, so its update is not zero)."""
    paths: dict[str, str] = {}
    for seed, name in enumerate(names):
        base = _lm()  # seeds the generator, so the expert's own draw comes after it
        lucid.manual_seed(100 + seed)
        peft = get_peft_model(
            base, LoraConfig(r=2, target_modules=["c_attn"], init_lora_weights=False)
        )
        peft.save_pretrained(root / name)
        paths[name] = str(root / name)
    return paths


def _xlora(root: Path, device: str = "cpu", **fields: object) -> PeftModel:
    config = XLoraConfig.from_dict(
        {"adapters": _experts(root), "task_type": "CAUSAL_LM", "xlora_dropout_p": 0.0, **fields}
    )
    return get_peft_model(_lm(device), config)


def _classifier(peft: PeftModel) -> XLoraClassifier:
    host = peft.base_model
    assert isinstance(host, XLoraModel)
    return host.internal_xlora_classifier


def _constant_scalings(peft: PeftModel, logits: list[float]) -> None:
    """Make the classifier ignore the hidden states and predict ``logits`` for every token."""
    last = _classifier(peft).layers[-1]
    with lucid.no_grad():
        last.weight.copy_(lucid.zeros_like(last.weight))
        last.bias.copy_(lucid.tensor(logits))


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


class TestConfig:
    def test_the_fields_are_the_references(self) -> None:
        assert set(XLoraConfig().to_dict()) == {
            *("adapters", "auto_mapping", "base_model_name_or_path", "enable_softmax"),
            *("enable_softmax_topk", "global_scaling_weight", "hidden_size", "inference_mode"),
            *("layerwise_scalings", "peft_type", "revision", "scaling_pass_value"),
            *("softmax_temperature", "task_type", "top_k_lora", "use_trainable_adapters"),
            *("xlora_depth", "xlora_dropout_p", "xlora_size"),
        }

    def test_the_defaults_are_the_references(self) -> None:
        config = XLoraConfig()
        assert config.peft_type == PeftType.XLORA and config.hidden_size is None
        assert config.enable_softmax and not config.enable_softmax_topk
        assert not config.layerwise_scalings and not config.use_trainable_adapters
        assert (config.xlora_depth, config.xlora_size, config.xlora_dropout_p) == (1, 2048, 0.2)
        assert (config.softmax_temperature, config.top_k_lora) == (1.0, None)
        assert (config.scaling_pass_value, config.global_scaling_weight) == (0.0, 1.0)

    def test_a_file_round_trips(self) -> None:
        config = XLoraConfig(hidden_size=8, adapters={"a": "/x/a", "b": "/x/b"}, top_k_lora=1)
        assert XLoraConfig.from_dict(config.to_dict()) == config

    def test_a_path_is_kept_as_its_string(self) -> None:
        assert XLoraConfig.from_dict({"adapters": {"a": Path("/x/a")}}).adapters == {"a": "/x/a"}

    def test_bad_values_are_refused(self) -> None:
        for bad in ({"xlora_depth": 0}, {"top_k_lora": 0}, {"xlora_size": -1}):
            with pytest.raises(ValueError, match="positive"):
                XLoraConfig.from_dict(bad)
        with pytest.raises(ValueError, match="positive"):
            XLoraConfig(softmax_temperature=0.0)
        with pytest.raises(ValueError, match="lie in"):
            XLoraConfig(xlora_dropout_p=1.0)
        with pytest.raises(TypeError, match="dict"):
            XLoraConfig.from_dict({"adapters": ["a"]})

    def test_conflicting_softmax_settings_warn_as_the_reference_does(self) -> None:
        with pytest.warns(UserWarning, match="top_k_lora. is not set"):
            XLoraConfig(enable_softmax_topk=True, enable_softmax=False)
        with pytest.warns(UserWarning, match="worse performance"):
            XLoraConfig(enable_softmax_topk=True, top_k_lora=1)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            XLoraConfig(enable_softmax_topk=True, enable_softmax=False, top_k_lora=1)


class TestClassifier:
    def _classifier(self, **fields: object) -> XLoraClassifier:
        config = XLoraConfig.from_dict({"hidden_size": 8, "xlora_size": 6, **fields})
        return XLoraClassifier(config, n_classes=3, n_layers=2, dtype=lucid.float32, device="cpu")

    def test_depth_one_is_a_single_linear_layer(self) -> None:
        classifier = self._classifier()
        assert sorted(classifier.state_dict()) == ["layers.0.bias", "layers.0.weight"]
        assert tuple(classifier.layers[0].weight.shape) == (3, 8)

    def test_deeper_stacks_have_the_references_layer_indices(self) -> None:
        with_dropout = self._classifier(xlora_depth=3, xlora_dropout_p=0.5)
        # Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear
        assert sorted(with_dropout.state_dict()) == [
            f"layers.{i}.{p}" for i in (0, 3, 6) for p in ("bias", "weight")
        ]
        without = self._classifier(xlora_depth=3, xlora_dropout_p=0.0)
        assert sorted(without.state_dict()) == [
            f"layers.{i}.{p}" for i in (0, 2, 4) for p in ("bias", "weight")
        ]
        assert [tuple(without.layers[i].weight.shape) for i in (0, 2, 4)] == [
            (6, 8),
            (6, 6),
            (3, 6),
        ]

    def test_the_output_is_a_softmax_over_the_experts_of_each_layer(self) -> None:
        scalings = self._classifier()(lucid.randn(2, 5, 8))
        assert tuple(scalings.shape) == (2, 5, 2, 3)
        assert bool(lucid.allclose(scalings.sum(dim=-1), lucid.ones(2, 5, 2), atol=1e-6))
        assert _same(scalings[:, :, 0], scalings[:, :, 1])  # one set shared by every layer

    def test_layerwise_scalings_give_each_layer_its_own(self) -> None:
        classifier = self._classifier(layerwise_scalings=True)
        assert tuple(classifier.layers[0].weight.shape) == (6, 8)  # n_classes * n_layers
        scalings = classifier(lucid.randn(1, 4, 8))
        assert not _same(scalings[:, :, 0], scalings[:, :, 1])

    def test_without_the_softmax_the_logits_come_out(self) -> None:
        raw = self._classifier(enable_softmax=False)(lucid.randn(1, 3, 8))
        assert float(raw.sum(dim=-1).abs().max().item()) != 1.0

    def test_a_low_temperature_sharpens(self) -> None:
        hidden = lucid.randn(1, 4, 8)
        sharp, flat = self._classifier(softmax_temperature=0.1), self._classifier(
            softmax_temperature=10.0
        )
        flat.load_state_dict(sharp.state_dict())
        assert float(sharp(hidden).max().item()) > float(flat(hidden).max().item())

    def test_the_scalings_pass_value_fills_the_dummy(self) -> None:
        dummy = self._classifier(scaling_pass_value=0.25).dummy_scalings(2, 3, "cpu")
        assert tuple(dummy.shape) == (2, 3, 2, 3) and float(dummy.min().item()) == 0.25

    def test_logging_records_each_call_until_stopped(self) -> None:
        classifier = self._classifier()
        classifier(lucid.randn(1, 2, 8))
        assert classifier.log_scalings == []
        classifier.scalings_logging = True
        classifier(lucid.randn(1, 2, 8))
        classifier(lucid.randn(1, 4, 8))
        assert [int(s.shape[1]) for s in classifier.log_scalings] == [2, 4]


class TestMixture:
    def _expected(self, tmp_path: Path, weights: list[float]) -> lucid.Tensor:
        """The logits of a plain model whose ``c_attn`` weights carry the experts' updates, weighted."""
        peft = _xlora(tmp_path / "again", xlora_depth=1)
        model = peft.get_base_model()
        plain = _lm()
        plain_layers = dict(_tuner_layers_by_name(plain))
        with lucid.no_grad():
            for key, layer in _tuner_layers(model):
                target = plain_layers[key]
                for weight, name in zip(weights, peft.base_model.experts, strict=True):
                    target.weight.copy_(target.weight + weight * layer.get_delta_weight(name))
        return plain(_ids()).logits

    def test_the_experts_are_mixed_by_the_predicted_weights(self, tmp_path: Path) -> None:
        peft = _xlora(tmp_path / "x").eval()
        _constant_scalings(peft, [0.5, -0.5, 1.0])
        s = lucid.softmax(lucid.tensor([0.5, -0.5, 1.0]), dim=0).tolist()
        got = peft(_ids()).logits
        expected = self._expected(tmp_path, [float(w) for w in s])
        assert bool(lucid.allclose(got, expected, atol=1e-5))
        assert bool(lucid.allclose(peft.get_latest_scalings()[0, 0, 0], lucid.tensor(s), atol=1e-6))

    def test_equal_weights_of_zero_are_the_base(self, tmp_path: Path) -> None:
        peft = _xlora(tmp_path).eval()
        _constant_scalings(peft, [0.0, 0.0, 0.0])
        peft.set_global_scaling_weight(0.0)
        assert bool(lucid.allclose(peft(_ids()).logits, _lm()(_ids()).logits, atol=1e-6))

    def test_the_global_weight_scales_every_update(self, tmp_path: Path) -> None:
        peft = _xlora(tmp_path / "x").eval()
        _constant_scalings(peft, [1.0, 0.0, -1.0])
        peft.set_global_scaling_weight(0.5)
        assert peft.get_global_scaling_weight() == 0.5
        s = [float(w) for w in lucid.softmax(lucid.tensor([1.0, 0.0, -1.0]), dim=0).tolist()]
        expected = self._expected(tmp_path, [0.5 * w for w in s])
        assert bool(lucid.allclose(peft(_ids()).logits, expected, atol=1e-5))
        assert peft.peft_config["default"].global_scaling_weight == 0.5

    def test_top_k_keeps_the_heaviest_experts(self, tmp_path: Path) -> None:
        peft = _xlora(tmp_path / "x").eval()
        _constant_scalings(peft, [0.0, 2.0, 1.0])
        peft.set_topk_lora(1)
        s = lucid.softmax(lucid.tensor([0.0, 2.0, 1.0]), dim=0).tolist()
        expected = self._expected(tmp_path, [0.0, float(s[1]), 0.0])
        assert bool(lucid.allclose(peft(_ids()).logits, expected, atol=1e-5))

    def test_a_softmax_over_the_kept_experts_renormalises_them(self, tmp_path: Path) -> None:
        peft = _xlora(
            tmp_path / "x", enable_softmax=False, enable_softmax_topk=True, top_k_lora=2
        ).eval()
        _constant_scalings(peft, [3.0, 0.0, 1.0])
        kept = lucid.softmax(lucid.tensor([3.0, 1.0]), dim=0).tolist()
        expected = self._expected(tmp_path, [float(kept[0]), 0.0, float(kept[1])])
        assert bool(lucid.allclose(peft(_ids()).logits, expected, atol=1e-5))

    def test_the_scalings_come_from_a_pass_at_the_fixed_value(self, tmp_path: Path) -> None:
        peft = _xlora(tmp_path).eval()
        zero = peft(_ids()).logits
        base_features = peft.get_latest_scalings()
        peft.set_scaling_pass_value(None)
        assert peft.peft_config["default"].scaling_pass_value == pytest.approx(1 / 3)
        peft(_ids())
        # A pass with the experts mixed in sees other hidden states than the base alone does.
        assert not _same(peft.get_latest_scalings(), base_features)
        assert not _same(zero, peft(_ids()).logits)

    def test_layerwise_scalings_reach_each_layer_separately(self, tmp_path: Path) -> None:
        peft = _xlora(tmp_path, layerwise_scalings=True).eval()
        peft(_ids())
        scalings = peft.get_latest_scalings()
        assert tuple(scalings.shape) == (2, 6, 2, 3)
        assert not _same(scalings[:, :, 0], scalings[:, :, 1])

    def test_a_disabled_model_is_the_base(self, tmp_path: Path) -> None:
        peft = _xlora(tmp_path).eval()
        _constant_scalings(peft, [1.0, 1.0, 1.0])
        expected = _lm()(_ids()).logits
        assert not bool(lucid.allclose(peft(_ids()).logits, expected, atol=1e-4))
        with peft.disable_adapter():
            assert bool(lucid.allclose(peft(_ids()).logits, expected, atol=1e-6))
        assert not bool(lucid.allclose(peft(_ids()).logits, expected, atol=1e-4))

    def test_every_token_has_its_own_weights(self, tmp_path: Path) -> None:
        peft = _xlora(tmp_path, xlora_depth=2, xlora_size=8).eval()
        peft(_ids())
        scalings = peft.get_latest_scalings()
        assert not _same(scalings[:, 0], scalings[:, 3])

    def test_the_scalings_are_logged_on_request(self, tmp_path: Path) -> None:
        peft = _xlora(tmp_path).eval()
        peft(_ids())
        assert peft.get_scalings_log() == []
        peft.enable_scalings_logging()
        peft(_ids())
        peft(lucid.randint(0, 32, (1, 4)))
        buckets = peft.get_bucketed_scalings_log()
        assert sorted(buckets) == [4, 6] and buckets[6][0] == [0]
        peft.disable_scalings_logging()
        peft(_ids())
        assert len(peft.get_scalings_log()) == 2
        peft.clear_scalings_log()
        assert peft.get_scalings_log() == []


def _tuner_layers_by_name(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.endswith("c_attn")
    ]


class TestTuner:
    def test_the_classifier_alone_trains(self, tmp_path: Path) -> None:
        peft = _xlora(tmp_path, xlora_depth=2, xlora_size=8)
        trained = {n for n, p in peft.named_parameters() if p.requires_grad}
        assert trained == {
            f"base_model.internal_xlora_classifier.layers.{i}.{p}"
            for i in (0, 2)
            for p in ("weight", "bias")
        }
        assert isinstance(peft.base_model, XLoraModel)

    def test_the_experts_train_too_when_asked(self, tmp_path: Path) -> None:
        peft = _xlora(tmp_path, use_trainable_adapters=True)
        trained = {n for n, p in peft.named_parameters() if p.requires_grad}
        assert any("lora_A" in n for n in trained) and any(
            "internal_xlora_classifier" in n for n in trained
        )
        assert not any("base_layer" in n for n in trained)

    def test_the_experts_are_loaded_by_index_and_all_active(self, tmp_path: Path) -> None:
        peft = _xlora(tmp_path)
        experts = peft.base_model.lora_model
        assert sorted(experts.peft_config) == ["0", "1", "2"]
        assert experts.active_adapters == ["0", "1", "2"]
        assert peft.base_model.experts == ["0", "1", "2"]

    def test_the_hidden_size_is_read_off_the_model(self, tmp_path: Path) -> None:
        peft = _xlora(tmp_path)
        assert peft.peft_config["default"].hidden_size == 16

    def test_training_lowers_the_loss_and_moves_the_classifier_alone(
        self, tmp_path: Path, device: str
    ) -> None:
        peft = _xlora(tmp_path, device, xlora_depth=2, xlora_size=16, xlora_dropout_p=0.0)
        ids = _ids(device)
        frozen = {n: p.clone() for n, p in peft.named_parameters() if not p.requires_grad}
        opt = optim.AdamW([p for p in peft.parameters() if p.requires_grad], lr=5e-2)
        peft.train()
        losses = []
        for _ in range(8):
            opt.zero_grad()
            loss = peft(ids, labels=ids).loss
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        assert losses[-1] < losses[0], losses
        assert all(_same(p, frozen[n]) for n, p in peft.named_parameters() if n in frozen)

    def test_generate_runs_without_a_cache_and_refuses_one(self, tmp_path: Path) -> None:
        peft = _xlora(tmp_path).eval()
        out = peft.generate(_ids(), max_new_tokens=3, do_sample=False, use_cache=False)
        assert tuple(out.shape) == (2, 9)
        with pytest.raises(ValueError, match="use_cache=False"):
            peft.generate(_ids(), max_new_tokens=3, do_sample=False)

    def test_unload_gives_the_model_back(self, tmp_path: Path) -> None:
        base = _lm()
        peft = get_peft_model(base, XLoraConfig(adapters=_experts(tmp_path), task_type="CAUSAL_LM"))
        expected = _lm()(_ids()).logits
        plain = peft.unload()
        assert plain is base
        assert not any(isinstance(m, Linear) for m in plain.modules())
        assert not plain.__dict__.get("peft_config")
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in plain.modules())
        assert all(p.requires_grad for p in plain.parameters())
        assert bool(lucid.allclose(plain(_ids()).logits, expected, atol=1e-6))

    def test_nothing_merges_adds_or_goes(self, tmp_path: Path) -> None:
        peft = _xlora(tmp_path)
        with pytest.raises(NotImplementedError, match="per token"):
            peft.merge_and_unload()
        with pytest.raises(NotImplementedError, match="one adapter"):
            peft.add_adapter("second", XLoraConfig(adapters={"a": "x"}))
        with pytest.raises(NotImplementedError, match="unload"):
            peft.delete_adapter("default")
        with pytest.raises(NotImplementedError, match="per token"):
            peft(_ids(), adapter_names=["default", "default"])

    def test_the_one_adapter_is_selectable(self, tmp_path: Path) -> None:
        peft = _xlora(tmp_path)
        peft.set_adapter("default", inference_mode=True)
        assert not any(p.requires_grad for p in _classifier(peft).parameters())
        peft.set_requires_grad("default", True)
        assert all(p.requires_grad for p in _classifier(peft).parameters())
        with pytest.raises(ValueError, match="not found"):
            peft.set_adapter("other")

    def test_it_is_not_applied_to_a_bare_model(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="get_peft_model"):
            inject_adapter_in_model(XLoraConfig(adapters=_experts(tmp_path)), _lm())


class TestRefusals:
    def test_no_experts(self) -> None:
        model = _lm()
        with pytest.raises(ValueError, match="`adapters` must name at least one"):
            get_peft_model(model, XLoraConfig(task_type="CAUSAL_LM"))
        assert "peft_config" not in model.__dict__

    def test_a_model_that_already_holds_adapters(self, tmp_path: Path) -> None:
        model = _lm()
        get_peft_model(model, LoraConfig(r=2, target_modules=["c_attn"]))
        with pytest.raises(ValueError, match="already holds adapters"):
            get_peft_model(model, XLoraConfig(adapters=_experts(tmp_path)))

    def test_an_expert_that_is_not_lora(self, tmp_path: Path) -> None:
        get_peft_model(
            _lm(), IA3Config(target_modules=["c_attn"], feedforward_modules=[])
        ).save_pretrained(tmp_path / "ia3")
        model = _lm()
        before = list(model.state_dict())
        with pytest.raises(ValueError, match="holds a IA3 one"):
            get_peft_model(model, XLoraConfig(adapters={"x": str(tmp_path / "ia3")}))
        assert list(model.state_dict()) == before and not any(
            isinstance(m, Linear) for m in model.modules()
        )

    def test_a_missing_expert_leaves_the_model_as_it_was(self, tmp_path: Path) -> None:
        paths = _experts(tmp_path, ("a",))
        model = _lm()
        before = list(model.state_dict())
        with pytest.raises(FileNotFoundError):
            get_peft_model(model, XLoraConfig(adapters={**paths, "gone": str(tmp_path / "gone")}))
        assert list(model.state_dict()) == before
        assert "peft_config" not in model.__dict__

    def test_a_dora_expert(self, tmp_path: Path) -> None:
        get_peft_model(
            _lm(), LoraConfig(r=2, target_modules=["c_attn"], use_dora=True)
        ).save_pretrained(tmp_path / "dora")
        with pytest.raises(ValueError, match="does not support DoRA"):
            get_peft_model(_lm(), XLoraConfig(adapters={"d": str(tmp_path / "dora")}))

    def test_an_expert_that_is_not_a_linear_layer(self, tmp_path: Path) -> None:
        get_peft_model(_lm(), LoraConfig(r=2, target_modules=["wte"])).save_pretrained(
            tmp_path / "embedding"
        )
        model = _lm()
        with pytest.raises(NotImplementedError, match="nn.Linear layers"):
            get_peft_model(model, XLoraConfig(adapters={"e": str(tmp_path / "embedding")}))
        assert "peft_config" not in model.__dict__


class TestFile:
    def test_the_file_holds_the_classifier_alone(self, tmp_path: Path) -> None:
        peft = _xlora(tmp_path, xlora_depth=2, xlora_size=8)
        assert sorted(get_peft_model_state_dict(peft)) == sorted(
            f"base_model.internal_xlora_classifier.layers.{i}.{p}"
            for i in (0, 2)
            for p in ("bias", "weight")
        )

    def test_a_saved_adapter_reloads_to_the_same_logits(self, tmp_path: Path) -> None:
        trained = _xlora(tmp_path / "e", xlora_depth=2, xlora_size=8).eval()
        with lucid.no_grad():
            for p in _classifier(trained).parameters():
                p.copy_(lucid.randn(*p.shape))
        expected = trained(_ids()).logits
        trained.save_pretrained(tmp_path / "saved")
        loaded = PeftModel.from_pretrained(_lm(), tmp_path / "saved")
        assert type(loaded).__name__ == "PeftModelForCausalLM"
        assert isinstance(loaded.base_model, XLoraModel)
        assert bool(lucid.allclose(loaded(_ids()).logits, expected, atol=1e-5))
        assert loaded.peft_config["default"].adapters == trained.peft_config["default"].adapters

    def test_a_loaded_adapter_can_keep_training(self, tmp_path: Path) -> None:
        _xlora(tmp_path / "e").save_pretrained(tmp_path / "saved")
        loaded = PeftModel.from_pretrained(_lm(), tmp_path / "saved", is_trainable=True)
        assert all(p.requires_grad for p in _classifier(loaded).parameters())

    def test_the_classifier_loads_into_a_fresh_model(self, tmp_path: Path) -> None:
        source = _xlora(tmp_path / "s")
        other = _xlora(tmp_path / "t")
        with lucid.no_grad():
            for p in _classifier(source).parameters():
                p.copy_(lucid.randn(*p.shape))
        set_peft_model_state_dict(other, get_peft_model_state_dict(source))
        assert all(
            _same(a, b)
            for a, b in zip(
                _classifier(source).parameters(), _classifier(other).parameters(), strict=True
            )
        )
