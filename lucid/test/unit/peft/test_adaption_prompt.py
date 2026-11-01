"""Adaption prompts (LLaMA-Adapter): learned prompts the top attention layers attend to, behind a zero gate.

What follows from the method: the adapted decoder starts as the base bit for bit (the gate is
zero), only the top ``adapter_layers`` blocks hold an adapter and only their prompts and gates
train, the layer computes the gated attention over the prompt (checked against a plain-ops
rewrite), one adapter applies at a time, and nothing merges.  A model type the method has no
attention strategy for is refused by name.
"""

import math
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.models import create_model
from lucid.peft import (
    AdaptionPromptConfig,
    AdaptionPromptModel,
    PeftModel,
    PeftType,
    get_peft_model,
    get_peft_model_state_dict,
)
from lucid.peft.tuners.adaption_prompt import AdaptedAttention

_TEXT: dict[str, object] = {
    "vocab_size": 64,
    "hidden_size": 32,
    "num_hidden_layers": 3,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "max_position_embeddings": 32,
    "hidden_dropout": 0.0,
    "attention_dropout": 0.0,
}


def _lm(factory: str = "gpt2_small_lm", device: str = "cpu") -> nn.Module:
    lucid.manual_seed(0)
    return create_model(factory, **_TEXT).to(device)


def _ids(device: str = "cpu") -> lucid.Tensor:
    lucid.manual_seed(1)
    return lucid.randint(0, 64, (2, 10), device=device)


def _config(
    adapter_len: int | None = 4,
    adapter_layers: int | None = 2,
    target_modules: str | None = None,
    exclude_modules: list[str] | None = None,
    modules_to_save: list[str] | None = None,
    layers_to_transform: list[int] | None = None,
    layers_pattern: str | None = None,
) -> AdaptionPromptConfig:
    return AdaptionPromptConfig(
        adapter_len=adapter_len,
        adapter_layers=adapter_layers,
        target_modules=target_modules,
        exclude_modules=exclude_modules,
        modules_to_save=modules_to_save,
        layers_to_transform=layers_to_transform,
        layers_pattern=layers_pattern,
        task_type="CAUSAL_LM",
    )


def _open_gates(peft: PeftModel, to: float = 0.7) -> None:
    with lucid.no_grad():
        for name, param in peft.named_parameters():
            if "adaption_gate" in name:
                param.copy_(lucid.full_like(param, to))


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


class TestConfig:
    def test_the_fields_are_the_references(self) -> None:
        assert set(_config().to_dict()) == {
            *("adapter_layers", "adapter_len", "auto_mapping", "base_model_name_or_path"),
            *("inference_mode", "peft_type", "revision", "target_modules", "task_type"),
        }

    def test_a_file_round_trips(self) -> None:
        config = _config(target_modules="attn")
        assert AdaptionPromptConfig.from_dict(config.to_dict()) == config
        assert config.to_dict()["peft_type"] == "ADAPTION_PROMPT"
        assert config.peft_type == PeftType.ADAPTION_PROMPT

    def test_sizes_are_positive_integers(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _config(adapter_len=0)
        with pytest.raises(ValueError, match="positive"):
            _config(adapter_layers=0)
        with pytest.raises(TypeError, match="int"):
            AdaptionPromptConfig.from_dict({"adapter_len": "4"})

    @pytest.mark.parametrize(
        "unused",
        [
            {"exclude_modules": ["x"]},
            {"modules_to_save": ["x"]},
            {"layers_to_transform": [0]},
            {"layers_pattern": "h"},
        ],
        ids=lambda unused: next(iter(unused)),
    )
    def test_targeting_fields_the_method_lacks_are_refused(self, unused: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="not a field of an adaption prompt"):
            AdaptionPromptConfig.from_dict({"adapter_len": 4, "adapter_layers": 2, **unused})

    def test_a_size_left_unset_is_refused_when_the_adapter_is_built(self) -> None:
        with pytest.raises(ValueError, match="`adapter_len` must be set"):
            get_peft_model(_lm(), _config(adapter_len=None))
        with pytest.raises(ValueError, match="`adapter_layers` must be set"):
            get_peft_model(_lm(), _config(adapter_layers=None))


class TestLayer:
    def _layer(self) -> tuple[AdaptedAttention, nn.Module]:
        lucid.manual_seed(2)
        attention = _lm().transformer.h[0].attn
        layer = AdaptedAttention(attention, "default", _config(adapter_len=3))
        return layer, attention

    def test_the_prompt_starts_random_and_the_gate_at_zero(self) -> None:
        layer, _ = self._layer()
        prompt, gate = layer.adaption_prompt["default"], layer.adaption_gate["default"]
        assert tuple(prompt.shape) == (1, 3, 32) and tuple(gate.shape) == (1,)
        assert float(prompt.std().item()) > 0.3
        assert float(gate.abs().max().item()) == 0.0

    def test_it_computes_the_gated_attention_over_the_prompt(self) -> None:
        layer, attention = self._layer()
        with lucid.no_grad():
            layer.adaption_gate["default"].copy_(lucid.tensor([0.6]))
        x = lucid.randn(2, 5, 32)
        heads, dim, length = attention.num_heads, attention.head_dim, 3
        # The same arithmetic in plain ops: queries of x, keys and values of the prompt.
        q, _, _ = attention.c_attn(x).chunk(3, dim=-1)
        _, k, v = attention.c_attn(layer.adaption_prompt["default"]).chunk(3, dim=-1)
        q = q.reshape(2, 5, heads, dim).permute(0, 2, 1, 3)
        k = k.reshape(1, length, heads, dim).permute(0, 2, 1, 3)
        v = v.reshape(1, length, heads, dim).permute(0, 2, 1, 3)
        weights = F.softmax(q @ k.transpose(2, 3) / math.sqrt(dim), dim=-1)
        expected = attention(x) + 0.6 * (weights @ v).permute(0, 2, 1, 3).reshape(2, 5, 32)
        assert bool(lucid.allclose(layer(x), expected, atol=1e-5))

    def test_every_query_attends_to_the_whole_prompt_but_the_gate_decides_how_much(self) -> None:
        layer, attention = self._layer()
        x = lucid.randn(1, 4, 32)
        with lucid.no_grad():
            layer.adaption_gate["default"].copy_(lucid.tensor([1.0]))
        gap = float((layer(x) - attention(x)).abs().max().item())
        with lucid.no_grad():
            layer.adaption_gate["default"].copy_(lucid.tensor([2.0]))
        assert float((layer(x) - attention(x)).abs().max().item()) > gap > 0

    def test_a_disabled_layer_is_the_base(self) -> None:
        layer, attention = self._layer()
        with lucid.no_grad():
            layer.adaption_gate["default"].copy_(lucid.tensor([1.0]))
        x = lucid.randn(1, 4, 32)
        layer.enable_adapters(False)
        assert _same(layer(x), attention(x))

    def test_it_wraps_a_gpt_attention_too_and_refuses_a_module_of_no_family(self) -> None:
        lucid.manual_seed(2)
        gpt = create_model("gpt_lm", **_TEXT).transformer.h[0].attn
        assert tuple(
            AdaptedAttention(gpt, "default", _config()).adaption_prompt["default"].shape
        ) == (1, 4, 32)
        with pytest.raises(TypeError, match="not an attention module an adaption prompt can wrap"):
            AdaptedAttention(nn.Linear(4, 4), "default", _config())

    def test_it_cannot_be_merged(self) -> None:
        layer, _ = self._layer()
        with pytest.raises(NotImplementedError, match="adaption prompt cannot be folded"):
            layer.merge()
        with pytest.raises(NotImplementedError, match="cannot be folded"):
            layer.get_delta_weight("default")


class TestTuner:
    def test_the_adapted_decoder_starts_as_the_base(self, device: str) -> None:
        base, ids = _lm(device=device), _ids(device)
        expected = base(ids).logits
        peft = get_peft_model(base, _config())
        assert _same(peft.eval()(ids).logits, expected)

    def test_the_top_layers_alone_are_adapted_and_trained(self) -> None:
        peft = get_peft_model(_lm(), _config(adapter_layers=2))
        assert peft.base_model.targeted_module_names == [
            "transformer.h.1.attn",
            "transformer.h.2.attn",
        ]
        trained = {n for n, p in peft.named_parameters() if p.requires_grad}
        assert trained == {
            f"base_model.model.transformer.h.{i}.attn.{table}.default"
            for i in (1, 2)
            for table in ("adaption_prompt", "adaption_gate")
        }
        assert isinstance(peft.base_model, AdaptionPromptModel)
        assert not any(
            isinstance(m, AdaptedAttention)
            for m in peft.get_base_model().transformer.h[0].modules()
        )

    def test_more_layers_than_the_model_has_is_refused_with_the_model_untouched(self) -> None:
        model = _lm()
        keys = list(model.state_dict())
        with pytest.raises(ValueError, match="more adapter layers '4' than the model has '3'"):
            get_peft_model(model, _config(adapter_layers=4))
        assert list(model.state_dict()) == keys

    def test_a_model_type_without_a_strategy_is_refused_by_name(self) -> None:
        bert = create_model("bert_base_mlm", **{**_TEXT, "type_vocab_size": 2})
        with pytest.raises(ValueError, match="Unsupported model type for adaption prompt: 'bert'"):
            get_peft_model(bert, _config())
        with pytest.raises(ValueError, match="gpt, gpt2"):
            get_peft_model(nn.Sequential(nn.Linear(4, 4)), _config())

    def test_the_target_name_is_the_models_unless_given(self) -> None:
        peft = get_peft_model(_lm(), _config())
        assert peft.peft_config["default"].target_modules == "attn"
        with pytest.raises(TypeError, match="not an attention module an adaption prompt can wrap"):
            get_peft_model(_lm(), _config(target_modules="mlp"))

    def test_training_moves_the_prompts_and_gates_alone_and_lowers_the_loss(
        self, device: str
    ) -> None:
        peft = get_peft_model(_lm(device=device), _config())
        ids = _ids(device)
        frozen = {n: p.clone() for n, p in peft.named_parameters() if not p.requires_grad}
        opt = optim.AdamW([p for p in peft.parameters() if p.requires_grad], lr=5e-2)
        losses = []
        peft.train()
        for _ in range(6):
            opt.zero_grad()
            loss = peft(ids, labels=ids).loss
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        assert losses[-1] < losses[0], losses
        assert all(_same(p, frozen[n]) for n, p in peft.named_parameters() if n in frozen)
        gate = peft.get_base_model().transformer.h[2].attn.adaption_gate["default"]
        assert float(gate.abs().item()) > 0

    def test_generate_runs_with_the_prompts_in_every_step(self) -> None:
        peft = get_peft_model(_lm(), _config()).eval()
        _open_gates(peft)
        ids = _ids()
        cached = peft.generate(ids, max_new_tokens=3, do_sample=False)
        assert tuple(cached.shape) == (2, 13)

    def test_disable_adapter_gives_the_base_logits(self) -> None:
        base, ids = _lm(), _ids()
        expected = base(ids).logits
        peft = get_peft_model(base, _config()).eval()
        _open_gates(peft)
        assert not _same(peft(ids).logits, expected)
        with peft.disable_adapter():
            assert _same(peft(ids).logits, expected)

    def test_unload_gives_the_model_back(self) -> None:
        base, ids = _lm(), _ids()
        expected = base(ids).logits
        peft = get_peft_model(base, _config()).eval()
        _open_gates(peft)
        plain = peft.unload()
        assert plain is base
        assert not any(isinstance(m, AdaptedAttention) for m in plain.modules())
        assert _same(plain(ids).logits, expected)
        assert all(p.requires_grad for p in plain.parameters())

    def test_merging_is_refused(self) -> None:
        peft = get_peft_model(_lm(), _config())
        with pytest.raises(NotImplementedError, match="cannot be folded"):
            peft.merge_and_unload()


class TestAdapters:
    def test_several_adapters_live_side_by_side_and_one_applies(self) -> None:
        base, ids = _lm(), _ids()
        expected = base(ids).logits
        peft = get_peft_model(base, _config(adapter_layers=1), adapter_name="a").eval()
        peft.add_adapter("b", _config(adapter_len=6, adapter_layers=3))
        _open_gates(peft)
        assert peft.active_adapter == "a"
        with_a = peft(ids).logits
        assert not _same(with_a, expected)
        peft.set_adapter("b")
        with_b = peft(ids).logits
        assert not _same(with_b, with_a)
        trained = {n for n, p in peft.named_parameters() if p.requires_grad}
        assert trained and all(".b" in n or n.endswith(".b") for n in trained)
        peft.set_adapter("a")
        assert _same(peft(ids).logits, with_a)

    def test_a_layer_an_adapter_does_not_reach_runs_its_base(self) -> None:
        base, ids = _lm(), _ids()
        expected = base(ids).logits
        peft = get_peft_model(base, _config(adapter_layers=1), adapter_name="a").eval()
        peft.add_adapter("b", _config(adapter_layers=2))
        _open_gates(peft)
        peft.set_adapter("a")
        top = peft(ids).logits
        peft.set_adapter("b")
        assert not _same(peft(ids).logits, top)
        with pytest.warns(UserWarning, match="Setting active adapter to a"):
            peft.delete_adapter("b")
        assert _same(peft(ids).logits, top)
        assert not _same(top, expected)

    def test_two_adapters_cannot_apply_together(self) -> None:
        peft = get_peft_model(_lm(), _config(), adapter_name="a")
        peft.add_adapter("b", _config())
        with pytest.raises(ValueError, match="one adapter at a time"):
            peft.base_model.set_adapter(["a", "b"])


class TestFile:
    def test_the_file_holds_the_references_keys(self) -> None:
        peft = get_peft_model(_lm(), _config())
        assert sorted(get_peft_model_state_dict(peft)) == sorted(
            f"base_model.model.transformer.h.{i}.attn.{table}"
            for i in (1, 2)
            for table in ("adaption_prompt", "adaption_gate")
        )

    def test_a_saved_adapter_reloads_to_the_same_logits(self, tmp_path: Path) -> None:
        trained = get_peft_model(_lm(), _config()).eval()
        _open_gates(trained, 0.4)
        ids = _ids()
        expected = trained(ids).logits
        trained.save_pretrained(tmp_path)
        loaded = PeftModel.from_pretrained(_lm(), tmp_path)
        assert type(loaded).__name__ == "PeftModelForCausalLM"
        assert bool(lucid.allclose(loaded(ids).logits, expected, atol=1e-6))
        assert loaded.peft_config["default"] == trained._config_to_save("default")

    def test_a_file_loads_as_a_trainable_adapter_too(self, tmp_path: Path) -> None:
        get_peft_model(_lm(), _config()).save_pretrained(tmp_path)
        loaded = PeftModel.from_pretrained(_lm(), tmp_path, is_trainable=True)
        assert sum(p.requires_grad for p in loaded.parameters()) == 4
