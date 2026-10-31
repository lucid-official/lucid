"""``lora_bias``: a bias on ``lora_B``, scaled with the rest of the update and merged into the base bias.

``y = W x + b + s (B A x + c)`` with ``c`` the bias of ``B``.  Merging adds
``s c`` to ``b`` beside ``s B A`` to ``W``, so a merged layer is the same
function; unmerging takes both out.  The state-dict key is the reference's,
``lora_B.<adapter>.bias``.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from lucid.peft.tuners.lora import Conv2d, Embedding, Linear, MultiheadAttention, ParamWrapper


def _close(a: lucid.Tensor, b: lucid.Tensor, atol: float = 1e-5) -> bool:
    return bool(lucid.allclose(a, b, atol=atol, rtol=1e-4))


def _linear(device: str, **kwargs: object) -> tuple[nn.Linear, Linear]:
    lucid.manual_seed(0)
    base = nn.Linear(6, 5).to(device)
    options: dict[str, object] = {
        "r": 3,
        "lora_alpha": 6,
        "lora_bias": True,
        "init_lora_weights": False,
    } | kwargs
    return base, Linear(base, "default", **options)  # type: ignore[arg-type]


class TestLinear:
    def test_the_bias_is_the_reference_s_key(self, device: str) -> None:
        _, layer = _linear(device)
        assert sorted(layer.state_dict()) == [
            "base_layer.bias",
            "base_layer.weight",
            "lora_A.default.weight",
            "lora_B.default.bias",
            "lora_B.default.weight",
        ]
        assert tuple(layer.lora_B["default"].bias.shape) == (5,)
        assert layer.lora_bias == {"default": True}

    def test_off_by_default(self, device: str) -> None:
        lucid.manual_seed(0)
        layer = Linear(nn.Linear(6, 5).to(device), "default", r=2)
        assert layer.lora_B["default"].bias is None and layer.lora_bias == {"default": False}

    def test_the_default_init_zeros_the_bias(self, device: str) -> None:
        _, layer = _linear(device, init_lora_weights=True)
        assert float(layer.lora_B["default"].bias.abs().max().item()) == 0.0

    def test_a_fresh_layer_is_the_base(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Linear(6, 5).to(device)
        layer = Linear(base, "default", r=3, lora_alpha=6, lora_bias=True)
        x = lucid.randn(4, 6, device=device)
        assert bool((layer(x) == base(x)).all().item())

    def test_the_forward_adds_the_scaled_bias(self, device: str) -> None:
        base, layer = _linear(device)
        x = lucid.randn(4, 6, device=device)
        a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
        c = layer.lora_B["default"].bias
        expected = base(x) + 2.0 * (F.linear(F.linear(x, a), b) + c)
        assert _close(layer(x), expected)

    def test_grads_reach_the_bias(self, device: str) -> None:
        _, layer = _linear(device)
        layer(lucid.randn(4, 6, device=device)).sum().backward()
        assert layer.lora_B["default"].bias.grad is not None

    def test_merge_folds_the_bias_into_the_base_bias(self, device: str) -> None:
        base, layer = _linear(device)
        x = lucid.randn(4, 6, device=device)
        before = layer(x)
        bias = base.bias.detach().clone()
        expected_bias = bias + 2.0 * layer.lora_B["default"].bias.detach()
        layer.merge()
        assert _close(base.bias, expected_bias) and _close(layer(x), before)

    def test_unmerge_takes_the_bias_back_out(self, device: str) -> None:
        base, layer = _linear(device)
        weight, bias = base.weight.detach().clone(), base.bias.detach().clone()
        layer.merge()
        layer.unmerge()
        assert _close(base.weight, weight) and _close(base.bias, bias)

    def test_safe_merge(self, device: str) -> None:
        base, layer = _linear(device)
        x = lucid.randn(4, 6, device=device)
        before = layer(x)
        layer.merge(safe_merge=True)
        assert _close(layer(x), before)

    def test_merge_targets_include_the_bias_only_when_an_adapter_needs_it(
        self, device: str
    ) -> None:
        _, plain = _linear(device, lora_bias=False)
        _, biased = _linear(device)
        assert list(plain._merge_targets()) == ["base_layer.weight"]
        assert list(biased._merge_targets()) == ["base_layer.weight", "base_layer.bias"]

    def test_a_non_finite_bias_fails_the_safe_merge_and_changes_nothing(self, device: str) -> None:
        base, layer = _linear(device)
        bias = base.bias.detach().clone()
        with lucid.no_grad():
            layer.lora_B["default"].bias.copy_(layer.lora_B["default"].bias * float("inf"))
        with pytest.raises(ValueError, match="NaNs detected"):
            layer.merge(safe_merge=True)
        assert bool((base.bias == bias).all().item())

    def test_biased_and_plain_adapters_together(self, device: str) -> None:
        base, layer = _linear(device)
        layer.update_layer("plain", 2, 4.0, 0.0, False, False)
        layer.set_adapter(["default", "plain"])
        x = lucid.randn(4, 6, device=device)
        before = layer(x)
        layer.merge()
        assert _close(layer(x), before)
        layer.unmerge()
        assert _close(layer(x), before)

    def test_deleting_the_biased_adapter_takes_the_bias_target_with_it(self, device: str) -> None:
        _, layer = _linear(device)
        layer.update_layer("other", 2, 4.0, 0.0, True, False)
        layer.delete_adapter("default")
        assert layer.lora_bias == {"other": False}
        assert list(layer._merge_targets()) == ["base_layer.weight"]


class TestBaseWithoutBias:
    def test_it_warns_at_creation_and_refuses_to_merge(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Linear(6, 5, bias=False).to(device)
        with pytest.warns(UserWarning, match="has no bias"):
            layer = Linear(base, "default", r=2, lora_alpha=4, lora_bias=True)
        weight = base.weight.detach().clone()
        with pytest.raises(RuntimeError, match="no bias"):
            layer.merge()
        assert bool((base.weight == weight).all().item()) and not layer.merged
        x = lucid.randn(3, 6, device=device)
        assert tuple(layer(x).shape) == (3, 5)  # still usable unmerged


class TestConv:
    def test_conv2d(self, device: str) -> None:
        lucid.manual_seed(0)
        base = nn.Conv2d(3, 5, kernel_size=3, padding=1).to(device)
        layer = Conv2d(base, "default", r=2, lora_alpha=4, lora_bias=True, init_lora_weights=False)
        assert tuple(layer.lora_B["default"].bias.shape) == (5,)
        x = lucid.randn(2, 3, 6, 6, device=device)
        before = layer(x)
        weight, bias = base.weight.detach().clone(), base.bias.detach().clone()
        layer.merge()
        assert _close(layer(x), before, atol=1e-4)
        layer.unmerge()
        assert _close(base.weight, weight) and _close(base.bias, bias)


class TestRefusals:
    def test_embedding(self, device: str) -> None:
        with pytest.raises(ValueError, match="lora_bias=True is not supported for Embedding"):
            Embedding(nn.Embedding(9, 4).to(device), "default", r=2, lora_bias=True)

    def test_attention_and_raw_parameters(self, device: str) -> None:
        attention = nn.MultiheadAttention(8, 2, batch_first=True).to(device)
        with pytest.raises(ValueError, match="lora_bias=True is not supported"):
            MultiheadAttention(attention, "default", r=2, lora_bias=True)

        class Gate(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = nn.Parameter(lucid.randn(4, 6))

        with pytest.raises(ValueError, match="lora_bias=True is not supported"):
            ParamWrapper(Gate().to(device), "default", "w", r=2, lora_bias=True)

    def test_with_dora_in_the_config(self) -> None:
        with pytest.raises(ValueError, match="not supported for DoRA"):
            LoraConfig(lora_bias=True, use_dora=True)

    @pytest.mark.parametrize("init", ["gaussian", "pissa", "olora"])
    def test_with_a_non_boolean_init_in_the_config(self, init: str) -> None:
        with pytest.raises(ValueError, match="only supported with init_lora_weights=True or False"):
            LoraConfig(lora_bias=True, init_lora_weights=init)

    def test_the_flag_must_be_a_bool(self) -> None:
        with pytest.raises(TypeError, match="lora_bias"):
            LoraConfig(lora_bias=1)  # type: ignore[arg-type]


class TestModel:
    def _net(self, device: str) -> nn.Sequential:
        lucid.manual_seed(0)
        return nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 3)).to(device)

    def test_the_config_round_trips(self) -> None:
        config = LoraConfig(lora_bias=True, target_modules=["0"])
        assert config.to_dict()["lora_bias"] is True
        assert LoraConfig.from_dict(config.to_dict()) == config

    def test_the_adapter_file_has_the_bias_key(self, device: str) -> None:
        peft = get_peft_model(
            self._net(device), LoraConfig(r=2, target_modules=["0"], lora_bias=True)
        )
        assert sorted(get_peft_model_state_dict(peft)) == [
            "base_model.model.0.lora_A.weight",
            "base_model.model.0.lora_B.bias",
            "base_model.model.0.lora_B.weight",
        ]

    def test_the_bias_trains_and_merges_and_unloads(self, device: str) -> None:
        net = self._net(device)
        peft = get_peft_model(
            net,
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["0", "2"],
                lora_bias=True,
                init_lora_weights=False,
            ),
        )
        trained = {n for n, p in peft.named_parameters() if p.requires_grad}
        assert any(n.endswith("lora_B.default.bias") for n in trained)
        x = lucid.randn(4, 6, device=device)
        expected = peft(x)
        merged = peft.merge_and_unload()
        assert type(merged[0]) is nn.Linear and _close(merged(x), expected, atol=1e-4)

    def test_weighted_adapters_refuse_a_biased_source(self, device: str) -> None:
        peft = get_peft_model(
            self._net(device),
            LoraConfig(r=2, target_modules=["0"], lora_bias=True, init_lora_weights=False),
            adapter_name="a",
        )
        peft.add_adapter("b", LoraConfig(r=2, target_modules=["0"], init_lora_weights=False))
        with pytest.raises(NotImplementedError, match="lora_bias"):
            peft.base_model.add_weighted_adapter(["a", "b"], [0.5, 0.5], "mix")
