"""Tests for LCD-827: EVA init, LoRA-GA init, adapter-to-LoRA conversion, and LoftQ refusal surface."""

from pathlib import Path
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.peft import (
    BOFTConfig,
    C3AConfig,
    EvaConfig,
    FourierFTConfig,
    IA3Config,
    LoHaConfig,
    LoKrConfig,
    LoftQConfig,
    LoraConfig,
    LoraGAConfig,
    MissConfig,
    OFTConfig,
    PeftModel,
    RandLoraConfig,
    ShiraConfig,
    VBLoRAConfig,
    VeraConfig,
    convert_to_lora,
    get_eva_state_dict,
    get_peft_model,
    initialize_lora_eva_weights,
    preprocess_loraga,
    replace_lora_weights_loftq,
    save_as_lora,
)


class _TwoLayerMLP(nn.Module):
    def __init__(self, d_in: int = 8, d_mid: int = 8, d_out: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_mid)
        self.fc2 = nn.Linear(d_mid, d_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class _ParallelLinearModel(nn.Module):
    def __init__(self, d: int = 8) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.head = nn.Linear(d, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.q_proj(x) + self.k_proj(x))


class _TinyLM(nn.Module):
    def __init__(self, vocab: int = 16, d: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.q_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None, labels: Tensor | None = None
    ) -> Tensor:
        del attention_mask, labels
        h = self.embed(input_ids)
        return self.q_proj(h) + self.v_proj(h)


class TestSubConfigsAndLoftQ:
    def test_eva_config_defaults_and_validation(self) -> None:
        cfg = EvaConfig()
        assert (cfg.rho, cfg.tau, cfg.use_label_mask, cfg.label_mask_value) == (
            2.0,
            0.99,
            True,
            -100,
        )
        assert (cfg.whiten, cfg.adjust_scaling_factors) == (False, True)
        with pytest.raises(ValueError, match="rho"):
            EvaConfig(rho=0.5)
        with pytest.raises(ValueError, match="tau"):
            EvaConfig(tau=1.5)

    def test_lora_ga_config_defaults_and_validation(self) -> None:
        cfg = LoraGAConfig()
        assert (cfg.direction, cfg.scale, cfg.stable_gamma) == ("ArB2r", "stable", 16)
        with pytest.raises(ValueError, match="direction"):
            LoraGAConfig(direction="bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="scale"):
            LoraGAConfig(scale="bad")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="stable_gamma"):
            LoraGAConfig(stable_gamma=0)

    def test_lora_config_subconfig_round_trip_and_warnings(self, tmp_path: Path) -> None:
        with pytest.warns(UserWarning, match="Using default EVA config"):
            c_eva = LoraConfig(init_lora_weights="eva", target_modules=["fc1"])
        assert c_eva.eva_config == EvaConfig()
        assert LoraConfig.from_dict(c_eva.to_dict()) == c_eva

        c_ga = LoraConfig(
            init_lora_weights="lora_ga",
            lora_ga_config=LoraGAConfig(direction="ArBr", scale="unit"),
            target_modules=["fc1"],
        )
        c_ga.save_pretrained(tmp_path)
        assert LoraConfig.from_pretrained(tmp_path) == c_ga

        with pytest.warns(UserWarning, match="eva_config"):
            LoraConfig(init_lora_weights=True, eva_config=EvaConfig())
        with pytest.warns(UserWarning, match="lora_ga_config"):
            LoraConfig(init_lora_weights=True, lora_ga_config=LoraGAConfig())

    def test_loftq_refusal_surface(self) -> None:
        cfg = LoftQConfig(loftq_bits=4, loftq_iter=2)
        assert (cfg.loftq_bits, cfg.loftq_iter) == (4, 2)
        with pytest.raises(NotImplementedError, match="loftq"):
            LoraConfig(init_lora_weights="loftq")
        with pytest.raises(NotImplementedError, match="loftq_config"):
            LoraConfig(loftq_config={"loftq_bits": 4})
        with pytest.raises(NotImplementedError, match="4-bit"):
            replace_lora_weights_loftq(_TwoLayerMLP())


class TestEvaInit:
    def test_eva_language_modeling_pipeline_and_rank_redistribution(self) -> None:
        lucid.manual_seed(0)
        base = _TinyLM(vocab=16, d=8)
        config = LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
            init_lora_weights="eva",
            eva_config=EvaConfig(rho=2.0, tau=0.9, whiten=True, adjust_scaling_factors=True),
        )
        peft_model = get_peft_model(base, config)
        batch = {
            "input_ids": lucid.randint(0, 16, (4, 6)),
            "attention_mask": lucid.ones(4, 6, dtype=lucid.int64),
            "labels": lucid.tensor(
                [
                    [1, 2, 3, -100, -100, -100],
                    [4, 5, 6, 7, -100, -100],
                    [1, 2, 3, 4, 5, -100],
                    [2, 3, 4, 5, 6, 7],
                ]
            ),
        }
        initialize_lora_eva_weights(peft_model, dataloader=[batch, batch])
        q_layer = peft_model.base_model.model.q_proj
        v_layer = peft_model.base_model.model.v_proj
        total_rank = q_layer.r["default"] + v_layer.r["default"]
        assert total_rank == 4
        assert not bool(lucid.all(q_layer.lora_A["default"].weight == 0).item())
        assert bool(lucid.all(q_layer.lora_B["default"].weight == 0).item())

    def test_get_eva_state_dict_on_bare_model_and_load(self) -> None:
        lucid.manual_seed(1)
        base = _ParallelLinearModel(d=8)
        config = LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "k_proj"],
            init_lora_weights="eva",
            eva_config=EvaConfig(rho=1.5, tau=0.95),
        )
        batches = [lucid.randn(12, 8), lucid.randn(12, 8)]
        state = get_eva_state_dict(
            base,
            dataloader=batches,
            peft_config=config,
            forward_fn=lambda m, x: m(x),
            prepare_model_inputs_fn=None,
            prepare_layer_inputs_fn=None,
        )
        assert set(state.keys()) == {"q_proj", "k_proj"}
        peft_model = get_peft_model(base, config)
        initialize_lora_eva_weights(peft_model, eva_state_dict=state)
        assert (
            peft_model.base_model.model.q_proj.r["default"]
            + peft_model.base_model.model.k_proj.r["default"]
            == 4
        )

    def test_eva_error_conditions(self) -> None:
        base = _TwoLayerMLP()
        with pytest.raises(ValueError, match="empty"):
            get_eva_state_dict(base, dataloader=[], peft_config=LoraConfig(target_modules=["fc1"]))
        with pytest.raises(ValueError, match="peft_config is required"):
            get_eva_state_dict(base, dataloader=[lucid.randn(4, 8)])
        with pytest.raises(ValueError, match="PeftModel"):
            initialize_lora_eva_weights(base, dataloader=[lucid.randn(4, 8)])
        with pytest.raises(ValueError, match="fewer samples than n_components"):
            get_eva_state_dict(
                base,
                dataloader=[lucid.randn(2, 8)],
                peft_config=LoraConfig(
                    r=2, target_modules=["fc1"], init_lora_weights="eva", eva_config=EvaConfig()
                ),
                forward_fn=lambda m, b: m(b),
                prepare_model_inputs_fn=None,
                prepare_layer_inputs_fn=None,
            )


class TestLoraGA:
    @pytest.mark.parametrize("direction", ["ArBr", "A2rBr", "ArB2r", "random"])
    @pytest.mark.parametrize("scale", ["stable", "weight_svd", "gd_scale", "unit"])
    def test_preprocess_and_init_preserves_forward_output(self, direction: str, scale: str) -> None:
        lucid.manual_seed(7)
        model = _TwoLayerMLP(d_in=8, d_mid=8, d_out=4)
        x = lucid.randn(4, 8)
        expected = model(x).detach().clone()

        config = LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["fc1", "fc2"],
            init_lora_weights="lora_ga",
            lora_ga_config=LoraGAConfig(direction=direction, scale=scale, stable_gamma=16),  # type: ignore[arg-type]
        )

        def step() -> None:
            out = model(x)
            loss = lucid.sum(out * out)
            loss.backward()

        preprocess_loraga(model, config, train_step=step)
        assert hasattr(model.fc1, "_peft_loraga_grad")
        assert hasattr(model.fc2, "_peft_loraga_grad")

        peft_model = get_peft_model(model, config)
        assert not hasattr(peft_model.base_model.model.fc1.get_base_layer(), "_peft_loraga_grad")
        actual = peft_model(x)
        assert bool(lucid.allclose(actual, expected, atol=1e-5))

    def test_preprocess_loraga_cache_file(self, tmp_path: Path) -> None:
        lucid.manual_seed(11)
        m1 = _TwoLayerMLP(d_in=8, d_mid=8, d_out=4)
        m2 = _TwoLayerMLP(d_in=8, d_mid=8, d_out=4)
        m2.load_state_dict(m1.state_dict())
        x = lucid.randn(4, 8)
        cache_path = tmp_path / "grads" / "loraga.pt"
        cfg = LoraConfig(
            r=2,
            target_modules=["fc1", "fc2"],
            init_lora_weights="lora_ga",
            lora_ga_config=LoraGAConfig(),
        )
        calls = 0

        def step() -> None:
            nonlocal calls
            calls += 1
            lucid.sum(m1(x) ** 2).backward()

        preprocess_loraga(m1, cfg, train_step=step, cache_file=cache_path)
        assert calls == 1 and cache_path.is_file()

        def should_not_run() -> None:
            raise AssertionError("train_step should not be called when cache_file exists")

        preprocess_loraga(m2, cfg, train_step=should_not_run, cache_file=cache_path)
        assert bool(
            lucid.allclose(
                getattr(m1.fc1, "_peft_loraga_grad"), getattr(m2.fc1, "_peft_loraga_grad")
            )
        )


class TestLoraConversion:
    def test_supports_lora_conversion_flags(self) -> None:
        for cfg in (
            LoHaConfig(r=2, target_modules=["fc1", "fc2"]),
            LoKrConfig(r=2, target_modules=["fc1", "fc2"]),
            VeraConfig(r=2, target_modules=["fc1", "fc2"]),
            VBLoRAConfig(r=2, num_vectors=4, vector_length=2, target_modules=["fc1", "fc2"]),
            RandLoraConfig(r=2, target_modules=["fc1", "fc2"]),
            FourierFTConfig(n_frequency=4, target_modules=["fc1", "fc2"]),
            C3AConfig(block_size=2, target_modules=["fc1", "fc2"]),
            MissConfig(r=2, target_modules=["fc1", "fc2"]),
            ShiraConfig(r=2, target_modules=["fc1", "fc2"]),
            LoraConfig(r=2, target_modules=["fc1", "fc2"]),
        ):
            m = get_peft_model(_TwoLayerMLP(), cfg)
            assert m.supports_lora_conversion() is True

        for unsupported_cfg in (
            OFTConfig(r=2, oft_block_size=0, target_modules=["fc1", "fc2"]),
            BOFTConfig(boft_block_size=2, target_modules=["fc1", "fc2"]),
            IA3Config(target_modules=["fc1", "fc2"], feedforward_modules=["fc1", "fc2"]),
        ):
            m = get_peft_model(_TwoLayerMLP(), unsupported_cfg)
            assert m.supports_lora_conversion() is False
            with pytest.raises(ValueError, match="does not support conversion to LoRA"):
                convert_to_lora(m, rank=2)

    @pytest.mark.parametrize(
        "cfg",
        [
            LoHaConfig(r=2, alpha=4, target_modules=["fc1"], init_weights=False),
            LoKrConfig(r=2, alpha=4, target_modules=["fc1"], init_weights=False),
            MissConfig(r=2, target_modules=["fc1"], init_weights=False),
            MissConfig(r=2, mini_r=2, target_modules=["fc1"], init_weights="mini"),
            MissConfig(r=2, target_modules=["fc1"], init_weights="bat"),
            ShiraConfig(r=2, target_modules=["fc1"], init_weights=False),
        ],
    )
    def test_save_as_lora_reproduces_adapted_forward(self, cfg: object, tmp_path: Path) -> None:
        lucid.manual_seed(19)
        base = _TwoLayerMLP(d_in=8, d_mid=8, d_out=4)
        base_state = {k: v.clone() for k, v in base.state_dict().items()}
        adapted = get_peft_model(base, cfg, "default")  # type: ignore[arg-type]
        for p in adapted.parameters():
            if p.requires_grad:
                with lucid.no_grad():
                    p.add_(0.1 * lucid.randn(*p.shape, device=p.device, dtype=p.dtype))

        x = lucid.randn(3, 8)
        expected = adapted(x)

        out_dir = tmp_path / "lora_out"
        save_as_lora(out_dir, adapted, rank=8)

        fresh = _TwoLayerMLP(d_in=8, d_mid=8, d_out=4)
        fresh.load_state_dict(base_state)
        loaded_lora = PeftModel.from_pretrained(fresh, out_dir)
        actual = loaded_lora(x)
        assert bool(lucid.allclose(actual, expected, atol=1e-4))

    def test_convert_to_lora_threshold_rank_and_modules_to_save(self) -> None:
        lucid.manual_seed(23)
        base = _TwoLayerMLP(d_in=8, d_mid=8, d_out=4)
        cfg = LoHaConfig(
            r=2, alpha=4, target_modules=["fc1"], modules_to_save=["fc2"], init_weights=False
        )
        adapted = get_peft_model(base, cfg)
        lora_cfg, state = convert_to_lora(adapted, rank=0.95)
        assert "fc1" in lora_cfg.rank_pattern
        assert lora_cfg.modules_to_save == ["fc2"]
        assert "base_model.model.fc2.weight" in state
        assert "base_model.model.fc2.bias" in state
