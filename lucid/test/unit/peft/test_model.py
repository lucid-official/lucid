"""Injection: which modules are adapted, what trains, and taking it all out again."""

import copy
import threading

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid.peft import (
    LoraConfig,
    LoraModel,
    PeftModel,
    TaskType,
    get_peft_model,
    inject_adapter_in_model,
)
from lucid.peft.tuners.lora import Linear, LoraLayer


class _Block(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Linear(d, d)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.mlp(self.norm(self.q_proj(x) + self.v_proj(x)))


class _TinyLM(nn.Module):
    """``model.layers.N.{q_proj,v_proj,mlp}`` and an LM head, tied or not."""

    def __init__(self, vocab: int = 12, d: int = 8, depth: int = 3, tie: bool = True) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([_Block(d) for _ in range(depth)])
        self.lm_head = nn.Linear(d, vocab, bias=False)
        if tie:
            self.lm_head.weight = self.embed.weight

    def forward(self, ids: lucid.Tensor) -> lucid.Tensor:
        h = self.embed(ids)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(h)


class _Classifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(8, 8)
        self.pre_classifier = nn.Linear(8, 8)
        self.classifier = nn.Linear(8, 3)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.classifier(F.relu(self.pre_classifier(self.encoder(x))))


def _lm(device: str = "cpu", **kwargs: object) -> _TinyLM:
    lucid.manual_seed(0)
    return _TinyLM(**kwargs).to(device)  # type: ignore[arg-type]


def _ids(device: str = "cpu") -> lucid.Tensor:
    return lucid.tensor([[1, 4, 2, 7], [3, 3, 0, 11]], device=device)


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def _targets(config: LoraConfig, model: nn.Module | None = None) -> list[str]:
    return sorted(LoraModel(model or _lm(), config).targeted_module_names)


class TestTargeting:
    def test_a_list_matches_by_dotted_suffix(self) -> None:
        names = _targets(LoraConfig(target_modules=["q_proj"]))
        assert names == [f"layers.{i}.q_proj" for i in range(3)]

    def test_a_list_entry_may_be_a_full_name(self) -> None:
        assert _targets(LoraConfig(target_modules=["layers.1.v_proj"])) == ["layers.1.v_proj"]

    def test_a_suffix_does_not_match_inside_a_name(self) -> None:
        model = _Classifier()
        names = _targets(LoraConfig(target_modules=["classifier"]), model)
        assert names == ["classifier"]

    def test_a_string_is_a_full_regex(self) -> None:
        names = _targets(LoraConfig(target_modules=r"layers\.[02]\.(q|v)_proj"))
        assert names == ["layers.0.q_proj", "layers.0.v_proj", "layers.2.q_proj", "layers.2.v_proj"]

    def test_exclude_modules_list_and_regex(self) -> None:
        listed = LoraConfig(
            target_modules=["q_proj", "v_proj"], exclude_modules=["layers.0.q_proj"]
        )
        assert "layers.0.q_proj" not in _targets(listed) and len(_targets(listed)) == 5
        regex = LoraConfig(target_modules=["q_proj", "v_proj"], exclude_modules=r".*v_proj")
        assert _targets(regex) == [f"layers.{i}.q_proj" for i in range(3)]

    @pytest.mark.parametrize(
        ("layers", "pattern", "expected"),
        [
            ([0, 2], None, [0, 2]),
            (1, None, [1]),
            ([1], "layers", [1]),
            ([1], ["blocks", "layers"], [1]),
        ],
    )
    def test_layers_to_transform_filters_by_index(
        self, layers: int | list[int], pattern: str | list[str] | None, expected: list[int]
    ) -> None:
        config = LoraConfig(
            target_modules=["q_proj"], layers_to_transform=layers, layers_pattern=pattern
        )
        assert _targets(config) == [f"layers.{i}.q_proj" for i in expected]

    @pytest.mark.parametrize(
        ("layers", "pattern", "expected"),
        [
            # The reference's current rule: the first (outer) index.
            (
                [0],
                None,
                ["encoder.layers.0.blocks.0.attn.qkv", "encoder.layers.0.blocks.1.attn.qkv"],
            ),
            (
                [1],
                "blocks",
                ["encoder.layers.0.blocks.1.attn.qkv", "encoder.layers.1.blocks.1.attn.qkv"],
            ),
        ],
    )
    def test_nested_lists_take_the_outer_index(
        self, layers: list[int], pattern: str | None, expected: list[str]
    ) -> None:
        def blocks() -> nn.ModuleList:
            return nn.ModuleList(
                [nn.ModuleDict({"attn": nn.ModuleDict({"qkv": nn.Linear(2, 2)})}) for _ in range(2)]
            )

        layer_list = nn.ModuleList([nn.ModuleDict({"blocks": blocks()}) for _ in range(2)])
        model = nn.ModuleDict({"encoder": nn.ModuleDict({"layers": layer_list})})
        config = LoraConfig(
            target_modules=["qkv"], layers_to_transform=layers, layers_pattern=pattern
        )
        assert _targets(config, model) == expected

    @pytest.mark.parametrize("layer", [0, 2])
    def test_a_pattern_takes_its_first_match(self, layer: int) -> None:
        def mlp() -> nn.ModuleDict:
            inner = nn.ModuleList([nn.ModuleDict({"fc": nn.Linear(2, 2)})])
            return nn.ModuleDict({"mlp": nn.ModuleDict({"layers": inner})})

        stack = [mlp() for _ in range(3)]
        model = nn.ModuleDict({"model": nn.ModuleDict({"layers": nn.ModuleList(stack)})})
        config = LoraConfig(
            target_modules=["fc"], layers_to_transform=[layer], layers_pattern="layers"
        )
        assert _targets(config, model) == [f"model.layers.{layer}.mlp.layers.0.fc"]

    def test_a_layer_pattern_that_is_absent_matches_nothing(self) -> None:
        config = LoraConfig(target_modules=["q_proj"], layers_to_transform=[0], layers_pattern="h")
        with pytest.raises(ValueError, match="layers_pattern"):
            LoraModel(_lm(), config)

    def test_a_layer_filter_on_an_unindexed_name_matches_nothing(self) -> None:
        model = nn.ModuleDict({"block": nn.ModuleDict({"proj": nn.Linear(2, 2)})})
        config = LoraConfig(target_modules=["proj"], layers_to_transform=[0])
        with pytest.raises(ValueError, match="layers_to_transform"):
            LoraModel(model, config)

    def test_an_exact_name_passes_the_layer_filter(self) -> None:
        config = LoraConfig(target_modules=["lm_head"], layers_to_transform=[0])
        assert _targets(config, _lm(tie=False)) == ["lm_head"]

    def test_rank_and_alpha_patterns(self) -> None:
        model = _lm()
        LoraModel(
            model,
            LoraConfig(
                r=2,
                lora_alpha=4,
                target_modules=["q_proj"],
                rank_pattern={"layers.1.q_proj": 6, r"layers\.2\.q_proj": 1},
                alpha_pattern={"q_proj": 12},
            ),
        )
        ranks = [model.layers[i].q_proj.r["default"] for i in range(3)]
        assert ranks == [2, 6, 1]
        assert [model.layers[i].q_proj.scaling["default"] for i in range(3)] == [6.0, 2.0, 12.0]


class TestAllLinear:
    def test_excludes_a_tied_head(self) -> None:
        names = _targets(LoraConfig(target_modules="all-linear"))
        assert "lm_head" not in names and len(names) == 9

    def test_excludes_an_untied_lm_head(self) -> None:
        names = _targets(LoraConfig(target_modules="ALL-LINEAR"), _lm(tie=False))
        assert "lm_head" not in names and len(names) == 9

    def test_excludes_any_linear_whose_weight_is_shared(self) -> None:
        model = _lm(tie=False)
        model.layers[0].mlp.weight = model.layers[1].mlp.weight
        names = _targets(LoraConfig(target_modules="all-linear"), model)
        assert "layers.0.mlp" not in names and "layers.1.mlp" not in names

    def test_a_sequence_classification_head_is_excluded(self) -> None:
        config = LoraConfig(target_modules="all-linear", task_type=TaskType.SEQ_CLS)
        assert _targets(config, _Classifier()) == ["encoder", "pre_classifier"]
        assert len(_targets(LoraConfig(target_modules="all-linear"), _Classifier())) == 3

    def test_get_output_embeddings_names_the_head(self) -> None:
        class WithGetter(_Classifier):
            def get_output_embeddings(self) -> nn.Module:
                return self.pre_classifier

        names = _targets(LoraConfig(target_modules="all-linear"), WithGetter())
        assert names == ["classifier", "encoder"]

    def test_the_stored_names_are_exactly_the_adapted_ones(self) -> None:
        config = LoraConfig(
            target_modules="all-linear", modules_to_save=["classifier"], exclude_modules=["encoder"]
        )
        lora = LoraModel(_Classifier(), config)
        assert lora.targeted_module_names == ["pre_classifier"]
        assert lora.peft_config["default"].target_modules == {"pre_classifier"}

    def test_the_stored_config_spells_out_the_names(self) -> None:
        config = LoraConfig(target_modules="all-linear")
        lora = LoraModel(_lm(), config)
        assert config.target_modules == "all-linear"
        assert lora.peft_config["default"].target_modules == set(lora.targeted_module_names)


class TestRefusals:
    def test_no_match_raises_and_leaves_the_model(self) -> None:
        model = _lm()
        before = list(model.state_dict())
        with pytest.raises(ValueError, match="not found in the base model"):
            LoraModel(model, LoraConfig(target_modules=["k_proj"]))
        assert list(model.state_dict()) == before

    def test_target_modules_is_required(self) -> None:
        with pytest.raises(ValueError, match="specify `target_modules`"):
            LoraModel(_lm(), LoraConfig())

    def test_a_non_linear_match_raises_before_anything_changes(self) -> None:
        model = _lm()
        with pytest.raises(ValueError, match=r"layers\.0 \(_Block\) is not supported"):
            LoraModel(model, LoraConfig(target_modules=r"layers\.0(\.q_proj)?"))
        assert type(model.layers[0].q_proj) is nn.Linear

    def test_a_module_type_with_no_layer_is_refused_naming_the_supported_ones(self) -> None:
        model = _lm()
        with pytest.raises(ValueError, match=r"norm \(LayerNorm\) is not supported.*nn.Embedding"):
            LoraModel(model, LoraConfig(target_modules=["layers.0.norm"]))
        assert type(model.layers[0].norm) is nn.LayerNorm

    def test_a_lazy_linear_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"0 \(LazyLinear\) is not supported"):
            LoraModel(nn.Sequential(nn.LazyLinear(4)), LoraConfig(target_modules=["0"]))

    def test_a_failure_while_planning_leaves_the_model(self) -> None:
        class Locked(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = nn.Linear(2, 2)
                self.b = nn.Linear(2, 2)
                self.c = nn.Linear(2, 2)
                self.b.lock = threading.Lock()  # cannot be deep-copied

        model = Locked()
        config = LoraConfig(target_modules=["c"], modules_to_save=["a", "b"])
        with pytest.raises(TypeError):
            LoraModel(model, config)
        assert all(p.requires_grad for p in model.parameters())
        assert type(model.a) is nn.Linear and type(model.c) is nn.Linear
        assert "peft_config" not in model.__dict__

    def test_double_injection_is_refused(self) -> None:
        model = _lm()
        LoraModel(model, LoraConfig(target_modules=["q_proj"]))
        with pytest.raises(NotImplementedError, match="already holds a LoRA adapter"):
            LoraModel(model, LoraConfig(target_modules=["v_proj"]))

    def test_modules_to_save_that_matches_nothing(self) -> None:
        config = LoraConfig(target_modules=["q_proj"], modules_to_save=["score"])
        with pytest.raises(ValueError, match="modules_to_save"):
            LoraModel(_lm(), config)

    @pytest.mark.parametrize(
        ("model", "config", "error"),
        [
            ("not a module", LoraConfig(target_modules=["q"]), TypeError),
            (nn.Linear(2, 2), {"target_modules": ["q"]}, TypeError),
        ],
    )
    def test_wrong_argument_types(
        self, model: object, config: object, error: type[Exception]
    ) -> None:
        with pytest.raises(error):
            LoraModel(model, config)  # type: ignore[arg-type]

    def test_peft_model_wants_a_lora_config(self) -> None:
        with pytest.raises(TypeError, match="LoraConfig"):
            PeftModel(_lm(), {"r": 8})  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="LoraConfig"):
            get_peft_model(_lm(), {"r": 8})  # type: ignore[arg-type]

    def test_fan_in_fan_out_warns_and_is_reset(self) -> None:
        with pytest.warns(UserWarning, match="fan_in_fan_out"):
            lora = LoraModel(_lm(), LoraConfig(target_modules=["q_proj"], fan_in_fan_out=True))
        assert lora.peft_config["default"].fan_in_fan_out is False


class TestTraining:
    def test_b_zero_model_is_the_base_bit_for_bit(self, device: str) -> None:
        model = _lm(device)
        expected = model(_ids(device))
        peft = get_peft_model(model, LoraConfig(target_modules=["q_proj", "v_proj", "mlp"]))
        assert _same(peft(_ids(device)), expected)

    def test_only_adapters_get_gradients(self, device: str) -> None:
        peft = get_peft_model(_lm(device), LoraConfig(target_modules=["q_proj", "v_proj"]))
        peft(_ids(device)).sum().backward()
        for name, param in peft.named_parameters():
            if ".lora_" in name:
                assert param.requires_grad and param.grad is not None, name
            else:
                assert not param.requires_grad and param.grad is None, name

    def test_sgd_changes_only_the_adapter(self, device: str) -> None:
        peft = get_peft_model(
            _lm(device), LoraConfig(target_modules=["q_proj"], init_lora_weights=False)
        )
        before = {name: p.clone() for name, p in peft.named_parameters()}
        opt = optim.SGD([p for p in peft.parameters() if p.requires_grad], lr=0.1)
        for _ in range(2):
            opt.zero_grad()
            peft(_ids(device)).square().mean().backward()
            opt.step()
        for name, param in peft.named_parameters():
            assert _same(param, before[name]) != (".lora_" in name), name

    @pytest.mark.parametrize(
        ("bias", "trainable"),
        [
            ("none", set()),
            ("lora_only", {"layers.0.q_proj.base_layer.bias", "layers.1.q_proj.base_layer.bias"}),
        ],
    )
    def test_bias_option(self, bias: str, trainable: set[str]) -> None:
        model = _lm(depth=2)
        LoraModel(model, LoraConfig(target_modules=["q_proj"], bias=bias))  # type: ignore[arg-type]
        found = {n for n, p in model.named_parameters() if p.requires_grad and ".lora_" not in n}
        assert found == trainable

    def test_bias_all_trains_every_bias(self) -> None:
        model = _lm(depth=1)
        LoraModel(model, LoraConfig(target_modules=["q_proj"], bias="all"))
        found = {n for n, p in model.named_parameters() if p.requires_grad and ".lora_" not in n}
        assert found == {
            "layers.0.q_proj.base_layer.bias",
            "layers.0.v_proj.bias",
            "layers.0.norm.bias",
            "layers.0.mlp.bias",
        }

    def test_inference_mode_freezes_everything(self) -> None:
        model = _lm()
        LoraModel(model, LoraConfig(target_modules=["q_proj"], inference_mode=True))
        assert not any(p.requires_grad for p in model.parameters())


class TestModulesToSave:
    def test_the_head_trains_as_a_copy(self, device: str) -> None:
        lucid.manual_seed(0)
        model = _Classifier().to(device)
        original = model.classifier
        peft = get_peft_model(
            model, LoraConfig(target_modules=["encoder"], modules_to_save=["classifier"])
        )
        wrapper = model.classifier
        assert wrapper.original_module is original and wrapper.active_adapter == "default"
        copy = wrapper.modules_to_save["default"]
        assert copy is not original and copy.weight is not original.weight
        assert copy.weight.requires_grad and not original.weight.requires_grad
        assert wrapper.out_features == 3 and wrapper.weight is copy.weight
        peft(lucid.randn(2, 8, device=device)).sum().backward()
        assert copy.weight.grad is not None and original.weight.grad is None

    def test_a_classification_task_trains_its_head(self) -> None:
        peft = get_peft_model(
            _Classifier(), LoraConfig(target_modules=["encoder"], task_type="SEQ_CLS")
        )
        assert peft.peft_config["default"].modules_to_save == ["classifier"]
        assert peft.get_base_model().classifier.modules_to_save["default"].weight.requires_grad

    def test_a_task_head_joins_user_entries(self) -> None:
        config = LoraConfig(
            target_modules=["encoder"], task_type="TOKEN_CLS", modules_to_save=["pre_classifier"]
        )
        peft = get_peft_model(_Classifier(), config)
        assert peft.peft_config["default"].modules_to_save == ["pre_classifier", "classifier"]
        assert config.modules_to_save == ["pre_classifier"]

    def test_a_task_without_a_head_adds_nothing(self) -> None:
        peft = get_peft_model(
            _lm(), LoraConfig(target_modules=["q_proj"], task_type=TaskType.QUESTION_ANS)
        )
        assert peft.peft_config["default"].modules_to_save is None

    def test_a_module_to_save_is_never_adapted(self) -> None:
        config = LoraConfig(target_modules=["mlp", "q_proj"], modules_to_save=["layers.1"])
        names = _targets(config)
        assert not any(name.startswith("layers.1.") for name in names) and len(names) == 4


class TestDisableAndUnload:
    def test_disable_adapter_gives_the_base_output(self, device: str) -> None:
        lucid.manual_seed(0)
        model = _Classifier().to(device)
        x = lucid.randn(2, 8, device=device)
        expected = model(x)
        peft = get_peft_model(
            model,
            LoraConfig(
                target_modules=["encoder"], modules_to_save=["classifier"], init_lora_weights=False
            ),
        )
        assert not _same(peft(x), expected)
        with peft.disable_adapter():
            assert model.classifier.disable_adapters
            assert _same(peft(x), expected)
        assert not _same(peft(x), expected)

    def test_training_the_copy_leaves_the_original(self, device: str) -> None:
        lucid.manual_seed(0)
        model = _Classifier().to(device)
        x = lucid.randn(2, 8, device=device)
        expected = model(x)
        peft = get_peft_model(
            model, LoraConfig(target_modules=["encoder"], modules_to_save=["classifier"])
        )
        original = model.classifier.original_module.weight.clone()
        opt = optim.SGD([p for p in peft.parameters() if p.requires_grad], lr=0.5)
        peft(x).square().sum().backward()
        opt.step()
        assert _same(model.classifier.original_module.weight, original)
        assert not _same(model.classifier.modules_to_save["default"].weight, original)
        with peft.disable_adapter():
            assert _same(peft(x), expected)
        assert _same(peft.unload()(x), expected)

    def test_disable_restores_after_an_exception(self) -> None:
        peft = get_peft_model(_lm(), LoraConfig(target_modules=["q_proj"]))
        with pytest.raises(KeyError), peft.disable_adapter():
            raise KeyError("x")
        assert not peft.get_base_model().layers[0].q_proj.disable_adapters

    def test_disabling_with_trained_biases_warns(self) -> None:
        peft = get_peft_model(_lm(), LoraConfig(target_modules=["q_proj"], bias="all"))
        with pytest.warns(UserWarning, match="bias configured"), peft.disable_adapter():
            pass

    def test_unload_returns_the_original_modules(self, device: str) -> None:
        model = _lm(device, tie=False)
        originals = [block.q_proj for block in model.layers]
        x = _ids(device)
        expected = model(x)
        peft = get_peft_model(model, LoraConfig(target_modules=["q_proj"], init_lora_weights=False))
        peft.base_model.merge_adapter()
        restored = peft.unload()
        assert restored is model
        assert [block.q_proj for block in model.layers] == originals
        assert all(type(block.q_proj) is nn.Linear for block in model.layers)
        assert bool(lucid.allclose(model(x), expected, atol=1e-5))
        assert all(p.requires_grad for p in model.parameters())
        assert "peft_config" not in model.__dict__

    def test_merge_and_unload_folds_the_adapter_in(self, device: str) -> None:
        model = _lm(device)
        peft = get_peft_model(
            model, LoraConfig(target_modules=["q_proj", "v_proj"], init_lora_weights=False)
        )
        x = _ids(device)
        adapted = peft(x)
        merged = peft.merge_and_unload(safe_merge=True)
        assert merged is model and not any(isinstance(m, LoraLayer) for m in model.modules())
        assert bool(lucid.allclose(model(x), adapted, atol=1e-4))

    def test_a_deep_copy_unloads_with_its_flags(self) -> None:
        model = _lm(tie=False)
        model.embed.weight.requires_grad_(False)
        peft = get_peft_model(model, LoraConfig(target_modules=["q_proj"]))
        restored = copy.deepcopy(peft).unload()
        flags = {name: p.requires_grad for name, p in restored.named_parameters()}
        assert flags.pop("embed.weight") is False and all(flags.values())

    def test_merge_and_unload_keeps_the_trained_copy(self) -> None:
        model = _Classifier()
        peft = get_peft_model(
            model, LoraConfig(target_modules=["encoder"], modules_to_save=["classifier"])
        )
        trained = model.classifier.modules_to_save["default"]
        peft.merge_and_unload()
        assert model.classifier is trained

    def test_unload_keeps_the_original_copy(self) -> None:
        model = _Classifier()
        original = model.classifier
        peft = get_peft_model(
            model, LoraConfig(target_modules=["encoder"], modules_to_save=["classifier"])
        )
        peft.unload()
        assert model.classifier is original and original.weight.requires_grad

    def test_a_tied_module_trained_in_full_cannot_be_merged_away(self) -> None:
        model = _lm()
        with pytest.warns(UserWarning, match="shares its weight with embed.weight"):
            peft = get_peft_model(
                model, LoraConfig(target_modules=["q_proj"], modules_to_save=["lm_head"])
            )
        with pytest.raises(RuntimeError, match="trained apart from embed.weight"):
            peft.merge_and_unload()
        assert isinstance(model.layers[0].q_proj, Linear) and not model.layers[0].q_proj.merged
        assert peft.unload() is model and model.lm_head.weight is model.embed.weight

    def test_merging_into_a_tied_weight_is_refused(self) -> None:
        model = _lm()
        peft = get_peft_model(model, LoraConfig(target_modules=["lm_head"]))
        before = model.embed.weight.clone()
        with pytest.raises(RuntimeError, match="embed.weight"):
            peft.merge_and_unload()
        assert _same(model.embed.weight, before) and isinstance(model.lm_head, Linear)

    def test_safe_merge_checks_every_layer_before_writing(self) -> None:
        model = _lm()
        peft = get_peft_model(model, LoraConfig(target_modules=["q_proj"], init_lora_weights=False))
        first = model.layers[0].q_proj.base_layer.weight.clone()
        with lucid.no_grad():
            model.layers[2].q_proj.lora_B["default"].weight.fill_(float("inf"))
        with pytest.raises(ValueError, match="NaNs detected"):
            peft.base_model.merge_adapter(safe_merge=True)
        assert _same(model.layers[0].q_proj.base_layer.weight, first)

    def test_unmerge_adapter(self) -> None:
        model = _lm()
        peft = get_peft_model(model, LoraConfig(target_modules=["q_proj"], init_lora_weights=False))
        peft.base_model.merge_adapter()
        assert model.layers[0].q_proj.merged
        peft.base_model.unmerge_adapter()
        assert not model.layers[0].q_proj.merged


class TestWrapper:
    def test_public_names_reach_the_model(self) -> None:
        model = _lm()
        peft = get_peft_model(model, LoraConfig(target_modules=["q_proj"]))
        assert peft.lm_head is model.lm_head and peft.base_model.lm_head is model.lm_head
        assert peft.get_base_model() is model and peft.base_model.model is model
        assert peft.active_adapter == "default" and peft.base_model.active_adapter == "default"
        with pytest.raises(AttributeError):
            _ = peft.no_such_name
        with pytest.raises(AttributeError):
            _ = peft._private_name

    def test_state_dict_has_the_reference_prefix(self) -> None:
        peft = get_peft_model(_lm(depth=1), LoraConfig(target_modules=["q_proj"]))
        keys = list(peft.state_dict())
        assert "base_model.model.layers.0.q_proj.lora_A.default.weight" in keys
        assert "base_model.model.layers.0.q_proj.base_layer.weight" in keys
        assert all(k.startswith("base_model.model.") for k in keys)

    def test_the_base_parameters_are_the_model_s_own(self) -> None:
        model = _lm()
        ids = {id(p) for p in model.parameters()}
        peft = get_peft_model(model, LoraConfig(target_modules=["q_proj"]))
        base = {id(p) for n, p in peft.named_parameters() if ".lora_" not in n}
        assert base == ids

    def test_trainable_counts(self) -> None:
        peft = get_peft_model(_lm(depth=1), LoraConfig(r=2, target_modules=["q_proj"]))
        trainable, total = peft.get_nb_trainable_parameters()
        assert trainable == 2 * 8 + 8 * 2
        assert total == sum(p.numel() for p in peft.parameters())

    def test_print_trainable_parameters(self, capsys: pytest.CaptureFixture[str]) -> None:
        peft = get_peft_model(_lm(depth=1), LoraConfig(r=2, target_modules=["q_proj"]))
        peft.print_trainable_parameters()
        assert capsys.readouterr().out.startswith("trainable params: 32 || all params: ")

    def test_name_or_path_fills_the_config(self) -> None:
        model = _lm()
        model.name_or_path = "tiny-lm"
        peft = get_peft_model(model, LoraConfig(target_modules=["q_proj"]))
        assert peft.peft_config["default"].base_model_name_or_path == "tiny-lm"

    def test_inject_returns_the_bare_model(self) -> None:
        model = _lm()
        assert inject_adapter_in_model(LoraConfig(target_modules=["q_proj"]), model) is model
        assert isinstance(model.layers[0].q_proj, Linear)
        assert model.peft_config["default"].target_modules == {"q_proj"}

    def test_lora_model_forward_runs_the_model(self) -> None:
        model = _lm()
        lora = LoraModel(model, LoraConfig(target_modules=["q_proj"]))
        assert _same(lora(_ids()), model(_ids()))
