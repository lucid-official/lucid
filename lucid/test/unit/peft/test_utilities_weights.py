"""``get_base_model_state_dict`` / ``set_base_model_state_dict`` and ``load_peft_weights``."""

import json
import os

import pytest

import lucid
import lucid.nn as nn
from lucid.models import create_model
from lucid.peft import (
    LoraConfig,
    PromptTuningConfig,
    TaskType,
    get_base_model_state_dict,
    get_peft_model,
    get_peft_model_state_dict,
    load_peft_weights,
    set_base_model_state_dict,
)

_TEXT: dict[str, object] = {
    "vocab_size": 50,
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 32,
    "max_position_embeddings": 64,
    "hidden_dropout": 0.0,
    "attention_dropout": 0.0,
}


def _net() -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))


def _same(a: dict[str, lucid.Tensor], b: dict[str, lucid.Tensor]) -> bool:
    return a.keys() == b.keys() and all(a[k].tolist() == b[k].tolist() for k in a)


class TestBaseModelStateDict:
    def test_a_lora_model_gives_back_the_original_names_and_values(self) -> None:
        net = _net()
        original = {k: v.clone() for k, v in net.state_dict().items()}
        peft = get_peft_model(net, LoraConfig(r=2, target_modules=["0", "1"]))
        assert any("base_layer" in k for k in peft.state_dict())
        assert _same(get_base_model_state_dict(peft), original)

    def test_modules_to_save_gives_the_original_not_the_copy(self) -> None:
        net = _net()
        original = {k: v.clone() for k, v in net.state_dict().items()}
        peft = get_peft_model(net, LoraConfig(r=2, target_modules=["0"], modules_to_save=["1"]))
        peft.get_base_model()[1].modules_to_save["default"].weight.data = lucid.zeros(2, 4)
        assert _same(get_base_model_state_dict(peft), original)

    def test_the_adapter_never_leaks_in(self) -> None:
        peft = get_peft_model(_net(), LoraConfig(r=2, target_modules=["0"]))
        assert not [k for k in get_base_model_state_dict(peft) if "lora_" in k]

    def test_a_module_named_base_layer_is_not_an_infix(self) -> None:
        class Holder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base_layer = nn.Linear(3, 3)
                self.head = nn.Linear(3, 3)

        model = Holder()
        peft = get_peft_model(model, LoraConfig(r=2, target_modules=["head"]))
        assert sorted(get_base_model_state_dict(peft)) == [
            "base_layer.bias",
            "base_layer.weight",
            "head.bias",
            "head.weight",
        ]

    def test_the_tensors_are_the_models_own(self) -> None:
        peft = get_peft_model(_net(), LoraConfig(r=2, target_modules=["0"]))
        got = get_base_model_state_dict(peft)
        peft.get_base_model()[0].base_layer.weight.data.zero_()
        assert float(got["0.weight"].abs().sum()) == 0.0

    def test_a_prompt_learning_model_is_all_base(self) -> None:
        model = create_model("gpt2_small_lm", use_cache=False, **_TEXT)
        names = set(model.state_dict())
        peft = get_peft_model(
            model, PromptTuningConfig(num_virtual_tokens=3, task_type=TaskType.CAUSAL_LM)
        )
        assert set(get_base_model_state_dict(peft)) == names

    def test_it_refuses_a_bare_module(self) -> None:
        with pytest.raises(TypeError, match="takes a PeftModel"):
            get_base_model_state_dict(_net())  # type: ignore[arg-type]  # the point of the test

    def test_a_second_adapter_changes_nothing(self) -> None:
        net = _net()
        original = {k: v.clone() for k, v in net.state_dict().items()}
        peft = get_peft_model(net, LoraConfig(r=2, target_modules=["0"]))
        peft.add_adapter("other", LoraConfig(r=3, target_modules=["0", "1"]))
        assert _same(get_base_model_state_dict(peft), original)


class TestSetBaseModelStateDict:
    def test_it_loads_original_names_into_the_wrapped_places(self) -> None:
        peft = get_peft_model(_net(), LoraConfig(r=2, target_modules=["0", "1"]))
        zeros = {k: lucid.zeros(*v.shape) for k, v in get_base_model_state_dict(peft).items()}
        result = set_base_model_state_dict(peft, zeros)
        assert result.missing_keys == [] and result.unexpected_keys == []
        layers = peft.get_base_model()
        assert float(layers[0].base_layer.weight.abs().sum()) == 0.0
        assert float(layers[1].base_layer.bias.abs().sum()) == 0.0

    def test_the_adapter_is_left_alone(self) -> None:
        peft = get_peft_model(
            _net(), LoraConfig(r=2, target_modules=["0"], init_lora_weights=False)
        )
        adapter = {k: v.clone() for k, v in get_peft_model_state_dict(peft).items()}
        base = get_base_model_state_dict(peft)
        set_base_model_state_dict(peft, {k: lucid.zeros(*v.shape) for k, v in base.items()})
        assert _same(get_peft_model_state_dict(peft), adapter)

    def test_get_then_set_is_the_identity(self) -> None:
        peft = get_peft_model(_net(), LoraConfig(r=2, target_modules=["0"], modules_to_save=["1"]))
        before = {k: v.clone() for k, v in get_base_model_state_dict(peft).items()}
        set_base_model_state_dict(peft, before)
        assert _same(get_base_model_state_dict(peft), before)

    def test_modules_to_save_loads_into_the_original(self) -> None:
        peft = get_peft_model(_net(), LoraConfig(r=2, target_modules=["0"], modules_to_save=["1"]))
        base = get_base_model_state_dict(peft)
        set_base_model_state_dict(peft, {k: lucid.ones(*v.shape) for k, v in base.items()})
        head = peft.get_base_model()[1]
        assert float(head.original_module.weight.sum()) == 8.0

    def test_strict_raises_on_missing_and_unexpected_and_loads_nothing(self) -> None:
        peft = get_peft_model(_net(), LoraConfig(r=2, target_modules=["0"]))
        base = get_base_model_state_dict(peft)
        before = {k: v.clone() for k, v in base.items()}
        partial = {k: lucid.zeros(*v.shape) for k, v in base.items() if k != "1.bias"}
        partial["ghost"] = lucid.zeros(1)
        with pytest.raises(RuntimeError, match=r"Missing key\(s\).*1\.bias") as hit:
            set_base_model_state_dict(peft, partial)
        assert "ghost" in str(hit.value)
        assert _same(get_base_model_state_dict(peft), before)

    def test_not_strict_reports_and_loads_what_matches(self) -> None:
        peft = get_peft_model(_net(), LoraConfig(r=2, target_modules=["0"]))
        base = get_base_model_state_dict(peft)
        partial = {"0.weight": lucid.zeros(*base["0.weight"].shape), "ghost": lucid.zeros(1)}
        result = set_base_model_state_dict(peft, partial, strict=False)
        assert sorted(result.missing_keys) == ["0.bias", "1.bias", "1.weight"]
        assert result.unexpected_keys == ["ghost"]
        assert float(peft.get_base_model()[0].base_layer.weight.abs().sum()) == 0.0

    def test_a_prompt_learning_model_loads_directly(self) -> None:
        model = create_model("gpt2_small_lm", use_cache=False, **_TEXT)
        peft = get_peft_model(
            model, PromptTuningConfig(num_virtual_tokens=3, task_type=TaskType.CAUSAL_LM)
        )
        base = get_base_model_state_dict(peft)
        key = next(iter(base))
        set_base_model_state_dict(peft, {k: lucid.zeros(*v.shape) for k, v in base.items()})
        assert float(model.state_dict()[key].abs().sum()) == 0.0

    def test_it_refuses_a_bare_module(self) -> None:
        with pytest.raises(TypeError, match="takes a PeftModel"):
            set_base_model_state_dict(_net(), {})  # type: ignore[arg-type]  # the point of the test


class TestLoadPeftWeights:
    @pytest.fixture
    def saved(self, tmp_path: os.PathLike[str]) -> tuple[str, dict[str, lucid.Tensor]]:
        peft = get_peft_model(
            _net(), LoraConfig(r=2, target_modules=["0", "1"], init_lora_weights=False)
        )
        directory = os.fspath(tmp_path)
        peft.save_pretrained(directory)
        return directory, get_peft_model_state_dict(peft)

    def test_it_reads_the_file_keys_and_values(
        self, saved: tuple[str, dict[str, lucid.Tensor]]
    ) -> None:
        directory, state = saved
        weights = load_peft_weights(directory)
        assert sorted(weights) == [
            "base_model.model.0.lora_A.weight",
            "base_model.model.0.lora_B.weight",
            "base_model.model.1.lora_A.weight",
            "base_model.model.1.lora_B.weight",
        ]
        assert weights["base_model.model.0.lora_A.weight"].tolist() == (
            state["base_model.model.0.lora_A.weight"].tolist()
        )

    def test_the_device_moves_the_tensors(self, saved: tuple[str, dict[str, lucid.Tensor]]) -> None:
        weights = load_peft_weights(saved[0], device="metal")
        assert {w.device.type for w in weights.values()} == {"metal"}

    def test_key_mapping_renames_by_the_first_matching_pattern(
        self, saved: tuple[str, dict[str, lucid.Tensor]]
    ) -> None:
        mapping = {r"^0\.": "first.", r"^(\d)\.": r"layer\1."}
        weights = load_peft_weights(saved[0], key_mapping=mapping)
        assert sorted(weights) == [
            "base_model.model.first.lora_A.weight",
            "base_model.model.first.lora_B.weight",
            "base_model.model.layer1.lora_A.weight",
            "base_model.model.layer1.lora_B.weight",
        ]

    def test_subfolder_names_a_directory_under_model_id(self, tmp_path: os.PathLike[str]) -> None:
        peft = get_peft_model(_net(), LoraConfig(r=2, target_modules=["0"]))
        peft.save_pretrained(os.fspath(tmp_path), selected_adapters=["default"])
        nested = os.path.join(os.fspath(tmp_path), "inner")
        os.makedirs(nested)
        for name in os.listdir(os.fspath(tmp_path)):
            if name.startswith("adapter_"):
                os.replace(os.path.join(os.fspath(tmp_path), name), os.path.join(nested, name))
        assert sorted(load_peft_weights(os.fspath(tmp_path), subfolder="inner"))

    def test_hub_options_are_refused_by_name(
        self, saved: tuple[str, dict[str, lucid.Tensor]]
    ) -> None:
        with pytest.raises(NotImplementedError, match="`revision`"):
            load_peft_weights(saved[0], revision="main")

    def test_a_directory_without_weights(self, tmp_path: os.PathLike[str]) -> None:
        with pytest.raises(FileNotFoundError, match="adapter_model.safetensors"):
            load_peft_weights(os.fspath(tmp_path))

    def test_a_pickle_is_refused_with_the_reason(self, tmp_path: os.PathLike[str]) -> None:
        directory = os.fspath(tmp_path)
        with open(os.path.join(directory, "adapter_model.bin"), "wb") as fh:
            fh.write(b"x")
        with open(os.path.join(directory, "adapter_config.json"), "w", encoding="utf-8") as fh:
            json.dump({"peft_type": "LORA"}, fh)
        with pytest.raises(NotImplementedError, match="pickle"):
            load_peft_weights(directory)

    def test_a_non_string_subfolder_is_a_type_error(self, tmp_path: os.PathLike[str]) -> None:
        with pytest.raises(TypeError, match="subfolder"):
            load_peft_weights(os.fspath(tmp_path), subfolder=3)
