"""LN tuning: a trainable copy per adapter, merged by exchanging values, and the reference's numbers.

The method adds no tensor of its own: an adapter is a copy of the wrapped module.
These tests pin what follows from that: the copy starts equal and trains apart, one
adapter applies (and merges) at a time, merging writes the base's own ``Parameter``\\ s
and unmerging gives the originals back, and a state dict reads as the reference's.
"""

from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
from lucid.models import create_model
from lucid.peft import (
    LNTuningConfig,
    LNTuningModel,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
)
from lucid.peft.tuners import ln_tuning
from lucid.peft.tuners.ln_tuning._defaults import LN_TUNING_DEFAULT_TARGET_MODULES
from lucid.test.unit.peft._parity import adapter_library, gap


def _layer(base: nn.Module | None = None) -> tuple[ln_tuning.LNTuningLayer, nn.Module]:
    lucid.manual_seed(1)
    wrapped = nn.LayerNorm(4) if base is None else base
    return ln_tuning.LNTuningLayer(
        wrapped, "default", LNTuningConfig(target_modules=["x"])
    ), wrapped


def _tune(layer: ln_tuning.LNTuningLayer, adapter: str = "default") -> None:
    with lucid.no_grad():
        for p in layer.ln_tuning_layers[adapter].parameters():
            p.copy_(p * 1.5 + 0.25)


class TestConfig:
    def test_the_fields_are_the_references(self) -> None:
        config = LNTuningConfig(target_modules=["a"])
        assert config.exclude_modules is None and config.modules_to_save is None
        assert set(config.to_dict()) == {
            *("auto_mapping", "base_model_name_or_path", "exclude_modules", "inference_mode"),
            *("modules_to_save", "peft_type", "revision", "target_modules", "task_type"),
        }

    def test_layer_selection_is_an_extension_written_only_when_used(self) -> None:
        plain = LNTuningConfig(target_modules=["a"]).to_dict()
        assert "layers_to_transform" not in plain
        chosen = LNTuningConfig(
            target_modules=["a"], layers_to_transform=[0, 2], layers_pattern="h"
        )
        assert chosen.to_dict()["layers_to_transform"] == [0, 2]
        assert LNTuningConfig.from_dict(chosen.to_dict()) == chosen

    def test_a_file_round_trips(self) -> None:
        config = LNTuningConfig(target_modules=["ln_1", "ln_2"], exclude_modules=["ln_f"])
        assert LNTuningConfig.from_dict(config.to_dict()) == config
        assert config.to_dict()["peft_type"] == "LN_TUNING"


class TestLayer:
    def test_the_copy_starts_equal_to_the_base_and_trains_apart(self) -> None:
        layer, base = _layer()
        copy = layer.ln_tuning_layers["default"]
        assert copy is not base and copy.weight is not base.weight
        x = lucid.randn(3, 4)
        assert bool((layer(x) == base(x)).all().item())
        _tune(layer)
        assert not bool(lucid.allclose(layer(x), base(x), atol=1e-3))
        assert bool((base.weight == 1).all().item())  # the base is untouched

    def test_the_tuned_copy_is_what_the_layer_computes(self) -> None:
        layer, _ = _layer()
        _tune(layer)
        x = lucid.randn(3, 4)
        assert bool((layer(x) == layer.ln_tuning_layers["default"](x)).all().item())

    def test_merging_writes_the_bases_own_parameters_and_unmerging_gives_them_back(self) -> None:
        layer, base = _layer()
        weight, bias = base.weight, base.bias
        _tune(layer)
        x = lucid.randn(3, 4)
        with lucid.no_grad():
            tuned = layer(x)
            layer.merge()
            assert layer.merged and base.weight is weight and base.bias is bias
            assert bool(lucid.allclose(layer(x), tuned, atol=1e-6))
            assert bool((weight == 1.75).all().item())
            layer.unmerge()
        assert bool((weight == 1).all().item()) and bool((bias == 0).all().item())
        assert not layer.merged

    def test_buffers_are_exchanged_with_the_parameters(self) -> None:
        layer, base = _layer(nn.BatchNorm1d(3))
        copy = layer.ln_tuning_layers["default"]
        with lucid.no_grad():
            copy.running_mean.copy_(lucid.tensor([1.0, 2.0, 3.0]))
            layer.merge()
        assert base.running_mean.tolist() == [1.0, 2.0, 3.0]
        with lucid.no_grad():
            layer.unmerge()
        assert base.running_mean.tolist() == [0.0, 0.0, 0.0]

    def test_a_non_finite_copy_is_not_merged_when_the_merge_is_safe(self) -> None:
        layer, base = _layer()
        with lucid.no_grad():
            layer.ln_tuning_layers["default"].weight.copy_(lucid.tensor([float("nan")] * 4))
        with pytest.raises(ValueError, match="NaNs detected"):
            layer.merge(safe_merge=True)
        assert bool((base.weight == 1).all().item()) and not layer.merged

    def test_one_adapter_merges_at_a_time(self) -> None:
        layer, base = _layer()
        layer.update_layer("other", LNTuningConfig(target_modules=["x"]))
        _tune(layer, "other")
        with pytest.raises(ValueError, match="more than one adapter"):
            layer.merge(adapter_names=["default", "other"])
        layer.merge(adapter_names=["default"])
        with pytest.warns(UserWarning, match="Unmerging first"):
            layer.merge(adapter_names=["other"])
        assert layer.merged_adapters == ["other"] and bool((base.weight == 1.75).all().item())

    def test_a_new_adapter_copies_the_original_even_when_another_is_merged(self) -> None:
        layer, base = _layer()
        _tune(layer)
        layer.merge()
        layer.update_layer("late", LNTuningConfig(target_modules=["x"]))
        assert bool((layer.ln_tuning_layers["late"].weight == 1).all().item())

    def test_two_active_adapters_cannot_run_together(self) -> None:
        layer, _ = _layer()
        layer.update_layer("other", LNTuningConfig(target_modules=["x"]))
        layer.set_adapter(["default", "other"])
        with pytest.raises(ValueError, match="more than one adapter"):
            layer(lucid.randn(2, 4))

    def test_the_delta_weight_is_the_tuned_weight_minus_the_original(self) -> None:
        layer, _ = _layer()
        _tune(layer)
        assert bool((layer.get_delta_weight("default") == 0.75).all().item())
        layer.merge()
        assert bool((layer.get_delta_weight("default") == 0.75).all().item())


class TestModel:
    def test_any_module_can_be_tuned_and_only_the_copy_trains(self) -> None:
        net = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), nn.RMSNorm(4))
        peft = get_peft_model(net, LNTuningConfig(target_modules=["1", "2"]))
        assert isinstance(peft, PeftModel) and isinstance(peft.base_model, LNTuningModel)
        trained = sorted(n for n, p in peft.named_parameters() if p.requires_grad)
        assert trained == [
            "base_model.model.1.ln_tuning_layers.default.bias",
            "base_model.model.1.ln_tuning_layers.default.weight",
            "base_model.model.2.ln_tuning_layers.default.weight",
        ]

    def test_the_zoo_defaults_name_the_norms_of_gpt2_and_t5(self) -> None:
        gpt2 = create_model(
            "gpt2_small_lm",
            **{
                "vocab_size": 16,
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
            },
            **{"intermediate_size": 16, "max_position_embeddings": 8},
        )
        peft = get_peft_model(gpt2, LNTuningConfig())
        names = peft.base_model.targeted_module_names
        assert sorted(names) == ["transformer.h.0.ln_1", "transformer.h.0.ln_2", "transformer.ln_f"]
        assert set(LN_TUNING_DEFAULT_TARGET_MODULES) == {"gpt2", "t5"}

    def test_a_model_type_without_a_row_asks_for_target_modules(self) -> None:
        bert = create_model(
            "bert_base_cls",
            **{
                "vocab_size": 16,
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
            },
            **{"intermediate_size": 16, "max_position_embeddings": 8},
        )
        with pytest.raises(ValueError, match="Please specify `target_modules`"):
            get_peft_model(bert, LNTuningConfig())

    def test_unload_gives_the_originals_back_and_merge_and_unload_the_tuned_ones(self) -> None:
        net = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        norm = net[1]
        peft = get_peft_model(net, LNTuningConfig(target_modules=["1"]))
        _tune(peft.base_model.model[1])
        merged = peft.merge_and_unload()
        assert merged[1] is norm and bool((norm.weight == 1.75).all().item())


# ── the reference adapter library ────────────────────────────────────────────


@pytest.mark.parity
def test_a_layer_matches_the_reference(ref: ModuleType) -> None:
    library = adapter_library()
    lucid.manual_seed(8)
    net = nn.Sequential(nn.Linear(6, 6), nn.LayerNorm(6), nn.Linear(6, 2))
    twin = ref.nn.Sequential(ref.nn.Linear(6, 6), ref.nn.LayerNorm(6), ref.nn.Linear(6, 2))
    peft = get_peft_model(net, LNTuningConfig(target_modules=["1"]))
    with ref.no_grad():
        for i in (0, 2):
            twin[i].weight.copy_(ref.tensor(peft.base_model.model[i].weight.tolist()))
            twin[i].bias.copy_(ref.tensor(peft.base_model.model[i].bias.tolist()))
    theirs = library.get_peft_model(twin, library.LNTuningConfig(target_modules=["1"]))
    mine = peft.base_model.model[1]
    other = theirs.base_model.model[1]
    _tune(mine)
    with ref.no_grad():
        for name in ("weight", "bias"):
            getattr(other.ln_tuning_layers["default"], name).copy_(
                ref.tensor(getattr(mine.ln_tuning_layers["default"], name).tolist())
            )
    x = lucid.randn(3, 6)
    assert gap(ref, peft(x), theirs(ref.tensor(x.tolist()))) < 1e-5
    assert sorted(library.get_peft_model_state_dict(theirs)) == sorted(
        get_peft_model_state_dict(peft)
    )
    peft.base_model.merge_adapter()
    theirs.base_model.merge_adapter()
    assert gap(ref, peft(x), theirs(ref.tensor(x.tolist()))) < 1e-5


@pytest.mark.parity
def test_the_config_file_has_the_references_keys() -> None:
    library = adapter_library()
    ours = LNTuningConfig(target_modules=["a"]).to_dict()
    theirs = library.LNTuningConfig(target_modules=["a"]).to_dict()
    assert set(ours) == set(theirs) - {"peft_version"}
