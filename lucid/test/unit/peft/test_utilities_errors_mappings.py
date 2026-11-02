"""The error classes, the two method tables, ``get_peft_config`` and the bloom prompt reshaper."""

import operator
import types
import warnings

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    PEFT_TYPE_TO_CONFIG_MAPPING,
    PEFT_TYPE_TO_TUNER_MAPPING,
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    IA3Config,
    LoraConfig,
    LoraModel,
    NoMatchingPeftModuleError,
    PeftConfig,
    PeftError,
    PeftType,
    PeftWarning,
    bloom_model_postprocess_past_key_value,
    get_peft_config,
    get_peft_model,
)
from lucid.peft._config import _CONFIG_CLASSES
from lucid.peft.tuners._base import _TUNERS


def _net() -> nn.Sequential:
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))


_BASE_CLAIMED = pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="LCD-821: tuners/_base.py (_not_found, _plan_modules_to_save) is claimed by LCD-829; "
    "the one-line retrofit to NoMatchingPeftModuleError follows its land",
)


class TestErrors:
    def test_hierarchy(self) -> None:
        assert issubclass(PeftError, Exception)
        assert issubclass(NoMatchingPeftModuleError, PeftError)
        assert issubclass(NoMatchingPeftModuleError, ValueError)
        assert issubclass(PeftWarning, UserWarning)

    def test_warning_is_filterable_as_a_user_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            warnings.warn("x", PeftWarning)
        assert [w.category for w in caught] == [PeftWarning]

    @_BASE_CLAIMED
    def test_missing_target_modules_raise_the_class_and_keep_value_error(self) -> None:
        net = _net()
        before = list(net.state_dict())
        with pytest.raises(NoMatchingPeftModuleError, match="not found in the base model") as hit:
            get_peft_model(net, LoraConfig(target_modules=["absent"]))
        assert isinstance(hit.value, ValueError) and isinstance(hit.value, PeftError)
        assert list(net.state_dict()) == before

    @_BASE_CLAIMED
    def test_the_layer_filters_are_named_in_the_message(self) -> None:
        config = LoraConfig(target_modules=["absent"], layers_to_transform=[0], layers_pattern="x")
        with pytest.raises(NoMatchingPeftModuleError, match="layers_to_transform"):
            get_peft_model(_net(), config)

    def test_an_except_value_error_still_catches_it(self) -> None:
        with pytest.raises(ValueError):
            LoraModel(_net(), LoraConfig(target_modules=["absent"]), adapter_name="default")

    @_BASE_CLAIMED
    def test_modules_to_save_that_matches_nothing(self) -> None:
        config = LoraConfig(target_modules=["0"], modules_to_save=["absent"])
        with pytest.raises(NoMatchingPeftModuleError, match="modules_to_save"):
            get_peft_model(_net(), config)

    def test_target_parameters_that_match_nothing(self) -> None:
        config = LoraConfig(target_parameters=["absent.w"])
        with pytest.raises(NoMatchingPeftModuleError, match="Target parameters"):
            get_peft_model(_net(), config)

    def test_other_value_errors_stay_plain(self) -> None:
        with pytest.raises(ValueError) as hit:
            get_peft_model(_net(), LoraConfig())
        assert not isinstance(hit.value, NoMatchingPeftModuleError)


class TestMappings:
    def test_they_cover_every_method(self) -> None:
        assert set(PEFT_TYPE_TO_CONFIG_MAPPING) == set(PeftType)
        assert set(PEFT_TYPE_TO_TUNER_MAPPING) == set(PeftType)

    def test_a_key_is_found_by_its_string(self) -> None:
        assert PEFT_TYPE_TO_CONFIG_MAPPING["LORA"] is LoraConfig
        assert PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.IA3] is IA3Config
        assert PEFT_TYPE_TO_TUNER_MAPPING["LORA"] is LoraModel

    def test_each_tuner_takes_the_config_of_its_key(self) -> None:
        for peft_type, tuner in PEFT_TYPE_TO_TUNER_MAPPING.items():
            assert PEFT_TYPE_TO_CONFIG_MAPPING[peft_type] is tuner._config_class

    def test_they_are_read_only_views(self) -> None:
        for table in (PEFT_TYPE_TO_CONFIG_MAPPING, PEFT_TYPE_TO_TUNER_MAPPING):
            assert isinstance(table, types.MappingProxyType)
            with pytest.raises(TypeError):
                operator.setitem(table, "LORA", object())

    def test_they_are_views_of_the_registries_not_copies(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(_CONFIG_CLASSES, PeftType.IA3, LoraConfig)
        monkeypatch.setitem(_TUNERS, PeftType.IA3, LoraModel)
        assert PEFT_TYPE_TO_CONFIG_MAPPING[PeftType.IA3] is LoraConfig
        assert PEFT_TYPE_TO_TUNER_MAPPING[PeftType.IA3] is LoraModel


class TestGetPeftConfig:
    def test_it_builds_the_class_the_type_names(self) -> None:
        config = get_peft_config({"peft_type": "LORA", "r": 4, "target_modules": ["q"]})
        assert type(config) is LoraConfig
        assert config.r == 4 and config.target_modules == {"q"}

    def test_a_config_round_trips_through_its_dict(self) -> None:
        for original in (
            LoraConfig(r=3, lora_alpha=5, target_modules=["a", "b"], task_type="CAUSAL_LM"),
            IA3Config(target_modules=["a"], feedforward_modules=["a"]),
        ):
            assert get_peft_config(original.to_dict()) == original

    def test_the_result_is_a_peft_config(self) -> None:
        assert isinstance(
            get_peft_config({"peft_type": PeftType.IA3, "target_modules": ["a"]}), PeftConfig
        )

    def test_a_reference_key_that_turns_nothing_on_is_dropped(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = get_peft_config(
                {"peft_type": "LORA", "target_modules": ["q"], "loftq_config": {}}
            )
        assert type(config) is LoraConfig

    def test_a_key_that_turns_a_missing_feature_on_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="use_qalora"):
            get_peft_config({"peft_type": "LORA", "target_modules": ["q"], "use_qalora": True})

    def test_no_peft_type_is_a_key_error(self) -> None:
        with pytest.raises(KeyError, match="peft_type"):
            get_peft_config({"r": 4})

    def test_an_unknown_peft_type_is_a_key_error(self) -> None:
        with pytest.raises(KeyError, match="NOT_A_METHOD"):
            get_peft_config({"peft_type": "NOT_A_METHOD"})


class TestBloomPostprocess:
    def test_it_is_registered_for_bloom_only(self) -> None:
        table = TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING
        assert table == {"bloom": bloom_model_postprocess_past_key_value}

    def test_shapes(self) -> None:
        prompt = tuple(lucid.zeros(1, 2, 4, 3, 8) for _ in range(6))  # 3 layers
        pairs = bloom_model_postprocess_past_key_value(prompt)
        assert len(pairs) == 3
        for keys, values in pairs:
            assert keys.shape == (8, 8, 3)  # (batch * heads, head_dim, tokens)
            assert values.shape == (8, 3, 8)  # (batch * heads, tokens, head_dim)

    def test_keys_and_values_come_from_the_two_halves(self) -> None:
        # 2 entries (1 layer), batch 1, 2 heads, 2 tokens, head_dim 1: keys are 0..3, values 4..7.
        joined = lucid.arange(8, dtype=lucid.float32).reshape(2, 1, 2, 2, 1)
        ((keys, values),) = bloom_model_postprocess_past_key_value((joined,))
        # the reference swaps the head and token axes of the keys before it flattens batch * heads
        assert keys.tolist() == [[[0.0, 2.0]], [[1.0, 3.0]]]
        assert values.tolist() == [[[4.0], [5.0]], [[6.0], [7.0]]]

    def test_a_stacked_tensor_works_as_a_one_tuple(self) -> None:
        stacked = lucid.randn(4, 1, 2, 3, 5)
        a = bloom_model_postprocess_past_key_value((stacked,))
        b = bloom_model_postprocess_past_key_value(tuple(stacked[i : i + 1] for i in range(4)))
        assert all(
            x.tolist() == y.tolist()
            for (xk, xv), (yk, yv) in zip(a, b, strict=True)
            for x, y in ((xk, yk), (xv, yv))
        )
