"""``LoraConfig``: validation at the boundary, and ``adapter_config.json`` as the reference writes it.

The golden file below is written by hand from the reference adapter
library's format — sorted keys, two-space indent, enums as strings,
module-name sets as lists — not produced by importing it.
"""

import json
import math
import warnings
from pathlib import Path

import pytest

from lucid.peft import LoraConfig, PeftConfig, PeftType, TaskType

#: What the reference writes for ``LoraConfig(r=8, lora_alpha=16,
#: lora_dropout=0.05, target_modules=["c_attn"], task_type="CAUSAL_LM")``
#: saved for inference, restricted to the keys v1 has.
GOLDEN_CONFIG = {
    "alpha_pattern": {},
    "auto_mapping": None,
    "base_model_name_or_path": None,
    "bias": "none",
    "exclude_modules": None,
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_lora_weights": True,
    "layers_pattern": None,
    "layers_to_transform": None,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "modules_to_save": None,
    "peft_type": "LORA",
    "r": 8,
    "rank_pattern": {},
    "revision": None,
    "target_modules": ["c_attn"],
    "task_type": "CAUSAL_LM",
    "use_dora": False,
    "use_rslora": False,
}

#: An ``adapter_config.json`` as a recent reference release writes it for a
#: GPT-2 adapter: keys for features v1 lacks, all at their "off" value, and
#: provenance keys.
REFERENCE_WRITTEN = {
    **GOLDEN_CONFIG,
    "base_model_name_or_path": "gpt2",
    "corda_config": None,
    "eva_config": None,
    "fan_in_fan_out": True,
    "layer_replication": None,
    "loftq_config": {},
    "lora_bias": False,
    "megatron_config": None,
    "megatron_core": "megatron.core",
    "peft_version": "0.17.1",
    "target_parameters": None,
    "trainable_token_indices": None,
    "use_qalora": False,
}


class TestDefaults:
    def test_reference_defaults(self) -> None:
        c = LoraConfig()
        assert (c.r, c.lora_alpha, c.lora_dropout, c.bias) == (8, 8, 0.0, "none")
        assert c.init_lora_weights is True and c.use_rslora is False and c.use_dora is False
        assert c.peft_type is PeftType.LORA and c.task_type is None
        assert c.rank_pattern == {} and c.alpha_pattern == {}

    def test_lists_of_names_become_sets_and_strings_stay(self) -> None:
        c = LoraConfig(target_modules=["q", "v", "q"], exclude_modules=("k",))
        assert c.target_modules == {"q", "v"} and c.exclude_modules == {"k"}
        assert LoraConfig(target_modules=r".*\.q").target_modules == r".*\.q"

    def test_task_type_string_becomes_the_enum(self) -> None:
        assert LoraConfig(task_type="SEQ_CLS").task_type is TaskType.SEQ_CLS

    def test_fields_are_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            LoraConfig(None)  # type: ignore[misc]

    def test_init_scheme_is_case_insensitive(self) -> None:
        assert LoraConfig(init_lora_weights="Gaussian").init_lora_weights == "gaussian"


BAD_FIELDS = [
    ({"r": 0}, ValueError, "positive integer"),
    ({"r": -2}, ValueError, "positive integer"),
    ({"r": 8.0}, TypeError, "must be an int"),
    ({"r": True}, TypeError, "must be an int"),
    ({"lora_alpha": "8"}, TypeError, "real number"),
    ({"lora_alpha": math.inf}, ValueError, "finite"),
    ({"lora_dropout": 1.0}, ValueError, r"\[0, 1\)"),
    ({"lora_dropout": -0.1}, ValueError, r"\[0, 1\)"),
    ({"bias": "some"}, ValueError, "`bias` must be one of"),
    ({"task_type": "TRANSLATION"}, ValueError, "Invalid task type"),
    ({"task_type": 3}, ValueError, "Invalid task type"),
    ({"peft_type": "IA3"}, ValueError, "LORA adapter"),
    ({"target_modules": 5}, TypeError, "target_modules"),
    ({"target_modules": ["q", 5]}, TypeError, "not a str"),
    ({"modules_to_save": "head"}, TypeError, "modules_to_save"),
    ({"inference_mode": "yes"}, TypeError, "inference_mode"),
    ({"base_model_name_or_path": 3}, TypeError, "base_model_name_or_path"),
    ({"init_lora_weights": "foo"}, ValueError, "Unknown initialization"),
    ({"init_lora_weights": 1}, TypeError, "init_lora_weights"),
    ({"layers_to_transform": "0"}, TypeError, "layers_to_transform"),
    ({"layers_pattern": 3}, TypeError, "layers_pattern"),
    ({"rank_pattern": [("q", 4)]}, TypeError, "rank_pattern"),
    ({"rank_pattern": {"q": 0}}, ValueError, "positive integer"),
    ({"alpha_pattern": "q"}, TypeError, "alpha_pattern"),
    ({"alpha_pattern": {"q": math.nan}}, ValueError, "finite"),
    (
        {"target_modules": "q", "layers_to_transform": [0]},
        ValueError,
        "cannot be used when `target_modules` is a str",
    ),
    (
        {"target_modules": "q", "layers_pattern": "h"},
        ValueError,
        "cannot be used when `target_modules` is a str",
    ),
    ({"target_modules": ["q"], "layers_pattern": "h"}, ValueError, "must also be specified"),
    (
        {"target_modules": ["q"], "layers_pattern": "h", "layers_to_transform": []},
        ValueError,
        "must also be specified",
    ),
    ({"target_modules": "q_proj("}, ValueError, "`target_modules` is not a valid regular"),
    ({"exclude_modules": "["}, ValueError, "`exclude_modules` is not a valid regular"),
    ({"rank_pattern": {"q(": 2}}, ValueError, "rank_pattern.*not a valid regular"),
    ({"alpha_pattern": {"*q": 2.0}}, ValueError, "alpha_pattern.*not a valid regular"),
    (
        {"target_modules": ["q"], "layers_pattern": "h(", "layers_to_transform": [0]},
        ValueError,
        "`layers_pattern` is not a valid regular",
    ),
]


def test_layer_zero_with_a_pattern_is_a_selection() -> None:
    c = LoraConfig(target_modules=["q"], layers_to_transform=0, layers_pattern="h")
    assert c.layers_to_transform == 0 and c.layers_pattern == "h"


@pytest.mark.parametrize(
    ("fields", "error", "match"), BAD_FIELDS, ids=[str(f[0]) for f in BAD_FIELDS]
)
def test_a_bad_field_is_refused_with_its_class(
    fields: dict[str, object], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        LoraConfig(**fields)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "fields",
    [{"use_dora": True}, {"init_lora_weights": "pissa"}, {"init_lora_weights": "pissa_niter_4"}],
)
def test_what_v1_lacks_raises_not_implemented(fields: dict[str, object]) -> None:
    with pytest.raises(NotImplementedError):
        LoraConfig(**fields)  # type: ignore[arg-type]


class TestDict:
    def test_to_dict_is_json_ready(self) -> None:
        d = LoraConfig(target_modules=["v", "q"], task_type=TaskType.CAUSAL_LM).to_dict()
        assert d["target_modules"] == ["q", "v"]
        assert type(d["task_type"]) is str and type(d["peft_type"]) is str
        json.dumps(d)

    def test_round_trip(self) -> None:
        c = LoraConfig(
            r=4,
            lora_alpha=12.5,
            target_modules=["a", "b"],
            exclude_modules="x.*",
            modules_to_save=["head"],
            bias="lora_only",
            rank_pattern={"a": 2},
            alpha_pattern={"b": 3.0},
            layers_to_transform=[0, 1],
            layers_pattern="layers",
            init_lora_weights="gaussian",
            use_rslora=True,
        )
        assert LoraConfig.from_dict(c.to_dict()) == c

    def test_a_file_from_the_reference_reads(self) -> None:
        with pytest.warns(UserWarning, match=r"Unexpected keyword arguments \['some_future_key'\]"):
            c = LoraConfig.from_dict({**REFERENCE_WRITTEN, "some_future_key": None})
        assert c.base_model_name_or_path == "gpt2" and c.target_modules == {"c_attn"}
        assert c.fan_in_fan_out is True and c.inference_mode is True

    @pytest.mark.parametrize("value", [True, 1, "on", {"k": 1}, [0]])
    def test_an_unknown_key_that_is_on_is_refused(self, value: object) -> None:
        with pytest.raises(NotImplementedError, match="some_future_key"):
            LoraConfig.from_dict({**GOLDEN_CONFIG, "some_future_key": value})

    @pytest.mark.parametrize("value", [None, False, {}, [], ""])
    def test_an_unknown_key_that_is_off_is_dropped(self, value: object) -> None:
        with pytest.warns(UserWarning, match="some_future_key"):
            c = LoraConfig.from_dict({**GOLDEN_CONFIG, "some_future_key": value})
        assert c.r == 8

    def test_provenance_keys_drop_silently(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            c = LoraConfig.from_dict(REFERENCE_WRITTEN)
        assert c.r == 8

    @pytest.mark.parametrize(
        ("key", "value"),
        [("lora_bias", True), ("loftq_config", {"loftq_bits": 4}), ("layer_replication", [[0, 4]])],
    )
    def test_a_feature_v1_lacks_turned_on_is_refused(self, key: str, value: object) -> None:
        with pytest.raises(NotImplementedError, match=key):
            LoraConfig.from_dict({**GOLDEN_CONFIG, key: value})

    @pytest.mark.parametrize("peft_type", ["IA3", "SOME_NEWER_METHOD"])
    def test_another_adapter_type_is_refused(self, peft_type: str) -> None:
        with pytest.raises(NotImplementedError, match=peft_type):
            LoraConfig.from_dict({"peft_type": peft_type, "target_modules": ["k"]})

    def test_use_dora_true_in_a_file_is_refused(self) -> None:
        with pytest.raises(NotImplementedError, match="DoRA"):
            LoraConfig.from_dict({**GOLDEN_CONFIG, "use_dora": True})


class TestFile:
    def test_the_file_is_the_reference_layout(self, tmp_path: Path) -> None:
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["c_attn"],
            task_type="CAUSAL_LM",
            inference_mode=True,
        ).save_pretrained(tmp_path)
        text = (tmp_path / "adapter_config.json").read_text(encoding="utf-8")
        assert text == json.dumps(GOLDEN_CONFIG, indent=2)

    def test_save_creates_the_directory_and_reads_back(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "adapter"
        c = LoraConfig(r=2, target_modules="all-linear")
        c.save_pretrained(str(target))
        assert LoraConfig.from_pretrained(target) == c

    def test_a_file_path_is_not_a_directory(self, tmp_path: Path) -> None:
        file = tmp_path / "f"
        file.write_text("x")
        with pytest.raises(NotADirectoryError):
            LoraConfig().save_pretrained(file)

    def test_a_missing_config_is_named(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="adapter_config.json"):
            LoraConfig.from_pretrained(tmp_path)

    def test_a_non_object_file_is_refused(self, tmp_path: Path) -> None:
        (tmp_path / "adapter_config.json").write_text("[1, 2]")
        with pytest.raises(ValueError, match="JSON object"):
            LoraConfig.from_pretrained(tmp_path)

    def test_a_subfolder_holds_the_file(self, tmp_path: Path) -> None:
        LoraConfig(r=3, target_modules=["q"]).save_pretrained(tmp_path / "style")
        assert isinstance(loaded := PeftConfig.from_pretrained(tmp_path, subfolder="style"), LoraConfig)
        assert loaded.r == 3
        with pytest.raises(FileNotFoundError, match="adapter_config.json"):
            PeftConfig.from_pretrained(tmp_path)
        with pytest.raises(FileNotFoundError, match="style"):
            PeftConfig.from_pretrained(tmp_path / "style", subfolder="style")

    def test_from_json_file_is_the_dict_the_file_holds(self, tmp_path: Path) -> None:
        LoraConfig(r=3, target_modules=["q"], task_type="SEQ_CLS").save_pretrained(tmp_path)
        raw = PeftConfig.from_json_file(tmp_path / "adapter_config.json")
        assert raw == json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
        assert isinstance(raw, dict) and raw["task_type"] == "SEQ_CLS"

    def test_from_json_file_drops_a_runtime_config_with_a_warning(self, tmp_path: Path) -> None:
        path = tmp_path / "adapter_config.json"
        path.write_text(json.dumps({**GOLDEN_CONFIG, "runtime_config": {"ephemeral": True}}))
        with pytest.warns(UserWarning, match="runtime_config"):
            raw = PeftConfig.from_json_file(path)
        assert "runtime_config" not in raw
        with pytest.warns(UserWarning, match="runtime_config"):
            assert LoraConfig.from_pretrained(tmp_path).r == 8

    def test_from_json_file_refuses_what_is_not_an_object(self, tmp_path: Path) -> None:
        path = tmp_path / "x.json"
        path.write_text('"text"')
        with pytest.raises(ValueError, match="JSON object"):
            PeftConfig.from_json_file(path)
        with pytest.raises(FileNotFoundError):
            PeftConfig.from_json_file(tmp_path / "missing.json")

    def test_the_file_names_the_class(self, tmp_path: Path) -> None:
        LoraConfig(target_modules=["q"]).save_pretrained(tmp_path)
        assert type(PeftConfig.from_pretrained(tmp_path)) is LoraConfig
        (tmp_path / "adapter_config.json").write_text(
            json.dumps({"peft_type": "PREFIX_TUNING", "task_type": "CAUSAL_LM"})
        )
        with pytest.raises(NotImplementedError, match="PREFIX_TUNING"):
            PeftConfig.from_pretrained(tmp_path)


class TestAutoMapping:
    MAPPING = {"base_model_class": "ViTForImageClassification", "parent_library": "lucid.models"}

    def test_it_is_kept_and_written(self, tmp_path: Path) -> None:
        LoraConfig(target_modules=["q"], auto_mapping=self.MAPPING).save_pretrained(tmp_path)
        assert PeftConfig.from_pretrained(tmp_path).auto_mapping == self.MAPPING
        assert LoraConfig().auto_mapping is None

    @pytest.mark.parametrize("bad", [["a"], {"a": 1}, {1: "a"}, "text"])
    def test_only_a_dict_of_strings_is_taken(self, bad: object) -> None:
        with pytest.raises(TypeError, match="auto_mapping"):
            LoraConfig(auto_mapping=bad)  # type: ignore[arg-type]


#: The reference's ``peft_type`` and ``task_type`` strings, member by member —
#: a file names its method and task with these.
REFERENCE_PEFT_TYPES = {
    PeftType.PROMPT_TUNING: "PROMPT_TUNING",
    PeftType.MULTITASK_PROMPT_TUNING: "MULTITASK_PROMPT_TUNING",
    PeftType.P_TUNING: "P_TUNING",
    PeftType.PREFIX_TUNING: "PREFIX_TUNING",
    PeftType.LORA: "LORA",
    PeftType.ADALORA: "ADALORA",
    PeftType.BOFT: "BOFT",
    PeftType.ADAPTION_PROMPT: "ADAPTION_PROMPT",
    PeftType.IA3: "IA3",
    PeftType.LOHA: "LOHA",
    PeftType.LOKR: "LOKR",
    PeftType.OFT: "OFT",
    PeftType.POLY: "POLY",
    PeftType.LN_TUNING: "LN_TUNING",
    PeftType.VERA: "VERA",
    PeftType.FOURIERFT: "FOURIERFT",
    PeftType.XLORA: "XLORA",
    PeftType.HRA: "HRA",
    PeftType.VBLORA: "VBLORA",
    PeftType.CPT: "CPT",
    PeftType.BONE: "BONE",
}

REFERENCE_TASK_TYPES = {
    TaskType.SEQ_CLS: "SEQ_CLS",
    TaskType.SEQ_2_SEQ_LM: "SEQ_2_SEQ_LM",
    TaskType.CAUSAL_LM: "CAUSAL_LM",
    TaskType.TOKEN_CLS: "TOKEN_CLS",
    TaskType.QUESTION_ANS: "QUESTION_ANS",
    TaskType.FEATURE_EXTRACTION: "FEATURE_EXTRACTION",
}


@pytest.mark.parametrize(
    ("enum", "reference"), [(PeftType, REFERENCE_PEFT_TYPES), (TaskType, REFERENCE_TASK_TYPES)]
)
def test_enum_members_are_the_reference_strings(
    enum: type[PeftType] | type[TaskType], reference: dict[str, str]
) -> None:
    assert {member: member.value for member in enum} == reference
    assert all(str(member) == value and member == value for member, value in reference.items())


@pytest.mark.parametrize("peft_type", [m for m in PeftType if m is not PeftType.LORA])
def test_every_other_adapter_type_is_not_implemented(peft_type: PeftType) -> None:
    with pytest.raises(NotImplementedError, match=str(peft_type)):
        LoraConfig.from_dict({"peft_type": str(peft_type)})


@pytest.mark.parametrize("task_type", list(TaskType))
def test_every_task_type_is_accepted(task_type: TaskType) -> None:
    assert LoraConfig(task_type=str(task_type)).task_type is task_type
