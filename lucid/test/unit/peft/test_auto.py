"""``AutoPeftModel*``: the adapter's directory says which base to build and which class to return.

What the zoo does with a real name or directory is held per family in
``lucid/test/unit/models/test_peft_zoo.py``.  Here the base is a class of the
test's own, reached through the same ``auto_mapping`` route the zoo's models
take, so the checks on the route itself — the task match, the allowlist, the
refusals — run without building a model of the zoo.
"""

import json
import sys
import types
from pathlib import Path
from typing import Self

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    AutoPeftModel,
    AutoPeftModelForCausalLM,
    AutoPeftModelForFeatureExtraction,
    AutoPeftModelForQuestionAnswering,
    AutoPeftModelForSeq2SeqLM,
    AutoPeftModelForSequenceClassification,
    AutoPeftModelForTokenClassification,
    LoraConfig,
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForFeatureExtraction,
    PeftModelForQuestionAnswering,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
    get_peft_model,
)

_PACKAGE = "lucid_test_package"


class _Net(nn.Module):
    """What the adapters here were trained on; built from a name the way a zoo class is."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(8, 8)
        self.classifier = nn.Linear(8, 3)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.classifier(self.encoder(x))

    @classmethod
    def from_pretrained(cls, name_or_path: str, *, strict: bool = True) -> Self:
        del strict
        assert name_or_path == "the-base", name_or_path
        lucid.manual_seed(1)
        return cls()


@pytest.fixture(autouse=True)
def package(monkeypatch: pytest.MonkeyPatch) -> None:
    """``lucid_test_package.Net`` — a package of the test's own, standing in for ``lucid.models``."""
    monkeypatch.setitem(sys.modules, _PACKAGE, types.SimpleNamespace(Net=_Net, answer=42))


def _adapter(path: Path, task: str | None, base: str | None = "the-base") -> lucid.Tensor:
    """Save an adapter of ``_Net`` for ``task``; return what it computes on ones."""
    lucid.manual_seed(1)
    config = LoraConfig(
        r=2,
        target_modules=["encoder"],
        task_type=task,
        init_lora_weights=False,
        base_model_name_or_path=base,
    )
    peft = get_peft_model(_Net(), config)
    peft.save_pretrained(path)
    with lucid.no_grad():
        return peft.eval()(lucid.ones(2, 8))


def _remap(path: Path, **fields: object) -> None:
    """Rewrite fields of the adapter's config file."""
    file = path / "adapter_config.json"
    file.write_text(json.dumps(json.loads(file.read_text()) | fields))


def _name_the_test_package(path: Path, name: str = "Net") -> None:
    _remap(path, auto_mapping={"base_model_class": name, "parent_library": _PACKAGE})


ALLOW = [_PACKAGE]


class TestFromPretrained:
    def test_an_adapter_with_no_task_is_rebuilt_from_its_mapping(self, tmp_path: Path) -> None:
        expected = _adapter(tmp_path, None)
        _name_the_test_package(tmp_path)
        loaded = AutoPeftModel.from_pretrained(tmp_path, import_allowlist=ALLOW)
        assert type(loaded) is PeftModel and type(loaded.get_base_model()) is _Net
        assert not any(p.requires_grad for p in loaded.parameters())
        assert bool((loaded(lucid.ones(2, 8)) == expected).all().item())

    def test_trainable_and_named_on_request(self, tmp_path: Path) -> None:
        _adapter(tmp_path, None)
        _name_the_test_package(tmp_path)
        loaded = AutoPeftModel.from_pretrained(
            tmp_path, adapter_name="mine", is_trainable=True, import_allowlist=ALLOW
        )
        assert list(loaded.peft_config) == ["mine"]
        assert loaded.get_nb_trainable_parameters()[0] > 0

    def test_a_subfolder_holds_the_adapter(self, tmp_path: Path) -> None:
        _adapter(tmp_path / "nested", None)
        _name_the_test_package(tmp_path / "nested")
        loaded = AutoPeftModel.from_pretrained(tmp_path, subfolder="nested", import_allowlist=ALLOW)
        assert type(loaded.get_base_model()) is _Net

    def test_a_config_given_replaces_the_adapters_own(self, tmp_path: Path) -> None:
        _adapter(tmp_path, None)
        _name_the_test_package(tmp_path)
        config = LoraConfig.from_pretrained(tmp_path)
        config.lora_alpha = 32
        loaded = AutoPeftModel.from_pretrained(tmp_path, config=config, import_allowlist=ALLOW)
        assert loaded.peft_config["default"].lora_alpha == 32

    def test_the_saved_mapping_names_the_class_of_the_base(self, tmp_path: Path) -> None:
        _adapter(tmp_path, "SEQ_CLS")
        mapping = LoraConfig.from_pretrained(tmp_path).auto_mapping
        assert mapping == {"base_model_class": "_Net", "parent_library": _Net.__module__}


class TestTheMappingIsFollowedOnlyWhereAllowed:
    def test_the_default_allowlist_is_the_zoo(self, tmp_path: Path) -> None:
        _adapter(tmp_path, None)
        _name_the_test_package(tmp_path)
        with pytest.raises(ValueError, match=rf"{_PACKAGE}.*import_allowlist=\['{_PACKAGE}'\]"):
            AutoPeftModel.from_pretrained(tmp_path)

    def test_a_subpackage_of_an_allowed_package_is_allowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _adapter(tmp_path, None)
        monkeypatch.setitem(sys.modules, f"{_PACKAGE}.models", types.SimpleNamespace(Net=_Net))
        _remap(
            tmp_path,
            auto_mapping={"base_model_class": "Net", "parent_library": f"{_PACKAGE}.models"},
        )
        loaded = AutoPeftModel.from_pretrained(tmp_path, import_allowlist=ALLOW)
        assert type(loaded.get_base_model()) is _Net

    def test_a_name_that_only_starts_like_an_allowed_package_is_not(self, tmp_path: Path) -> None:
        _adapter(tmp_path, None)
        _remap(
            tmp_path,
            auto_mapping={"base_model_class": "Net", "parent_library": f"{_PACKAGE}_other"},
        )
        with pytest.raises(ValueError, match="allowlist"):
            AutoPeftModel.from_pretrained(tmp_path, import_allowlist=ALLOW)

    @pytest.mark.parametrize("name", ["answer", "missing", "create_model"])
    def test_what_is_found_must_be_a_model_class(self, tmp_path: Path, name: str) -> None:
        _adapter(tmp_path, None)
        _name_the_test_package(tmp_path, name)
        with pytest.raises(TypeError, match="not a model class"):
            AutoPeftModel.from_pretrained(tmp_path, import_allowlist=[_PACKAGE, "lucid.models"])

    def test_the_typed_classes_pass_over_a_mapping_into_another_package(
        self, tmp_path: Path
    ) -> None:
        """A file the reference wrote maps into its own package; the zoo's class is used instead."""
        _adapter(tmp_path, "SEQ_CLS", base="no-such-zoo-model")
        _remap(
            tmp_path,
            auto_mapping={"base_model_class": "BertForSequenceClassification",
                          "parent_library": "transformers.models.bert.modeling_bert"},
        )
        with pytest.raises(ValueError, match="no-such-zoo-model"):
            AutoPeftModelForSequenceClassification.from_pretrained(tmp_path)

    def test_the_typed_classes_follow_an_allowed_one(self, tmp_path: Path) -> None:
        expected = _adapter(tmp_path, "SEQ_CLS")
        _name_the_test_package(tmp_path)
        loaded = AutoPeftModelForSequenceClassification.from_pretrained(
            tmp_path, import_allowlist=ALLOW
        )
        assert type(loaded) is PeftModelForSequenceClassification
        assert bool((loaded(lucid.ones(2, 8)) == expected).all().item())


class TestTheTasksMatch:
    SHELLS = [
        (AutoPeftModelForCausalLM, PeftModelForCausalLM, "CAUSAL_LM"),
        (AutoPeftModelForSeq2SeqLM, PeftModelForSeq2SeqLM, "SEQ_2_SEQ_LM"),
        (AutoPeftModelForSequenceClassification, PeftModelForSequenceClassification, "SEQ_CLS"),
        (AutoPeftModelForTokenClassification, PeftModelForTokenClassification, "TOKEN_CLS"),
        (AutoPeftModelForQuestionAnswering, PeftModelForQuestionAnswering, "QUESTION_ANS"),
        (AutoPeftModelForFeatureExtraction, PeftModelForFeatureExtraction, "FEATURE_EXTRACTION"),
    ]

    @pytest.mark.parametrize(("shell", "wrapper", "task"), SHELLS, ids=lambda x: getattr(x, "__name__", x))
    def test_each_shell_returns_its_wrapper(
        self, shell: type, wrapper: type[PeftModel], task: str, tmp_path: Path
    ) -> None:
        _adapter(tmp_path, task)
        _name_the_test_package(tmp_path)
        loaded = shell.from_pretrained(tmp_path, import_allowlist=ALLOW)
        assert type(loaded) is wrapper

    @pytest.mark.parametrize(("shell", "wrapper", "task"), SHELLS, ids=lambda x: getattr(x, "__name__", x))
    def test_each_shell_refuses_another_task(
        self, shell: type, wrapper: type[PeftModel], task: str, tmp_path: Path
    ) -> None:
        other = "CAUSAL_LM" if task != "CAUSAL_LM" else "SEQ_CLS"
        _adapter(tmp_path, other)
        _name_the_test_package(tmp_path)
        with pytest.raises(ValueError, match=rf"Expected target PEFT class: .*, but you have asked for: {wrapper.__name__}"):
            shell.from_pretrained(tmp_path, import_allowlist=ALLOW)

    def test_a_shell_takes_an_adapter_with_no_task(self, tmp_path: Path) -> None:
        _adapter(tmp_path, None)
        _name_the_test_package(tmp_path)
        loaded = AutoPeftModelForCausalLM.from_pretrained(tmp_path, import_allowlist=ALLOW)
        assert type(loaded) is PeftModelForCausalLM

    def test_the_task_less_class_refuses_a_task(self, tmp_path: Path) -> None:
        _adapter(tmp_path, "CAUSAL_LM")
        with pytest.raises(ValueError, match="Cannot use `AutoPeftModel` with a task type"):
            AutoPeftModel.from_pretrained(tmp_path)

    @pytest.mark.parametrize("mapping", [None, {"base_model_class": "Net"}], ids=["none", "partial"])
    def test_the_task_less_class_needs_a_mapping(
        self, tmp_path: Path, mapping: dict[str, str] | None
    ) -> None:
        _adapter(tmp_path, None)
        _remap(tmp_path, auto_mapping=mapping)
        with pytest.raises(ValueError, match="Cannot infer the auto class"):
            AutoPeftModel.from_pretrained(tmp_path)


class TestRefusals:
    @pytest.mark.parametrize("base", [None, ""])
    def test_an_adapter_that_names_no_base_says_what_to_do(
        self, tmp_path: Path, base: str | None
    ) -> None:
        _adapter(tmp_path, "SEQ_CLS", base=base)
        _name_the_test_package(tmp_path)
        with pytest.raises(ValueError, match="names no base model.*PeftModel.from_pretrained"):
            AutoPeftModelForSequenceClassification.from_pretrained(tmp_path, import_allowlist=ALLOW)

    def test_a_missing_directory_is_the_files_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="adapter_config.json"):
            AutoPeftModelForCausalLM.from_pretrained(tmp_path / "nowhere")

    @pytest.mark.parametrize(
        "shell",
        [AutoPeftModel, AutoPeftModelForCausalLM, AutoPeftModelForFeatureExtraction],
    )
    def test_the_shells_are_never_instantiated(self, shell: type) -> None:
        with pytest.raises(OSError, match=rf"{shell.__name__} is designed to be instantiated"):
            shell()
