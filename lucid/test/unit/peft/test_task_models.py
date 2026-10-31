"""The ``PeftModelFor*`` wrappers: found by ``task_type``, with the head and ``generate`` of their task.

``get_peft_model`` and ``PeftModel.from_pretrained`` look the wrapper up in a
registry the wrappers join as they are defined; nothing is a branch they
take.  What is held here, per task: the class returned, the head that trains,
that ``generate`` calls the adapted model's own, and that at initialisation
(``B = 0``) every wrapper computes what the base model computed — on every
device.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    MODEL_TYPE_TO_PEFT_MODEL_MAPPING,
    LoraConfig,
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForFeatureExtraction,
    PeftModelForQuestionAnswering,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
    TaskType,
    get_peft_model,
)


class _Net(nn.Module):
    """An encoder with every conventional head, and a ``generate`` that unrolls it."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(8, 8)
        self.classifier = nn.Linear(8, 3)
        self.qa_outputs = nn.Linear(8, 2)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.classifier(self.encoder(x)) + self.qa_outputs(self.encoder(x)).sum()

    def generate(self, x: lucid.Tensor, steps: int = 2) -> lucid.Tensor:
        for _ in range(steps):
            x = self.encoder(x)
        return x


def _net(device: str = "cpu") -> _Net:
    lucid.manual_seed(0)
    return _Net().to(device)


class _Row(NamedTuple):
    task: TaskType
    wrapper: type[PeftModel]
    heads: list[str]


_ROWS = [
    _Row(TaskType.CAUSAL_LM, PeftModelForCausalLM, []),
    _Row(TaskType.SEQ_2_SEQ_LM, PeftModelForSeq2SeqLM, []),
    _Row(TaskType.SEQ_CLS, PeftModelForSequenceClassification, ["classifier"]),
    _Row(TaskType.TOKEN_CLS, PeftModelForTokenClassification, ["classifier"]),
    _Row(TaskType.QUESTION_ANS, PeftModelForQuestionAnswering, ["qa_outputs"]),
    _Row(TaskType.FEATURE_EXTRACTION, PeftModelForFeatureExtraction, []),
]
_BY_TASK = pytest.mark.parametrize("row", _ROWS, ids=lambda r: r.task.value)


def _config(task: TaskType | None = None, **fields: object) -> LoraConfig:
    base: dict[str, object] = {"r": 2, "target_modules": ["encoder"], "task_type": task}
    return LoraConfig(**(base | fields))  # type: ignore[arg-type]


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[TaskType, type[PeftModel]]]:
    """A private copy of the registry, so a wrapper defined in a test is gone after it."""
    copy = dict(PeftModel._task_models)
    monkeypatch.setattr(PeftModel, "_task_models", copy)
    yield copy


class TestRegistry:
    def test_every_task_has_exactly_its_wrapper(self) -> None:
        assert set(MODEL_TYPE_TO_PEFT_MODEL_MAPPING) == set(TaskType)
        for row in _ROWS:
            assert MODEL_TYPE_TO_PEFT_MODEL_MAPPING[row.task] is row.wrapper

    def test_the_table_is_read_only_and_keyed_like_the_reference(self) -> None:
        assert MODEL_TYPE_TO_PEFT_MODEL_MAPPING["SEQ_CLS"] is PeftModelForSequenceClassification
        with pytest.raises(TypeError):
            MODEL_TYPE_TO_PEFT_MODEL_MAPPING[TaskType.CAUSAL_LM] = PeftModel  # type: ignore[index]

    @_BY_TASK
    def test_every_wrapper_is_a_peft_model(self, row: _Row) -> None:
        assert issubclass(row.wrapper, PeftModel) and row.wrapper is not PeftModel

    def test_a_wrapper_defined_later_joins_the_table(
        self, registry: dict[TaskType, type[PeftModel]]
    ) -> None:
        class Mine(PeftModel, task_type=TaskType.FEATURE_EXTRACTION):
            pass

        assert registry[TaskType.FEATURE_EXTRACTION] is Mine
        assert type(get_peft_model(_net(), _config("FEATURE_EXTRACTION"))) is Mine

        # A subclass that names no task is not registered.
        class Unnamed(PeftModel):
            pass

        assert Mine in registry.values() and Unnamed not in registry.values()

    def test_a_task_with_no_wrapper_gets_the_plain_class(
        self, registry: dict[TaskType, type[PeftModel]]
    ) -> None:
        del registry[TaskType.SEQ_CLS]
        assert type(get_peft_model(_net(), _config("SEQ_CLS"))) is PeftModel


class TestGetPeftModel:
    @_BY_TASK
    def test_the_wrapper_follows_the_task_type(self, row: _Row) -> None:
        peft = get_peft_model(_net(), _config(row.task))
        assert type(peft) is row.wrapper
        assert peft.peft_config["default"].task_type is row.task

    @pytest.mark.parametrize("task", [None, ""], ids=["none", "empty"])
    def test_no_task_is_the_plain_class(self, task: str | None) -> None:
        config = _config(None) if task is None else _config(None, base_model_name_or_path=task)
        assert type(get_peft_model(_net(), config)) is PeftModel

    @_BY_TASK
    def test_a_wrapper_can_be_built_directly(self, row: _Row) -> None:
        peft = row.wrapper(_net(), _config())
        assert type(peft) is row.wrapper and peft.peft_config["default"].task_type is None

    @_BY_TASK
    def test_at_initialisation_it_computes_what_the_base_computed(
        self, row: _Row, device: str
    ) -> None:
        base = _net(device)
        x = lucid.randn(4, 8, device=device)
        expected = base(x)
        peft = get_peft_model(base, _config(row.task))
        assert bool((peft(x) == expected).all().item())

    @_BY_TASK
    def test_the_heads_train_and_the_config_passed_in_is_left_alone(self, row: _Row) -> None:
        config = _config(row.task)
        peft = get_peft_model(_net(), config)
        assert config.modules_to_save is None
        assert (peft.peft_config["default"].modules_to_save or []) == row.heads
        model = peft.get_base_model()
        for head in ("classifier", "qa_outputs"):
            held = model.get_submodule(head)
            if head in row.heads:
                assert held.modules_to_save["default"].weight.requires_grad
            else:
                assert not held.weight.requires_grad

    def test_a_head_the_model_lacks_is_not_named(self) -> None:
        model = nn.Sequential(nn.Linear(4, 4))
        peft = get_peft_model(model, LoraConfig(r=2, target_modules=["0"], task_type="SEQ_CLS"))
        assert type(peft) is PeftModelForSequenceClassification
        assert peft.peft_config["default"].modules_to_save is None

    def test_the_users_own_modules_to_save_comes_first(self) -> None:
        config = _config("SEQ_CLS", modules_to_save=["qa_outputs"])
        peft = get_peft_model(_net(), config)
        assert peft.peft_config["default"].modules_to_save == ["qa_outputs", "classifier"]
        assert config.modules_to_save == ["qa_outputs"]

    def test_a_head_already_named_is_not_named_twice(self) -> None:
        peft = get_peft_model(_net(), _config("SEQ_CLS", modules_to_save=["classifier"]))
        assert peft.peft_config["default"].modules_to_save == ["classifier"]

    def test_a_plain_model_does_not_train_a_head(self) -> None:
        peft = PeftModel(_net(), _config("SEQ_CLS"))
        assert peft.peft_config["default"].modules_to_save is None
        assert not peft.get_base_model().classifier.weight.requires_grad

    def test_a_non_module_is_the_tuner_s_type_error(self) -> None:
        with pytest.raises(TypeError, match="nn.Module"):
            PeftModelForSequenceClassification("not a module", _config())  # type: ignore[arg-type]


class TestAddAdapter:
    @_BY_TASK
    def test_every_adapter_the_wrapper_takes_trains_the_head(self, row: _Row) -> None:
        peft = row.wrapper(_net(), _config())
        peft.add_adapter("second", _config())
        assert (peft.peft_config["second"].modules_to_save or []) == row.heads
        for head in row.heads:
            assert "second" in peft.get_base_model().get_submodule(head).modules_to_save

    def test_the_wrapper_not_the_config_decides(self) -> None:
        peft = PeftModelForQuestionAnswering(_net(), _config("SEQ_CLS"))
        assert peft.peft_config["default"].modules_to_save == ["qa_outputs"]
        peft.add_adapter("second", _config("SEQ_CLS"))
        assert peft.peft_config["second"].modules_to_save == ["qa_outputs"]

    def test_a_plain_model_adds_none(self) -> None:
        peft = PeftModel(_net(), _config("SEQ_CLS"))
        peft.add_adapter("second", _config("SEQ_CLS"))
        assert peft.peft_config["second"].modules_to_save is None


class TestGenerate:
    @pytest.mark.parametrize("wrapper", [PeftModelForCausalLM, PeftModelForSeq2SeqLM])
    def test_generate_is_the_adapted_model_s_own(
        self, wrapper: type[PeftModel], device: str
    ) -> None:
        base = _net(device)
        x = lucid.randn(2, 8, device=device)
        plain = base.generate(x, steps=3)
        peft = wrapper(base, _config(init_lora_weights=False, lora_alpha=64))
        adapted = peft.generate(x, steps=3)  # type: ignore[operator]
        assert not bool((adapted == plain).all().item())
        with peft.disable_adapter():
            assert bool((peft.generate(x, steps=3) == plain).all().item())  # type: ignore[operator]

    def test_arguments_are_passed_as_given(self) -> None:
        seen: list[tuple[tuple[object, ...], dict[str, object]]] = []

        class Recorder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = nn.Linear(2, 2)

            def generate(self, *args: object, **kwargs: object) -> str:
                seen.append((args, kwargs))
                return "decoded"

        peft = PeftModelForCausalLM(Recorder(), _config())
        assert peft.generate(1, 2, beams=3) == "decoded"  # type: ignore[operator]
        assert seen == [((1, 2), {"beams": 3})]

    @pytest.mark.parametrize("wrapper", [PeftModelForCausalLM, PeftModelForSeq2SeqLM])
    def test_a_model_that_cannot_decode_says_so(self, wrapper: type[PeftModel]) -> None:
        peft = wrapper(nn.Sequential(nn.Linear(2, 2)), LoraConfig(r=2, target_modules=["0"]))
        with pytest.raises(AttributeError, match="Sequential has no generate"):
            peft.generate(1)  # type: ignore[operator]

    def test_the_other_wrappers_leave_generate_to_the_model(self) -> None:
        with_generate = PeftModelForSequenceClassification(_net(), _config())
        assert callable(with_generate.generate)  # forwarded from the model
        bare = nn.Sequential(nn.Linear(2, 2))
        without = PeftModelForFeatureExtraction(bare, _config(None, target_modules=["0"]))
        assert not hasattr(without, "generate")


class _Base(nn.Module):
    """A tiny model with a conventional head, rebuilt identically for every adapter loaded."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(8, 8)
        self.classifier = nn.Linear(8, 3)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.classifier(self.encoder(x))


def _base(device: str = "cpu") -> _Base:
    lucid.manual_seed(1)
    return _Base().to(device)


def _saved(path: Path, task: str | None, device: str = "cpu") -> lucid.Tensor:
    """Save a trained-looking adapter for ``task``; return its output on a fixed input."""
    peft = get_peft_model(
        _base(device),
        LoraConfig(r=2, target_modules=["encoder"], task_type=task, init_lora_weights=False),
    )
    peft.save_pretrained(path)
    with lucid.no_grad():
        return peft.eval()(lucid.ones(2, 8, device=device))


class TestFromPretrained:
    @pytest.mark.parametrize("entry", [PeftModel, PeftModelForCausalLM])
    def test_the_adapters_task_picks_the_class_whoever_is_asked(
        self, entry: type[PeftModel], tmp_path: Path, device: str
    ) -> None:
        expected = _saved(tmp_path, "SEQ_CLS", device)
        loaded = entry.from_pretrained(_base(device), tmp_path)
        assert type(loaded) is PeftModelForSequenceClassification
        assert bool((loaded(lucid.ones(2, 8, device=device)) == expected).all().item())

    @pytest.mark.parametrize(
        "entry", [PeftModel, PeftModelForCausalLM, PeftModelForFeatureExtraction]
    )
    def test_an_adapter_with_no_task_is_built_as_the_class_asked_for(
        self, entry: type[PeftModel], tmp_path: Path
    ) -> None:
        _saved(tmp_path, None)
        assert type(entry.from_pretrained(_base(), tmp_path)) is entry

    def test_a_subfolder_holds_the_adapter(self, tmp_path: Path) -> None:
        _saved(tmp_path / "nested" / "style", "TOKEN_CLS")
        loaded = PeftModel.from_pretrained(_base(), tmp_path, subfolder="nested/style")
        assert type(loaded) is PeftModelForTokenClassification
        with pytest.raises(FileNotFoundError, match="adapter_config.json"):
            PeftModel.from_pretrained(_base(), tmp_path)

    def test_load_adapter_reads_a_subfolder_too(self, tmp_path: Path) -> None:
        _saved(tmp_path / "style", "SEQ_CLS")
        peft = get_peft_model(_base(), LoraConfig(r=2, target_modules=["encoder"]))
        peft.load_adapter(tmp_path, "style", subfolder="style")
        assert sorted(peft.peft_config) == ["default", "style"]

    def test_loading_into_a_wrapper_adds_the_head_to_the_adapter(self, tmp_path: Path) -> None:
        _saved(tmp_path, None)
        # The file holds no copy of the head, which the reference warns of as well.
        with pytest.warns(UserWarning, match="missing adapter keys"):
            loaded = PeftModelForSequenceClassification.from_pretrained(_base(), tmp_path)
        assert loaded.peft_config["default"].modules_to_save == ["classifier"]
