"""A saved zoo directory loads back as the class it was saved from, and models know their source.

LCD-757.  ``AutoModelFor*`` resolves a directory through ``(task, model_type)``,
and a family registers several heads under one pair — ``BERTForQuestionAnswering``
and ``BERTForSequenceClassification`` are both ``sequence-classification``
``bert`` — so the first registered one was loaded whatever had been saved, and the
strict load of the others' weights failed.  ``save_pretrained`` now records the
class under ``architectures`` and the registry resolves by it.  The cases are
enumerated from the registry, not listed: a family that registers a second head
under a pair is a case here the day it does.
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path

import pytest

from lucid.models import (
    AutoConfig,
    AutoModelForLanguageModeling,
    AutoModelForSequenceClassification,
    BERTConfig,
    BERTForMaskedLM,
    PretrainedModel,
    create_model,
)
from lucid.models import _auto as _auto_mod
from lucid.models._registry import _REGISTRY, _registry_resolve_saved
from lucid.nn._shadow import shadow_alloc
from lucid.peft import AutoPeftModelForQuestionAnswering, LoraConfig, TaskType, get_peft_model

_AUTO_BY_TASK = {cls._task: cls for cls in vars(_auto_mod).values() if hasattr(cls, "_task")}

#: ``model_type`` -> config overrides that shrink a model of that type to unit-test size.
_TEXT = {
    "vocab_size": 64,
    "hidden_size": 16,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "intermediate_size": 32,
    "max_position_embeddings": 32,
}
_TINY: dict[str, dict[str, object]] = {
    "bert": _TEXT,
    "roformer": {**_TEXT, "embedding_size": 16},
}


def _ambiguous_pairs() -> dict[tuple[str, str], dict[str, str]]:
    """``(task, model_type)`` pairs with more than one model class -> class name -> a factory."""
    classes: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for name, entry in _REGISTRY.items():
        assert entry.model_class is not None
        classes[(entry.task, entry.model_type)].setdefault(entry.model_class.__name__, name)
    return {pair: found for pair, found in classes.items() if len(found) > 1}


_CASES = [
    pytest.param(task, model_type, cls, factory, id=f"{task}-{model_type}-{cls}")
    for (task, model_type), found in sorted(_ambiguous_pairs().items())
    for cls, factory in sorted(found.items())
]


def _same_weights(a: PretrainedModel, b: PretrainedModel) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(
        float((sa[k] - sb[k]).abs().max().item()) == 0.0 for k in sa
    )


def test_the_ambiguous_pairs_all_have_a_unit_test_size() -> None:
    """A family that registers a second head under a pair must say how to shrink it here."""
    assert {model_type for _, model_type in _ambiguous_pairs()} <= set(_TINY)


@pytest.mark.parametrize(("task", "model_type", "cls", "factory"), _CASES)
def test_a_saved_directory_loads_as_the_class_it_was_saved_from(
    task: str, model_type: str, cls: str, factory: str, tmp_path: Path
) -> None:
    model = create_model(factory, **_TINY[model_type]).eval()
    assert type(model).__name__ == cls
    model.save_pretrained(str(tmp_path))

    loaded = _AUTO_BY_TASK[task].from_pretrained(str(tmp_path))

    assert type(loaded) is type(model)
    assert _same_weights(loaded, model)


@pytest.mark.parametrize("name", sorted(_REGISTRY))
def test_every_registered_class_resolves_to_itself(name: str) -> None:
    """The invariant behind the round trip, over the whole registry and with no model built."""
    entry = _REGISTRY[name]
    assert entry.model_class is not None
    found = _registry_resolve_saved(entry.task, entry.model_type, [entry.model_class.__name__])
    assert found.model_class is entry.model_class


def test_the_directory_names_its_class(tmp_path: Path) -> None:
    create_model("bert_base_qa", **_TINY["bert"]).save_pretrained(str(tmp_path))
    document = json.loads((tmp_path / "config.json").read_text())
    assert document["architectures"] == ["BERTForQuestionAnswering"]
    assert document["model_type"] == "bert"


def test_the_class_name_is_not_a_config_field(tmp_path: Path) -> None:
    """``architectures`` rides beside the fields; reading them back must not warn about it."""
    create_model("bert_base_qa", **_TINY["bert"]).save_pretrained(str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = AutoConfig.from_pretrained(str(tmp_path))
        reread = BERTConfig.load(str(tmp_path / "config.json"))
    assert isinstance(config, BERTConfig)
    assert reread.hidden_size == _TEXT["hidden_size"]


def test_a_directory_without_architectures_takes_the_first_class_for_its_pair(
    tmp_path: Path,
) -> None:
    """Directories saved before the key existed keep loading as they did."""
    model = create_model("bert_base_cls", **_TINY["bert"]).eval()
    model.save_pretrained(str(tmp_path))
    path = tmp_path / "config.json"
    document = json.loads(path.read_text())
    del document["architectures"]
    path.write_text(json.dumps(document))

    loaded = AutoModelForSequenceClassification.from_pretrained(str(tmp_path))

    assert type(loaded) is type(model)


def test_an_architecture_of_another_task_is_refused_naming_its_task(tmp_path: Path) -> None:
    create_model("bert_base_mlm", **_TINY["bert"]).save_pretrained(str(tmp_path))
    with pytest.raises(ValueError, match="registered for task 'language-modeling'"):
        AutoModelForSequenceClassification.from_pretrained(str(tmp_path))
    assert isinstance(AutoModelForLanguageModeling.from_pretrained(str(tmp_path)), BERTForMaskedLM)


def test_an_unknown_architecture_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    create_model("bert_base_cls", **_TINY["bert"]).save_pretrained(str(tmp_path))
    path = tmp_path / "config.json"
    document = json.loads(path.read_text())
    document["architectures"] = ["NoSuchHead"]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="NoSuchHead"):
        AutoModelForSequenceClassification.from_pretrained(str(tmp_path))


def test_architectures_must_be_a_list_of_names(tmp_path: Path) -> None:
    create_model("bert_base_cls", **_TINY["bert"]).save_pretrained(str(tmp_path))
    path = tmp_path / "config.json"
    document = json.loads(path.read_text())
    document["architectures"] = "BERTForSequenceClassification"
    path.write_text(json.dumps(document))
    with pytest.raises(TypeError, match="architectures"):
        AutoModelForSequenceClassification.from_pretrained(str(tmp_path))


# ── name_or_path ─────────────────────────────────────────────────────────────


def test_every_factory_names_its_model() -> None:
    """A factory's model carries the registered name — built without storage, over the registry."""
    wrong: list[str] = []
    with shadow_alloc():
        for name, entry in sorted(_REGISTRY.items()):
            if create_model(name).name_or_path != name:
                wrong.append(name)
    assert wrong == []


def test_the_factory_function_names_its_model_as_create_model_does() -> None:
    from lucid.models import bert_base_cls

    model = bert_base_cls(**_TINY["bert"])
    assert model.name_or_path == "bert_base_cls"
    assert model.__dict__["name_or_path"] == "bert_base_cls"


def test_from_pretrained_by_name_keeps_the_name() -> None:
    model = create_model("lenet_5", pretrained=False)
    assert model.name_or_path == "lenet_5"


def test_from_pretrained_of_a_directory_names_the_directory(tmp_path: Path) -> None:
    created = create_model("bert_base_cls", **_TINY["bert"])
    created.save_pretrained(str(tmp_path))
    direct = type(created).from_pretrained(str(tmp_path))
    through_auto = AutoModelForSequenceClassification.from_pretrained(str(tmp_path))
    assert direct.name_or_path == through_auto.name_or_path == str(tmp_path)


def test_a_model_built_from_a_config_has_no_source() -> None:
    model = BERTForMaskedLM(BERTConfig(**_TEXT))
    assert model.name_or_path == ""
    assert "name_or_path" not in model.__dict__


def test_the_name_is_not_state(tmp_path: Path) -> None:
    model = create_model("bert_base_cls", **_TINY["bert"])
    assert not any("name_or_path" in key for key in model.state_dict())


def test_peft_takes_its_base_model_from_the_factory_name() -> None:
    model = create_model("bert_base_cls", **_TINY["bert"])
    peft = get_peft_model(model, LoraConfig(r=2))
    assert peft.peft_config["default"].base_model_name_or_path == "bert_base_cls"


def test_peft_takes_its_base_model_from_the_directory(tmp_path: Path) -> None:
    create_model("bert_base_cls", **_TINY["bert"]).save_pretrained(str(tmp_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(tmp_path))
    peft = get_peft_model(model, LoraConfig(r=2))
    assert peft.peft_config["default"].base_model_name_or_path == str(tmp_path)


def test_a_saved_adapter_reloads_the_head_its_base_was_saved_with(tmp_path: Path) -> None:
    """The point of both fields: ``AutoPeftModel`` rebuilds the base from what the adapter recorded."""
    create_model("bert_base_qa", **_TINY["bert"]).save_pretrained(str(tmp_path / "base"))
    base = AutoModelForSequenceClassification.from_pretrained(str(tmp_path / "base"))
    adapter = get_peft_model(base, LoraConfig(r=2, task_type=TaskType.QUESTION_ANS))
    adapter.save_pretrained(str(tmp_path / "adapter"))

    reloaded = AutoPeftModelForQuestionAnswering.from_pretrained(str(tmp_path / "adapter"))

    assert type(reloaded.get_base_model()).__name__ == "BERTForQuestionAnswering"
