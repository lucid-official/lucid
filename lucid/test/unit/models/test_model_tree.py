"""``lucid/models/MODELS.tree`` is the registry, drawn, and the check can go red."""

import pytest

import lucid.models as models
from tools.model_tree import TREE_PATH, Entry, check, check_tree, draw, parse, registered


def _text() -> str:
    return TREE_PATH.read_text(encoding="utf-8")


def _placed_task(entries: list[Entry], name: str) -> str:
    return next(e.task for e in parse(draw(entries)) if e.name == name)


def test_the_committed_tree_is_the_registry_drawn() -> None:
    assert check_tree() == []
    assert _text() == draw(registered())


def test_every_factory_is_drawn_exactly_once() -> None:
    names = [entry.name for entry in parse(_text())]
    assert sorted(names) == sorted(models.list_models())


def test_a_task_branch_shows_what_the_public_query_returns() -> None:
    for entry in registered():
        if entry.task == "base":
            continue
        drawn = {
            e.name for e in parse(_text()) if e.task == entry.task and e.family == entry.family
        }
        assert set(models.list_models(task=entry.task, family=entry.family)) <= drawn


def test_a_base_factory_follows_its_family_primary_task() -> None:
    entries = [
        Entry("text", "bert", "base", "bert_base"),
        Entry("text", "bert", "sequence-classification", "bert_base_cls"),
        Entry("text", "bert", "sequence-classification", "bert_base_qa"),
        Entry("text", "bert", "language-modeling", "bert_base_mlm"),
    ]
    assert _placed_task(entries, "bert_base") == "sequence-classification"
    assert _placed_task(entries, "bert_base_mlm") == "language-modeling"


def test_a_tie_for_primary_goes_to_the_alphabetically_first_task() -> None:
    entries = [
        Entry("text", "gpt", "base", "gpt"),
        Entry("text", "gpt", "sequence-classification", "gpt_cls"),
        Entry("text", "gpt", "language-modeling", "gpt_lm"),
    ]
    assert _placed_task(entries, "gpt") == "language-modeling"


def test_a_family_with_only_base_factories_stays_under_base() -> None:
    assert _placed_task([Entry("vision", "plain", "base", "plain")], "plain") == "base"


def test_factories_sort_numerically() -> None:
    entries = [Entry("vision", "resnet", "base", f"resnet_{size}") for size in (101, 18, 50)]
    assert [e.name for e in parse(draw(entries))] == ["resnet_18", "resnet_50", "resnet_101"]


@pytest.mark.parametrize(
    ("old", "new", "complaint"),
    [
        ("└── lenet_5_cls", "└── lenet_7_cls", "'lenet_5_cls' is registered"),
        ("└── lenet_5_cls", "└── lenet_7_cls", "'lenet_7_cls' is drawn"),
        ("└── lenet_5_cls", "└── lenet_5", "'lenet_5' is drawn 2 times"),
        ("── lenet/", "── not_a_family/", "family 'not_a_family'"),
        ("── image-classification/", "── image-generation/", "is registered as multimodal/image-c"),
        ("── vision/", "── visual/", "is registered as vision/"),
        ("494 models", "495 models", "out of order"),
    ],
)
def test_each_way_the_tree_can_drift_is_caught(old: str, new: str, complaint: str) -> None:
    text = _text()
    assert old in text
    problems = check(text.replace(old, new, 1), registered())
    assert any(complaint in problem for problem in problems), problems


def test_a_factory_missing_from_the_tree_is_caught() -> None:
    entries = [*registered(), Entry("vision", "a_new_family", "base", "a_new_family_base")]
    assert any("'a_new_family_base' is registered" in p for p in check(_text(), entries))


@pytest.mark.parametrize("text", ["somewhere/\n", "lucid/models/\nnot a tree line\n"])
def test_text_it_cannot_read_is_an_error(text: str) -> None:
    with pytest.raises(ValueError, match="MODELS.tree"):
        parse(text)
