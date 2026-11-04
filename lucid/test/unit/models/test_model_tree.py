"""``lucid/models/MODELS.tree`` stays exactly true to the registry, and the check can go red."""

import pytest

from tools.model_tree import (
    TREE_PATH,
    check,
    check_tree,
    parse,
    redrawn,
    registered_by_family,
    render,
)


def _text() -> str:
    return TREE_PATH.read_text(encoding="utf-8")


def test_the_committed_tree_matches_the_registry() -> None:
    assert check_tree() == []


def test_every_registered_factory_is_drawn_by_its_full_name() -> None:
    lines = {line.rsplit(" ", 1)[-1] for line in _text().splitlines()}
    for names in registered_by_family().values():
        assert set(names) <= lines


def test_drawing_what_was_read_reproduces_the_file() -> None:
    assert render(parse(_text())) == _text()


@pytest.mark.parametrize(
    ("old", "new", "complaint"),
    [
        ("└── lenet_5_cls", "└── lenet_7_cls", "'lenet_5_cls' is registered but not listed"),
        ("└── lenet_5_cls", "└── lenet_7_cls", "'lenet_7_cls' is listed but not registered"),
        ("└── lenet_5_cls", "└── lenet_5", "'lenet_5' is listed twice"),
        ("── lenet/", "── not_a_family/", "'not_a_family' is listed but registers nothing"),
        ("── lenet/", "── not_a_family/", "'lenet' is registered but not listed"),
        ("── alexnet/", "── lenet/", "family 'lenet' is listed twice"),
        ("494 models", "495 models", "not canonical"),
        ("├── alexnet/", "├── lenet_also/", "'lenet_also' is listed"),
    ],
)
def test_each_way_the_tree_can_drift_is_caught(old: str, new: str, complaint: str) -> None:
    text = _text()
    assert old in text
    problems = check(text.replace(old, new, 1), registered_by_family())
    assert any(complaint in problem for problem in problems), problems


def test_factories_out_of_order_are_caught() -> None:
    text = _text()
    swapped = text.replace("├── lenet_5\n", "├── @\n", 1).replace(
        "└── lenet_5_cls\n", "└── lenet_5\n", 1
    )
    swapped = swapped.replace("├── @\n", "├── lenet_5_cls\n", 1)
    assert swapped != text
    assert any("not canonical" in p for p in check(swapped, registered_by_family()))


def test_a_family_under_the_wrong_domain_is_caught() -> None:
    problems = check(_text(), registered_by_family(), {"lenet": "text"})
    assert problems == ["family 'lenet' is under 'vision', but lives in 'text'"]


def test_a_family_missing_from_the_tree_is_caught() -> None:
    registered = registered_by_family()
    registered["a_new_family"] = ["a_new_family_base"]
    assert "family 'a_new_family' is registered but not listed" in check(_text(), registered)


def test_a_new_factory_is_drawn_in_numeric_order() -> None:
    registered = registered_by_family()
    registered["resnet"] = [*registered["resnet"], "resnet_60"]
    drawn = render(redrawn(parse(_text()), registered))
    assert (
        drawn.index("── resnet_50_cls\n")
        < drawn.index("── resnet_60\n")
        < drawn.index("── resnet_101\n")
    )
    assert drawn.endswith("4 domains, 74 families, 495 models\n")


def test_a_line_it_cannot_read_is_an_error() -> None:
    with pytest.raises(ValueError, match="cannot read"):
        parse("lucid/models/\nnot a tree line\n")
