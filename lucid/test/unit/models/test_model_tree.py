"""``lucid/models/MODELS.tree`` stays true to the registry, and the check can go red."""

import pytest

from tools.model_tree import TREE_PATH, check, check_tree, registered_by_family


def test_the_committed_tree_matches_the_registry() -> None:
    assert check_tree() == []


@pytest.mark.parametrize(
    ("edit", "complaint"),
    [
        (("[lenet] (2)", "[lenet] (3)"), "says 3 models, 2 are registered"),
        (("[lenet] (2)", "[not_a_family] (2)"), "'not_a_family' is listed"),
        (("[lenet] (2)", "[alexnet] (2)"), "listed twice"),
        (("lenet_5  +cls", "lenet_7  +cls"), "no factory matches 'lenet_7'"),
        (("Classic CNN (22)", "Classic CNN (23)"), "'Classic CNN (23)'"),
        (("— 74 families", "— 75 families"), "says 75 families"),
    ],
)
def test_each_way_the_tree_can_drift_is_caught(edit: tuple[str, str], complaint: str) -> None:
    text = TREE_PATH.read_text(encoding="utf-8")
    assert edit[0] in text
    problems = check(text.replace(edit[0], edit[1], 1), registered_by_family())
    assert any(complaint in problem for problem in problems), problems


def test_a_family_missing_from_the_tree_is_caught() -> None:
    registered = registered_by_family()
    registered["a_new_family"] = ["a_new_family_base"]
    problems = check(TREE_PATH.read_text(encoding="utf-8"), registered)
    assert "family 'a_new_family' is registered but not listed" in problems
