"""A spelling the sweep no longer lists can still name an object it does.

The sweep lists an object once, under the shallowest module that exports it.
Exporting a class from a shallower module therefore leaves the old spelling
unlisted although every caller's import of it still works (LCD-959).
``_surface.moved`` is the one place that says where such a spelling went; the
released-surface gate and the coverage baseline both ask it.  The fixtures are
plain namespaces, so what it does is read off the fixture and not the tree.
"""

import types

from lucid.test.audit import _surface


class _Thing:
    def run(self, x: int) -> int:
        return x

    @classmethod
    def build(cls) -> object:
        return cls()


class _Helper:
    pass


def _roots(**attrs: object) -> dict[str, object]:
    return {"lucid": types.SimpleNamespace(**attrs)}


_LISTED: list[tuple[str, object]] = [("lucid.Thing", _Thing)]


def test_a_spelling_of_a_listed_object_maps_to_the_listed_one() -> None:
    roots = _roots(Thing=_Thing, deep=types.SimpleNamespace(Thing=_Thing))
    assert _surface.moved(["lucid.deep.Thing"], _LISTED, roots) == {
        "lucid.deep.Thing": "lucid.Thing"
    }


def test_a_member_maps_to_the_same_member_of_the_listed_class() -> None:
    roots = _roots(Thing=_Thing, deep=types.SimpleNamespace(Thing=_Thing))
    names = ["lucid.deep.Thing.run", "lucid.deep.Thing.build"]
    # ``build`` is a bound method each time it is read: it is the class that is looked up.
    assert _surface.moved(names, _LISTED, roots) == {
        "lucid.deep.Thing.run": "lucid.Thing.run",
        "lucid.deep.Thing.build": "lucid.Thing.build",
    }


def test_a_listed_spelling_has_not_moved() -> None:
    roots = _roots(Thing=_Thing)
    assert _surface.moved(["lucid.Thing", "lucid.Thing.run"], _LISTED, roots) == {}


def test_a_second_listed_spelling_of_one_object_has_not_moved() -> None:
    # ``Tensor.add`` and ``lucid.add`` are listed both: each is answered in its own right.
    listed = [("Tensor.Thing", _Thing), ("lucid.Thing", _Thing)]
    roots = _roots(Thing=_Thing)
    assert _surface.moved(["lucid.Thing"], listed, roots) == {}


def test_a_spelling_that_reaches_nothing_has_not_moved() -> None:
    roots = _roots(Thing=_Thing, deep=types.SimpleNamespace())
    names = ["lucid.deep.Thing", "lucid.deep.Thing.run", "lucid.gone.Thing", "other.Thing"]
    assert _surface.moved(names, _LISTED, roots) == {}


def test_a_spelling_that_reaches_an_unlisted_object_has_not_moved() -> None:
    roots = _roots(Thing=_Thing, deep=types.SimpleNamespace(Thing=_Helper))
    assert _surface.moved(["lucid.deep.Thing"], _LISTED, roots) == {}


def test_a_spelling_rebound_to_another_listed_object_maps_to_that_one() -> None:
    # Which object a recorded spelling once named is not recorded; what is
    # judged is the object it reaches now.  A caller of the old name gets it.
    listed = [("lucid.Thing", _Thing), ("lucid.Other", _Helper)]
    roots = _roots(Thing=_Thing, Other=_Helper, deep=types.SimpleNamespace(Thing=_Helper))
    assert _surface.moved(["lucid.deep.Thing"], listed, roots) == {
        "lucid.deep.Thing": "lucid.Other"
    }


def test_the_real_tree_lists_one_spelling_of_an_object_two_modules_export() -> None:
    import lucid
    import lucid.autograd

    assert lucid.autograd.enable_grad is lucid.enable_grad
    assert _surface.moved(["lucid.autograd.enable_grad", "lucid.enable_grad"]) == {
        "lucid.autograd.enable_grad": "lucid.enable_grad"
    }
