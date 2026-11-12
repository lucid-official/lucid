"""Which free-function argument a method's receiver is — and that an unreadable annotation cannot move it.

``EntryPointAxis`` compares an op's free function with its method.  For an op whose
receiver is not the first argument (``lucid.where(condition, x, y)`` is
``x.where(condition, y)``) it derives the position from the two signatures' parameter
names.  The derivation used to read the signatures with their annotations evaluated and
answered ``0`` when that raised ``NameError``, so a signature naming a type nothing binds
put the receiver first and the axis compared two different questions.  Names and kinds
need only forward references.
"""

import pytest

from lucid.test.audit import _surface
from lucid.test.audit._axes import _receiver_position


def _define(source: str) -> object:
    namespace: dict[str, object] = {"__name__": "lucid.scratch"}
    exec(source, namespace)
    return namespace["fn"]


#: ``where``'s shape: the method names every free parameter but the one it became.
_FREE = "def fn(condition{c}, x{x}, y{y}): ...\n"
_METHOD = "def fn(self, condition{c}, y{y}): ...\n"


@pytest.mark.parametrize(
    "annotations",
    [
        {"c": "", "x": "", "y": ""},
        {"c": ": int", "x": ": int", "y": ": int"},
        {"c": ": Gone", "x": ": Gone", "y": ": Gone"},
        {"c": ": int", "x": ": Gone | None", "y": ": int"},
    ],
    ids=["bare", "readable", "nothing-binds-them", "partly-unreadable"],
)
def test_an_unreadable_annotation_does_not_change_the_receiver_position(
    annotations: dict[str, str],
) -> None:
    free = _define(_FREE.format(**annotations))
    method = _define(_METHOD.format(**annotations))
    assert _receiver_position(free, method) == 1


def test_a_receiver_first_pair_is_still_position_zero_when_unreadable() -> None:
    free = _define("def fn(x: Gone, other: Gone): ...\n")
    method = _define("def fn(self, other: Gone): ...\n")
    assert _receiver_position(free, method) == 0


def test_a_registry_injected_method_names_nothing_and_stays_first() -> None:
    free = _define("def fn(a: Gone, b: Gone, c: Gone): ...\n")
    method = _define("def fn(self, *args: Gone, **kwargs: Gone): ...\n")
    assert _receiver_position(free, method) == 0


def test_a_callable_with_no_signature_still_answers_first() -> None:
    assert _receiver_position(len, len) == 0


def _method_receiver_positions() -> dict[str, int]:
    """``_receiver_position`` for every op the entry axis compares, keyed by qualname."""
    found: dict[str, int] = {}
    for symbol in _surface.enumerate_surface():
        routes = list(_surface.counterparts(symbol))
        if len(routes) < 2:
            continue
        probe_fn = routes[0][1]
        for label, fn in routes:
            if label == "method":
                found[symbol.qualname] = _receiver_position(probe_fn, fn)
    return found


def test_the_released_surface_has_one_receiver_in_the_middle_and_none_unanswered() -> None:
    """The answer for every op, so a signature that stops reading shows as a change.

    Every method takes its receiver first except ``where``; ``-1`` ("no alignment to
    compare") would turn an op's entry-axis cell into not-applicable without a word.
    """
    positions = _method_receiver_positions()
    assert positions, "the entry axis found no op with a method spelling"
    assert {name: at for name, at in positions.items() if at != 0} == {
        "lucid.where": 1,
        "Tensor.where": 1,
    }
