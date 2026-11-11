"""Nothing the last release shipped stops accepting the calls it accepted.

Lucid 3.x renamed zoo factories, dropped names and changed constructors in
minor releases, so code that ran on one 3.x release stopped importing on
the next.  ``lucid/test/audit/released_surface.json`` holds the last
release's surface — every public name and what a caller could pass it —
and this holds the tree to it.  See :mod:`lucid.test.audit._released` for
what counts as a break and :mod:`lucid._deprecation` for how a name leaves.
"""

import types

import pytest

import lucid
from lucid.test.audit import _released


def test_the_snapshot_is_the_release_this_tree_is_versioned_as() -> None:
    # The release commit bumps the version and rewrites the snapshot
    # together; a bump without the rewrite would hold the next release to
    # the previous one's surface, and a rewrite without the bump would let
    # a development tree erase what the last release promised.
    assert _released.load()["version"] == lucid.__version__, (
        "the release commit refreshes the released surface: "
        "python -m lucid.test.audit._released --update"
    )


def test_every_released_call_still_binds() -> None:
    released = _released.load()["symbols"]
    problems = _released.breaks(released, _released.collect(released=released), lucid.__version__)
    assert not problems, (
        f"{len(problems)} released call(s) no longer bind — keep the old "
        "spelling working and mark it with lucid._deprecation.deprecated:\n  "
        + "\n  ".join(problems[:40])
    )


def test_the_surface_it_reads_is_the_whole_surface() -> None:
    # An empty or truncated collection would pass the check above by
    # comparing against nothing.
    current = _released.collect()
    assert len(current) >= 3500
    for name in ("lucid.add", "nn.Module.state_dict", "lucid.models.resnet_50"):
        assert name in current


COMPATIBLE = [
    (["a:x"], ["a:x", "a:dim="], "a new optional parameter"),
    (["a:x", "k:keepdim="], ["a:x", "k:keepdim=", "k:out="], "a new optional keyword"),
    (["p:x"], ["p:input"], "a positional-only rename"),
    (["a:x", "a:dim"], ["a:x", "a:dim="], "a parameter given a default"),
    (["a:x", "k:flag="], ["a:x", "a:flag="], "a keyword-only parameter opened to position"),
    (["a:x"], ["v:args", "w:kwargs"], "a catch-all"),
    (None, ["a:x"], "a builtin that became readable"),
]

BREAKING = [
    (["a:x", "a:dim"], ["a:x", "a:axis"], "was renamed"),
    (["a:x", "a:dim=", "a:keepdim="], ["a:x", "a:keepdim=", "a:dim="], "moved"),
    (["a:x", "a:dim="], ["a:x", "k:dim="], "no longer be passed by position"),
    (["a:x", "a:dim="], ["a:x", "a:dim"], "is required now"),
    (["a:x"], ["a:x", "a:dim"], "new parameter 'dim' is required"),
    (["a:x"], ["a:x", "k:dim"], "new keyword 'dim' is required"),
    (["a:x", "k:flag="], ["a:x"], "keyword 'flag' is no longer accepted"),
    (["a:x", "w:kwargs"], ["a:x"], "extra keyword arguments"),
    (["a:x", "v:args"], ["a:x"], "extra positional arguments"),
]


@pytest.mark.parametrize("old,new,why", COMPATIBLE, ids=[c[2] for c in COMPATIBLE])
def test_a_call_that_still_binds_is_not_a_break(
    old: list[str] | None, new: list[str], why: str
) -> None:
    assert _released.incompatible(old, new) is None


@pytest.mark.parametrize("old,new,why", BREAKING, ids=[b[2] for b in BREAKING])
def test_a_call_that_stops_binding_is_a_break(old: list[str], new: list[str], why: str) -> None:
    reason = _released.incompatible(old, new)
    assert reason is not None and why in reason


def _released_with(**marks: str) -> dict[str, dict[str, object]]:
    return {
        "lucid.models.old_net": {"params": ["a:pretrained="], "deprecated": marks},
        "lucid.models.OldNet": {"params": ["a:config"], "deprecated": marks},
        "lucid.models.OldNet.forward": {"params": ["a:self", "a:x"]},
        "lucid.models.kept": {"params": ["a:x"]},
    }


def test_a_name_leaves_only_in_the_release_it_was_deprecated_until() -> None:
    released = _released_with(since="3.16.0", removal="3.18.0")
    current = {"lucid.models.kept": {"params": ["a:x"]}}
    early = _released.breaks(released, current, "3.17.4")
    assert len(early) == 3 and all("deprecated until" in p for p in early)
    # A member leaves with the class it belongs to.
    assert _released.breaks(released, current, "3.18.0") == []


def test_a_name_removed_without_warning_is_a_break() -> None:
    released = {"lucid.models.mobilenet_v1": {"params": ["a:pretrained="]}}
    assert _released.breaks(released, {}, "4.0.0") == [
        "lucid.models.mobilenet_v1: removed without a deprecation"
    ]


# A class's surface is what lookup resolves on it, not what it defines
# itself.  ``__module__`` puts these fixtures inside Lucid: the collector
# follows the MRO only through Lucid's own classes.
class _PrivateBase:
    __module__ = "lucid.fixture"

    def extra_repr(self) -> str:
        return ""


class _OwnMethod:
    __module__ = "lucid.fixture"

    def extra_repr(self) -> str:
        return ""


class _MovedToPrivateBase(_PrivateBase):
    __module__ = "lucid.fixture"


class _Deleted:
    __module__ = "lucid.fixture"


class _OnlyStr(str):
    __module__ = "lucid.fixture"


def _members_of(cls: type, held: tuple[type, ...] = ()) -> _released.Surface:
    out: _released.Surface = {}
    _released._members("lucid.fixture.Layer", cls, out, held)
    return out


def test_a_method_moved_to_a_private_base_is_not_a_removal() -> None:
    released = _members_of(_OwnMethod)
    assert released == {"lucid.fixture.Layer.extra_repr": {"params": ["a:self"]}}
    assert _released.breaks(released, _members_of(_MovedToPrivateBase), "3.99.0") == []


def test_a_deleted_method_is_still_a_removal() -> None:
    released = _members_of(_OwnMethod)
    assert _released.breaks(released, _members_of(_Deleted), "3.99.0") == [
        "lucid.fixture.Layer.extra_repr: removed without a deprecation"
    ]


def test_a_method_inherited_from_a_held_class_is_recorded_once() -> None:
    # The check reads it on the subclass; the snapshot holds it on the base.
    assert "lucid.fixture.Layer.extra_repr" in _members_of(_MovedToPrivateBase)
    assert _members_of(_MovedToPrivateBase, held=(_PrivateBase,)) == {}


def test_methods_defined_outside_lucid_are_not_surface() -> None:
    assert _members_of(_OnlyStr) == {}


def test_the_collected_surface_follows_the_mro() -> None:
    current = _released.collect()
    assert "nn.ConvTranspose2d.extra_repr" in current
    assert "lucid.nn.quantized.dynamic.LSTM.from_float" in current
    assert "nn.Linear.state_dict" in current
    compact = _released.collect(inherited=False)
    assert "nn.Module.state_dict" in compact
    assert "nn.Linear.state_dict" not in compact


# A released spelling is judged by what it reaches today, not by whether the
# walk lists it: the walk keeps one spelling per object, the shallowest, so an
# object exported from a shallower module reads as gone from the old one
# (LCD-959: ``LoraRuntimeConfig`` could not be named in ``lucid.peft``).
# ``roots`` and ``listed`` below are the two things the real collection reads —
# the namespace attribute access starts from, and the spellings the walk lists.
class _Thing:
    __module__ = "lucid.fixture"

    def run(self, x: int) -> int:
        return x


class _ThingWithoutRun:
    __module__ = "lucid.fixture"


class _Helper:
    __module__ = "lucid.fixture"

    def run(self, y: int) -> int:
        return y


class _Compatible:
    __module__ = "lucid.fixture"

    def run(self, x: int, scale: int = 1) -> int:
        return x * scale


def _namespace(**attrs: object) -> types.SimpleNamespace:
    return types.SimpleNamespace(**attrs)


def _snapshot(listed: list[tuple[str, object]], **top: object) -> _released.Surface:
    return _released._collect(listed, {"lucid": _namespace(**top)}, inherited=False, released=())


def _today(
    listed: list[tuple[str, object]], released: _released.Surface, **top: object
) -> _released.Surface:
    roots = {"lucid": _namespace(**top)}
    return _released._collect(listed, roots, inherited=True, released=released)


def _shipped() -> _released.Surface:
    """A release that listed ``Thing`` under ``deep``, the only module that exported it."""
    return _snapshot([("lucid.deep.Thing", _Thing)], deep=_namespace(Thing=_Thing))


def test_an_object_exported_from_a_shallower_module_keeps_its_old_spelling() -> None:
    released = _shipped()
    assert set(released) == {"lucid.deep.Thing", "lucid.deep.Thing.run"}
    # Now ``lucid`` exports it too, and the walk lists only that.
    listed = [("lucid.Thing", _Thing)]
    top = {"Thing": _Thing, "deep": _namespace(Thing=_Thing)}
    # What the check read before, which is what held the promotion back:
    unread = _today(listed, {}, **top)
    assert [p.split(": ")[0] for p in _released.breaks(released, unread, "3.99.0")] == [
        "lucid.deep.Thing",
        "lucid.deep.Thing.run",
    ]
    current = _today(listed, released, **top)
    assert _released.breaks(released, current, "3.99.0") == []
    # ... under the entry the object has now, so a break in it is still one.
    assert current["lucid.deep.Thing"] == current["lucid.Thing"]
    assert current["lucid.deep.Thing.run"] == current["lucid.Thing.run"]


def test_a_promoted_spelling_is_still_held_by_the_signature_rules() -> None:
    released = _shipped()
    listed = [("lucid.Thing", _Helper)]
    current = _today(listed, released, Thing=_Helper, deep=_namespace(Thing=_Helper))
    assert _released.breaks(released, current, "3.99.0") == [
        "lucid.deep.Thing.run: parameter 'x' was renamed, moved or made positional-only"
    ]


def test_a_spelling_that_reaches_nothing_is_removed() -> None:
    released = _shipped()
    listed = [("lucid.Thing", _Thing)]
    for deep in (_namespace(), _namespace(Thing=None)):
        current = _today(listed, released, Thing=_Thing, deep=deep)
        assert [p.split(": ")[0] for p in _released.breaks(released, current, "3.99.0")] == [
            "lucid.deep.Thing",
            "lucid.deep.Thing.run",
        ]
    # A member the class no longer has goes under it, wherever the class is listed.
    current = _today(
        [("lucid.Thing", _ThingWithoutRun)],
        released,
        Thing=_ThingWithoutRun,
        deep=_namespace(Thing=_ThingWithoutRun),
    )
    assert _released.breaks(released, current, "3.99.0") == [
        "lucid.deep.Thing.run: removed without a deprecation"
    ]


def test_a_spelling_that_reaches_an_object_the_walk_does_not_list_is_removed() -> None:
    released = _shipped()
    # ``deep.Thing`` is still an attribute, but of a helper nobody exports.
    current = _today(
        [("lucid.Thing", _Thing)], released, Thing=_Thing, deep=_namespace(Thing=_Helper)
    )
    assert [p.split(": ")[0] for p in _released.breaks(released, current, "3.99.0")] == [
        "lucid.deep.Thing",
        "lucid.deep.Thing.run",
    ]


def test_a_spelling_rebound_to_another_public_object_must_still_bind() -> None:
    released = _shipped()
    # ``deep.Thing`` now reaches an object the walk lists as ``other``.
    other = _today(
        [("lucid.other", _Helper)], released, other=_Helper, deep=_namespace(Thing=_Helper)
    )
    assert any("renamed" in p for p in _released.breaks(released, other, "3.99.0"))
    # Calls that bound to the old object and still bind to the new one are all a caller sees.
    same = _today(
        [("lucid.other", _Compatible)],
        released,
        other=_Compatible,
        deep=_namespace(Thing=_Compatible),
    )
    assert _released.breaks(released, same, "3.99.0") == []


def test_a_deprecated_spelling_leaves_when_it_reaches_nothing_and_its_release_has_come() -> None:
    marked = {"since": "3.16.0", "removal": "3.18.0"}
    released = {"lucid.deep.Thing": {"params": [], "deprecated": marked}}
    gone = _today([("lucid.Thing", _Thing)], released, Thing=_Thing, deep=_namespace())
    early = _released.breaks(released, gone, "3.17.4")
    assert len(early) == 1 and "deprecated until" in early[0]
    assert _released.breaks(released, gone, "3.18.0") == []
    # Still reaching the object, it needs no deprecation at all.
    kept = _today([("lucid.Thing", _Thing)], released, Thing=_Thing, deep=_namespace(Thing=_Thing))
    assert _released.breaks(released, kept, "3.17.4") == []


def test_an_update_carries_a_promoted_spelling_forward_and_drops_a_removed_one() -> None:
    released = _snapshot(
        [("lucid.deep.Thing", _Thing), ("lucid.deep.Gone", _Helper)],
        deep=_namespace(Thing=_Thing, Gone=_Helper),
    )
    roots = {"lucid": _namespace(Thing=_Thing, deep=_namespace(Thing=_Thing))}
    written = _released._collect(
        [("lucid.Thing", _Thing)], roots, inherited=False, released=released
    )
    assert set(written) == {
        "lucid.Thing",
        "lucid.Thing.run",
        "lucid.deep.Thing",
        "lucid.deep.Thing.run",
    }


def test_the_collected_surface_reads_released_spellings_from_the_real_tree() -> None:
    # ``lucid.autograd`` exports the object ``lucid`` does; the walk lists it once.
    assert lucid.autograd.enable_grad is lucid.enable_grad
    assert "lucid.autograd.enable_grad" not in _released.collect()
    current = _released.collect(released=["lucid.autograd.enable_grad", "lucid.no_such_name"])
    assert current["lucid.autograd.enable_grad"] == current["lucid.enable_grad"]
    assert "lucid.no_such_name" not in current
