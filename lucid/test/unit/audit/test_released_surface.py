"""Nothing the last release shipped stops accepting the calls it accepted.

Lucid 3.x renamed zoo factories, dropped names and changed constructors in
minor releases, so code that ran on one 3.x release stopped importing on
the next.  ``lucid/test/audit/released_surface.json`` holds the last
release's surface — every public name and what a caller could pass it —
and this holds the tree to it.  See :mod:`lucid.test.audit._released` for
what counts as a break and :mod:`lucid._deprecation` for how a name leaves.
"""

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
    problems = _released.breaks(_released.load()["symbols"], _released.collect(), lucid.__version__)
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
    (
        ["a:x", "k:flag="],
        ["a:x", "a:flag="],
        "a keyword-only parameter opened to position",
    ),
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
