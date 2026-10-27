"""A deprecated name keeps working, says so, and says when it goes."""

import warnings

import pytest

from lucid._deprecation import LucidDeprecationWarning, deprecated
from lucid.test.audit import _released


def _new(x: int, scale: int = 2) -> int:
    return x * scale


@deprecated(since="3.16.0", removal="3.18.0", alternative="lucid.new")
def _old(x: int, scale: int = 2) -> int:
    return _new(x, scale)


class _Base:
    def __init__(self, width: int, depth: int = 1) -> None:
        self.width, self.depth = width, depth


@deprecated(since="3.16.0", removal="4.0.0", alternative="_Base")
class _Renamed(_Base):
    pass


def test_a_deprecated_function_warns_and_still_answers() -> None:
    with pytest.warns(LucidDeprecationWarning, match=r"removed in 3\.18\.0; use lucid\.new"):
        assert _old(3, scale=4) == 12


def test_a_deprecated_class_warns_and_leaves_its_base_alone() -> None:
    with pytest.warns(LucidDeprecationWarning, match=r"_Renamed is deprecated since"):
        instance = _Renamed(8, depth=3)
    assert (instance.width, instance.depth) == (8, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _Base(8)


def test_the_warning_reaches_the_person_running_the_code() -> None:
    # Python shows FutureWarning by default and hides DeprecationWarning
    # outside __main__.
    assert issubclass(LucidDeprecationWarning, FutureWarning)
    assert not issubclass(LucidDeprecationWarning, DeprecationWarning)


def test_the_snapshot_records_the_promise() -> None:
    assert _released._entry(_old)["deprecated"] == {"since": "3.16.0", "removal": "3.18.0"}
    assert _released._entry(_Renamed)["deprecated"]["removal"] == "4.0.0"
    assert "deprecated" not in _released._entry(_Base)
    # The signature a caller sees is the old one, not the warning's wrapper.
    assert _released._entry(_old)["params"] == ["a:x", "a:scale="]


@pytest.mark.parametrize(
    "since,removal",
    [("3.16.0", "3.17.0"), ("3.16.0", "3.16.5"), ("3.16.0", "2.0.0"), ("3.16", "3.18.0")],
)
def test_a_removal_too_soon_or_malformed_is_refused(since: str, removal: str) -> None:
    with pytest.raises(ValueError):
        deprecated(since=since, removal=removal)
