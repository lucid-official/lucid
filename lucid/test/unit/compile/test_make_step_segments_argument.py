"""``make_step(segments=...)`` is checked where it enters (LCD-690).

``0`` and ``-1`` used to run as the monolithic step without a word, and
``2.5`` reached the splitter and failed as a bare ``TypeError`` about
``float`` and ``int``.  Each wrong value now fails at the entry, by what is
wrong with it, before any model is touched.
"""

import pytest

import lucid.nn as nn
import lucid.nn.functional as F
from lucid.compile import make_step


class _Two:
    """Anything that converts to an index counts as an integer."""

    def __index__(self) -> int:
        return 2


def _make(segments: object) -> object:
    return make_step(nn.Linear(4, 2), F.mse_loss, segments=segments)  # type: ignore[arg-type]


# value, the exception it raises, what the message names.
INVALID_SEGMENTS = [
    pytest.param(0, ValueError, "positive int", id="zero"),
    pytest.param(-1, ValueError, "positive int", id="negative"),
    pytest.param("two", ValueError, "'two'", id="unknown-string"),
    pytest.param("AUTO", ValueError, "'AUTO'", id="auto-is-lowercase"),
    pytest.param(2.5, TypeError, "float", id="float"),
    pytest.param(2.0, TypeError, "float", id="whole-float"),
    pytest.param(None, TypeError, "NoneType", id="none"),
    pytest.param(True, TypeError, "bool", id="bool"),
    pytest.param([2], TypeError, "list", id="list"),
]


@pytest.mark.parametrize("segments, exc, names", INVALID_SEGMENTS)
def test_a_wrong_segments_value_is_refused_at_the_entry(
    segments: object, exc: type[Exception], names: str
) -> None:
    with pytest.raises(exc, match=names):
        _make(segments)


def test_the_message_says_what_is_accepted() -> None:
    with pytest.raises(TypeError, match=r"positive int or 'auto'"):
        _make(1.5)


def test_one_is_the_monolithic_step() -> None:
    assert callable(_make(1))


@pytest.mark.parametrize(
    "segments", [pytest.param(3, id="int"), pytest.param(_Two(), id="index"), "auto"]
)
def test_a_valid_count_goes_on_to_the_splitter(segments: object) -> None:
    """A good value passes the entry: the splitter then has its own say (Sequential only)."""
    with pytest.raises(TypeError, match="nn.Sequential"):
        _make(segments)
