"""Refuse a tolerance that makes a tiny expected value impossible to fail (LCD-949).

Two defaults put an *absolute floor* under a relative check:

* ``pytest.approx(x, rel=R)`` still compares within ``max(R * |x|, 1e-12)``;
* ``np.allclose`` compares within ``atol + rtol * |b|`` with ``atol = 1e-8``.

When the expected value is far below the floor, the floor *is* the tolerance:
``approx(4.1e-37, rel=1e-10) == 0.0`` is true, and so is
``np.allclose(0.0, 1.4e-200, rtol=1e-12)``.  A guard written for "this tail value
came back 0" then cannot fail, and a reviewer cannot see it -- the assertion reads
as a tight one -- so the harness checks it, at the comparison.

The rules, for a nonzero finite expected value ``e``:

* ``pytest.approx``: with no ``abs=`` given, the floor (1e-12) is more than 10x
  looser than the relative term (``rel * |e|``, ``rel`` defaulting to 1e-6).  Naming
  an ``abs=`` is the way out: ``abs=0`` for an exact expectation, an explicit value
  (with a comment) when ``e`` is genuinely ~0.
* ``np.allclose`` with its default ``atol``: the same, ``atol >= max|e| / 10``.
* ``np.allclose`` / ``np.testing.assert_allclose`` with an ``atol`` given: it is the
  author's choice, so only a tolerance the all-zeros array would pass is refused
  (``atol + rtol * max|e| >= max|e|``).  A looser band that still rejects zero
  (half precision, an accumulating reduction) is not a vacuous one.  When the
  expected operand really is roundoff -- an analytic zero such as a window's end
  point -- say so with ``expects_noise("why")``.

Violations raise :class:`VacuousToleranceError` (``LUCID_TEST_TOLERANCE_GUARD=warn``
turns it into a warning, to triage a tree at once; ``off`` disables it).  The numpy
functions are checked only when called from inside ``lucid/test``; the approx check
applies wherever ``pytest.approx`` is used.  Installed once, from the root
``conftest.py`` and before any test module is imported, so ``from numpy import
allclose`` bindings are checked too.
"""

import contextlib
import functools
import math
import numbers
import os
import sys
import warnings
from collections.abc import Callable, Iterator
from contextvars import ContextVar
from pathlib import Path
from typing import TypeVar

import _pytest
import numpy as np
from _pytest.python_api import ApproxScalar

_R = TypeVar("_R")

MODE_ENV = "LUCID_TEST_TOLERANCE_GUARD"
_THIS_FILE = str(Path(__file__).resolve())
_TEST_ROOT = str(Path(_THIS_FILE).parent.parent)
_PYTEST_ROOT = str(Path(_pytest.__file__).resolve().parent)
_APPROX_FLOOR = 1e-12
_RATIO = 10.0
_MARK = "_lucid_tolerance_guard"

_noise: ContextVar[tuple[str, ...]] = ContextVar("lucid_expected_noise", default=())


class VacuousToleranceError(Exception):
    """A tolerance so loose that the expected value cannot fail the comparison.

    Not an ``AssertionError``: ``pytest.raises(AssertionError)`` in a test of the
    code under test must not swallow it.
    """


class VacuousToleranceWarning(UserWarning):
    """The ``warn`` mode of :class:`VacuousToleranceError`."""


@contextlib.contextmanager
def expects_noise(reason: str) -> Iterator[None]:
    """Declare that the expected operand in this block is roundoff, not a value.

    ``atol`` then *is* the comparison: both sides are meant to be ~0 within it.  The
    reason is the comment an ``abs=`` would carry; it must say what is analytically
    zero.
    """
    if not reason.strip():
        raise ValueError("expects_noise needs the reason the expected value is ~0")
    token = _noise.set((*_noise.get(), reason))
    try:
        yield
    finally:
        _noise.reset(token)


def noise_when_below(
    expected: object, atol: float, reason: str
) -> contextlib.AbstractContextManager[None]:
    """``expects_noise(reason)`` when everything in ``expected`` is within ``atol`` of zero.

    For a sweep whose expected operand is a real value in most cells and analytic-zero
    roundoff in a few: the cells that are not noise keep the guard.
    """
    if _expected_scale(expected) <= atol:
        return expects_noise(reason)
    return contextlib.nullcontext()


def _mode() -> str:
    return os.environ.get(MODE_ENV, "error")


def _outside_stacklevel() -> int:
    """The ``warnings.warn`` stacklevel, from ``_report``'s caller, of the first test frame."""
    level, frame = 1, sys._getframe(2)
    while frame.f_back is not None and frame.f_code.co_filename.startswith(
        (_THIS_FILE, _PYTEST_ROOT)
    ):
        level, frame = level + 1, frame.f_back
    return level


def _report(message: str) -> None:
    mode = _mode()
    if mode == "off":
        return
    if mode == "warn":
        warnings.warn(message, VacuousToleranceWarning, stacklevel=_outside_stacklevel() + 1)
        return
    raise VacuousToleranceError(message)


def _approx_message(expected: object, rel: float, relative_term: float) -> str:
    return (
        f"pytest.approx({expected!r}, rel={rel:g}) has no abs=, so the "
        f"{_APPROX_FLOOR:g} default floor decides, not rel*|expected| = "
        f"{relative_term:.3g}: a 0.0 result passes.  Give abs=0 for an exact "
        f"expectation, or an explicit abs= (with a comment) when the value is "
        f"genuinely ~0."
    )


def _checked_approx_eq(original: Callable[[ApproxScalar, object], bool]) -> Callable[..., bool]:
    @functools.wraps(original)
    def __eq__(self: ApproxScalar, actual: object) -> bool:
        expected = self.expected
        if (
            self.abs is None
            and isinstance(expected, numbers.Complex)
            and not isinstance(expected, bool)
            and expected != 0
        ):
            magnitude = abs(expected)
            if math.isfinite(magnitude):
                rel = self.rel if self.rel is not None else self.DEFAULT_RELATIVE_TOLERANCE
                relative_term = float(rel) * magnitude
                if _APPROX_FLOOR > _RATIO * relative_term:
                    _report(_approx_message(expected, float(rel), relative_term))
        return original(self, actual)

    return __eq__


def _expected_scale(expected: object) -> float:
    """Largest finite magnitude of ``expected``; 0.0 if there is none or it is not numeric."""
    try:
        arr = np.asarray(expected)
        if arr.dtype.kind not in "fiuc":
            return 0.0
        mags = np.abs(arr)
        return float(np.max(mags, initial=0.0, where=np.isfinite(mags)))
    except (TypeError, ValueError):
        return 0.0


def _largest(value: object) -> float:
    try:
        return float(np.max(np.asarray(value)))
    except (TypeError, ValueError):
        return 0.0


def _vacuous_closeness(
    args: tuple[object, ...],
    kwargs: dict[str, object],
    *,
    expected_kw: str,
    default_rtol: float,
    default_atol: float,
) -> str | None:
    """Why this call's tolerance cannot fail, or ``None`` if it can."""
    given = len(args) > 3 or "atol" in kwargs
    atol = _largest(args[3] if len(args) > 3 else kwargs.get("atol", default_atol))
    if atol <= 0.0:
        return None
    expected = args[1] if len(args) > 1 else kwargs.get(expected_kw)
    scale = _expected_scale(expected)
    if scale <= 0.0:
        return None
    if given:
        rtol = _largest(args[2] if len(args) > 2 else kwargs.get("rtol", default_rtol))
        if atol + rtol * scale < scale:
            return None
        return (
            f"atol={atol:g} accepts an all-zeros result against an expected operand "
            f"of magnitude {scale:.3g}"
        )
    if atol < scale / _RATIO:
        return None
    return f"the default atol={atol:g} swamps an expected operand of magnitude {scale:.3g}"


def _checked_closeness(
    original: Callable[..., _R], name: str, expected_kw: str, rtol: float, atol: float
) -> Callable[..., _R]:
    @functools.wraps(original)
    def checked(*args: object, **kwargs: object) -> _R:
        if (
            _mode() != "off"
            and not _noise.get()
            and sys._getframe(1).f_code.co_filename.startswith(_TEST_ROOT)
        ):
            why = _vacuous_closeness(
                args, kwargs, expected_kw=expected_kw, default_rtol=rtol, default_atol=atol
            )
            if why is not None:
                _report(
                    f"{name}: {why}, so a 0 result passes.  Use atol=0 for an exact "
                    f"expectation, an atol taken from the operand's own scale "
                    f"(e.g. rtol * max|expected|), or wrap a comparison of "
                    f"roundoff in expects_noise('why')."
                )
        return original(*args, **kwargs)

    setattr(checked, _MARK, True)
    return checked


def install() -> None:
    """Wrap the three comparison entry points; idempotent."""
    if not getattr(ApproxScalar.__eq__, _MARK, False):
        checked = _checked_approx_eq(ApproxScalar.__eq__)
        setattr(checked, _MARK, True)
        setattr(ApproxScalar, "__eq__", checked)
    if not getattr(np.allclose, _MARK, False):
        setattr(np, "allclose", _checked_closeness(np.allclose, "np.allclose", "b", 1e-5, 1e-8))
    if not getattr(np.testing.assert_allclose, _MARK, False):
        setattr(
            np.testing,
            "assert_allclose",
            _checked_closeness(
                np.testing.assert_allclose, "np.testing.assert_allclose", "desired", 1e-7, 0.0
            ),
        )
