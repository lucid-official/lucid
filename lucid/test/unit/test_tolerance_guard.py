"""The harness refuses a tolerance that cannot fail (LCD-949).

``approx(4.1e-37, rel=1e-10) == 0.0`` is true: the 1e-12 floor, not the relative term,
is the tolerance.  ``lucid/test/_fixtures/tolerance_guard.py`` makes that a loud error at
the comparison; these tests pin what it refuses and what it must leave alone.
"""

import math
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose as imported_assert_allclose

from lucid.test._fixtures import tolerance_guard
from lucid.test._fixtures.tolerance_guard import (
    VacuousToleranceError,
    VacuousToleranceWarning,
    expects_noise,
)


@pytest.fixture(autouse=True)
def _guard_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests are about the default mode, whatever the run's environment says."""
    monkeypatch.delenv(tolerance_guard.MODE_ENV, raising=False)


# ── pytest.approx ────────────────────────────────────────────────────────────


def test_the_guard_is_installed_in_this_run() -> None:
    assert getattr(pytest.approx(1.0).__class__.__eq__, "_lucid_tolerance_guard", False)
    assert getattr(np.allclose, "_lucid_tolerance_guard", False)
    assert getattr(np.testing.assert_allclose, "_lucid_tolerance_guard", False)
    assert getattr(imported_assert_allclose, "_lucid_tolerance_guard", False)


@pytest.mark.parametrize(
    "expected, kwargs",
    [
        (4.14e-37, {"rel": 1e-10}),
        (2.2e-17, {"rel": 3e-6}),
        (1e-30, {}),  # default rel 1e-6
        (-1e-20, {"rel": 1e-4}),
        (1.0, {"rel": 1e-14}),  # rel*|e| = 1e-14: the 1e-12 floor is 100x looser
        (1.5, {"rel": 5e-14}),
    ],
)
def test_approx_without_abs_refuses_a_floor_that_swamps_the_relative_term(
    expected: float, kwargs: dict[str, float]
) -> None:
    with pytest.raises(VacuousToleranceError, match="no abs="):
        assert expected == pytest.approx(expected, **kwargs)


def test_approx_refuses_a_sequence_and_an_array_element_by_element() -> None:
    with pytest.raises(VacuousToleranceError):
        assert [0.0] * 3 == pytest.approx([1e-20] * 3, rel=1e-4)
    with pytest.raises(VacuousToleranceError):
        assert np.zeros(3) == pytest.approx(np.full(3, 1e-20), rel=1e-4)
    with pytest.raises(VacuousToleranceError):
        assert {"a": 1e-20} == pytest.approx({"a": 1e-20}, rel=1e-4)


@pytest.mark.parametrize(
    "expected, kwargs",
    [
        (4.14e-37, {"rel": 1e-10, "abs": 0}),  # exact expectation, said so
        (2e-15, {"abs": 1e-14}),  # genuinely ~0, floor chosen
        (0.0, {}),  # the floor is the whole point of approx(0)
        (0.0, {"rel": 1e-6}),
        (1.0, {}),
        (1.0, {"rel": 1e-9}),
        (1.5, {"rel": 1e-14, "abs": 0}),
        (1e-6, {"rel": 1e-3}),  # rel*|e| = 1e-9, floor 1e3x below it
        (math.inf, {}),
        (math.nan, {"nan_ok": True}),
    ],
)
def test_approx_legitimate_forms_are_left_alone(expected: float, kwargs: dict[str, float]) -> None:
    assert expected == pytest.approx(expected, **kwargs) or math.isnan(expected)


def test_approx_ignores_non_numbers() -> None:
    assert "a" == pytest.approx("a")
    assert True == pytest.approx(True)  # noqa: E712
    assert (1.0, "x") == pytest.approx((1.0, "x"))


def test_a_fixed_approx_can_fail() -> None:
    assert not (0.0 == pytest.approx(4.14e-37, rel=1e-10, abs=0))
    assert not (4.14e-36 == pytest.approx(4.14e-37, rel=1e-10, abs=0))
    assert 4.14e-37 == pytest.approx(4.14e-37, rel=1e-10, abs=0)


# ── np.allclose / np.testing.assert_allclose ─────────────────────────────────


def test_allclose_refuses_a_default_atol_over_a_tiny_expected() -> None:
    with pytest.raises(VacuousToleranceError, match="default atol"):
        np.allclose(0.0, 1.414e-200, rtol=1e-12)
    with pytest.raises(VacuousToleranceError):
        np.allclose(np.zeros(3), np.full(3, 1e-12))
    with pytest.raises(VacuousToleranceError):
        np.allclose(a=np.zeros(1), b=np.full(1, 5e-8))  # within 10x of the 1e-8 default


def test_an_explicit_atol_is_refused_when_an_all_zeros_result_would_pass() -> None:
    expected = np.array([1e-8, -2e-8, 0.0])
    with pytest.raises(VacuousToleranceError, match="all-zeros"):
        np.testing.assert_allclose(np.zeros(3), expected, rtol=1e-6, atol=1e-7)
    with pytest.raises(VacuousToleranceError):
        imported_assert_allclose(np.zeros(3), desired=expected, rtol=1e-6, atol=1e-7)
    with pytest.raises(VacuousToleranceError):
        np.testing.assert_allclose(np.zeros(3), expected, 1e-6, 1e-7)  # positional atol
    with pytest.raises(VacuousToleranceError):
        np.allclose(np.zeros(3), expected, atol=1e-7)


@pytest.mark.parametrize(
    "expected, kwargs",
    [
        (np.array([1.0, 2.0]), {}),  # O(1) values: the default is a rounding floor
        (np.array([1.0, 1e-30]), {}),  # the array's scale decides, as for allclose itself
        (np.zeros(3), {}),  # comparing noise against exact zero is what atol is for
        (np.array([np.nan, np.inf]), {}),
        (np.array([1e-200]), {"atol": 0.0}),  # exact expectation, said so
        (np.array([1e-8]), {"atol": 1e-12}),  # an atol the operand's own scale can see
        (np.array([0.479]), {"atol": 0.06, "rtol": 1e-3}),  # a half-precision band: rejects zero
    ],
)
def test_allclose_legitimate_forms_are_left_alone(
    expected: np.ndarray, kwargs: dict[str, float]
) -> None:
    assert np.allclose(expected, expected, equal_nan=True, **kwargs)
    np.testing.assert_allclose(expected, expected, equal_nan=True, **kwargs)


def test_assert_allclose_without_an_atol_is_exact_and_unflagged() -> None:
    np.testing.assert_allclose(np.array([1e-200]), np.array([1e-200]), rtol=1e-12)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(np.zeros(1), np.array([1e-200]), rtol=1e-12)


def test_expects_noise_declares_a_comparison_of_roundoff() -> None:
    noise = np.array([1.4e-17, 0.0])
    with expects_noise("the window's end points are analytically zero"):
        np.testing.assert_allclose(np.zeros(2), noise, atol=1e-6)
        assert np.allclose(noise, noise)
    with pytest.raises(VacuousToleranceError):  # and it ends with the block
        np.testing.assert_allclose(np.zeros(2), noise, atol=1e-6)


def test_expects_noise_wants_a_reason() -> None:
    with pytest.raises(ValueError, match="reason"), expects_noise("  "):
        pass


def test_a_fixed_allclose_can_fail() -> None:
    want = np.hypot(1e-200, 1e-200)
    assert np.allclose(want, want, rtol=1e-12, atol=0)
    assert not np.allclose(0.0, want, rtol=1e-12, atol=0)
    assert not np.allclose(10 * want, want, rtol=1e-12, atol=0)


def test_numpy_calls_from_outside_the_test_tree_are_not_checked() -> None:
    # A compiled-from-elsewhere frame stands in for library code that calls allclose.
    outside = compile("np.allclose(0.0, 1.414e-200, rtol=1e-12)", "/elsewhere/lib.py", "eval")
    assert eval(outside, {"np": np}) is True  # noqa: S307


# ── modes ────────────────────────────────────────────────────────────────────


def test_warn_mode_reports_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(tolerance_guard.MODE_ENV, "warn")
    with pytest.warns(VacuousToleranceWarning, match="no abs="):
        assert 0.0 == pytest.approx(4.14e-37, rel=1e-10)
    with pytest.warns(VacuousToleranceWarning, match="default atol"):
        assert np.allclose(0.0, 1e-200)


def test_off_mode_checks_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(tolerance_guard.MODE_ENV, "off")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert 0.0 == pytest.approx(4.14e-37, rel=1e-10)
        assert np.allclose(0.0, 1e-200)


def test_install_is_idempotent() -> None:
    before = (pytest.approx(1.0).__class__.__eq__, np.allclose, np.testing.assert_allclose)
    tolerance_guard.install()
    after = (pytest.approx(1.0).__class__.__eq__, np.allclose, np.testing.assert_allclose)
    assert before == after
