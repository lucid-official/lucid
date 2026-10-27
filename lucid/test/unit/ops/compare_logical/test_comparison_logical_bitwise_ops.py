"""Comparison + logical + bitwise + shifts."""

import numpy as np
import pytest

import lucid

# ── comparison ───────────────────────────────────────────────────────────


class TestComparison:
    def test_eq(self, device: str) -> None:
        a = lucid.tensor([1.0, 2.0, 3.0], device=device)
        b = lucid.tensor([1.0, 0.0, 3.0], device=device)
        np.testing.assert_array_equal(lucid.eq(a, b).numpy(), [True, False, True])

    def test_ne(self, device: str) -> None:
        a = lucid.tensor([1.0, 2.0], device=device)
        b = lucid.tensor([2.0, 2.0], device=device)
        np.testing.assert_array_equal(lucid.ne(a, b).numpy(), [True, False])

    def test_lt(self, device: str) -> None:
        a = lucid.tensor([1.0, 2.0, 3.0], device=device)
        b = lucid.tensor([2.0, 2.0, 2.0], device=device)
        np.testing.assert_array_equal(lucid.lt(a, b).numpy(), [True, False, False])

    def test_le(self, device: str) -> None:
        a = lucid.tensor([1.0, 2.0, 3.0], device=device)
        b = lucid.tensor([2.0, 2.0, 2.0], device=device)
        np.testing.assert_array_equal(lucid.le(a, b).numpy(), [True, True, False])

    def test_gt(self, device: str) -> None:
        a = lucid.tensor([1.0, 2.0, 3.0], device=device)
        b = lucid.tensor([2.0, 2.0, 2.0], device=device)
        np.testing.assert_array_equal(lucid.gt(a, b).numpy(), [False, False, True])

    def test_ge(self, device: str) -> None:
        a = lucid.tensor([1.0, 2.0, 3.0], device=device)
        b = lucid.tensor([2.0, 2.0, 2.0], device=device)
        np.testing.assert_array_equal(lucid.ge(a, b).numpy(), [False, True, True])

    def test_isclose(self, device: str) -> None:
        a = lucid.tensor([1.0, 2.0], device=device)
        b = lucid.tensor([1.000001, 2.5], device=device)
        out = lucid.isclose(a, b, atol=1e-3).numpy()
        np.testing.assert_array_equal(out, [True, False])


# ── logical ──────────────────────────────────────────────────────────────


class TestLogical:
    def test_logical_and(self, device: str) -> None:
        a = lucid.tensor([True, True, False], dtype=lucid.bool_, device=device)
        b = lucid.tensor([True, False, False], dtype=lucid.bool_, device=device)
        np.testing.assert_array_equal(lucid.logical_and(a, b).numpy(), [True, False, False])

    def test_logical_or(self, device: str) -> None:
        a = lucid.tensor([True, False, False], dtype=lucid.bool_, device=device)
        b = lucid.tensor([False, True, False], dtype=lucid.bool_, device=device)
        np.testing.assert_array_equal(lucid.logical_or(a, b).numpy(), [True, True, False])

    def test_logical_xor(self, device: str) -> None:
        a = lucid.tensor([True, True, False], dtype=lucid.bool_, device=device)
        b = lucid.tensor([True, False, False], dtype=lucid.bool_, device=device)
        np.testing.assert_array_equal(lucid.logical_xor(a, b).numpy(), [False, True, False])

    def test_logical_not(self, device: str) -> None:
        a = lucid.tensor([True, False, True], dtype=lucid.bool_, device=device)
        np.testing.assert_array_equal(lucid.logical_not(a).numpy(), [False, True, False])


# ── bitwise (integer / bool) ────────────────────────────────────────────


class TestBitwise:
    def test_bitwise_and(self, device: str) -> None:
        a = lucid.tensor([0b1100, 0b1010], dtype=lucid.int32, device=device)
        b = lucid.tensor([0b1010, 0b0110], dtype=lucid.int32, device=device)
        out = lucid.bitwise_and(a, b).numpy()
        np.testing.assert_array_equal(out, [0b1000, 0b0010])

    def test_bitwise_or(self, device: str) -> None:
        a = lucid.tensor([0b1100, 0b1010], dtype=lucid.int32, device=device)
        b = lucid.tensor([0b0011, 0b0101], dtype=lucid.int32, device=device)
        np.testing.assert_array_equal(lucid.bitwise_or(a, b).numpy(), [0b1111, 0b1111])

    def test_bitwise_xor(self, device: str) -> None:
        a = lucid.tensor([0b1100, 0b1010], dtype=lucid.int32, device=device)
        b = lucid.tensor([0b1010, 0b0110], dtype=lucid.int32, device=device)
        np.testing.assert_array_equal(lucid.bitwise_xor(a, b).numpy(), [0b0110, 0b1100])

    def test_bitwise_not(self, device: str) -> None:
        a = lucid.tensor([0, 1, 2], dtype=lucid.int32, device=device)
        np.testing.assert_array_equal(lucid.bitwise_not(a).numpy(), [-1, -2, -3])


class TestShifts:
    def test_left_shift(self, device: str) -> None:
        a = lucid.tensor([1, 2, 4, 8], dtype=lucid.int32, device=device)
        b = lucid.tensor([1, 1, 2, 3], dtype=lucid.int32, device=device)
        np.testing.assert_array_equal(lucid.bitwise_left_shift(a, b).numpy(), [2, 4, 16, 64])

    def test_right_shift(self, device: str) -> None:
        a = lucid.tensor([8, 16, 64], dtype=lucid.int32, device=device)
        b = lucid.tensor([1, 2, 3], dtype=lucid.int32, device=device)
        np.testing.assert_array_equal(lucid.bitwise_right_shift(a, b).numpy(), [4, 4, 8])

    def test_oob_clamps_to_zero_cpu(self, device_cpu_only: str) -> None:
        # OOB shift contract is CPU-only: the CPU kernel explicitly
        # clamps shifts ≥ width to 0.  MLX's GPU shift uses the natural
        # ``count mod width`` wraparound, so this contract isn't held
        # on metal — documented divergence, not a bug.
        a = lucid.tensor([1], dtype=lucid.int32, device=device_cpu_only)
        b = lucid.tensor([100], dtype=lucid.int32, device=device_cpu_only)
        np.testing.assert_array_equal(lucid.bitwise_left_shift(a, b).numpy(), [0])

    def test_negative_right_shift_signed(self, device: str) -> None:
        # Arithmetic right shift on a signed negative preserves sign.
        a = lucid.tensor([-8], dtype=lucid.int8, device=device)
        b = lucid.tensor([1], dtype=lucid.int8, device=device)
        np.testing.assert_array_equal(lucid.bitwise_right_shift(a, b).numpy(), [-4])

    def test_shift_rejects_bool(self, device: str) -> None:
        with pytest.raises(Exception):
            lucid.bitwise_left_shift(
                lucid.tensor([True], dtype=lucid.bool_, device=device),
                lucid.tensor([True], dtype=lucid.bool_, device=device),
            )

    def test_shift_rejects_float(self, device: str) -> None:
        with pytest.raises(Exception):
            lucid.bitwise_left_shift(
                lucid.tensor([1.0], device=device),
                lucid.tensor([1.0], device=device),
            )


# ── boolean reductions all/any ──────────────────────────────────────────


class TestBoolReductions:
    def test_all_true(self, device: str) -> None:
        t = lucid.tensor([True, True, True], dtype=lucid.bool_, device=device)
        assert bool(lucid.all(t).item())

    def test_all_false(self, device: str) -> None:
        t = lucid.tensor([True, False, True], dtype=lucid.bool_, device=device)
        assert not bool(lucid.all(t).item())

    def test_any_true(self, device: str) -> None:
        t = lucid.tensor([False, False, True], dtype=lucid.bool_, device=device)
        assert bool(lucid.any(t).item())

    def test_any_false(self, device: str) -> None:
        t = lucid.tensor([False, False, False], dtype=lucid.bool_, device=device)
        assert not bool(lucid.any(t).item())
