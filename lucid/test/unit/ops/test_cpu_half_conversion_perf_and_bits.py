"""Bit-exactness and bulk conversion paths for CPU float16 and bfloat16 (LCD-298, LCD-462).

CPU ``float16`` and ``bfloat16`` widen/narrow/cast paths share ``core/Half.h``:
* ``float16 <-> float32`` uses ARM64 hardware ``__fp16`` (``fcvt`` / NEON
  auto-vectorized loops) without depending on ``<Accelerate/Accelerate.h>``.
* ``bfloat16 <-> float32`` uses branchless round-to-nearest-even shifts with
  quiet-NaN preservation so the compiler auto-vectorizes bulk loops with NEON.
"""

from pathlib import Path
import numpy as np
import pytest

import lucid


def _f32_from_u32(bits: list[int] | np.ndarray) -> np.ndarray:
    return np.asarray(bits, dtype=np.uint32).view(np.float32)


def _u16_of_f16_tensor(t: lucid.Tensor) -> np.ndarray:
    return t.numpy().view(np.uint16)


def _u32_of_f32_tensor(t: lucid.Tensor) -> np.ndarray:
    return t.numpy().view(np.uint32)


def test_core_headers_do_not_include_accelerate_or_undef_nil() -> None:
    core_dir = Path(lucid.__file__).resolve().parent / "_C" / "core"
    headers = sorted(core_dir.glob("*.h"))
    assert headers
    for hdr in headers:
        text = hdr.read_text(encoding="utf-8")
        assert "<Accelerate/" not in text, f"{hdr.name} must not include Accelerate"
        assert "#undef nil" not in text, f"{hdr.name} must not #undef nil"


def test_f16_to_f32_all_65536_bit_patterns() -> None:
    all_u16 = np.arange(65536, dtype=np.uint16)
    np_f16 = all_u16.view(np.float16)
    with np.errstate(invalid="ignore"):
        expected_f32 = np_f16.astype(np.float32)

    t_f16 = lucid.from_numpy(np_f16)
    got_f32 = t_f16.to(lucid.float32).numpy()

    nan_mask = np.isnan(expected_f32)
    assert np.array_equal(np.isnan(got_f32), nan_mask)
    assert np.array_equal(
        got_f32[~nan_mask].view(np.uint32),
        expected_f32[~nan_mask].view(np.uint32),
    )

    roundtrip_u16 = _u16_of_f16_tensor(lucid.from_numpy(got_f32).to(lucid.float16))
    assert np.array_equal(roundtrip_u16[~nan_mask], all_u16[~nan_mask])
    assert np.all((roundtrip_u16[nan_mask] & 0x7C00) == 0x7C00)
    assert np.all((roundtrip_u16[nan_mask] & 0x03FF) != 0)


def test_f32_to_f16_special_subnormal_and_ties() -> None:
    u32_cases = [
        # +0.0, -0.0
        (0x00000000, 0x0000),
        (0x80000000, 0x8000),
        # +inf, -inf, overflow -> +/-inf, max normal -> 0x7bff
        (0x7F800000, 0x7C00),
        (0xFF800000, 0xFC00),
        (0x47800000, 0x7C00),  # 65536.0 -> +inf
        (0xC7800000, 0xFC00),  # -65536.0 -> -inf
        (0x477FE000, 0x7BFF),  # 65504.0 -> max f16 normal
        # Subnormal ties (2^-24 is smallest f16 subnormal = 0x33800000 in f32):
        # 0.5 * 2^-24 (tie to even 0) -> 0x0000
        (0x33000000, 0x0000),
        # -0.5 * 2^-24 (tie to even -0) -> 0x8000
        (0xB3000000, 0x8000),
        # (0.5 + 2^-23) * 2^-24 (just above tie) -> 0x0001
        (0x33000001, 0x0001),
        # 1.5 * 2^-24 (tie to even 2) -> 0x0002
        (0x33C00000, 0x0002),
        # 2.5 * 2^-24 (tie to even 2) -> 0x0002
        (0x34200000, 0x0002),
        # 3.5 * 2^-24 (tie to even 4) -> 0x0004
        (0x34600000, 0x0004),
        # Normal ties around 1.0 (0x3c00): f32 mantissa bit 13 is f16 LSB, bit 12 is tie bit
        (0x3F801000, 0x3C00),  # tie with even LSB (0) -> 0x3c00
        (0x3F803000, 0x3C02),  # tie with odd LSB (1) -> rounds up to 0x3c02
        (0x3F801001, 0x3C01),  # tie + sticky bit -> rounds up to 0x3c01
    ]
    f32_arr = _f32_from_u32([u for u, _ in u32_cases])
    expected_u16 = np.array([e for _, e in u32_cases], dtype=np.uint16)

    got_u16 = _u16_of_f16_tensor(lucid.from_numpy(f32_arr).to(lucid.float16))
    assert np.array_equal(got_u16, expected_u16)

    # Signaling and quiet NaNs must stay NaN (including tiny payload in bit 0)
    nan_u32 = _f32_from_u32([0x7FC00000, 0x7F800001, 0x7FFFFFFF, 0xFFC00000, 0xFF800001])
    nan_u16 = _u16_of_f16_tensor(lucid.from_numpy(nan_u32).to(lucid.float16))
    assert np.all((nan_u16 & 0x7C00) == 0x7C00)
    assert np.all((nan_u16 & 0x03FF) != 0)


def test_f32_to_bf16_and_back_special_ties_and_nans() -> None:
    # Each pair is (input f32 u32 bits, expected widened f32 u32 bits after f32->bf16->f32)
    u32_cases = [
        # +0.0, -0.0
        (0x00000000, 0x00000000),
        (0x80000000, 0x80000000),
        # +inf, -inf
        (0x7F800000, 0x7F800000),
        (0xFF800000, 0xFF800000),
        # Exact midpoint ties on bit 16:
        # LSB (bit 16) = 0, tie (bit 15) = 1, sticky (bits 0..14) = 0 -> rounds to even (0x3f80)
        (0x3F808000, 0x3F800000),
        # LSB (bit 16) = 1, tie (bit 15) = 1, sticky = 0 -> rounds up to even (0x3f82)
        (0x3F818000, 0x3F820000),
        # LSB (bit 16) = 0, tie (bit 15) = 1, sticky = 1 -> rounds up (0x3f81)
        (0x3F808001, 0x3F810000),
        # Negative midpoint ties:
        (0xBF808000, 0xBF800000),
        (0xBF818000, 0xBF820000),
        # Carry from mantissa into exponent (0x3fffffff -> 0x40000000 = 2.0):
        (0x3FFFFFFF, 0x40000000),
        # Subnormal f32 values survive in bf16 (same 8-bit exponent):
        (0x00400000, 0x00400000),
        (0x00018000, 0x00020000),
    ]
    f32_arr = _f32_from_u32([u for u, _ in u32_cases])
    expected_u32 = np.array([e for _, e in u32_cases], dtype=np.uint32)

    bf16_t = lucid.from_numpy(f32_arr).to(lucid.bfloat16)
    got_u32 = _u32_of_f32_tensor(bf16_t.to(lucid.float32))
    assert np.array_equal(got_u32, expected_u32)

    # NaNs with high payload bits (0x7fffffff, 0xffffffff) or low payload bit (0x7f800001)
    # must stay NaN and never carry into +/-inf.
    nan_f32 = _f32_from_u32(
        [0x7FC00000, 0x7F800001, 0x7FFFFFFF, 0xFFC00000, 0xFF800001, 0xFFFFFFFF]
    )
    nan_rt = _u32_of_f32_tensor(lucid.from_numpy(nan_f32).to(lucid.bfloat16).to(lucid.float32))
    assert np.all((nan_rt & 0x7F800000) == 0x7F800000)
    assert np.all((nan_rt & 0x007FFFFF) != 0)


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16])
def test_empty_scalar_and_widen_narrow_ops(dtype: lucid.Dtype) -> None:
    empty = lucid.zeros((0, 4), dtype=lucid.float32, device="cpu").to(dtype)
    assert empty.shape == (0, 4)
    assert empty.to(lucid.float32).shape == (0, 4)

    scalar = lucid.tensor(-0.0, dtype=lucid.float32, device="cpu").to(dtype).to(lucid.float32)
    assert np.copysign(1.0, scalar.item()) == -1.0

    # Exercise unary_op, binary_op, and axis reduction widen/narrow paths on CPU.
    x = lucid.linspace(-2.0, 2.0, 257, dtype=lucid.float32, device="cpu").to(dtype)
    y = (lucid.sin(x) + x).sum(dim=0)
    assert y.dtype == dtype
    assert np.isfinite(y.to(lucid.float32).item())
