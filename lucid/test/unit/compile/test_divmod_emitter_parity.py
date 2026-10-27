"""Compiled (Metal) and Core ML emitter parity for ``floordiv``, ``remainder``, and ``fmod`` (LCD-444).

Verifies that ``lucid.compile`` (MPSGraph) and ``lucid.coreml`` (MIL) lower
``//`` (``floordiv`` / ``rfloordiv``), ``%`` (``remainder`` / ``rmod``), and
``fmod`` without eager fallback and match eager bit-for-bit across the #232
exact value matrix (integers including ``int64`` / zero divisors / ``MIN // -1``,
floats including ``±0.0`` / ``±inf`` / ``NaN`` / large quotients, and half
op-math).
"""

from collections.abc import Callable
import math
import warnings

import numpy as np
import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.compile._core.fallback import CompileFallbackWarning
from lucid.test._fixtures.devices import metal_available
from lucid.test.unit.ops.binary.test_divmod_exact import (
    _BITS,
    _FLOAT_DENS,
    _FLOAT_NUMS,
    _INT_DTYPES,
    _bits,
    _int_operands,
    _int_oracle,
)

OpFn = Callable[[lucid.Tensor, lucid.Tensor], lucid.Tensor]

_OPS: dict[str, OpFn] = {
    "floordiv": lambda a, b: a // b,
    "remainder": lucid.remainder,
    "fmod": lucid.fmod,
}
_KIND = {"floordiv": "floor", "remainder": "remainder", "fmod": "fmod"}
_HAS_COREML = hasattr(_C_engine, "coreml")


class _BinaryMod(nn.Module):
    def __init__(self, fn: OpFn) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fn(x[0], x[1])


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
@pytest.mark.parametrize("op_name", sorted(_OPS))
@pytest.mark.parametrize("dtype", _INT_DTYPES)
def test_compiled_integer_divmod_matches_eager_and_oracle(dtype: lucid.dtype, op_name: str) -> None:
    """Compiled integer ``floordiv``, ``remainder``, and ``fmod`` match eager exactly."""
    _C_engine.compile.session_cache_clear()
    bits = _BITS[dtype]
    nums, dens = _int_operands(bits)
    a = lucid.tensor(nums, dtype=dtype, device="metal")
    b = lucid.tensor(dens, dtype=dtype, device="metal")
    fn = _OPS[op_name]
    compiled = lucid.compile(fn)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        got = compiled(a, b)
    eager = fn(a, b)
    info = compiled.cache_info()
    assert info["entries"] == 1
    assert info["eager_only"] == ()
    assert got.dtype == dtype
    want = [_int_oracle(_KIND[op_name], x, y, bits) for x, y in zip(nums, dens, strict=True)]
    assert got.tolist() == want
    assert got.tolist() == eager.tolist()


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
@pytest.mark.parametrize("op_name", sorted(_OPS))
@pytest.mark.parametrize("dtype", _INT_DTYPES)
def test_compiled_integer_zero_divisor_matches_metal_contract(
    dtype: lucid.dtype, op_name: str
) -> None:
    """Compiled Metal integer zero divisor returns ``-1`` (``//``) or ``a`` (``%`` / ``fmod``)."""
    _C_engine.compile.session_cache_clear()
    nums = [5, -5, 6, 0, -(2**7)]
    a = lucid.tensor([nums], dtype=dtype, device="metal")
    b = lucid.tensor([[3], [0]], dtype=dtype, device="metal")
    fn = _OPS[op_name]
    compiled = lucid.compile(fn)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        got = compiled(a, b)
    eager = fn(a, b)
    assert compiled.cache_info()["eager_only"] == ()
    assert got.tolist() == eager.tolist()


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
@pytest.mark.parametrize("op_name", sorted(_OPS))
@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16, lucid.float32])
def test_compiled_float_divmod_matches_eager_bit_for_bit(dtype: lucid.dtype, op_name: str) -> None:
    """Compiled floating-point ``floordiv``, ``remainder``, and ``fmod`` match eager bit-for-bit."""
    _C_engine.compile.session_cache_clear()
    pairs = [(x, y) for x in _FLOAT_NUMS for y in _FLOAT_DENS]
    nums = [x for x, _ in pairs]
    dens = [y for _, y in pairs]
    a = lucid.tensor(nums, dtype=dtype, device="metal")
    b = lucid.tensor(dens, dtype=dtype, device="metal")
    fn = _OPS[op_name]
    compiled = lucid.compile(fn)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        got = compiled(a, b)
    eager = fn(a, b)
    assert compiled.cache_info()["eager_only"] == ()
    assert got.dtype == dtype
    g_arr = got.to(lucid.float32).numpy()
    e_arr = eager.to(lucid.float32).numpy()
    np.testing.assert_array_equal(_bits(g_arr), _bits(e_arr))


@pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
@pytest.mark.parametrize("divisor", [2.0, -0.3, 1e-3, math.inf, -math.inf, 0.0])
@pytest.mark.parametrize(
    "op_fn",
    [
        lambda x, s: x // s,
        lambda x, s: s // x,
        lambda x, s: x % s,
        lambda x, s: s % x,
        lambda x, s: lucid.fmod(x, s),
    ],
    ids=["floordiv", "rfloordiv", "mod", "rmod", "fmod"],
)
@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16])
def test_compiled_half_with_python_scalar_matches_eager(
    dtype: lucid.dtype, op_fn: Callable[[lucid.Tensor, float], lucid.Tensor], divisor: float
) -> None:
    """Compiled half tensor with Python float scalar matches eager bit-for-bit."""
    _C_engine.compile.session_cache_clear()
    nums = [0.0, -0.0, 1.0, -1.0, 7.5, -7.5, 1000.0, -3.0, 0.1]
    a = lucid.tensor(nums, dtype=dtype, device="metal")
    fn = lambda t: op_fn(t, divisor)
    compiled = lucid.compile(fn)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        got = compiled(a)
    eager = fn(a)
    assert compiled.cache_info()["eager_only"] == ()
    assert got.dtype == dtype
    np.testing.assert_array_equal(
        _bits(got.to(lucid.float32).numpy()), _bits(eager.to(lucid.float32).numpy())
    )


@pytest.mark.skipif(not _HAS_COREML, reason="built without Core ML writer")
@pytest.mark.parametrize("op_name", sorted(_OPS))
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_coreml_float_divmod_matches_eager_bit_for_bit(
    op_name: str, precision: str, tmp_path: object
) -> None:
    """Core ML ``floordiv``, ``remainder``, and ``fmod`` match eager bit-for-bit."""
    dtype = lucid.float16 if precision == "fp16" else lucid.float32
    pairs = [(x, y) for x in _FLOAT_NUMS for y in _FLOAT_DENS]
    inp = lucid.tensor([[x for x, _ in pairs], [y for _, y in pairs]], dtype=dtype, device="cpu")
    model = _BinaryMod(_OPS[op_name]).eval()
    want = model(inp)
    cml_prec = cml.Precision.FLOAT16 if precision == "fp16" else cml.Precision.FLOAT32
    pkg = cml.export(
        model,
        inp,
        f"{tmp_path}/{op_name}_{precision}.mlpackage",
        precision=cml_prec,
        io_precision=cml_prec,
    )
    try:
        got = pkg.predict(inp)
    finally:
        pkg.close()
    assert isinstance(got, lucid.Tensor)
    assert got.dtype == dtype
    np.testing.assert_array_equal(
        _bits(got.to(lucid.float32).numpy()), _bits(want.to(lucid.float32).numpy())
    )


@pytest.mark.skipif(not _HAS_COREML, reason="built without Core ML writer")
@pytest.mark.parametrize("op_name", sorted(_OPS))
def test_coreml_int32_divmod_matches_eager_exactly(op_name: str, tmp_path: object) -> None:
    """Core ML ``int32`` ``floordiv``, ``remainder``, and ``fmod`` match eager across _int_operands(32) and 2^24..2^31."""
    nums, dens = _int_operands(32)
    extra_nums = [-100_000_001, 100_000_003, -100_000_003] + [
        sign * ((1 << k) + delta)
        for k in range(24, 31)
        for delta in (-3, -1, 1, 3)
        for sign in (1, -1)
    ]
    extra_dens = [1, -1, 2, -2, 3, -3, 7, -7, 10, -10]
    for n in extra_nums:
        for d in extra_dens:
            nums.append(n)
            dens.append(d)
    inp = lucid.tensor([nums, dens], dtype=lucid.int32, device="cpu")
    model = _BinaryMod(_OPS[op_name]).eval()
    want = model(inp)
    pkg = cml.export(model, inp, f"{tmp_path}/{op_name}_i32.mlpackage")
    try:
        got = pkg.predict(inp)
    finally:
        pkg.close()
    assert isinstance(got, lucid.Tensor)
    assert got.tolist() == want.tolist()


@pytest.mark.skipif(not _HAS_COREML, reason="built without Core ML writer")
@pytest.mark.parametrize("op_name", sorted(_OPS))
def test_coreml_constant_folded_divmod_and_wide_int64_refusal(
    op_name: str, tmp_path: object
) -> None:
    """Core ML folds constant divmod buffers and refuses wide ``int64`` constants outside ``int32``."""
    fn = _OPS[op_name]

    class _FoldedMod(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("num", lucid.tensor([17, -17, 25, -25], dtype=lucid.int32))
            self.register_buffer("den", lucid.tensor([5, 5, -7, -7], dtype=lucid.int32))

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            c = fn(self.get_buffer("num"), self.get_buffer("den")).to(lucid.float32)
            return x + c

    x = lucid.tensor([1.0, 2.0, 3.0, 4.0], dtype=lucid.float32)
    m_fold = _FoldedMod().eval()
    pkg = cml.export(m_fold, x, f"{tmp_path}/{op_name}_fold.mlpackage")
    try:
        got = pkg.predict(x)
    finally:
        pkg.close()
    assert isinstance(got, lucid.Tensor)
    np.testing.assert_array_equal(got.numpy(), m_fold(x).numpy())

    class _WideInt64Mod(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("wide", lucid.tensor([2**40 + 3], dtype=lucid.int64))

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            return fn(x.to(lucid.int64), self.get_buffer("wide")).to(lucid.float32)

    with pytest.raises(ValueError, match="outside int32"):
        cml.export(_WideInt64Mod().eval(), x, f"{tmp_path}/{op_name}_wide.mlpackage")
