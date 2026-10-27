"""Half op-math parity across ``lucid.compile`` and ``lucid.coreml`` (LCD-344).

Since ENG-21 and API-03, eager ``mul`` / ``div`` / ``floordiv`` on a half
tensor (``float16`` or ``bfloat16``) paired with a Python ``float`` or a 0-d
``float32`` tensor compute in ``float32`` and round once to the output half
dtype.  Previously, both ``lucid.compile`` (``Arith.mm``) and ``lucid.coreml``
(``_emit.py`` / ``_build.py``) narrowed every operand to the output half dtype
before applying the binary operation, rounding the scalar first and diverging
from eager (e.g. ``bf16(3.09375) * 0.1``).
"""

import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn
from lucid.compile._core import fallback
from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")

VALUES: list[float] = [3.09375, -3.09375, 1.5, -1.5, 2.0, -2.0, 0.3, -0.3, 1.7, -1.7]

SCALARS: list[float] = [0.1, 0.3, 1.7]

OPS: dict[str, Callable[[lucid.Tensor, float | lucid.Tensor], lucid.Tensor]] = {
    "mul": lambda x, s: x * s,
    "rmul": lambda x, s: s * x,
    "truediv": lambda x, s: x / s,
    "rtruediv": lambda x, s: s / x,
    "floordiv": lambda x, s: x // s,
    "rfloordiv": lambda x, s: s // x,
}


def _assert_bits_equal(got: lucid.Tensor, want: lucid.Tensor) -> None:
    """Assert exact bit-for-bit equality (with ``equal_nan=True``)."""
    assert got.dtype == want.dtype
    assert got.shape == want.shape
    got_f32 = got.to(lucid.float32).cpu().numpy()
    want_f32 = want.to(lucid.float32).cpu().numpy()
    assert np.array_equal(got_f32, want_f32, equal_nan=True), (
        f"dtype={got.dtype}: got {got_f32.tolist()} "
        f"(bits={got_f32.view(np.uint32).tolist()}), "
        f"want {want_f32.tolist()} "
        f"(bits={want_f32.view(np.uint32).tolist()})"
    )
    non_nan = ~np.isnan(want_f32)
    assert np.array_equal(got_f32.view(np.uint32)[non_nan], want_f32.view(np.uint32)[non_nan])


class _ScalarOpModule(nn.Module):
    def __init__(
        self,
        op_fn: Callable[[lucid.Tensor, float | lucid.Tensor], lucid.Tensor],
        scalar: float | lucid.Tensor,
    ) -> None:
        super().__init__()
        self._op_fn = op_fn
        self._scalar = scalar

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self._op_fn(x, self._scalar)


def test_bf16_3_09375_mul_0_1_compiles_without_pre_rounding() -> None:
    """``bf16(3.09375) * 0.1`` must round once from ``f32``, not pre-round ``0.1``."""
    x = lucid.tensor([3.09375], dtype=lucid.bfloat16, device=COMPILE_DEVICE)
    pre_rounded = x * lucid.tensor(0.1, dtype=lucid.bfloat16, device=COMPILE_DEVICE)
    eager = x * 0.1
    assert eager.to(lucid.float32).item() != pre_rounded.to(lucid.float32).item()

    model = _ScalarOpModule(OPS["mul"], 0.1).eval()
    compiled = lucid.compile(model)
    with warnings.catch_warnings():
        warnings.simplefilter("error", fallback.CompileFallbackWarning)
        got = compiled(x)
    assert not compiled.cache_info()["eager_only"]
    _assert_bits_equal(got, eager)


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=["f16", "bf16"])
@pytest.mark.parametrize("op_name", sorted(OPS))
@pytest.mark.parametrize("scalar_val", SCALARS)
@pytest.mark.parametrize("as_0d_tensor", [False, True], ids=["py_float", "0d_f32"])
def test_compile_half_op_math_matches_eager_bit_for_bit(
    dtype: object, op_name: str, scalar_val: float, as_0d_tensor: bool
) -> None:
    x = lucid.tensor(VALUES, dtype=dtype, device=COMPILE_DEVICE)
    scalar: float | lucid.Tensor = (
        lucid.tensor(scalar_val, dtype=lucid.float32, device=COMPILE_DEVICE)
        if as_0d_tensor
        else scalar_val
    )
    model = _ScalarOpModule(OPS[op_name], scalar).eval()
    want = model(x)

    compiled = lucid.compile(model)
    with warnings.catch_warnings():
        warnings.simplefilter("error", fallback.CompileFallbackWarning)
        got = compiled(x)
    assert not compiled.cache_info()["eager_only"]
    _assert_bits_equal(got, want)


@pytest.mark.parametrize("precision", [cml.Precision.FLOAT32, cml.Precision.FLOAT16])
@pytest.mark.parametrize("op_name", sorted(OPS))
@pytest.mark.parametrize("scalar_val", SCALARS)
@pytest.mark.parametrize("as_0d_tensor", [False, True], ids=["py_float", "0d_f32"])
def test_coreml_float16_op_math_matches_eager_bit_for_bit(
    tmp_path: Path, precision: cml.Precision, op_name: str, scalar_val: float, as_0d_tensor: bool
) -> None:
    x = lucid.tensor(VALUES, dtype=lucid.float16)
    scalar: float | lucid.Tensor = (
        lucid.tensor(scalar_val, dtype=lucid.float32) if as_0d_tensor else scalar_val
    )
    model = _ScalarOpModule(OPS[op_name], scalar).eval()
    want = model(x)

    pkg_path = tmp_path / f"{op_name}_{scalar_val}_{as_0d_tensor}_{precision.value}.mlpackage"
    exported = cml.export(
        model, x, str(pkg_path), precision=precision, compute_units=cml.ComputeUnits.CPU_ONLY
    )
    try:
        got = exported.predict(x)
    finally:
        exported.close()
    assert isinstance(got, lucid.Tensor)
    _assert_bits_equal(got, want)


class _LearnableScalarOpModule(nn.Module):
    def __init__(self, op_name: str, scalar_val: float) -> None:
        super().__init__()
        self._op_fn = OPS[op_name]
        self.scale = nn.Parameter(
            lucid.tensor(scalar_val, dtype=lucid.float32, device=COMPILE_DEVICE)
        )

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self._op_fn(x, self.scale).sum()


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=["f16", "bf16"])
@pytest.mark.parametrize("op_name", ["mul", "rmul", "truediv", "rtruediv"])
def test_compiled_step_with_learnable_0d_f32_parameter_matches_eager(
    dtype: object, op_name: str
) -> None:
    x = lucid.tensor(VALUES, dtype=dtype, device=COMPILE_DEVICE)
    eager_model = _LearnableScalarOpModule(op_name, 0.1)
    eager_loss = eager_model(x)
    eager_loss.backward()
    assert eager_model.scale.grad is not None

    comp_model = _LearnableScalarOpModule(op_name, 0.1)
    step = lucid.compile.make_step(comp_model, lambda pred, _: pred)
    with warnings.catch_warnings():
        warnings.simplefilter("error", fallback.CompileFallbackWarning)
        comp_loss = step(x, x)
        comp_loss.backward()
    assert comp_model.scale.grad is not None
    assert comp_model.scale.grad.dtype == lucid.float32
    _assert_bits_equal(comp_loss.detach(), eager_loss.detach())
    assert comp_model.scale.grad.item() == pytest.approx(
        eager_model.scale.grad.item(), rel=1e-5, abs=1e-5
    )
