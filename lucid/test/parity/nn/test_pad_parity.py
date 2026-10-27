"""Parity tests for F.pad modes and Conv padding_mode.

Verifies reflect / replicate / circular padding against the reference
framework for 1-D, 2-D, and 3-D inputs, and that Conv1d/2d/3d forward
passes with all padding_mode values produce matching outputs.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.test._helpers.compare import assert_close


@pytest.mark.parity
class TestFPadParity:
    """F.pad mode parity: constant / reflect / replicate / circular."""

    @pytest.mark.parametrize("mode", ["constant", "reflect", "replicate", "circular"])
    def test_pad_1d(self, ref: Any, mode: str) -> None:
        np.random.seed(0)
        x = np.random.randn(2, 3, 8).astype(np.float32)
        pad = (2, 2)
        kw = {"value": 0.0} if mode == "constant" else {}
        l_out = F.pad(lucid.tensor(x.copy()), pad, mode=mode, **kw)
        r_out = ref.nn.functional.pad(ref.tensor(x.copy()), pad, mode=mode, **kw)
        assert_close(l_out, r_out, atol=1e-6)

    @pytest.mark.parametrize("mode", ["constant", "reflect", "replicate", "circular"])
    def test_pad_2d(self, ref: Any, mode: str) -> None:
        np.random.seed(1)
        x = np.random.randn(2, 3, 5, 5).astype(np.float32)
        pad = (1, 1, 2, 2)
        kw = {"value": 0.0} if mode == "constant" else {}
        l_out = F.pad(lucid.tensor(x.copy()), pad, mode=mode, **kw)
        r_out = ref.nn.functional.pad(ref.tensor(x.copy()), pad, mode=mode, **kw)
        assert_close(l_out, r_out, atol=1e-6)

    @pytest.mark.parametrize("mode", ["constant", "reflect", "replicate", "circular"])
    def test_pad_3d(self, ref: Any, mode: str) -> None:
        np.random.seed(2)
        x = np.random.randn(1, 2, 4, 4, 4).astype(np.float32)
        pad = (1, 1, 1, 1, 1, 1)
        kw = {"value": 0.0} if mode == "constant" else {}
        l_out = F.pad(lucid.tensor(x.copy()), pad, mode=mode, **kw)
        r_out = ref.nn.functional.pad(ref.tensor(x.copy()), pad, mode=mode, **kw)
        assert_close(l_out, r_out, atol=1e-6)

    def test_constant_nonzero_value(self, ref: Any) -> None:
        x = np.ones((1, 1, 3), dtype=np.float32)
        l_out = F.pad(lucid.tensor(x.copy()), (2, 2), mode="constant", value=-1.0)
        r_out = ref.nn.functional.pad(ref.tensor(x.copy()), (2, 2), mode="constant", value=-1.0)
        assert_close(l_out, r_out, atol=1e-6)


@pytest.mark.parity
class TestConvPaddingModeParity:
    """Conv2d padding_mode forward parity."""

    def _run(self, ref: Any, padding_mode: str) -> tuple[Any, Any]:
        from lucid._C import engine as _C_engine

        np.random.seed(42)
        w_np = np.random.randn(4, 2, 3, 3).astype(np.float32)
        b_np = np.random.randn(4).astype(np.float32)
        x_np = np.random.randn(1, 2, 7, 7).astype(np.float32)

        l_conv = nn.Conv2d(2, 4, 3, padding=1, padding_mode=padding_mode)
        r_conv = ref.nn.Conv2d(2, 4, 3, padding=1, padding_mode=padding_mode)

        # Copy weights
        l_conv.weight._impl = _C_engine.TensorImpl(w_np.copy(), _C_engine.Device.CPU, True)
        l_conv.bias._impl = _C_engine.TensorImpl(b_np.copy(), _C_engine.Device.CPU, True)
        r_conv.weight.data = ref.tensor(w_np.copy())
        r_conv.bias.data = ref.tensor(b_np.copy())

        l_y = l_conv(lucid.tensor(x_np.copy()))
        r_y = r_conv(ref.tensor(x_np.copy()))
        return l_y, r_y

    @pytest.mark.parametrize("padding_mode", ["zeros", "reflect", "replicate", "circular"])
    def test_conv2d_padding_mode(self, ref: Any, padding_mode: str) -> None:
        l_y, r_y = self._run(ref, padding_mode)
        assert_close(l_y, r_y, atol=1e-4)
