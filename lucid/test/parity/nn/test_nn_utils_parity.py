"""Parity: ``lucid.nn.utils`` vs reference framework utilities."""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.utils as nn_utils
from lucid.test._helpers.compare import assert_close


@pytest.mark.parity
class TestFuseConvBnEvalParity:
    """fuse_conv_bn_eval must produce the same output as the fused reference pair."""

    def _make_pair(
        self, ref: Any, *, cin: int = 4, cout: int = 8, k: int = 3, bias: bool = True
    ) -> tuple[Any, Any, Any, Any]:
        np.random.seed(42)
        w = np.random.standard_normal((cout, cin, k, k)).astype(np.float32)
        b = np.random.standard_normal((cout,)).astype(np.float32) if bias else None

        gamma = np.random.standard_normal((cout,)).astype(np.float32)
        beta = np.random.standard_normal((cout,)).astype(np.float32)
        mean = np.random.standard_normal((cout,)).astype(np.float32)
        var = np.abs(np.random.standard_normal((cout,)).astype(np.float32)) + 0.1

        # Lucid side.
        l_conv = nn.Conv2d(cin, cout, k, bias=bias)
        l_conv.weight = nn.Parameter(lucid.tensor(w.copy()))
        if bias and b is not None:
            l_conv.bias = nn.Parameter(lucid.tensor(b.copy()))
        l_bn = nn.BatchNorm2d(cout)
        l_bn.weight = nn.Parameter(lucid.tensor(gamma.copy()))
        l_bn.bias = nn.Parameter(lucid.tensor(beta.copy()))
        l_bn.running_mean = lucid.tensor(mean.copy())
        l_bn.running_var = lucid.tensor(var.copy())
        l_conv.eval()
        l_bn.eval()

        # Reference side.
        r_conv = ref.nn.Conv2d(cin, cout, k, bias=bias)
        r_conv.weight = ref.nn.Parameter(ref.tensor(w.copy()))
        if bias and b is not None:
            r_conv.bias = ref.nn.Parameter(ref.tensor(b.copy()))
        r_bn = ref.nn.BatchNorm2d(cout)
        r_bn.weight = ref.nn.Parameter(ref.tensor(gamma.copy()))
        r_bn.bias = ref.nn.Parameter(ref.tensor(beta.copy()))
        r_bn.running_mean.copy_(ref.tensor(mean.copy()))
        r_bn.running_var.copy_(ref.tensor(var.copy()))
        r_conv.eval()
        r_bn.eval()

        return l_conv, l_bn, r_conv, r_bn

    def test_forward_matches(self, ref: Any) -> None:
        l_conv, l_bn, r_conv, r_bn = self._make_pair(ref)

        np.random.seed(7)
        x_np = np.random.standard_normal((2, 4, 10, 10)).astype(np.float32)
        x_l = lucid.tensor(x_np.copy())
        x_r = ref.tensor(x_np.copy())

        # Fuse on both sides.
        fused_l = nn_utils.fusion.fuse_conv_bn_eval(l_conv, l_bn)
        fused_r = ref.nn.utils.fusion.fuse_conv_bn_eval(r_conv, r_bn)

        out_l = fused_l(x_l)
        out_r = fused_r(x_r)
        assert_close(out_l, out_r, atol=1e-4)

    def test_no_bias_forward_matches(self, ref: Any) -> None:
        l_conv, l_bn, r_conv, r_bn = self._make_pair(ref, bias=False)

        np.random.seed(8)
        x_np = np.random.standard_normal((2, 4, 10, 10)).astype(np.float32)
        x_l = lucid.tensor(x_np.copy())
        x_r = ref.tensor(x_np.copy())

        fused_l = nn_utils.fusion.fuse_conv_bn_eval(l_conv, l_bn)
        fused_r = ref.nn.utils.fusion.fuse_conv_bn_eval(r_conv, r_bn)

        assert_close(fused_l(x_l), fused_r(x_r), atol=1e-4)


@pytest.mark.parity
class TestCopyParametersParity:
    """copy_parameters_and_buffers: dest weights must equal source after copy."""

    def test_linear_copy(self, ref: Any) -> None:  # noqa: ARG002
        np.random.seed(0)
        w = np.random.standard_normal((8, 4)).astype(np.float32)
        b = np.random.standard_normal((8,)).astype(np.float32)

        src = nn.Linear(4, 8)
        src.weight = nn.Parameter(lucid.tensor(w.copy()))
        src.bias = nn.Parameter(lucid.tensor(b.copy()))

        dst = nn.Linear(4, 8)
        nn_utils.copy_parameters_and_buffers(src, dst)

        assert_close(dst.weight, src.weight, atol=0.0)
        assert_close(dst.bias, src.bias, atol=0.0)
