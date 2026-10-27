"""Parity: Conv* layers (including bias=False) vs reference framework."""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.test._helpers.compare import assert_close


def _sync_conv2d(ref: Any, *, bias: bool = True) -> tuple[Any, Any, np.ndarray]:
    np.random.seed(0)
    w = np.random.standard_normal((8, 4, 3, 3)).astype(np.float32)
    b = np.random.standard_normal((8,)).astype(np.float32)
    x = np.random.standard_normal((2, 4, 12, 12)).astype(np.float32)

    l_conv = nn.Conv2d(4, 8, 3, bias=bias)
    r_conv = ref.nn.Conv2d(4, 8, 3, bias=bias)

    l_conv.weight = nn.Parameter(lucid.tensor(w.copy()))
    r_conv.weight = ref.nn.Parameter(ref.tensor(w.copy()))
    if bias:
        l_conv.bias = nn.Parameter(lucid.tensor(b.copy()))
        r_conv.bias = ref.nn.Parameter(ref.tensor(b.copy()))

    return l_conv, r_conv, x


@pytest.mark.parity
class TestConvBiasParity:
    def test_conv2d_with_bias(self, ref: Any) -> None:
        l_conv, r_conv, x_np = _sync_conv2d(ref, bias=True)
        assert_close(
            l_conv(lucid.tensor(x_np.copy())),
            r_conv(ref.tensor(x_np.copy())),
            atol=1e-4,
        )

    def test_conv2d_no_bias(self, ref: Any) -> None:
        l_conv, r_conv, x_np = _sync_conv2d(ref, bias=False)
        assert_close(
            l_conv(lucid.tensor(x_np.copy())),
            r_conv(ref.tensor(x_np.copy())),
            atol=1e-4,
        )

    def test_conv2d_no_bias_backward(self, ref: Any) -> None:
        l_conv, r_conv, x_np = _sync_conv2d(ref, bias=False)

        x_l = lucid.tensor(x_np.copy(), requires_grad=True)
        x_r = ref.tensor(x_np.copy(), requires_grad=True)

        l_conv(x_l).sum().backward()
        r_conv(x_r).sum().backward()

        assert_close(x_l.grad, x_r.grad, atol=1e-4)
        assert_close(l_conv.weight.grad, r_conv.weight.grad, atol=1e-4)

    def test_conv1d_no_bias(self, ref: Any) -> None:
        np.random.seed(1)
        w = np.random.standard_normal((8, 4, 3)).astype(np.float32)
        x = np.random.standard_normal((2, 4, 16)).astype(np.float32)

        l_conv = nn.Conv1d(4, 8, 3, bias=False)
        r_conv = ref.nn.Conv1d(4, 8, 3, bias=False)
        l_conv.weight = nn.Parameter(lucid.tensor(w.copy()))
        r_conv.weight = ref.nn.Parameter(ref.tensor(w.copy()))

        assert_close(
            l_conv(lucid.tensor(x.copy())),
            r_conv(ref.tensor(x.copy())),
            atol=1e-4,
        )

    def test_conv_transpose2d_no_bias(self, ref: Any) -> None:
        np.random.seed(2)
        w = np.random.standard_normal((4, 8, 3, 3)).astype(np.float32)
        x = np.random.standard_normal((2, 4, 6, 6)).astype(np.float32)

        l_ct = nn.ConvTranspose2d(4, 8, 3, bias=False)
        r_ct = ref.nn.ConvTranspose2d(4, 8, 3, bias=False)
        l_ct.weight = nn.Parameter(lucid.tensor(w.copy()))
        r_ct.weight = ref.nn.Parameter(ref.tensor(w.copy()))

        assert_close(
            l_ct(lucid.tensor(x.copy())),
            r_ct(ref.tensor(x.copy())),
            atol=1e-4,
        )


@pytest.mark.parity
class TestConvGroupedParity:
    @pytest.mark.parametrize(
        "in_ch, out_ch, groups",
        [
            (4, 8, 2),  # 2 groups, asymmetric channel counts per group
            (8, 8, 4),  # 4 groups, equal in/out per group
            (16, 16, 16),  # depthwise (groups == in_ch == out_ch)
        ],
        ids=["g2", "g4", "depthwise"],
    )
    def test_conv2d_grouped_forward(self, ref: Any, in_ch: int, out_ch: int, groups: int) -> None:
        np.random.seed(in_ch * 31 + groups)
        w = np.random.standard_normal((out_ch, in_ch // groups, 3, 3)).astype(np.float32)
        b = np.random.standard_normal((out_ch,)).astype(np.float32)
        x = np.random.standard_normal((2, in_ch, 8, 8)).astype(np.float32)

        l_conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=groups)
        r_conv = ref.nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=groups)
        l_conv.weight = nn.Parameter(lucid.tensor(w.copy()))
        r_conv.weight = ref.nn.Parameter(ref.tensor(w.copy()))
        l_conv.bias = nn.Parameter(lucid.tensor(b.copy()))
        r_conv.bias = ref.nn.Parameter(ref.tensor(b.copy()))

        assert_close(
            l_conv(lucid.tensor(x.copy())),
            r_conv(ref.tensor(x.copy())),
            atol=1e-4,
        )

    @pytest.mark.parametrize(
        "in_ch, out_ch, groups",
        [
            (4, 8, 2),
            (8, 8, 4),
            (
                16,
                16,
                16,
            ),  # the depthwise case that used to raise inside engine_backward
        ],
        ids=["g2", "g4", "depthwise"],
    )
    def test_conv2d_grouped_backward(self, ref: Any, in_ch: int, out_ch: int, groups: int) -> None:
        np.random.seed(in_ch * 17 + groups + 1)
        w = np.random.standard_normal((out_ch, in_ch // groups, 3, 3)).astype(np.float32)
        b = np.random.standard_normal((out_ch,)).astype(np.float32)
        x = np.random.standard_normal((2, in_ch, 8, 8)).astype(np.float32)

        l_conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=groups)
        r_conv = ref.nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=groups)
        l_conv.weight = nn.Parameter(lucid.tensor(w.copy()))
        r_conv.weight = ref.nn.Parameter(ref.tensor(w.copy()))
        l_conv.bias = nn.Parameter(lucid.tensor(b.copy()))
        r_conv.bias = ref.nn.Parameter(ref.tensor(b.copy()))

        x_l = lucid.tensor(x.copy(), requires_grad=True)
        x_r = ref.tensor(x.copy(), requires_grad=True)

        l_conv(x_l).sum().backward()
        r_conv(x_r).sum().backward()

        assert_close(x_l.grad, x_r.grad, atol=1e-4)
        assert_close(l_conv.weight.grad, r_conv.weight.grad, atol=1e-4)
        assert_close(l_conv.bias.grad, r_conv.bias.grad, atol=1e-4)
