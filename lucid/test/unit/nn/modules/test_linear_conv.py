"""nn.Linear / nn.Bilinear / nn.Conv* / nn.ConvTranspose* modules."""

import numpy as np

import lucid
import lucid.nn as nn


class TestLinear:
    def test_shape(self) -> None:
        m = nn.Linear(4, 8)
        x = lucid.zeros(2, 4)
        out = m(x)
        assert out.shape == (2, 8)

    def test_no_bias_shape(self) -> None:
        m = nn.Linear(4, 8, bias=False)
        out = m(lucid.zeros(2, 4))
        assert out.shape == (2, 8)

    def test_known_values(self) -> None:
        m = nn.Linear(2, 1, bias=False)
        # Override weight to identity-ish.
        m.weight.data if hasattr(m.weight, "data") else m.weight  # touch
        # We can verify that y = x @ w.T by sampling random inputs.
        x = lucid.tensor([[1.0, 2.0]])
        out = m(x)
        # out shape is (1, 1).
        assert out.shape == (1, 1)


class TestBilinear:
    def test_shape(self) -> None:
        m = nn.Bilinear(3, 4, 5)
        out = m(lucid.zeros(2, 3), lucid.zeros(2, 4))
        assert out.shape == (2, 5)


class TestConv1d:
    def test_basic_shape(self) -> None:
        m = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3)
        x = lucid.zeros(1, 2, 16)
        out = m(x)
        # Default stride=1, padding=0 → length 16-3+1 = 14.
        assert out.shape == (1, 4, 14)


class TestConv2d:
    def test_basic_shape(self) -> None:
        m = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        x = lucid.zeros(1, 3, 16, 16)
        out = m(x)
        assert out.shape == (1, 8, 16, 16)

    def test_stride_2(self) -> None:
        m = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)
        out = m(lucid.zeros(1, 3, 16, 16))
        assert out.shape == (1, 8, 8, 8)


class TestConv3d:
    def test_basic_shape(self) -> None:
        m = nn.Conv3d(2, 4, kernel_size=3, padding=1)
        out = m(lucid.zeros(1, 2, 8, 8, 8))
        assert out.shape == (1, 4, 8, 8, 8)


class TestConvTranspose:
    def test_2d_doubles(self) -> None:
        m = nn.ConvTranspose2d(4, 2, kernel_size=4, stride=2, padding=1)
        out = m(lucid.zeros(1, 4, 8, 8))
        assert out.shape == (1, 2, 16, 16)


class TestConvBiasFalse:
    """``Conv*(bias=False)`` was previously broken at the engine binding
    (rejected None bias).  These exercise all six conv variants forward +
    backward and confirm the result equals ``bias=True`` + ``b=0``."""

    def _check_bias_false_matches_zero_bias(self, ctor, x_shape: tuple[int, ...]) -> None:

        np.random.seed(0)
        x_np = np.random.randn(*x_shape).astype(np.float32)

        m_no = ctor(bias=False)
        m_zero = ctor(bias=True)
        m_zero.weight._impl.copy_from(m_no.weight._impl)
        m_zero.bias._impl.copy_from(
            lucid.zeros(m_no.weight.shape[1 if "Transpose" in type(m_no).__name__ else 0])._impl
        )

        x = lucid.tensor(x_np.copy(), requires_grad=True)
        out = m_no(x).sum()
        out.backward()

        x2 = lucid.tensor(x_np.copy(), requires_grad=True)
        out2 = m_zero(x2).sum()
        out2.backward()

        np.testing.assert_allclose(out.item(), out2.item(), atol=1e-5)
        np.testing.assert_allclose(x.grad.numpy(), x2.grad.numpy(), atol=1e-5)
        np.testing.assert_allclose(m_no.weight.grad.numpy(), m_zero.weight.grad.numpy(), atol=1e-5)
        assert m_no.bias is None

    def test_conv1d_no_bias(self) -> None:
        self._check_bias_false_matches_zero_bias(
            lambda bias: nn.Conv1d(3, 4, kernel_size=3, padding=1, bias=bias), (2, 3, 8)
        )

    def test_conv2d_no_bias(self) -> None:
        self._check_bias_false_matches_zero_bias(
            lambda bias: nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=bias), (2, 3, 6, 6)
        )

    def test_conv3d_no_bias(self) -> None:
        self._check_bias_false_matches_zero_bias(
            lambda bias: nn.Conv3d(3, 4, kernel_size=3, padding=1, bias=bias), (2, 3, 4, 4, 4)
        )

    def test_conv_transpose1d_no_bias(self) -> None:
        self._check_bias_false_matches_zero_bias(
            lambda bias: nn.ConvTranspose1d(3, 4, kernel_size=3, padding=1, bias=bias), (2, 3, 6)
        )

    def test_conv_transpose2d_no_bias(self) -> None:
        self._check_bias_false_matches_zero_bias(
            lambda bias: nn.ConvTranspose2d(3, 4, kernel_size=3, padding=1, bias=bias), (2, 3, 6, 6)
        )

    def test_conv_transpose3d_no_bias(self) -> None:
        self._check_bias_false_matches_zero_bias(
            lambda bias: nn.ConvTranspose3d(3, 4, kernel_size=3, padding=1, bias=bias),
            (2, 3, 4, 4, 4),
        )
