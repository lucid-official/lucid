"""Unit tests for fused_linear_relu/gelu + FusedLinear."""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F


class TestFusedLinearRelu:
    """F.fused_linear_relu parity and gradient tests."""

    def test_inference_matches_unfused(self) -> None:
        np.random.seed(0)
        x = lucid.randn(4, 8)
        w = lucid.randn(16, 8)
        b = lucid.randn(16)
        with lucid.no_grad():
            y_fused = F.fused_linear_relu(x, w, b)
        y_ref = lucid.relu(F.linear(x, w, b))
        np.testing.assert_allclose(y_fused.numpy(), y_ref.numpy(), atol=1e-5)

    def test_training_backward(self) -> None:
        x = lucid.randn(3, 4, requires_grad=True)
        w = lucid.randn(8, 4, requires_grad=True)
        b = lucid.randn(8, requires_grad=True)
        F.fused_linear_relu(x, w, b).sum().backward()
        assert x.grad is not None
        assert w.grad is not None
        assert b.grad is not None

    def test_training_gradients_correct(self) -> None:
        """Training grads must match the reference (unfused) backward."""
        np.random.seed(1)
        x_np = np.random.randn(2, 4).astype(np.float32)
        w_np = np.random.randn(6, 4).astype(np.float32)
        b_np = np.random.randn(6).astype(np.float32)

        x1 = lucid.tensor(x_np.copy(), requires_grad=True)
        w1 = lucid.tensor(w_np.copy(), requires_grad=True)
        b1 = lucid.tensor(b_np.copy(), requires_grad=True)
        F.fused_linear_relu(x1, w1, b1).sum().backward()

        x2 = lucid.tensor(x_np.copy(), requires_grad=True)
        w2 = lucid.tensor(w_np.copy(), requires_grad=True)
        b2 = lucid.tensor(b_np.copy(), requires_grad=True)
        lucid.relu(F.linear(x2, w2, b2)).sum().backward()

        np.testing.assert_allclose(x1.grad.numpy(), x2.grad.numpy(), atol=1e-5)
        np.testing.assert_allclose(w1.grad.numpy(), w2.grad.numpy(), atol=1e-5)
        np.testing.assert_allclose(b1.grad.numpy(), b2.grad.numpy(), atol=1e-5)

    def test_output_shape(self) -> None:
        x = lucid.randn(2, 3, 8)
        w = lucid.randn(16, 8)
        b = lucid.randn(16)
        with lucid.no_grad():
            y = F.fused_linear_relu(x, w, b)
        assert y.shape == (2, 3, 16)

    def test_nonnegative_output(self) -> None:
        """ReLU output must be ≥ 0."""
        x = lucid.randn(10, 5)
        w = lucid.randn(8, 5)
        b = lucid.randn(8)
        with lucid.no_grad():
            y = F.fused_linear_relu(x, w, b)
        assert float(y.min().item()) >= 0.0


class TestFusedLinearGelu:
    """F.fused_linear_gelu parity and gradient tests."""

    def test_inference_matches_unfused(self) -> None:
        np.random.seed(2)
        x = lucid.randn(4, 8)
        w = lucid.randn(16, 8)
        b = lucid.randn(16)
        with lucid.no_grad():
            y_fused = F.fused_linear_gelu(x, w, b)
        y_ref = F.gelu(F.linear(x, w, b), approximate="tanh")
        np.testing.assert_allclose(y_fused.numpy(), y_ref.numpy(), atol=1e-5)

    def test_training_backward(self) -> None:
        x = lucid.randn(3, 4, requires_grad=True)
        w = lucid.randn(8, 4, requires_grad=True)
        b = lucid.randn(8, requires_grad=True)
        F.fused_linear_gelu(x, w, b).sum().backward()
        assert x.grad is not None

    def test_exact_erf_fallback(self) -> None:
        """approximate='none' falls back to unfused erf path."""
        x = lucid.randn(3, 4)
        w = lucid.randn(8, 4)
        b = lucid.randn(8)
        with lucid.no_grad():
            y = F.fused_linear_gelu(x, w, b, approximate="none")
        y_ref = F.gelu(F.linear(x, w, b), approximate="none")
        np.testing.assert_allclose(y.numpy(), y_ref.numpy(), atol=1e-5)


class TestFusedLinearModule:
    """nn.FusedLinear module tests."""

    def test_relu_forward_shape(self) -> None:
        m = nn.FusedLinear(8, 16, activation="relu")
        x = lucid.randn(4, 8)
        with lucid.no_grad():
            assert m(x).shape == (4, 16)

    def test_gelu_forward_shape(self) -> None:
        m = nn.FusedLinear(8, 16, activation="gelu")
        x = lucid.randn(4, 8)
        with lucid.no_grad():
            assert m(x).shape == (4, 16)

    def test_no_bias(self) -> None:
        m = nn.FusedLinear(8, 16, activation="relu", bias=False)
        assert m.bias is None
        x = lucid.randn(2, 8)
        with lucid.no_grad():
            assert m(x).shape == (2, 16)

    def test_backward(self) -> None:
        m = nn.FusedLinear(8, 16, activation="relu")
        x = lucid.randn(3, 8, requires_grad=True)
        m(x).sum().backward()
        assert x.grad is not None
        assert m.weight.grad is not None
        assert m.bias.grad is not None

    def test_output_matches_linear_relu(self) -> None:
        """FusedLinear(relu) output == relu(Linear(x)) with same weights."""
        from lucid._C import engine as _C_engine

        m_lin = nn.Linear(8, 16)
        m_fus = nn.FusedLinear(8, 16, activation="relu")
        m_fus.weight._impl = _C_engine.TensorImpl(
            m_lin.weight.numpy().copy(), _C_engine.Device.CPU, True
        )
        m_fus.bias._impl = _C_engine.TensorImpl(
            m_lin.bias.numpy().copy(), _C_engine.Device.CPU, True
        )
        x = lucid.randn(4, 8)
        with lucid.no_grad():
            y_lin = lucid.relu(m_lin(x))
            y_fus = m_fus(x)
        np.testing.assert_allclose(y_fus.numpy(), y_lin.numpy(), atol=1e-5)

    def test_extra_repr(self) -> None:
        m = nn.FusedLinear(4, 8, activation="gelu")
        r = repr(m)
        assert "gelu" in r
        assert "in_features=4" in r

    def test_invalid_activation_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported activation"):
            nn.FusedLinear(4, 8, activation="sigmoid")

    def test_parameters_count(self) -> None:
        m = nn.FusedLinear(4, 8)
        params = list(m.parameters())
        assert len(params) == 2  # weight + bias


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to("metal")
        return True
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False


_DEVICES = [
    "cpu",
    pytest.param("metal", marks=pytest.mark.skipif(not _metal_ok(), reason="no Metal")),
]
_DTYPES = [lucid.float32, lucid.float16, lucid.bfloat16]
_FUSED = {
    "relu": (F.fused_linear_relu, F.relu),
    "gelu": (F.fused_linear_gelu, lambda p: F.gelu(p, approximate="tanh")),
}


class TestFusedKernelEveryDevice:
    """The fused kernels ran only on the CPU in float32: Metal and the half
    formats raised ``NotImplementedError``, so ``nn.FusedLinear().eval()``
    did not run on Metal at all (LCD-450). Elsewhere the binding composes
    the backend's linear and activation, the same function."""

    @pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f16", "bf16"])
    @pytest.mark.parametrize("device", _DEVICES)
    @pytest.mark.parametrize("act", sorted(_FUSED))
    def test_matches_the_unfused_composition(
        self, act: str, device: str, dtype: lucid.dtype
    ) -> None:
        fused, activation = _FUSED[act]
        lucid.manual_seed(0)
        x = lucid.randn(2, 5, 8).to(device).to(dtype)
        w = lucid.randn(3, 8).to(device).to(dtype)
        b = lucid.randn(3).to(device).to(dtype)
        with lucid.no_grad():
            got = fused(x, w, b)
            want = activation(F.linear(x, w, b))
        assert got.shape == want.shape == (2, 5, 3)
        assert got.dtype == dtype and got.device == want.device
        # CPU float32 runs the fused BLAS kernel; elsewhere it is the same ops.
        tol = 1e-6 if (device, dtype) == ("cpu", lucid.float32) else 0.0
        diff = float((got.float() - want.float()).abs().max().item())
        assert diff <= tol * max(1.0, float(want.float().abs().max().item()))

    @pytest.mark.parametrize("dtype", _DTYPES, ids=["f32", "f16", "bf16"])
    @pytest.mark.parametrize("device", _DEVICES)
    @pytest.mark.parametrize("act", sorted(_FUSED))
    def test_the_module_runs_in_eval(
        self, act: str, device: str, dtype: lucid.dtype
    ) -> None:
        m = nn.FusedLinear(8, 4, activation=act).to(device).to(dtype).eval()
        x = lucid.randn(3, 8).to(device).to(dtype)
        with lucid.no_grad():
            got = m(x)
            want = _FUSED[act][1](F.linear(x, m.weight, m.bias))
        diff = float((got.float() - want.float()).abs().max().item())
        assert diff <= 1e-6 * max(1.0, float(want.float().abs().max().item()))

    def test_an_empty_batch(self) -> None:
        x, w, b = lucid.randn(0, 8), lucid.randn(3, 8), lucid.randn(3)
        with lucid.no_grad():
            assert F.fused_linear_gelu(x, w, b).shape == (0, 3)


class TestFusedKernelChecksItsOperands:
    """The CPU kernel derived K from the input's byte size, so a weight with
    fewer input features was read past its end and the answer was garbage."""

    @pytest.mark.parametrize(
        "w_shape, b_shape", [((3, 5), (3,)), ((3, 8), (4,))], ids=["weight", "bias"]
    )
    def test_a_shape_mismatch_raises(
        self, w_shape: tuple[int, ...], b_shape: tuple[int, ...]
    ) -> None:
        x, w, b = lucid.randn(4, 8), lucid.randn(*w_shape), lucid.randn(*b_shape)
        with lucid.no_grad(), pytest.raises(ValueError, match="shape"):
            F.fused_linear_relu(x, w, b)

    def test_a_dtype_mismatch_raises(self) -> None:
        x, w, b = lucid.randn(4, 8), lucid.randn(3, 8).half(), lucid.randn(3)
        with lucid.no_grad(), pytest.raises(TypeError):
            F.fused_linear_relu(x, w, b)

    def test_an_integer_input_raises(self) -> None:
        x, w, b = lucid.randint(0, 3, (4, 8)), lucid.randn(3, 8), lucid.randn(3)
        with lucid.no_grad(), pytest.raises(TypeError):
            F.fused_linear_relu(x, w, b)
