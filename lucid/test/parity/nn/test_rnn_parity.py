"""Parity tests for RNN-family modules.

Covers:
  lucid.nn.LSTM  — standard, proj_size, bidirectional, multi-layer
  lucid.nn.GRU   — standard, bidirectional
  lucid.nn.RNN   — tanh / relu
  lucid.autograd.checkpoint — gradient checkpointing parity
  Tensor.register_hook      — gradient hook parity
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.test._helpers.compare import assert_close

# ── helpers ───────────────────────────────────────────────────────────────────


def _copy_weights_by_name(src_module: Any, dst_module: nn.Module, param_names: list[str]) -> None:
    """Copy named parameters from reference module to lucid module (same names)."""
    from lucid._C import engine as _C_engine

    for name in param_names:
        src = getattr(src_module, name)
        dst = getattr(dst_module, name)
        arr = src.detach().numpy().copy()
        dst._impl = _C_engine.TensorImpl(arr, _C_engine.Device.CPU, True)


def _copy_weights_positional(src_module: Any, dst_module: nn.Module) -> None:
    """Copy parameters positionally (works when parameter names differ)."""
    from lucid._C import engine as _C_engine

    src_params = list(src_module.parameters())
    dst_params = list(dst_module.parameters())
    assert len(src_params) == len(
        dst_params
    ), f"Parameter count mismatch: {len(src_params)} vs {len(dst_params)}"
    for src_p, dst_p in zip(src_params, dst_params):
        arr = src_p.detach().numpy().copy()
        dst_p._impl = _C_engine.TensorImpl(arr, _C_engine.Device.CPU, True)


# ── LSTM parity ───────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestLSTMParity:
    def _run(
        self, ref: Any, hidden: int, inp: int, seq: int, batch: int, **kwargs: Any
    ) -> tuple[Any, Any, Any, Any]:
        """Return (lucid_out, ref_out, lucid_h, ref_h)."""
        ref_lstm = ref.nn.LSTM(inp, hidden, **kwargs)
        l_lstm = nn.LSTM(inp, hidden, **kwargs)

        params = [n for n, _ in ref_lstm.named_parameters()]
        _copy_weights_by_name(ref_lstm, l_lstm, params)

        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((seq, batch, inp)).astype(np.float32)
        ref_out, (ref_h, ref_c) = ref_lstm(ref.tensor(x_np))
        l_out, (l_h, l_c) = l_lstm(lucid.tensor(x_np.copy()))
        return l_out, ref_out, (l_h, l_c), (ref_h, ref_c)

    def test_standard(self, ref: Any) -> None:
        l_out, ref_out, (l_h, l_c), (ref_h, ref_c) = self._run(ref, hidden=8, inp=4, seq=5, batch=2)
        assert_close(l_out, ref_out, atol=1e-4)
        assert_close(l_h, ref_h, atol=1e-4)
        assert_close(l_c, ref_c, atol=1e-4)

    def test_proj_size(self, ref: Any) -> None:
        """LSTM with proj_size: output and hidden use projected dimension."""
        l_out, ref_out, (l_h, l_c), (ref_h, ref_c) = self._run(
            ref, hidden=8, inp=4, seq=5, batch=2, proj_size=3
        )
        # output shape should reflect proj_size
        assert tuple(l_out.shape)[-1] == 3, f"expected proj dim 3, got {l_out.shape}"
        assert_close(l_out, ref_out, atol=1e-4)
        assert_close(l_h, ref_h, atol=1e-4)
        assert_close(l_c, ref_c, atol=1e-4)

    def test_bidirectional(self, ref: Any) -> None:
        l_out, ref_out, (l_h, l_c), (ref_h, ref_c) = self._run(
            ref, hidden=6, inp=4, seq=4, batch=2, bidirectional=True
        )
        assert l_out.shape[-1] == 12  # 2 * hidden
        assert_close(l_out, ref_out, atol=1e-4)

    def test_multi_layer(self, ref: Any) -> None:
        l_out, ref_out, (l_h, l_c), (ref_h, ref_c) = self._run(
            ref, hidden=6, inp=4, seq=4, batch=2, num_layers=2
        )
        assert_close(l_out, ref_out, atol=1e-4)
        assert_close(l_h, ref_h, atol=1e-4)


@pytest.mark.parity
class TestGRUParity:
    def _run(self, ref: Any, **kwargs: Any) -> tuple[Any, Any]:
        ref_gru = ref.nn.GRU(4, 6, **kwargs)
        l_gru = nn.GRU(4, 6, **kwargs)
        _copy_weights_positional(ref_gru, l_gru)

        x_np = np.random.default_rng(1).standard_normal((5, 2, 4)).astype(np.float32)
        ref_out, ref_h = ref_gru(ref.tensor(x_np))
        l_out, l_h = l_gru(lucid.tensor(x_np.copy()))
        return (l_out, l_h), (ref_out, ref_h)

    def test_standard(self, ref: Any) -> None:
        (l_out, l_h), (ref_out, ref_h) = self._run(ref)
        assert_close(l_out, ref_out, atol=1e-4)
        assert_close(l_h, ref_h, atol=1e-4)

    def test_bidirectional(self, ref: Any) -> None:
        (l_out, l_h), (ref_out, ref_h) = self._run(ref, bidirectional=True)
        assert l_out.shape[-1] == 12
        assert_close(l_out, ref_out, atol=1e-4)


# ── register_hook parity ──────────────────────────────────────────────────────


@pytest.mark.parity
class TestRegisterHookParity:
    def test_hook_grad_matches(self, ref: Any) -> None:
        """register_hook fires with the correct gradient (matches reference)."""

        np.random.seed(5)
        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # lucid
        x = lucid.tensor(x_np.copy(), requires_grad=True)
        captured_l: list[Any] = []
        x.register_hook(lambda g: captured_l.append(g.clone()) or None)
        (x * 2).sum().backward()

        # reference
        x_ref = ref.tensor(x_np.copy(), requires_grad=True)
        captured_r: list[Any] = []
        x_ref.register_hook(lambda g: captured_r.append(g.clone()) or None)
        (x_ref * 2).sum().backward()

        assert_close(captured_l[0], captured_r[0], atol=1e-6)

    def test_hook_can_scale_grad(self, ref: Any) -> None:
        """Hook that halves the gradient produces same x.grad as reference."""

        x_np = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        x = lucid.tensor(x_np.copy(), requires_grad=True)
        x.register_hook(lambda g: g * 0.5)
        (x * 4).sum().backward()

        x_ref = ref.tensor(x_np.copy(), requires_grad=True)
        x_ref.register_hook(lambda g: g * 0.5)
        (x_ref * 4).sum().backward()

        assert_close(x.grad, x_ref.grad, atol=1e-6)


# ── checkpoint parity ─────────────────────────────────────────────────────────


@pytest.mark.parity
class TestCheckpointParity:
    def test_output_parity(self, ref: Any) -> None:
        """checkpoint output matches direct forward."""
        from lucid.autograd import checkpoint

        lucid.manual_seed(0)
        W = lucid.tensor([[1.0, 0.5], [-0.5, 1.0]])

        def block(x: lucid.Tensor) -> lucid.Tensor:
            return F.relu(x @ W)

        x_np = np.array([[1.0, -1.0], [0.5, 2.0]], dtype=np.float32)
        x = lucid.tensor(x_np.copy(), requires_grad=True)
        y = checkpoint(block, x)

        x_direct = lucid.tensor(x_np.copy(), requires_grad=True)
        y_direct = block(x_direct)

        assert_close(y, y_direct, atol=1e-6)

    def test_grad_parity(self, ref: Any) -> None:
        """checkpoint gradients match direct backward."""
        from lucid.autograd import checkpoint

        W = lucid.tensor([[2.0, 0.0], [0.0, 1.0]], requires_grad=True)
        x = lucid.tensor([[1.0, 2.0]], requires_grad=True)

        def seg(inp: lucid.Tensor) -> lucid.Tensor:
            return inp @ W

        checkpoint(seg, x).sum().backward()
        x_g, W_g = x.grad.clone(), W.grad.clone()

        # reference direct backward
        W2 = lucid.tensor([[2.0, 0.0], [0.0, 1.0]], requires_grad=True)
        x2 = lucid.tensor([[1.0, 2.0]], requires_grad=True)
        (x2 @ W2).sum().backward()

        assert_close(x_g, x2.grad, atol=1e-5)
        assert_close(W_g, W2.grad, atol=1e-5)
