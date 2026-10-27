"""LSTM compile coverage.

The MPSGraph LSTM emitter handles the F32 / no-projection envelope.
Multi-layer and bidirectional stacks are orchestrated in Python as a
sequence of single-layer engine calls plus reverse / concat glue, so
they compile as soon as the single-layer ``lstm`` op and that glue do.

Three bugs used to make every LSTM fall back to eager (the emitter was
effectively dead): (1) the tracer wiped a multi-output op's inputs on the
follow-up ``on_op_io({}, h_n/c_n)`` calls; (2) the emitter expected three
MPSGraph LSTM outputs but inference mode returns two (``[hidden, cell]``);
(3) a *returned* ``split_at`` piece (the per-layer state slice / ``y[-1]``)
was never marked consumed, so it went unbound.  With those fixed, the
whole family compiles.

The MPSGraph LSTM is its own Metal kernel (not the eager MLX path), so the
results agree to fp32 tolerance, not bit-for-bit.  ``proj_size > 0`` is
still outside the envelope and must fall back to eager with a correct
result.
"""

import pytest

import lucid
import lucid.nn as nn

from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor

# Compiled forwards under grad mode run eagerly (LCD-582); these test them.
pytestmark = pytest.mark.usefixtures("inference")


def _compile_and_check(model: nn.Module, x: lucid.Tensor, *, should_compile: bool) -> None:
    """Compile ``model``, assert compile-vs-eager parity + the compile state."""
    model.to(COMPILE_DEVICE).eval()
    x = x.to(COMPILE_DEVICE)
    eager = model(x)
    cm = lucid.compile(model)
    out = cm(x)
    out.eval()
    diff = float((eager - out).abs().max().item())
    assert diff < 1e-4, f"compile-vs-eager parity broken: {diff:.3e}"
    info = cm.cache_info()
    if should_compile:
        assert info["entries"] >= 1 and not info["eager_only"], info
    else:
        assert info["entries"] == 0 or info["eager_only"], info


class _LSTMHead(nn.Module):
    def __init__(self, num_layers: int = 1, bidirectional: bool = False) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=16, hidden_size=32, num_layers=num_layers, bidirectional=bidirectional
        )
        self.fc = nn.Linear(32 * (2 if bidirectional else 1), 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        y, _ = self.lstm(x)
        return self.fc(y[-1])


def test_lstm_single_layer_compiles() -> None:
    _compile_and_check(_LSTMHead(), metal_tensor(8, 4, 16), should_compile=True)


def test_lstm_multi_layer_compiles() -> None:
    _compile_and_check(_LSTMHead(num_layers=2), metal_tensor(8, 4, 16), should_compile=True)
    _compile_and_check(_LSTMHead(num_layers=3), metal_tensor(8, 4, 16), should_compile=True)


def test_lstm_bidirectional_compiles() -> None:
    _compile_and_check(_LSTMHead(bidirectional=True), metal_tensor(8, 4, 16), should_compile=True)


def test_lstm_multi_layer_bidirectional_compiles() -> None:
    _compile_and_check(
        _LSTMHead(num_layers=2, bidirectional=True), metal_tensor(8, 4, 16), should_compile=True
    )


def test_lstm_consumes_hidden_state_compiles() -> None:
    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=8, hidden_size=16)
            self.fc = nn.Linear(16, 4)

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            _, (hn, _cn) = self.lstm(x)
            return self.fc(hn.squeeze(0))

    _compile_and_check(M(), metal_tensor(6, 3, 8), should_compile=True)


def test_lstm_consumes_cell_state_falls_back() -> None:
    # Consuming c_n needs MPSGraph's produceCell trajectory, which crashes the
    # LSTM kernel on some Metal drivers — the emitter bails to eager (correct
    # result) when the cell state is read.
    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=8, hidden_size=16)

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            _, (_hn, cn) = self.lstm(x)
            return cn.sum() + cn.flatten().mean()

    _compile_and_check(M(), metal_tensor(6, 3, 8), should_compile=False)


def test_lstm_proj_size_falls_back_cleanly() -> None:
    """Projected LSTM (``proj_size > 0``) is outside the envelope → eager."""

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=8, hidden_size=16, proj_size=4)
            self.fc = nn.Linear(4, 4)

        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            y, _ = self.lstm(x)
            return self.fc(y[-1])

    _compile_and_check(M(), metal_tensor(6, 3, 8), should_compile=False)
