"""``lucid.quantization`` Phase-6 — rank-generic QAT Conv+BN fold (1d/2d/3d)."""

import contextlib
from collections.abc import Iterator

import pytest

import lucid
import lucid.backends as backends
import lucid.nn as nn
import lucid.nn.intrinsic.qat as nniqat
import lucid.nn.quantized as nnq
import lucid.optim as optim
import lucid.quantization as Q
from lucid.nn.intrinsic.qat.modules import _convbn_to_quantized


@contextlib.contextmanager
def _reference_engine() -> Iterator[None]:
    prev = backends.quantized.engine
    backends.quantized.engine = "reference"
    try:
        yield
    finally:
        backends.quantized.engine = prev


def _build(rank: int, relu: bool):
    qc = Q.get_default_qat_qconfig()
    conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[rank](3, 8, 3, padding=1, bias=False)
    bn = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}[rank](8)
    cls = {
        (1, False): nniqat.ConvBn1d,
        (2, False): nniqat.ConvBn2d,
        (3, False): nniqat.ConvBn3d,
        (1, True): nniqat.ConvBnReLU1d,
        (2, True): nniqat.ConvBnReLU2d,
        (3, True): nniqat.ConvBnReLU3d,
    }[(rank, relu)]
    return cls(conv, bn, qconfig=qc)


_SHAPES = {1: (2, 3, 16), 2: (2, 3, 8, 8), 3: (2, 3, 4, 4, 4)}
_QCONV = {
    (1, False): nnq.Conv1d,
    (2, False): nnq.Conv2d,
    (3, False): nnq.Conv3d,
    (1, True): nnq.ConvReLU1d,
    (2, True): nnq.ConvReLU2d,
    (3, True): nnq.ConvReLU3d,
}


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("relu", [False, True])
class TestConvBnRanks:
    def test_fold_and_forward(self, rank: int, relu: bool) -> None:
        lucid.manual_seed(rank)
        m = _build(rank, relu)
        w, b = m._fold()
        assert w.shape == (8, 3) + (3,) * rank  # (out, in, *kernel) for the rank
        assert b.shape == (8,)
        y = m(lucid.randn(*_SHAPES[rank]))  # QAT forward runs
        assert y.shape == (2, 8) + _SHAPES[rank][2:]

    def test_train_then_convert(self, rank: int, relu: bool) -> None:
        lucid.manual_seed(10 + rank)
        m = _build(rank, relu)
        opt = optim.SGD(m.parameters(), lr=0.01)
        x = lucid.randn(*_SHAPES[rank])
        w0 = m.conv.weight.data.numpy().copy()
        for _ in range(3):
            loss = (m(x) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        import numpy as np

        assert not np.allclose(w0, m.conv.weight.data.numpy())  # weights move (STE)
        m.eval()
        with _reference_engine():
            q = _convbn_to_quantized(m)
        assert isinstance(q, _QCONV[(rank, relu)])
        assert q.weight_int8.dtype is lucid.int8
        assert q(x).shape == (2, 8) + _SHAPES[rank][2:]
