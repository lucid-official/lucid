"""``lucid.quantization`` — QAT fused modules (LinearReLU/ConvReLU/Embedding) + fuse_modules_qat."""

import contextlib
from collections.abc import Iterator

import numpy as np
import pytest

import lucid
import lucid.backends as backends
import lucid.nn as nn
import lucid.nn.qat as nnqat
import lucid.nn.quantized as nnq
import lucid.optim as optim
import lucid.quantization as Q


@contextlib.contextmanager
def _reference_engine() -> Iterator[None]:
    prev = backends.quantized.engine
    backends.quantized.engine = "reference"
    try:
        yield
    finally:
        backends.quantized.engine = prev


class TestQATFusedConvReLU:
    @pytest.mark.parametrize("rank", [1, 2, 3])
    def test_conv_relu_fuse_prepare_convert(self, rank: int) -> None:
        with _reference_engine():
            conv = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[rank](3, 8, 3, padding=1)
            shape = {1: (2, 3, 8), 2: (2, 3, 8, 8), 3: (2, 3, 4, 4, 4)}[rank]
            q_relu = {1: nnq.ConvReLU1d, 2: nnq.ConvReLU2d, 3: nnq.ConvReLU3d}[rank]
            qat_relu = {1: nnqat.ConvReLU1d, 2: nnqat.ConvReLU2d, 3: nnqat.ConvReLU3d}[rank]
            lucid.manual_seed(rank)
            m = nn.Sequential(conv, nn.ReLU())
            fused = Q.fuse_modules(m, [["0", "1"]])
            qat = Q.prepare_qat(fused, Q.get_default_qat_qconfig_mapping())
            assert isinstance(qat[0], qat_relu)
            opt = optim.SGD(qat.parameters(), lr=0.01)
            for _ in range(3):
                loss = (qat(lucid.randn(*shape)) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            qat.eval()
            qc = Q.convert(qat)
            assert isinstance(qc[0], q_relu)
            assert qc[0].weight_int8.dtype is lucid.int8


class TestQATLinearReLU:
    def test_linear_relu(self) -> None:
        with _reference_engine():
            lucid.manual_seed(0)
            m = nn.Sequential(nn.Linear(32, 16), nn.ReLU())
            fused = Q.fuse_modules(m, [["0", "1"]])
            qat = Q.prepare_qat(fused, Q.get_default_qat_qconfig_mapping())
            assert isinstance(qat[0], nnqat.LinearReLU)
            w0 = qat[0].weight.data.numpy().copy()
            opt = optim.SGD(qat.parameters(), lr=0.05)
            for _ in range(4):
                loss = (qat(lucid.randn(8, 32)) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            assert not np.allclose(w0, qat[0].weight.data.numpy())
            qat.eval()
            qc = Q.convert(qat)
            assert isinstance(qc[0], nnq.LinearReLU)


class TestQATEmbedding:
    def test_embedding(self) -> None:
        with _reference_engine():
            lucid.manual_seed(0)
            m = nn.Sequential(nn.Embedding(20, 8))
            qat = Q.prepare_qat(m, Q.get_default_qat_qconfig_mapping())
            assert isinstance(qat[0], nnqat.Embedding)
            qat.eval()
            qc = Q.convert(qat)
            assert isinstance(qc[0], nnq.Embedding)
            assert qc[0].weight_int8.dtype is lucid.int8


class TestFuseModulesQAT:
    def test_conv_bn_relu_trainable_bn(self) -> None:
        with _reference_engine():
            lucid.manual_seed(0)

            class M(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
                    self.bn = nn.BatchNorm2d(8)
                    self.relu = nn.ReLU()

                def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                    return self.relu(self.bn(self.conv(x)))

            qc = Q.get_default_qat_qconfig()
            fused = Q.fuse_modules_qat(M(), [["conv", "bn", "relu"]], qconfig=qc)
            import lucid.nn.intrinsic.qat as nniqat

            assert isinstance(fused.conv, nniqat.ConvBnReLU2d)  # trainable, not folded
            opt = optim.SGD(fused.parameters(), lr=0.01)
            g0 = fused.conv.bn.weight.data.numpy().copy()
            for _ in range(3):
                loss = (fused(lucid.randn(2, 3, 8, 8)) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            assert not np.allclose(g0, fused.conv.bn.weight.data.numpy())  # BN trains
            fused.eval()
            conv_q = Q.convert(fused).conv
            assert isinstance(conv_q, nnq.ConvReLU2d)
