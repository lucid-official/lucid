"""``lucid.quantization`` Phase-4 — quantization-aware training."""

import numpy as np

import lucid
import lucid.nn as nn
import lucid.nn.intrinsic.qat as nniqat
import lucid.nn.qat as nnqat
import lucid.nn.quantized as nnq
import lucid.optim as optim
import lucid.quantization as Q


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quant = nnq.QuantStub()
        self.fc1 = nn.Linear(16, 32)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(32, 4)
        self.dequant = nnq.DeQuantStub()

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.dequant(self.fc2(self.act(self.fc1(self.quant(x)))))


class TestPrepareQAT:
    def test_swaps_to_qat_and_trains(self) -> None:
        lucid.manual_seed(0)
        qat = Q.prepare_qat(_Net())
        assert isinstance(qat.fc1, nnqat.Linear)
        assert qat.training  # QAT runs in train mode
        assert isinstance(qat.fc1.weight_fake_quant, Q.FakeQuantize)

    def test_ste_training_updates_weights(self) -> None:
        lucid.manual_seed(1)
        qat = Q.prepare_qat(_Net())
        w0 = qat.fc1.weight.numpy().copy()
        opt = optim.SGD(qat.parameters(), lr=0.1)
        for _ in range(5):
            x = lucid.randn(8, 16)
            loss = (qat(x) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        # STE let gradients reach the float weight → it moved.
        assert not np.allclose(w0, qat.fc1.weight.numpy())


class TestQATConvert:
    def test_convert_to_int8(self) -> None:
        lucid.manual_seed(2)
        qat = Q.prepare_qat(_Net())
        # The data has a stream of its own, so the initialisation's draws
        # ahead of it cannot change it (LCD-615).
        data = lucid.Generator(2)
        opt = optim.SGD(qat.parameters(), lr=0.05)
        for _ in range(5):
            loss = (qat(lucid.randn(16, 16, generator=data)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        qat.eval()
        qm = Q.convert(qat)
        assert isinstance(qm.fc1, nnq.Linear)
        assert qm.fc1.weight_int8.dtype is lucid.int8
        # QAT-then-convert output tracks the fake-quant training model closely.
        # Over 64 inputs, not 8: across 60 seeds the error on 8 reached 0.125,
        # past the bound, and on 64 at most 0.077.
        x = lucid.randn(64, 16, generator=data)
        yq_train = qat(x).numpy()
        yq_int8 = qm(x).numpy()
        assert np.abs(yq_train - yq_int8).mean() / (np.abs(yq_train).mean() + 1e-9) < 0.1


class TestConvBnReLUQAT:
    def _module(self) -> nniqat.ConvBnReLU2d:
        return nniqat.ConvBnReLU2d(
            nn.Conv2d(3, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            relu=True,
            qconfig=Q.get_default_qat_qconfig(),
        )

    def test_fold_shapes(self) -> None:
        lucid.manual_seed(3)
        cbr = self._module()
        w, b = cbr._fold()
        assert w.shape == (8, 3, 3, 3)
        assert b.shape == (8,)

    def test_train_and_convert(self) -> None:
        lucid.manual_seed(4)
        cbr = self._module()
        opt = optim.SGD(cbr.parameters(), lr=0.01)
        for _ in range(4):
            loss = (cbr(lucid.randn(2, 3, 8, 8)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        cbr.eval()
        from lucid.nn.intrinsic.qat.modules import convbnrelu2d_to_quantized

        qc = convbnrelu2d_to_quantized(cbr)
        assert isinstance(qc, nnq.ConvReLU2d)
        assert qc.weight_int8.dtype is lucid.int8
        assert qc(lucid.randn(2, 3, 8, 8)).shape == (2, 8, 8, 8)
