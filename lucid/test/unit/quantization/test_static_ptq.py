"""``lucid.quantization`` Phase-2 — eager static PTQ end-to-end."""

import contextlib
from collections.abc import Iterator

import numpy as np

import lucid
import lucid.backends as backends
import lucid.nn as nn
import lucid.nn.quantized as nnq
import lucid.quantization as Q


@contextlib.contextmanager
def _reference_engine() -> Iterator[None]:
    """Pin the dequant (reference) path so int8-structure asserts skip MLX routing."""
    prev = backends.quantized.engine
    backends.quantized.engine = "reference"
    try:
        yield
    finally:
        backends.quantized.engine = prev


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.quant = nnq.QuantStub()
        self.fc1 = nn.Linear(16, 32)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(32, 8)
        self.dequant = nnq.DeQuantStub()

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.dequant(self.fc2(self.act(self.fc1(self.quant(x)))))


class _ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(8, 10)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc(self.flat(self.pool(self.relu(self.bn(self.conv(x))))))


def _calibrate_mlp() -> tuple[_MLP, nn.Module]:
    lucid.manual_seed(0)
    m = _MLP()
    m.eval()
    prepared = Q.prepare(m, Q.get_default_qconfig())
    for _ in range(20):
        prepared(lucid.randn(64, 16))
    return m, Q.convert(prepared)


class TestConvertStructure:
    def test_module_swaps(self) -> None:
        _, qm = _calibrate_mlp()
        assert isinstance(qm.fc1, nnq.Linear)
        assert isinstance(qm.quant, nnq.Quantize)
        assert isinstance(qm.dequant, nnq.DeQuantize)

    def test_weight_is_int8(self) -> None:
        with _reference_engine():
            _, qm = _calibrate_mlp()
        assert qm.fc1.weight_int8.dtype is lucid.int8
        assert qm.fc2.weight_int8.dtype is lucid.int8


class TestAccuracy:
    def test_mlp_error_bounded(self) -> None:
        m, qm = _calibrate_mlp()
        errs = []
        for _ in range(10):
            xe = lucid.randn(64, 16)
            yf, yq = m(xe).numpy(), qm(xe).numpy()
            errs.append(np.abs(yf - yq).mean() / (np.abs(yf).mean() + 1e-9))
        # int8 PTQ on an in-distribution set: a few percent at most.
        assert np.mean(errs) < 0.05


class TestFusion:
    def test_conv_bn_fold_matches(self) -> None:
        lucid.manual_seed(1)
        net = _ConvNet()
        net.eval()
        x = lucid.randn(2, 3, 8, 8)
        y_ref = net(x).numpy()
        fused = Q.fuse_modules(net, [["conv", "bn"]])
        # Folding BN into the conv is numerically exact at eval time.
        assert np.allclose(fused(x).numpy(), y_ref, atol=1e-4)
        assert isinstance(fused.bn, nn.Identity)

    def test_conv_bn_relu_becomes_intrinsic(self) -> None:
        import lucid.nn.intrinsic as nni

        lucid.manual_seed(2)
        net = _ConvNet()
        net.eval()
        fused = Q.fuse_modules(net, [["conv", "bn", "relu"]])
        assert isinstance(fused.conv, nni.ConvReLU2d)
        assert isinstance(fused.bn, nn.Identity)
        assert isinstance(fused.relu, nn.Identity)


class TestConvPTQ:
    def test_conv_net_quantizes_with_fusion(self) -> None:
        lucid.manual_seed(3)
        net = _ConvNet()
        net.eval()
        fused = Q.fuse_modules(net, [["conv", "bn"]])
        wrapped = nnq.QuantWrapper(fused)
        prepared = Q.prepare(wrapped)
        for _ in range(8):
            prepared(lucid.randn(4, 3, 8, 8))
        qm = Q.convert(prepared)
        assert isinstance(qm.module.conv, nnq.Conv2d)
        assert qm.module.conv.weight_int8.dtype is lucid.int8
        xe = lucid.randn(8, 3, 8, 8)
        yf, yq = net(xe).numpy(), qm(xe).numpy()
        assert np.abs(yf - yq).mean() / (np.abs(yf).mean() + 1e-9) < 0.05


class TestCheckpointRoundTrip:
    def test_state_dict_preserves_int8(self, tmp_path: object) -> None:
        _, qm = _calibrate_mlp()
        x = lucid.randn(4, 16)
        y_before = qm(x).numpy()

        path = str(tmp_path) + "/qmodel.lucid"  # type: ignore[operator]
        lucid.save(qm.state_dict(), path)
        loaded = lucid.load(path)
        assert loaded["fc1.weight_int8"].dtype is lucid.int8

        _, qm2 = _calibrate_mlp()  # same architecture + calibration → same structure
        qm2.load_state_dict(loaded)
        y_after = qm2(x).numpy()
        assert np.array_equal(y_before, y_after)


class TestQuantWrapper:
    def test_wraps_and_quantizes(self) -> None:
        lucid.manual_seed(4)
        inner = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
        wrapped = nnq.QuantWrapper(inner)
        prepared = Q.prepare(wrapped)
        for _ in range(8):
            prepared(lucid.randn(16, 8))
        qm = Q.convert(prepared)
        out = qm(lucid.randn(2, 8))
        assert out.shape == (2, 4)


class TestSequentialOrderPreserved:
    def test_convert_keeps_child_order(self) -> None:
        # Regression: replacing a Sequential child via setattr would move it to
        # the end (Module.__setattr__ delete-then-re-add), scrambling exec order.
        lucid.manual_seed(5)
        model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU())
        model.eval()
        prepared = Q.prepare(nnq.QuantWrapper(model))
        for _ in range(4):
            prepared(lucid.randn(2, 3, 8, 8))
        qm = Q.convert(prepared)
        keys = list(qm.module._modules.keys())
        assert keys == ["0", "1", "2"]  # conv, bn, relu — original order
        assert qm(lucid.randn(2, 3, 8, 8)).shape == (2, 4, 8, 8)


class TestZooIntegration:
    def test_resnet_18_static_ptq(self) -> None:
        from lucid.models import resnet_18

        lucid.manual_seed(6)
        m = resnet_18(num_classes=10)
        m.eval()
        prepared = Q.prepare(m)
        for _ in range(2):
            prepared(lucid.randn(2, 3, 32, 32))
        qm = Q.convert(prepared)

        qconvs = [mod for _, mod in qm.named_modules() if isinstance(mod, nnq.Conv2d)]
        assert len(qconvs) == 20  # every conv quantized
        assert all(c.weight_int8.dtype is lucid.int8 for c in qconvs)

        xe = lucid.randn(2, 3, 32, 32)
        lf = m(xe).last_hidden_state.numpy()
        lq = qm(xe).last_hidden_state.numpy()
        rel = np.abs(lf - lq).mean() / (np.abs(lf).mean() + 1e-9)
        assert rel < 0.25  # unfused int8 PTQ on a real backbone
