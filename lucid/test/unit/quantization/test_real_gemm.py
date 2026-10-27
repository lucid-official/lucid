"""``lucid.quantization`` Phase-6 — real int4/int8 GEMM via MLX (Metal only)."""

import numpy as np
import pytest

import lucid
import lucid.backends as backends
import lucid.nn as nn
import lucid.nn.quantized as nnq
import lucid.quantization as Q


def _metal_ok() -> bool:
    try:
        lucid.zeros((1,)).to("metal")
        return True
    except Exception:
        return False


_ENABLED = backends.quantized.available and _metal_ok()

pytestmark = pytest.mark.skipif(
    not _ENABLED, reason="engine lacks quantized ops or Metal unavailable"
)


class TestRealGEMM:
    def test_metal_forward_has_no_python_host_observation(self, monkeypatch) -> None:
        layer = nnq.QuantizedLinearMLX.from_float(nn.Linear(128, 64), bits=4)
        x = lucid.randn(1, 128, device="metal")
        expected = layer(x).numpy().copy()

        def reject(*args, **kwargs):
            raise AssertionError("Metal quantized forward must not observe a host value")

        with monkeypatch.context() as guard:
            for name in ("numpy", "item", "cpu", "tolist"):
                guard.setattr(lucid.Tensor, name, reject)
            output = layer(x)
        assert output.is_metal
        np.testing.assert_allclose(output.numpy(), expected, atol=1e-5)

    def test_int8_accuracy(self) -> None:
        lucid.manual_seed(0)
        lin = nn.Linear(256, 256)
        lin.eval()
        x = lucid.randn(16, 256)
        yf = lin(x).numpy()
        q = nnq.QuantizedLinearMLX.from_float(lin, bits=8)
        yq = q(x.to("metal")).numpy()
        assert np.abs(yf - yq).max() / (np.abs(yf).max() + 1e-9) < 0.02

    def test_int4_accuracy(self) -> None:
        lucid.manual_seed(1)
        lin = nn.Linear(256, 256)
        lin.eval()
        x = lucid.randn(16, 256)
        yf = lin(x).numpy()
        q = nnq.QuantizedLinearMLX.from_float(lin, bits=4)
        yq = q(x.to("metal")).numpy()
        # 4-bit is coarser but still tracks the float output.
        assert np.abs(yf - yq).max() / (np.abs(yf).max() + 1e-9) < 0.15

    def test_weight_memory_smaller(self) -> None:
        lucid.manual_seed(2)
        lin = nn.Linear(512, 512, bias=False)
        q = nnq.QuantizedLinearMLX.from_float(lin, bits=8)
        float_bytes = 512 * 512 * 4
        packed_bytes = (
            q.packed_weight.numpy().nbytes + q.scales.numpy().nbytes + q.biases.numpy().nbytes
        )
        assert packed_bytes < float_bytes / 2  # meaningfully smaller

    def test_state_dict_round_trip(self, tmp_path: object) -> None:
        # Packed weight is uint32 tagged I32; must survive save→reload→to(metal)
        # (the engine bitcasts int32→uint32 for the kernel).
        for bits in (8, 4):
            lucid.manual_seed(10 + bits)
            lin = nn.Linear(128, 64)
            lin.eval()
            x = lucid.randn(8, 128).to("metal")
            q = nnq.QuantizedLinearMLX.from_float(lin, bits=bits)
            y_before = q(x).numpy()

            path = str(tmp_path) + f"/mlxq{bits}.lucid"  # type: ignore[operator]
            lucid.save(q.state_dict(), path)
            q2 = nnq.QuantizedLinearMLX(128, 64, bits=bits).to("metal")
            q2.load_state_dict(lucid.load(path))
            assert np.allclose(y_before, q2(x).numpy(), atol=1e-4)

    @pytest.mark.parametrize("bits", [4, 8])
    @pytest.mark.parametrize("rows", [1, 2, 8, 32])
    def test_engine_op_matches_dequant_path(self, bits: int, rows: int) -> None:
        # The real GEMM should agree with the reference dequant→float path.
        from lucid.quantization import _qgemm

        lucid.manual_seed(3)
        w = lucid.randn(64, 128).to("metal")
        x = lucid.randn(rows, 128).to("metal")
        packed, scales, biases = _qgemm.quantize(w, group_size=64, bits=bits)
        y_kernel = _qgemm.quantized_matmul(
            x, packed, scales, biases, transpose=True, group_size=64, bits=bits
        ).numpy()
        w_deq = _qgemm.dequantize(packed, scales, biases, group_size=64, bits=bits)
        y_deq = lucid.matmul(x, w_deq.mT).numpy()
        assert np.allclose(y_kernel, y_deq, atol=1e-3)


class TestConvertRoutesToMLX:
    """`convert` / `quantize_dynamic` route Linear to the real MLX GEMM (auto engine)."""

    def test_static_convert_routes_linear(self) -> None:
        lucid.manual_seed(0)
        m = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 64))
        m.eval()
        x = lucid.randn(16, 128)
        yf = m(x).numpy()
        prepared = Q.prepare(m, Q.get_default_qconfig_mapping())
        for _ in range(5):
            prepared(lucid.randn(16, 128))
        qm = Q.convert(prepared)  # auto engine + Metal → weight-only MLX GEMM
        assert isinstance(qm[0], nnq.QuantizedLinearMLX)
        assert isinstance(qm[2], nnq.QuantizedLinearMLX)
        yq = qm(x).numpy()
        assert np.abs(yf - yq).max() / (np.abs(yf).max() + 1e-9) < 0.05

    def test_dynamic_convert_routes_linear(self) -> None:
        lucid.manual_seed(1)
        m = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 64))
        m.eval()
        qm = Q.quantize_dynamic(m)
        assert isinstance(qm[0], nnq.QuantizedLinearMLX)

    def test_indivisible_dim_falls_back_to_dequant(self) -> None:
        # in_features=8 divides no MLX group size (32/64/128) → dequant, not a crash.
        lucid.manual_seed(2)
        m = nn.Sequential(nn.Linear(8, 16))
        m.eval()
        prepared = Q.prepare(m, Q.get_default_qconfig_mapping())
        for _ in range(3):
            prepared(lucid.randn(4, 8))
        qm = Q.convert(prepared)
        assert isinstance(qm[0], nnq.Linear)
        assert not isinstance(qm[0], nnq.QuantizedLinearMLX)

    def test_relu_fused_variant_routes(self) -> None:
        lucid.manual_seed(3)
        import lucid.nn.intrinsic as nni

        seq = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        fused = nni.LinearReLU(seq[0], seq[1])
        m = nn.Sequential(fused)
        m.eval()
        prepared = Q.prepare(m, Q.get_default_qconfig_mapping())
        for _ in range(3):
            prepared(lucid.randn(8, 64))
        qm = Q.convert(prepared)
        assert isinstance(qm[0], nnq.QuantizedLinearMLX)
        assert qm[0].relu is True  # ReLU folded into the fast layer


class TestBackendsToggle:
    def test_toggle_values(self) -> None:
        assert backends.quantized.engine == "auto"
        backends.quantized.engine = "mlx_group"
        assert backends.quantized.use_mlx()
        backends.quantized.engine = "reference"
        assert not backends.quantized.use_mlx()
        backends.quantized.engine = "auto"  # restore
        with pytest.raises(ValueError):
            backends.quantized.engine = "nonsense"
