"""Parity tests: normalization layers + transformer / attention modules.

Covers:
  lucid.nn.LayerNorm          — forward + backward
  lucid.nn.RMSNorm            — forward (when available in ref)
  lucid.nn.GroupNorm          — forward
  lucid.nn.BatchNorm1d/2d     — eval-mode forward
  lucid.nn.InstanceNorm2d     — no-affine forward
  lucid.nn.LocalResponseNorm  — forward
  lucid.nn.MultiheadAttention — eval-mode, weight-copied forward
  lucid.nn.TransformerEncoderLayer — eval-mode, weight-copied forward
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.test._helpers.compare import assert_close

# ── weight copying helpers ────────────────────────────────────────────────────


def _copy_weights_positional(src: Any, dst: nn.Module) -> None:
    from lucid._C import engine as _C_engine

    src_params = list(src.parameters())
    dst_params = list(dst.parameters())
    assert len(src_params) == len(
        dst_params
    ), f"param count mismatch: ref={len(src_params)} lucid={len(dst_params)}"
    for sp, dp in zip(src_params, dst_params):
        arr = sp.detach().cpu().numpy().copy()
        dp._impl = _C_engine.TensorImpl(arr, _C_engine.Device.CPU, True)


def _copy_named(src: Any, dst: nn.Module, mapping: list[tuple[str, str]]) -> None:
    from lucid._C import engine as _C_engine

    for src_name, dst_name in mapping:
        obj = src
        for attr in src_name.split("."):
            obj = getattr(obj, attr)
        arr = obj.detach().cpu().numpy().copy()
        dst_obj = dst
        attrs = dst_name.split(".")
        for attr in attrs[:-1]:
            dst_obj = getattr(dst_obj, attr)
        p = getattr(dst_obj, attrs[-1])
        p._impl = _C_engine.TensorImpl(arr, _C_engine.Device.CPU, True)


# ── LayerNorm ─────────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestLayerNormParity:
    def test_forward(self, ref: Any) -> None:
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((4, 8)).astype(np.float32)

        ref_ln = ref.nn.LayerNorm(8)
        lucid_ln = nn.LayerNorm(8)
        _copy_weights_positional(ref_ln, lucid_ln)

        ref_out = ref_ln(ref.tensor(x_np.copy()))
        lucid_out = lucid_ln(lucid.tensor(x_np.copy()))
        assert_close(lucid_out, ref_out, atol=1e-5)

    def test_backward(self, ref: Any) -> None:
        rng = np.random.default_rng(1)
        x_np = rng.standard_normal((3, 6)).astype(np.float32)

        ref_ln = ref.nn.LayerNorm(6)
        lucid_ln = nn.LayerNorm(6)
        _copy_weights_positional(ref_ln, lucid_ln)

        xr = ref.tensor(x_np.copy(), requires_grad=True)
        ref_ln(xr).sum().backward()

        xl = lucid.tensor(x_np.copy())
        xl.requires_grad_(True)
        lucid_ln(xl).sum().backward()

        assert xl.grad is not None and xr.grad is not None
        assert_close(xl.grad, xr.grad, atol=1e-5)

    def test_no_elementwise_affine(self, ref: Any) -> None:
        rng = np.random.default_rng(2)
        x_np = rng.standard_normal((2, 10)).astype(np.float32)

        ref_ln = ref.nn.LayerNorm(10, elementwise_affine=False)
        lucid_ln = nn.LayerNorm(10, elementwise_affine=False)

        assert_close(
            lucid_ln(lucid.tensor(x_np.copy())),
            ref_ln(ref.tensor(x_np.copy())),
            atol=1e-5,
        )

    def test_3d_input(self, ref: Any) -> None:
        rng = np.random.default_rng(3)
        x_np = rng.standard_normal((2, 4, 8)).astype(np.float32)

        ref_ln = ref.nn.LayerNorm([4, 8])
        lucid_ln = nn.LayerNorm([4, 8])
        _copy_weights_positional(ref_ln, lucid_ln)

        assert_close(
            lucid_ln(lucid.tensor(x_np.copy())),
            ref_ln(ref.tensor(x_np.copy())),
            atol=1e-5,
        )


# ── RMSNorm ───────────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestRMSNormParity:
    def test_forward(self, ref: Any) -> None:
        if not hasattr(ref.nn, "RMSNorm"):
            pytest.skip("RMSNorm not available in this version of the ref framework")
        rng = np.random.default_rng(4)
        x_np = rng.standard_normal((3, 8)).astype(np.float32)

        ref_rms = ref.nn.RMSNorm(8, eps=1e-8)
        lucid_rms = nn.RMSNorm(8, eps=1e-8)
        _copy_weights_positional(ref_rms, lucid_rms)

        assert_close(
            lucid_rms(lucid.tensor(x_np.copy())),
            ref_rms(ref.tensor(x_np.copy())),
            atol=1e-5,
        )


# ── GroupNorm ─────────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestGroupNormParity:
    def test_forward(self, ref: Any) -> None:
        rng = np.random.default_rng(5)
        x_np = rng.standard_normal((2, 8, 4, 4)).astype(np.float32)

        ref_gn = ref.nn.GroupNorm(4, 8)
        lucid_gn = nn.GroupNorm(4, 8)
        _copy_weights_positional(ref_gn, lucid_gn)

        assert_close(
            lucid_gn(lucid.tensor(x_np.copy())),
            ref_gn(ref.tensor(x_np.copy())),
            atol=1e-5,
        )

    def test_no_affine(self, ref: Any) -> None:
        rng = np.random.default_rng(6)
        x_np = rng.standard_normal((2, 6, 4)).astype(np.float32)

        ref_gn = ref.nn.GroupNorm(3, 6, affine=False)
        lucid_gn = nn.GroupNorm(3, 6, affine=False)

        assert_close(
            lucid_gn(lucid.tensor(x_np.copy())),
            ref_gn(ref.tensor(x_np.copy())),
            atol=1e-5,
        )


# ── BatchNorm ─────────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestBatchNormParity:
    def test_bn1d_eval(self, ref: Any) -> None:
        rng = np.random.default_rng(7)
        x_np = rng.standard_normal((4, 8)).astype(np.float32)

        ref_bn = ref.nn.BatchNorm1d(8)
        lucid_bn = nn.BatchNorm1d(8)
        ref_bn.eval()
        lucid_bn.eval()
        _copy_weights_positional(ref_bn, lucid_bn)

        assert_close(
            lucid_bn(lucid.tensor(x_np.copy())),
            ref_bn(ref.tensor(x_np.copy())),
            atol=1e-5,
        )

    def test_bn2d_eval(self, ref: Any) -> None:
        rng = np.random.default_rng(8)
        x_np = rng.standard_normal((2, 4, 6, 6)).astype(np.float32)

        ref_bn = ref.nn.BatchNorm2d(4)
        lucid_bn = nn.BatchNorm2d(4)
        ref_bn.eval()
        lucid_bn.eval()
        _copy_weights_positional(ref_bn, lucid_bn)

        assert_close(
            lucid_bn(lucid.tensor(x_np.copy())),
            ref_bn(ref.tensor(x_np.copy())),
            atol=1e-5,
        )

    def test_bn2d_train_running_stats(self, ref: Any) -> None:
        rng = np.random.default_rng(9)
        x_np = rng.standard_normal((4, 4, 6, 6)).astype(np.float32)

        ref_bn = ref.nn.BatchNorm2d(4)
        lucid_bn = nn.BatchNorm2d(4)
        _copy_weights_positional(ref_bn, lucid_bn)

        ref_out = ref_bn(ref.tensor(x_np.copy()))
        lucid_out = lucid_bn(lucid.tensor(x_np.copy()))
        assert_close(lucid_out, ref_out, atol=1e-4)


# ── InstanceNorm ──────────────────────────────────────────────────────────────


@pytest.mark.parity
class TestInstanceNormParity:
    def test_in2d_no_affine(self, ref: Any) -> None:
        rng = np.random.default_rng(10)
        x_np = rng.standard_normal((2, 4, 6, 6)).astype(np.float32)

        ref_in = ref.nn.InstanceNorm2d(4)
        lucid_in = nn.InstanceNorm2d(4)

        assert_close(
            lucid_in(lucid.tensor(x_np.copy())),
            ref_in(ref.tensor(x_np.copy())),
            atol=1e-5,
        )

    def test_in1d_with_affine(self, ref: Any) -> None:
        rng = np.random.default_rng(11)
        x_np = rng.standard_normal((3, 6, 10)).astype(np.float32)

        ref_in = ref.nn.InstanceNorm1d(6, affine=True)
        lucid_in = nn.InstanceNorm1d(6, affine=True)
        _copy_weights_positional(ref_in, lucid_in)

        assert_close(
            lucid_in(lucid.tensor(x_np.copy())),
            ref_in(ref.tensor(x_np.copy())),
            atol=1e-5,
        )


# ── LocalResponseNorm ─────────────────────────────────────────────────────────


@pytest.mark.parity
class TestLocalResponseNormParity:
    def test_default_params(self, ref: Any) -> None:
        rng = np.random.default_rng(12)
        x_np = rng.standard_normal((2, 8, 4, 4)).astype(np.float32)

        ref_lrn = ref.nn.LocalResponseNorm(5)
        lucid_lrn = nn.LocalResponseNorm(5)

        # A 2e-3 tolerance once hid a missing ``/ size`` (LCD-440); the
        # float32 rounding of a five-channel window is far below 1e-5.
        assert_close(
            lucid_lrn(lucid.tensor(x_np.copy())),
            ref_lrn(ref.tensor(x_np.copy())),
            atol=1e-5,
        )

    def test_custom_params(self, ref: Any) -> None:
        rng = np.random.default_rng(13)
        x_np = rng.standard_normal((1, 6, 8)).astype(np.float32)

        ref_lrn = ref.nn.LocalResponseNorm(3, alpha=1e-3, beta=0.75, k=2.0)
        lucid_lrn = nn.LocalResponseNorm(3, alpha=1e-3, beta=0.75, k=2.0)

        assert_close(
            lucid_lrn(lucid.tensor(x_np.copy())),
            ref_lrn(ref.tensor(x_np.copy())),
            atol=1e-5,
        )


# ── MultiheadAttention ────────────────────────────────────────────────────────


@pytest.mark.parity
class TestMultiheadAttentionParity:
    def _copy_mha_weights(self, ref_mha: Any, lucid_mha: nn.MultiheadAttention) -> None:
        _copy_named(
            ref_mha,
            lucid_mha,
            [
                ("in_proj_weight", "in_proj_weight"),
                ("in_proj_bias", "in_proj_bias"),
                ("out_proj.weight", "out_proj_weight"),
                ("out_proj.bias", "out_proj_bias"),
            ],
        )

    def test_self_attention_forward(self, ref: Any) -> None:
        E, H = 8, 2
        rng = np.random.default_rng(14)
        x_np = rng.standard_normal((5, 2, E)).astype(np.float32)  # (T, N, E)

        ref_mha = ref.nn.MultiheadAttention(E, H, dropout=0.0)
        lucid_mha = nn.MultiheadAttention(E, H, dropout=0.0)
        ref_mha.eval()
        lucid_mha.eval()
        self._copy_mha_weights(ref_mha, lucid_mha)

        xr = ref.tensor(x_np.copy())
        xl = lucid.tensor(x_np.copy())

        ref_out, _ = ref_mha(xr, xr, xr)
        lucid_out, _ = lucid_mha(xl, xl, xl)

        assert_close(lucid_out, ref_out, atol=1e-4)

    def test_cross_attention_forward(self, ref: Any) -> None:
        E, H = 8, 2
        rng = np.random.default_rng(15)
        q_np = rng.standard_normal((3, 1, E)).astype(np.float32)
        kv_np = rng.standard_normal((6, 1, E)).astype(np.float32)

        ref_mha = ref.nn.MultiheadAttention(E, H, dropout=0.0)
        lucid_mha = nn.MultiheadAttention(E, H, dropout=0.0)
        ref_mha.eval()
        lucid_mha.eval()
        self._copy_mha_weights(ref_mha, lucid_mha)

        ref_out, _ = ref_mha(
            ref.tensor(q_np.copy()), ref.tensor(kv_np.copy()), ref.tensor(kv_np.copy())
        )
        lucid_out, _ = lucid_mha(
            lucid.tensor(q_np.copy()),
            lucid.tensor(kv_np.copy()),
            lucid.tensor(kv_np.copy()),
        )

        assert_close(lucid_out, ref_out, atol=1e-4)

    def test_batch_first(self, ref: Any) -> None:
        E, H = 8, 2
        rng = np.random.default_rng(16)
        x_np = rng.standard_normal((2, 5, E)).astype(np.float32)  # (N, T, E)

        ref_mha = ref.nn.MultiheadAttention(E, H, dropout=0.0, batch_first=True)
        lucid_mha = nn.MultiheadAttention(E, H, dropout=0.0, batch_first=True)
        ref_mha.eval()
        lucid_mha.eval()
        self._copy_mha_weights(ref_mha, lucid_mha)

        xr = ref.tensor(x_np.copy())
        xl = lucid.tensor(x_np.copy())

        ref_out, _ = ref_mha(xr, xr, xr)
        lucid_out, _ = lucid_mha(xl, xl, xl)

        assert_close(lucid_out, ref_out, atol=1e-4)

    def test_gqa_repeat_matches_ref_enable_gqa(self, ref: Any) -> None:
        # lucid's repeat_kv + SDPA must match the reference framework's grouped-
        # query attention (enable_gqa repeats K/V internally), confirming the
        # head-grouping convention is identical.  Skips on older ref SDPA.
        ref_sdpa = ref.nn.functional.scaled_dot_product_attention
        B, H, KV, T, D = 2, 8, 2, 5, 4
        rng = np.random.default_rng(21)
        q = rng.standard_normal((B, H, T, D)).astype(np.float32)
        k = rng.standard_normal((B, KV, T, D)).astype(np.float32)
        v = rng.standard_normal((B, KV, T, D)).astype(np.float32)

        try:
            ref_out = ref_sdpa(
                ref.tensor(q.copy()),
                ref.tensor(k.copy()),
                ref.tensor(v.copy()),
                enable_gqa=True,
            )
        except (TypeError, RuntimeError):  # fmt: skip
            pytest.skip("reference SDPA has no enable_gqa")
        kl = nn.functional.repeat_kv(lucid.tensor(k.copy()), H // KV)
        vl = nn.functional.repeat_kv(lucid.tensor(v.copy()), H // KV)
        lucid_out = nn.functional.scaled_dot_product_attention(lucid.tensor(q.copy()), kl, vl)
        assert_close(lucid_out, ref_out, atol=1e-4)


# ── TransformerEncoderLayer ───────────────────────────────────────────────────


@pytest.mark.parity
class TestTransformerEncoderLayerParity:
    def test_forward_eval(self, ref: Any) -> None:
        D, H = 8, 2
        rng = np.random.default_rng(17)
        x_np = rng.standard_normal((6, 2, D)).astype(np.float32)  # (T, N, D)

        ref_layer = ref.nn.TransformerEncoderLayer(D, H, dim_feedforward=16, dropout=0.0)
        lucid_layer = nn.TransformerEncoderLayer(D, H, dim_feedforward=16, dropout=0.0)
        ref_layer.eval()
        lucid_layer.eval()
        _copy_weights_positional(ref_layer, lucid_layer)

        ref_out = ref_layer(ref.tensor(x_np.copy()))
        lucid_out = lucid_layer(lucid.tensor(x_np.copy()))
        assert_close(lucid_out, ref_out, atol=1e-4)

    def test_norm_first_eval(self, ref: Any) -> None:
        D, H = 8, 2
        rng = np.random.default_rng(18)
        x_np = rng.standard_normal((4, 2, D)).astype(np.float32)

        ref_layer = ref.nn.TransformerEncoderLayer(
            D, H, dim_feedforward=16, dropout=0.0, norm_first=True
        )
        lucid_layer = nn.TransformerEncoderLayer(
            D, H, dim_feedforward=16, dropout=0.0, norm_first=True
        )
        ref_layer.eval()
        lucid_layer.eval()
        _copy_weights_positional(ref_layer, lucid_layer)

        ref_out = ref_layer(ref.tensor(x_np.copy()))
        lucid_out = lucid_layer(lucid.tensor(x_np.copy()))
        assert_close(lucid_out, ref_out, atol=1e-4)


# ── TransformerDecoderLayer ───────────────────────────────────────────────────


@pytest.mark.parity
class TestTransformerDecoderLayerParity:
    def test_forward_eval(self, ref: Any) -> None:
        D, H = 8, 2
        rng = np.random.default_rng(19)
        tgt_np = rng.standard_normal((4, 2, D)).astype(np.float32)
        mem_np = rng.standard_normal((6, 2, D)).astype(np.float32)

        ref_layer = ref.nn.TransformerDecoderLayer(D, H, dim_feedforward=16, dropout=0.0)
        lucid_layer = nn.TransformerDecoderLayer(D, H, dim_feedforward=16, dropout=0.0)
        ref_layer.eval()
        lucid_layer.eval()
        _copy_weights_positional(ref_layer, lucid_layer)

        ref_out = ref_layer(ref.tensor(tgt_np.copy()), ref.tensor(mem_np.copy()))
        lucid_out = lucid_layer(lucid.tensor(tgt_np.copy()), lucid.tensor(mem_np.copy()))
        assert_close(lucid_out, ref_out, atol=1e-4)

    def test_norm_first_eval(self, ref: Any) -> None:
        D, H = 8, 2
        rng = np.random.default_rng(20)
        tgt_np = rng.standard_normal((3, 1, D)).astype(np.float32)
        mem_np = rng.standard_normal((5, 1, D)).astype(np.float32)

        ref_layer = ref.nn.TransformerDecoderLayer(
            D, H, dim_feedforward=16, dropout=0.0, norm_first=True
        )
        lucid_layer = nn.TransformerDecoderLayer(
            D, H, dim_feedforward=16, dropout=0.0, norm_first=True
        )
        ref_layer.eval()
        lucid_layer.eval()
        _copy_weights_positional(ref_layer, lucid_layer)

        ref_out = ref_layer(ref.tensor(tgt_np.copy()), ref.tensor(mem_np.copy()))
        lucid_out = lucid_layer(lucid.tensor(tgt_np.copy()), lucid.tensor(mem_np.copy()))
        assert_close(lucid_out, ref_out, atol=1e-4)


# ── TransformerDecoder ────────────────────────────────────────────────────────


@pytest.mark.parity
class TestTransformerDecoderParity:
    def test_two_layer_eval(self, ref: Any) -> None:
        D, H = 8, 2
        rng = np.random.default_rng(21)
        tgt_np = rng.standard_normal((4, 2, D)).astype(np.float32)
        mem_np = rng.standard_normal((6, 2, D)).astype(np.float32)

        ref_proto = ref.nn.TransformerDecoderLayer(D, H, dim_feedforward=16, dropout=0.0)
        lucid_proto = nn.TransformerDecoderLayer(D, H, dim_feedforward=16, dropout=0.0)

        ref_dec = ref.nn.TransformerDecoder(ref_proto, num_layers=2)
        lucid_dec = nn.TransformerDecoder(lucid_proto, num_layers=2)
        ref_dec.eval()
        lucid_dec.eval()
        _copy_weights_positional(ref_dec, lucid_dec)

        ref_out = ref_dec(ref.tensor(tgt_np.copy()), ref.tensor(mem_np.copy()))
        lucid_out = lucid_dec(lucid.tensor(tgt_np.copy()), lucid.tensor(mem_np.copy()))
        assert_close(lucid_out, ref_out, atol=1e-4)


# ── Transformer (full encoder-decoder) ───────────────────────────────────────


@pytest.mark.parity
class TestTransformerParity:
    def test_forward_eval(self, ref: Any) -> None:
        D, H = 8, 2
        rng = np.random.default_rng(22)
        src_np = rng.standard_normal((6, 2, D)).astype(np.float32)
        tgt_np = rng.standard_normal((4, 2, D)).astype(np.float32)

        ref_t = ref.nn.Transformer(
            d_model=D,
            nhead=H,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=16,
            dropout=0.0,
        )
        lucid_t = nn.Transformer(
            d_model=D,
            nhead=H,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=16,
            dropout=0.0,
        )
        ref_t.eval()
        lucid_t.eval()
        _copy_weights_positional(ref_t, lucid_t)

        ref_out = ref_t(ref.tensor(src_np.copy()), ref.tensor(tgt_np.copy()))
        lucid_out = lucid_t(lucid.tensor(src_np.copy()), lucid.tensor(tgt_np.copy()))
        assert_close(lucid_out, ref_out, atol=1e-4)
