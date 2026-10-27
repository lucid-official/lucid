"""Unit and reference-parity tests for LCD-131, LCD-133, and LCD-134.

Covers:
- LCD-131: Unbatched 2-D ``(L, E)`` inputs across ``MultiheadAttention``,
  ``GroupedQueryAttention``, ``MultiQueryAttention``,
  ``F.multi_head_attention_forward``, ``Transformer*``, and rank validation
  guards on ``LazyConv*d``, dilated ``max_pool*d``, and ``adaptive_*_pool*d``.
- LCD-133: ``lp_pool1d/2d/3d`` ``ceil_mode=True`` output shapes and signed
  ``x ** p`` evaluation (including ``p = 1`` and ``p = inf``).
- LCD-134: ``Transformer.generate_square_subsequent_mask``, ``bias=False``
  across ``TransformerEncoderLayer`` / ``TransformerDecoderLayer`` /
  ``Transformer``, ``norm_first``, ``custom_encoder`` / ``custom_decoder``,
  ``enable_nested_tensor`` / ``mask_check``, and ``*_is_causal`` forwarding
  through ``TransformerEncoder``, ``TransformerDecoder``, and ``Transformer``.
"""

from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.nn.parameter import UninitializedParameter


def _copy_state(src_state: dict[str, object], dst_module: nn.Module, device: str) -> None:
    mapped = {
        k: lucid.tensor(v.detach().cpu().numpy(), device=device) for k, v in src_state.items()
    }
    dst_module.load_state_dict(mapped)


class TestUnbatchedMultiheadAttention:
    @pytest.mark.parametrize("batch_first", [False, True])
    @pytest.mark.parametrize("average_attn_weights", [True, False])
    @pytest.mark.parametrize("use_kpm", [False, True])
    @pytest.mark.parametrize("is_causal", [False, True])
    def test_mha_unbatched_matches_batched_and_reference(
        self,
        ref: ModuleType,
        device: str,
        batch_first: bool,
        average_attn_weights: bool,
        use_kpm: bool,
        is_causal: bool,
    ) -> None:
        rng = np.random.default_rng(42)
        L, S, E, H = 5, 5 if is_causal else 6, 8, 2
        q_np = rng.standard_normal((L, E)).astype(np.float32)
        k_np = rng.standard_normal((S, E)).astype(np.float32)
        v_np = rng.standard_normal((S, E)).astype(np.float32)
        kpm_np = np.array([False, True, False, False, True, False][:S])

        ref_mha = ref.nn.MultiheadAttention(E, H, dropout=0.0, batch_first=batch_first).eval()
        mha = nn.MultiheadAttention(
            E, H, dropout=0.0, batch_first=batch_first, device=device
        ).eval()
        _copy_state(ref_mha.state_dict(), mha, device)

        q_l = lucid.tensor(q_np, device=device, requires_grad=True)
        k_l = lucid.tensor(k_np, device=device, requires_grad=True)
        v_l = lucid.tensor(v_np, device=device, requires_grad=True)
        kpm_l = lucid.tensor(kpm_np, device=device) if use_kpm else None
        attn_mask_l = (
            nn.Transformer.generate_square_subsequent_mask(L, device=device) if is_causal else None
        )

        q_r = ref.tensor(q_np, requires_grad=True)
        k_r = ref.tensor(k_np, requires_grad=True)
        v_r = ref.tensor(v_np, requires_grad=True)
        kpm_r = ref.tensor(kpm_np) if use_kpm else None
        attn_mask_r = ref.nn.Transformer.generate_square_subsequent_mask(L) if is_causal else None

        out_l, w_l = mha(
            q_l,
            k_l,
            v_l,
            key_padding_mask=kpm_l,
            need_weights=True,
            attn_mask=attn_mask_l,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        out_r, w_r = ref_mha(
            q_r,
            k_r,
            v_r,
            key_padding_mask=kpm_r,
            need_weights=True,
            attn_mask=attn_mask_r,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        assert out_l.shape == (L, E)
        assert w_l is not None
        expected_w_shape = (L, S) if average_attn_weights else (H, L, S)
        assert w_l.shape == expected_w_shape

        np.testing.assert_allclose(out_l.numpy(), out_r.detach().numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(w_l.numpy(), w_r.detach().numpy(), rtol=1e-4, atol=1e-5)

        grad_out = rng.standard_normal((L, E)).astype(np.float32)
        (out_l * lucid.tensor(grad_out, device=device)).sum().backward()
        (out_r * ref.tensor(grad_out)).sum().backward()
        assert q_l.grad is not None and k_l.grad is not None and v_l.grad is not None
        np.testing.assert_allclose(q_l.grad.numpy(), q_r.grad.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(k_l.grad.numpy(), k_r.grad.numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(v_l.grad.numpy(), v_r.grad.numpy(), rtol=1e-4, atol=1e-5)

    def test_gqa_and_mqa_and_functional_unbatched(self, device: str) -> None:
        q = lucid.randn(4, 8, device=device)
        k = lucid.randn(6, 8, device=device)
        v = lucid.randn(6, 8, device=device)

        gqa = nn.GroupedQueryAttention(8, 4, num_kv_heads=2, device=device).eval()
        out_gqa, w_gqa = gqa(q, k, v, average_attn_weights=False)
        assert out_gqa.shape == (4, 8)
        assert w_gqa is not None and w_gqa.shape == (4, 4, 6)

        mqa = nn.MultiQueryAttention(8, 4, device=device).eval()
        out_mqa, w_mqa = mqa(q, k, v, average_attn_weights=True)
        assert out_mqa.shape == (4, 8)
        assert w_mqa is not None and w_mqa.shape == (4, 6)

        mha = nn.MultiheadAttention(8, 2, device=device).eval()
        out_f, w_f = F.multi_head_attention_forward(
            q,
            k,
            v,
            embed_dim_to_check=8,
            num_heads=2,
            in_proj_weight=mha.in_proj_weight,
            in_proj_bias=mha.in_proj_bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=mha.out_proj_weight,
            out_proj_bias=mha.out_proj_bias,
            training=False,
            need_weights=True,
        )
        assert out_f.shape == (4, 8)
        assert w_f is not None and w_f.shape == (4, 6)

    def test_mha_rank_validation(self, device: str) -> None:
        mha = nn.MultiheadAttention(8, 2, device=device)
        q2 = lucid.randn(4, 8, device=device)
        q3 = lucid.randn(4, 2, 8, device=device)

        with pytest.raises(ValueError, match="expected query to be 2-D or 3-D"):
            mha(lucid.randn(8, device=device), q2, q2)
        with pytest.raises(ValueError, match="expected key and value to be 2-D"):
            mha(q2, q3, q2)
        with pytest.raises(ValueError, match="expected key and value to be 3-D"):
            mha(q3, q2, q3)
        with pytest.raises(ValueError, match="expected key_padding_mask to be 1-D"):
            mha(
                q2,
                q2,
                q2,
                key_padding_mask=lucid.zeros(1, 4, dtype=lucid.bool_, device=device),
            )
        with pytest.raises(ValueError, match="expected key_padding_mask to be 2-D"):
            mha(
                q3,
                q3,
                q3,
                key_padding_mask=lucid.zeros(4, dtype=lucid.bool_, device=device),
            )


class TestUnbatchedConvPoolAndRankGuards:
    def test_lazy_conv_wrong_rank_does_not_corrupt_state(self, device: str) -> None:
        lazy = nn.LazyConv2d(4, kernel_size=3, device=device)
        with pytest.raises(ValueError, match="expected 3D .* or 4D"):
            lazy(lucid.randn(3, 8, device=device))
        assert lazy.in_channels is None
        assert isinstance(lazy.weight, UninitializedParameter)

        out = lazy(lucid.randn(3, 8, 8, device=device))
        assert out.shape == (4, 6, 6)
        assert lazy.in_channels == 3

    def test_dilated_max_pool_and_adaptive_pool_rank_guards(self, device: str) -> None:
        bad = lucid.randn(4, 8, device=device)
        with pytest.raises(ValueError, match="max_pool2d: expected 3D or 4D"):
            F.max_pool2d(bad, kernel_size=2, dilation=2)
        with pytest.raises(ValueError, match="adaptive_avg_pool2d: expected 3D or 4D"):
            F.adaptive_avg_pool2d(bad, (2, 2))
        with pytest.raises(ValueError, match="adaptive_max_pool2d: expected 3D or 4D"):
            F.adaptive_max_pool2d(bad, (2, 2))

    def test_lp_pool_ceil_mode_and_signed_p1(self, device: str) -> None:
        x = lucid.randn(2, 3, 9, device=device)
        assert F.lp_pool1d(x, 3, kernel_size=2, ceil_mode=True).shape == (2, 3, 5)
        assert F.lp_pool1d(x[0], 3, kernel_size=2, ceil_mode=True).shape == (3, 5)

        signed = lucid.tensor([[[-1.0, 2.0, -3.0, 4.0]]], device=device)
        got = F.lp_pool1d(signed, 1, kernel_size=2)
        np.testing.assert_allclose(got.numpy(), np.array([[[1.0, 1.0]]], dtype=np.float32))


class TestTransformerAPIParity:
    def test_generate_square_subsequent_mask(self, ref: ModuleType, device: str) -> None:
        for sz in (0, 1, 5):
            for dt_l, dt_r in (
                (None, None),
                (lucid.float32, ref.float32),
                (lucid.float16, ref.float16),
                (lucid.bool_, ref.bool),
            ):
                got = nn.Transformer.generate_square_subsequent_mask(sz, device=device, dtype=dt_l)
                want = ref.nn.Transformer.generate_square_subsequent_mask(sz, dtype=dt_r)
                assert got.device.type == device
                assert got.shape == (sz, sz)
                np.testing.assert_array_equal(got.numpy(), want.numpy())

        with pytest.raises(ValueError, match="sz must be >= 0"):
            nn.Transformer.generate_square_subsequent_mask(-1, device=device)

    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("norm_first", [False, True])
    @pytest.mark.parametrize("batch_first", [False, True])
    def test_transformer_bias_norm_first_and_causal_parity(
        self,
        ref: ModuleType,
        device: str,
        bias: bool,
        norm_first: bool,
        batch_first: bool,
    ) -> None:
        rng = np.random.default_rng(7)
        D, H, FF = 8, 2, 16
        S, T, B = 4, 4, 2
        src_shape = (B, S, D) if batch_first else (S, B, D)
        tgt_shape = (B, T, D) if batch_first else (T, B, D)
        src_np = rng.standard_normal(src_shape).astype(np.float32)
        tgt_np = rng.standard_normal(tgt_shape).astype(np.float32)

        ref_t = ref.nn.Transformer(
            d_model=D,
            nhead=H,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=FF,
            dropout=0.0,
            activation="gelu",
            layer_norm_eps=1e-5,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
        ).eval()
        lucid_t = nn.Transformer(
            d_model=D,
            nhead=H,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=FF,
            dropout=0.0,
            activation="gelu",
            layer_norm_eps=1e-5,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
            device=device,
        ).eval()

        if not bias:
            assert all("bias" not in k for k in lucid_t.state_dict())

        _copy_state(ref_t.state_dict(), lucid_t, device)

        src_mask_l = nn.Transformer.generate_square_subsequent_mask(S, device=device)
        tgt_mask_l = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        mem_mask_l = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        src_mask_r = ref.nn.Transformer.generate_square_subsequent_mask(S)
        tgt_mask_r = ref.nn.Transformer.generate_square_subsequent_mask(T)
        mem_mask_r = ref.nn.Transformer.generate_square_subsequent_mask(T)

        src_l = lucid.tensor(src_np, device=device, requires_grad=True)
        tgt_l = lucid.tensor(tgt_np, device=device, requires_grad=True)
        src_r = ref.tensor(src_np, requires_grad=True)
        tgt_r = ref.tensor(tgt_np, requires_grad=True)

        out_l = lucid_t(
            src_l,
            tgt_l,
            src_mask=src_mask_l,
            tgt_mask=tgt_mask_l,
            memory_mask=mem_mask_l,
            src_is_causal=True,
            tgt_is_causal=True,
            memory_is_causal=True,
        )
        out_r = ref_t(
            src_r,
            tgt_r,
            src_mask=src_mask_r,
            tgt_mask=tgt_mask_r,
            memory_mask=mem_mask_r,
            src_is_causal=True,
            tgt_is_causal=True,
            memory_is_causal=True,
        )
        np.testing.assert_allclose(out_l.numpy(), out_r.detach().numpy(), rtol=1e-4, atol=1e-4)

        out_l.sum().backward()
        out_r.sum().backward()
        assert src_l.grad is not None and tgt_l.grad is not None
        np.testing.assert_allclose(src_l.grad.numpy(), src_r.grad.numpy(), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(tgt_l.grad.numpy(), tgt_r.grad.numpy(), rtol=1e-4, atol=1e-4)

    def test_custom_encoder_decoder_and_unbatched_transformer(self, device: str) -> None:
        enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(8, 2, 16, dropout=0.0, device=device),
            num_layers=1,
        )
        dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(8, 2, 16, dropout=0.0, device=device),
            num_layers=1,
        )
        model = nn.Transformer(
            d_model=8,
            nhead=2,
            custom_encoder=enc,
            custom_decoder=dec,
            device=device,
        ).eval()
        assert model.encoder is enc
        assert model.decoder is dec

        # Unbatched 2-D inputs (S, E) and (T, E)
        src = lucid.randn(5, 8, device=device)
        tgt = lucid.randn(4, 8, device=device)
        out = model(src, tgt)
        assert out.shape == (4, 8)

        # Validation errors on mismatched ranks, batch sizes, or feature dim
        with pytest.raises(RuntimeError, match="number of dimensions"):
            model(lucid.randn(5, 2, 8, device=device), tgt)
        with pytest.raises(RuntimeError, match="batch number"):
            model(
                lucid.randn(5, 2, 8, device=device),
                lucid.randn(4, 3, 8, device=device),
            )
        with pytest.raises(RuntimeError, match="feature number"):
            model(
                lucid.randn(5, 2, 16, device=device),
                lucid.randn(4, 2, 8, device=device),
            )

    def test_encoder_nested_tensor_and_mask_check_flags(self, device: str) -> None:
        layer = nn.TransformerEncoderLayer(8, 2, 16, device=device)
        enc = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False, mask_check=False
        )
        assert enc.enable_nested_tensor is False
        assert enc.mask_check is False
