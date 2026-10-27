"""Dedicated unit coverage for ``nn`` transformer modules.

Shape preservation through encoder / decoder stacks + a couple of config
variants (``norm_first``) + backward — these had no dedicated test file.
All use ``batch_first=True`` so inputs are ``(N, S, E)``.
"""

import lucid
import lucid.nn as nn


def _enc_layer(**kw: object) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(d_model=16, nhead=4, batch_first=True, **kw)


def _dec_layer() -> nn.TransformerDecoderLayer:
    return nn.TransformerDecoderLayer(d_model=16, nhead=4, batch_first=True)


class TestTransformerEncoder:
    def test_encoder_layer_shape(self) -> None:
        out = _enc_layer()(lucid.randn(2, 5, 16))
        assert out.shape == (2, 5, 16)

    def test_encoder_layer_norm_first(self) -> None:
        out = _enc_layer(norm_first=True)(lucid.randn(2, 5, 16))
        assert out.shape == (2, 5, 16)

    def test_encoder_stack_shape(self) -> None:
        out = nn.TransformerEncoder(_enc_layer(), num_layers=3)(lucid.randn(2, 5, 16))
        assert out.shape == (2, 5, 16)

    def test_encoder_backward(self) -> None:
        x = lucid.randn(2, 5, 16, requires_grad=True)
        _enc_layer()(x).sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 5, 16)


class TestTransformerDecoder:
    def test_decoder_layer_shape(self) -> None:
        tgt = lucid.randn(2, 6, 16)
        memory = lucid.randn(2, 5, 16)
        assert _dec_layer()(tgt, memory).shape == (2, 6, 16)

    def test_decoder_stack_shape(self) -> None:
        dec = nn.TransformerDecoder(_dec_layer(), num_layers=2)
        out = dec(lucid.randn(2, 6, 16), lucid.randn(2, 5, 16))
        assert out.shape == (2, 6, 16)


class TestTransformerFull:
    def test_full_shape(self) -> None:
        model = nn.Transformer(
            d_model=16, nhead=4, num_encoder_layers=2, num_decoder_layers=2, batch_first=True
        )
        out = model(lucid.randn(2, 5, 16), lucid.randn(2, 6, 16))
        assert out.shape == (2, 6, 16)
