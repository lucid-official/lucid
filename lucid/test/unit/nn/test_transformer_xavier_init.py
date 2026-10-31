"""``nn.Transformer`` starts from the reference's whole-model Xavier init (LCD-762).

The reference ends ``Transformer.__init__`` with ``_reset_parameters()``:
every parameter of two or more dimensions is drawn Xavier-uniform, over the
sub-layers' own defaults.  Lucid had the method (LCD-749) but never called it,
so a freshly built model kept the per-layer defaults.
"""

import math

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.init as init

SHAPES = [
    dict(d_model=8, nhead=2, num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=16),
    dict(d_model=16, nhead=4, num_encoder_layers=2, num_decoder_layers=3, dim_feedforward=32),
    dict(
        d_model=8,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=24,
        norm_first=True,
        bias=False,
    ),
]


def _bound(p: lucid.Tensor) -> float:
    fan_out, fan_in = int(p.shape[0]), int(p.shape[1])
    receptive = math.prod(int(d) for d in p.shape[2:])
    return math.sqrt(6.0 / ((fan_in + fan_out) * receptive))


@pytest.mark.parametrize("kw", SHAPES)
def test_matrices_lie_within_the_xavier_bound(kw: dict[str, object]) -> None:
    lucid.manual_seed(0)
    model = nn.Transformer(**kw)
    seen = 0
    for name, p in model.named_parameters():
        if p.ndim > 1:
            seen += 1
            limit = _bound(p)
            assert float(p.abs().max()) <= limit * (1 + 1e-6), name
            # Xavier fills its bound; the layer-local Linear default stops at
            # 0.5-0.7 of it, so a matrix this large tells the two apart.
            assert float(p.abs().max()) > 0.9 * limit, name
    assert seen > 0


@pytest.mark.parametrize("kw", SHAPES)
def test_vectors_keep_their_layer_defaults(kw: dict[str, object]) -> None:
    model = nn.Transformer(**kw)
    for name, p in model.named_parameters():
        if p.ndim == 1 and name.endswith("norm1.weight"):
            assert set(p.tolist()) == {1.0}, name


@pytest.mark.parametrize("kw", SHAPES)
def test_seeded_construction_equals_manual_xavier_in_parameter_order(
    kw: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    lucid.manual_seed(123)
    built = nn.Transformer(**kw)

    # Same seed with the constructor's own xavier pass switched off: every
    # layer default is drawn identically, then the pass is applied by hand in
    # the order ``parameters()`` yields -- which is what the constructor does.
    monkeypatch.setattr(nn.Transformer, "_reset_parameters", lambda self: None)
    lucid.manual_seed(123)
    manual = nn.Transformer(**kw)
    for p in manual.parameters():
        if p.ndim > 1:
            init.xavier_uniform_(p)

    for (name, a), (_, b) in zip(built.named_parameters(), manual.named_parameters(), strict=True):
        assert a.tolist() == b.tolist(), name


def test_custom_encoder_and_decoder_are_reset_too() -> None:
    enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(8, 2, 16), 1)
    dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(8, 2, 16), 1)
    model = nn.Transformer(8, 2, custom_encoder=enc, custom_decoder=dec)
    for name, p in model.named_parameters():
        if p.ndim > 1:
            assert float(p.abs().max()) <= _bound(p) * (1 + 1e-6), name
