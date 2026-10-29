"""A size, count or probability that no layer can use is refused where the layer reads it (LCD-672).

Every row is one invalid argument for one module or functional, run on both
libraries.  What is compared is *when* the refusal comes -- while the module
is built, or when it is first called -- and *what class* it is:

- Lucid may refuse **earlier** than the reference (at construction where the
  reference waits for the forward) but never later, and never answer where
  the reference refuses.
- The class must be the reference's.  Lucid's ``InvalidArgument`` is a
  ``ValueError`` and a ``RuntimeError`` at once, so a bad value the
  reference reports as either one is caught by either ``except``.
- A handful of rows are listed in ``_DIVERGENT`` with the reason: the
  reference's own class there is an accident (an ``assert``, a bare
  ``ZeroDivisionError``, ``NotImplementedError`` for an unknown mode string),
  and Lucid answers with the error class its taxonomy gives that kind of
  value.  Each row pins what Lucid does instead.
- A degenerate value the reference accepts (a cell with no hidden units) is
  accepted here too.

The gaps this closed: ``Conv2d(1, 1, 0)`` returned a ``(1, 1, 5, 5)`` tensor;
``LSTM(4, 0)`` raised a bare ``ZeroDivisionError`` and ``LSTM(4, 8,
num_layers=0)`` was built; ``GroupNorm(3, 4)`` failed only at the forward;
``affine_grid`` of a zero-size grid returned an empty tensor.
"""

import warnings
from collections.abc import Callable
from types import ModuleType

import pytest

import lucid
from lucid._C import engine as _C_engine

_Build = Callable[[ModuleType], object]
_Run = Callable[[ModuleType, object], object]
_Row = tuple[_Build | None, _Run | None]

_BUILD, _RUN, _OK = "build", "run", "ok"
_ORDER = {_BUILD: 0, _RUN: 1, _OK: 2}


def _fwd(*shape: int) -> _Run:
    return lambda lib, m: m(lib.randn(*shape))


def _call(fn: Callable[[ModuleType], object]) -> _Run:
    """A functional row: nothing to build, one call to make."""
    return lambda lib, _: fn(lib)


def _nn(name: str, *args: object, **kwargs: object) -> _Build:
    return lambda lib: getattr(lib.nn, name)(*args, **kwargs)


_SPATIAL = {1: (1, 1, 8), 2: (1, 1, 6, 6), 3: (1, 1, 4, 4, 4)}
_ROWS: dict[str, _Row] = {}


def _row(name: str, build: _Build | None, run: _Run | None = None) -> None:
    assert name not in _ROWS, name
    _ROWS[name] = (build, run)


# ── convolution ────────────────────────────────────────────────────────────
for _d, _shape in _SPATIAL.items():
    for _n in (f"Conv{_d}d", f"ConvTranspose{_d}d"):
        _row(f"{_n} kernel 0", _nn(_n, 1, 1, 0), _fwd(*_shape))
        _row(f"{_n} kernel -1", _nn(_n, 1, 1, -1), _fwd(*_shape))
        _row(f"{_n} stride 0", _nn(_n, 1, 1, 3, stride=0), _fwd(*_shape))
        _row(f"{_n} dilation 0", _nn(_n, 1, 1, 3, dilation=0), _fwd(*_shape))
        _row(f"{_n} padding -1", _nn(_n, 1, 1, 3, padding=-1), _fwd(*_shape))
        _row(f"{_n} groups 0", _nn(_n, 2, 2, 3, groups=0), _fwd(*_shape))
        _row(f"{_n} in_channels % groups", _nn(_n, 3, 4, 3, groups=2))
        _row(f"{_n} out_channels % groups", _nn(_n, 4, 3, 3, groups=2))
        _row(f"{_n} padding_mode", _nn(_n, 1, 1, 3, padding_mode="bogus"))
    _n = f"Conv{_d}d"
    _row(f"{_n} stride -1", _nn(_n, 1, 1, 3, stride=-1), _fwd(*_shape))
    _row(f"{_n} padding='same' stride 2", _nn(_n, 1, 1, 3, padding="same", stride=2))
    _row(f"{_n} padding string", _nn(_n, 1, 1, 3, padding="bogus"))
    _n = f"ConvTranspose{_d}d"
    _row(f"{_n} output_padding >= stride", _nn(_n, 1, 1, 3, output_padding=1), _fwd(*_shape))
    _row(f"{_n} output_padding -1", _nn(_n, 1, 1, 3, output_padding=-1), _fwd(*_shape))
_row("LazyConv2d kernel 0", _nn("LazyConv2d", 1, 0), _fwd(1, 1, 4, 4))
_row("LazyConv2d groups 0", _nn("LazyConv2d", 3, 3, groups=0), _fwd(1, 2, 4, 4))
_row("LazyConv2d out_channels % groups", _nn("LazyConv2d", 3, 3, groups=2), _fwd(1, 2, 4, 4))
_row("LazyConv2d in_channels % groups", _nn("LazyConv2d", 2, 3, groups=2), _fwd(1, 3, 4, 4))
for _f, _args in (
    ("conv1d", ((1, 1, 4), (1, 1, 0))),
    ("conv2d", ((1, 1, 4, 4), (1, 1, 0, 0))),
    ("conv3d", ((1, 1, 4, 4, 4), (1, 1, 0, 1, 1))),
    ("conv_transpose1d", ((1, 1, 4), (1, 1, 0))),
    ("conv_transpose2d", ((1, 1, 4, 4), (1, 1, 0, 0))),
):
    _row(
        f"F.{_f} kernel 0",
        None,
        lambda lib, _, f=_f, a=_args: getattr(lib.nn.functional, f)(
            lib.randn(*a[0]), lib.randn(*a[1])
        ),
    )
_row(
    "F.conv2d groups 0",
    None,
    _call(
        lambda lib: lib.nn.functional.conv2d(lib.randn(1, 2, 4, 4), lib.randn(2, 2, 3, 3), groups=0)
    ),
)
_row(
    "F.conv2d groups does not divide channels",
    None,
    _call(
        lambda lib: lib.nn.functional.conv2d(lib.randn(1, 2, 4, 4), lib.randn(2, 1, 3, 3), groups=3)
    ),
)

# ── pooling ────────────────────────────────────────────────────────────────
for _d, _shape in _SPATIAL.items():
    for _n in (f"MaxPool{_d}d", f"AvgPool{_d}d"):
        _row(f"{_n} kernel 0", _nn(_n, 0), _fwd(*_shape))
        _row(f"{_n} kernel -1", _nn(_n, -1), _fwd(*_shape))
        _row(f"{_n} stride 0", _nn(_n, 2, stride=0), _fwd(*_shape))
        _row(f"{_n} stride -1", _nn(_n, 2, stride=-1), _fwd(*_shape))
        _row(f"{_n} padding -1", _nn(_n, 2, padding=-1), _fwd(*_shape))
        _row(f"{_n} padding > kernel / 2", _nn(_n, 2, padding=2), _fwd(*_shape))
    _row(f"MaxPool{_d}d dilation 0", _nn(f"MaxPool{_d}d", 2, dilation=0), _fwd(*_shape))
    _row(f"AdaptiveAvgPool{_d}d output 0", _nn(f"AdaptiveAvgPool{_d}d", 0), _fwd(*_shape))
    _row(f"AdaptiveAvgPool{_d}d output -1", _nn(f"AdaptiveAvgPool{_d}d", -1), _fwd(*_shape))
    _row(f"AdaptiveMaxPool{_d}d output 0", _nn(f"AdaptiveMaxPool{_d}d", 0), _fwd(*_shape))
    _row(f"AdaptiveMaxPool{_d}d output -1", _nn(f"AdaptiveMaxPool{_d}d", -1), _fwd(*_shape))
for _d in (1, 2):
    _n = f"LPPool{_d}d"
    _row(f"{_n} kernel 0", _nn(_n, 2.0, 0), _fwd(*_SPATIAL[_d]))
    _row(f"{_n} stride 0", _nn(_n, 2.0, 2, stride=0), _fwd(*_SPATIAL[_d]))
    _row(f"{_n} norm_type 0", _nn(_n, 0.0, 2), _fwd(*_SPATIAL[_d]))
_row("AvgPool2d divisor_override 0", _nn("AvgPool2d", 2, divisor_override=0), _fwd(*_SPATIAL[2]))
_row("MaxUnpool2d kernel 0", _nn("MaxUnpool2d", 0))
_row("MaxUnpool2d stride 0", _nn("MaxUnpool2d", 2, stride=0))
for _d in (2, 3):
    _n = f"FractionalMaxPool{_d}d"
    _row(f"{_n} no size and no ratio", _nn(_n, 2))
    _row(f"{_n} size and ratio", _nn(_n, 2, output_size=4, output_ratio=0.5))
    _row(f"{_n} ratio 0", _nn(_n, 2, output_ratio=0.0))
    _row(f"{_n} ratio 1.5", _nn(_n, 2, output_ratio=1.5))
for _f in ("max_pool2d", "avg_pool2d", "adaptive_avg_pool2d", "adaptive_max_pool2d"):
    _row(
        f"F.{_f} 0",
        None,
        lambda lib, _, f=_f: getattr(lib.nn.functional, f)(lib.randn(1, 1, 4, 4), 0),
    )

# ── linear / embedding / dropout ───────────────────────────────────────────
_row("Linear in_features -1", _nn("Linear", -1, 3))
_row("Linear out_features -1", _nn("Linear", 3, -1))
_row("Bilinear in1_features -1", _nn("Bilinear", -1, 2, 3))
_row("Bilinear out_features -1", _nn("Bilinear", 2, 2, -1))
_row("Embedding num_embeddings -1", _nn("Embedding", -1, 3))
_row("Embedding embedding_dim -1", _nn("Embedding", 3, -1))
_row("Embedding padding_idx out of range", _nn("Embedding", 3, 2, padding_idx=5))
_row("Embedding padding_idx below range", _nn("Embedding", 3, 2, padding_idx=-5))
_row("EmbeddingBag num_embeddings -1", _nn("EmbeddingBag", -1, 3))
_row("EmbeddingBag padding_idx out of range", _nn("EmbeddingBag", 3, 2, padding_idx=5))
_row("EmbeddingBag mode", _nn("EmbeddingBag", 3, 2, mode="bogus"))
_row(
    "F.embedding padding_idx out of range",
    None,
    _call(lambda lib: lib.nn.functional.embedding(lib.tensor([0]), lib.randn(3, 2), padding_idx=9)),
)
_row(
    "F.embedding index out of range",
    None,
    _call(lambda lib: lib.nn.functional.embedding(lib.tensor([9]), lib.randn(3, 2))),
)
_row(
    "F.embedding float index",
    None,
    _call(lambda lib: lib.nn.functional.embedding(lib.tensor([0.5]), lib.randn(3, 2))),
)
_row(
    "F.one_hot num_classes -2",
    None,
    _call(lambda lib: lib.nn.functional.one_hot(lib.tensor([0, 1]), num_classes=-2)),
)
_row(
    "F.one_hot num_classes 0",
    None,
    _call(lambda lib: lib.nn.functional.one_hot(lib.tensor([0, 1]), num_classes=0)),
)
_row(
    "F.one_hot index >= num_classes",
    None,
    _call(lambda lib: lib.nn.functional.one_hot(lib.tensor([0, 5]), num_classes=3)),
)
_row(
    "F.one_hot negative index",
    None,
    _call(lambda lib: lib.nn.functional.one_hot(lib.tensor([-1, 1]), num_classes=3)),
)
for _n, _shape in (
    ("Dropout", (2, 3)),
    ("Dropout1d", (2, 3, 4)),
    ("Dropout2d", (2, 3, 4, 4)),
    ("Dropout3d", (2, 3, 2, 2, 2)),
    ("AlphaDropout", (2, 3)),
    ("FeatureAlphaDropout", (2, 3, 4)),
):
    _row(f"{_n} p -0.1", _nn(_n, -0.1), _fwd(*_shape))
    _row(f"{_n} p 1.5", _nn(_n, 1.5), _fwd(*_shape))
for _f in ("dropout", "alpha_dropout"):
    _row(
        f"F.{_f} p 2", None, lambda lib, _, f=_f: getattr(lib.nn.functional, f)(lib.randn(3), p=2.0)
    )
_row(
    "F.scaled_dot_product_attention dropout_p 2",
    None,
    _call(
        lambda lib: lib.nn.functional.scaled_dot_product_attention(
            lib.randn(1, 2, 3, 4), lib.randn(1, 2, 3, 4), lib.randn(1, 2, 3, 4), dropout_p=2.0
        )
    ),
)
_row(
    "F.scaled_dot_product_attention 1-D operands",
    None,
    _call(
        lambda lib: lib.nn.functional.scaled_dot_product_attention(
            lib.randn(4), lib.randn(4), lib.randn(4)
        )
    ),
)

# ── normalization ──────────────────────────────────────────────────────────
for _d, _shape in ((1, (4, 3)), (2, (2, 3, 4, 4)), (3, (2, 3, 2, 2, 2))):
    for _n in (f"BatchNorm{_d}d", f"InstanceNorm{_d}d"):
        _row(f"{_n} num_features 0", _nn(_n, 0))
        _row(f"{_n} num_features -1", _nn(_n, -1))
        _row(f"{_n} wrong num_features", _nn(_n, 5), _fwd(*_shape))
    _row(f"BatchNorm{_d}d eps 0", _nn(f"BatchNorm{_d}d", 3, eps=0.0), _fwd(*_shape))
    _row(f"BatchNorm{_d}d eps -1", _nn(f"BatchNorm{_d}d", 3, eps=-1.0), _fwd(*_shape))
_row("GroupNorm num_groups does not divide num_channels", _nn("GroupNorm", 3, 4), _fwd(2, 4, 3))
_row("GroupNorm num_channels -4", _nn("GroupNorm", 2, -4))
_row("GroupNorm wrong channels", _nn("GroupNorm", 2, 4), _fwd(2, 6, 3))
_row("GroupNorm 1-D input", _nn("GroupNorm", 2, 4), _fwd(4))
_row("LayerNorm normalized_shape -1", _nn("LayerNorm", -1))
_row("LayerNorm normalized_shape ()", _nn("LayerNorm", ()), _fwd(2, 3))
_row("LayerNorm wrong shape", _nn("LayerNorm", 4), _fwd(2, 3))
_row("RMSNorm normalized_shape -1", _nn("RMSNorm", -1))
_row("RMSNorm wrong shape", _nn("RMSNorm", 4), _fwd(2, 3))
_row("LocalResponseNorm size 0", _nn("LocalResponseNorm", 0), _fwd(2, 3, 4, 4))
_row("LocalResponseNorm 2-D input", _nn("LocalResponseNorm", 2), _fwd(2, 3))
for _label, _fn in (
    (
        "group_norm num_groups 3 of 4 channels",
        lambda lib: lib.nn.functional.group_norm(lib.randn(2, 4, 3), 3),
    ),
    ("group_norm num_groups 0", lambda lib: lib.nn.functional.group_norm(lib.randn(2, 4, 3), 0)),
    ("group_norm num_groups -2", lambda lib: lib.nn.functional.group_norm(lib.randn(2, 4, 3), -2)),
    ("group_norm 1-D input", lambda lib: lib.nn.functional.group_norm(lib.randn(4), 2)),
    (
        "layer_norm normalized_shape ()",
        lambda lib: lib.nn.functional.layer_norm(lib.randn(2, 3), ()),
    ),
    ("layer_norm wrong shape", lambda lib: lib.nn.functional.layer_norm(lib.randn(2, 3), (4,))),
    ("rms_norm wrong shape", lambda lib: lib.nn.functional.rms_norm(lib.randn(2, 3), (4,))),
    (
        "batch_norm training eps -1",
        lambda lib: lib.nn.functional.batch_norm(
            lib.randn(4, 3), None, None, training=True, eps=-1.0
        ),
    ),
    (
        "local_response_norm size 0",
        lambda lib: lib.nn.functional.local_response_norm(lib.randn(1, 3, 4, 4), 0),
    ),
):
    _row(f"F.{_label}", None, _call(_fn))

# ── recurrent ──────────────────────────────────────────────────────────────
for _n in ("RNN", "LSTM", "GRU"):
    _row(f"{_n} hidden_size 0", _nn(_n, 4, 0))
    _row(f"{_n} hidden_size -1", _nn(_n, 4, -1))
    _row(f"{_n} input_size 0", _nn(_n, 0, 8))
    _row(f"{_n} num_layers 0", _nn(_n, 4, 8, num_layers=0))
    _row(f"{_n} num_layers -1", _nn(_n, 4, 8, num_layers=-1))
    _row(f"{_n} dropout 2", _nn(_n, 4, 8, dropout=2.0))
    _row(f"{_n} dropout -1", _nn(_n, 4, 8, dropout=-1.0))
    _row(f"{_n} wrong input size", _nn(_n, 4, 8), _fwd(3, 2, 5))
    _row(f"{_n} 1-D input", _nn(_n, 4, 8), _fwd(4))
    _row(f"{_n} 4-D input", _nn(_n, 4, 8), _fwd(1, 2, 3, 4))
_row("RNN nonlinearity", _nn("RNN", 4, 8, nonlinearity="bogus"))
_row("LSTM proj_size -1", _nn("LSTM", 4, 8, proj_size=-1))
_row("LSTM proj_size == hidden_size", _nn("LSTM", 4, 8, proj_size=8))
_row("LSTM proj_size > hidden_size", _nn("LSTM", 4, 8, proj_size=9))
for _n in ("RNNCell", "LSTMCell", "GRUCell"):
    # The reference builds a cell with no hidden units; an empty weight is
    # not a division by zero in the init.
    _row(f"{_n} hidden_size 0", _nn(_n, 4, 0))
    _row(f"{_n} wrong input size", _nn(_n, 4, 4), _fwd(2, 5))
    _row(f"{_n} 3-D input", _nn(_n, 4, 4), _fwd(2, 3, 4))
_row("RNNCell nonlinearity", _nn("RNNCell", 4, 4, nonlinearity="bogus"), _fwd(2, 4))

# ── attention / transformer ────────────────────────────────────────────────
_row("MultiheadAttention embed_dim 0", _nn("MultiheadAttention", 0, 1))
_row("MultiheadAttention embed_dim -4", _nn("MultiheadAttention", -4, 2))
_row("MultiheadAttention num_heads 0", _nn("MultiheadAttention", 4, 0))
_row("MultiheadAttention num_heads -1", _nn("MultiheadAttention", 4, -1))
_row("MultiheadAttention embed_dim % num_heads", _nn("MultiheadAttention", 5, 2))
_row("TransformerEncoderLayer d_model 0", _nn("TransformerEncoderLayer", 0, 1))
_row("TransformerEncoderLayer nhead 0", _nn("TransformerEncoderLayer", 4, 0))
_row("TransformerEncoderLayer d_model % nhead", _nn("TransformerEncoderLayer", 5, 2))
_row("TransformerEncoderLayer activation", _nn("TransformerEncoderLayer", 4, 2, activation="bogus"))
_row("TransformerEncoderLayer dropout 2", _nn("TransformerEncoderLayer", 4, 2, dropout=2.0))
_row("TransformerDecoderLayer d_model % nhead", _nn("TransformerDecoderLayer", 5, 2))
_row("TransformerDecoderLayer activation", _nn("TransformerDecoderLayer", 4, 2, activation="bogus"))
_row("Transformer d_model 0", _nn("Transformer", d_model=0, nhead=1))
_row("Transformer d_model % nhead", _nn("Transformer", d_model=5, nhead=2))
_row("Transformer activation", _nn("Transformer", d_model=4, nhead=2, activation="bogus"))
_row("Transformer dropout 2", _nn("Transformer", d_model=4, nhead=2, dropout=2.0))

# ── resampling / rearranging ───────────────────────────────────────────────
_X4 = (1, 1, 4, 4)
_row("Upsample neither size nor scale_factor", _nn("Upsample"), _fwd(*_X4))
_row("Upsample size and scale_factor", _nn("Upsample", size=4, scale_factor=2), _fwd(*_X4))
_row("Upsample scale_factor 0", _nn("Upsample", scale_factor=0), _fwd(*_X4))
_row("Upsample scale_factor -1", _nn("Upsample", scale_factor=-1), _fwd(*_X4))
_row("Upsample size 0", _nn("Upsample", size=0), _fwd(*_X4))
_row("Upsample size -1", _nn("Upsample", size=-1), _fwd(*_X4))
_row(
    "Upsample align_corners with nearest",
    _nn("Upsample", scale_factor=2, align_corners=True),
    _fwd(*_X4),
)
_row("Upsample mode", _nn("Upsample", scale_factor=2, mode="bogus"), _fwd(*_X4))
_row("Upsample 2-D input", _nn("Upsample", scale_factor=2), _fwd(4, 4))
_row("UpsamplingNearest2d size 0", _nn("UpsamplingNearest2d", size=0), _fwd(*_X4))
_row("PixelShuffle upscale_factor 0", _nn("PixelShuffle", 0), _fwd(1, 4, 2, 2))
_row("PixelShuffle upscale_factor -1", _nn("PixelShuffle", -1), _fwd(1, 4, 2, 2))
_row("PixelShuffle channels % r^2", _nn("PixelShuffle", 2), _fwd(1, 3, 2, 2))
_row("PixelShuffle 2-D input", _nn("PixelShuffle", 2), _fwd(4, 4))
_row("PixelUnshuffle downscale_factor 0", _nn("PixelUnshuffle", 0), _fwd(*_X4))
_row("PixelUnshuffle downscale_factor -1", _nn("PixelUnshuffle", -1), _fwd(*_X4))
_row("PixelUnshuffle size % r", _nn("PixelUnshuffle", 3), _fwd(*_X4))
_row("ChannelShuffle groups 0", _nn("ChannelShuffle", 0), _fwd(1, 4, 2, 2))
_row("ChannelShuffle groups -1", _nn("ChannelShuffle", -1), _fwd(1, 4, 2, 2))
_row("ChannelShuffle channels % groups", _nn("ChannelShuffle", 3), _fwd(1, 4, 2, 2))
_row("Unfold kernel 0", _nn("Unfold", 0), _fwd(*_X4))
_row("Unfold kernel -1", _nn("Unfold", -1), _fwd(*_X4))
_row("Unfold stride 0", _nn("Unfold", 2, stride=0), _fwd(*_X4))
_row("Unfold dilation 0", _nn("Unfold", 2, dilation=0), _fwd(*_X4))
_row("Unfold padding -1", _nn("Unfold", 2, padding=-1), _fwd(*_X4))
_row("Unfold kernel > input", _nn("Unfold", 9), _fwd(*_X4))
_row("Unfold 5-D input", _nn("Unfold", 2), _fwd(1, 1, 1, 4, 4))
_row("Fold kernel 0", _nn("Fold", 4, 0), _fwd(1, 4, 9))
_row("Fold output_size 0", _nn("Fold", 0, 2), _fwd(1, 4, 9))
_row("Fold stride 0", _nn("Fold", 4, 2, stride=0), _fwd(1, 4, 9))
_row("Fold dilation 0", _nn("Fold", 4, 2, dilation=0), _fwd(1, 4, 9))
_row("Fold padding -1", _nn("Fold", 4, 2, padding=-1), _fwd(1, 4, 9))
_row("Fold wrong block count", _nn("Fold", 4, 2), _fwd(1, 4, 8))
_row("Fold channels % kernel", _nn("Fold", 4, 2), _fwd(1, 5, 9))
_row("Flatten start_dim > end_dim", _nn("Flatten", 2, 1), _fwd(2, 3, 4))
_row("Flatten dim out of range", _nn("Flatten", 5), _fwd(2, 3, 4))
_row("Unflatten sizes do not multiply to the axis", _nn("Unflatten", 1, (2, 2)), _fwd(2, 3))
_row("Unflatten zero size", _nn("Unflatten", 1, (0, 3)), _fwd(2, 3))
_row("ReflectionPad2d wider than the input", _nn("ReflectionPad2d", 5), _fwd(*_X4))
_row("ZeroPad2d wrong length", _nn("ZeroPad2d", (1, 2, 3)), _fwd(*_X4))
for _label, _fn in (
    (
        "affine_grid zero-size grid",
        lambda lib: lib.nn.functional.affine_grid(
            lib.randn(1, 2, 3), (1, 1, 0, 0), align_corners=False
        ),
    ),
    (
        "affine_grid negative size",
        lambda lib: lib.nn.functional.affine_grid(
            lib.randn(1, 2, 3), (1, 1, -2, 2), align_corners=False
        ),
    ),
    (
        "affine_grid zero-size volume",
        lambda lib: lib.nn.functional.affine_grid(
            lib.randn(1, 3, 4), (1, 1, 0, 2, 2), align_corners=False
        ),
    ),
    (
        "affine_grid theta shape",
        lambda lib: lib.nn.functional.affine_grid(
            lib.randn(1, 3, 3), (1, 1, 2, 2), align_corners=False
        ),
    ),
    (
        "affine_grid batch mismatch",
        lambda lib: lib.nn.functional.affine_grid(
            lib.randn(2, 2, 3), (1, 1, 2, 2), align_corners=False
        ),
    ),
    ("interpolate size 0", lambda lib: lib.nn.functional.interpolate(lib.randn(*_X4), size=0)),
    (
        "interpolate size (0, 2)",
        lambda lib: lib.nn.functional.interpolate(lib.randn(*_X4), size=(0, 2)),
    ),
    (
        "interpolate scale_factor 0",
        lambda lib: lib.nn.functional.interpolate(lib.randn(*_X4), scale_factor=0),
    ),
    (
        "interpolate scale_factor -1",
        lambda lib: lib.nn.functional.interpolate(lib.randn(*_X4), scale_factor=-1),
    ),
    (
        "interpolate size and scale_factor",
        lambda lib: lib.nn.functional.interpolate(lib.randn(*_X4), size=4, scale_factor=2),
    ),
    ("interpolate neither", lambda lib: lib.nn.functional.interpolate(lib.randn(*_X4))),
    (
        "pixel_shuffle factor 0",
        lambda lib: lib.nn.functional.pixel_shuffle(lib.randn(1, 4, 2, 2), 0),
    ),
    ("pixel_unshuffle factor 0", lambda lib: lib.nn.functional.pixel_unshuffle(lib.randn(*_X4), 0)),
    (
        "channel_shuffle groups 0",
        lambda lib: lib.nn.functional.channel_shuffle(lib.randn(1, 4, 2, 2), 0),
    ),
    ("unfold kernel 0", lambda lib: lib.nn.functional.unfold(lib.randn(*_X4), 0)),
    ("fold kernel 0", lambda lib: lib.nn.functional.fold(lib.randn(1, 4, 9), 4, 0)),
    ("pdist 1-D input", lambda lib: lib.nn.functional.pdist(lib.randn(3))),
    ("pdist p -1", lambda lib: lib.nn.functional.pdist(lib.randn(3, 2), p=-1.0)),
    ("glu odd axis", lambda lib: lib.nn.functional.glu(lib.randn(2, 3))),
    ("rrelu lower > upper", lambda lib: lib.nn.functional.rrelu(lib.randn(3), 0.5, 0.1)),
    ("softplus beta 0", lambda lib: lib.nn.functional.softplus(lib.randn(3), beta=0.0)),
    (
        "max_unpool2d index out of range",
        lambda lib: lib.nn.functional.max_unpool2d(
            lib.randn(1, 1, 2, 2), lib.zeros(1, 1, 2, 2).long() + 99, 2
        ),
    ),
):
    _row(f"F.{_label}", None, _call(_fn))
_row("GLU odd axis", _nn("GLU"), _fwd(2, 3))
_row("RReLU lower > upper", _nn("RReLU", 0.5, 0.1), _fwd(3))
_row("Softplus beta 0", _nn("Softplus", beta=0.0), _fwd(3))
_row("PReLU wrong num_parameters", _nn("PReLU", 3), _fwd(2, 4, 5))
_row("PReLU num_parameters -1", _nn("PReLU", -1))

# ── losses ─────────────────────────────────────────────────────────────────
_row(
    "CrossEntropyLoss label_smoothing 2",
    _nn("CrossEntropyLoss", label_smoothing=2.0),
    lambda lib, m: m(lib.randn(3, 4), lib.tensor([0, 1, 2])),
)
_row(
    "CrossEntropyLoss weight length",
    lambda lib: lib.nn.CrossEntropyLoss(weight=lib.ones(5)),
    lambda lib, m: m(lib.randn(3, 4), lib.tensor([0, 1, 2])),
)
_row(
    "MarginRankingLoss shape",
    _nn("MarginRankingLoss"),
    lambda lib, m: m(lib.randn(3), lib.randn(4), lib.ones(3)),
)
_row("HuberLoss delta 0", _nn("HuberLoss", delta=0.0), lambda lib, m: m(lib.randn(3), lib.randn(3)))
_row(
    "SmoothL1Loss beta -1",
    _nn("SmoothL1Loss", beta=-1.0),
    lambda lib, m: m(lib.randn(3), lib.randn(3)),
)
for _n in (
    "MSELoss",
    "L1Loss",
    "SmoothL1Loss",
    "HuberLoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "KLDivLoss",
):
    _row(f"{_n} reduction", _nn(_n, reduction="bogus"), lambda lib, m: m(lib.rand(3), lib.rand(3)))

#: Rows where the reference's class is an accident of how it checks, and what
#: Lucid answers instead: (stage, class).  The reason is the dict key's comment.
_DIVERGENT: dict[str, tuple[str, type[BaseException]]] = {
    # An ``assert`` in the reference -- stripped under ``python -O``.
    "Embedding padding_idx out of range": (_BUILD, ValueError),
    "Embedding padding_idx below range": (_BUILD, ValueError),
    "EmbeddingBag padding_idx out of range": (_BUILD, ValueError),
    "F.embedding padding_idx out of range": (_RUN, ValueError),
    "MultiheadAttention embed_dim % num_heads": (_BUILD, ValueError),
    "TransformerEncoderLayer d_model % nhead": (_BUILD, ValueError),
    "TransformerDecoderLayer d_model % nhead": (_BUILD, ValueError),
    "Transformer d_model % nhead": (_BUILD, ValueError),
    # A bare ``ZeroDivisionError`` falling out of ``num_channels % 0`` or ``1 / p``.
    "GroupNorm num_groups does not divide num_channels": (_BUILD, ValueError),
    "F.group_norm num_groups 0": (_RUN, ValueError),
    "LPPool1d norm_type 0": (_RUN, ValueError),
    "LPPool2d norm_type 0": (_RUN, ValueError),
    # ``NotImplementedError`` for a mode string or input rank it does not list;
    # the taxonomy calls an unknown mode string a value error.
    "Upsample mode": (_RUN, ValueError),
    "Upsample 2-D input": (_RUN, ValueError),
}

#: What an invalid argument raises, in either library.
_REFUSALS = (ValueError, RuntimeError, TypeError, LookupError, AssertionError, ArithmeticError)


def _outcome(
    lib: ModuleType, build: _Build | None, run: _Run | None
) -> tuple[str, BaseException | None]:
    """Where ``lib`` refuses the row, and with what: ``(stage, exception)``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            obj = build(lib) if build is not None else None
        except _REFUSALS as exc:
            return _BUILD, exc
        if run is None:
            return _OK, None
        try:
            run(lib, obj)
        except _REFUSALS as exc:
            return _RUN, exc
    return _OK, None


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_ROWS))
def test_an_invalid_argument_is_refused_as_the_reference_refuses_it(
    ref: ModuleType, name: str
) -> None:
    build, run = _ROWS[name]
    want_stage, want_exc = _outcome(ref, build, run)
    got_stage, got_exc = _outcome(lucid, build, run)

    if name in _DIVERGENT:
        stage, cls = _DIVERGENT[name]
        assert want_stage != _OK, f"{name}: the reference no longer refuses; drop the row"
        assert got_stage == stage and isinstance(
            got_exc, cls
        ), f"{name}: Lucid gave {got_stage}:{type(got_exc).__name__}"
        return

    if want_stage == _OK:
        assert got_stage == _OK, f"{name}: the reference accepts it, Lucid refused: {got_exc!r}"
        return
    assert got_stage != _OK, f"{name}: Lucid answered where the reference refuses ({want_exc!r})"
    assert (
        _ORDER[got_stage] <= _ORDER[want_stage]
    ), f"{name}: Lucid refuses at {got_stage}, later than the reference ({want_stage})"
    assert isinstance(
        got_exc, type(want_exc)
    ), f"{name}: Lucid raised {type(got_exc).__name__}, the reference {type(want_exc).__name__}"


# ── the reference's own spellings, without a reference installed ───────────


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda: lucid.nn.LSTM(4, 0), "hidden_size"),
        (lambda: lucid.nn.GRU(4, 0), "hidden_size"),
        (lambda: lucid.nn.RNN(4, 0), "hidden_size"),
        (lambda: lucid.nn.LSTM(4, 8, num_layers=0), "num_layers"),
        (lambda: lucid.nn.LSTM(0, 8), "input_size"),
        (lambda: lucid.nn.GroupNorm(3, 4), "divisible"),
        (lambda: lucid.nn.Conv2d(3, 4, 3, groups=2), "in_channels"),
        (lambda: lucid.nn.Conv2d(2, 3, 3, groups=2), "out_channels"),
        (lambda: lucid.nn.Conv2d(2, 2, 3, groups=0), "groups"),
        (lambda: lucid.nn.MultiheadAttention(4, 0), "greater than 0"),
        (lambda: lucid.nn.FractionalMaxPool2d(2, output_size=4, output_ratio=0.5), "exactly one"),
    ],
    ids=str,
)
def test_a_module_refuses_a_bad_size_when_it_is_built(
    make: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        make()


@pytest.mark.filterwarnings("ignore:Initializing zero-element")
def test_a_convolution_with_an_empty_kernel_is_refused_at_the_forward() -> None:
    layer = lucid.nn.Conv2d(1, 1, 0)
    x = lucid.randn(1, 1, 4, 4)
    for catch in (_C_engine.InvalidArgument, ValueError, RuntimeError):
        with pytest.raises(catch, match="kernel_size"):
            layer(x)


def test_a_zero_size_affine_grid_is_refused() -> None:
    theta = lucid.randn(1, 2, 3)
    with pytest.raises(ValueError, match="positive"):
        lucid.nn.functional.affine_grid(theta, (1, 1, 0, 0), align_corners=False)


def test_a_cell_with_no_hidden_units_is_built() -> None:
    for cell in (lucid.nn.RNNCell, lucid.nn.LSTMCell, lucid.nn.GRUCell):
        assert cell(4, 0).hidden_size == 0
