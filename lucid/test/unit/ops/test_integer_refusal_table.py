"""The bool and integer refusals of the Python entry points, pinned row by row (LCD-1026).

``lucid._ops._bool_refusal.ROWS`` is the one table of what the Python entry points -- the
composites, the adapters, the losses -- turn away: a row per ``(op, role)`` says, for bool and
each signed integer, the class the reference raises, or that the reference answers.  This file is
the test of the table:

* every row has a call (a *pin*), and every pin names a row -- a refusal cannot be added and left
  unpinned, and a row cannot be dropped unnoticed;
* each pin is replayed on the reference for bool, int8, int16, int32 and int64 (``-m parity``): it
  answers where the row says it answers, and raises the row's class where the row refuses -- a
  verdict *or* a class that drifts from the reference's fails;
* each pin is replayed on Lucid, on both devices, and does the same;
* the table is well formed: one owner per ``(op, dtype)`` against the engine's schemas, one
  spelling of a dtype in a message across the two layers, and a float input pays for no lookup.

The classes, named after how the reference refuses:

``kernel``
    ``NotImplementedError``: a kernel with no instantiation for the dtype.
``guard``
    a ``RuntimeError`` that is not a ``NotImplementedError``: a check on the input dtype.
``value``
    a ``ValueError``: a check Python makes on an argument.

A pin that says nothing about its dtypes takes them from its row.  A call whose answer differs
from the row's -- a loss beside a floating target, an operand of another dtype -- names the dtypes
it refuses and the class.  Every Lucid refusal is a ``RuntimeError``, as the reference's are.
"""

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._ops import _bool_refusal as _table
from lucid._ops._bool_refusal import BOOL_REFUSED, ROWS, Rule

pytestmark = pytest.mark.filterwarnings("ignore")

_DTYPES = ("bool", "int8", "int16", "int32", "int64")
_ALL = frozenset(_DTYPES)
_NONE: frozenset[str] = frozenset()

#: The class each Lucid error names, as the reference's classes are told apart.
_LABEL: dict[type[BaseException], str] = {
    _C_engine.NotImplementedError: "kernel",
    _C_engine.DtypeMismatch: "guard",
    _C_engine.InvalidArgument: "value",
}

# ``fw`` is the framework under test, ``x`` a vector and ``m`` a 2x3 matrix of the dtype being
# probed.  Both frameworks spell every call below the same way.
_Call = Callable[[ModuleType, object, object], object]


@dataclass(frozen=True)
class _Pin:
    """One call of an op, and what the table says of it."""

    op: str
    role: str
    call: _Call
    #: The class when it is not the row's rule (the row's ``InvalidArgument`` is a ``value`` for
    #: ``affine_grid`` and a ``guard`` for ``one_hot``), or by dtype where it varies with it.
    kind: str | dict[str, str] | None = None
    #: The dtypes this call refuses, when it is not what the row says (a loss beside a float).
    refused: frozenset[str] | None = None


def _pin(
    op: str,
    call: _Call,
    *,
    role: str = "",
    kind: str | dict[str, str] | None = None,
    refused: frozenset[str] | None = None,
) -> _Pin:
    return _Pin(op, role, call, kind, refused)


def _F(fw: ModuleType) -> ModuleType:
    return fw.nn.functional  # type: ignore[no-any-return]


def _floats(fw: ModuleType) -> object:
    return fw.tensor([1.0, 0.5, 2.0, 1.5])


_PINS: dict[str, _Pin] = {
    # ── order statistics, counting and membership ───────────────────────────────────────────────
    "quantile": _pin("quantile", lambda fw, x, m: fw.quantile(x, 0.5)),
    "nanquantile": _pin("nanquantile", lambda fw, x, m: fw.nanquantile(x, 0.5)),
    "searchsorted": _pin("searchsorted", lambda fw, x, m: fw.searchsorted(x, x[:1])),
    "bucketize": _pin("bucketize", lambda fw, x, m: fw.bucketize(x, x)),
    "isin": _pin("isin", lambda fw, x, m: fw.isin(x, x)),
    "histogram": _pin("histogram", lambda fw, x, m: fw.histogram(x, 2)),
    "histogram-bins": _pin(
        "histogram", lambda fw, x, m: fw.histogram(x.float(), x), role="bins"
    ),
    "histogram-weight": _pin(
        "histogram", lambda fw, x, m: fw.histogram(x.float(), 2, weight=x), role="weight"
    ),
    "poisson": _pin("poisson", lambda fw, x, m: fw.poisson(x)),
    "multinomial": _pin("multinomial", lambda fw, x, m: fw.multinomial(x, 1)),
    # ── moments ─────────────────────────────────────────────────────────────────────────────────
    "std_mean": _pin("std_mean", lambda fw, x, m: fw.std_mean(x)),
    "var_mean": _pin("var_mean", lambda fw, x, m: fw.var_mean(x)),
    "nanmean": _pin("nanmean", lambda fw, x, m: fw.nanmean(x)),
    "cov": _pin("cov", lambda fw, x, m: fw.cov(m)),
    "corrcoef": _pin("corrcoef", lambda fw, x, m: fw.corrcoef(m)),
    "gradient": _pin("gradient", lambda fw, x, m: fw.gradient(x)),
    # ── elementwise ─────────────────────────────────────────────────────────────────────────────
    "clamp": _pin("clamp", lambda fw, x, m: fw.clamp(x, x, x)),
    "pow": _pin("pow", lambda fw, x, m: fw.pow(x, x)),
    "frexp": _pin("frexp", lambda fw, x, m: fw.frexp(x)),
    "polar": _pin("polar", lambda fw, x, m: fw.polar(x, x)),
    "polar-first": _pin("polar", lambda fw, x, m: fw.polar(x, x.float())),
    "polar-second": _pin("polar", lambda fw, x, m: fw.polar(x.float(), x)),
    # ── special functions and linear algebra ────────────────────────────────────────────────────
    # A pair of integers meets no floating operand and has no kernel; an integer beside a float
    # promotes and is answered.
    "gammainc": _pin("gammainc", lambda fw, x, m: fw.special.gammainc(x, x)),
    "gammainc-float": _pin(
        "gammainc", lambda fw, x, m: fw.special.gammainc(x, x.float() + 1.0), refused=_NONE
    ),
    "gammainc-float-first": _pin(
        "gammainc", lambda fw, x, m: fw.special.gammainc(x.float() + 1.0, x), refused=_NONE
    ),
    "gammainc-integer-pair": _pin(
        "gammainc", lambda fw, x, m: fw.special.gammainc(x, x.long())
    ),
    "gammaincc": _pin("gammaincc", lambda fw, x, m: fw.special.gammaincc(x, x)),
    "gammaincc-float": _pin(
        "gammaincc", lambda fw, x, m: fw.special.gammaincc(x.float() + 1.0, x), refused=_NONE
    ),
    "multigammaln": _pin("multigammaln", lambda fw, x, m: fw.special.multigammaln(x, 1)),
    "vecdot": _pin("vecdot", lambda fw, x, m: fw.linalg.vecdot(x, x)),
    "vecdot-first": _pin("vecdot", lambda fw, x, m: fw.linalg.vecdot(x, x.float())),
    "vecdot-second": _pin("vecdot", lambda fw, x, m: fw.linalg.vecdot(x.float(), x)),
    "vander": _pin("vander", lambda fw, x, m: fw.linalg.vander(x)),
    "matrix_exp": _pin("matrix_exp", lambda fw, x, m: fw.linalg.matrix_exp(m[:, :2])),
    "ldl_solve": _pin(
        "ldl_solve",
        lambda fw, x, m: fw.linalg.ldl_solve(
            m[:, :2], fw.tensor([1, 2], dtype=fw.int32), fw.tensor([[1.0], [2.0]])
        ),
    ),
    # ── activations, dropout and normalisation composed from other ops ──────────────────────────
    "celu": _pin("celu", lambda fw, x, m: _F(fw).celu(x)),
    "celu-inplace": _pin("celu", lambda fw, x, m: _F(fw).celu_(x.clone())),
    "hardshrink": _pin("hardshrink", lambda fw, x, m: _F(fw).hardshrink(x)),
    "softshrink": _pin("softshrink", lambda fw, x, m: _F(fw).softshrink(x)),
    "tanhshrink": _pin("tanhshrink", lambda fw, x, m: _F(fw).tanhshrink(x)),
    "hardtanh": _pin("hardtanh", lambda fw, x, m: _F(fw).hardtanh(x)),
    "threshold": _pin("threshold", lambda fw, x, m: _F(fw).threshold(x, 0.1, 20.0)),
    "rrelu": _pin("rrelu", lambda fw, x, m: _F(fw).rrelu(x)),
    "rrelu-training": _pin("rrelu", lambda fw, x, m: _F(fw).rrelu(x, training=True)),
    "prelu": _pin("prelu", lambda fw, x, m: _F(fw).prelu(x, x[:1])),
    "prelu-weight": _pin(
        "prelu", lambda fw, x, m: _F(fw).prelu(x.float(), x[:1]), role="mixed"
    ),
    "prelu-input": _pin(
        "prelu", lambda fw, x, m: _F(fw).prelu(x, x[:1].float()), role="mixed"
    ),
    "glu": _pin("glu", lambda fw, x, m: _F(fw).glu(x)),
    "gumbel_softmax": _pin("gumbel_softmax", lambda fw, x, m: _F(fw).gumbel_softmax(x)),
    "feature_alpha_dropout": _pin(
        "feature_alpha_dropout",
        lambda fw, x, m: _F(fw).feature_alpha_dropout(x.reshape(1, 2, 2), p=0.5, training=True),
    ),
    # Its integers are the engine's (the schema declares them), its bool is the entry's.
    "rms_norm": _pin(
        "rms_norm", lambda fw, x, m: _F(fw).rms_norm(x, (4,)), kind="kernel", refused=_ALL
    ),
    "rms_norm-eps": _pin(
        "rms_norm",
        lambda fw, x, m: _F(fw).rms_norm(x, (4,), eps=1e-5),
        kind="kernel",
        refused=_ALL,
    ),
    "instance_norm": _pin(
        "instance_norm", lambda fw, x, m: _F(fw).instance_norm(m.reshape(1, 2, 3))
    ),
    "local_response_norm": _pin(
        "local_response_norm", lambda fw, x, m: _F(fw).local_response_norm(m.reshape(1, 2, 3), 2)
    ),
    # ── pooling ─────────────────────────────────────────────────────────────────────────────────
    "max_unpool1d": _pin(
        "max_unpool1d",
        lambda fw, x, m: _F(fw).max_unpool1d(
            x[:2].reshape(1, 1, 2), fw.tensor([[[0, 2]]], dtype=fw.int64), 2
        ),
    ),
    "max_unpool2d": _pin(
        "max_unpool2d",
        lambda fw, x, m: _F(fw).max_unpool2d(
            x.reshape(1, 1, 2, 2), fw.tensor([[[[0, 5], [10, 15]]]], dtype=fw.int64), 2
        ),
    ),
    "max_unpool3d": _pin(
        "max_unpool3d",
        lambda fw, x, m: _F(fw).max_unpool3d(
            x.reshape(1, 1, 1, 2, 2), fw.tensor([[[[[0, 5], [10, 15]]]]], dtype=fw.int64), 2
        ),
    ),
    "adaptive_avg_pool1d": _pin(
        "adaptive_avg_pool1d", lambda fw, x, m: _F(fw).adaptive_avg_pool1d(x.reshape(1, 1, 4), 2)
    ),
    "adaptive_avg_pool1d-global": _pin(
        "adaptive_avg_pool1d",
        lambda fw, x, m: _F(fw).adaptive_avg_pool1d(x.reshape(1, 1, 4), 1),
        role="global",
    ),
    "adaptive_avg_pool2d": _pin(
        "adaptive_avg_pool2d",
        lambda fw, x, m: _F(fw).adaptive_avg_pool2d(m.reshape(1, 1, 2, 3), (1, 2)),
    ),
    "adaptive_avg_pool2d-global": _pin(
        "adaptive_avg_pool2d",
        lambda fw, x, m: _F(fw).adaptive_avg_pool2d(m.reshape(1, 1, 2, 3), 1),
        role="global",
    ),
    "adaptive_avg_pool3d": _pin(
        "adaptive_avg_pool3d",
        lambda fw, x, m: _F(fw).adaptive_avg_pool3d(x.reshape(1, 1, 1, 2, 2), (1, 1, 2)),
    ),
    "adaptive_avg_pool3d-global": _pin(
        "adaptive_avg_pool3d",
        lambda fw, x, m: _F(fw).adaptive_avg_pool3d(x.reshape(1, 1, 1, 2, 2), 1),
        role="global",
    ),
    "adaptive_max_pool1d": _pin(
        "adaptive_max_pool1d", lambda fw, x, m: _F(fw).adaptive_max_pool1d(x.reshape(1, 1, 4), 2)
    ),
    "adaptive_max_pool2d": _pin(
        "adaptive_max_pool2d",
        lambda fw, x, m: _F(fw).adaptive_max_pool2d(m.reshape(1, 1, 2, 3), 1),
    ),
    "adaptive_max_pool2d-indices": _pin(
        "adaptive_max_pool2d",
        lambda fw, x, m: _F(fw).adaptive_max_pool2d(m.reshape(1, 1, 2, 3), 1, return_indices=True),
    ),
    "adaptive_max_pool3d": _pin(
        "adaptive_max_pool3d",
        lambda fw, x, m: _F(fw).adaptive_max_pool3d(x.reshape(1, 1, 1, 2, 2), 1),
    ),
    # ── initialisers and samplers drawn through a float tensor ──────────────────────────────────
    "orthogonal": _pin("orthogonal", lambda fw, x, m: fw.nn.init.orthogonal_(m)),
    "sparse": _pin("sparse", lambda fw, x, m: fw.nn.init.sparse_(m, 0.5)),
    "cauchy": _pin("cauchy", lambda fw, x, m: x.cauchy_()),
    "exponential": _pin("exponential", lambda fw, x, m: x.exponential_()),
    # ── indices and masks ───────────────────────────────────────────────────────────────────────
    # The reference wants int64 class indices: a plain RuntimeError for the rest.
    "one_hot": _pin("one_hot", lambda fw, x, m: _F(fw).one_hot(x, 2), kind="guard"),
    "masked_select-mask": _pin(
        "masked_select", lambda fw, x, m: fw.masked_select(x.float(), x), role="mask"
    ),
    "masked_select-input": _pin(
        "masked_select", lambda fw, x, m: fw.masked_select(x, x.float() > 0.5), role="mask", refused=_NONE
    ),
    "attn_mask": _pin(
        "scaled_dot_product_attention",
        lambda fw, x, m: _F(fw).scaled_dot_product_attention(
            m.float().reshape(1, 2, 3),
            m.float().reshape(1, 2, 3),
            m.float().reshape(1, 2, 3),
            attn_mask=x.reshape(2, 2),
        ),
        role="attn_mask",
    ),
    # An integer table: the engine op declares it for the one path that reaches it, the entry
    # for the rest; a bool one is the entry's.
    "embedding_bag": _pin(
        "embedding_bag",
        lambda fw, x, m: _F(fw).embedding_bag(
            fw.tensor([0, 1], dtype=fw.int64), m[:, :2], fw.tensor([0], dtype=fw.int64)
        ),
        kind="guard",
        refused=_ALL,
    ),
    "embedding_bag-weighted": _pin(
        "embedding_bag",
        lambda fw, x, m: _F(fw).embedding_bag(
            fw.tensor([0, 1], dtype=fw.int64),
            m[:, :2],
            fw.tensor([0], dtype=fw.int64),
            per_sample_weights=fw.tensor([1.0, 2.0]),
            mode="sum",
        ),
        kind="guard",
        refused=_ALL,
    ),
    # ── resampling ──────────────────────────────────────────────────────────────────────────────
    "interpolate": _pin(
        "interpolate", lambda fw, x, m: _F(fw).interpolate(m.reshape(1, 2, 3), scale_factor=2)
    ),
    "interpolate-bilinear": _pin(
        "interpolate",
        lambda fw, x, m: _F(fw).interpolate(m.reshape(1, 1, 2, 3), scale_factor=2, mode="bilinear"),
    ),
    "interpolate-area": _pin(
        "interpolate",
        lambda fw, x, m: _F(fw).interpolate(m.reshape(1, 2, 3), size=2, mode="area"),
    ),
    "upsample": _pin(
        "interpolate", lambda fw, x, m: _F(fw).upsample(m.reshape(1, 1, 2, 3), scale_factor=2)
    ),
    "grid_sample": _pin(
        "grid_sample",
        lambda fw, x, m: _F(fw).grid_sample(
            m.reshape(1, 1, 2, 3), m.float().reshape(1, 1, 3, 2) * 0.5, align_corners=False
        ),
    ),
    "grid_sample-grid": _pin(
        "grid_sample",
        lambda fw, x, m: _F(fw).grid_sample(
            m.float().reshape(1, 1, 2, 3), m.reshape(1, 1, 3, 2), align_corners=False
        ),
        role="grid",
    ),
    "affine_grid": _pin(
        "affine_grid",
        lambda fw, x, m: _F(fw).affine_grid(m.reshape(1, 2, 3), [1, 1, 2, 2], align_corners=False),
        kind="value",
    ),
    # ── losses ──────────────────────────────────────────────────────────────────────────────────
    # An integer pair has no kernel; beside a floating operand the reference answers it
    # (``mse_loss``) or reads it as a dtype mismatch (``huber_loss``, the BCE losses).
    "mse_loss": _pin("mse_loss", lambda fw, x, m: _F(fw).mse_loss(x, x)),
    "mse_loss-pair-of-widths": _pin("mse_loss", lambda fw, x, m: _F(fw).mse_loss(x, x.long())),
    "mse_loss-float-target": _pin(
        "mse_loss", lambda fw, x, m: _F(fw).mse_loss(x, x.float()), refused=_NONE
    ),
    "mse_loss-float-input": _pin(
        "mse_loss", lambda fw, x, m: _F(fw).mse_loss(x.float(), x), refused=_NONE
    ),
    "l1_loss": _pin("l1_loss", lambda fw, x, m: _F(fw).l1_loss(x, x)),
    "l1_loss-float-target": _pin(
        "l1_loss", lambda fw, x, m: _F(fw).l1_loss(x, x.float()), refused=frozenset({"bool"})
    ),
    "l1_loss-float-input": _pin(
        "l1_loss", lambda fw, x, m: _F(fw).l1_loss(x.float(), x), refused=frozenset({"bool"})
    ),
    "smooth_l1_loss": _pin("smooth_l1_loss", lambda fw, x, m: _F(fw).smooth_l1_loss(x, x)),
    "smooth_l1_loss-float-target": _pin(
        "smooth_l1_loss", lambda fw, x, m: _F(fw).smooth_l1_loss(x, x.float()), refused=_NONE
    ),
    "huber_loss": _pin("huber_loss", lambda fw, x, m: _F(fw).huber_loss(x, x)),
    "huber_loss-pair-of-widths": _pin(
        "huber_loss",
        lambda fw, x, m: _F(fw).huber_loss(x, x.long()),
        kind={"bool": "guard"},
    ),
    "huber_loss-float-target": _pin(
        "huber_loss", lambda fw, x, m: _F(fw).huber_loss(x, x.float()), kind="guard"
    ),
    "huber_loss-float-input": _pin(
        "huber_loss", lambda fw, x, m: _F(fw).huber_loss(x.float(), x), refused=_NONE
    ),
    "binary_cross_entropy": _pin(
        "binary_cross_entropy", lambda fw, x, m: _F(fw).binary_cross_entropy(x, x)
    ),
    "binary_cross_entropy-float-target": _pin(
        "binary_cross_entropy",
        lambda fw, x, m: _F(fw).binary_cross_entropy(x, x.float()),
        kind="guard",
    ),
    "binary_cross_entropy-float-input": _pin(
        "binary_cross_entropy",
        lambda fw, x, m: _F(fw).binary_cross_entropy(x.float() * 0.5, x),
        kind="guard",
    ),
    "binary_cross_entropy-pair-of-widths": _pin(
        "binary_cross_entropy",
        lambda fw, x, m: _F(fw).binary_cross_entropy(x, x.long()),
        kind={"int64": "kernel", "bool": "guard", "int8": "guard", "int16": "guard", "int32": "guard"},
    ),
    "binary_cross_entropy_with_logits": _pin(
        "binary_cross_entropy_with_logits",
        lambda fw, x, m: _F(fw).binary_cross_entropy_with_logits(x, x),
    ),
    "binary_cross_entropy_with_logits-float-target": _pin(
        "binary_cross_entropy_with_logits",
        lambda fw, x, m: _F(fw).binary_cross_entropy_with_logits(x, x.float()),
    ),
    "binary_cross_entropy_with_logits-float-input": _pin(
        "binary_cross_entropy_with_logits",
        lambda fw, x, m: _F(fw).binary_cross_entropy_with_logits(x.float(), x),
        kind="guard",
    ),
    "cross_entropy": _pin(
        "cross_entropy", lambda fw, x, m: _F(fw).cross_entropy(m, x[:2].long())
    ),
    "cross_entropy-probabilities": _pin(
        "cross_entropy",
        lambda fw, x, m: _F(fw).cross_entropy(m.float(), m),
        kind="guard",
    ),
    "nll_loss": _pin("nll_loss", lambda fw, x, m: _F(fw).nll_loss(m, x[:2].long())),
    "ctc_loss": _pin(
        "ctc_loss",
        lambda fw, x, m: _F(fw).ctc_loss(
            m.reshape(3, 1, 2),
            fw.tensor([[1]], dtype=fw.int64),
            fw.tensor([3], dtype=fw.int64),
            fw.tensor([1], dtype=fw.int64),
        ),
    ),
    "kl_div": _pin("kl_div", lambda fw, x, m: _F(fw).kl_div(x, x)),
    "kl_div-float-target": _pin(
        "kl_div", lambda fw, x, m: _F(fw).kl_div(x.float(), x), kind="guard"
    ),
    "soft_margin_loss": _pin(
        "soft_margin_loss", lambda fw, x, m: _F(fw).soft_margin_loss(x, x)
    ),
    "gaussian_nll_loss": _pin(
        "gaussian_nll_loss", lambda fw, x, m: _F(fw).gaussian_nll_loss(x, x, x)
    ),
    "gaussian_nll_loss-var": _pin(
        "gaussian_nll_loss",
        lambda fw, x, m: _F(fw).gaussian_nll_loss(x.float(), x.float(), x),
    ),
    "gaussian_nll_loss-integer-operands": _pin(
        "gaussian_nll_loss",
        lambda fw, x, m: _F(fw).gaussian_nll_loss(x, x, x.float()),
        refused=frozenset({"bool"}),
    ),
    "hinge_embedding_loss": _pin(
        "hinge_embedding_loss", lambda fw, x, m: _F(fw).hinge_embedding_loss(x, x)
    ),
    "margin_ranking_loss": _pin(
        "margin_ranking_loss", lambda fw, x, m: _F(fw).margin_ranking_loss(x, x, x)
    ),
    "multi_margin_loss": _pin(
        "multi_margin_loss", lambda fw, x, m: _F(fw).multi_margin_loss(m, x[:2].long())
    ),
    "multilabel_margin_loss": _pin(
        "multilabel_margin_loss", lambda fw, x, m: _F(fw).multilabel_margin_loss(m, m.long())
    ),
    "multilabel_soft_margin_loss": _pin(
        "multilabel_soft_margin_loss", lambda fw, x, m: _F(fw).multilabel_soft_margin_loss(m, m)
    ),
    "poisson_nll_loss": _pin(
        "poisson_nll_loss", lambda fw, x, m: _F(fw).poisson_nll_loss(x, x)
    ),
    "triplet_margin_loss": _pin(
        "triplet_margin_loss", lambda fw, x, m: _F(fw).triplet_margin_loss(m, m, m)
    ),
    "triplet_margin_with_distance_loss": _pin(
        "triplet_margin_with_distance_loss",
        lambda fw, x, m: _F(fw).triplet_margin_with_distance_loss(m, m, m),
    ),
}

#: The ops (and, for a role, ``(op, role)``) the pins above name, for the files that must
#: tell what is pinned here from what is not.
PINNED: frozenset[tuple[str, str]] = frozenset((pin.op, pin.role) for pin in _PINS.values())
PINNED_OPS: frozenset[str] = frozenset(op for op, _ in PINNED)

_VECTOR = [1, 0, 1, 1]
_MATRIX = [[1, 0, 1], [0, 0, 1]]


def _dtype_of(fw: ModuleType, name: str) -> object:
    return getattr(fw, "bool_" if fw is lucid and name == "bool" else name)


def _refused_dtypes(pin: _Pin) -> frozenset[str]:
    """The dtypes ``pin``'s call is refused for: its own say, or its row's."""
    if pin.refused is not None:
        return pin.refused
    rules = ROWS[(pin.op, pin.role)].by_code()
    return frozenset(name for name, rule in zip(_DTYPES, rules, strict=True) if rule is not None)


def _kind_of(pin: _Pin, dtype: str) -> str:
    """The class the reference raises for ``pin`` on ``dtype``."""
    if isinstance(pin.kind, str):
        return pin.kind
    if pin.kind is not None and dtype in pin.kind:
        return pin.kind[dtype]
    rule = ROWS[(pin.op, pin.role)].by_code()[_DTYPES.index(dtype)]
    assert rule is not None, (pin.op, pin.role, dtype)
    return _LABEL[rule.error]


def _lucid_matches(kind: str, error: BaseException) -> bool:
    """Whether ``error`` is a Lucid refusal of class ``kind``."""
    if not isinstance(error, RuntimeError):
        return False
    if kind == "kernel":
        return isinstance(error, NotImplementedError)
    if kind == "value":
        return isinstance(error, ValueError)
    return not isinstance(error, NotImplementedError)


def _reference_matches(kind: str, error: BaseException) -> bool:
    """Whether ``error`` is the reference's refusal of class ``kind``."""
    if kind == "kernel":
        return isinstance(error, NotImplementedError)
    if kind == "value":
        return isinstance(error, ValueError)
    return isinstance(error, RuntimeError) and not isinstance(error, NotImplementedError)


_CASES = [(name, dtype) for name in _PINS for dtype in _DTYPES]


@pytest.mark.parametrize(("name", "dtype"), _CASES)
def test_lucid_does_what_the_table_says(device: str, name: str, dtype: str) -> None:
    pin = _PINS[name]
    x = lucid.tensor(_VECTOR, dtype=_dtype_of(lucid, dtype), device=device)  # type: ignore[arg-type]
    m = lucid.tensor(_MATRIX, dtype=_dtype_of(lucid, dtype), device=device)  # type: ignore[arg-type]
    if dtype not in _refused_dtypes(pin):
        assert pin.call(lucid, x, m) is not None
        return
    kind = _kind_of(pin, dtype)
    with pytest.raises(RuntimeError) as caught:
        pin.call(lucid, x, m)
    assert _lucid_matches(kind, caught.value), (kind, type(caught.value), str(caught.value))


@pytest.mark.parity
@pytest.mark.parametrize(("name", "dtype"), _CASES)
def test_reference_does_what_the_table_says(ref: ModuleType, name: str, dtype: str) -> None:
    pin = _PINS[name]
    x = ref.tensor(_VECTOR, dtype=_dtype_of(ref, dtype))
    m = ref.tensor(_MATRIX, dtype=_dtype_of(ref, dtype))
    if dtype not in _refused_dtypes(pin):
        assert pin.call(ref, x, m) is not None
        return
    kind = _kind_of(pin, dtype)
    with pytest.raises(Exception) as caught:
        pin.call(ref, x, m)
    assert _reference_matches(kind, caught.value), (kind, type(caught.value), str(caught.value))


# ── the table is pinned, and well formed ────────────────────────────────────────────────────────


def test_every_row_of_the_table_has_a_pin_and_every_pin_a_row() -> None:
    assert PINNED == set(ROWS), (
        f"rows with no pin: {sorted(set(ROWS) - PINNED)}; "
        f"pins with no row: {sorted(PINNED - set(ROWS))}"
    )


def test_a_pin_that_refuses_what_its_row_answers_says_so_on_purpose() -> None:
    # An override that restates the row hides a stale pin: it must differ from what the row says.
    for name, pin in _PINS.items():
        if pin.refused is None:
            continue
        row_refused = frozenset(
            dtype
            for dtype, rule in zip(_DTYPES, ROWS[(pin.op, pin.role)].by_code(), strict=True)
            if rule is not None
        )
        assert pin.refused != row_refused, f"{name} restates its row"


def test_the_rows_are_the_bool_table_the_old_consumers_read() -> None:
    primary = {op for (op, role), row in ROWS.items() if role == "" and row.on_bool is not None}
    assert set(BOOL_REFUSED) == primary
    for op in BOOL_REFUSED:
        assert BOOL_REFUSED[op] == ROWS[(op, "")].on_bool


def test_an_op_has_one_owner_per_dtype() -> None:
    # The engine owns the dtypes its schema declares and Python the rest; ``grid_sample`` is the
    # one op both refuse, because its entry would promote the operand before the engine looks.
    engine = _table.engine_refusals()
    both: set[tuple[str, str]] = set()
    for (op, role), row in ROWS.items():
        if role != "" or op not in engine:
            continue
        for dtype, mine, theirs in zip(_DTYPES, row.by_code(), engine[op], strict=True):
            if mine is not None and theirs is not None:
                both.add((op, dtype))
    assert both == {("grid_sample", dtype) for dtype in _DTYPES}
    for dtype, mine, theirs in zip(
        _DTYPES, ROWS[("grid_sample", "")].by_code(), engine["grid_sample"], strict=True
    ):
        assert mine == theirs, dtype


def test_a_rule_names_the_cast_of_its_class() -> None:
    hints = {rule.hint for row in ROWS.values() for rule in row.by_code() if rule is not None}
    engine_hints = {
        rule.hint
        for rules in _table.engine_refusals().values()
        for rule in rules
        if rule is not None
    }
    assert engine_hints <= {
        _table._FLOATING,
        _table._INTEGER,
        _table._NUMBER,
    }
    assert hints <= {
        _table._FLOATING,
        _table._INTEGER,
        _table._NUMBER,
        _table._INT64,
        _table._BOOL_MASK,
    }


def test_a_row_sets_one_integer_slot() -> None:
    rule = Rule(_table._KERNEL, _table._FLOATING)
    with pytest.raises(ValueError, match="not both"):
        _table.Row("x", on_ints=rule, on_narrow=rule)


def test_a_dtype_is_refused_by_its_width() -> None:
    # ``local_response_norm`` answers int64 and refuses the narrower integers and bool.
    rules = ROWS[("local_response_norm", "")].by_code()
    assert [rule is not None for rule in rules] == [True, True, True, True, False]
    # ``one_hot`` is the same by width, with its own cast.
    assert {rule.hint for rule in ROWS[("one_hot", "")].by_code() if rule} == {_table._INT64}


def test_a_role_with_no_row_reads_the_primary_one() -> None:
    error = _table.refusal("huber_loss", 1, "target")
    assert isinstance(error, _C_engine.NotImplementedError)
    assert "an int8 target" in str(error)


def test_an_in_place_twin_takes_the_row_of_its_op() -> None:
    assert isinstance(_table.refusal("pow_", 0), _C_engine.NotImplementedError)
    assert str(_table.refusal("celu_", 3)).startswith("celu_ does not accept int32")


def test_an_integer_is_told_to_cast_where_its_row_says() -> None:
    assert str(_table.refusal("quantile", 3)) == (
        "quantile does not accept int32 — cast to a floating dtype, e.g. x.float()"
    )
    assert str(_table.refusal("cross_entropy", 4, "target", index=True)).endswith(
        "cast to an integer dtype, e.g. x.long()"
    )
    assert "a bool target" in str(_table.refusal("cross_entropy", 0, "target"))


def test_a_refusal_the_table_does_not_hold_is_a_lookup_error() -> None:
    with pytest.raises(KeyError, match="no refusal row"):
        _table.refusal("not_an_op", 0)
    with pytest.raises(KeyError, match="does not refuse"):
        _table.refusal("cov", 3)


def test_a_float_pays_for_no_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    # The check runs on every float tensor of every op that carries it; it must not reach the
    # table for one.
    def boom(*args: object) -> None:
        raise AssertionError("a float input reached the table")

    monkeypatch.setattr(_table, "_rule", boom)
    x = lucid.tensor([1.0, 2.0])
    _table.refuse("quantile", x)
    _table.refuse("quantile", x, "weight")
    _table.refuse("quantile", 3.0)
    _table.refuse("quantile", None)
    _table.refuse_common("gammainc", x, x)
    _table.refuse_mixed("prelu", x, x)
    _table.refuse_declared("embedding_bag", x)
    _table.refuse_bool("cov", x)


def test_what_the_reference_answers_is_not_refused() -> None:
    # ``refuse`` answers an operand its row has no rule for.
    ints = lucid.tensor([1, 2])
    _table.refuse("cov", ints)
    _table.refuse("masked_select", lucid.tensor([True, False]), "mask")
    _table.refuse_common("gammainc", ints, lucid.tensor([1.0, 2.0]))
    _table.refuse_common("searchsorted", ints, lucid.tensor([True, False]))
    with pytest.raises(NotImplementedError):
        _table.refuse_common("gammainc", ints, lucid.tensor([True, False]))
    with pytest.raises(NotImplementedError):
        _table.refuse_common("searchsorted", lucid.tensor([True]), lucid.tensor([False]))


def test_a_refused_in_place_celu_leaves_the_tensor_as_it_was(device: str) -> None:
    x = lucid.tensor(_VECTOR, dtype=lucid.int32, device=device)
    version = x._impl.version
    with pytest.raises(NotImplementedError, match="celu does not accept int32"):
        lucid.nn.functional.celu_(x)
    assert x.tolist() == _VECTOR
    assert x._impl.version == version


# ── what the reference passes through is passed through ─────────────────────────────────────────

#: ``feature_alpha_dropout`` draws a channel mask only when something is dropped; at eval, or at
#: ``p == 0``, the reference hands the input back whatever its dtype, so the refusal stays on the
#: path that draws (the engine's dropout rows sit inside ``training && p > 0`` for the same reason).
_NOTHING_DROPPED = {"eval": {"p": 0.5, "training": False}, "p0": {"p": 0.0, "training": True}}


@pytest.mark.parametrize("case", list(_NOTHING_DROPPED))
@pytest.mark.parametrize("dtype", _DTYPES)
def test_feature_alpha_dropout_returns_an_input_nothing_is_dropped_from(
    device: str, dtype: str, case: str
) -> None:
    x = lucid.tensor([[[1, 0], [1, 1]]], dtype=_dtype_of(lucid, dtype), device=device)  # type: ignore[arg-type]
    out = lucid.nn.functional.feature_alpha_dropout(x, **_NOTHING_DROPPED[case])
    assert out.dtype == x.dtype
    assert out.tolist() == x.tolist()


@pytest.mark.parametrize("dtype", _DTYPES)
def test_feature_alpha_dropout_refuses_what_it_drops_from(device: str, dtype: str) -> None:
    x = lucid.tensor([[[1, 0], [1, 1]]], dtype=_dtype_of(lucid, dtype), device=device)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match=f"does not accept {dtype}") as caught:
        lucid.nn.functional.feature_alpha_dropout(x, p=0.5, training=True)
    assert not isinstance(caught.value, NotImplementedError)


@pytest.mark.parity
@pytest.mark.parametrize("case", list(_NOTHING_DROPPED))
@pytest.mark.parametrize("dtype", _DTYPES)
def test_the_reference_returns_what_it_drops_nothing_from(
    ref: ModuleType, dtype: str, case: str
) -> None:
    x = ref.tensor([[[1, 0], [1, 1]]], dtype=_dtype_of(ref, dtype))
    out = ref.nn.functional.feature_alpha_dropout(x, **_NOTHING_DROPPED[case])
    assert out.dtype == x.dtype
    assert out.tolist() == x.tolist()


# ── a mask is a bool, and nothing else ──────────────────────────────────────────────────────────

_NOT_BOOL = ("int8", "int16", "int32", "int64", "float16", "float32", "float64")


@pytest.mark.parametrize("dtype", _NOT_BOOL)
def test_masked_select_refuses_a_mask_of_any_dtype_but_bool(device: str, dtype: str) -> None:
    if device == "metal" and dtype == "float64":
        pytest.skip("Metal holds no float64")
    values = lucid.tensor([1.0, 2.0, 3.0], device=device)
    mask = lucid.tensor([1, 0, 1], dtype=_dtype_of(lucid, dtype), device=device)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match=f"does not accept (an|a) {dtype} mask") as caught:
        lucid.masked_select(values, mask)
    assert not isinstance(caught.value, NotImplementedError)
    assert lucid.masked_select(values, mask.bool()).tolist() == [1.0, 3.0]


@pytest.mark.parametrize("dtype", _NOT_BOOL)
def test_the_method_and_the_function_refuse_a_mask_alike(dtype: str) -> None:
    values = lucid.tensor([1.0, 2.0, 3.0])
    mask = lucid.tensor([1, 0, 1], dtype=_dtype_of(lucid, dtype))  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="does not accept"):
        values.masked_select(mask)


@pytest.mark.parity
@pytest.mark.parametrize("dtype", _NOT_BOOL)
def test_the_reference_refuses_a_mask_of_any_dtype_but_bool(ref: ModuleType, dtype: str) -> None:
    values = ref.tensor([1.0, 2.0, 3.0])
    mask = ref.tensor([1, 0, 1], dtype=_dtype_of(ref, dtype))
    with pytest.raises(RuntimeError) as caught:
        ref.masked_select(values, mask)
    assert not isinstance(caught.value, NotImplementedError)
