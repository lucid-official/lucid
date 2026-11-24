"""The calls that pin each row of the Python refusal table (LCD-1026).

``lucid._ops._bool_refusal.ROWS`` says, per ``(op, role)``, which of bool and the integers an
entry turns away.  A *pin* is one call of that op, written once for both frameworks, with what the
table says of it: the dtypes it refuses and the class.  ``test_integer_refusal_table.py`` replays
every pin on Lucid and on the reference; the bool table test reads :data:`PINNED_OPS` to know which
ops that file already holds, so the data lives here and neither test module imports the other.

A pin with no row is an op the engine's schema owns that the Python entry asks about itself
(``grid_sample``'s image): it names the dtypes and the class, since there is no row to read them
from.
"""

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

DTYPES = ("bool", "int8", "int16", "int32", "int64")
ALL = frozenset(DTYPES)
NONE: frozenset[str] = frozenset()

# ``fw`` is the framework under test, ``x`` a vector and ``m`` a 2x3 matrix of the dtype being
# probed.  Both frameworks spell every call below the same way.
Call = Callable[[ModuleType, object, object], object]


@dataclass(frozen=True)
class Pin:
    """One call of an op, and what the table says of it."""

    op: str
    role: str
    call: Call
    #: The class when it is not the row's rule (the row's ``InvalidArgument`` is a ``value`` for
    #: ``affine_grid`` and a ``guard`` for ``one_hot``), or by dtype where it varies with it.
    kind: str | dict[str, str] | None = None
    #: The dtypes this call refuses, when it is not what the row says (a loss beside a float).
    refused: frozenset[str] | None = None


def pin(
    op: str,
    call: Call,
    *,
    role: str = "",
    kind: str | dict[str, str] | None = None,
    refused: frozenset[str] | None = None,
) -> Pin:
    return Pin(op, role, call, kind, refused)


def functional(fw: ModuleType) -> ModuleType:
    """``fw.nn.functional`` of either framework."""
    module: ModuleType = fw.nn.functional
    return module


PINS: dict[str, Pin] = {
    # ── order statistics, counting and membership ───────────────────────────────────────────────
    "quantile": pin("quantile", lambda fw, x, m: fw.quantile(x, 0.5)),
    "nanquantile": pin("nanquantile", lambda fw, x, m: fw.nanquantile(x, 0.5)),
    "searchsorted": pin("searchsorted", lambda fw, x, m: fw.searchsorted(x, x[:1])),
    "bucketize": pin("bucketize", lambda fw, x, m: fw.bucketize(x, x)),
    "isin": pin("isin", lambda fw, x, m: fw.isin(x, x)),
    "histogram": pin("histogram", lambda fw, x, m: fw.histogram(x, 2)),
    "histogram-bins": pin("histogram", lambda fw, x, m: fw.histogram(x.float(), x), role="bins"),
    "histogram-weight": pin(
        "histogram", lambda fw, x, m: fw.histogram(x.float(), 2, weight=x), role="weight"
    ),
    "histogramdd": pin("histogramdd", lambda fw, x, m: fw.histogramdd(m, 2)),
    "poisson": pin("poisson", lambda fw, x, m: fw.poisson(x)),
    "multinomial": pin("multinomial", lambda fw, x, m: fw.multinomial(x, 1)),
    # ── moments ─────────────────────────────────────────────────────────────────────────────────
    "std_mean": pin("std_mean", lambda fw, x, m: fw.std_mean(x)),
    "var_mean": pin("var_mean", lambda fw, x, m: fw.var_mean(x)),
    "nanmean": pin("nanmean", lambda fw, x, m: fw.nanmean(x)),
    "cov": pin("cov", lambda fw, x, m: fw.cov(m)),
    "corrcoef": pin("corrcoef", lambda fw, x, m: fw.corrcoef(m)),
    "gradient": pin("gradient", lambda fw, x, m: fw.gradient(x)),
    # ── elementwise ─────────────────────────────────────────────────────────────────────────────
    "clamp": pin("clamp", lambda fw, x, m: fw.clamp(x, x, x)),
    "pow": pin("pow", lambda fw, x, m: fw.pow(x, x)),
    "frexp": pin("frexp", lambda fw, x, m: fw.frexp(x)),
    "polar": pin("polar", lambda fw, x, m: fw.polar(x, x)),
    "polar-first": pin("polar", lambda fw, x, m: fw.polar(x, x.float())),
    "polar-second": pin("polar", lambda fw, x, m: fw.polar(x.float(), x)),
    # ── special functions and linear algebra ────────────────────────────────────────────────────
    # A pair of integers meets no floating operand and has no kernel; an integer beside a float
    # promotes and is answered.
    "gammainc": pin("gammainc", lambda fw, x, m: fw.special.gammainc(x, x)),
    "gammainc-float": pin(
        "gammainc", lambda fw, x, m: fw.special.gammainc(x, x.float() + 1.0), refused=NONE
    ),
    "gammainc-float-first": pin(
        "gammainc", lambda fw, x, m: fw.special.gammainc(x.float() + 1.0, x), refused=NONE
    ),
    "gammainc-integer-pair": pin("gammainc", lambda fw, x, m: fw.special.gammainc(x, x.long())),
    "gammaincc": pin("gammaincc", lambda fw, x, m: fw.special.gammaincc(x, x)),
    "gammaincc-float": pin(
        "gammaincc", lambda fw, x, m: fw.special.gammaincc(x.float() + 1.0, x), refused=NONE
    ),
    "multigammaln": pin("multigammaln", lambda fw, x, m: fw.special.multigammaln(x, 1)),
    "vecdot": pin("vecdot", lambda fw, x, m: fw.linalg.vecdot(x, x)),
    "vecdot-first": pin("vecdot", lambda fw, x, m: fw.linalg.vecdot(x, x.float())),
    "vecdot-second": pin("vecdot", lambda fw, x, m: fw.linalg.vecdot(x.float(), x)),
    "vander": pin("vander", lambda fw, x, m: fw.linalg.vander(x)),
    # ``householder_product`` has an engine route (one matrix, a reflector per column) and a
    # composed one (a batch, fewer reflectors): the reference has no kernel for an integral
    # dtype on any of them, so every route is a pin.
    "householder_product": pin(
        "householder_product", lambda fw, x, m: fw.linalg.householder_product(m.mT, x[:2])
    ),
    "householder_product-batched": pin(
        "householder_product",
        lambda fw, x, m: fw.linalg.householder_product(m.mT.unsqueeze(0), x[:2].unsqueeze(0)),
    ),
    "householder_product-fewer": pin(
        "householder_product", lambda fw, x, m: fw.linalg.householder_product(m.mT, x[:1])
    ),
    "householder_product-none": pin(
        "householder_product", lambda fw, x, m: fw.linalg.householder_product(m.mT, x[:0])
    ),
    # An integer pair of 3-vectors is answered; a bool pair is not.
    "cross": pin("cross", lambda fw, x, m: fw.linalg.cross(m[0], m[1])),
    # The integers of ``solve`` are the engine's schema (``refuse_declared``), the bool this table's:
    # every dtype is refused, and the reference judges ``A``'s dtype before ``B``'s and before
    # any shape.
    "solve": pin(
        "solve", lambda fw, x, m: fw.linalg.solve(m[:, :2], m[:, :2]), kind="guard", refused=ALL
    ),
    "solve-float-rhs": pin(
        "solve",
        lambda fw, x, m: fw.linalg.solve(m[:, :2], m[:, :2].float()),
        kind="guard",
        refused=ALL,
    ),
    "solve-not-square": pin(
        "solve", lambda fw, x, m: fw.linalg.solve(m, m), kind="guard", refused=ALL
    ),
    "solve-left-false-vector": pin(
        "solve",
        lambda fw, x, m: fw.linalg.solve(m[:, :2], x[:2], left=False),
        kind="guard",
        refused=ALL,
    ),
    # An empty matrix is its own inverse and never reaches the engine; the reference refuses an
    # integral one all the same.
    "inv-empty": pin("inv", lambda fw, x, m: fw.linalg.inv(m[:0, :0]), kind="guard", refused=ALL),
    # The ``hermitian`` route of both goes through ``eigh``, a kernel: the reference refuses it with
    # its own guard first, on every route.
    "pinv": pin("pinv", lambda fw, x, m: fw.linalg.pinv(m)),
    "pinv-hermitian": pin("pinv", lambda fw, x, m: fw.linalg.pinv(m[:, :2], hermitian=True)),
    "matrix_rank": pin("matrix_rank", lambda fw, x, m: fw.linalg.matrix_rank(m)),
    "matrix_rank-hermitian": pin(
        "matrix_rank", lambda fw, x, m: fw.linalg.matrix_rank(m[:, :2], hermitian=True)
    ),
    "matrix_exp": pin("matrix_exp", lambda fw, x, m: fw.linalg.matrix_exp(m[:, :2])),
    "ldl_solve": pin(
        "ldl_solve",
        lambda fw, x, m: fw.linalg.ldl_solve(
            m[:, :2], fw.tensor([1, 2], dtype=fw.int32), fw.tensor([[1.0], [2.0]])
        ),
    ),
    # ── activations, dropout and normalisation composed from other ops ──────────────────────────
    "celu": pin("celu", lambda fw, x, m: functional(fw).celu(x)),
    "celu-inplace": pin("celu", lambda fw, x, m: functional(fw).celu_(x.clone())),
    "hardshrink": pin("hardshrink", lambda fw, x, m: functional(fw).hardshrink(x)),
    "softshrink": pin("softshrink", lambda fw, x, m: functional(fw).softshrink(x)),
    "tanhshrink": pin("tanhshrink", lambda fw, x, m: functional(fw).tanhshrink(x)),
    "hardtanh": pin("hardtanh", lambda fw, x, m: functional(fw).hardtanh(x)),
    "threshold": pin("threshold", lambda fw, x, m: functional(fw).threshold(x, 0.1, 20.0)),
    "rrelu": pin("rrelu", lambda fw, x, m: functional(fw).rrelu(x)),
    "rrelu-training": pin("rrelu", lambda fw, x, m: functional(fw).rrelu(x, training=True)),
    "prelu": pin("prelu", lambda fw, x, m: functional(fw).prelu(x, x[:1])),
    "prelu-weight": pin(
        "prelu", lambda fw, x, m: functional(fw).prelu(x.float(), x[:1]), role="mixed"
    ),
    "prelu-input": pin(
        "prelu", lambda fw, x, m: functional(fw).prelu(x, x[:1].float()), role="mixed"
    ),
    "glu": pin("glu", lambda fw, x, m: functional(fw).glu(x)),
    "gumbel_softmax": pin("gumbel_softmax", lambda fw, x, m: functional(fw).gumbel_softmax(x)),
    "feature_alpha_dropout": pin(
        "feature_alpha_dropout",
        lambda fw, x, m: functional(fw).feature_alpha_dropout(
            x.reshape(1, 2, 2), p=0.5, training=True
        ),
    ),
    # Its integers are the engine's (the schema declares them), its bool is the entry's.
    "rms_norm": pin(
        "rms_norm", lambda fw, x, m: functional(fw).rms_norm(x, (4,)), kind="kernel", refused=ALL
    ),
    "rms_norm-eps": pin(
        "rms_norm",
        lambda fw, x, m: functional(fw).rms_norm(x, (4,), eps=1e-5),
        kind="kernel",
        refused=ALL,
    ),
    "instance_norm": pin(
        "instance_norm", lambda fw, x, m: functional(fw).instance_norm(m.reshape(1, 2, 3))
    ),
    "local_response_norm": pin(
        "local_response_norm",
        lambda fw, x, m: functional(fw).local_response_norm(m.reshape(1, 2, 3), 2),
    ),
    # ── pooling ─────────────────────────────────────────────────────────────────────────────────
    "max_unpool1d": pin(
        "max_unpool1d",
        lambda fw, x, m: functional(fw).max_unpool1d(
            x[:2].reshape(1, 1, 2), fw.tensor([[[0, 2]]], dtype=fw.int64), 2
        ),
    ),
    "max_unpool2d": pin(
        "max_unpool2d",
        lambda fw, x, m: functional(fw).max_unpool2d(
            x.reshape(1, 1, 2, 2), fw.tensor([[[[0, 5], [10, 15]]]], dtype=fw.int64), 2
        ),
    ),
    "max_unpool3d": pin(
        "max_unpool3d",
        lambda fw, x, m: functional(fw).max_unpool3d(
            x.reshape(1, 1, 1, 2, 2), fw.tensor([[[[[0, 5], [10, 15]]]]], dtype=fw.int64), 2
        ),
    ),
    "adaptive_avg_pool1d": pin(
        "adaptive_avg_pool1d",
        lambda fw, x, m: functional(fw).adaptive_avg_pool1d(x.reshape(1, 1, 4), 2),
    ),
    "adaptive_avg_pool1d-global": pin(
        "adaptive_avg_pool1d",
        lambda fw, x, m: functional(fw).adaptive_avg_pool1d(x.reshape(1, 1, 4), 1),
        role="global",
    ),
    "adaptive_avg_pool2d": pin(
        "adaptive_avg_pool2d",
        lambda fw, x, m: functional(fw).adaptive_avg_pool2d(m.reshape(1, 1, 2, 3), (1, 2)),
    ),
    "adaptive_avg_pool2d-global": pin(
        "adaptive_avg_pool2d",
        lambda fw, x, m: functional(fw).adaptive_avg_pool2d(m.reshape(1, 1, 2, 3), 1),
        role="global",
    ),
    "adaptive_avg_pool3d": pin(
        "adaptive_avg_pool3d",
        lambda fw, x, m: functional(fw).adaptive_avg_pool3d(x.reshape(1, 1, 1, 2, 2), (1, 1, 2)),
    ),
    "adaptive_avg_pool3d-global": pin(
        "adaptive_avg_pool3d",
        lambda fw, x, m: functional(fw).adaptive_avg_pool3d(x.reshape(1, 1, 1, 2, 2), 1),
        role="global",
    ),
    "adaptive_max_pool1d": pin(
        "adaptive_max_pool1d",
        lambda fw, x, m: functional(fw).adaptive_max_pool1d(x.reshape(1, 1, 4), 2),
    ),
    "adaptive_max_pool2d": pin(
        "adaptive_max_pool2d",
        lambda fw, x, m: functional(fw).adaptive_max_pool2d(m.reshape(1, 1, 2, 3), 1),
    ),
    "adaptive_max_pool2d-indices": pin(
        "adaptive_max_pool2d",
        lambda fw, x, m: functional(fw).adaptive_max_pool2d(
            m.reshape(1, 1, 2, 3), 1, return_indices=True
        ),
    ),
    "adaptive_max_pool3d": pin(
        "adaptive_max_pool3d",
        lambda fw, x, m: functional(fw).adaptive_max_pool3d(x.reshape(1, 1, 1, 2, 2), 1),
    ),
    # ── initialisers and samplers drawn through a float tensor ──────────────────────────────────
    "orthogonal": pin("orthogonal", lambda fw, x, m: fw.nn.init.orthogonal_(m)),
    "sparse": pin("sparse", lambda fw, x, m: fw.nn.init.sparse_(m, 0.5)),
    "cauchy": pin("cauchy", lambda fw, x, m: x.cauchy_()),
    "exponential": pin("exponential", lambda fw, x, m: x.exponential_()),
    # ── indices and masks ───────────────────────────────────────────────────────────────────────
    # The reference wants int64 class indices: a plain RuntimeError for the rest.
    "one_hot": pin("one_hot", lambda fw, x, m: functional(fw).one_hot(x, 2), kind="guard"),
    # ``take_along_dim`` reads its positions as int64 when it is given a dim (flattened it is a
    # ``gather``, which takes int32 too).
    "take_along_dim-indices": pin(
        "take_along_dim", lambda fw, x, m: fw.take_along_dim(x.float(), x, 0), role="indices"
    ),
    # ``take`` reads int64 too, ``gather`` int32 or int64 (so an int32 index answers there).
    "take-index": pin("take", lambda fw, x, m: fw.take(x.float(), x), role="index"),
    "gather-index": pin("gather", lambda fw, x, m: fw.gather(x.float(), 0, x), role="index"),
    # ``ldexp`` of a real float has no kernel for a bool exponent; an integral exponent answers.
    "ldexp-exponent": pin("ldexp", lambda fw, x, m: fw.ldexp(x.float(), x), role="exponent"),
    "masked_select-mask": pin(
        "masked_select", lambda fw, x, m: fw.masked_select(x.float(), x), role="mask"
    ),
    "masked_select-input": pin(
        "masked_select",
        lambda fw, x, m: fw.masked_select(x, x.float() > 0.5),
        role="mask",
        refused=NONE,
    ),
    "attn_mask": pin(
        "scaled_dot_product_attention",
        lambda fw, x, m: functional(fw).scaled_dot_product_attention(
            m.float().reshape(1, 2, 3),
            m.float().reshape(1, 2, 3),
            m.float().reshape(1, 2, 3),
            attn_mask=x.reshape(2, 2),
        ),
        role="attn_mask",
    ),
    # An integer table: the engine op declares it for the one path that reaches it, the entry
    # for the rest; a bool one is the entry's.
    "embedding_bag": pin(
        "embedding_bag",
        lambda fw, x, m: functional(fw).embedding_bag(
            fw.tensor([0, 1], dtype=fw.int64), m[:, :2], fw.tensor([0], dtype=fw.int64)
        ),
        kind="guard",
        refused=ALL,
    ),
    "embedding_bag-weighted": pin(
        "embedding_bag",
        lambda fw, x, m: functional(fw).embedding_bag(
            fw.tensor([0, 1], dtype=fw.int64),
            m[:, :2],
            fw.tensor([0], dtype=fw.int64),
            per_sample_weights=fw.tensor([1.0, 2.0]),
            mode="sum",
        ),
        kind="guard",
        refused=ALL,
    ),
    # ── resampling ──────────────────────────────────────────────────────────────────────────────
    "interpolate": pin(
        "interpolate",
        lambda fw, x, m: functional(fw).interpolate(m.reshape(1, 2, 3), scale_factor=2),
    ),
    "interpolate-bilinear": pin(
        "interpolate",
        lambda fw, x, m: functional(fw).interpolate(
            m.reshape(1, 1, 2, 3), scale_factor=2, mode="bilinear"
        ),
    ),
    "interpolate-area": pin(
        "interpolate",
        lambda fw, x, m: functional(fw).interpolate(m.reshape(1, 2, 3), size=2, mode="area"),
    ),
    "upsample": pin(
        "interpolate",
        lambda fw, x, m: functional(fw).upsample(m.reshape(1, 1, 2, 3), scale_factor=2),
    ),
    # The image is the engine schema's row, asked of it by the entry (``refuse_declared``): no
    # Python row, so the pin says what it refuses.
    "grid_sample": pin(
        "grid_sample",
        lambda fw, x, m: functional(fw).grid_sample(
            m.reshape(1, 1, 2, 3), m.float().reshape(1, 1, 3, 2) * 0.5, align_corners=False
        ),
        kind="kernel",
        refused=ALL,
    ),
    "grid_sample-grid": pin(
        "grid_sample",
        lambda fw, x, m: functional(fw).grid_sample(
            m.float().reshape(1, 1, 2, 3), m.reshape(1, 1, 3, 2), align_corners=False
        ),
        role="grid",
    ),
    "affine_grid": pin(
        "affine_grid",
        lambda fw, x, m: functional(fw).affine_grid(
            m.reshape(1, 2, 3), [1, 1, 2, 2], align_corners=False
        ),
        kind="value",
    ),
    # ── losses ──────────────────────────────────────────────────────────────────────────────────
    # An integer pair has no kernel; beside a floating operand the reference answers it
    # (``mse_loss``) or reads it as a dtype mismatch (``huber_loss``, the BCE losses).
    "mse_loss": pin("mse_loss", lambda fw, x, m: functional(fw).mse_loss(x, x)),
    "mse_loss-pair-of-widths": pin(
        "mse_loss", lambda fw, x, m: functional(fw).mse_loss(x, x.long())
    ),
    "mse_loss-float-target": pin(
        "mse_loss", lambda fw, x, m: functional(fw).mse_loss(x, x.float()), refused=NONE
    ),
    "mse_loss-float-input": pin(
        "mse_loss", lambda fw, x, m: functional(fw).mse_loss(x.float(), x), refused=NONE
    ),
    "l1_loss": pin("l1_loss", lambda fw, x, m: functional(fw).l1_loss(x, x)),
    "l1_loss-float-target": pin(
        "l1_loss",
        lambda fw, x, m: functional(fw).l1_loss(x, x.float()),
        refused=frozenset({"bool"}),
    ),
    "l1_loss-float-input": pin(
        "l1_loss",
        lambda fw, x, m: functional(fw).l1_loss(x.float(), x),
        refused=frozenset({"bool"}),
    ),
    "smooth_l1_loss": pin("smooth_l1_loss", lambda fw, x, m: functional(fw).smooth_l1_loss(x, x)),
    "smooth_l1_loss-float-target": pin(
        "smooth_l1_loss", lambda fw, x, m: functional(fw).smooth_l1_loss(x, x.float()), refused=NONE
    ),
    "huber_loss": pin("huber_loss", lambda fw, x, m: functional(fw).huber_loss(x, x)),
    "huber_loss-pair-of-widths": pin(
        "huber_loss",
        lambda fw, x, m: functional(fw).huber_loss(x, x.long()),
        kind={"bool": "guard"},
    ),
    "huber_loss-float-target": pin(
        "huber_loss", lambda fw, x, m: functional(fw).huber_loss(x, x.float()), kind="guard"
    ),
    "huber_loss-float-input": pin(
        "huber_loss", lambda fw, x, m: functional(fw).huber_loss(x.float(), x), refused=NONE
    ),
    "binary_cross_entropy": pin(
        "binary_cross_entropy", lambda fw, x, m: functional(fw).binary_cross_entropy(x, x)
    ),
    "binary_cross_entropy-float-target": pin(
        "binary_cross_entropy",
        lambda fw, x, m: functional(fw).binary_cross_entropy(x, x.float()),
        kind="guard",
    ),
    "binary_cross_entropy-float-input": pin(
        "binary_cross_entropy",
        lambda fw, x, m: functional(fw).binary_cross_entropy(x.float() * 0.5, x),
        kind="guard",
    ),
    "binary_cross_entropy-pair-of-widths": pin(
        "binary_cross_entropy",
        lambda fw, x, m: functional(fw).binary_cross_entropy(x, x.long()),
        kind={
            "int64": "kernel",
            "bool": "guard",
            "int8": "guard",
            "int16": "guard",
            "int32": "guard",
        },
    ),
    "binary_cross_entropy_with_logits": pin(
        "binary_cross_entropy_with_logits",
        lambda fw, x, m: functional(fw).binary_cross_entropy_with_logits(x, x),
    ),
    "binary_cross_entropy_with_logits-float-target": pin(
        "binary_cross_entropy_with_logits",
        lambda fw, x, m: functional(fw).binary_cross_entropy_with_logits(x, x.float()),
    ),
    "binary_cross_entropy_with_logits-float-input": pin(
        "binary_cross_entropy_with_logits",
        lambda fw, x, m: functional(fw).binary_cross_entropy_with_logits(x.float(), x),
        kind="guard",
    ),
    "cross_entropy": pin(
        "cross_entropy", lambda fw, x, m: functional(fw).cross_entropy(m, x[:2].long())
    ),
    "cross_entropy-probabilities": pin(
        "cross_entropy", lambda fw, x, m: functional(fw).cross_entropy(m.float(), m), kind="guard"
    ),
    "nll_loss": pin("nll_loss", lambda fw, x, m: functional(fw).nll_loss(m, x[:2].long())),
    "ctc_loss": pin(
        "ctc_loss",
        lambda fw, x, m: functional(fw).ctc_loss(
            m.reshape(3, 1, 2),
            fw.tensor([[1]], dtype=fw.int64),
            fw.tensor([3], dtype=fw.int64),
            fw.tensor([1], dtype=fw.int64),
        ),
    ),
    "kl_div": pin("kl_div", lambda fw, x, m: functional(fw).kl_div(x, x)),
    "kl_div-float-target": pin(
        "kl_div", lambda fw, x, m: functional(fw).kl_div(x.float(), x), kind="guard"
    ),
    "soft_margin_loss": pin(
        "soft_margin_loss", lambda fw, x, m: functional(fw).soft_margin_loss(x, x)
    ),
    "gaussian_nll_loss": pin(
        "gaussian_nll_loss", lambda fw, x, m: functional(fw).gaussian_nll_loss(x, x, x)
    ),
    "gaussian_nll_loss-var": pin(
        "gaussian_nll_loss",
        lambda fw, x, m: functional(fw).gaussian_nll_loss(x.float(), x.float(), x),
    ),
    "gaussian_nll_loss-integer-operands": pin(
        "gaussian_nll_loss",
        lambda fw, x, m: functional(fw).gaussian_nll_loss(x, x, x.float()),
        refused=frozenset({"bool"}),
    ),
    "hinge_embedding_loss": pin(
        "hinge_embedding_loss", lambda fw, x, m: functional(fw).hinge_embedding_loss(x, x)
    ),
    "margin_ranking_loss": pin(
        "margin_ranking_loss", lambda fw, x, m: functional(fw).margin_ranking_loss(x, x, x)
    ),
    "multi_margin_loss": pin(
        "multi_margin_loss", lambda fw, x, m: functional(fw).multi_margin_loss(m, x[:2].long())
    ),
    "multilabel_margin_loss": pin(
        "multilabel_margin_loss",
        lambda fw, x, m: functional(fw).multilabel_margin_loss(m, m.long()),
    ),
    "multilabel_soft_margin_loss": pin(
        "multilabel_soft_margin_loss",
        lambda fw, x, m: functional(fw).multilabel_soft_margin_loss(m, m),
    ),
    "poisson_nll_loss": pin(
        "poisson_nll_loss", lambda fw, x, m: functional(fw).poisson_nll_loss(x, x)
    ),
    "triplet_margin_loss": pin(
        "triplet_margin_loss", lambda fw, x, m: functional(fw).triplet_margin_loss(m, m, m)
    ),
    "triplet_margin_with_distance_loss": pin(
        "triplet_margin_with_distance_loss",
        lambda fw, x, m: functional(fw).triplet_margin_with_distance_loss(m, m, m),
    ),
}

#: The ops (and, for a role, ``(op, role)``) the pins above name, for the files that must
#: tell what is pinned here from what is not.
PINNED: frozenset[tuple[str, str]] = frozenset((pin.op, pin.role) for pin in PINS.values())
PINNED_OPS: frozenset[str] = frozenset(op for op, _ in PINNED)


#: The ``(op, role)`` the pins above name, for the files that must tell what is pinned here from
#: what is not.
PINNED: frozenset[tuple[str, str]] = frozenset((p.op, p.role) for p in PINS.values())
PINNED_OPS: frozenset[str] = frozenset(op for op, _ in PINNED)
