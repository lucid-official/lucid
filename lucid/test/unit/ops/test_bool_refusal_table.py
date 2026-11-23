"""The ops that refuse a bool tensor, pinned op by op against the reference (LCD-1013).

The reference's CPU turns a bool operand away from a long list of ops where
Lucid used to promote the mask silently (``mask.mean()``, ``mask.abs()``,
``softmax(mask)``).  Two lists own those refusals, one per layer
(:mod:`lucid._ops._bool_refusal`): the ops with an engine schema are refused
by the engine (:func:`engine_bool_refused` reads them back), and the ops with
none by their Python entry (:data:`BOOL_REFUSED`).  This file is the test of
both:

* every row here names the op it exercises, and a final test fails when an op
  of either list has no row -- a refusal cannot be added and left unpinned,
  and an op cannot be dropped unnoticed;
* every row is replayed on the reference (``-m parity``), so a row that drifts
  from the reference's answer -- the verdict *or* the error class -- fails;
* a refusal is the same on both devices, whichever spelling reached it (free
  function, ``Tensor`` method, in-place twin, ``abs()``, ``nn.functional``,
  an ``nn`` module) and whether the call is traced by ``lucid.compile``.

Kinds, named after how the reference refuses:

``kernel``
    a ``NotImplementedError`` (a kernel with no bool instantiation).
``guard``
    a plain ``RuntimeError`` that is *not* a ``NotImplementedError`` (a check
    on the input dtype).
``refused``
    some ``RuntimeError``: the operand is refused for a reason of its own
    (a bool ``bins`` where the reference wants the input's dtype).
``accepted``
    answered, and answered with the reference's values: the spelling the
    reference offers in place of the refused one.

Every Lucid refusal is a ``RuntimeError``, as the reference's are.
"""

import operator
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._ops._bool_refusal import (
    _BOOL_MASK,
    _FLOATING,
    _INT64,
    _INTEGER,
    _NUMBER,
    BOOL_REFUSED,
    engine_bool_refused,
    refusal,
)
from lucid.test._helpers.dtype_refusal import engine_refusal
from lucid.test._helpers.dtype_refusal_pins import PINNED_OPS
from lucid.nn.functional._loss_checks import (
    _BOOL_BY_CALL,
    _FLOATING_RESULT,
    _KINDS,
    _OPERAND_DTYPES,
)

# ``upsample*`` are the reference's deprecated spellings of ``interpolate``: they warn by design.
pytestmark = pytest.mark.filterwarnings("ignore:.*nn.functional.upsample.*:UserWarning")

Kind = Literal["kernel", "guard", "refused", "accepted"]

# ``fw`` is the framework module under test, ``t`` a bool vector, ``m`` a bool
# matrix.  Both frameworks spell every call below the same way; the extra
# tensors a call needs are made from ``t`` (``t[:1]``, ``t.long()``).
_Call = Callable[[ModuleType, object, object], object]


@dataclass(frozen=True)
class _Row:
    """One call: the op of ``BOOL_REFUSED`` it pins (``None`` for an accepted
    spelling), how the reference answers it, and the call."""

    op: str | None
    kind: Kind
    call: _Call
    #: False for a spelling only Lucid has, which the reference cannot replay.
    parity: bool = True


def _kernel(op: str, call: _Call, *, parity: bool = True) -> _Row:
    return _Row(op, "kernel", call, parity)


def _guard(op: str, call: _Call) -> _Row:
    return _Row(op, "guard", call)


def _refused(op: str, call: _Call) -> _Row:
    return _Row(op, "refused", call)


def _accepted(call: _Call) -> _Row:
    return _Row(None, "accepted", call)


# The losses refuse a bool operand at their door (LCD-1024).  Each entry is the class the reference
# raises for a bool *input*, the ``nn.functional`` call and the ``nn`` module's, on the bool vector
# ``t`` and matrix ``m`` (a class label is ``t[:2].long()``).  The reference words a bool *target*
# beside a floating input as a dtype mismatch, a plain ``RuntimeError``, whatever the input's
# class: the ``-target`` rows further down pin those, and Lucid raises that class for them too.
_LossCalls = tuple[Kind, _Call, _Call]

_LOSSES: dict[str, _LossCalls] = {
    "mse_loss": (
        "kernel",
        lambda fw, t, m: fw.nn.functional.mse_loss(t, t),
        lambda fw, t, m: fw.nn.MSELoss()(t, t),
    ),
    "l1_loss": (
        "guard",
        lambda fw, t, m: fw.nn.functional.l1_loss(t, t),
        lambda fw, t, m: fw.nn.L1Loss()(t, t),
    ),
    "smooth_l1_loss": (
        "kernel",
        lambda fw, t, m: fw.nn.functional.smooth_l1_loss(t, t),
        lambda fw, t, m: fw.nn.SmoothL1Loss()(t, t),
    ),
    "huber_loss": (
        "kernel",
        lambda fw, t, m: fw.nn.functional.huber_loss(t, t),
        lambda fw, t, m: fw.nn.HuberLoss()(t, t),
    ),
    "binary_cross_entropy": (
        "kernel",
        lambda fw, t, m: fw.nn.functional.binary_cross_entropy(t, t),
        lambda fw, t, m: fw.nn.BCELoss()(t, t),
    ),
    "binary_cross_entropy_with_logits": (
        "kernel",
        lambda fw, t, m: fw.nn.functional.binary_cross_entropy_with_logits(t, t),
        lambda fw, t, m: fw.nn.BCEWithLogitsLoss()(t, t),
    ),
    "cross_entropy": (
        "kernel",
        lambda fw, t, m: fw.nn.functional.cross_entropy(m, t[:2].long()),
        lambda fw, t, m: fw.nn.CrossEntropyLoss()(m, t[:2].long()),
    ),
    "nll_loss": (
        "kernel",
        lambda fw, t, m: fw.nn.functional.nll_loss(m, t[:2].long()),
        lambda fw, t, m: fw.nn.NLLLoss()(m, t[:2].long()),
    ),
    "kl_div": (
        "guard",
        lambda fw, t, m: fw.nn.functional.kl_div(t, t),
        lambda fw, t, m: fw.nn.KLDivLoss()(t, t),
    ),
    "poisson_nll_loss": (
        "guard",
        lambda fw, t, m: fw.nn.functional.poisson_nll_loss(t, t),
        lambda fw, t, m: fw.nn.PoissonNLLLoss()(t, t),
    ),
    "gaussian_nll_loss": (
        "guard",
        lambda fw, t, m: fw.nn.functional.gaussian_nll_loss(t, t, t),
        lambda fw, t, m: fw.nn.GaussianNLLLoss()(t, t, t),
    ),
    "soft_margin_loss": (
        "guard",
        lambda fw, t, m: fw.nn.functional.soft_margin_loss(t, t),
        lambda fw, t, m: fw.nn.SoftMarginLoss()(t, t),
    ),
    "hinge_embedding_loss": (
        "guard",
        lambda fw, t, m: fw.nn.functional.hinge_embedding_loss(t, t),
        lambda fw, t, m: fw.nn.HingeEmbeddingLoss()(t, t),
    ),
    "margin_ranking_loss": (
        "guard",
        lambda fw, t, m: fw.nn.functional.margin_ranking_loss(t, t, t),
        lambda fw, t, m: fw.nn.MarginRankingLoss()(t, t, t),
    ),
    "multi_margin_loss": (
        "kernel",
        lambda fw, t, m: fw.nn.functional.multi_margin_loss(m, t[:2].long()),
        lambda fw, t, m: fw.nn.MultiMarginLoss()(m, t[:2].long()),
    ),
    "multilabel_margin_loss": (
        "kernel",
        lambda fw, t, m: fw.nn.functional.multilabel_margin_loss(m, m.long()),
        lambda fw, t, m: fw.nn.MultiLabelMarginLoss()(m, m.long()),
    ),
    "multilabel_soft_margin_loss": (
        "kernel",
        lambda fw, t, m: fw.nn.functional.multilabel_soft_margin_loss(m, m),
        lambda fw, t, m: fw.nn.MultiLabelSoftMarginLoss()(m, m),
    ),
    "triplet_margin_loss": (
        "guard",
        lambda fw, t, m: fw.nn.functional.triplet_margin_loss(m, m, m),
        lambda fw, t, m: fw.nn.TripletMarginLoss()(m, m, m),
    ),
    "triplet_margin_with_distance_loss": (
        "guard",
        lambda fw, t, m: fw.nn.functional.triplet_margin_with_distance_loss(m, m, m),
        lambda fw, t, m: fw.nn.TripletMarginWithDistanceLoss()(m, m, m),
    ),
    # The CTC log-probabilities are (time, batch, classes); the lengths are integer tensors.
    "ctc_loss": (
        "kernel",
        lambda fw, t, m: fw.nn.functional.ctc_loss(
            m.reshape(3, 1, 2),
            t[:1].reshape(1, 1).long(),
            fw.tensor([3]).long(),
            fw.tensor([1]).long(),
        ),
        lambda fw, t, m: fw.nn.CTCLoss()(
            m.reshape(3, 1, 2),
            t[:1].reshape(1, 1).long(),
            fw.tensor([3]).long(),
            fw.tensor([1]).long(),
        ),
    ),
}


def _loss_rows() -> dict[str, _Row]:
    rows: dict[str, _Row] = {}
    for loss, (kind, functional, module) in _LOSSES.items():
        rows[loss] = _Row(loss, kind, functional)
        rows[f"{loss}-module"] = _Row(loss, kind, module)
    return rows


_TABLE: dict[str, _Row] = {
    # ── elementwise ──────────────────────────────────────────────────────────
    "abs": _kernel("abs", lambda fw, t, m: fw.abs(t)),
    "abs-method": _kernel("abs", lambda fw, t, m: t.abs()),
    "abs-builtin": _kernel("abs", lambda fw, t, m: abs(t)),
    "abs-inplace": _kernel("abs", lambda fw, t, m: t.clone().abs_()),
    "absolute": _kernel("abs", lambda fw, t, m: fw.absolute(t)),
    "ceil": _kernel("ceil", lambda fw, t, m: fw.ceil(t)),
    "ceil-method": _kernel("ceil", lambda fw, t, m: t.ceil()),
    "ceil-inplace": _kernel("ceil", lambda fw, t, m: t.clone().ceil_()),
    "floor": _kernel("floor", lambda fw, t, m: fw.floor(t)),
    "floor-method": _kernel("floor", lambda fw, t, m: t.floor()),
    "floor-inplace": _kernel("floor", lambda fw, t, m: t.clone().floor_()),
    "round": _kernel("round", lambda fw, t, m: fw.round(t)),
    "round-method": _kernel("round", lambda fw, t, m: t.round()),
    "round-decimals": _kernel("round", lambda fw, t, m: fw.round(t, decimals=1)),
    "round-inplace": _kernel("round", lambda fw, t, m: t.clone().round_()),
    "trunc": _kernel("trunc", lambda fw, t, m: fw.trunc(t)),
    "trunc-method": _kernel("trunc", lambda fw, t, m: t.trunc()),
    "trunc-inplace": _kernel("trunc", lambda fw, t, m: t.clone().trunc_()),
    "fix": _kernel("trunc", lambda fw, t, m: fw.fix(t)),
    "fix-method": _kernel("trunc", lambda fw, t, m: t.fix()),
    "frac": _kernel("frac", lambda fw, t, m: fw.frac(t)),
    "frac-method": _kernel("frac", lambda fw, t, m: t.frac()),
    "frac-inplace": _kernel("frac", lambda fw, t, m: t.clone().frac_()),
    "frexp": _guard("frexp", lambda fw, t, m: fw.frexp(t)),
    "frexp-method": _guard("frexp", lambda fw, t, m: t.frexp()),
    "polar": _guard("polar", lambda fw, t, m: fw.polar(t, t)),
    "polar-first": _guard("polar", lambda fw, t, m: fw.polar(t, t.float())),
    "polar-second": _guard("polar", lambda fw, t, m: fw.polar(t.float(), t)),
    "nextafter": _kernel("nextafter", lambda fw, t, m: fw.nextafter(t, t)),
    "nextafter-method": _kernel("nextafter", lambda fw, t, m: t.nextafter(t)),
    "nextafter-inplace": _kernel("nextafter", lambda fw, t, m: t.clone().nextafter_(t)),
    # ── activations ──────────────────────────────────────────────────────────
    "relu": _guard("relu", lambda fw, t, m: fw.relu(t)),
    "relu-method": _guard("relu", lambda fw, t, m: t.relu()),
    "relu-inplace": _guard("relu", lambda fw, t, m: t.clone().relu_()),
    "relu-functional": _guard("relu", lambda fw, t, m: fw.nn.functional.relu(t)),
    "relu-functional-inplace": _guard(
        "relu", lambda fw, t, m: fw.nn.functional.relu(t.clone(), inplace=True)
    ),
    "relu-module": _guard("relu", lambda fw, t, m: fw.nn.ReLU()(t)),
    "selu": _kernel("selu", lambda fw, t, m: fw.selu(t)),
    "selu-method": _kernel("selu", lambda fw, t, m: t.selu(), parity=False),
    "selu-functional": _kernel("selu", lambda fw, t, m: fw.nn.functional.selu(t)),
    "selu-module": _kernel("selu", lambda fw, t, m: fw.nn.SELU()(t)),
    "softmax": _kernel("softmax", lambda fw, t, m: fw.softmax(t, 0)),
    "softmax-dim": _kernel("softmax", lambda fw, t, m: fw.softmax(m, -1)),
    "softmax-method": _kernel("softmax", lambda fw, t, m: t.softmax(0)),
    "softmax-functional": _kernel("softmax", lambda fw, t, m: fw.nn.functional.softmax(t, 0)),
    "softmax-module": _kernel("softmax", lambda fw, t, m: fw.nn.Softmax(0)(t)),
    "log_softmax": _kernel("log_softmax", lambda fw, t, m: fw.log_softmax(t, 0)),
    "log_softmax-method": _kernel("log_softmax", lambda fw, t, m: t.log_softmax(0)),
    "log_softmax-functional": _kernel(
        "log_softmax", lambda fw, t, m: fw.nn.functional.log_softmax(t, 0)
    ),
    "log_softmax-module": _kernel("log_softmax", lambda fw, t, m: fw.nn.LogSoftmax(0)(t)),
    # ── moments ──────────────────────────────────────────────────────────────
    "mean": _guard("mean", lambda fw, t, m: fw.mean(t)),
    "mean-dim": _guard("mean", lambda fw, t, m: fw.mean(m, dim=1)),
    "mean-method": _guard("mean", lambda fw, t, m: t.mean()),
    # ``std`` is the root of the variance, so the engine's ``var`` refuses it.
    "std": _guard("var", lambda fw, t, m: fw.std(t)),
    "std-dim": _guard("var", lambda fw, t, m: fw.std(m, dim=1)),
    "std-method": _guard("var", lambda fw, t, m: t.std()),
    "var": _guard("var", lambda fw, t, m: fw.var(t)),
    "var-correction": _guard("var", lambda fw, t, m: fw.var(t, correction=0)),
    "var-method": _guard("var", lambda fw, t, m: t.var()),
    "std_mean": _guard("std_mean", lambda fw, t, m: fw.std_mean(t)),
    "std_mean-dim": _guard("std_mean", lambda fw, t, m: fw.std_mean(m, dim=1)),
    "var_mean": _guard("var_mean", lambda fw, t, m: fw.var_mean(t)),
    "var_mean-dim": _guard("var_mean", lambda fw, t, m: fw.var_mean(m, dim=1)),
    "nanmean": _guard("nanmean", lambda fw, t, m: fw.nanmean(t)),
    "nanmean-method": _guard("nanmean", lambda fw, t, m: t.nanmean()),
    # ``nanmean`` has no spelling that takes a bool, ``mean`` does (below).
    "nanmean-dtype": _guard("nanmean", lambda fw, t, m: fw.nanmean(t, dtype=fw.float32)),
    "cov": _guard("cov", lambda fw, t, m: fw.cov(m)),
    "corrcoef": _guard("corrcoef", lambda fw, t, m: fw.corrcoef(m)),
    "gradient": _guard("gradient", lambda fw, t, m: fw.gradient(t)),
    "gradient-spacing": _guard("gradient", lambda fw, t, m: fw.gradient(t, spacing=2.0)),
    "gradient-dim": _guard("gradient", lambda fw, t, m: fw.gradient(m, dim=1)),
    "histogram": _kernel("histogram", lambda fw, t, m: fw.histogram(t, 2)),
    "histogram-range": _kernel("histogram", lambda fw, t, m: fw.histogram(t, 2, range=(0.0, 1.0))),
    "histogram-bins": _refused("histogram", lambda fw, t, m: fw.histogram(t.float(), t)),
    "histogram-weight": _refused(
        "histogram", lambda fw, t, m: fw.histogram(t.float(), 2, weight=t)
    ),
    "poisson": _kernel("poisson", lambda fw, t, m: fw.poisson(t)),
    # ── order statistics, search and membership ──────────────────────────────
    "quantile": _guard("quantile", lambda fw, t, m: fw.quantile(t, 0.5)),
    "quantile-dim": _guard("quantile", lambda fw, t, m: fw.quantile(m, 0.5, dim=1)),
    "quantile-method": _guard("quantile", lambda fw, t, m: t.quantile(0.5)),
    "nanquantile": _guard("nanquantile", lambda fw, t, m: fw.nanquantile(t, 0.5)),
    "nanquantile-method": _guard("nanquantile", lambda fw, t, m: t.nanquantile(0.5)),
    "isin": _guard("isin", lambda fw, t, m: fw.isin(t, t)),
    "isin-elements": _guard("isin", lambda fw, t, m: fw.isin(t, t.long())),
    "isin-test-elements": _guard("isin", lambda fw, t, m: fw.isin(t.long(), t)),
    # A sequence and its values meet at their common dtype: both bool is the
    # one pairing the reference has no kernel for.
    "searchsorted": _kernel("searchsorted", lambda fw, t, m: fw.searchsorted(t, t[:1])),
    "bucketize": _kernel("bucketize", lambda fw, t, m: fw.bucketize(t, t)),
    # ``clamp`` refuses a bool *number* bound on a bool tensor: a number bound
    # that is an int or a float promotes the answer, and a tensor bound is
    # answered by ``minimum`` / ``maximum``.
    "clamp": _kernel("clamp", lambda fw, t, m: fw.clamp(t, False, True)),
    "clamp-min": _kernel("clamp", lambda fw, t, m: fw.clamp(t, min=False)),
    "clamp-max": _kernel("clamp", lambda fw, t, m: fw.clamp(t, max=True)),
    "clamp-method": _kernel("clamp", lambda fw, t, m: t.clamp(False, True)),
    "clip": _kernel("clamp", lambda fw, t, m: fw.clip(t, False, True)),
    "clamp_min": _kernel("clamp", lambda fw, t, m: fw.clamp_min(t, True)),
    "clamp_max": _kernel("clamp", lambda fw, t, m: fw.clamp_max(t, True)),
    "clamp-inplace": _kernel("clamp", lambda fw, t, m: t.clone().clamp_(False, True)),
    "clip-inplace": _kernel("clamp", lambda fw, t, m: t.clone().clip_(False, True)),
    "clamp_min-inplace": _kernel("clamp", lambda fw, t, m: t.clone().clamp_min_(True)),
    "clamp-tensor-bounds": _kernel("clamp", lambda fw, t, m: fw.clamp(t, t, t)),
    "clamp-tensor-bounds-method": _kernel("clamp", lambda fw, t, m: t.clamp(t, t)),
    "clip-tensor-bounds": _kernel("clamp", lambda fw, t, m: fw.clip(t, t, t)),
    "clamp-tensor-bounds-inplace": _kernel("clamp", lambda fw, t, m: t.clone().clamp_(t, t)),
    # ``pow`` refuses a bool beside a tensor exponent, and answers a number one.
    "pow": _kernel("pow", lambda fw, t, m: fw.pow(t, t)),
    "pow-method": _kernel("pow", lambda fw, t, m: t.pow(t)),
    "pow-operator": _kernel("pow", lambda fw, t, m: t**t),
    "pow-inplace": _kernel("pow", lambda fw, t, m: t.clone().pow_(t)),
    "pow-inplace-operator": _kernel("pow", lambda fw, t, m: operator.ipow(t.clone(), t)),
    "pow-bool-base": _kernel("pow", lambda fw, t, m: fw.pow(True, t)),
    "pow-bool-base-operator": _kernel("pow", lambda fw, t, m: operator.pow(True, t)),
    "gammainc": _kernel("gammainc", lambda fw, t, m: fw.special.gammainc(t, t)),
    "gammaincc": _kernel("gammaincc", lambda fw, t, m: fw.special.gammaincc(t, t)),
    "multigammaln": _guard("multigammaln", lambda fw, t, m: fw.special.multigammaln(t, 1)),
    "vecdot": _guard("vecdot", lambda fw, t, m: fw.linalg.vecdot(t, t)),
    "vecdot-first": _guard("vecdot", lambda fw, t, m: fw.linalg.vecdot(t, t.float())),
    "vecdot-second": _guard("vecdot", lambda fw, t, m: fw.linalg.vecdot(t.float(), t)),
    # ── activations composed from other ops ──────────────────────────────────
    "celu": _kernel("celu", lambda fw, t, m: fw.nn.functional.celu(t)),
    "celu-alpha": _kernel("celu", lambda fw, t, m: fw.nn.functional.celu(t, alpha=2.0)),
    "celu-inplace": _kernel("celu", lambda fw, t, m: fw.nn.functional.celu_(t.clone())),
    "celu-inplace-alpha": _kernel(
        "celu", lambda fw, t, m: fw.nn.functional.celu(t.clone(), alpha=2.0, inplace=True)
    ),
    "celu-module": _kernel("celu", lambda fw, t, m: fw.nn.CELU(alpha=2.0)(t)),
    "hardshrink": _kernel("hardshrink", lambda fw, t, m: fw.nn.functional.hardshrink(t)),
    "hardshrink-module": _kernel("hardshrink", lambda fw, t, m: fw.nn.Hardshrink()(t)),
    "softshrink": _kernel("softshrink", lambda fw, t, m: fw.nn.functional.softshrink(t)),
    "softshrink-module": _kernel("softshrink", lambda fw, t, m: fw.nn.Softshrink()(t)),
    "hardtanh": _guard("hardtanh", lambda fw, t, m: fw.nn.functional.hardtanh(t)),
    "hardtanh-inplace": _guard("hardtanh", lambda fw, t, m: fw.nn.functional.hardtanh_(t.clone())),
    "hardtanh-inplace-flag": _guard(
        "hardtanh", lambda fw, t, m: fw.nn.functional.hardtanh(t.clone(), inplace=True)
    ),
    "hardtanh-module": _guard("hardtanh", lambda fw, t, m: fw.nn.Hardtanh()(t)),
    "threshold": _kernel("threshold", lambda fw, t, m: fw.nn.functional.threshold(t, 0.1, 20.0)),
    "threshold-inplace": _kernel(
        "threshold", lambda fw, t, m: fw.nn.functional.threshold_(t.clone(), 0.1, 20.0)
    ),
    "threshold-module": _kernel("threshold", lambda fw, t, m: fw.nn.Threshold(0.1, 20.0)(t)),
    "rrelu": _kernel("rrelu", lambda fw, t, m: fw.nn.functional.rrelu(t)),
    "rrelu-training": _kernel("rrelu", lambda fw, t, m: fw.nn.functional.rrelu(t, training=True)),
    "rrelu-inplace": _kernel("rrelu", lambda fw, t, m: fw.nn.functional.rrelu_(t.clone())),
    "rrelu-inplace-training": _kernel(
        "rrelu", lambda fw, t, m: fw.nn.functional.rrelu_(t.clone(), training=True)
    ),
    "rrelu-module": _kernel("rrelu", lambda fw, t, m: fw.nn.RReLU()(t)),
    # The reference refuses a bool beside a float weight for its own reason (a
    # type promotion it will not do), and a bool weight with the kernel.
    "prelu": _kernel("prelu", lambda fw, t, m: fw.nn.functional.prelu(t, t[:1])),
    "prelu-float-weight": _refused(
        "prelu", lambda fw, t, m: fw.nn.functional.prelu(t, t[:1].float())
    ),
    "prelu-module": _refused("prelu", lambda fw, t, m: fw.nn.PReLU()(t)),
    # ── resampling ───────────────────────────────────────────────────────────
    "interpolate": _kernel(
        "interpolate",
        lambda fw, t, m: fw.nn.functional.interpolate(m.reshape(1, 2, 3), scale_factor=2),
    ),
    "interpolate-bilinear": _kernel(
        "interpolate",
        lambda fw, t, m: fw.nn.functional.interpolate(
            m.reshape(1, 1, 2, 3), scale_factor=2, mode="bilinear"
        ),
    ),
    "interpolate-area": _kernel(
        "interpolate",
        lambda fw, t, m: fw.nn.functional.interpolate(m.reshape(1, 2, 3), size=2, mode="area"),
    ),
    "upsample": _kernel(
        "interpolate",
        lambda fw, t, m: fw.nn.functional.upsample(m.reshape(1, 1, 2, 3), scale_factor=2),
    ),
    "upsample_nearest": _kernel(
        "interpolate",
        lambda fw, t, m: fw.nn.functional.upsample_nearest(m.reshape(1, 1, 2, 3), scale_factor=2),
    ),
    "upsample_bilinear": _kernel(
        "interpolate",
        lambda fw, t, m: fw.nn.functional.upsample_bilinear(m.reshape(1, 1, 2, 3), scale_factor=2),
    ),
    "upsample-module": _kernel(
        "interpolate", lambda fw, t, m: fw.nn.Upsample(scale_factor=2)(m.reshape(1, 1, 2, 3))
    ),
    "upsampling-nearest-module": _kernel(
        "interpolate",
        lambda fw, t, m: fw.nn.UpsamplingNearest2d(scale_factor=2)(m.reshape(1, 1, 2, 3)),
    ),
    "upsampling-bilinear-module": _kernel(
        "interpolate",
        lambda fw, t, m: fw.nn.UpsamplingBilinear2d(scale_factor=2)(m.reshape(1, 1, 2, 3)),
    ),
    "grid_sample": _kernel(
        "grid_sample",
        lambda fw, t, m: fw.nn.functional.grid_sample(
            m.reshape(1, 1, 2, 3), m.float().reshape(1, 1, 3, 2), align_corners=False
        ),
    ),
    # The reference refuses a bool grid beside a float image with a mixed-dtype
    # error; Lucid, whose cast would have hidden it, refuses it as it does the image.
    "grid_sample-grid": _refused(
        "grid_sample",
        lambda fw, t, m: fw.nn.functional.grid_sample(
            m.float().reshape(1, 1, 2, 3), m.reshape(1, 1, 3, 2), align_corners=False
        ),
    ),
    # ── losses, and the other ops that refused a bool with a plain TypeError (LCD-1024) ─
    **_loss_rows(),
    # A bool target, label or var beside a floating input: the reference's plain dtype-mismatch
    # RuntimeError in every loss, whichever class the loss gives a bool *input* (``guard`` fails a
    # NotImplementedError).  A bool class label is told to be an integer, not a float (below).
    "binary_cross_entropy-target": _guard(
        "binary_cross_entropy", lambda fw, t, m: fw.nn.functional.binary_cross_entropy(t.float(), t)
    ),
    "binary_cross_entropy_with_logits-target": _guard(
        "binary_cross_entropy_with_logits",
        lambda fw, t, m: fw.nn.functional.binary_cross_entropy_with_logits(t.float(), t),
    ),
    "cross_entropy-target": _guard(
        "cross_entropy", lambda fw, t, m: fw.nn.functional.cross_entropy(m.float(), t[:2])
    ),
    "cross_entropy-probabilities": _guard(
        "cross_entropy", lambda fw, t, m: fw.nn.functional.cross_entropy(m.float(), m)
    ),
    "nll_loss-target": _guard(
        "nll_loss", lambda fw, t, m: fw.nn.functional.nll_loss(m.float(), t[:2])
    ),
    "multi_margin_loss-target": _guard(
        "multi_margin_loss", lambda fw, t, m: fw.nn.functional.multi_margin_loss(m.float(), t[:2])
    ),
    "multilabel_margin_loss-target": _guard(
        "multilabel_margin_loss",
        lambda fw, t, m: fw.nn.functional.multilabel_margin_loss(m.float(), m),
    ),
    "multilabel_soft_margin_loss-target": _guard(
        "multilabel_soft_margin_loss",
        lambda fw, t, m: fw.nn.functional.multilabel_soft_margin_loss(m.float(), m),
    ),
    "kl_div-target": _guard("kl_div", lambda fw, t, m: fw.nn.functional.kl_div(t.float(), t)),
    "l1_loss-target": _guard("l1_loss", lambda fw, t, m: fw.nn.functional.l1_loss(t.float(), t)),
    "margin_ranking_loss-target": _guard(
        "margin_ranking_loss",
        lambda fw, t, m: fw.nn.functional.margin_ranking_loss(t.float(), t.float(), t),
    ),
    "triplet_margin_loss-negative": _guard(
        "triplet_margin_loss",
        lambda fw, t, m: fw.nn.functional.triplet_margin_loss(m.float(), m.float(), m),
    ),
    "gaussian_nll_loss-var": _guard(
        "gaussian_nll_loss",
        lambda fw, t, m: fw.nn.functional.gaussian_nll_loss(t.float(), t.float(), t),
    ),
    # A bool *input* beside a floating target takes the class its loss gives one in the reference,
    # which varies: these two are the kernel's NotImplementedError there.  ``binary_cross_entropy``
    # and ``huber_loss`` are the kernel's class for an all-bool call but a dtype mismatch beside a
    # target of another dtype (LCD-1046), as the next rows pin.
    "binary_cross_entropy_with_logits-input": _kernel(
        "binary_cross_entropy_with_logits",
        lambda fw, t, m: fw.nn.functional.binary_cross_entropy_with_logits(t, t.float()),
    ),
    "multilabel_soft_margin_loss-input": _kernel(
        "multilabel_soft_margin_loss",
        lambda fw, t, m: fw.nn.functional.multilabel_soft_margin_loss(m, m.float()),
    ),
    "binary_cross_entropy-input": _guard(
        "binary_cross_entropy", lambda fw, t, m: fw.nn.functional.binary_cross_entropy(t, t.float())
    ),
    "binary_cross_entropy-input-integer": _guard(
        "binary_cross_entropy", lambda fw, t, m: fw.nn.functional.binary_cross_entropy(t, t.long())
    ),
    "huber_loss-input": _guard(
        "huber_loss", lambda fw, t, m: fw.nn.functional.huber_loss(t, t.float())
    ),
    "huber_loss-input-integer": _guard(
        "huber_loss", lambda fw, t, m: fw.nn.functional.huber_loss(t, t.long())
    ),
    # A class weight beside floating logits is read at the logits' dtype by the reference's
    # class-index kernels, so a bool one is a plain dtype ``RuntimeError`` (LCD-1046).  Class
    # probabilities multiply it in instead and take it (the accepted rows below).
    "cross_entropy-weight": _guard(
        "cross_entropy",
        lambda fw, t, m: fw.nn.functional.cross_entropy(m.float(), t[:2].long(), weight=t[:3]),
    ),
    "cross_entropy-weight-module": _guard(
        "cross_entropy",
        lambda fw, t, m: fw.nn.CrossEntropyLoss(weight=t[:3])(m.float(), t[:2].long()),
    ),
    "nll_loss-weight": _guard(
        "nll_loss",
        lambda fw, t, m: fw.nn.functional.nll_loss(m.float(), t[:2].long(), weight=t[:3]),
    ),
    "nll_loss-weight-module": _guard(
        "nll_loss", lambda fw, t, m: fw.nn.NLLLoss(weight=t[:3])(m.float(), t[:2].long())
    ),
    "multi_margin_loss-weight": _guard(
        "multi_margin_loss",
        lambda fw, t, m: fw.nn.functional.multi_margin_loss(m.float(), t[:2].long(), weight=t[:3]),
    ),
    "multi_margin_loss-weight-module": _guard(
        "multi_margin_loss",
        lambda fw, t, m: fw.nn.MultiMarginLoss(weight=t[:3])(m.float(), t[:2].long()),
    ),
    # ``pos_weight - 1`` is a subtraction, and the reference's takes no bool.
    "binary_cross_entropy_with_logits-pos_weight": _guard(
        "binary_cross_entropy_with_logits",
        lambda fw, t, m: fw.nn.functional.binary_cross_entropy_with_logits(
            m.float(), m.float(), pos_weight=t[:3]
        ),
    ),
    # The CTC length tensors must be integral: a bool one is a dtype ``RuntimeError`` (LCD-1046).
    "ctc_loss-input_lengths": _guard(
        "ctc_loss",
        lambda fw, t, m: fw.nn.functional.ctc_loss(
            m.float().reshape(3, 1, 2), t[:1].reshape(1, 1).long(), t[:1], fw.tensor([1]).long()
        ),
    ),
    "ctc_loss-target_lengths": _guard(
        "ctc_loss",
        lambda fw, t, m: fw.nn.functional.ctc_loss(
            m.float().reshape(3, 1, 2), t[:1].reshape(1, 1).long(), fw.tensor([3]).long(), t[:1]
        ),
    ),
    "ctc_loss-lengths-module": _guard(
        "ctc_loss",
        lambda fw, t, m: fw.nn.CTCLoss()(
            m.float().reshape(3, 1, 2), t[:1].reshape(1, 1).long(), t[:1], fw.tensor([1]).long()
        ),
    ),
    # ``poisson_nll_loss`` takes a bool beside another dtype -- ``exp`` of one is a float -- and
    # refuses it only where the reference subtracts one: a bool input that is not a log, and a
    # bool target under ``full`` (a bool input beside a bool target is the row above).
    "poisson_nll_loss-rate": _guard(
        "poisson_nll_loss",
        lambda fw, t, m: fw.nn.functional.poisson_nll_loss(t, t.float(), log_input=False),
    ),
    "poisson_nll_loss-full": _guard(
        "poisson_nll_loss",
        lambda fw, t, m: fw.nn.functional.poisson_nll_loss(t.float(), t, full=True),
    ),
    # A bool beside an integer meets no floating operand: the kernel's class, as for two bools.
    "mse_loss-integer": _kernel(
        "mse_loss", lambda fw, t, m: fw.nn.functional.mse_loss(t, t.long())
    ),
    "mse_loss-integer-target": _kernel(
        "mse_loss", lambda fw, t, m: fw.nn.functional.mse_loss(t.long(), t)
    ),
    "smooth_l1_loss-integer": _kernel(
        "smooth_l1_loss", lambda fw, t, m: fw.nn.functional.smooth_l1_loss(t, t.long())
    ),
    "smooth_l1_loss-integer-target": _kernel(
        "smooth_l1_loss", lambda fw, t, m: fw.nn.functional.smooth_l1_loss(t.long(), t)
    ),
    "rms_norm": _kernel("rms_norm", lambda fw, t, m: fw.nn.functional.rms_norm(t, (4,))),
    # An explicit ``eps`` skipped the ``finfo`` that refused the default one: the bool computed.
    "rms_norm-eps": _kernel(
        "rms_norm", lambda fw, t, m: fw.nn.functional.rms_norm(t, (4,), eps=1e-5)
    ),
    "rms_norm-weight": _refused(
        "rms_norm", lambda fw, t, m: fw.nn.functional.rms_norm(t, (4,), t.float())
    ),
    "rms_norm-module": _refused("rms_norm", lambda fw, t, m: fw.nn.RMSNorm(4)(t)),
    "tanhshrink": _guard("tanhshrink", lambda fw, t, m: fw.nn.functional.tanhshrink(t)),
    "tanhshrink-module": _guard("tanhshrink", lambda fw, t, m: fw.nn.Tanhshrink()(t)),
    "vander": _guard("vander", lambda fw, t, m: fw.linalg.vander(t)),
    "vander-columns": _guard("vander", lambda fw, t, m: fw.linalg.vander(t, N=2)),
    # The channel mask is drawn only when something is dropped: a bool refuses there, as dropout's.
    "feature_alpha_dropout": _guard(
        "feature_alpha_dropout",
        lambda fw, t, m: fw.nn.functional.feature_alpha_dropout(
            t.reshape(1, 2, 2), p=0.5, training=True
        ),
    ),
    "feature_alpha_dropout-inplace": _guard(
        "feature_alpha_dropout",
        lambda fw, t, m: fw.nn.functional.feature_alpha_dropout(
            t.clone().reshape(1, 2, 2), p=0.5, training=True, inplace=True
        ),
    ),
    # Below rank 2 the reference refuses for its own reason (a feature needs a channel axis).
    "feature_alpha_dropout-rank1": _refused(
        "feature_alpha_dropout",
        lambda fw, t, m: fw.nn.functional.feature_alpha_dropout(t, p=0.5, training=True),
    ),
    "feature_alpha_dropout-module": _guard(
        "feature_alpha_dropout", lambda fw, t, m: fw.nn.FeatureAlphaDropout()(t.reshape(1, 2, 2))
    ),
    # ── refused by the engine's schema: no Python entry repeats these ────────
    "silu-functional": _kernel("silu", lambda fw, t, m: fw.nn.functional.silu(t)),
    "silu-method": _kernel("silu", lambda fw, t, m: t.silu(), parity=False),
    "silu-module": _kernel("silu", lambda fw, t, m: fw.nn.SiLU()(t)),
    "gelu": _kernel("gelu_exact", lambda fw, t, m: fw.nn.functional.gelu(t)),
    "gelu-tanh": _kernel("gelu", lambda fw, t, m: fw.nn.functional.gelu(t, approximate="tanh")),
    "gelu-module": _kernel("gelu_exact", lambda fw, t, m: fw.nn.GELU()(t)),
    "mish": _kernel("mish", lambda fw, t, m: fw.nn.functional.mish(t)),
    "mish-module": _kernel("mish", lambda fw, t, m: fw.nn.Mish()(t)),
    "relu6": _guard("relu6", lambda fw, t, m: fw.nn.functional.relu6(t)),
    "relu6-module": _guard("relu6", lambda fw, t, m: fw.nn.ReLU6()(t)),
    "hard_sigmoid": _kernel("hard_sigmoid", lambda fw, t, m: fw.nn.functional.hardsigmoid(t)),
    "hard_sigmoid-module": _kernel("hard_sigmoid", lambda fw, t, m: fw.nn.Hardsigmoid()(t)),
    "hard_swish": _kernel("hard_swish", lambda fw, t, m: fw.nn.functional.hardswish(t)),
    "hard_swish-module": _kernel("hard_swish", lambda fw, t, m: fw.nn.Hardswish()(t)),
    "leaky_relu": _kernel("leaky_relu", lambda fw, t, m: fw.nn.functional.leaky_relu(t, 0.1)),
    "leaky_relu-module": _kernel("leaky_relu", lambda fw, t, m: fw.nn.LeakyReLU(0.1)(t)),
    "elu": _kernel("elu", lambda fw, t, m: fw.nn.functional.elu(t)),
    "elu-module": _kernel("elu", lambda fw, t, m: fw.nn.ELU()(t)),
    "softsign-module": _kernel("abs", lambda fw, t, m: fw.nn.Softsign()(t)),
    "max_pool1d": _kernel(
        "max_pool1d", lambda fw, t, m: fw.nn.functional.max_pool1d(t.reshape(1, 1, 4), 2)
    ),
    "max_pool2d": _kernel(
        "max_pool2d", lambda fw, t, m: fw.nn.functional.max_pool2d(m.reshape(1, 1, 2, 3), 2)
    ),
    "max_pool3d": _kernel(
        "max_pool3d", lambda fw, t, m: fw.nn.functional.max_pool3d(t.reshape(1, 1, 1, 2, 2), 1)
    ),
    "max_pool-module": _kernel(
        "max_pool2d", lambda fw, t, m: fw.nn.MaxPool2d(2)(m.reshape(1, 1, 2, 3))
    ),
    "avg_pool1d": _kernel(
        "avg_pool1d", lambda fw, t, m: fw.nn.functional.avg_pool1d(t.reshape(1, 1, 4), 2)
    ),
    "avg_pool2d": _kernel(
        "avg_pool2d", lambda fw, t, m: fw.nn.functional.avg_pool2d(m.reshape(1, 1, 2, 3), 2)
    ),
    "avg_pool3d": _kernel(
        "avg_pool3d", lambda fw, t, m: fw.nn.functional.avg_pool3d(t.reshape(1, 1, 1, 2, 2), 1)
    ),
    "layer_norm": _kernel("layer_norm", lambda fw, t, m: fw.nn.functional.layer_norm(t, (4,))),
    # A float weight beside the bool, which the reference refuses first, as a mixed dtype.
    "layer_norm-weight": _refused(
        "layer_norm", lambda fw, t, m: fw.nn.functional.layer_norm(t, (4,), t.float())
    ),
    "group_norm": _kernel(
        "group_norm", lambda fw, t, m: fw.nn.functional.group_norm(m.reshape(1, 2, 3), 2)
    ),
    "batch_norm1d": _kernel(
        "batch_norm1d",
        lambda fw, t, m: fw.nn.functional.batch_norm(m.reshape(1, 2, 3), None, None, training=True),
    ),
    "batch_norm": _kernel(
        "batch_norm",
        lambda fw, t, m: fw.nn.functional.batch_norm(
            m.reshape(1, 1, 2, 3), None, None, training=True
        ),
    ),
    "batch_norm3d": _kernel(
        "batch_norm3d",
        lambda fw, t, m: fw.nn.functional.batch_norm(
            t.reshape(1, 1, 1, 2, 2), None, None, training=True
        ),
    ),
    "batch_norm_eval": _kernel(
        "batch_norm_eval",
        lambda fw, t, m: fw.nn.functional.batch_norm(
            m.reshape(1, 1, 2, 3), t[:1].float(), t[:1].float(), training=False
        ),
    ),
    # The GEMM and convolution kernels have no bool instantiation in the reference (LCD-1025);
    # Metal used to answer with MLX's own error for these.
    "linear": _kernel("linear", lambda fw, t, m: fw.nn.functional.linear(m, m)),
    # A float input beside a bool weight: the reference reports the mismatch first.
    "linear-weight": _refused("linear", lambda fw, t, m: fw.nn.functional.linear(m.float(), m)),
    "bilinear": _kernel(
        "bilinear_layer",
        lambda fw, t, m: fw.nn.functional.bilinear(
            t.reshape(2, 2), t.reshape(2, 2), t.reshape(1, 2, 2)
        ),
    ),
    "conv1d": _kernel(
        "conv1d",
        lambda fw, t, m: fw.nn.functional.conv1d(t.reshape(1, 1, 4), t[:2].reshape(1, 1, 2)),
    ),
    "conv2d": _kernel(
        "conv2d",
        lambda fw, t, m: fw.nn.functional.conv2d(m.reshape(1, 1, 2, 3), t.reshape(1, 1, 2, 2)),
    ),
    "conv3d": _kernel(
        "conv3d",
        lambda fw, t, m: fw.nn.functional.conv3d(
            t.reshape(1, 1, 1, 2, 2), t[:1].reshape(1, 1, 1, 1, 1)
        ),
    ),
    "conv_transpose1d": _kernel(
        "conv_transpose1d",
        lambda fw, t, m: fw.nn.functional.conv_transpose1d(
            t.reshape(1, 1, 4), t[:2].reshape(1, 1, 2)
        ),
    ),
    "conv_transpose2d": _kernel(
        "conv_transpose2d",
        lambda fw, t, m: fw.nn.functional.conv_transpose2d(
            m.reshape(1, 1, 2, 3), t.reshape(1, 1, 2, 2)
        ),
    ),
    "conv_transpose3d": _kernel(
        "conv_transpose3d",
        lambda fw, t, m: fw.nn.functional.conv_transpose3d(
            t.reshape(1, 1, 1, 2, 2), t[:1].reshape(1, 1, 1, 1, 1)
        ),
    ),
    # Dropout draws a mask in the input's dtype; the reference refuses a bool at its dtype guard
    # (LCD-1025).  Nothing is drawn in eval or at ``p == 0``, and those still answer.
    "dropout": _guard("dropout", lambda fw, t, m: fw.nn.functional.dropout(t)),
    "dropout-module": _guard("dropout", lambda fw, t, m: fw.nn.Dropout()(t)),
    "dropout-inplace": _guard(
        "dropout", lambda fw, t, m: fw.nn.functional.dropout(t.clone(), inplace=True)
    ),
    "dropout1d": _guard(
        "dropoutnd", lambda fw, t, m: fw.nn.functional.dropout1d(t.reshape(1, 2, 2))
    ),
    "dropout1d-module": _guard("dropoutnd", lambda fw, t, m: fw.nn.Dropout1d()(t.reshape(1, 2, 2))),
    "dropout2d": _guard(
        "dropoutnd", lambda fw, t, m: fw.nn.functional.dropout2d(t.reshape(1, 1, 2, 2))
    ),
    "dropout2d-module": _guard(
        "dropoutnd", lambda fw, t, m: fw.nn.Dropout2d()(t.reshape(1, 1, 2, 2))
    ),
    "dropout3d": _guard(
        "dropoutnd", lambda fw, t, m: fw.nn.functional.dropout3d(t.reshape(1, 1, 1, 2, 2))
    ),
    "dropout3d-module": _guard(
        "dropoutnd", lambda fw, t, m: fw.nn.Dropout3d()(t.reshape(1, 1, 1, 2, 2))
    ),
    "alpha_dropout": _guard(
        "alpha_dropout", lambda fw, t, m: fw.nn.functional.alpha_dropout(t, training=True)
    ),
    "alpha_dropout-module": _guard("alpha_dropout", lambda fw, t, m: fw.nn.AlphaDropout()(t)),
    # ── what the reference answers, and so does Lucid ────────────────────────
    "dropout-eval": _accepted(lambda fw, t, m: fw.nn.functional.dropout(t, training=False)),
    "dropout-p0": _accepted(lambda fw, t, m: fw.nn.functional.dropout(t, p=0.0)),
    "dropout2d-eval": _accepted(
        lambda fw, t, m: fw.nn.functional.dropout2d(t.reshape(1, 1, 2, 2), training=False)
    ),
    "feature_alpha_dropout-eval": _accepted(
        lambda fw, t, m: fw.nn.functional.feature_alpha_dropout(t.reshape(1, 2, 2), training=False)
    ),
    "feature_alpha_dropout-default": _accepted(
        lambda fw, t, m: fw.nn.functional.feature_alpha_dropout(t.reshape(1, 2, 2))
    ),
    "feature_alpha_dropout-p0": _accepted(
        lambda fw, t, m: fw.nn.functional.feature_alpha_dropout(
            t.reshape(1, 2, 2), p=0.0, training=True
        )
    ),
    "feature_alpha_dropout-module-eval": _accepted(
        lambda fw, t, m: fw.nn.FeatureAlphaDropout().eval()(t.reshape(1, 2, 2))
    ),
    # Two bools compare as truth values, with ``isclose``'s tolerances.
    "allclose": _accepted(lambda fw, t, m: fw.allclose(t, t)),
    "allclose-different": _accepted(lambda fw, t, m: fw.allclose(t, ~t)),
    "allclose-method": _accepted(lambda fw, t, m: t.allclose(t)),
    "allclose-equal-nan": _accepted(lambda fw, t, m: fw.allclose(t, t, equal_nan=True)),
    "allclose-atol": _accepted(lambda fw, t, m: fw.allclose(t, ~t, atol=1.0)),
    "allclose-rtol": _accepted(lambda fw, t, m: fw.allclose(t, ~t, rtol=1.0, atol=0.0)),
    "allclose-broadcast": _accepted(lambda fw, t, m: fw.allclose(m, t[:3])),
    # A bool target beside a floating input is a number to these, as to the reference.
    "mse_loss-target": _accepted(lambda fw, t, m: fw.nn.functional.mse_loss(t.float(), t)),
    "smooth_l1_loss-target": _accepted(
        lambda fw, t, m: fw.nn.functional.smooth_l1_loss(t.float(), t)
    ),
    "huber_loss-target": _accepted(lambda fw, t, m: fw.nn.functional.huber_loss(t.float(), t)),
    "soft_margin_loss-target": _accepted(
        lambda fw, t, m: fw.nn.functional.soft_margin_loss(t.float(), t)
    ),
    "hinge_embedding_loss-target": _accepted(
        lambda fw, t, m: fw.nn.functional.hinge_embedding_loss(t.float(), t)
    ),
    "cosine_embedding_loss": _accepted(
        lambda fw, t, m: fw.nn.functional.cosine_embedding_loss(m, m.flip(0), t[:2])
    ),
    # A bool beside another dtype is a number to ``poisson_nll_loss`` (LCD-1046), and a float16
    # target does not narrow what a non-floating input makes of it: float32, as the reference.
    "poisson_nll_loss-bool-input": _accepted(
        lambda fw, t, m: fw.nn.functional.poisson_nll_loss(t, t.float())
    ),
    "poisson_nll_loss-bool-input-integer": _accepted(
        lambda fw, t, m: fw.nn.functional.poisson_nll_loss(t, t.long())
    ),
    "poisson_nll_loss-bool-input-half": _accepted(
        lambda fw, t, m: fw.nn.functional.poisson_nll_loss(t, t.half())
    ),
    "poisson_nll_loss-bool-target": _accepted(
        lambda fw, t, m: fw.nn.functional.poisson_nll_loss(t.float(), t)
    ),
    "poisson_nll_loss-bool-target-integer": _accepted(
        lambda fw, t, m: fw.nn.functional.poisson_nll_loss(t.long(), t)
    ),
    # A bool weight is a mask of the elements that count, to the losses that scale one by weight,
    # and to a class-probability target (the class-index kernels' dtype rule does not apply).
    "binary_cross_entropy-weight": _accepted(
        lambda fw, t, m: fw.nn.functional.binary_cross_entropy(m.float() * 0.5, m.float(), weight=m)
    ),
    "binary_cross_entropy_with_logits-weight": _accepted(
        lambda fw, t, m: fw.nn.functional.binary_cross_entropy_with_logits(
            m.float(), m.float(), weight=m
        )
    ),
    "multilabel_soft_margin_loss-weight": _accepted(
        lambda fw, t, m: fw.nn.functional.multilabel_soft_margin_loss(
            m.float(), m.float(), weight=t[:3]
        )
    ),
    "mse_loss-weight": _accepted(
        lambda fw, t, m: fw.nn.functional.mse_loss(m.float(), m.float() * 0.5, weight=m)
    ),
    "cross_entropy-probabilities-weight": _accepted(
        lambda fw, t, m: fw.nn.functional.cross_entropy(m.float(), m.float() * 0.5, weight=t[:3])
    ),
    "mean-dtype": _accepted(lambda fw, t, m: fw.mean(t, dtype=fw.float32)),
    "mean-dim-dtype": _accepted(lambda fw, t, m: fw.mean(m, dim=1, dtype=fw.float32)),
    "softmax-dtype": _accepted(lambda fw, t, m: fw.softmax(t, 0, dtype=fw.float32)),
    "softmax-dtype-positional": _accepted(lambda fw, t, m: fw.softmax(t, 0, fw.float32)),
    "softmax-dtype-method": _accepted(lambda fw, t, m: t.softmax(0, dtype=fw.float32)),
    "log_softmax-dtype": _accepted(lambda fw, t, m: fw.log_softmax(t, 0, dtype=fw.float32)),
    "softmax-functional-dtype": _accepted(
        lambda fw, t, m: fw.nn.functional.softmax(t, 0, dtype=fw.float32)
    ),
    "clamp-int-bounds": _accepted(lambda fw, t, m: fw.clamp(t, 0, 1)),
    "clamp-float-bound": _accepted(lambda fw, t, m: fw.clamp(t, min=0.5)),
    "clamp-tensor-bound": _accepted(lambda fw, t, m: fw.clamp(t, min=t)),
    "clamp-tensor-bound-max": _accepted(lambda fw, t, m: fw.clamp(t, max=t)),
    "clamp-tensor-bound-inplace": _accepted(lambda fw, t, m: t.clone().clamp_(min=t)),
    # A bool beside a tensor of another dtype meets it there: the result is not bool.
    "clamp-tensor-bounds-int": _accepted(lambda fw, t, m: fw.clamp(t, t, t.long())),
    "pow-bool-exponent": _accepted(lambda fw, t, m: t**True),
    "pow-int-exponent": _accepted(lambda fw, t, m: fw.pow(t, 2)),
    "pow-float-exponent": _accepted(lambda fw, t, m: fw.pow(t, 2.0)),
    "pow-number-base": _accepted(lambda fw, t, m: fw.pow(2, t)),
    "pow-number-base-operator": _accepted(lambda fw, t, m: 2**t),
    "pow-inplace-bool-exponent": _accepted(lambda fw, t, m: t.clone().pow_(True)),
    "pow-int-tensor-exponent": _accepted(lambda fw, t, m: fw.pow(t, t.long())),
    "pow-int-tensor-base": _accepted(lambda fw, t, m: fw.pow(t.long(), t)),
    # P(0, 0) is NaN, which no comparison of values takes; keep the float operand off zero.
    "gammainc-float": _accepted(lambda fw, t, m: fw.special.gammainc(t, t.float() + 1.0)),
    "gammaincc-float": _accepted(lambda fw, t, m: fw.special.gammaincc(t.float() + 1.0, t)),
    # The reference promotes a bool to a float for these, or counts it.
    "exp2": _accepted(lambda fw, t, m: fw.exp2(t)),
    "sigmoid": _accepted(lambda fw, t, m: fw.sigmoid(t)),
    "sort": _accepted(lambda fw, t, m: fw.sort(t)[0]),
    "sum": _accepted(lambda fw, t, m: fw.sum(t)),
    "cov-float": _accepted(lambda fw, t, m: fw.cov(m.float())),
    "abs-int": _accepted(lambda fw, t, m: fw.abs(t.long())),
    "relu-int": _accepted(lambda fw, t, m: fw.relu(t.long())),
}

_VECTOR = [True, False, True, True]
_MATRIX = [[True, False, True], [False, False, True]]


def _flat(result: object) -> list[float]:
    """``result``'s values as one flat list of floats, whichever framework made it."""
    if isinstance(result, (tuple, list)):
        return [v for part in result for v in _flat(part)]
    tolist = getattr(result, "tolist", None)
    return _flat(tolist()) if callable(tolist) else [float(result)]  # type: ignore[arg-type]


def _refusals() -> dict[str, _Row]:
    return {name: row for name, row in _TABLE.items() if row.kind != "accepted"}


def _error_class_matches(kind: Kind, error: BaseException) -> bool:
    """Whether ``error`` is of the class ``kind`` names."""
    if not isinstance(error, RuntimeError):
        return False
    if kind == "kernel":
        return isinstance(error, NotImplementedError)
    if kind == "guard":
        return not isinstance(error, NotImplementedError)
    return True


@pytest.mark.parametrize("name", list(_refusals()))
def test_refusal_is_the_same_on_every_device(device: str, name: str) -> None:
    row = _TABLE[name]
    t = lucid.tensor(_VECTOR, device=device)
    m = lucid.tensor(_MATRIX, device=device)
    with pytest.raises(RuntimeError) as caught:
        row.call(lucid, t, m)
    assert _error_class_matches(row.kind, caught.value), (row.kind, type(caught.value))
    # The message names the op and the cast to make, not a kernel.
    assert "does not accept" in str(caught.value) and "cast to" in str(caught.value)


@pytest.mark.parametrize("name", [n for n, row in _TABLE.items() if row.kind == "accepted"])
def test_the_spelling_the_reference_offers_still_answers(device: str, name: str) -> None:
    t = lucid.tensor(_VECTOR, device=device)
    m = lucid.tensor(_MATRIX, device=device)
    assert _TABLE[name].call(lucid, t, m) is not None


@pytest.mark.parity
@pytest.mark.parametrize("name", [n for n, row in _TABLE.items() if row.parity])
def test_table_row_is_what_the_reference_does_on_cpu(ref: ModuleType, name: str) -> None:
    row = _TABLE[name]
    t = ref.tensor(_VECTOR)
    m = ref.tensor(_MATRIX)
    if row.kind == "accepted":
        row.call(ref, t, m)
        return
    with pytest.raises(RuntimeError) as caught:
        row.call(ref, t, m)
    assert _error_class_matches(row.kind, caught.value), (row.kind, type(caught.value))


@pytest.mark.parity
@pytest.mark.parametrize("name", [n for n, row in _TABLE.items() if row.kind == "accepted"])
def test_accepted_rows_return_the_reference_values(ref: ModuleType, device: str, name: str) -> None:
    call = _TABLE[name].call
    got = call(lucid, lucid.tensor(_VECTOR, device=device), lucid.tensor(_MATRIX, device=device))
    want = call(ref, ref.tensor(_VECTOR), ref.tensor(_MATRIX))
    assert _flat(got) == pytest.approx(_flat(want))
    if hasattr(got, "dtype"):
        assert str(got.dtype).split(".")[-1] == str(want.dtype).split(".")[-1]  # type: ignore[attr-defined]


#: Ops the engine refuses that no ``LUCID_REGISTER_OP`` publishes, so the registry does not list
#: them: ``var`` builds its own forward and ``frac`` is composed (``test_engine_bool_refusal_schema``
#: calls both).  ``std`` is not one of them: it runs through ``var``.
_ENGINE_UNREGISTERED = frozenset({"var", "frac"})


def test_every_op_of_the_table_is_pinned() -> None:
    # The ops that came with the integer refusals are pinned for bool and integers alike in
    # ``test_integer_refusal_table.py``, which replays every row of the Python table.
    owned = set(BOOL_REFUSED) | set(engine_bool_refused()) | _ENGINE_UNREGISTERED
    pinned = {row.op for row in _TABLE.values() if row.op is not None}
    elsewhere = set(BOOL_REFUSED) & PINNED_OPS
    missing = sorted(owned - pinned - elsewhere)
    assert pinned | elsewhere == owned, (
        f"missing a row: {missing}; not in either list: {sorted(pinned - owned)}"
    )


def test_every_loss_that_can_refuse_a_bool_has_a_row() -> None:
    # A loss whose input or target kind leaves bool out, or whose result must be floating, reaches
    # ``bool_refusal`` for its name: a loss of that kind with no row would be a ``KeyError`` there.
    # ``cosine_embedding_loss`` takes bool for every operand, and so has none.  A loss whose answer
    # turns on the call (``_BOOL_BY_CALL``) says ``numeric`` in the table and refuses in its own check.
    can_refuse = (
        {
            loss
            for loss, (input_kind, target_kind) in _OPERAND_DTYPES.items()
            if "bool" not in _KINDS[input_kind]
            or (target_kind and "bool" not in _KINDS[target_kind])
        }
        | _FLOATING_RESULT
        | _BOOL_BY_CALL
    )
    assert can_refuse == set(_LOSSES)
    assert set(_LOSSES) <= set(BOOL_REFUSED)


@pytest.mark.parametrize(
    ("name", "role", "cast"),
    [
        ("cross_entropy-target", "target", _INTEGER),
        ("nll_loss-target", "target", _INTEGER),
        ("multi_margin_loss-target", "target", _INTEGER),
        ("cross_entropy-probabilities", "target", _FLOATING),
        ("kl_div-target", "target", _FLOATING),
        ("cross_entropy-weight", "weight", _FLOATING),
        ("nll_loss-weight", "weight", _FLOATING),
        ("multi_margin_loss-weight", "weight", _FLOATING),
        ("binary_cross_entropy_with_logits-pos_weight", "pos_weight", _FLOATING),
        ("ctc_loss-input_lengths", "input_lengths", _INTEGER),
        ("ctc_loss-target_lengths", "target_lengths", _INTEGER),
    ],
)
def test_a_bool_operand_is_told_the_cast_it_wants(
    device: str, name: str, role: str, cast: str
) -> None:
    # A class label or a length is an integer, class probabilities, the other targets and a weight
    # are floating.
    t = lucid.tensor(_VECTOR, device=device)
    m = lucid.tensor(_MATRIX, device=device)
    with pytest.raises(RuntimeError) as caught:
        _TABLE[name].call(lucid, t, m)
    assert f"a bool {role}" in str(caught.value)
    assert f"cast to {cast}" in str(caught.value)


def test_an_op_has_one_owner() -> None:
    # The engine refuses what its schema declares and Python what has no schema; an entry that
    # reaches an engine op on some paths only (``grid_sample``) asks the schema, not a row.
    assert not set(BOOL_REFUSED) & set(engine_bool_refused())
    assert not set(BOOL_REFUSED) & _ENGINE_UNREGISTERED


def test_the_two_layers_name_the_same_casts() -> None:
    # ``OpSchema.h`` (``cast_hint_text``) and ``_bool_refusal.py`` each spell the casts out; a
    # message that sent the caller to a different dtype by layer would be a defect.  The entries
    # that want one dtype (an int64 class index, a bool mask) have a cast of their own.
    casts = {_FLOATING, _INTEGER, _NUMBER}
    engine_hints = {hint for _, hint in engine_bool_refused().values()}
    assert engine_hints <= casts
    assert {hint for _, hint in BOOL_REFUSED.values()} <= casts | {_INT64, _BOOL_MASK}
    assert engine_hints == {_FLOATING, _NUMBER}


@pytest.mark.parametrize(
    ("code", "name"), [(0, "bool"), (1, "int8"), (2, "int16"), (3, "int32"), (4, "int64")]
)
def test_the_two_layers_spell_the_dtype_alike(code: int, name: str) -> None:
    # The engine says ``int32`` where the enum says ``I32``: a refusal reads the same from either
    # layer, the op aside.  ``leaky_relu`` is the engine's and ``celu`` Python's, both kernel
    # refusals of every dtype here with the floating cast.
    dtype = lucid.bool_ if name == "bool" else getattr(lucid, name)
    engine = engine_refusal("leaky_relu", dtype)
    assert engine is not None
    assert str(refusal("celu", code)) == engine.message.replace("leaky_relu", "celu")
    assert f"does not accept {name} " in str(refusal("celu", code))


def test_the_two_layers_raise_the_same_classes() -> None:
    engine = {_C_engine.NotImplementedError, _C_engine.DtypeMismatch}
    assert {cls for cls, _ in engine_bool_refused().values()} == engine
    # Python adds the engine's argument class: ``one_hot`` and ``affine_grid`` are a Python
    # entry's own check, not a kernel's or a dtype guard's.
    assert {cls for cls, _ in BOOL_REFUSED.values()} == engine | {_C_engine.InvalidArgument}


@pytest.mark.parametrize(
    "build",
    [
        lambda t: lucid.mean(t),
        lambda t: lucid.abs(t),
        lambda t: lucid.softmax(t, 0),
        lambda t: lucid.relu(t),
        lambda t: lucid.quantile(t, 0.5),
        lambda t: lucid.clamp(t, False, True),
    ],
    ids=["mean", "abs", "softmax", "relu", "quantile", "clamp"],
)
def test_a_compiled_bool_op_refuses_as_eager_does(
    device: str, build: Callable[[lucid.Tensor], lucid.Tensor]
) -> None:
    # The trace runs the eager entry, so the refusal fires while the function
    # is traced, before any graph holds the op.
    x = (lucid.randint(0, 2, (4, 6)) == 1).to(device)
    with pytest.raises(RuntimeError, match="does not accept"):
        lucid.compile(build)(x)


def test_a_non_tensor_still_gets_the_engines_own_error() -> None:
    # The refusal looks only at tensors: a number or a list falls through to the
    # engine op, whose ``TypeError`` is the right answer for them.
    for bad in (3.0, [1.0], None):
        with pytest.raises(TypeError):
            lucid.abs(bad)  # type: ignore[arg-type]


#: The in-place spellings of the refused ops, each given the tensor to write.
_IN_PLACE: dict[str, Callable[[lucid.Tensor], object]] = {
    "abs_": lambda t: t.abs_(),
    "relu_": lambda t: t.relu_(),
    "floor_": lambda t: t.floor_(),
    "ceil_": lambda t: t.ceil_(),
    "round_": lambda t: t.round_(),
    "trunc_": lambda t: t.trunc_(),
    "frac_": lambda t: t.frac_(),
    "nextafter_": lambda t: t.nextafter_(t),
    "clamp_": lambda t: t.clamp_(t, t),
    "pow_": lambda t: t.pow_(t),
    "ipow": lambda t: operator.ipow(t, t),
    "relu-inplace": lambda t: lucid.nn.functional.relu(t, inplace=True),
    "hardtanh_": lambda t: lucid.nn.functional.hardtanh_(t),
    "threshold_": lambda t: lucid.nn.functional.threshold_(t, 0.1, 20.0),
    "celu_": lambda t: lucid.nn.functional.celu_(t),
    "rrelu_": lambda t: lucid.nn.functional.rrelu_(t),
    "dropout-inplace": lambda t: lucid.nn.functional.dropout(t, inplace=True),
    "feature_alpha_dropout-inplace": lambda t: lucid.nn.functional.feature_alpha_dropout(
        t.reshape(1, 2, 2), training=True, inplace=True
    ),
}


@pytest.mark.parametrize("name", list(_IN_PLACE))
def test_a_refused_in_place_twin_leaves_the_tensor_as_it_was(device: str, name: str) -> None:
    # The refusal comes before the write: the values, and the version the autograd check of a
    # saved tensor reads, are the ones the caller had.
    t = lucid.tensor(_VECTOR, device=device)
    version = t._impl.version
    with pytest.raises(RuntimeError, match="does not accept"):
        _IN_PLACE[name](t)
    assert t.tolist() == _VECTOR
    assert t._impl.version == version


# ── a bool beside an integer: not a refusal, the two are promoted and searched ──────────────

#: ``(call, what the reference answers)``: the reference promotes the bool and searches at int64.
_PROMOTED_SEARCH: dict[str, tuple[Callable[[ModuleType, object, object], object], list[int]]] = {
    "searchsorted-bool-sequence": (lambda fw, b, i: fw.searchsorted(b, i[:2]), [0, 1]),
    "searchsorted-bool-values": (lambda fw, b, i: fw.searchsorted(i, b), [0, 1]),
    "bucketize-bool-values": (lambda fw, b, i: fw.bucketize(b, i), [0, 1]),
    "bucketize-bool-boundaries": (lambda fw, b, i: fw.bucketize(i, b), [0, 1, 2]),
}


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_PROMOTED_SEARCH))
def test_the_reference_answers_a_bool_beside_an_integer(ref: ModuleType, name: str) -> None:
    call, want = _PROMOTED_SEARCH[name]
    got = call(ref, ref.tensor([False, True]), ref.tensor([0, 1, 2]))
    assert got.tolist() == want  # type: ignore[attr-defined]


@pytest.mark.parametrize("name", list(_PROMOTED_SEARCH))
def test_a_bool_beside_an_integer_is_promoted_not_refused(device: str, name: str) -> None:
    call, want = _PROMOTED_SEARCH[name]
    got = call(
        lucid, lucid.tensor([False, True], device=device), lucid.tensor([0, 1, 2], device=device)
    )
    assert got.tolist() == want  # type: ignore[attr-defined]
