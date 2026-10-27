"""The autocast policy table and how the engine applies it (LCD-500).

``lucid/_C/core/AmpPolicyTable.def`` is the one place an op's autocast
behaviour is decided: one row per op, a Metal column (the reference
framework's MPS list) and a CPU column (its CPU list).  An op with no row
keeps its inputs' dtype.  Before the table each primitive carried one
policy for both devices, and a composite re-applied its parts' policies:
inside ``autocast(metal, float16)`` a float32 ``x * y`` came back float16,
``log1p`` raised ``DtypeMismatch``, and the CPU computed ``exp`` and
``softmax`` in float32 where the reference keeps bfloat16.

The checks:

- the table is sorted, unique and names ops Lucid has;
- every row has a behaviour case, a strict xfail naming the card that
  wires it, or is one of the engine-only rows;
- each wired row gives the dtype it says, for a float32 and a half input,
  under ``autocast(metal, float16)`` and ``autocast(cpu, bfloat16)`` — and,
  with ``-m parity``, the dtype the reference gives;
- every WIDEST row runs a half and a float32 operand in float32;
- autocast is back on after an op returns or raises, since the engine
  turns it off inside the op.
"""

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.amp import autocast
from lucid.test._fixtures.devices import metal_available

_P = _C_engine.AmpPolicy

# Rows whose names are schemas the engine declares but does not register,
# with the engine function that runs each one.
_UNREGISTERED_SCHEMAS = {
    "cumprod": "cumprod",
    "cumsum": "cumsum",
    "dot_1d": "dot",
    "trace": "trace",
}

# Rows applied by an engine function that has no schema, with the function.
_ENGINE_ENTRIES = {
    "fftn": "fft.fftn",
    "ifftn": "fft.ifftn",
    "irfftn": "fft.irfftn",
    "rfftn": "fft.rfftn",
    "scatter_add": "scatter_add",
}

# Rows no public Python spelling reaches.
_ENGINE_ONLY = {"cube", "cube_root"}

# Rows that name a composite rather than a schema, with its public path.
# The engine composites apply their row through ``AmpScope``; the Python
# ones are wired by LCD-500-A1.
_COMPOSITES = {
    "adaptive_avg_pool3d": "lucid.nn.functional.adaptive_avg_pool3d",
    "adaptive_max_pool3d": "lucid.nn.functional.adaptive_max_pool3d",
    "addbmm": "lucid.addbmm",
    "addcdiv": "lucid.addcdiv",
    "addcmul": "lucid.addcmul",
    "addmm": "lucid.addmm",
    "addmv": "lucid.addmv",
    "addr": "lucid.addr",
    "atan2": "lucid.atan2",
    "baddbmm": "lucid.baddbmm",
    "bmm": "lucid.bmm",
    "cdist": "lucid.cdist",
    "cholesky": "lucid.linalg.cholesky",
    "cholesky_ex": "lucid.linalg.cholesky_ex",
    "cond": "lucid.linalg.cond",
    "cosine_embedding_loss": "lucid.nn.functional.cosine_embedding_loss",
    "cosine_similarity": "lucid.nn.functional.cosine_similarity",
    "eig": "lucid.linalg.eig",
    "eigh": "lucid.linalg.eigh",
    "eigvals": "lucid.linalg.eigvals",
    "eigvalsh": "lucid.linalg.eigvalsh",
    "einsum": "lucid.einops.einsum",
    "expm1": "lucid.expm1",
    "fractional_max_pool2d": "lucid.nn.functional.fractional_max_pool2d",
    "fractional_max_pool3d": "lucid.nn.functional.fractional_max_pool3d",
    "gaussian_nll_loss": "lucid.nn.functional.gaussian_nll_loss",
    "hinge_embedding_loss": "lucid.nn.functional.hinge_embedding_loss",
    "householder_product": "lucid.linalg.householder_product",
    "index_copy": "lucid.index_copy",
    "index_put": "lucid.index_put",
    "instance_norm": "lucid.nn.functional.instance_norm",
    "inv_ex": "lucid.linalg.inv_ex",
    "kl_div": "lucid.nn.functional.kl_div",
    "l1_loss": "lucid.nn.functional.l1_loss",
    "local_response_norm": "lucid.nn.functional.local_response_norm",
    "log10": "lucid.log10",
    "log1p": "lucid.log1p",
    "logsumexp": "lucid.logsumexp",
    "lstm_cell": "lucid.nn.LSTMCell",
    "lstsq": "lucid.linalg.lstsq",
    "margin_ranking_loss": "lucid.nn.functional.margin_ranking_loss",
    "matrix_norm": "lucid.linalg.matrix_norm",
    "matrix_rank": "lucid.linalg.matrix_rank",
    "max_unpool2d": "lucid.nn.functional.max_unpool2d",
    "max_unpool3d": "lucid.nn.functional.max_unpool3d",
    "mm": "lucid.mm",
    "multi_dot": "lucid.linalg.multi_dot",
    "multi_margin_loss": "lucid.nn.functional.multi_margin_loss",
    "multilabel_margin_loss": "lucid.nn.functional.multilabel_margin_loss",
    "mv": "lucid.mv",
    "nanquantile": "lucid.nanquantile",
    "poisson_nll_loss": "lucid.nn.functional.poisson_nll_loss",
    "polar": "lucid.polar",
    "prelu": "lucid.nn.functional.prelu",
    "qr": "lucid.linalg.qr",
    "quantile": "lucid.quantile",
    "smooth_l1_loss": "lucid.nn.functional.smooth_l1_loss",
    "soft_margin_loss": "lucid.nn.functional.soft_margin_loss",
    "softmin": "lucid.nn.functional.softmin",
    "stft": "lucid.stft",
    "svd": "lucid.linalg.svd",
    "svdvals": "lucid.linalg.svdvals",
    "tensordot": "lucid.tensordot",
    "triplet_margin_loss": "lucid.nn.functional.triplet_margin_loss",
    "vecdot": "lucid.linalg.vecdot",
    "vector_norm": "lucid.linalg.vector_norm",
}

# The two autocast settings the table is checked under: the Metal column
# with float16 and the CPU column with bfloat16 (CPU float16 computes in
# float32, so it would hide the LOWER rows).
_METAL = pytest.mark.skipif(not metal_available(), reason="Metal unavailable")
_SETTINGS = [
    pytest.param("metal", "float16", id="metal-float16", marks=_METAL),
    pytest.param("cpu", "bfloat16", id="cpu-bfloat16"),
]
_AUTOCAST_DT = {"metal": "float16", "cpu": "bfloat16"}
_INPUTS = ["float32", "half"]


def _rows() -> dict[str, tuple[_C_engine.AmpPolicy, _C_engine.AmpPolicy]]:
    return {name: (metal, cpu) for name, metal, cpu in _C_engine.amp_policy_table()}


def _resolve(path: str) -> object:
    module_name, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


def _engine_attr(path: str) -> object:
    obj: object = _C_engine
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


# ── the table itself ─────────────────────────────────────────────────────────


def test_rows_are_sorted_unique_and_valid() -> None:
    names = [name for name, _, _ in _C_engine.amp_policy_table()]
    assert names == sorted(names)
    assert len(names) == len(set(names))
    valid = {_P.Promote, _P.KeepInput, _P.ForceFP32, _P.PromoteWidest}
    for name, (metal, cpu) in _rows().items():
        assert metal in valid and cpu in valid, name
        # A row that keeps on both devices is the default, written out.
        assert (metal, cpu) != (_P.KeepInput, _P.KeepInput), name


def test_every_row_names_an_op_lucid_has() -> None:
    for name in _rows():
        if _C_engine.op_lookup(name) is not None:
            continue
        if name in _UNREGISTERED_SCHEMAS:
            assert hasattr(_C_engine, _UNREGISTERED_SCHEMAS[name]), name
        elif name in _ENGINE_ENTRIES:
            assert callable(_engine_attr(_ENGINE_ENTRIES[name])), name
        else:
            assert name in _COMPOSITES, f"{name}: no schema and no composite of that name"
            assert callable(_resolve(_COMPOSITES[name])), name


def test_the_name_lists_hold_only_rows() -> None:
    rows = _rows()
    for name in (*_COMPOSITES, *_ENGINE_ENTRIES, *_UNREGISTERED_SCHEMAS):
        # A name that gains a registered schema leaves these lists.
        assert name in rows and _C_engine.op_lookup(name) is None, name


def test_amp_policy_for_reads_the_table() -> None:
    for name, (metal, cpu) in _rows().items():
        assert _C_engine.amp_policy_for(name, _C_engine.Device.GPU) == metal, name
        assert _C_engine.amp_policy_for(name, _C_engine.Device.CPU) == cpu, name
    for device in (_C_engine.Device.GPU, _C_engine.Device.CPU):
        assert _C_engine.amp_policy_for("not_an_op", device) == _P.KeepInput


def test_schemas_report_the_table_not_their_declaration() -> None:
    rows = _rows()
    for schema in _C_engine.op_registry_all():
        metal, cpu = rows.get(schema.name, (_P.KeepInput, _P.KeepInput))
        assert (schema.amp_policy, schema.amp_policy_cpu) == (metal, cpu), schema.name
    # ``add`` was declared Promote; it has no row, so autocast keeps it.
    add = _C_engine.op_lookup("add")
    assert add.declared_amp_policy == _P.Promote
    assert (add.amp_policy, add.amp_policy_cpu) == (_P.KeepInput, _P.KeepInput)


# ── cases: one call per op, the same code for Lucid and the reference ───────


class _Make:
    """Inputs for one case, in one library, on one device and dtype."""

    def __init__(self, lib: ModuleType, device: str, dtype: object) -> None:
        self.lib, self.device, self.dtype = lib, device, dtype

    def __call__(self, *shape: int, pos: bool = False) -> Any:
        t = self.lib.rand(*shape, device=self.device) + (0.5 if pos else -0.5)
        return t.to(self.dtype)

    def unit(self, *shape: int) -> Any:
        """Values in (0.1, 0.9) — probabilities."""
        return (self.lib.rand(*shape, device=self.device) * 0.8 + 0.1).to(self.dtype)

    def signs(self, n: int) -> Any:
        return self.lib.tensor([1.0, -1.0] * (n // 2), device=self.device).to(self.dtype)

    def ints(self, values: list[Any]) -> Any:
        return self.lib.tensor(values, device=self.device)

    def spd(self, n: int) -> Any:
        """A symmetric, diagonally dominant matrix — built without a matmul,
        which is LOWER and would change the dtype under test."""
        a = self(n, n)
        return a + a.mT + self.lib.eye(n, device=self.device).to(self.dtype) * 4


_Fn = Callable[[ModuleType, Any, _Make], Any]


@dataclass(frozen=True)
class _Case:
    fn: _Fn
    # The reference's spelling, where it differs from Lucid's.
    ref_fn: _Fn | None = None
    # Where the row is checked.  A CPU-only row (a KEEP Metal column) skips
    # Metal, where half linear algebra and complex results are the
    # kernel's business, not autocast's.
    devices: tuple[str, ...] = ("metal", "cpu")
    # The output dtype when the op's output is not floating (a rank).
    want: str | None = None


def _bn(train: bool, rank: int) -> _Fn:
    def fn(lib: ModuleType, F: Any, m: _Make) -> Any:
        x = m(2, 3, *([4] * (rank - 2)))
        if train:
            return F.batch_norm(x, None, None, m(3), m(3), training=True)
        return F.batch_norm(x, m(3), m(3, pos=True), m(3), m(3), training=False)

    return fn


def _cpu(fn: _Fn, want: str | None = None) -> _Case:
    return _Case(fn, devices=("cpu",), want=want)


_CASES: dict[str, _Case] = {
    # ── LOWER ──
    "affine_grid": _Case(
        lambda lib, F, m: F.affine_grid(m(1, 2, 3), (1, 1, 4, 4), align_corners=False)
    ),
    "bmm": _Case(lambda lib, F, m: lib.bmm(m(1, 3, 4), m(1, 4, 2))),
    "conv1d": _Case(lambda lib, F, m: F.conv1d(m(1, 2, 8), m(3, 2, 3))),
    "conv2d": _Case(lambda lib, F, m: F.conv2d(m(1, 2, 8, 8), m(3, 2, 3, 3))),
    "conv3d": _Case(lambda lib, F, m: F.conv3d(m(1, 2, 4, 4, 4), m(3, 2, 3, 3, 3))),
    "conv_transpose1d": _Case(lambda lib, F, m: F.conv_transpose1d(m(1, 2, 8), m(2, 3, 3))),
    "conv_transpose2d": _Case(lambda lib, F, m: F.conv_transpose2d(m(1, 2, 8, 8), m(2, 3, 3, 3))),
    "einsum": _Case(
        lambda lib, F, m: lib.einops.einsum("ij,jk->ik", m(3, 4), m(4, 2)),
        lambda lib, F, m: lib.einsum("ij,jk->ik", m(3, 4), m(4, 2)),
    ),
    "linear": _Case(lambda lib, F, m: F.linear(m(2, 4), m(3, 4), m(3))),
    "lstm_cell": _Case(lambda lib, F, m: lib.nn.LSTMCell(4, 5).to(m.device)(m(2, 4))[0]),
    "matmul": _Case(lambda lib, F, m: m(3, 4) @ m(4, 2)),
    "mm": _Case(lambda lib, F, m: lib.mm(m(3, 4), m(4, 2))),
    "multi_dot": _Case(lambda lib, F, m: lib.linalg.multi_dot([m(3, 4), m(4, 2), m(2, 2)])),
    "scaled_dot_product_attention": _Case(
        lambda lib, F, m: F.scaled_dot_product_attention(
            m(1, 2, 4, 8), m(1, 2, 4, 8), m(1, 2, 4, 8), attn_mask=m(4, 4)
        )
    ),
    # ── FP32 on Metal, LOWER on the CPU ──
    "conv_transpose3d": _Case(
        lambda lib, F, m: F.conv_transpose3d(m(1, 2, 4, 4, 4), m(2, 3, 3, 3, 3))
    ),
    # ── FP32 on both ──
    "bce_with_logits": _Case(lambda lib, F, m: F.binary_cross_entropy_with_logits(m(4), m.unit(4))),
    "cosine_embedding_loss": _Case(
        lambda lib, F, m: F.cosine_embedding_loss(m(2, 4), m(2, 4), m.signs(2))
    ),
    "cross_entropy_loss": _Case(lambda lib, F, m: F.cross_entropy(m(3, 5), m.ints([0, 2, 4]))),
    "ctc_loss": _Case(
        lambda lib, F, m: F.ctc_loss(
            m(6, 1, 4).log_softmax(-1), m.ints([[1, 2]]), lib.tensor([6]), lib.tensor([2])
        )
    ),
    "hinge_embedding_loss": _Case(lambda lib, F, m: F.hinge_embedding_loss(m(4), m.signs(4))),
    "huber_loss": _Case(lambda lib, F, m: F.huber_loss(m(4), m(4))),
    "kl_div": _Case(lambda lib, F, m: F.kl_div(m(2, 4), m(2, 4, pos=True), reduction="batchmean")),
    "l1_loss": _Case(lambda lib, F, m: F.l1_loss(m(4), m(4))),
    "margin_ranking_loss": _Case(lambda lib, F, m: F.margin_ranking_loss(m(4), m(4), m.signs(4))),
    "mse_loss": _Case(lambda lib, F, m: F.mse_loss(m(4), m(4))),
    "multi_margin_loss": _Case(lambda lib, F, m: F.multi_margin_loss(m(3, 5), m.ints([0, 2, 4]))),
    "multilabel_margin_loss": _Case(
        lambda lib, F, m: F.multilabel_margin_loss(m(2, 4), m.ints([[3, 0, -1, 1], [0, 1, 2, -1]]))
    ),
    "nll_loss": _Case(lambda lib, F, m: F.nll_loss(m(3, 5), m.ints([0, 2, 4]))),
    "norm": _Case(lambda lib, F, m: lib.linalg.norm(m(4))),
    "prod": _Case(lambda lib, F, m: m(4, 3).prod(0)),
    "smooth_l1_loss": _Case(lambda lib, F, m: F.smooth_l1_loss(m(4), m(4))),
    "trace": _Case(lambda lib, F, m: lib.trace(m(3, 3))),
    "triplet_margin_loss": _Case(
        lambda lib, F, m: F.triplet_margin_loss(m(2, 4), m(2, 4), m(2, 4))
    ),
    # ── FP32 on Metal, KEEP on the CPU ──
    "arccos": _Case(lambda lib, F, m: lib.arccos(m(4))),
    "arcsin": _Case(lambda lib, F, m: lib.arcsin(m(4))),
    "batch_norm": _Case(_bn(train=True, rank=4)),
    "batch_norm1d": _Case(_bn(train=True, rank=3)),
    "batch_norm3d": _Case(_bn(train=True, rank=5)),
    "batch_norm_eval": _Case(_bn(train=False, rank=4)),
    "cosh": _Case(lambda lib, F, m: lib.cosh(m(4))),
    "cosine_similarity": _Case(lambda lib, F, m: F.cosine_similarity(m(2, 4), m(2, 4))),
    "cumprod": _Case(lambda lib, F, m: lib.cumprod(m(4), 0)),
    "cumsum": _Case(lambda lib, F, m: lib.cumsum(m(4), 0)),
    "erfinv": _Case(lambda lib, F, m: lib.erfinv(m(4))),
    "exp": _Case(lambda lib, F, m: lib.exp(m(4))),
    "expm1": _Case(lambda lib, F, m: lib.expm1(m(4))),
    "group_norm": _Case(lambda lib, F, m: F.group_norm(m(2, 4, 3), 2)),
    "layer_norm": _Case(lambda lib, F, m: F.layer_norm(m(2, 4), (4,))),
    "local_response_norm": _Case(lambda lib, F, m: F.local_response_norm(m(1, 4, 3), 2)),
    "log": _Case(lambda lib, F, m: lib.log(m(4, pos=True))),
    "log10": _Case(lambda lib, F, m: lib.log10(m(4, pos=True))),
    "log1p": _Case(lambda lib, F, m: lib.log1p(m(4, pos=True))),
    "log2": _Case(lambda lib, F, m: lib.log2(m(4, pos=True))),
    "log_softmax": _Case(lambda lib, F, m: F.log_softmax(m(4), 0)),
    "logsumexp": _Case(lambda lib, F, m: lib.logsumexp(m(4), 0)),
    "pow": _Case(lambda lib, F, m: lib.pow(m(4, pos=True), m(4))),
    "pow_scalar": _Case(lambda lib, F, m: m(4, pos=True) ** 2.5),
    "reciprocal": _Case(lambda lib, F, m: lib.reciprocal(m(4, pos=True))),
    "rpow_scalar": _Case(lambda lib, F, m: 2.0 ** m(4)),
    "rsqrt": _Case(lambda lib, F, m: lib.rsqrt(m(4, pos=True))),
    "sinh": _Case(lambda lib, F, m: lib.sinh(m(4))),
    "softmax": _Case(lambda lib, F, m: F.softmax(m(4), 0)),
    "softmin": _Case(lambda lib, F, m: F.softmin(m(4), 0)),
    "square": _Case(lambda lib, F, m: lib.square(m(4))),
    "sum": _Case(lambda lib, F, m: m(4, 3).sum(0)),
    "tan": _Case(lambda lib, F, m: lib.tan(m(4))),
    # ── KEEP on Metal, FP32 on the CPU ──
    "avg_pool3d": _Case(lambda lib, F, m: F.avg_pool3d(m(1, 2, 4, 4, 4), 2)),
    "cholesky": _cpu(lambda lib, F, m: lib.linalg.cholesky(m.spd(3))),
    "cholesky_ex": _cpu(lambda lib, F, m: lib.linalg.cholesky_ex(m.spd(3))[0]),
    "cond": _cpu(lambda lib, F, m: lib.linalg.cond(m.spd(3))),
    "eig": _cpu(lambda lib, F, m: lib.linalg.eig(m.spd(3))[0].abs()),
    "eigh": _cpu(lambda lib, F, m: lib.linalg.eigh(m.spd(3))[0]),
    "eigvals": _cpu(lambda lib, F, m: lib.linalg.eigvals(m.spd(3)).abs()),
    "eigvalsh": _cpu(lambda lib, F, m: lib.linalg.eigvalsh(m.spd(3))),
    "fftn": _cpu(lambda lib, F, m: lib.fft.fft(m(8)).abs()),
    "grid_sample": _Case(
        lambda lib, F, m: F.grid_sample(m(1, 1, 4, 4), m(1, 3, 3, 2), align_corners=False)
    ),
    "householder_product": _cpu(lambda lib, F, m: lib.linalg.householder_product(m(3, 2), m(2))),
    "ifftn": _cpu(lambda lib, F, m: lib.fft.ifft(m(8)).abs()),
    "inv": _cpu(lambda lib, F, m: lib.linalg.inv(m.spd(3))),
    "inv_ex": _cpu(lambda lib, F, m: lib.linalg.inv_ex(m.spd(3))[0]),
    "irfftn": _cpu(lambda lib, F, m: lib.fft.irfft(m(8))),
    "matrix_rank": _cpu(lambda lib, F, m: lib.linalg.matrix_rank(m.spd(3)), want="int64"),
    "max_pool3d": _Case(lambda lib, F, m: F.max_pool3d(m(1, 2, 4, 4, 4), 2)),
    "polar": _cpu(lambda lib, F, m: lib.polar(m(3, pos=True), m(3)).abs()),
    "qr": _cpu(lambda lib, F, m: lib.linalg.qr(m(3, 3))[0]),
    "rfftn": _cpu(lambda lib, F, m: lib.fft.rfft(m(8)).abs()),
    "solve": _cpu(lambda lib, F, m: lib.linalg.solve(m.spd(3), m(3))),
    "stft": _cpu(lambda lib, F, m: lib.stft(m(64), 16, return_complex=True).abs()),
    "svd": _cpu(lambda lib, F, m: lib.linalg.svd(m(3, 3))[1]),
    "svdvals": _cpu(lambda lib, F, m: lib.linalg.svdvals(m(3, 3))),
    "view_as_complex": _cpu(lambda lib, F, m: lib.view_as_complex(m(3, 2)).abs()),
    # ── WIDEST: one dtype in, that dtype out (mixed operands below) ──
    "addcdiv": _Case(lambda lib, F, m: lib.addcdiv(m(3), m(3), m(3, pos=True))),
    "addcmul": _Case(lambda lib, F, m: lib.addcmul(m(3), m(3), m(3))),
    "atan2": _Case(lambda lib, F, m: lib.atan2(m(4), m(4))),
    "bilinear_layer": _Case(lambda lib, F, m: F.bilinear(m(2, 3), m(2, 4), m(5, 3, 4))),
    "concatenate": _Case(lambda lib, F, m: lib.cat([m(4), m(4)])),
    "dot_1d": _Case(
        lambda lib, F, m: lib.linalg.dot(m(4), m(4)), lambda lib, F, m: lib.dot(m(4), m(4))
    ),
    "index_copy": _Case(lambda lib, F, m: lib.index_copy(m(4), 0, m.ints([0, 2]), m(2))),
    "index_put": _Case(
        lambda lib, F, m: lib.index_put(m(4), (m.ints([0, 2, 1]),), m(3)),
        lambda lib, F, m: m(4).index_put((m.ints([0, 2, 1]),), m(3)),
    ),
    "scatter_add": _Case(lambda lib, F, m: m(4).scatter_add(0, m.ints([0, 2, 1]), m(3))),
    "stack": _Case(lambda lib, F, m: lib.stack([m(4), m(4)])),
    # WIDEST on Metal, LOWER on the CPU.
    "tensordot": _Case(lambda lib, F, m: lib.tensordot(m(3, 4), m(4, 2), 1)),
    # ── Python composites wired via _autocast_entry (A1) ─────────────
    "adaptive_avg_pool3d": _cpu(lambda lib, F, m: F.adaptive_avg_pool3d(m(1, 2, 5, 5, 5), 2)),
    "adaptive_max_pool3d": _cpu(lambda lib, F, m: F.adaptive_max_pool3d(m(1, 2, 5, 5, 5), 2)),
    "addbmm": _Case(lambda lib, F, m: lib.addbmm(m(3, 2), m(1, 3, 4), m(1, 4, 2))),
    "addmm": _Case(lambda lib, F, m: lib.addmm(m(3, 2), m(3, 4), m(4, 2))),
    "addmv": _Case(lambda lib, F, m: lib.addmv(m(3), m(3, 4), m(4))),
    "addr": _Case(lambda lib, F, m: lib.addr(m(3, 4), m(3), m(4))),
    "baddbmm": _Case(lambda lib, F, m: lib.baddbmm(m(1, 3, 2), m(1, 3, 4), m(1, 4, 2))),
    "bce_loss": _Case(lambda lib, F, m: F.binary_cross_entropy(m.unit(4), m.unit(4))),
    "cdist": _Case(lambda lib, F, m: lib.cdist(m(3, 4), m(2, 4))),
    "fractional_max_pool2d": _cpu(
        lambda lib, F, m: F.fractional_max_pool2d(m(1, 1, 8, 8), 2, output_size=4)
    ),
    "fractional_max_pool3d": _cpu(
        lambda lib, F, m: F.fractional_max_pool3d(m(1, 1, 8, 8, 8), 2, output_size=4)
    ),
    "gaussian_nll_loss": _Case(lambda lib, F, m: F.gaussian_nll_loss(m(4), m(4), m(4, pos=True))),
    "instance_norm": _Case(lambda lib, F, m: F.instance_norm(m(2, 3, 4))),
    "lstsq": _cpu(lambda lib, F, m: lib.linalg.lstsq(m.spd(3), m(3, 1))[0]),
    "matrix_norm": _Case(lambda lib, F, m: lib.linalg.matrix_norm(m(3, 3))),
    "max_unpool2d": _cpu(
        lambda lib, F, m: F.max_unpool2d(*F.max_pool2d(m(1, 1, 4, 4), 2, return_indices=True), 2)
    ),
    "max_unpool3d": _cpu(
        lambda lib, F, m: F.max_unpool3d(
            m(1, 1, 2, 2, 2), m.ints([[[[[0, 1], [8, 9]], [[32, 33], [40, 41]]]]]), 2
        )
    ),
    "mv": _Case(lambda lib, F, m: lib.mv(m(3, 4), m(4))),
    "nanquantile": _cpu(lambda lib, F, m: lib.nanquantile(m(4), 0.5)),
    "poisson_nll_loss": _Case(lambda lib, F, m: F.poisson_nll_loss(m(4), m(4, pos=True))),
    "prelu": _Case(lambda lib, F, m: F.prelu(m(1, 2, 3), m(2))),
    "quantile": _cpu(lambda lib, F, m: lib.quantile(m(4), 0.5)),
    "soft_margin_loss": _Case(lambda lib, F, m: F.soft_margin_loss(m(4), m.signs(4))),
    "softplus": _Case(lambda lib, F, m: F.softplus(m(4))),
    "vecdot": _Case(lambda lib, F, m: lib.linalg.vecdot(m(3, 4), m(3, 4))),
    "vector_norm": _Case(lambda lib, F, m: lib.linalg.vector_norm(m(4))),
}

# Ops with no row keep their inputs' dtype — several of them were LOWER or
# FP32 for both devices before the table.
_KEEP_CASES: dict[str, _Case] = {
    "abs": _Case(lambda lib, F, m: lib.abs(m(4))),
    "add": _Case(lambda lib, F, m: m(4) + m(4)),
    "cross": _Case(lambda lib, F, m: lib.linalg.cross(m(3), m(3))),
    "div": _Case(lambda lib, F, m: m(4) / m(4, pos=True)),
    "elu": _Case(lambda lib, F, m: F.elu(m(4))),
    "exp2": _Case(lambda lib, F, m: lib.exp2(m(4))),
    "floordiv": _Case(lambda lib, F, m: lib.floor_divide(m(4), m(4, pos=True))),
    "fmod": _Case(lambda lib, F, m: lib.fmod(m(4), m(4, pos=True))),
    "frac": _Case(lambda lib, F, m: lib.frac(m(4))),
    "gelu": _Case(lambda lib, F, m: F.gelu(m(4))),
    "gru": _cpu(lambda lib, F, m: lib.nn.GRU(4, 5).to(m.device)(m(3, 2, 4))[0]),
    "gru_cell": _Case(lambda lib, F, m: lib.nn.GRUCell(4, 5).to(m.device)(m(2, 4))),
    "hypot": _Case(lambda lib, F, m: lib.hypot(m(4), m(4))),
    "interpolate_bilinear": _Case(
        lambda lib, F, m: F.interpolate(m(1, 1, 4, 4), scale_factor=2, mode="bilinear")
    ),
    "kron": _Case(lambda lib, F, m: lib.kron(m(2, 2), m(2, 2))),
    "logaddexp": _Case(lambda lib, F, m: lib.logaddexp(m(4), m(4))),
    "lp_normalize": _Case(lambda lib, F, m: F.normalize(m(3, 4))),
    "lstm": _cpu(lambda lib, F, m: lib.nn.LSTM(4, 5).to(m.device)(m(3, 2, 4))[0]),
    "maximum": _Case(lambda lib, F, m: lib.maximum(m(4), m(4))),
    "mean": _Case(lambda lib, F, m: m(4).mean()),
    "mish": _Case(lambda lib, F, m: F.mish(m(4))),
    "mul": _Case(lambda lib, F, m: m(4) * m(4)),
    "mul_scalar": _Case(lambda lib, F, m: m(4) * 0.5),
    "neg": _Case(lambda lib, F, m: -m(4)),
    "rms_norm": _Case(lambda lib, F, m: F.rms_norm(m(2, 4), (4,))),
    "selu": _Case(lambda lib, F, m: F.selu(m(4))),
    "sigmoid": _Case(lambda lib, F, m: lib.sigmoid(m(4))),
    "silu": _Case(lambda lib, F, m: F.silu(m(4))),
    "sin": _Case(lambda lib, F, m: lib.sin(m(4))),
    "sqrt": _Case(lambda lib, F, m: lib.sqrt(m(4, pos=True))),
    "sub": _Case(lambda lib, F, m: m(4) - m(4)),
    "tanh": _Case(lambda lib, F, m: lib.tanh(m(4))),
    "vdot": _Case(lambda lib, F, m: lib.vdot(m(4), m(4))),
}

_POLICY_DTYPE = {
    _P.Promote: lambda autocast_dt, input_dt: autocast_dt,
    _P.ForceFP32: lambda autocast_dt, input_dt: "float32",
    _P.KeepInput: lambda autocast_dt, input_dt: input_dt,
    _P.PromoteWidest: lambda autocast_dt, input_dt: input_dt,
}


def _name(dtype: object) -> str:
    return str(dtype).split(".")[-1]


def _run_lucid(case: _Case, device: str, autocast_dt: str, input_dt: str) -> str:
    dtype = getattr(lucid, autocast_dt if input_dt == "half" else "float32")
    lucid.manual_seed(0)
    with autocast(device_type=device, dtype=getattr(lucid, autocast_dt)):
        out = case.fn(lucid, lucid.nn.functional, _Make(lucid, device, dtype))
    return _name(out.dtype)


def _want(name: str, case: _Case, device: str, autocast_dt: str, input_dt: str) -> str:
    if case.want is not None:
        return case.want
    metal, cpu = _rows().get(name, (_P.KeepInput, _P.KeepInput))
    policy = metal if device == "metal" else cpu
    return _POLICY_DTYPE[policy](autocast_dt, autocast_dt if input_dt == "half" else "float32")


def _check(name: str, case: _Case, device: str, autocast_dt: str, input_dt: str) -> None:
    if device not in case.devices:
        pytest.skip(f"{name} is checked on {case.devices} only")
    got = _run_lucid(case, device, autocast_dt, input_dt)
    assert got == _want(name, case, device, autocast_dt, input_dt)


def test_every_row_has_a_case() -> None:
    covered = set(_CASES) | _ENGINE_ONLY
    assert set(_rows()) == covered, {
        "rows without a case": set(_rows()) - covered,
        "cases without a row": covered - set(_rows()),
    }
    assert not set(_KEEP_CASES) & set(_rows())


@pytest.mark.parametrize("input_dt", _INPUTS)
@pytest.mark.parametrize(("device", "autocast_dt"), _SETTINGS)
@pytest.mark.parametrize("name", sorted(_CASES))
def test_a_row_gives_its_dtype(name: str, device: str, autocast_dt: str, input_dt: str) -> None:
    _check(name, _CASES[name], device, autocast_dt, input_dt)


@pytest.mark.parametrize("input_dt", _INPUTS)
@pytest.mark.parametrize(("device", "autocast_dt"), _SETTINGS)
@pytest.mark.parametrize("name", sorted(_KEEP_CASES))
def test_an_op_without_a_row_keeps_its_dtype(
    name: str, device: str, autocast_dt: str, input_dt: str
) -> None:
    _check(name, _KEEP_CASES[name], device, autocast_dt, input_dt)


@pytest.mark.parametrize(("device", "autocast_dt"), _SETTINGS)
@pytest.mark.parametrize("name", sorted(_ENGINE_ONLY))
def test_an_engine_only_row_gives_its_dtype(name: str, device: str, autocast_dt: str) -> None:
    """``cube`` and ``cube_root`` have no public spelling; they follow ``pow``."""
    engine_dtype = {"float32": _C_engine.F32, "float16": _C_engine.F16, "bfloat16": _C_engine.BF16}
    for input_dt in _INPUTS:
        dtype = getattr(lucid, autocast_dt if input_dt == "half" else "float32")
        x = lucid.rand(4, device=device).to(dtype)
        with autocast(device_type=device, dtype=getattr(lucid, autocast_dt)):
            out = getattr(_C_engine, name)(x._impl)
        want = _want(name, _Case(lambda lib, F, m: None), device, autocast_dt, input_dt)
        assert out.dtype == engine_dtype[want], input_dt


# ── WIDEST rows with mixed operands ──────────────────────────────────────────

_MixedFn = Callable[[ModuleType, Any, _Make, _Make], Any]

# ``h`` makes the autocast's half dtype, ``f`` float32.
_MIXED: dict[str, _MixedFn] = {
    "addcdiv": lambda lib, F, h, f: lib.addcdiv(h(3), f(3), f(3, pos=True)),
    "addcmul": lambda lib, F, h, f: lib.addcmul(h(3), f(3), f(3)),
    "atan2": lambda lib, F, h, f: lib.atan2(h(4), f(4)),
    "bilinear_layer": lambda lib, F, h, f: F.bilinear(h(2, 3), f(2, 4), f(5, 3, 4), f(5)),
    "concatenate": lambda lib, F, h, f: lib.cat([h(4), f(4)]),
    "dot_1d": lambda lib, F, h, f: lib.linalg.dot(h(4), f(4)),
    "grid_sample": lambda lib, F, h, f: F.grid_sample(
        h(1, 1, 4, 4), f(1, 3, 3, 2), align_corners=False
    ),
    "index_copy": lambda lib, F, h, f: lib.index_copy(h(4), 0, h.ints([0, 2]), f(2)),
    "index_put": lambda lib, F, h, f: lib.index_put(h(4), (h.ints([0, 2]),), f(2)),
    "scatter_add": lambda lib, F, h, f: h(4).scatter_add(0, h.ints([0, 2, 1]), f(3)),
    "stack": lambda lib, F, h, f: lib.stack([h(4), f(4)]),
    "tensordot": lambda lib, F, h, f: lib.tensordot(h(3, 4), f(4, 2), 1),
}


def _widest_columns() -> list[Any]:
    params = []
    for name, (metal, cpu) in sorted(_rows().items()):
        for device, policy in (("metal", metal), ("cpu", cpu)):
            if policy != _P.PromoteWidest:
                continue
            marks = [_METAL] if device == "metal" else []
            params.append(pytest.param(name, device, id=f"{name}-{device}", marks=marks))
    return params


def test_every_widest_row_has_a_mixed_case() -> None:
    widest = {name for name, (metal, cpu) in _rows().items() if _P.PromoteWidest in (metal, cpu)}
    assert widest == set(_MIXED)


@pytest.mark.parametrize(("name", "device"), _widest_columns())
def test_a_widest_row_runs_mixed_operands_in_float32(name: str, device: str) -> None:
    half = getattr(lucid, _AUTOCAST_DT[device])
    lucid.manual_seed(0)
    with autocast(device_type=device, dtype=half):
        out = _MIXED[name](
            lucid,
            lucid.nn.functional,
            _Make(lucid, device, half),
            _Make(lucid, device, lucid.float32),
        )
    assert out.dtype == lucid.float32


@pytest.mark.parametrize("device", [pytest.param("metal", marks=_METAL), "cpu"])
def test_cosine_similarity_promotes_mixed_inputs(device: str) -> None:
    """The composite promotes two floating dtypes as a binary op would —
    outside autocast too, where it used to raise ``DtypeMismatch``."""
    half = getattr(lucid, _AUTOCAST_DT[device])
    a = lucid.randn(2, 4, device=device).to(half)
    b = lucid.randn(2, 4, device=device)
    want = lucid.nn.functional.cosine_similarity(a.float(), b)
    got = lucid.nn.functional.cosine_similarity(a, b)
    with autocast(device_type=device, dtype=half):
        inside = lucid.nn.functional.cosine_similarity(a, b)
    assert got.dtype == inside.dtype == lucid.float32
    assert float((got - want).abs().max().item()) < 1e-6
    assert float((inside - want).abs().max().item()) < 1e-6


@pytest.mark.parametrize("device", [pytest.param("metal", marks=_METAL), "cpu"])
def test_sdpa_casts_q_k_v_and_an_additive_mask_together(device: str) -> None:
    """RoFormer hands attention a float32 Q and K (rotated by float32
    tables) and a half V; the LOWER row casts all of them, and an additive
    mask, to the autocast dtype rather than refusing the mixed set."""
    half = getattr(lucid, _AUTOCAST_DT[device])
    q = lucid.randn(1, 2, 4, 8, device=device)
    v = lucid.randn(1, 2, 4, 8, device=device).to(half)
    mask = lucid.randn(4, 4, device=device)
    keep = lucid.rand(4, 4, device=device) > 0.3
    F = lucid.nn.functional
    with autocast(device_type=device, dtype=half):
        additive = F.scaled_dot_product_attention(q, q, v, attn_mask=mask)
        selecting = F.scaled_dot_product_attention(q, q, v, attn_mask=keep)
    assert additive.dtype == selecting.dtype == half


@_METAL
def test_bilinear_bias_joins_the_widest_row() -> None:
    """``F.bilinear`` hands the bias to the engine with the other operands,
    so under Metal autocast a float32 bias beside float16 inputs runs the
    whole op in float32 — it used to be added afterwards, outside the row."""
    lucid.manual_seed(0)
    x1, x2 = lucid.randn(2, 3, device="metal"), lucid.randn(2, 4, device="metal")
    w, b = lucid.randn(5, 3, 4, device="metal"), lucid.randn(5, device="metal")
    half = lucid.float16
    with autocast(device_type="metal", dtype=half):
        out = lucid.nn.functional.bilinear(x1.to(half), x2.to(half), w.to(half), b)
    assert out.dtype == lucid.float32
    want = lucid.nn.functional.bilinear(
        x1.to(half).float(), x2.to(half).float(), w.to(half).float(), b
    )
    assert float((out - want).abs().max().item()) < 1e-5


@pytest.mark.parametrize("device", [pytest.param("metal", marks=_METAL), "cpu"])
def test_bilinear_with_bias_matches_the_unbiased_op_plus_bias(device: str) -> None:
    lucid.manual_seed(0)
    x1 = lucid.randn(2, 3, device=device, requires_grad=True)
    x2 = lucid.randn(2, 4, device=device, requires_grad=True)
    w = lucid.randn(5, 3, 4, device=device, requires_grad=True)
    b = lucid.randn(5, device=device, requires_grad=True)
    out = lucid.nn.functional.bilinear(x1, x2, w, b)
    want = lucid.nn.functional.bilinear(x1, x2, w) + b
    assert float((out - want).abs().max().item()) < 1e-6
    grads = lucid.autograd.grad(out.sum(), [x1, x2, w, b])
    want_grads = lucid.autograd.grad(want.sum(), [x1, x2, w, b])
    for g, h in zip(grads, want_grads, strict=True):
        assert float((g - h).abs().max().item()) < 1e-6


@_METAL
def test_bilinear_refuses_a_bias_on_another_device() -> None:
    x1, x2 = lucid.randn(2, 3, device="metal"), lucid.randn(2, 4, device="metal")
    w = lucid.randn(5, 3, 4, device="metal")
    with pytest.raises(_C_engine.DeviceMismatch):
        lucid.nn.functional.bilinear(x1, x2, w, lucid.randn(5))


def test_bilinear_refuses_a_bias_of_another_dtype() -> None:
    x1, x2, w = lucid.randn(2, 3), lucid.randn(2, 4), lucid.randn(5, 3, 4)
    with pytest.raises(_C_engine.DtypeMismatch):
        lucid.nn.functional.bilinear(x1, x2, w, lucid.randn(5, dtype=lucid.float64))


# ── documented deviations ────────────────────────────────────────────────────


def test_a_cnn_in_eval_mode_runs_under_a_cpu_bfloat16_autocast() -> None:
    """The conv is LOWER and hands batch_norm_eval a bfloat16 activation,
    which the CPU column keeps, as the reference's does.  Until LCD-487 the
    CPU eval kernel had no half path and the row was a FP32 departure."""
    model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU()).eval()
    x = lucid.randn(2, 3, 8, 8)
    with autocast(device_type="cpu", dtype=lucid.bfloat16):
        y = model(x)
    assert y.dtype == lucid.bfloat16
    with lucid.no_grad():
        want = model(x)
    assert float((y.float() - want).abs().max().item()) < 5e-2


@pytest.mark.xfail(
    strict=True,
    reason="LCD-500 E2: pad is one schema for every mode, so the reference's FP32 "
    "reflect / replicate rows have no row here (AmpPolicyTable.def header)",
)
@pytest.mark.parametrize("mode", ["reflect", "replicate"])
def test_cpu_reflect_and_replicate_pad_run_in_float32(mode: str) -> None:
    x = lucid.randn(1, 2, 4).to(lucid.bfloat16)
    with autocast(device_type="cpu", dtype=lucid.bfloat16):
        out = lucid.nn.functional.pad(x, (1, 1), mode=mode)
    assert out.dtype == lucid.float32


def test_cpu_ctc_loss_takes_a_bfloat16_log_softmax() -> None:
    """Under a CPU autocast ``log_softmax`` keeps bfloat16 now; ``ctc_loss``
    is FP32 and casts it, where it used to reach a kernel without a half
    path."""
    logits = lucid.randn(6, 1, 4).to(lucid.bfloat16).requires_grad_()
    targets = lucid.tensor([[1, 2]])
    with autocast(device_type="cpu", dtype=lucid.bfloat16):
        log_probs = logits.log_softmax(-1)
        loss = lucid.nn.functional.ctc_loss(
            log_probs, targets, lucid.tensor([6]), lucid.tensor([2])
        )
    assert log_probs.dtype == lucid.bfloat16
    assert loss.dtype == lucid.float32
    loss.backward()
    assert logits.grad is not None and logits.grad.dtype == lucid.bfloat16

    want = lucid.nn.functional.ctc_loss(
        logits.detach().float().log_softmax(-1), targets, lucid.tensor([6]), lucid.tensor([2])
    )
    assert abs(loss.item() - want.item()) < 5e-2


@pytest.mark.parametrize(
    ("kind", "kwargs"),
    [
        ("LSTM", {}),
        ("LSTM", {"proj_size": 3}),
        ("LSTM", {"num_layers": 2, "bidirectional": True}),
        ("GRU", {}),
        ("RNN", {}),
    ],
)
def test_cpu_recurrent_modules_cast_weights_for_bfloat16_input(
    kind: str, kwargs: dict[str, Any]
) -> None:
    """Under a CPU bfloat16 autocast, a bfloat16 input to an RNN / GRU / LSTM
    with float32 weights casts the weights to bfloat16 and returns bfloat16
    outputs and states, while gradients flow back to the float32 parameters."""
    lucid.manual_seed(0)
    mod = getattr(nn, kind)(4, 8, **kwargs)
    x = lucid.randn(5, 2, 4).to(lucid.bfloat16).detach().requires_grad_()
    with autocast(device_type="cpu", dtype=lucid.bfloat16):
        out, state = mod(x)
    assert out.dtype == lucid.bfloat16
    states = list(state) if isinstance(state, tuple) else [state]
    assert all(s.dtype == lucid.bfloat16 for s in states)
    out.float().sum().backward()
    assert x.grad is not None and x.grad.dtype == lucid.bfloat16
    for p in mod.parameters():
        assert p.grad is not None and p.grad.dtype == lucid.float32


# ── the engine turns autocast off inside an op, and back on after it ────────


@pytest.mark.parametrize(("device", "autocast_dt"), _SETTINGS)
def test_autocast_is_back_on_after_an_op_returns_or_raises(device: str, autocast_dt: str) -> None:
    dt = getattr(lucid, autocast_dt)
    x = lucid.randn(3, 4, device=device)
    with autocast(device_type=device, dtype=dt):
        (x @ x.T).sum()
        lucid.log1p(x.abs())
        lucid.nn.functional.cosine_similarity(x, x)
        assert autocast.is_autocast_enabled(device)
        assert autocast.get_autocast_dtype(device) == dt
        with pytest.raises(ValueError):
            x @ x
        assert autocast.is_autocast_enabled(device)
        assert (x @ x.T).dtype == dt


# ── parity: the dtype the reference's autocast gives ────────────────────────


def _ref_device(ref: ModuleType, device: str) -> str:
    if device == "metal":
        if not ref.backends.mps.is_available():
            pytest.skip("the reference's MPS backend is unavailable")
        return "mps"
    return "cpu"


@pytest.mark.parity
@pytest.mark.parametrize("input_dt", _INPUTS)
@pytest.mark.parametrize(("device", "autocast_dt"), _SETTINGS)
@pytest.mark.parametrize("name", sorted(_CASES) + sorted(_KEEP_CASES))
def test_the_dtype_matches_the_reference(
    ref: ModuleType, name: str, device: str, autocast_dt: str, input_dt: str
) -> None:
    case = _CASES.get(name) or _KEEP_CASES[name]
    if device not in case.devices:
        pytest.skip(f"{name} is checked on {case.devices} only")
    got = _run_lucid(case, device, autocast_dt, input_dt)

    ref_device = _ref_device(ref, device)
    ref_dt = getattr(ref, autocast_dt)
    dtype = ref_dt if input_dt == "half" else ref.float32
    try:
        with ref.autocast(device_type=ref_device, dtype=ref_dt):
            out = (case.ref_fn or case.fn)(ref, ref.nn.functional, _Make(ref, ref_device, dtype))
    except NotImplementedError:
        # Its MPS backend lacks some losses; nothing to compare against.
        pytest.skip(f"the reference has no {ref_device} kernel for {name}")
    assert got == _name(out.dtype)


def _outcome(fn: Callable[[], Any]) -> str:
    """The output dtype, or ``"raises"`` — what a mixed pair comes to."""
    try:
        return _name(fn().dtype)
    except (RuntimeError, TypeError, ValueError):
        return "raises"


@pytest.mark.parity
@pytest.mark.parametrize("name", ["cross", "vdot"])
@_METAL
def test_dropped_metal_rows_meet_a_mixed_pair_as_the_reference_does(
    ref: ModuleType, name: str
) -> None:
    """``cross`` and ``vdot`` are not on the reference's MPS list (its
    ``cross`` is ``linalg_cross``, and its ``vdot`` row is not an MPS one),
    so they have no Metal row."""
    ref_device = _ref_device(ref, "metal")

    def run(lib: ModuleType, device: str) -> str:
        half = lib.float16
        h, f = _Make(lib, device, half), _Make(lib, device, lib.float32)
        op = {"cross": lib.linalg.cross, "vdot": lib.vdot}[name]
        n = 3 if name == "cross" else 4
        with lib.amp.autocast(device_type=device, dtype=half):
            return _outcome(lambda: op(h(n), f(n)))

    assert run(lucid, "metal") == run(ref, ref_device)
