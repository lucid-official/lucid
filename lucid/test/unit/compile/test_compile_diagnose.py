"""Compile diagnose() — manual-VJP coverage observability (Phase 2).

Three buckets must populate correctly:

* **Registered** — ops with a real :class:`VjpEmitter`.  An all-linear
  MLP has 100 % coverage (linear / relu / mse_loss / mean / sub / etc).
* **GradSink** — ops in the walker's ``no_grad_ops`` list (factories,
  comparisons, arg-reduce).  ``argmax`` is the canonical sample.
* **Missing** — ops with no emitter AND not in the sink set.  ``var``
  is the canonical sample (no manual VJP, real fallback path).

Also exercises ``LUCID_MANUAL_VJP_DEBUG=1`` — when set, a coverage gap
triggers a structured stderr line carrying the op name and signature.
"""

import subprocess
import sys

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.compile import DiagnosisReport, OpInfo, diagnose

from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor

# ── DiagnosisReport shape ───────────────────────────────────────────


def test_diagnose_returns_report_dataclass() -> None:
    """Return type is :class:`DiagnosisReport` with the documented fields."""
    model = nn.Linear(8, 4).to(COMPILE_DEVICE)
    x = metal_tensor(2, 8)
    rpt = diagnose(model, x)

    assert isinstance(rpt, DiagnosisReport)
    assert isinstance(rpt.total_ops, int) and rpt.total_ops > 0
    assert isinstance(rpt.registered, list)
    assert isinstance(rpt.grad_sinks, list)
    assert isinstance(rpt.uncovered, list)
    assert isinstance(rpt.recommendation, str)
    for bucket in (rpt.registered, rpt.grad_sinks, rpt.uncovered):
        for info in bucket:
            assert isinstance(info, OpInfo)
            assert info.count >= 1
            # sample_shape / sample_dtype may be None for output-less ops, but
            # every op in this MLP has at least one output.
            assert info.sample_shape is not None
            assert info.sample_dtype is not None


# ── Fully-covered model ─────────────────────────────────────────────


def test_diagnose_covered_mlp_reports_no_fallback() -> None:
    """An all-linear MLP loss has 100 % manual VJP coverage."""
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).to(COMPILE_DEVICE)
    x = metal_tensor(2, 8)
    t = metal_tensor(2, 4)

    def loss_fn(inp: lucid.Tensor) -> lucid.Tensor:
        return F.mse_loss(model(inp), t)

    rpt = diagnose(loss_fn, x)
    assert rpt.uncovered == [], (
        f"expected zero uncovered ops on a Linear→ReLU→Linear→MSE "
        f"graph; got {[i.name for i in rpt.uncovered]}"
    )
    assert "100% manual VJP coverage" in rpt.recommendation
    # linear, relu, mse_loss should all be registered (count >= 1 each).
    reg_names = {i.name for i in rpt.registered}
    assert "linear" in reg_names
    assert "relu" in reg_names


# ── Grad-sink classification (NOT uncovered) ────────────────────────


def test_diagnose_argmax_is_grad_sink_not_uncovered() -> None:
    """``argmax`` produces integer indices; walker treats it as a sink."""
    x = metal_tensor(4, 8)

    def fn(inp: lucid.Tensor) -> lucid.Tensor:
        return lucid.argmax(inp, dim=-1)

    rpt = diagnose(fn, x)
    sink_names = {i.name for i in rpt.grad_sinks}
    uncov_names = {i.name for i in rpt.uncovered}
    assert "argmax" in sink_names, (
        f"argmax should be classified as grad-sink, not uncovered. "
        f"sinks={sink_names}, uncovered={uncov_names}"
    )
    assert "argmax" not in uncov_names
    assert "100% manual VJP coverage" in rpt.recommendation


# ── Coverage gap ────────────────────────────────────────────────────


# ``F.unfold`` (im2col) stands in for "an op with no manual VJP".  These
# tests used ``var``, then ``conv_transpose3d``, then trilinear interpolation,
# until each got its VJP.  The op matrix's gap tables are empty now, so when
# this one gets its VJP, find the next with ``diagnose`` — ``affine_grid``,
# ``fold`` and ``nan_to_num`` had none on 2026-09-25.
_UNCOVERED = "unfold"


def _uncovered_fn() -> tuple[object, lucid.Tensor]:
    def fn(inp: lucid.Tensor) -> lucid.Tensor:
        return F.unfold(inp, 2)

    return fn, metal_tensor(1, 2, 4, 4)


def test_diagnose_uncovered_op_with_sample_shape() -> None:
    """An op with no manual VJP surfaces as uncovered + sample shape."""
    fn, x = _uncovered_fn()
    rpt = diagnose(fn, x)
    uncov_names = {i.name for i in rpt.uncovered}
    assert _UNCOVERED in uncov_names, f"expected {_UNCOVERED}; got {uncov_names}"
    # Recommendation mentions both the op name and the LUCID_MANUAL_VJP_*
    # env-var pointers the user should try next.
    assert _UNCOVERED in rpt.recommendation
    assert "LUCID_MANUAL_VJP_DEBUG" in rpt.recommendation
    assert "LUCID_MANUAL_VJP_REQUIRE" in rpt.recommendation

    # Sample shape / dtype must be populated for the offending op.
    info = next(i for i in rpt.uncovered if i.name == _UNCOVERED)
    assert info.sample_shape is not None
    assert info.sample_dtype is not None


# ── str(report) renders a readable summary ──────────────────────────


def test_diagnose_str_summary_includes_uncovered_ops() -> None:
    """``str(report)`` lists every uncovered op with count + sample shape."""
    fn, x = _uncovered_fn()
    rpt = diagnose(fn, x)
    s = str(rpt)
    assert "Diagnosis:" in s
    assert "Uncovered ops:" in s
    assert f"{_UNCOVERED} (x" in s  # "<op> (x1, sample (...)" line


# ── LUCID_MANUAL_VJP_DEBUG=1 stderr capture ─────────────────────────


def _trigger_gap_stderr() -> str:
    """Spawn a sub-process with LUCID_MANUAL_VJP_DEBUG=1 + REQUIRE=1 set.

    Capturing C++ stderr from the parent pytest process is unreliable
    (``capfd`` doesn't always intercept ``std::cerr`` from Objective-C
    on macOS).  Running in a subprocess gives us a clean stderr stream
    we can grep deterministically.
    """
    script = """
import os, sys
os.environ['LUCID_MANUAL_VJP_DEBUG'] = '1'
os.environ['LUCID_MANUAL_VJP_REQUIRE'] = '1'
import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.compile import fused_step
import lucid.optim as optim

# unfold (im2col) has no manual VJP → walker hits a gap →
# REQUIRE=1 raises and DEBUG=1 logs to stderr.
class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix = nn.Conv2d(2, 2, 1)
    def forward(self, x):
        return F.unfold(self.mix(x), 2).sum()

model = M().to('metal')
opt = optim.SGD(model.parameters(), lr=1e-3)
step = fused_step(model, lambda y, _: y, opt)
x = lucid.randn(1, 2, 4, 4).to('metal')
t = lucid.zeros(()).to('metal')
try:
    step(x, t)
except Exception as e:
    print('caught:', type(e).__name__, file=sys.stderr)
"""
    # Repo root = three levels up from this file
    # (lucid/test/unit/compile/test_compile_diagnose.py).
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[4]
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(repo_root),
    )
    return result.stderr


def test_debug_env_var_emits_structured_stderr_on_gap() -> None:
    """``LUCID_MANUAL_VJP_DEBUG=1`` writes op name + signature to stderr."""
    err = _trigger_gap_stderr()
    # Unconditional: this was ``if "manual_vjp" in err`` — and once ``var``
    # got a VJP the script stopped reaching a gap, so the test passed while
    # checking nothing.
    assert "lucid.compile manual_vjp" in err, (
        f"expected the bracketed marker [lucid.compile manual_vjp] in " f"stderr but got:\n{err}"
    )
    assert "verdict:" in err, f"expected fallback verdict line; got:\n{err}"
