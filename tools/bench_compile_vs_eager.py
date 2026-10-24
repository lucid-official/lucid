"""Compile-vs-eager wall-clock characterisation sweep.

Companion to ``tools/bench_compile.py`` (which generates a
fixed-format table for the docs site).  This one is the
characterisation harness — sweeps every workload × precision × batch
size × path combination and writes a structured CSV the
``obsidian/perf/perf-compile-vs-eager.md`` note consumes.

Usage::

    # Local smoke (fast subset, validates harness end-to-end):
    python -m tools.bench_compile_vs_eager --quick

    # Full sweep — runs on Mac Studio M4 Max per the macstudio-bench
    # skill workflow; ~10 min wall, ~90 measurements:
    python -m tools.bench_compile_vs_eager \\
        --csv /Users/chanlee/Desktop/bench_compile_vs_eager.csv

    # Specific workload only:
    python -m tools.bench_compile_vs_eager \\
        --workload mlp,resnet18_train

Closes the data gap identified in
the vault's compile parity architecture note §4.2 —
Lucid eager vs Lucid compile end-to-end measurements have never
been recorded.

Methodology
-----------
* Wall-clock via ``time.perf_counter`` after every region; matches
  ``tools/bench_compile.py`` convention.
* Materialise every output via ``sum().item()`` (or ``loss.item()``)
  and then call ``lucid.metal.synchronize()``. Synchronisation alone
  does not submit a lazy graph. These are end-to-end latencies,
  including the host observation, not isolated kernel timings.
* **Cold** = one call after model/input transfer to metal, before
  any other invocation.  Captures trace + MPSGraph compile + first
  run.  Reported in its own CSV column.
* **Warm** = median of ``--iter`` timed runs (default 10)
  after 3 warmup calls + synchronize.  Reported with p5/p95.
* For multiple trials, rerun the command with a distinct CSV path.
  There is no automatic interleaving or thermal-throttling correction.
* **Parity guard before timing.**  Compile-out vs eager-out
  ``max|diff| / max|eager|`` > 1e-3 aborts the row with note
  ``parity-diverged``.  A compile that produces wrong outputs
  cannot appear in the speedup column.

Output schema (CSV)
-------------------
``workload, precision, batch, path, cold_ms, warm_ms_median,
warm_ms_p5, warm_ms_p95, speedup_vs_eager, parity_rel_diff, note``

* ``path`` ∈ {``eager``, ``compile``, ``fused_step``}
* ``precision`` ∈ {``f32``, ``f16``} (f16 = manual cast since
  X4.3/X4.4 autocast threading deferred — see parity diagnosis)
* ``speedup_vs_eager`` is computed against the same workload/
  precision/batch's *eager* row; column blank for the eager row
  itself.
* ``note`` carries any annotation (``ok``, ``parity-diverged``,
  ``skip-amp``, ``skip-dropout-on-compile``, etc.).
"""

import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import traceback
from typing import Callable

import lucid
import lucid.metal as _metal
import lucid.nn as nn
import lucid.optim as optim
from lucid.compile import fused_step
from tools._bench_timing import cold_ms, warm_ms

# Import works both as a package module (``python -m tools.bench_*``
# from the repo root) and as a flat script (``python bench_*.py`` from
# the Mac Studio's ``/Users/chanlee/lucid_smoke/`` deployment dir).
try:
    from tools._bench_workloads import WORKLOADS, Workload, get as get_workload
except ImportError:
    from _bench_workloads import WORKLOADS, Workload, get as get_workload  # type: ignore[no-redef]


COMPILE_DEVICE = "metal"


# ── Tensor-unwrap helper (mirror of tools/bench_compile.py) ────────


def _unwrap(out: object) -> lucid.Tensor:
    """Extract the raw Tensor from a model output (Tensor or BaseModelOutput).

    Lucid model factories return ``BaseModelOutput``-style dataclasses
    with one of ``logits`` / ``prediction`` / ``last_hidden_state``.
    Forward-only workloads return raw Tensors.  Both supported.
    """
    if isinstance(out, lucid.Tensor):
        return out
    for attr in ("logits", "prediction", "last_hidden_state"):
        v = getattr(out, attr, None)
        if isinstance(v, lucid.Tensor):
            return v
    raise TypeError(f"unwrap: unrecognised output type {type(out).__name__}")


# ── Precision helper ───────────────────────────────────────────────


def _cast_inputs(
    inputs: tuple[lucid.Tensor, ...], dtype: object
) -> tuple[lucid.Tensor, ...]:
    """Cast float inputs to ``dtype``.  Integer targets (CE indices) untouched."""
    out: list[lucid.Tensor] = []
    for t in inputs:
        if not t.is_floating_point():
            out.append(t)
        else:
            out.append(t.to(dtype))
    return tuple(out)


def _cast_model(model: nn.Module, dtype: object) -> nn.Module:
    """Cast parameters and floating buffers without replacing Parameter objects."""
    return model.to(dtype=dtype)


# ── Timing primitives ──────────────────────────────────────────────


def _sync() -> None:
    """Block until every queued Metal op completes."""
    _metal.synchronize()


def _time_cold(call: Callable[[], object]) -> float:
    """Single cold call — captures trace + compile + first dispatch."""
    return cold_ms(call)


def _time_warm(
    call: Callable[[], object], n_warmup: int, n_iter: int
) -> tuple[float, float, float]:
    """Return (median, p5, p95) wall-ms over ``n_iter`` calls after warmup."""
    return warm_ms(call, n_warmup, n_iter)


# ── Parity guard ───────────────────────────────────────────────────


def _check_parity(
    eager_out: lucid.Tensor, compile_out: lucid.Tensor, threshold: float = 1e-3
) -> tuple[float, bool]:
    """Return (rel_diff, ok).  ok = rel_diff <= threshold.

    **Critical**: explicit ``metal.synchronize()`` after each operand's
    materialisation is required.  Without it, the ``eager_out -
    compile_out`` subtraction sees lazy tensors that may not have
    flushed through the MLX async dispatch queue yet — producing a
    spurious all-zeros comparison for one of the operands and a 1.0
    rel_diff false positive (which is what tripped the original
    bench harness at BS ≥ 32 on Conv workloads; see
    ``obsidian/perf/perf-compile-vs-eager-2026-05-26.md`` §"Open
    questions" §1).  Force evaluation + sync before subtracting.
    """
    if eager_out.shape != compile_out.shape:
        return float("inf"), False
    # Force evaluation of both operands BEFORE the subtraction so MLX
    # async dispatch can't leave one of them stale during the diff.
    _ = float(eager_out.sum().item())
    _ = float(compile_out.sum().item())
    _metal.synchronize()
    abs_diff = float((eager_out - compile_out).abs().max().item())
    scale = float(eager_out.abs().max().item())
    _metal.synchronize()
    rel_diff = abs_diff / max(scale, 1e-9)
    return rel_diff, rel_diff <= threshold


# ── Per-row benchmark ──────────────────────────────────────────────


def _bench_forward_eager(
    w: Workload, bs: int, dtype: object, n_iter: int
) -> tuple[float, float, float, float, lucid.Tensor]:
    """Eager forward.  Returns (cold_ms, warm_med, warm_p5, warm_p95, last_out).

    Each timed call ends with ``out.sum().item()`` to force MLX to
    fully evaluate the lazy forward graph — without it,
    ``metal.synchronize()`` alone doesn't drain MLX's deferred ops
    for tiny / submission-bound workloads, and the measured time
    collapses to launch-overhead only.  See the docstring on
    ``_bench_eager_train`` for the full history.
    """
    lucid.manual_seed(0)
    model = w.mk_model().to(COMPILE_DEVICE)
    model.eval()
    if dtype != lucid.float32:
        _cast_model(model, dtype)
    inputs = _cast_inputs(tuple(t.to(COMPILE_DEVICE) for t in w.mk_input(bs)), dtype)

    def call() -> object:
        # The same grad mode as the compiled call it is compared with.
        with lucid.no_grad():
            y = _unwrap(model(*inputs))
        return y

    cold = _time_cold(call)
    med, p5, p95 = _time_warm(call, n_warmup=3, n_iter=n_iter)
    out = call()
    return cold, med, p5, p95, out


def _bench_forward_compile(
    w: Workload, bs: int, dtype: object, n_iter: int
) -> tuple[float, float, float, float, lucid.Tensor, str]:
    """``lucid.compile(model)`` forward.  Returns same tuple + a note."""
    lucid.manual_seed(0)
    model = w.mk_model().to(COMPILE_DEVICE)
    model.eval()
    if dtype != lucid.float32:
        _cast_model(model, dtype)
    inputs = _cast_inputs(tuple(t.to(COMPILE_DEVICE) for t in w.mk_input(bs)), dtype)

    cm = lucid.compile(model)

    # Force lazy-eval drain after each call for measurement honesty
    # (compile path's sync-default already blocks for GPU work, but
    # ``.item()`` ensures the result tensor is actually materialised
    # — matches the apples-to-apples convention of the other paths).
    def call() -> object:
        # A compiled forward needs no_grad since LCD-582: under grad mode a
        # trainable model's call runs eagerly, and this would time eager.
        with lucid.no_grad():
            y = _unwrap(cm(*inputs))
        return y

    cold = _time_cold(call)
    med, p5, p95 = _time_warm(call, n_warmup=3, n_iter=n_iter)
    out = call()

    # Detect eager fallback — compile cache empty means trace aborted.
    info = cm.cache_info()
    note = "ok"
    if info["entries"] == 0:
        note = "eager-fallback"
    elif len(info.get("eager_only", [])) > 0:
        note = "partial-fallback"
    return cold, med, p5, p95, out, note


def _bench_fused_step(
    w: Workload, bs: int, dtype: object, n_iter: int
) -> tuple[float, float, float, float, str]:
    """``fused_step`` — forward + backward + opt step in one executable.

    Skips with note ``"no-loss"`` for forward-only workloads.
    This sweep times F32 training only. Manual F16 forward casting is not
    equivalent to autocast training; use ``bench_autocast`` for the latter.
    """
    if w.loss_fn is None or w.mk_target is None:
        return float("nan"), float("nan"), float("nan"), float("nan"), "no-loss"
    if dtype != lucid.float32:
        return float("nan"), float("nan"), float("nan"), float("nan"), "skip-amp-fused"

    lucid.manual_seed(0)
    model = w.mk_model().to(COMPILE_DEVICE)
    model.train()
    inputs = tuple(t.to(COMPILE_DEVICE) for t in w.mk_input(bs))
    target = w.mk_target(bs).to(COMPILE_DEVICE)
    opt = optim.SGD(list(model.parameters()), lr=1e-3)
    step = fused_step(model, w.loss_fn, opt)

    # Force lazy-eval drain so the timing matches the eager_train
    # baseline's honest cost.  fused_step's sync-default already
    # blocks for GPU work, but explicit ``.item()`` keeps the
    # apples-to-apples convention across all paths.
    def call() -> object:
        loss = step(*inputs, target)
        return loss

    cold = _time_cold(call)
    med, p5, p95 = _time_warm(call, n_warmup=3, n_iter=n_iter)
    return cold, med, p5, p95, "ok"


# ── Eager training counterpart (for fused_step speedup comparison) ──


def _bench_eager_train(
    w: Workload, bs: int, dtype: object, n_iter: int
) -> tuple[float, float, float, float, str]:
    """Eager training step: zero_grad → fwd → loss → backward → step.

    Comparison baseline for the fused_step row (so speedup is
    eager-train / fused-step, not eager-forward / fused-step which
    would be unfair).
    """
    if w.loss_fn is None or w.mk_target is None:
        return float("nan"), float("nan"), float("nan"), float("nan"), "no-loss"
    if dtype != lucid.float32:
        return float("nan"), float("nan"), float("nan"), float("nan"), "skip-amp-fused"

    lucid.manual_seed(0)
    model = w.mk_model().to(COMPILE_DEVICE)
    model.train()
    inputs = tuple(t.to(COMPILE_DEVICE) for t in w.mk_input(bs))
    target = w.mk_target(bs).to(COMPILE_DEVICE)
    opt = optim.SGD(list(model.parameters()), lr=1e-3)

    # Critical: end each timed step with ``loss.item()`` to force MLX
    # to fully evaluate the lazy graph.  Without it, ``metal.synchronize()``
    # alone only drains submitted GPU work — MLX defers backward + opt
    # for tiny / "submission-bound" workloads (GPT-2 block, MLP), and
    # the measured time collapses to ~0.2 ms regardless of batch.  This
    # bit P0.5 of the parity diagnosis where the original sweep
    # reported GPT-2 fused_step "30-440× slower" — the real eager_train
    # cost is 27 ms @ BS=8 / 366 ms @ BS=128 (4× scaling, exactly
    # matching what ``.item()`` reveals), and fused_step is actually
    # **3.8-4× faster** than honest eager training.  See
    # ``obsidian/perf/perf-compile-vs-eager-2026-05-26.md`` §"Open
    # questions" §3 → resolved by this fix.
    def call() -> object:
        opt.zero_grad()
        out = _unwrap(model(*inputs))
        loss = w.loss_fn(out, target)
        loss.backward()
        opt.step()
        return loss

    cold = _time_cold(call)
    med, p5, p95 = _time_warm(call, n_warmup=3, n_iter=n_iter)
    return cold, med, p5, p95, "ok"


# ── Sweep runner ───────────────────────────────────────────────────


def _iterations_for(warm_seed_ms: float) -> int:
    """Pick iter count so tiny workloads get more samples (lower noise)."""
    if warm_seed_ms < 5.0:
        return 50
    return 10


def _bench_one_row(
    w: Workload, dtype: object, bs: int, path: str, n_iter_hint: int
) -> dict[str, object]:
    """Run one (workload, precision, batch, path) row."""
    base = {
        "workload": w.name,
        "precision": "f32" if dtype == lucid.float32 else "f16",
        "batch": bs,
        "path": path,
        "cold_ms": float("nan"),
        "warm_ms_median": float("nan"),
        "warm_ms_p5": float("nan"),
        "warm_ms_p95": float("nan"),
        "parity_rel_diff": float("nan"),
        "note": "",
    }

    # Skip non-amp workloads at f16.
    if dtype != lucid.float32 and not w.supports_amp:
        base["note"] = "skip-amp"
        return base

    try:
        if path == "eager":
            cold, med, p5, p95, _ = _bench_forward_eager(w, bs, dtype, n_iter_hint)
            base.update(
                cold_ms=cold,
                warm_ms_median=med,
                warm_ms_p5=p5,
                warm_ms_p95=p95,
                note="ok",
            )
        elif path == "compile":
            # First get the eager output for parity check.
            _, _, _, _, eager_out = _bench_forward_eager(w, bs, dtype, n_iter=2)
            cold, med, p5, p95, comp_out, comp_note = _bench_forward_compile(
                w, bs, dtype, n_iter_hint
            )
            rel_diff, ok = _check_parity(eager_out, comp_out)
            base["parity_rel_diff"] = rel_diff
            if not ok:
                base["note"] = "parity-diverged"
            else:
                base.update(
                    cold_ms=cold,
                    warm_ms_median=med,
                    warm_ms_p5=p5,
                    warm_ms_p95=p95,
                    note=comp_note,
                )
        elif path == "fused_step":
            # Skip dropout workload at f16 (X4.3 GradScaler missing).
            cold, med, p5, p95, note = _bench_fused_step(w, bs, dtype, n_iter_hint)
            base.update(
                cold_ms=cold,
                warm_ms_median=med,
                warm_ms_p5=p5,
                warm_ms_p95=p95,
                note=note,
            )
        elif path == "eager_train":
            cold, med, p5, p95, note = _bench_eager_train(w, bs, dtype, n_iter_hint)
            base.update(
                cold_ms=cold,
                warm_ms_median=med,
                warm_ms_p5=p5,
                warm_ms_p95=p95,
                note=note,
            )
        else:
            base["note"] = f"unknown-path:{path}"
    except Exception as e:
        base["note"] = f"error:{type(e).__name__}:{str(e)[:60]}"
        traceback.print_exc(file=sys.stderr)
    return base


def _compute_speedups(rows: list[dict[str, object]]) -> None:
    """Annotate each row's speedup_vs_eager column in-place.

    eager and compile speedup is vs the eager *forward* row of the
    same (workload, precision, batch).  fused_step speedup is vs the
    eager *train* row (apples-to-apples — both include backward + opt).
    """
    # Build lookup table: (workload, precision, batch) → {path: warm_ms}
    lookup: dict[tuple[str, str, int], dict[str, float]] = {}
    for r in rows:
        key = (str(r["workload"]), str(r["precision"]), int(r["batch"]))
        lookup.setdefault(key, {})[str(r["path"])] = float(r["warm_ms_median"])

    for r in rows:
        key = (str(r["workload"]), str(r["precision"]), int(r["batch"]))
        warm = float(r["warm_ms_median"])
        if warm != warm:  # NaN
            r["speedup_vs_eager"] = float("nan")
            continue
        path = str(r["path"])
        baseline_path = "eager_train" if path in ("fused_step", "eager_train") else "eager"
        baseline = lookup.get(key, {}).get(baseline_path, float("nan"))
        if baseline != baseline or baseline <= 0.0:
            r["speedup_vs_eager"] = float("nan")
        elif path == baseline_path:
            r["speedup_vs_eager"] = 1.0
        else:
            r["speedup_vs_eager"] = baseline / warm


# ── CSV output ─────────────────────────────────────────────────────


_CSV_FIELDS = [
    "workload",
    "precision",
    "batch",
    "path",
    "cold_ms",
    "warm_ms_median",
    "warm_ms_p5",
    "warm_ms_p95",
    "speedup_vs_eager",
    "parity_rel_diff",
    "note",
]


def _emit_row(
    writer: csv.DictWriter, row: dict[str, object], file_handle: object
) -> None:
    """Write one row and flush — partial CSV is still useful on crash."""
    cleaned = {
        k: ("" if (isinstance(v, float) and v != v) else v)
        for k, v in row.items()
        if k in _CSV_FIELDS
    }
    for k in _CSV_FIELDS:
        cleaned.setdefault(k, "")
    writer.writerow(cleaned)
    if hasattr(file_handle, "flush"):
        file_handle.flush()


# ── Main ───────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Subset for local smoke (mlp + wide_mlp_dropout @ BS=8 F32 only)",
    )
    ap.add_argument(
        "--workload",
        type=str,
        default="",
        help="Comma-separated subset (default: all 5).  Names from "
        "tools/_bench_workloads.WORKLOADS.",
    )
    ap.add_argument(
        "--batch",
        type=str,
        default="8,32,128",
        help="Comma-separated batch sizes.",
    )
    ap.add_argument(
        "--precision",
        type=str,
        default="f32",
        help="Comma-separated precisions.  AMP F16 sweep is deferred until "
        "X4.3 GradScaler + X4.4 autocast threading land — see "
        "the compile parity architecture note §1.4. Pass `f32,f16` to "
        "force the F16 rows anyway (will show DtypeMismatch under the "
        "current Lucid cast surface).",
    )
    ap.add_argument(
        "--path",
        type=str,
        default="eager,compile,fused_step,eager_train",
        help="Comma-separated paths.  eager_train is the fused_step baseline.",
    )
    ap.add_argument(
        "--csv",
        type=str,
        default="out/bench_compile_vs_eager.csv",
        help="Output CSV path (will be created/overwritten).",
    )
    ap.add_argument(
        "--iter",
        type=int,
        default=10,
        help="Warm iterations per row (default 10, after 3 warmup calls).",
    )
    args = ap.parse_args()
    if args.iter < 1:
        ap.error("--iter must be positive")

    # Workload selection
    if args.quick:
        workloads = [get_workload("mlp"), get_workload("wide_mlp_dropout")]
        precisions = [lucid.float32]
        batches = [8]
        paths = ["eager", "compile", "fused_step", "eager_train"]
    else:
        if args.workload:
            workloads = [get_workload(n.strip()) for n in args.workload.split(",")]
        else:
            workloads = list(WORKLOADS)
        precisions = []
        for p in args.precision.split(","):
            p = p.strip()
            if p == "f32":
                precisions.append(lucid.float32)
            elif p == "f16":
                precisions.append(lucid.float16)
            else:
                raise ValueError(f"unknown precision {p!r}; use f32 or f16")
        batches = [int(b) for b in args.batch.split(",")]
        paths = [p.strip() for p in args.path.split(",")]

    # Open CSV
    import os

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    f = open(args.csv, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
    writer.writeheader()
    f.flush()

    print(f"# Lucid compile-vs-eager sweep — {args.csv}")
    print(
        f"# {len(workloads)} workloads × {len(precisions)} precisions × "
        f"{len(batches)} batches × {len(paths)} paths "
        f"= {len(workloads) * len(precisions) * len(batches) * len(paths)} rows"
    )

    all_rows: list[dict[str, object]] = []
    for w in workloads:
        for dt in precisions:
            for bs in batches:
                for path in paths:
                    sys.stdout.write(
                        f"  {w.name:24s} "
                        f"{'f32' if dt == lucid.float32 else 'f16':4s} "
                        f"BS={bs:4d} {path:12s} ... "
                    )
                    sys.stdout.flush()
                    row = _bench_one_row(w, dt, bs, path, args.iter)
                    warm = row["warm_ms_median"]
                    note = row["note"]
                    if isinstance(warm, float) and warm == warm:
                        sys.stdout.write(f"warm={warm:.2f}ms  {note}\n")
                    else:
                        sys.stdout.write(f"—         {note}\n")
                    sys.stdout.flush()
                    all_rows.append(row)
                    _emit_row(writer, row, f)

    # Speedups
    _compute_speedups(all_rows)
    # Rewrite the CSV with speedups populated.
    f.seek(0)
    f.truncate()
    writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
    writer.writeheader()
    for r in all_rows:
        _emit_row(writer, r, f)
    f.close()
    metadata = {
        "version": lucid.__version__, "platform": platform.platform(),
        "python": platform.python_version(), "arguments": vars(args),
        "warmup_calls": 3, "timed_calls_per_row": args.iter,
        "materialization": "output sum/loss item then Metal synchronize on every call",
        "timing_scope": "end-to-end host-observed latency, not isolated kernels",
        "trials": 1,
    }
    Path(args.csv + ".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\n# CSV written to {args.csv}")
    return 1 if any(str(row["note"]).startswith(("error:", "parity-diverged", "unknown-path:"))
                    for row in all_rows) else 0


if __name__ == "__main__":
    sys.exit(main())
