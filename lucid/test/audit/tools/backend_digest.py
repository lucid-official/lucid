"""lucid/test/audit/tools/backend_digest.py — golden runtime digest over audit invocations.

Replays the audit's working invocation for each surveyed symbol across
``device in {cpu, metal}`` and ``dtype in {float32, float64 (cpu), float16,
bfloat16, int32, int64, bool}`` and records a deterministic digest of:

- every output tensor's ``(dtype, shape, sha256(bytes))`` (or scalar/sequence
  repr when the return value is not a tensor),
- any raised exception's ``(type, message)``, and
- when the primary operand and output are floating-point, the backward gradient
  digest of the primary operand after contracting the output against a fixed
  covector.

Usage::

    python -m lucid.test.audit.tools.backend_digest --out before.json --repeat 2
    python -m lucid.test.audit.tools.backend_digest --out after.json --plan before.json
    python -m lucid.test.audit.tools.backend_digest --compare before.json after.json
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

import lucid
from lucid.test.audit import _probe, _specs, _surface
from lucid.test.audit._axes import Context

DIGEST_DTYPES: dict[str, tuple[str, ...]] = {
    "cpu": ("float32", "float64", "float16", "bfloat16", "int32", "int64", "bool"),
    "metal": ("float32", "float16", "bfloat16", "int32", "int64", "bool"),
}

FLOAT_DTYPES: frozenset[str] = frozenset({"float16", "bfloat16", "float32", "float64"})


def _tensor_digest(t: lucid.Tensor) -> dict[str, Any]:
    dt_str = str(t.dtype)
    shape = list(t.shape)
    if t.dtype is lucid.bfloat16:
        arr = np.ascontiguousarray(t.to("cpu").astype(lucid.float32).numpy())
    else:
        arr = np.ascontiguousarray(t.to("cpu").numpy())
    digest = hashlib.sha256(arr.tobytes()).hexdigest()
    return {"kind": "tensor", "dtype": dt_str, "shape": shape, "sha256": digest}


def _value_digest(val: Any) -> dict[str, Any]:
    if isinstance(val, lucid.Tensor):
        return _tensor_digest(val)
    if isinstance(val, np.ndarray):
        arr = np.ascontiguousarray(val)
        return {
            "kind": "ndarray",
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        }
    if isinstance(val, (tuple, list)):
        return {"kind": "seq", "items": [_value_digest(v) for v in val]}
    if isinstance(val, (bool, int, float, complex, str)) or val is None:
        return {"kind": "scalar", "type": type(val).__name__, "repr": repr(val)}
    arr_opt = _probe.to_numpy(val)
    if arr_opt is not None:
        arr = np.ascontiguousarray(arr_opt)
        return {
            "kind": "coerced",
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        }
    return {"kind": "other", "type": type(val).__name__}


def _first_tensor(val: Any) -> lucid.Tensor | None:
    if isinstance(val, lucid.Tensor):
        return val
    if isinstance(val, (tuple, list)):
        for item in val:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _contract_device_safe(out: lucid.Tensor) -> lucid.Tensor:
    flat = out.reshape(-1)
    n = int(flat.shape[0])
    weights = _probe.rng(_probe.SEED_A).standard_normal(max(n, 1))[:n]
    work_dt = lucid.float64 if out.dtype is lucid.float64 else lucid.float32
    np_dt = np.float64 if work_dt is lucid.float64 else np.float32
    w_t = _probe.tensor_of(weights.astype(np_dt), dtype=work_dt, device=str(out.device))
    if flat.is_complex():
        imag_w = _probe.rng(_probe.SEED_A + 1).standard_normal(max(n, 1))[:n]
        wi_t = _probe.tensor_of(imag_w.astype(np_dt), dtype=work_dt, device=str(out.device))
        return (
            lucid.real(flat).astype(work_dt) * w_t + lucid.imag(flat).astype(work_dt) * wi_t
        ).sum()
    return (flat.astype(work_dt) * w_t).sum()


def _build_at(array: Any, follow: bool, dtype_name: str, device: str) -> Any:
    if not follow:
        return _probe.tensor_of(array, device=device)
    dt = _probe.dtype_of(dtype_name)
    np_dt = np.float32 if dtype_name == "bfloat16" else _probe.numpy_of(dtype_name)
    return _probe.tensor_of(np.asarray(array, dtype=np_dt), dtype=dt, device=device)


def _run_one_cell(
    qualname: str, fn: Any, call: Any, device: str, dtype_name: str
) -> dict[str, Any]:
    lucid.manual_seed(1337)

    def build(array: Any, follow: bool) -> Any:
        return _build_at(array, follow, dtype_name, device)

    with _probe.preserved_globals():
        try:
            args = _probe.dtype_args(call, dtype_name, build)
            kwargs = _probe.dtype_kwargs(call, build)
            out = fn(*args, **kwargs)
            if qualname == "Tensor.data_ptr" and isinstance(out, int):
                fwd: dict[str, Any] = {"kind": "pointer", "nonzero": out > 0}
            else:
                fwd = _value_digest(out)
        except Exception as exc:  # noqa: BLE001
            return {"status": "exc", "exc_type": type(exc).__name__, "exc_msg": str(exc)}

        result: dict[str, Any] = {"status": "ok", "fwd": fwd}

        if dtype_name in FLOAT_DTYPES and 0 <= call.primary < len(args):
            try:
                args_bwd = _probe.dtype_args(call, dtype_name, build)
                kwargs_bwd = _probe.dtype_kwargs(call, build)
                prim = args_bwd[call.primary]
                if isinstance(prim, lucid.Tensor) and prim.is_floating_point():
                    prim.requires_grad_(True)
                    out_bwd = fn(*args_bwd, **kwargs_bwd)
                    t_out = _first_tensor(out_bwd)
                    if (
                        t_out is not None
                        and (t_out.is_floating_point() or t_out.is_complex())
                        and t_out.numel() > 0
                        and t_out.requires_grad
                    ):
                        loss = _contract_device_safe(t_out)
                        loss.backward()
                        if prim.grad is not None:
                            result["bwd"] = _tensor_digest(prim.grad)
            except Exception as exc:  # noqa: BLE001
                result["bwd_exc"] = f"{type(exc).__name__}: {exc}"

        return result


def _select_call(
    symbol: _surface.Symbol,
    fn: Any,
    domains: tuple[str, ...],
    plan_entry: dict[str, Any] | None = None,
) -> tuple[Any | None, dict[str, Any] | None]:
    if plan_entry is not None:
        domain = plan_entry["domain"]
        idx = int(plan_entry["index"])
        inv_list = list(_specs.invocations(symbol.short, domain, symbol.qualname, fn))
        if 0 <= idx < len(inv_list):
            return inv_list[idx], plan_entry
        return None, None

    with _probe.preserved_globals():
        for domain in domains:
            for idx, candidate in enumerate(
                _specs.invocations(symbol.short, domain, symbol.qualname, fn)
            ):
                try:
                    _ = candidate.base
                    probe_call = candidate.fresh()
                    out = fn(*probe_call.args, **probe_call.kwargs)
                except Exception:  # noqa: BLE001
                    continue
                if _probe.to_numpy(out) is not None:
                    return candidate, {"domain": domain, "index": idx}
    return None, None


def generate_digest(
    symbols_filter: set[str] | None = None,
    limit: int = 0,
    repeat: int = 1,
    plan: dict[str, dict[str, Any]] | None = None,
    devices: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    has_metal = _probe.metal_available()
    ctx = Context(metal=has_metal)
    if devices is None:
        devices = ("cpu", "metal") if has_metal else ("cpu",)

    chosen_plan: dict[str, dict[str, Any]] = {}
    cells: dict[str, dict[str, Any]] = {}
    unstable: list[str] = []

    surveyed = 0
    for symbol in _surface.enumerate_surface():
        if symbols_filter and (
            symbol.qualname not in symbols_filter and symbol.short not in symbols_filter
        ):
            continue
        fn = _surface.resolve(symbol)
        if fn is None:
            continue
        call, p_entry = _select_call(
            symbol, fn, ctx.domains, plan.get(symbol.qualname) if plan else None
        )
        if call is None or p_entry is None:
            continue

        chosen_plan[symbol.qualname] = p_entry
        surveyed += 1

        for device in devices:
            for dt_name in DIGEST_DTYPES[device]:
                cell_key = f"{symbol.qualname}|{device}|{dt_name}"
                first = _run_one_cell(symbol.qualname, fn, call, device, dt_name)
                is_unstable = False
                for _ in range(1, repeat):
                    again = _run_one_cell(symbol.qualname, fn, call, device, dt_name)
                    if again != first:
                        is_unstable = True
                        break
                if is_unstable:
                    unstable.append(cell_key)
                cells[cell_key] = first

        if limit and surveyed >= limit:
            break

    return {"plan": chosen_plan, "unstable": sorted(unstable), "cells": cells}


def compare_digests(before: dict[str, Any], after: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return ``(diffs, unstable_skipped)`` between ``before`` and ``after``."""
    unstable = set(before.get("unstable", [])) | set(after.get("unstable", []))
    b_cells: dict[str, Any] = before.get("cells", {})
    a_cells: dict[str, Any] = after.get("cells", {})
    all_keys = sorted(set(b_cells) | set(a_cells))

    diffs: list[str] = []
    skipped: list[str] = []
    for k in all_keys:
        if k in unstable:
            skipped.append(k)
            continue
        if k not in b_cells:
            diffs.append(f"+ {k}: missing in before")
        elif k not in a_cells:
            diffs.append(f"- {k}: missing in after")
        elif b_cells[k] != a_cells[k]:
            diffs.append(f"~ {k}:\n    before: {b_cells[k]}\n    after:  {a_cells[k]}")
    return diffs, skipped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--out", type=Path, help="Write digest JSON to this path")
    ap.add_argument(
        "--plan", type=Path, help="Reuse invocation choices from an existing digest JSON"
    )
    ap.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("BEFORE", "AFTER"),
        help="Compare two digest JSON files",
    )
    ap.add_argument("--symbols", help="Comma-separated symbol names (short or qualname) to run")
    ap.add_argument("--limit", type=int, default=0, help="Stop after N symbols")
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run each cell N times and mark non-deterministic cells unstable",
    )
    args = ap.parse_args(argv)

    if args.compare:
        before_data = json.loads(args.compare[0].read_text(encoding="utf-8"))
        after_data = json.loads(args.compare[1].read_text(encoding="utf-8"))
        diffs, skipped = compare_digests(before_data, after_data)
        print(
            f"[backend_digest] compared {len(before_data.get('cells', {}))} cells "
            f"({len(skipped)} unstable skipped): {len(diffs)} diffs"
        )
        if skipped:
            print(f"  unstable cells skipped: {', '.join(skipped[:20])}")
        if diffs:
            for d in diffs[:30]:
                print(f"  {d}", file=sys.stderr)
            return 1
        return 0

    if not args.out:
        ap.error("Either --out or --compare is required")

    sym_filter = {s.strip() for s in args.symbols.split(",") if s.strip()} if args.symbols else None
    plan_data = None
    if args.plan and args.plan.exists():
        plan_data = json.loads(args.plan.read_text(encoding="utf-8")).get("plan")

    t0 = time.time()
    payload = generate_digest(
        symbols_filter=sym_filter, limit=args.limit, repeat=max(1, args.repeat), plan=plan_data
    )
    elapsed = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[backend_digest] wrote {len(payload['cells'])} cells "
        f"({len(payload['plan'])} symbols, {len(payload['unstable'])} unstable) "
        f"to {args.out} in {elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
