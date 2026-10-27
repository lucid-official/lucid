"""Snapshot registrations and scoped verification evidence without conflating them.

Run ``python -m tools.support_manifest --output build/support.json``. Optional
``--audit`` and ``--pretrained`` attach actual reports, not inferred support.
An emitter or public name being present says nothing about arbitrary shapes,
dtypes, devices, higher derivatives, training convergence, or paper metrics.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import lucid
from lucid._C import engine as _C_engine
from lucid.models._registry import _REGISTRY
from lucid.test.audit._surface import enumerate_surface
from lucid.weights._registry import _WEIGHTS_BY_MODEL
from tools._support_contracts import contract_links

ROOT = Path(__file__).resolve().parents[1]


def read_evidence(path: Path, kind: str) -> dict[str, Any]:
    """Keep scope and all verdicts, including refusal, skip and failure."""
    payload = path.read_bytes()
    report = json.loads(payload)
    required = {
        "audit": {"config", "findings", "coverage", "platform"},
        "pretrained": {"selected", "evidence", "scope", "complete"},
    }[kind]
    if not isinstance(report, dict) or not required.issubset(report):
        raise ValueError(f"{path}: not a {kind} evidence report")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source_match": "not_verified",
        "report": report,
    }


def snapshot(audit: Path | None = None, pretrained: Path | None = None) -> dict[str, object]:
    # The model registry is populated by its public package import, without
    # constructing any model or allocating checkpoint-sized parameters.
    symbols = enumerate_surface()
    audit_evidence = read_evidence(audit, "audit") if audit else None
    native = sorted(_C_engine.op_registry_all(), key=lambda schema: schema.name)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=normal"],
                cwd=ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        dirty = None
    return {
        "schema": 1,
        "provenance": {
            "version": lucid.__version__,
            "commit": commit,
            "dirty": dirty,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "abi": _C_engine.ABI_VERSION,
        },
        "contract": {
            "registrations": "presence only; not device/dtype/shape support",
            "evidence": "applies only to attached report inputs, scope and environment",
            "missing_evidence": "unverified, not unsupported",
            "source_match": "attached reports are historical; source identity is not inferred",
            "paper_metrics": "not measured by this manifest or checkpoint output parity",
            "static_test_calls": "source navigation only; not executed test evidence",
            "missing_contract_links": "empty means not linked, never implicitly supported",
        },
        "counts": {
            "public_symbols": len(symbols),
            "native_schemas": len(native),
            "model_factories": len(_REGISTRY),
            "weight_factories": len(_WEIGHTS_BY_MODEL),
        },
        "registrations": {
            "public_symbols": [
                {
                    "name": s.qualname,
                    "subsystem": s.subsystem,
                    "kind": s.kind,
                    "flags": sorted(s.flags),
                }
                for s in symbols
            ],
            "native_schemas": [
                {
                    "name": s.name,
                    "vjp_registration": str(_C_engine.compile.vjp_registration_status(s.name)),
                }
                for s in native
            ],
            "model_factories": [
                {
                    "name": name,
                    "family": entry.family,
                    "task": entry.task,
                    "params": entry.params,
                    "weight_entries": name in _WEIGHTS_BY_MODEL,
                }
                for name, entry in sorted(_REGISTRY.items())
            ],
        },
        "api_contracts": contract_links(
            ROOT, symbols, audit_evidence["report"] if audit_evidence else None
        ),
        "evidence": {
            "audit": audit_evidence,
            "pretrained": read_evidence(pretrained, "pretrained") if pretrained else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = snapshot(args.audit, args.pretrained)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(json.dumps(report["counts"]))
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
