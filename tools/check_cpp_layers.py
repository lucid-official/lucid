#!/usr/bin/env python3
"""tools/check_cpp_layers.py — Enforce the C++ engine layer DAG and boundary rules.

Enforces the C++ architecture invariants documented in ``obsidian/DEVELOPMENT.md``
§2.2 and §6.1 (AE-8 / GitHub #314):

1. ``upward-include``:
   The C++ layer DAG is::

       bindings -> nn -> {ops, random, optim, interop, coreml, utils}
                -> kernel -> compile -> autograd -> backend -> tensor -> core

   A file in a lower layer must not ``#include`` a header from a higher layer
   or an unrelated peer domain layer, and ``backend/cpu`` and ``backend/gpu``
   must not ``#include`` each other.

2. ``extern-op``:
   Forward declarations of ``*_op(...)`` functions (``extern ... _op(`` anywhere,
   or ``TensorImplPtr *_op(`` prototypes in layers below ``ops``) hide upward
   calls across the DAG. They are only permitted in ``autograd/GraphOps.h``
   (or on the dated ``_ALLOWLIST`` while AE-7 lands).

3. ``cblas-outside-blas``:
   Direct ``cblas_*`` calls are only permitted in ``backend/cpu/Blas.cpp``.

4. ``lapack-outside-lapack``:
   Direct Accelerate / Fortran LAPACK symbols (``sgetrf_``, ``dgesdd_``,
   ``__LAPACK_int``, ``__CLPK_integer``, etc.) are only permitted in
   ``backend/cpu/Lapack.cpp``.

5. ``gpu-cpu-lapack``:
   ``backend/gpu/`` must not call ``cpu::lapack_*`` directly (AE-3 / #301);
   GPU linear algebra delegates through ``Dispatcher::for_device(Device::CPU)``.

6. ``goto``:
   Hard Rule H13 — ``goto`` statements are forbidden across ``lucid/_C/``.

Existing legacy violations are recorded in ``_ALLOWLIST`` with a date and issue
citation. The allow-list is a strict ratchet: any new violation OR any stale
allow-list entry whose violation has been resolved causes the check to fail.
"""

import argparse
import posixpath
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "lucid" / "_C"
SUFFIXES = (".h", ".hpp", ".cpp", ".mm")

# Layer rank in the C++ DAG (lower number = lower layer in the dependency graph).
# A layer may only #include headers from its own layer or strictly lower layers.
LAYER_RANK: dict[str, int] = {
    "<root>": -1,
    "core": 0,
    "tensor": 1,
    "backend": 2,
    "autograd": 3,
    "compile": 4,
    "kernel": 5,
    "ops": 6,
    "random": 6,
    "optim": 6,
    "interop": 6,
    "coreml": 6,
    "utils": 6,
    "nn": 7,
    "bindings": 8,
    "test": 9,
}

GRAPH_OPS_HEADER = "autograd/GraphOps.h"
BLAS_IMPL = "backend/cpu/Blas.cpp"
LAPACK_IMPL = "backend/cpu/Lapack.cpp"

# Dated allow-list of legacy violations:
#   (rel_path, rule, target) -> (date_added, issue_and_reason)
# Note: backend/gpu/GpuBackend.h cpu::lapack_* calls were retired by AE-3 (#301).
_ALLOWLIST: dict[tuple[str, str, str], tuple[str, str]] = {
    # Upward #includes (to be retired by AE-7 / #314):
    ("autograd/Engine.cpp", "upward-include", "../ops/gfunc/Gfunc.h"): (
        "2026-10-14",
        "AE-7 (#314): Engine::grad includes Gfunc.h for zeros_like_op; move to autograd/GraphOps.h",
    ),
    ("compile/RngFeeds.cpp", "upward-include", "../random/Random.h"): (
        "2026-10-14",
        "AE-8 (#314): RngFeeds replays random factory ops; move hook/declaration to lower layer",
    ),
    ("core/Scope.h", "upward-include", "../compile/Tracer.h"): (
        "2026-10-14",
        "AE-7 (#314): OpScope queries compile::current_tracer directly from core/Scope.h",
    ),
    ("core/TensorImpl.cpp", "upward-include", "../backend/gpu/MlxBridge.h"): (
        "2026-10-14",
        "AE-7 (#314): TensorImpl reaches into backend/gpu/MlxBridge.h for storage materialization",
    ),
    ("core/TensorImpl.cpp", "upward-include", "../backend/gpu/SharedBufferWrites.h"): (
        "2026-10-14",
        "AE-7 (#314): TensorImpl reaches into backend/gpu/SharedBufferWrites.h for shared-buffer sync",
    ),
    ("core/TensorImpl.cpp", "upward-include", "../compile/Tracer.h"): (
        "2026-10-14",
        "AE-7 (#314): TensorImpl queries compile::current_tracer directly from core/",
    ),
    # Upward extern *_op( forward declarations (to be consolidated into autograd/GraphOps.h by AE-7 / #314):
    ("autograd/AccumulateGrad.cpp", "extern-op", "real_op"): (
        "2026-10-14",
        "AE-7 (#314): complex-to-real gradient projection in AccumulateGrad; move to autograd/GraphOps.h",
    ),
    ("autograd/AccumulateGrad.cpp", "extern-op", "astype_op"): (
        "2026-10-14",
        "AE-7 (#314): dtype cast in AccumulateGrad; move to autograd/GraphOps.h",
    ),
    ("core/TensorImpl.cpp", "extern-op", "add_op"): (
        "2026-10-14",
        "AE-7 (#314): leaf grad accumulation in TensorImpl::set_grad; move out of core/",
    ),
    ("kernel/BinaryKernel.h", "extern-op", "astype_op"): (
        "2026-10-14",
        "AE-7 (#314): AMP cast forward declaration in BinaryKernel.h; move to autograd/GraphOps.h",
    ),
    ("kernel/BinaryKernel.h", "extern-op", "to_device_op"): (
        "2026-10-14",
        "AE-7 (#314): 0-d CPU scalar promotion in BinaryKernel.h; move to autograd/GraphOps.h",
    ),
    ("kernel/BinaryKernel.h", "extern-op", "sum_op"): (
        "2026-10-14",
        "AE-7 (#314): broadcast reduction in BinaryKernel::reduce_impl_to_shape; move to autograd/GraphOps.h",
    ),
    ("kernel/BinaryKernel.h", "extern-op", "reshape_op"): (
        "2026-10-14",
        "AE-7 (#314): broadcast reshape in BinaryKernel::reduce_impl_to_shape; move to autograd/GraphOps.h",
    ),
    ("kernel/BinaryKernel.h", "extern-op", "broadcast_to_op"): (
        "2026-10-14",
        "AE-7 (#314): input broadcast in BinaryKernel::apply_for_graph; move to autograd/GraphOps.h",
    ),
    ("kernel/Contig.h", "extern-op", "contiguous_op"): (
        "2026-10-14",
        "AE-7 (#314): contiguous_op forward declaration in kernel/Contig.h; move to autograd/GraphOps.h",
    ),
    ("kernel/ReduceKernel.h", "extern-op", "broadcast_to_op"): (
        "2026-10-14",
        "AE-7 (#314): broadcast_to_op in ReduceKernel.h; move to autograd/GraphOps.h",
    ),
    ("kernel/ReduceKernel.h", "extern-op", "unsqueeze_op"): (
        "2026-10-14",
        "AE-7 (#314): unsqueeze_op in ReduceKernel.h; move to autograd/GraphOps.h",
    ),
    ("kernel/ReduceKernel.h", "extern-op", "reshape_op"): (
        "2026-10-14",
        "AE-7 (#314): reshape_op in ReduceKernel.h; move to autograd/GraphOps.h",
    ),
    ("kernel/UnaryKernel.h", "extern-op", "sum_op"): (
        "2026-10-14",
        "AE-7 (#314): sum_op in UnaryKernel.h; move to autograd/GraphOps.h",
    ),
    ("kernel/UnaryKernel.h", "extern-op", "reshape_op"): (
        "2026-10-14",
        "AE-7 (#314): reshape_op in UnaryKernel.h; move to autograd/GraphOps.h",
    ),
    ("ops/bfunc/Matmul.cpp", "extern-op", "mT_op"): (
        "2026-10-14",
        "AE-7 (#314): local extern mT_op in Matmul.cpp; include header instead",
    ),
    ("ops/bfunc/Matmul.cpp", "extern-op", "sum_op"): (
        "2026-10-14",
        "AE-7 (#314): local extern sum_op in Matmul.cpp; include Reductions.h instead",
    ),
    ("ops/bfunc/Matmul.cpp", "extern-op", "reshape_op"): (
        "2026-10-14",
        "AE-7 (#314): local extern reshape_op in Matmul.cpp; include View.h instead",
    ),
}


@dataclass(frozen=True)
class Violation:
    path: str
    line: int
    rule: str
    target: str
    detail: str


_COMMENTS_AND_STRINGS = re.compile(
    r"//[^\n]*"
    r"|/\*.*?\*/"
    r'|(?<!\w)R"(?P<delim>[^(\s"]{0,16})\(.*?\)(?P=delim)"'
    r'|"(?:\\.|[^"\\\n])*"'
    r"|(?<!\w)'(?:\\.|[^'\\\n])+'",
    re.DOTALL,
)
_PREPROCESSOR = re.compile(r"^[ \t]*#(?:\\\n|[^\n])*", re.MULTILINE)
_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*["<]([^">]+)[">]', re.MULTILINE)

# Matches explicit `extern ... <name>_op(...)` anywhere, or implicit-extern
# `TensorImplPtr <name>_op(...);` forward declarations in lower layers.
_EXTERN_OP = re.compile(
    r"\b(?P<kw>extern\s+)?(?:LUCID_API\s+)?"
    r"(?:TensorImplPtr|std::shared_ptr<\s*TensorImpl\s*>|"
    r"std::vector<\s*TensorImplPtr\s*>|std::pair<[^;{}]+>|std::tuple<[^;{}]+>)\s+"
    r"(?P<name>[A-Za-z_]\w*_op)\s*\([^;{}]*\)\s*;"
)
_CBLAS = re.compile(r"\bcblas_[A-Za-z0-9_]+\b")
_LAPACK = re.compile(
    r"\b(?:"
    r"__LAPACK_[A-Za-z0-9_]+"
    r"|__CLPK_[A-Za-z0-9_]+"
    r"|[sdcz](?:ge(?:trf|tri|trs|sdd|svd|qrf|qlf|rqf|lqf|ls|lsd|lss|lsy|ev|evd|evr|evx|es|esx|qp3|con|equ|rfs|sv|svx|hrd|bal|bak)"
    r"|po(?:trf|tri|trs|sv|svx|con|equ|rfs)"
    r"|sy(?:trf|tri|trs|ev|evd|evr|evx|sv|svx|con|rfs|td2|trd|gst)"
    r"|he(?:trf|tri|trs|ev|evd|evr|evx|sv|svx|con|rfs|td2|trd|gst)"
    r"|tr(?:trs|tri|sm|sv|con|rfs|sen|syl)"
    r"|org(?:qr|lq|ql|rq|br|hr|tr)"
    r"|ung(?:qr|lq|ql|rq|br|hr|tr)"
    r"|orm(?:qr|lq|ql|rq|br|hr|tr)"
    r"|unm(?:qr|lq|ql|rq|br|hr|tr)"
    r"|gb(?:trf|trs|sv|svx|con|equ|rfs)"
    r"|pt(?:trf|trs|sv|svx|con|rfs)"
    r"|sp(?:trf|tri|trs|sv|svx|ev|evd|evr|evx|con|rfs)"
    r"|ste(?:qr|vd|vr|vx|rf)"
    r"|bds(?:qr|dc)"
    r"|la(?:arn|cm|cpy|ebz|ed0|ed1|ev2|exc|f2|g2|ge|gs2|g2s|lag|lamch|lange|lanhe|lansy|lantr|larf|larfb|larfg|larft|lartg|lascl|laset|lasr|lassq|laswp|latrd|lauum))_"
    r")\b"
)
_GPU_CPU_LAPACK = re.compile(r"\b(?:cpu::\s*)?lapack_[A-Za-z0-9_]+\b")
_GOTO = re.compile(r"\bgoto\b")


def _blank_match(m: re.Match[str]) -> str:
    text = m.group(0)
    return "".join("\n" if c == "\n" else " " for c in text)


def _strip_comments(text: str) -> str:
    """Blank out comments while keeping `#include` lines intact."""
    return re.sub(r"//[^\n]*|/\*.*?\*/", _blank_match, text, flags=re.DOTALL)


def _strip_code(text: str) -> str:
    """Blank out comments, string/char literals, and preprocessor directives."""
    no_cs = _COMMENTS_AND_STRINGS.sub(_blank_match, text)
    return _PREPROCESSOR.sub(_blank_match, no_cs)


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _layer_of(rel_posix: str) -> str:
    parts = rel_posix.split("/")
    return parts[0] if len(parts) >= 2 else "<root>"


def _resolve_include(rel_posix: str, inc: str, engine: Path) -> str | None:
    """Resolve an ``#include`` path to its target path relative to ``engine``."""
    parent = posixpath.dirname(rel_posix)
    if inc.startswith("."):
        norm = posixpath.normpath(posixpath.join(parent, inc))
        if norm.startswith(".."):
            return None
        return norm
    # Relative to the including file's directory if it exists on disk.
    local_norm = posixpath.normpath(posixpath.join(parent, inc)) if parent else inc
    if (engine / local_norm).exists():
        return local_norm
    # Otherwise relative to engine root if the first segment is a known layer
    # or the file exists at engine root.
    first = inc.split("/")[0]
    if ("/" in inc and first in LAYER_RANK) or (engine / inc).exists():
        return posixpath.normpath(inc)
    return None


def _check_includes(rel_posix: str, text: str, engine: Path) -> list[Violation]:
    src_layer = _layer_of(rel_posix)
    if src_layer not in LAYER_RANK:
        return []
    src_rank = LAYER_RANK[src_layer]
    stripped = _strip_comments(text)
    out: list[Violation] = []

    for m in _INCLUDE.finditer(stripped):
        inc = m.group(1).strip()
        target_rel = _resolve_include(rel_posix, inc, engine)
        if target_rel is None:
            continue
        dst_layer = _layer_of(target_rel)
        if dst_layer not in LAYER_RANK or dst_layer == "<root>":
            continue
        dst_rank = LAYER_RANK[dst_layer]
        line = _line_of(stripped, m.start())

        if dst_layer == src_layer:
            # Intra-backend rule: backend/cpu and backend/gpu must not include each other.
            if src_layer == "backend":
                src_parts = rel_posix.split("/")
                dst_parts = target_rel.split("/")
                if (
                    len(src_parts) >= 3
                    and len(dst_parts) >= 3
                    and src_parts[1] in ("cpu", "gpu")
                    and dst_parts[1] in ("cpu", "gpu")
                    and src_parts[1] != dst_parts[1]
                ):
                    out.append(
                        Violation(
                            rel_posix,
                            line,
                            "upward-include",
                            inc,
                            f"backend/{src_parts[1]} must not #include backend/{dst_parts[1]} ({inc})",
                        )
                    )
            continue

        if dst_rank > src_rank:
            out.append(
                Violation(
                    rel_posix,
                    line,
                    "upward-include",
                    inc,
                    f"upward #include from '{src_layer}' (rank {src_rank}) to '{dst_layer}' (rank {dst_rank}): {inc}",
                )
            )
        elif dst_rank == src_rank:
            out.append(
                Violation(
                    rel_posix,
                    line,
                    "upward-include",
                    inc,
                    f"cross-domain #include between peer layers '{src_layer}' and '{dst_layer}' (rank {src_rank}): {inc}",
                )
            )
    return out


def _check_extern_ops(rel_posix: str, code: str) -> list[Violation]:
    if rel_posix == GRAPH_OPS_HEADER:
        return []
    src_layer = _layer_of(rel_posix)
    src_rank = LAYER_RANK.get(src_layer, -1)
    out: list[Violation] = []
    for m in _EXTERN_OP.finditer(code):
        has_extern = bool(m.group("kw"))
        name = m.group("name")
        # Flag any explicit `extern ... *_op(...)` anywhere (except GraphOps.h),
        # or any implicit-extern `TensorImplPtr *_op(...);` forward declaration
        # in layers below `ops` (rank < 6).
        if has_extern or (0 <= src_rank < LAYER_RANK["ops"]):
            line = _line_of(code, m.start())
            out.append(
                Violation(
                    rel_posix,
                    line,
                    "extern-op",
                    name,
                    f"forward declaration of '{name}' outside {GRAPH_OPS_HEADER} hides an upward op call",
                )
            )
    return out


def _check_cblas(rel_posix: str, code: str) -> list[Violation]:
    if rel_posix == BLAS_IMPL:
        return []
    out: list[Violation] = []
    for m in _CBLAS.finditer(code):
        sym = m.group(0)
        out.append(
            Violation(
                rel_posix,
                _line_of(code, m.start()),
                "cblas-outside-blas",
                sym,
                f"'{sym}' called outside {BLAS_IMPL}",
            )
        )
    return out


def _check_lapack(rel_posix: str, code: str) -> list[Violation]:
    if rel_posix == LAPACK_IMPL:
        return []
    out: list[Violation] = []
    for m in _LAPACK.finditer(code):
        sym = m.group(0)
        out.append(
            Violation(
                rel_posix,
                _line_of(code, m.start()),
                "lapack-outside-lapack",
                sym,
                f"LAPACK symbol '{sym}' used outside {LAPACK_IMPL}",
            )
        )
    return out


def _check_gpu_lapack(rel_posix: str, code: str) -> list[Violation]:
    if not rel_posix.startswith("backend/gpu/"):
        return []
    out: list[Violation] = []
    for m in _GPU_CPU_LAPACK.finditer(code):
        sym = m.group(0)
        out.append(
            Violation(
                rel_posix,
                _line_of(code, m.start()),
                "gpu-cpu-lapack",
                sym,
                f"'{sym}' called in backend/gpu/ (delegate via Dispatcher::for_device(Device::CPU) instead)",
            )
        )
    return out


def _check_goto(rel_posix: str, code: str) -> list[Violation]:
    out: list[Violation] = []
    for m in _GOTO.finditer(code):
        out.append(
            Violation(
                rel_posix,
                _line_of(code, m.start()),
                "goto",
                "goto",
                "goto statement violates Hard Rule H13",
            )
        )
    return out


def scan_file(path: Path, engine: Path = ENGINE) -> list[Violation]:
    """Scan a single C++/ObjC++ file under ``engine`` for all 6 invariants."""
    rel_posix = path.resolve().relative_to(engine.resolve()).as_posix()
    raw = path.read_text(encoding="utf-8", errors="replace")
    code = _strip_code(raw)
    violations: list[Violation] = []
    violations.extend(_check_includes(rel_posix, raw, engine))
    violations.extend(_check_extern_ops(rel_posix, code))
    violations.extend(_check_cblas(rel_posix, code))
    violations.extend(_check_lapack(rel_posix, code))
    violations.extend(_check_gpu_lapack(rel_posix, code))
    violations.extend(_check_goto(rel_posix, code))
    return violations


def scan_engine(
    engine: Path = ENGINE, allowlist: dict[tuple[str, str, str], tuple[str, str]] | None = None
) -> tuple[list[Violation], list[tuple[str, str, str]], list[Violation]]:
    """Scan ``engine`` and partition findings into ``(new_violations, stale_allowlist, allowed_hits)``."""
    table = _ALLOWLIST if allowlist is None else allowlist
    files = sorted(p for p in engine.rglob("*") if p.is_file() and p.suffix in SUFFIXES)

    by_key: dict[tuple[str, str, str], list[Violation]] = {}
    for f in files:
        for v in scan_file(f, engine):
            by_key.setdefault((v.path, v.rule, v.target), []).append(v)

    new_violations: list[Violation] = []
    allowed_hits: list[Violation] = []
    for key, hits in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[1][0].line)):
        if key in table and len(hits) == 1:
            allowed_hits.extend(hits)
        else:
            new_violations.extend(hits)

    checked_files = {f.resolve().relative_to(engine.resolve()).as_posix() for f in files}
    stale: list[tuple[str, str, str]] = [
        key for key in sorted(table) if key[0] not in checked_files or key not in by_key
    ]
    return new_violations, stale, allowed_hits


def check(
    engine: Path = ENGINE,
    allowlist: dict[tuple[str, str, str], tuple[str, str]] | None = None,
    *,
    list_allowed: bool = False,
) -> int:
    """Run all C++ layer checks on ``engine``. Returns 0 if clean, 1 on violation."""
    table = _ALLOWLIST if allowlist is None else allowlist
    new_violations, stale, allowed_hits = scan_engine(engine, table)

    if list_allowed:
        for v in allowed_hits:
            date_added, reason = table[(v.path, v.rule, v.target)]
            print(f"  allowed [{date_added}] {v.path}:{v.line} [{v.rule}] {v.target} — {reason}")

    if new_violations:
        print("[check_cpp_layers] C++ layer / boundary violations found:")
        for v in new_violations:
            print(f"  {v.path}:{v.line}: [{v.rule}] {v.detail}")

    if stale:
        print("[check_cpp_layers] Stale _ALLOWLIST entries (resolved — remove from _ALLOWLIST):")
        for path, rule, target in stale:
            print(f"  {path} [{rule}] {target}")

    if new_violations or stale:
        total = len(new_violations) + len(stale)
        print(
            f"\n[check_cpp_layers] {total} problem(s) ({len(new_violations)} new, {len(stale)} stale)."
        )
        return 1

    print(
        f"[check_cpp_layers] all C++ layer rules OK "
        f"(6 invariants checked, {len(allowed_hits)} legacy site(s) on _ALLOWLIST)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate C++ engine layer DAG and boundary invariants."
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=ENGINE,
        help="Path to lucid/_C directory (default: %(default)s)",
    )
    parser.add_argument("--list", action="store_true", help="List existing allow-listed violations")
    args = parser.parse_args(argv)
    return check(args.engine.resolve(), list_allowed=args.list)


if __name__ == "__main__":
    sys.exit(main())
