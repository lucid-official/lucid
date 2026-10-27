#!/usr/bin/env python3
"""tools/check_backend_layout.py — Enforce the split CpuBackend / GpuBackend layout (LCD-264).

Enforces three structural invariants on ``lucid/_C/backend/{cpu,gpu}/``:

1. ``inline-body`` / ``inline-body-too-long``:
   Inside ``CpuBackend.h`` and ``GpuBackend.h``, only members on the explicit
   ``KEEP_INLINE`` allow-list (tiny hot leaf helpers and class templates) may
   define an inline body in the class header, and every inline body must be at
   most ``MAX_INLINE_BODY_LINES`` lines (10 lines).

2. ``untracked-methods-cpp``:
   Because ``CMakeLists.txt`` collects ``backend/{cpu,gpu}/methods/*.cpp`` via
   ``file(GLOB ... CONFIGURE_DEPENDS)``, every ``.cpp`` file under ``methods/``
   must be tracked by git so an untracked scratch file is never linked into the
   engine.

3. ``methods-preamble`` / ``non-member-definition``:
   Every ``backend/{cpu,gpu}/methods/<Family>.cpp`` file must have a single
   ``#include "../<Backend>.h"`` directive before ``namespace lucid { namespace backend {``
   and contain only ``<Backend>::<member>(...)`` out-of-line member definitions
   (zero namespace-scope free functions or variables, eliminating ODR risk).

4. ``cpu-scalar-libm`` (LCD-608 / Hard Rule H3):
   Files under ``lucid/_C/backend/cpu/`` must route element-wise transcendental
   math (``sin``, ``cos``, ``tan``, ``asin``, ``acos``, ``atan``, ``sinh``,
   ``cosh``, ``tanh``, ``exp``, ``log``, ``erf``) through ``Vforce.h`` rather
   than calling ``std::<fn>`` in a loop.  Any remaining ``std::<fn>`` call must
   be accounted for on the explicit ``CPU_LIBM_ALLOWLIST`` with its exact count
   and documented reason (scalar setup, ``int32`` unary fallback, or ``verf_f64``
   where Accelerate has no double ``vForce`` symbol).
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys

try:
    from tools import backend_split
except ImportError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools import backend_split

ROOT = backend_split.ROOT
MAX_INLINE_BODY_LINES = 10

_STD_LIBM_RE = re.compile(r"\bstd::(sin|cos|tan|asin|acos|atan|sinh|cosh|tanh|exp|log|erf)\b")
_METAL_NARROW_RE = re.compile(
    r"\bfits_one_launch\b|\bstatic_cast<\s*(?:std::)?int(?:32_t)?\s*>\s*\(\s*n\s*\)"
)
_CPP_RAW_STR_RE = re.compile(r'R"([A-Za-z0-9_]*)\(.*?\)\1"', re.DOTALL)


def _strip_cpp_noncode(raw: str) -> str:
    blanked = _CPP_RAW_STR_RE.sub(
        lambda m: '""' + "".join("\n" if c == "\n" else " " for c in m.group(0)[2:]), raw
    )
    return backend_split.strip_noncode(blanked)


#: Strict ratchet allow-list of permitted ``std::<fn>`` sites in ``lucid/_C/backend/cpu/``:
#: ``(rel_path, fn_name) -> (count, reason)``.
CPU_LIBM_ALLOWLIST: dict[tuple[str, str], tuple[int, str]] = {
    ("lucid/_C/backend/cpu/Vforce.cpp", "erf"): (
        1,
        "verf_f64: Accelerate has no double vForce erf symbol; runs std::erf across parallel_for chunks",
    ),
    ("lucid/_C/backend/cpu/Softmax.h", "log"): (
        2,
        "log_softmax: single scalar log of the per-row sum_shifted_exp reduction",
    ),
    ("lucid/_C/backend/cpu/CtcLoss.h", "exp"): (
        3,
        "CTC trellis logsumexp and backward posterior occupancy over (T, C) states",
    ),
    ("lucid/_C/backend/cpu/methods/Spatial.cpp", "cos"): (
        1,
        "rotate_bilinear: single scalar rotation angle cosine per call",
    ),
    ("lucid/_C/backend/cpu/methods/Spatial.cpp", "sin"): (
        1,
        "rotate_bilinear: single scalar rotation angle sine per call",
    ),
    ("lucid/_C/backend/cpu/methods/Unary.cpp", "exp"): (
        2,
        "int32 fallback in CpuBackend::exp (1) and erfinv Newton step (1)",
    ),
    ("lucid/_C/backend/cpu/methods/Unary.cpp", "log"): (
        2,
        "int32 fallback in CpuBackend::log (1) and erfinv Winitzki initial guess (1)",
    ),
    ("lucid/_C/backend/cpu/methods/Unary.cpp", "erf"): (
        2,
        "int32 fallback in CpuBackend::erf (1) and erfinv Newton step (1)",
    ),
    ("lucid/_C/backend/cpu/methods/Unary.cpp", "sin"): (1, "int32 fallback in CpuBackend::sin"),
    ("lucid/_C/backend/cpu/methods/Unary.cpp", "cos"): (1, "int32 fallback in CpuBackend::cos"),
    ("lucid/_C/backend/cpu/methods/Unary.cpp", "tan"): (1, "int32 fallback in CpuBackend::tan"),
    ("lucid/_C/backend/cpu/methods/Unary.cpp", "asin"): (1, "int32 fallback in CpuBackend::asin"),
    ("lucid/_C/backend/cpu/methods/Unary.cpp", "acos"): (1, "int32 fallback in CpuBackend::acos"),
    ("lucid/_C/backend/cpu/methods/Unary.cpp", "atan"): (1, "int32 fallback in CpuBackend::atan"),
    ("lucid/_C/backend/cpu/methods/Unary.cpp", "sinh"): (1, "int32 fallback in CpuBackend::sinh"),
    ("lucid/_C/backend/cpu/methods/Unary.cpp", "cosh"): (1, "int32 fallback in CpuBackend::cosh"),
    ("lucid/_C/backend/cpu/methods/Unary.cpp", "tanh"): (1, "int32 fallback in CpuBackend::tanh"),
    ("lucid/_C/backend/cpu/methods/Activation.cpp", "erf"): (
        2,
        "gelu_exact / gelu_exact_backward F64 branch (Accelerate has no double vForce erf)",
    ),
    ("lucid/_C/backend/cpu/methods/Activation.cpp", "exp"): (
        13,
        "silu_backward (2), gelu_exact_backward F64 (1), softplus (2), elu/elu_backward (4), selu/selu_backward (4)",
    ),
    ("lucid/_C/backend/cpu/methods/Loss.cpp", "exp"): (
        5,
        "bce_with_logits (2), cross_entropy (1), soft_margin (2) scalar reductions/loops",
    ),
    ("lucid/_C/backend/cpu/methods/Loss.cpp", "log"): (
        9,
        "bce (4), cross_entropy (1), binary_cross_entropy (4) scalar reductions/loops",
    ),
}


@dataclass(frozen=True)
class LayoutViolation:
    path: str
    line: int
    rule: str
    detail: str


def _git_tracked_methods(root: Path, rel_dir: str) -> set[str] | None:
    """Return git-tracked paths under ``rel_dir``, or ``None`` if ``root`` is not a git repo.

    Before the initial backend-split commit lands in ``HEAD`` (or on a synced
    remote checkout whose ``.git/index`` is not synced), if ``git ls-files``
    reports zero tracked files under ``rel_dir`` while every ``.cpp`` file in
    ``rel_dir`` is named ``<Family>.cpp`` for ``FAMILY_NAMES``, falls back to
    the canonical family set.
    """
    if not (root / ".git").exists():
        return None
    res = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--", rel_dir],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        return None
    tracked = {line.strip() for line in res.stdout.splitlines() if line.strip()}
    if not tracked:
        return {f"{rel_dir}/{fam}.cpp" for fam in backend_split.FAMILY_NAMES}
    return tracked


def scan_backend_layout(
    root: Path = ROOT, tracked_files: set[str] | None = None
) -> list[LayoutViolation]:
    """Scan ``lucid/_C/backend/{cpu,gpu}`` and return any layout violations."""
    violations: list[LayoutViolation] = []

    for bk, (rel_header, class_name) in backend_split.BACKENDS.items():
        hdr_path = root / rel_header
        if not hdr_path.is_file():
            violations.append(
                LayoutViolation(rel_header, 1, "missing-header", "header file not found")
            )
            continue

        src = hdr_path.read_text(encoding="utf-8")
        try:
            parsed = backend_split.parse_header(src, class_name)
        except ValueError as exc:
            violations.append(LayoutViolation(rel_header, 1, "parse-error", str(exc)))
            continue

        keep_allowed = backend_split.KEEP_INLINE[bk]
        for m in parsed.members:
            if not m.is_function or not m.has_body:
                continue
            line = src.count("\n", 0, m.sig_start) + 1
            body_inner = backend_split.strip_noncode(src[m.body_open + 1 : m.seg_end - 1])
            code_lines = sum(1 for ln in body_inner.splitlines() if ln.strip())
            if m.name not in keep_allowed:
                violations.append(
                    LayoutViolation(
                        rel_header,
                        line,
                        "inline-body",
                        f"{class_name}::{m.name} has an inline body in {rel_header}; "
                        f"move its definition to lucid/_C/backend/{bk}/methods/<Family>.cpp",
                    )
                )
            elif code_lines > MAX_INLINE_BODY_LINES:
                violations.append(
                    LayoutViolation(
                        rel_header,
                        line,
                        "inline-body-too-long",
                        f"{class_name}::{m.name} inline body is {code_lines} lines "
                        f"(cap: {MAX_INLINE_BODY_LINES})",
                    )
                )

        rel_methods_dir = f"lucid/_C/backend/{bk}/methods"
        methods_dir = root / rel_methods_dir
        if not methods_dir.is_dir():
            violations.append(
                LayoutViolation(
                    rel_methods_dir,
                    1,
                    "missing-methods-dir",
                    f"expected directory {rel_methods_dir}",
                )
            )
            continue

        git_set = (
            tracked_files
            if tracked_files is not None
            else _git_tracked_methods(root, rel_methods_dir)
        )
        expected_hdr = hdr_path.name
        for cpp_file in sorted(methods_dir.glob("*.cpp")):
            rel_cpp = cpp_file.relative_to(root).as_posix()
            if git_set is not None and rel_cpp not in git_set:
                violations.append(
                    LayoutViolation(
                        rel_cpp,
                        1,
                        "untracked-methods-cpp",
                        f"{rel_cpp} is not tracked by git (GLOB would link stray TU)",
                    )
                )
            cpp_src = cpp_file.read_text(encoding="utf-8")
            try:
                backend_split.parse_methods_cpp(cpp_src, rel_cpp, class_name, expected_hdr)
            except ValueError as exc:
                msg = str(exc)
                rule = (
                    "non-member-definition"
                    if f"non-{class_name} definition" in msg or "unexpected top-level ';'" in msg
                    else "methods-preamble"
                )
                violations.append(LayoutViolation(rel_cpp, 1, rule, msg))

    cpu_dir = root / "lucid/_C/backend/cpu"
    if cpu_dir.is_dir():
        for src_file in sorted(cpu_dir.rglob("*")):
            if not src_file.is_file() or src_file.suffix not in (".h", ".hpp", ".cpp"):
                continue
            rel_path = src_file.relative_to(root).as_posix()
            raw = src_file.read_text(encoding="utf-8")
            stripped = backend_split.strip_noncode(raw)
            by_fn: dict[str, list[int]] = {}
            for m_libm in _STD_LIBM_RE.finditer(stripped):
                fn = m_libm.group(1)
                line = stripped.count("\n", 0, m_libm.start()) + 1
                by_fn.setdefault(fn, []).append(line)
            for fn, lines in sorted(by_fn.items()):
                allowed = CPU_LIBM_ALLOWLIST.get((rel_path, fn))
                allowed_count = allowed[0] if allowed is not None else 0
                if len(lines) > allowed_count:
                    violations.append(
                        LayoutViolation(
                            rel_path,
                            lines[allowed_count],
                            "cpu-scalar-libm",
                            f"std::{fn} in {rel_path} ({len(lines)} site(s), allowed {allowed_count}); "
                            f"route element-wise math through Vforce.h (LCD-608 / H3)",
                        )
                    )
            for (allow_path, allow_fn), (allow_count, _) in sorted(CPU_LIBM_ALLOWLIST.items()):
                if allow_path != rel_path:
                    continue
                actual_count = len(by_fn.get(allow_fn, []))
                if actual_count < allow_count:
                    violations.append(
                        LayoutViolation(
                            rel_path,
                            1,
                            "stale-libm-allowlist",
                            f"CPU_LIBM_ALLOWLIST({rel_path!r}, {allow_fn!r}) expects {allow_count}, "
                            f"found {actual_count}; lower the allow-list entry",
                        )
                    )

    for scan_rel in ("lucid/_C/backend/gpu", "lucid/_C/ops"):
        scan_dir = root / scan_rel
        if not scan_dir.is_dir():
            continue
        for src_file in sorted(scan_dir.rglob("*")):
            if (
                not src_file.is_file()
                or src_file.suffix not in (".h", ".hpp", ".cpp", ".mm")
                or src_file.name == "HalfAccumulation.h"
            ):
                continue
            rel_path = src_file.relative_to(root).as_posix()
            raw = src_file.read_text(encoding="utf-8")
            stripped = _strip_cpp_noncode(raw)
            if "metal_kernel" not in stripped and "fits_one_launch" not in stripped:
                continue
            for m_narrow in _METAL_NARROW_RE.finditer(stripped):
                line = stripped.count("\n", 0, m_narrow.start()) + 1
                violations.append(
                    LayoutViolation(
                        rel_path,
                        line,
                        "metal-narrow-launch",
                        f"{m_narrow.group(0)!r} in {rel_path}; route custom Metal kernel "
                        f"launches through metal_launch_1d(n) in BinaryKernelLoad.h (LCD-591)",
                    )
                )

    return violations


def check(root: Path = ROOT, quiet: bool = False) -> int:
    violations = scan_backend_layout(root=root)
    if not violations:
        if not quiet:
            print("[check_backend_layout] OK — CpuBackend/GpuBackend layout clean.")
        return 0
    for v in violations:
        print(f"[check_backend_layout] {v.path}:{v.line}: [{v.rule}] {v.detail}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    return check(root=args.root.resolve(), quiet=args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
