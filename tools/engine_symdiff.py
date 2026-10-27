#!/usr/bin/env python3
"""tools/engine_symdiff.py — binary ABI and per-function codegen equivalence checker.

Compares two built ``engine.cpython-*.so`` Mach-O shared objects across four
layers:

1. **Exports (ABI)** — ``nm -gU`` global defined symbol set.
2. **Class function symbols** — ``nm -n`` text symbols (``T`` / ``t``) for
   ``CpuBackend`` and ``GpuBackend`` (or ``--class``), demangled via
   ``c++filt`` and normalized for compiler-internal suffixes (``.cold.N``,
   ``.llvm.N``, and anonymous lambda indices ``$_N``).
3. **Per-function normalized disassembly** — ``/usr/bin/objdump -d
   --no-show-raw-insn --disassemble-symbols=...`` with addresses stripped,
   intra-function branch targets normalized to ``<SELF+0x...>``, inter-function
   call targets normalized to ``<callee>``, ``adrp`` + ``add``/``ldr``/``str``
   page-relative immediates normalized to ``PAGE`` / ``PAGEOFF``, and ``nop``
   alignment padding removed.
4. **Section & file size** — ``__TEXT,__text`` byte count and total ``.so``
   file size.

Usage::

    python tools/engine_symdiff.py before.so after.so
    python tools/engine_symdiff.py before.so after.so --class CpuBackend
    python tools/engine_symdiff.py before.so after.so --allow-body-diff
"""

import argparse
from dataclasses import dataclass
import difflib
from pathlib import Path
import re
import subprocess

DEFAULT_CLASSES: tuple[str, ...] = ("CpuBackend", "GpuBackend")

_COLD_OR_LLVM = re.compile(r"\.(?:cold|llvm)\.\d+\b")
_ANON_LAMBDA = re.compile(r"\d*\$_\d+")
_HEADER_LINE = re.compile(r"^[0-9a-fA-F]+\s+<([^>]+)>:\s*$")
_INSN_LINE = re.compile(r"^\s*[0-9a-fA-F]+:\s+(\S+)(?:\s+(.*))?$")
_BRANCH_TARGET = re.compile(r"\b0x[0-9a-fA-F]+\s+<([^+>]+)(?:\+0x([0-9a-fA-F]+))?>")
_ADRP_OPERAND = re.compile(r"^([wx]\d+),\s*0x[0-9a-fA-F]+(?:\s+<[^>]+>)?$")
_ADD_PAGEOFF = re.compile(r"^([wx]\d+,\s*{reg}),\s*#0x[0-9a-fA-F]+(?:\s+.*)?$")
_MEM_PAGEOFF = re.compile(r"^([wxdsq]\d+,\s*\[{reg})(?:,\s*#0x[0-9a-fA-F]+)?\](?:\s+.*)?$")


def normalize_symbol_name(sym: str) -> str:
    """Strip compiler-internal ``.cold.N`` / ``.llvm.N`` suffixes and ``$_N`` numbers."""
    s = _COLD_OR_LLVM.sub("", sym)
    s = _ANON_LAMBDA.sub("$_", s)
    return s


def normalize_disassembly(lines: list[str], current_symbol: str) -> list[str]:
    """Normalize raw ``objdump -d --no-show-raw-insn`` lines for one function."""
    norm_self = normalize_symbol_name(current_symbol)
    out: list[str] = []
    adrp_reg: str | None = None

    for raw in lines:
        m = _INSN_LINE.match(raw)
        if m is None:
            continue
        mnemonic = m.group(1)
        operands = (m.group(2) or "").strip()

        if mnemonic == "nop":
            continue

        if mnemonic == "adrp":
            am = _ADRP_OPERAND.match(operands)
            if am is not None:
                adrp_reg = am.group(1)
                out.append(f"adrp\t{adrp_reg}, PAGE")
                continue

        if adrp_reg is not None:
            if mnemonic == "add":
                add_m = re.match(
                    rf"^([wx]\d+,\s*{re.escape(adrp_reg)}),\s*#0x[0-9a-fA-F]+(?:\s+.*)?$", operands
                )
                if add_m is not None:
                    out.append(f"add\t{add_m.group(1)}, PAGEOFF")
                    adrp_reg = None
                    continue
            if mnemonic.startswith(("ldr", "str")):
                mem_m = re.match(
                    rf"^([wxdsq]\d+,\s*\[{re.escape(adrp_reg)})(?:,\s*#0x[0-9a-fA-F]+)?\](?:\s+.*)?$",
                    operands,
                )
                if mem_m is not None:
                    out.append(f"{mnemonic}\t{mem_m.group(1)}, PAGEOFF]")
                    adrp_reg = None
                    continue
            adrp_reg = None

        def _replace_target(bm: re.Match[str]) -> str:
            target = normalize_symbol_name(bm.group(1))
            offset = bm.group(2)
            if target == norm_self:
                return f"<SELF+0x{offset}>" if offset else "<SELF>"
            return f"<{target}>"

        operands = _BRANCH_TARGET.sub(_replace_target, operands)
        if operands:
            out.append(f"{mnemonic}\t{operands}")
        else:
            out.append(mnemonic)

    return out


def parse_objdump_output(text: str) -> dict[str, list[str]]:
    """Parse ``objdump -d --no-show-raw-insn`` output into ``{norm_sym: norm_insns}``."""
    raw_blocks: dict[str, list[str]] = {}
    current_sym: str | None = None
    current_lines: list[str] = []

    for line in text.splitlines():
        hm = _HEADER_LINE.match(line)
        if hm is not None:
            if current_sym is not None:
                raw_blocks[current_sym] = current_lines
            current_sym = hm.group(1)
            current_lines = []
            continue
        if current_sym is not None:
            current_lines.append(line)

    if current_sym is not None:
        raw_blocks[current_sym] = current_lines

    result: dict[str, list[str]] = {}
    for sym, blk in raw_blocks.items():
        # Skip .cold sub-blocks if a primary symbol exists, or key by normalized symbol
        if ".cold." in sym:
            continue
        norm_sym = normalize_symbol_name(sym)
        result[norm_sym] = normalize_disassembly(blk, sym)
    return result


def get_exports(so_path: Path) -> set[str]:
    """Return global defined symbols from ``nm -gU``."""
    out = subprocess.check_output(["nm", "-gU", str(so_path)], text=True)
    exports: set[str] = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            exports.add(parts[2])
    return exports


def get_class_text_symbols(so_path: Path, classes: tuple[str, ...]) -> dict[str, str]:
    """Return ``{normalized_mangled: raw_mangled}`` for text symbols matching ``classes``."""
    out = subprocess.check_output(["nm", "-n", str(so_path)], text=True)
    result: dict[str, str] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[1] not in ("T", "t"):
            continue
        mangled = parts[2]
        if ".cold." in mangled:
            continue
        if any(f"{len(cls)}{cls}" in mangled for cls in classes):
            result[normalize_symbol_name(mangled)] = mangled
    return result


def demangle_symbols(mangled_list: list[str]) -> dict[str, str]:
    """Demangle ``mangled_list`` in one ``c++filt`` invocation."""
    if not mangled_list:
        return {}
    proc = subprocess.run(
        ["c++filt"],
        input="\n".join(mangled_list) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    demangled = proc.stdout.splitlines()
    return dict(zip(mangled_list, demangled, strict=False))


def disassemble_symbols(so_path: Path, mangled_syms: list[str]) -> dict[str, list[str]]:
    """Disassemble ``mangled_syms`` from ``so_path`` in chunks."""
    if not mangled_syms:
        return {}
    merged: dict[str, list[str]] = {}
    chunk_size = 500
    for i in range(0, len(mangled_syms), chunk_size):
        chunk = mangled_syms[i : i + chunk_size]
        out = subprocess.check_output(
            [
                "/usr/bin/objdump",
                "-d",
                "--no-show-raw-insn",
                f"--disassemble-symbols={','.join(chunk)}",
                str(so_path),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        merged.update(parse_objdump_output(out))
    return merged


def get_text_section_size(so_path: Path) -> int:
    """Return the byte size of ``__TEXT,__text`` from ``objdump -h``."""
    out = subprocess.check_output(["/usr/bin/objdump", "-h", str(so_path)], text=True)
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "__text":
            return int(parts[2], 16)
    return 0


@dataclass(frozen=True)
class SymdiffReport:
    exports_added: list[str]
    exports_removed: list[str]
    exports_count: int
    funcs_added: list[str]
    funcs_removed: list[str]
    funcs_common: int
    bodies_identical: int
    bodies_changed: dict[str, tuple[int, int, str]]
    text_before: int
    text_after: int
    file_before: int
    file_after: int


def compare_binaries(
    before_so: Path, after_so: Path, classes: tuple[str, ...] = DEFAULT_CLASSES
) -> SymdiffReport:
    exp_before = get_exports(before_so)
    exp_after = get_exports(after_so)
    exp_added = sorted(exp_after - exp_before)
    exp_removed = sorted(exp_before - exp_after)

    syms_before = get_class_text_symbols(before_so, classes)
    syms_after = get_class_text_symbols(after_so, classes)
    set_before = set(syms_before)
    set_after = set(syms_after)
    funcs_added = sorted(set_after - set_before)
    funcs_removed = sorted(set_before - set_after)
    common_norm = sorted(set_before & set_after)

    dis_before = disassemble_symbols(before_so, [syms_before[k] for k in common_norm])
    dis_after = disassemble_symbols(after_so, [syms_after[k] for k in common_norm])

    bodies_identical = 0
    bodies_changed: dict[str, tuple[int, int, str]] = {}
    for norm_k in common_norm:
        b_insns = dis_before.get(norm_k, [])
        a_insns = dis_after.get(norm_k, [])
        if b_insns == a_insns:
            bodies_identical += 1
        else:
            diff_lines = list(
                difflib.unified_diff(
                    b_insns, a_insns, fromfile="before", tofile="after", lineterm="", n=2
                )
            )[:20]
            bodies_changed[norm_k] = (len(b_insns), len(a_insns), "\n".join(diff_lines))

    return SymdiffReport(
        exports_added=exp_added,
        exports_removed=exp_removed,
        exports_count=len(exp_before),
        funcs_added=funcs_added,
        funcs_removed=funcs_removed,
        funcs_common=len(common_norm),
        bodies_identical=bodies_identical,
        bodies_changed=bodies_changed,
        text_before=get_text_section_size(before_so),
        text_after=get_text_section_size(after_so),
        file_before=before_so.stat().st_size,
        file_after=after_so.stat().st_size,
    )


def _pct(before: int, after: int) -> str:
    if before == 0:
        return "0.00%"
    return f"{(after - before) / before * 100.0:+.2f}%"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("before", type=Path, help="Path to baseline engine .so")
    ap.add_argument("after", type=Path, help="Path to candidate engine .so")
    ap.add_argument(
        "--class",
        dest="classes",
        action="append",
        default=[],
        help="Class name filter (default: CpuBackend and GpuBackend)",
    )
    ap.add_argument(
        "--allow-body-diff",
        action="store_true",
        help="Treat body diffs as warnings if EXPORTS and FUNCTIONS are identical",
    )
    args = ap.parse_args(argv)

    classes = tuple(args.classes) if args.classes else DEFAULT_CLASSES
    rep = compare_binaries(args.before, args.after, classes=classes)

    if not rep.exports_added and not rep.exports_removed:
        print(f"EXPORTS identical ({rep.exports_count})")
    else:
        print(f"EXPORTS DIFF +{len(rep.exports_added)} -{len(rep.exports_removed)}")
        for s in rep.exports_added[:10]:
            print(f"  + {s}")
        for s in rep.exports_removed[:10]:
            print(f"  - {s}")

    if not rep.funcs_added and not rep.funcs_removed:
        print(f"FUNCTIONS identical ({rep.funcs_common})")
    else:
        print(
            f"FUNCTIONS DIFF +{len(rep.funcs_added)} -{len(rep.funcs_removed)} (common {rep.funcs_common})"
        )
        dem = demangle_symbols(rep.funcs_added[:10] + rep.funcs_removed[:10])
        for s in rep.funcs_added[:10]:
            print(f"  + {dem.get(s, s)}")
        for s in rep.funcs_removed[:10]:
            print(f"  - {dem.get(s, s)}")

    print(f"BODIES identical {rep.bodies_identical}, changed {len(rep.bodies_changed)}")
    if rep.bodies_changed:
        dem = demangle_symbols(list(rep.bodies_changed.keys()))
        for sym, (n_b, n_a, diff_txt) in sorted(rep.bodies_changed.items()):
            print(f"  ~ {dem.get(sym, sym)} ({n_b} -> {n_a} insns)")
            for dline in diff_txt.splitlines():
                print(f"      {dline}")

    print(
        f"TEXT {rep.text_before} -> {rep.text_after} ({_pct(rep.text_before, rep.text_after)})  "
        f"FILE {rep.file_before} -> {rep.file_after} ({_pct(rep.file_before, rep.file_after)})"
    )

    if rep.exports_added or rep.exports_removed:
        return 1
    if rep.funcs_added or rep.funcs_removed:
        return 1
    if rep.bodies_changed and not args.allow_body_diff:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
