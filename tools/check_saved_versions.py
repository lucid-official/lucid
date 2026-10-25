#!/usr/bin/env python3
"""tools/check_saved_versions.py — every tensor a backward node holds is accounted for.

A node must refuse backward when a tensor whose values it reads was written in
place after forward (``validate_versions``), and must not refuse when the
write cannot reach the gradient.  The engine keeps that rule in one place for
most nodes: ``AutogradNode`` guards the inputs saved by value in
``saved_inputs_`` (LCD-320 / LCD-306).  What it cannot see is a node class
that holds a tensor in a member of its own — an input's values kept outside
``saved_inputs_`` (so never guarded), or a tensor held by reference and read
at backward (``LstmBackward``, the ``scatter_reduce`` backwards).

This check lists every member of a node class whose type holds tensor data —
``Storage`` (and its vector / optional forms), ``CpuStorage``, ``GpuStorage``,
``TensorImplPtr`` / ``std::shared_ptr<TensorImpl>`` (and vectors of them) —
and requires each to appear in :data:`HELD` with its reason:

* :data:`GUARDED` — the class's own ``validate_versions`` checks the tensor;
  the class must declare ``validate_versions``.
* any other string — why no write into a user tensor can reach the gradient
  through it (computed by the forward, a by-value copy the CPU refuses to
  write under and Metal never aliases, an integer selector, ...).

A new member fails the check until it is classified, so a reviewer decides
for each one whether it is guarded.  An input's values belong in
``saved_inputs_`` — classify such a member by moving it there instead.
Entries that no longer match a member fail too, so the table stays true.

An empty ``validate_versions() override {}`` is refused as well: it was the
per-node patch for a node the base over-checked, and it switches off the
guard for whatever the node may save later.  The base already leaves an input
it did not save alone.

A node class built on ``AutogradNode`` or ``VariadicKernel`` (directly,
through ``FuncOp`` or a kernel template) that holds a tensor of its own must
also declare ``static constexpr bool kKeepsOwnValues = true``.  Their
``saved_kind`` sees only the canonical slots; without the declaration a node
whose backward reads a mask or an index it keeps itself would be run again
by a second pass the reference refuses (LCD-350).  And the canonical slots
are written only by those two headers — every other site keeps values with
``keep_input`` / ``keep_inputs`` / ``keep_output``, which record that the node
kept them whatever their size: a direct write of a tensor of no elements read
as "saved nothing".

Usage::

    python3 tools/check_saved_versions.py          # exit 1 on a finding
    python3 tools/check_saved_versions.py --list   # every member found
"""

import re
import sys
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "lucid" / "_C"
SKIP_DIRS = ("coreml", "compile")

GUARDED = "guarded"

# The bases a node class derives from, directly or through a kernel template.
_NODE_BASE = re.compile(
    r"\b(?:Node|FuncOp|AutogradNode|UnaryKernel|BinaryKernel|NaryKernel|ReduceKernel|"
    r"VariadicKernel|UnaryOp|BinaryOp|ReduceOp|ScatterReduceNode)\b"
)
_HELD_TYPE = (
    r"(?:std::array<\s*Storage\s*,[^>]*>|"
    r"std::array<\s*(?:TensorImplPtr|std::shared_ptr<\s*TensorImpl\s*>)\s*,[^>]*>|"
    r"std::vector<\s*Storage\s*>|"
    r"std::optional<\s*Storage\s*>|Storage|CpuStorage|"
    r"GpuStorage|TensorImplPtr|std::shared_ptr<\s*TensorImpl\s*>|"
    r"std::vector<\s*TensorImplPtr\s*>|std::vector<\s*std::shared_ptr<\s*TensorImpl\s*>\s*>)"
)
_MEMBER = re.compile(
    r"^(?:mutable\s+)?(?P<type>" + _HELD_TYPE + r")\s+"
    r"(?P<names>[A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*(?:=.*|\{.*\})?$",
    re.S,
)
_EMPTY_VALIDATE = re.compile(r"\bvalidate_versions\s*\(\s*\)\s*override\s*\{\s*\}")
# The bases whose saved_kind reads only the canonical slots.
_SLOT_BASE = re.compile(
    r"\b(?:FuncOp|AutogradNode|UnaryKernel|BinaryKernel|NaryKernel|ReduceKernel|"
    r"VariadicKernel|UnaryOp|BinaryOp|ReduceOp)\b"
)
_KEEPS_OWN = "kKeepsOwnValues"
# The classes that own the canonical slots themselves.
_SLOT_OWNERS = ("AutogradNode", "VariadicKernel")
# The headers that define them, the only places their slots may be written.
_SLOT_HEADERS = ("autograd/AutogradNode.h", "kernel/VariadicKernel.h")
# A write into a canonical slot: an assignment, or a vector slot grown.
_SLOT_WRITE = re.compile(
    r"\bsaved_(?:inputs|output|inputs_v)_\s*(?:\[[^\]]*\]\s*)?=(?!=)"
    r"|\bsaved_inputs_v_\s*\.\s*(?:push_back|emplace_back|assign|resize|insert)\s*\("
)
_CLASS = re.compile(
    r"\b(?:class|struct)\s+(?:LUCID_API\s+)?(?P<name>[A-Za-z_]\w*)\s*(?:final\s*)?"
    r":(?P<bases>[^;{]*)\{"
)

_BY_VALUE = (
    "a by-value copy of a tensor that takes no gradient: the CPU refuses a write "
    "into bytes a node holds and Metal never aliases them, so no write reaches it"
)
_DERIVED = "computed by the forward; no user tensor aliases it"
_OUTPUT = "the forward's output by value, read only as data"

#: ``Class::member`` -> why a write after forward cannot corrupt the gradient
#: through it, or :data:`GUARDED`.
HELD: dict[str, str] = {
    # The canonical slots — an input's values live here, guarded by the base.
    "AutogradNode::saved_inputs_": GUARDED,
    "AutogradNode::saved_impl_inputs_": GUARDED,
    "AutogradNode::saved_output_": _OUTPUT,
    "VariadicKernel::saved_inputs_v_": GUARDED,
    "VariadicKernel::saved_output_": _OUTPUT,
    # Held by reference and read at backward: the node checks their versions.
    "LstmBackward::saved_input": GUARDED,
    "LstmBackward::saved_h0": GUARDED,
    "LstmBackward::saved_weights": GUARDED,
    "LstmBackward::gates_all": _DERIVED,
    "LstmBackward::cells_all": _DERIVED,
    "ScatterReduceNode::saved_impl_inputs_": GUARDED,
    "ScatterReduceNode::saved_base_": GUARDED,
    "ScatterReduceNode::saved_indices_": GUARDED,
    "ScatterReduceNode::saved_src_": GUARDED,
    "ScatterAddNode::saved_indices_": GUARDED,
    "ScatterSetNode::saved_indices_": GUARDED,
    "LUSolveBackward::pivots_": GUARDED,
    "WhereBackward::cond_": GUARDED,
    "MaskedFillBackward::mask_": GUARDED,
    "GatherBackward::indices_": GUARDED,
    "EmbeddingBagBackward::saved_indices_": _BY_VALUE,
    "EmbeddingBagBackward::saved_offsets_": _BY_VALUE,
    # Computed by the forward.
    "CumprodBackward::saved_y_": _OUTPUT,
    "CummaxBackward::saved_y_": _OUTPUT,
    "CumminBackward::saved_y_": _OUTPUT,
    "VarBackward::saved_mean_": _DERIVED,
    "IndexScatterBackward::indices_": _DERIVED,
    "AsStridedBackward::indices_": _DERIVED,
    "ScaledDotProductAttentionBackward::saved_weights_": _DERIVED,
    "RMSNormBackward::saved_rstd_": _DERIVED,
    "BatchNormNdBackward::saved_mean_": _DERIVED,
    "BatchNormNdBackward::saved_rstd_": _DERIVED,
    "BatchNormNdBackward::saved_xnorm_": _DERIVED,
    "LayerNormBackward::saved_mean_": _DERIVED,
    "LayerNormBackward::saved_rstd_": _DERIVED,
    "GroupNormBackward::saved_mean_": _DERIVED,
    "GroupNormBackward::saved_rstd_": _DERIVED,
    "BatchNormEvalBackward::rstd_": _DERIVED,
    "LpNormalizeBackward::saved_norm_": _DERIVED,
    "GlobalResponseNormBackward::saved_Nx_": _DERIVED,
    "DropoutBackward::mask_": _DERIVED,
    "DropoutNdBackward::mask_": _DERIVED,
    "AlphaDropoutBackward::mask_": _DERIVED,
    "DropBlockBackward::mask_": _DERIVED,
    "DropPathBackward::mask_": _DERIVED,
    "RotaryPosEmbeddingBackward::saved_cos_": _DERIVED,
    "RotaryPosEmbeddingBackward::saved_sin_": _DERIVED,
    "CrossEntropyBackward::saved_softmax_": _DERIVED,
    "CrossEntropyBackward::saved_valid_count_": _DERIVED,
    "NLLLossBackward::saved_valid_count_": _DERIVED,
    "MaxPoolNdBackward::saved_argmax_": _DERIVED,
    # By-value copies of tensors that take no gradient.
    "EmbeddingBackward::saved_indices_": _BY_VALUE,
    "CrossEntropyBackward::saved_target_": _BY_VALUE,
    "CrossEntropyBackward::saved_weight_": _BY_VALUE,
    "NLLLossBackward::saved_target_": _BY_VALUE,
    "NLLLossBackward::saved_weight_": _BY_VALUE,
    "CtcLossBackward::saved_targets_": _BY_VALUE,
    "CtcLossBackward::saved_input_lengths_": _BY_VALUE,
    "CtcLossBackward::saved_target_lengths_": _BY_VALUE,
    "ScaledDotProductAttentionBackward::saved_mask_": _BY_VALUE,
}


def _strip(src: str) -> str:
    """``src`` with comments and string / char literals blanked, offsets kept."""
    out = []
    i, n = 0, len(src)
    while i < n:
        two = src[i : i + 2]
        if two == "//":
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif two == "/*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r"[^\n]", " ", src[i:j]))
            i = j
        elif src[i] in "\"'":
            q = src[i]
            j = i + 1
            while j < n and src[j] != q:
                j += 2 if src[j] == "\\" else 1
            out.append(q + " " * (min(j, n) - i - 1) + (q if j < n else ""))
            i = j + 1
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


def _body(text: str, open_at: int) -> str:
    """The text between the brace at ``open_at`` and its match."""
    depth = 0
    for k in range(open_at, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[open_at + 1 : k]
    return text[open_at + 1 :]


def _top_level(body: str) -> str:
    """``body`` with every nested ``{...}`` collapsed to ``{}``."""
    out = []
    depth = 0
    for ch in body:
        if ch == "{":
            if depth == 0:
                out.append("{")
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                out.append("}")
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def _members(flat: str) -> Iterator[str]:
    """Names of the tensor-holding members declared in a collapsed class body."""
    for stmt in flat.split(";"):
        stmt = stmt.rsplit("}", 1)[-1]
        stmt = re.sub(r"\b(?:public|protected|private)\s*:", " ", stmt).strip()
        m = _MEMBER.match(stmt)
        if m:
            for name in m.group("names").split(","):
                yield name.strip()


def _sources(engine: Path) -> Iterator[tuple[Path, str]]:
    """Each engine source under ``engine`` with comments and literals blanked."""
    for path in sorted(engine.rglob("*")):
        if path.suffix not in (".h", ".cpp", ".mm"):
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(engine).parts):
            continue
        yield path, _strip(path.read_text(encoding="utf-8", errors="replace"))


def scan(engine: Path = ENGINE) -> Iterator[tuple[str, str, bool, Path]]:
    """Yield ``(class, member, declares_validate, file)`` for each held tensor."""
    for path, text in _sources(engine):
        for m in _CLASS.finditer(text):
            flat = _top_level(_body(text, m.end() - 1))
            if not (_NODE_BASE.search(m.group("bases")) or "apply(Storage" in flat):
                continue
            validates = "validate_versions" in flat
            for member in _members(flat):
                yield m.group("name"), member, validates, path


def undeclared_own_values(engine: Path = ENGINE) -> Iterator[tuple[str, Path]]:
    """Yield ``(class, file)`` for each slot-based node holding a tensor of
    its own without declaring :data:`_KEEPS_OWN`."""
    for path, text in _sources(engine):
        for m in _CLASS.finditer(text):
            name = m.group("name")
            if name in _SLOT_OWNERS or not _SLOT_BASE.search(m.group("bases")):
                continue
            flat = _top_level(_body(text, m.end() - 1))
            if _KEEPS_OWN not in flat and any(True for _ in _members(flat)):
                yield name, path


def unreleased_own_values(engine: Path = ENGINE) -> Iterator[tuple[str, Path]]:
    """Yield ``(class, file)`` for each slot-based node declaring
    ``kKeepsOwnValues = true`` without defining ``release_own_saved``."""
    for path, text in _sources(engine):
        for m in _CLASS.finditer(text):
            name = m.group("name")
            if name in _SLOT_OWNERS or not _SLOT_BASE.search(m.group("bases")):
                continue
            flat = _top_level(_body(text, m.end() - 1))
            if (
                re.search(r"\bkKeepsOwnValues\s*=\s*true\b", flat)
                and "release_own_saved" not in flat
            ):
                yield name, path


def direct_slot_writes(engine: Path = ENGINE) -> Iterator[tuple[Path, int]]:
    """Yield ``(file, line)`` for each write into a canonical slot outside
    :data:`_SLOT_HEADERS`."""
    for path, text in _sources(engine):
        if path.relative_to(engine).as_posix() in _SLOT_HEADERS:
            continue
        for m in _SLOT_WRITE.finditer(text):
            yield path, text.count("\n", 0, m.start()) + 1


def check(engine: Path = ENGINE, held: dict[str, str] | None = None) -> list[str]:
    """Every finding under ``engine`` against ``held`` (default :data:`HELD`)."""
    table = HELD if held is None else held
    problems: list[str] = []
    for path, line in direct_slot_writes(engine):
        problems.append(
            f"{path.relative_to(engine)}:{line}: writes a saved slot directly — keep "
            f"values with keep_input / keep_inputs / keep_output, which record that the "
            f"node kept them (a slot of no elements holds no bytes, but a release still "
            f"frees what the backward reads)"
        )
    for cls, path in undeclared_own_values(engine):
        problems.append(
            f"{path.relative_to(engine)}: {cls} keeps a tensor of its own but does not "
            f"declare {_KEEPS_OWN} — AutogradNode::saved_kind sees only the canonical "
            f"slots, so a second pass would run it after its values were meant to be gone"
        )
    for cls, path in unreleased_own_values(engine):
        problems.append(
            f"{path.relative_to(engine)}: {cls} declares {_KEEPS_OWN} = true but does not "
            f"define release_own_saved() — release_saved() would leave its own members "
            f"allocated after backward"
        )
    seen: set[str] = set()
    for cls, member, validates, path in scan(engine):
        key = f"{cls}::{member}"
        seen.add(key)
        where = path.relative_to(engine)
        reason = table.get(key)
        if reason is None:
            problems.append(
                f"{where}: {key} holds tensor data and is not classified — an input's "
                f"values belong in saved_inputs_ (guarded by AutogradNode); anything "
                f"else needs an entry in HELD saying how a write cannot reach the "
                f"gradient through it, or '{GUARDED}' with a validate_versions that "
                f"checks it"
            )
        elif reason == GUARDED and not validates:
            problems.append(
                f"{where}: {key} is listed as guarded but {cls} declares no "
                f"validate_versions — the version is recorded and never compared"
            )
    for path, text in _sources(engine):
        for m in _EMPTY_VALIDATE.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            problems.append(
                f"{path.relative_to(engine)}:{line}: an empty validate_versions override "
                f"opts the node out of the guard — AutogradNode already skips an input "
                f"it did not save"
            )
    for key in sorted(set(table) - seen):
        problems.append(f"HELD lists {key}, which no longer exists")
    return problems


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if "--list" in args:
        for cls, member, validates, path in scan():
            mark = "validates" if validates else ""
            print(f"{cls}::{member:<28} {mark:<10} {path.relative_to(ROOT)}")
        return 0
    problems = check()
    if problems:
        print("[check_saved_versions] node members that hold tensors without a guard:")
        for p in problems:
            print("  " + p)
        return 1
    print(f"[check_saved_versions] {len(HELD)} held tensors, every one accounted for.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
