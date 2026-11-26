"""A buffer's pointer is offset through ``offset_or_null`` or the site has been reviewed (LCD-1047).

An empty CPU tensor's buffer is null.  ``<buffer>.ptr.get() + offset`` with a non-zero offset on it
is undefined behaviour even when nothing is read through the result; the nightly UBSan job found
that in ``make_numpy_view`` and the p-norm kernel.  ``core/PointerOffset.h`` is the one owner of
the rule ("a null base stays null"), which the optional gradient pointers of a fused backward
already followed.

This scans ``lucid/_C`` for the expression that rule is about, a byte buffer's ``.get()`` followed
by ``+`` or ``-``, and holds its count per file to what has been reviewed.  Every site below is
either integer arithmetic on an address, behind an emptiness test on the same path, or an offset
that is a product with the tensor's own extents (an empty tensor's is 0).  A new one fails here:
use ``offset_or_null(base, offset)``, or review it and say why it is safe in ``REVIEWED``.

What this cannot see is a *typed* kernel pointer (``x + base`` in a loop nest): telling it from
integer arithmetic needs types, and a pattern broad enough to find those is mostly false positives.
They are covered by behaviour instead, under UBSan: ``ops/test_zero_extent_cpu_kernels.py`` puts a
zero-length axis in every position of every reduction, normalisation, scan and sort the CPU has,
``tensor/test_empty_view_buffers.py`` does the same for views at an offset, and
``linalg/test_p_norm_cpu_kernel.py`` for the kernel where this was found.
"""

import re
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[2] / "_C"

#: ``x.ptr.get() + n``, ``ptr.get() - n`` and ``reinterpret_cast<T*>(p.get()) + n``.
RAW_OFFSET = re.compile(r"\.get\(\)\)?\s*[-+]\s*[^=\s]")

#: file (relative to ``lucid/_C``) -> (reviewed count, why those are safe)
REVIEWED: dict[str, tuple[int, str]] = {
    "backend/cpu/methods/Factory.cpp": (
        2,
        "``contiguous`` returns for ``nbytes == 0`` first; ``meshgrid`` loops over the output",
    ),
    "backend/cpu/methods/Indexing.cpp": (
        1,
        "``gather``'s coordinate walk, over the output's elements after the index is checked",
    ),
    "backend/cpu/methods/Layout.cpp": (
        4,
        "offsets are products with the tensor's own extents (0 when it is empty) or sit in loops "
        "over its elements; remaining layout ops use ``offset_or_null`` (LCD-1119)",
    ),
    "backend/cpu/methods/Sequence.cpp": (
        4,
        "an attention mask's slice offset is a multiple of ``Lq * Lk``; embedding rows are "
        "indexed after the id is checked against the table",
    ),
    "bindings/bind_tensor.cpp": (1, "an address (``uintptr_t``) plus an integer, not a pointer"),
    "core/TensorImpl.cpp": (
        13,
        "addresses as integers (2), ``item()`` of exactly one element (2), and packs and copies "
        "behind ``numel > 0`` / ``nbytes > 0`` or an early return (9); the offsets an empty view "
        "reaches (``make_numpy_view``, ``to_string``, the block and allocation views) use "
        "``offset_or_null``",
    ),
    "ops/utils/InplaceGraph.cpp": (2, "loops over index lists that are empty for an empty tensor"),
}


def _raw_offsets() -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(ENGINE.rglob("*")):
        relative = path.relative_to(ENGINE)
        if path.suffix not in {".h", ".cpp", ".mm"} or relative.parts[0] == "test":
            continue
        found = sum(
            1
            for line in path.read_text(encoding="utf-8").splitlines()
            if RAW_OFFSET.search(line.split("//")[0])
        )
        if found:
            counts[relative.as_posix()] = found
    return counts


def test_every_raw_buffer_offset_has_been_reviewed() -> None:
    reviewed = {name: count for name, (count, _) in REVIEWED.items()}
    assert _raw_offsets() == reviewed, (
        "a ``<buffer>.get() + offset`` was added or removed.  On an empty tensor the buffer is null "
        "and any non-zero offset is undefined behaviour: use ``offset_or_null(base, offset)`` "
        "(core/PointerOffset.h), or review the site and update REVIEWED with why it is safe"
    )


def test_the_helper_is_the_only_definition_of_the_rule() -> None:
    # a second spelling of the same rule elsewhere would be a second place to get it wrong
    definitions = [
        path.relative_to(ENGINE).as_posix()
        for path in ENGINE.rglob("*.h")
        if re.search(r"\bT\*\s+offset_or_null\s*\(", path.read_text(encoding="utf-8"))
    ]
    assert definitions == ["core/PointerOffset.h"]
