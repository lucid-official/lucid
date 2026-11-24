"""Every shapeless ``compile`` in the engine goes through ``ShapelessFused`` (LCD-1062).

MLX's shapeless trace broadcasts a ``0`` extent against a ``1`` as ``1``, so a result that should
be empty comes back with elements.  ``backend/gpu/ShapelessFused.h`` runs the steps eagerly when an
operand is empty; a site that calls ``mlx::core::compile(..., /*shapeless=*/true)`` itself skips
that and brings the defect back, unseen until an empty operand reaches it.  Fifteen sites were
found compiling directly, three of them reachable with an empty broadcast (``maximum``/``minimum``,
integer ``pow``, the shifts).  This keeps the next one from being the sixteenth.
"""

import re
from pathlib import Path

import pytest

_ENGINE = Path(__file__).resolve().parents[2] / "_C"
_OWNER = _ENGINE / "backend" / "gpu" / "ShapelessFused.h"
_SUFFIXES = {".h", ".cpp", ".mm"}

# ``mlx::core::compile(`` or ``mx::compile(``; the arguments run to the matching parenthesis, so
# the ``/*shapeless=*/true`` that follows a lambda body on a later line is inside them.
_COMPILE_CALL = re.compile(r"(?:mlx::core|\bmx)::compile\s*\(")


def _arguments(text: str, opened: int) -> str:
    """The text between the parenthesis at ``opened - 1`` and its match."""
    depth = 1
    for i in range(opened, len(text)):
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        if depth == 0:
            return text[opened:i]
    return text[opened:]


def _sources() -> list[Path]:
    if not _OWNER.is_file():
        pytest.skip("the engine sources are not part of this install")
    return sorted(p for p in _ENGINE.rglob("*") if p.suffix in _SUFFIXES)


def _shapeless_compiles(path: Path) -> list[int]:
    """Line numbers of the shapeless ``compile`` calls in ``path``, line comments dropped."""
    text = re.sub(r"//[^\n]*", "", path.read_text(encoding="utf-8"))
    return [
        text.count("\n", 0, m.start()) + 1
        for m in _COMPILE_CALL.finditer(text)
        if "shapeless" in _arguments(text, m.end())
    ]


def test_the_scan_sees_the_owner_s_compile() -> None:
    _sources()
    assert len(_shapeless_compiles(_OWNER)) == 1


def test_no_site_compiles_shapeless_outside_the_owner() -> None:
    stray = {
        str(p.relative_to(_ENGINE)): lines
        for p in _sources()
        if p != _OWNER and (lines := _shapeless_compiles(p))
    }
    assert not stray, (
        "compile shapeless through gpu::ShapelessFused (backend/gpu/ShapelessFused.h): "
        f"MLX's shapeless trace gives an empty broadcast one element — {stray}"
    )
