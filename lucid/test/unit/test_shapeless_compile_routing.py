"""Every MLX ``compile`` in the engine goes through ``ShapelessFused`` (LCD-1062, LCD-1072).

MLX's shapeless trace broadcasts a ``0`` extent against a ``1`` as ``1``, so a result that should
be empty comes back with elements.  ``backend/gpu/ShapelessFused.h`` runs the steps eagerly when an
operand is empty; a site that calls MLX's ``compile`` itself skips that and brings the defect back,
unseen until an empty operand reaches it.  Fifteen sites were found compiling directly, three of
them reachable with an empty broadcast (``maximum``/``minimum``, integer ``pow``, the shifts).

The rule is ``mlx-compile`` in ``tools/check_cpp_layers.py``, where it runs at commit; this file
runs it over the tree as a test and holds the spellings it has to see.  LCD-1062's scan looked for
``shapeless`` inside the call, so ``compile(fn, true)`` and a bare ``compile(`` after a ``using``
got past it; the rule now flags every call of MLX's ``compile``, whatever it is passed.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tools"))

import check_cpp_layers

_ENGINE = _ROOT / "lucid" / "_C"
_OWNER = _ENGINE / check_cpp_layers.SHAPELESS_OWNER


def _calls(source: str) -> list[str]:
    """The MLX ``compile`` uses the rule finds in ``source``."""
    code = check_cpp_layers._strip_code(source)
    return [m.group(0) for m in check_cpp_layers.mlx_compile_calls(code)]


def _engine_present() -> None:
    if not _OWNER.is_file():
        pytest.skip("the engine sources are not part of this install")


def test_the_rule_sees_the_owner_s_compile() -> None:
    _engine_present()
    assert _calls(_OWNER.read_text(encoding="utf-8")) == ["mlx::core::compile"]


def test_no_site_compiles_outside_the_owner() -> None:
    _engine_present()
    new, _stale, allowed = check_cpp_layers.scan_engine()
    stray = [f"{v.path}:{v.line} {v.target}" for v in new + allowed if v.rule == "mlx-compile"]
    assert not stray, (
        "compile through gpu::ShapelessFused (backend/gpu/ShapelessFused.h): MLX's shapeless trace "
        f"gives an empty broadcast one element — {stray}"
    )


#: (id, source, how many uses of MLX's compile the rule finds).  The spellings LCD-1062's scan
#: missed come first; the last rows are what must not be taken for one.
_SPELLINGS = [
    ("qualified-keyword", "auto f = mlx::core::compile(g, /*shapeless=*/true);", 1),
    ("global-qualified", "auto f = ::mlx::core::compile(g, true);", 1),
    ("alias-namespace-positional", "auto f = mx::compile(g, true);", 1),
    ("lambda-over-lines", "auto f = mx::compile(\n    [](auto x) { return x; },\n    true);", 1),
    ("no-shapeless-argument", "auto f = mx::compile(g);", 1),
    (
        "unqualified-after-using-namespace",
        "using namespace mlx::core;\nauto f = compile(g, true);",
        1,
    ),
    (
        "qualified-through-a-namespace-alias",
        "namespace m = mlx::core;\nauto f = m::compile(g, true);",
        1,
    ),
    ("using-declaration-and-its-call", "using mlx::core::compile;\nauto f = compile(g, true);", 2),
    ("named-not-called", "auto f = mlx::core::compile;", 1),
    ("a-member-or-another-name", "x.compile(a); p->compile(a); recompile(a); is_compile(a);", 0),
    (
        "lucid-compile-namespace",
        "namespace lucid::compile {}\nusing lucid::compile::Tracer;\nauto* t = compile::run();",
        0,
    ),
    (
        "comment-and-string",
        'const char* s = "mx::compile(f, true)"; // mlx::core::compile(f)\n/* compile(g) */',
        0,
    ),
]


@pytest.mark.parametrize(
    ("source", "found"), [(s, n) for _, s, n in _SPELLINGS], ids=[i for i, _, _ in _SPELLINGS]
)
def test_every_spelling_of_an_mlx_compile_is_seen(source: str, found: int) -> None:
    assert len(_calls(source)) == found
