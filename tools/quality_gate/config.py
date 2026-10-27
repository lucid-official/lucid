"""What the gate measures, and why each threshold sits where it does.

Every setting here is part of the ratchet's contract: changing one changes
what every count in ``tools/quality_baseline.json`` means, so a change here
ships with ``--rebaseline --reason`` in the same commit.  ``contract()``
lists, per collector, what its counts depend on; the baseline records it,
and a run refuses a baseline whose record differs for a collector it runs.
"""

import ast
import hashlib
from pathlib import Path

BASELINE_PATH = "tools/quality_baseline.json"
BASELINE_VERSION = 1

# ── Scope ─────────────────────────────────────────────────────────────────
# Production Python and the C++ engine.  Tests are excluded from the counts
# (they are scanned by vulture as *users* of the API, see deadcode.py).
PY_ROOT = "lucid/"
PY_EXCLUDE_PREFIXES = ("lucid/test/",)
CPP_ROOT = "lucid/_C/"
CPP_SUFFIXES = (".cpp", ".h", ".hpp", ".mm", ".cc")

# ── ruff ──────────────────────────────────────────────────────────────────
# Added on top of the project's own [tool.ruff.lint] (E, F, W) with
# --extend-select, so RUF100 judges a `noqa` against the rules the project
# actually enables.  Deliberately off:
#   FBT      — reference-compatible signatures take positional bools.
#   TRY003   — a library's exceptions carry their own messages.
#   PLR0913  — public signatures mirror the reference framework's; argument
#              counts are judged by lizard on `_` internals only.
RUFF_EXTEND_SELECT = (
    "B,BLE,S110,TRY,PGH,C90,PLR0911,PLR0912,PLR0915,ERA,ARG,SIM,RET,PIE,PERF,UP,RUF100"
)
RUFF_IGNORE = "TRY003"

# ── lizard (Python and C++) ───────────────────────────────────────────────
# Physical length in lines.  100 keeps the long tail visible (260 functions
# on 2026-10-05; 83 over 150) while a new function's budget stays ~80.
FUNCTION_LENGTH_MAX = 100
# Cyclomatic complexity: lizard's own default warning level.
FUNCTION_CCN_MAX = 15
# Parameters, for `_`-prefixed Python internals only — public signatures
# follow the reference framework and are not ours to shorten.
INTERNAL_ARGS_MAX = 7
# `function-length` / `function-ccn` count the functions over a limit, so a
# function already over it could grow without the count moving (DS-29:
# `_build_plan` 425 -> 480 lines, CCN 76 -> 85).  The `-excess` rules count
# how far over: per file, the sum of max(0, value - limit).

# ── file size cap ─────────────────────────────────────────────────────────
# The count is the number of lines *over* the cap, so a file already over it
# may not grow and a new file may not start over it.
LOC_CAP_PY = 1500
LOC_CAP_CPP = 3000
LOC_CAP_BACKEND_METHODS = 1500
BACKEND_METHODS_PREFIXES = ("lucid/_C/backend/cpu/methods/", "lucid/_C/backend/gpu/methods/")
# Rules whose count is a size, not a number of findings: class totals count
# one per file so 14954-line headers do not drown every other class.
PER_FILE_RULES = frozenset({"file-loc-over-cap", "function-length-excess", "function-ccn-excess"})

# ── vulture ───────────────────────────────────────────────────────────────
VULTURE_MIN_CONFIDENCE = 60
VULTURE_IGNORE_NAMES = ("__*__",)
# Registration decorators: the function is reached through the registry the
# decorator fills, never by its name.  Each one hides every function it
# decorates, so the list is contract.
VULTURE_IGNORE_DECORATORS = (
    "@_emitter",
    "@register_kl",
    "@register_model",
    "@register_weights",
    "@_register_preset",
    "@_register",
)
# vulture matches a whitelisted name in *every* module (its used-name set is
# global), so each entry hides any dead symbol of that name anywhere.  Its
# entries are contract: recorded by a hash of the file's AST, so comments
# and formatting stay free and a name added or removed is a rebaseline (DS-18).
VULTURE_WHITELIST = Path(__file__).with_name("vulture_whitelist.py")

# ── engine bindings ───────────────────────────────────────────────────────
# A function in the generated engine stub that no production module reaches
# is a dead entry point (DS-24): built, bound and stubbed for nobody.  The
# bindings that exist for tests and dev tools are listed with their consumer.
# Only their names are contract.
ENGINE_STUB = "lucid/_C/engine.pyi"
ENGINE_TEST_ONLY_BINDINGS = {
    "_has_tensor_hooks": "tensor-hook table probe, lucid/test/unit/autograd",
    "amp_policy_for": "autocast table checks, lucid/test/unit/amp (until LCD-500-A1)",
    "amp_policy_table": "tools/gen_pyi.py",
    "host_sync_count": "host-sync budgets, lucid/test/unit/{cost,nn}",
    "metal_active_memory": "memory probes, lucid/test/unit/{interop,optim}",
    "metal_peak_memory": "peak-memory asserts, lucid/test/unit/{interop,nn/functional}",
    "metal_reset_peak_memory": "resets between those peak-memory asserts",
    "mlx_runtime_version": "tools/doctor.py, scripts/ci_mlx_compat.sh",
    "op_registry_all": "tools/gen_pyi.py, tools/support_manifest.py",
}

# ── jscpd ─────────────────────────────────────────────────────────────────
# 8 lines / 50 tokens: the 2026-10-05 probe's "8-line window repeated in 3+
# files" class.  Pinned: a different jscpd tokenises differently.
JSCPD_VERSION = "5.4.0"
JSCPD_MIN_LINES = 8
JSCPD_MIN_TOKENS = 50
JSCPD_FORMATS = "python,cpp,objectivec,c-header,cpp-header"
JSCPD_CROSS_FORMATS = "cpp,c-header,cpp-header,objectivec"

# ── mypy (full mode only — the slow one) ──────────────────────────────────
# The documented type check (CLAUDE.md, tools/README.md), run verbatim: the
# gate only appends the auditing codes below and its output format, and it
# counts every error the command prints.  Before LCD-317 it counted lucid/
# files only, so an error in a tools/ module that lucid/ imports
# (tools/bench_compile.py via a test) failed the documented command while
# the gate stayed green.
MYPY_COMMAND = ("mypy", "--strict", "lucid/")
# On top of mypy.ini (strict).  `--platform darwin` so a Linux CI runner
# sees the same `sys.platform` branches as the Macs.
MYPY_FLAGS = (
    "--warn-unused-ignores",
    "--enable-error-code=ignore-without-code",
    "--enable-error-code=redundant-cast",
    "--enable-error-code=redundant-expr",
    "--enable-error-code=truthy-bool",
    "--enable-error-code=possibly-undefined",
    "--platform=darwin",
)

# ── semgrep ───────────────────────────────────────────────────────────────
SEMGREP_RULES_DIR = "tools/semgrep/rules"

# ── clang-tidy ────────────────────────────────────────────────────────────
# DS-3 (LCD-262): bugprone-*, the safe subset of cppcoreguidelines-*
# (pro-bounds, init-variables, narrowing), readability-function-size, and
# performance-*.  Opt-in (--collectors clang-tidy): needs compile_commands.json
# and ~30 s across all 249 translation units.
CLANG_TIDY_CHECKS = (
    "-*,"
    "bugprone-*,"
    "cppcoreguidelines-pro-bounds-*,"
    "cppcoreguidelines-init-variables,"
    "cppcoreguidelines-narrowing-conversions,"
    "readability-function-size,"
    "performance-*"
)
CLANG_TIDY_HEADER_FILTER = r".*lucid/_C/.*"

# ── defect classes (obsidian roadmap-deslop-quality-gate) ─────────────────
CLASSES = {
    "D1": "type laundering",
    "D2": "duplication",
    "D3": "god units",
    "D4": "silent errors",
    "D5": "error taxonomy",
    "D7": "dead code",
    "S": "style / modernise",
}

# (tool, rule prefix) -> class; the longest matching prefix wins.
_CLASS_RULES = (
    ("counters", "type-ignore", "D1"),
    ("counters", "cast", "D1"),
    ("counters", "noqa", "D1"),
    ("counters", "any", "D1"),
    ("counters", "file-loc", "D3"),
    ("ruff", "PGH", "D1"),
    ("ruff", "C90", "D3"),
    ("ruff", "PLR", "D3"),
    ("ruff", "B", "D4"),
    ("ruff", "BLE", "D4"),
    ("ruff", "S110", "D4"),
    ("ruff", "TRY", "D4"),
    ("ruff", "ARG", "D7"),
    ("ruff", "ERA", "D7"),
    ("ruff", "RUF100", "D7"),
    ("ruff", "", "S"),
    ("mypy", "unused-ignore", "D1"),
    ("mypy", "ignore-without-code", "D1"),
    ("mypy", "redundant-cast", "D1"),
    ("mypy", "", "D4"),
    ("lizard", "", "D3"),
    ("vulture", "", "D7"),
    ("engine", "", "D7"),
    ("jscpd", "", "D2"),
    ("semgrep", "", "D5"),
    ("clang-tidy", "readability-function-size", "D3"),
    ("clang-tidy", "performance-", "S"),
    ("clang-tidy", "", "D4"),
)


def class_of(tool: str, rule: str) -> str:
    best = ("", "S")
    for t, prefix, cls in _CLASS_RULES:
        if t == tool and rule.startswith(prefix) and len(prefix) >= len(best[0]):
            best = (prefix, cls)
    return best[1]


def ast_digest(path: Path) -> str:
    """A Python file's AST, hashed: what it says, not how it is laid out or commented."""
    tree = ast.dump(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    return hashlib.sha256(tree.encode()).hexdigest()[:16]


def contract() -> dict[str, dict[str, object]]:
    """Per collector, the settings its counts depend on; the baseline records them.

    Counts taken under one contract are not comparable with counts taken
    under another, so a run refuses a baseline whose record differs for a
    collector it runs, and only ``--rebaseline`` writes the record.  An entry
    added to an allowlist lowers a count exactly as a looser threshold does:
    before DS-18 a whitelist edit reached the baseline as a plain ``--update``.
    """
    return {
        "lizard": {
            "function_length_max": FUNCTION_LENGTH_MAX,
            "function_ccn_max": FUNCTION_CCN_MAX,
            "internal_args_max": INTERNAL_ARGS_MAX,
        },
        "counters": {
            "loc_cap_py": LOC_CAP_PY,
            "loc_cap_cpp": LOC_CAP_CPP,
            "loc_cap_backend_methods": LOC_CAP_BACKEND_METHODS,
        },
        "ruff": {"ruff_extend_select": RUFF_EXTEND_SELECT, "ruff_ignore": RUFF_IGNORE},
        "vulture": {
            "vulture_min_confidence": VULTURE_MIN_CONFIDENCE,
            "vulture_ignore_names": ",".join(VULTURE_IGNORE_NAMES),
            "vulture_ignore_decorators": ",".join(VULTURE_IGNORE_DECORATORS),
            "vulture_whitelist_ast_sha256": ast_digest(VULTURE_WHITELIST),
        },
        "jscpd": {
            "jscpd": (
                f"{JSCPD_VERSION} lines>={JSCPD_MIN_LINES} tokens>={JSCPD_MIN_TOKENS} "
                f"formats={JSCPD_FORMATS} cross={JSCPD_CROSS_FORMATS}"
            )
        },
        "mypy": {"mypy_flags": " ".join(MYPY_FLAGS)},
        "engine": {"engine_test_only_bindings": ",".join(sorted(ENGINE_TEST_ONLY_BINDINGS))},
        "clang-tidy": {
            "clang_tidy_checks": CLANG_TIDY_CHECKS,
            "clang_tidy_header_filter": CLANG_TIDY_HEADER_FILTER,
        },
    }
