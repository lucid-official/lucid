#!/usr/bin/env bash
# tools/check_format.sh — Run clang-format compliance check on C++ sources.
#
# Usage:
#   tools/check_format.sh              # check only (exits non-zero on violation)
#   tools/check_format.sh --format-only  # same as above (alias for pre-commit)
#   tools/check_format.sh --tidy       # also run clang-tidy if available
#   tools/check_format.sh --which      # print the clang-format the check runs
#
# Exit status: 0 every file is formatted, 1 a file needs format, 2 cannot
# check.  "Cannot check" means no clang-format could be run, or it failed
# for a reason other than format (a bad .clang-format, an unreadable file).
# A tool that never ran gives no verdict: a missing clang-format used to be
# reported as every file needing format (LCD-572).
#
# The clang-format is the first that exists of: $CLANG_FORMAT (that one or
# none); this checkout's .venv; the main checkout's .venv (a worktree's
# light venv has none); `clang-format` on PATH, where CI's pinned pip
# install lands; Homebrew's LLVM.  PATH goes before Homebrew because a
# runner image's Homebrew LLVM is not the major CI pins, and an older
# major formats differently.

set -euo pipefail

cd "$(dirname "$0")/.."

TIDY=0
WHICH=0
for arg in "$@"; do
  case "$arg" in
    --tidy) TIDY=1 ;;
    --which) WHICH=1 ;;
  esac
done

cannot_check() {
  echo "[check_format] cannot check: $*" >&2
  exit 2
}

# The first clang-format candidate that is an executable file, or nothing.
find_clang_format() {
  if [[ -n "${CLANG_FORMAT:-}" ]]; then
    echo "$CLANG_FORMAT"
    return
  fi
  local candidates=("$PWD/.venv/bin/clang-format")
  local common
  common="$(git rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
  if [[ -n "$common" ]]; then
    candidates+=("$(dirname "$common")/.venv/bin/clang-format")
  fi
  candidates+=("$(command -v clang-format || true)")
  candidates+=("${HOMEBREW_PREFIX:-/opt/homebrew}/opt/llvm/bin/clang-format")
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -f "$candidate" && -x "$candidate" ]]; then
      echo "$candidate"
      return
    fi
  done
}

CLANG_FORMAT="$(find_clang_format)"
if [[ -z "$CLANG_FORMAT" ]] || ! command -v "$CLANG_FORMAT" >/dev/null; then
  cannot_check "clang-format not found${CLANG_FORMAT:+ at $CLANG_FORMAT}" \
    "(set CLANG_FORMAT=/path/to/clang-format)"
fi
if ! VERSION="$("$CLANG_FORMAT" --version 2>&1)" || [[ "$VERSION" != *"clang-format version"* ]]; then
  cannot_check "$CLANG_FORMAT does not run as clang-format (${VERSION:-no output})" \
    "— set CLANG_FORMAT=/path/to/clang-format"
fi
if [[ $WHICH -eq 1 ]]; then
  echo "$CLANG_FORMAT"
  exit 0
fi

CPP_FILES=$(find lucid/_C -name "*.h" -o -name "*.hpp" -o -name "*.cpp" | grep -v "/_C/test/" || true)

if [[ -z "$CPP_FILES" ]]; then
  echo "[check_format] No C++ files found."
  exit 0
fi

echo "[check_format] ${VERSION} (${CLANG_FORMAT})"
FAILED=0
for f in $CPP_FILES; do
  if ! OUT="$("$CLANG_FORMAT" --dry-run --Werror "$f" 2>&1)"; then
    # A format difference is reported under this warning flag; any other
    # failure is the tool not reading the file or the style.
    if [[ "$OUT" != *"-Wclang-format-violations"* ]]; then
      cannot_check "$CLANG_FORMAT failed on $f: ${OUT%%$'\n'*}"
    fi
    echo "[check_format] NEEDS FORMAT: $f"
    FAILED=1
  fi
done

if [[ $FAILED -ne 0 ]]; then
  echo "[check_format] Run: clang-format -i lucid/_C/**/*.{h,hpp,cpp}"
  exit 1
fi

echo "[check_format] all C++ files formatted correctly"

if [[ $TIDY -eq 1 ]]; then
  CLANG_TIDY="${CLANG_TIDY:-clang-tidy}"
  if command -v "$CLANG_TIDY" &>/dev/null; then
    echo "[check_format] Running clang-tidy..."
    if [[ -f "build/compile_commands.json" ]]; then
      BUILD_DB="build/compile_commands.json"
    else
      BUILD_DB="build/temp.macosx-10.15-universal2-cpython-314/lucid__C_engine/compile_commands.json"
    fi
    if [[ -f "$BUILD_DB" ]]; then
      $CLANG_TIDY -p "$BUILD_DB" $CPP_FILES 2>&1 | grep -E "warning:|error:" | head -40 || true
    else
      echo "[check_format] compile_commands.json not found — skipping clang-tidy"
    fi
  else
    echo "[check_format] clang-tidy not found — skipping"
  fi
fi
