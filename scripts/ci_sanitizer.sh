#!/usr/bin/env bash
# CI gate: build the engine extension with ASan + UBSan (or UBSan-only), verify
# that a seeded heap out-of-bounds write fails under AddressSanitizer, and run
# the targeted engine suites (ops, autograd, nn, linalg, indexing/tensor, and
# audit smoke) under sanitizers.
#
# Usage:
#   ./scripts/ci_sanitizer.sh                # ASan + UBSan (default, LUCID_SANITIZE=address,undefined)
#   ./scripts/ci_sanitizer.sh asan           # ASan + UBSan (LUCID_SANITIZE=address,undefined)
#   ./scripts/ci_sanitizer.sh ubsan          # UBSan only (LUCID_SANITIZE=undefined)
#
# On macOS, the standard Python launcher in /Library/Frameworks/Python.framework
# carries the Hardened Runtime flag, which causes dyld to strip
# DYLD_INSERT_LIBRARIES on execve.  In `asan` mode this script links a tiny
# C launcher (`Py_BytesMain`) with `-fsanitize=address,undefined` into
# `build/asan-venv/bin/python3` so `libclang_rt.asan_osx_dynamic.dylib` is an
# LC_LOAD_DYLIB of the main executable itself — requiring no DYLD_INSERT_LIBRARIES
# and propagating automatically to child `sys.executable` subprocesses.

set -euo pipefail

cd "$(dirname "$0")/.."

MODE="${1:-asan}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
    if [ -x ".venv/bin/python3" ]; then
        PYTHON_BIN=".venv/bin/python3"
    else
        PYTHON_BIN="python3"
    fi
fi
PYTEST_TARGET="${PYTEST_TARGET:-lucid/test/unit/ops lucid/test/unit/autograd lucid/test/unit/nn lucid/test/unit/linalg lucid/test/unit/tensor lucid/test/audit/test_audit_smoke.py}"

case "$MODE" in
    ubsan)
        SANITIZE="undefined"
        EXTRA_OPTS=("UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1")
        ;;
    asan)
        SANITIZE="address,undefined"
        # detect_leaks=0 because Python's deferred GC creates apparent leaks
        # that aren't real (we have an explicit gc-aware leak test elsewhere).
        EXTRA_OPTS=(
            "ASAN_OPTIONS=detect_leaks=0:abort_on_error=0:halt_on_error=1"
            "UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1"
        )
        ;;
    *)
        echo "Unknown mode: $MODE (use 'asan' or 'ubsan')" >&2
        exit 1
        ;;
esac

echo "==> Building engine with LUCID_SANITIZE=$SANITIZE"
if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    LUCID_SANITIZE="$SANITIZE" "$PYTHON_BIN" -m pip install -e . --no-build-isolation --no-deps \
        2>&1 | tail -5
else
    LUCID_SANITIZE="$SANITIZE" uv pip install --python "$PYTHON_BIN" --no-build-isolation --no-deps -e . \
        2>&1 | tail -5
fi

RUN_PYTHON="$PYTHON_BIN"
if [ "$MODE" = "asan" ]; then
    echo "==> Verifying seeded heap out-of-bounds write fails under AddressSanitizer"
    mkdir -p build/asan-venv/bin
    PROBE_BIN="build/asan-venv/oob_probe"
    cat <<'CPP' | clang++ -x c++ -std=c++20 -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined - -o "$PROBE_BIN"
#include <cstdlib>
__attribute__((noinline)) void write_oob(float* p, int idx) {
    volatile float* sink = p;
    sink[idx] = 1.0f;
}
int main() {
    auto* buf = new float[4];
    write_oob(buf, 4);
    delete[] buf;
    return 0;
}
CPP
    set +e
    PROBE_OUT="$(env "${EXTRA_OPTS[@]}" "$PROBE_BIN" 2>&1)"
    PROBE_RC=$?
    set -e
    if [ "$PROBE_RC" -eq 0 ] || ! printf '%s' "$PROBE_OUT" | grep -q "AddressSanitizer: heap-buffer-overflow"; then
        echo "ERROR: seeded OOB write did not fail under AddressSanitizer (rc=$PROBE_RC)" >&2
        printf '%s\n' "$PROBE_OUT" >&2
        exit 1
    fi
    echo "    ✓ seeded heap-buffer-overflow caught by AddressSanitizer (exit $PROBE_RC)"

    echo "==> Linking ASan-enabled Python launcher in build/asan-venv/bin/python3"
    PY_META="$("$PYTHON_BIN" - <<'PY'
import sys, sysconfig
from pathlib import Path
print(sysconfig.get_config_var("LIBDIR") or "")
print(sysconfig.get_config_var("LDVERSION") or sysconfig.get_python_version())
print(Path(sys.prefix).resolve())
PY
)"
    PY_LINES=()
    while IFS= read -r line; do
        PY_LINES+=("$line")
    done <<EOF
$PY_META
EOF
    LIBDIR="${PY_LINES[0]}"
    PYVER="${PY_LINES[1]}"
    PREFIX_DIR="${PY_LINES[2]}"
    rm -f build/asan-venv/lib build/asan-venv/pyvenv.cfg
    ln -s "$PREFIX_DIR/lib" build/asan-venv/lib
    if [ -f "$PREFIX_DIR/pyvenv.cfg" ]; then
        cp "$PREFIX_DIR/pyvenv.cfg" build/asan-venv/pyvenv.cfg
    fi
    printf 'int Py_BytesMain(int, char**);\nint main(int argc, char** argv) { return Py_BytesMain(argc, argv); }\n' \
        | clang -x c - -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined \
            -L"$LIBDIR" -lpython"$PYVER" -Wl,-rpath,"$LIBDIR" \
            -o build/asan-venv/bin/python3
    ln -sf python3 build/asan-venv/bin/python
    RUN_PYTHON="$(pwd)/build/asan-venv/bin/python3"
fi

echo "==> Running targeted engine suites under $MODE ($SANITIZE)"
# A sanitizer writes its report to fd 2 and exits.  pytest's fd capture holds fd 2 in
# a temporary file during each test, which the dying process never prints, so the job
# ended with a bare exit 1 mid-line.  The pytest run therefore sends each report to a
# per-process file (log_path) and prints those files when the run fails.  Only this run:
# the seeded probe above reads its report from stderr.
SAN_LOG_DIR="$(pwd)/build/sanitizer-logs"
rm -rf "$SAN_LOG_DIR"
mkdir -p "$SAN_LOG_DIR"
RUN_OPTS=()
for opt in "${EXTRA_OPTS[@]}"; do
    name="${opt%%=*}"
    RUN_OPTS+=("$opt:log_path=$SAN_LOG_DIR/${name%%_OPTIONS}")
done
set +e
# shellcheck disable=SC2086 — the target is a list of paths
env "${RUN_OPTS[@]}" "$RUN_PYTHON" -m pytest $PYTEST_TARGET --tb=short -q
PYTEST_RC=$?
set -e
shopt -s nullglob
SAN_LOGS=("$SAN_LOG_DIR"/*)
shopt -u nullglob
if [ "$PYTEST_RC" -ne 0 ] || [ "${#SAN_LOGS[@]}" -gt 0 ]; then
    for log in "${SAN_LOGS[@]}"; do
        echo "==> sanitizer report: $log"
        cat "$log"
    done
    [ "$PYTEST_RC" -ne 0 ] && exit "$PYTEST_RC"
    echo "ERROR: the run passed but a sanitizer wrote a report" >&2
    exit 1
fi

echo "==> $MODE ($SANITIZE): green"
