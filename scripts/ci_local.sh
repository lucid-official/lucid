#!/usr/bin/env bash
# CI's push gate on this machine, before the push.
#
# The push gate is three parts of scripts/ci_full.sh — core, coreml and
# checks — each a job on a hosted runner.  A push that fails there costs the
# queue, the runner's minutes and the round trip before anyone knows, and
# the push that fixes it starts the clock again.  This runs the same parts
# here, side by side, in interpreters held to what CI installs
# (scripts/ci_mirror), and says which part failed before anything leaves.
#
#   ./scripts/ci_local.sh                     # core, coreml and checks at once
#   ./scripts/ci_local.sh checks zoo:1/2      # any parts ci_full.sh knows
#   LUCID_LOCAL_NO_BUILD=1 ./scripts/ci_local.sh    # the engine is current
#
# Different from CI on purpose: the engine is built with uv, as every local
# build is, and the publish gate is skipped — a uv venv has no pip to build
# the wheel with, and CI still runs it.  One log per part, in
# $LUCID_LOCAL_LOGS (default: a fresh temporary directory).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
VENV="${VENV:-$ROOT/.venv}"

PARTS=("$@")
if [ "${#PARTS[@]}" -eq 0 ]; then
    PARTS=(core coreml checks)
fi
LOGS="${LUCID_LOCAL_LOGS:-$(mktemp -d -t lucid-ci-local)}"
mkdir -p "$LOGS"

export PATH="$VENV/bin:$PATH"
export PYTHON_BIN="$VENV/bin/python"
# The pinned MLX build's minos ([tool.lucid.mlx] macos), as CI exports it.
MACOSX_DEPLOYMENT_TARGET="$("$PYTHON_BIN" tools/fetch_mlx.py --print-macos)"
export MACOSX_DEPLOYMENT_TARGET
# The major ci.yml pins, where this venv has it.  A worktree's light venv
# has none, and tools/check_format.sh then finds the main checkout's; set
# to a missing file, it would have no clang-format to run (LCD-572).
if [ -x "$VENV/bin/clang-format" ]; then
    export CLANG_FORMAT="$VENV/bin/clang-format"
fi
export PYTHONPATH="$ROOT/scripts/ci_mirror${PYTHONPATH:+:$PYTHONPATH}"
# CI's pytest loads one plugin, pytest-benchmark.  Any other a venv carries
# would be refused by the mirror as pytest starts.
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTEST_PLUGINS=pytest_benchmark.plugin,hypothesis.extra.pytestplugin
export LUCID_CI_SKIP_BUILD=1
export LUCID_CI_SKIP_PUBLISH=1

trap 'kill 0' INT TERM

if [ "${LUCID_LOCAL_NO_BUILD:-0}" = "1" ]; then
    echo "==> Engine build — skipped (LUCID_LOCAL_NO_BUILD=1)"
else
    echo "==> Engine build"
    # Where Xcode is installed its linker wants its own SDK: the Command
    # Line Tools' SDK 27 lists an architecture Xcode 26's ld rejects.
    # Without Xcode.app the default SDK is the right one.
    XCODE_SDK=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
    if [ -z "${SDKROOT:-}" ] && [ -d "$XCODE_SDK" ]; then
        SDKROOT="$XCODE_SDK" VIRTUAL_ENV="$VENV" \
            uv pip install -e . --no-build-isolation >"$LOGS/build.log" 2>&1 || build_status=$?
    else
        VIRTUAL_ENV="$VENV" \
            uv pip install -e . --no-build-isolation >"$LOGS/build.log" 2>&1 || build_status=$?
    fi
    if [ "${build_status:-0}" -ne 0 ]; then
        tail -30 "$LOGS/build.log"
        echo "==> ci_local: the engine did not build ($LOGS/build.log)" >&2
        exit 1
    fi
fi

echo "==> ${PARTS[*]} (logs: $LOGS)"
started=$(date +%s)
for part in "${PARTS[@]}"; do
    name="${part//[:\/]/_}"
    (
        t0=$(date +%s)
        set +e
        ./scripts/ci_full.sh "$part" >"$LOGS/$name.log" 2>&1
        echo "$? $(($(date +%s) - t0))" >"$LOGS/$name.status"
    ) &
done
wait

failed=()
for part in "${PARTS[@]}"; do
    name="${part//[:\/]/_}"
    read -r status seconds <"$LOGS/$name.status"
    if [ "$status" = "0" ]; then
        mark="ok"
    else
        mark="FAIL ($status)"
        failed+=("$part")
    fi
    printf '    %-10s %-10s %4ds\n' "$part" "$mark" "$seconds"
done

if [ "${#failed[@]}" -gt 0 ]; then
    for part in "${failed[@]}"; do
        name="${part//[:\/]/_}"
        echo
        echo "── $part: the last 40 lines of $LOGS/$name.log"
        tail -40 "$LOGS/$name.log"
    done
    echo
    echo "==> ci_local: ${failed[*]} failed" >&2
    exit 1
fi
echo "==> ci_local: green in $(($(date +%s) - started))s"
