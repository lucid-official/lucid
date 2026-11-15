#!/usr/bin/env bash
# Lucid beside each `mlx` a user may have: none at all, the oldest the [mlx]
# extra admits, and the newest on PyPI.  One fresh venv per case, the same
# wheel in each, then the interop and Metal tests against the installed copy.
#
#   ./scripts/ci_mlx_compat.sh <wheel | dist dir> [none|floor|latest|<version>]...
#
# The engine never loads the `mlx` package's libmlx (LCD-292): it ships its
# own pinned build, renamed so the two coexist.  Every case checks that the
# engine still runs that pin; where `mlx` is present the tests also check
# that both run in one process, in either import order, and share Metal
# tensors by DLPack (test_vendored_mlx, test_mlx_bridge).
#
# The tests are left out of the wheel, so lucid/test is copied into each
# venv's installed lucid/ and run from outside the checkout: they import the
# wheel, never this source tree.  Venvs go in $LUCID_COMPAT_DIR (kept), or a
# temporary directory removed on exit.  uv makes them when it is on PATH,
# venv + pip otherwise.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
    echo "usage: $0 <wheel | directory of wheels> [none|floor|latest|<mlx version>]..." >&2
    exit 2
}
[ $# -ge 1 ] || usage
if [ -d "$1" ]; then
    # The newest Lucid wheel there, chosen as scripts/ci_publish.sh does.
    WHEEL=$(ls "$1"/lucid*.whl 2>/dev/null | sort -V | tail -1 || true)
    [ -n "$WHEEL" ] || { echo "$0: no lucid wheel in $1" >&2; exit 2; }
else
    WHEEL=$1
fi
[ -f "$WHEEL" ] || usage
WHEEL="$(cd "$(dirname "$WHEEL")" && pwd)/$(basename "$WHEEL")"
shift
CASES=("$@")
if [ "${#CASES[@]}" -eq 0 ]; then
    CASES=(none floor latest)
fi
# A glob that matched two wheels would pass the second as an mlx version.
for case in "${CASES[@]}"; do
    case "$case" in
        none | floor | latest | [0-9]*[0-9]) ;;
        *) echo "$0: '$case' is not none, floor, latest or an mlx version" >&2; usage ;;
    esac
done

PIN=$("$PYTHON_BIN" tools/fetch_mlx.py --print-version)
# The oldest `mlx` the [mlx] extra admits ("mlx>=0.32; ..." -> 0.32).
FLOOR=$("$PYTHON_BIN" -c "from tools.doctor import mlx_interop_floor; print(mlx_interop_floor())")

if [ -n "${LUCID_COMPAT_DIR:-}" ]; then
    WORK="$LUCID_COMPAT_DIR"
    mkdir -p "$WORK"
else
    WORK=$(mktemp -d -t lucid-mlx-compat)
    trap 'rm -rf "$WORK"' EXIT
fi
# The suite's conftest imports tools/ (benchmark timing).  The checkout's root
# cannot go on sys.path — its lucid/ would shadow the wheel — so the tests run
# from $WORK, where this link is the only import root besides the venv.
ln -sfn "$ROOT/tools" "$WORK/tools"

# pytest and pytest-benchmark exactly as the `test` extra pins them, read from
# pyproject.toml so the pin lives in one place.
TEST_TOOLS=()
while IFS= read -r req; do TEST_TOOLS+=("$req"); done < <(
    "$PYTHON_BIN" - "$ROOT/pyproject.toml" <<'PY'
import re
import sys
import tomllib

with open(sys.argv[1], "rb") as fh:
    extra = tomllib.load(fh)["project"]["optional-dependencies"]["test"]
for req in extra:
    if re.match(r"pytest(-benchmark)?\b(?!-)", req):
        print(req)
PY
)
if [ "${#TEST_TOOLS[@]}" -ne 2 ]; then
    echo "ci_mlx_compat: expected pytest and pytest-benchmark in the test extra, got: ${TEST_TOOLS[*]}" >&2
    exit 2
fi

make_venv() { # <venv dir> <requirement>...
    local venv=$1
    shift
    if command -v uv >/dev/null; then
        uv venv -q --python "$PYTHON_BIN" "$venv"
        uv pip install -q --python "$venv/bin/python" "$@"
    else
        "$PYTHON_BIN" -m venv "$venv"
        "$venv/bin/python" -m pip install --disable-pip-version-check -q "$@"
    fi
}

# Runs in the case's venv: the engine's MLX is the pin whatever `mlx` is there.
CHECK_RUNTIME='
import importlib.metadata
import importlib.util
import os

from lucid._C import engine as _C_engine

case, pin = os.environ["COMPAT_CASE"], os.environ["LUCID_MLX_PIN"]
assert _C_engine.mlx_runtime_version() == pin, (_C_engine.mlx_runtime_version(), pin)
present = importlib.util.find_spec("mlx") is not None
assert present == (case != "none"), f"case {case}: mlx installed = {present}"
version = importlib.metadata.version("mlx") if present else "absent"
print(f"    engine MLX {pin}; mlx package {version}")
'

failed=()
for case in "${CASES[@]}"; do
    case "$case" in
        none) spec="" ;;
        floor) spec="mlx==$FLOOR" ;;
        latest) spec="mlx" ;;
        *) spec="mlx==$case" ;;
    esac
    echo "==> mlx: $case${spec:+ ($spec)}"
    rm -rf "${WORK:?}/$case"
    venv="$WORK/$case/venv"
    # The test tools at the project's own pins (the `test` extra), never bare
    # `pytest`: an unpinned install took pytest 9.2 on 2026-11-14, which
    # pyproject holds back (pytest-benchmark reads Node._nodeid, 0517725afc),
    # and every case failed to import its conftest.
    make_venv "$venv" "$WHEEL" "${TEST_TOOLS[@]}" ${spec:+"$spec"}
    site=$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    # Into, not onto: the wheel already carries some lucid/test data files.
    mkdir -p "$site/lucid/test"
    cp -R lucid/test/. "$site/lucid/test/"

    if ! (cd "$WORK" && COMPAT_CASE="$case" LUCID_MLX_PIN="$PIN" "$venv/bin/python" -c "$CHECK_RUNTIME"); then
        failed+=("$case")
        continue
    fi
    # --confcutdir: the conftests live in the venv, outside the config's rootdir.
    if ! (cd "$WORK" && LUCID_MLX_PIN="$PIN" "$venv/bin/python" -m pytest -q -p no:cacheprovider \
        -c "$ROOT/pyproject.toml" --rootdir "$site" --confcutdir "$site" \
        "$site/lucid/test/unit/interop" "$site/lucid/test/unit/metal"); then
        failed+=("$case")
    fi
done

if [ "${#failed[@]}" -gt 0 ]; then
    echo "==> ci_mlx_compat: failed with mlx ${failed[*]}" >&2
    exit 1
fi
echo "==> ci_mlx_compat: green (${CASES[*]})"
