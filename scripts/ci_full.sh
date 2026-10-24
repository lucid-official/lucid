#!/usr/bin/env bash
# Full local CI gate.
# Runs: release build, all Python tests, C++ unit tests, sanitizer build,
#       all validator tools, format check.
#
# In parts, so CI can run them as parallel jobs (ci.yml ``phase-gate``):
#
#   core      the fast tier's main process
#   coreml    the Core ML tests and the family export smoke
#   checks    the parity / integration / perf tiers, every validator, the
#             audit and doctest stages, the zoo's contract and checkpoints,
#             the docs summaries, format, and the publish gate
#   zoo[:K/N] zoo compiled training — shard K of N, or all of it     (slow)
#   native    C++ unit tests, then the UBSan build                    (slow)
#
#   ./scripts/ci_full.sh                  # every part, one after another
#   ./scripts/ci_full.sh core checks      # just these
#
# The parts split what one job did in sequence: the push gate was 21
# minutes, 14 of them the fast tier, and a full run 137, of which the
# checkpoint fit took 67 and the zoo sweep 29 — each waiting on the others
# for no reason but order.  Together the parts run exactly what the single
# job did.
#
# LUCID_CI_SLOW_STAGES=0 drops the slow parts (pushes set it).  Two more
# switches serve scripts/ci_local.sh: LUCID_CI_SKIP_BUILD=1 (the caller
# installed the engine already) and LUCID_CI_SKIP_PUBLISH=1 (no pip to
# build a wheel with).

set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"

PARTS=("$@")
if [ "${#PARTS[@]}" -eq 0 ]; then
    PARTS=(core coreml checks zoo native)
fi
ZOO_SHARD=""
for part in "${PARTS[@]}"; do
    case "$part" in
        core | coreml | checks | zoo | native) ;;
        zoo:*/*) ZOO_SHARD="${part#zoo:}" ;;
        *)
            echo "unknown part: $part (core, coreml, checks, zoo[:K/N], native)" >&2
            exit 2
            ;;
    esac
done

want() {
    local part
    for part in "${PARTS[@]}"; do
        [ "${part%%:*}" = "$1" ] && return 0
    done
    return 1
}

slow() {
    [ "${LUCID_CI_SLOW_STAGES:-1}" = "1" ] && return 0
    echo "==> $1 — skipped (LUCID_CI_SLOW_STAGES=0)"
    return 1
}

# ── 1. Release build ──────────────────────────────────────────────────────────
if [ "${LUCID_CI_SKIP_BUILD:-0}" = "1" ]; then
    echo "==> Release build — skipped (LUCID_CI_SKIP_BUILD=1)"
else
    echo "==> Release build"
    "$PYTHON_BIN" -m pip install -e . --no-build-isolation
fi

# ── 2. Python fast tier (unit + numerical + stubs, no reference framework) ─
# Model-zoo tests are excluded here and run nightly instead
# (ci.yml ``nightly-models``, one process per file: 12.5 minutes on an M4
# Max, 7.4 GB at the heaviest file).  A single process carrying the whole
# zoo is what once needed 50–60 GB; per file it does not.
#
# Three processes rather than one, so each starts from a clean heap. The
# Core ML tests export real models — mask2former, detr, clip — and a
# single process carrying 1,300 earlier tests into them was SIGKILLed on
# the hosted runner partway through the export smoke file, which alone
# peaks near 5 GB on an M1 Pro. The ignores are complementary: the three
# together select exactly what the single run did (7,587 tests).
FAST_TIER_IGNORES=(
    --ignore=lucid/test/parity
    --ignore=lucid/test/integration
    --ignore=lucid/test/perf
    --ignore=lucid/test/unit/models
)
COREML_TESTS=lucid/test/unit/coreml
EXPORT_SMOKE=$COREML_TESTS/test_family_export_smoke.py

# No fast-tier process stops at its first failure.  ``-x`` once hid 91% of
# core behind whichever test failed first (LCD-530), and on coreml it went
# on hiding the export smoke: one red Core ML test meant the 61 export
# checks behind it never ran.  Fifty failures is a broken build, not a list
# worth reading on; the cap keeps a red run near its green length.
FAST_TIER_FAILURES=(--maxfail=50 --continue-on-collection-errors)

part_core() {
    echo "==> Python fast tier (non-models)"
    "$PYTHON_BIN" -m pytest lucid/test/ "${FAST_TIER_IGNORES[@]}" --ignore="$COREML_TESTS" \
        "${FAST_TIER_FAILURES[@]}" -q
}

part_coreml() {
    # Both processes run however the first one ends, so a red Core ML test
    # cannot cost the export smoke its run; the part is red if either is.
    echo "==> Python fast tier (Core ML)"
    local status=0
    "$PYTHON_BIN" -m pytest "$COREML_TESTS" --ignore="$EXPORT_SMOKE" \
        "${FAST_TIER_FAILURES[@]}" -q || status=1
    echo "==> Python fast tier (Core ML export smoke)"
    "$PYTHON_BIN" -m pytest "$EXPORT_SMOKE" "${FAST_TIER_FAILURES[@]}" -q || status=1
    return "$status"
}

part_checks() {
    local failed_stages=()
    stage() {
        local name="$1"
        shift
        echo "==> $name"
        set +e
        "$@"
        local rc=$?
        set -e
        if [ "$rc" -ne 0 ]; then
            echo "  [FAIL] $name (exit $rc)" >&2
            failed_stages+=("$name")
        fi
    }

    # ── 3. Parity tier (requires reference framework; auto-skips when missing) ──
    #
    # The two outcomes this used to fold together are not the same thing.  A
    # missing oracle means the tier skips and verifies nothing, which is a
    # warning; a parity failure means Lucid and the reference disagree about
    # what a model computes, which is a stop.  Swallowing both as "continuing"
    # meant that when the oracle *was* installed, five real defects passed the
    # gate — se_resnet computing a different function than its own published
    # weights among them.
    if ! "$PYTHON_BIN" -c "
import sys
sys.path.insert(0, '.')
from lucid.test._fixtures.ref_framework import ref_module, zoo_module
sys.exit(0 if ref_module() and zoo_module() else 1)
" 2>/dev/null; then
        echo "  [WARN] the reference framework or the model-zoo oracle is not"
        echo "         installed — the parity tier will skip, not verify."
    fi
    stage "Parity tier (vs reference framework)" \
        "$PYTHON_BIN" -m pytest lucid/test/parity/ --tb=short -q -rs

    # ── 4. Integration tier ──────────────────────────────────────────────────
    stage "Integration tier" "$PYTHON_BIN" -m pytest lucid/test/integration/ --tb=short -q

    # ── 4b. Perf tier (opt-in; uses pytest-benchmark when installed) ────────
    echo "==> Perf tier"
    "$PYTHON_BIN" -m pytest lucid/test/perf/ -m perf --tb=short -q || \
        echo "[WARN] Perf tier failed — continuing."

    # ── 5. Validator tools ───────────────────────────────────────────────────
    stage "Layer dependency check" "$PYTHON_BIN" tools/check_layers.py
    stage "Op API check" "$PYTHON_BIN" tools/check_op_api.py
    stage "Phase 1 foundation check" "$PYTHON_BIN" tools/check_phase1.py
    stage "Doxygen coverage" "$PYTHON_BIN" tools/check_doxygen.py --threshold 70
    stage "Op template conformance" "$PYTHON_BIN" tools/check_op_template.py
    stage "Kernel template coverage" "$PYTHON_BIN" tools/check_kernel_template.py
    stage "Storage API compliance" "$PYTHON_BIN" tools/check_storage_api.py
    stage "Dtype dispatch refusals" "$PYTHON_BIN" tools/check_dtype_dispatch.py
    stage "Backend method family layout" "$PYTHON_BIN" tools/check_backend_layout.py
    stage "Dead model-config fields" "$PYTHON_BIN" tools/check_dead_config_fields.py
    stage "H4 numpy guard (sanctioned bridge files only)" "$PYTHON_BIN" tools/check_numpy_h4.py

    # ── The symbol x axis sweep ──────────────────────────────────────────────
    #
    # 1,512 symbols against 33 contract axes, plus the self-check that asks
    # whether the instruments can still go red.  Half the test suite carries
    # the ``audit`` marker and ``addopts`` deselects it, so none of this ran
    # here — while its own README calls it a gate and says that one running
    # only part of it "reports clean over half a framework".
    #
    # ``--audit-only`` is the self-check and the sweep: a minute, because the
    # sweep is static contract probing rather than model work.  The suite is
    # covered by the tiers above; the doctest stage is not, and runs next.
    #
    # Exit is 0 only when every stage that ran is clean, so no output needs
    # reading.  2 means the harness broke, which is not the same as 1 — the
    # framework — and is worth saying out loud rather than folding together.
    echo "==> Symbol x axis audit"
    set +e
    "$PYTHON_BIN" -m lucid.test.audit --audit-only --json "${LUCID_AUDIT_JSON:-build/audit-report.json}"
    audit_status=$?
    set -e
    if [ "$audit_status" -eq 2 ]; then
        echo "  [ERROR] the audit harness itself failed — the sweep proved nothing"
        exit 2
    elif [ "$audit_status" -ne 0 ]; then
        echo "  [FAIL] Symbol x axis audit (exit $audit_status)" >&2
        failed_stages+=("Symbol x axis audit")
    fi

    # Docstring examples, each module against its recorded failure count in
    # lucid/test/audit/doctest.json.  Nothing above runs them: the tiers
    # collect tests, not docstrings, and --audit-only skips this stage — which
    # is how lucid.coreml shipped 63 examples that had never run.  A module
    # whose count goes up fails the gate; the floor itself is rewritten
    # locally with ``python -m lucid.test.audit --doctests-only --update-doctests``.
    echo "==> Docstring examples (against the doctest floor)"
    set +e
    "$PYTHON_BIN" -m lucid.test.audit --doctests-only
    doctest_status=$?
    set -e
    if [ "$doctest_status" -eq 2 ]; then
        echo "  [ERROR] the doctest harness itself failed — the examples proved nothing"
        exit 2
    elif [ "$doctest_status" -ne 0 ]; then
        echo "  [FAIL] Docstring examples (against the doctest floor) (exit $doctest_status)" >&2
        failed_stages+=("Docstring examples (against the doctest floor)")
    fi

    # Model-zoo family contract — verifies the 5-slot structure and Protocol
    # conformance of every family under lucid/models/.  Strict mode is OFF so
    # advisory warnings don't fail CI; flip to --strict when ready to enforce.
    # Spec: obsidian/architecture/arch-models-family-contract.md
    # The model tier as a whole is excluded above — a full pass wants 50-60
    # GB, past any hosted runner.  But the four files that gate the zoo's
    # *contract* rather than its numbers are cheap: 377 tests, 29 seconds,
    # 2.4 GB peak.  Leaving them out meant the rules the zoo states about
    # itself were enforced only on whoever remembered to run them locally —
    # "every family takes a training step" among them, which is how a family
    # reached a release branch untrained.
    stage "Model-zoo contract tests" \
        "$PYTHON_BIN" -m pytest -q -p no:randomly \
        lucid/test/unit/models/test_family_contract.py \
        lucid/test/unit/models/test_models_train_step.py \
        lucid/test/unit/models/test_models_eval_determinism.py \
        lucid/test/unit/models/test_video_preprocessing.py

    stage "Model-zoo family contract" "$PYTHON_BIN" -m tools.validate_model_zoo --runtime

    # Published checkpoints — that each one still fits the factory offering it.
    # `validate_model_zoo` checks the shape of the *code* (a factory takes
    # `pretrained: bool = False`, the family has its five slots); it never asks
    # whether the checkpoint that factory points at would actually load.  Nothing
    # did, and three families were shipping weights that could not: sk_resnet_18/34
    # (a paper floor the checkpoints were trained without), maskformer_resnet50/101
    # (a 6-layer encoder with no weights behind it, against a config comment that
    # said 0), resnest_200/269 (a dropout that moved the classifier one level down).
    # Each built, trained and exported fine — only `pretrained=True` raised.
    #
    # Reads safetensors headers over range requests, so it downloads a few KB per
    # checkpoint rather than the weights, and builds each model in shadow mode, so
    # it allocates none either: a minute for every checkpoint, where building the
    # real weights took 67 and kept this stage off pushes.  Exit 2 means some URL
    # was unreachable, which is not the same as a mismatch and must not fail the
    # gate.
    echo "==> Published checkpoint fit"
    set +e
    "$PYTHON_BIN" -m tools.check_weight_fit
    weight_fit_status=$?
    set -e
    if [ "$weight_fit_status" -eq 1 ]; then
        echo "  [FAIL] Published checkpoint fit (exit 1)" >&2
        failed_stages+=("Published checkpoint fit")
    elif [ "$weight_fit_status" -ne 0 ]; then
        echo "  [WARN] some checkpoints were unreachable — fit unverified for those"
    fi

    # Model summaries — the layer tree and parameter count the docs site renders
    # per factory.  ``validate_model_zoo`` checks the *declared* ``params=`` against
    # what a factory actually builds, and passes; it never looks at this cache, and
    # neither does the api-data drift gate, which only hashes the Griffe slugs.  So
    # nothing tied web/public/api-data/_summaries.json to the code, and it drifted
    # unseen: by 2026-08-14 it was missing 5 factories, still listed 3 that had been
    # deleted, and had the wrong parameter count for dozens — efficientdet_d7 read
    # 16.1M against a real 74.3M.  A rebuild is ~1 min because the shadow path never
    # allocates real storage, so just do it and require the result to be committed.
    echo "==> Model summaries (docs layer trees)"
    # Incremental, not --force.  The fingerprint the cache turns on hashes
    # file *contents*, so a fresh checkout reaches the same values the
    # committed sidecar holds and every unchanged factory is a hit; a
    # changed one misses and is rebuilt.  This stage was 34 of the gate's 40
    # minutes when the fingerprint keyed on mtime, which a checkout rewrites
    # — the cache could never hit on a runner, so --force was the only
    # honest option and the whole zoo was re-instantiated every run.
    #
    # Both files are diffed.  _summaries.json catches a stale tree; the
    # sidecar catches a fingerprint that disagrees with the source it claims
    # to describe, which is the one way a hand-edited cache could hide one.
    set +e
    "$PYTHON_BIN" -m tools.build_model_summaries >/dev/null
    summary_status=$?
    set -e
    local _summary_files=(
        web/public/api-data/_summaries.json
        web/public/api-data/_summaries.meta.json
    )
    if [ "$summary_status" -ne 0 ]; then
        echo "  [FAIL] Model summaries (docs layer trees) (exit $summary_status)" >&2
        failed_stages+=("Model summaries (docs layer trees)")
    elif ! git diff --quiet -- "${_summary_files[@]}"; then
        echo "  ✗ model summaries are stale — regenerated output differs from the commit." >&2
        echo "    Run: python -m tools.build_model_summaries  and commit the result." >&2
        git --no-pager diff --stat -- "${_summary_files[@]}" >&2
        failed_stages+=("Model summaries (docs layer trees)")
    else
        echo "  ✓ summaries match the current factories"
    fi

    # ── 6. Build tools ───────────────────────────────────────────────────────
    stage "Compile commands" ./scripts/build_compile_commands.sh
    stage "Format + clang-tidy" tools/check_format.sh --tidy

    # ── 7. Publish gate ──────────────────────────────────────────────────────
    # Builds its own wheel into dist/ and installs it into a fresh venv, so it
    # reads nothing the stages above installed.
    if [ "${LUCID_CI_SKIP_PUBLISH:-0}" = "1" ]; then
        echo "==> Publish gate — skipped (LUCID_CI_SKIP_PUBLISH=1)"
    else
        stage "Publish gate" ./scripts/ci_publish.sh
    fi

    if [ "${#failed_stages[@]}" -ne 0 ]; then
        echo "==> part_checks: ${#failed_stages[@]} stage(s) failed:" >&2
        for s in "${failed_stages[@]}"; do
            echo "  - $s" >&2
        done
        if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
            {
                echo "### \`checks\` failed stages (${#failed_stages[@]})"
                for s in "${failed_stages[@]}"; do
                    echo "- \`$s\`"
                done
            } >> "$GITHUB_STEP_SUMMARY"
        fi
        exit 1
    fi
}

part_zoo() {
    # Zoo compiled training — one training step of every model-zoo family,
    # compiled (make_step) and compared with eager: the loss, every gradient and
    # every buffer the step updates.  The op matrices prove each op alone; a model
    # is where they meet, and where a missing VJP or a value frozen at trace time
    # shows.  One child process per family, since an MPSGraph abort kills the
    # interpreter.  The pairs that run eager are listed with their reasons in
    # ``_zoo_matrix.EXPECTED``, strict both ways — a pair that starts compiling
    # fails until its entry is deleted.  ~10 minutes on an M1 Pro, 29 on a
    # hosted runner, so nightly, and sharded across jobs there.
    slow "Zoo compiled training" || return 0
    echo "==> Zoo compiled training${ZOO_SHARD:+ (shard $ZOO_SHARD)}"
    "$PYTHON_BIN" -m lucid.test.unit.compile._zoo_matrix --sweep \
        ${ZOO_SHARD:+--shard "$ZOO_SHARD"}
}

part_native() {
    # ── C++ unit tests (GoogleTest) ──────────────────────────────────────────
    # Its own build tree — the old step looked for the one setup.py leaves, under
    # a path pip no longer uses, and skipped with a warning on every run: the
    # tests never ran in CI.  A failure fails the gate.  Slow stage: the tree is a
    # second full engine build.
    if slow "C++ unit tests"; then
        echo "==> C++ unit tests"
        ./scripts/ci_cpp_tests.sh
    fi

    # ── UBSan build ──────────────────────────────────────────────────────────
    # LUCID_CI_SLOW_STAGES=0 skips the stages that cost the most and almost
    # never change with a push: this sanitizer build (a debug engine, then the
    # ops / autograd / nn unit tests under it — a UB report fails the gate), the
    # zoo's compiled-training sweep (a process per family) and the C++ unit tests
    # above (a second engine build).  CI sets it for pushes; the nightly
    # schedule, manual runs and a local ``ci_full.sh`` run everything.
    # Last of all: it replaces the installed engine with the sanitizer build,
    # and every stage before it reads the release one.
    if slow "ASan + UBSan build + fast tests"; then
        echo "==> ASan + UBSan build + fast tests"
        ./scripts/ci_sanitizer.sh asan
    fi
}

for part in core coreml checks zoo native; do
    if want "$part"; then
        "part_$part"
    fi
done

echo "==> ci_full: green (${PARTS[*]})"
