#!/usr/bin/env bash
# Finish a regenerated api-data tree, commit it, and push it onto a main
# that may have moved while it was built.
#
# docs-api-data.yml rebuilds every page for about two hours.  Whatever
# lands on main in that window used to make the closing ``git push`` fail
# with "fetch first" (10-20, 10-23), and the rebuild was thrown away.  This
# carries the refresh onto the new main instead:
#
#   1. Run the post-processors and re-baseline the drift cache, so the
#      commit is the whole deployable tree that ``pnpm prebuild`` would
#      make — the Docs workflow fails on any post-processor drift.
#   2. Commit and push.
#   3. On a rejection, fetch main and apply the refresh to its new tip as
#      a three-way merge, the refresh winning a hunk both sides changed.
#      Then rebuild, at that tip, only the pages its new commits made
#      stale: the slugs whose sources moved, by the drift check run
#      against the refresh's own baseline; the synthesised pages, which
#      read the whole tree, when anything under lucid/ moved; and the
#      model pages embedding a factory summary that _summaries.json
#      changed, which no source hash sees.  Building at the tip keeps
#      every permalink on a commit that stays on main.  Post-process,
#      re-baseline, commit, and push again.
#
# The push is tried LUCID_API_DATA_PUSH_ATTEMPTS times (default 3).  It
# fails loudly — an ::error:: line and the step summary, naming the slugs
# that moved on main during the rebuild — when the attempts run out, when
# the generator itself changed on main (every page is then stale), when
# the merge leaves a conflict it cannot settle, when a push is refused
# while main has not moved, since that is no race, and when any command
# fails (the ERR trap).
#
# Exit 0: pushed, or nothing to push.  Exit 1: not pushed.
#
# Bash 3.2 compatible (macOS /bin/bash runs its test), so no mapfile and no
# arrays that may be empty: the slug list is a newline-separated string,
# word-split where it is passed on.

# -E: the ERR trap reaches functions, subshells and command substitutions.
set -eEuo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
REMOTE="${LUCID_API_DATA_REMOTE:-origin}"
BRANCH="${LUCID_API_DATA_BRANCH:-main}"
ATTEMPTS="${LUCID_API_DATA_PUSH_ATTEMPTS:-3}"

API=web/public/api-data
GENERATOR=web/scripts/build-api-data.py
# The pages build-api-data.py synthesises from the whole lucid/ tree.  The
# drift check cannot see them: they have no source directory to hash.
SYNTH="lucid.ops lucid.creation lucid.ops.composite"
MESSAGE="docs(api-data): nightly refresh against the current source"

# A failure is reported once.  It can be raised in a subshell — a command
# substitution — and then fails its caller too, so the mark is a file.
REPORTED="$(mktemp -d "${TMPDIR:-/tmp}/api-data-push.XXXXXX")"
trap 'rm -rf "$REPORTED"' EXIT

output() {
    if [ -n "${GITHUB_OUTPUT:-}" ]; then
        echo "$1" >>"$GITHUB_OUTPUT"
    fi
}

fail() { # <why> [moved slug ...]
    local why="$1"
    shift
    touch "$REPORTED/done"
    echo "::error::api-data was not pushed: $why" >&2
    if [ "$#" -gt 0 ]; then
        echo "  moved on $BRANCH during the rebuild:" >&2
        printf '    - %s\n' "$@" >&2
    fi
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
        {
            echo "### api-data was not pushed"
            echo "$why"
            if [ "$#" -gt 0 ]; then
                echo
                echo "Moved on \`$BRANCH\` during the rebuild:"
                printf -- '- `%s`\n' "$@"
            fi
        } >>"$GITHUB_STEP_SUMMARY"
    fi
    output "pushed=false"
    exit 1
}

on_error() { # <status> <line> <command>
    if [ -e "$REPORTED/done" ]; then
        exit "$1"
    fi
    fail "\`$3\` failed with exit $1 (scripts/ci_api_data_push.sh:$2)"
}
trap 'on_error $? $LINENO "$BASH_COMMAND"' ERR

count() { echo "$#"; }

finish() {
    (
        cd web
        for step in build-cross-links build-usedby link-citations build-bibtex build-meta; do
            "$PYTHON_BIN" "scripts/$step.py"
        done
        "$PYTHON_BIN" scripts/build-api-data-cached.py --write-cache
    )
}

commit() { # 0 committed, 1 nothing to commit
    # Called as a condition, where neither set -e nor the ERR trap acts,
    # so each failure is raised here: a failed commit is not "nothing new".
    local status=0
    git add -A -- "$API" || fail "git add of $API failed"
    git diff --cached --quiet || status=$?
    case "$status" in
        0) return 1 ;;
        1) ;;
        *) fail "git diff --cached failed with exit $status" ;;
    esac
    git commit -q -m "$MESSAGE" || fail "git commit failed"
}

carry() { # <ours> <tip> — the work tree becomes <tip> with <ours> applied
    local ours="$1" tip="$2" unmerged path
    git checkout -q --detach "$tip"
    if git cherry-pick --no-commit -X theirs "$ours" >/dev/null; then
        return 0
    fi
    unmerged="$(git diff --name-only --diff-filter=U)"
    if [ -z "$unmerged" ]; then
        fail "the refresh could not be applied to $BRANCH at $tip"
    fi
    # -X theirs settles every hunk; what is left is a page one side
    # deleted.  Main deleting it means the module is gone, so it stays gone.
    while IFS= read -r path; do
        if ! git cat-file -e "$tip:$path" 2>/dev/null; then
            git rm -q -- "$path"
        elif git cat-file -e "$ours:$path" 2>/dev/null; then
            fail "$path: a conflict -X theirs did not settle"
        else
            fail "$path: the refresh deleted a page $BRANCH still has"
        fi
    done <<<"$unmerged"
    # A pick stopped on a conflict stays "in progress"; commit() makes the
    # commit, not the pick.
    git cherry-pick --quit
}

summary_pages() { # <base> <tip> — model pages embedding a summary that moved
    if git diff --quiet "$1" "$2" -- "$API/_summaries.json"; then
        return 0
    fi
    # build-api-data.py copies _summaries.json[<factory>] into the page
    # entry of the factory of that name, so a page is stale when it lists
    # a factory whose summary differs between the two commits.
    "$PYTHON_BIN" - "$API" "$1" "$2" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

api, base, tip = sys.argv[1:]


def summaries(rev):
    blob = f"{rev}:{api}/_summaries.json"
    if subprocess.run(["git", "cat-file", "-e", blob], capture_output=True).returncode:
        return {}
    shown = subprocess.run(["git", "show", blob], capture_output=True, text=True, check=True)
    return json.loads(shown.stdout)


def factories(node):
    if isinstance(node, dict):
        if "signature" in node and isinstance(node.get("name"), str):
            yield node["name"]
        for value in node.values():
            yield from factories(value)
    elif isinstance(node, list):
        for value in node:
            yield from factories(value)


old, new = summaries(base), summaries(tip)
moved = {name for name in old.keys() | new.keys() if old.get(name) != new.get(name)}
for page in sorted(Path(api).glob("lucid.models*.json")):
    if moved & set(factories(json.loads(page.read_text()))):
        print(page.stem)
PY
}

stale_slugs() { # <base> <tip> <ours> — the pages to rebuild, one per line
    local base="$1" tip="$2" ours="$3" report status slugs slug
    # The refresh's baseline describes the source at <base>; checked
    # against the source at <tip>, it names exactly the slugs that moved.
    git show "$ours:$API/.cache/drift-hashes.json" >"$API/.cache/drift-hashes.json" ||
        fail "the refresh commit $ours carries no drift baseline"
    # A condition, so the exit 1 that means "stale" trips no trap.
    if report="$(cd web && "$PYTHON_BIN" scripts/build-api-data-cached.py --check)"; then
        status=0
    else
        status=$?
    fi
    slugs="$(printf '%s\n' "$report" | sed -n 's/^    - \([^ ]*\)  (.*)$/\1/p')"
    if [ "$status" -ne 0 ] && [ -z "$slugs" ]; then
        printf '%s\n' "$report" >&2
        fail "the drift check failed without naming a slug, so what $BRANCH made stale is unknown"
    fi
    if [ -n "$slugs" ]; then
        printf '%s\n' "$slugs"
    fi
    if ! git diff --quiet "$base" "$tip" -- lucid; then
        for slug in $SYNTH; do
            if [ -f "$API/$slug.json" ]; then
                echo "$slug"
            fi
        done
    fi
    summary_pages "$base" "$tip" ||
        fail "could not tell which model pages embed a summary that moved"
}

finish
if ! commit; then
    echo "the regenerated api-data is what is committed already — nothing to push"
    output "pushed=false"
    exit 0
fi

attempt=1
until git push -q "$REMOTE" "HEAD:refs/heads/$BRANCH"; do
    git fetch -q "$REMOTE" "$BRANCH"
    tip="$(git rev-parse FETCH_HEAD)"
    ours="$(git rev-parse HEAD)"
    base="$(git rev-parse HEAD^)"
    if [ "$tip" = "$base" ]; then
        fail "the push was refused but $BRANCH has not moved — not a race; see the error above"
    fi
    if ! git merge-base --is-ancestor "$base" "$tip"; then
        fail "$BRANCH no longer contains $base, the commit the refresh was built from"
    fi
    if ! git diff --quiet "$base" "$tip" -- "$GENERATOR"; then
        fail "$GENERATOR changed on $BRANCH during the rebuild ($(git log --format=%h "$base..$tip" -- "$GENERATOR" | tr '\n' ' ')) — every page is stale; run the workflow again" \
            "every page (the generator moved)"
    fi
    carry "$ours" "$tip"
    stale="$(stale_slugs "$base" "$tip" "$ours" | sort -u)"
    # $stale is word-split on purpose below — one argument per slug, none
    # when there are none.
    # shellcheck disable=SC2086
    if [ "$attempt" -ge "$ATTEMPTS" ]; then
        fail "the push was refused $attempt times; $BRANCH kept moving" $stale
    fi
    # shellcheck disable=SC2086
    echo "==> $BRANCH moved to $(git rev-parse --short "$tip"); rebuilding $(count $stale) page(s) there"
    if [ -n "$stale" ]; then
        # shellcheck disable=SC2086
        (cd web && "$PYTHON_BIN" scripts/build-api-data.py --slugs $stale)
    fi
    finish
    if ! commit; then
        echo "$BRANCH already carries this refresh — nothing to push"
        output "pushed=false"
        exit 0
    fi
    attempt=$((attempt + 1))
done

echo "==> pushed $(git rev-parse --short HEAD) to $REMOTE/$BRANCH"
output "pushed=true"
