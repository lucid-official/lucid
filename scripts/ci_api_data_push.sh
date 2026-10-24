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
#      against the refresh's own baseline, and the synthesised pages,
#      which read the whole tree, when anything under lucid/ moved.
#      Building at the tip keeps every permalink on a commit that stays
#      on main.  Post-process, re-baseline, commit, and push again.
#
# The push is tried LUCID_API_DATA_PUSH_ATTEMPTS times (default 3).  It
# fails loudly — an ::error:: line and the step summary, naming the slugs
# that are stale against main — when the attempts run out, when the
# generator itself changed on main (every page is then stale), when the
# merge leaves a conflict it cannot settle, and when a push is refused
# while main has not moved, since that is no race.
#
# Exit 0: pushed, or nothing to push.  Exit 1: not pushed.
#
# Bash 3.2 compatible (macOS /bin/bash runs its test), so no mapfile and no
# arrays that may be empty: the slug list is a newline-separated string,
# word-split where it is passed on.

set -euo pipefail

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

output() {
    if [ -n "${GITHUB_OUTPUT:-}" ]; then
        echo "$1" >>"$GITHUB_OUTPUT"
    fi
}

fail() { # <why> [stale slug ...]
    local why="$1"
    shift
    echo "::error::api-data was not pushed: $why" >&2
    if [ "$#" -gt 0 ]; then
        echo "  stale against $REMOTE/$BRANCH:" >&2
        printf '    - %s\n' "$@" >&2
    fi
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
        {
            echo "### api-data was not pushed"
            echo "$why"
            if [ "$#" -gt 0 ]; then
                echo
                echo "Stale against \`$REMOTE/$BRANCH\`:"
                printf -- '- `%s`\n' "$@"
            fi
        } >>"$GITHUB_STEP_SUMMARY"
    fi
    output "pushed=false"
    exit 1
}

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

commit() {
    git add -A -- "$API"
    if git diff --cached --quiet; then
        return 1
    fi
    git commit -q -m "$MESSAGE"
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
        if git cat-file -e "$tip:$path" 2>/dev/null; then
            fail "$path: the refresh deleted a page $BRANCH still has"
        fi
        git rm -q -- "$path"
    done <<<"$unmerged"
    # A pick stopped on a conflict stays "in progress"; commit() makes the
    # commit, not the pick.
    git cherry-pick --quit
}

stale_slugs() { # <base> <tip> <ours> — the pages to rebuild, one per line
    local base="$1" tip="$2" ours="$3" report status slugs slug
    # The refresh's baseline describes the source at <base>; checked
    # against the source at <tip>, it names exactly the slugs that moved.
    git show "$ours:$API/.cache/drift-hashes.json" >"$API/.cache/drift-hashes.json" ||
        fail "the refresh commit $ours carries no drift baseline"
    set +e
    report="$(cd web && "$PYTHON_BIN" scripts/build-api-data-cached.py --check)"
    status=$?
    set -e
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
            "every page"
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
