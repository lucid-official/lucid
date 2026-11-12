# `.githooks/` — Lucid local git hooks

Project-tracked git hooks: the gates a commit and a push to `main` go through, and
the `CHANGELOG.md` fold for user-facing commits.

## Enable for this clone

Run **once** per checkout:

```bash
./tools/install_hooks.sh
```

This tells git to look for hook scripts under `.githooks/` (which is git-tracked)
instead of the default `.git/hooks/` (which is local-only and not shareable). The
hooks run Python from `.venv/bin/python3`, the virtualenv `CONTRIBUTING.md` sets up,
or the main checkout's from a worktree. Without one they refuse: the rules need
Python 3.14, and a `python3` found on PATH could be anything.

## What's installed

| Hook          | Stage                 | Behavior |
|---------------|-----------------------|----------|
| `pre-commit`  | before commit drafted | Writes a receipt. Hard-fails on the claim gate, then `python -m tools.rules run --level commit --staged`: everything `tools/rules/registry` claims at commit (source rules on the staged blobs, H4, the C++ layer rules, the quality ratchet, ruff on a staged `web/scripts/*.py`). Then the checks that are not records yet: annotation table, audit baselines, stubs, phantom backend, dtype dispatch, backend layout, docstring coverage, cross-references. Finishes the receipt, then refreshes the Lines-of-Code badge in `README.md` via `cloc`, a soft failure. |
| `commit-msg`  | before commit lands   | Hard-fails on the commit convention. Warns if a user-facing subject has no `CHANGELOG.md` entry. |
| `post-commit` | after commit lands    | Runs even under `--no-verify`. Reads the receipt and records a bypass (below). Then **auto-injects + amends `CHANGELOG.md`** for every user-facing conventional commit, outside worktrees: `tools/changelog.py propose --auto --message-file .git/COMMIT_EDITMSG`, then `git commit --amend --no-edit --no-verify`. A guard variable stops the amend from recursing. |
| `pre-push`    | before refs leave     | `python -m tools.rules push` judges again what is going to `refs/heads/main` on any remote, or to a `v*` tag; other refs pass. It fails closed: no interpreter, a crash and "cannot measure" all refuse, as does a working copy that is not the tip (`lucid/ tools/ scripts/ .githooks/ pyproject.toml` dirty, or the tip is not `HEAD`), because the checks read the working copy. A range that changes `lucid/_C` must also bump `LUCID_ABI_VERSION` when it moves the binary interface, and must have a recorded pass of the C++ suites for its tree; the refusal prints the command (`tools/README.md`, "C++ at push"). Nothing here runs a suite. |

### Why post-commit and not prepare-commit-msg?

Git's commit pipeline finalises the *tree* of the commit at the end of
the `pre-commit` hook — any `git add` performed in
`prepare-commit-msg` or `commit-msg` modifies the index for the *next*
commit, not the current one.  `post-commit` runs after the commit
hash has been computed but before push, so amending in place is safe.

The post-commit auto-amend keeps `[Unreleased]` **in sync with the
conventional-commit history with zero human steps**.  Manual
``tools/changelog.py add`` is reserved for entries that don't map
1-to-1 to a commit subject (e.g. a multi-commit feature grouped under
one bullet).

## What a bypass leaves behind

Three commits on 2026-11-04 skipped pre-commit and reached origin within 21 s, one
with a 37-cell rise, and nothing recorded it. Since then:

- **A permanent `Lucid-Bypass:` trailer** in the commit's message, naming what was
  skipped: `no-verify`, `pre-commit-unfinished` (it exited early), `missing:<script>`,
  or a gate variable.
- **A line in `.git/lucid-bypass.log`**: time, commit, what, subject.
- **A line at push.** `pre-push` prints each trailer in the range. It asks for no
  reason and refuses nothing for the trailer itself: it judges the commits again,
  whatever happened at commit.

A rebase, a cherry-pick and a revert run `post-commit` without `pre-commit` and are not
bypasses: they replay commits that were judged, and push judges the result.

A linked worktree whose branch predates `tools/rules` is refused outright, since main
has the rules and a rebase brings them. A tree with no `tools/rules` at all, in the
main checkout, cannot be judged here: its commits go on and say `missing:rules/__main__.py`.

What this does not catch: `-c core.hooksPath=…` turns `post-commit` off with the rest,
so that commit carries no trailer, and only the push gate sees it. `git push
--no-verify` skips the gate itself; there the record is the commit's trailer, if any.

## Adding entries

```bash
python tools/changelog.py propose                       # interactive — reads HEAD
python tools/changelog.py add added   "Tensor.foo()"    # explicit
python tools/changelog.py add fixed   "edge case in X"
python tools/changelog.py add perf    "Y kernel 30% faster"
python tools/changelog.py add tooling "ruff config tweak"
```

Categories follow [Keep a Changelog](https://keepachangelog.com/) with two
project-specific extensions: `performance` and `tooling`.

Releasing:

```bash
python tools/changelog.py release 3.0.1
```

This promotes the current `[Unreleased]` content into a dated `[3.0.1] -
YYYY-MM-DD` section, resets `[Unreleased]` to a placeholder, and updates the
compare links at the bottom of the file.
