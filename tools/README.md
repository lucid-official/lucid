# tools/

Developer and CI tooling for Lucid. Nothing here is imported by `lucid/`.
Most scripts carry their own usage in their module docstring; this file
documents the ones whose rules other people's work depends on.

## Quality gate

`tools/quality_gate/` plus the committed `tools/quality_baseline.json`: a
**ratchet** on measured code-health debt. Existing debt never blocks a
commit, and new debt is zero.

### What it counts

Each tool's findings become `(tool, rule, file) -> count`. Production
Python is `lucid/` minus `lucid/test/`; C++ is `lucid/_C/`.

| collector | measures | defect class |
|---|---|---|
| `ruff` | the project's lint config plus `B BLE S110 TRY PGH C90 PLR0911/0912/0915 ERA ARG SIM RET PIE PERF UP RUF100` (`TRY003` off) | D1, D3, D4, D7, style |
| `counters` | `type: ignore[code]` per code (a bare ignore is `type-ignore-bare`), `cast(`, `noqa[CODE]`, `Any` as a name (`any`: annotations, `typing.Any` and the import alike), lines over the file cap (1500 py / 3000 C++) | D1, D3 |
| `lizard` | functions over 100 lines or CCN 15 (py + C++), and per file how far over (`function-length-excess` / `function-ccn-excess`: the sum of `max(0, value − limit)`); `_`-internal Python functions over 7 parameters | D3 |
| `vulture` | dead code at 60 % confidence; scans tests as users of the API, counts only production files | D7 |
| `engine` | functions of `lucid/_C/engine.pyi` (top level and submodules, `nn.bce_loss`) that no production module reaches through an engine-rooted path (`_C_engine.x`, an alias such as `_R = _C_engine`, `from lucid._C.engine import x`, `getattr(_C_engine, "x")`); bindings for tests and tools are listed in `config.ENGINE_TEST_ONLY_BINDINGS`. `python -m tools.quality_gate.collectors.bindings` lists them | D7 |
| `jscpd` | clone pairs of 8+ lines / 50+ tokens (py + C++), via `npx jscpd@5.4.0` | D2 |
| `mypy` | the documented `mypy --strict lucid/` verbatim (`config.MYPY_COMMAND`) plus `warn-unused-ignores`, `ignore-without-code`, `redundant-cast`, `redundant-expr`, `truthy-bool`, `possibly-undefined`; every file it reports counts, including the `tools/` modules `lucid/` imports | D1, D4 |
| `semgrep` | Lucid rules in `tools/semgrep/rules/` (none until DS-4) | D5 |
| `clang-tidy` | `bugprone-*`, `cppcoreguidelines-pro-bounds-*`, `cppcoreguidelines-init-variables`, `cppcoreguidelines-narrowing-conversions`, `readability-function-size`, `performance-*` on `build/compile_commands.json` (opt-in via `--collectors clang-tidy`; DS-3) | D3, D4, style |

Deliberately off: `FBT` and public-API `PLR0913` (signatures follow the
reference framework's), `TRY003` (a library's exceptions carry their own
messages). The thresholds live in `tools/quality_gate/config.py`, and
`config.contract()` records them in the baseline per collector, together
with what else a count depends on: the vulture whitelist (a hash of its
AST), the ignored decorators and the engine allowlist (see "The rules it
enforces"). The tool versions are pinned in the `quality`
dependency group in `pyproject.toml` (`pip install --group quality`),
because a different version counts differently.

### Modes

```
python -m tools.quality_gate --fast        # pre-commit: files changed since HEAD; ruff, counters, lizard + jscpd (~3 s)
python -m tools.quality_gate --full        # CI: whole tree, every collector (~40 s cold on an M1 Pro)
python -m tools.quality_gate --diff main   # SLOP DELTA for a worker report and for land: every --full collector, both trees (~1–2 min)
python -m tools.quality_gate --report      # totals per defect class (deslop SWEEP)
python -m tools.quality_gate --update      # record decreases: lowers the baseline, never raises
python -m tools.quality_gate --rebaseline --reason "<why>"   # orchestrator only: may raise
```

`--collectors ruff,counters` narrows any mode. Exit codes: `0` passes,
`1` means a count rose (or the baseline is out of date), and `2` means the
gate could not measure. A tool that crashes is never read as "no findings".

### The rules it enforces

- **No count may rise.** A new `type: ignore`, a new `except Exception: pass`,
  a 200-line function or a pasted block each fails the gate.
- **A count that fell must be recorded in the same commit.** `check` fails
  on a baseline that is higher than the code, because that slack would let
  slop come back unnoticed. Run `--update` (or `--update --fast` after a
  fast check) and stage `tools/quality_baseline.json`. CI passes
  `--allow-slack`, so slack there is only a warning. The fast path cannot
  see a fall in a cross-file count (mypy, vulture), and two lands can
  combine into one. The orchestrator clears such slack with `--update` on
  main after a land. Nothing can slip back in through it in the meantime:
  `land` measures the counts on main itself, not main's baseline.
- **Moving a file is not new debt.** Renames since the commit that last
  wrote the baseline (git's `-M`) carry their counts to the new path, and
  `--update` rewrites the keys. A move combined with a heavy rewrite falls
  below git's similarity threshold and reads as delete + add. Record a move
  in its own commit when that matters.
- **A count is never netted across files.** Only a git rename carries a
  count to another path. Code that moves into a new file (a split, an
  extracted helper) brings its findings there as new debt, even when the
  source file's count falls in the same diff: the key `(tool, rule, new
  file)` had no baseline row, so it starts at 0. Fix the finding while
  moving it. Netting would need a cross-file ledger, and it would let a
  branch trade a cheap removal for a new finding elsewhere.
- **Counts, not sites.** Moving a finding within a file is free. Swapping
  one finding for another of the same rule in the same file is not
  detected. That is the price of not churning on every edit. Swapping
  across rules is: a `type: ignore` traded for an `Any` lowers one key and
  raises another, and the raise fails (DS-16).
- **The contract may not change silently.** A looser threshold, a new
  whitelist name or a new ignored decorator lowers counts without
  removing any slop. Before DS-18 such an edit reached the baseline as a
  plain `--update`. Now `check`, `--update` and `--diff` refuse a baseline
  whose recorded contract differs for a collector they run, and name the
  entry. Only `--rebaseline --reason` records a new contract, so such an
  edit ships with one (orchestrator only). A rebaseline narrowed with
  `--collectors` records the contract of the collectors it measured and
  keeps the rest as recorded, so it cannot bless another collector's
  change. Comments and layout in the whitelist are free; its names are not.
- **A whitelist entry hides its name everywhere.** vulture keeps one global
  set of used names, so `_.type_as` keeps every dead `type_as` in
  `lucid/` alive, not just the one it was written for. An entry is
  therefore admitted only when vulture needs it (a run without it reports
  a production definition: `test_every_whitelist_entry_is_needed`) and
  something outside Python's view reaches that definition, named on the
  same line. A name Python already uses needs no entry; an unused one
  belongs in a deletion.

### Where it runs

| layer | command | blocks |
|---|---|---|
| pre-commit (`.githooks/pre-commit` §1b, `.pre-commit-config.yaml`) | `--fast` when `lucid/` or the baseline is staged | the commit (`LUCID_SKIP_QUALITY_GATE=1` skips; CI and land still check) |
| CI (`ci.yml` job `quality-gate`) | `--full --allow-slack` | the push |
| `agent_ws.py land` | `--diff main` on the rebased branch; `land --check` diffs against the fork point | the land (exit 6). `--allow-slop "<reason>"` overrides, and every use is appended to `~/Library/Caches/lucid-agents/slop-overrides.log` |

`--diff REF` materialises REF's tree with `git archive` (no worktree, so the
claim gate never sees it). It runs **every collector `--full` runs**, mypy
included, and refuses `--fast`. It measures both trees: the local
collectors on the files that differ, the cross-file ones on everything.
Until DS-GATE-1 (2026-10-11) `--diff` skipped mypy and semgrep to save
time, so a new file's mypy `redundant-expr` passed two lands and failed
only CI's `--full` on main. The ref tree's mypy cache is cold (~30 s on an
M1 Pro); that is the price of land seeing what CI sees. It then also
checks the working tree against the committed baseline. On a land, that
means the baseline the rebase merged.

### Parallel branches and the baseline

Every branch that lowers a count writes the baseline, so the file is
claim-exempt (`agent_ws.GENERATED`). It merges through the
`lucid-quality-baseline` git merge driver (`.gitattributes`). The driver
does a per-key 3-way merge:

- a key that one side left alone takes the other side's value;
- a key that both sides lowered takes the lower one;
- a key deleted on either side stays deleted.

Each side can only lower counts, so the result can never be higher than the
real combined state. A side that **raises** a key aborts the merge, unless
that side also carries a new `--rebaseline` record. `agent_ws.py setup`
registers the driver in repo-local config, and `land` re-checks it before
every rebase. Without the driver, git falls back to a text merge, which
conflicts on neighbouring keys.

### Unlaundering (redundant casts, unused ignores)

When an owner's types are fixed (LCD-260: `Module.__call__` typed from the
subclass's `forward`), the casts and `type: ignore`s written around the
old types become dead, and mypy can prove which ones. Those are removed
mechanically, never by hand:

```
python -m tools.quality_gate.unlaunder --paths lucid/models/vision          # edit
python -m tools.quality_gate.unlaunder --paths lucid/models/vision --check  # count, exit 1 if any
```

Each round runs mypy over `lucid/` with the gate's flags. It then unwraps
`cast(T, e)` to `e` at each `redundant-cast` under `--paths`, and drops
the codes each `unused-ignore` names. An ignore left with no codes goes
together with its trailing reason; a `# noqa` or `# pragma` tail stays.
mypy reports one finding per line and message, so the next cast on a line
only shows up after the first one is gone. Rounds therefore repeat until
nothing more applies, usually in 2–3 rounds. A round takes about 6 s
once mypy's cache (`.mypy_cache/unlaunder`) is warm, and about 25 s cold.
The heavy lock is not needed. The
touched files then get `ruff --select F401 --fix` (for a `cast` import
that became unused) and black at 88. Rules that keep the edit safe:

- every file's result is checked against the original AST with the casts
  replaced, and parentheses are added when the value binds looser than
  its new context;
- a cast whose removed parts hold a comment is left in place and listed;
- a mypy error of any other code that appears during the run fails it
  (exit 2);
- a `--paths` entry outside `lucid/` (or under `lucid/test/`) is
  refused, because mypy would never report there and the result would
  read as "0 left".

Exit codes are `0` (nothing left), `1` (sites left) and `2` (could not
proceed).

**The transitional-section convention.** A change to an owner's types
turns hundreds of existing suppressions into errors at once under the
gate's `--warn-unused-ignores` / `redundant-cast`. Those suppressions
cannot all be removed in the same commit. The owner's commit therefore
adds a `mypy.ini` section for the affected packages that turns those two
checks off, and the comment right above its header starts with
`# transitional (<issue>)`. unlaunder runs mypy on a copy of `mypy.ini`
without every section introduced that way, so it sees exactly what the
section hides. Only the comment block directly above a header counts: a
key line between the marker and the header cancels it. The last sweep
card deletes the section (LCD-260-C4).

**Per sweep card** (one path set, for example `lucid/models/vision`):

1. `python -m tools.quality_gate.unlaunder --paths <prefix>...`
2. `python -m tools.quality_gate.unlaunder --paths <prefix>... --check` → `0 redundant-cast, 0 unused-ignore`.
3. `mypy --strict lucid/` → 0 errors (with the real `mypy.ini`).
4. `python -m tools.quality_gate --update --fast` to record the drop. Stage
   `tools/quality_baseline.json` together with the edits.

The edits change no behaviour, so the card's tests are the touched
package's own suite.

### Adding a collector

1. Put a subclass of `collectors.Collector` in `tools/quality_gate/collectors/`.
   Set `local` if a file's count depends on that file alone, `fast` if it
   is cheap enough for pre-commit, and `default` (a default collector runs
   in both `--full` and `--diff`; there is no CI-only set). Register it in
   `all_collectors()`.
2. Map its rules to a defect class in `config._CLASS_RULES`, and put the
   settings its counts depend on in `config.contract()` under its `name`.
3. Its existing findings raise counts, so the orchestrator runs
   `--rebaseline --reason "<collector> added"` in the same commit.
4. Seed a regression for it in `tools/test_quality_gate.py`.

Tests: `.venv/bin/python3 -m pytest tools/test_quality_gate.py` (scratch git
repositories; the jscpd case skips without Node;
`test_every_whitelist_entry_is_needed` runs vulture over the real `lucid/`).
The dead-code collectors' and the contract's unit tests are in
`lucid/test/unit/test_quality_gate_deadcode.py` and run with the suite.
