# tools/

Developer and CI tooling for Lucid. Nothing here is imported by `lucid/`.
Most scripts carry their own usage in their module docstring; this file
documents the ones whose rules other people's work depends on.

## Rule registry

`tools/rules/` holds rules as records in `registry/*.toml`, one file per domain,
and answers what prose could not: is this rule enforced, and can its check fail.
TOML because the hook and push paths must not fail on a missing package.

    python -m tools.rules list --verdict         # every rule, how far it is enforced
    python -m tools.rules show H1                # one rule, with why
    python -m tools.rules run --level push       # the checks one level runs, whole tree
    python -m tools.rules run --level push --range A..B    # what those commits changed
    python -m tools.rules run --level commit --staged      # what the index changes
    python -m tools.rules push --range A..B      # what pre-push does with those commits
    python -m tools.rules audit                  # is each claim true, can each check fail
    python -m tools.rules audit --check numpy-h4 --rule H4 # only these controls

Every check exits **0 passes, 1 violation, 2 cannot measure**; the run returns
the worst, and 2 is never read as a pass. A crash, a killed process and a test
that was skipped, deselected or not collected are all 2.

With `--staged`, a check with `--staged` in its `run` reads the blobs itself. Any
other check reads the working copy, so it exits 2, naming the path, when that
copy is not what is staged. A check with `--staged` has nothing to say without it.

With `--range A..B` a check reads the working copy too, so the run refuses unless
that copy is `B`: exit 2 when `B` is not `HEAD`, or when `lucid/ tools/ scripts/
.githooks/ pyproject.toml` hold anything uncommitted. `{since}` in a `run` is `A`
there, `HEAD` under `--staged`, and the empty tree otherwise.

`.githooks/pre-push` runs `python -m tools.rules push`, which takes those ranges
from the refs git writes to it. Only `refs/heads/main` and `v*` tags are judged. A new
tag is judged only for what the remote's main does not hold. Every exit but 0 refuses
the push. See `.githooks/README.md` for what a bypass leaves behind.

A `[[rule]]` holds exactly one of `enforce` with a `control`, `advisory = "<why
no check can exist>"`, or `pending = "LCD-n"`; anything else fails load. A rule
with `gap` is enforced only partly.

| `audit` section | exit 1 | exit 2 |
|---|---|---|
| controls | a check stayed green on its seed, went red for another reason, or left the seed outside its trigger; a test node failed | a control could not run; a check is red on the near miss; a test node did not run |
| wiring | a hook level's real commit or push did not run the tool, or went through with it failing; a CI level does not name the tool; a `pending` level is wired now; a trigger or scope glob matches no file; a check's trigger leaves out a file in the scope of a rule that claims it | |
| allowlists | a table's size differs from its `ceiling` | a table could not be read |
| budgets | a ceiling is passed; the `pending` count fell and its ceiling did not | a budget's paths match nothing |

**What a green audit does not prove.**

- A hook level is proven by the real hook in a scratch repository, whose python is a
  stub, in a main checkout and in a linked worktree: git must run the tool, and must
  stop when it fails. What the stub cannot show is whether the runner runs each check
  it claims, which the controls and `tools/test_rules.py` answer.
- A CI level is proven only by its name in `scripts/ci_full.sh` or a workflow, outside
  a comment (or by `tools.rules run --level ci`). A job with `if: false` still passes;
  LCD-841 owes the rest. `.pre-commit-config.yaml` counts for nothing: it never runs.
- A test node is proven to run and pass, not to contain an assertion. An empty body
  passes.
- `pending = "LCD-n"` is checked for shape; the tracker is out of reach.
- A trigger is held to its rule's scope over what `git ls-files -co --exclude-standard`
  lists, so every checkout of a tree gets one answer. A copy with no git is walked as it
  is, ignored files and all.

The whole audit took 40 to 68 s on a loaded Studio and 92 s on an M1 Pro (2026-11-06/07),
most of it the C++ records' controls: the real-git push case alone is about 15 s.

**Adding a rule.** Write the record. Each check it claims needs a control: `seed`
(the tool and `tree` in a scratch directory; the `clean` near miss at `path` must
pass, `text` there must fail with `expect` in the output) or `pytest` (nodes that
assert the failure). A claimed `test` level's own nodes run with the controls. A
seed that may not be written down in the file, because the file is held to the
rule, is a list of parts. Seed the mode the level really calls, and below the top
of the tree. A new kind of check or control is a subclass.

Tests: `.venv/bin/python3 -m pytest tools/test_rules.py`, exposed to CI by
`lucid/test/unit/test_rules_tooling.py`. Its mutant cases fail when a broken line
goes unseen; add one when you add a way for a claim to be false.

## C++ at push

A push whose range changes `lucid/_C` meets two more records in `registry/engine.toml`.
Neither runs at commit: a worker may not touch `version.h` or run the suites. A passing
check still shows its `note: ` lines at push, so an excuse it took is printed.

**`E-ABI-BUMP`**, `tools/check_abi_bump.py`. It reads the two ends of the range, not the
diff lines, so a reflow, a comment, a renamed parameter or a type moved between headers
is no move. A move is a data member or virtual of any class or struct in a header (order
and a member's `alignas` included; a specifier on the struct's own head is not read), any
member of a `LUCID_API` type, a free `LUCID_API`
declaration, a `using` alias that comes to mean another type, a name a binding gives
Python, or a name in `engine.pyi`. The range passes when `LUCID_ABI_VERSION` rose between
its ends, or when every commit that moved it is excused. A commit's reason speaks for what
that commit moved, as a trailer on it: `ABI: none (<reason>)`. It speaks for another
commit's move only by naming it, in the reason: that commit's sha (seven digits or more),
or every symbol it moved (the names in the brackets after `moved by` in the failure; a
name of fewer than three letters needs the sha). An unrelated `ABI: none (Python only)`
excuses nothing but itself. `ABI: bump needed; <what>` is excused by nothing: the range
fails until `LUCID_ABI_VERSION` moves. A bare `ABI: none` is a habit, not a reason. What
it cannot see is the record's `gap`. A move that a later commit undid holds nothing.

    python tools/check_abi_bump.py --range A..B            # as pre-push judges it
    python tools/check_abi_bump.py --range A..B --each     # every commit on its own

Backtest, 2026-11-06, `--each` over main's last 300 commits up to `4db11a36a`: 19 move
the interface; 2 bump in the same commit, and 17 do not. Fourteen of those went out in a
range that held their bump (LCD-853's nine with ABI 44, LCD-837's Tracer, LCD-834's grad
masks, LCD-717's new node), or undid themselves before it (LCD-850). Judged as main's
real pushes were (origin/main's reflog, 104 pushes up to `a9ffe63aa`), it refuses 3:
`578247379` and `48f57990e`, which shipped at ABI 40 with no bump, and `78d57bf38`, which
added a member to a struct private to `CpuBackend`. That last one is what `version.h` asks
a bump for, and the one a reader may call needless. Every bump in the history has a move
before it.

**`E-VERIFY-CPP`**, the attestation. `lucid/test/unit/compile` and `unit/cost` take
minutes, so the run and the check are apart:

    .venv/bin/python3 -m tools.rules attest            # run them here, record the result
    .venv/bin/python3 -m tools.rules attest --verify   # what pre-push asks; instant
    .venv/bin/python3 -m tools.rules attest --plan     # the chunks (below), run nothing

`attest` refuses (exit 2) when `lucid/`, `pyproject.toml`, `setup.py` or
`tools/fetch_mlx.py` holds anything uncommitted; when the engine has no build stamp, or a
stamp that is not this tree's digest (below); when the checkout's python imports lucid or its engine from anywhere else; when a
suite ran no test, every one skipped, or skipped more than 5 % (a sound run skips 0.17 %;
the memory governor's summary in the output says why, so run it again when memory is free);
when a chunk's process died (below); and when HEAD or what the suites read moved during the
run. It runs each suite as several processes, without `PYTEST_ADDOPTS`, `PYTEST_PLUGINS` or
`PYTEST_DISABLE_PLUGIN_AUTOLOAD` (it says when it drops one), and
writes `<git common dir>/lucid-attest/<key>.json`: counts, commit, host, time. A failing
run is recorded too, and exits 1. The key digests what the engine is built from:
`HEAD:lucid/_C`, `setup.py`, `tools/fetch_mlx.py` and `[tool.lucid.mlx]`. So one record
covers any number of lands that leave those alone, a Python-only push needs none, and
every worktree reads the same ledger. A record is a local file: write it where the commit
is checked out and the engine is built, in a checkout whose `.git` the pushing one shares
or syncs with, or copy the one file. In the Studio's main checkout run `git reset -q`
first: `.git/index` is not synced, so a stale index reads as uncommitted edits (the
working tree is left alone).

**Chunks.** One pytest process holds what its tests leave behind, and `unit/compile` (14 280
tests) in one was SIGKILLed on the 16 GB MacBook at about 80 %: no traceback, no junit, and
`attest` read the missing report as "ran no test". Now each suite is dealt out, in sorted
order of the test files `git ls-files` lists at HEAD, over the fewest processes that hold at
most `--chunk-files N` files each (default 6: 31 chunks for compile, 1 for cost). Dealt out
in turn, not cut in runs, because files named alike cost alike (the `rank_canonical_*`
family) and neighbours should land apart. The same tree is always cut the same way, and
`attest --plan` lists who is where. The chunks' junit totals are the suite's: the record,
its key and its format are what they were, a record from the old runner stays valid, and
every refusal above is judged on the merged totals (a chunk that skipped every test, or found
none, fails nothing when the suite ran). A process that **dies** (a signal, an exit that left
no report, a report that does not parse) is neither a failing test nor a zero: the run stops,
no record is written (a transient kill must not overwrite a green record), and it exits 2 with

    chunk 7/31 killed by signal 9 (SIGKILL), likely memory; it was running
    lucid/test/unit/compile/test_x.py::test_y[mid] (6 files: ...): rerun when memory is free,
    or split it with `attest --chunk-files 3`

where the test is the one the chunk's pytest had started, left in a file at each test start
(the killed process can say nothing else). That is the test the footprint was on, not
proof it is the one that grew it: a leak shows on the test that ran last. Measured
2026-11-13 on the MacBook, peak memory footprint per chunk (`/usr/bin/time -l`): of the 31
compile chunks (18 minutes, 14 238 tests, 0 failed, 46 skipped) all but four stay under 2 GB.
Those hold the files that peak that high alone: LCD-995's `test_rank_canonical_widening` 38.9
GB (the swap grew to 37 GB), `test_norm_pool_rank_canonical` 5.9, `test_rank_canonical_movement`
4.9 and `_attention` 2.7. Memory is per file, not per test count, so a smaller `--chunk-files`
helps only by keeping such files apart; one file that needs more than the machine has needs
the file changed.

**The engine build stamp** (LCD-881). "This engine was built from this tree" cannot be
read off mtimes: `unison` keeps times, and a `cp -c` of an `.so` is "now". An in-place
build leaves `build/.engine_stamp`, one digest of `lucid/_C` (sources and CMake files),
`setup.py`, `tools/fetch_mlx.py` and `[tool.lucid.mlx]`, computed by `tools/engine_stamp.py`
alone. Every build path writes it through the CMake target `lucid_engine_stamp`, which
runs after each `cmake --build`: `uv pip install -e .` (main), `python3 tools/agent_ws.py
build` (a worktree), and a plain `cmake --build build/temp.macosx-*/lucid__C_engine/`.
`agent_ws._engine_is_stale` reads the stamp first and falls to mtimes only for a checkout
with none; `attest` takes no fallback. `python3 tools/engine_stamp.py check` says whether a
checkout's stamp matches. The stamp is local (`build/` is not synced): a checkout that
received its engine by sync or copy has none until it builds.

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
python -m tools.quality_gate --fast --since REF   # pre-push: the same, from REF; also sees a baseline raised since REF
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
| pre-commit (`tools.rules run --level commit --staged`) | `--fast --since HEAD` when `lucid/`, the gate or the baseline is staged | the commit (`--no-verify` skips it, and leaves a `Lucid-Bypass` trailer: see `.githooks/README.md`) |
| pre-push (`tools.rules push`) | `--fast --since <remote> --allow-slack` over the range | the push to `main` or a `v*` tag, whatever happened at commit |
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
