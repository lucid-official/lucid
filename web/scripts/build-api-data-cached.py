"""Cached wrapper around build-api-data.py.

Two-tier caching:

  * **Global key** — whole-tree fingerprint over every ``.py`` under
    ``lucid/`` plus the build script itself.  Hit → skip everything,
    no Griffe walk, no JSON writes.  Used for the common "I'm editing
    docs site code" loop where the Python tree is untouched.

  * **Per-module keys** — for each emitted module slug, hash the
    Python files the slug's JSON depends on.  When the global key
    misses, we still skip rebuilding individual JSONs whose
    fingerprints match.  Cuts a typical "I edited one lucid file" loop
    from ~9 min to ~30 s.

Cache files live under ``public/api-data/.cache/``:
  * ``key`` — global fingerprint (string).
  * ``modules.json`` — ``{slug: {hash, sources: [files], deps: [deps]}}``.

Env vars:
  SKIP_API_BUILD=1    force skip even on cache miss
  FORCE_API_BUILD=1   force full rebuild, ignore both caches
  FORCE_FULL_REBUILD=1  alias of FORCE_API_BUILD

Exit code mirrors the underlying build-api-data.py.
"""

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
WEB_ROOT = HERE.parent
REPO_ROOT = WEB_ROOT.parent
LUCID_SRC = REPO_ROOT / "lucid"
OUT_DIR = WEB_ROOT / "public" / "api-data"
BUILD_SCRIPT = HERE / "build-api-data.py"
CACHE_DIR = OUT_DIR / ".cache"
CACHE_FILE = OUT_DIR / ".cache-key"
MODULE_CACHE = CACHE_DIR / "modules.json"
# Portable per-slug CONTENT hashes for the drift gate (``--check``).  Distinct
# from ``modules.json`` which fingerprints mtimes — mtimes aren't preserved by
# git, so that cache only works within one machine's build.  This one hashes
# file bytes, so it's identical on macOS dev and the Linux CI runner.
DRIFT_CACHE = CACHE_DIR / "drift-hashes.json"


def _compute_key() -> str:
    """Hash (path, mtime_ns) for every .py under lucid/ plus every
    build / post-build script the prebuild chain invokes.

    Tracking the *full* set of scripts (not just ``build-api-data.py``)
    means that editing ``build-usedby.py`` or ``link-citations.py``
    invalidates the cache even when the Lucid sources are untouched —
    otherwise the user would see stale output from a refactored
    post-build pass and have to ``FORCE_API_BUILD=1`` to recover.
    """
    h = hashlib.sha256()
    sources: list[Path] = sorted(LUCID_SRC.rglob("*.py"))
    # Every script the prebuild / build-meta chain runs.  Sorted for
    # determinism — relative order of script edits shouldn't change
    # the cache key, only the set of mtimes.
    for script_name in sorted(HERE.glob("*.py")):
        sources.append(script_name)
    for p in sources:
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        rel = p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p
        h.update(f"{rel}:{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def _outputs_exist() -> bool:
    return OUT_DIR.is_dir() and any(OUT_DIR.glob("*.json"))


def _slug_to_source_dir(slug: str) -> Path | None:
    """Map a slug (``lucid.nn.functional``) to its directory under lucid/.

    Used by the per-module cache to fingerprint only the files that
    contribute to that slug's JSON — touching ``lucid/optim/sgd.py``
    shouldn't invalidate the ``lucid.nn.functional`` slug's cache.
    Returns ``None`` for synthesized slugs (``lucid.ops`` /
    ``lucid.creation`` / ``lucid.ops.composite``) — those depend on
    the whole lucid tree and fall back to the global key.
    """
    if slug in ("lucid.ops", "lucid.creation", "lucid.ops.composite"):
        return None
    # ``lucid.tensor`` reads from lucid/_tensor/tensor.py.
    if slug == "lucid.tensor":
        return LUCID_SRC / "_tensor"
    # Synthetic ``lucid._C.engine`` depends on engine.pyi + bindings —
    # safer to treat as whole-tree.
    if slug == "lucid._C.engine":
        return None
    # Strip leading ``lucid.``, swap dots for slashes.
    rel = slug.removeprefix("lucid.").replace(".", "/")
    candidate = LUCID_SRC / rel
    if candidate.is_dir():
        return candidate
    py = candidate.with_suffix(".py")
    if py.is_file():
        return py
    return None


def _model_factory_names_in_dir(d: Path) -> set[str]:
    """Names of functions decorated with ``@register_model(...)`` under ``d``.

    Mirrors the decorator-call detection ``build-api-data.py``'s
    ``_extract_register_model_params`` uses (same AST shape: a ``Call``
    decorator whose name/attr is ``register_model``), but only needs the
    function name — and reads raw source with :mod:`ast` so this
    stdlib-only script (no Griffe on the docs CI runner) can run it too.
    """
    names: set[str] = set()
    files = list(d.rglob("*.py")) if d.is_dir() else [d] if d.is_file() else []
    for p in files:
        try:
            tree = ast.parse(p.read_text())
        except OSError, SyntaxError, UnicodeDecodeError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                if not isinstance(dec, ast.Call):
                    continue
                func = dec.func
                fname = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr if isinstance(func, ast.Attribute) else None
                )
                if fname == "register_model":
                    names.add(node.name)
    return names


def _load_summaries_json() -> dict[str, Any]:
    """Read ``public/api-data/_summaries.json`` fresh (no in-process cache —
    a cached value would go stale across the ``--write-cache`` / mutate /
    ``--check`` sequence a single test or CI run performs)."""
    cache = OUT_DIR / "_summaries.json"
    if not cache.is_file():
        return {}
    try:
        return json.loads(cache.read_text())
    except OSError, json.JSONDecodeError:
        return {}


def _model_summary_fingerprint(slug: str, d: Path) -> str:
    """The ``_summaries.json`` entries this slug's factories embed, as a
    stable string — ``""`` for slugs with no ``register_model`` factory.

    Model pages embed ``_summaries.json[<factory name>]`` (see
    ``build-api-data.py:_load_model_summaries`` /
    ``_serialise_function``), but neither per-slug fingerprint below walks
    anything but ``*.py`` sources.  A commit that only reruns
    ``tools/build_model_summaries.py`` therefore left the embedding page's
    fingerprint — and the committed drift baseline — unchanged, so
    ``--check`` stayed green over a stale page (LCD-628).  Mixed into both
    fingerprints below so such a change marks exactly the family slugs
    whose factories moved, not every slug.
    """
    names = _model_factory_names_in_dir(d)
    if not names:
        return ""
    summaries = _load_summaries_json()
    entries = {n: summaries[n] for n in sorted(names) if n in summaries}
    if not entries:
        return ""
    return json.dumps(entries, sort_keys=True)


def _hash_paths(paths: list[Path], extra: str = "") -> str:
    """sha256 over (rel path, mtime_ns) pairs — same shape as the global key.

    ``extra`` mixes in state that isn't a file under ``paths`` but should
    still invalidate the fingerprint (the slug's ``_summaries.json``
    entries — see :func:`_model_summary_fingerprint`).
    """
    h = hashlib.sha256()
    for p in sorted(paths):
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        rel = p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p
        h.update(f"{rel}:{st.st_mtime_ns}\n".encode())
    h.update(f"build_script:{BUILD_SCRIPT.stat().st_mtime_ns}\n".encode())
    if extra:
        h.update(b"\0summaries\0")
        h.update(extra.encode())
    return h.hexdigest()


def _documented_slugs() -> list[str]:
    """Every slug that *should* have a page, not just those that already do.

    Deriving the list from the emitted ``*.json`` files alone makes a
    brand-new module invisible: it has no JSON yet, so it never lands in
    ``stale_slugs`` (never gets built) and never lands in the drift
    baseline (``--check`` reports green while a whole page is missing).
    The build script's manifest is the source of truth, so union the two —
    the manifest brings in new modules, the glob keeps synthesized slugs
    that have no source directory of their own.
    """
    emitted: set[str] = set()
    if OUT_DIR.is_dir():
        emitted = {
            jp.stem
            for jp in OUT_DIR.glob("*.json")
            if not jp.name.startswith(("_", "."))
        }

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_api_build", BUILD_SCRIPT)
        if spec is None or spec.loader is None:
            return sorted(emitted)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        emitted |= set(module.MODULE_MANIFEST)
    except Exception:  # noqa: BLE001 — discovery is best-effort
        # A broken build script is reported by the build itself; falling
        # back to the emitted set keeps the cache usable meanwhile.
        pass
    return sorted(emitted)


def _per_module_fingerprints() -> dict[str, str]:
    """Compute a fingerprint per slug, keyed by slug.

    Walks each slug's source directory recursively.  Synth slugs return
    no fingerprint (they always rebuild when the global key misses).
    Used for the post-build cache write — the actual rebuild decision
    happens module-by-module in :func:`_should_skip_slug`.
    """
    result: dict[str, str] = {}
    if not OUT_DIR.is_dir():
        return result
    for slug in _documented_slugs():
        d = _slug_to_source_dir(slug)
        if d is None:
            continue
        files = list(d.rglob("*.py")) if d.is_dir() else [d]
        result[slug] = _hash_paths(files, _model_summary_fingerprint(slug, d))
    return result


def _load_module_cache() -> dict[str, str]:
    if not MODULE_CACHE.is_file():
        return {}
    try:
        return json.loads(MODULE_CACHE.read_text())
    except OSError, json.JSONDecodeError:
        return {}


def _save_module_cache(fingerprints: dict[str, str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODULE_CACHE.write_text(json.dumps(fingerprints, indent=2, sort_keys=True))


def _content_fingerprints() -> dict[str, str]:
    """Per-slug fingerprint over source-file CONTENT (bytes), keyed by slug.

    Portable across machines/checkouts — unlike :func:`_per_module_fingerprints`
    (mtimes) — so the committed baseline computed on a dev box matches what the
    CI runner recomputes.  Drives the drift gate only; the build-speed cache is
    unaffected.  Synth slugs (``lucid.ops`` / ``lucid.creation`` /
    ``lucid.ops.composite`` / ``lucid._C.engine``) have no isolated source dir
    and are skipped.

    Also mixes in the slug's ``_summaries.json`` entries (see
    :func:`_model_summary_fingerprint`) — model pages embed those, and no
    ``*.py`` source hash sees a ``tools/build_model_summaries.py`` rerun.
    """
    result: dict[str, str] = {}
    if not OUT_DIR.is_dir():
        return result
    for slug in _documented_slugs():
        d = _slug_to_source_dir(slug)
        if d is None:
            continue
        files = list(d.rglob("*.py")) if d.is_dir() else [d]

        def _rel(p: Path) -> str:
            return (
                str(p.relative_to(LUCID_SRC)).replace("\\", "/")
                if p.is_relative_to(LUCID_SRC)
                else p.name
            )

        # Sort by the lucid-relative path string so iteration order — and thus
        # the combined hash — is identical on macOS dev and the Linux runner
        # (absolute prefixes differ, relative paths don't).
        h = hashlib.sha256()
        for p in sorted(files, key=_rel):
            try:
                data = p.read_bytes()
            except OSError:
                continue
            h.update(_rel(p).encode())  # catches renames too
            h.update(b"\0")
            h.update(data)
            h.update(b"\0")
        extra = _model_summary_fingerprint(slug, d)
        if extra:
            h.update(b"\0summaries\0")
            h.update(extra.encode())
        result[slug] = h.hexdigest()
    return result


def _check_drift() -> int:
    """Report (without rebuilding) any slug whose source CONTENT differs from the
    committed drift baseline (``.cache/drift-hashes.json``) — i.e. the committed
    api-data is stale w.r.t. its Lucid source.

    Pure stdlib (content hashes), so it runs on the docs CI runner where Griffe
    / Lucid / pip are intentionally absent.  This is the "source → api-data" half
    of the drift gate; the "post-processor" half (cross-links / used-by /
    link-citations / meta) is caught by a plain ``git diff`` after the prebuild.
    """
    cached: dict[str, str] = {}
    if DRIFT_CACHE.is_file():
        try:
            cached = json.loads(DRIFT_CACHE.read_text())
        except OSError, json.JSONDecodeError:
            cached = {}
    current = _content_fingerprints()
    if not current:
        print("[api-data] --check: no emitted *.json found; nothing to verify")
        return 0
    if not cached:
        print(
            "[api-data] ✗ no drift baseline — run "
            "`python3 scripts/build-api-data-cached.py --write-cache` and commit "
            f"{DRIFT_CACHE.relative_to(REPO_ROOT)}"
        )
        return 1
    stale = sorted(s for s, h in current.items() if cached.get(s) != h)
    if not stale:
        print(
            f"[api-data] ✓ no source drift — {len(current)} module slugs "
            "match the committed baseline"
        )
        return 0
    print("[api-data] ✗ STALE — committed api-data is out of date with source:")
    for s in stale:
        why = (
            "source changed since last regen"
            if s in cached
            else "new slug, never built"
        )
        print(f"    - {s}  ({why})")
    print("\n  Regenerate + commit:")
    print("    cd web && FORCE_API_BUILD=1 python3 scripts/build-api-data-cached.py")
    print("    npm run rebaseline:api-cache   # refresh the drift baseline")
    print("    git add web/public/api-data")
    return 1


def _write_cache() -> int:
    """(Re)write ``.cache/drift-hashes.json`` to the CURRENT source content,
    asserting the committed api-data already reflects it (e.g. after a direct
    ``build-api-data.py`` regen).  Re-baselines the drift gate.
    """
    fp = _content_fingerprints()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DRIFT_CACHE.write_text(json.dumps(fp, indent=2, sort_keys=True))
    print(f"[api-data] re-baselined {len(fp)} content fingerprints -> {DRIFT_CACHE}")
    return 0


def main() -> int:
    if "--check" in sys.argv:
        return _check_drift()
    if "--write-cache" in sys.argv:
        return _write_cache()

    if os.environ.get("SKIP_API_BUILD") == "1":
        print("[api-data] SKIP_API_BUILD=1 — skipping build")
        return 0

    force = (
        os.environ.get("FORCE_API_BUILD") == "1"
        or os.environ.get("FORCE_FULL_REBUILD") == "1"
    )
    current_key = _compute_key()

    if not force and _outputs_exist() and CACHE_FILE.is_file():
        try:
            cached = CACHE_FILE.read_text().strip()
        except OSError:
            cached = ""
        if cached == current_key:
            print("[api-data] cache hit — skipping rebuild")
            return 0

    if force:
        print("[api-data] FORCE rebuild")
    else:
        print("[api-data] global cache miss — checking per-module cache…")

    # ── Per-module cache — only invoke build-api-data.py for slugs
    # whose source-dir fingerprint has changed.  The build script
    # already supports ``--module <slug>`` so we drive it once per
    # stale slug.  Synth slugs and lucid._C.engine fall back to a
    # whole-tree rebuild when *anything* changes.
    prev_fingerprints = _load_module_cache() if not force else {}
    current_per_module: dict[str, str] = {}
    if OUT_DIR.is_dir():
        # Manifest-driven, not glob-driven: a module that has never been
        # built has no JSON to glob, and enumerating the emitted files
        # would silently exclude it from ``stale_slugs`` forever.
        for slug in _documented_slugs():
            d = _slug_to_source_dir(slug)
            if d is None:
                continue
            # ``_slug_to_source_dir`` returns the canonical source dir
            # but doesn't itself check existence — the recursive
            # ``rglob`` would silently emit zero files when the dir
            # was renamed / deleted, which then collides with the
            # previous cache entry's hash and forces unnecessary
            # rebuilds.  Skip the slug here so the orphan-cleanup
            # branch below handles it instead.
            if d.is_dir():
                files = list(d.rglob("*.py"))
            elif d.is_file():
                files = [d]
            else:
                continue
            if not files:
                continue
            current_per_module[slug] = _hash_paths(
                files, _model_summary_fingerprint(slug, d)
            )

    # ── Orphan cleanup — slugs that used to be in the cache but no
    # longer have a source directory.  Could happen on file deletes /
    # renames / branch switches.  We delete the stale JSON + drop the
    # slug from the cache so subsequent builds don't try to rebuild
    # ghosts.  Synth-slug names are exempt because their source dirs
    # don't map 1:1 to file system locations.
    SYNTH_NAMES = {
        "lucid.ops",
        "lucid.creation",
        "lucid.ops.composite",
        "lucid._C.engine",
    }
    orphans: list[str] = []
    for slug in list(prev_fingerprints.keys()):
        if slug in SYNTH_NAMES:
            continue
        if _slug_to_source_dir(slug) is None:
            orphans.append(slug)
            continue
        d = _slug_to_source_dir(slug)
        if d is not None and not (d.is_dir() or d.is_file()):
            orphans.append(slug)
    for slug in orphans:
        jp = OUT_DIR / f"{slug}.json"
        if jp.exists():
            try:
                jp.unlink()
            except OSError:
                pass
        prev_fingerprints.pop(slug, None)
    if orphans:
        print(
            f"[api-data] pruned {len(orphans)} orphan slug(s): "
            + ", ".join(orphans[:3])
            + (f" + {len(orphans) - 3} more" if len(orphans) > 3 else ""),
        )

    # Decide rebuild strategy.  If we have NO prior cache or the per-
    # module cache is empty, fall back to a full rebuild — it's faster
    # than scheduling 76 ``--module`` invocations on a cold start.
    stale_slugs: list[str] = []
    needs_full = force or not prev_fingerprints
    if not needs_full:
        for slug, fp in current_per_module.items():
            if prev_fingerprints.get(slug) != fp:
                stale_slugs.append(slug)
        # Synth slugs depend on global tree — rebuild them whenever
        # ANY manifest slug is stale (cheap, the source data is
        # already loaded into memory at that point anyway).
        if stale_slugs:
            for synth in ("lucid.ops", "lucid.creation", "lucid.ops.composite"):
                if synth not in stale_slugs and (OUT_DIR / f"{synth}.json").exists():
                    stale_slugs.append(synth)

    if needs_full:
        print("[api-data] running full rebuild")
        result = subprocess.run([sys.executable, str(BUILD_SCRIPT)])
    elif not stale_slugs:
        # Per-module cache is fully warm — global mtime changed (maybe
        # the build script itself) but no module source changed.  Just
        # refresh the global key and exit.
        print("[api-data] per-module cache fully warm — skipping rebuild")
        CACHE_FILE.write_text(current_key)
        return 0
    else:
        print(
            f"[api-data] rebuilding {len(stale_slugs)} stale slug(s): "
            f"{', '.join(stale_slugs[:5])}"
            + (f" + {len(stale_slugs) - 5} more" if len(stale_slugs) > 5 else "")
        )
        result = subprocess.run(
            [
                sys.executable,
                str(BUILD_SCRIPT),
                "--slugs",
                *stale_slugs,
            ]
        )

    if result.returncode == 0:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(_compute_key())
        # Refresh per-module fingerprints from what's actually on disk
        # — captures any slugs the build emitted, regardless of which
        # branch we took above.
        _save_module_cache(_per_module_fingerprints())
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
