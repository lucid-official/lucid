"""Walk the model-zoo registry and emit one JSON file containing the
expandable layer-summary tree for every ``@register_model``-decorated
factory whose entry sets ``summary="auto"``.

The cache lives at ``web/public/api-data/_summaries.json`` (a mapping
of normalised factory name → tree).  The web build pipeline
(``web/scripts/build-api-data.py``) merges these into the per-module
JSON files at the same path the rest of the docs data lives.

Why a separate CLI and not the main build script?
- Instantiating every model burns a few minutes and ~GBs of RAM —
  exactly the kind of pass we want to amortise via a cache and run
  only when factory code actually changes.
- The web build script is JS-developer-facing and shouldn't have to
  import the entire Lucid runtime.

Usage
-----
::

    python -m tools.build_model_summaries                # full sweep
    python -m tools.build_model_summaries --family resnet     # filter by family
    python -m tools.build_model_summaries --factory resnet_50 # one factory
    python -m tools.build_model_summaries --skip-large 5e9    # raise runtime-fallback limit
"""

import argparse
import gc
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import lucid.models  # noqa: F401 — imported for the registry it populates
from lucid.models._registry import _REGISTRY, _RegistryEntry
from lucid.models._summary import compute_model_summary
from lucid.nn._shadow import shadow_alloc

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = REPO_ROOT / "web" / "public" / "api-data" / "_summaries.json"
# Sidecar holding the per-factory fingerprint that gates incremental
# rebuilds.  Kept separate from the tree cache so the main JSON stays a
# clean ``{factory_name: tree}`` map for the web build pipeline.
META_PATH = REPO_ROOT / "web" / "public" / "api-data" / "_summaries.meta.json"

# Files outside the family directory that *also* invalidate every
# family's cache — touching the summary extractor itself or the
# registry-entry layout means the cached trees are stale even if no
# family source changed.
_GLOBAL_FINGERPRINT_FILES = (
    REPO_ROOT / "lucid" / "models" / "_summary.py",
    REPO_ROOT / "lucid" / "models" / "_registry.py",
)


def _factory_family_dir(entry: _RegistryEntry) -> Path | None:
    """Derive the family source directory from the factory function's
    ``__module__`` (e.g. ``"lucid.models.vision.alexnet._pretrained"``
    → ``<repo>/lucid/models/vision/alexnet``).  Trailing private
    submodule (``_pretrained``, ``_v1``, …) is stripped."""
    mod = getattr(entry.factory, "__module__", "")
    if not mod:
        return None
    parts = mod.split(".")
    # Drop the final private submodule so we land on the family dir.
    if parts and parts[-1].startswith("_"):
        parts = parts[:-1]
    p = REPO_ROOT.joinpath(*parts)
    return p if p.is_dir() else None


_FAMILY_FP_CACHE: dict[Path, str] = {}


def _family_fingerprint(fam_dir: Path) -> str:
    """SHA-256 over the *contents* of every ``.py`` in the family
    directory plus the two global files.

    Content rather than ``mtime``: a fresh checkout rewrites every
    mtime, so an mtime fingerprint misses on every factory in CI and the
    cache it gates can never hit there.  That is the whole reason the
    gate ran ``--force`` — the incremental path was dead on a runner and
    the full rebuild was the only honest option.  Hashing bytes makes
    the fingerprint reproducible across checkouts, so a runner reaches
    the same values the committed sidecar holds.

    Reading each file costs a few milliseconds per family against the
    seconds-to-minutes of instantiating the models it gates.  Cached
    per-call so a tree is read once per CLI run.
    """
    if fam_dir in _FAMILY_FP_CACHE:
        return _FAMILY_FP_CACHE[fam_dir]
    h = hashlib.sha256()
    for f in sorted(fam_dir.rglob("*.py")):
        try:
            payload = f.read_bytes()
        except OSError:
            continue
        h.update(f"{f.relative_to(REPO_ROOT)}:".encode())
        h.update(hashlib.sha256(payload).digest())
        h.update(b"\n")
    for extra in _GLOBAL_FINGERPRINT_FILES:
        try:
            payload = extra.read_bytes()
        except OSError:
            continue
        h.update(f"{extra.relative_to(REPO_ROOT)}:".encode())
        h.update(hashlib.sha256(payload).digest())
        h.update(b"\n")
    fp = h.hexdigest()[:16]
    _FAMILY_FP_CACHE[fam_dir] = fp
    return fp


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", help="only factories whose family matches")
    p.add_argument("--factory", help="only this factory")
    p.add_argument(
        "--skip-large",
        type=float,
        default=2e9,
        help="when shadow construction fails and the runtime fallback would "
        "allocate real Storage, skip factories whose declared 'params' "
        "exceeds this value (default 2e9 ≈ avoid materializing >2B-param "
        "models on hosted runners).  Does NOT affect the shadow path, "
        "which never materializes regardless of param count.",
    )
    p.add_argument(
        "--out", default=str(CACHE_PATH), help=f"output JSON path (default: {CACHE_PATH})"
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="ignore the on-disk fingerprint cache and " "re-instantiate every factory",
    )
    args = p.parse_args()

    out_path = Path(args.out)
    meta_path = out_path.with_name("_summaries.meta.json")

    # Existing cache — fall back to empty dicts on first run / corruption.
    prev_tree: dict[str, Any] = {}
    prev_meta: dict[str, str] = {}
    if not args.force:
        try:
            prev_tree = json.loads(out_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass
        try:
            prev_meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            pass

    # When filtering to a subset (--family / --factory), SEED from the existing
    # cache so untouched families survive.  Without this, a per-family run wrote
    # a cache containing ONLY that family, silently dropping every other model's
    # summary on the next ``build-api-data`` regen (the resnet/vit/… "model
    # structure" cards vanished).  A full run (no filter) still starts empty so
    # removed factories drop out.
    filtered = bool(args.family or args.factory)
    out: dict[str, Any] = dict(prev_tree) if (filtered and not args.force) else {}
    meta: dict[str, str] = dict(prev_meta) if (filtered and not args.force) else {}
    ok = 0
    cache_hits = 0
    skipped = 0
    failed = 0
    t0 = time.perf_counter()

    for name, entry in sorted(_REGISTRY.items()):
        if args.factory and name != args.factory:
            continue
        if args.family and entry.family != args.family:
            continue
        if entry.summary is None:
            skipped += 1
            continue
        if isinstance(entry.summary, dict):
            # Pre-built declarative tree — pass through verbatim,
            # fingerprinted by the literal JSON form so changes to the
            # decorator dict invalidate the cache.
            fp = hashlib.sha256(json.dumps(entry.summary, sort_keys=True).encode()).hexdigest()[:16]
            if prev_meta.get(name) == fp and name in prev_tree:
                out[name] = prev_tree[name]
                cache_hits += 1
            else:
                out[name] = entry.summary
                ok += 1
            meta[name] = fp
            continue
        if entry.summary != "auto":
            print(f"  ! {name}: unknown summary={entry.summary!r}; skip")
            skipped += 1
            continue

        # Cache check: skip re-instantiation when (factory module, family
        # source dir, _summary.py, _registry.py) all unchanged.
        fam_dir = _factory_family_dir(entry)
        fp = _family_fingerprint(fam_dir) if fam_dir is not None else ""
        if fp and prev_meta.get(name) == fp and name in prev_tree:
            out[name] = prev_tree[name]
            meta[name] = fp
            cache_hits += 1
            continue

        try:
            t_start = time.perf_counter()
            # Fast path: construct under ``shadow_alloc`` — no Storage
            # allocation, megabytes instead of gigabytes of RAM, ~200x
            # faster.  Works for the vast majority of factories.  A
            # handful of architectures hit op-dispatch edge cases that
            # shadow mode can't intercept (custom indexing, dtype
            # casting on phantom impls); for those we fall back to a
            # full runtime instantiation so coverage stays at 100%.
            tag = "shadow"
            try:
                with shadow_alloc():
                    model = entry.factory(pretrained=False)
                tree = compute_model_summary(model)
            except Exception:
                # Shadow couldn't intercept this architecture; the only way
                # to summarize is a full runtime instantiation that allocates
                # real Storage.  ``--skip-large`` guards *this* path only —
                # shadow never materializes regardless of declared param
                # count (a 2.4B-param model summarizes in ~20 ms / a few MB),
                # so the limit must not pre-empt the shadow attempt.
                if entry.params and entry.params > args.skip_large:
                    print(
                        f"  - {name}: shadow failed, params={entry.params:,} "
                        f"> runtime-fallback limit; skip"
                    )
                    skipped += 1
                    continue
                tag = "runtime"
                model = entry.factory(pretrained=False)
                tree = compute_model_summary(model)
            dt = time.perf_counter() - t_start
            out[name] = tree
            if fp:
                meta[name] = fp
            ok += 1
            print(
                f"  ✓ {name:38s}  {tree['params']:>14,}  "
                f"({_depth(tree)} levels, {dt*1000:.0f} ms, {tag})"
            )
            del model, tree
            gc.collect()
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  ✗ {name:38s}  {type(exc).__name__}: {exc}")
            if "--debug" in sys.argv:
                traceback.print_exc()

    dt_total = time.perf_counter() - t0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print()
    print(f"Wrote {len(out)} summaries to {out_path}")
    print(f"  cache hit : {cache_hits}")
    print(f"  rebuilt   : {ok}")
    print(f"  skipped   : {skipped}")
    print(f"  failed    : {failed}")
    print(f"  elapsed   : {dt_total:.1f}s")
    return 0 if failed == 0 else 1


def _depth(node: dict[str, Any], cur: int = 0) -> int:
    children = node.get("children") or []
    if not children:
        return cur
    return max(_depth(c, cur + 1) for c in children)


if __name__ == "__main__":
    sys.exit(main())
