#!/usr/bin/env python3
"""Regression tests for build-api-data-cached.py's per-slug fingerprints.

LCD-628: ``_per_module_fingerprints`` / ``_content_fingerprints`` hashed only
each slug's ``*.py`` sources (``d.rglob("*.py")``), so a commit that touches
only ``public/api-data/_summaries.json`` -- which model pages embed via
``build-api-data.py``'s ``_load_model_summaries`` / ``_serialise_function`` --
left the embedding slug's fingerprint, and the committed drift baseline,
unchanged.  ``--check`` (the ``docs.yml`` build-job gate) stayed green over a
page that was actually stale.

Standalone by design, like ``test_rst_converter.py``: runs with no pytest on
the docs CI runner (``python3 web/scripts/test_api_data_cached.py``) and is
also pytest-collectable for local runs.
"""

import contextlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

_HERE = Path(__file__).resolve().parent


def _load_cached() -> Any:
    """Import build-api-data-cached.py (hyphenated filename) fresh."""
    path = _HERE / "build-api-data-cached.py"
    spec = importlib.util.spec_from_file_location("_build_api_data_cached", path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_M = _load_cached()

_PATCHED = (
    "REPO_ROOT",
    "LUCID_SRC",
    "WEB_ROOT",
    "OUT_DIR",
    "CACHE_DIR",
    "CACHE_FILE",
    "MODULE_CACHE",
    "DRIFT_CACHE",
    "BUILD_SCRIPT",
)


def _write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


@contextlib.contextmanager
def _fixture_tree() -> Iterator[tuple[Path, Path]]:
    """A minimal ``lucid/`` + ``web/public/api-data/`` laid out like the real
    repo: one model family (``gamma``) with a ``register_model``'d factory,
    and one plain family with none -- so a summaries-only change should move
    only the former's fingerprint, never the latter's.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(
            root,
            "lucid/models/gamma/_pretrained.py",
            "from lucid.models._registry import register_model\n\n"
            "@register_model(params=1)\n"
            "def gamma_small():\n"
            "    ...\n",
        )
        _write(root, "lucid/models/gamma/__init__.py", "")
        _write(root, "lucid/plain/__init__.py", "PLAIN = 1\n")
        out = root / "web" / "public" / "api-data"
        _write(out, "lucid.models.gamma.json", "{}")
        _write(out, "lucid.plain.json", "{}")
        _write(out, "_summaries.json", json.dumps({"gamma_small": {"params": 1}}))

        saved = {name: getattr(_M, name) for name in _PATCHED}
        saved_documented_slugs = _M._documented_slugs
        _M.REPO_ROOT = root
        _M.LUCID_SRC = root / "lucid"
        _M.WEB_ROOT = root / "web"
        _M.OUT_DIR = out
        _M.CACHE_DIR = out / ".cache"
        _M.CACHE_FILE = out / ".cache-key"
        _M.MODULE_CACHE = _M.CACHE_DIR / "modules.json"
        _M.DRIFT_CACHE = _M.CACHE_DIR / "drift-hashes.json"
        _M.BUILD_SCRIPT = _HERE / "build-api-data.py"
        # Fixed, not discovered from the real repo's MODULE_MANIFEST -- keeps
        # the fixture deterministic and independent of what lucid/ currently
        # documents.
        _M._documented_slugs = lambda: ["lucid.models.gamma", "lucid.plain"]
        try:
            yield root, out
        finally:
            for name, value in saved.items():
                setattr(_M, name, value)
            _M._documented_slugs = saved_documented_slugs


def test_summaries_only_change_moves_only_the_embedding_slugs_fingerprint() -> None:
    with _fixture_tree() as (_root, out):
        content_before = _M._content_fingerprints()
        mtime_before = _M._per_module_fingerprints()
        (out / "_summaries.json").write_text(json.dumps({"gamma_small": {"params": 2}}))
        content_after = _M._content_fingerprints()
        mtime_after = _M._per_module_fingerprints()

    assert content_before["lucid.models.gamma"] != content_after["lucid.models.gamma"], (
        "a summaries-only change must move the content fingerprint of the "
        "slug whose factory the summary describes"
    )
    assert content_before["lucid.plain"] == content_after["lucid.plain"], (
        "a family with no register_model factory must not move on an "
        "unrelated family's summary change"
    )
    assert mtime_before["lucid.models.gamma"] != mtime_after["lucid.models.gamma"], (
        "the mtime-based speed-cache fingerprint must see the same change "
        "as the content-based drift fingerprint"
    )


def test_check_drift_catches_a_summaries_only_change() -> None:
    with _fixture_tree() as (_root, out):
        assert _M._write_cache() == 0
        assert _M._check_drift() == 0, "freshly baselined -- must read clean"
        (out / "_summaries.json").write_text(json.dumps({"gamma_small": {"params": 9}}))
        assert _M._check_drift() == 1, (
            "a summaries-only change must fail --check (docs.yml's drift "
            "gate), not pass it silently over a stale model page"
        )


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    failed = 0
    for t in _TESTS:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}\n        {exc!r}", file=sys.stderr)
    print(f"\n{len(_TESTS) - failed}/{len(_TESTS)} api-data-cached tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
