#!/usr/bin/env python3
"""Completeness guard for the ``lucid.peft`` reference (LCD-725).

The package is documented on one page per method
(``lucid.peft.tuners.<method>``) plus the package page itself, grouped into
sections by ``web/data/api-sections.json``.  Nothing about that is automatic
enough to trust unattended: a tuner added without a section, an export that
slips off every page, or a docs generator that stops walking a level too early
(the original defect: ``/api/lucid.peft.tuners`` built with 0 members and no
tuner page was built at all) would each ship a silently thinner reference.
This test reads the source tree, the sections file and the *committed*
``public/api-data`` JSON and fails on any of them.

What "documented" means, per symbol:

  - an entry in the ``members`` of exactly one ``lucid.peft*`` page — a class
    or a function, which is what the sidebar draws a leaf for — or, for an
    exported constant, a row of ``lucid.peft``'s ``attributes`` table;
  - for a tuner: on that tuner's own page.

Standalone by design, like ``test_rst_converter.py``: stdlib only (the docs CI
runner has no Griffe, no pytest), so it runs as
``python3 web/scripts/test_peft_docs.py`` and exits non-zero on the first
failure.  It is also pytest-collectable.  It checks the generated data, so run
``python3 web/scripts/build-api-data.py --package lucid.peft`` first if the
JSON is older than the source.
"""

import ast
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WEB = _HERE.parent
_REPO = _WEB.parent
_PEFT = _REPO / "lucid" / "peft"
_TUNERS = _PEFT / "tuners"
_API_DATA = _WEB / "public" / "api-data"
_SECTIONS = _WEB / "data" / "api-sections.json"


def _load_builder():
    """Import build-api-data.py (hyphenated filename) and return its module."""
    path = _HERE / "build-api-data.py"
    spec = importlib.util.spec_from_file_location("_build_api_data_peft", path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # guarded by `if __name__ == "__main__"`
    return mod


_M = _load_builder()


# ── source-tree facts ───────────────────────────────────────────────────────

def _dunder_all(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "__all__" for t in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise AssertionError(f"{path} has no __all__")


def _tuner_dirs() -> list[str]:
    return sorted(
        d.name
        for d in _TUNERS.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / "__init__.py").exists()
    )


def _page(slug: str) -> dict:
    path = _API_DATA / f"{slug}.json"
    assert path.is_file(), (
        f"{path.relative_to(_REPO)} is missing — the generator did not build {slug}. "
        "Run `python3 web/scripts/build-api-data.py --package lucid.peft`."
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _sections() -> list[dict]:
    return json.loads(_SECTIONS.read_text(encoding="utf-8"))["lucid.peft"]["sections"]


# ── the generator walks every level (defect class 1) ────────────────────────

def test_manifest_discovery_has_no_depth_limit() -> None:
    """``_discover_manifest`` found packages two levels down only, so
    ``lucid.peft.tuners.lora`` (depth 4) and everything like it was never
    built.  Pins discovery at depth 5, the pruning rules, and the real tree."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for rel in (
            "lucid/a", "lucid/a/b", "lucid/a/b/c", "lucid/a/b/c/d",
            "lucid/a/_private", "lucid/a/_private/pub", "lucid/a/b/no_init",
            "lucid/test", "lucid/test/sub",
        ):
            (root / rel).mkdir(parents=True, exist_ok=True)
        for rel in (
            "lucid/a", "lucid/a/b", "lucid/a/b/c", "lucid/a/b/c/d",
            "lucid/a/_private", "lucid/a/_private/pub", "lucid/test", "lucid/test/sub",
        ):
            (root / rel / "__init__.py").write_text("")
        saved = _M.LUCID_SRC
        _M.LUCID_SRC = root
        try:
            found = _M._discover_manifest()
        finally:
            _M.LUCID_SRC = saved
    for slug in ("lucid.a", "lucid.a.b", "lucid.a.b.c", "lucid.a.b.c.d"):
        assert slug in found, (slug, sorted(found))
    assert not any("_private" in s for s in found), sorted(found)        # private branch pruned
    assert "lucid.a.b.no_init" not in found, sorted(found)                # not a package
    assert not any(s.startswith("lucid.test") for s in found), sorted(found)  # excluded subtree

    real = _M.MODULE_MANIFEST
    assert "lucid.peft.tuners" in real
    for name in _tuner_dirs():
        assert f"lucid.peft.tuners.{name}" in real, name


# ── the sections file lists every tuner, once ───────────────────────────────

def test_sections_list_every_tuner_exactly_once() -> None:
    listed = [p["slug"] for s in _sections() for p in s.get("packages", [])]
    assert len(listed) == len(set(listed)), f"a tuner is listed twice: {sorted(listed)}"
    on_disk = {f"lucid.peft.tuners.{n}" for n in _tuner_dirs()}
    assert set(listed) == on_disk, (
        f"web/data/api-sections.json and lucid/peft/tuners disagree — "
        f"unlisted: {sorted(on_disk - set(listed))}, stale: {sorted(set(listed) - on_disk)}"
    )
    ids = [s["id"] for s in _sections()]
    assert len(ids) == len(set(ids)), ids
    for section in _sections():
        for pkg in section.get("packages", []):
            assert pkg.get("label"), f"{pkg['slug']} has no label"
        assert bool(section.get("packages")) != bool(section.get("members")), (
            f"section {section['id']!r} must hold packages or members, not both or neither"
        )


# ── every export has an api-data entry (and so a sidebar leaf) ──────────────

def _members_by_path() -> dict[str, tuple[str, dict]]:
    """``{member path: (page slug, member)}`` over every lucid.peft* page,
    failing on a symbol documented on two pages."""
    seen: dict[str, tuple[str, dict]] = {}
    for slug in ["lucid.peft", "lucid.peft.tuners", *(f"lucid.peft.tuners.{n}" for n in _tuner_dirs())]:
        for member in _page(slug)["members"]:
            path = member["path"]
            assert path not in seen, (
                f"{path} is documented on both {seen[path][0]} and {slug} — "
                "one canonical page per symbol"
            )
            seen[path] = (slug, member)
    return seen


def test_sidebar_draws_a_leaf_for_every_member() -> None:
    """The sidebar builds a leaf for each ``class`` / ``function`` member and
    nothing else: a member of any other kind would be on the page and in no
    navigation."""
    for _path, (slug, member) in _members_by_path().items():
        assert member["kind"] in ("class", "function"), (slug, member["name"], member["kind"])


def test_every_lucid_peft_export_is_documented() -> None:
    by_path = _members_by_path()
    documented = {m["name"] for _slug, m in by_path.values()}
    constants = {a["name"] for a in _page("lucid.peft")["attributes"]}
    missing = [n for n in _dunder_all(_PEFT / "__init__.py") if n not in documented | constants]
    assert not missing, (
        f"lucid.peft.__all__ names with no api-data entry on any lucid.peft* page: {missing}"
    )


def test_every_tuner_export_is_on_that_tuners_page() -> None:
    for name in _tuner_dirs():
        slug = f"lucid.peft.tuners.{name}"
        page = {m["name"] for m in _page(slug)["members"]}
        exported = _dunder_all(_TUNERS / name / "__init__.py")
        missing = [n for n in exported if n not in page]
        assert not missing, f"{slug}: __all__ names missing from its page: {missing}"


def test_every_tuners_config_and_model_is_on_that_tuners_page() -> None:
    """Some tuners (lora, ia3, loha, lokr) keep their config and model out of
    the subpackage ``__all__`` and export them from ``lucid.peft`` only; the
    generator hands them to the subpackage page that defines them."""
    top = _dunder_all(_PEFT / "__init__.py")
    for name in _tuner_dirs():
        slug = f"lucid.peft.tuners.{name}"
        page = {m["name"] for m in _page(slug)["members"]}
        owned = []
        for src in sorted((_TUNERS / name).glob("*.py")):
            for node in ast.parse(src.read_text(encoding="utf-8")).body:
                if isinstance(node, ast.ClassDef) and node.name in top:
                    owned.append(node.name)
        missing = [n for n in owned if n not in page]
        assert not missing, f"{slug}: exported from lucid.peft, defined here, not on this page: {missing}"
        configs = [n for n in page if n.endswith("Config")]
        assert configs, f"{slug}: no *Config class on the page"


def test_no_public_tuner_symbol_is_left_unexported() -> None:
    """A public-named class or function defined in a tuner's modules that no
    ``__all__`` exports is on no page: export it or make it ``_private``."""
    top = set(_dunder_all(_PEFT / "__init__.py"))
    for name in _tuner_dirs():
        exported = top | set(_dunder_all(_TUNERS / name / "__init__.py"))
        for src in sorted((_TUNERS / name).glob("*.py")):
            if src.name == "__init__.py":
                continue
            for node in ast.parse(src.read_text(encoding="utf-8")).body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and not node.name.startswith("_"):
                    assert node.name in exported, (
                        f"{src.relative_to(_REPO)}: public {node.name} is exported nowhere"
                    )


# ── the generator defects of LCD-725 stay fixed on the real pages ───────────

def _member(slug: str, name: str) -> dict:
    return next(m for m in _page(slug)["members"] if m["name"] == name)


def test_members_inherited_from_private_bases_are_listed() -> None:
    config = {m["name"]: m for m in _member("lucid.peft.tuners.lora", "LoraConfig")["methods"]}
    for name in ("to_dict", "from_dict", "save_pretrained"):
        assert name in config, f"LoraConfig lost {name}: {sorted(config)}"
        assert config[name]["inherited_from"] == "PeftConfig", config[name]
    model = {m["name"] for m in _member("lucid.peft.tuners.lora", "LoraModel")["methods"]}
    for name in ("merge_and_unload", "unload", "merge_adapter", "unmerge_adapter"):
        assert name in model, f"LoraModel lost {name}: {sorted(model)}"


def test_documented_instance_attributes_are_listed() -> None:
    layer = {m["name"]: m for m in _member("lucid.peft.tuners.lora", "LoraLayer")["methods"]}
    for name in ("disable_adapters", "merged", "active_adapter"):
        assert name in layer, f"LoraLayer lost the attribute {name}: {sorted(layer)}"
        assert layer[name]["labels"] == ["attribute"], layer[name]
        assert layer[name]["summary"], f"{name} has no description"


def test_module_overview_code_is_fenced_not_flowed() -> None:
    """The package overview's usage examples are reST literal blocks."""
    text = _page("lucid.peft")["extended"]
    assert "```python\nfrom lucid.peft import LoraConfig" in text, text[:400]
    assert "::" not in text.replace("```", "").replace("std::", ""), "a literal-block marker leaked"


def test_exported_constant_is_in_the_attributes_table() -> None:
    names = {a["name"] for a in _page("lucid.peft")["attributes"]}
    assert "MODEL_TYPE_TO_PEFT_MODEL_MAPPING" in names, sorted(names)


def test_tuner_pages_carry_their_paper_and_overview() -> None:
    for name in _tuner_dirs():
        page = _page(f"lucid.peft.tuners.{name}")
        assert page.get("canonical_name"), name
        assert page.get("theory"), f"{name}: no overview lifted from its config"
        if name != "trainable_tokens":
            assert page.get("citation"), name


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]


def main() -> int:
    failed = 0
    for t in _TESTS:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {t.__name__}\n        {exc}", file=sys.stderr)
    print(f"\n{len(_TESTS) - failed}/{len(_TESTS)} peft docs tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
