#!/usr/bin/env python3
"""Regression tests for small, easily-drifted pieces of build-api-data.py.

Standalone by design, like ``test_rst_converter.py``: runs with no pytest on
the docs CI runner (``python3 web/scripts/test_build_api_data_subcategory.py``)
and is also pytest-collectable for local runs.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent


def _load_builder() -> Any:
    """Import build-api-data.py (hyphenated filename) and return its module."""
    path = _HERE / "build-api-data.py"
    spec = importlib.util.spec_from_file_location("_build_api_data_subcat", path)
    assert spec and spec.loader, f"cannot load {path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # guarded by `if __name__ == "__main__":`
    return mod


_M = _load_builder()


class _FakeMember:
    """Minimal stand-in for a Griffe ``Function``/``Attribute``.

    ``_subcategory`` (with ``is_lucid_toplevel=False``, the default) only
    reads ``.filepath`` off the object, so a real Griffe object isn't needed
    to exercise its module-root-relative-path branch.
    """

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath


def test_softmax_distance_shrinkage_group_under_activations() -> None:
    """LCD-614 split ``nn/functional/activations.py``'s softmax, distance and
    shrinkage families out into their own private modules (for file size)
    that still re-export through ``activations.py``.  LCD-628: the docs
    sidebar must keep them grouped under "Activations" -- not splinter into
    three new one-function groups named after the private file stems.
    """
    root = Path("/repo/lucid/nn/functional")
    for stem in ("_softmax", "_distance", "_shrinkage"):
        got = _M._subcategory(_FakeMember(root / f"{stem}.py"), module_root=root)
        assert got == "activations", (stem, got)
    # The re-export point itself is unaliased (its stem already IS the
    # group name) -- confirms the alias doesn't collide with the identity
    # case.
    got = _M._subcategory(_FakeMember(root / "activations.py"), module_root=root)
    assert got == "activations", got


def test_rng_docs_text_does_not_claim_separate_generators() -> None:
    """LCD-628 (an LCD-583 follow-up): the published ``lucid.creation`` page
    claimed CPU and Metal draw from separate generators -- false since
    LCD-583 made the default RNG stream a single engine singleton that both
    devices read, just advancing its counter differently per draw (see
    obsidian/op-contracts/op-rng-default-stream.md).  Pins the corrected
    claim so a future edit can't silently regress it back.
    """
    _summary, extended = _M._SYNTH_SUMMARIES["lucid.creation"]
    assert "separate generators" not in extended, extended
    assert "one default generator" in extended, extended
    assert "get_rng_state" in extended, extended
    assert "set_rng_state" in extended, extended


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
    print(f"\n{len(_TESTS) - failed}/{len(_TESTS)} build-api-data tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
