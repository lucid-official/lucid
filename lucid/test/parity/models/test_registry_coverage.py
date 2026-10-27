"""Every registered spec must be reached by some test.

``_registry.py`` is the single source of truth for what the model zoo is
compared against, and reading it that way is the trap: registering a
spec does not run it.  Each test module hand-writes a ``_FACTORIES`` set
and picks the registry entries matching it, so a spec whose factory
nobody listed is registered, counted, and never once compared.

Three were, out of 106.  Two of the three had drifted from the
implementation their weights came from — cspdarknet_53 activating at
darknet's leaky slope of 0.1 against a checkpoint trained at 0.01, and
cspresnext_50 splitting two cross-stages with an activation the
checkpoint leaves linear.  Neither shows up at load time, because an
activation carries no parameters, so nothing in the repository could
have caught them and nothing did; they were found by hand.

The list is maintained by hand and will be edited again, which is the
argument for checking it rather than trusting it.
"""

import importlib
import pathlib

from lucid.test.parity.models._registry import SPECS


def _collected_spec_ids() -> tuple[set[str], list[str]]:
    """Ids every parity test module parametrises over, and what would not import.

    Imports each sibling test module and reads the spec lists it builds
    at import time — the same lists ``pytest.mark.parametrize`` is handed
    — so this measures what actually runs rather than what looks
    reachable.

    Some modules are *meant* to be unimportable: the directory conftest
    drops tests whose model family is declared here but not implemented
    yet (``_PENDING_MODULES`` in ``vision/conftest.py``).  Those are
    returned rather than swallowed, and a module that breaks for any
    other reason surfaces as its specs going missing — a loud failure,
    which is the right way for this to be wrong.
    """
    collected: set[str] = set()
    unimportable: list[str] = []
    root = pathlib.Path(__file__).parent
    for path in sorted(root.rglob("test_*.py")):
        if path.name == pathlib.Path(__file__).name:
            continue
        relative = path.relative_to(root.parents[3]).with_suffix("")
        try:
            module = importlib.import_module(str(relative).replace("/", "."))
        except (ImportError, AttributeError) as exc:
            unimportable.append(f"{path.name}: {type(exc).__name__}: {exc}"[:120])
            continue
        for attribute in ("_TIMM", "_SC", "_SPECS"):
            for spec in getattr(module, attribute, None) or ():
                collected.add(spec.id)
    return collected, unimportable


class TestTheRegistryIsFullyReached:
    def test_no_spec_is_registered_and_never_run(self) -> None:
        collected, unimportable = _collected_spec_ids()
        orphans = sorted({spec.id for spec in SPECS} - collected)
        note = f"  (modules that did not import: {'; '.join(unimportable)})" if unimportable else ""
        assert not orphans, (
            "registered but collected by no test — add the factory to the "
            f"matching module's _FACTORIES: {', '.join(orphans)}{note}"
        )

    def test_the_check_can_actually_see_the_specs(self) -> None:
        """A guard on the guard.

        If the import walk above silently found nothing — a moved
        directory, a renamed attribute — the orphan set would come out
        empty and this file would pass while checking nothing, which is
        the failure it exists to prevent.
        """
        collected, _ = _collected_spec_ids()
        assert len(collected) > len(SPECS) // 2, (
            f"only {len(collected)} of {len(SPECS)} specs were seen; the "
            "collection walk is broken, not the registry"
        )
