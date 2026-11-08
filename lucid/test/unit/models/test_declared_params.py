"""A factory's declared ``params`` is what it builds, or the table below says why not.

``params`` is the figure the docs site shows as the model's size, so a
factory that builds a different model than the one it names the size of
misleads every reader of the card.  LCD-884 found fourteen of them, and the
causes were of three kinds:

* the config did not build the paper's variant (``efficientdet_d7`` took
  the B7 backbone that belongs to D7x, 74M against the paper's 52M);
* the base factory returns the headless trunk but declared the paper's
  figure, which counts the classifier (``cspresnext_50``, ``coatnet_0``,
  ``efficientformer_l1``);
* the paper gives a rounded label and no exact count, or states two figures
  that do not agree.

The first two are fixed at the factory.  The third cannot be, so it is held
to an explicit per-factory tolerance (``_declared_params``) instead of being skipped: a label
that drifts further than what was measured fails here.
"""

import pytest

from lucid.models import _registry  # importing lucid.models registers every factory
from lucid.test.audit._declared_params import (
    DEFAULT_TOLERANCE,
    PAPER_FIGURE_TOLERANCE,
    ParamsVerdict,
    judge,
)
from lucid.test.audit._zoo_budget import shadow_params


def _declared() -> dict[str, int]:
    return {name: entry.params for name, entry in _registry._REGISTRY.items() if entry.params}


@pytest.mark.parametrize("name", sorted(_declared()))
def test_declared_params_match_the_build(name: str) -> None:
    verdict = judge(name, shadow_params(name), _declared()[name])
    assert verdict.within, (
        f"{name}: {verdict.describe()}; fix the config if it does not build the "
        f"paper's variant, or the figure if the paper means a different count"
    )


@pytest.mark.parametrize("name", sorted(PAPER_FIGURE_TOLERANCE))
def test_a_tolerance_entry_is_still_needed(name: str) -> None:
    """An entry the default tolerance already covers is a stale excuse; remove it."""
    assert name in _declared(), f"{name} is no longer a factory with a declared count"
    by_default = ParamsVerdict(shadow_params(name), _declared()[name], DEFAULT_TOLERANCE, "")
    assert not by_default.within, (
        f"{name} is within {DEFAULT_TOLERANCE:.0%} of its declared figure; "
        "drop it from PAPER_FIGURE_TOLERANCE"
    )
