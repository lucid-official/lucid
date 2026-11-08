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
to an explicit per-factory tolerance below instead of being skipped: a label
that drifts further than what was measured fails here.
"""

import pytest

from lucid.models import _registry  # importing lucid.models registers every factory
from lucid.test.audit._zoo_budget import shadow_params

#: Relative gap allowed between the declared figure and the counted one.
DEFAULT_TOLERANCE = 0.05

_DREAMER_LABEL = (
    "Table 3 names the sizes 12M ... 400M and gives no exact count; the rounded "
    "label is declared as stated and the build is Table 3's own dimensions"
)

#: The factories whose declared figure is deliberately further than
#: ``DEFAULT_TOLERANCE`` from the build: ``name -> (tolerance, why)``.  Each is
#: measured, then rounded up to the next percent.
PAPER_FIGURE_TOLERANCE: dict[str, tuple[float, str]] = {
    **{
        name: (0.08, _DREAMER_LABEL)
        for size in ("25m", "100m", "200m", "400m")
        for name in (f"dreamer_v3_{size}", f"dreamer_v3_{size}_world_model")
    },
    "coatnet_6_cls": (
        0.09,
        "Table 5 states 1.47B; Table 12's sizes, which the config follows, count 1.35B "
        "(S4 D=3072 would count 1.46B, but the table says 2048)",
    ),
}


def _declared() -> dict[str, int]:
    return {name: entry.params for name, entry in _registry._REGISTRY.items() if entry.params}


def _gap(name: str, declared: int) -> float:
    return abs(shadow_params(name) - declared) / declared


@pytest.mark.parametrize("name", sorted(_declared()))
def test_declared_params_match_the_build(name: str) -> None:
    declared = _declared()[name]
    tolerance, why = PAPER_FIGURE_TOLERANCE.get(name, (DEFAULT_TOLERANCE, ""))
    built = shadow_params(name)
    gap = abs(built - declared) / declared
    assert gap <= tolerance, (
        f"{name}: declared {declared:,}, built {built:,} ({gap:.1%} apart, tolerance "
        f"{tolerance:.0%}{' - ' + why if why else ''}); fix the config if it does not "
        f"build the paper's variant, or the figure if the paper means a different count"
    )


@pytest.mark.parametrize("name", sorted(PAPER_FIGURE_TOLERANCE))
def test_a_tolerance_entry_is_still_needed(name: str) -> None:
    """An entry the default tolerance already covers is a stale excuse; remove it."""
    assert name in _declared(), f"{name} is no longer a factory with a declared count"
    assert _gap(name, _declared()[name]) > DEFAULT_TOLERANCE, (
        f"{name} is within {DEFAULT_TOLERANCE:.0%} of its declared figure; "
        "drop it from PAPER_FIGURE_TOLERANCE"
    )
