"""How far a factory's declared ``params`` may sit from what it builds (LCD-884, LCD-933).

``params`` is the paper's figure, and papers round, so an exact comparison
flags every real build of a model whose paper gives a label instead of a
count.  The unit test (``unit/models/test_declared_params.py``) and the zoo
audit probe (``_zoo_probes``) both hold a build to the figure, and they must
hold it to the *same* tolerance: this module is its one owner.  It lives under
``lucid/test`` because it is test infrastructure, never part of the runtime.
"""

from dataclasses import dataclass

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


@dataclass(frozen=True)
class ParamsVerdict:
    """One build held to its declared figure."""

    built: int
    declared: int
    tolerance: float
    #: Why this factory's tolerance is not the default; empty for the default.
    why: str

    @property
    def gap(self) -> float:
        return abs(self.built - self.declared) / self.declared

    @property
    def within(self) -> bool:
        return self.gap <= self.tolerance

    def describe(self) -> str:
        """``built X params, declared Y (Z% apart, tolerance T% ...)``."""
        source = f"its table entry: {self.why}" if self.why else "the default"
        return (
            f"built {self.built:,} params, declared {self.declared:,} "
            f"({self.gap:.1%} apart, tolerance {self.tolerance:.0%} from {source})"
        )


def tolerance_for(name: str) -> tuple[float, str]:
    """``(tolerance, why)`` for factory ``name``; ``why`` is empty for the default."""
    return PAPER_FIGURE_TOLERANCE.get(name, (DEFAULT_TOLERANCE, ""))


def judge(name: str, built: int, declared: int) -> ParamsVerdict:
    """Hold ``built`` to ``declared`` under ``name``'s tolerance."""
    tolerance, why = tolerance_for(name)
    return ParamsVerdict(built=built, declared=declared, tolerance=tolerance, why=why)
