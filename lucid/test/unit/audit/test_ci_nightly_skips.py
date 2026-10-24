"""The nightly fails on a skip unless the skip is listed as the design (LCD-530).

``scripts/ci_nightly_suite.py`` used to fail on every skip, so files whose
skips were all by design kept it red every night.  Now a skip passes only
when ``DESIGNED`` lists it, in the number it lists, and the list is held both
ways — an unexpected skip, a changed count and an entry that no longer
happens each fail.
"""

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[4]


def _suite() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "ci_nightly_suite", ROOT / "scripts/ci_nightly_suite.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves the module through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


suite = _suite()

# As the 10-22 nightly printed them.
_REPORT = [
    "SKIPPED [33] lucid/test/parity/nn/test_interpolate_sample_positions_parity.py:35:"
    " Metal has no float64",
    "SKIPPED [1] lucid/test/unit/models/test_models_inplace_independence.py:70:"
    " coatnet: no module built with inplace=True",
    "SKIPPED [1] lucid/test/unit/models/test_models_inplace_independence.py:70:"
    " vit: no module built with inplace=True",
    "SKIPPED [2] lucid/test/a.py: the oracle is not installed",
    "3 passed, 36 skipped in 1.20s",
]

_FILE = "lucid/test/unit/x/test_x.py"
_CPU_ONLY = suite.Designed(r"^Metal has no float64$", None, "CPU only")
_TWO = suite.Designed(r"^too big \S+$", 2, "two variants explode")


def test_the_skip_lines_of_a_report_are_read_with_their_counts() -> None:
    assert suite.skips_of(_REPORT) == [
        (33, "Metal has no float64"),
        (1, "coatnet: no module built with inplace=True"),
        (1, "vit: no module built with inplace=True"),
        (2, "the oracle is not installed"),
    ]


def test_a_designed_skip_passes_and_is_reported() -> None:
    problems, lines = suite.judge_skips(
        _FILE,
        [(33, "Metal has no float64"), (1, "too big 9e4"), (1, "too big 2e8")],
        {_FILE: (_CPU_ONLY, _TWO)},
    )
    assert problems == []
    assert lines == [
        "33 × '^Metal has no float64$' — CPU only",
        "2 × '^too big \\\\S+$' — two variants explode",
    ]


@pytest.mark.parametrize(
    ("skips", "problem"),
    [
        # A skip no entry describes: a test that stopped running.
        (
            [(33, "Metal has no float64"), (2, "too big 1"), (1, "no oracle")],
            "unexpected skip (1): no oracle",
        ),
        # More than designed: something new stopped running.
        (
            [(33, "Metal has no float64"), (3, "too big 1")],
            "3 skips match",
        ),
        # Fewer: the count is stale.
        (
            [(33, "Metal has no float64"), (1, "too big 1")],
            "1 skips match",
        ),
        # None at all: the entry is stale.
        (
            [(2, "too big 1")],
            "designed skip no longer happens",
        ),
    ],
)
def test_the_list_is_held_both_ways(skips: list[tuple[int, str]], problem: str) -> None:
    problems, _ = suite.judge_skips(_FILE, skips, {_FILE: (_CPU_ONLY, _TWO)})
    assert len(problems) == 1 and problems[0].startswith(problem), problems


def test_a_file_with_no_entry_may_not_skip() -> None:
    problems, _ = suite.judge_skips("lucid/test/unit/y.py", [(1, "anything")], {})
    assert problems == ["unexpected skip (1): anything"]


def test_every_entry_names_a_test_file_and_a_pattern() -> None:
    for path, entries in suite.DESIGNED.items():
        assert (ROOT / path).is_file(), path
        assert entries, path
        for entry in entries:
            re.compile(entry.reason)
            assert entry.why, (path, entry.reason)


def test_a_real_run_passes_designed_skips_and_fails_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through pytest itself, so the report format is the one parsed."""
    test = tmp_path / "test_case.py"
    test.write_text(
        "import pytest\n\n"
        "@pytest.mark.parametrize('n', range(3))\n"
        "def test_by_design(n):\n"
        "    pytest.skip('Metal has no float64')\n\n"
        "def test_runs():\n"
        "    assert True\n"
    )
    designed = {suite._relative(test): (_CPU_ONLY,)}
    monkeypatch.setattr(suite, "DESIGNED", designed)
    verdict, detail, _ = suite._run(test)
    assert verdict == "ok", detail
    assert detail[1:] == ["3 × '^Metal has no float64$' — CPU only"]

    with test.open("a") as f:
        f.write("\ndef test_drifted():\n    pytest.skip('the API moved')\n")
    verdict, detail, _ = suite._run(test)
    assert verdict == "skip", detail
    assert detail[1:] == ["unexpected skip (1): the API moved"]
