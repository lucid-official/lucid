"""Every perf ceiling is one ``tools/record_perf_ceilings`` can read, and none is a hand-typed one.

``record_perf_ceilings`` finds a ceiling's history by the key the pytest-benchmark JSON gives it:
the test's name, ``test_<case>[<device>]`` -> ``<case>_<device>``.  A test that asserts under a
different key leaves its ceiling unreadable, and the recorder used to skip it without a word
(LCD-947: ``matmul_64``, ``matmul_256`` and ``transformer_block_fwd`` were never re-recordable).
A ceiling typed in from one sample is the LCD-867 defect, so a new one has to come with a
``_provenance`` entry, which only the recorder writes.
"""

import ast
import json
from pathlib import Path

import pytest

_PERF = Path(__file__).resolve().parents[2] / "perf"
_AREAS = ("microbench", "e2e")

# Hand-set before the recorder existed: one MacBook sample, rounded up.  `--audit` over the
# nightly artifacts (LCD-947) found every one at least 2x above the slowest hosted run, so
# they cannot flake; they are loose, not wrong.  Re-record one with the tool and delete it here.
_LEGACY_UNRECORDED = {
    "microbench": {
        "add_2048_cpu",
        "mul_2048_cpu",
        "matmul_64x64_cpu",
        "matmul_256x256_cpu",
        "sum_4096_cpu",
        "softmax_1024_cpu",
        "tanh_1m_cpu",
        "gelu_tanh_1m_cpu",
        "argsort_100k_cpu",
        "unique_100k_cpu",
    },
    "e2e": {"mlp_step_cpu", "block_forward_cpu"},
}


def _template(key: ast.JoinedStr) -> str:
    """``f"mlp_step_{device}"`` -> ``mlp_step_{device}``."""
    parts: list[str] = []
    for part in key.values:
        if isinstance(part, ast.Constant):
            parts.append(str(part.value))
        elif isinstance(part, ast.FormattedValue) and isinstance(part.value, ast.Name):
            parts.append("{" + part.value.id + "}")
        else:
            raise AssertionError(f"unreadable perf key part: {ast.dump(part)}")
    return "".join(parts)


def _asserted_keys(area: str) -> list[tuple[str, str]]:
    """[(test function name, key template)] for each ``assert_no_regression`` call in an area."""
    found: list[tuple[str, str]] = []
    for path in sorted((_PERF / area).glob("test_*.py")):
        for fn in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(fn, ast.FunctionDef) and fn.name.startswith("test_")):
                continue
            for call in ast.walk(fn):
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Name)
                    and call.func.id == "assert_no_regression"
                ):
                    continue
                key = call.args[0]
                assert isinstance(
                    key, ast.JoinedStr
                ), f"{path.name}::{fn.name}: key is not an f-string"
                found.append((fn.name, _template(key)))
    return found


@pytest.mark.parametrize("area", _AREAS)
def test_every_ceiling_key_is_the_test_name_the_recorder_reads(area: str) -> None:
    calls = _asserted_keys(area)
    assert calls, f"no assert_no_regression call found under perf/{area}"
    for name, template in calls:
        case = name.removeprefix("test_")
        assert template == case + "_{device}", (
            f"{name} asserts under {template}; the recorder reads the history of "
            f"{case}_<device> (rename one to match the other)"
        )


@pytest.mark.parametrize("area", _AREAS)
def test_every_ceiling_belongs_to_a_test_that_asserts_it(area: str) -> None:
    cases = {name.removeprefix("test_") for name, _ in _asserted_keys(area)}
    raw = json.loads((_PERF / area / "_thresholds.json").read_text())
    orphans = [key for key in raw if not key.startswith("_") and key.rsplit("_", 1)[0] not in cases]
    assert not orphans, f"ceilings no test in perf/{area} asserts under: {orphans}"


@pytest.mark.parametrize("area", _AREAS)
def test_a_ceiling_without_provenance_is_a_known_legacy_one(area: str) -> None:
    raw = json.loads((_PERF / area / "_thresholds.json").read_text())
    provenance = raw.get("_provenance", {})
    unrecorded = {k for k in raw if not k.startswith("_") and k not in provenance}
    assert unrecorded == _LEGACY_UNRECORDED[area], (
        "record a new ceiling with tools/record_perf_ceilings.py (it writes the provenance); "
        "a legacy one that was re-recorded leaves _LEGACY_UNRECORDED"
    )
    assert set(provenance) <= set(raw), "a provenance entry outlived its ceiling"
