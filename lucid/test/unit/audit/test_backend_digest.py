"""Unit tests for ``lucid/test/audit/tools/backend_digest.py``."""

import copy
import json
from pathlib import Path

from lucid.test.audit.tools import backend_digest


def test_subset_digest_is_reproducible_and_detects_bit_change(tmp_path: Path) -> None:
    symbols = {"lucid.add", "lucid.sub", "lucid.mul", "F.relu", "lucid.sum", "F.layer_norm"}
    first = backend_digest.generate_digest(symbols_filter=symbols, repeat=2, devices=("cpu",))
    second = backend_digest.generate_digest(
        symbols_filter=symbols, repeat=1, plan=first["plan"], devices=("cpu",)
    )
    diffs, _ = backend_digest.compare_digests(first, second)
    assert diffs == []
    assert len(first["cells"]) >= 30

    # Perturb one cell and verify compare_digests catches the change
    mutated = copy.deepcopy(second)
    sample_key = sorted(mutated["cells"].keys())[0]
    mutated["cells"][sample_key] = {
        "status": "exc",
        "exc_type": "RuntimeError",
        "exc_msg": "seeded regression",
    }
    diffs_mut, _ = backend_digest.compare_digests(first, mutated)
    assert len(diffs_mut) == 1
    assert sample_key in diffs_mut[0]

    # If marked unstable, compare_digests skips it and reports it in skipped
    mutated["unstable"] = [sample_key]
    diffs_unstable, skipped = backend_digest.compare_digests(first, mutated)
    assert diffs_unstable == []
    assert skipped == [sample_key]

    # Verify CLI --compare return codes
    p_a = tmp_path / "a.json"
    p_b = tmp_path / "b.json"
    p_m = tmp_path / "m.json"
    p_a.write_text(json.dumps(first), encoding="utf-8")
    p_b.write_text(json.dumps(second), encoding="utf-8")
    mutated["unstable"] = []
    p_m.write_text(json.dumps(mutated), encoding="utf-8")

    assert backend_digest.main(["--compare", str(p_a), str(p_b)]) == 0
    assert backend_digest.main(["--compare", str(p_a), str(p_m)]) == 1
