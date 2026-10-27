"""Every compile emitter, over float32 / int64 / int32 / bool, against eager.

The cases live in :mod:`._op_matrix`.  Each is compiled, traced on one
input and replayed on another, and must match eager on the second — read
back through ``.numpy()``, which is the path that once reported every
compiled integer output as float32.

A fallback to eager is allowed only where :data:`EXPECTED_EAGER` says why;
any other fallback fails, so a graph that stops compiling is seen the day
it happens.  An expected fallback that starts compiling fails as stale —
delete its entry.

The dtypes eager itself refuses are not generated: they are listed in
:data:`EAGER_REJECTS` and held to it by ``test_op_eager_rejects``.  A refusal
that is not listed fails here.

The matrix found, in its first rounds, answers that were silently wrong
(integer ``topk``, ``scatter`` on int64, ``erfinv(±1)``, float math on
integers, one executable shared by two callables returning different
tensors), graphs that aborted the process (``~`` on bool, ``all`` returned
directly), and ops eager itself got wrong (``arcsin`` on integers).
"""

import pytest

import lucid

from lucid.test.unit.compile._helpers import COMPILE_DEVICE
from lucid.test.unit.compile._op_matrix import CASE_BY_NAME, EXPECTED_EAGER, all_params, run


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")


@pytest.mark.parametrize(("name", "dtype"), all_params(), ids=lambda v: str(v))
def test_compiled_matches_eager(name: str, dtype: str) -> None:
    outcome = run(CASE_BY_NAME[name], dtype)
    if outcome.status == "rejected":
        pytest.fail(f"eager rejects {dtype}, not listed in EAGER_REJECTS: {outcome.detail}")
    listed = (name, dtype) in EXPECTED_EAGER
    if outcome.status == "eager":
        if listed:
            return
        pytest.fail(f"fell back to eager, not listed in EXPECTED_EAGER: {outcome.detail}")
    # A listed case that compiles now is a stale entry — ``meshgrid`` and
    # training-mode dropout sat here long after they compiled.
    assert not listed, "compiles now; remove it from EXPECTED_EAGER"
    assert outcome.status == "ok", f"{outcome.status}: {outcome.detail}"
