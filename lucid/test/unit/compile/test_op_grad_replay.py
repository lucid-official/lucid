"""Compiled training (``make_step``) against eager backward, case by case.

The cases are the float32 cases of :mod:`._op_matrix`, each wrapped as
``fn(x * w)`` so the gradient reaches the parameter ``w`` through the op —
see :mod:`._grad_matrix`.  A fallback to eager is allowed only where
:data:`EXPECTED_EAGER_GRAD` says why.

Before this test, a manual-VJP gap handed the whole backward to MPSGraph's
automatic differentiation, which aborted the process for 28 of these cases
(``where``, ``leaky_relu``, ``elu``, ``max``, ``cumsum``, ``bce``, …) and
returned a wrong gradient for 3 (``arcsin``, ``triu`` / ``tril`` off the
diagonal).  That fallback is now taken only for traces made of ops measured
to differentiate correctly.

A case with nothing to differentiate — no floating output, or one that does
not depend on ``w`` — is not generated: it is listed in :data:`NO_GRAD`, and
:func:`test_no_grad_case_still_has_none` holds it there.  An unlisted case
that finds nothing to differentiate fails.
"""

import pytest

import lucid

from lucid.test.unit.compile._grad_matrix import (
    EXPECTED_EAGER_GRAD,
    GRAD_CASES,
    MANUAL_VJP_GAPS,
    NO_GRAD,
    run_grad,
)
from lucid.test.unit.compile._helpers import COMPILE_DEVICE
from lucid.test.unit.compile._op_matrix import CASE_BY_NAME


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")


@pytest.mark.parametrize("case", GRAD_CASES, ids=lambda c: c.name)
def test_compiled_gradient_matches_eager(case: object) -> None:
    outcome = run_grad(case)  # type: ignore[arg-type]
    if outcome.status == "nograd":
        pytest.fail(f"nothing to differentiate, not listed in NO_GRAD: {outcome.detail}")
    listed = case.name in EXPECTED_EAGER_GRAD  # type: ignore[attr-defined]
    if outcome.status == "eager":
        if listed:
            return
        pytest.fail(f"fell back to eager, not listed: {outcome.detail}")
    # A listed case that compiles now is a stale entry — drop it, or the
    # list stops saying what still runs eager.
    assert not listed, "compiles now; remove it from EXPECTED_EAGER_GRAD"
    assert outcome.status == "ok", f"{outcome.status}: {outcome.detail}"


@pytest.mark.parametrize("case", GRAD_CASES, ids=lambda c: c.name)
def test_every_case_trains_through_a_manual_vjp(
    case: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same cases with MPSGraph's autodiff forbidden as a fallback.

    Autodiff covers a missing VJP only while every op in the graph is on its
    safe list; a train-mode batch norm or an interpolation is not, and one
    op without a VJP then sends the whole model's step eager.  Twenty ops
    were in that state — MobileNet trained eager over ``relu6``, CLIP over
    ``minimum`` — and nothing reported it, because this matrix took the
    autodiff path.  Forbidding it makes every missing VJP a failure here.
    """
    from lucid._C import engine as _C_engine

    monkeypatch.setenv("LUCID_MANUAL_VJP_REQUIRE", "1")
    # An executable built earlier through autodiff would be served from the
    # process-wide cache without the VJP walk ever running.
    _C_engine.compile.session_cache_clear()
    outcome = run_grad(case)  # type: ignore[arg-type]
    if outcome.status == "nograd":
        pytest.fail(f"nothing to differentiate, not listed in NO_GRAD: {outcome.detail}")
    if outcome.status == "eager":
        why = EXPECTED_EAGER_GRAD.get(case.name) or MANUAL_VJP_GAPS.get(case.name)  # type: ignore[attr-defined]
        if why:
            return
        pytest.fail(f"no manual VJP on the path: {outcome.detail}")
    assert outcome.status == "ok", f"{outcome.status}: {outcome.detail}"


@pytest.mark.parametrize("name", sorted(NO_GRAD))
def test_no_grad_case_still_has_none(name: str) -> None:
    """A case left out as having nothing to differentiate still has nothing.

    Runs :func:`run_grad` itself, so the detection is the one the matrix
    makes.  A case that has a gradient now is a stale entry: drop it from
    :data:`NO_GRAD` and the two tests above check it.
    """
    outcome = run_grad(CASE_BY_NAME[name])
    assert outcome.status == "nograd", (
        f"{outcome.status}: has a gradient now; drop it from NO_GRAD " f"({outcome.detail})"
    )
    assert outcome.detail == NO_GRAD[name], f"{outcome.detail!r}, NO_GRAD says {NO_GRAD[name]!r}"
