"""Compiled training (``make_step``) against eager autograd, case by case.

Reuses the float32 cases of :mod:`._op_matrix`.  Each case is wrapped as
``fn(x * w)`` with ``w`` a parameter, so the gradient reaches ``w`` through
the case's op and its compiled VJP.  The loss weights every floating output
by a fixed, non-uniform probe — with a plain ``sum`` a transposed or
permuted gradient can come out the same.

The compiled step is traced on one input and replayed on another; the
gradient on the second is compared with eager backward on the same input.
"""

import io
import sys
from dataclasses import dataclass

import numpy as np

import lucid
import lucid.nn as nn

from lucid.test.unit.compile._op_matrix import CASES, DEV, Case, _leaves, make_input


@dataclass
class GradOutcome:
    status: str  # ok | wrong | eager | error | nograd | recipe
    detail: str = ""


#: The two ways :func:`run_grad` finds nothing to differentiate (``nograd``).
NO_FLOAT = "no floating output"
NO_DEP = "output does not depend on w"


_PROBES: dict[tuple[int, ...], lucid.Tensor] = {}


def _probe(shape: tuple[int, ...]) -> lucid.Tensor:
    if shape not in _PROBES:
        state = lucid.get_rng_state()
        lucid.manual_seed(11)
        _PROBES[shape] = (lucid.rand(*shape) + 0.5).to(DEV) if shape else lucid.tensor(1.3).to(DEV)
        lucid.set_rng_state(state)
    return _PROBES[shape]


class _Wrapped(nn.Module):
    def __init__(self, case: Case) -> None:
        super().__init__()
        lucid.manual_seed(5)
        self.w = nn.Parameter((lucid.rand(*case.shape) + 0.5).to(DEV))
        self._fn = case.fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        out = self._fn(x * self.w)
        floats = [t for t in _leaves(out) if t.is_floating_point()]
        if not floats:
            raise _NoGrad
        total = (floats[0] * _probe(tuple(floats[0].shape))).sum()
        for t in floats[1:]:
            total = total + (t * _probe(tuple(t.shape))).sum()
        return total


class _NoGrad(Exception):
    pass


def _loss(out: lucid.Tensor) -> lucid.Tensor:
    return out


def run_grad(case: Case) -> GradOutcome:
    if case.random or "f32" not in case.dtypes:
        return GradOutcome("recipe", "random or no float32")
    x1 = make_input(case.kind, "f32", case.shape, 1)
    x2 = make_input(case.kind, "f32", case.shape, 2)

    model = _Wrapped(case)
    try:
        model.w.grad = None
        loss = model(x2)
        if not loss.requires_grad:
            return GradOutcome("nograd", NO_DEP)
        loss.backward()
    except _NoGrad:
        return GradOutcome("nograd", NO_FLOAT)
    except Exception as e:  # noqa: BLE001
        return GradOutcome("recipe", f"eager backward: {type(e).__name__}: {str(e)[:80]}")
    if model.w.grad is None:
        return GradOutcome("nograd", "no gradient reaches w")
    want = model.w.grad.numpy().copy()

    step = lucid.compile.make_step(model, _loss)
    err, old = io.StringIO(), sys.stderr
    sys.stderr = err
    try:
        model.w.grad = None
        step(x1).backward()
        model.w.grad = None
        step(x2).backward()
    except Exception as e:  # noqa: BLE001
        return GradOutcome("error", f"{type(e).__name__}: {str(e)[:160]}")
    finally:
        sys.stderr = old
    fallback = step.eager_only  # type: ignore[attr-defined]
    if len(fallback.snapshot()) if hasattr(fallback, "snapshot") else len(fallback):
        why = [
            ln for ln in err.getvalue().splitlines() if "fallback" in ln or "compile_trace" in ln
        ]
        return GradOutcome("eager", (why[-1] if why else "")[:160])
    if model.w.grad is None:
        return GradOutcome("wrong", "compiled step left w.grad empty")
    got = model.w.grad.numpy()
    if got.shape != want.shape:
        return GradOutcome("wrong", f"grad shape {got.shape}, want {want.shape}")
    if not np.allclose(got, want, rtol=1e-3, atol=1e-4, equal_nan=True):
        d = np.nanmax(np.abs(got.astype(np.float64) - want.astype(np.float64)))
        return GradOutcome("wrong", f"grad max|diff| {d:.3g}")
    return GradOutcome("ok")


#: Float32 case → which of :data:`NO_FLOAT` / :data:`NO_DEP` it is, for every
#: case with no gradient to compare.  They are left out of
#: :data:`GRAD_CASES`.  Strict both ways: ``test_no_grad_case_still_has_none``
#: runs :func:`run_grad` on each and fails when one has a gradient now (drop
#: the entry and the matrix checks it), and a case not listed here that finds
#: nothing to differentiate fails the matrix.
NO_GRAD: dict[str, str] = {
    # Bool predicates and masks.
    **dict.fromkeys(
        (
            "isnan",
            "isinf",
            "isfinite",
            "logical_not",
            "astype_bool",
            "eq",
            "ne",
            "lt",
            "le",
            "gt",
            "ge",
            "logical_and",
            "logical_or",
            "logical_xor",
            "isclose",
            "all",
            "any",
        ),
        NO_FLOAT,
    ),
    # Integer results: a cast, indices, a one-hot code.
    **dict.fromkeys(("astype_i64", "argmax", "argmin", "argsort", "one_hot"), NO_FLOAT),
}

GRAD_CASES = [c for c in CASES if "f32" in c.dtypes and not c.random and c.name not in NO_GRAD]

#: Case → why its compiled training step runs eager.  The manual VJPs for
#: prod, cumprod, cummax/cummin, sort/kthvalue, repeat_interleave,
#: scatter/scatter_add and det (2026-09-24) emptied this of all but the one
#: op that has no forward emitter to differentiate.
EXPECTED_EAGER_GRAD: dict[str, str] = {
    # Float ``//``, ``%`` and ``fmod`` are exact engine kernels now (LCD-159)
    # with no compile VJP, and the last two no forward emitter either.
    **dict.fromkeys(
        ("floordiv", "remainder", "fmod"),
        "LCD-444: no compile VJP (or emitter) for floordiv / remainder / fmod",
    ),
}

#: Case → why it has no manual VJP yet, although MPSGraph's autodiff
#: differentiates it correctly where nothing else rules autodiff out — so it
#: trains compiled in :func:`test_compiled_gradient_matches_eager`, and would
#: run eager in a graph with a train-mode batch norm.  Empty since
#: ``conv_transpose3d`` got its VJP.
MANUAL_VJP_GAPS: dict[str, str] = {}


def _main() -> None:  # pragma: no cover — manual triage
    import os

    os.environ["LUCID_COMPILE_VERBOSE"] = "1"
    only = set(sys.argv[1:])
    counts: dict[str, int] = {}
    for case in GRAD_CASES:
        if only and case.name not in only:
            continue
        print(f"RUN {case.name}", file=sys.stderr, flush=True)
        o = run_grad(case)
        counts[o.status] = counts.get(o.status, 0) + 1
        if o.status not in ("ok",):
            print(f"{o.status:7s} {case.name:18s} {o.detail}", flush=True)
    print("counts", counts)


if __name__ == "__main__":  # pragma: no cover
    _main()
