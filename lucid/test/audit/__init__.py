"""Exhaustive correctness diagnosis of Lucid, the model zoo included.

Why this exists
---------------
Auditing by hand does not scale and does not repeat.  A one-off sweep in
August 2026 found four defects — seven in-place activations returning the
pre-activation gradient, three reductions silently returning the incoming
seed under ``create_graph=True``, a whole-tensor assignment taking a
general scatter, and a CPU ``relu`` that turned NaN into zero — but it
reached only **4.7%** of the symbol × axis matrix, and nothing in its
output said so.  A survey that cannot report its own coverage is not a
survey; it is a sample presented as one.

So this package does three things the hand-rolled version could not:

1. **Enumerates the denominator.**  Every public symbol outside
   ``lucid.models`` is counted whether or not it can be probed, and the
   summary always prints reached-over-total.  The model zoo is not swept
   symbol by symbol; it is a package of its own with a family-by-family
   stage (:mod:`~lucid.test.audit._zoo`), so "the audit is clean" covers it.
2. **Names what it could not reach.**  ``SKIP`` is a first-class outcome
   with a reason, and ``--list-uncovered`` prints the work queue for
   extending :mod:`~lucid.test.audit._specs`.
3. **Interrogates a disagreement before reporting it.**  Most of what a
   naive sweep flags is an artefact of the probe — a finite difference
   taken near a pole, a quantity defined only up to a gauge, a gradient
   that is identically zero.  Each of those has its own status, so the
   FAIL list stays short enough to read.

Usage
-----
::

    python -m lucid.test.audit --help
    python -m lucid.test.audit --list-axes
    python -m lucid.test.audit --quick
    python -m lucid.test.audit --axis grad --nn.functional
    python -m lucid.test.audit --models --family lenet

Under pytest, :mod:`lucid.test.audit.test_audit_smoke` runs a small slice
so the harness itself cannot rot; the full sweep is deliberately not a
default test, because it is minutes long and its output is a report
rather than an assertion.
"""

from lucid.test.audit._axes import ALL_AXES, Axis, Context, axis_by_name, axis_names
from lucid.test.audit._result import Baseline, Finding, Report, Status
from lucid.test.audit._surface import Symbol, enumerate_surface

__all__ = [
    "ALL_AXES",
    "Axis",
    "Baseline",
    "Context",
    "Finding",
    "Report",
    "Status",
    "Symbol",
    "axis_by_name",
    "axis_names",
    "enumerate_surface",
]
