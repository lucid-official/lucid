"""
lucid.compile._debug.diagnose — manual-VJP coverage diagnostics.

Provides :func:`diagnose` which traces a callable, classifies every op
in the captured graph against the manual-VJP registry, and returns a
human-readable :class:`DiagnosisReport` summarising:

* **Registered** ops — covered by a `VjpEmitter`; manual-VJP path will
  produce a real gradient.
* **GradSink** ops — in the walker's ``no_grad_ops`` list (factories,
  integer casts, comparisons, arg-reduce).  Gradient flow stops here
  by design; this is *not* a fallback.
* **Missing** ops — neither.  Compile path will soft-fallback to
  ``gradientForPrimaryTensor:`` (or hard-fail under
  ``LUCID_MANUAL_VJP_REQUIRE=1``).

The report runs in seconds without compiling or executing anything —
it just walks the captured :class:`TraceGraph` and queries the static
C++ VJP registry per op name.

Usage
-----
::

    import lucid
    import lucid.nn as nn
    from lucid.compile import diagnose

    model = nn.Linear(8, 4).to('metal')
    x = lucid.zeros(2, 8).to('metal')

    rpt = diagnose(model, x)
    print(rpt)
    # → "Diagnosis: 2 ops total, 100% manual VJP coverage."

If a model triggers a fallback in production, the same call surfaces
the offending op + a sample shape::

    rpt = diagnose(model_with_var, x)
    # → "Diagnosis: 5 ops total. 1 op will trigger autograd fallback:
    #    var (x1, sample (2, 8) float32)."
"""

import dataclasses
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING, override

from lucid._C import engine as _C_engine

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


__all__ = ["OpInfo", "DiagnosisReport", "diagnose"]


@dataclasses.dataclass(frozen=True, slots=True)
class OpInfo:
    """Per-op-name summary inside a :class:`DiagnosisReport`.

    A single :class:`OpInfo` aggregates *all* nodes in the trace with
    the same op name into one record (``count`` is the number of
    occurrences; ``sample_shape`` / ``sample_dtype`` are taken from the
    first occurrence's primary output, useful when reporting a coverage
    gap).

    Attributes
    ----------
    name : str
        The trace op name (e.g. ``"linear"``, ``"var"``, ``"argmax"``).
    count : int
        Number of times the op appears in the captured trace.
    sample_shape : tuple[int, ...] | None
        Shape of the first occurrence's primary output, or ``None`` if
        the op produces no outputs (rare; defensive fallback).
    sample_dtype : str | None
        Dtype name of the first occurrence's primary output
        (``"float32"`` / ``"int64"`` / ...) — uses
        :meth:`lucid._C.engine.Dtype.name`.
    """

    name: str
    count: int
    sample_shape: tuple[int, ...] | None
    sample_dtype: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class DiagnosisReport:
    """Result of :func:`diagnose`.

    Holds three parallel buckets of :class:`OpInfo` plus a one-line
    ``recommendation``.  The class is immutable; ``__str__`` renders a
    short human-readable summary suitable for logs.

    Attributes
    ----------
    total_ops : int
        Sum of every op occurrence in the trace.
    registered : list[OpInfo]
        Ops with a registered `VjpEmitter` — real gradient.
    grad_sinks : list[OpInfo]
        Ops in the walker's ``no_grad_ops`` set — gradient flow stops
        here by design (not a fallback).
    uncovered : list[OpInfo]
        Ops without an emitter and not in the sink set — will trigger
        soft-fallback to MPSGraph autograd (or hard-fail under
        ``LUCID_MANUAL_VJP_REQUIRE=1``).
    recommendation : str
        One-line summary of fallback risk.
    """

    total_ops: int
    registered: list[OpInfo]
    grad_sinks: list[OpInfo]
    uncovered: list[OpInfo]
    recommendation: str

    @override
    def __str__(self) -> str:
        """Render a short human-readable summary."""
        lines = [
            f"Diagnosis: {self.total_ops} op(s) total "
            f"({len(self.registered)} kinds registered, "
            f"{len(self.grad_sinks)} kinds grad-sink, "
            f"{len(self.uncovered)} kinds uncovered).",
            self.recommendation,
        ]
        if self.uncovered:
            lines.append("Uncovered ops:")
            for info in self.uncovered:
                lines.append(
                    f"  - {info.name} (x{info.count}, "
                    f"sample {info.sample_shape}:{info.sample_dtype})"
                )
        return "\n".join(lines)


def diagnose(fn: Callable[..., object], *example_inputs: Tensor) -> DiagnosisReport:
    """Trace ``fn(*example_inputs)`` and report manual-VJP coverage.

    Runs ``fn`` once inside a :func:`_tracing` context to capture the
    op DAG, then classifies every op against the registry exposed by
    `vjp_registration_status` (C++ engine binding).  Does
    **not** compile or execute the captured graph — purely
    introspective, runs in milliseconds even for large models.

    Parameters
    ----------
    fn : Callable
        Any callable that returns a :class:`Tensor` (or a tuple of
        them).  An :class:`nn.Module` instance works because
        ``Module.__call__`` is the callable.
    *example_inputs : Tensor
        Positional arguments forwarded to ``fn``.  Their concrete
        shape / dtype are recorded into the trace, so the diagnosis
        reflects exactly what would be compiled for these inputs.

    Returns
    -------
    DiagnosisReport
        Three-bucket classification + recommendation string.  Inspect
        ``.uncovered`` to see which ops will fall back, or print
        ``str(report)`` for a quick log line.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> from lucid.compile import diagnose
    >>> model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    >>> print(diagnose(model, lucid.zeros(2, 8)))           # doctest: +SKIP
    Diagnosis: 5 op(s) total ...
    100% manual VJP coverage — no fallback expected ...

    See Also
    --------
    lucid.compile.fused_step : the production training-step compile
        path that consults the same registry under the hood.
    """
    # Local imports avoid the package-init cycle that ``lucid.compile``
    # carefully sidesteps via its own lazy ``__getattr__``.
    from lucid.compile._core.trace_context import compile_attempt, recording

    with compile_attempt(), recording(watch=example_inputs) as tracer:
        fn(*example_inputs)

    op_counts: Counter[str] = Counter()
    op_samples: dict[str, tuple[tuple[int, ...], str]] = {}
    for node in tracer.graph.ops:
        op_counts[node.name] += 1
        if node.name not in op_samples and node.outputs:
            primary = node.outputs[0]
            op_samples[node.name] = (
                tuple(int(d) for d in primary.shape),
                primary.dtype.name,
            )

    registered: list[OpInfo] = []
    grad_sinks: list[OpInfo] = []
    uncovered: list[OpInfo] = []

    Reg = _C_engine.compile.VjpRegistration
    for name, count in sorted(op_counts.items()):
        sample = op_samples.get(name)
        info = OpInfo(
            name=name,
            count=count,
            sample_shape=sample[0] if sample else None,
            sample_dtype=sample[1] if sample else None,
        )
        status = _C_engine.compile.vjp_registration_status(name)
        if status == Reg.Registered:
            registered.append(info)
        elif status == Reg.GradSink:
            grad_sinks.append(info)
        else:
            uncovered.append(info)

    total = sum(op_counts.values())
    if not uncovered:
        rec = (
            "100% manual VJP coverage — no fallback expected under "
            "LUCID_MANUAL_VJP_REQUIRE=1."
        )
    else:
        names = ", ".join(f"{i.name} (x{i.count})" for i in uncovered)
        rec = (
            f"{len(uncovered)} op kind(s) will trigger autograd fallback: "
            f"{names}.  Set LUCID_MANUAL_VJP_DEBUG=1 for per-call stderr "
            "diagnostics, or LUCID_MANUAL_VJP_REQUIRE=1 to surface as "
            "a hard RuntimeError."
        )

    return DiagnosisReport(
        total_ops=total,
        registered=registered,
        grad_sinks=grad_sinks,
        uncovered=uncovered,
        recommendation=rec,
    )
