"""Pre-compile gate: is a traced graph safe to compile with a symbolic batch axis?

A symbolic-batch compile aborts (uncatchably) or silently mis-shapes when the
graph bakes the batch dimension into a constant shape MPSGraph cannot infer:

* explicit ``broadcast_to`` / ``expand`` / ``repeat`` / ``tile`` (the target
  shape carries the batch);
* a ``concatenate`` / ``stack`` on the **batch axis** (dim 0) — the joined dim
  is ``2*B``, which the dim-0 symbolic machinery can't represent;
* a **batch-shaped constant** — anything the call builds out of nothing that
  is sized like the input (``zeros_like(x)``, an RNN's zero hidden init,
  ``arange(x.shape[0])``).  A Metal factory stays an op in the trace, sized
  by its output; a constant built elsewhere is folded into one feed
  (``Tracer.fold_constants``) holding the trace-time value, and the gate is
  given those feeds' shapes;
* a **permutation that moves the batch axis** off dim 0 (``x.T``,
  ``permute(1, 0)``) — the symbolic machinery assumes the batch stays at
  dim 0, and the executable answered a traced-batch-shaped result for every
  batch, or aborted MPSGraph on the next ``reshape``;
* a **reduction over the batch axis** — a count that comes from the batch
  is a trace-time constant: ``std(x, dim=0)``'s Bessel factor ``n/(n-1)``
  is a ``full`` in the graph, and was applied to every batch size;
* an **axis-sensitive op on the batch axis** — splitting it (the number of
  pieces is the batch's), rolling it, taking its diagonal, gathering or
  scattering along it (the index's shape is the traced batch): wrong values,
  a wrong shape, or MPSGraph aborting on a gather.

This gate scans the trace once.  A ``False`` result routes the model to robust
per-shape static caching (correct, never crashes) instead of attempting
symbolic.  It is the safety net that lets ``dynamic=True`` default to *attempting*
symbolic: only graphs that pass the gate ride the single-executable path.

The *view* ops (reshape / flatten / squeeze / contiguous / reduce-squeeze) are
deliberately NOT gated here — they fail gracefully in the emitter (returning nil
when the batch isn't provably preserved at dim 0), so the compile simply retries
static.  Only the ops above, which abort uncatchably or corrupt silently, need
the up-front gate.
"""

from collections.abc import Iterable, Sequence

# Ops whose target shape carries the batch dimension verbatim.
_BROADCAST_OPS: frozenset[str] = frozenset({"broadcast_to", "expand", "repeat", "tile"})
# Concatenate / stack: unsafe only when the join axis IS the batch axis (dim 0).
_JOIN_OPS: frozenset[str] = frozenset({"concatenate", "concat", "cat", "stack"})
# Factory ops: unsafe only when the produced shape's leading dim is the batch.
_FACTORY_OPS: frozenset[str] = frozenset({"zeros", "ones", "full", "arange", "eye", "linspace"})

# Ops whose answer depends on the length of the axis they act along, so the
# batch axis may not be that axis.  (``embedding`` is its own op, and a table
# lookup along the weight's axis 0 is not caught here.)
_AXIS_SENSITIVE_OPS: frozenset[str] = frozenset(
    {
        "split",
        "split_at",
        "unbind",
        "roll",
        "diagonal",
        "gather",
        "scatter",
        "scatter_add",
        # The overwrite: ``x[key] = v`` (always along axis 0 of the flattened
        # tensor), ``lucid.scatter`` and ``index_copy`` trace as it.
        "scatter_set",
        "index_select",
        "take",
    }
)
_AXIS_ATTRS = ("axis", "axes", "dim", "dims", "axis1", "axis2")

__all__ = ["graph_symbolic_safe"]


def _touches_axis_zero(attrs: dict[str, object]) -> bool:
    for key in _AXIS_ATTRS:
        value = attrs.get(key)
        if value == 0 or (isinstance(value, (list, tuple)) and 0 in value):
            return True
    return False


def _out_shape(op: object) -> tuple[int, ...]:
    """Leading-output shape of ``op`` as a tuple, or ``()`` if unavailable."""
    outs = getattr(op, "outputs", None)
    if not outs:
        return ()
    shp = getattr(outs[0], "shape", None)
    return tuple(shp) if shp else ()


def graph_symbolic_safe(
    graph: object,
    trace_batch: int,
    batch_ids: set[int] | None = None,
    constants: Iterable[Sequence[int]] = (),
) -> bool:
    """Return ``True`` if ``graph`` can be compiled with a symbolic batch axis.

    Parameters
    ----------
    graph : TraceGraph
        The recorded trace (its ``ops`` are scanned by name + attrs + shape).
    trace_batch : int
        The leading (batch) size of the user input at trace time — used to tell a
        batch-shaped constant (``zeros_like(x)`` → an axis of ``trace_batch``)
        from a concrete-shaped one (a fixed positional-encoding table, say).
    batch_ids : set of int, optional
        Trace ids of the inputs that carry the batch.  The axis rules —
        permuting, reducing, splitting along axis 0 — apply only to ops fed,
        however indirectly, by one of them: splitting a fused q/k/v weight
        along its axis 0 has nothing to do with the batch.  ``None`` treats
        every op as fed by the batch.
    constants : iterable of sequence of int, optional
        Shapes of the feeds the call built itself — neither an argument nor
        the model's state.  One with an axis as long as ``trace_batch`` is
        taken for a constant sized from the batch: its trace-time value would
        be fed to every batch size.

    Returns
    -------
    bool
        ``False`` if any op would bake the batch into a constant MPSGraph can't
        symbolicise; ``True`` otherwise.
    """
    if any(trace_batch in tuple(shape) for shape in constants):
        return False
    carrying = set(batch_ids) if batch_ids is not None else None
    for op in getattr(graph, "ops", []):
        name = getattr(op, "name", "")
        if name in _BROADCAST_OPS:
            return False
        if name in _JOIN_OPS:
            attrs = getattr(op, "attrs", None) or {}
            axis = attrs.get("dim", attrs.get("axis", 0))
            rank = len(_out_shape(op))
            if isinstance(axis, int):
                if axis < 0:
                    axis += rank
                if axis == 0:  # joining along the batch axis
                    return False
            else:
                return False  # unknown axis → conservatively unsafe
        if name in _FACTORY_OPS:
            shape = _out_shape(op)
            if shape and shape[0] == trace_batch:
                return False
        inputs = list(getattr(op, "inputs", []))
        fed = carrying is None or any(i in carrying for i in inputs)
        if carrying is not None and fed:
            carrying.update(o.id for o in getattr(op, "outputs", []))
        if not fed:
            continue
        attrs = getattr(op, "attrs", None) or {}
        perm = attrs.get("permutation")
        if isinstance(perm, (list, tuple)) and perm and perm[0] != 0:
            return False  # the batch axis leaves dim 0
        dims = attrs.get("dims")
        if isinstance(dims, (list, tuple)) and 0 in dims:
            return False  # a reduction over the batch axis
        if name in _AXIS_SENSITIVE_OPS and _touches_axis_zero(attrs):
            return False
    return True
