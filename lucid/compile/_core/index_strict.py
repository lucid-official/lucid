"""A compiled replay does not repeat an index check, so strict mode compiles no index op (LCD-991).

``LUCID_STRICT_INDEX=1`` makes an out-of-range index on Metal raise the CPU's ``IndexError``
(LCD-892).  The check is a read of the index back to the host, made when the op is dispatched.
A compiled replay dispatches nothing: the graph isolates a bad index in place (policy B, a
gather answers NaN, a scatter drops its update) and never reads it back, so a replay answered
what strict mode exists to refuse.  Only the call that traced the program was checked.

Two ways to close it were measured on embedding + linear, B8 x S128, V50257, MacBook M1 Pro
(ms per call, the result evaluated every call, best of 150):

* a compiled replay with strict mode off: 0.43-0.49;
* the same replay plus an in-graph ``all(in_range)`` read after every call: 0.46-0.52;
* the call run eagerly under strict mode: 0.77-0.85.

The in-graph flag is nearly free, but it is C++ in every index emitter and in the output list
of every entry point's executable.  This module takes the other way: under strict mode a graph
that holds an index op is not compiled, and its signature runs eagerly, where the eager gates
raise.  Strict mode is a debugging switch, and an eager call is what any call that needs
gradients already is.

Strict mode is in the cache key (the last entry of ``CacheKey.policy``).  The verdict is
remembered per signature, so a program that was declined under strict mode must neither
poison the same signature once the switch is off nor be replayed after it is turned on.
"""

from typing import TYPE_CHECKING

from lucid.compile._core.fallback import _fallback_notice
from lucid.nn.functional._index_checks import _strict_index

if TYPE_CHECKING:
    from lucid._C.engine.compile import TraceGraph

__all__ = ["INDEX_OPS", "StrictIndexInTrace", "declines_for_strict_index", "strict_index_detail"]

#: The ops of a trace whose eager Metal path refuses an index in strict mode: every gate of
#: ``backend/gpu/AxisIndex.h`` (``gather`` carries ``index_select``, ``take``,
#: ``take_along_dim`` and ``x[idx]``; ``scatter_add`` carries ``index_add`` and
#: ``index_fill``; ``scatter_set`` carries ``scatter`` and ``index_copy``) and the ones the
#: Python guards of ``_index_checks`` stand in front of (``one_hot``, ``embedding``).
#: ``F.cross_entropy`` and ``F.nll_loss`` gather in Python, so the engine's own
#: ``nll_loss`` / ``cross_entropy_loss`` (which have emitters, behind ``graph_class_target``)
#: are reached only through ``_C_engine``.  ``embedding_bag`` has no emitter yet: its calls
#: run eagerly and are checked.  ``test_index_bounds_contract.py`` compiles every index op
#: the contract knows and fails for one that is missing here, and ``test_strict_index.py``
#: fails for an emitter that gates an index and is not known to this module.
INDEX_OPS: frozenset[str] = frozenset(
    {
        "gather",
        "embedding",
        "embedding_bag",
        "one_hot",
        "scatter_add",
        "scatter_set",
        "scatter_amax",
        "scatter_amin",
        "scatter_prod",
        "nll_loss",
        "cross_entropy_loss",
    }
)


class StrictIndexInTrace(RuntimeError):
    """A trace holds an index op while ``LUCID_STRICT_INDEX`` is on.

    Raised by an entry point that declines a plan by raising (``fused_step``); the others
    return ``None`` as they do for any lowering they decline.
    """


def strict_index_detail(graph: TraceGraph) -> str | None:
    """Why ``graph`` is not compiled under strict mode, or ``None`` when it may be.

    Reads the switch now, so a program traced with it off is compiled and one traced with it
    on is not.  With the switch off this costs one environment lookup: the ops are not read.

    Parameters
    ----------
    graph : TraceGraph
        What a trace recorded.

    Returns
    -------
    str or None
        The detail for the fallback notice, naming the index ops the graph holds.
    """
    if not _strict_index():
        return None
    held = sorted({node.name for node in graph.ops} & INDEX_OPS)
    if not held:
        return None
    return (
        f"LUCID_STRICT_INDEX=1 and the graph indexes with {', '.join(held)}: a compiled replay "
        "never reads an index back, so the call runs eagerly, where an out-of-range index "
        "raises IndexError"
    )


def declines_for_strict_index(graph: TraceGraph) -> bool:
    """Whether the entry point must not compile ``graph``; says so once when it must not.

    Parameters
    ----------
    graph : TraceGraph
        What a trace recorded.

    Returns
    -------
    bool
        ``True`` when strict mode is on and ``graph`` holds an index op.
    """
    detail = strict_index_detail(graph)
    if detail is None:
        return False
    _fallback_notice("strict-index", detail)
    return True
