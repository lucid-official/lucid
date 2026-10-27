"""Whether a traced attention is better left to eager's fused kernel (CHA-14).

The compiled graph lowers ``scaled_dot_product_attention`` to matmul,
softmax and matmul, and so writes the whole score matrix — batch × heads ×
queries × keys — to memory and reads it back twice.  Eager MLX runs a fused
kernel that never writes it.  Measured on an M4 Max, fp16, dim 1536 / 12
heads (a Wan DiT block over a KV cache, 4,680 queries):

==========  ======  ========  ===================
keys        eager   compiled  score matrix
==========  ======  ========  ===================
4,680       37.0    37.8 ms   0.5 GiB
9,360       53.9    58.6 ms   1.0 GiB
18,720      87.5    120.8 ms  2.0 GiB
==========  ======  ========  ===================

Self-attention up to 4,096 tokens (≤ 0.4 GiB) matched eager to within 1%,
and MPSGraph's own fused attention (``LUCID_COMPILE_FUSED_SDPA=1``) changed
nothing — it materialises the scores as well.  So past half a GiB of scores
the call is routed to eager, where it is faster, with a one-time notice.
``LUCID_COMPILE_LONG_ATTENTION=compile`` keeps such calls compiled.
"""

import math
import os

from lucid._C import engine as _C_engine

#: Score-matrix size above which the compiled decomposition loses to eager.
SCORE_LIMIT_BYTES = 512 * 2**20


def _itemsize(dtype: object) -> int:
    """Bytes per element, read off a dtype's name (engine or graph dtype)."""
    name = str(getattr(dtype, "name", dtype)).lower()
    if "16" in name:
        return 2
    if "64" in name:
        return 8
    if "8" in name and "128" not in name:
        return 1
    return 4


def long_attention(tracer: _C_engine.compile.Tracer) -> str | None:
    """Why the traced call should run eagerly, or None when compiling pays.

    Reads every ``scaled_dot_product_attention`` in the trace and sizes its
    score matrix from the query and key shapes.
    """
    if os.environ.get("LUCID_COMPILE_LONG_ATTENTION") == "compile":
        return None
    shapes: dict[int, tuple[list[int], object]] = {
        int(fid): (list(impl.shape), impl.dtype) for fid, impl in tracer.external_feeds.items()
    }
    for node in tracer.graph.ops:
        for out in node.outputs:
            shapes.setdefault(int(out.id), (list(out.shape), out.dtype))
    worst: tuple[int, int, int] | None = None
    for node in tracer.graph.ops:
        if node.name != "scaled_dot_product_attention" or len(node.inputs) < 2:
            continue
        q = shapes.get(int(node.inputs[0]))
        k = shapes.get(int(node.inputs[1]))
        if q is None or k is None or len(q[0]) < 2 or len(k[0]) < 2:
            continue
        queries, keys = q[0][-2], k[0][-2]
        size = math.prod(q[0][:-2]) * queries * keys * _itemsize(q[1])
        if worst is None or size > worst[0]:
            worst = (size, queries, keys)
    if worst is None or worst[0] <= SCORE_LIMIT_BYTES:
        return None
    size, queries, keys = worst
    return (
        f"attention of {queries} queries over {keys} keys writes a "
        f"{size / 2**20:.0f} MiB score matrix in the compiled graph, which "
        "eager's fused kernel never materialises — running it eagerly, where "
        "it is faster (LUCID_COMPILE_LONG_ATTENTION=compile keeps it compiled)"
    )
