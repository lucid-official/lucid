"""Shape generators — broadcasting cases, edge shapes, sweep matrices.

Used by the per-op test template so every op is exercised across a
representative shape spectrum without each file re-deriving the same
list.
"""

from collections.abc import Iterator

# Compact shape sweep used by the canonical per-op template.  Covers:
#   - 0-D scalar
#   - 1-D small / large
#   - 2-D square / rectangular
#   - 3-D batch
#   - 4-D image-like (1, C, H, W)
BASIC_SHAPES: tuple[tuple[int, ...], ...] = (
    (),
    (1,),
    (8,),
    (4, 4),
    (3, 5),
    (2, 3, 4),
    (1, 3, 8, 8),
)


# Pairs of shapes that broadcast to a common output.  Used by binary-op
# tests to validate the broadcasting rules without re-stating them.
BROADCAST_PAIRS: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (
    ((3,), (3,)),
    ((1,), (4,)),
    ((4, 1), (1, 5)),
    ((2, 3, 1), (1, 3, 5)),
    ((1, 3, 8, 8), (3, 1, 1)),
    ((), (2, 3)),
)


# Shapes that exercise edge conditions (empty, 1-element, 0-D).
EDGE_SHAPES: tuple[tuple[int, ...], ...] = ((), (1,), (0,), (0, 3), (3, 0, 5), (1, 1, 1))


def reduction_axes(shape: tuple[int, ...]) -> Iterator[int | tuple[int, ...]]:
    """Yield interesting reduction axis specs for a given shape:
    each individual dim, then a multi-dim sweep when ``ndim >= 2``."""
    n = len(shape)
    for i in range(n):
        yield i
    if n >= 2:
        yield (0, n - 1)
    if n >= 3:
        yield (0, 1, n - 1)
