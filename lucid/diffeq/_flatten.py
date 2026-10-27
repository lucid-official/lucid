"""Packing a group of differently-shaped tensors into one flat state.

An ODE solver integrates a single tensor, but two things want to integrate a
*collection*: the adjoint method, whose augmented state is the solution
alongside its adjoint and one accumulator per parameter, and a caller whose
state is naturally a tuple.  Both reduce to the same trick — concatenate into
one vector, integrate that, split it back.
"""

import math
from typing import Callable, Sequence

import lucid
from lucid._tensor.tensor import Tensor
from lucid.diffeq._typing import EventFunction, RightHandSide

__all__: list[str] = []


def flatten(tensors: Sequence[Tensor]) -> Tensor:
    """Concatenate a group of tensors into one 1-D vector.

    Parameters
    ----------
    tensors : sequence of Tensor
        Non-empty group, all on the same device.  Shapes may differ.

    Returns
    -------
    Tensor
        A 1-D tensor holding every element in order.

    Notes
    -----
    Differentiable: the concatenation and the reshapes are ordinary ops, so a
    gradient reaching the flat vector routes back to each source tensor.
    """
    return lucid.concat([t.reshape(-1) for t in tensors])


def shapes_of(tensors: Sequence[Tensor]) -> list[tuple[int, ...]]:
    """Record the shapes needed to undo :func:`flatten`.

    Parameters
    ----------
    tensors : sequence of Tensor
        The group about to be flattened.

    Returns
    -------
    list of tuple of int
        One shape per tensor, in order.  Hand this to :func:`unflatten` to
        split a flat state back into its parts.
    """
    return [tuple(t.shape) for t in tensors]


def unflatten(flat: Tensor, shapes: Sequence[tuple[int, ...]]) -> list[Tensor]:
    """Split a flat vector back into tensors of the recorded shapes.

    Parameters
    ----------
    flat : Tensor
        1-D tensor produced by :func:`flatten`.
    shapes : sequence of tuple of int
        The shapes to rebuild, in the order they were flattened.

    Returns
    -------
    list of Tensor
        One tensor per entry of ``shapes``.

    Raises
    ------
    ValueError
        If ``flat`` does not hold exactly as many elements as ``shapes``
        describe.

    Notes
    -----
    Also differentiable, so the pair round-trips inside an autograd graph.
    """
    total = sum(math.prod(shape) for shape in shapes)
    if flat.shape != (total,):
        raise ValueError(
            f"flat state has shape {tuple(flat.shape)} but the recorded shapes "
            f"describe {total} elements"
        )

    pieces: list[Tensor] = []
    offset = 0
    for shape in shapes:
        size = math.prod(shape)
        pieces.append(flat[offset : offset + size].reshape(*shape))
        offset += size
    return pieces


def unflatten_rows(flat: Tensor, shapes: Sequence[tuple[int, ...]]) -> list[Tensor]:
    """Split every row of a stacked flat state back into its parts.

    Parameters
    ----------
    flat : Tensor
        Shape ``(n, total)`` — one flattened state per row.
    shapes : sequence of tuple of int
        The shapes each row is composed of.

    Returns
    -------
    list of Tensor
        One tensor of shape ``(n, *shape)`` per entry of ``shapes``.

    Notes
    -----
    The trajectory counterpart of :func:`unflatten`: a solve over a tuple
    state is run flat and split back once at the end, so the split happens
    per component rather than per time point.
    """
    rows = flat.shape[0]
    pieces: list[Tensor] = []
    offset = 0
    for shape in shapes:
        size = math.prod(shape)
        pieces.append(flat[:, offset : offset + size].reshape(rows, *shape))
        offset += size
    return pieces


def wrap_rhs(
    func: Callable[[Tensor, tuple[Tensor, ...]], Sequence[Tensor]],
    shapes: Sequence[tuple[int, ...]],
) -> RightHandSide:
    """Adapt a tuple-valued right-hand side to the flat state the solver uses.

    Parameters
    ----------
    func : callable
        ``f(t, y_tuple) -> derivative tuple``, the caller's own signature.
    shapes : sequence of tuple of int
        Component shapes of the state.

    Returns
    -------
    callable
        ``f(t, y_flat) -> dy_flat``.

    Raises
    ------
    TypeError
        If ``func`` does not return a sequence of tensors.
    ValueError
        If it returns the wrong number of components.
    """

    def flat_func(t: Tensor, y: Tensor) -> Tensor:
        parts = func(t, tuple(unflatten(y, shapes)))
        if isinstance(parts, Tensor) or not isinstance(parts, (list, tuple)):
            raise TypeError(
                f"func must return a tuple of tensors when y0 is a tuple, "
                f"got {type(parts).__name__}"
            )
        if len(parts) != len(shapes):
            raise ValueError(f"func returned {len(parts)} components but y0 has {len(shapes)}")
        return flatten(list(parts))

    return flat_func


def wrap_event(
    event_fn: Callable[[Tensor, tuple[Tensor, ...]], Tensor], shapes: Sequence[tuple[int, ...]]
) -> EventFunction:
    """Adapt a tuple-valued event function to the flat state.

    Parameters
    ----------
    event_fn : callable
        ``g(t, y_tuple) -> scalar tensor``.
    shapes : sequence of tuple of int
        Component shapes of the state.

    Returns
    -------
    callable
        ``g(t, y_flat) -> scalar tensor``.

    Notes
    -----
    Only the input is adapted — the event value is a scalar either way.
    """

    def flat_event(t: Tensor, y: Tensor) -> Tensor:
        return event_fn(t, tuple(unflatten(y, shapes)))

    return flat_event


def check_state(y0: object) -> tuple[Tensor, ...]:
    """Validate a tuple state and return it as a tuple of tensors.

    Parameters
    ----------
    y0 : object
        The caller's initial state, already known not to be a tensor.

    Returns
    -------
    tuple of Tensor
        The validated components.

    Raises
    ------
    TypeError
        If ``y0`` is not a tuple or list of tensors.
    ValueError
        If it is empty.
    """
    if not isinstance(y0, (tuple, list)):
        raise TypeError(f"y0 must be a Tensor or a tuple of Tensors, got {type(y0).__name__}")
    if not y0:
        raise ValueError("y0 tuple must not be empty")
    for index, part in enumerate(y0):
        if not isinstance(part, Tensor):
            raise TypeError(f"y0[{index}] must be a Tensor, got {type(part).__name__}")
    return tuple(y0)
