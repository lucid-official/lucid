"""Flatten parameters into a single 1-D vector and back.

Mirrors ``reference framework.nn.utils.{parameters_to_vector, vector_to_parameters}``.
Useful when an algorithm needs the full parameter set as a flat vector — most
commonly L-BFGS / second-order methods, hyper-gradient solvers, and reinforcement
learning trust-region updates.
"""

from typing import Iterable

import lucid
from lucid._tensor.tensor import Tensor
from lucid.nn.parameter import Parameter


def parameters_to_vector(parameters: Iterable[Parameter]) -> Tensor:
    r"""Flatten an iterable of parameters into a single 1-D ``Tensor``.

    Every parameter is detached, reshaped to 1-D, and concatenated in
    iteration order.  Useful when an algorithm needs to view the whole
    parameter set as one vector — most commonly second-order optimisers
    (L-BFGS, conjugate-gradient, trust-region methods), hyper-gradient
    solvers, and natural-gradient pre-conditioners.

    Parameters
    ----------
    parameters : iterable of Parameter
        Parameters to flatten.  Each must be a :class:`~lucid._tensor.tensor.Tensor`
        (or subclass).  All entries must share device and dtype; a mismatch
        raises :class:`ValueError`.

    Returns
    -------
    Tensor
        1-D tensor of length :math:`\sum_k \text{numel}(p_k)` on the same
        device / dtype as the first parameter.  It is part of
        the autograd graph, as in the reference framework — a gradient
        taken through it reaches every parameter.  Call ``.detach()`` for a
        snapshot.

    Notes
    -----
    Given parameters :math:`p_1, p_2, \dots, p_K` with sizes :math:`n_k`,
    the returned vector is

    .. math::

        v \;=\; \bigl[\,\operatorname{vec}(p_1);\,
        \operatorname{vec}(p_2);\, \dots;\,
        \operatorname{vec}(p_K)\,\bigr] \in \mathbb{R}^{\sum_k n_k}.

    The inverse operation is :func:`vector_to_parameters`, which copies a
    1-D tensor of the same length back into the original parameter shapes.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import parameters_to_vector
    >>> model = nn.Linear(4, 2)                 # 4 * 2 weights + 2 biases
    >>> vec = parameters_to_vector(model.parameters())
    >>> vec.shape
    (10,)
    """
    flats: list[Tensor] = []
    target_device: str | None = None
    target_dtype: object | None = None
    for p in parameters:
        if not isinstance(p, Tensor):
            raise TypeError(f"expected Parameter or Tensor, got {type(p).__name__}")
        if target_device is None:
            target_device = str(p._impl.device)
            target_dtype = p.dtype
        elif str(p._impl.device) != target_device:
            raise ValueError(
                f"all parameters must live on the same device; saw "
                f"{target_device} then {p._impl.device}"
            )
        elif p.dtype is not target_dtype:
            raise ValueError(f"all parameters must share dtype; saw {target_dtype} then {p.dtype}")
        flats.append(p.reshape(-1))
    if not flats:
        return lucid.zeros(0)
    return lucid.cat(flats)


def vector_to_parameters(vec: Tensor, parameters: Iterable[Parameter]) -> None:
    r"""Copy a flat 1-D tensor back into the supplied parameters in place.

    Inverse of :func:`parameters_to_vector`.  ``vec`` is sliced according
    to the size of each parameter (in iteration order) and the slices are
    written into the parameters' storage.  Typically called by line-search
    routines and second-order optimisers that operate on the concatenated
    vector and need to push the result back into the model.

    Parameters
    ----------
    vec : Tensor
        1-D source tensor.  Length must equal :math:`\sum_k \text{numel}(p_k)`
        for the supplied parameter iterator; mismatches raise
        :class:`ValueError`.
    parameters : iterable of Parameter
        Destination parameters.  Each parameter's storage is overwritten
        with the matching slice of ``vec``, reshaped to the parameter's
        own shape.

    Returns
    -------
    None
        The copy is performed in place; no value is returned.

    Notes
    -----
    Conceptually the operation splits ``vec`` at the parameter
    boundaries and reshapes each chunk:

    .. math::

        p_k \;\leftarrow\; \operatorname{reshape}\bigl(
            v[s_{k-1}\!:\!s_k],\, \text{shape}(p_k)\bigr),
        \qquad s_k = \sum_{j \leq k} n_j.

    Autograd graph state is left intact — the assignment touches storage
    directly, not the parameter's identity, so existing references in
    optimiser state remain valid.

    Examples
    --------
    >>> import lucid
    >>> import lucid.nn as nn
    >>> from lucid.nn.utils import parameters_to_vector, vector_to_parameters
    >>> model = nn.Linear(4, 2)
    >>> lr = 0.1
    >>> v = parameters_to_vector(model.parameters())
    >>> search_direction = lucid.ones(v.shape[0])
    >>> v = v + lr * search_direction
    >>> vector_to_parameters(v, model.parameters())
    >>> bool((parameters_to_vector(model.parameters()) == v).all())
    True
    """
    if len(vec.shape) != 1:
        raise ValueError(f"vec must be 1-D, got shape {tuple(vec.shape)}")
    params: list[Parameter] = list(parameters)
    expected_total: int = sum(int(p._impl.numel()) for p in params)
    if int(vec._impl.numel()) != expected_total:
        raise ValueError(
            f"vec length {int(vec._impl.numel())} does not match total parameter "
            f"size {expected_total}"
        )
    offset: int = 0
    for p in params:
        n: int = int(p._impl.numel())
        slice_view: Tensor = vec[offset : offset + n].reshape(p.shape)
        p._impl.copy_from(slice_view._impl)
        offset += n
