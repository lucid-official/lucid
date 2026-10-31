"""``lucid.func`` — functional transforms (reference-framework func-compatible API).

Provides composable, higher-order transformations over Lucid functions:

* :func:`vmap`           — map a function of one example over a batch dimension
* :func:`grad`           — gradient transform (returns a function)
* :func:`grad_and_value` — gradient + primal output
* :func:`vjp`            — vector-Jacobian product (functional style)
* :func:`jvp`            — Jacobian-vector product (exact via double-backward)
* :func:`jacrev`         — reverse-mode Jacobian
* :func:`jacfwd`         — forward-mode Jacobian via jvp
* :func:`hessian`        — second-order derivative matrix
* :func:`linearize`      — linearisation: (primals_out, tangent_fn)

All transforms compose: ``vmap(grad(fn))``, ``grad(vmap(fn))``,
``grad(grad(fn))``, ``jacfwd(jacrev(fn))``, ``vmap(grad(grad(fn)))``.  A
derivative taken inside another transform's function stays attached to the
graph the outer transform differentiates, to any depth.

Pytrees
-------
Inputs and outputs may be nested tuples, lists, dicts and namedtuples of
tensors, not just tensors.  A transform differentiates or batches every
tensor leaf and returns its result in the structure of what it came from:
``grad(f)({"w": w, "b": b})`` is a dict with the keys ``"w"`` and ``"b"``,
``jacrev`` of a ``func`` returning a tuple is a tuple of Jacobians, and
``vmap``'s ``in_dims`` / ``out_dims`` may be a pytree matching (a prefix
of) the arguments / results.

How vmap runs the function
--------------------------
By default (``strategy="auto"``) ``vmap`` calls the function once per batch
element, on that element's logical shapes, and stacks the results.  A
reduction without a ``dim``, a loss, a shape query or a transform inside the
function therefore sees one example, as written.  ``strategy="vectorized"``
instead moves the batch axis to the front and calls the function **once** on
the whole batch, so the C++ engine ops run on the complete batch: one Metal
dispatch on the GPU, batched BLAS calls on the CPU.  A function that is
itself a derivative transform, or that draws independent random numbers
(``randomness="different"``), is still run element by element.

Limitations
-----------
* With ``strategy="vectorized"`` a reduction inside the function must use an
  explicit ``dim``: an unqualified ``.sum()`` / ``.mean()`` reduces over the
  batch dimension as well.
* In-place ops inside a vmapped function are not supported.
* ``randomness='error'`` (default) raises if the function calls any random op;
  ``'different'`` gives every batch element its own draws and ``'same'`` the
  one draw all of them share.
"""

from collections.abc import Callable
from typing import Literal, overload

from lucid._tensor.tensor import Tensor as _Tensor
from lucid.func._batching import Vmap as _Vmap
from lucid.func._forward import Jacfwd as _Jacfwd, jvp_at as _jvp_at, linearize_at as _linearize_at
from lucid.func._reverse import (
    Grad as _Grad,
    GradAndValue as _GradAndValue,
    Hessian as _Hessian,
    Jacrev as _Jacrev,
    vjp_at as _vjp_at,
)
from lucid.func._transform import Dims as _Dims, PyTree as _PyTree

__all__ = [
    "vmap",
    "grad",
    "grad_and_value",
    "vjp",
    "jvp",
    "jacrev",
    "jacfwd",
    "hessian",
    "linearize",
]


# ── vmap ─────────────────────────────────────────────────────────────────────


def vmap(
    func: Callable[..., _PyTree],
    in_dims: _Dims = 0,
    out_dims: _Dims = 0,
    randomness: str = "error",
    *,
    chunk_size: int | None = None,
    strategy: str = "auto",
) -> Callable[..., _PyTree]:
    r"""Vectorise ``func`` over a batch axis to produce a batched function.

    Lifts a function of a single example into one of a whole batch.  If
    :math:`f : \mathbb{R}^d \to \mathbb{R}^e`, ``vmap(f)`` realises
    :math:`F : \mathbb{R}^{B \times d} \to \mathbb{R}^{B \times e}` with
    ``F(x)[b] = f(x[b])``.  It composes with :func:`grad`, :func:`jacrev`,
    :func:`jvp` and :func:`vjp` to yield per-sample gradients, batched
    Jacobians and higher-order constructs.

    Parameters
    ----------
    func : Callable
        Function to vectorise. Must accept and return pytrees of tensors:
        tensors, or tuples, lists, dicts and namedtuples nesting them.
    in_dims : int, None or pytree of (int or None), optional
        Batch axis for each input. An ``int`` applies to every tensor of
        every positional argument; a tuple gives per-argument control, and
        inside an argument a dict or list may give per-leaf control — any
        pytree that is a prefix of the arguments, whose leaf stands for the
        whole subtree under it. Use ``None`` to hand an argument to
        ``func`` unchanged. A tuple has one entry per positional argument.
        Default ``0``.
    out_dims : int, None or pytree of (int or None), optional
        Where the batch axis appears in the output(s): an ``int`` for every
        tensor, or a pytree that is a prefix of ``func``'s output.  ``None``
        marks an output that does not depend on the batch: it comes back
        as ``func`` returned it, and a tensor that differs between batch
        elements is an error.  Default ``0``.
    randomness : {"error", "different", "same"}, optional
        ``"error"`` (default) forbids random ops inside ``func``;
        ``"different"`` gives every batch element its own draws; ``"same"``
        gives every element the draw the first one gets, and leaves the
        generator where a single call would.
    chunk_size : int, optional
        If set, process the batch in chunks of this size to cap peak
        memory. Applies in both vectorised and isolated strategies; it
        must be positive.
    strategy : str, optional
        ``"auto"`` (default) and ``"isolated"`` evaluate ``func`` per
        batch element so ``func`` always observes logical per-sample
        shapes, reductions, and differentiation transforms.
        ``"vectorized"`` moves the batch axis to the front and calls
        ``func`` once on the physical batched tensor when ``func`` is
        not marked for isolation and ``randomness`` is not
        ``"different"`` (one call would share a single draw).

    Returns
    -------
    Callable
        ``func``'s output structure with every tensor batched; for a batch
        of no elements, the shapes ``func`` makes for one example with an
        axis of length 0 where the batch goes.

    Raises
    ------
    ValueError
        ``in_dims`` / ``out_dims`` do not fit the arguments / results; a
        batched argument is not a tensor; the batch sizes disagree; a
        result with an ``out_dims`` axis is not a tensor; ``chunk_size``
        is not positive; ``strategy`` is unknown.
    RuntimeError
        ``randomness`` is none of the three modes.

    Notes
    -----
    In ``"auto"`` and ``"isolated"`` modes each batch element is sliced
    along its ``in_dims`` axis before ``func`` is called, so unqualified
    reductions, losses, shape queries and nested transforms
    (:func:`grad`, :func:`jacrev`, :func:`vjp`, :func:`jvp`) see the
    logical per-sample tensor.  In-place ops in ``"vectorized"`` mode are
    unsupported.

    Examples
    --------
    Per-sample gradients:

    >>> import lucid
    >>> from lucid.func import grad, vmap
    >>> f = lambda x: (x ** 2).sum()
    >>> X = lucid.randn(32, 4)
    >>> per_sample = vmap(grad(f))(X)  # shape (32, 4)

    Pytree arguments are batched leaf by leaf:

    >>> g = lambda d: d["x"].sum() + d["y"].sum()
    >>> vmap(g)({"x": lucid.ones(3, 2), "y": lucid.ones(3, 4)})
    tensor([6., 6., 6.])
    """
    return _Vmap(func, in_dims, out_dims, randomness, chunk_size, strategy)


# ── grad & grad_and_value ────────────────────────────────────────────────────


@overload
def grad[**P](
    func: Callable[P, object], argnums: int | tuple[int, ...] = 0, has_aux: Literal[False] = False
) -> Callable[P, _PyTree]: ...
@overload
def grad[**P](
    func: Callable[P, object], argnums: int | tuple[int, ...] = 0, *, has_aux: Literal[True]
) -> Callable[P, tuple[_PyTree, object]]: ...
@overload
def grad[**P](
    func: Callable[P, object], argnums: int | tuple[int, ...], has_aux: bool
) -> Callable[P, _PyTree | tuple[_PyTree, object]]: ...
def grad[**P](
    func: Callable[P, object], argnums: int | tuple[int, ...] = 0, has_aux: bool = False
) -> Callable[P, _PyTree | tuple[_PyTree, object]]:
    r"""Build a function returning the gradient of ``func``.

    The cornerstone of functional-style autograd: rather than calling
    ``.backward()`` and reading ``.grad``, ``grad(func)`` produces a new
    callable that, when invoked, returns the gradient tensor directly.
    Transforms compose, so ``grad(grad(func))`` yields a second
    derivative and ``vmap(grad(func))`` computes per-sample gradients.

    Parameters
    ----------
    func : Callable
        Function returning a **scalar** Tensor (or ``(scalar, aux)`` when
        ``has_aux=True``).
    argnums : int or tuple of int, optional
        Positional argument index/indices to differentiate. Each selected
        argument may be a tensor or a pytree of tensors. Default ``0``.
    has_aux : bool, optional
        If ``True``, ``func`` must return ``(loss, aux)``; the wrapped
        callable then returns ``(grads, aux)`` with ``aux`` forwarded
        through without differentiation. Default ``False``.

    Returns
    -------
    Callable
        Function with the same signature as ``func`` returning the
        gradient in the structure of the differentiated argument — a
        tensor for a tensor, a dict for a dict — or a tuple of them when
        ``argnums`` is a tuple. A tensor the loss does not depend on has
        gradient ``None``.

    Notes
    -----
    For :math:`f : \mathbb{R}^n \to \mathbb{R}`, ``grad(f)`` realises

    .. math::

        \nabla f : \mathbb{R}^n \to \mathbb{R}^n,
        \quad (\nabla f)_i (x) = \frac{\partial f}{\partial x_i}(x).

    Implementation is reverse-mode AD: one forward + one backward pass
    through ``func``, independent of input dimensionality.

    The gradient is itself differentiable when it needs to be: whenever an
    argument requires grad, or ``grad`` is called inside the function of
    another transform, the backward pass records its own graph. That is
    what makes ``grad(grad(f))`` the second derivative. Otherwise the
    graph is dropped as soon as the gradient is computed. A gradient taken
    at plain arguments through a tensor that ``func`` merely captures
    (a module's parameters, say) is returned detached from it — pass such
    a tensor in as an argument to differentiate through it.

    Examples
    --------
    >>> import lucid
    >>> from lucid.func import grad
    >>> f = lambda x: (x ** 3).sum()
    >>> df = grad(f)
    >>> df(lucid.tensor([1.0, 2.0, 3.0]))  # 3 * x ** 2
    tensor([3., 12., 27.])

    Second derivative, and a dict of parameters:

    >>> grad(grad(lambda a: a ** 3))(lucid.tensor(2.0))
    tensor(12.)
    >>> grad(lambda p: (p["w"] ** 2).sum())({"w": lucid.tensor([1.0, 2.0])})
    {'w': tensor([2., 4.])}
    """
    return _Grad(func, argnums, has_aux)


@overload
def grad_and_value[**P](
    func: Callable[P, object], argnums: int | tuple[int, ...] = 0, has_aux: Literal[False] = False
) -> Callable[P, tuple[_PyTree, _Tensor]]: ...
@overload
def grad_and_value[**P](
    func: Callable[P, object], argnums: int | tuple[int, ...] = 0, *, has_aux: Literal[True]
) -> Callable[P, tuple[_PyTree, tuple[_Tensor, object]]]: ...
@overload
def grad_and_value[**P](
    func: Callable[P, object], argnums: int | tuple[int, ...], has_aux: bool
) -> Callable[P, tuple[_PyTree, _Tensor | tuple[_Tensor, object]]]: ...
def grad_and_value[**P](
    func: Callable[P, object], argnums: int | tuple[int, ...] = 0, has_aux: bool = False
) -> Callable[P, tuple[_PyTree, _Tensor | tuple[_Tensor, object]]]:
    r"""Return a function computing both the gradient and the primal value.

    Equivalent to combining :func:`grad` with an evaluation of ``func``,
    but evaluates the forward pass exactly once. This is the canonical
    pattern for training loops where the loss value is also needed for
    logging or learning-rate scheduling.

    Parameters
    ----------
    func : Callable
        Function returning a scalar Tensor (or ``(scalar, aux)`` when
        ``has_aux=True``).
    argnums : int or tuple of int, optional
        Positional argument index/indices to differentiate. Default ``0``.
    has_aux : bool, optional
        If ``True``, ``func`` must return ``(loss, aux)`` and the wrapped
        callable returns ``(grads, (loss, aux))``. Default ``False``.

    Returns
    -------
    Callable
        Function returning ``(grads, value)`` — or ``(grads, (value, aux))``
        when ``has_aux=True``.

    Notes
    -----
    Mathematically computes both :math:`f(x)` and :math:`\nabla f(x)` in
    a single forward + backward sweep, saving redundant work compared to
    calling ``func`` and ``grad(func)`` separately.

    The value comes back detached, like the gradients: the backward pass
    that produced them has already consumed the graph behind it, so a
    value that still claimed ``requires_grad`` would only fail later, on
    a second ``backward``. Where :func:`grad` keeps the graph — an
    argument requires grad, or this is called inside another transform —
    the value keeps it too, so an outer transform can differentiate it.

    Examples
    --------
    >>> import lucid
    >>> from lucid.func import grad_and_value
    >>> f = lambda x: (x ** 2).sum()
    >>> gv = grad_and_value(f)
    >>> grads, value = gv(lucid.tensor([1.0, 2.0, 3.0]))
    >>> value  # 14.0
    tensor(14.)
    """
    return _GradAndValue(func, argnums, has_aux)


# ── vjp ──────────────────────────────────────────────────────────────────────


@overload
def vjp(
    func: Callable[..., object], *primals: _PyTree, has_aux: Literal[False] = False
) -> tuple[_PyTree, Callable[..., _PyTree]]: ...
@overload
def vjp(
    func: Callable[..., object], *primals: _PyTree, has_aux: Literal[True]
) -> tuple[tuple[_PyTree, object], Callable[..., _PyTree]]: ...
def vjp(
    func: Callable[..., object], *primals: _PyTree, has_aux: bool = False
) -> tuple[_PyTree | tuple[_PyTree, object], Callable[..., _PyTree]]:
    r"""Compute the vector-Jacobian product of ``func`` at ``primals``.

    Evaluates ``func`` at the supplied primal inputs and returns both the
    output and a callable ``vjp_fn`` that, given cotangent vectors
    :math:`v`, returns :math:`v^\top J(\mathrm{primals})` — the
    backward-mode contraction of the Jacobian against ``v``. This is the
    workhorse used internally by :func:`grad` and :func:`jacrev`, and is
    useful directly when many backward passes are needed against the same
    forward computation.

    Parameters
    ----------
    func : Callable
        Differentiable function taking the primals as positional inputs;
        it returns a tensor or a pytree of tensors.
    *primals : Tensor or pytree of Tensor
        Points at which to linearise ``func``.
    has_aux : bool, optional
        If ``True``, ``func`` must return ``(output, aux)``; the call
        then yields ``((output, aux), vjp_fn)``. Default ``False``.

    Returns
    -------
    tuple
        ``(outputs, vjp_fn)`` with ``outputs`` in ``func``'s output
        structure. ``vjp_fn(cotangents)`` takes a cotangent for every
        output tensor, in the structure of the output, and returns a
        tuple with one entry per primal, each in that primal's structure
        (``None`` for a tensor the outputs do not depend on). For
        convenience the cotangents may also be given as separate positional
        arguments, one per output tensor, and left out for a scalar output.

    Notes
    -----
    For :math:`f : \mathbb{R}^n \to \mathbb{R}^m` and cotangent
    :math:`v \in \mathbb{R}^m`,

    .. math::

        \mathrm{vjp\_fn}(v) = v^\top J_f(x), \quad
        J_f(x) \in \mathbb{R}^{m \times n}.

    Cost is one backward pass per call to ``vjp_fn`` — the forward
    graph is retained so multiple cotangents can be applied cheaply.

    Examples
    --------
    >>> import lucid
    >>> from lucid.func import vjp
    >>> f = lambda x: x ** 2
    >>> x = lucid.tensor([1.0, 2.0, 3.0])
    >>> y, vjp_fn = vjp(f, x)
    >>> (grads,) = vjp_fn(lucid.ones_like(y))  # 2 * x
    """
    return _vjp_at(func, primals, has_aux)


# ── jvp ──────────────────────────────────────────────────────────────────────


def jvp(
    func: Callable[..., _PyTree],
    primals: tuple[_PyTree, ...],
    tangents: tuple[_PyTree, ...],
    strict: bool = False,
) -> tuple[_PyTree, _PyTree]:
    r"""Compute the Jacobian-vector product of ``func`` at ``primals``.

    Returns both the primal output ``func(*primals)`` and the directional
    derivative :math:`J(\mathrm{primals}) \cdot \mathrm{tangents}` — the
    forward-mode contraction of the Jacobian against a tangent vector.
    Forward-mode is preferred over reverse-mode when the input dimension
    is small relative to the output dimension.

    Parameters
    ----------
    func : Callable
        Differentiable function returning a Tensor or a pytree of Tensors.
    primals : tuple of (Tensor or pytree of Tensor)
        Points at which to evaluate ``func`` and its Jacobian. A leaf that
        is not a tensor is held fixed.
    tangents : tuple of (Tensor or pytree of Tensor)
        Tangent vectors, one per primal, in the same structure as
        ``primals`` and each tensor matching the shape of the one it
        belongs to.
    strict : bool, optional
        If ``True``, refuse what :func:`lucid.autograd.jvp` refuses under
        ``strict`` — an output independent of the inputs among them —
        instead of giving it a zero tangent. Default ``False``.

    Returns
    -------
    tuple
        ``(primals_out, tangents_out)``, each with the same nested structure
        as ``func(*primals)``.

    Raises
    ------
    RuntimeError
        ``primals`` and ``tangents`` differ in length, or ``strict``
        refuses a result.
    ValueError
        ``tangents`` is not in the structure of ``primals``.
    NotImplementedError
        A primal or a tangent requires grad, so the tangent has to stay
        differentiable, and an op on the path has no graph-mode backward.

    Notes
    -----
    :math:`J v` is exact, not a finite difference: it is computed by
    :func:`lucid.autograd.jvp` in two reverse passes (the double-VJP
    trick), or — through an op whose backward has no graph mode, such as
    ``LSTM`` or a padded ``avg_pool`` — in one reverse pass per output
    element.

    The results join the caller's autograd graph as the reference's do.
    Under ``no_grad`` neither does.  Otherwise the primal output is attached
    when it depends on a tensor that requires grad — a primal, or a
    parameter ``func`` captures — and the tangent is attached when a
    primal, a tangent or such a parameter requires grad.  Through an op
    with no graph-mode backward the tangent cannot be attached: when only
    captured parameters would attach it, it comes back detached with a
    warning.

    ``func`` runs once when grad mode is off or a primal requires grad.
    With grad mode on and no primal requiring grad it runs **twice**: once
    at the primals, which is the only way to learn whether the output
    reaches the caller's graph through tensors ``func`` captures, and once
    for the tangent.  The second run replays the first one's random draws,
    so a dropout mask is the same in both, but any other side effect
    happens twice — a ``BatchNorm`` in training mode updates its running
    statistics twice, and an in-place buffer update is applied twice.

    Examples
    --------
    >>> import lucid
    >>> from lucid.func import jvp
    >>> f = lambda x: x ** 2
    >>> x = lucid.tensor([1.0, 2.0, 3.0])
    >>> t = lucid.ones_like(x)
    >>> y, dy = jvp(f, (x,), (t,))
    >>> dy  # 2 * x
    tensor([2., 4., 6.])
    """
    return _jvp_at(func, tuple(primals), tuple(tangents), strict)


# ── linearize ────────────────────────────────────────────────────────────────


def linearize(
    func: Callable[..., _PyTree], *primals: _PyTree
) -> tuple[_PyTree, Callable[..., _PyTree]]:
    r"""Linearise ``func`` around ``primals`` for reuse across many tangents.

    Returns the primal output together with a callable ``linear_fn`` that
    applies the first-order Taylor expansion of ``func`` at ``primals``.
    Mathematically ``linear_fn(t)`` evaluates :math:`J_f(\mathrm{primals})
    \, t` — the same quantity as :func:`jvp` — but the linearisation cost
    is paid once even if many tangent vectors are queried subsequently.

    Parameters
    ----------
    func : Callable
        Differentiable function returning a Tensor or a pytree of Tensors.
    *primals : Tensor or pytree of Tensor
        Points at which to linearise ``func``.

    Returns
    -------
    tuple
        ``(primals_out, linear_fn)``. Calling ``linear_fn(*tangents)``
        returns the JVP at ``primals`` against ``tangents``.

    Notes
    -----
    Conceptually equivalent to a Taylor expansion truncated at first
    order:

    .. math::

        f(x + t) \approx f(x) + J_f(x) \, t.

    ``func`` runs once here and once per call of ``linear_fn``.  Whether
    the results join the caller's autograd graph follows :func:`jvp`, and
    is decided here, from the primal output, once.  The returned
    ``linear_fn`` is flagged for vmap isolation so that composing
    ``vmap(linear_fn)`` slices tangents one at a time.

    Examples
    --------
    >>> import lucid
    >>> from lucid.func import linearize
    >>> f = lambda x: x ** 2
    >>> x = lucid.tensor([1.0, 2.0, 3.0])
    >>> y, lin = linearize(f, x)
    >>> lin(lucid.ones_like(x))  # 2 * x
    tensor([2., 4., 6.])
    """
    return _linearize_at(func, primals)


# ── jacrev & jacfwd ──────────────────────────────────────────────────────────


@overload
def jacrev[**P](
    func: Callable[P, object],
    argnums: int | tuple[int, ...] = 0,
    has_aux: Literal[False] = False,
    *,
    chunk_size: int | None = None,
) -> Callable[P, _PyTree]: ...
@overload
def jacrev[**P](
    func: Callable[P, object],
    argnums: int | tuple[int, ...] = 0,
    *,
    has_aux: Literal[True],
    chunk_size: int | None = None,
) -> Callable[P, tuple[_PyTree, object]]: ...
@overload
def jacrev[**P](
    func: Callable[P, object],
    argnums: int | tuple[int, ...],
    has_aux: bool,
    *,
    chunk_size: int | None = None,
) -> Callable[P, _PyTree | tuple[_PyTree, object]]: ...
def jacrev[**P](
    func: Callable[P, object],
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    *,
    chunk_size: int | None = None,
) -> Callable[P, _PyTree | tuple[_PyTree, object]]:
    r"""Build a function returning the reverse-mode Jacobian of ``func``.

    Assembles the full Jacobian matrix one row at a time, each row
    obtained from a backward pass seeded with a one-hot cotangent. For a
    scalar-output ``func`` this collapses to :func:`grad`.

    Parameters
    ----------
    func : Callable
        Differentiable function returning a Tensor or a pytree of Tensors
        (or ``(output, aux)`` when ``has_aux=True``).
    argnums : int or tuple of int, optional
        Argument(s) to differentiate with respect to; each may be a
        tensor or a pytree of tensors. Default ``0``.
    has_aux : bool, optional
        Whether ``func`` returns auxiliary data. Default ``False``.
    chunk_size : int, optional
        Reserved for future vmap-batched row computation.

    Returns
    -------
    Callable
        Function returning the Jacobian. For each output tensor of
        ``func`` the Jacobian block has shape ``output.shape + arg.shape``
        for each differentiated tensor; the result has ``func``'s output
        structure, with the structure of the differentiated argument (a
        tuple, for a tuple ``argnums``) at each leaf.

    Notes
    -----
    For :math:`f : \mathbb{R}^n \to \mathbb{R}^m` produces

    .. math::

        J_{ij} = \frac{\partial f_i}{\partial x_j}, \quad
        J \in \mathbb{R}^{m \times n}.

    Cost scales with the output dimension :math:`m` (one backward pass
    per row). Prefer :func:`jacfwd` when :math:`n < m`.  Like :func:`grad`
    it keeps the graph of the Jacobian when an argument requires grad or
    when it is called inside another transform, so ``jacrev(jacrev(f))``
    is the Hessian.

    Examples
    --------
    >>> import lucid
    >>> from lucid.func import jacrev
    >>> f = lambda x: lucid.stack([x.sum(), (x ** 2).sum()])
    >>> jacrev(f)(lucid.tensor([1.0, 2.0, 3.0]))  # shape (2, 3)
    tensor([[1., 1., 1.], [2., 4., 6.]])
    """
    return _Jacrev(func, argnums, has_aux, chunk_size)


@overload
def jacfwd[**P](
    func: Callable[P, object],
    argnums: int | tuple[int, ...] = 0,
    has_aux: Literal[False] = False,
    *,
    randomness: str = "error",
) -> Callable[P, _PyTree]: ...
@overload
def jacfwd[**P](
    func: Callable[P, object],
    argnums: int | tuple[int, ...] = 0,
    *,
    has_aux: Literal[True],
    randomness: str = "error",
) -> Callable[P, tuple[_PyTree, object]]: ...
@overload
def jacfwd[**P](
    func: Callable[P, object],
    argnums: int | tuple[int, ...],
    has_aux: bool,
    *,
    randomness: str = "error",
) -> Callable[P, _PyTree | tuple[_PyTree, object]]: ...
def jacfwd[**P](
    func: Callable[P, object],
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    *,
    randomness: str = "error",
) -> Callable[P, _PyTree | tuple[_PyTree, object]]:
    r"""Build a function returning the forward-mode Jacobian of ``func``.

    Materialises the Jacobian one column at a time by repeatedly calling
    :func:`jvp` with one-hot tangent vectors. Each column corresponds to
    the partial derivative of every output with respect to a single input
    coordinate.

    Parameters
    ----------
    func : Callable
        Differentiable function returning a Tensor or a pytree of Tensors
        (or ``(output, aux)`` when ``has_aux=True``).
    argnums : int or tuple of int, optional
        Argument(s) to differentiate with respect to; each may be a
        tensor or a pytree of tensors. Default ``0``.
    has_aux : bool, optional
        Whether ``func`` returns auxiliary data. Default ``False``.
    randomness : str, optional
        Randomness policy forwarded to internal vmap calls. Default
        ``"error"``.

    Returns
    -------
    Callable
        Function returning the Jacobian, in the structure :func:`jacrev`
        gives it: for each output tensor of ``func`` a block of shape
        ``output.shape + arg.shape`` per differentiated tensor.

    Notes
    -----
    For :math:`f : \mathbb{R}^n \to \mathbb{R}^m` produces

    .. math::

        J_{ij} = \frac{\partial f_i}{\partial x_j}, \quad
        J \in \mathbb{R}^{m \times n}.

    Cost scales with the input dimension :math:`n` (one forward / JVP per
    column, and one forward at the arguments themselves). Prefer this
    transform over :func:`jacrev` when :math:`n \ll m`.  Whether the
    Jacobian joins the caller's autograd graph follows :func:`jvp`, and is
    decided once, from that first forward.

    Examples
    --------
    >>> import lucid
    >>> from lucid.func import jacfwd
    >>> f = lambda x: lucid.stack([x.sum(), (x ** 2).sum()])
    >>> jacfwd(f)(lucid.tensor([1.0, 2.0, 3.0]))  # shape (2, 3)
    tensor([[1., 1., 1.], [2., 4., 6.]])
    """
    return _Jacfwd(func, argnums, has_aux, randomness)


# ── hessian ───────────────────────────────────────────────────────────────────


def hessian[**P](
    func: Callable[P, _Tensor], argnums: int | tuple[int, ...] = 0
) -> Callable[P, _PyTree]:
    r"""Build a function returning the Hessian of a scalar-valued ``func``.

    Computes the matrix of second partial derivatives — the Jacobian of
    the gradient — which is the value ``jacfwd(jacrev(func))`` (or
    ``jacrev(jacrev(func))``) has. The result captures the local
    curvature used by Newton-style optimisers, natural-gradient methods,
    and second-order analyses such as eigenvalue spectra of the loss.

    Parameters
    ----------
    func : Callable
        Function returning a scalar Tensor.
    argnums : int or tuple of int, optional
        Argument(s) to differentiate twice; each may be a tensor or a
        pytree of tensors. Default ``0``.

    Returns
    -------
    Callable
        Function returning the Hessian tensor with shape
        ``arg.shape + arg.shape``; for a pytree argument the structure of
        the argument holding, at each leaf, the structure of the argument
        again (``H["a"]["b"]`` is the block of leaves ``a`` and ``b``).

    Notes
    -----
    For :math:`f : \mathbb{R}^n \to \mathbb{R}`,

    .. math::

        H_{ij} = \frac{\partial^2 f}{\partial x_i \, \partial x_j},
        \quad H \in \mathbb{R}^{n \times n}.

    Cost is dominated by :math:`n` reverse passes over the graph of the
    gradient, which is built with ``create_graph=True``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.func import hessian
    >>> f = lambda x: (x ** 3).sum()  # H = diag(6 * x)
    >>> hessian(f)(lucid.tensor([1.0, 2.0, 3.0]))
    tensor([[6., 0., 0.], [0., 12., 0.], [0., 0., 18.]])
    """
    return _Hessian(func, argnums)
