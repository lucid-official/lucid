"""``lucid.func`` — functional transforms (reference-framework func-compatible API).

Provides composable, higher-order transformations over Lucid functions:

* :func:`vmap`           — vectorise over a batch dimension (C++ vmapped dispatch)
* :func:`grad`           — gradient transform (returns a function)
* :func:`grad_and_value` — gradient + primal output
* :func:`vjp`            — vector-Jacobian product (functional style)
* :func:`jvp`            — Jacobian-vector product (exact via double-backward)
* :func:`jacrev`         — reverse-mode Jacobian
* :func:`jacfwd`         — forward-mode Jacobian via jvp
* :func:`hessian`        — second-order derivative matrix
* :func:`linearize`      — linearisation: (primals_out, tangent_fn)

All transforms compose: ``vmap(grad(fn))``, ``grad(vmap(fn))``, etc.

C++ vmapped dispatch
--------------------
``vmap`` does **not** loop over batch elements in Python.  It moves the batch
dimension to the front and calls the function **once** with the full batched
tensor.  The underlying C++ engine ops then run on the complete batch:

* **GPU (Metal/MLX)**: a single Metal dispatch handles all batch elements in
  parallel via MLX's native batched kernels.
* **CPU (Accelerate)**: BLAS calls (``SGEMM``, ``cblas_sgemm_batch``, etc.)
  operate on the full batched tensor with SIMD parallelism.

This gives real vectorisation — not a sequential Python loop.

Limitations
-----------
* Reduction ops inside the vmapped function must use an explicit ``dim``
  argument.  An unqualified ``.sum()`` / ``.mean()`` reduces over the batch
  dimension as well, which is almost never desired.
* In-place ops inside a vmapped function are not supported.
* ``randomness='error'`` (default) raises if the function calls any random op.
"""

import warnings
from typing import Callable, TYPE_CHECKING
from lucid._ctx import _RandomnessGuard as _RandGuard
from lucid.autograd._functional import jvp as _jvp_by_double_backward

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

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


# ── isolation marker ─────────────────────────────────────────────────────────

# Attribute set on functions returned by jacrev / jacfwd / hessian so that
# vmap knows to use per-element isolation instead of the move-batch-dim path.
# Those transforms materialise per-output backward passes that must not see
# the batch dimension as part of the input — Stage 1 would give wrong shapes
# like (B, out, B, in) instead of (B, out, in).
_ISOLATION_ATTR = "_lucid_needs_vmap_isolation"
_MIN_AUX_TUPLE_LEN = 2

#: One result per differentiated argument — bare for an int ``argnums``, a
#: tuple for a tuple — and ``None`` for one the output does not reach: what
#: :func:`grad` and :func:`jacfwd` return.
type _PerArgnum = Tensor | tuple[Tensor | None, ...] | None
#: A Jacobian packed over inputs and outputs as
#: :func:`lucid.autograd.jacobian` packs it: what :func:`jacrev` returns.
type _Jacobian = Tensor | tuple[Tensor | tuple[Tensor, ...], ...]


def _mark_isolation[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Tag *fn* so :func:`vmap` uses the per-element isolation path.

    Used by functional transforms (:func:`grad`, :func:`grad_and_value`,
    :func:`vjp`, :func:`linearize`, :func:`jacrev`, :func:`jacfwd`,
    :func:`hessian`, :func:`vmap`) whose execution must not observe an
    enclosing batch axis as part of the input.
    """
    setattr(fn, _ISOLATION_ATTR, True)
    return fn


# ── internal helpers ──────────────────────────────────────────────────────────


def _normalise_argnums(
    argnums: int | tuple[int, ...],
) -> tuple[bool, tuple[int, ...]]:
    if isinstance(argnums, int):
        return True, (argnums,)
    return False, tuple(argnums)


def _prepare_grad_args(args: tuple[object, ...], argnums: tuple[int, ...]) -> list[object]:
    from lucid._tensor.tensor import Tensor as _T

    args_list = list(args)
    for i in argnums:
        a = args_list[i]
        if isinstance(a, _T) and not a.requires_grad:
            args_list[i] = a.detach().requires_grad_(True)
    return args_list


def _tensor_argument(args: list[object], i: int, name: str) -> Tensor:
    """Argument ``i``, which ``name`` differentiates and so must be a tensor."""
    from lucid._tensor.tensor import Tensor as _T

    a = args[i]
    if not isinstance(a, _T):
        raise TypeError(
            f"lucid.func.{name}: argument {i} is differentiated, so it must be "
            f"a Tensor, got {type(a).__name__}"
        )
    return a


def _output_tensors(value: object, name: str) -> Tensor | tuple[Tensor, ...]:
    """``value`` when it is a tensor or a tuple (or list) of tensors — what a
    transform can differentiate; refused with ``TypeError`` otherwise."""
    from lucid._tensor.tensor import Tensor as _T

    if isinstance(value, _T):
        return value
    if isinstance(value, (tuple, list)) and all(isinstance(v, _T) for v in value):
        return tuple(item for item in value if isinstance(item, _T))
    raise TypeError(
        f"lucid.func.{name}: func must return a Tensor or a tuple of Tensors, "
        f"got {type(value).__name__}"
    )


def _output_tensor(value: object, name: str) -> Tensor:
    """``value`` when it is one tensor; refused with ``TypeError`` otherwise."""
    out = _output_tensors(value, name)
    if isinstance(out, tuple):
        raise TypeError(
            f"lucid.func.{name}: func must return a single Tensor, got a tuple " f"of {len(out)}"
        )
    return out


def _split_aux(
    output: object, *, has_aux: bool, name: str, label: str = "output"
) -> tuple[object, object]:
    """``(output, aux)`` from what ``func`` returned — ``aux`` is ``None``
    unless ``has_aux``, and then everything after the first item."""
    if not has_aux:
        return output, None
    if not isinstance(output, tuple) or len(output) < _MIN_AUX_TUPLE_LEN:
        raise ValueError(f"lucid.func.{name}: has_aux=True requires func to return ({label}, aux)")
    aux = output[1] if len(output) == _MIN_AUX_TUPLE_LEN else output[1:]
    return output[0], aux


def _requires_grad(out: Tensor | tuple[Tensor, ...]) -> bool:
    """Whether any tensor of ``out`` is attached to a graph."""
    return any(o.requires_grad for o in (out if isinstance(out, tuple) else (out,)))


def _attachment(primals: tuple[Tensor, ...], out: Tensor | tuple[Tensor, ...]) -> tuple[bool, bool]:
    """``(attach, required)`` for tangents at ``primals``, under the current
    grad mode, given ``func``'s output there evaluated in that mode.

    Required when a primal requires grad; attached then, and also when
    ``out`` reaches the caller's graph through tensors ``func`` captures.
    """
    import lucid

    if not lucid.is_grad_enabled():
        return False, False
    required = any(p.requires_grad for p in primals)
    return required or _requires_grad(out), required


def _replay_draws() -> Callable[[], None]:
    """A rewind of the default generator to where it stands now.

    Calling the rewind before ``func`` is evaluated again makes that
    evaluation draw exactly what the previous one drew — the same dropout
    mask — and leaves the stream where one evaluation leaves it, so a
    transform that has to run ``func`` twice still looks like one call.
    """
    from lucid._factories.random import get_rng_state, set_rng_state

    state = get_rng_state()

    def rewind() -> None:
        set_rng_state(state)

    return rewind


def _normalise_in_dims(
    in_dims: int | tuple[int | None, ...],
    n_args: int,
) -> list[int | None]:
    if isinstance(in_dims, int):
        return [in_dims] * n_args
    dims: list[int | None] = list(in_dims[:n_args])
    while len(dims) < n_args:
        dims.append(0)
    return dims


def _move_to_front(x: Tensor, dim: int) -> Tensor:
    import lucid

    ndim = len(x.shape)
    d = dim if dim >= 0 else ndim + dim
    if d == 0:
        return x
    return lucid.moveaxis(x, d, 0)


def _move_from_front(x: Tensor, dim: int) -> Tensor:
    import lucid

    if dim == 0:
        return x
    return lucid.moveaxis(x, 0, dim)


def _combine_outputs(
    items: list[Tensor | tuple[Tensor, ...]],
    out_dims: int | tuple[int, ...],
    combine: Callable[[list[Tensor], int], Tensor],
) -> Tensor | tuple[Tensor, ...]:
    from lucid._tensor.tensor import Tensor as _T

    first = items[0]
    if isinstance(first, tuple):
        n_o = len(first)
        ods: list[int] = [out_dims] * n_o if isinstance(out_dims, int) else list(out_dims)
        return tuple(
            combine(
                [r[i] for r in items if isinstance(r, tuple)],
                ods[i],
            )
            for i in range(n_o)
        )
    od_i: int = out_dims if isinstance(out_dims, int) else out_dims[0]
    return combine([t for t in items if isinstance(t, _T)], od_i)


def _validate_vmap_batch_size(args: tuple[object, ...], in_dims: list[int | None]) -> int | None:
    from lucid._tensor.tensor import Tensor as _T

    batch_size: int | None = None
    for a, d in zip(args, in_dims, strict=True):
        if d is None or not isinstance(a, _T):
            continue
        ndim = len(a.shape)
        bd = d if d >= 0 else ndim + d
        if bd < 0 or bd >= ndim:
            raise ValueError(f"vmap: in_dim {d} is out of range for a {ndim}-D tensor")
        bs = int(a.shape[bd])
        if batch_size is None:
            batch_size = bs
        elif batch_size != bs:
            raise ValueError(f"vmap: inconsistent batch sizes {batch_size} vs {bs}")
    return batch_size


def _vectorized_vmap_call(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    in_dims: list[int | None],
    out_dims: int | tuple[int, ...],
) -> Tensor | tuple[Tensor, ...]:
    from lucid._tensor.tensor import Tensor as _T

    moved = [
        a if d is None or not isinstance(a, _T) else _move_to_front(a, d)
        for a, d in zip(args, in_dims, strict=True)
    ]
    output = func(*moved, **kwargs)
    if isinstance(output, tuple):
        od_list = [out_dims] * len(output) if isinstance(out_dims, int) else list(out_dims)
        # A non-tensor item passes through, as it always has.
        return tuple(
            _move_from_front(o, od) if isinstance(o, _T) else o
            for o, od in zip(output, od_list, strict=True)
        )
    if isinstance(output, _T) and isinstance(out_dims, int) and out_dims != 0:
        return _move_from_front(output, out_dims)
    return output


# ── vmap ─────────────────────────────────────────────────────────────────────


def vmap(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    in_dims: int | tuple[int | None, ...] = 0,
    out_dims: int | tuple[int, ...] = 0,
    randomness: str = "error",
    *,
    chunk_size: int | None = None,
    strategy: str = "auto",
) -> Callable[..., Tensor | tuple[Tensor, ...]]:
    r"""Vectorise ``func`` over a batch axis to produce a batched function.

    Lifts a function operating on a single example into one operating on a
    whole batch in a single dispatch, without writing an explicit Python
    loop. Mathematically, if :math:`f : \mathbb{R}^d \to \mathbb{R}^e`, then
    ``vmap(f)`` realises :math:`F : \mathbb{R}^{B \times d} \to
    \mathbb{R}^{B \times e}` such that ``F(x)[b] = f(x[b])`` for every
    batch index :math:`b`. The transform composes with :func:`grad`,
    :func:`jacrev`, :func:`jvp`, and :func:`vjp` to yield per-sample
    gradients, batched Jacobians, and higher-order constructs.

    Parameters
    ----------
    func : Callable
        Function to vectorise. Must accept and return tensors (or tuples of
        tensors).
    in_dims : int or tuple of (int or None), optional
        Batch axis for each input. An ``int`` applies to all positional
        arguments; a tuple gives per-argument control. Use ``None`` to
        broadcast an argument unchanged across the batch. Default ``0``.
    out_dims : int or tuple of int, optional
        Where the batch axis appears in the output(s). Default ``0``.
    randomness : str, optional
        ``"error"`` (default) forbids random ops inside ``func``;
        ``"different"`` and ``"same"`` allow them with shared RNG state.
    chunk_size : int, optional
        If set, process the batch in chunks of this size to cap peak
        memory. Applies in both vectorised and isolated strategies.
    strategy : str, optional
        ``"auto"`` (default) and ``"isolated"`` evaluate ``func`` per
        batch element so ``func`` always observes logical per-sample
        shapes, reductions, and differentiation transforms.
        ``"vectorized"`` moves the batch axis to the front and calls
        ``func`` once on the physical batched tensor when ``func`` is
        not marked for isolation.

    Returns
    -------
    Callable
        A new function with the batched semantics described above.

    Notes
    -----
    In ``"auto"`` (default) and ``"isolated"`` modes, each batch element
    is sliced along its ``in_dims`` axis before calling ``func``, so
    unqualified reductions (``.sum()``, ``.mean()``), loss functions,
    shape queries, and nested transforms (:func:`grad`, :func:`jacrev`,
    :func:`vjp`, :func:`jvp`) operate on the logical per-sample tensor.
    In ``"vectorized"`` mode, the batch axis is moved to the front for a
    single engine dispatch across the full batch. In-place ops inside the
    vectorised function are unsupported.

    Examples
    --------
    Per-sample gradients:

    >>> import lucid
    >>> from lucid.func import grad, vmap
    >>> f = lambda x: (x ** 2).sum()
    >>> X = lucid.randn(32, 4)
    >>> per_sample = vmap(grad(f))(X)  # shape (32, 4)
    """
    if strategy not in ("auto", "isolated", "vectorized"):
        raise ValueError(
            f"vmap: strategy must be 'auto', 'isolated', or 'vectorized'; " f"got {strategy!r}"
        )

    def vectorized(
        *args: Tensor,
        **kwargs: object,
    ) -> Tensor | tuple[Tensor, ...]:
        """Run ``func`` across the batched axis of each input.

        Inputs whose ``in_dims`` entry is not ``None`` must share the
        same batch length on the indicated axis. The unwrapped outputs
        are stacked along ``out_dims`` to form the vectorised result.
        """
        _in = _normalise_in_dims(in_dims, len(args))
        batch_size = _validate_vmap_batch_size(args, _in)

        with _RandGuard(randomness):
            if batch_size is None:
                return func(*args, **kwargs)
            isolate = batch_size > 0 and (
                strategy != "vectorized" or getattr(func, _ISOLATION_ATTR, False)
            )
            if isolate:
                return _isolated_vmap(func, args, kwargs, _in, out_dims, batch_size, chunk_size)
            if chunk_size is not None and batch_size > chunk_size:
                return _chunked_vmap(func, args, kwargs, _in, out_dims, chunk_size, batch_size)
            return _vectorized_vmap_call(func, args, kwargs, _in, out_dims)

    return _mark_isolation(vectorized)


def _chunked_vmap(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    args: tuple[Tensor, ...],
    kwargs: dict[str, object],
    in_dims: list[int | None],
    out_dims: int | tuple[int, ...],
    chunk_size: int,
    batch_size: int,
) -> Tensor | tuple[Tensor, ...]:
    import lucid
    from lucid._tensor.tensor import Tensor as _T

    chunks: list[Tensor | tuple[Tensor, ...]] = []
    for start in range(0, batch_size, chunk_size):
        end = min(start + chunk_size, batch_size)
        cargs: list[object] = []
        for a, d in zip(args, in_dims, strict=True):
            if d is None or not isinstance(a, _T):
                cargs.append(a)
            else:
                ndim = len(a.shape)
                bd = d if d >= 0 else ndim + d
                sl: list[slice | int | None] = [slice(None)] * ndim
                sl[bd] = slice(start, end)
                cargs.append(a[tuple(sl)])
        chunks.append(func(*cargs, **kwargs))

    return _combine_outputs(chunks, out_dims, lambda ts, d: lucid.cat(ts, dim=d))


def _isolated_vmap(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    args: tuple[Tensor, ...],
    kwargs: dict[str, object],
    in_dims: list[int | None],
    out_dims: int | tuple[int, ...],
    batch_size: int,
    chunk_size: int | None = None,
) -> Tensor | tuple[Tensor, ...]:
    """Per-element isolation: call func(x[b]) for b in 0..batch_size-1.

    Each element is sliced out independently before calling func so that
    func observes logical per-sample shapes and transforms cannot observe
    the batch axis as part of the input.

    When *chunk_size* is given, results are partially stacked after each chunk
    to free intermediate autograd-graph memory before the next chunk starts.
    """
    import lucid
    from lucid._tensor.tensor import Tensor as _T

    def _run_element(b: int) -> Tensor | tuple[Tensor, ...]:
        sliced: list[object] = []
        for a, d in zip(args, in_dims, strict=True):
            if d is None or not isinstance(a, _T):
                sliced.append(a)
            else:
                nd = len(a.shape)
                bd = d if d >= 0 else nd + d
                idx: list[int | slice] = [slice(None)] * nd
                idx[bd] = b
                sliced.append(a[tuple(idx)])
        return func(*sliced, **kwargs)

    if chunk_size is not None and batch_size > chunk_size:
        parts: list[Tensor | tuple[Tensor, ...]] = []
        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            chunk_res = [_run_element(b) for b in range(start, end)]
            parts.append(
                _combine_outputs(chunk_res, out_dims, lambda ts, d: lucid.stack(ts, dim=d))
            )
        return _combine_outputs(parts, out_dims, lambda ts, d: lucid.cat(ts, dim=d))

    results = [_run_element(b) for b in range(batch_size)]
    return _combine_outputs(results, out_dims, lambda ts, d: lucid.stack(ts, dim=d))


# ── grad & grad_and_value ────────────────────────────────────────────────────


def _eval_grad_and_value(
    func: Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    argnums: tuple[int, ...],
    *,
    has_aux: bool,
    name: str,
) -> tuple[_PerArgnum, Tensor, object]:
    from lucid.autograd._backward import grad as _ag
    from lucid.autograd._grad_mode import enable_grad

    args_list = _prepare_grad_args(args, argnums)
    with enable_grad():
        output = func(*args_list, **kwargs)
    first, aux = _split_aux(output, has_aux=has_aux, name=name, label="loss")
    loss = _output_tensor(first, name)
    grads = _ag(
        [loss],
        [_tensor_argument(args_list, i, name) for i in argnums],
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    g = grads[0] if len(argnums) == 1 else tuple(grads)
    return g, loss.detach(), aux


def grad[**P](
    func: Callable[P, Tensor | tuple[object, ...]],
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
) -> Callable[P, _PerArgnum | tuple[_PerArgnum, object]]:
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
        Positional argument index/indices to differentiate. Default ``0``.
    has_aux : bool, optional
        If ``True``, ``func`` must return ``(loss, aux)``; the wrapped
        callable then returns ``(grads, aux)`` with ``aux`` forwarded
        through without differentiation. Default ``False``.

    Returns
    -------
    Callable
        Function with the same signature as ``func`` returning the
        gradient tensor — or a tuple of gradients when ``argnums`` is a
        tuple.

    Notes
    -----
    For :math:`f : \mathbb{R}^n \to \mathbb{R}`, ``grad(f)`` realises

    .. math::

        \nabla f : \mathbb{R}^n \to \mathbb{R}^n,
        \quad (\nabla f)_i (x) = \frac{\partial f}{\partial x_i}(x).

    Implementation is reverse-mode AD: one forward + one backward pass
    through ``func``, independent of input dimensionality.

    Examples
    --------
    >>> import lucid
    >>> from lucid.func import grad
    >>> f = lambda x: (x ** 3).sum()
    >>> df = grad(f)
    >>> df(lucid.tensor([1.0, 2.0, 3.0]))  # 3 * x ** 2
    tensor([3., 12., 27.])
    """
    _, _argnums = _normalise_argnums(argnums)

    def grad_fn(*args: P.args, **kwargs: P.kwargs) -> _PerArgnum | tuple[_PerArgnum, object]:
        """Closure that computes the gradient via reverse-mode autograd."""
        g, _, aux = _eval_grad_and_value(func, args, kwargs, _argnums, has_aux=has_aux, name="grad")
        return (g, aux) if has_aux else g

    return _mark_isolation(grad_fn)


def grad_and_value[**P](
    func: Callable[P, Tensor | tuple[object, ...]],
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
) -> Callable[P, tuple[_PerArgnum, Tensor | tuple[Tensor, object]]]:
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
    a second ``backward``.

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
    _, _argnums = _normalise_argnums(argnums)

    def gv_fn(
        *args: P.args, **kwargs: P.kwargs
    ) -> tuple[_PerArgnum, Tensor | tuple[Tensor, object]]:
        """Closure that computes the gradient and primal value."""
        g, val, aux = _eval_grad_and_value(
            func, args, kwargs, _argnums, has_aux=has_aux, name="grad_and_value"
        )
        return (g, (val, aux)) if has_aux else (g, val)

    return _mark_isolation(gv_fn)


# ── vjp ──────────────────────────────────────────────────────────────────────


def vjp(
    func: Callable[..., Tensor | tuple[object, ...]],
    *primals: Tensor,
    has_aux: bool = False,
) -> tuple[
    Tensor | tuple[Tensor, ...] | tuple[Tensor | tuple[Tensor, ...], object],
    Callable[..., tuple[Tensor | None, ...]],
]:
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
        Differentiable function taking the primals as positional inputs.
    *primals : Tensor
        Points at which to linearise ``func``.
    has_aux : bool, optional
        If ``True``, ``func`` must return ``(output, aux)``; the call
        then yields ``((output, aux), vjp_fn)``. Default ``False``.

    Returns
    -------
    tuple
        ``(outputs, vjp_fn)``. ``vjp_fn(*cotangents)`` returns a tuple of
        input-gradient tensors of the same shapes as ``primals``.

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
    from lucid.autograd._backward import grad as _ag
    from lucid.autograd._grad_mode import enable_grad

    prepared = _prepare_grad_args(primals, tuple(range(len(primals))))
    primals_rg = [_tensor_argument(prepared, i, "vjp") for i in range(len(prepared))]
    with enable_grad():
        raw_out = func(*primals_rg)

    first, aux = _split_aux(raw_out, has_aux=has_aux, name="vjp")
    outputs = _output_tensors(first, "vjp")
    out_list = list(outputs) if isinstance(outputs, tuple) else [outputs]

    def _vjp(*cotangents: Tensor) -> tuple[Tensor | None, ...]:
        cots: list[Tensor | None] = list(cotangents) if cotangents else [None] * len(out_list)
        return tuple(
            _ag(
                out_list,
                primals_rg,
                grad_outputs=cots,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
        )

    vjp_fn = _mark_isolation(_vjp)
    return ((outputs, aux), vjp_fn) if has_aux else (outputs, vjp_fn)


# ── jvp ──────────────────────────────────────────────────────────────────────


def _on_tensors(
    func: Callable[..., Tensor | tuple[Tensor, ...]], primals: tuple[Tensor, ...]
) -> tuple[Callable[..., Tensor | tuple[Tensor, ...]], tuple[int, ...]]:
    """``func`` as a function of its tensor primals alone, and their
    positions — a primal that is not a tensor is held fixed."""
    from lucid._tensor.tensor import Tensor as _T

    idx = tuple(i for i, p in enumerate(primals) if isinstance(p, _T))
    if len(idx) == len(primals):
        return func, idx
    return (lambda *sel: func(*_splice(primals, idx, sel))), idx


def _tangent_pass(
    fn: Callable[..., Tensor | tuple[Tensor, ...]],
    primals: tuple[Tensor, ...],
    tangents: tuple[Tensor, ...],
    *,
    attach: bool,
    required: bool,
    strict: bool,
) -> tuple[Tensor | tuple[Tensor, ...], Tensor | tuple[Tensor, ...]]:
    """``fn``'s outputs and tangents at ``primals``, the tangents attached
    to the caller's graph when ``attach``.

    The attachment is ``required`` when a primal or a tangent requires grad:
    the caller differentiates through the tangent, and an op on the path
    with no graph-mode backward is then an error.  Otherwise only tensors
    ``fn`` captures would attach it, and through such an op the tangent is
    taken first-order and returned detached, with a warning — ``fn`` runs
    once more for it, on the same random draws.
    """
    rewind = _replay_draws() if attach and not required else None
    try:
        return _jvp_by_double_backward(fn, primals, tangents, create_graph=attach, strict=strict)
    except NotImplementedError:
        if rewind is None:
            raise
    warnings.warn(
        "lucid.func: an op on the path has no graph-mode backward, so the "
        "tangent cannot reach the tensors func captures and is returned "
        "detached",
        stacklevel=4,
    )
    rewind()
    return _jvp_by_double_backward(fn, primals, tangents, create_graph=False, strict=strict)


def _jvp_attached(
    fn: Callable[..., Tensor | tuple[Tensor, ...]],
    primals: tuple[Tensor, ...],
    tangents: tuple[Tensor, ...],
    strict: bool,
) -> tuple[Tensor | tuple[Tensor, ...], Tensor | tuple[Tensor, ...]]:
    """:func:`jvp` at tensor primals, its results attached to the caller's
    graph exactly where the reference attaches them."""
    import lucid

    grad_on = lucid.is_grad_enabled()
    if not grad_on or any(p.requires_grad for p in primals):
        return _tangent_pass(fn, primals, tangents, attach=grad_on, required=grad_on, strict=strict)
    # Nothing passed in requires grad, yet ``fn`` can still reach the
    # caller's graph through the tensors it captures — a model's parameters.
    # The backward passes run ``fn`` at copies of the primals that require
    # grad, so their output is attached whatever ``fn`` captures; only an
    # evaluation at the primals themselves tells.
    replay = _replay_draws()
    primals_out = _output_tensors(fn(*primals), "jvp")
    replay()
    required = any(t.requires_grad for t in tangents)
    _, tangents_out = _tangent_pass(
        fn,
        primals,
        tangents,
        attach=required or _requires_grad(primals_out),
        required=required,
        strict=strict,
    )
    return primals_out, tangents_out


def jvp(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    primals: tuple[Tensor, ...],
    tangents: tuple[Tensor, ...],
    strict: bool = False,
) -> tuple[Tensor | tuple[Tensor, ...], Tensor | tuple[Tensor, ...]]:
    r"""Compute the Jacobian-vector product of ``func`` at ``primals``.

    Returns both the primal output ``func(*primals)`` and the directional
    derivative :math:`J(\mathrm{primals}) \cdot \mathrm{tangents}` — the
    forward-mode contraction of the Jacobian against a tangent vector.
    Forward-mode is preferred over reverse-mode when the input dimension
    is small relative to the output dimension.

    Parameters
    ----------
    func : Callable
        Differentiable function returning a Tensor or tuple of Tensors.
    primals : tuple of Tensor
        Points at which to evaluate ``func`` and its Jacobian.
    tangents : tuple of Tensor
        Tangent vectors, one per primal, each matching the shape of the
        corresponding primal.
    strict : bool, optional
        If ``True``, refuse what :func:`lucid.autograd.jvp` refuses under
        ``strict`` — an output independent of the inputs among them —
        instead of giving it a zero tangent. Default ``False``.

    Returns
    -------
    tuple
        ``(primals_out, tangents_out)`` with the same nested structure as
        ``func(*primals)``.

    Raises
    ------
    RuntimeError
        ``primals`` and ``tangents`` differ in length, or ``strict``
        refuses a result.
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
    primals_t = tuple(primals)
    tangents_t = tuple(tangents)
    if len(primals_t) != len(tangents_t):
        raise RuntimeError(
            f"jvp: expected one tangent per primal, got {len(primals_t)} primals "
            f"and {len(tangents_t)} tangents"
        )
    fn, idx = _on_tensors(func, primals_t)
    return _jvp_attached(
        fn,
        tuple(primals_t[i] for i in idx),
        tuple(tangents_t[i] for i in idx),
        strict,
    )


# ── linearize ────────────────────────────────────────────────────────────────


def linearize(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    *primals: Tensor,
) -> tuple[Tensor | tuple[Tensor, ...], Callable[..., Tensor | tuple[Tensor, ...]]]:
    r"""Linearise ``func`` around ``primals`` for reuse across many tangents.

    Returns the primal output together with a callable ``linear_fn`` that
    applies the first-order Taylor expansion of ``func`` at ``primals``.
    Mathematically ``linear_fn(t)`` evaluates :math:`J_f(\mathrm{primals})
    \, t` — the same quantity as :func:`jvp` — but the linearisation cost
    is paid once even if many tangent vectors are queried subsequently.

    Parameters
    ----------
    func : Callable
        Differentiable function.
    *primals : Tensor
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
    import lucid

    fn, idx = _on_tensors(func, primals)
    primals_sel = tuple(primals[i] for i in idx)
    primals_out = _output_tensors(func(*primals), "linearize")
    attached, primal_attached = _attachment(primals_sel, primals_out)

    def linear_fn(
        *tangents: Tensor,
    ) -> Tensor | tuple[Tensor, ...]:
        """Closure that applies the linear approximation."""
        if len(tangents) != len(primals):
            raise RuntimeError(
                f"linearize: expected one tangent per primal, got {len(primals)} "
                f"primals and {len(tangents)} tangents"
            )
        on = lucid.is_grad_enabled()
        required = on and (primal_attached or any(tangents[i].requires_grad for i in idx))
        _, tangents_out = _tangent_pass(
            fn,
            primals_sel,
            tuple(tangents[i] for i in idx),
            attach=required or (on and attached),
            required=required,
            strict=False,
        )
        return tangents_out

    return primals_out, _mark_isolation(linear_fn)


# ── jacrev & jacfwd ──────────────────────────────────────────────────────────


def _one_hot_tangent(x: Tensor, col: int) -> Tensor:
    from lucid._C import engine as _C_engine
    from lucid._tensor.tensor import Tensor as _TT

    x_numel = x.numel()
    x_shape = list(x.shape)
    dev = x._impl.device
    dtype = x._impl.dtype
    t_flat = _C_engine.zeros([x_numel], dtype, dev)
    one = _C_engine.ones([1], dtype, dev)
    idx = _C_engine.full([1], float(col), _C_engine.I32, dev)
    t_flat = _C_engine.scatter_add(t_flat, idx, one, 0)
    return _TT.__new_from_impl__(_C_engine.reshape(t_flat, x_shape) if x_shape else t_flat)


def _jacfwd_block(
    wrapped_fn: Callable[..., Tensor | tuple[Tensor, ...]],
    primals: tuple[Tensor, ...],
    target: int,
    out_shape: list[int],
    *,
    attach: bool,
    required: bool,
) -> Tensor | None:
    """The Jacobian of ``wrapped_fn``'s output in ``primals[target]``, one
    column — one :func:`jvp` along a unit tangent — at a time."""
    import lucid

    x = primals[target]
    cols: list[Tensor] = []
    for col in range(x.numel()):
        tans = tuple(
            _one_hot_tangent(x, col) if k == target else lucid.zeros_like(p)
            for k, p in enumerate(primals)
        )
        _, jvp_col = _tangent_pass(
            wrapped_fn,
            primals,
            tans,
            attach=attach,
            required=required,
            strict=False,
        )
        jc = jvp_col[0] if isinstance(jvp_col, tuple) else jvp_col
        cols.append(jc.reshape(-1))
    if not cols:
        return None
    return lucid.stack(cols, dim=-1).reshape(out_shape + list(x.shape))


def jacrev[**P](
    func: Callable[P, Tensor | tuple[object, ...]],
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    *,
    chunk_size: int | None = None,
) -> Callable[P, _Jacobian | tuple[_Jacobian, object]]:
    r"""Build a function returning the reverse-mode Jacobian of ``func``.

    Assembles the full Jacobian matrix one row at a time, each row
    obtained from a backward pass seeded with a one-hot cotangent. For a
    scalar-output ``func`` this collapses to :func:`grad`.

    Parameters
    ----------
    func : Callable
        Differentiable function returning a Tensor (or ``(output, aux)``
        when ``has_aux=True``).
    argnums : int or tuple of int, optional
        Argument(s) to differentiate with respect to. Default ``0``.
    has_aux : bool, optional
        Whether ``func`` returns auxiliary data. Default ``False``.
    chunk_size : int, optional
        Reserved for future vmap-batched row computation.

    Returns
    -------
    Callable
        Function returning the Jacobian tensor (or tuple of tensors)
        with shape ``output.shape + arg.shape``.

    Notes
    -----
    For :math:`f : \mathbb{R}^n \to \mathbb{R}^m` produces

    .. math::

        J_{ij} = \frac{\partial f_i}{\partial x_j}, \quad
        J \in \mathbb{R}^{m \times n}.

    Cost scales with the output dimension :math:`m` (one backward pass
    per row). Prefer :func:`jacfwd` when :math:`n < m`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.func import jacrev
    >>> f = lambda x: lucid.stack([x.sum(), (x ** 2).sum()])
    >>> jacrev(f)(lucid.tensor([1.0, 2.0, 3.0]))  # shape (2, 3)
    tensor([[1., 1., 1.], [2., 4., 6.]])
    """
    from lucid.autograd._functional import jacobian as _jacobian

    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("jacrev: chunk_size must be positive")
    _scalar_argnum, _argnums = _normalise_argnums(argnums)
    call: Callable[..., object] = func

    def jac_fn(*args: P.args, **kwargs: P.kwargs) -> _Jacobian | tuple[_Jacobian, object]:
        """Closure that computes the Jacobian."""
        arg_list = list(args)
        selected = tuple(_tensor_argument(arg_list, i, "jacrev") for i in _argnums)
        aux_holder: list[object] = []

        def _wrapped(*sel: Tensor) -> Tensor | tuple[Tensor, ...]:
            out, aux = _split_aux(
                call(*_splice(args, _argnums, sel), **kwargs),
                has_aux=has_aux,
                name="jacrev",
            )
            aux_holder.append(aux)
            return _output_tensors(out, "jacrev")

        jac = _jacobian(_wrapped, selected[0] if _scalar_argnum else selected)
        return (jac, aux_holder[0]) if has_aux else jac

    return _mark_isolation(jac_fn)


def jacfwd[**P](
    func: Callable[P, Tensor | tuple[object, ...]],
    argnums: int | tuple[int, ...] = 0,
    has_aux: bool = False,
    *,
    randomness: str = "error",
) -> Callable[P, _PerArgnum | tuple[_PerArgnum, object]]:
    r"""Build a function returning the forward-mode Jacobian of ``func``.

    Materialises the Jacobian one column at a time by repeatedly calling
    :func:`jvp` with one-hot tangent vectors. Each column corresponds to
    the partial derivative of every output with respect to a single input
    coordinate.

    Parameters
    ----------
    func : Callable
        Differentiable function returning a Tensor (or ``(output, aux)``
        when ``has_aux=True``).
    argnums : int or tuple of int, optional
        Argument(s) to differentiate with respect to. Default ``0``.
    has_aux : bool, optional
        Whether ``func`` returns auxiliary data. Default ``False``.
    randomness : str, optional
        Randomness policy forwarded to internal vmap calls. Default
        ``"error"``.

    Returns
    -------
    Callable
        Function returning the Jacobian tensor (or tuple of tensors)
        with shape ``output.shape + arg.shape``.

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
    _scalar_argnum, _argnums = _normalise_argnums(argnums)
    call: Callable[..., object] = func

    def jac_fn(*args: P.args, **kwargs: P.kwargs) -> _PerArgnum | tuple[_PerArgnum, object]:
        """Closure that computes the Jacobian."""
        arg_list = list(args)
        primals = tuple(_tensor_argument(arg_list, i, "jacfwd") for i in _argnums)
        with _RandGuard(randomness):
            # At the arguments, in the caller's grad mode: the output's shape,
            # the aux, and the attachment — decided once, not per column.
            first, aux = _split_aux(call(*args, **kwargs), has_aux=has_aux, name="jacfwd")
            sample = _output_tensors(first, "jacfwd")
            out_shape = [] if isinstance(sample, tuple) else list(sample.shape)
            attach, required = _attachment(primals, sample)

            def _wrapped(*sel: Tensor) -> Tensor | tuple[Tensor, ...]:
                out, _ = _split_aux(
                    call(*_splice(args, _argnums, sel), **kwargs),
                    has_aux=has_aux,
                    name="jacfwd",
                )
                return _output_tensors(out, "jacfwd")

            results = tuple(
                _jacfwd_block(
                    _wrapped,
                    primals,
                    k,
                    out_shape,
                    attach=attach,
                    required=required,
                )
                for k in range(len(_argnums))
            )

        r = results[0] if _scalar_argnum else results
        return (r, aux) if has_aux else r

    return _mark_isolation(jac_fn)


# ── hessian ───────────────────────────────────────────────────────────────────


def hessian[**P](
    func: Callable[P, Tensor],
    argnums: int | tuple[int, ...] = 0,
) -> Callable[P, Tensor | tuple[tuple[Tensor, ...], ...]]:
    r"""Build a function returning the Hessian of a scalar-valued ``func``.

    Computes the matrix of second partial derivatives by composing
    forward-mode over reverse-mode differentiation
    (``jacfwd(jacrev(func))``). The result captures the local curvature
    used by Newton-style optimisers, natural-gradient methods, and
    second-order analyses such as eigenvalue spectra of the loss.

    Parameters
    ----------
    func : Callable
        Function returning a scalar Tensor.
    argnums : int or tuple of int, optional
        Argument(s) to differentiate twice. Default ``0``.

    Returns
    -------
    Callable
        Function returning the Hessian tensor with shape
        ``arg.shape + arg.shape``.

    Notes
    -----
    For :math:`f : \mathbb{R}^n \to \mathbb{R}`,

    .. math::

        H_{ij} = \frac{\partial^2 f}{\partial x_i \, \partial x_j},
        \quad H \in \mathbb{R}^{n \times n}.

    Cost is dominated by :math:`n` JVPs over the reverse-mode gradient
    graph, retained via ``create_graph=True``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.func import hessian
    >>> f = lambda x: (x ** 3).sum()  # H = diag(6 * x)
    >>> hessian(f)(lucid.tensor([1.0, 2.0, 3.0]))
    tensor([[6., 0., 0.], [0., 12., 0.], [0., 0., 18.]])
    """
    from lucid.autograd._functional import hessian as _hessian

    _scalar_argnum, _argnums = _normalise_argnums(argnums)
    call: Callable[..., Tensor] = func

    def hessian_fn(*args: P.args, **kwargs: P.kwargs) -> Tensor | tuple[tuple[Tensor, ...], ...]:
        """Closure that computes the Hessian (second-order derivative tensor)."""
        arg_list = list(args)
        selected = tuple(_tensor_argument(arg_list, i, "hessian") for i in _argnums)
        return _hessian(
            lambda *sel: call(*_splice(args, _argnums, sel), **kwargs),
            selected[0] if _scalar_argnum else selected,
        )

    return _mark_isolation(hessian_fn)


def _splice(
    args: tuple[object, ...],
    argnums: tuple[int, ...],
    selected: tuple[object, ...],
) -> tuple[object, ...]:
    """Replace args at argnums positions with selected values."""
    lst = list(args)
    for i, s in zip(argnums, selected, strict=True):
        lst[i] = s
    return tuple(lst)
