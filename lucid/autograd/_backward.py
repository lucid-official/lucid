"""
autograd.backward() and autograd.grad() free functions.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.autograd._grad_mode import no_grad

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _refuse_trace() -> None:
    """Mark an active compile trace as one no executable can stand for.

    A backward pass inside a traced function — MeanFlow's ``jvp``, a
    gradient penalty — computes storage the tracer never records: compiled,
    it would replay the trace-time values, and ``make_step`` hung building
    a VJP of the recorded backward.  The trace is marked, every compile
    entry point refuses it, and the call runs eager.
    """
    tracer = _C_engine.compile.current_tracer()
    if tracer is not None:
        tracer.mark_unsupported("a backward pass ran inside the traced function")


def _as_tensors(value: Tensor | Sequence[Tensor]) -> list[Tensor]:
    """One tensor or a sequence of them, as a list."""
    from lucid._tensor.tensor import Tensor

    return [value] if isinstance(value, Tensor) else list(value)


def _check_seed(where: str, index: int, out: Tensor, seed: object) -> Tensor | None:
    """Hold one seed gradient to the output it is for.

    ``None`` stands for an implicit ``1`` and is allowed only for a real
    output of one element.  A tensor must have the output's shape and
    device, and be complex exactly when the output is; another real dtype
    is cast to the output's, as the reference casts it.
    """
    from lucid._tensor.tensor import Tensor

    if seed is None:
        if out.numel() != 1:
            raise RuntimeError(
                f"{where}(): grad can be implicitly created only for scalar "
                f"outputs, but output {index} has shape {tuple(out.shape)}; "
                "pass its gradient explicitly"
            )
        if out.is_complex():
            raise RuntimeError(
                f"{where}(): grad can be implicitly created only for real "
                f"scalar outputs, but output {index} is {out.dtype}"
            )
        return None
    if not isinstance(seed, Tensor):
        raise TypeError(
            f"{where}(): gradient {index} must be a Tensor or None, " f"got {type(seed).__name__}"
        )
    if tuple(seed.shape) != tuple(out.shape):
        raise _C_engine.ShapeMismatch(
            f"{where}(): mismatch in shape: gradient {index} has shape "
            f"{tuple(seed.shape)} and output {index} has shape {tuple(out.shape)}"
        )
    if seed.is_complex() != out.is_complex():
        raise _C_engine.DtypeMismatch(
            f"{where}(): a complex output and its gradient must both be "
            f"complex: gradient {index} is {seed.dtype} and output {index} "
            f"is {out.dtype}"
        )
    if seed.device != out.device:
        raise _C_engine.DeviceMismatch(
            f"{where}(): gradient {index} is on {seed.device.type} but "
            f"output {index} is on {out.device.type}"
        )
    return seed if seed.dtype == out.dtype else seed.to(out.dtype)


def _make_seeds(
    where: str, outputs: Sequence[Tensor], grads: Tensor | Sequence[Tensor | None] | None
) -> list[Tensor | None]:
    """Match the seed gradients a caller passed to the outputs, one each.

    The one boundary for ``grad_outputs`` / ``grad_tensors`` /
    ``Tensor.backward(gradient)``: a single tensor or ``None`` stands for
    a one-element sequence, the count must equal the number of outputs,
    and each seed is checked by :func:`_check_seed`.

    Raises
    ------
    TypeError
        ``grads`` or one of its entries is neither a tensor nor ``None``.
    RuntimeError
        The counts differ, or ``None`` stands for the gradient of an
        output that is not a real scalar.
    ShapeMismatch, DtypeMismatch, DeviceMismatch
        A seed does not fit its output.
    """
    seeds = _raw_grad_list(where, outputs, grads)
    return [
        _check_seed(where, i, out, seed)
        for i, (out, seed) in enumerate(zip(outputs, seeds, strict=True))
    ]


def _engine_seeds(
    seeds: list[Tensor | None], keep_seed_graph: bool
) -> list[_C_engine.TensorImpl | None]:
    """The seeds as the engine takes them, one per root: ``None`` is ones.

    The engine seeds each root with its gradient as given — the reference's
    ``grad_tensors``.  ``keep_seed_graph`` keeps a seed's own graph
    attached, so that under ``create_graph`` a gradient is differentiable
    with respect to whatever computed the seed.
    """
    return [None if s is None else _unwrap(s if keep_seed_graph else s.detach()) for s in seeds]


def backward(
    tensors: Tensor | Sequence[Tensor],
    grad_tensors: Tensor | Sequence[Tensor | None] | None = None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
    inputs: Tensor | Sequence[Tensor] | None = None,
) -> None:
    r"""Compute gradients of ``tensors`` w.r.t. the leaf variables in their graph.

    Top-level entry point that triggers reverse-mode automatic
    differentiation across the computation graph rooted at
    ``tensors``. For every leaf tensor ``x`` reachable from
    ``tensors`` whose ``requires_grad`` is ``True``, this function
    accumulates :math:`\partial \mathcal{L} / \partial x` into
    ``x.grad``, where :math:`\mathcal{L}` is the (possibly weighted)
    sum of the root tensors.

    The chain rule is applied edge-by-edge during a topological
    walk of the graph in reverse order, so each intermediate
    Jacobian-vector product fires exactly once.

    Parameters
    ----------
    tensors : Tensor or list of Tensor
        Root tensors at which the backward pass starts. When more
        than one root is supplied each receives its own seed and the
        contributions are summed at every shared leaf.
    grad_tensors : Tensor, sequence of (Tensor or None), or None, optional
        Seed cotangent vectors, one per root tensor, each of its root's
        shape and device (another real dtype is cast to the root's).
        A single tensor stands for a one-element sequence.  ``None`` —
        for the whole argument or one entry — is an implicit ``1``,
        allowed only for a real root of one element.
    retain_graph : bool, optional
        If ``True`` the intermediate saved tensors are not freed after
        the backward pass, so the same graph can be traversed again.
        Defaults to ``create_graph``.
    create_graph : bool, optional
        If ``True`` the operations performed during backward are
        themselves recorded in the graph, enabling higher-order
        differentiation (e.g. Hessian-vector products, meta-learning).
        Implies stronger memory usage. Defaults to ``False``.
    inputs : Tensor, sequence of Tensor, or None, optional
        Restrict accumulation to these tensors: only their ``.grad``
        changes, every other leaf in the graph keeps what it had.  An
        input that is not a leaf has its ``.grad`` filled too (it is
        marked :meth:`~Tensor.retain_grad`, as the reference does).  An
        input the roots do not reach is left alone.  ``None`` (default)
        accumulates into every leaf.

    Returns
    -------
    None
        Gradients are accumulated in-place onto each leaf tensor's
        ``.grad`` attribute. Existing ``.grad`` values are added to,
        not overwritten — call :meth:`Tensor.zero_grad` (or the
        optimizer's ``zero_grad``) between successive backward passes
        if accumulation is undesired.

    Raises
    ------
    RuntimeError
        A root does not require grad, the number of gradients differs
        from the number of roots, a non-scalar root has no gradient,
        ``inputs`` is empty, or one of ``inputs`` does not require grad.
    TypeError
        A gradient is neither a tensor nor ``None``, or an entry of
        ``inputs`` is not a tensor.
    ShapeMismatch, DtypeMismatch, DeviceMismatch
        A gradient's shape or device differs from its root's, or only
        one of the two is complex.

    Notes
    -----
    Reverse-mode AD computes

    .. math::

        \frac{\partial \mathcal{L}}{\partial x}
        = \sum_{i}
            \left(
                \frac{\partial \mathcal{L}}{\partial t_i}
            \right)^{\!\top}
            \frac{\partial t_i}{\partial x},

    propagating cotangents :math:`\bar t_i = \partial \mathcal{L} /
    \partial t_i` from the roots through each saved op contract
    until every reachable leaf has received its contribution.

    Memory/compute trade-off:

    * ``retain_graph=False`` (default) is the cheapest mode — once
      the walk finishes, every saved tensor is freed.
    * ``retain_graph=True, create_graph=False`` keeps activations
      so the same graph can be traversed again.
    * ``create_graph=True`` additionally records the backward ops
      in a new graph, doubling memory in the worst case but
      enabling :math:`\nabla^2 \mathcal{L}` and beyond.

    Examples
    --------
    >>> import lucid
    >>> from lucid.autograd import backward
    >>> x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> y = (x * x).sum()
    >>> backward(y)
    >>> x.grad
    tensor([2., 4., 6.])

    Only the tensors in ``inputs`` accumulate:

    >>> a = lucid.ones(2, requires_grad=True)
    >>> b = lucid.ones(2, requires_grad=True)
    >>> backward((a * b * 3).sum(), inputs=[a])
    >>> a.grad, b.grad
    (tensor([3., 3.]), None)
    """
    roots = _as_tensors(tensors)
    for i, t in enumerate(roots):
        if not t.requires_grad:
            # Returning quietly turned "this loss is not connected to
            # anything trainable" into a training loop that never moved a
            # parameter.
            raise RuntimeError(
                f"backward(): element {i} of tensors does not require grad "
                "and has no grad_fn, so nothing upstream of it can receive "
                "a gradient. If it is a loss, check that it was not computed "
                "under lucid.no_grad() and that every op on its path tracks "
                "gradients (requires_grad is True on its inputs)."
            )
    seeds = _make_seeds("backward", roots, grad_tensors)
    _retain = retain_graph if retain_graph is not None else create_graph
    if inputs is not None:
        _backward_into(roots, seeds, _backward_inputs(inputs), _retain, create_graph=create_graph)
        return
    _refuse_trace()
    # One pass from every root, as the reference runs it: a pass per root
    # freed the graph the roots share under the second one, and ran the
    # hooks of a tensor they share once per root, on part of its gradient.
    _C_engine.engine_backward(
        [_unwrap(root) for root in roots],
        _engine_seeds(seeds, keep_seed_graph=False),
        retain_graph=_retain,
        create_graph=create_graph,
    )


def _backward_inputs(inputs: Tensor | Sequence[Tensor]) -> list[Tensor]:
    """Check the ``inputs`` of :func:`backward`: tensors that require grad.

    Raises
    ------
    RuntimeError
        ``inputs`` is empty, or one of them does not require grad.
    TypeError
        An entry is not a tensor.
    """
    from lucid._tensor.tensor import Tensor

    targets = _as_tensors(inputs)
    if not targets:
        raise RuntimeError("backward(): inputs cannot be empty; pass None for every leaf")
    for i, t in enumerate(targets):
        if not isinstance(t, Tensor):
            raise TypeError(f"backward(): inputs[{i}] must be a Tensor, got {type(t).__name__}")
        if not t.requires_grad:
            raise RuntimeError(
                f"backward(): inputs[{i}] does not require grad, so no "
                "gradient can be accumulated into it"
            )
    return targets


def _backward_into(
    roots: list[Tensor],
    seeds: list[Tensor | None],
    targets: list[Tensor],
    retain_graph: bool,
    *,
    create_graph: bool = False,
) -> None:
    """``backward`` restricted to ``targets``: the engine's ``grad`` walks
    only the paths to them, then each gradient is accumulated.

    A target that is not a leaf is marked to retain its gradient first, so
    the traversal itself fills its ``.grad`` (the reference does the same);
    a leaf's gradient is added here, after its hooks ran in the traversal.
    """
    for t in targets:
        if not t.is_leaf:
            t.retain_grad()
    grads = grad(
        roots,
        targets,
        grad_outputs=seeds,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )
    done: set[int] = set()
    for t, g in zip(targets, grads, strict=True):
        # A tensor named twice is still one accumulation, as one leaf's
        # AccumulateGrad runs once per pass.
        if g is None or not t.is_leaf or id(t) in done:
            continue
        done.add(id(t))
        current = t.grad
        if create_graph or (current is not None and current.requires_grad):
            _C_engine._accumulate_grad(_unwrap(t), _unwrap(g))
        elif current is None:
            t.grad = g
        else:
            # In place, as the engine accumulates: a ``.grad`` the caller
            # holds sees the new total.
            with no_grad():
                current.add_(g)


def _raw_grad_list(
    where: str, outs: Sequence[Tensor], grads: Tensor | Sequence[Tensor | None] | None
) -> list[object]:
    """Normalize ``grads`` into a list matching ``len(outs)``."""
    from lucid._tensor.tensor import Tensor

    if grads is None:
        raw: list[object] = [None] * len(outs)
    elif isinstance(grads, Tensor):
        raw = [grads]
    elif isinstance(grads, (list, tuple)):
        raw = list(grads)
    else:
        raise TypeError(
            f"{where}(): gradients must be a Tensor, a sequence of Tensors "
            f"or None, got {type(grads).__name__}"
        )
    if len(raw) != len(outs):
        raise RuntimeError(f"{where}(): got {len(outs)} tensors and {len(raw)} gradients")
    return raw


def _check_batched_seeds(
    outs: list[Tensor], grad_outputs: Tensor | Sequence[Tensor | None] | None
) -> tuple[list[Tensor], int]:
    """Validate batched ``grad_outputs`` and return ``(batched_seeds, batch_size)``."""
    from lucid._tensor.tensor import Tensor

    raw = _raw_grad_list("grad", outs, grad_outputs)
    batched_seeds: list[Tensor] = []
    batch_size: int | None = None
    for i, (out, s) in enumerate(zip(outs, raw, strict=True)):
        if not isinstance(s, Tensor) or s.ndim == 0:
            raise ValueError(
                "grad(): expected each grad_output to be a batched Tensor with "
                "at least 1 dimension when is_grads_batched=True"
            )
        if tuple(s.shape[1:]) != tuple(out.shape):
            raise RuntimeError(
                "If `is_grads_batched=True`, we interpret the first dimension of "
                "each grad_output as the batch dimension. The sizes of the "
                "remaining dimensions are expected to match the shape of "
                f"corresponding output, but a mismatch was detected: grad_output[{i}] "
                f"has a shape of {tuple(s.shape[1:])} and output[{i}] has a shape of "
                f"{tuple(out.shape)}."
            )
        b = int(s.shape[0])
        if batch_size is not None and b != batch_size:
            raise RuntimeError(
                "grad(): all batched grad_outputs must share the same batch size "
                f"in dimension 0, got {batch_size} and {b}"
            )
        batch_size = b
        batched_seeds.append(s)
    assert batch_size is not None
    return batched_seeds, batch_size


def _batched_grad(
    outs: list[Tensor],
    ins: list[Tensor],
    grad_outputs: Tensor | Sequence[Tensor | None] | None,
    retain_graph: bool,
    create_graph: bool,
) -> tuple[list[Tensor | None], int]:
    """Run one engine grad pass per batch element when ``is_grads_batched=True``."""
    import lucid as _lucid

    batched_seeds, batch_size = _check_batched_seeds(outs, grad_outputs)
    rows: list[list[Tensor | None]] = []
    for b in range(batch_size):
        slice_seeds = _make_seeds("grad", outs, [s[b] for s in batched_seeds])
        partials = _C_engine.engine_grad(
            [_unwrap(out) for out in outs],
            [_unwrap(inp) for inp in ins],
            _engine_seeds(slice_seeds, keep_seed_graph=create_graph),
            retain_graph if b == batch_size - 1 else True,
            create_graph,
        )
        rows.append([None if impl is None else _wrap(impl) for impl in partials])

    totals: list[Tensor | None] = []
    for j in range(len(ins)):
        col = [g for row in rows if (g := row[j]) is not None]
        totals.append(_lucid.stack(col, dim=0) if len(col) == batch_size else None)
    return totals, batch_size


def _finalize_grads(
    totals: list[Tensor | None],
    ins: list[Tensor],
    allow_unused: bool,
    materialize_grads: bool,
    batch_size: int | None,
) -> tuple[Tensor | None, ...]:
    """Check unused inputs and optionally materialize zero gradients."""
    if not allow_unused and any(g is None for g in totals):
        raise RuntimeError(
            "One of the differentiated tensors does not require grad "
            "and is not reachable from outputs. "
            "Set allow_unused=True to suppress this error."
        )
    if not materialize_grads:
        return tuple(totals)
    import lucid as _lucid

    if batch_size is not None:
        return tuple(
            (
                _lucid.zeros((batch_size, *inp.shape), dtype=inp.dtype, device=inp.device)
                if g is None
                else g
            )
            for g, inp in zip(totals, ins, strict=True)
        )
    return tuple(
        _lucid.zeros_like(inp) if g is None else g for g, inp in zip(totals, ins, strict=True)
    )


def grad(
    outputs: Tensor | Sequence[Tensor],
    inputs: Tensor | Sequence[Tensor],
    grad_outputs: Tensor | Sequence[Tensor | None] | None = None,
    retain_graph: bool | None = None,
    create_graph: bool = False,
    only_inputs: bool = True,
    allow_unused: bool | None = None,
    is_grads_batched: bool = False,
    materialize_grads: bool = False,
) -> tuple[Tensor | None, ...]:
    r"""Compute gradients of outputs w.r.t. inputs, returning them as a tuple.

    The "functional" gradient interface — invoke once to get the partial
    derivatives back without touching ``.grad`` on any leaf tensor.
    Useful for higher-order differentiation, gradient-based meta-learning,
    or any pattern where you want to use the gradients as input to a new
    computation rather than to update parameters in-place.

    Parameters
    ----------
    outputs : Tensor or list of Tensor
        Output tensors to differentiate.  Each must have ``requires_grad``
        set in the graph that produced it.
    inputs : Tensor or list of Tensor
        Input tensors w.r.t. which gradients are requested.  Each must be
        a leaf (or non-leaf with ``requires_grad=True`` if you want grads
        flowing into intermediate nodes).
    grad_outputs : Tensor, sequence of (Tensor or None), or None, optional
        Seed gradients :math:`\partial \mathcal{L} / \partial \text{outputs}`,
        one per output, each of its output's shape and device (another
        real dtype is cast to the output's).  A single tensor stands for
        a one-element sequence.  ``None`` — for the whole argument or one
        entry — is an implicit ``1``, allowed only for a real output of
        one element.
    retain_graph : bool, optional
        Keep the autograd graph alive after this call so additional
        backward passes are possible.  Defaults to ``create_graph``.
    create_graph : bool, optional
        If ``True``, build the autograd graph of the gradient itself so
        the returned tensors are differentiable — used by
        :func:`gradgradcheck` and other higher-order recipes.
    only_inputs : bool, optional
        Reserved for reference-framework compatibility; gradients are
        always restricted to the requested ``inputs``.
    allow_unused : bool, optional
        If ``True``, return ``None`` for any ``inputs`` entry that lies
        outside the computation graph of ``outputs``.  Defaults to
        ``materialize_grads``.
    is_grads_batched : bool, optional
        If ``True``, interpret the leading dimension of each tensor in
        ``grad_outputs`` as a batch dimension and compute a batch of
        vector-Jacobian products.  Defaults to ``False``.
    materialize_grads : bool, optional
        If ``True``, replace ``None`` gradients for unused inputs with
        zero-filled tensors matching each input's shape, dtype, and device.
        Defaults to ``False``.

    Returns
    -------
    tuple[Tensor or None, ...]
        One gradient per element of ``inputs``, in the same order.
        Entries are ``None`` only when ``allow_unused=True`` (and
        ``materialize_grads=False``) and the input is disconnected.

    Raises
    ------
    RuntimeError
        The number of ``grad_outputs`` differs from the number of
        outputs, a non-scalar output has no seed, or (without
        ``allow_unused``) an input is unreachable from ``outputs``.
    ValueError
        If ``allow_unused=False`` and ``materialize_grads=True``.
    TypeError
        A seed is neither a tensor nor ``None``.
    ShapeMismatch, DtypeMismatch, DeviceMismatch
        A seed's shape or device differs from its output's, or only one
        of the two is complex.

    Notes
    -----
    Mathematically, ``grad`` computes the vector-Jacobian product

    .. math::

        \frac{\partial}{\partial \mathbf{x}}
        \left(\sum_k \text{grad\_outputs}_k \cdot \text{outputs}_k\right)

    via one reverse-mode pass.  Unlike :meth:`Tensor.backward`, it does
    NOT accumulate into ``.grad`` — leaf tensors' existing ``.grad``
    values are preserved across the call.  For chained gradient
    computations (Hessian-vector products, MAML inner loops) this is the
    right primitive.

    Examples
    --------
    Scalar output — no seed needed:

    >>> import lucid
    >>> from lucid.autograd import grad
    >>> x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> y = (x * x).sum()
    >>> (gx,) = grad(y, [x])
    >>> gx                         # equals 2 * x
    tensor([2., 4., 6.])

    Vector output — explicit seed:

    >>> z = x * x
    >>> seed = lucid.ones_like(z)
    >>> (gx,) = grad(z, [x], grad_outputs=[seed])

    Higher-order with ``create_graph=True``:

    >>> y = (x ** 3).sum()
    >>> (g,) = grad(y, [x], create_graph=True)
    >>> (gg,) = grad(g.sum(), [x])    # second derivative: 6x
    >>> gg
    tensor([6., 12., 18.])
    """
    del only_inputs  # reference-signature parity; grad never populates leaf .grad
    if allow_unused is False and materialize_grads:
        raise ValueError(
            "Expected allow_unused to be True or not passed when "
            "materialize_grads=True, but got: allow_unused=False."
        )
    _allow = bool(materialize_grads) if allow_unused is None else bool(allow_unused)
    outs, ins = _as_tensors(outputs), _as_tensors(inputs)
    _retain = retain_graph if retain_graph is not None else create_graph
    _refuse_trace()
    if is_grads_batched:
        totals, bsz = _batched_grad(outs, ins, grad_outputs, _retain, create_graph)
        return _finalize_grads(totals, ins, _allow, materialize_grads, bsz)
    totals = _single_grad(outs, ins, grad_outputs, _retain, create_graph)
    return _finalize_grads(totals, ins, _allow, materialize_grads, None)


def _single_grad(
    outs: list[Tensor],
    ins: list[Tensor],
    grad_outputs: Tensor | Sequence[Tensor | None] | None,
    retain_graph: bool,
    create_graph: bool,
) -> list[Tensor | None]:
    """Run one unbatched engine grad pass."""
    seeds = _make_seeds("grad", outs, grad_outputs)
    partials = _C_engine.engine_grad(
        [_unwrap(out) for out in outs],
        [_unwrap(inp) for inp in ins],
        _engine_seeds(seeds, keep_seed_graph=create_graph),
        retain_graph,
        create_graph,
    )
    return [None if impl is None else _wrap(impl) for impl in partials]
