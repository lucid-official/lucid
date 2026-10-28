"""
autograd.Function: base class for custom differentiable operations.
"""

from collections.abc import Callable
from typing import Any, ClassVar, NamedTuple, Protocol, cast, override
from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid._engine_scope import untraced
from lucid._tensor.tensor import Tensor
from lucid.autograd._python_node import _register

#: What ``Function.apply`` hands back.
type _Outputs = Tensor | tuple[Tensor, ...]


class _Saved(NamedTuple):
    """One tensor ``save_for_backward`` was given, once the node holds the call.

    Attributes
    ----------
    slot : int
        Where the node's carrier context holds the tensor's impl.
    version : int
        That impl's version once ``forward`` returned.
    where : str
        Where it came from ("Mul input 1"), for the ``VersionMismatch``
        message.
    tensor : Tensor or None
        The object that was saved, handed back while it still has that
        impl; ``None`` for one of the node's own outputs, which the context
        must not hold (:meth:`FunctionCtx._hold_saved`).
    """

    slot: int
    version: int
    where: str
    tensor: Tensor | None


def _impl_of(t: object) -> _C_engine.TensorImpl | None:
    """The impl behind a saved entry, or ``None`` for one that is not a tensor."""
    if isinstance(t, _C_engine.TensorImpl):
        return t
    if isinstance(t, Tensor):
        return t._impl
    return None


def _position(impl: _C_engine.TensorImpl, impls: list[_C_engine.TensorImpl | None]) -> int | None:
    """Index of ``impl`` in ``impls`` by identity, or ``None``."""
    for k, other in enumerate(impls):
        if other is impl:
            return k
    return None


class _FunctionClass(Protocol):
    """Protocol describing a Function subclass (has forward/backward classmethods)."""

    @classmethod
    def forward(
        cls, ctx: object, *args: object, **kwargs: object
    ) -> Tensor | tuple[Tensor, ...]: ...

    @classmethod
    def backward(
        cls, ctx: object, *grad_outputs: Tensor
    ) -> Tensor | tuple[Tensor | None, ...] | list[Tensor | None]: ...


class FunctionCtx:
    r"""Per-call context shared between :meth:`Function.forward` and
    :meth:`Function.backward`.

    A fresh ``FunctionCtx`` is created on every :meth:`Function.apply`
    invocation. ``forward`` populates it with anything ``backward``
    will need: saved tensors (via :meth:`save_for_backward`),
    non-differentiable output markers (via
    :meth:`mark_non_differentiable`), and arbitrary user-defined
    attributes (cached shapes, axis indices, scalar hyperparameters,
    ...) set with ordinary attribute assignment. The context is the
    only legal channel for passing state from forward to backward —
    capturing tensors through Python closures bypasses autograd's
    bookkeeping and leaks memory.

    Parameters
    ----------
    None
        ``FunctionCtx`` is instantiated by :meth:`Function.apply`
        with no arguments. User code never constructs one
        directly; it receives the instance as the first
        positional argument of ``forward`` / ``backward``.

    Attributes
    ----------
    needs_input_grad : tuple of bool
        One flag per positional ``Tensor`` input to ``forward``,
        indicating whether autograd would propagate a gradient to
        that input. Use this to skip unneeded branches in
        ``backward``.
    saved_tensors : tuple of Tensor
        Read-only view of the tensors stored via
        :meth:`save_for_backward`, in registration order.

    Methods
    -------
    save_for_backward(\*tensors)
        Persist tensors needed for the backward pass.
    mark_non_differentiable(\*outputs)
        Declare specific outputs as carrying no gradient (e.g.
        integer indices, masks).
    set_materialize_grads(value)
        Reserved hook for controlling whether ``None`` upstream
        gradients are materialised as zero tensors before
        ``backward`` is invoked.

    Notes
    -----
    The context is what ties the forward and backward halves of a
    custom node together in the chain rule:

    .. math::

        \mathbf{y} = f(\mathbf{x}; \text{ctx}),
        \qquad
        \bar{\mathbf{x}} = g(\mathbf{ctx}, \bar{\mathbf{y}}),

    where :math:`\bar{\mathbf{y}} = \partial \mathcal{L} /
    \partial \mathbf{y}` is the upstream gradient and the same
    ``ctx`` object is passed to both halves so :math:`g` can read
    back whatever :math:`f` saved.

    Examples
    --------
    >>> import lucid
    >>> from lucid.autograd import Function
    >>> class Square(Function):
    ...     @staticmethod
    ...     def forward(ctx, x):
    ...         ctx.save_for_backward(x)
    ...         ctx.shape = x.shape
    ...         return x * x
    ...     @staticmethod
    ...     def backward(ctx, grad_out):
    ...         (x,) = ctx.saved_tensors
    ...         return 2 * x * grad_out
    """

    def __init__(self) -> None:
        """Initialise an empty context with no saved tensors or extras."""
        # What ``save_for_backward`` was given.  Once the node is registered
        # (``_hold_saved``), only the values that are not tensors (``None``,
        # say) are left here, in order: the node's carrier holds the tensors.
        self._saved_tensors: list[Tensor] = []
        # Written at registration: one entry per saved value, a ``_Saved``
        # for a tensor and ``None`` for a value that is not one (the next
        # left in ``_saved_tensors``).  ``None`` before that, so a
        # ``forward`` reading its own ``saved_tensors`` gets the objects it
        # saved, unchecked.
        self._saved_record: list[_Saved | None] | None = None
        self._carrier: _C_engine.FunctionCtx | None = None
        self.needs_input_grad: tuple[bool, ...] = ()
        self._non_differentiable: list[Tensor] = []
        self._extra: dict[str, object] = {}

    def save_for_backward(self, *tensors: Tensor) -> None:
        """Store tensors needed to compute the backward pass.

        Parameters
        ----------
        *tensors : Tensor
            Activations / inputs the ``backward`` implementation will
            require. They are retrieved later via the :attr:`saved_tensors`
            property as a tuple in the same order.

        Notes
        -----
        Each call replaces any tensors previously saved on this context.

        A saved tensor must not be written in place between the end of
        ``forward`` and the ``backward`` that reads it: reading
        :attr:`saved_tensors` then raises ``VersionMismatch``, as a built-in
        op's backward does, instead of handing ``backward`` the new values.
        Writes inside ``forward`` itself are fine — the version is taken
        once ``forward`` has returned.

        ``forward``'s own outputs may be saved too: the engine context holds
        their data and a weak reference to the node (re-attaching the node on
        read), so a call that runs no backward is freed by refcounting the
        moment its last output is dropped.  A tensor assigned to an attribute
        of the context (``ctx.out = y``) gets none of this — save tensors
        through this method.
        """
        self._saved_tensors = list(tensors)

    def _hold_saved(
        self,
        name: str,
        inputs: tuple[object, ...],
        returned: list[object],
        on_node: list[Tensor],
        carrier: _C_engine.FunctionCtx,
    ) -> None:
        """Hand the saved tensors to the node's carrier now that it exists.

        Parameters
        ----------
        name : str
            The ``Function`` subclass's name, for the error message.
        inputs : tuple of object
            ``forward``'s positional arguments.
        returned : list of object
            What ``forward`` returned, one entry per result.
        on_node : list of Tensor
            The tensors registered as the node's outputs.
        carrier : FunctionCtx
            The engine context the node owns (``_register``).

        Notes
        -----
        Each version is taken here, after ``forward`` — the reference
        framework saves at the same point — so a ``forward`` that writes a
        tensor in place and then returns it is not refused for its own
        write.

        Every saved impl goes to ``carrier``, and this context keeps no
        reference to one of the node's outputs.  ``carrier`` stores an output
        as a detached storage alias plus a weak reference to the node and its
        output index (re-attaching the node when :attr:`saved_tensors` is
        read), so dropping the outputs of a forward-only call frees both the
        outputs and the node immediately by refcount (LCD-588, LCD-602).
        """
        input_impls = [_impl_of(a) for a in inputs]
        output_impls = [_impl_of(o) for o in returned]
        node_impls: list[_C_engine.TensorImpl | None] = [t._impl for t in on_node]
        record: list[_Saved | None] = []
        held: list[_C_engine.TensorImpl] = []
        others: list[Tensor] = []
        for i, t in enumerate(self._saved_tensors):
            impl = _impl_of(t)
            if impl is None:
                record.append(None)
                others.append(t)
                continue
            # Named the way the engine names a node's input — "mul input 1" —
            # by the position the caller knows it under.
            k = _position(impl, input_impls)
            j = _position(impl, output_impls)
            if k is not None:
                where = f"{name} input {k}"
            elif j is not None:
                where = f"{name} output {j}"
            else:
                where = f"{name} saved tensor {i}"
            own = _position(impl, node_impls) is not None
            keep = t if isinstance(t, Tensor) and not own else None
            record.append(_Saved(len(held), impl.version, where, keep))
            held.append(impl)
        carrier.save_for_backward(*held)
        self._carrier = carrier
        self._saved_record = record
        self._saved_tensors = others

    @property
    def saved_tensors(self) -> tuple[Tensor, ...]:
        """Read back the tensors saved during ``forward``.

        Returns
        -------
        tuple of Tensor
            The tensors stored by :meth:`save_for_backward`, wrapped
            back into Python ``Tensor`` instances if the engine stored
            raw ``TensorImpl`` handles.

        Raises
        ------
        VersionMismatch
            A saved tensor was written in place after ``forward``
            returned — its values are no longer the ones ``forward``
            saw, so a gradient computed from them would be wrong.
            Suppressed inside
            ``lucid.autograd.graph.allow_mutation_on_saved_tensors``.
        """
        record, carrier = self._saved_record, self._carrier
        if record is None or carrier is None:
            return tuple(
                _wrap(t) if isinstance(t, _C_engine.TensorImpl) else t for t in self._saved_tensors
            )
        check = not _C_engine.is_mutation_on_saved_allowed()
        held = carrier.saved_tensors
        others = iter(self._saved_tensors)
        result: list[Tensor] = []
        for entry in record:
            if entry is None:
                result.append(next(others))
                continue
            impl = held[entry.slot]
            if check and impl.version != entry.version:
                # Worded as the engine's own ``check_version_match`` words it.
                raise _C_engine.VersionMismatch(
                    f"VersionMismatch ({entry.where}): saved version "
                    f"{entry.version} but tensor is now at version {impl.version} "
                    "(in-place mutation between forward and backward?)"
                )
            # What was saved is the impl, as for a built-in op: a write that
            # rebinds the Python tensor to a new impl (``t[i] = v`` on a
            # tensor that needs no grad) leaves the saved values where they
            # were, and ``backward`` reads those, not the new ones.  An
            # output comes back as a new tensor over its impl, ``grad_fn``
            # and all, so a second derivative still runs through the node.
            t = entry.tensor
            result.append(t if t is not None and t._impl is impl else _wrap(impl))
        return tuple(result)

    def mark_non_differentiable(self, *tensors: Tensor) -> None:
        """Declare that the given output tensors carry no gradient.

        Parameters
        ----------
        *tensors : Tensor
            Outputs of :meth:`Function.forward` for which autograd should
            not propagate gradients (e.g., integer indices, masks).

        Notes
        -----
        :meth:`Function.apply` hands a marked output back detached:
        ``requires_grad=False`` and no ``grad_fn``, and a compiled step
        sends no gradient through it either.  ``backward`` is still given a
        gradient for it — zeros — so its arguments keep one slot per
        output.  What comes back is a new tensor over the same storage, and
        the tensor ``forward`` returned is left as it was: a marked input
        returned as-is keeps the caller's ``requires_grad``, and a buffer
        ``forward`` read from a module keeps its place there.

        Unlike the reference framework, that holds for every marked output.
        The reference hands back the very object ``forward`` returned when
        it needs no gradient, and detaches one that does in place: a leaf
        parameter ``forward`` closed over, returned and marked, loses its
        ``requires_grad`` there.  Here the parameter keeps it, and the
        caller gets a new detached tensor instead.

        An output of a dtype that carries no gradient — bool or an integer
        — is treated as marked without being named here, as the reference
        framework treats it.

        Examples
        --------
        >>> import lucid
        >>> from lucid.autograd import Function
        >>> class ScaleAndMask(Function):
        ...     @staticmethod
        ...     def forward(ctx, x, mask):
        ...         ctx.mark_non_differentiable(mask)
        ...         return x * 2, mask
        ...     @staticmethod
        ...     def backward(ctx, grad, grad_mask):
        ...         return grad * 2, None
        >>> x = lucid.ones(2, requires_grad=True)
        >>> mask = lucid.ones(2, requires_grad=True)
        >>> y, out = ScaleAndMask.apply(x, mask)
        >>> out.requires_grad, mask.requires_grad
        (False, True)
        """
        self._non_differentiable = list(tensors)

    @override
    def __setattr__(self, name: str, value: object) -> None:
        """Route user-defined attributes onto the ``_extra`` dict.

        Reserved private names and ``needs_input_grad`` use normal
        instance storage so they remain accessible through descriptors;
        everything else falls back to the extras bag, keeping the
        context permissive while preserving the public schema.
        """
        if name.startswith("_") or name in ("needs_input_grad",):
            object.__setattr__(self, name, value)
        else:
            try:
                object.__setattr__(self, name, value)
            except AttributeError:
                self._extra[name] = value

    def __getattr__(self, name: str) -> object:
        """Look up overflow attributes that were stored in ``_extra``.

        Called only when normal attribute resolution fails, so it
        complements :meth:`__setattr__` without shadowing class members.
        """
        extra = object.__getattribute__(self, "_extra")
        if name in extra:
            return extra[name]
        raise AttributeError(f"FunctionCtx has no attribute '{name}'")


def _owned_outputs(
    entries: list[object],
    ctx: FunctionCtx,
    input_impls: list[_C_engine.TensorImpl | None],
    tracked: bool,
) -> tuple[list[object], list[Tensor], list[bool]]:
    """``forward``'s results, with every non-differentiable or aliased one settled.

    An output is non-differentiable when ``forward`` marked it, or when
    its dtype carries no gradient — bool or an integer — as the reference
    framework decides it.  Each comes back as the detached alias
    :meth:`Tensor.detach` hands out: a new tensor over the same storage,
    which a compiled step traces as a gradient sink.  The tensor
    ``forward`` returned is left alone — a caller's input keeps its
    ``requires_grad``, a module buffer its place — where detaching it in
    place rebound whatever else held it.

    An unmarked input returned as-is (or a tensor returned more than once)
    comes back as a view of it (``TensorImpl.custom_function_view``),
    leaving the caller's tensor untouched; when ``tracked`` is ``True``, that
    view is marked with ``is_custom_function_output`` so in-place writes to it
    or its base are refused as in the reference (LCD-558).

    Parameters
    ----------
    entries : list of object
        What ``forward`` returned, one entry per result.
    ctx : FunctionCtx
        The context ``forward`` filled in, with its marks.
    input_impls : list of TensorImpl or None
        The ``TensorImpl`` handles of ``apply``'s positional and keyword
        arguments.
    tracked : bool
        Whether the call records a backward node.

    Returns
    -------
    tuple
        The settled entries, their tensors in order, and whether each tensor
        is non-differentiable.
    """
    marked_impls: list[_C_engine.TensorImpl | None] = [t._impl for t in ctx._non_differentiable]
    seen_output_impls: list[_C_engine.TensorImpl | None] = []
    owned: list[object] = []
    tensors: list[Tensor] = []
    marked: list[bool] = []
    for entry in entries:
        if isinstance(entry, Tensor):
            # Marks name what ``forward`` returned, so they are read before
            # the output is swapped for its alias.
            impl = entry._impl
            named = _position(impl, marked_impls) is not None
            is_marked = named or _promotion.is_integral(impl.dtype)
            if is_marked:
                entry = _wrap(impl.data_alias())
            elif (
                _position(impl, input_impls) is not None
                or _position(impl, seen_output_impls) is not None
            ):
                entry = _wrap(impl.custom_function_view(tracked))
            seen_output_impls.append(impl)
            marked.append(is_marked)
            tensors.append(entry)
        owned.append(entry)
    return owned, tensors, marked


def _wrap_outputs(
    output: object,
    klass: type,
    ctx: FunctionCtx,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    tracked: bool,
) -> object:
    """What ``apply`` hands back for what ``forward`` returned.

    A non-differentiable output comes back as a new detached tensor
    (:func:`_owned_outputs`).  When the call is tracked, the node is
    registered and handed the saved tensors (:meth:`FunctionCtx._hold_saved`).

    Parameters
    ----------
    output : object
        What ``forward`` returned: a tensor, or a tuple that may hold
        non-tensor entries alongside the tensors.
    klass : type
        The ``Function`` subclass, whose ``backward`` the node calls.
    ctx : FunctionCtx
        The context ``forward`` filled in.
    args : tuple of object
        ``forward``'s positional arguments, the ones gradients go to.
    kwargs : dict of str to object
        ``forward``'s keyword arguments.
    tracked : bool
        Whether this call is recorded — grad mode is on and an input needs
        a gradient.

    Returns
    -------
    object
        ``output`` with each non-differentiable tensor entry detached and, if
        tracked, the backward node attached to the rest.
    """
    is_tuple = isinstance(output, tuple)
    returned = list(output) if isinstance(output, tuple) else [output]
    input_impls = [_impl_of(a) for a in (*args, *kwargs.values())]
    entries, tensors, marked = _owned_outputs(returned, ctx, input_impls, tracked)
    if tracked and not all(marked):
        # ``backward`` is handed one gradient per *tensor* output, zeros for
        # a non-differentiable one.  Its slot goes to a stand-in alias that
        # dies here, so neither what ``forward`` returned nor the detached
        # tensor handed back gets the node.  The stand-in alias is untraced so
        # it adds no extra ``detach`` op to a grad-mode trace (LCD-612).
        with untraced():
            slots = [
                _wrap(t._impl.data_alias()) if is_marked else t
                for t, is_marked in zip(tensors, marked, strict=True)
            ]
        # Which positional arguments were tensors, so ``backward`` may
        # answer with one gradient per argument (``None`` for the rest).
        is_tensor_arg = [isinstance(a, Tensor) for a in args]
        carrier = _register(
            tuple(slots) if is_tuple else slots[0],
            klass,  # type: ignore[arg-type]
            ctx,
            [a for a in args if isinstance(a, Tensor)],
            is_tensor_arg,
        )
        # ``backward`` will read the saved tensors now: note their versions,
        # as a built-in op does when it saves its inputs, so a write in
        # between is refused there.
        ctx._hold_saved(klass.__name__, args, returned, slots, carrier)
    return tuple(entries) if is_tuple else entries[0]


def _prep_traced_vjp(
    tracer: _C_engine.compile.Tracer,
    ctx: FunctionCtx,
    args: tuple[object, ...],
    output: object,
    bounds: tuple[int, int, bool],
) -> tuple[object, tuple[FunctionCtx, tuple[Tensor, ...], list[bool], int]]:
    """Ensure traced custom-VJP outputs have distinct post-``id_begin`` ids."""
    fwd_begin, id_begin, tracked = bounds
    entries = list(output) if isinstance(output, tuple) else [output]
    marked_impls: list[_C_engine.TensorImpl | None] = [t._impl for t in ctx._non_differentiable]
    input_impls = [_impl_of(a) for a in args]
    marked: list[bool] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, Tensor):
            continue
        impl = entry._impl
        is_marked = _position(impl, marked_impls) is not None or _promotion.is_integral(impl.dtype)
        marked.append(is_marked)
        tid = tracer.lookup_id(impl)
        if tid is None or int(tid) < id_begin:
            tracer.ensure_id(impl)
            if not is_marked and _position(impl, input_impls) is None:
                entries[idx] = _wrap(impl.custom_function_view(tracked))
    raw_saved = tuple(ctx._saved_tensors)
    updated = tuple(entries) if isinstance(output, tuple) else entries[0]
    return updated, (ctx, raw_saved, marked, fwd_begin)


def _lucid_defined(klass: type) -> bool:
    """Whether ``klass`` is defined in a ``lucid.`` module outside ``lucid.test``.

    The rule every private ``Function`` marker is trusted by
    (:attr:`Function._trace_form`, :attr:`Function._exact_derivative`): a
    marker changes what a compile recording keeps of the call, so only a
    class whose ``backward`` Lucid wrote and tests may set one.  A user's
    class, or a test's, that sets the attribute is treated as unmarked.
    ``__module__`` is the class's own claim, so this keeps an ordinary
    subclass from opting in by accident; it is not a security boundary.
    """
    module = klass.__module__
    if module != "lucid" and not module.startswith("lucid."):
        return False
    return module != "lucid.test" and not module.startswith("lucid.test.")


def _trace_form_of(klass: type[Function]) -> Callable[..., _Outputs] | None:
    """``klass``'s trusted :attr:`Function._trace_form`, or ``None``."""
    return klass._trace_form if _lucid_defined(klass) else None


def _is_exact_derivative(klass: type[Function]) -> bool:
    """Whether ``klass`` carries a trusted :attr:`Function._exact_derivative`."""
    return klass._exact_derivative and _lucid_defined(klass)


def _make_apply() -> classmethod:  # type: ignore[type-arg]
    """Build the per-subclass ``apply`` classmethod injected by ``FunctionMeta``."""

    def apply(klass: type[Function], *args: object, **kwargs: object) -> _Outputs:
        """Run ``forward`` and register the autograd node when needed."""
        tracer = _C_engine.compile.current_tracer()
        if tracer is not None and (trace_form := _trace_form_of(klass)) is not None:
            # The node exists for the eager graph's version checks, and a
            # compile cannot differentiate through it: the recording takes
            # the plain ops, in the caller's grad mode.
            return trace_form(*args, **kwargs)
        ctx = FunctionCtx()
        ctx.needs_input_grad = tuple(isinstance(a, Tensor) and a.requires_grad for a in args)
        tracked = _C_engine.grad_enabled() and any(ctx.needs_input_grad)
        bounds: tuple[int, int, bool] | None = None
        if (
            tracer is not None
            and not tracer.fold_constants
            and getattr(klass, "_traced_backward", False)
        ):
            for a in args:
                if isinstance(a, Tensor):
                    tracer.ensure_id(a._impl)
            bounds = (int(tracer.op_count()), int(tracer.graph.next_id), tracked)

        # Under ``no_grad``, as the reference runs it: the node registered
        # below is the call's whole graph.  A graph ``forward`` built itself
        # was thrown away by every tracked call and kept by an untracked one,
        # which sent a gradient to whatever tensor ``forward`` closed over.
        with _C_engine.NoGradGuard():
            output = klass.forward(ctx, *args, **kwargs)
        vjp_call: tuple[FunctionCtx, tuple[Tensor, ...], list[bool], int] | None = None
        if tracer is not None and bounds is not None:
            output, vjp_call = _prep_traced_vjp(tracer, ctx, args, output, bounds)
        settled = _wrap_outputs(output, klass, ctx, args, kwargs, tracked)
        if tracer is not None:
            # Imported here: ``lucid.compile`` imports this package.
            from lucid.compile._core.custom_backward import note_applied

            note_applied(tracer, klass, args, settled, vjp_call=vjp_call)

        return cast(_Outputs, settled)

    return classmethod(apply)  # type: ignore[arg-type]


class FunctionMeta(type):
    """Metaclass that wires a fresh :meth:`apply` onto each ``Function`` subclass.

    The base ``Function`` class is left untouched (so ``Function.apply`` keeps
    its explanatory stub); every concrete subclass receives a closure that
    instantiates a :class:`FunctionCtx`, calls the subclass's ``forward``,
    and — when grad is enabled and any input requests it — registers the
    autograd node.
    """

    def __init__(cls, name: str, bases: tuple[type, ...], dct: dict[str, object]) -> None:
        """Install the dispatching :meth:`apply` on the freshly built subclass."""
        super().__init__(name, bases, dct)
        if name != "Function":
            cls.apply = _make_apply()


class Function(metaclass=FunctionMeta):
    r"""Base class for custom differentiable operations.

    Subclass to define an operation whose forward and backward
    passes Lucid's autograd cannot deduce automatically — for
    example, an operation that wraps an external library, an op
    with a custom backward formula for numerical stability, or a
    piecewise-defined function whose gradient differs from naïve
    autograd.

    Define :py:meth:`forward` and :py:meth:`backward` as
    ``@staticmethod`` on the subclass, then invoke the op via
    :py:meth:`apply` (NOT by calling ``forward`` directly — that
    would skip autograd registration).

    Parameters
    ----------
    None
        ``Function`` itself is never instantiated. Subclasses
        are stateless — their behaviour is defined entirely by
        the ``forward`` / ``backward`` static methods, and ops
        are invoked through the :py:meth:`apply` classmethod
        rather than ``__init__``.

    Attributes
    ----------
    forward : staticmethod
        ``forward(ctx, *args, **kwargs) -> Tensor or tuple[Tensor, ...]``.
        Computes the primal value; stores anything ``backward`` will
        need on ``ctx``.
    backward : staticmethod
        ``backward(ctx, *grad_outputs) -> tuple of Tensors (or None per
        non-tensor input)``. Returns the cotangents matching the
        positional inputs of ``forward``.
    apply : classmethod
        Bind ``forward`` to the autograd graph, returning the forward
        output and registering ``backward`` as the gradient closure.
    _traced_backward : bool, default=False
        Class attribute opting the subclass into traced custom-VJP lowering
        under :mod:`lucid.compile`.  When ``True`` and both ``forward`` and
        ``backward`` consist of traceable tensor operations on GPU,
        ``make_step``, ``fused_step``, and ``compiled_step`` trace
        ``backward`` into a ``VjpRegion`` inside the compiled executable
        instead of falling back to eager execution.

    Notes
    -----
    The :py:class:`FunctionCtx` passed to ``forward`` carries the
    bookkeeping autograd needs to call ``backward`` later — saved
    tensors plus per-input non-differentiable flags.

    ``forward`` runs with gradient recording off, as in the reference
    framework: the node ``apply`` registers is the whole graph of the call,
    and a tensor ``forward`` computes needs no gradient of its own.

    An output that is non-differentiable — marked so, or of a bool or
    integer dtype — comes back as a new, detached tensor over the same
    storage, and the tensor ``forward`` returned is left as it was: an
    input passed through keeps the caller's ``requires_grad``.  Unlike the
    reference framework, which hands back the very object ``forward``
    returned when it needs no gradient and detaches a closed-over leaf in
    place, every such output here is a new tensor.  An unmarked
    input returned as-is comes back as the same tensor, now the node's
    output: where the reference framework hands back a view of it, the
    caller's tensor here is no longer a leaf after the call.

    Each :py:class:`Function` defines a node in the computation graph

    .. math::

        \mathbf{y} = f(\mathbf{x}_1, \ldots, \mathbf{x}_n; \theta)

    and supplies the chain-rule contribution

    .. math::

        \frac{\partial \mathcal{L}}{\partial \mathbf{x}_i}
        = \sum_j \frac{\partial \mathcal{L}}{\partial y_j}
          \cdot \frac{\partial y_j}{\partial x_{i}}

    for each input :math:`\mathbf{x}_i`. The ``backward`` method
    implements precisely this Jacobian-vector product.

    Examples
    --------
    Custom ReLU with an explicit backward:

    >>> import lucid
    >>> from lucid.autograd import Function
    >>> class MyReLU(Function):
    ...     @staticmethod
    ...     def forward(ctx, x):
    ...         ctx.save_for_backward(x)
    ...         return x.clamp(min=0.0)
    ...     @staticmethod
    ...     def backward(ctx, grad_out):
    ...         (x,) = ctx.saved_tensors
    ...         return grad_out * (x > 0).float()
    >>> y = MyReLU.apply(lucid.tensor([-1.0, 2.0], requires_grad=True))
    >>> y.sum().backward()
    """

    #: What a ``lucid.compile`` recording runs in place of the node: a
    #: callable taking ``apply``'s arguments and returning ``forward``'s
    #: values from plain ops, in the caller's grad mode, whose own derivative
    #: is the one ``backward`` gives.  For a node that exists only for the
    #: eager graph's bookkeeping — saving nothing, so an in-place write after
    #: it is not refused — whose own node a compile cannot differentiate
    #: through.  The plain ops must save no tensor the traced code can write
    #: either, or an eager backward run inside the recording (``autograd.grad``
    #: of an energy, a gradient penalty) refuses the write the node allowed:
    #: they read graph-carrying copies of the operands.  ``apply`` then
    #: registers no node and the call is not reported to the compile's
    #: custom-backward check.  Trusted only on a
    #: class defined in a ``lucid.`` module outside ``lucid.test``
    #: (:func:`_lucid_defined`); on any other class it is ignored.
    _trace_form: ClassVar[Callable[..., _Outputs] | None] = None

    #: Whether the derivative a compile takes of the recorded ``forward`` is
    #: ``backward``: the recorded ops' VJP equals it, and ``forward`` reads no
    #: tensor that carries a gradient besides its positional inputs (or
    #: ``backward`` answers for that tensor too), so a compiled backward may
    #: differentiate the recording instead of calling ``backward``.  Trusted
    #: only as :attr:`_trace_form` is.
    _exact_derivative: ClassVar[bool] = False

    #: Whether ``lucid.compile`` may trace :meth:`backward` into the compiled
    #: graph as a custom VJP region when ``apply`` runs inside a recording.
    #: Both ``forward`` and ``backward`` must be pure traceable Lucid tensor
    #: operations with no host-side control flow or side effects on runtime
    #: tensor values.
    _traced_backward: ClassVar[bool] = False

    # The base signature is ``(ctx, *Any, **Any) -> Any`` on purpose: every
    # subclass declares its own concrete parameters and return type, and only
    # this shape is override-compatible with all of them (LSP), so subclasses
    # need no ``ignore[override]``.  The runtime contract is documented below
    # and enforced by ``apply`` / the backward node, not by this annotation.
    @staticmethod
    def forward(ctx: FunctionCtx, *args: Any, **kwargs: Any) -> Any:
        """Compute the forward result of the custom op.

        Subclasses override this static method. Save anything needed
        during backward on ``ctx`` (via ``ctx.save_for_backward(...)``
        or by setting attributes like ``ctx.shape = x.shape``); do not
        capture tensors via closure.

        It runs under ``no_grad``: ``lucid.is_grad_enabled()`` is ``False``
        inside it and the tensors it computes record no graph.  A
        ``forward`` that differentiates something itself does so inside
        ``lucid.enable_grad()``.

        Parameters
        ----------
        ctx : FunctionCtx
            Per-call context; will be passed unchanged to :meth:`backward`.
        *args : Tensor
            Positional inputs to the custom op. Non-Tensor positional
            arguments are allowed but receive no gradients.
        **kwargs
            Keyword inputs forwarded unchanged from :meth:`apply`; they
            receive no gradients.

        Returns
        -------
        Tensor or tuple of Tensor
            One or more output tensors.
        """
        raise NotImplementedError

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_outputs: Any) -> Any:
        r"""Compute gradients of the loss w.r.t. each input of :meth:`forward`.

        Subclasses override this static method. The returned tuple must
        contain one gradient per **positional argument** of ``forward``,
        in the same order; non-Tensor arguments receive ``None``.

        Parameters
        ----------
        ctx : FunctionCtx
            The same context that was populated during ``forward``.
        *grad_outputs : Tensor
            Upstream gradients :math:`\partial L / \partial y_i`, one per
            output that ``forward`` returned.

        Returns
        -------
        Tensor or tuple of Tensor
            Downstream gradients matching the positional inputs of
            ``forward``. Use ``None`` for inputs that have no gradient.
            One gradient per *tensor* input — leaving out the non-tensor
            arguments — is accepted as well, and trailing ``None``
            entries past the last argument are ignored.

        Raises
        ------
        TypeError
            When a gradient is neither a tensor nor ``None``.
        RuntimeError
            When the number of gradients fits neither form, when a
            non-tensor argument is given a gradient other than ``None``,
            or when a gradient does not fit its input: a shape the input
            does not broadcast to, or a device other than the input's.

        Notes
        -----
        Each gradient is checked against the input it is for, once
        ``backward`` returns.  A shape the input broadcasts to is summed
        back down to the input's shape, another dtype is cast to the
        input's (a complex gradient for a real input keeps its real part),
        and a 0-d gradient on another device is moved to the input's.
        """
        raise NotImplementedError

    @classmethod
    def apply(cls, *args: object, **kwargs: object) -> object:
        """Run the function — autograd-tracked when ``forward`` requests grad.

        ``FunctionMeta`` rewrites this on every concrete subclass to dispatch
        through ``forward``/``backward``; the base implementation here exists
        only so ``Function.apply`` is a real attribute (matters for tooling
        and ``hasattr`` checks).  Calling it on the base class is meaningless
        and raises.
        """
        raise NotImplementedError(
            "Function.apply must be called on a concrete subclass that "
            "implements forward / backward."
        )
