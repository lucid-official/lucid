"""
nn.Parameter: a Tensor that is automatically registered by Module.
"""

import copy
from typing import cast, override

from lucid._tensor.tensor import Tensor
from lucid._C import engine as _C_engine
from lucid._dispatch import _impl_with_grad
from lucid._factories.converters import _to_impl


class Parameter(Tensor):
    r"""A :class:`Tensor` subclass that :class:`Module` recognises as a learnable parameter.

    When an instance is bound as an attribute of a ``Module``, the module's
    ``__setattr__`` sees that it is a ``Parameter`` and routes it into the
    module's ``_parameters`` registry instead of treating it as a plain
    Tensor attribute.  This is what makes ``Module.parameters()`` and
    ``Module.named_parameters()`` discover the tensor automatically — no
    manual registration call is needed.

    By default the underlying tensor has ``requires_grad=True``, so it
    participates in the autograd graph from the moment of construction.
    Pass ``requires_grad=False`` for "buffer-like" learnables (e.g.
    BatchNorm's running mean) that should still be tracked under
    ``state_dict`` but shouldn't accumulate gradients.

    Parameters
    ----------
    data : Tensor | list | None, optional
        Initial values.  ``None`` (default) yields a zero-sized
        placeholder F32/CPU parameter, useful when the actual shape is
        deferred until first ``forward`` (lazy modules).  A ``Tensor``
        is wrapped without a copy: the parameter shares the tensor's
        storage, as in the reference, so an in-place write through one is
        seen by the other (``copy.deepcopy`` is what makes an independent
        one).  A list / nested-list is converted via :func:`_to_impl`.
    requires_grad : bool, optional
        Whether to record gradient flow through this parameter
        (default: ``True``).

    Attributes
    ----------
    _is_parameter : bool
        Class-level marker (always ``True``; ``False`` on ``Tensor``) that
        :mod:`lucid.serialization` reads to tell a parameter from a tensor
        without importing :mod:`lucid.nn`.

    Notes
    -----
    Subclassing ``Tensor`` (rather than wrapping one) keeps every tensor
    operation usable on a ``Parameter`` without unwrap boilerplate — e.g.
    ``p + 1`` returns a ``Tensor``, while ``p.data`` and ``p.grad`` still
    work.  Being a ``Parameter`` is the sole difference visible to
    ``Module``: a registered parameter name takes only a ``Parameter`` or
    ``None``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.nn import Parameter, Module
    >>> class Affine(Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.w = Parameter(lucid.randn(3))  # auto-registered
    ...         self.b = Parameter(lucid.zeros(3))
    >>> m = Affine()
    >>> list(m.named_parameters())   # both surface here automatically
    [('w', Parameter ...), ('b', Parameter ...)]
    """

    _is_parameter: bool = True

    def __new__(
        cls, data: Tensor | list[object] | None = None, requires_grad: bool = True
    ) -> Parameter:
        """Allocate a new :class:`Parameter` instance.

        Bypasses :meth:`Tensor.__init__` (which expects raw data) so a
        Parameter can be built over an existing Tensor without an extra
        copy of metadata.  A ``Tensor`` input is aliased, not copied: the
        Parameter shares its buffer.

        Parameters
        ----------
        data : Tensor | list | None, optional
            See class docstring.
        requires_grad : bool, optional
            See class docstring.

        Returns
        -------
        Parameter
            Fresh instance ready to be bound as a Module attribute.
        """
        if data is None:
            impl = _C_engine.zeros([0], _C_engine.F32, _C_engine.CPU)
            impl = _impl_with_grad(impl, requires_grad)
        elif isinstance(data, Tensor):
            impl = _impl_with_grad(data._impl, requires_grad)
        else:
            impl = _to_impl(data, requires_grad=requires_grad)
        obj = object.__new__(cls)
        obj._impl = impl
        return obj

    def __init__(
        self, data: Tensor | list[object] | None = None, requires_grad: bool = True
    ) -> None:
        """Initialise the instance.  See the class docstring for parameter semantics."""
        pass

    def __deepcopy__(self, memo: dict[int, object]) -> Parameter:
        """Deep-copy into an independent buffer, keeping the subclass.

        ``copy.deepcopy`` on a bare ``Tensor`` subclass falls back to the
        Tensor reconstruction path, which drops the ``Parameter`` marker and
        silently demotes the result to a plain ``Tensor``.  For a ``Module``
        that is fatal: ``deepcopy(model)`` turns every entry of
        ``_parameters`` into a non-Parameter, so the next attribute rebind
        evicts it from the registry and the copy trains as a no-op.

        The data is cloned *before* it is wrapped: ``Parameter(tensor)`` binds
        the tensor's own buffer, so wrapping ``self`` would hand back a
        parameter sharing storage with the original, and a step on the copy
        would move the source (LCD-686).  The copy keeps the concrete class
        (so an :class:`UninitializedParameter` stays pending), its
        ``requires_grad``, and its other instance attributes, each deep-copied
        through ``memo``.  Like the reference, the accumulated ``.grad`` is
        not carried over.  ``memo`` guards shared references and self-cycles
        exactly as the deepcopy protocol expects: a weight tied across two
        modules is copied once and stays tied.
        """
        existing = memo.get(id(self))
        if existing is not None:
            return cast(Parameter, existing)
        result = object.__new__(type(self))
        result._impl = _impl_with_grad(self.data.clone()._impl, self.requires_grad)
        memo[id(self)] = result
        for name, value in self.__dict__.items():
            if name != "_impl":
                result.__dict__[name] = copy.deepcopy(value, memo)
        return result

    @override
    def __repr__(self) -> str:
        """Return a developer-facing string representation of the instance."""
        return f"Parameter containing:\n{super().__repr__()}"


class UninitializedParameter(Parameter):
    """A parameter that exists before its shape does.

    A lazy layer cannot build its weights until it sees an input, but
    ``parameters()`` is read before that — by an optimizer, an EMA, a
    parameter-server shard — and whatever is read then is what those
    hold for the rest of the run.  Returning nothing means they hold
    nothing, and the layer silently never trains.

    So the object is created up front over a zero-element buffer and
    :meth:`materialize` fills it in place.  What matters is that the
    *object* survives: ``id(param)`` is the same before and after, so a
    list captured early still names the weights that eventually exist.

    Notes
    -----
    Lucid's engine optimizers capture ``TensorImpl`` pointers, one level
    below the ``Parameter``, so preserving the Python object is
    necessary but not sufficient — see the deferred binding in
    :class:`~lucid.optim.Optimizer`, which is the other half.
    """

    def materialize(self, data: Tensor) -> None:
        """Take on ``data``'s storage, shape and dtype, in place.

        Parameters
        ----------
        data : Tensor
            The real parameter, already the right shape and initialised.

        Raises
        ------
        RuntimeError
            If ``data`` is bool or an integer dtype, which cannot require
            gradients.

        Notes
        -----
        Rebinds ``_impl`` rather than copying into it: the buffer is a
        different size, and ``copy_from`` is shape-strict by design.  The
        instance's class is rewritten to :class:`Parameter` so nothing
        downstream keeps treating a fully-formed weight as pending.
        """
        self._impl = _impl_with_grad(data._impl, True)
        # Rewriting the class is the point: nothing downstream should keep
        # treating a fully-formed weight as pending.  mypy cannot express
        # an instance narrowing its own type, hence the ignore.
        self.__class__ = Parameter  # type: ignore[assignment]


__all__ = ["Parameter", "UninitializedParameter"]
