from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, ClassVar, Self, Iterator, final, overload

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import DTypeLike as NpDTypeLike
else:

    class _LazyNumpy:
        """Stands in for ``numpy`` until an annotation actually asks.

        ``Tensor.__init__`` accepts an ``np.ndarray``, and says so. Under
        PEP 649 that annotation is evaluated in this module's globals the
        moment anyone reads the signature — ``help()``, an editor, the
        documentation build — so a name bound only for type checking
        makes the constructor of the package's central class impossible
        to describe.

        Importing numpy here instead would make an optional dependency a
        required one: this file is a sanctioned bridge, but the bridge is
        deliberately lazy so that an install without numpy still works.
        Resolving on attribute access keeps both — nothing is imported
        until something reads the annotation, and then only what a caller
        holding a numpy array would have imported anyway.
        """

        def __getattr__(self, name: str) -> object:
            import numpy

            return getattr(numpy, name)

    np, NpDTypeLike = _LazyNumpy(), object

from lucid._C import engine as _C_engine
from lucid._dtype import (
    dtype as _dtype_cls,
    _ENGINE_TO_DTYPE,
    float16,
    float32,
    float64,
    bfloat16,
    complex64,
    complex128,
)
from lucid._device import device as _device_cls, _device_from_engine
from lucid._dispatch import (
    _check_grad_dtype,
    _impl_with_grad,
    _read_grad,
    _set_grad,
    _swap_data,
    _wrap,
)
from lucid._factories.creation import (
    zeros as _zeros,
    ones as _ones,
    empty as _empty,
    full as _full,
)
from lucid._factories.converters import tensor as _tensor_fn, _to_impl

if TYPE_CHECKING:
    from lucid.autograd._hooks import RemovableHandle as _RemovableHandle


# DLPack device type for Metal — what a GPU-resident tensor reports and
# what MLX tags its own capsules with.
_DLPACK_METAL = 8


def _set_requires_grad(t: Tensor, requires_grad: bool, *, setter: bool) -> None:
    """Change ``t``'s ``requires_grad`` flag in place, by the leaf rules.

    The flag flips on the tensor's own ``TensorImpl``.  This used to swap in
    a ``clone_with_grad`` alias, and every holder of the old impl was left
    behind: an engine optimizer built before the swap kept stepping a tensor
    that no longer received gradients, so a parameter frozen and unfrozen
    after the first ``step()`` (a GAN discriminator, a backbone) silently
    stopped training.  The gradient already accumulated stayed on the old
    impl too; in place, ``.grad`` survives a toggle.

    Only a leaf's flag is the user's to change.  A computed tensor's flag
    follows its graph: clearing it would not detach the tensor, so that is
    refused, as is the property setter on any computed tensor.
    ``requires_grad_(True)`` on one is a no-op, since it already requires
    grad.

    Parameters
    ----------
    t : Tensor
        Tensor whose flag changes.
    requires_grad : bool
        New flag value.
    setter : bool
        ``True`` for the property setter, which refuses any computed
        tensor; ``False`` for :meth:`Tensor.requires_grad_`.

    Raises
    ------
    RuntimeError
        If ``t`` was computed by an operation and the change is refused, or
        if ``requires_grad`` is set on a bool or integer tensor.
    """
    impl = t._impl
    if not isinstance(impl, _C_engine.TensorImpl):
        t._impl = _impl_with_grad(impl, requires_grad)
        return
    _check_grad_dtype(impl.dtype, requires_grad)
    if not impl.is_leaf:
        if requires_grad and not setter:
            return
        hint = (
            ""
            if requires_grad
            else "; to use a computed tensor without gradient tracking, "
            "take x.detach()"
        )
        raise RuntimeError(f"requires_grad can only be changed on a leaf tensor{hint}.")
    impl._set_requires_grad(requires_grad)


def _write_part(part: Tensor, value: Tensor | float) -> None:
    """Assign ``value`` to ``part`` — the view ``Tensor.real`` or
    ``Tensor.imag`` returned — in place, broadcast to its shape.

    The view shares the complex tensor's storage, so the write lands in it;
    a real tensor's real part is the tensor itself.
    """
    if isinstance(value, Tensor):
        part.copy_(value)
    else:
        part.fill_(value)


class Tensor:
    """Multi-dimensional array with automatic differentiation support.

    The central data structure of the Lucid framework. Wraps a C++ ``TensorImpl``
    via composition. Tensors live on either ``cpu`` (Apple Accelerate) or
    ``metal`` (Apple Metal GPU) devices.

    Parameters
    ----------
    data : array_like
        Input data. Accepts nested Python lists, NumPy arrays, or scalars.
    dtype : lucid.dtype, optional
        Desired data type. Defaults to ``lucid.float32``.
    device : str or lucid.device, optional
        Target device (``"cpu"`` or ``"metal"``). Defaults to the global default.
    requires_grad : bool, optional
        If ``True``, operations on this tensor are recorded for autograd.
        Default is ``False``.

    Examples
    --------
    >>> import lucid
    >>> x = lucid.Tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> x.shape
    (2, 2)
    >>> x.dtype
    lucid.float32
    """

    _is_parameter: ClassVar[bool] = False
    __lucid_function__: ClassVar[None] = None

    def __init__(
        self,
        data: np.ndarray | list[object] | int | float | bool | Tensor,
        *,
        # Spelled through the aliases, not the bare names: the class body
        # defines ``dtype`` and ``device`` properties, and PEP 649
        # evaluates these annotations with the class namespace in scope,
        # so the bare names resolve to those properties and
        # ``inspect.signature(Tensor)`` raises rather than answering.
        dtype: _dtype_cls | _C_engine.Dtype | str | None = None,
        device: _device_cls | str | None = None,
        requires_grad: bool = False,
    ) -> None:
        r"""Construct a Tensor from Python data, a NumPy array, or another Tensor.

        The data is funnelled through :func:`lucid._factories.converters._to_impl`,
        the canonical bridge between the outside world and Lucid's C++
        ``TensorImpl``. Python scalars become 0-d tensors; nested lists are
        recursively flattened with shape inference; NumPy arrays cross the
        single sanctioned host-to-engine boundary; and existing ``Tensor``
        sources are rewrapped (optionally with cast and/or device transfer).

        Parameters
        ----------
        data : ndarray | list | int | float | bool | Tensor
            Source data. Python scalars become 0-d tensors; lists are
            recursively converted; NumPy arrays use the bridge boundary
            documented in :mod:`lucid._factories.converters`; another Tensor
            is shallow-copied to a new Tensor with possibly different
            ``dtype`` / ``device`` / ``requires_grad``.
        dtype : dtype | str | None, optional
            Element type. If ``None``, inferred from ``data``.
        device : device | str | None, optional
            Target device (``"cpu"`` or ``"metal"``). If ``None``, defaults
            to the source device or ``"cpu"`` for host data.
        requires_grad : bool, optional
            Whether to record operations on this tensor for autograd.

        Returns
        -------
        Tensor
            A freshly constructed tensor whose storage is owned by the
            engine.

        Notes
        -----
        Constructor is one of the six sanctioned host-to-engine bridge
        boundaries (rule **H4**). NumPy arrays cross the bridge exactly
        once here; afterwards the data is owned by Lucid's engine. The
        resulting layout is C-contiguous row-major:

        .. math::

            \text{stride}[i] = e \cdot \prod_{j=i+1}^{d-1} s_j

        where :math:`e` is the element size in bytes and :math:`s_j` are
        the dimension sizes.

        Examples
        --------
        >>> import lucid
        >>> lucid.Tensor([1.0, 2.0, 3.0])
        tensor([1., 2., 3.])
        >>> lucid.Tensor([[1, 2], [3, 4]], dtype=lucid.int64).shape
        (2, 2)
        >>> x = lucid.Tensor(3.14, requires_grad=True)
        >>> x.requires_grad
        True
        """
        self._impl: _C_engine.TensorImpl = _to_impl(
            data, dtype=dtype, device=device, requires_grad=requires_grad
        )

    @classmethod
    def __new_from_impl__(cls, impl: _C_engine.TensorImpl) -> Self:
        r"""Wrap an existing ``TensorImpl`` as a ``Tensor`` with zero copy.

        Internal factory used pervasively by ops, autograd, and dispatch to
        promote a freshly-produced C++ ``TensorImpl`` into a Python
        :class:`Tensor` without re-running the full constructor (which would
        re-validate / re-convert the data). Bypasses :meth:`__init__`
        entirely by calling :func:`object.__new__` and binding ``_impl``
        directly.

        Parameters
        ----------
        impl : lucid._C.engine.TensorImpl
            Engine-side tensor object produced by a C++ kernel, autograd
            node, or factory. Ownership transfers to the new ``Tensor``.

        Returns
        -------
        Tensor
            Python-side wrapper aliasing ``impl``; no data is copied.

        Notes
        -----
        This is an **internal API**. End users should construct tensors via
        :class:`Tensor` (the public constructor) or factory functions like
        :func:`lucid.zeros` / :func:`lucid.tensor`. The factory exists so
        that the per-op overhead is one ``object.__new__`` plus one attribute
        assignment, rather than the full ``_to_impl`` validation pipeline.

        On the hot dispatch path the asymptotic cost is
        :math:`\mathcal{O}(1)` independent of ``numel``.

        Examples
        --------
        >>> import lucid
        >>> from lucid._C import engine as _C_engine
        >>> impl = _C_engine.zeros([3], _C_engine.Dtype.F32, _C_engine.Device.CPU)
        >>> t = lucid.Tensor.__new_from_impl__(impl)
        >>> t.shape
        (3,)
        """
        obj = object.__new__(cls)
        obj._impl = impl
        return obj

    @property
    def impl(self) -> _C_engine.TensorImpl:
        r"""Access the underlying C++ ``TensorImpl`` object.

        Provides read-only access to the engine-side tensor that backs this
        Python wrapper. Used by ops and the autograd engine to fetch the
        native handle without going through Python-level conversions.

        Returns
        -------
        lucid._C.engine.TensorImpl
            The engine tensor object. Lifetime is tied to ``self``; do not
            retain references beyond ``self``'s lifetime.

        Notes
        -----
        This is an internal accessor exposed for advanced use cases such as
        writing custom ops that bind directly against the engine. Public
        APIs should compose existing ``Tensor`` operations instead.
        Logically the identity projection
        :math:`\pi: \text{Tensor} \to \text{TensorImpl}` with
        :math:`\pi(t) = t.\_impl`.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3)
        >>> type(x.impl).__name__
        'TensorImpl'
        """
        return self._impl

    # ── metadata ─────────────────────────────────────────────────────────────

    @property
    def shape(self) -> tuple[int, ...]:
        r"""Shape of the tensor as a tuple of integers.

        Each element gives the size of the corresponding dimension, with
        ``shape[0]`` being the outermost (batch) dimension.  A scalar tensor
        (0-d) has ``shape == ()``.

        Returns
        -------
        tuple[int, ...]
            Immutable tuple of dimension sizes.

        Notes
        -----
        For a tensor with :math:`N` elements arranged in :math:`d` dimensions
        :math:`(s_0, s_1, \ldots, s_{d-1})`, the total element count satisfies:

        .. math::

            \prod_{i=0}^{d-1} s_i = N

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3, 4, 5)
        >>> x.shape
        (3, 4, 5)
        >>> lucid.tensor(1.0).shape
        ()
        """
        return tuple(self._impl.shape)

    @property
    def dtype(self) -> _dtype_cls:
        r"""Data type of the tensor elements.

        Reflects the numeric format used to store each element in memory.
        Lucid supports the following dtypes:

        * ``lucid.float32`` (default for floating-point)
        * ``lucid.float64``
        * ``lucid.float16``
        * ``lucid.bfloat16``
        * ``lucid.int8``, ``lucid.int16``, ``lucid.int32``, ``lucid.int64``
        * ``lucid.bool_``
        * ``lucid.complex64``

        Returns
        -------
        lucid.dtype
            The element type of this tensor.

        Notes
        -----
        The memory footprint of a tensor is :math:`N \times e` bytes, where
        :math:`N` is the total number of elements and :math:`e` is
        ``dtype.itemsize``.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3).dtype
        lucid.float32
        >>> lucid.zeros(3, dtype=lucid.int64).dtype
        lucid.int64
        """
        return _ENGINE_TO_DTYPE[self._impl.dtype]

    @property
    def device(self) -> _device_cls:
        r"""Device on which this tensor is stored.

        Lucid tensors reside on one of two devices:

        * ``cpu`` — Apple Accelerate (vDSP / vForce / BLAS / LAPACK).
        * ``metal`` — Apple Metal GPU via the MLX backend.

        Returns
        -------
        lucid.device
            The device object for this tensor.

        Notes
        -----
        On Apple Silicon the CPU and GPU share the same physical DRAM
        (unified memory architecture).  Moving a tensor between devices
        copies the *logical* dispatch target, not physical bytes, unless
        the tensor is in a non-shared Metal buffer. The dispatch routing
        rule is

        .. math::

            \text{backend}(t) = \begin{cases}
                \text{Accelerate} & \text{if } t.\text{device} = \text{cpu} \\
                \text{MLX/Metal} & \text{if } t.\text{device} = \text{metal}.
            \end{cases}

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3)
        >>> x.device
        device('cpu')
        >>> x.metal().device
        device('metal')
        """
        return _device_from_engine(self._impl.device)

    @property
    def ndim(self) -> int:
        r"""Number of dimensions (rank) of the tensor.

        Equivalent to ``len(tensor.shape)``.  A scalar tensor has ``ndim == 0``,
        a vector has ``ndim == 1``, a matrix has ``ndim == 2``, and so on.

        Returns
        -------
        int
            The number of dimensions.

        Examples
        --------
        >>> import lucid
        >>> lucid.tensor(3.14).ndim
        0
        >>> lucid.zeros(5).ndim
        1
        >>> lucid.zeros(2, 3).ndim
        2

        Notes
        -----
        For a shape tuple :math:`(s_0, \ldots, s_{d-1})` the rank is simply

        .. math::

            \text{ndim} = d = |\text{shape}|.
        """
        return len(self._impl.shape)

    @property
    def T(self) -> Tensor:
        r"""Tensor with all dimensions reversed.

        For a 2-D tensor this is the standard matrix transpose.  For tensors
        with more than two dimensions, the axis order is fully reversed:
        axis ``i`` maps to axis ``ndim - 1 - i``.

        Returns
        -------
        Tensor
            ``self`` with its dimension order reversed — a view on the
            CPU, a copy on metal.

        Notes
        -----
        For a tensor of shape :math:`(s_0, s_1, \ldots, s_{d-1})`, the
        transposed tensor has shape :math:`(s_{d-1}, \ldots, s_1, s_0)`.
        This differs from :attr:`mT`, which only transposes the final two axes.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.arange(6).reshape(2, 3)
        >>> x.shape
        (2, 3)
        >>> x.T.shape
        (3, 2)
        >>> lucid.arange(24).reshape(2, 3, 4).T.shape
        (4, 3, 2)
        """
        return Tensor.__new_from_impl__(_C_engine.T(self._impl))

    @property
    def mT(self) -> Tensor:
        r"""Tensor with the last two dimensions transposed.

        Equivalent to calling ``lucid.swapaxes(x, -2, -1)``.  Useful for
        batched linear-algebra operations where the batch dimensions should
        remain untouched.

        Returns
        -------
        Tensor
            ``self`` with axes ``-2`` and ``-1`` swapped — a view on the
            CPU, a copy on metal.

        Notes
        -----
        For a tensor of shape :math:`(\ldots, m, n)` the result has shape
        :math:`(\ldots, n, m)`.  The leading batch dimensions are unchanged.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(4, 3, 5)
        >>> x.mT.shape
        (4, 5, 3)
        """
        return Tensor.__new_from_impl__(_C_engine.mT(self._impl))

    @property
    def real(self) -> Tensor:
        r"""The real part of ``self``: a view of a complex tensor, or ``self``.

        For a complex tensor the result is real, of the matching precision
        (``float32`` for ``complex64``), and reads and writes ``self``'s
        own storage — on Metal as on the CPU — so writing into it changes
        ``self``.  A real tensor is its own real part: ``x.real is x``.
        Assigning to the attribute writes the real part in place.

        Returns
        -------
        Tensor
            :func:`lucid.real` of ``self``.

        Notes
        -----
        For :math:`z = a + ib`, ``z.real`` is :math:`a`; the setter keeps
        :math:`b` and replaces :math:`a` with the assigned value, broadcast
        to ``self``'s shape.

        Examples
        --------
        >>> import lucid
        >>> z = lucid.tensor([1 + 2j, 3 - 1j])
        >>> z.real
        tensor([1., 3.])
        >>> z.real = lucid.tensor([5.0, 6.0])
        >>> z
        tensor([(5.+2.j), (6.-1.j)], dtype=lucid.complex64)
        >>> x = lucid.ones(2)
        >>> x.real is x
        True
        """
        from lucid._ops import real as _real

        return _real(self)

    @real.setter
    def real(self, value: Tensor | float) -> None:
        _write_part(self.real, value)

    @property
    def imag(self) -> Tensor:
        r"""The imaginary part of a complex ``self``, as a view of its storage.

        The result is real, of the matching precision (``float32`` for
        ``complex64``), and reads and writes ``self``'s own storage — on
        Metal as on the CPU — so writing into it changes ``self``.
        Assigning to the attribute writes the imaginary part in place.

        Returns
        -------
        Tensor
            :func:`lucid.imag` of ``self``.

        Raises
        ------
        NotImplementedError
            ``self`` is not complex: a real tensor has no imaginary part to
            view (a ``RuntimeError`` too, as the reference raises).

        Examples
        --------
        >>> import lucid
        >>> z = lucid.tensor([1 + 2j, 3 - 1j])
        >>> z.imag
        tensor([2., -1.])
        >>> z.imag = 0.0
        >>> z
        tensor([(1.+0.j), (3.+0.j)], dtype=lucid.complex64)
        """
        from lucid._ops import imag as _imag

        return _imag(self)

    @imag.setter
    def imag(self, value: Tensor | float) -> None:
        _write_part(self.imag, value)

    @property
    def is_metal(self) -> bool:
        r"""Whether this tensor resides on the Metal (GPU) device.

        On Apple Silicon the GPU backend is Apple Metal accessed through the
        MLX library.  CPU tensors use Apple Accelerate instead.

        Returns
        -------
        bool
            ``True`` if the tensor is on the Metal device, ``False`` for CPU.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3)
        >>> x.is_metal
        False
        >>> x.metal().is_metal
        True

        Notes
        -----
        Equivalent predicate :math:`\text{is\_metal}(t) = (t.\text{device} = \text{GPU})`.
        On Apple Silicon the Metal backend dispatches via MLX, which lazily
        builds a graph and evaluates on the GPU when results are observed.
        """
        return self._impl.device == _C_engine.Device.GPU

    @property
    def is_shared(self) -> bool:
        r"""True when backed by a Metal ``MTLResourceStorageModeShared`` buffer.

        Shared-memory tensors live in Apple Silicon unified DRAM and are
        simultaneously accessible from CPU and GPU without a ``memcpy``.
        Create them with ``lucid.metal.shared_tensor()`` or promote an
        existing tensor with ``lucid.metal.to_shared()``; ``.to()`` between
        CPU and Metal then returns an alias of the same buffer.  A view that
        covers only part of the buffer answers ``False``, and so does a tensor
        whose storage an in-place op recording a graph has replaced.

        Returns
        -------
        bool
            ``True`` if the tensor's storage uses Metal's shared storage
            mode, ``False`` otherwise (including all pure-CPU tensors and
            private-mode Metal buffers).

        Notes
        -----
        On Apple Silicon, shared storage permits zero-copy hand-off between
        CPU and GPU code paths because both engines see the same physical
        DRAM page. The trade-off is that Metal's GPU caches cannot be used
        as aggressively as with private storage; for compute-bound GPU
        kernels, private mode is often preferable. Predicate:
        :math:`\text{is\_shared}(t) = \mathbf{1}\{\text{storage\_mode}(t) = \text{Shared}\}`.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3)
        >>> x.is_shared
        False
        >>> y = lucid.metal.shared_tensor((3,))
        >>> y.is_shared
        True
        >>> y.to("metal").is_shared  # an alias of the same buffer
        True
        """
        return self._impl.is_metal_shared

    @property
    def is_leaf(self) -> bool:
        r"""Whether this tensor is a leaf in the autograd computation graph.

        A tensor is a leaf if it was created directly by the user (not as the
        result of an operation) or if it does not require gradients.  Only
        leaf tensors accumulate gradients into their ``.grad`` attribute during
        :meth:`backward`.

        Returns
        -------
        bool
            ``True`` for leaf tensors, ``False`` for intermediate results.

        Notes
        -----
        In the computation graph, leaf nodes are the "inputs" to the
        forward pass.  All tensors created with ``requires_grad=False``
        are leaves by definition.  Tensors created with
        ``requires_grad=True`` directly by the user are also leaves.
        Tensors produced by differentiable operations on
        ``requires_grad=True`` inputs are *not* leaves — they are
        intermediate nodes and their ``.grad`` is not retained unless
        :meth:`retain_grad` is called. Predicate:

        .. math::

            \text{is\_leaf}(t) = \mathbf{1}\{\text{grad\_fn}(t) = \bot\}.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0], requires_grad=True)
        >>> x.is_leaf
        True
        >>> y = x * 2          # result of an op — not a leaf
        >>> y.is_leaf
        False
        """
        return self._impl.is_leaf

    @property
    def requires_grad(self) -> bool:
        """Whether gradient computation is enabled for this tensor.

        When ``True``, operations involving this tensor are recorded in the
        computation graph so that :meth:`backward` can propagate gradients
        back through them.

        Setting this attribute on a leaf tensor promotes or demotes it from
        the autograd graph.  Calling ``requires_grad = True`` on an
        intermediate (non-leaf) tensor raises a ``RuntimeError`` because
        those tensors are not user-controlled inputs.

        Returns
        -------
        bool
            ``True`` if gradients are tracked for this tensor.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0])
        >>> x.requires_grad
        False
        >>> x.requires_grad = True
        >>> x.requires_grad
        True

        Notes
        -----
        Tensors with ``requires_grad=True`` become nodes in the autograd DAG.
        Operations consuming them produce non-leaf tensors that also require
        gradients (transitive closure of the flag along the forward graph).
        """
        return self._impl.requires_grad

    @requires_grad.setter
    def requires_grad(self, v: bool) -> None:
        """Set whether this tensor participates in autograd, in place.

        The flag changes on this tensor itself: its storage, its identity
        and any gradient already accumulated in ``.grad`` are kept, so an
        optimizer that holds the tensor keeps updating it after a freeze
        and unfreeze.

        Parameters
        ----------
        v : bool
            New value for the ``requires_grad`` flag. ``True`` enrolls the
            tensor in the autograd graph; ``False`` removes it.

        Raises
        ------
        RuntimeError
            If ``self`` is not a leaf.  An intermediate tensor's flag
            follows the operation that computed it; use :meth:`detach` for
            a tensor outside the graph.

        Notes
        -----
        For chaining, prefer :meth:`requires_grad_` which returns ``self``.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3)
        >>> x.requires_grad = True
        >>> x.requires_grad
        True
        """
        _set_requires_grad(self, v, setter=True)

    def numel(self) -> int:
        r"""Return the total number of elements in the tensor.

        Equivalent to the product of all dimension sizes, i.e. the result of
        evaluating :math:`\prod_{i} s_i` over the shape tuple.  For a scalar
        (0-d) tensor this is ``1``; for an empty tensor it is ``0``.

        Returns
        -------
        int
            Total element count.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3, 4).numel()
        12
        >>> lucid.tensor(5.0).numel()
        1

        Notes
        -----
        Used by the engine to allocate storage of size
        :math:`N \cdot e` bytes, where :math:`e = \text{dtype.itemsize}`.
        Convenient invariant: ``tensor.numel() == tensor.nbytes // tensor.itemsize``.
        """
        return int(self._impl.numel())

    def dim(self) -> int:
        r"""Return the number of dimensions (rank) of the tensor.

        This is identical to the :attr:`ndim` property.  It exists as a
        method for API compatibility with code that calls ``tensor.dim()``.

        Returns
        -------
        int
            Number of dimensions.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(2, 3, 4).dim()
        3
        >>> lucid.tensor(1.0).dim()
        0

        Notes
        -----
        Equivalent to ``len(tensor.shape)``; matches the conventional rank
        :math:`d` such that :math:`\text{shape} \in \mathbb{N}^d`.
        """
        return len(self._impl.shape)

    @overload
    def size(self, dim: int) -> int:
        """Overload: with an integer ``dim``, returns the size of that axis as ``int``."""
        ...

    @overload
    def size(self, dim: None = ...) -> tuple[int, ...]:
        """Overload: with no argument, returns the full shape tuple."""
        ...

    def size(self, dim: int | None = None) -> int | tuple[int, ...]:
        r"""Return the size of a specific dimension, or the full shape tuple.

        Parameters
        ----------
        dim : int, optional
            Dimension index to query.  Supports negative indexing.
            If omitted, the full shape tuple is returned.

        Returns
        -------
        int
            Size of the requested dimension (when ``dim`` is given).
        tuple[int, ...]
            Full shape tuple (when ``dim`` is omitted).

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3, 4, 5)
        >>> x.size()
        (3, 4, 5)
        >>> x.size(0)
        3
        >>> x.size(-1)
        5

        Notes
        -----
        The full shape and the element count are related by

        .. math::

            \text{numel} = \prod_{i=0}^{d-1} \text{size}(i).
        """
        s = tuple(self._impl.shape)
        if dim is not None:
            return s[dim]
        return s

    def is_contiguous(self) -> bool:
        r"""Return ``True`` if the tensor's data is stored contiguously in memory.

        A contiguous tensor stores its elements in a single unbroken block of
        memory in C (row-major) order — i.e. the stride of each dimension equals
        the product of all *later* dimension sizes times the element size.

        On the CPU a transpose, a slice along any axis, :meth:`diagonal`,
        :meth:`unfold`, ``expand`` and ``broadcast_to`` are views of their
        input's buffer, and a transposed or column view is not contiguous.
        Every op reads such a view correctly and a write through it follows
        its strides, so the answer matters only to code that walks the memory
        itself.  A slice with a positive step is a view too.  Metal tensors
        are always packed, and a negative step still copies.

        Returns
        -------
        bool
            ``True`` if the tensor uses a single contiguous memory block in
            C-order, ``False`` otherwise.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3, 4)
        >>> x.is_contiguous()
        True
        >>> x.T.is_contiguous()   # a transpose is a view with swapped strides
        False

        Notes
        -----
        A tensor of shape :math:`(s_0, \ldots, s_{d-1})` is contiguous iff
        its strides satisfy the C-order recurrence

        .. math::

            \text{stride}[d-1] = 1, \quad
            \text{stride}[i] = \text{stride}[i+1] \cdot s_{i+1}

        (in element units). Many Accelerate/MLX kernels require contiguous
        inputs; if not, call :meth:`contiguous` first.
        """
        return self._impl.is_contiguous()

    # ── autograd ─────────────────────────────────────────────────────────────

    @property
    def grad(self) -> Tensor | None:
        r"""Accumulated gradient tensor, or ``None`` if not yet computed.

        After calling :meth:`backward`, this attribute holds the gradient of
        the scalar loss with respect to this tensor — i.e.
        :math:`\frac{\partial \mathcal{L}}{\partial \mathbf{x}}` where
        :math:`\mathbf{x}` is this tensor.

        Gradients are **accumulated** (added) across multiple :meth:`backward`
        calls.  Zero the gradient before each optimisation step with
        ``tensor.grad = None`` or ``tensor.grad.zero_()``, or use an optimiser
        that calls ``zero_grad()`` automatically.

        Only **leaf** tensors (those created directly by user code with
        ``requires_grad=True``) populate this attribute by default.
        Non-leaf (intermediate) tensors discard their gradient after the
        backward pass unless :meth:`retain_grad` was called.

        Returns
        -------
        Tensor or None
            The accumulated gradient, or ``None`` if :meth:`backward` has not
            been called yet or if this tensor does not require gradients.

        Notes
        -----
        The gradient has the same shape as the tensor itself:

        .. math::

            \text{grad.shape} = \text{self.shape}

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> y = (x ** 2).sum()
        >>> y.backward()
        >>> x.grad          # d(sum(x^2))/dx = 2x
        tensor([2., 4., 6.])
        """
        # Prefer graph-mode gradient (set when backward was run with create_graph=True).
        g_impl = _read_grad(self._impl)
        return None if g_impl is None else Tensor.__new_from_impl__(g_impl)

    @grad.setter
    def grad(self, v: Tensor | None) -> None:
        """Assign or clear the accumulated gradient buffer.

        Allows external code (optimisers, gradient-clipping utilities,
        custom training loops) to write directly into the autograd
        ``.grad`` slot. Assigning ``None`` zeroes the buffer; assigning a
        ``Tensor`` overwrites its contents.

        Parameters
        ----------
        v : Tensor or None
            New gradient value. If a ``Tensor``, its shape and dtype must
            match ``self``; the value is unwrapped to its engine impl and
            installed in place. If ``None``, the engine's ``zero_grad`` is
            called, releasing the gradient storage.

        Notes
        -----
        Common training-loop idiom — clear gradients before each step:

        .. code-block:: python

            for p in model.parameters():
                p.grad = None

        Assigning ``None`` is preferred over ``p.grad.zero_()`` because it
        frees the gradient buffer entirely (smaller memory footprint
        between steps) and avoids an unnecessary write to zero-fill.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0], requires_grad=True)
        >>> (x ** 2).sum().backward()
        >>> x.grad is not None
        True
        >>> x.grad = None
        >>> x.grad is None
        True
        """
        _set_grad(self._impl, v)

    @property
    def grad_fn(self) -> _C_engine.Node | None:
        r"""The autograd graph node that created this tensor, or ``None``.

        Every tensor produced by a differentiable operation holds a reference
        to the C++ ``Node`` (gradient function) that can propagate gradients
        back through that operation.  The graph is a directed acyclic graph
        (DAG) of such nodes; :meth:`backward` traverses it in reverse
        topological order.

        Leaf tensors always have ``grad_fn = None``, also once used in a graph,
        where the engine keeps the leaf's gradient accumulator in that slot.

        Returns
        -------
        lucid._C.engine.Node or None
            The gradient-computing node, or ``None`` for leaf tensors or
            tensors that do not require gradients.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0], requires_grad=True)
        >>> x.grad_fn is None      # leaf — no grad_fn
        True
        >>> y = x * 3
        >>> y.grad_fn is None      # result of an op — has a grad_fn
        False

        Notes
        -----
        The graph forms the chain-rule factorisation used by :meth:`backward`:

        .. math::

            \frac{\partial \mathcal{L}}{\partial \mathbf{x}}
            = \frac{\partial \mathcal{L}}{\partial \mathbf{y}}
              \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}},

        where :math:`\mathbf{y} = f(\mathbf{x})` and ``grad_fn`` carries the
        Jacobian-vector product :math:`v \mapsto v \cdot J_f`.
        """
        return None if self._impl.is_leaf else self._impl.grad_fn

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        r"""Enable or disable gradient tracking for this tensor, in-place.

        Unlike the :attr:`requires_grad` property setter, this method
        returns ``self`` so it can be chained inline:

        .. code-block:: python

            x = lucid.randn(3, 4).requires_grad_(True)

        Parameters
        ----------
        requires_grad : bool, optional
            New value for gradient tracking.  Defaults to ``True``.

        Returns
        -------
        Tensor
            ``self`` with the updated ``requires_grad`` flag.

        Raises
        ------
        RuntimeError
            If ``requires_grad`` is ``False`` and ``self`` is not a leaf.
            Use :meth:`detach` for a computed tensor outside the graph.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3)
        >>> x.requires_grad_(True).requires_grad
        True

        Notes
        -----
        In-place flag flip on this tensor itself: its storage, its identity
        and any accumulated ``.grad`` are kept, so freezing and unfreezing a
        parameter between optimizer steps does not detach it from the
        optimizer.  On a non-leaf tensor ``requires_grad_(True)`` is a no-op
        (it already requires grad), and clearing the flag is refused.
        """
        _set_requires_grad(self, requires_grad, setter=False)
        return self

    def retain_grad(self) -> None:
        r"""Retain the gradient on this non-leaf tensor after :meth:`backward`.

        By default, gradients are only stored on **leaf** tensors.  Intermediate
        (non-leaf) results in the computation graph have their ``.grad``
        discarded after the backward pass to save memory.  Calling
        ``retain_grad()`` on an intermediate tensor before :meth:`backward`
        instructs the engine to keep that gradient so it can be inspected
        afterwards.

        Raises
        ------
        RuntimeError
            If ``self`` does not require grad.

        Notes
        -----
        Call it before the :meth:`backward` whose gradient you want to
        inspect; it may come after the ops that consume ``self``.  Calling
        it after :meth:`backward` has no effect on that pass.  Conceptually
        retains :math:`\frac{\partial \mathcal{L}}{\partial \mathbf{y}}`
        for the intermediate node :math:`\mathbf{y}` instead of discarding
        it after its parents have consumed it.  The retained gradient is
        the one left after the hooks registered with :meth:`register_hook`
        have run, whichever was registered first, and it accumulates
        across backward passes as a leaf's does.  A leaf keeps its
        gradient anyway.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> y = x * 2          # intermediate — grad normally discarded
        >>> y.retain_grad()
        >>> y.sum().backward()
        >>> y.grad             # now available: d(sum(y))/dy = [1., 1., 1.]
        tensor([1., 1., 1.])
        """
        if not self.requires_grad:
            raise RuntimeError(
                "retain_grad(): this tensor does not require grad, so no "
                "gradient will ever reach it"
            )
        self._impl.retain_grad_()

    def register_hook(
        self, hook: Callable[[Tensor], Tensor | None]
    ) -> _RemovableHandle:
        r"""Register a hook that runs on this tensor's gradient during backward.

        The hook receives the gradient of ``self`` in the current backward
        pass, once every contribution to it has arrived.  If it returns a
        :class:`~lucid.Tensor`, that tensor replaces the gradient for
        everything upstream of ``self``; returning ``None`` leaves it as is.

        Parameters
        ----------
        hook : callable
            ``hook(grad: Tensor) -> Tensor | None``.  A returned tensor must
            have the gradient's dtype, device and shape.

        Returns
        -------
        RemovableHandle
            Call ``.remove()`` to de-register the hook, or use it as a
            context manager.

        Raises
        ------
        RuntimeError
            If ``self`` does not require grad.

        Notes
        -----
        * On a **non-leaf** the hook runs once its producer's output slot
          has received every consumer's gradient — consumers created before
          or after the hook alike — and before the producer's backward runs.
          No :meth:`retain_grad` is needed.  :func:`lucid.autograd.grad`
          runs it too when the tensor lies on a path to the requested
          inputs.
        * On a **leaf** the hook runs just before the pass's gradient is
          added into ``.grad``, so it sees that pass's gradient alone, not
          the accumulated one, and an unrelated backward never runs it.
        * Several hooks run in registration order, each on what the one
          before it returned.  :meth:`retain_grad` keeps the gradient left
          after all of them.
        * The hook runs with grad mode off, or on under
          ``backward(create_graph=True)``, where its result stays part of
          the graph.
        * After an in-place op on ``self``, a hook registered earlier still
          sees the gradient of the value before the write.
        * On a non-leaf, the hook is kept by the producer graph node and
          released after backward without ``retain_graph``.  On a **leaf**,
          the hook runner lives in the leaf's C++ ``AutogradMeta``, so a
          closure capturing the leaf forms a cycle (``leaf → AutogradMeta →
          runner → closure → leaf``) until ``handle.remove()`` is called.

        Chain-rule effect: if the hook returns a tensor :math:`\tilde g`,
        the engine substitutes :math:`\frac{\partial \mathcal{L}}{\partial \mathbf{x}}
        \leftarrow \tilde g` before continuing backward propagation.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> grads = []
        >>> h = x.register_hook(lambda g: grads.append(g.clone()))
        >>> (x * 2).sum().backward()
        >>> grads[0]
        tensor([2., 2., 2.])
        >>> h.remove()        # de-register

        On an intermediate, the returned gradient flows on upstream:

        >>> x = lucid.tensor([1.0, 2.0], requires_grad=True)
        >>> y = x * 3
        >>> _ = y.register_hook(lambda g: g * 10)
        >>> y.sum().backward()
        >>> x.grad
        tensor([30., 30.])
        """
        if not self.requires_grad:
            raise RuntimeError(
                "register_hook called on a tensor that does not require grad. "
                "Only tensors with requires_grad=True can have gradient hooks."
            )
        from lucid.autograd._hooks import _register_tensor_hook

        return _register_tensor_hook(self, hook)

    def backward(
        self,
        gradient: Tensor | None = None,
        retain_graph: bool = False,
        create_graph: bool = False,
        inputs: Tensor | Sequence[Tensor] | None = None,
    ) -> None:
        r"""Compute gradients via reverse-mode automatic differentiation.

        Traverses the computation graph in reverse topological order, applying
        the chain rule at each node to accumulate
        :math:`\frac{\partial \mathcal{L}}{\partial \mathbf{x}}` into the
        ``.grad`` attribute of every leaf tensor that has
        ``requires_grad=True``.

        Parameters
        ----------
        gradient : Tensor, optional
            The seed gradient — the gradient of some external scalar with
            respect to ``self``.  Must have the same shape as ``self``.
            Omitted on scalar tensors (implicit ``1.0``); required on
            non-scalar tensors.
        retain_graph : bool, optional
            If ``False`` (default), intermediate tensors and gradient functions
            in the computation graph are freed after the backward pass.  Set
            to ``True`` to call ``backward()`` again on the same graph.
        create_graph : bool, optional
            When ``True`` the backward pass itself is recorded, so the
            gradients it leaves in ``.grad`` are differentiable.
        inputs : Tensor, sequence of Tensor, or None, optional
            Accumulate only into these tensors' ``.grad``; every other
            leaf keeps what it had.  A non-leaf input has its ``.grad``
            filled as if :meth:`retain_grad` had been called.  See
            :func:`lucid.autograd.backward`.

        Raises
        ------
        TypeError
            If ``gradient`` is neither a :class:`Tensor` nor ``None``, or
            ``inputs`` is not a tensor or sequence of tensors.
        RuntimeError
            If ``self`` does not require grad, if ``gradient`` is omitted for
            a non-scalar or complex output, or if ``inputs`` is empty or
            contains a tensor that does not require grad.
        lucid._C.engine.ShapeMismatch
            If ``gradient.shape != self.shape``.
        lucid._C.engine.DtypeMismatch
            If ``self`` and ``gradient`` disagree on complex vs. real dtype.
        lucid._C.engine.DeviceMismatch
            If ``gradient.device != self.device``.

        Notes
        -----
        **The chain rule and reverse-mode AD**

        Given a scalar loss :math:`\mathcal{L}` and a sequence of operations
        :math:`\mathbf{z} = f_n(\cdots f_2(f_1(\mathbf{x})) \cdots)`,
        the chain rule gives:

        .. math::

            \frac{\partial \mathcal{L}}{\partial \mathbf{x}}
            = \frac{\partial \mathcal{L}}{\partial \mathbf{z}}
              \cdot \frac{\partial \mathbf{z}}{\partial \mathbf{x}}
            = \frac{\partial \mathcal{L}}{\partial \mathbf{z}}
              \cdot J_{f_n} \cdots J_{f_1}

        where :math:`J_{f_i}` is the Jacobian of the :math:`i`-th operation.
        Reverse-mode AD (backpropagation) evaluates this product right-to-left,
        starting from the scalar output with seed gradient
        :math:`\frac{\partial \mathcal{L}}{\partial \mathcal{L}} = 1`, so
        it computes gradients for *all* inputs in a single backward pass —
        :math:`\mathcal{O}(1)` passes regardless of the number of parameters.

        **Gradient accumulation**

        Gradients are *added* to ``tensor.grad`` rather than overwritten.
        This is intentional: it supports patterns like accumulated gradient
        steps.  Zero gradients explicitly before each optimisation step:

        .. code-block:: python

            for param in model.parameters():
                param.grad = None   # or param.grad.zero_()

        Examples
        --------
        Scalar output — no explicit gradient seed needed:

        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
        >>> loss = (x ** 2).sum()   # scalar
        >>> loss.backward()
        >>> x.grad                  # d(sum(x^2))/dx = 2x
        tensor([2., 4., 6.])

        Non-scalar output — must supply a gradient seed:

        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0], requires_grad=True)
        >>> y = x * 3               # shape (2,) — not a scalar
        >>> y.backward(lucid.ones(2))
        >>> x.grad                  # d(3x)/dx = 3 for each element
        tensor([3., 3.])

        Multiple backward passes with ``retain_graph=True``:

        >>> import lucid
        >>> x = lucid.tensor([2.0], requires_grad=True)
        >>> y = x ** 3
        >>> y.backward(retain_graph=True)   # first pass
        >>> y.backward()                    # second pass — accumulates
        >>> x.grad                          # 3*x^2 + 3*x^2 = 2 * 12 = 24
        tensor([24.])
        """
        # No ``self._impl.eval()`` first: a sync point between the passes cost
        # 5-10% of an eager step on M4 Max (May 2026), and MLX's backward
        # evaluates what it needs.
        from lucid.autograd._backward import backward as _backward

        _backward(self, gradient, retain_graph, create_graph, inputs)

    def detach_(self) -> Self:
        r"""Detach this tensor from the autograd graph in-place.

        Clears ``requires_grad`` and removes the ``grad_fn`` so that future
        operations on ``self`` are not tracked.  Returns ``self`` for chaining.

        Unlike :meth:`detach`, this modifies the tensor in-place and does
        **not** create a new object.

        Returns
        -------
        Tensor
            ``self`` after detaching.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0], requires_grad=True)
        >>> y = x * 3
        >>> _ = y.detach_()
        >>> y.requires_grad
        False

        Notes
        -----
        In-place counterpart of :meth:`detach`. Mutates ``self`` so future
        ops are not tracked; equivalent to setting
        :math:`\text{requires\_grad} \leftarrow \text{False}` and
        clearing ``grad_fn``. Returns ``self`` to permit fluent chaining.
        """
        self._impl = self._impl.data_alias()
        return self

    def clamp_(
        self,
        min: Tensor | float | None = None,
        max: Tensor | float | None = None,
    ) -> Self:
        r"""Clamp all elements of this tensor to ``[min, max]``, in-place.

        Each element :math:`x_i` is replaced by:

        .. math::

            x_i \leftarrow \max(\text{min},\, \min(\text{max},\, x_i))

        Parameters
        ----------
        min : Tensor or scalar, optional
            Lower bound.  Elements below this value are raised to ``min``.
        max : Tensor or scalar, optional
            Upper bound.  Elements above this value are lowered to ``max``.

        Returns
        -------
        Tensor
            ``self`` after clamping.

        Raises
        ------
        ValueError
            If neither bound is given, as :meth:`clamp` refuses it.
        RuntimeError
            If the bounds promote ``self``'s dtype, as :meth:`clamp` promotes
            it, to one ``self`` cannot hold: ``int64.clamp_(0.5, 1.5)``.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([-2.0, 0.5, 3.0])
        >>> x.clamp_(min=-1.0, max=2.0)
        tensor([-1., 0.5, 2.])
        """
        return _clamp_inplace(self, min, max, "clamp_")

    def clamp_min_(self, min: Tensor | float) -> Self:
        r"""Raise all elements below ``min`` to ``min``, in-place.

        Shorthand for ``clamp_(min=min)``.

        Parameters
        ----------
        min : Tensor or scalar
            Lower bound.  Elements strictly less than ``min`` are set to
            ``min``.

        Returns
        -------
        Tensor
            ``self`` after clamping.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([-1.0, 0.0, 1.0])
        >>> x.clamp_min_(0.0)
        tensor([0., 0., 1.])

        Notes
        -----
        Pointwise rectification:

        .. math::

            x_i \leftarrow \max(\text{min}, x_i).

        The non-smooth point at :math:`x = \text{min}` has subgradient
        :math:`[0, 1]`; autograd selects ``1`` for :math:`x > \text{min}`
        and ``0`` otherwise.
        """
        return _clamp_inplace(self, min, None, "clamp_min_")

    def clamp_max_(self, max: Tensor | float) -> Self:
        r"""Lower all elements above ``max`` to ``max``, in-place.

        Shorthand for ``clamp_(max=max)``.

        Parameters
        ----------
        max : Tensor or scalar
            Upper bound.  Elements strictly greater than ``max`` are set to
            ``max``.

        Returns
        -------
        Tensor
            ``self`` after clamping.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([0.5, 1.5, 2.5])
        >>> x.clamp_max_(2.0)
        tensor([0.5, 1.5, 2.])

        Notes
        -----
        Pointwise saturation:

        .. math::

            x_i \leftarrow \min(\text{max}, x_i).

        The gradient is the indicator :math:`\mathbf{1}\{x_i < \text{max}\}`.
        """
        return _clamp_inplace(self, None, max, "clamp_max_")

    # ── conversion ───────────────────────────────────────────────────────────

    def item(self) -> float | int | bool:
        r"""Return the value of a single-element tensor as a Python scalar.

        Delegates to the engine's ``TensorImpl::item`` which performs the
        single-element extraction (including IEEE-754 binary16 to float
        decoding) without going through numpy.

        Returns
        -------
        float or int or bool
            The unboxed Python scalar. Floating-point dtypes (including
            ``float16`` / ``bfloat16``) return ``float``; integer dtypes
            return ``int``; ``bool_`` returns ``bool``.

        Raises
        ------
        RuntimeError
            If the tensor has more than one element (``numel() != 1``).

        Notes
        -----
        Triggers a device-to-host synchronisation when called on a Metal
        tensor — the value cannot be inspected until any pending MLX
        computation has completed. Avoid calling in tight loops on GPU
        tensors; prefer ``tensor.cpu().tolist()`` for batch extraction.

        Defined only when :math:`\text{numel} = \prod_i s_i = 1`. ``item``
        is one of the sanctioned engine-to-host bridge points (rule **H4**)
        and detaches from autograd: the returned Python scalar carries no
        gradient information.

        Examples
        --------
        >>> import lucid
        >>> lucid.tensor(3.14).item()
        3.140000104904175
        >>> lucid.tensor(7, dtype=lucid.int64).item()
        7
        """
        return self._impl.item()

    def numpy(self) -> np.ndarray:  # type: ignore[type-arg]
        r"""Return the tensor as a NumPy array (CPU only).

        Imports numpy lazily — the rest of Lucid stays numpy-free unless
        the user explicitly bridges through this method. When numpy is
        not installed, raises an ``ImportError`` pointing at
        ``pip install lucid[numpy]``.

        Returns
        -------
        numpy.ndarray
            A NumPy view (or copy) of the tensor's data. Shape and dtype
            mirror the source; the array lives in host memory.

        Raises
        ------
        ImportError
            If NumPy is not installed.
        RuntimeError
            If the tensor lives on the Metal device and cannot be evaluated
            to host memory.

        Notes
        -----
        This is one of the sanctioned bridge points between Lucid and the
        outside world (see project rule **H4**). The returned ``ndarray``
        does **not** participate in autograd; downstream NumPy operations
        will not produce gradients.

        Layout is C-contiguous; for a tensor of shape :math:`(s_0, \ldots, s_{d-1})`
        and itemsize :math:`e`, the NumPy strides equal

        .. math::

            \text{nstride}[i] = e \cdot \prod_{j=i+1}^{d-1} s_j.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> arr = x.numpy()
        >>> arr.shape
        (2, 2)
        >>> arr.dtype
        dtype('float32')
        """
        from lucid._factories.converters import _require_numpy

        np = _require_numpy("Tensor.numpy()")
        raw = self._impl.data_as_python()
        return np.asarray(raw)

    def __array__(
        self, dtype: NpDTypeLike | None = None, *, copy: bool | None = None
    ) -> np.ndarray:
        r"""NumPy's array-conversion hook — makes ``np.asarray(t)`` give real numbers.

        Without this method NumPy falls back to the sequence protocol and
        walks the tensor element by element, producing an
        ``dtype=object`` array whose entries are 0-d :class:`Tensor`
        objects.  That fails no test and raises nothing; it just silently
        stops being numeric, so every downstream ``mean`` / ``astype`` /
        plot behaves differently from the same call on
        :meth:`numpy`.  Routing the hook through :meth:`numpy` keeps the
        three spellings — ``t.numpy()``, ``np.asarray(t)``,
        ``np.from_dlpack(t)`` — on one code path.

        Parameters
        ----------
        dtype : numpy.typing.DTypeLike or None, optional, default=None
            Requested element type.  ``None`` keeps the tensor's own.
        copy : bool or None, optional, keyword-only, default=None
            NumPy 2 copy semantics.  ``None`` copies when needed, ``True``
            always copies, and ``False`` demands a view.  A Lucid tensor
            has to be read out through the host to become an
            ``ndarray``, so ``copy=False`` is unsatisfiable and raises.

        Returns
        -------
        numpy.ndarray
            The tensor's values, on the host.

        Raises
        ------
        ValueError
            ``copy=False`` was requested — see above.
        ImportError
            NumPy is not installed (raised by :meth:`numpy`).

        Notes
        -----
        Like :meth:`numpy`, a Metal tensor round-trips through the CPU.
        This is one of the bridge boundaries where numpy is allowed
        (project rule **H4**).

        Examples
        --------
        >>> import lucid
        >>> import numpy as np
        >>> np.asarray(lucid.ones(2, 2)).dtype
        dtype('float32')
        """
        if copy is False:
            raise ValueError(
                "Tensor.__array__: copy=False was requested, but converting a "
                "Lucid tensor to a NumPy array always reads the values out "
                "through the host, so no view is possible.  Use "
                "np.asarray(t) / t.numpy() and accept the copy."
            )
        arr = self.numpy()
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    # ── DLPack protocol ──────────────────────────────────────────────────
    #
    # Both methods route through ``self.numpy()`` so any DLPack-aware
    # consumer (reference framework / JAX / etc) can ingest a Lucid tensor directly:
    #
    #     >>> any_dlpack_consumer.from_dlpack(lucid.zeros(3))
    #
    # Metal tensors silently round-trip through CPU (numpy() already
    # does that).  A native engine-side DLPack export is filed as
    # future work.

    def __dlpack__(
        self,
        stream: object | None = None,
        *,
        max_version: tuple[int, int] | None = None,
        dl_device: tuple[int, int] | None = None,
        copy: bool | None = None,
    ) -> object:
        r"""Export this tensor as a DLPack ``PyCapsule`` for zero-copy interop.

        DLPack is the open cross-framework tensor exchange specification —
        any consumer implementing the protocol (the reference framework,
        JAX, CuPy, TVM, and many others) can ingest a Lucid tensor without
        a Python-level data copy by calling ``their_framework.from_dlpack(t)``.

        Two dialects, picked by where the tensor lives and what the
        consumer asks for:

        * **Metal** — the capsule carries the tensor's ``MTLBuffer``
          (device ``kDLMetal``). An MLX consumer reads the very pages
          Lucid computed into; no download happens. NumPy cannot read a
          capsule of this device type at all.
        * **Host** — a CPU tensor, or a Metal one whose consumer named
          the host in ``dl_device``, exports through :meth:`numpy`.

        A host consumer therefore has to say so — ``np.from_dlpack(t)``
        on a Metal tensor raises, while ``np.from_dlpack(t, device="cpu")``
        downloads and works. That is the same bargain every framework's
        GPU tensors make, and the reason :meth:`__dlpack_device__` can now
        answer truthfully.

        Parameters
        ----------
        stream : object, optional
            Stream / queue handle for synchronisation, per the DLPack v0.8+
            protocol. Forwarded to NumPy's ``__dlpack__`` on the host path.
            Ignored on the Metal path, which blocks until the producing
            command buffer completes before it hands over the handle.
        max_version : tuple[int, int] or None, optional, keyword-only, default=None
            Highest DLPack version the consumer supports. Accepted for
            protocol compatibility; this exporter emits the unversioned
            ``DLManagedTensor`` both dialects agree on.
        dl_device : tuple[int, int] or None, optional, keyword-only, default=None
            Device the consumer wants the data on. ``None`` means the
            tensor's own device. Naming the host (``(1, 0)``) forces the
            NumPy path even for a Metal tensor.
        copy : bool or None, optional, keyword-only, default=None
            ``False`` demands a view. Refused when it would require
            moving a Metal tensor to the host, since that cannot be done
            without copying.

        Returns
        -------
        PyCapsule
            A capsule wrapping a ``DLManagedTensor`` struct. The capsule
            must be consumed (renamed to ``"used_dltensor"``) by the
            receiver; otherwise its destructor releases the underlying
            buffer.

        Notes
        -----
        See the DLPack specification at
        https://dmlc.github.io/dlpack/latest/ for the precise wire format
        and version semantics. Lucid follows the v0.8 protocol revision.

        Bridge boundary (rule **H4**): no autograd information crosses the
        capsule. Logically the export is the identity on values,
        :math:`\text{dlpack}(\mathbf{x}) \equiv \mathbf{x}`, with the
        consumer responsible for interpreting strides/dtype correctly.

        Examples
        --------
        >>> import lucid
        >>> import numpy as np
        >>> x = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> arr = np.from_dlpack(x)
        >>> arr.shape
        (2, 2)
        """
        del max_version  # DLPack protocol compatibility; emits unversioned DLManagedTensor
        wants_host = dl_device is not None and dl_device[0] != _DLPACK_METAL
        if self.is_metal and not wants_host:
            # Metal tensors export natively: the capsule points at the
            # MTLBuffer MLX is already holding, so a consumer on the same
            # unified memory reads the very pages Lucid computed into.
            # ``stream`` has no meaning here — the exporter blocks until
            # the producing command buffer completes before handing the
            # handle over.
            from lucid._C import engine as _C_engine

            return _C_engine.to_dlpack_metal(self._impl)
        # A consumer that asked for the host (``dl_device``) gets the
        # NumPy dialect, download and all.  ``copy=False`` cannot be
        # honoured across that download, so it is refused rather than
        # quietly ignored.
        if wants_host and self.is_metal and copy is False:
            raise BufferError(
                "Tensor.__dlpack__: a Metal tensor cannot be exported to the "
                "host without a copy; pass copy=None or import it on Metal"
            )
        arr = self.numpy()
        return arr.__dlpack__(stream=stream) if stream is not None else arr.__dlpack__()

    def __dlpack_device__(self) -> tuple[int, int]:
        r"""Return ``(device_type, device_id)`` per the DLPack specification.

        Companion to :meth:`__dlpack__` — DLPack consumers query this method
        before calling ``__dlpack__`` so they know how to synchronise and
        which memory space the capsule will reference.

        Returns
        -------
        tuple[int, int]
            ``(device_type, device_id)``: ``(8, 0)`` for a Metal tensor
            (``8 == kDLMetal``) and ``(1, 0)`` for a host one, where
            ``1 == kDLCPU``. The ``device_id`` of ``0`` is conventional
            for non-indexed devices.

        Notes
        -----
        DLPack device types of interest:

        * ``1`` — ``kDLCPU``
        * ``8`` — ``kDLMetal``

        This used to answer ``(1, 0)`` for every tensor, Metal ones
        included, because the export went through NumPy and downloaded to
        make that true. A consumer had no way to learn that the data was
        already on the GPU it was about to upload it back to.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3).__dlpack_device__()
        (1, 0)
        """
        return (8, 0) if self.is_metal else (1, 0)

    def tolist(self) -> list[object] | int | float | bool:
        r"""Return the tensor contents as a nested Python list (or scalar).

        Converts the tensor to a standard Python object:

        * 0-d tensor → a Python scalar (``int``, ``float``, ``bool``, or
          ``complex``).
        * 1-d tensor → a flat ``list``.
        * N-d tensor → a nested ``list`` of depth ``N``.

        Returns
        -------
        list or int or float or bool or complex
            Nested Python representation of the tensor data.

        Examples
        --------
        >>> import lucid
        >>> lucid.tensor([[1, 2], [3, 4]]).tolist()
        [[1, 2], [3, 4]]
        >>> lucid.tensor(3.14).tolist()   # the float32 value, widened exactly
        3.140000104904175

        Notes
        -----
        Numpy-free for every supported dtype (F16 / F32 / F64 / I8 / I16 /
        I32 / I64 / Bool / C64).  Delegates to the engine-side
        ``TensorImpl.tolist()``, which mirrors ``item()``'s dtype dispatch
        but walks the full shape recursively and yields Python ``complex``
        for C64 leaves.  Forces a device-to-host synchronisation for
        Metal tensors via the underlying ``to_bytes()`` snapshot.
        Autograd information is dropped — the returned Python objects are
        pure value copies.
        """
        return self._impl.tolist()  # type: ignore[no-any-return]

    def unfold(self, dimension: int, size: int, step: int) -> Tensor:
        r"""Return a view with an extra dimension containing sliding-window slices.

        Extracts non-overlapping or overlapping windows of length ``size``
        along ``dimension``, advancing by ``step`` elements between windows.
        The output has one extra trailing dimension compared to the input.

        Parameters
        ----------
        dimension : int
            The axis along which to slide the window.
        size : int
            Number of elements in each window.
        step : int
            Gap between the start positions of successive windows.

        Returns
        -------
        Tensor
            Tensor of shape
            :math:`(\ldots,\, L,\, \text{size})` where
            :math:`L = \lfloor (s_{\text{dim}} - \text{size}) /
            \text{step} \rfloor + 1`
            and :math:`s_{\text{dim}}` is the original size along
            ``dimension``.

        Notes
        -----
        Unfold is the fundamental primitive behind 1-D convolution and
        sliding-window aggregations.  On the CPU the result is a view of
        ``self``'s buffer, so a write through a window reaches ``self``; the
        windows overlap when ``step < size``, and a write through
        overlapping windows is refused, since an element then repeats.  On
        metal the result is a copy.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.arange(8, dtype=lucid.float32)
        >>> x.unfold(0, size=3, step=2).shape
        (3, 3)
        >>> # windows: [0,1,2], [2,3,4], [4,5,6]
        """
        return _wrap(_C_engine.unfold_dim(self._impl, dimension, size, step))

    @property
    def data(self) -> Tensor:
        r"""The tensor's underlying data, detached from gradient tracking.

        Returns a tensor over the same storage as ``self``, with
        ``requires_grad=False`` and no ``grad_fn``.  On the CPU it is one of
        ``self``'s views, so an in-place write through it reaches ``self``
        without autograd recording it.  Outside ``no_grad`` a write through
        the ``.data`` of a leaf that requires grad is refused, as a write to
        the leaf itself would be.

        ``t.data = value`` gives ``t`` the storage, shape, dtype and device of
        the Tensor ``value``, uncopied.  ``t`` stays the same object — a
        Parameter stays registered — with its ``requires_grad`` and a ``.grad``
        that still fits, and no graph records the swap.  A non-leaf ``t``, or
        an integer ``value`` for a ``t`` that requires grad, raises RuntimeError.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0], requires_grad=True)
        >>> x.data.requires_grad
        False
        >>> x.data = lucid.zeros(3)
        >>> x.shape, x.requires_grad
        ((3,), True)
        """
        return Tensor.__new_from_impl__(self._impl.data_alias())

    @data.setter
    def data(self, value: Tensor) -> None:
        """Give this tensor ``value``'s storage in place; see :attr:`data`."""
        if not isinstance(value, Tensor):
            raise TypeError(f"data has to be a Tensor, but got {type(value).__name__}")
        self._impl = _swap_data(self._impl, value._impl)

    # ── device/dtype conversion ───────────────────────────────────────────────
    # Injected by _tensor/_to.py after class definition.

    # ── shape helpers ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        r"""Return the size of the first dimension (``shape[0]``).

        Implements the standard Python ``len()`` protocol. Mirrors
        sequence semantics: the length is the number of top-level elements
        (rows) the tensor iterates over.

        Returns
        -------
        int
            ``self.shape[0]`` — the size of the outermost dimension.

        Raises
        ------
        TypeError
            If the tensor is 0-d (scalar). Python forbids ``len()`` on
            sized-zero scalar objects.

        Examples
        --------
        >>> import lucid
        >>> len(lucid.zeros(5, 3))
        5
        >>> len(lucid.zeros(7))
        7

        Notes
        -----
        Mirrors NumPy/sequence semantics: :math:`\text{len}(t) = s_0`.
        Iterating with :meth:`__iter__` yields exactly :math:`s_0` slices.
        """
        if not self._impl.shape:
            raise TypeError("len() of a 0-d tensor")
        return self._impl.shape[0]

    def __bool__(self) -> bool:
        r"""Convert a single-element tensor to a Python ``bool``.

        Implements the truthiness protocol used by ``if t:``, ``while t:``,
        and ``bool(t)``. Only defined for single-element tensors because
        the truth value of a multi-element tensor is ambiguous: "is any
        element true?" (``.any()``) versus "are all elements true?"
        (``.all()``) are different reductions.

        Returns
        -------
        bool
            The unboxed value of the single element interpreted as a
            Python ``bool`` (non-zero numerics map to ``True``).

        Raises
        ------
        RuntimeError
            If ``numel() != 1``. Use :meth:`any` / :meth:`all` explicitly
            on multi-element tensors.

        Notes
        -----
        This matches the convention adopted by every mainstream tensor
        framework. The error guards against subtle bugs such as
        ``if pred_tensor:`` silently truncating to the first element.

        Examples
        --------
        >>> import lucid
        >>> bool(lucid.tensor(1.0))
        True
        >>> bool(lucid.tensor(0))
        False

        Mathematically: defined only when :math:`\prod_i s_i = 1`. The
        single element :math:`x` maps to :math:`\text{bool}(x) = (x \neq 0)`.

        >>> bool(lucid.tensor([1.0, 0.0]))  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        RuntimeError: Boolean value of Tensor with more than one element is ambiguous
        """
        if self._impl.numel() != 1:
            raise RuntimeError(
                "Boolean value of Tensor with more than one element is ambiguous"
            )
        return bool(self.item())

    def __repr__(self) -> str:
        r"""Return a string representation suitable for debugging.

        Delegates to the formatted printer in :mod:`lucid._tensor._repr`,
        which renders shape, dtype, and (truncated) element values in a
        readable layout.

        Returns
        -------
        str
            Multi-line repr showing element values with appropriate
            precision, plus shape and dtype metadata when non-default.

        Examples
        --------
        >>> import lucid
        >>> repr(lucid.zeros(2, 3))  # doctest: +SKIP
        'tensor([[0., 0., 0.],\n        [0., 0., 0.]])'

        Notes
        -----
        Display only — does **not** participate in autograd and is one of
        the bridge boundaries permitted by rule **H4** (``_repr.py``).
        For Metal-resident tensors this forces a host synchronisation to
        materialise element values for printing.
        """
        return _tensor_repr(self)

    # hash: identity-based so tensors can be used as dict keys
    __hash__ = object.__hash__  # type: ignore[assignment]

    # ── iteration & formatting ────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Tensor]:
        r"""Iterate over the first dimension of the tensor.

        Yields successive slices ``self[i]`` for ``i`` in ``range(shape[0])``,
        so iterating an :math:`N`-d tensor produces :math:`(N-1)`-d slices.
        Implements the standard Python iterator protocol.

        Yields
        ------
        Tensor
            The ``i``-th slice along axis 0.

        Raises
        ------
        TypeError
            If the tensor is 0-d (scalar). Scalars have no first dimension
            to iterate over.

        Examples
        --------
        >>> import lucid
        >>> for row in lucid.arange(6, dtype=lucid.int64).reshape(3, 2):
        ...     print(row.tolist())
        [0, 1]
        [2, 3]
        [4, 5]

        Notes
        -----
        Yields :math:`s_0` slices each of shape :math:`(s_1, \ldots, s_{d-1})`.
        For very large leading axes prefer chunked iteration via
        :func:`lucid.split` to amortise per-slice overhead.
        """
        if not self._impl.shape:
            raise TypeError("iteration over a 0-d tensor")
        for i in range(self._impl.shape[0]):
            yield self[i]  # type: ignore[misc]

    def __format__(self, format_spec: str) -> str:
        """Apply a Python format spec, supporting single-element tensors.

        For 1-element tensors, the spec is applied to the unboxed Python
        scalar via :meth:`item` — useful inside f-strings to control
        precision. For multi-element tensors the spec is ignored and the
        regular :meth:`__repr__` output is returned.

        Parameters
        ----------
        format_spec : str
            Standard Python format spec (e.g. ``".3f"``, ``"+.2e"``).

        Returns
        -------
        str
            Formatted scalar value or the full repr.

        Examples
        --------
        >>> import lucid
        >>> f"{lucid.tensor(3.14159):.2f}"
        '3.14'
        >>> f"{lucid.tensor(2.5):+.1e}"
        '+2.5e+00'

        Notes
        -----
        For single-element tensors the call is equivalent to
        ``format(t.item(), spec)``. For multi-element tensors the spec is
        ignored to avoid ambiguous element-wise formatting semantics.
        """
        if self._impl.numel() == 1:
            return format(self.item(), format_spec)
        return repr(self)

    # ── new_* convenience constructors ────────────────────────────────────────

    def new_empty(
        self,
        *size: int,
        dtype: _dtype_cls | None = None,
        device: _device_cls | str | None = None,
        requires_grad: bool = False,
    ) -> Tensor:
        r"""Return an uninitialized tensor of the given shape.

        The returned tensor inherits this tensor's ``dtype`` and ``device``
        unless overridden.  The contents are **undefined** — do not read
        values without first writing them.

        Parameters
        ----------
        *size : int
            Dimensions of the output tensor.
        dtype : lucid.dtype, optional
            Element type.  Defaults to ``self.dtype``.
        device : str or lucid.device, optional
            Target device.  Defaults to ``self.device``.
        requires_grad : bool, optional
            Enable gradient tracking on the result.  Default ``False``.

        Returns
        -------
        Tensor
            Uninitialized tensor of shape ``size``.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(2, 3, dtype=lucid.float64)
        >>> y = x.new_empty(4, 5)
        >>> y.shape, y.dtype
        ((4, 5), lucid.float64)

        Notes
        -----
        Allocates :math:`\prod_i s_i \cdot e` bytes of uninitialised
        memory where :math:`e` is ``dtype.itemsize``. Faster than
        :meth:`new_zeros` because the engine skips the zero-fill kernel.
        Must be followed by a write before any read.
        """
        _dtype = dtype or self.dtype
        _device = device or self.device
        return _empty(
            *size,
            dtype=_dtype,
            device=_device,
            requires_grad=requires_grad,
        )

    def new_zeros(
        self,
        *size: int,
        dtype: _dtype_cls | None = None,
        device: _device_cls | str | None = None,
        requires_grad: bool = False,
    ) -> Tensor:
        """Return a zero-filled tensor of the given shape.

        The returned tensor inherits this tensor's ``dtype`` and ``device``
        unless overridden.  All elements are initialised to ``0``.

        Parameters
        ----------
        *size : int
            Dimensions of the output tensor.
        dtype : lucid.dtype, optional
            Element type.  Defaults to ``self.dtype``.
        device : str or lucid.device, optional
            Target device.  Defaults to ``self.device``.
        requires_grad : bool, optional
            Enable gradient tracking on the result.  Default ``False``.

        Returns
        -------
        Tensor
            Zero tensor of shape ``size``.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.ones(2, dtype=lucid.int32)
        >>> x.new_zeros(3, 3)
        tensor([[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]], dtype=lucid.int32)

        Notes
        -----
        Initialises every element to the additive identity :math:`0`. The
        zero-fill is delegated to a fused engine kernel — Accelerate
        ``catlas_*set`` on CPU and MLX broadcast-fill on Metal.
        """
        _dtype = dtype or self.dtype
        _device = device or self.device
        return _zeros(
            *size,
            dtype=_dtype,
            device=_device,
            requires_grad=requires_grad,
        )

    def new_ones(
        self,
        *size: int,
        dtype: _dtype_cls | None = None,
        device: _device_cls | str | None = None,
        requires_grad: bool = False,
    ) -> Tensor:
        """Return an all-ones tensor of the given shape.

        The returned tensor inherits this tensor's ``dtype`` and ``device``
        unless overridden.  All elements are initialised to ``1``.

        Parameters
        ----------
        *size : int
            Dimensions of the output tensor.
        dtype : lucid.dtype, optional
            Element type.  Defaults to ``self.dtype``.
        device : str or lucid.device, optional
            Target device.  Defaults to ``self.device``.
        requires_grad : bool, optional
            Enable gradient tracking on the result.  Default ``False``.

        Returns
        -------
        Tensor
            All-ones tensor of shape ``size``.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(2, dtype=lucid.float16)
        >>> x.new_ones(2, 4)
        tensor([[1., 1., 1., 1.],
                [1., 1., 1., 1.]], dtype=lucid.float16)

        Notes
        -----
        Initialises every element to the multiplicative identity :math:`1`.
        For arbitrary fill values use :meth:`new_full`.
        """
        _dtype = dtype or self.dtype
        _device = device or self.device
        return _ones(
            *size,
            dtype=_dtype,
            device=_device,
            requires_grad=requires_grad,
        )

    def new_full(
        self,
        size: tuple[int, ...],
        fill_value: float,
        *,
        dtype: _dtype_cls | None = None,
        device: _device_cls | str | None = None,
        requires_grad: bool = False,
    ) -> Tensor:
        r"""Return a tensor of the given shape filled with a constant value.

        The returned tensor inherits this tensor's ``dtype`` and ``device``
        unless overridden.  Every element is set to ``fill_value``.

        Parameters
        ----------
        size : tuple[int, ...]
            Shape of the output tensor.
        fill_value : float
            Scalar value to fill the tensor with.
        dtype : lucid.dtype, optional
            Element type.  Defaults to ``self.dtype``.
        device : str or lucid.device, optional
            Target device.  Defaults to ``self.device``.
        requires_grad : bool, optional
            Enable gradient tracking on the result.  Default ``False``.

        Returns
        -------
        Tensor
            Constant-filled tensor of shape ``size``.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(1)
        >>> x.new_full((2, 3), fill_value=7.0)
        tensor([[7., 7., 7.],
                [7., 7., 7.]])

        Notes
        -----
        Constant tensor :math:`f \cdot \mathbf{1}` where :math:`f` is
        ``fill_value`` and :math:`\mathbf{1}` is the all-ones tensor of
        the given shape. ``fill_value`` is promoted to ``dtype`` before
        the broadcast fill.
        """
        _dtype = dtype or self.dtype
        _device = device or self.device
        return _full(
            size,
            fill_value,
            dtype=_dtype,
            device=_device,
            requires_grad=requires_grad,
        )

    def new_tensor(
        self,
        data: np.ndarray | list[object] | int | float | bool | Tensor,
        *,
        dtype: _dtype_cls | None = None,
        device: _device_cls | str | None = None,
        requires_grad: bool = False,
    ) -> Tensor:
        """Return a new tensor constructed from ``data``, inheriting dtype/device.

        Creates a fresh tensor from the provided data, using this tensor's
        ``dtype`` and ``device`` as defaults.  The data is always **copied**;
        the result does not share storage with ``data`` even if ``data`` is
        already a :class:`Tensor`.

        Parameters
        ----------
        data : array_like or Tensor
            Input data — nested Python lists, scalars, or an existing tensor.
        dtype : lucid.dtype, optional
            Element type.  Defaults to ``self.dtype``.
        device : str or lucid.device, optional
            Target device.  Defaults to ``self.device``.
        requires_grad : bool, optional
            Enable gradient tracking on the result.  Default ``False``.

        Returns
        -------
        Tensor
            New tensor containing a copy of ``data``.

        Examples
        --------
        >>> import lucid
        >>> proto = lucid.zeros(1, dtype=lucid.float64)
        >>> t = proto.new_tensor([[1, 2], [3, 4]])
        >>> t.dtype
        lucid.float64

        Notes
        -----
        Always copies — never aliases ``data``'s storage even when ``data``
        is already a ``Tensor``. Routes through the same
        ``_to_impl`` bridge boundary as the public constructor (rule **H4**).
        """
        _dtype = dtype or self.dtype
        _device = device or self.device
        return _tensor_fn(
            data,
            dtype=_dtype,
            device=_device,
            requires_grad=requires_grad,
        )

    # ── size / element info ───────────────────────────────────────────────────

    def element_size(self) -> int:
        r"""Return the size in bytes of a single element.

        Determined entirely by :attr:`dtype`.  Common values:

        * ``float32`` / ``int32`` → 4 bytes
        * ``float64`` / ``int64`` → 8 bytes
        * ``float16`` / ``bfloat16`` / ``int16`` → 2 bytes
        * ``int8`` / ``bool_`` → 1 byte
        * ``complex64`` → 8 bytes (two 32-bit floats)

        Returns
        -------
        int
            Number of bytes per element.

        Notes
        -----
        The total memory footprint of the tensor is:

        .. math::

            \text{nbytes} = \text{numel()} \times \text{element\_size()}

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3, dtype=lucid.float32).element_size()
        4
        >>> lucid.zeros(3, dtype=lucid.float64).element_size()
        8
        """
        return self.dtype.itemsize

    @property
    def itemsize(self) -> int:
        r"""Bytes per element — alias for :meth:`element_size`.

        Provided as a property (rather than a method) for NumPy-style
        attribute access:  ``tensor.itemsize`` instead of
        ``tensor.element_size()``.

        Returns
        -------
        int
            Number of bytes per element.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(5, dtype=lucid.int16).itemsize
        2

        Notes
        -----
        Total footprint of the tensor satisfies
        :math:`\text{nbytes} = \text{numel} \cdot \text{itemsize}`.
        """
        return self.dtype.itemsize

    @property
    def nbytes(self) -> int:
        r"""Total number of bytes occupied by the tensor's data buffer.

        Equals ``numel() * element_size()``.  Useful for estimating memory
        usage and for setting buffer sizes when interoperating with C or
        Metal shaders.

        Returns
        -------
        int
            Total byte count of the data storage.

        Notes
        -----
        .. math::

            \text{nbytes} = \prod_{i} s_i \times e

        where :math:`s_i` are the dimension sizes and :math:`e` is the
        element size in bytes.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(4, 4, dtype=lucid.float32).nbytes
        64
        >>> lucid.zeros(4, 4, dtype=lucid.float64).nbytes
        128
        """
        return self._impl.numel() * self.dtype.itemsize

    def stride(self, dim: int | None = None) -> tuple[int, ...] | int:
        r"""Return the strides of the tensor in *element* counts.

        Parameters
        ----------
        dim : int, optional
            If given, return the stride along that dimension only.

        Returns
        -------
        tuple[int, ...] or int
            Element-count strides (same semantics as the reference framework).

        Notes
        -----
        For a C-contiguous tensor of shape :math:`(s_0, \ldots, s_{d-1})`
        the element strides satisfy the row-major recurrence

        .. math::

            \text{stride}[d-1] = 1, \quad
            \text{stride}[i] = \text{stride}[i+1] \cdot s_{i+1}.

        Non-contiguous tensors (e.g. transposed or sliced views) may have
        arbitrary strides; the address of element ``[i_0, \ldots, i_{d-1}]``
        is ``base + \sum_k i_k \cdot \text{stride}[k]``.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3, 4).stride()
        (4, 1)
        >>> lucid.zeros(3, 4).stride(0)
        4
        """
        byte_strides: list[int] = list(self._impl.stride)
        itemsz: int = self.dtype.itemsize
        elem_strides = tuple(s // itemsz for s in byte_strides)
        if dim is None:
            return elem_strides
        return elem_strides[dim]

    def data_ptr(self) -> int:
        r"""Return the address of the first element as an integer.

        The address is the start of :meth:`untyped_storage` plus
        :meth:`storage_offset` elements.  A metal tensor is evaluated first,
        so its buffer exists and can be read in unified memory.

        Returns
        -------
        int
            Address of the first element.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3)
        >>> y = x
        >>> x.data_ptr() == y.data_ptr()
        True
        >>> z = lucid.zeros(3)
        >>> x.data_ptr() != z.data_ptr()
        True

        Notes
        -----
        An in-place op that computes its result into a new buffer moves the
        address; compare addresses to ask whether two tensors read the same
        memory at this moment.
        """
        return int(self._impl.data_ptr())

    def storage_offset(self) -> int:
        r"""Return the offset, in elements, of the first element in its storage.

        A tensor that owns its allocation starts at ``0``.  A view reports
        where its first element sits in :meth:`untyped_storage`, and
        :meth:`as_strided` counts offsets from that same allocation start:
        :math:`\text{data\_ptr}(t) = \text{untyped\_storage}(t).\text{data\_ptr}()
        + \text{storage\_offset}(t) \times \text{element\_size}(t)`.

        Returns
        -------
        int
            Element offset of the first element within the underlying storage.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3, 4).storage_offset()
        0
        >>> lucid.arange(6.).reshape(2, 3)[1].storage_offset()
        3
        """
        return int(self._impl.storage_offset_bytes()) // self.element_size()

    @property
    def H(self) -> Tensor:
        r"""Conjugate (Hermitian) transpose of the last two axes.

        For real-valued tensors this is identical to :attr:`mT`. For
        complex tensors the elements are conjugated before transposing,
        producing the Hermitian adjoint :math:`A^{\mathsf{H}}` familiar
        from linear algebra:

        .. math::

            (A^{\mathsf{H}})_{ij} = \overline{A_{ji}}

        Returns
        -------
        Tensor
            A tensor of shape :math:`(\ldots, n, m)` for an input of
            shape :math:`(\ldots, m, n)`. Real inputs round-trip through
            :attr:`mT`; complex inputs are first conjugated by
            ``lucid._ops.composite.conj``.

        Notes
        -----
        Equivalent to ``x.conj().mT`` for complex tensors. For batched
        linear algebra the leading dimensions are untouched.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.arange(6).reshape(2, 3)
        >>> x.H.shape
        (3, 2)
        """
        if self.is_complex():
            from lucid._ops.composite import conj as _conj

            return _conj(self).mT
        return self.mT

    def type(
        self,
        dtype: _dtype_cls | _C_engine.Dtype | str | None = None,
        non_blocking: bool = False,
    ) -> str | Tensor:
        """Return the legacy type name, or cast to a dtype or legacy type.

        * ``t.type()`` — the name, ``'lucid.FloatTensor'`` (``lucid.metal.`` on Metal).
        * ``t.type(lucid.float64)`` / ``t.type('lucid.DoubleTensor')`` — cast as
          :meth:`to` casts, a name to its device too; ``self`` if it already is.

        Parameters
        ----------
        dtype : lucid.dtype or str, optional
            The dtype, or its legacy name: ``FloatTensor``, ``DoubleTensor``,
            ``HalfTensor``, ``BFloat16Tensor``, ``CharTensor`` (int8),
            ``ShortTensor``, ``IntTensor``, ``LongTensor``, ``BoolTensor``,
            ``ComplexFloatTensor`` or ``ComplexDoubleTensor``, prefixed with
            ``lucid.`` (CPU) or ``lucid.metal.``.  ``None`` returns ``self``'s.
        non_blocking : bool, optional
            Read and ignored: a transfer is synchronous.

        Returns
        -------
        str or Tensor
            The name when ``dtype is None``, otherwise the cast tensor: on the
            name's device for a name, on ``self``'s for a dtype.

        Raises
        ------
        ValueError
            If ``dtype`` is a string or a Python type that names no legacy
            type — ``"float64"`` and ``float`` included, as in the reference.
        TypeError
            If ``dtype`` is neither a dtype, a string nor a type.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3).type()
        'lucid.FloatTensor'
        >>> lucid.zeros(3).type('lucid.LongTensor').dtype
        lucid.int64
        >>> lucid.zeros(3).type(lucid.float16).type()
        'lucid.HalfTensor'
        """
        del non_blocking  # the reference's parameter; transfers are synchronous
        if dtype is None:
            return _legacy_type_name(self._impl)
        return self.to(*_type_target(dtype))

    def get_device(self) -> int:
        r"""Return the device index.

        Returns ``0`` for Metal (GPU) tensors and ``-1`` for CPU tensors,
        following the convention adopted by the reference framework.

        Returns
        -------
        int
            ``0`` for Metal-resident tensors; ``-1`` for CPU-resident tensors.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3).get_device()
        -1
        >>> lucid.zeros(3).metal().get_device()  # doctest: +SKIP
        0

        Notes
        -----
        Indicator-style encoding:
        :math:`\text{idx}(t) = 0 \cdot \mathbf{1}\{\text{is\_metal}\} + (-1) \cdot \mathbf{1}\{\text{is\_cpu}\}`.
        Lucid only supports a single Metal device on Apple Silicon, so a
        positive index is always ``0``.
        """
        return 0 if self.is_metal else -1

    def pin_memory(self, device: object = None) -> Tensor:
        r"""Return ``self`` — pinned memory is a no-op on Apple Silicon.

        Apple Silicon uses unified memory: CPU and GPU share the same
        physical DRAM, so the "page-lock host memory to accelerate
        host-to-device DMA" concept from discrete-GPU frameworks does not
        apply. This method exists for API compatibility and simply
        returns the tensor unchanged.

        Parameters
        ----------
        device : object, optional
            Accepted but ignored. Present for signature compatibility.

        Returns
        -------
        Tensor
            ``self``, untouched.

        Notes
        -----
        See also :meth:`is_pinned` (always ``False``) and
        :meth:`share_memory_` (also a no-op). The function is the
        identity: :math:`\text{pin\_memory}(t) \equiv t`.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3)
        >>> x.pin_memory() is x
        True
        """
        del device  # reference-signature parity; unified memory on Apple Silicon
        return self

    def is_pinned(self, device: object = None) -> bool:
        r"""Return ``False`` — pinned memory is not applicable on Apple Silicon.

        Apple Silicon's unified-memory architecture makes the distinction
        between "pageable" and "page-locked" host memory irrelevant: CPU
        and GPU already see the same DRAM. This predicate is provided for
        API compatibility and always reports ``False``.

        Parameters
        ----------
        device : object, optional
            Accepted but ignored.

        Returns
        -------
        bool
            Always ``False``.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3).is_pinned()
        False

        Notes
        -----
        Identically false: :math:`\text{is\_pinned}(t) \equiv \text{False}`.
        Unified memory makes the concept moot — all host buffers are
        already DMA-accessible to the GPU without page-locking.
        """
        del device  # reference-signature parity; unified memory on Apple Silicon
        return False

    def reshape_as(self, other: Tensor) -> Tensor:
        r"""Return a tensor with the same data reshaped to ``other.shape``.

        Convenience wrapper around :func:`reshape` that takes the target
        shape from another tensor instead of as a tuple. Element count
        must agree:

        .. math::

            \prod_i \text{self.shape}[i] = \prod_j \text{other.shape}[j]

        Parameters
        ----------
        other : Tensor
            Tensor whose shape will be adopted. Only ``other.shape`` is
            consulted; the values and dtype of ``other`` are ignored.

        Returns
        -------
        Tensor
            A view (or copy if storage layout requires) of ``self`` with
            shape ``other.shape``.

        Raises
        ------
        RuntimeError
            If ``self.numel() != other.numel()``.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.arange(12)
        >>> proto = lucid.zeros(3, 4)
        >>> x.reshape_as(proto).shape
        (3, 4)

        Notes
        -----
        Reshape preserves element order under row-major (C) traversal;
        the element at flat index :math:`k = \sum_i i_k \cdot \prod_{j>k} s_j`
        in ``self`` becomes the element at the same flat index in the
        result. A view is returned when the source is contiguous;
        otherwise a contiguous copy is materialised first.
        """
        return Tensor.__new_from_impl__(
            _C_engine.reshape(self._impl, list(other._impl.shape))
        )

    @final
    class _UntypedStorage:
        """Minimal storage object returned by :meth:`Tensor.untyped_storage`."""

        def __init__(self, tensor: Tensor) -> None:
            self._tensor = tensor

        def data_ptr(self) -> int:
            """Address of the first byte of the underlying allocation."""
            return int(self._tensor._impl.storage_data_ptr())

        def size(self) -> int:
            """Total allocation size in bytes."""
            return int(self._tensor._impl.storage_nbytes())

        def nbytes(self) -> int:
            return self.size()

        def __len__(self) -> int:
            return self.size()

        def __repr__(self) -> str:
            return f"UntypedStorage(nbytes={self.size()}, device={self._tensor.device})"

    def untyped_storage(self) -> _UntypedStorage:
        r"""Return a minimal storage view of the underlying allocation.

        Exposes :meth:`data_ptr`, :meth:`size`, and :meth:`nbytes` for
        aliasing checks and memory accounting.  A base tensor and all views
        of it report the same allocation base address and byte size.

        Returns
        -------
        Tensor._UntypedStorage
            Storage handle reporting the byte count and base address.

        Notes
        -----
        Introspection-only; mutating through this handle is not supported.
        On CPU the engine tracks the root allocation window across views; on
        Metal a linked view reports the root of its live view chain (an
        unlinked tensor or dead root reports the last live tensor in its chain).

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(4, dtype=lucid.float32)
        >>> s = x.untyped_storage()
        >>> (s.size(), len(s), x[2:].untyped_storage().size())
        (16, 16, 16)
        """
        return Tensor._UntypedStorage(self)

    def is_floating_point(self) -> bool:
        r"""Return ``True`` if the dtype is a floating-point type.

        Floating-point dtypes recognised by Lucid are ``float16``,
        ``float32``, ``float64``, and ``bfloat16``.

        Returns
        -------
        bool
            ``True`` for the four floating-point dtypes above; ``False``
            for integer, boolean, and complex dtypes.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3).is_floating_point()
        True
        >>> lucid.zeros(3, dtype=lucid.int64).is_floating_point()
        False

        Notes
        -----
        Set-membership predicate
        :math:`\text{is\_float}(t) = (t.\text{dtype} \in \mathcal{F})`
        with :math:`\mathcal{F} = \{\text{float16}, \text{float32}, \text{float64}, \text{bfloat16}\}`.
        Many differentiable ops are only defined on :math:`\mathcal{F}`;
        integer and boolean dtypes block autograd at the operator level.
        """
        return self.dtype in (float16, float32, float64, bfloat16)

    def is_complex(self) -> bool:
        r"""Return ``True`` if the dtype is a complex type.

        Two complex dtypes exist: ``complex64`` (two 32-bit floats per
        element) and ``complex128`` (two 64-bit floats).

        Returns
        -------
        bool
            ``True`` for ``lucid.complex64`` and ``lucid.complex128``;
            ``False`` otherwise.

        Examples
        --------
        >>> import lucid
        >>> lucid.zeros(3, dtype=lucid.complex64).is_complex()
        True
        >>> lucid.zeros(3).is_complex()
        False

        Notes
        -----
        Complex tensors store interleaved real/imag pairs:
        :math:`z_i = a_i + b_i \mathrm{i}` with :math:`a_i, b_i \in \mathbb{R}`.
        ``complex128`` is backed by two 64-bit floats and exists for
        ``linalg.eig``, whose output dtype is not a choice — the
        eigenvalues of an ``f64`` matrix need the wider lanes, and
        narrowing them to ``complex64`` would shed eight decimal digits.
        """
        return self.dtype in (complex64, complex128)

    def share_memory_(self) -> Self:
        r"""Mark storage as shareable across processes — a no-op on Apple Silicon.

        On platforms with separate CPU and GPU address spaces this method
        moves the storage into a shared-memory segment so that worker
        processes (e.g. DataLoader workers) can read it without copying.
        Apple Silicon's unified-memory architecture makes this unnecessary:
        all tensors are already addressable from every process that holds
        a reference to the buffer. Returns ``self`` for chaining.

        Returns
        -------
        Tensor
            ``self``, unchanged.

        Notes
        -----
        Provided for API compatibility. See also :meth:`is_pinned` and
        :meth:`pin_memory`, which are no-ops for the same reason. The
        operation is the identity:
        :math:`\text{share\_memory\_}(t) \equiv t`.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3)
        >>> x.share_memory_() is x
        True
        """
        return self

    # ── Phase N convenience methods ───────────────────────────────────────────

    def fill_(self, value: float) -> Self:
        r"""Fill the tensor with a scalar value in-place.

        Mutates the tensor's storage so every element becomes ``value``.
        Implemented by materialising a constant tensor with
        ``_C_engine.full`` and copying it into ``self``'s storage; the
        original ``TensorImpl`` (and thus identity) is preserved.

        Parameters
        ----------
        value : float
            Scalar value to broadcast into every position. Promoted to the
            tensor's dtype on copy.

        Returns
        -------
        Tensor
            ``self`` (in-place); useful for method chaining.

        Notes
        -----
        In-place operations bypass autograd's view tracking for
        performance. Calling ``fill_`` on a leaf tensor with
        ``requires_grad=True`` may raise a runtime error from the
        autograd engine.

        Mathematically the result is the constant tensor
        :math:`v \cdot \mathbf{1}` with the same shape as ``self``;
        every entry satisfies :math:`x_i \leftarrow v`.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.empty(3)
        >>> _ = x.fill_(0.5)
        >>> x.tolist()
        [0.5, 0.5, 0.5]
        """
        filled = _C_engine.full(
            list(self._impl.shape), value, self._impl.dtype, self._impl.device
        )
        if (
            self._impl.requires_grad
            and not self._impl.is_leaf
            and _C_engine.grad_enabled()
        ):
            # The result no longer depends on the old values, but it stays in
            # the graph with that zero derivative, as the reference keeps it:
            # detached, a loss built from it could not be differentiated.
            from lucid.autograd._overwritten import overwritten

            filled = overwritten(self, _wrap(filled))._impl
        # ``assign_from``, not ``copy_from``: the values are overwritten, so
        # the tensor no longer depends on what it was.  Written through the
        # raw buffer copy, the graph position stayed put and the gradient
        # flowed back through it as if the fill had not happened.  A tensor
        # that shares its buffer with a view goes the in-place ops' way,
        # Re-derives views' graphs along with its own.
        if self._impl.is_aliased():
            _C_engine.assign_inplace(self._impl, filled, "fill_")
        else:
            self._impl.assign_from(filled, "fill_")
        return self

    def copy_(self, other: Tensor, non_blocking: bool = False) -> Self:
        r"""Copy data from ``other`` into ``self`` in-place.

        Overwrites ``self``'s storage with ``other``'s values. Broadcasting
        is permitted: ``other`` may have a shape that broadcasts to
        ``self.shape``. The dtype of ``self`` is preserved; ``other``
        values are cast as necessary.

        Parameters
        ----------
        other : Tensor
            Source tensor. Made contiguous before the copy to guarantee
            a stride-compatible memory layout.
        non_blocking : bool, optional
            Accepted for the reference's signature; a copy here is
            synchronous.  Default ``False``.

        Returns
        -------
        Tensor
            ``self`` after the copy.

        Notes
        -----
        Unlike :meth:`clone`, ``copy_`` does not allocate a new tensor —
        only ``self``'s storage is written. The autograd graph *is*
        extended: after the copy ``self`` depends on ``other``, not on
        whatever produced ``self``, so a backward pass through ``self``
        deposits its gradient on ``other``. Element-wise:

        .. math::

            \text{self}_i \leftarrow \text{cast}_{\text{self.dtype}}(\text{other}_i),

        with right-aligned broadcasting from ``other.shape`` to ``self.shape``.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.zeros(3)
        >>> b = lucid.tensor([1.0, 2.0, 3.0])
        >>> _ = a.copy_(b)
        >>> a.tolist()
        [1.0, 2.0, 3.0]
        """
        del non_blocking  # a copy is synchronous; the reference's flag is read
        dst, src = self._impl, other._impl
        if src.device != dst.device and dst.device == _C_engine.Device.CPU:
            src = _C_engine.to_device(src, dst.device)
        if src.dtype != dst.dtype:
            src = _C_engine.astype(src, dst.dtype)
        if src.device != dst.device:
            src = _C_engine.to_device(src, dst.device)
        if list(src.shape) != list(dst.shape):
            src = _C_engine.broadcast_to(src, list(dst.shape))
        src = _C_engine.contiguous(src)
        if dst.is_aliased():
            _C_engine.assign_inplace(dst, src, "copy_")
        else:
            dst.assign_from(src, "copy_")
        return self

    # ── index writes (index_add / index_copy / index_fill / index_put) ────────
    # Each in-place form runs its out-of-place op and writes the result into
    # ``self`` through ``_write_inplace``: into the buffer ``self``'s views
    # share, with the result's place in the autograd graph, refused on a leaf
    # that requires grad while autograd records — as ``fill_`` and ``copy_``.

    def index_add(
        self, dim: int, index: Tensor, source: Tensor, *, alpha: float = 1.0
    ) -> Tensor:
        r"""Return a copy of ``self`` with ``alpha * source`` added at ``index`` along ``dim``.

        Slice ``i`` of ``source`` along ``dim`` is added to slice
        ``index[i]`` of the copy.  Repeated indices accumulate.

        Parameters
        ----------
        dim : int
            Axis along which ``index`` addresses slices.  Negative values
            count from the end.
        index : Tensor
            0-D or 1-D ``int32`` / ``int64`` tensor of ``m`` positions
            naming the slices to add to; a 0-D one names a single slice.
        source : Tensor
            Same shape as ``self`` except ``source.shape[dim] == m``, and
            ``self``'s dtype: another dtype is refused
            (``lucid._C.engine.DtypeMismatch``), not cast.
        alpha : float, optional
            Scale applied to ``source`` before it is added.  Default ``1.0``.

        Returns
        -------
        Tensor
            A new tensor with ``self``'s shape and dtype; ``self`` is
            unchanged.

        Notes
        -----
        With ``dim = 0``,

        .. math::

            \text{out}[\text{index}[i]] \mathrel{+}= \alpha \cdot \text{source}[i].

        Differentiable through ``self`` and ``source``.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.ones(5)
        >>> x.index_add(0, lucid.tensor([0, 0, 4]), lucid.tensor([1.0, 2.0, 3.0]))
        tensor([4., 1., 1., 1., 4.])
        """
        from lucid._ops.composite.indexing import index_add as _index_add

        return _index_add(self, dim, index, source, alpha=alpha)

    def index_add_(
        self, dim: int, index: Tensor, source: Tensor, *, alpha: float = 1.0
    ) -> Self:
        r"""Add ``alpha * source`` into ``self`` at ``index`` along ``dim``, in place.

        In-place form of :meth:`index_add`: the sums are written into
        ``self``'s storage, so a view of ``self`` sees them.

        Parameters
        ----------
        dim : int
            Axis along which ``index`` addresses slices.  Negative values
            count from the end.
        index : Tensor
            0-D or 1-D ``int32`` / ``int64`` tensor of ``m`` positions
            naming the slices to add to; a 0-D one names a single slice.
        source : Tensor
            Same shape as ``self`` except ``source.shape[dim] == m``, and
            ``self``'s dtype: another dtype is refused
            (``lucid._C.engine.DtypeMismatch``), not cast.
        alpha : float, optional
            Scale applied to ``source`` before it is added.  Default ``1.0``.

        Returns
        -------
        Tensor
            ``self``.

        Raises
        ------
        RuntimeError
            If ``self`` is a leaf that requires grad and autograd is
            recording — wrap the call in :func:`lucid.no_grad`.

        Notes
        -----
        With ``dim = 0``,

        .. math::

            \text{self}[\text{index}[i]] \mathrel{+}= \alpha \cdot \text{source}[i],

        repeated indices accumulating.  ``self`` takes the sum's place in
        the autograd graph, so the gradient reaches ``source`` and whatever
        produced ``self``.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3, 2)
        >>> src = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> _ = x.index_add_(0, lucid.tensor([0, 2]), src, alpha=2.0)
        >>> x
        tensor([[2., 4.], [0., 0.], [6., 8.]])
        """
        from lucid._ops.composite.indexing import _write_inplace, index_add

        return _write_inplace(
            self,
            "index_add_",
            lambda base: index_add(base, dim, index, source, alpha=alpha),
            source,
        )

    def index_copy(self, dim: int, index: Tensor, source: Tensor) -> Tensor:
        r"""Return a copy of ``self`` with the slices at ``index`` along ``dim`` replaced.

        Slice ``i`` of ``source`` along ``dim`` replaces slice ``index[i]``
        of the copy.

        Parameters
        ----------
        dim : int
            Axis along which ``index`` addresses slices.  Negative values
            count from the end.
        index : Tensor
            0-D or 1-D ``int32`` / ``int64`` tensor of ``m`` positions
            naming the slices to replace; a 0-D one names a single slice.
        source : Tensor
            Same shape as ``self`` except ``source.shape[dim] == m``, and
            ``self``'s dtype: another dtype is refused
            (``lucid._C.engine.DtypeMismatch``), not cast.

        Returns
        -------
        Tensor
            A new tensor with ``self``'s shape and dtype; ``self`` is
            unchanged.

        Notes
        -----
        With ``dim = 0``,

        .. math::

            \text{out}[\text{index}[i]] = \text{source}[i].

        The gradient reaches ``source`` at the copied slices and ``self``
        everywhere else.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(2, 3)
        >>> src = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> x.index_copy(1, lucid.tensor([2, 0]), src)
        tensor([[2., 0., 1.], [4., 0., 3.]])
        """
        from lucid._ops.composite.indexing import index_copy as _index_copy

        return _index_copy(self, dim, index, source)

    def index_copy_(self, dim: int, index: Tensor, source: Tensor) -> Self:
        r"""Replace the slices of ``self`` at ``index`` along ``dim`` with ``source``, in place.

        In-place form of :meth:`index_copy`: the slices are written into
        ``self``'s storage, so a view of ``self`` sees them.

        Parameters
        ----------
        dim : int
            Axis along which ``index`` addresses slices.  Negative values
            count from the end.
        index : Tensor
            0-D or 1-D ``int32`` / ``int64`` tensor of ``m`` positions
            naming the slices to replace; a 0-D one names a single slice.
        source : Tensor
            Same shape as ``self`` except ``source.shape[dim] == m``, and
            ``self``'s dtype: another dtype is refused
            (``lucid._C.engine.DtypeMismatch``), not cast.

        Returns
        -------
        Tensor
            ``self``.

        Raises
        ------
        RuntimeError
            If ``self`` is a leaf that requires grad and autograd is
            recording — wrap the call in :func:`lucid.no_grad`.

        Notes
        -----
        With ``dim = 0``,

        .. math::

            \text{self}[\text{index}[i]] \leftarrow \text{source}[i].

        The overwritten slices pass no gradient back to what produced
        ``self``; ``source`` receives it there instead.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(2, 3)
        >>> src = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> _ = x.index_copy_(1, lucid.tensor([2, 0]), src)
        >>> x
        tensor([[2., 0., 1.], [4., 0., 3.]])
        """
        from lucid._ops.composite.indexing import _write_inplace, index_copy

        return _write_inplace(
            self,
            "index_copy_",
            lambda base: index_copy(base, dim, index, source),
            source,
        )

    def index_fill(self, dim: int, index: Tensor, value: float) -> Tensor:
        r"""Return a copy of ``self`` with the slices at ``index`` along ``dim`` set to ``value``.

        Parameters
        ----------
        dim : int
            Axis along which ``index`` addresses slices.  Negative values
            count from the end.
        index : Tensor
            0-D or 1-D ``int32`` / ``int64`` tensor naming the slices to
            fill; a 0-D one names a single slice.
        value : float
            The scalar every element of those slices becomes, in ``self``'s
            dtype.

        Returns
        -------
        Tensor
            A new tensor with ``self``'s shape and dtype; ``self`` is
            unchanged.

        Notes
        -----
        With ``dim = 0``,

        .. math::

            \text{out}[j] = \begin{cases} v & j \in \text{index} \\
            \text{self}[j] & \text{otherwise.} \end{cases}

        The filled slices pass no gradient back to ``self``.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.arange(6, dtype=lucid.float32).reshape(2, 3)
        >>> x.index_fill(1, lucid.tensor([0, 2]), -1.0)
        tensor([[-1., 1., -1.], [-1., 4., -1.]])
        """
        from lucid._ops.composite.indexing import index_fill as _index_fill

        return _index_fill(self, dim, index, value)

    def index_fill_(self, dim: int, index: Tensor, value: float) -> Self:
        r"""Set the slices of ``self`` at ``index`` along ``dim`` to ``value``, in place.

        In-place form of :meth:`index_fill`: the value is written into
        ``self``'s storage, so a view of ``self`` sees it.

        Parameters
        ----------
        dim : int
            Axis along which ``index`` addresses slices.  Negative values
            count from the end.
        index : Tensor
            0-D or 1-D ``int32`` / ``int64`` tensor naming the slices to
            fill; a 0-D one names a single slice.
        value : float
            The scalar every element of those slices becomes, in ``self``'s
            dtype.

        Returns
        -------
        Tensor
            ``self``.

        Raises
        ------
        RuntimeError
            If ``self`` is a leaf that requires grad and autograd is
            recording — wrap the call in :func:`lucid.no_grad`.

        Notes
        -----
        With ``dim = 0``, :math:`\text{self}[j] \leftarrow v` for every
        :math:`j \in \text{index}`.  The filled slices pass no gradient
        back to what produced ``self``.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.ones(2, 3)
        >>> row = x[1]
        >>> _ = x.index_fill_(1, lucid.tensor([1]), 0.0)
        >>> row
        tensor([1., 0., 1.])
        """
        from lucid._ops.composite.indexing import _write_inplace, index_fill

        return _write_inplace(
            self, "index_fill_", lambda base: index_fill(base, dim, index, value)
        )

    def index_put(
        self,
        indices: tuple[Tensor, ...] | list[Tensor],
        values: Tensor,
        accumulate: bool = False,
    ) -> Tensor:
        r"""Return a copy of ``self`` with ``values`` written at ``indices``.

        The out-of-place spelling of ``out[indices] = values``: one
        integer index tensor per leading dimension, broadcast together;
        the dimensions after them are taken whole.

        Parameters
        ----------
        indices : tuple of Tensor or list of Tensor
            ``int32`` / ``int64`` index tensors, one per leading dimension
            of ``self``, or a bool mask; another integer width is refused
            with ``IndexError``.
        values : Tensor
            Values to write, broadcastable to the indexed shape, of
            ``self``'s dtype: another dtype is refused
            (``lucid._C.engine.DtypeMismatch``), not cast.
        accumulate : bool, optional
            ``True`` adds at each position, so repeated positions sum;
            ``False`` (default) overwrites.

        Returns
        -------
        Tensor
            A new tensor with ``self``'s shape and dtype; ``self`` is
            unchanged.

        Notes
        -----
        For two index tensors :math:`r, c`,

        .. math::

            \text{out}[r_k, c_k] = v_k
            \quad\text{or, accumulating,}\quad
            \text{out}[r_k, c_k] \mathrel{+}= v_k.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(2, 3)
        >>> rows, cols = lucid.tensor([0, 1]), lucid.tensor([2, 0])
        >>> x.index_put((rows, cols), lucid.tensor([5.0, 7.0]))
        tensor([[0., 0., 5.], [7., 0., 0.]])
        """
        from lucid._ops.composite.indexing import index_put as _index_put

        return _index_put(self, indices, values, accumulate=accumulate)

    def index_put_(
        self,
        indices: tuple[Tensor, ...] | list[Tensor],
        values: Tensor,
        accumulate: bool = False,
    ) -> Self:
        r"""Write ``values`` into ``self`` at ``indices``, in place.

        In-place form of :meth:`index_put`, and the spelling of
        ``self[indices] = values`` (or ``+=`` with ``accumulate=True``)
        that returns ``self``.

        Parameters
        ----------
        indices : tuple of Tensor or list of Tensor
            ``int32`` / ``int64`` index tensors, one per leading dimension
            of ``self``, or a bool mask; another integer width is refused
            with ``IndexError``.
        values : Tensor
            Values to write, broadcastable to the indexed shape, of
            ``self``'s dtype: another dtype is refused
            (``lucid._C.engine.DtypeMismatch``), not cast.
        accumulate : bool, optional
            ``True`` adds at each position, so repeated positions sum;
            ``False`` (default) overwrites.

        Returns
        -------
        Tensor
            ``self``.

        Raises
        ------
        RuntimeError
            If ``self`` is a leaf that requires grad and autograd is
            recording — wrap the call in :func:`lucid.no_grad`.

        Notes
        -----
        For one index tensor :math:`r`,
        :math:`\text{self}[r_k] \mathrel{+}= v_k` when accumulating, so a
        position named twice receives both values.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(4)
        >>> rows = lucid.tensor([1, 1, 3])
        >>> _ = x.index_put_((rows,), lucid.tensor([1.0, 2.0, 3.0]), accumulate=True)
        >>> x
        tensor([0., 3., 0., 3.])
        """
        from lucid._ops.composite.indexing import index_put_

        return index_put_(self, indices, values, accumulate)

    def expand_as(self, other: Tensor) -> Tensor:
        r"""Broadcast ``self`` to match ``other.shape``.

        Convenience wrapper around :meth:`expand` that takes the target
        shape from another tensor.  On the CPU the result is a view of
        ``self``'s buffer: each stretched axis has stride 0, so every repeated
        element is one of ``self``'s and a write through the result is
        refused.  On metal it is a copy.

        Parameters
        ----------
        other : Tensor
            Tensor whose shape will be adopted. Only ``other.shape`` is
            consulted.

        Returns
        -------
        Tensor
            ``self`` under ``other.shape`` — a read-only view on the CPU, a
            copy on metal.

        Notes
        -----
        Broadcasting rules follow the standard right-aligned semantics:
        each dimension of ``self.shape`` must either equal the
        corresponding entry of ``other.shape`` or be ``1``. Size-1 axes
        are stretched by repeating their single entry. Formally, for each
        axis

        .. math::

            s'_i = \begin{cases}
                s_i & \text{if } s_i = t_i \\
                t_i & \text{if } s_i = 1
            \end{cases}

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0, 3.0]).reshape(1, 3)
        >>> proto = lucid.zeros(4, 3)
        >>> x.expand_as(proto).shape
        (4, 3)
        """
        return Tensor.__new_from_impl__(
            _C_engine.expand(self._impl, list(other._impl.shape))
        )

    def view_as(self, other: Tensor) -> Tensor:
        r"""Return a view of ``self`` with the same shape as ``other``.

        Equivalent to ``self.view(other.shape)``: never copies storage and
        raises ``RuntimeError`` when ``self``'s size and stride are not
        compatible with ``other.shape``.

        Parameters
        ----------
        other : Tensor
            Tensor whose ``shape`` is adopted.

        Returns
        -------
        Tensor
            A view of ``self`` with shape ``other.shape``.

        Raises
        ------
        RuntimeError
            If ``self.numel() != other.numel()`` or ``self``'s strides are
            not compatible with ``other.shape``.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.arange(6)
        >>> proto = lucid.zeros(2, 3)
        >>> x.view_as(proto).shape
        (2, 3)
        """
        return Tensor.__new_from_impl__(
            _C_engine.view(self._impl, list(other._impl.shape))
        )

    def type_as(self, other: Tensor) -> Tensor:
        r"""Cast ``self`` to the dtype and device of ``other``.

        Equivalent to ``self.type(other.type())``: converts both precision
        and device placement to match ``other``.

        Parameters
        ----------
        other : Tensor
            Tensor whose ``dtype`` and ``device`` will be adopted.

        Returns
        -------
        Tensor
            ``self`` on ``other.device`` with ``other.dtype``. If both
            already match, returns ``self`` unchanged.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.zeros(3, dtype=lucid.int32)
        >>> y = lucid.zeros(1, dtype=lucid.float64)
        >>> x.type_as(y).dtype
        lucid.float64

        Notes
        -----
        Values are cast element-wise:
        :math:`x_i \leftarrow \text{cast}_{\tau}(x_i)` where
        :math:`\tau = \text{other.dtype}`. Casts between floating-point
        types preserve gradients; integer→float→integer round-trips lose
        information in the integer truncation step.
        """
        return self.to(*_type_target(other.type()))

    def lerp(self, end: Tensor, weight: float | Tensor) -> Tensor:
        r"""Linearly interpolate between ``self`` and ``end``.

        Computes

        .. math::

            \text{out} = \text{self} + \text{weight} \times (\text{end} - \text{self})

        element-wise. ``weight`` may be a Python scalar or a tensor
        broadcastable to ``end - self``; with ``weight = 0`` the result
        equals ``self``, with ``weight = 1`` it equals ``end``.

        Parameters
        ----------
        end : Tensor
            Target tensor. Must broadcast against ``self``.
        weight : float or Tensor
            Interpolation coefficient. Float values are broadcast to a
            constant tensor matching the difference's shape; tensor
            values are used as-is.

        Returns
        -------
        Tensor
            Interpolated tensor with the broadcast shape of
            ``self``, ``end``, and ``weight``.

        Raises
        ------
        NotImplementedError
            If ``self`` or ``end`` is not floating point.  The weight is a
            fraction, and building it in an integer dtype truncated it to
            zero, so an integer ``lerp`` used to return ``self`` unchanged.

        Notes
        -----
        Linear interpolation is affine: ``lerp(a, b, t)`` lies on the
        line segment from ``a`` to ``b``. The formula is numerically
        preferable to ``(1 - weight) * self + weight * end`` because it
        avoids loss of precision at small ``weight`` values.

        Examples
        --------
        >>> import lucid
        >>> a = lucid.tensor([0.0, 0.0, 0.0])
        >>> b = lucid.tensor([10.0, 20.0, 30.0])
        >>> a.lerp(b, 0.5).tolist()
        [5.0, 10.0, 15.0]
        """
        if not (self.is_floating_point() and end.is_floating_point()):
            raise _C_engine.NotImplementedError(
                f"lerp expects floating-point tensors, got {self.dtype} and "
                f"{end.dtype} — cast them first"
            )
        diff = Tensor.__new_from_impl__(_C_engine.sub(end._impl, self._impl))
        if isinstance(weight, Tensor):
            scaled = Tensor.__new_from_impl__(_C_engine.mul(weight._impl, diff._impl))
        else:
            w_impl = _C_engine.full(
                list(diff._impl.shape),
                float(weight),
                diff._impl.dtype,
                diff._impl.device,
            )
            scaled = Tensor.__new_from_impl__(_C_engine.mul(w_impl, diff._impl))
        return Tensor.__new_from_impl__(_C_engine.add(self._impl, scaled._impl))

    def where(self, condition: Tensor, other: Tensor | float) -> Tensor:
        r"""Element-wise piecewise selection between ``self`` and ``other``.

        For each position, returns the value from ``self`` if the
        corresponding entry in ``condition`` is truthy, otherwise the
        value from ``other``:

        .. math::

            \text{out}_i = \begin{cases}
                \text{self}_i  & \text{if } \text{condition}_i \\
                \text{other}_i & \text{otherwise}
            \end{cases}

        Parameters
        ----------
        condition : Tensor
            Boolean (or truthy-valued) mask, broadcastable to ``self``.
        other : Tensor or float
            Fall-back values when ``condition`` is falsy.  A scalar is
            broadcast, and is weak as in ``self + other``: it can widen the
            kind (a float beside an integer tensor gives float), never the
            width.

        Returns
        -------
        Tensor
            Tensor with the broadcast shape of the three inputs, at the
            common dtype of ``self`` and ``other``.

        Notes
        -----
        Differentiable in both branches: gradients flow into ``self``
        where ``condition`` is true and into ``other`` where it is false.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0, 3.0, 4.0])
        >>> mask = lucid.tensor([True, False, True, False])
        >>> x.where(mask, 0.0).tolist()
        [1.0, 0.0, 3.0, 0.0]
        """
        # The same path as ``lucid.where(condition, self, other)``.  Building
        # ``other`` at ``self``'s dtype truncated a float scalar beside an
        # integer tensor (``ints.where(m, 1.5)`` filled in 1), and a tensor
        # ``other`` of another dtype raised ``DtypeMismatch``.
        from lucid._ops._adapters import _where_adapter

        return Tensor.__new_from_impl__(_where_adapter(condition, self, other))

    def diff(
        self,
        n: int = 1,
        dim: int = -1,
        prepend: Tensor | None = None,
        append: Tensor | None = None,
    ) -> Tensor:
        r"""Compute the ``n``-th order discrete difference along ``dim``.

        The first-order difference operator :math:`\Delta` is defined as

        .. math::

            (\Delta x)_i = x_{i+1} - x_i,

        which reduces the length of the chosen axis by 1. Higher orders
        are obtained by composition: :math:`\Delta^n = \Delta \circ \Delta^{n-1}`,
        equivalent to

        .. math::

            (\Delta^n x)_i = \sum_{k=0}^{n} (-1)^k \binom{n}{k} x_{i+n-k}.

        Parameters
        ----------
        n : int, optional
            Order of the difference operator. Default ``1``. Must be
            non-negative; ``n = 0`` returns ``self`` unchanged.
        dim : int, optional
            Axis along which to compute the difference. Negative values
            count from the end. Default ``-1`` (last axis).
        prepend : Tensor, optional
            Values to prepend to ``self`` along ``dim`` before computing the
            difference.
        append : Tensor, optional
            Values to append to ``self`` along ``dim`` before computing the
            difference.

        Returns
        -------
        Tensor
            Tensor whose size along ``dim`` is ``self.shape[dim] - n`` (plus any
            prepended/appended elements) and whose other axes match ``self``.

        Notes
        -----
        Useful as a discrete analogue of differentiation. For sample
        spacing :math:`h`, finite-difference derivatives are obtained by
        dividing by :math:`h^n`.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.tensor([1.0, 2.0, 4.0, 7.0, 11.0])
        >>> x.diff().tolist()
        [1.0, 2.0, 3.0, 4.0]
        >>> x.diff(n=2).tolist()
        [1.0, 1.0, 1.0]
        """
        from lucid._ops.composite.reductions import _diff

        return _diff(self, n=n, dim=dim, prepend=prepend, append=append)

    def addmm(
        self, mat1: Tensor, mat2: Tensor, *, beta: float = 1.0, alpha: float = 1.0
    ) -> Tensor:
        r"""Fused matrix multiply-add.

        Computes

        .. math::

            \text{out} = \beta \cdot \text{self} + \alpha \cdot (\text{mat1} \cdot \text{mat2})

        where :math:`\cdot` is matrix multiplication. ``mat1`` must have
        shape :math:`(n, k)` and ``mat2`` shape :math:`(k, p)`; the
        product has shape :math:`(n, p)`. ``self`` must broadcast to
        :math:`(n, p)`.

        Parameters
        ----------
        mat1 : Tensor
            Left matrix of shape :math:`(n, k)`.
        mat2 : Tensor
            Right matrix of shape :math:`(k, p)`.
        beta : float, optional
            Scaling factor applied to ``self``. Default ``1.0``.
            ``beta = 0`` zeroes the additive term.
        alpha : float, optional
            Scaling factor applied to the matrix product. Default ``1.0``.

        Returns
        -------
        Tensor
            Resulting tensor of shape :math:`(n, p)`.

        Notes
        -----
        This is the same operation as the BLAS ``GEMM`` routine
        :math:`C \leftarrow \beta C + \alpha A B`. In Lucid it is
        currently decomposed as ``add(beta * self, alpha * (mat1 @ mat2))``;
        a future fused-kernel path may exploit the Accelerate / MLX
        GEMM-with-bias primitives directly.

        Examples
        --------
        >>> import lucid
        >>> bias = lucid.zeros(2, 2)
        >>> a = lucid.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> b = lucid.tensor([[1.0, 0.0], [0.0, 1.0]])
        >>> bias.addmm(a, b, alpha=2.0).tolist()
        [[2.0, 4.0], [6.0, 8.0]]
        """
        from lucid._ops.composite.blas import addmm as _addmm

        return _addmm(self, mat1, mat2, beta=beta, alpha=alpha)

    # ── zero_() helper ───────────────────────────────────────────────────────

    def zero_(self) -> Self:
        r"""Fill the tensor with zeros in-place.

        Writes ``0`` into every element of ``self``'s storage.  It is
        :meth:`fill_` with ``0`` — a multiply by zero, which this used to
        be, turns ``inf`` and ``NaN`` into ``NaN`` rather than ``0``.

        Returns
        -------
        Tensor
            ``self``, after zeroing.

        Notes
        -----
        As with :meth:`fill_`, in-place mutation bypasses autograd's
        view tracking. Calling ``zero_`` on a leaf tensor with
        ``requires_grad=True`` may raise a runtime error from the
        autograd engine.

        Element-wise: :math:`x_i \leftarrow 0` for every position. The
        result is the additive identity of the tensor algebra at the
        same shape and dtype.

        Examples
        --------
        >>> import lucid
        >>> x = lucid.ones(3)
        >>> _ = x.zero_()
        >>> x.tolist()
        [0.0, 0.0, 0.0]
        """
        return self.fill_(0)

    # ── pickling support (required for multiprocessing DataLoader) ────────────

    def __reduce__(self) -> tuple:
        r"""Pickle hook for cross-process Tensor serialisation.

        Implements the Python pickle protocol (PEP 307). The returned
        ``(callable, args)`` pair is invoked by ``pickle.loads`` /
        ``copy.deepcopy`` to reconstruct the tensor; the unpickler
        receives raw element bytes plus enough metadata to rebuild the
        ``TensorImpl`` without going through NumPy.

        Returns
        -------
        tuple
            ``(_tensor_unpickle, (raw_bytes, shape, dtype_name, device,
            requires_grad))``. The first element is a module-level
            callable so that ``pickle`` can locate it by qualified name;
            the second element is the positional argument tuple.

        Notes
        -----
        The wire format mirrors the ``lucid.serialization`` v3 contract,
        so DataLoader workers spawned via ``multiprocessing`` (which
        defaults to the ``spawn`` start method on macOS) can transfer
        tensors without importing NumPy. The byte buffer is produced by
        ``TensorImpl::to_bytes`` and consumed by ``TensorImpl::from_bytes``.

        The pickle protocol requires the callable to be importable by
        fully qualified name — that is why :func:`_tensor_unpickle` is a
        module-level function rather than a static method. Logically
        :math:`\text{loads}(\text{dumps}(t)) = t` on values; autograd
        history is **not** preserved across the boundary.

        Examples
        --------
        >>> import pickle, lucid
        >>> x = lucid.tensor([1.0, 2.0, 3.0])
        >>> y = pickle.loads(pickle.dumps(x))
        >>> y.tolist()
        [1.0, 2.0, 3.0]
        """
        # Wire format mirrors the lucid.serialization v3 contract — raw
        # bytes + (shape, dtype name, device) — so multiprocessing pickle
        # is numpy-free.
        return (
            _tensor_unpickle,
            (
                self._impl.to_bytes(),
                list(self._impl.shape),
                self.dtype._name,
                self._impl.device,
                self._impl.requires_grad,
            ),
        )


def _tensor_unpickle(
    raw_bytes: bytes,
    shape: list[int],
    dtype_name: str,
    device: object,
    requires_grad: bool,
) -> Tensor:
    """Top-level helper so multiprocessing (spawn) can pickle/unpickle Tensor."""
    from lucid._dtype import _resolve_dtype_name, to_engine_dtype

    eng_dtype = to_engine_dtype(_resolve_dtype_name(dtype_name))
    impl = _C_engine.TensorImpl.from_bytes(
        raw_bytes, list(shape), eng_dtype, device, requires_grad
    )
    return Tensor.__new_from_impl__(impl)


# ── inject dunders and methods after class definition ────────────────────────
from lucid._tensor._repr import tensor_repr as _tensor_repr
from lucid._tensor._dunders import _inject_dunders
from lucid._tensor._methods import _clamp_inplace, _inject_methods
from lucid._tensor._to import _inject_to, _legacy_type_name, _type_target

# ``register_hook`` is annotated ``-> _RemovableHandle``, imported under
# TYPE_CHECKING to keep the module's import graph acyclic.  PEP 649 then
# has nothing to resolve it against, so ``inspect.signature`` on that one
# method raised NameError.  Bound here rather than at the top: by this
# point the class exists and the import cannot cycle back through it.
from lucid.autograd._hooks import (
    RemovableHandle as _RemovableHandle,
)

_inject_dunders(Tensor)
_inject_methods(Tensor)
_inject_to(Tensor)
