"""
lucid._types: canonical type aliases, protocols, and TypeVars for the lucid API.

These serve three purposes:
  1. **Readability** — ``def to(self, device: DeviceLike)`` is clearer than
     a raw union.
  2. **DRY** — each underlying union is written once and referenced everywhere.
  3. **Extensibility** — :class:`TensorLikeProtocol` lets user-defined types
     participate in lucid's functional interface without subclassing ``Tensor``.

Import policy
-------------
Every module — including those imported during ``lucid._tensor.tensor``
construction — can import from here.  The ``Tensor`` reference itself is
held under ``TYPE_CHECKING`` and the Tensor-dependent aliases are PEP 695
``type`` statements, which lazily evaluate their value, so importing this
module does not trigger an early ``Tensor`` import.

Naming conventions
------------------
- No leading underscore → public API (stable).
- Leading underscore   → private / implementation detail (may change).
"""

from types import EllipsisType
from typing import Callable, Literal, Protocol, TYPE_CHECKING, runtime_checkable

from lucid._dtype import dtype as _DType
from lucid._device import device as _Device

if TYPE_CHECKING:
    import numpy as np  # typing-only — no runtime import.
    from lucid._tensor.tensor import Tensor
    from lucid._C import engine as _C_engine  # for DeviceLike / DTypeLike widening


# ── TypeVars ──────────────────────────────────────────────────────────────────
# (none — generics use inline PEP 695 type parameters; no module-level TypeVars.)


# ── Public type aliases (no Tensor dependency) ───────────────────────────────

# Scalar numeric types — valid operands alongside Tensor in arithmetic.
type Scalar = int | float | bool

# Device specifier: Python device object, C++ Device enum, string ('cpu'/'metal'), or None.
# The C++ engine type is accepted by _parse_device at runtime; widening here avoids
# spurious mypy errors when internal code passes tensor._impl.device directly.
type DeviceLike = _Device | _C_engine.Device | str | None

# DType specifier: Python dtype object, dtype class, C++ Dtype enum, or None.
type DTypeLike = _DType | type[_DType] | _C_engine.Dtype | None

# Shape / size specifier used in factory functions and reshape.
type ShapeLike = int | tuple[int, ...]

# Spatial size variants for 1-D / 2-D / 3-D ops (Conv, Pool, …).
type _Size1d = int | tuple[int]
type _Size2d = int | tuple[int, int]
type _Size3d = int | tuple[int, int, int]

# String-enum modes — call-site Literal checking. Each set is the *complete*
# authoritative value list its op accepts (see the validating helper beside the
# consumer). These are domain-specific: a same-named param in another context
# (einops ``reduce`` reduction, ``grid_sample`` padding_mode / mode, image
# interpolation) has a different value set and intentionally stays ``str``.
type Reduction = Literal["none", "mean", "sum"]
type ReductionKL = Literal["none", "mean", "sum", "batchmean"]
type PaddingMode = Literal["zeros", "reflect", "replicate", "circular"]
type GeluApproximate = Literal["none", "tanh"]


# ── Protocols ─────────────────────────────────────────────────────────────────


@runtime_checkable
class HasShape(Protocol):
    """Structural protocol — any object that exposes ``.shape`` / ``.ndim``.

    Used as a duck-typing bound where Lucid only needs shape introspection
    (e.g. broadcasting helpers, validation code) and doesn't care
    whether the object is a Lucid :class:`Tensor`, a ``numpy.ndarray``,
    an :class:`mlx.core.array`, or a user-defined array proxy.
    ``@runtime_checkable`` means ``isinstance(x, HasShape)`` works at
    runtime in addition to static type checkers honouring it.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Tuple of dimension sizes."""
        ...

    @property
    def ndim(self) -> int:
        """Number of dimensions (length of :attr:`shape`)."""
        ...


@runtime_checkable
class SupportsNumpyConversion(Protocol):
    """Structural protocol — exposes ``.numpy()`` returning ``ndarray``.

    The narrow type Lucid uses at the ``Tensor → ndarray`` boundary
    (``_factories/converters.py``, ``_tensor/_repr.py``).  Any object
    that fulfils it can be fed through Lucid's numpy bridge regardless
    of whether it's a Lucid :class:`Tensor`, an :class:`mlx.core.array`,
    or a user proxy — ``@runtime_checkable`` makes ``isinstance``
    dispatch correctly.
    """

    def numpy(self) -> np.ndarray:
        """Return a NumPy view / copy of the underlying buffer."""
        ...


@runtime_checkable
class SupportsGrad(Protocol):
    """Structural protocol — autograd participants.

    Lucid's autograd machinery is structural, not nominal: anything
    that exposes ``requires_grad`` / ``backward`` / ``detach`` qualifies
    for the autograd graph.  This protocol formalises that contract
    so users can build custom Tensor-like wrappers without subclassing
    :class:`Tensor` directly.
    """

    @property
    def requires_grad(self) -> bool:
        """Whether autograd tracks operations on this object."""
        ...

    def backward(self, *, retain_graph: bool = ...) -> None:
        """Trigger reverse-mode differentiation through the graph rooted here."""
        ...

    def detach(self) -> SupportsGrad:
        """Return a view of this object detached from the autograd graph."""
        ...


@runtime_checkable
class TensorLikeProtocol(HasShape, SupportsNumpyConversion, Protocol):
    """Structural protocol for objects that behave like :class:`Tensor`.

    Any object satisfying this protocol can be used where a Tensor is expected
    in lucid's functional interfaces — enabling custom array types without
    subclassing.

    Example
    -------
    >>> class MyArray:
    ...     @property
    ...     def shape(self) -> tuple[int, ...]: return (3,)
    ...     @property
    ...     def ndim(self) -> int: return 1
    ...     @property
    ...     def dtype(self) -> lucid.dtype: return lucid.float32
    ...     @property
    ...     def device(self) -> lucid.device: return lucid.device("cpu")
    ...     def numpy(self): return np.zeros(3)
    ...     def to(self, *a, **kw): return self
    >>> isinstance(MyArray(), TensorLikeProtocol)
    True
    """

    @property
    def dtype(self) -> _DType:
        """Element :class:`~lucid.dtype` of the underlying buffer."""
        ...

    @property
    def device(self) -> _Device:
        """:class:`~lucid.device` on which this object resides."""
        ...

    def to(self, *args: object, **kwargs: object) -> TensorLikeProtocol:
        """Return a (possibly converted) version of this object on a new device/dtype."""
        ...


# ── Public type aliases (Tensor-dependent — PEP 695 lazy values) ─────────────

# Any value that lucid can meaningfully convert to a Tensor.
type TensorLike = Tensor | np.ndarray | list[object] | int | float | bool

# Right-hand operand for arithmetic dunders (+, -, *, /, **, …).
type TensorOrScalar = Tensor | Scalar

# All legal index forms for Tensor.__getitem__ / __setitem__: an int; a bool
# (a new axis of length 1 or 0); a slice; ``...``; ``None``; an integer or bool
# tensor; a list or NumPy array (one index tensor); or a tuple of them.  Read
# by ``lucid._tensor._indexing._normalize_key``, the one place a key is
# interpreted.
type _IndexType = (
    int
    | slice
    | EllipsisType
    | Tensor
    | np.ndarray
    | list[object]
    | tuple[int | slice | EllipsisType | Tensor | np.ndarray | list[object] | None, ...]
    | None
)

# Reduction / op axis specifier
type DimLike = int | list[int] | tuple[int, ...] | None


# ── Neural network ────────────────────────────────────────────────────────────

# Return type of Module.forward: either a single Tensor or a tuple of Tensors.
type _ModuleOutput = Tensor | tuple[Tensor, ...]


class _HasForward[**P, R](Protocol):
    """Anything whose ``forward`` takes ``P`` and returns ``R``.

    ``Module.__call__`` binds its ``self`` to this, so the checker reads the
    parameters and the return type of a call off the concrete subclass's own
    ``forward`` instead of the base class's catch-all signature.
    """

    def forward(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """The computation a call runs."""
        ...


# State dict: flat mapping from dotted parameter / buffer name to its tensor.
type StateDict = dict[str, Tensor]

# Hook signatures for Module.register_forward_pre_hook / register_forward_hook.
# Module is typed as object here to avoid a circular import (module.py → _types.py → module.py).
type _ForwardPreHook = Callable[[object, tuple[Tensor, ...]], tuple[Tensor, ...] | None]
type _ForwardHook = Callable[[object, tuple[Tensor, ...], _ModuleOutput], _ModuleOutput | None]
type _BackwardHook = Callable[..., None]


# ── Optimizers ────────────────────────────────────────────────────────────────

# A single parameter group dict holding hyperparams and a list of Parameters.
type _ParamGroup = dict[str, object]

# Optional re-evaluation closure passed to Optimizer.step().
type _OptimizerClosure = Callable[[], Tensor] | None
