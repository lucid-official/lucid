"""The one place a ``lucid.fft`` call's arguments are read and checked.

Every public function of :mod:`lucid.fft` — the 1-D, 2-D and n-D forms of
each transform, and the frequency / shift helpers — turns its arguments into
a :class:`_Spec` here before any kernel runs.  Nothing below this module
re-checks an argument, so the rules the reference states once live once:

* ``n`` / ``s`` / ``dim`` are integers.  ``2.5``, ``"4"`` and ``True`` are a
  ``TypeError`` naming the public op, not a silent ``int()`` truncation.
* ``-1`` in ``s`` keeps the input's size along that axis.  For the
  half-spectrum readers (``irfftn`` / ``hfftn``) the last axis keeps the
  default output length ``2 * (m - 1)``.  ``n`` of the 1-D forms has no such
  spelling, as in the reference.
* An empty ``dim`` transforms no axis: ``fftn`` / ``ifftn`` return the input
  as a complex tensor, and a 0-d input is that case.  The real-valued
  families refuse it — a half-spectrum is of an axis.
* Half-precision input is promoted to ``float32`` and integer / boolean
  input to the default floating dtype, whichever entry point is called.
"""

import numbers
import operator
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import SupportsIndex

import lucid
from lucid._C import engine as _C_engine
from lucid._types import DTypeLike
from lucid._ops._dims import _to_axes, _wrap_dim, _wrap_dims
from lucid._tensor.tensor import Tensor

# ── Normalisation ────────────────────────────────────────────────────────────

_VALID_NORMS = ("backward", "forward", "ortho")


def _check_norm(norm: str | None) -> str:
    if norm is None:
        return "backward"
    if norm not in _VALID_NORMS:
        raise ValueError(f"norm must be one of {_VALID_NORMS} or None, got {norm!r}")
    return norm


def _dual_norm(norm: str) -> str:
    if norm == "ortho":
        return "ortho"
    return "forward" if norm == "backward" else "backward"


# ── Families ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _Family:
    """What one family of transforms asks of its arguments.

    ``real_input`` — the transform reads a real signal (``rfftn`` /
    ``ihfftn``), so a complex input is refused.  ``half_output`` — it reads
    a half-spectrum and writes ``2 * (m - 1)`` samples by default
    (``irfftn`` / ``hfftn``).  ``empty_is_identity`` — an empty axis list
    is the identity rather than an error (``fftn`` / ``ifftn``).
    """

    real_input: bool = False
    half_output: bool = False
    empty_is_identity: bool = False


_COMPLEX = _Family(empty_is_identity=True)
_REAL_IN = _Family(real_input=True)
_REAL_OUT = _Family(half_output=True)

# ── Argument readers ─────────────────────────────────────────────────────────

_HALF_DTYPES = (lucid.float16, lucid.bfloat16)


def _index(value: object, op: str, what: str) -> int:
    """``value`` as an int — an index, not a bool, float or string."""
    if isinstance(value, bool) or not isinstance(value, SupportsIndex):
        raise TypeError(f"{op}(): {what} must be an int, got {type(value).__name__}")
    return operator.index(value)


def _sizes(s: object, op: str) -> list[int]:
    """``s`` — ``None``, an int or a sequence of ints — as a list (empty when omitted)."""
    if s is None:
        return []
    if isinstance(s, Sequence) and not isinstance(s, str):
        return [_index(v, op, "s entries") for v in s]
    return [_index(s, op, "s")]


def _check_points(size: int, op: str) -> int:
    if size < 1:
        raise _C_engine.InvalidArgument(f"{op}(): invalid number of data points ({size}) specified")
    return size


def _canonical_input(op: str, x: object, family: _Family) -> Tensor:
    """``x`` as a tensor the kernels take: floating (≥ float32) or complex."""
    if not isinstance(x, Tensor):
        raise TypeError(f"{op}(): input must be a Tensor, got {type(x).__name__}")
    if x.is_complex():
        if family.real_input:
            raise TypeError(f"{op}(): expects a real-valued input, got {x.dtype}")
        return x
    if not x.is_floating_point():
        x = x.to(lucid.get_default_dtype())
    if x.dtype in _HALF_DTYPES:
        x = x.to(lucid.float32)
    return x


@dataclass(frozen=True, slots=True)
class _Spec:
    """A checked call: what to transform, along which axes, at which sizes.

    ``s`` is empty when the call named no sizes — each axis then keeps its
    own — and otherwise holds one positive size per axis.
    """

    op: str
    family: _Family
    input: Tensor
    axes: list[int]
    s: list[int]
    norm: str

    @property
    def is_identity(self) -> bool:
        """No axis to transform (only ``fftn`` / ``ifftn`` get this far)."""
        return not self.axes

    def hermitian(self) -> _Spec:
        """The same call as the conjugate input under the dual norm.

        ``hfftn`` is ``irfftn`` of the conjugate with the dual norm.
        """
        return replace(self, input=lucid.conj(self.input), norm=_dual_norm(self.norm))

    def with_dual_norm(self) -> _Spec:
        """The same call under the dual norm (``ihfftn`` is a conjugated ``rfftn``)."""
        return replace(self, norm=_dual_norm(self.norm))


def _refuse_axes_of_0d(op: str, rank: int, axes: list[int]) -> None:
    """A 0-d input has no axis for a transform to name (unlike ``wrap_dim``'s notional one)."""
    if rank == 0 and axes:
        raise IndexError(
            f"{op}(): axis out of range: got {axes[0]}, but a 0-d input has no axis to transform"
        )


def _wrap_axes(op: str, rank: int, axes: list[int]) -> list[int]:
    _refuse_axes_of_0d(op, rank, axes)
    return _wrap_dims(axes, rank, op)


def _spec(
    op: str,
    family: _Family,
    input: object,
    axes: list[int] | None,
    sizes: list[int],
    norm: str | None,
    *,
    keep_minus_one: bool,
) -> _Spec:
    """Resolve ``axes`` and ``sizes`` against ``input`` and check them together."""
    norm_v = _check_norm(norm)
    x = _canonical_input(op, input, family)
    rank = x.ndim
    if axes is None:
        if len(sizes) > rank:
            raise ValueError(f"{op}(): len(s)={len(sizes)} exceeds the input's {rank} dimensions")
        resolved = list(range(rank - len(sizes), rank)) if sizes else list(range(rank))
    else:
        resolved = _wrap_axes(op, rank, axes)
    if sizes and len(sizes) != len(resolved):
        raise ValueError(f"{op}(): len(s)={len(sizes)} must match len(dim)={len(resolved)}")
    if not resolved and not family.empty_is_identity:
        raise _C_engine.InvalidArgument(f"{op}(): must transform at least one axis")
    final: list[int] = []
    for i, size in enumerate(sizes):  # the lengths agree, checked above
        if size == -1 and keep_minus_one:
            last = i == len(resolved) - 1
            in_size = int(x.shape[resolved[i]])
            final.append(2 * (in_size - 1) if family.half_output and last else in_size)
        else:
            final.append(_check_points(size, op))
    return _Spec(op, family, x, resolved, final, norm_v)


def _spec_nd(
    op: str,
    family: _Family,
    input: object,
    s: object,
    dim: object,
    norm: str | None,
) -> _Spec:
    """The spec of an n-D or 2-D call (``dim=None`` names every axis)."""
    axes = None if dim is None else _to_axes(dim, op)
    return _spec(op, family, input, axes, _sizes(s, op), norm, keep_minus_one=True)


def _spec_1d(
    op: str,
    family: _Family,
    input: object,
    n: object,
    dim: object,
    norm: str | None,
) -> _Spec:
    """The spec of a 1-D call: ``n`` and ``dim`` are single ints."""
    sizes = [] if n is None else [_index(n, op, "n")]
    axes = [_index(dim, op, "dim")]
    return _spec(op, family, input, axes, sizes, norm, keep_minus_one=False)


# ── Helpers' arguments ───────────────────────────────────────────────────────


def _shift_axes(op: str, input: object, dim: object) -> tuple[Tensor, list[int]]:
    """The tensor and the wrapped axes ``fftshift`` / ``ifftshift`` roll."""
    if not isinstance(input, Tensor):
        raise TypeError(f"{op}(): input must be a Tensor, got {type(input).__name__}")
    rank = input.ndim
    axes = list(range(rank)) if dim is None else _to_axes(dim, op)
    _refuse_axes_of_0d(op, rank, axes)
    return input, [_wrap_dim(a, rank, op) for a in axes]


def _freq_args(op: str, n: object, d: object, dtype: DTypeLike) -> tuple[int, float]:
    """``n``, ``d`` and ``dtype`` of ``fftfreq`` / ``rfftfreq``, checked."""
    n_v = _index(n, op, "n")
    if n_v < 0:
        raise ValueError(f"{op} requires n >= 0, got {n_v}")
    if isinstance(d, bool) or not isinstance(d, numbers.Real):
        raise TypeError(f"{op}(): d must be a number, got {type(d).__name__}")
    resolved = lucid.zeros(0, dtype=dtype)
    if not resolved.is_floating_point():
        raise TypeError(f"{op}(): dtype must be a floating-point dtype, got {resolved.dtype}")
    return n_v, float(d)
