"""The Euclidean ``cdist`` (and so ``pdist``): its two forms and the scale both take.

``cdist(p=2)`` has two evaluations, and the reference picks between them the
way :func:`_use_matmul` does:

* **the difference form** — materialise ``a_i - b_j`` and take the norm of
  each.  It cancels nothing (a point's distance to itself is exactly 0), but
  holds a ``(..., P, R, M)`` tensor;
* **the matmul form** — ``|a|^2 + |b|^2 - 2 a.b``, which holds only the
  ``(..., P, R)`` result, at the cost of cancelling when two points are close.

Both square the coordinates, and a float32 square leaves the range long before
the distance does: ``1e25`` squares to ``1e50``, and ``1e-20`` to ``1e-40``,
which the CPU keeps as a subnormal with a few digits lost and Metal flushes to
zero.  The distance is homogeneous (``|c x| = |c| |x|``), so each form divides
its operands by a magnitude first and multiplies the root by it afterwards:

* the difference form by each pair's own largest ``|a_ik - b_jk|``;
* the matmul form by the largest magnitude in either operand of a batch entry
  (a divisor per pair would break the matmul).  A pair whose points are tiny
  next to the batch's largest entry loses them, as the form's cancellation
  already does.

The divisor is a constant of the call — no gradient flows through it — and is 1
wherever the magnitude is inside ``2^±E`` (:data:`_RANGE_EXPONENT`), or is
``nan`` or infinite.  Inside the window the division and the product are by 1
and the result is bit for bit the unscaled form's; a ``nan`` or infinite entry
leaves its batch entry (matmul form) or pair (difference form) to the unscaled
evaluation, whose answer there is the same ``nan`` / ``inf`` it always was.
Because it is a constant, every derivative is the unscaled form's: the distance
equals ``s * |x / s|`` for any constant ``s``, so scaling changes the rounding
and nothing else.

An eager CPU tensor reads whether any magnitude is outside the window (free
there: nothing is queued) and, when none is, skips the division and the product
altogether.  Metal, a compile trace and a phantom tensor have no value to read
cheaply, so their graph is branch-free — the divisor is computed on the device
and is simply 1 — and they take the same route as eager.  Half precision is not
scaled: its range ends near ``6.5e4``, far below any window.
"""

from typing import TYPE_CHECKING

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._ops._reduce import _p_norm, _zero_safe_root

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

_Impl = _C_engine.TensorImpl

#: The reference switches ``use_mm_for_euclid_dist_if_necessary`` to the matmul
#: form when either operand has more than this many rows (measured: 25 rows
#: take the difference form, 26 the matmul form).
_MATMUL_ROWS = 25

#: A magnitude inside ``2^±E`` squares, and a few thousand of them sum, inside the
#: dtype's range (float32 ``2^±126``, float64 ``2^±1022``), and Metal flushes the
#: float32 subnormals below ``2^-126``.  Only these two dtypes are scaled.
_RANGE_EXPONENT = {_C_engine.F32: 30, _C_engine.F64: 300}


def _use_matmul(compute_mode: str, rows1: int, rows2: int) -> bool:
    """Whether ``compute_mode`` takes the matmul form for operands of these row counts."""
    if compute_mode == "use_mm_for_euclid_dist":
        return True
    if compute_mode == "donot_use_mm_for_euclid_dist":
        return False
    return rows1 > _MATMUL_ROWS or rows2 > _MATMUL_ROWS


def _euclidean_cdist(x1: Tensor, x2: Tensor, compute_mode: str) -> Tensor:
    """``|x1[..., i, :] - x2[..., j, :]|`` for every row pair, in the form the mode picks.

    Both operands are real floating tensors of one dtype, so that a mode does
    not change which inputs are refused.
    """
    rows1 = int(x1.shape[-2]) if x1.ndim >= 2 else 1
    rows2 = int(x2.shape[-2]) if x2.ndim >= 2 else 1
    if _use_matmul(compute_mode, rows1, rows2):
        return _matmul_form(x1, x2)
    return _wrap(_difference_form(x1, x2))


def _readable(x: _Impl) -> bool:
    """Whether reading a value of ``x`` is free: an eager CPU tensor."""
    return (
        x.device == _C_engine.Device.CPU
        and not x.is_phantom
        and _C_engine.compile.current_tracer() is None
    )


def _divisor(top: _Impl, exponent: int) -> _Impl | None:
    """The divisor for the largest magnitudes ``top``, or ``None`` when all are in range.

    ``top`` where it is outside ``2^±exponent``, else 1.  A ``nan`` or infinite
    one is 1 too (dividing would turn every finite entry into 0 or ``nan``), and a
    zero is a tiny positive number — there is nothing to divide, so any positive
    divisor leaves the zeros and gives the root a factor ``0 * floor = 0``.

    Call under ``no_grad``: the divisor is a constant of the call.
    """
    inside = _C_engine.equal(top, _C_engine.clip(top, 2.0**-exponent, 2.0**exponent))
    # A zero needs no divisor either: a coincident pair is a zero largest entry.
    if _readable(top):
        settled = _C_engine.logical_or(inside, _C_engine.equal(top, _C_engine.zeros_like(top)))
        if _C_engine.all(settled).item():
            return None
    divisor = _C_engine.where(inside, _C_engine.ones_like(top), top)
    divisor = _C_engine.nan_to_num(divisor, 1.0, 1.0, 1.0)
    floor = _C_engine.full([], 2.0 ** -(3 * exponent + 10), top.dtype, top.device)
    return _C_engine.maximum(divisor, floor)


def _largest(x: _Impl, axes: list[int]) -> _Impl:
    """The largest ``|x|`` over ``axes``, each kept as a size-1 axis."""
    return _C_engine.max(_C_engine.abs(x), axes, True)


def _matmul_form(x1: Tensor, x2: Tensor) -> Tensor:
    """The matmul form, over operands divided by the batch entry's largest magnitude."""
    a, b = _unwrap(x1), _unwrap(x2)
    scale = _matmul_scale(a, b)
    if scale is not None:
        x1, x2 = _wrap(_C_engine.div(a, scale)), _wrap(_C_engine.div(b, scale))
    # ||a - b||² = ||a||² + ||b||² − 2 a·bᵀ, which never materialises the
    # (..., P, R, M) difference.
    x1_sq = (x1 * x1).sum(dim=-1, keepdim=True)  # (..., P, 1)
    x2_sq = (x2 * x2).sum(dim=-1, keepdim=True)  # (..., R, 1)
    cross = lucid.matmul(x1, x2.swapaxes(-1, -2))  # (..., P, R)
    sq = x1_sq + x2_sq.swapaxes(-1, -2) - 2.0 * cross
    sq = lucid.clamp(sq, 0.0, float("inf"))  # clamp numerical noise
    # Coincident points — every diagonal entry of cdist(x, x) — sit at the
    # root's infinite slope; the shared zero-safe root gives them the 0
    # subgradient.
    root = _zero_safe_root(_unwrap(sq), 2)
    return _wrap(root if scale is None else _C_engine.mul(root, scale))


def _matmul_scale(a: _Impl, b: _Impl) -> _Impl | None:
    """The ``(..., 1, 1)`` divisor of a batch entry's two operands, or ``None``."""
    exponent = _RANGE_EXPONENT.get(a.dtype)
    nd_a, nd_b = len(a.shape), len(b.shape)
    # A 1-D operand has no row axis to reduce, and an empty one no largest entry.
    if exponent is None or min(nd_a, nd_b) < 2 or a.numel() == 0 or b.numel() == 0:
        return None
    with lucid.no_grad():
        top = _C_engine.maximum(
            _largest(a, [nd_a - 2, nd_a - 1]), _largest(b, [nd_b - 2, nd_b - 1])
        )
        return _divisor(top, exponent)


def _difference_form(x1: Tensor, x2: Tensor) -> _Impl:
    """The difference form: each pair's difference divided by its largest entry."""
    diff = _unwrap(x1.unsqueeze(-2) - x2.unsqueeze(-3))  # (..., P, R, M)
    last = [len(diff.shape) - 1]
    exponent = _RANGE_EXPONENT.get(diff.dtype)
    scale = None
    if exponent is not None and diff.numel() > 0:
        with lucid.no_grad():
            scale = _divisor(_largest(diff, last), exponent)  # (..., P, R, 1)
    if scale is None:
        return _p_norm(diff, 2.0, last, False)
    scaled = _C_engine.div(diff, scale)
    root = _zero_safe_root(_C_engine.sum(_C_engine.mul(scaled, scaled), last, False), 2)
    return _C_engine.mul(root, _C_engine.reshape(scale, list(root.shape)))
