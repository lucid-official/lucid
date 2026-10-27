"""BLAS-level composites (``addmm``, ``addbmm``, ``mv``, ``ger``, ...)."""

from typing import TYPE_CHECKING

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._ops._autocast import _amp_op, _autocast_entry, _autocast_suspended
import lucid.linalg as _la

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _scaled_add(
    input: Tensor, beta: float, prod: Tensor, alpha: float, *, op: str | None = None
) -> Tensor:
    """``beta * input + alpha * prod``, skipping ``input``'s values when ``beta == 0``."""
    if op is not None and input.dtype != prod.dtype:
        raise _C_engine.DtypeMismatch(
            f"DtypeMismatch ({op}): expected {input.dtype._name}, got {prod.dtype._name}"
        )
    if beta == 0:
        acc = (
            lucid.where(lucid.zeros_like(input, dtype=lucid.bool_), input, 0)
            if input.requires_grad
            else lucid.zeros_like(input)
        )
        return acc + prod * alpha
    return input * beta + prod * alpha


@_amp_op("addmm")
def addmm(
    input: Tensor, mat1: Tensor, mat2: Tensor, *, beta: float = 1.0, alpha: float = 1.0
) -> Tensor:
    r"""General matrix multiply with a scaled accumulator (GEMM).

    Computes :math:`\beta \cdot \text{input} + \alpha \cdot (\text{mat1} \times \text{mat2})`,
    the canonical BLAS-3 ``gemm`` operation. Useful for fused linear
    layers: an output bias plus a weight multiplication in one call.

    Parameters
    ----------
    input : Tensor
        Accumulator of shape ``(M, N)`` (or broadcastable to it).
    mat1 : Tensor
        Left matrix of shape ``(M, K)``.
    mat2 : Tensor
        Right matrix of shape ``(K, N)``.
    beta : float, optional
        Scalar multiplier on ``input``. Defaults to ``1.0``.
    alpha : float, optional
        Scalar multiplier on ``mat1 @ mat2``. Defaults to ``1.0``.

    Returns
    -------
    Tensor
        Tensor of shape ``(M, N)``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out} = \beta \cdot \text{input}
        + \alpha \cdot (\text{mat1} \cdot \text{mat2}).

    The inner dimension ``K`` must match between ``mat1`` and ``mat2``.
    Standard dtype promotion applies; the matmul itself uses Accelerate
    (CPU stream) or MLX (GPU stream).

    Examples
    --------
    >>> import lucid
    >>> M = lucid.zeros((2, 2))
    >>> a = lucid.tensor([[1., 2.], [3., 4.]])
    >>> b = lucid.tensor([[5., 6.], [7., 8.]])
    >>> lucid.addmm(M, a, b, beta=0.0, alpha=1.0)
    tensor([[19., 22.], [43., 50.]])
    """
    return _scaled_add(input, beta, lucid.matmul(mat1, mat2), alpha, op="addmm")


@_amp_op("addbmm")
def addbmm(
    input: Tensor, batch1: Tensor, batch2: Tensor, *, beta: float = 1.0, alpha: float = 1.0
) -> Tensor:
    r"""Batched matmul with reduction over the batch axis.

    Computes
    :math:`\beta \cdot \text{input} + \alpha \cdot \sum_{k} (\text{batch1}[k] \cdot \text{batch2}[k])`.
    Useful for accumulating multiple parallel matrix products into a
    single output matrix (e.g. summing per-head attention contributions).

    Parameters
    ----------
    input : Tensor
        Accumulator of shape ``(M, N)``.
    batch1 : Tensor
        Batched left matrices of shape ``(B, M, K)``.
    batch2 : Tensor
        Batched right matrices of shape ``(B, K, N)``.
    beta : float, optional
        Scalar multiplier on ``input``. Defaults to ``1.0``.
    alpha : float, optional
        Scalar multiplier on the batched matmul sum. Defaults to ``1.0``.

    Returns
    -------
    Tensor
        Tensor of shape ``(M, N)``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out} = \beta \cdot \text{input}
        + \alpha \sum_{k = 0}^{B - 1}
        (\text{batch1}[k] \cdot \text{batch2}[k]).

    Differs from :func:`baddbmm` in that the batch dimension is reduced
    away; :func:`baddbmm` keeps the per-batch results.

    Examples
    --------
    >>> import lucid
    >>> b1 = lucid.ones((3, 2, 4))
    >>> b2 = lucid.ones((3, 4, 2))
    >>> M = lucid.zeros((2, 2))
    >>> lucid.addbmm(M, b1, b2)
    tensor([[12., 12.], [12., 12.]])
    """
    return _scaled_add(input, beta, lucid.sum(lucid.bmm(batch1, batch2), 0), alpha, op="addbmm")


@_amp_op("baddbmm")
def baddbmm(
    input: Tensor, batch1: Tensor, batch2: Tensor, *, beta: float = 1.0, alpha: float = 1.0
) -> Tensor:
    r"""Batched GEMM with a batched accumulator.

    Computes
    :math:`\beta \cdot \text{input} + \alpha \cdot \operatorname{bmm}(\text{batch1}, \text{batch2})`
    where the batch axis is preserved (unlike :func:`addbmm` which reduces
    it).

    Parameters
    ----------
    input : Tensor
        Accumulator of shape ``(B, M, N)``.
    batch1 : Tensor
        Batched left matrices of shape ``(B, M, K)``.
    batch2 : Tensor
        Batched right matrices of shape ``(B, K, N)``.
    beta : float, optional
        Scalar multiplier on ``input``. Defaults to ``1.0``.
    alpha : float, optional
        Scalar multiplier on ``bmm(batch1, batch2)``. Defaults to ``1.0``.

    Returns
    -------
    Tensor
        Tensor of shape ``(B, M, N)``.

    Notes
    -----
    Per-batch definition:

    .. math::

        \text{out}[k] = \beta \cdot \text{input}[k]
        + \alpha \cdot (\text{batch1}[k] \cdot \text{batch2}[k]).

    The implementation defers to :func:`lucid.bmm`, which dispatches to
    Accelerate-batched GEMM on CPU and MLX on GPU.

    Examples
    --------
    >>> import lucid
    >>> b1 = lucid.ones((3, 2, 4))
    >>> b2 = lucid.ones((3, 4, 2))
    >>> M = lucid.zeros((3, 2, 2))
    >>> lucid.baddbmm(M, b1, b2).shape
    (3, 2, 2)
    """
    return _scaled_add(input, beta, lucid.bmm(batch1, batch2), alpha, op="baddbmm")


@_amp_op("addmv")
def addmv(
    input: Tensor, mat: Tensor, vec: Tensor, *, beta: float = 1.0, alpha: float = 1.0
) -> Tensor:
    r"""Matrix-vector multiply with a scaled accumulator (BLAS-2 gemv).

    Computes
    :math:`\beta \cdot \text{input} + \alpha \cdot (\text{mat} \cdot \text{vec})`,
    the canonical BLAS-2 ``gemv`` operation.

    Parameters
    ----------
    input : Tensor
        Accumulator vector of shape ``(M,)``.
    mat : Tensor
        Matrix of shape ``(M, N)``.
    vec : Tensor
        Vector of shape ``(N,)``.
    beta : float, optional
        Scalar multiplier on ``input``. Defaults to ``1.0``.
    alpha : float, optional
        Scalar multiplier on ``mat @ vec``. Defaults to ``1.0``.

    Returns
    -------
    Tensor
        Vector of shape ``(M,)``.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i = \beta \cdot \text{input}_i
        + \alpha \cdot \sum_{j = 0}^{N - 1} \text{mat}_{ij} \cdot \text{vec}_j.

    Implemented by promoting ``vec`` to a column ``(N, 1)``, calling
    :func:`lucid.matmul`, and squeezing the trailing axis.

    Examples
    --------
    >>> import lucid
    >>> M = lucid.zeros(2)
    >>> A = lucid.tensor([[1., 2.], [3., 4.]])
    >>> v = lucid.tensor([5., 6.])
    >>> lucid.addmv(M, A, v)
    tensor([17., 39.])
    """
    mv_out = lucid.matmul(mat, vec.unsqueeze(-1)).squeeze(-1)
    return _scaled_add(input, beta, mv_out, alpha, op="addmv")


@_amp_op("addr")
def addr(
    input: Tensor, vec1: Tensor, vec2: Tensor, *, beta: float = 1.0, alpha: float = 1.0
) -> Tensor:
    r"""Rank-1 update of a matrix by an outer product (BLAS-2 ger).

    Computes
    :math:`\beta \cdot \text{input} + \alpha \cdot (\text{vec1} \otimes \text{vec2})`,
    i.e. updates the matrix ``input`` by adding the (scaled) outer product
    of two vectors. The classical BLAS ``ger`` routine.

    Parameters
    ----------
    input : Tensor
        Accumulator matrix of shape ``(M, N)``.
    vec1 : Tensor
        First vector of length ``M``.
    vec2 : Tensor
        Second vector of length ``N``.
    beta : float, optional
        Scalar multiplier on ``input``. Defaults to ``1.0``.
    alpha : float, optional
        Scalar multiplier on the outer product. Defaults to ``1.0``.

    Returns
    -------
    Tensor
        Updated matrix of shape ``(M, N)``.

    Notes
    -----
    Element-wise:

    .. math::

        \text{out}_{ij} = \beta \cdot \text{input}_{ij}
        + \alpha \cdot \text{vec1}_i \cdot \text{vec2}_j.

    The outer product is a rank-1 matrix, so this update can be used to
    build rank-1 corrections in optimisation algorithms (Broyden / BFGS).

    Examples
    --------
    >>> import lucid
    >>> M = lucid.zeros((2, 3))
    >>> u = lucid.tensor([1., 2.])
    >>> v = lucid.tensor([3., 4., 5.])
    >>> lucid.addr(M, u, v)
    tensor([[3., 4., 5.], [6., 8., 10.]])
    """
    return _scaled_add(input, beta, lucid.linalg.outer(vec1, vec2), alpha)


def _can_mix_devices(a: Tensor, b: Tensor) -> bool:
    return a.device == b.device or (
        (a.ndim == 0 and a.device.type == "cpu") or (b.ndim == 0 and b.device.type == "cpu")
    )


def _check_ternary_devices(op: str, input: Tensor, tensor1: Tensor, tensor2: Tensor) -> None:
    for name, peer in (("tensor1", tensor1), ("tensor2", tensor2)):
        if not _can_mix_devices(input, peer):
            raise _C_engine.DeviceMismatch(
                f"DeviceMismatch ({op}: input/{name}): "
                f"expected {input.device.type}, got {peer.device.type}"
            )
    if not _can_mix_devices(tensor1, tensor2):
        raise _C_engine.DeviceMismatch(
            f"DeviceMismatch ({op}: tensor1/tensor2): "
            f"expected {tensor1.device.type}, got {tensor2.device.type}"
        )


def _lerp(input: Tensor, end: Tensor, weight: float | Tensor) -> Tensor:
    """Linearly interpolate ``input`` and ``end`` by ``weight``."""
    if not (input.is_floating_point() and end.is_floating_point()):
        raise _C_engine.NotImplementedError(
            f"lerp expects floating-point tensors, got {input.dtype} and "
            f"{end.dtype} — cast them first"
        )
    if end.dtype != input.dtype:
        raise _C_engine.DtypeMismatch(
            f"DtypeMismatch (lerp: input/end): expected {input.dtype._name}, got {end.dtype._name}"
        )
    if not _can_mix_devices(input, end):
        raise _C_engine.DeviceMismatch(
            f"DeviceMismatch (lerp: input/end): expected {input.device.type}, got {end.device.type}"
        )
    diff = _wrap(_C_engine.sub(_unwrap(end), _unwrap(input)))
    if isinstance(weight, lucid.Tensor):
        if weight.dtype != input.dtype:
            raise _C_engine.DtypeMismatch(
                f"DtypeMismatch (lerp: input/weight): "
                f"expected {input.dtype._name}, got {weight.dtype._name}"
            )
        if not _can_mix_devices(diff, weight):
            raise _C_engine.DeviceMismatch(
                f"DeviceMismatch (lerp: input/weight): "
                f"expected {diff.device.type}, got {weight.device.type}"
            )
        scaled = _wrap(_C_engine.mul(_unwrap(weight), _unwrap(diff)))
    else:
        w_impl = _C_engine.full(
            list(_unwrap(diff).shape), float(weight), _unwrap(diff).dtype, _unwrap(diff).device
        )
        scaled = _wrap(_C_engine.mul(w_impl, _unwrap(diff)))
    return _wrap(_C_engine.add(_unwrap(input), _unwrap(scaled)))


def addcmul(input: Tensor, tensor1: Tensor, tensor2: Tensor, *, value: float = 1) -> Tensor:
    r"""Element-wise fused multiply-add with a scalar weight.

    Computes :math:`\text{input} + \text{value} \cdot (t_1 \odot t_2)`
    element-wise, where :math:`\odot` denotes the Hadamard (element-wise)
    product.  Provided as a single named op for readability and to give
    the dispatcher the chance to fuse the three steps into one kernel on
    backends that support it.

    Parameters
    ----------
    input : Tensor
        Tensor that is added to the scaled element-wise product.
    tensor1 : Tensor
        Left operand of the element-wise product.  Must broadcast with
        ``input`` and ``tensor2``.
    tensor2 : Tensor
        Right operand of the element-wise product.  Must broadcast with
        ``input`` and ``tensor1``.
    value : float, optional
        Scalar multiplier applied to the product ``t1 * t2`` before
        addition.  Defaults to ``1`` (plain multiply-and-add).

    Returns
    -------
    Tensor
        Element-wise result with the broadcast shape of the three input
        tensors.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_i =
        \text{input}_i + \text{value} \cdot (t_{1,i} \cdot t_{2,i}).

    Sister op of :func:`addcdiv` (``input + value * t1 / t2``).  Common
    in optimisers (e.g. running-mean / running-variance updates) where a
    bias-correction term is added to an existing accumulator.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0, 3.0])
    >>> b = lucid.tensor([4.0, 5.0, 6.0])
    >>> c = lucid.tensor([0.5, 0.5, 0.5])
    >>> lucid.addcmul(a, b, c, value=2.0)
    tensor([5., 7., 9.])
    """
    _check_ternary_devices("addcmul", input, tensor1, tensor2)
    with _autocast_entry("addcmul", _unwrap(input), _unwrap(tensor1), _unwrap(tensor2)) as (
        inp,
        t1,
        t2,
    ):
        input, tensor1, tensor2 = _wrap(inp), _wrap(t1), _wrap(t2)
        return input + (tensor1 * tensor2) * value


def addcdiv(input: Tensor, tensor1: Tensor, tensor2: Tensor, *, value: float = 1) -> Tensor:
    r"""Element-wise fused divide-add.

    Computes :math:`\text{input} + \text{value} \cdot (t_1 / t_2)` in
    one expression. Common in optimiser updates (e.g. Adam's
    parameter step uses the form ``param -= lr * m / sqrt(v)``).

    Parameters
    ----------
    input : Tensor
        Accumulator tensor.
    tensor1 : Tensor
        Numerator. Must broadcast with ``input`` and ``tensor2``.
    tensor2 : Tensor
        Denominator. Must broadcast with ``input`` and ``tensor1``.
    value : float, optional
        Scalar multiplier on the quotient. Defaults to ``1``.

    Returns
    -------
    Tensor
        Tensor with the broadcast shape of the inputs.

    Notes
    -----
    Element-wise definition:

    .. math::

        \text{out}_i = \text{input}_i
        + \text{value} \cdot \frac{(t_1)_i}{(t_2)_i}.

    Division by zero follows standard IEEE 754 rules — no clamping is
    applied. Gradients flow through all three tensor operands.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1.0, 2.0])
    >>> t1 = lucid.tensor([4.0, 9.0])
    >>> t2 = lucid.tensor([2.0, 3.0])
    >>> lucid.addcdiv(a, t1, t2, value=0.5)
    tensor([2., 3.5])
    """
    from lucid import _promotion

    _check_ternary_devices("addcdiv", input, tensor1, tensor2)
    if _promotion.is_integral(_unwrap(tensor1).dtype) and _promotion.is_integral(
        _unwrap(tensor2).dtype
    ):
        raise RuntimeError(
            "Integer division with addcdiv is no longer supported; use floor_divide instead."
        )
    with _autocast_entry("addcdiv", _unwrap(input), _unwrap(tensor1), _unwrap(tensor2)) as (
        inp,
        t1,
        t2,
    ):
        input, tensor1, tensor2 = _wrap(inp), _wrap(t1), _wrap(t2)
        return input + (tensor1 / tensor2) * value


@_amp_op("mv")
def mv(input: Tensor, vec: Tensor) -> Tensor:
    r"""Matrix-vector product.

    Computes the standard matrix-vector product
    :math:`y = M v`. Equivalent to ``M @ v`` for 2-D / 1-D inputs.

    Parameters
    ----------
    input : Tensor
        2-D matrix of shape ``(M, N)``.
    vec : Tensor
        1-D vector of shape ``(N,)``.

    Returns
    -------
    Tensor
        1-D tensor of shape ``(M,)``.

    Notes
    -----
    Element-wise definition:

    .. math::

        y_i = \sum_{j = 0}^{N - 1} \text{mat}_{ij} \cdot \text{vec}_j.

    Implemented by promoting ``vec`` to a column vector, calling
    :func:`lucid.matmul`, and then squeezing the trailing axis. Dispatches
    to BLAS-2 ``gemv`` on the CPU stream and the matmul kernel on the
    GPU stream.

    Examples
    --------
    >>> import lucid
    >>> M = lucid.tensor([[1., 2.], [3., 4.]])
    >>> v = lucid.tensor([5., 6.])
    >>> lucid.mv(M, v)
    tensor([17., 39.])
    """
    return lucid.matmul(input, vec.unsqueeze(-1)).squeeze(-1)


def ger(input: Tensor, vec2: Tensor) -> Tensor:
    r"""Outer product of two 1-D tensors.

    Forwards to :func:`lucid.linalg.outer`. Provided under the legacy BLAS
    name (``ger`` = "general rank-one update") for API parity.

    Parameters
    ----------
    input : Tensor
        1-D tensor of length ``M``.
    vec2 : Tensor
        1-D tensor of length ``N``.

    Returns
    -------
    Tensor
        2-D tensor of shape ``(M, N)`` holding the outer product.

    Notes
    -----
    Mathematical definition:

    .. math::

        \text{out}_{ij} = \text{vec1}_i \cdot \text{vec2}_j.

    The result is always rank-1 (in the linear-algebra sense). Use
    :func:`addr` to perform a rank-1 update in place of an accumulator.

    Examples
    --------
    >>> import lucid
    >>> u = lucid.tensor([1., 2., 3.])
    >>> v = lucid.tensor([4., 5.])
    >>> lucid.ger(u, v)
    tensor([[4., 5.], [8., 10.], [12., 15.]])
    """
    return lucid.linalg.outer(input, vec2)


def vdot(input: Tensor, other: Tensor) -> Tensor:
    r"""Vector dot product (complex-conjugating on the first argument).

    Forwards to :func:`lucid.linalg.dot`. For real tensors this is the
    usual inner product; for complex tensors (when supported) the first
    argument is conjugated, matching NumPy's ``vdot`` convention.

    Parameters
    ----------
    input : Tensor
        1-D tensor.
    other : Tensor
        1-D tensor of the same length as ``input``.

    Returns
    -------
    Tensor
        0-D scalar tensor holding the dot product.

    Notes
    -----
    Mathematical definition (general case):

    .. math::

        \text{out} = \sum_{i = 0}^{N - 1} \overline{a_i} \cdot b_i.

    For real ``input`` this reduces to the standard inner product
    :math:`\sum_i a_i b_i`. Tensors must be 1-D and of equal length.

    Examples
    --------
    >>> import lucid
    >>> a = lucid.tensor([1., 2., 3.])
    >>> b = lucid.tensor([4., 5., 6.])
    >>> lucid.vdot(a, b)
    tensor(32.)
    """
    # ``vdot`` has no row, so under autocast a mixed pair is refused as it is
    # outside, rather than taking ``dot``'s WIDEST Metal row inside.
    with _autocast_suspended(_unwrap(input).device):
        if (
            input.is_complex()
            and input.ndim == 1
            and other.ndim == 1
            and input.shape == other.shape
            and input.dtype == other.dtype
            and input.device == other.device
        ):
            return _la.vecdot(input, other)
        return _la.dot(input, other)


def block_diag(*tensors: Tensor) -> Tensor:
    r"""Construct a block-diagonal matrix from a sequence of blocks.

    Each input is placed on the diagonal of a larger matrix, with zeros
    everywhere off-block. Scalars (0-D) become :math:`1 \times 1` blocks
    and 1-D inputs become :math:`1 \times n` blocks before stacking.

    Parameters
    ----------
    *tensors : Tensor
        One or more 0-, 1-, or 2-D tensors to place on the diagonal,
        in order from top-left to bottom-right.

    Returns
    -------
    Tensor
        2-D block-diagonal matrix of shape
        ``(sum(h_i), sum(w_i))`` where ``h_i, w_i`` are the (promoted)
        block heights and widths.

    Notes
    -----
    For blocks :math:`A_1, A_2, \dots, A_k`, the result is

    .. math::

        \operatorname{blockdiag}(A_1, A_2, \dots, A_k) =
        \begin{pmatrix}
            A_1 & 0 & \cdots & 0 \\
            0 & A_2 & \cdots & 0 \\
            \vdots & & \ddots & \vdots \\
            0 & 0 & \cdots & A_k
        \end{pmatrix}.

    Called with zero arguments, returns an empty ``(0, 0)`` matrix.

    Examples
    --------
    >>> import lucid
    >>> A = lucid.tensor([[1., 2.], [3., 4.]])
    >>> B = lucid.tensor([[5.]])
    >>> lucid.block_diag(A, B)
    tensor([[1., 2., 0.], [3., 4., 0.], [0., 0., 5.]])
    """
    if not tensors:
        return lucid.zeros(0, 0)

    parts: list[Tensor] = []
    for t_i in tensors:
        if t_i.ndim == 0:
            t_i = t_i.reshape(1, 1)
        elif t_i.ndim == 1:
            t_i = t_i.unsqueeze(0)
        parts.append(t_i)

    total_cols = sum(t_i.shape[1] for t_i in parts)
    rows: list[Tensor] = []
    cum = 0
    for blk in parts:
        h, w = blk.shape
        pieces: list[Tensor] = []
        if cum:
            pieces.append(lucid.zeros(h, cum, dtype=blk.dtype, device=blk.device))
        pieces.append(blk)
        right = total_cols - cum - w
        if right:
            pieces.append(lucid.zeros(h, right, dtype=blk.dtype, device=blk.device))
        rows.append(lucid.cat(pieces, 1))
        cum += w
    return lucid.cat(rows, 0)


def logdet(input: Tensor) -> Tensor:
    """Log-determinant of a square matrix (or batch).

    Returns ``log(det(A))``.  Defined only for matrices with ``det > 0``;
    returns NaN otherwise.  Uses ``linalg.slogdet`` internally.
    """
    sign, logabsdet = _la.slogdet(input)
    zero = lucid.zeros_like(sign)
    nan = lucid.full_like(logabsdet, float("nan"))
    return lucid.where(sign > zero, logabsdet, nan)


__all__ = [
    "addmm",
    "addbmm",
    "baddbmm",
    "addmv",
    "addr",
    "addcmul",
    "addcdiv",
    "mv",
    "ger",
    "vdot",
    "block_diag",
    "logdet",
]

# Bound at runtime, not only for type checking: PEP 649 evaluates a
# function's annotations in its own module globals when something asks
# for them, so a name that exists under ``TYPE_CHECKING`` alone makes
# ``inspect.signature`` raise NameError — which is what help(), an IDE
# and the docs build all call. Imported at the bottom because this
# module is inside the cycle ``Tensor`` closes.
from lucid._tensor.tensor import Tensor
