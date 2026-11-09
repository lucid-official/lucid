"""``matrix_exp``: the matrix exponential by Padé [6/6] with scaling and squaring.

Re-exported by :mod:`lucid.linalg`, whose ``inv`` it solves with.
"""

import lucid
from lucid._tensor.tensor import Tensor
from lucid.linalg import inv


def matrix_exp(input: Tensor) -> Tensor:
    r"""Matrix exponential :math:`e^A`.

    Returns the matrix exponential

    .. math::

        e^A \,=\, \sum_{k=0}^{\infty} \frac{A^k}{k!},

    which solves the matrix ODE :math:`\dot Y = A Y` with initial
    condition :math:`Y(0) = I`.  Computed by the scaling-and-squaring
    algorithm of Higham (2005) with a Padé [6/6] rational approximant:

    1. Scale :math:`A' = A / 2^s` so that :math:`\|A'\| \le \theta_6`.
    2. Approximate :math:`R \approx e^{A'}` via Padé [6/6] using the
      even/odd polynomial splitting :math:`R = D^{-1} N`.
    3. Square :math:`R` a total of :math:`s` times to recover
       :math:`e^A = R^{2^s}`.

    Parameters
    ----------
    input : Tensor
        Square matrix of shape ``(*, n, n)``.

    Returns
    -------
    Tensor
        :math:`e^A`, shape ``(*, n, n)``.

    Notes
    -----
    Computational cost is :math:`O((s + 6)\, n^3)` where the scaling
    parameter :math:`s = \lceil \log_2 (\|A\| / \theta_6) \rceil`.  The
    Frobenius norm is used here as a (conservative) proxy for the
    1-norm — this may add one extra squaring step but keeps the result
    correct.

    Differentiable: built entirely on :func:`matmul` and :func:`inv`,
    so autograd flows naturally.  Batched inputs (``ndim > 2``) are
    handled element-wise.

    Examples
    --------
    >>> import lucid
    >>> from lucid.linalg import matrix_exp
    >>> A = lucid.tensor([[0.0, 1.0], [-1.0, 0.0]])  # 90-deg rotation generator
    >>> matrix_exp(A)
    tensor([[0.5403, 0.8415], [-0.8415, 0.5403]])
    """
    import math as _math

    A = input
    sh: tuple[int, ...] = tuple(A.shape)
    if len(sh) < 2 or sh[-1] != sh[-2]:
        raise ValueError(
            f"matrix_exp requires a square matrix in the last two dims, got shape {sh}"
        )
    n: int = int(sh[-1])

    # -- Padé [6/6] coefficients c_j = (12-j)! * 6! / (12! * j! * (6-j)!) ----
    _c0: float = 1.0
    _c1: float = 0.5
    _c2: float = 5.0 / 44.0
    _c3: float = 1.0 / 66.0
    _c4: float = 1.0 / 792.0
    _c5: float = 1.0 / 15840.0
    _c6: float = 1.0 / 665280.0
    # theta_6: Frobenius-norm threshold at which Padé [6/6] is machine-accurate
    _theta: float = 2.0

    # -- Scaling ---------------------------------------------------------------
    # Upper bound for the 1-norm: max element × n (safe over-estimate).
    # An empty matrix has nothing to bound (and ``max`` of nothing is refused).
    norm_bound: float = float(A.abs().max().item()) * n if A.numel() else 0.0
    s: int = max(0, _math.ceil(_math.log2(norm_bound / _theta))) if norm_bound > _theta else 0
    A_sc: Tensor = A * (2.0 ** (-s))

    # -- Matrix powers ---------------------------------------------------------
    A2, A4 = A_sc @ A_sc, (A_sc @ A_sc) @ (A_sc @ A_sc)
    A6: Tensor = A2 @ A4

    # -- Identity (broadcasts over any batch dims) -----------------------------
    eye_2d: Tensor = lucid.eye(n, dtype=A.dtype, device=A.device)
    I: Tensor = (
        eye_2d + lucid.zeros(*sh[:-2], n, n, dtype=A.dtype, device=A.device)
        if len(sh) > 2
        else eye_2d
    )

    # -- Padé polynomials (even/odd split for efficiency) ----------------------
    Even: Tensor = _c0 * I + _c2 * A2 + _c4 * A4 + _c6 * A6
    Odd: Tensor = A_sc @ (_c1 * I + _c3 * A2 + _c5 * A4)
    N, D = Even + Odd, Even - Odd
    R: Tensor = inv(D) @ N

    # -- Squaring step ---------------------------------------------------------
    for _ in range(s):
        R = R @ R

    return R
