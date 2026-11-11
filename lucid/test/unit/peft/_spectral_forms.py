"""The updates of RoAd, PSOFT, OSF and FRoD, written out entry by entry from the papers' formulas.

Plain lists and loops, never the layers' code: a test expects a matrix from here and compares the
layer to it.  Matrices are lists of rows (``_delta_forms`` has the helpers).  Each function takes
the adapter's tensors as an adapter file holds them and returns what the paper says the adapted
weight, or the matrix that multiplies it, is.
"""

import math

from lucid.test.unit.peft._delta_forms import Matrix, add, eye, matmul, sub, transpose

# ── RoAd (Liao and Monz, 2024, section 3.2, Eq. 3 and Table 1) ───────────────


def road_matrix(variant: str, group_size: int, theta: list[float], alpha: list[float]) -> Matrix:
    r""":math:`R`, ``out x out``: a block ``[[a11 cos t11, -a12 sin t12], [a21 sin t21, a22 cos t22]]`` per pair.

    Pair ``k`` of a group of ``g`` is ``(k, k + g/2)``.  ``road_1`` gives the four angles of a block
    one value and the four scales one value; ``road_2`` one value to each row of the block;
    ``road_4`` gives the diagonal and the off-diagonal terms values of their own.  The file holds
    them as the reference lays them out: per pair (``road_1``), per element (``road_2``), or per
    element twice, a group's first ``g`` for the diagonal and its next ``g`` for the off-diagonal
    (``road_4``).
    """
    half = group_size // 2
    width = {"road_1": len(theta) * 2, "road_2": len(theta), "road_4": len(theta) // 2}[variant]
    matrix = [[0.0] * width for _ in range(width)]
    for group in range(width // group_size):
        for k in range(half):
            i, j = group * group_size + k, group * group_size + k + half
            if variant == "road_1":
                p = group * half + k
                scales, angles = (alpha[p],) * 4, (theta[p],) * 4
            elif variant == "road_2":
                scales = (alpha[i], alpha[i], alpha[j], alpha[j])
                angles = (theta[i], theta[i], theta[j], theta[j])
            else:
                base = group * 2 * group_size
                scales = (
                    alpha[base + k],
                    alpha[base + group_size + k],
                    alpha[base + group_size + k + half],
                    alpha[base + k + half],
                )
                angles = (
                    theta[base + k],
                    theta[base + group_size + k],
                    theta[base + group_size + k + half],
                    theta[base + k + half],
                )
            a11, a12, a21, a22 = scales
            t11, t12, t21, t22 = angles
            matrix[i][i] = a11 * math.cos(t11)
            matrix[i][j] = -a12 * math.sin(t12)
            matrix[j][i] = a21 * math.sin(t21)
            matrix[j][j] = a22 * math.cos(t22)
    return matrix


def road_adapted(
    weight: Matrix,
    bias: list[float],
    variant: str,
    group_size: int,
    theta: list[float],
    alpha: list[float],
) -> tuple[Matrix, list[float]]:
    r""":math:`(R W, R b)`: the weight and the bias of the layer with its output turned by :math:`R`."""
    rotation = road_matrix(variant, group_size, theta, alpha)
    column = matmul(rotation, [[b] for b in bias])
    return matmul(rotation, weight), [row[0] for row in column]


# ── PSOFT (Wu et al., 2025, section 4, Eq. 6-8 and Appendix C) ───────────────


def invert(a: Matrix) -> Matrix:
    """``a^-1`` by Gauss-Jordan elimination with partial pivoting."""
    n = len(a)
    work = [list(row) + eye(n)[i] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(work[r][col]))
        work[col], work[pivot] = work[pivot], work[col]
        divisor = work[col][col]
        work[col] = [v / divisor for v in work[col]]
        for r in range(n):
            if r != col:
                factor = work[r][col]
                work[r] = [v - factor * w for v, w in zip(work[r], work[col], strict=True)]
    return [row[n:] for row in work]


def skew(upper: list[float], size: int) -> Matrix:
    """The skew-symmetric :math:`Q` with ``upper`` on its entries above the diagonal, row by row."""
    q = [[0.0] * size for _ in range(size)]
    entries = iter(upper)
    for i in range(size):
        for j in range(i + 1, size):
            q[i][j] = next(entries)
            q[j][i] = -q[i][j]
    return q


def cayley(upper: list[float], size: int) -> Matrix:
    r""":math:`(I + Q)(I - Q)^{-1}` of the skew-symmetric :math:`Q` that ``upper`` fills.

    The paper writes the transform :math:`(I - Q)(I + Q)^{-1}` (section 4.2); the reference's, which a
    saved adapter was trained with, is the same family of matrices with :math:`Q` negated.
    """
    q = skew(upper, size)
    return matmul(add(eye(size), q), invert(sub(eye(size), q)))


def psoft_inner(rotation: Matrix, mag_b: list[float] | None, mag_a: list[float] | None) -> Matrix:
    r""":math:`\operatorname{diag}(\beta)\,R\,\operatorname{diag}(\alpha)` (Eq. 8): ``mag_b`` on the left of ``rotation``, ``mag_a`` on its right.

    ``None`` is a vector of ones.
    """
    size = len(rotation)
    return [
        [
            (mag_b[i] if mag_b else 1.0) * rotation[i][j] * (mag_a[j] if mag_a else 1.0)
            for j in range(size)
        ]
        for i in range(size)
    ]


def psoft_delta(
    u: Matrix, s: list[float], vt: Matrix, inner: Matrix, rank: int, init: str
) -> Matrix:
    r""":math:`\Delta W = U_r\Sigma_r\,(C - I)\,V_r^\top` (Eq. 6-8, in the weight's ``out x in`` layout), from the SVD ``u diag(s) vt``.

    The paper writes the weight as ``d x n`` and the layer as :math:`h = W^\top x`; transposed, its
    :math:`A' (\cdot) B'` with :math:`B' = \Sigma V^\top` becomes :math:`U\Sigma\,(\cdot)^\top V^\top`:
    the singular values sit on the output side of :math:`C`.  ``"pissa_init"`` splits them,
    :math:`U\sqrt\Sigma\,(C - I)\sqrt\Sigma V^\top` (Eq. 3).  ``inner`` is :math:`C`.
    """
    power = 0.5 if init == "pissa_init" else 0.0
    left = [[u[i][k] * s[k] ** (1.0 - power) for k in range(rank)] for i in range(len(u))]
    right = [[vt[k][j] * s[k] ** power for j in range(len(vt[0]))] for k in range(rank)]
    return matmul(matmul(left, sub(inner, eye(rank))), right)


# ── OSF (Nayak et al., 2025, section 3.2) ────────────────────────────────────


def osf_adapted(
    u: Matrix,
    s: list[float],
    vt: Matrix,
    u_low: Matrix,
    s_low: list[float],
    v_low: Matrix,
    frozen: int,
) -> Matrix:
    r""":math:`W' = U_{high}\Sigma_{high}V_{high}^\top + U_{low}\Sigma_{low}V_{low}^\top`.

    The top ``frozen`` triplets of the SVD ``u diag(s) vt`` of the weight are kept, whatever signs the
    decomposition gave them (the products are the same); the low part is the one the file holds.
    """
    rows, columns = len(u), len(vt[0])
    return [
        [
            sum(u[i][k] * s[k] * vt[k][j] for k in range(frozen))
            + sum(u_low[i][k] * s_low[k] * v_low[k][j] for k in range(len(s_low)))
            for j in range(columns)
        ]
        for i in range(rows)
    ]


# ── FRoD (Wan et al., 2025, Eq. 6-8) ─────────────────────────────────────────


def frod_adapted(
    weight: Matrix,
    basis: Matrix,
    indices: list[list[int]],
    norms: list[float],
    entries: list[float],
) -> Matrix:
    r""":math:`W' = U(\Sigma + S)V^\top` of the layer weight ``weight`` (``out x in``), the basis :math:`V` and the file's tensors.

    :math:`U` is :math:`B = W V^{-\top}` with its columns divided by their norms (Eq. 6; a column of zero norm
    stays zero); ``norms`` is the file's trained :math:`\operatorname{diag}\Sigma`, which is *not* what
    :math:`U` was divided by; ``entries`` sit at ``indices`` (``[rows, columns]``) of :math:`S`.
    """
    columns = matmul(weight, transpose(invert(basis)))
    scale = [math.sqrt(sum(row[j] ** 2 for row in columns)) for j in range(len(columns[0]))]
    u = [[v / scale[j] if scale[j] > 1e-12 else 0.0 for j, v in enumerate(row)] for row in columns]
    size = len(basis)
    middle = [[norms[i] if i == j else 0.0 for j in range(size)] for i in range(size)]
    for row, column, value in zip(indices[0], indices[1], entries, strict=True):
        middle[row][column] += value
    return matmul(matmul(u, middle), transpose(basis))
