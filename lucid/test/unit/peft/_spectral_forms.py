"""The updates of RoAd, PSOFT, OSF and FRoD, written out entry by entry from the papers' formulas.

Plain lists and loops, never the layers' code: a test expects a matrix from here and compares the
layer to it.  Matrices are lists of rows (``_delta_forms`` has the helpers).  Each function takes
the adapter's tensors as an adapter file holds them and returns what the paper says the adapted
weight, or the matrix that multiplies it, is.
"""

import math

from lucid.test.unit.peft._delta_forms import Matrix, matmul

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
