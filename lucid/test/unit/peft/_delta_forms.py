"""The updates of DeLoRA, GraLoRA, HiRA, GLoRA, DEFT and PEANuT, written out entry by entry from the papers' formulas.

Plain lists and loops, never the layers' code: a test expects an update from here and compares the
layer to it.  Matrices are lists of rows.  Each function takes the adapter's tensors as the layer
holds them and returns what the paper says the update (or the adapted weight) is.
"""

import math

type Matrix = list[list[float]]


def eye(n: int) -> Matrix:
    """The ``n x n`` identity."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    """``a @ b``."""
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def transpose(a: Matrix) -> Matrix:
    """``a.T``."""
    return [list(row) for row in zip(*a, strict=True)]


def add(a: Matrix, b: Matrix) -> Matrix:
    """``a + b``."""
    return [[x + y for x, y in zip(ra, rb, strict=True)] for ra, rb in zip(a, b, strict=True)]


def sub(a: Matrix, b: Matrix) -> Matrix:
    """``a - b``."""
    return [[x - y for x, y in zip(ra, rb, strict=True)] for ra, rb in zip(a, b, strict=True)]


def hadamard(a: Matrix, b: Matrix) -> Matrix:
    """The elementwise product ``a * b``."""
    return [[x * y for x, y in zip(ra, rb, strict=True)] for ra, rb in zip(a, b, strict=True)]


def scaled(a: Matrix, s: float) -> Matrix:
    """``s * a``."""
    return [[s * x for x in row] for row in a]


def frobenius(a: Matrix) -> float:
    """The Frobenius norm of ``a``."""
    return math.sqrt(sum(x * x for row in a for x in row))


# ── DeLoRA (Bini et al., 2025, Eq. 5-8, 15) ──────────────────────────────────


def delora_delta(a: Matrix, b: Matrix, boundary: float, w_norm: list[float]) -> Matrix:
    r""":math:`\Delta W = \frac{\lambda}{r}\sum_i \frac{b_i a_i^\top}{\lVert b_i\rVert\lVert a_i\rVert}` with column :math:`j` scaled by ``w_norm[j]``.

    ``a`` is ``(r, in)``, ``b`` is ``(out, r)``; a norm is floored at 1e-4 as the layer floors it.
    """
    r = len(a)
    out, inn = len(b), len(a[0])
    delta = [[0.0] * inn for _ in range(out)]
    for k in range(r):
        a_norm = max(math.sqrt(sum(v * v for v in a[k])), 1e-4)
        b_norm = max(math.sqrt(sum(b[o][k] ** 2 for o in range(out))), 1e-4)
        for o in range(out):
            for i in range(inn):
                delta[o][i] += boundary / r * b[o][k] * a[k][i] / (a_norm * b_norm)
    return [[delta[o][i] * w_norm[i] for i in range(inn)] for o in range(out)]


# ── GraLoRA (Jung et al., 2025, Eq. 5) ───────────────────────────────────────


def gralora_delta(
    a: list[Matrix],
    b: list[Matrix],
    scaling: float,
    general: tuple[Matrix, Matrix] | None = None,
) -> Matrix:
    r"""The block grid of the paper: block ``(j, n)`` of the update is :math:`B_{j,n} A_{j,n}^\top`.

    ``a[n]`` is ``(in / k, r)`` and ``b[j]`` is ``(r, out / k)``, as the layer keeps them; the
    paper's :math:`A_{j,n}` is the ``j``-th run of ``r / k`` columns of ``a[n]`` and
    :math:`B_{j,n}^\top` the ``n``-th run of ``r / k`` rows of ``b[j]``.  ``general`` is the hybrid
    LoRA pair ``(A, B)`` of shapes ``(h, in)`` and ``(out, h)``; the whole sum is scaled.
    """
    k, sub_in, r = len(a), len(a[0]), len(a[0][0])
    sub_out, s = len(b[0][0]), r // k
    delta = [[0.0] * (k * sub_in) for _ in range(k * sub_out)]
    for j in range(k):
        for n in range(k):
            for o in range(sub_out):
                for i in range(sub_in):
                    delta[j * sub_out + o][n * sub_in + i] = sum(
                        b[j][n * s + t][o] * a[n][i][j * s + t] for t in range(s)
                    )
    if general is not None:
        delta = add(delta, matmul(general[1], general[0]))
    return scaled(delta, scaling)
