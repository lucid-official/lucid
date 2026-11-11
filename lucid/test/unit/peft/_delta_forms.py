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


# ── HiRA (Huang et al., 2025) ────────────────────────────────────────────────


def hira_delta(weight: Matrix, a: Matrix, b: Matrix) -> Matrix:
    r""":math:`\Delta W = W \odot (B A)` for ``a`` of shape ``(r, in)`` and ``b`` of shape ``(out, r)``."""
    return hadamard(weight, matmul(b, a))


def hira_conv_delta(weight: list[Matrix], a: list[Matrix], b: Matrix) -> list[Matrix]:
    r""":math:`\Delta W_{o,i,k} = W_{o,i,k} \sum_s B_{o,s} A_{s,i,k}`.

    ``weight`` is ``(out, in, K)``, ``a`` is ``(r, in, K)`` (the kernel's axes flattened to ``K``) and
    ``b`` is ``(out, r)`` (its trailing ones dropped).
    """
    return [
        [
            [
                weight[o][i][k] * sum(b[o][s] * a[s][i][k] for s in range(len(a)))
                for k in range(len(weight[o][i]))
            ]
            for i in range(len(weight[o]))
        ]
        for o in range(len(weight))
    ]


# ── GLoRA (Chavan et al., 2023, Eq. 10, 13, 14) ──────────────────────────────

type Support = tuple[str, list[Matrix]]
"""A support tensor as its adapter holds it: its kind and its parts (``Xd, Xu`` / ``X`` / ``zeros``)."""


def glora_value(support: Support, rows: int, cols: int) -> Matrix:
    """The ``rows x cols`` matrix a support tensor stands for: a product, a column or a scalar broadcast, or zeros."""
    kind, parts = support
    if kind == "lora":
        return matmul(parts[0], parts[1])
    if kind == "vector":
        flat = [v[0] if isinstance(v, list) else v for v in parts[0]]
        return [[flat[o]] * cols for o in range(rows)]
    if kind == "constant":
        return [[parts[0][0]] * cols for _ in range(rows)]
    if not parts:
        return [[0.0] * cols for _ in range(rows)]
    held = parts[0]  # an absent support's buffer, which a file may fill with anything
    return [[v] for v in held] if not isinstance(held[0], list) else held


def glora_changes(
    weight: Matrix,
    bias: list[float] | None,
    a: Support,
    b: Support,
    c: Support,
    d: Support,
    e: Support,
) -> tuple[Matrix, list[float] | None]:
    r""":math:`\Delta W = W \odot A + B` and :math:`\Delta b = b \odot D + E + W C`, entry by entry."""
    out, inn = len(weight), len(weight[0])
    delta_a, delta_b = glora_value(a, out, inn), glora_value(b, out, inn)
    delta_w = [
        [weight[o][i] * delta_a[o][i] + delta_b[o][i] for i in range(inn)] for o in range(out)
    ]
    if bias is None:
        return delta_w, None
    column = glora_value(c, inn, 1)
    scale, shift = glora_value(d, out, 1), glora_value(e, out, 1)
    delta_bias = [
        bias[o] * scale[o][0] + shift[o][0] + sum(weight[o][i] * column[i][0] for i in range(inn))
        for o in range(out)
    ]
    return delta_w, delta_bias


# ── DEFT (Kumar et al., 2025, Eq. 1, 2) ──────────────────────────────────────


def householder_q(a: Matrix) -> Matrix:
    """The first ``r`` columns of ``Q`` of the QR decomposition of the ``m x r`` matrix ``a`` by Householder reflections.

    The reflector of each column maps it onto ``-sign(x_0) |x|`` times a basis vector, which is
    the sign convention of LAPACK's ``geqrf`` and so of the reference's QR: the injection ``Q_P R``
    depends on the sign of every column of ``Q_P``.
    """
    m, r = len(a), len(a[0])
    work = [list(row) for row in a]
    q = eye(m)
    for j in range(r):
        x = [work[i][j] for i in range(j, m)]
        norm = math.sqrt(sum(v * v for v in x))
        if norm == 0.0:
            continue
        v = list(x)
        v[0] -= -math.copysign(norm, x[0])  # x - beta e_0, beta = -sign(x_0) |x|
        scale = sum(c * c for c in v)
        for k in range(j, r):  # H applied to the trailing columns
            dot = sum(v[i - j] * work[i][k] for i in range(j, m))
            for i in range(j, m):
                work[i][k] -= 2.0 * v[i - j] * dot / scale
        for row in q:  # Q <- Q H
            dot = sum(row[j + i] * v[i] for i in range(len(v)))
            for i in range(len(v)):
                row[j + i] -= 2.0 * dot * v[i] / scale
    return [row[:r] for row in q]


def deft_delta(
    weight: Matrix,
    direction: Matrix,
    injection: Matrix | None,
    *,
    method: str,
    scaling: float,
) -> Matrix:
    r""":math:`\Delta W = Q_P\,(s R - \mathrm{right}^\top W)`, or :math:`-Q_P\,\mathrm{right}^\top W` without an injection.

    ``direction`` is ``P`` (``out x r``), ``injection`` ``R`` (``r x in``); ``method`` is ``"relu"``
    (``Q_P = P``, ``right = relu(P)``) or ``"qr"`` (``Q_P = right = Q`` of ``P``).
    """
    if method == "relu":
        basis, right = direction, [[max(v, 0.0) for v in row] for row in direction]
    else:
        basis = right = householder_q(direction)
    removed = matmul(transpose(right), weight)
    inner = scaled(removed, -1.0) if injection is None else sub(scaled(injection, scaling), removed)
    return matmul(basis, inner)
