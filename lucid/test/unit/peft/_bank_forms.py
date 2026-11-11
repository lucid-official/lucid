"""The updates of Uni-LoRA, PVeRA, TinyLoRA and Lily, written out entry by entry from the papers' formulas.

Plain lists and loops, never the layers' code: a test expects an update from here and compares the
layer to it.  Matrices are lists of rows.  Each function takes the adapter's tensors as a layer
holds them and returns what the paper says the update is.
"""

import math

from lucid.test.unit.peft._delta_forms import Matrix, matmul, transpose

# ── Uni-LoRA (Li et al., 2025, Sec. 3.2, Eq. 2 and Algorithm 1) ──────────────


def unilora_factors(
    theta: list[float],
    index_a: list[list[int]],
    index_b: list[list[int]],
    scale_a: list[list[float]],
    scale_b: list[list[float]],
) -> tuple[Matrix, Matrix]:
    """``A[i][j] = theta[index_a[i][j]] * scale_a[i][j]`` and likewise ``B``."""
    a = [
        [theta[k] * s for k, s in zip(ik, sk, strict=True)]
        for ik, sk in zip(index_a, scale_a, strict=True)
    ]
    b = [
        [theta[k] * s for k, s in zip(ik, sk, strict=True)]
        for ik, sk in zip(index_b, scale_b, strict=True)
    ]
    return a, b


def unilora_delta(
    theta: list[float],
    index_a: list[list[int]],
    index_b: list[list[int]],
    scale_a: list[list[float]],
    scale_b: list[list[float]],
) -> Matrix:
    """``B A``: no ``alpha / r`` in Uni-LoRA."""
    a, b = unilora_factors(theta, index_a, index_b, scale_a, scale_b)
    return matmul(b, a)


def unilora_projection(indices: list[int], theta_d_length: int) -> tuple[Matrix, list[float]]:
    """The dense ``P`` (``D x d``) the paper defines for an assignment, and its column scales ``1 / sqrt(n_j)``."""
    counts = [indices.count(j) for j in range(theta_d_length)]
    scales = [1.0 / math.sqrt(n) if n else 0.0 for n in counts]
    p = [
        [scales[j] if indices[row] == j else 0.0 for j in range(theta_d_length)]
        for row in range(len(indices))
    ]
    return p, scales


# ── PVeRA (Fillioux et al., 2025, Sec. 3.2 and 3.3, Eq. 5-7 and 10) ──────────


def pvera_mean_delta(
    proj_a: Matrix,
    proj_b: Matrix,
    lambda_d: list[float],
    lambda_b: list[float],
    size: tuple[int, int],
) -> Matrix:
    """The merged update ``(lambda_b * B) (A_mu * lambda_d_mu)`` of Eq. 10: ``mu`` is the first ``r`` of the ``2r`` latent entries.

    ``proj_a`` is ``(2r, in_max)`` and ``proj_b`` ``(out_max, r)``, both sliced to the layer's
    ``(out, in)``.
    """
    out, inn = size
    r = len(lambda_d) // 2
    a = [[lambda_d[i] * proj_a[i][j] for j in range(inn)] for i in range(r)]
    b = [[lambda_b[o] * proj_b[o][k] for k in range(r)] for o in range(out)]
    return matmul(b, a)


def pvera_latent(
    x: Matrix, proj_a: Matrix, lambda_d: list[float], in_features: int
) -> tuple[Matrix, Matrix]:
    """``mu, logvar = split(x A^T * lambda_d)`` (Eq. 5): the two halves of the ``2r`` latent entries."""
    r = len(lambda_d) // 2
    a = [row[:in_features] for row in proj_a]
    hidden = matmul(x, transpose(a))
    scaled = [[h * d for h, d in zip(row, lambda_d, strict=True)] for row in hidden]
    return [row[:r] for row in scaled], [row[r:] for row in scaled]


# ── TinyLoRA (Morris et al., 2026, Sec. 4.0.2) ───────────────────────────────


def tinylora_r(v: list[float], p: list[Matrix]) -> Matrix:
    """``R = sum_i v_i P_i``."""
    r = len(p[0])
    return [[sum(v[i] * p[i][j][k] for i in range(len(v))) for k in range(r)] for j in range(r)]


def tinylora_delta(a: Matrix, b: Matrix, p: list[Matrix], v: list[float]) -> Matrix:
    """``B R A``: the frozen SVD factors (singular values split over ``A`` and ``B``) around ``R``."""
    return matmul(matmul(b, tinylora_r(v, p)), a)


# ── Lily (Zhong et al., 2024, Sec. 3.1-3.2, Eq. 1-4, and Fig. 7) ─────────────


def softmax(values: list[float]) -> list[float]:
    """``softmax`` of a list."""
    peak = max(values)
    exps = [math.exp(v - peak) for v in values]
    return [e / sum(exps) for e in exps]


def lily_update(
    x: Matrix, lp: Matrix, hp: Matrix, router: Matrix, num_b: int, scaling: float
) -> Matrix:
    """``s * h * sum_e S_e B_e`` for ``h = x A``; ``S`` is the router's softmax averaged over every row of the batch.

    ``lp`` is ``A``'s weight ``(r, in)``, ``hp`` the ``B`` experts' weight ``(num_b * r, out)`` (expert
    ``e`` is rows ``e r .. (e + 1) r``) and ``router`` its weight ``(num_b, r)``.
    """
    hidden = matmul(x, transpose(lp))
    r = len(lp)
    probabilities = [softmax(row) for row in matmul(hidden, transpose(router))]
    mean = [sum(p[e] for p in probabilities) / len(probabilities) for e in range(num_b)]
    out = len(hp[0])
    combined = [
        [sum(mean[e] * hp[e * r + i][o] for e in range(num_b)) for o in range(out)]
        for i in range(r)
    ]
    return [[scaling * value for value in row] for row in matmul(hidden, combined)]
