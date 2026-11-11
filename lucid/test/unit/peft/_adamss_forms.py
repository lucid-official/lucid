"""AdaMSS's closed forms and k-means, written as plain loops over Python floats.

The layers' own code builds the update with tensor products; the tests hold it to these sums, taken
entry by entry from the paper's formulas (Zheng et al., 2025, NeurIPS, section 3.3) and to Lloyd's
algorithm as the library that the reference calls for the clustering writes it down, so neither
borrows a line from what it checks.  Matrices are nested lists.
"""

Matrix = list[list[float]]


def matmul(left: Matrix, right: Matrix) -> Matrix:
    """``left @ right`` as three nested sums."""
    return [
        [sum(left[i][k] * right[k][j] for k in range(len(right))) for j in range(len(right[0]))]
        for i in range(len(left))
    ]


def joined_delta(
    members: list[list[int]],
    factors: list[Matrix],
    updates: list[Matrix],
    projection: Matrix,
    out: int,
) -> Matrix:
    """:math:`\\Delta\\tilde W`, ``(out, columns)``: row ``members[k][i]`` is row ``i`` of :math:`B_k A_k V_r^\\top`.

    Each subspace ``k`` has the rows ``members[k]`` of the weight, the factors ``factors[k]``
    (:math:`A_k`, ``r_k x r``) and ``updates[k]`` (:math:`B_k`, ``n_k x r_k``), and every subspace
    reads the same ``projection`` (:math:`V_r^\\top`, ``r x columns``).
    """
    delta = [[0.0] * len(projection[0]) for _ in range(out)]
    for rows, a, b in zip(members, factors, updates, strict=True):
        block = matmul(matmul(b, a), projection)
        for i, row in enumerate(rows):
            delta[row] = block[i]
    return delta


def layer(x: Matrix, weight: Matrix, bias: list[float] | None, delta: Matrix) -> Matrix:
    """``y = [W | b] [x; 1] + delta [x; 1]`` row by row: the base layer plus the update of the joined weight."""
    joined = [row + ([bias[i]] if bias is not None else []) for i, row in enumerate(weight)]
    full = [[joined[i][c] + delta[i][c] for c in range(len(joined[i]))] for i in range(len(joined))]
    outputs: Matrix = []
    for sample in x:
        augmented = sample + ([1.0] if bias is not None else [])
        outputs.append([sum(row[c] * augmented[c] for c in range(len(augmented))) for row in full])
    return outputs


def target(total: int, goal: int, init: int, final: int, exponent: float, step: int) -> int:
    """The subspaces ASA allows at ``step``: ``goal + (total - goal) * (1 - progress) ** exponent``, truncated."""
    if step > final:
        return goal
    remaining = max(0.0, min(1.0, 1.0 - (step - init) / (final - init)))
    return int(goal + (total - goal) * remaining**exponent)


def lloyd(points: Matrix, seeds: list[int], iterations: int, tolerance: float) -> list[int]:
    """Lloyd's k-means from the rows ``seeds`` of ``points``, as the reference's library runs it.

    Rows are centred first; each round assigns every row to the nearest center (the first on a
    tie), averages, and stops when the labels repeat (the labels as they are) or the centers have
    moved ``tolerance`` or less in squared distance in all (labelled once more against the new
    centers); after ``iterations`` rounds it also labels once more.
    """
    n, d = len(points), len(points[0])
    mean = [sum(row[c] for row in points) / n for c in range(d)]
    rows = [[row[c] - mean[c] for c in range(d)] for row in points]
    centers = [list(rows[s]) for s in seeds]

    def nearest() -> list[int]:
        labels = []
        for row in rows:
            distances = [sum((row[c] - center[c]) ** 2 for c in range(d)) for center in centers]
            labels.append(distances.index(min(distances)))
        return labels

    previous: list[int] = []
    for _ in range(iterations):
        labels = nearest()
        updated = []
        for k, old in enumerate(centers):
            held = [rows[i] for i in range(n) if labels[i] == k]
            updated.append(
                [sum(r[c] for r in held) / len(held) for c in range(d)] if held else list(old)
            )
        shift = sum(
            (updated[k][c] - centers[k][c]) ** 2 for k in range(len(centers)) for c in range(d)
        )
        centers = updated
        if labels == previous:
            return labels
        if shift <= tolerance:
            break
        previous = labels
    return nearest()
