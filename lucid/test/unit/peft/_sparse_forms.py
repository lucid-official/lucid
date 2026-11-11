"""The adapted weights of the sparse methods, from the papers' formulas, with plain loops and no adapter code.

``test_sparse_bias_contract`` loads adapter files written by hand in the reference's layout and needs
what the loaded model must compute; these functions say it entry by entry.  The tiny MLP is
``Linear(6, 8) -> ReLU -> Linear(8, 4)``.
"""


def scattered(weight: list[list[float]], flat: list[int], values: list[float]) -> list[list[float]]:
    """``weight`` with ``values`` added at the row-major positions ``flat`` (Super-Tuning's sparse update)."""
    columns = len(weight[0])
    out = [row[:] for row in weight]
    for place, value in zip(flat, values, strict=True):
        out[place // columns][place % columns] += value
    return out


def low_rank_added(
    weight: list[list[float]], a: list[list[float]], b: list[list[float]], scale: float
) -> list[list[float]]:
    """``weight + scale * B A`` (the Supra hybrid's LoRA part, ``scale = alpha / r``)."""
    rank = len(a)
    return [
        [
            weight[i][j] + scale * sum(b[i][k] * a[k][j] for k in range(rank))
            for j in range(len(weight[0]))
        ]
        for i in range(len(weight))
    ]
