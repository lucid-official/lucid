"""The adapted weights of the sparse methods, from the papers' formulas, with plain loops and no adapter code.

``test_sparse_bias_contract`` loads adapter files written by hand in the reference's layout and needs
what the loaded model must compute; these functions say it entry by entry.  The tiny MLP is
``Linear(6, 8) -> ReLU -> Linear(8, 4)``.
"""

from lucid.test.unit.peft._wavelet_forms import synthesis_2d


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


def _padded(size: int, reach: int) -> int:
    side = size + reach
    return side + side % 2


def waveft_delta(
    shape: tuple[int, int],
    rows: list[int],
    columns: list[int],
    spectrum: list[float],
    filters: tuple[tuple[float, ...], tuple[float, ...]],
    scaling: float,
    inverse: bool = True,
) -> list[list[float]]:
    """WaveFT's update: the coefficients at ``(rows, columns)`` of a zero matrix, inverse-transformed one level (``filters = (rec_lo, rec_hi)``) and scaled.

    The coefficient matrix is padded by the filters' reach so the transform returns the weight's
    shape, split into quadrants ``cA`` (top left), ``cH`` (top right), ``cV`` (bottom left) and
    ``cD`` (bottom right), and the result is cut to the weight's shape from its middle.  With
    ``inverse=False`` the coefficients are the entries of the update.
    """
    out, inn = shape
    if not inverse:
        direct = [[0.0] * inn for _ in range(out)]
        for r, c, v in zip(rows, columns, spectrum, strict=True):
            direct[r][c] = scaling * v
        return direct
    rec_lo, rec_hi = filters
    reach = len(rec_lo) - 2
    height, width = _padded(out, reach), _padded(inn, reach)
    top, left = (height - out) // 2, (width - inn) // 2
    dense = [[0.0] * width for _ in range(height)]
    for r, c, v in zip(rows, columns, spectrum, strict=True):
        dense[r + top][c + left] = v
    half_h, half_w = height // 2, width // 2
    bands = (
        [row[:half_w] for row in dense[:half_h]],
        [row[half_w:] for row in dense[:half_h]],
        [row[:half_w] for row in dense[half_h:]],
        [row[half_w:] for row in dense[half_h:]],
    )
    full = synthesis_2d(bands, rec_lo, rec_hi)
    first_r, first_c = (len(full) - out) // 2, (len(full[0]) - inn) // 2
    return [[scaling * full[first_r + i][first_c + j] for j in range(inn)] for i in range(out)]
