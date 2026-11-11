"""What the methods that train a few entries of a weight matrix share: putting those entries into a dense matrix.

Super-Tuning trains the entries of largest magnitude and WaveFT, without its inverse transform,
entries at random positions.  Both hold them as a vector of values beside a vector of positions into
the row-major flattened weight, and the update a layer applies is that vector laid into a zero
matrix.  That is all this module does.
"""

import lucid
from lucid._tensor.tensor import Tensor


def _matrix_from_entries(shape: tuple[int, int], flat: Tensor, values: Tensor) -> Tensor:
    """The matrix of ``shape`` holding ``values`` at the row-major positions ``flat`` and zero elsewhere.

    Position ``p`` is the entry at row ``p // columns`` and column ``p % columns``.  Values at a
    repeated position add, as the reference's ``scatter_add`` does.  The result is differentiable in
    ``values`` and nothing is read back from the device, so a compiled forward that calls it stays
    compiled.

    Parameters
    ----------
    shape : tuple of int
        ``(rows, columns)`` of the matrix.
    flat : Tensor
        Integer positions ``(K,)``, any integer dtype; moved to the device of ``values``.
    values : Tensor
        ``(K,)``; the matrix takes its dtype and device.

    Returns
    -------
    Tensor
        ``shape``, in the dtype of ``values``.
    """
    dense = lucid.zeros(shape[0] * shape[1], device=values.device, dtype=values.dtype)
    where = flat.to(device=values.device, dtype=lucid.int64)
    return dense.index_put((where,), values, accumulate=True).reshape(shape)
