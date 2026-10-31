r"""Block-circulant matrices: the weight update of C3A, and applying it with FFTs.

A kernel ``w`` of shape ``(m, n, b)`` stands for the ``(m b) x (n b)`` matrix whose
``(i, j)`` block is the ``b x b`` circulant :math:`C_{st} = w_{ij}[(t - s) \bmod b]`.
Applied to a vector cut into ``n`` blocks :math:`x_j`, block :math:`i` of the result is
:math:`\sum_j C_{ij} x_j`, a sum of circular convolutions
:math:`y_s = \sum_t x_t\, w[(t - s) \bmod b]`, which is
:math:`\mathrm{irfft}(\mathrm{rfft}(x)\,\mathrm{rfft}(\tilde w))` for the reversed kernel
:math:`\tilde w[u] = w[-u \bmod b]`.  Applying costs :math:`O(m n\, b \log b)` against the
dense matrix's :math:`O(m n\, b^2)`, and needs no ``(m b) x (n b)`` matrix.
"""

import lucid
import lucid.fft as fft
from lucid._tensor.tensor import Tensor


def _circulant_matrix(kernel: Tensor) -> Tensor:
    """The dense ``(m b, n b)`` matrix a ``(m, n, b)`` kernel stands for."""
    m, n, b = kernel.shape
    steps = lucid.arange(b, device=kernel.device)
    rolled = (steps.unsqueeze(0) - steps.unsqueeze(1)) % b  # entry (s, t) is (t - s) mod b
    blocks = kernel[:, :, rolled]  # (m, n, b, b)
    return blocks.permute(0, 2, 1, 3).reshape(m * b, n * b)


def _circulant_apply(x: Tensor, kernel: Tensor) -> Tensor:
    """``x @ _circulant_matrix(kernel).T`` for ``x`` of shape ``(..., n b)``, by FFTs."""
    m, n, b = kernel.shape
    blocks = x.reshape(*x.shape[:-1], n, b)
    reversed_kernel = kernel.flip(-1).roll(1, -1)
    spectrum = fft.rfft(blocks).unsqueeze(-3) * fft.rfft(reversed_kernel)  # (..., m, n, b/2+1)
    out = fft.irfft(spectrum.sum(-2), n=b)  # (..., m, b)
    return out.reshape(*x.shape[:-1], m * b)
