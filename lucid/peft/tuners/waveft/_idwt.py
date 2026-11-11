r"""The one-level inverse 2-D discrete wavelet transform WaveFT turns its coefficients into a weight update with.

A one-level 2-D transform splits a matrix into four quarter-size sub-bands: the approximation
:math:`cA` (low-pass along both axes) and the details :math:`cH` (high-pass along rows, low-pass
along columns), :math:`cV` (the reverse) and :math:`cD` (high-pass along both).  The inverse puts
each band back by *upsampling* it (zeros between its entries), *filtering* it with the matching
synthesis filter and adding the four results.  Because the 2-D filters are outer products of the 1-D
ones, the whole step is one transposed convolution with stride 2 over the four bands as channels, and
a kernel of four :math:`L \times L` filters

.. math:: \phi_{cA} = h \otimes h, \quad \phi_{cH} = g \otimes h, \quad \phi_{cV} = h \otimes g,
    \quad \phi_{cD} = g \otimes g,

with :math:`h` the synthesis low-pass and :math:`g` the synthesis high-pass filter of length
:math:`L`.  The full result is :math:`2m + L - 2` wide for a band of width :math:`m`; the
:math:`L - 2` samples at each end are the transform's boundary effect and are cut off, which leaves
:math:`2m - L + 2`.  Hence a matrix that is to come out :math:`n` wide is made from bands
:math:`(n + L - 2)/2` wide: ``WaveFTLayer`` pads its coefficient matrix to that size first.
"""

import functools

import lucid
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike
from lucid.peft.tuners.waveft._filters import _filter_bank


@functools.cache
def _synthesis_kernel(family: str, dtype: DTypeLike, device: DeviceLike) -> Tensor:
    """The ``(4, 1, L, L)`` kernel of the wavelet ``family``: the filters of ``cA, cH, cV, cD`` in that order, as a transposed convolution reads them."""
    bank = _filter_bank(family)
    lo = lucid.tensor(bank.rec_lo, dtype=dtype, device=device)
    hi = lucid.tensor(bank.rec_hi, dtype=dtype, device=device)
    pairs = ((lo, lo), (hi, lo), (lo, hi), (hi, hi))
    return lucid.stack([rows.unsqueeze(1) * columns for rows, columns in pairs]).unsqueeze(1)


def _idwt2(
    approximation: Tensor, horizontal: Tensor, vertical: Tensor, diagonal: Tensor, family: str
) -> Tensor:
    """The matrix whose one-level 2-D wavelet transform with ``family`` has the four given sub-bands.

    Differentiable in the sub-bands, and computed in their dtype.

    Parameters
    ----------
    approximation, horizontal, vertical, diagonal : Tensor
        The bands :math:`cA, cH, cV, cD`, all ``(m, n)`` and one dtype and device.
    family : str
        A wavelet of ``WAVELET_FAMILIES``.

    Returns
    -------
    Tensor
        ``(2m - L + 2, 2n - L + 2)`` for filters of length ``L``.

    Raises
    ------
    ValueError
        ``family`` is not a wavelet this package has filters for.
    """
    kernel = _synthesis_kernel(family, approximation.dtype, approximation.device)
    bands = lucid.stack([approximation, horizontal, vertical, diagonal]).unsqueeze(0)
    taps = int(kernel.shape[-1])
    full = F.conv_transpose2d(bands, kernel, stride=2, padding=taps - 2)
    return full[0, 0]
