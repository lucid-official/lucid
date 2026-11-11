"""Wavelet filters and the one-level 2-D transform, from their definitions, with plain loops and no adapter code.

The filters are checked against the equations that define them (``test_waveft_filters``), and the
inverse transform WaveFT builds its update with is checked against the sums written out here: a
coefficient at ``(i, j)`` of a band puts a copy of that band's 2-D filter at ``(2i, 2j)`` of the full
result, and the result is cut by the filters' reach at each end.  Nothing here calls a Lucid op.
"""

import cmath
import math
from math import comb

Matrix = list[list[float]]


def orthonormality_gap(h: tuple[float, ...]) -> float:
    """The largest ``|sum_k h_k h_{k+2m} - delta_m0|`` over the double shifts ``m``."""
    taps = len(h)
    return max(
        abs(sum(h[k] * h[k + 2 * m] for k in range(taps - 2 * m)) - (1.0 if m == 0 else 0.0))
        for m in range(taps // 2)
    )


def wavelet_moment(h: tuple[float, ...], order: int) -> float:
    """``sum_k (-1)^k k^order h_k``: zero for the orders below the wavelet's vanishing moments."""
    return sum((-1) ** k * k**order * h[k] for k in range(len(h)))


def scaling_moment(h: tuple[float, ...], order: int, centre: int) -> float:
    """``sum_k (k - centre)^order h_k``: zero for a coiflet's orders ``1 .. 2N - 1`` about ``centre = 2N``."""
    return sum((k - centre) ** order * h[k] for k in range(len(h)))


def response_energy(h: tuple[float, ...], omega: float) -> float:
    """``|H(e^{i omega})|^2`` of the filter ``h``."""
    return abs(sum(c * cmath.exp(-1j * omega * k) for k, c in enumerate(h))) ** 2


def daubechies_energy(n: int, omega: float) -> float:
    """``2 cos^{2n}(omega/2) sum_{k<n} C(n-1+k, k) sin^{2k}(omega/2)``: the squared response every ``n``-moment orthogonal filter of length ``2n`` has."""
    s = math.sin(omega / 2) ** 2
    return 2 * math.cos(omega / 2) ** (2 * n) * sum(comb(n - 1 + k, k) * s**k for k in range(n))


def phase_deviation(h: tuple[float, ...]) -> float:
    """The largest distance of the unwrapped phase of ``H(e^{i omega})`` on ``(0, pi)`` from the best straight line through the origin."""
    count = 400
    omegas = [math.pi * (k + 0.5) / count for k in range(count)]
    phases: list[float] = []
    for omega in omegas:
        value = sum(c * cmath.exp(-1j * omega * k) for k, c in enumerate(h))
        phase = cmath.phase(value)
        if phases:
            while phase - phases[-1] > math.pi:
                phase -= 2 * math.pi
            while phase - phases[-1] < -math.pi:
                phase += 2 * math.pi
        phases.append(phase)
    slope = sum(w * p for w, p in zip(omegas, phases, strict=True)) / sum(w * w for w in omegas)
    return max(abs(p - slope * w) for w, p in zip(omegas, phases, strict=True))


def front_energy(h: tuple[float, ...], count: int) -> float:
    """The energy of the first ``count`` taps."""
    return sum(c * c for c in h[:count])


def _analysis_1d(signal: list[float], filt: tuple[float, ...]) -> list[float]:
    """The full convolution of ``signal`` with ``filt``, every second sample from the second on (length ``(n + L - 2) / 2`` for even ``n`` and ``L``)."""
    taps = len(filt)
    full = [
        sum(signal[i - k] * filt[k] for k in range(taps) if 0 <= i - k < len(signal))
        for i in range(len(signal) + taps - 1)
    ]
    return full[1::2]


def _along_columns(matrix: Matrix, filt: tuple[float, ...]) -> Matrix:
    return [_analysis_1d(row, filt) for row in matrix]


def _along_rows(matrix: Matrix, filt: tuple[float, ...]) -> Matrix:
    columns = [_analysis_1d([row[j] for row in matrix], filt) for j in range(len(matrix[0]))]
    return [[column[i] for column in columns] for i in range(len(columns[0]))]


def analysis_2d(
    matrix: Matrix, dec_lo: tuple[float, ...], dec_hi: tuple[float, ...]
) -> tuple[Matrix, Matrix, Matrix, Matrix]:
    """The bands ``(cA, cH, cV, cD)`` of one level: ``cH`` is high-pass along the rows' index and low-pass along the columns'."""
    low_cols, high_cols = _along_columns(matrix, dec_lo), _along_columns(matrix, dec_hi)
    return (
        _along_rows(low_cols, dec_lo),
        _along_rows(low_cols, dec_hi),
        _along_rows(high_cols, dec_lo),
        _along_rows(high_cols, dec_hi),
    )


def synthesis_2d(
    bands: tuple[Matrix, Matrix, Matrix, Matrix],
    rec_lo: tuple[float, ...],
    rec_hi: tuple[float, ...],
) -> Matrix:
    """The inverse of ``analysis_2d``: each coefficient puts its band's ``L x L`` filter at twice its position, and the ends are cut by ``L - 2``."""
    taps = len(rec_lo)
    filters = [
        [[a * b for b in cols] for a in rows]
        for rows, cols in ((rec_lo, rec_lo), (rec_hi, rec_lo), (rec_lo, rec_hi), (rec_hi, rec_hi))
    ]
    height, width = len(bands[0]), len(bands[0][0])
    out_h, out_w = 2 * height - taps + 2, 2 * width - taps + 2
    out = [[0.0] * out_w for _ in range(out_h)]
    for band, filt in zip(bands, filters, strict=True):
        for i in range(height):
            for j in range(width):
                for r in range(out_h):
                    a = r + taps - 2 - 2 * i
                    if not 0 <= a < taps:
                        continue
                    for c in range(out_w):
                        b = c + taps - 2 - 2 * j
                        if 0 <= b < taps:
                            out[r][c] += band[i][j] * filt[a][b]
    return out
