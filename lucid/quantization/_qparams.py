"""Quantization parameters and their derivation from an observed range.

:class:`QParams` bundles the four things a quantized tensor needs —
``scale``, ``zero_point``, the :class:`~lucid.quantization.QScheme`, and
the :class:`~lucid.quantization.QDtype` — plus the channel axis for
per-channel schemes.  :func:`calculate_qparams` turns an observed
``[min_val, max_val]`` range into the ``(scale, zero_point)`` pair, and
is the shared math behind every observer.
"""

from dataclasses import dataclass

import lucid
from lucid._tensor.tensor import Tensor
from lucid.quantization._qscheme import QDtype, QScheme

# A ``type`` alias, evaluated when read: under ``TYPE_CHECKING`` it was
# absent at run time, and every signature naming it raised ``NameError``.
type _RangeLike = Tensor | float


@dataclass(frozen=True)
class QParams:
    r"""Immutable bundle of everything that defines a quantized tensor's grid.

    A quantized tensor in Lucid's *sidecar* representation is an ordinary integer
    :class:`~lucid.Tensor` plus the metadata needed to interpret its codes as real
    numbers. :class:`QParams` is that metadata gathered into one frozen value object:
    the ``scale`` / ``zero_point`` produced by calibration, together with the
    :class:`~lucid.quantization.QScheme` (grid geometry) and
    :class:`~lucid.quantization.QDtype` (grid range) they were derived under, plus the
    channel axis for per-channel schemes.

    It is the return-shape of an observer's calibration: an observer records a range,
    calls :func:`calculate_qparams` to get ``(scale, zero_point)``, and packages the
    result — with the scheme / dtype it used — into a :class:`QParams` so downstream
    quantize / dequantize / fake-quant calls have a single self-describing object to
    read from.

    Attributes
    ----------
    scale : Tensor
        The quantization step size — how many real units one integer code spans. A
        scalar for per-tensor schemes; a length-``C`` vector for per-channel schemes.
    zero_point : Tensor
        The integer code that real 0 maps to, stored as an integer-valued float tensor.
        0 (or the grid midpoint) for symmetric schemes; solved for under affine
        schemes. Same shape as ``scale``.
    qscheme : QScheme
        The mapping family (per-tensor / per-channel × affine / symmetric) the
        parameters were derived under.
    qdtype : QDtype
        The target integer grid (bit width + ``[quant_min, quant_max]``) the codes
        clamp to.
    ch_axis : int or None, default None
        The channel axis for per-channel schemes; ``None`` for per-tensor.

    Notes
    -----
    - Frozen (``@dataclass(frozen=True)``): a :class:`QParams` is a value, safe to share
      and compare; re-calibration produces a new instance rather than mutating one.
    - ``scale`` and ``zero_point`` always share a shape, and it is that shape (scalar vs
      length-``C``) — not ``qscheme`` alone — that a kernel broadcasts against.
    - The pair is consumed by :func:`~lucid.quantization.quantize` /
      :func:`~lucid.quantization.dequantize` /
      :func:`~lucid.quantization.fake_quantize`, which broadcast ``scale`` /
      ``zero_point`` along ``ch_axis``.

    Examples
    --------
    >>> import lucid.quantization as Q
    >>> scale, zp = Q.calculate_qparams(-2.0, 6.0, Q.per_tensor_affine, Q.quint8)
    >>> p = Q.QParams(scale, zp, Q.per_tensor_affine, Q.quint8)
    >>> p.qscheme.is_per_channel, p.ch_axis
    (False, None)
    >>> round(float(p.scale.item()), 4)        # (6 - (-2)) / 255
    0.0314

    See Also
    --------
    calculate_qparams : Derives the ``scale`` / ``zero_point`` a :class:`QParams` holds.
    lucid.quantization.QScheme : The grid-geometry field.
    lucid.quantization.QDtype : The grid-range field.
    """

    scale: Tensor
    zero_point: Tensor
    qscheme: QScheme
    qdtype: QDtype
    ch_axis: int | None = None


def calculate_qparams(
    min_val: _RangeLike, max_val: _RangeLike, qscheme: QScheme, qdtype: QDtype, eps: float = 1e-8
) -> tuple[Tensor, Tensor]:
    r"""Derive ``(scale, zero_point)`` from an observed value range.

    The shared calibration math behind every observer. Given the ``[min_val, max_val]``
    range an observer recorded, it solves for the affine map
    :math:`q = \operatorname{round}(x / s) + z` that packs that range into the target
    grid ``[q_{\min}, q_{\max}]``. The observed range is first **extended to include zero**
    (``min <- min(min, 0)``, ``max <- max(max, 0)``) so that real 0 is always exactly
    representable — a property the padding / masking paths rely on.

    The scheme's symmetry then selects the formula:

    .. math::

        \textbf{affine:}\quad
        s = \frac{\max - \min}{q_{\max} - q_{\min}},\qquad
        z = \operatorname{clip}\!\bigl(q_{\min} - \operatorname{round}(\min / s),\;
            q_{\min},\; q_{\max}\bigr)

    .. math::

        \textbf{symmetric:}\quad
        s = \frac{\max(\lvert\min\rvert,\ \lvert\max\rvert)}{(q_{\max} - q_{\min}) / 2},
        \qquad
        z = \begin{cases}
            0 & \text{signed} \\
            \lfloor (q_{\max} + q_{\min} + 1)/2 \rfloor & \text{unsigned}
        \end{cases}

    ``scale`` is floored at ``eps`` in every branch so a degenerate (constant) range
    never yields a zero or infinite step. The whole computation is elementwise, so
    scalar ``min_val`` / ``max_val`` produce per-tensor (scalar) parameters and
    length-``C`` inputs produce per-channel (vector) parameters — the function does not
    itself branch on granularity, only on symmetry.

    Parameters
    ----------
    min_val, max_val : Tensor or float
        The observed minimum / maximum of the tensor being calibrated — per-tensor
        scalars, or length-``C`` vectors for a per-channel scheme. Plain Python floats
        are lifted to scalar tensors.
    qscheme : QScheme
        The mapping family; only its :attr:`~QScheme.is_symmetric` flag is consulted
        here (granularity is implied by the shape of ``min_val`` / ``max_val``).
    qdtype : QDtype
        The target grid, supplying ``quant_min`` / ``quant_max`` and ``signed``.
    eps : float, default 1e-8
        Lower floor on ``scale`` to avoid a division-by-zero step on a degenerate
        (constant-valued) range.

    Returns
    -------
    (Tensor, Tensor)
        The ``scale`` and ``zero_point`` tensors, matching the shape of the inputs
        (scalar for per-tensor, length-``C`` for per-channel). ``zero_point`` is
        integer-valued but carried as a float tensor.

    Notes
    -----
    - **Zero is always representable.** Extending the range to include 0 guarantees a
      code for real 0, which matters wherever padding / masking injects exact zeros.
    - **Symmetric ``zero_point``.** Signed grids pin it to 0 (so integer matmuls need no
      weight-offset correction); unsigned grids pin it to the grid midpoint.
    - **Granularity is shape-driven.** There is no per-channel branch — passing vectors
      in yields vectors out, so a single code path serves both granularities.
    - Mirrors the reference framework's calibration but stays entirely on Lucid tensor
      ops (no external numeric library).

    Examples
    --------
    >>> import lucid.quantization as Q
    >>> # affine quint8 over an asymmetric activation range:
    >>> s, z = Q.calculate_qparams(-2.0, 6.0, Q.per_tensor_affine, Q.quint8)
    >>> round(float(s.item()), 4), int(z.item())     # (6 - (-2)) / 255, then clip
    (0.0314, 64)
    >>> # symmetric qint8 weight — zero_point pinned to 0:
    >>> s, z = Q.calculate_qparams(-0.8, 0.5, Q.per_channel_symmetric, Q.qint8)
    >>> int(z.item())
    0
    >>> # degenerate (constant) range floors scale at eps, never 0:
    >>> s, _ = Q.calculate_qparams(3.0, 3.0, Q.per_tensor_affine, Q.qint8)
    >>> bool(s.item() > 0)
    True

    See Also
    --------
    QParams : Bundles the returned ``(scale, zero_point)`` with its scheme / dtype.
    lucid.quantization.QScheme : Selects affine vs symmetric here.
    lucid.quantization.QDtype : Supplies ``quant_min`` / ``quant_max`` / ``signed``.
    lucid.quantization.quantize : Applies the derived parameters to produce codes.
    """
    min_t: Tensor = min_val if isinstance(min_val, Tensor) else lucid.tensor(float(min_val))
    max_t: Tensor = max_val if isinstance(max_val, Tensor) else lucid.tensor(float(max_val))

    quant_min, quant_max = qdtype.quant_min, qdtype.quant_max
    # Extend the range so that zero is always representable.
    min_t = lucid.minimum(min_t, lucid.zeros_like(min_t))
    max_t = lucid.maximum(max_t, lucid.zeros_like(max_t))
    eps_t = lucid.full_like(max_t, eps)

    if qscheme.is_symmetric:
        max_abs = lucid.maximum(lucid.abs(min_t), lucid.abs(max_t))
        scale = lucid.maximum(max_abs / ((quant_max - quant_min) / 2.0), eps_t)
        if qdtype.signed:
            zero_point = lucid.zeros_like(scale)
        else:
            zero_point = lucid.full_like(scale, float((quant_max + quant_min + 1) // 2))
    else:
        scale = lucid.maximum((max_t - min_t) / (quant_max - quant_min), eps_t)
        zero_point = lucid.clip(quant_min - lucid.round(min_t / scale), quant_min, quant_max)
    return scale, zero_point
