"""Core quantization math: ``quantize`` / ``dequantize`` / ``fake_quantize``.

These three primitives are the arithmetic core of the whole subsystem.
Everything else — observers, quantized modules, QAT — is expressed on
top of them.

``quantize`` maps reals to integer codes,

.. math::

    q = \\operatorname{clip}\\!\\big(\\operatorname{round}(x / s) + z,\\;
        q_\\min,\\; q_\\max\\big),

``dequantize`` inverts the affine map, :math:`\\hat{x} = (q - z)\\,s`,
and ``fake_quantize`` composes the two in the float domain
(:math:`\\hat{x} = \\text{dequantize}(\\text{quantize}(x))`) so a network
can *simulate* quantization while still training in float.

The rounding step has no useful derivative, so ``fake_quantize`` carries
a **straight-through estimator** (STE): gradient passes unchanged where
the code lands inside ``[quant_min, quant_max]`` and is zeroed where the
value saturates.  This is why it is implemented as a custom
:class:`~lucid.autograd.Function` rather than a plain composite.

All math runs through the public ``lucid.*`` op surface — no external
libraries (H4) — so the same code path is correct on both the Accelerate
(CPU) and MLX (GPU) streams.  Rounding and clipping are done in the
**float domain** and only the final cast lands in the integer storage
dtype, which sidesteps the missing CPU integer-clamp kernel.
"""

from typing import TYPE_CHECKING, override

import lucid
from lucid._tensor.tensor import Tensor
from lucid.autograd import Function, FunctionCtx

if TYPE_CHECKING:
    from lucid.quantization._qscheme import QDtype

# ``type`` aliases, evaluated when read: under ``TYPE_CHECKING`` they were
# absent at run time, and every signature naming them raised ``NameError``.
type _ScaleLike = Tensor | float
type _ZeroPointLike = Tensor | float | int


def _reshape_for_channel(param: Tensor, ch_axis: int, ndim: int) -> Tensor:
    """Reshape a 1-D per-channel ``param`` to broadcast along ``ch_axis``.

    A length-``C`` vector of scales/zero-points becomes a shape with ``C``
    at ``ch_axis`` and ``1`` everywhere else, so it broadcasts against a
    rank-``ndim`` activation/weight tensor.
    """
    shape: list[int] = [1] * ndim
    shape[ch_axis] = -1
    return param.reshape(shape)


def _grid_code(x: Tensor, s: _ScaleLike, z: _ZeroPointLike) -> Tensor:
    """Return the unclamped integer code ``round(x * (1 / s)) + z`` of ``x``.

    :func:`quantize` and :func:`fake_quantize` both read their codes from
    here, so a quantized model meets the exact rounding it trained against.
    The reciprocal is multiplied in, as the reference does, rather than
    ``x`` being divided: at a symmetric range's extremum ``x / s`` is
    exactly ``q_max + 0.5`` and rounds off the grid, so the clamp moved the
    value and the straight-through gradient there was dropped.
    """
    return lucid.round(x * (1.0 / s)) + z


def quantize(
    x: Tensor,
    scale: _ScaleLike,
    zero_point: _ZeroPointLike,
    qdtype: QDtype,
    ch_axis: int | None = None,
) -> Tensor:
    r"""Encode a real tensor into integer codes on an affine quantization grid.

    ``quantize`` is the forward half of the affine quantization map and the
    lowest-level primitive in the whole subsystem — observers, quantized
    modules, and QAT all bottom out here. Given a real tensor ``x`` and an
    already-chosen ``(scale, zero_point)`` pair, it snaps every value to the
    nearest lattice point of the integer grid described by ``qdtype`` and
    returns the integer *codes* that physically get stored. This is the step
    that actually shrinks memory: a ``float32`` weight becomes an int8 (or int4)
    code tensor roughly ``4x`` (or ``8x``) smaller, with the tiny ``scale`` /
    ``zero_point`` sidecar carried alongside.

    The map is affine — real values are divided by the step ``s``, offset by the
    integer ``zero_point`` ``z``, rounded to the nearest integer, and clamped
    into the representable ``[quant_min, quant_max]`` range before the storage
    cast:

    .. math::

        q = \operatorname{clip}\!\bigl(
            \operatorname{round}(x / s) + z,\ q_{\min},\ q_{\max}\bigr)

    Rounding and clipping happen in the **float domain**; only the final cast
    lands in ``qdtype.storage``. That ordering is deliberate — it keeps the path
    on the public ``lucid.*`` op surface (no external libraries, H4) so it is
    bit-for-bit identical on the Accelerate (CPU) and MLX (GPU) streams, and it
    sidesteps the missing CPU integer-clamp kernel. Pass ``ch_axis`` to quantize
    per channel: a length-``C`` ``scale`` / ``zero_point`` vector is reshaped to
    broadcast along that axis, giving each channel its own step — the
    accuracy-preserving default for conv / linear weights.

    Parameters
    ----------
    x : Tensor
        Real-valued input tensor to encode.
    scale : Tensor or float
        Quantization step ``s`` — the width of one code interval. A Python float
        (or scalar tensor) for per-tensor; a length-``C`` tensor when ``ch_axis``
        is given.
    zero_point : Tensor, float, or int
        Integer offset ``z`` mapping the real value ``0`` onto a code, so zero is
        represented exactly (matters for zero-padded tensors).
    qdtype : QDtype
        Target grid descriptor — supplies ``quant_min`` / ``quant_max`` and the
        ``storage`` dtype the codes are cast to (``qint8`` = ``[-128, 127]`` in
        ``int8``, ``quint8`` = ``[0, 255]``, ``qint4`` = ``[-8, 7]``).
    ch_axis : int, optional
        Channel axis for per-channel quantization; ``None`` (default) quantizes
        the whole tensor with a single scalar ``scale`` / ``zero_point``.

    Returns
    -------
    Tensor
        Integer codes with dtype ``getattr(lucid, qdtype.storage)`` and the same
        shape as ``x``.

    Notes
    -----
    - Pure encode with **no gradient** — it emits an integer tensor and is not
      meant to sit inside a differentiable path. For a differentiable
      simulate-quantization op (used by QAT), use :func:`fake_quantize`, which
      adds a straight-through estimator.
    - ``quint8`` currently maps to ``int16`` storage (the engine has no
      ``uint8``) and ``qint4`` packs into ``int8``; codes stay within the grid
      bounds regardless of the wider physical dtype.
    - Round-tripping ``quantize`` → :func:`dequantize` is lossy by the rounding
      step; the per-element reconstruction error is at most ``s / 2``.
    - Device-agnostic: there is no stream-specific branch here, so CPU and Metal
      produce identical codes.

    Examples
    --------
    >>> import lucid
    >>> import lucid.quantization as Q
    >>> x = lucid.tensor([-1.0, 0.0, 0.5, 2.0])
    >>> codes = Q.quantize(x, scale=0.05, zero_point=0, qdtype=Q.qint8)
    >>> codes.shape
    (4,)

    Values outside the grid saturate at ``quant_min`` / ``quant_max`` rather than
    wrapping — a common gotcha when ``scale`` is chosen too small:

    >>> hot = Q.quantize(lucid.tensor([100.0]), scale=0.05, zero_point=0, qdtype=Q.qint8)
    >>> int(hot.item())            # clamped to qint8's max, not 2000
    127

    Per-channel encode of a ``(out, in)`` weight — one step per output row:

    >>> w = lucid.randn(4, 8)
    >>> s = lucid.ones(4) * 0.02   # a scale for each output channel
    >>> zp = lucid.zeros(4)
    >>> Q.quantize(w, s, zp, Q.qint8, ch_axis=0).shape
    (4, 8)

    See Also
    --------
    lucid.quantization.dequantize : The inverse affine decode ``(q - z) * s``.
    lucid.quantization.fake_quantize : Differentiable quantize→dequantize round-trip.
    lucid.quantization.QDtype : Grid descriptor (bounds + storage dtype).
    """
    s: _ScaleLike = scale
    z: _ZeroPointLike = zero_point
    # qparams may be calibrated on a different device than the data — e.g. a
    # HistogramObserver derives its range from host floats, so its scale lands on
    # CPU while a Metal model feeds GPU activations.  The affine math runs on the
    # data's device, so pull a tensor scale / zero-point onto it (scalars are
    # device-agnostic and pass through).
    if isinstance(s, Tensor) and s.device != x.device:
        s = s.to(x.device)
    if isinstance(z, Tensor) and z.device != x.device:
        z = z.to(x.device)
    if ch_axis is not None and isinstance(s, Tensor):
        ndim = len(x.shape)
        s = _reshape_for_channel(s, ch_axis, ndim)
        if isinstance(z, Tensor):
            z = _reshape_for_channel(z, ch_axis, ndim)
    codes = lucid.clip(_grid_code(x, s, z), qdtype.quant_min, qdtype.quant_max)
    storage_dtype: lucid.dtype = getattr(lucid, qdtype.storage)
    return codes.to(storage_dtype)


def dequantize(
    q: Tensor, scale: _ScaleLike, zero_point: _ZeroPointLike, ch_axis: int | None = None
) -> Tensor:
    r"""Decode integer codes back to a ``float32`` approximation of the original.

    ``dequantize`` is the inverse of :func:`quantize` — the *read* side of the
    affine map. It undoes the offset and scaling to recover a floating-point
    tensor from the stored integer codes. It is the operation that lets a
    quantized weight participate in an ordinary float matmul: the sidecar-design
    quantized modules keep their weight as int8 codes, then call ``dequantize``
    on every forward to reconstruct a float weight before the GEMM. The result
    is not the exact original — it is the original rounded to the nearest grid
    point — but the error per element is bounded by ``s / 2``.

    The decode is the affine inverse:

    .. math::

        \hat{x} = (q - z)\, s

    where the subtraction removes the ``zero_point`` offset ``z`` and the
    multiply restores the physical step ``s``. As in :func:`quantize`, a
    per-channel ``scale`` / ``zero_point`` (given ``ch_axis``) is reshaped to
    broadcast along the channel axis, so each channel is decoded with its own
    step. The whole computation is a subtract-then-multiply on the public
    ``lucid.*`` op surface (H4), identical on the Accelerate and MLX streams.

    Parameters
    ----------
    q : Tensor
        Integer codes previously produced by :func:`quantize` (or an equivalent
        encoder). Cast up to ``float32`` internally before the affine decode.
    scale : Tensor or float
        The same quantization step ``s`` used to produce ``q``. A length-``C``
        tensor for per-channel; a scalar for per-tensor.
    zero_point : Tensor, float, or int
        The same zero-point offset ``z`` used to produce ``q``.
    ch_axis : int, optional
        Channel axis for per-channel decode; ``None`` (default) uses a single
        scalar ``scale`` / ``zero_point`` for the whole tensor.

    Returns
    -------
    Tensor
        A ``float32`` reconstruction of the original tensor, the same shape as
        ``q``.

    Notes
    -----
    - Lossy by construction — ``dequantize(quantize(x)) ≈ x`` only up to the
      grid resolution; values that saturated during encode are *not* recovered.
    - Must be called with the **same** ``scale`` / ``zero_point`` / ``ch_axis``
      used to encode; mismatched qparams silently produce wrong magnitudes.
    - Output is always ``float32`` regardless of the input storage dtype, and
      the op carries no straight-through gradient (it is a plain composite, not
      a custom autograd Function).

    Examples
    --------
    >>> import lucid
    >>> import lucid.quantization as Q
    >>> x = lucid.tensor([-1.0, 0.0, 0.5, 2.0])
    >>> q = Q.quantize(x, scale=0.05, zero_point=0, qdtype=Q.qint8)
    >>> x_hat = Q.dequantize(q, scale=0.05, zero_point=0)
    >>> bool((lucid.abs(x_hat - x) <= 0.05).all().item())   # within one step
    True

    Per-channel decode mirrors the encode axis exactly:

    >>> w = lucid.randn(4, 8)
    >>> s, zp = lucid.ones(4) * 0.02, lucid.zeros(4)
    >>> codes = Q.quantize(w, s, zp, Q.qint8, ch_axis=0)
    >>> Q.dequantize(codes, s, zp, ch_axis=0).shape
    (4, 8)

    See Also
    --------
    lucid.quantization.quantize : The forward affine encode ``round(x/s) + z``.
    lucid.quantization.fake_quantize : Fused encode→decode with an STE gradient.
    lucid.nn.quantized.Linear : Uses ``dequantize`` on its int8 weight each forward.
    """
    s: _ScaleLike = scale
    z: _ZeroPointLike = zero_point
    # Align tensor qparams onto the code tensor's device (see quantize()).
    if isinstance(s, Tensor) and s.device != q.device:
        s = s.to(q.device)
    if isinstance(z, Tensor) and z.device != q.device:
        z = z.to(q.device)
    if ch_axis is not None and isinstance(s, Tensor):
        ndim = len(q.shape)
        s = _reshape_for_channel(s, ch_axis, ndim)
        if isinstance(z, Tensor):
            z = _reshape_for_channel(z, ch_axis, ndim)
    return (q.to(lucid.float32) - z) * s


class _FakeQuantizeAffine(Function):
    """``dequantize(quantize(x))`` with a straight-through backward."""

    _once_differentiable = True  # second derivatives through it are unverified

    @override
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        x: Tensor,
        scale: _ScaleLike,
        zero_point: _ZeroPointLike,
        quant_min: int,
        quant_max: int,
        ch_axis: int | None,
    ) -> Tensor:
        """Fake-quantize ``x`` and cache the saturation mask for STE."""
        s: _ScaleLike = scale
        z: _ZeroPointLike = zero_point
        # Align tensor qparams onto the input's device (see quantize()).
        if isinstance(s, Tensor) and s.device != x.device:
            s = s.to(x.device)
        if isinstance(z, Tensor) and z.device != x.device:
            z = z.to(x.device)
        if ch_axis is not None and isinstance(s, Tensor):
            ndim = len(x.shape)
            s = _reshape_for_channel(s, ch_axis, ndim)
            if isinstance(z, Tensor):
                z = _reshape_for_channel(z, ch_axis, ndim)
        q_pre = _grid_code(x, s, z)
        q = lucid.clip(q_pre, quant_min, quant_max)
        dq = (q - z) * s
        # STE mask: 1 where the code is inside the grid, 0 where it saturates.
        in_lo = (q_pre >= quant_min).to(lucid.float32)
        in_hi = (q_pre <= quant_max).to(lucid.float32)
        ctx.save_for_backward(in_lo * in_hi)
        return dq

    @override
    @staticmethod
    def backward(ctx: FunctionCtx, grad_out: Tensor) -> Tensor:
        """Pass gradient through the non-saturated region only.

        ``x`` is the only positional (tensor) input to ``forward`` — the
        scale / zero-point / bounds are passed as keyword arguments so
        autograd does not track them — hence a single gradient is
        returned, matching the one registered input edge.
        """
        (mask,) = ctx.saved_tensors
        return grad_out * mask


def fake_quantize(
    x: Tensor,
    scale: _ScaleLike,
    zero_point: _ZeroPointLike,
    quant_min: int,
    quant_max: int,
    ch_axis: int | None = None,
) -> Tensor:
    r"""Simulate quantization in the float domain with a straight-through gradient.

    ``fake_quantize`` composes :func:`quantize` and :func:`dequantize` into a
    single float→float round-trip: it quantizes ``x`` to the integer grid and
    immediately decodes it back, returning a ``float32`` tensor that carries the
    *rounding error* quantization would introduce while staying differentiable.
    That is what makes quantization-aware training (QAT) possible — a network can
    *feel* the low-precision grid on every forward, and thus learn weights that
    are robust to it, while gradients continue to flow and the optimizer keeps
    working in float. It is the operation behind :class:`FakeQuantize`, every
    ``nn.qat`` layer, and the output re-quantization step of the sidecar
    quantized inference modules.

    The forward is the encode→decode round-trip:

    .. math::

        \hat{x} = \bigl(\operatorname{clip}(\operatorname{round}(x/s) + z,\
            q_{\min},\ q_{\max}) - z\bigr)\, s

    The round has no useful derivative, so the backward is the **straight-through
    estimator** (STE): the gradient passes unchanged where the pre-clip code
    lands inside the grid and is zeroed where the value saturates:

    .. math::

        \frac{\partial \hat{x}}{\partial x} =
        \begin{cases}
            1 & q_{\min} \le \operatorname{round}(x/s) + z \le q_{\max} \\
            0 & \text{otherwise}
        \end{cases}

    Because of that custom backward it is implemented as a
    :class:`~lucid.autograd.Function` rather than a plain composite; only ``x``
    is registered as a differentiable input (the qparams / bounds are passed as
    keyword arguments), so a single gradient edge flows back to ``x`` alone.

    Parameters
    ----------
    x : Tensor
        Real-valued input; may require grad (this is the QAT training path).
    scale : Tensor or float
        Quantization step ``s`` (typically produced by an observer / learnable
        in QAT). A length-``C`` tensor for per-channel.
    zero_point : Tensor, float, or int
        Integer zero-point offset ``z``.
    quant_min, quant_max : int
        Inclusive integer grid bounds (e.g. ``0, 255`` for ``quint8``, ``-128,
        127`` for ``qint8``); values rounding outside this range saturate.
    ch_axis : int, optional
        Channel axis for per-channel fake-quant; ``None`` (default) uses a
        single scalar ``scale`` / ``zero_point`` for the whole tensor.

    Returns
    -------
    Tensor
        A fake-quantized ``float32`` tensor, the same shape as ``x``, carrying
        the STE gradient described above.

    Notes
    -----
    - Output stays ``float32`` — nothing is physically stored as int here. Use
      :func:`quantize` when you actually want integer codes; use this when you
      want the *effect* of quantization inside a differentiable graph.
    - The STE saturation mask is computed from the **pre-clip** code, so it
      correctly zeroes the gradient for inputs that round to exactly the grid
      edge from outside.
    - At inference the round-trip makes a module's output *bit-equivalent* to a
      true integer kernel without needing one — the reason the sidecar
      :class:`~lucid.nn.quantized.Linear` fake-quantizes its output.
    - Runs on the public ``lucid.*`` op surface (H4), so the STE is identical on
      the Accelerate and MLX streams.

    Examples
    --------
    >>> import lucid
    >>> import lucid.quantization as Q
    >>> x = lucid.tensor([0.02, 0.31, 0.77], requires_grad=True)
    >>> y = Q.fake_quantize(x, scale=0.1, zero_point=0, quant_min=0, quant_max=255)
    >>> y.shape
    (3,)

    STE in action — the gradient is ``1`` in-range and ``0`` where the input
    saturates past ``quant_max``:

    >>> x = lucid.tensor([0.5, 99.0], requires_grad=True)   # 99.0 rounds past 255*0.1
    >>> Q.fake_quantize(x, scale=0.1, zero_point=0, quant_min=0, quant_max=255).sum().backward()
    >>> [float(g) for g in x.grad]
    [1.0, 0.0]

    See Also
    --------
    lucid.quantization.quantize : The (non-differentiable) integer encode.
    lucid.quantization.dequantize : The affine decode half of the round-trip.
    lucid.quantization.FakeQuantize : Observer-driven module wrapper of this op.
    lucid.quantization.prepare_qat : Inserts fake-quant layers for QAT.
    """
    # Only ``x`` is positional; the rest are keyword arguments so autograd
    # registers a single input edge (STE flows to ``x`` alone).
    return _FakeQuantizeAffine.apply(  # type: ignore[return-value]
        x,
        scale=scale,
        zero_point=zero_point,
        quant_min=quant_min,
        quant_max=quant_max,
        ch_axis=ch_axis,
    )
