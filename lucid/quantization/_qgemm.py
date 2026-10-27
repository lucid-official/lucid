"""Python surface over the engine's MLX low-precision GEMM ops (GPU only).

Thin ``_wrap`` / ``_unwrap`` adapters around ``_C_engine.quantized.*`` (built
in Phase 6, E1).  These are the *real* int4/int8 kernels — MLX group-wise
affine quantize + quantized_matmul — as opposed to the dequantize-to-float
path.  Available only when the engine was compiled with the quantized ops
(``is_available()``) and only for Metal tensors.
"""

from typing import TYPE_CHECKING

from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

# ``_C_engine.quantized`` is an optional C++ submodule — present only when the
# engine was built with the E1 ops.  It is declared in ``engine.pyi`` (so the
# calls type-check), but accessed inside the functions (not at import) so this
# module stays importable against an engine that lacks it.


def is_available() -> bool:
    """Report whether the engine exposes the MLX quantized-GEMM ops (Metal only).

    The genuine low-precision path in Lucid — :class:`~lucid.nn.quantized.QuantizedLinearMLX`
    and the weight-only routing that :func:`~lucid.quantization.convert` performs when
    the MLX backend is active — is backed by MLX's group-wise ``quantized_matmul``
    kernel, exposed as an **optional** C++ submodule ``_C_engine.quantized``. That
    submodule is only compiled in when the engine build includes the quantized ops, so
    every caller of the fast path must gate on this predicate first and fall back to the
    dequantize-to-float reference path when it returns ``False``. It is a pure,
    side-effect-free capability probe — cheap enough to call on every dispatch.

    Returns
    -------
    bool
        ``True`` if the engine was built with the quantized submodule (and the real
        int4/int8 GEMM is therefore usable on Metal tensors), ``False`` otherwise.

    Notes
    -----
    - Availability is a **build-time** property of the engine binary, not a runtime
      toggle — it does not depend on the current device. Even when it returns ``True``
      the kernels still require the operand tensors to live on a Metal device.
    - A ``False`` result is not an error: the quantization subsystem stays fully
      functional through the dequantize-to-float path, which is also the exact-numerics
      reference (``backends.quantized.engine = "reference"``). Only the *compute*
      speed-up is unavailable.

    Examples
    --------
    >>> import lucid
    >>> from lucid.quantization import _qgemm
    >>> w = lucid.randn(64, 128, device="metal")
    >>> if _qgemm.is_available():
    ...     packed, scales, biases = _qgemm.quantize(w)      # real fast path
    ... else:
    ...     w_ref = w                                        # dequantize-to-float fallback

    See Also
    --------
    quantize : Group-wise pack a weight (requires this to be ``True``).
    quantized_matmul : The fused low-precision GEMM this predicate guards.
    lucid.nn.quantized.QuantizedLinearMLX : The layer that consumes these kernels.
    """
    return hasattr(_C_engine, "quantized")


def quantize(w: Tensor, group_size: int = 64, bits: int = 8) -> tuple[Tensor, Tensor, Tensor]:
    r"""Group-wise quantize a weight to MLX's packed int4/int8 form (Metal only).

    The encode half of the genuine low-precision path. Whereas the sidecar
    :class:`~lucid.nn.quantized.Linear` quantizes per *output channel* (one scale per
    row) for a dequantize-to-float matmul, this routine uses MLX's **group-wise affine**
    scheme tailored to the fused kernel: the quantized axis is sliced into contiguous
    blocks of ``group_size`` elements, each block gets its own ``(scale, bias)``, and the
    integer codes are bit-packed into a ``uint32`` tensor. The three returned tensors are
    exactly the operands :func:`quantized_matmul` consumes (and :func:`dequantize`
    inverts). This is called once, at weight-load time, by
    :meth:`lucid.nn.quantized.QuantizedLinearMLX.from_float`.

    For each group :math:`g` with element range :math:`[\min_g, \max_g]`:

    .. math::

        s_g = \frac{\max_g - \min_g}{2^{\text{bits}} - 1},
        \qquad b_g = \min_g,
        \qquad
        c_i = \operatorname{clip}\!\Bigl(
            \operatorname{round}\!\bigl((w_i - b_g)/s_g\bigr),\ 0,\ 2^{\text{bits}}-1
        \Bigr)

    so each code :math:`c_i` is an unsigned ``bits``-wide integer; :math:`2^{\text{bits}}`
    consecutive codes are then packed into one ``uint32`` word (8 codes for ``bits=4``,
    4 for ``bits=8``).

    Parameters
    ----------
    w : Tensor
        Float weight residing on a Metal device, typically ``(out_features, in_features)``.
    group_size : int, default 64
        Number of elements along the quantized axis that share one ``(scale, bias)``.
        Smaller groups track local range more tightly (less error) at the cost of more
        scale/bias metadata; ``in_features`` must be divisible by this.
    bits : int, default 8
        Bit-width of the packed codes; either ``4`` (int4, ~6.4x smaller) or ``8``
        (int8, ~3.55x smaller, higher accuracy).

    Returns
    -------
    tuple of Tensor
        ``(packed_weight, scales, biases)`` — the bit-packed ``uint32`` (I32-tagged)
        weight of shape ``(out, in·bits/32)``, and the per-group ``scales`` / ``biases``
        of shape ``(out, in/group_size)``. All three live on the same Metal device.

    Notes
    -----
    - **Metal-only.** Requires :func:`is_available` to be ``True`` and ``w`` on a Metal
      device; there is no CPU implementation (the CPU stream uses the reference dequant
      path per the Accelerate/MLX backend split).
    - **Group-wise, not per-channel.** The grouping axis and block size are dictated by
      the MLX kernel, so these qparams are *not* interchangeable with the per-channel
      ``(weight_scale, weight_zero_point)`` of the sidecar :class:`~lucid.nn.quantized.Linear`.
    - **Divisibility.** ``group_size`` must divide the quantized-axis length and be one of
      the kernel-supported sizes (32 / 64 / 128); otherwise the caller must fall back to
      the dequantize-to-float path.

    Examples
    --------
    >>> import lucid
    >>> from lucid.quantization import _qgemm
    >>> w = lucid.randn(256, 512).to("metal")
    >>> packed, scales, biases = _qgemm.quantize(w, group_size=64, bits=8)
    >>> packed.shape, scales.shape        # (out, in*bits/32), (out, in/group_size)
    ((256, 128), (256, 8))

    int4 packs twice as densely (8 codes per uint32):

    >>> packed4, _, _ = _qgemm.quantize(w, group_size=64, bits=4)
    >>> packed4.shape
    (256, 64)

    See Also
    --------
    quantized_matmul : Contracts a float activation against this packed weight.
    dequantize : Reconstructs the (approximate) float weight from this form.
    lucid.nn.quantized.QuantizedLinearMLX : The layer built on top of these ops.
    """
    wq, scales, biases = _C_engine.quantized.quantize(_unwrap(w), group_size, bits)
    return _wrap(wq), _wrap(scales), _wrap(biases)


def dequantize(
    wq: Tensor, scales: Tensor, biases: Tensor, group_size: int = 64, bits: int = 8
) -> Tensor:
    r"""Reconstruct a float weight from its packed group-wise form (Metal only).

    The exact inverse of :func:`quantize`'s packing: every ``bits``-wide code is
    unpacked from its ``uint32`` word and mapped back to float through its group's
    affine ``(scale, bias)``. Because quantization is lossy (the round step discards
    sub-step detail), the result is an *approximation* of the original weight, not a
    bit-exact copy — the reconstruction error per element is bounded by half a step,
    :math:`|w_i - \hat{w}_i| \le s_g / 2`.

    .. math::

        \hat{w}_i = c_i\, s_g + b_g

    where :math:`c_i` is the unpacked code, :math:`s_g` and :math:`b_g` the scale and
    bias of the group :math:`i` falls in. This materializes the full float weight, so it
    is used for inspection or as the reference path when the fused
    :func:`quantized_matmul` kernel is unavailable — the fused kernel folds this same
    decode *inside* the GEMM and never forms the float weight.

    Parameters
    ----------
    wq : Tensor
        Bit-packed ``uint32`` (I32-tagged) weight produced by :func:`quantize`.
    scales : Tensor
        Per-group scales returned by :func:`quantize`.
    biases : Tensor
        Per-group biases returned by :func:`quantize`.
    group_size : int, default 64
        Elements per group — **must** match the value passed to :func:`quantize`, or the
        codes are decoded against the wrong ``(scale, bias)`` and the result is garbage.
    bits : int, default 8
        Bit-width of the packed codes; either ``4`` or ``8`` — must match :func:`quantize`.

    Returns
    -------
    Tensor
        The reconstructed float weight, same shape and Metal device as the original.

    Notes
    -----
    - **Round-trip is lossy.** ``dequantize(quantize(w))`` approximates ``w`` to within a
      half-step per element; a larger ``group_size`` or fewer ``bits`` widens the error.
    - **Metal-only**, like the rest of this module; requires :func:`is_available`.
    - For the compute path you almost never call this directly — prefer
      :func:`quantized_matmul`, which fuses decode + matmul and skips materializing
      :math:`\hat{w}` entirely.

    Examples
    --------
    >>> import lucid
    >>> from lucid.quantization import _qgemm
    >>> w = lucid.randn(128, 256).to("metal")
    >>> packed, scales, biases = _qgemm.quantize(w, group_size=64, bits=8)
    >>> w_hat = _qgemm.dequantize(packed, scales, biases, group_size=64, bits=8)
    >>> w_hat.shape
    (128, 256)
    >>> bool((lucid.abs(w_hat - w).max() < 0.1).item())   # small round-trip error
    True

    See Also
    --------
    quantize : The forward pack this inverts.
    quantized_matmul : Fuses this decode into the GEMM (no float weight materialized).
    """
    out = _C_engine.quantized.dequantize(
        _unwrap(wq), _unwrap(scales), _unwrap(biases), group_size, bits
    )
    return _wrap(out)


def quantized_matmul(
    x: Tensor,
    wq: Tensor,
    scales: Tensor,
    biases: Tensor,
    transpose: bool = True,
    group_size: int = 64,
    bits: int = 8,
) -> Tensor:
    r"""Multiply a float activation by a packed low-precision weight (Metal only).

    The compute payoff of the whole low-precision path: MLX's **fused** group-wise
    GEMM. The packed weight is decoded *inside* the kernel, one tile at a time, and
    contracted with the float activation in a single pass — the full float weight is
    never materialized, so the layer reads ``bits/32`` as much weight memory from DRAM
    as a float matmul. Because a decode-shape matmul (batch/sequence ``M`` small) is
    **bandwidth-bound**, cutting weight traffic translates almost directly into speed:
    ~``3.15x`` faster at ``M = 1`` on a fully-Metal model. The win narrows toward parity
    (~``0.9-1x``) for large compute-bound GEMMs, where arithmetic — not weight traffic —
    dominates. With ``transpose=True`` (the default) it evaluates

    .. math::

        y = x\, \hat{w}^{\top} + \text{(bias handled by the caller)},
        \qquad \hat{w}_{ij} = c_{ij}\, s_g + b_g

    where :math:`\hat{w}` is the group-wise-decoded weight and :math:`x` the float
    activation — matching the ``(out, in)`` layout a linear layer stores its weight in.

    Parameters
    ----------
    x : Tensor
        Float activation on a Metal device, shape ``(M, in_features)``.
    wq : Tensor
        Bit-packed ``uint32`` (I32-tagged) weight produced by :func:`quantize`.
    scales : Tensor
        Per-group scales returned by :func:`quantize`.
    biases : Tensor
        Per-group biases returned by :func:`quantize`.
    transpose : bool, default True
        If ``True`` contract against the transposed weight (``x @ ŵᵀ`` → ``(M, out)``),
        the layout in which a linear layer stores ``(out, in)`` weights; if ``False``
        contract against ``ŵ`` directly.
    group_size : int, default 64
        Elements per group — must match the value used to :func:`quantize` the weight.
    bits : int, default 8
        Bit-width of the packed codes; ``4`` or ``8`` — must match :func:`quantize`.

    Returns
    -------
    Tensor
        The float product on the same Metal device — shape ``(M, out_features)`` when
        ``transpose=True``. Any bias / activation is applied by the caller afterward.

    Notes
    -----
    - **Bandwidth-bound win.** The speed-up lands in the memory-bound decode / generation
      regime (small ``M``) and fades in compute-bound training GEMMs; this is why the
      Linear family is routed here only when the MLX backend is active.
    - **In-kernel decode.** Unlike :func:`dequantize` → float matmul, no float weight is
      formed, which is both the memory *and* the speed advantage.
    - **W8A16, not W8A8.** The activation ``x`` stays float — this is weight-only
      quantization. For exact reference numerics on any device, use the dequantize path
      (``backends.quantized.engine = "reference"``).
    - **Metal-only**; requires :func:`is_available` and all operands on a Metal device.

    Examples
    --------
    >>> import lucid
    >>> from lucid.quantization import _qgemm
    >>> w = lucid.randn(256, 512).to("metal")               # (out, in)
    >>> packed, scales, biases = _qgemm.quantize(w, group_size=64, bits=8)
    >>> x = lucid.randn(1, 512).to("metal")                 # decode shape M=1
    >>> y = _qgemm.quantized_matmul(x, packed, scales, biases, transpose=True)
    >>> y.shape
    (1, 256)

    See Also
    --------
    quantize : Produces the packed weight + qparams this consumes.
    dequantize : The un-fused decode (materializes the float weight).
    lucid.nn.quantized.QuantizedLinearMLX : Wraps this into a drop-in linear layer.
    """
    out = _C_engine.quantized.quantized_matmul(
        _unwrap(x),
        _unwrap(wq),
        _unwrap(scales),
        _unwrap(biases),
        transpose,
        group_size,
        bits,
    )
    return _wrap(out)
