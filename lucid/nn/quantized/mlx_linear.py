"""``QuantizedLinearMLX`` — real int4/int8 GEMM Linear (Metal only).

Unlike :class:`~lucid.nn.quantized.Linear` (which dequantizes to float and runs
a float matmul), this layer stores the weight in MLX's group-wise packed format
and runs the genuine low-precision kernel (``quantized_matmul``) — the actual
compute + memory win.  It is Metal-only (the kernel is GPU-only) and is built
from a float ``Linear`` via :meth:`from_float`, choosing ``bits`` (4 or 8).
"""

from typing import TYPE_CHECKING, cast, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid.quantization._qgemm import quantize, quantized_matmul

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


class QuantizedLinearMLX(nn.Module):
    r"""Weight-only int4/int8 linear backed by a real Metal low-precision GEMM.

    The *compute-and-memory* quantized linear, and the layer
    :func:`lucid.quantization.convert` routes the Linear family to when the MLX
    backend is active. Where :class:`~lucid.nn.quantized.Linear` dequantizes its
    int8 weight back to float and runs an ordinary float matmul (a memory-only
    win), ``QuantizedLinearMLX`` stores the weight in MLX's group-wise packed
    format and runs the genuine low-precision kernel — the packed weight is
    dequantized *inside* the GEMM, in one pass, so a full float weight is never
    materialized. Build it from a float :class:`~lucid.nn.Linear` via
    :meth:`from_float`, choosing ``bits`` (4 or 8).

    **Group-wise packed representation.** The float weight ``(out_features,
    in_features)`` is split, along the input dim, into contiguous groups of
    ``group_size`` elements; every ``(output-channel, group)`` block gets its own
    affine ``scale`` :math:`s` and ``bias`` :math:`\beta` (an offset, distinct from
    the layer's additive bias term), and the codes are bit-packed into a ``uint32``
    tensor (tagged ``I32``). Finer groups track a channel's local dynamic range
    more tightly than one per-row scale, at the cost of more metadata. This is the
    packing MLX's ``quantized_matmul`` consumes directly.

    Encoding happens once (at :meth:`from_float`); the packed decode + matmul run
    fused on every forward:

    .. math::

        q_{o,i} = \operatorname{clamp}\!\Bigl(
            \operatorname{round}\!\bigl((w_{o,i} - \beta_{o,g}) / s_{o,g}\bigr),\
            0,\ 2^{b}-1\Bigr),
        \qquad
        \hat{w}_{o,i} = s_{o,g}\, q_{o,i} + \beta_{o,g}

    .. math::

        y = x\, \hat{w}^{\top} + \mathbf{b},
        \qquad
        g = \bigl\lfloor i / \texttt{group\_size} \bigr\rfloor

    where :math:`o` indexes ``out_features``, :math:`i` the input dim, :math:`b`
    the bit-width (grid ``0 .. 2^b - 1``), and :math:`g` maps input index
    :math:`i` to its group. A ReLU is optionally fused after ``+ b``. In the
    kernel, :math:`\hat{w}` is never fully realized — the packed codes and the
    per-group :math:`(s, \beta)` are streamed into the matmul directly.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, optional
        Whether a learned **float** bias term is added after the GEMM. Defaults
        to ``True``.
    bits : int, optional
        Weight bit-width — ``4`` (aggressive memory) or ``8``. Defaults to ``8``.
    group_size : int, optional
        Number of input-dim elements sharing one ``scale`` / ``bias`` block in
        the packed layout. Must divide ``in_features``. Smaller groups are tighter
        but carry more metadata. Defaults to ``64``.
    relu : bool, optional
        If ``True``, a ReLU is fused after the GEMM (+ bias). Defaults to
        ``False``.

    Attributes
    ----------
    packed_weight : Tensor
        The bit-packed codes, shape ``(out_features, in_features·bits/32)``,
        ``int32`` (I32-tagged uint32) — ``bits/32`` codes packed per word.
    scales : Tensor
        Per-``(output-channel, group)`` affine scale, shape
        ``(out_features, in_features/group_size)``.
    biases : Tensor
        Per-``(output-channel, group)`` affine offset, shape
        ``(out_features, in_features/group_size)``.
    bias : Tensor or None
        The float bias of shape ``(out_features,)``, or ``None``.

    Notes
    -----
    - The kernel is **Metal-only** but the layer is *device-transparent*: a
      CPU-carried activation is moved onto the GPU for the GEMM and the result is
      moved **back to the input's device**, so this layer accelerates inside an
      otherwise-CPU model (or residual skip branch) without forcing the whole
      model onto Metal.
    - Build via :meth:`from_float` (which quantizes on Metal), **not** by
      constructing directly: a fresh instance has an all-zero ``packed_weight`` and
      zeroed ``scales`` / ``biases``, so it returns zeros until ``from_float`` /
      ``load_state_dict`` populates the buffers.
    - The ``state_dict`` carries the packed ``int32`` weight + per-group ``scales``
      / ``biases`` + the float bias — never a float weight. The weight payload
      shrinks ~``3.55x`` at ``bits=8`` and ~``6.40x`` at ``bits=4`` (the packed
      codes plus the small per-group metadata).
    - This is the layer that wins on **compute**, not just memory: it skips the
      dequantize-to-float hop entirely. The speed-up is **bandwidth-bound** — it is
      largest in the memory-bound decode / generation regime (measured ~``3.15x``
      at ``M = 1``) and fades toward parity (~``0.9-1x``) in compute-bound
      (large-``M``) training GEMMs, where the matmul, not weight bandwidth,
      dominates.
    - The real int4/int8 kernels are an optional C++ submodule; gate on
      :func:`lucid.quantization._qgemm.is_available` before relying on this fast
      path. :func:`lucid.quantization.convert` routes the Linear family here only
      when the MLX backend is active.

    Examples
    --------
    >>> import lucid, lucid.nn as nn
    >>> from lucid.nn.quantized import QuantizedLinearMLX
    >>> lin = nn.Linear(512, 256)
    >>> qlin = QuantizedLinearMLX.from_float(lin, bits=8)  # quantize on Metal
    >>> x = lucid.randn(4, 512)
    >>> y = qlin(x)          # GEMM runs on Metal, y returns on x's device
    >>> y.shape
    (4, 256)

    The layer is device-transparent — a CPU input is round-tripped through Metal
    for the GEMM and the result comes back on the input's device, while a Metal
    input stays on the GPU throughout:

    >>> qlin(x).is_metal                       # CPU in -> CPU out
    False
    >>> qlin(x.to("metal")).is_metal           # Metal in -> Metal out
    True

    Constructing directly (instead of via :meth:`from_float`) yields a zeroed,
    no-op layer — the packed weight is all zero until the buffers are filled:

    >>> broken = QuantizedLinearMLX(512, 256, bits=8)
    >>> bool((broken.packed_weight == 0).all().item())
    True

    See Also
    --------
    lucid.nn.quantized.Linear : Sidecar int8 Linear (memory win, float compute).
    lucid.quantization.convert : Routes the Linear family here on the MLX backend.
    lucid.quantization._qgemm.is_available : Whether the Metal kernels are built.
    """

    packed_weight: Tensor
    scales: Tensor
    biases: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        bits: int = 8,
        group_size: int = 64,
        relu: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.relu = relu
        # Buffers sized to MLX's group-wise packed layout so a fresh module can
        # ``load_state_dict`` (strict) without a shape mismatch: the packed
        # weight is ``(out, in·bits/32)`` uint32 (tagged I32) and there is one
        # scale + bias per ``group_size`` block along the input dim.
        packed_cols = in_features * bits // 32
        num_groups = in_features // group_size
        self.register_buffer(
            "packed_weight", lucid.zeros((out_features, packed_cols), dtype=lucid.int32)
        )
        self.register_buffer("scales", lucid.zeros((out_features, num_groups)))
        self.register_buffer("biases", lucid.zeros((out_features, num_groups)))
        if bias:
            self.register_buffer("bias", lucid.zeros(out_features))
        else:
            self.bias = None

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Run the MLX low-precision GEMM ``x @ packed_wᵀ`` (+ bias, +ReLU).

        The kernel is Metal-only, so a CPU-carried activation is moved onto the
        GPU for the GEMM and the result is moved **back to the input's device**.
        That keeps this layer device-transparent — it accelerates inside an
        otherwise-CPU model without forcing the whole model (or residual skip
        branches) onto Metal.
        """
        on_metal = x.is_metal
        if not on_metal:
            x = x.to("metal")
        y = quantized_matmul(
            x,
            self.packed_weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        if self.bias is not None:
            y = y + self.bias
        if self.relu:
            y = F.relu(y)
        return y if on_metal else y.to("cpu")

    @classmethod
    def from_float(
        cls, mod: nn.Module, bits: int = 8, group_size: int = 64, relu: bool = False
    ) -> QuantizedLinearMLX:
        """Quantize a float ``Linear``'s weight into MLX packed form (on Metal)."""
        lin = cast("nn.Linear", mod)
        m = cls(
            lin.in_features,
            lin.out_features,
            bias=lin.bias is not None,
            bits=bits,
            group_size=group_size,
            relu=relu,
        )
        weight = lin.weight.to("metal")  # (out, in)
        packed, scales, biases = quantize(weight, group_size=group_size, bits=bits)
        m.register_buffer("packed_weight", packed)
        m.register_buffer("scales", scales)
        m.register_buffer("biases", biases)
        if lin.bias is not None:
            m.register_buffer("bias", lin.bias.detach().to("metal"))
        return m

    @override
    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}, relu={self.relu}, "
            f"backend=mlx"
        )
