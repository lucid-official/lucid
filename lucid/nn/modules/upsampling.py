"""
Upsampling and pixel-shuffle modules.
"""

from typing import override

from lucid._tensor.tensor import Tensor
from lucid.nn.module import Module
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.nn.functional.sampling import channel_shuffle, interpolate


class Upsample(Module):
    r"""Upsample an input tensor to a given spatial size or scale factor.

    ``Upsample`` resizes the spatial dimensions of an input tensor using one
    of several interpolation algorithms.  The batch and channel dimensions
    are never modified; only the trailing spatial axes are scaled.

    Exactly one of ``size`` or ``scale_factor`` must be provided.

    **Nearest-neighbour (mode='nearest'):**

    .. math::

        y[i] = x\!\left[\left\lfloor \frac{i}{\text{scale}} \right\rfloor\right]

    Each output element copies the value of the nearest input element.

    **Bilinear (mode='bilinear', 2-D only):**

    .. math::

        y[i, j] =
          (1 - t_h)(1 - t_w)\, x[i_0, j_0]
          + t_h(1 - t_w)\, x[i_1, j_0]
          + (1 - t_h) t_w\, x[i_0, j_1]
          + t_h t_w\, x[i_1, j_1]

    where :math:`i_0, i_1` are the two nearest row indices in the input and
    :math:`t_h` is the fractional offset between them (analogously for
    :math:`j_0, j_1, t_w`).

    Parameters
    ----------
    size : int or tuple[int, ...] or None, optional
        Output spatial size.  Pass a single ``int`` to set all spatial
        dimensions to the same value, or a tuple matching the number of
        spatial dimensions.  Mutually exclusive with ``scale_factor``.
    scale_factor : float or tuple[float, ...] or None, optional
        Multiplier for each spatial dimension.  Values ``> 1`` upsample;
        values ``< 1`` downsample.  Mutually exclusive with ``size``.
    mode : str, optional
        Interpolation algorithm.  One of:

        - ``'nearest'`` — no learnable parameters, fastest (default).
        - ``'linear'`` — 1-D linear interpolation (3-D input).
        - ``'bilinear'`` — 2-D bilinear interpolation (4-D input).
        - ``'bicubic'`` — 2-D bicubic interpolation (4-D input).
        - ``'trilinear'`` — 3-D trilinear interpolation (5-D input).
        - ``'nearest-exact'`` — nearest with half-pixel centres.
        - ``'area'`` — adaptive average pooling to the output size.
    align_corners : bool or None, optional
        When ``True``, the corner pixels of input and output are aligned,
        meaning the interpolation grid spans exactly from the first to the
        last pixel centre.  When ``False`` (or ``None``), the grid is
        scaled by the ratio of sizes, which may shift the grid by half a
        pixel relative to the corners.  Only accepted by the
        interpolating modes (``'linear'``, ``'bilinear'``, ``'bicubic'``,
        ``'trilinear'``).
    recompute_scale_factor : bool or None, optional
        When ``True``, sample positions step by ``in / out`` of the
        resolved output size instead of ``1 / scale_factor``; the two
        differ when ``in * scale_factor`` is not a whole number.  Not
        allowed with ``size``.

    Attributes
    ----------
    size : int or tuple[int, ...] or None
        Stored output size.
    scale_factor : float or tuple[float, ...] or None
        Stored scale factor.
    mode : str
        Stored interpolation mode.
    align_corners : bool or None
        Stored corner-alignment flag.
    recompute_scale_factor : bool or None
        Stored flag for how a scale factor maps sample positions.

    Shape
    -----
    - **Input:** :math:`(N, C, d_1, d_2, \dots, d_k)`.
    - **Output:** :math:`(N, C, d_1', d_2', \dots, d_k')` where each
      :math:`d_i'` is either given by ``size`` or
      :math:`d_i' = \lfloor d_i \cdot \text{scale\_factor} \rfloor`.

    Notes
    -----
    - The ``align_corners`` flag has a significant effect on output values
      for bilinear / bicubic / trilinear modes.  When ``True``, corner
      pixels map exactly onto each other; when ``False`` the mapping
      preserves the overall scale but shifts by 0.5 input pixels at the
      boundary.
    - For 4-D inputs with ``mode='nearest'`` this module is equivalent to
      ``UpsamplingNearest2d``; for ``mode='bilinear'`` with
      ``align_corners=True`` it is equivalent to ``UpsamplingBilinear2d``.
    - Delegates to ``nn.functional.interpolate``.

    Examples
    --------
    **2× nearest-neighbour upsampling of a feature map:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> up = nn.Upsample(scale_factor=2, mode="nearest")
    >>> x = lucid.zeros(1, 16, 14, 14)
    >>> up(x).shape
    (1, 16, 28, 28)

    **Fixed output size with bilinear interpolation:**

    >>> up = nn.Upsample(size=(256, 256), mode="bilinear", align_corners=False)
    >>> x = lucid.zeros(4, 3, 64, 64)
    >>> up(x).shape
    (4, 3, 256, 256)

    **Trilinear upsampling for 3-D volumetric data:**

    >>> up_3d = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
    >>> x = lucid.zeros(2, 8, 16, 16, 16)
    >>> up_3d(x).shape
    (2, 8, 32, 32, 32)
    """

    def __init__(
        self,
        size: int | tuple[int, ...] | None = None,
        scale_factor: float | tuple[float, ...] | None = None,
        mode: str = "nearest",
        align_corners: bool | None = None,
        recompute_scale_factor: bool | None = None,
    ) -> None:
        """Initialise the Upsample module. See the class docstring for parameter semantics."""
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Upsample the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)`.

        Returns
        -------
        Tensor
            Upsampled output tensor.
        """
        return interpolate(
            input,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
        )

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        parts = []
        if self.size is not None:
            parts.append(f"size={self.size}")
        if self.scale_factor is not None:
            parts.append(f"scale_factor={self.scale_factor}")
        parts.append(f"mode={self.mode!r}")
        return ", ".join(parts)


class UpsamplingNearest2d(Upsample):
    r"""Nearest-neighbour upsampling for 4-D tensors (N, C, H, W).

    A convenience wrapper around ``Upsample`` with ``mode='nearest'`` fixed.
    Accepts either a target ``size`` or a ``scale_factor``; the semantics
    are identical to ``Upsample``.

    Nearest-neighbour interpolation assigns each output pixel the value of
    the spatially nearest input pixel:

    .. math::

        y[n, c, i, j]
        = x\!\left[n,\, c,\,
            \left\lfloor \frac{i}{\text{scale}_H} \right\rfloor,\,
            \left\lfloor \frac{j}{\text{scale}_W} \right\rfloor
          \right]

    Parameters
    ----------
    size : int or tuple[int, int] or None, optional
        Target output spatial size ``(H_out, W_out)``.  Mutually exclusive
        with ``scale_factor``.
    scale_factor : float or tuple[float, float] or None, optional
        Spatial scale multiplier ``(s_H, s_W)``.  Mutually exclusive with
        ``size``.

    Shape
    -----
    - **Input:** :math:`(N, C, H, W)`.
    - **Output:** :math:`(N, C, H_{\text{out}}, W_{\text{out}})`.

    Notes
    -----
    - ``align_corners`` is not applicable for ``'nearest'`` mode and is
      always ``None``.
    - This class is marked as *deprecated* in some reference implementations
      but remains widely used in legacy codebases and super-resolution
      architectures.
    - Equivalent to ``Upsample(size=size, scale_factor=scale_factor, mode='nearest')``.

    Examples
    --------
    **Double the spatial resolution of a feature map:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> up = nn.UpsamplingNearest2d(scale_factor=2)
    >>> x = lucid.zeros(1, 32, 7, 7)
    >>> up(x).shape
    (1, 32, 14, 14)

    **Upsample to a fixed size:**

    >>> up = nn.UpsamplingNearest2d(size=(224, 224))
    >>> x = lucid.zeros(4, 3, 56, 56)
    >>> up(x).shape
    (4, 3, 224, 224)
    """

    def __init__(
        self,
        size: int | tuple[int, int] | None = None,
        scale_factor: float | tuple[float, float] | None = None,
    ) -> None:
        """Initialise the UpsamplingNearest2d module. See the class docstring for parameter semantics."""
        super().__init__(size=size, scale_factor=scale_factor, mode="nearest")


class UpsamplingBilinear2d(Upsample):
    r"""Bilinear upsampling for 4-D tensors (N, C, H, W) with aligned corners.

    A convenience wrapper around ``Upsample`` with ``mode='bilinear'`` and
    ``align_corners=True`` fixed.  The ``align_corners=True`` default matches
    the reference framework's ``UpsamplingBilinear2d`` — note that plain
    ``Upsample(mode='bilinear')`` defaults to ``align_corners=False``, which
    gives different output values near the image boundary.

    Bilinear interpolation performs 2-D linear interpolation between the four
    nearest input neighbours:

    .. math::

        y[n, c, i, j]
        = (1 - t_h)(1 - t_w)\, x[\dots, i_0, j_0]
        + t_h(1 - t_w)\, x[\dots, i_1, j_0]
        + (1 - t_h) t_w\, x[\dots, i_0, j_1]
        + t_h t_w\, x[\dots, i_1, j_1]

    where :math:`t_h, t_w \in [0, 1)` are the fractional offsets within the
    input grid cell (computed with corner-aligned coordinates when
    ``align_corners=True``).

    Parameters
    ----------
    size : int or tuple[int, int] or None, optional
        Target output spatial size ``(H_out, W_out)``.  Mutually exclusive
        with ``scale_factor``.
    scale_factor : float or tuple[float, float] or None, optional
        Spatial scale multiplier ``(s_H, s_W)``.  Mutually exclusive with
        ``size``.

    Shape
    -----
    - **Input:** :math:`(N, C, H, W)`.
    - **Output:** :math:`(N, C, H_{\text{out}}, W_{\text{out}})`.

    Notes
    -----
    - ``align_corners=True`` means that the corner pixels of the input and
      output grids coincide exactly; this is the legacy behaviour used in
      many pre-trained segmentation and super-resolution models.
    - When precise sub-pixel alignment matters (e.g. image restoration),
      prefer ``Upsample(mode='bilinear', align_corners=False)`` which avoids
      the half-pixel offset at image boundaries.

    Examples
    --------
    **4× bilinear upsampling with corner alignment:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> up = nn.UpsamplingBilinear2d(scale_factor=4)
    >>> x = lucid.zeros(1, 64, 8, 8)
    >>> up(x).shape
    (1, 64, 32, 32)

    **Upsample to a fixed output size:**

    >>> up = nn.UpsamplingBilinear2d(size=(512, 512))
    >>> x = lucid.zeros(2, 3, 128, 128)
    >>> up(x).shape
    (2, 3, 512, 512)
    """

    def __init__(
        self,
        size: int | tuple[int, int] | None = None,
        scale_factor: float | tuple[float, float] | None = None,
    ) -> None:
        """Initialise the UpsamplingBilinear2d module. See the class docstring for parameter semantics."""
        super().__init__(size=size, scale_factor=scale_factor, mode="bilinear", align_corners=True)


class PixelShuffle(Module):
    r"""Rearrange channel-stacked sub-pixel planes into a high-resolution output.

    ``PixelShuffle`` implements the sub-pixel convolution (ESPCN) operation:
    given an input with :math:`C \cdot r^2` channels and spatial size
    :math:`(H, W)`, it reshapes and transposes to produce an output with
    :math:`C` channels and spatial size :math:`(H \cdot r, W \cdot r)`.

    .. math::

        \text{output}[n,\, c,\, H \cdot s_h + i,\, W \cdot s_w + j]
        = \text{input}[n,\, c \cdot r^2 + s_h \cdot r + s_w,\, H,\, W]

    where :math:`s_h, s_w \in \{0, \dots, r-1\}` index the sub-pixel offset
    and :math:`c` indexes the output channel.

    Parameters
    ----------
    upscale_factor : int
        Spatial upscaling factor :math:`r`.  The number of input channels
        must be divisible by :math:`r^2`.

    Attributes
    ----------
    upscale_factor : int
        Stored value of the ``upscale_factor`` constructor argument.

    Shape
    -----
    - **Input:** :math:`(N, C \cdot r^2, H, W)`.
    - **Output:** :math:`(N, C, H \cdot r, W \cdot r)`.

    Notes
    -----
    - ``PixelShuffle`` is the core operation in the Efficient Sub-Pixel CNN
      (ESPCN) super-resolution architecture (Shi et al., 2016).  The idea is
      to learn the upsampling entirely through convolution on the
      low-resolution grid, then rearrange the :math:`r^2` feature channels
      into a single high-resolution plane, which is cheaper than computing
      features at high resolution.
    - Implemented as reshape → permute → reshape on the C++ engine; on
      the CPU the permute is a view, so the only copy is the final
      reshape's.
    - ``PixelUnshuffle`` is the exact inverse.

    Examples
    --------
    **2× super-resolution decoder block:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> r = 2
    >>> # Conv outputs C*r^2 channels at LR spatial size, PixelShuffle expands
    >>> block = nn.Sequential(
    ...     nn.Conv2d(64, 3 * r * r, kernel_size=3, padding=1),
    ...     nn.PixelShuffle(upscale_factor=r),
    ... )
    >>> x = lucid.zeros(1, 64, 56, 56)
    >>> block(x).shape
    (1, 3, 112, 112)

    **Verify channel / spatial trade-off:**

    >>> ps = nn.PixelShuffle(upscale_factor=3)
    >>> x = lucid.zeros(2, 36, 10, 10)   # 36 = 4 * 3^2
    >>> ps(x).shape
    (2, 4, 30, 30)
    """

    def __init__(self, upscale_factor: int) -> None:
        """Initialise the PixelShuffle module. See the class docstring for parameter semantics."""
        super().__init__()
        self.upscale_factor = upscale_factor

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Upsample the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)`.

        Returns
        -------
        Tensor
            Upsampled output tensor.
        """
        r = self.upscale_factor
        n, c_r2, h, w = input.shape
        c = c_r2 // (r * r)
        impl = _unwrap(input)
        t = _C_engine.reshape(impl, [n, c, r, r, h, w])
        t = _C_engine.permute(t, [0, 1, 4, 2, 5, 3])
        return _wrap(_C_engine.reshape(t, [n, c, h * r, w * r]))

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"upscale_factor={self.upscale_factor}"


class PixelUnshuffle(Module):
    r"""Reverse the sub-pixel rearrangement performed by ``PixelShuffle``.

    ``PixelUnshuffle`` is the exact inverse of ``PixelShuffle``: it takes a
    high-resolution input with :math:`C` channels and spatial size
    :math:`(H \cdot r, W \cdot r)` and folds it into a low-resolution tensor
    with :math:`C \cdot r^2` channels and spatial size :math:`(H, W)`.

    .. math::

        \text{output}[n,\, c \cdot r^2 + s_h \cdot r + s_w,\, H,\, W]
        = \text{input}[n,\, c,\, H \cdot r + s_h,\, W \cdot r + s_w]

    Parameters
    ----------
    downscale_factor : int
        Spatial downscaling factor :math:`r`.  Both ``H`` and ``W`` must be
        divisible by :math:`r`.

    Attributes
    ----------
    downscale_factor : int
        Stored value of the ``downscale_factor`` constructor argument.

    Shape
    -----
    - **Input:** :math:`(N, C, H \cdot r, W \cdot r)`.
    - **Output:** :math:`(N, C \cdot r^2, H, W)`.

    Notes
    -----
    - Commonly used in the *analysis* (encoder) side of learned image
      compression codecs to efficiently represent high-frequency spatial
      detail in the channel dimension before further processing.
    - Implemented as reshape → permute → reshape, symmetrically to
      ``PixelShuffle``.

    Examples
    --------
    **Fold high-resolution channels into low-resolution feature maps:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> pus = nn.PixelUnshuffle(downscale_factor=2)
    >>> x = lucid.zeros(1, 3, 112, 112)
    >>> pus(x).shape    # 12 = 3 * 2^2
    (1, 12, 56, 56)

    **Round-trip with PixelShuffle:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> x   = lucid.randn(2, 4, 30, 30)
    >>> pus = nn.PixelUnshuffle(downscale_factor=3)
    >>> ps  = nn.PixelShuffle(upscale_factor=3)
    >>> x_hat = ps(pus(x))
    >>> x_hat.shape == x.shape
    True
    """

    def __init__(self, downscale_factor: int) -> None:
        """Initialise the PixelUnshuffle module. See the class docstring for parameter semantics."""
        super().__init__()
        self.downscale_factor = downscale_factor

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Upsample the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)`.

        Returns
        -------
        Tensor
            Upsampled output tensor.
        """
        r = self.downscale_factor
        n, c, h_r, w_r = input.shape
        h, w = h_r // r, w_r // r
        impl = _unwrap(input)
        t = _C_engine.reshape(impl, [n, c, h, r, w, r])
        t = _C_engine.permute(t, [0, 1, 3, 5, 2, 4])
        return _wrap(_C_engine.reshape(t, [n, c * r * r, h, w]))

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"downscale_factor={self.downscale_factor}"


class ChannelShuffle(Module):
    r"""Shuffle channels across groups to enable cross-group information flow.

    ``ChannelShuffle`` implements the channel shuffle operation introduced in
    ShuffleNet (Zhang et al., 2018).  When depth-wise group convolutions are
    stacked, each group's output depends only on its own input channels;
    channel shuffling breaks this isolation by interleaving outputs from
    different groups before the next layer.

    The operation reshapes the channel dimension into
    :math:`(\text{groups}, C / \text{groups})`, transposes those two axes,
    and then flattens back to :math:`C` channels:

    .. math::

        \text{reshape:} \quad
        (N, C, H, W)
        \;\to\;
        (N,\, g,\, C/g,\, H,\, W)

    .. math::

        \text{transpose axes 1 and 2:} \quad
        (N,\, g,\, C/g,\, H,\, W)
        \;\to\;
        (N,\, C/g,\, g,\, H,\, W)

    .. math::

        \text{flatten back:} \quad
        (N,\, C/g,\, g,\, H,\, W)
        \;\to\;
        (N,\, C,\, H,\, W)

    Parameters
    ----------
    groups : int
        Number of channel groups :math:`g`.  The channel count :math:`C`
        must be divisible by ``groups``.

    Attributes
    ----------
    groups : int
        Stored value of the ``groups`` constructor argument.

    Shape
    -----
    - **Input:** :math:`(N, C, H, W)` where :math:`C` is divisible by
      ``groups``.
    - **Output:** :math:`(N, C, H, W)` — same shape as input.

    Notes
    -----
    - When ``groups == 1`` the output is identical to the input (identity).
    - The operation has **no learnable parameters** and is implemented as a
      sequence of reshape + transpose + reshape operations on the C++ engine.
    - In ShuffleNet, channel shuffle is placed between two consecutive
      group-wise point-wise convolutions to allow information exchange across
      groups without the cost of a full dense convolution.

    Examples
    --------
    **ShuffleNet-style block with group convolutions:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> g = 4
    >>> block = nn.Sequential(
    ...     nn.Conv2d(64, 64, kernel_size=1, groups=g),   # group-wise PW
    ...     nn.BatchNorm2d(64),
    ...     nn.ReLU(),
    ...     nn.ChannelShuffle(groups=g),                  # shuffle across groups
    ...     nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),  # DW
    ...     nn.BatchNorm2d(64),
    ...     nn.Conv2d(64, 64, kernel_size=1, groups=g),   # group-wise PW
    ...     nn.BatchNorm2d(64),
    ... )
    >>> x = lucid.zeros(2, 64, 28, 28)
    >>> block(x).shape
    (2, 64, 28, 28)

    **Verify shuffle is a permutation (no information lost):**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> x  = lucid.randn(1, 8, 4, 4)
    >>> cs = nn.ChannelShuffle(groups=4)
    >>> y  = cs(x)
    >>> y.shape == x.shape
    True
    """

    def __init__(self, groups: int) -> None:
        """Initialise the ChannelShuffle module. See the class docstring for parameter semantics."""
        super().__init__()
        self.groups = groups

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Upsample the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)`.

        Returns
        -------
        Tensor
            Upsampled output tensor.
        """
        return channel_shuffle(input, groups=self.groups)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"groups={self.groups}"
