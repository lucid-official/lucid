"""
Pooling modules.
"""

from typing import TYPE_CHECKING, override

from lucid._tensor.tensor import Tensor
from lucid._types import _Size2d, _Size3d
from lucid.nn.module import Module
from lucid.nn.functional._spatial_args import _ntuple
from lucid.nn.functional.pooling import (
    max_pool1d,
    max_pool2d,
    max_pool3d,
    avg_pool1d,
    avg_pool2d,
    avg_pool3d,
    adaptive_avg_pool1d,
    adaptive_avg_pool2d,
    adaptive_avg_pool3d,
    adaptive_max_pool2d,
    adaptive_max_pool1d,
    adaptive_max_pool3d,
    max_unpool1d,
    max_unpool2d,
    max_unpool3d,
    fractional_max_pool2d,
    fractional_max_pool3d,
    lp_pool1d,
    lp_pool2d,
    lp_pool3d,
)

# An adaptive pool's ``output_size``; ``None`` keeps that axis's extent.
type _OutSize = int | tuple[int | None, ...]


class MaxPool1d(Module):
    r"""Applies 1-D max pooling over a sequence.

    For each position in the output, the module selects the maximum value
    from a sliding window of length ``kernel_size`` applied along the last
    (length) dimension of the input.

    .. math::

        y[n, c, i] = \max_{0 \le k < k_s}
            x\!\left[n,\, c,\, i \cdot s + k \cdot d\right]

    where :math:`k_s` is ``kernel_size``, :math:`s` is ``stride``, and
    :math:`d` is ``dilation``.

    The output length is computed as:

    .. math::

        L_{out} = \left\lfloor
            \frac{L_{in} + 2p - d(k_s - 1) - 1}{s} + 1
        \right\rfloor

    When ``ceil_mode=True`` the floor is replaced by a ceiling, which may
    include one extra partial window at the right boundary.

    Parameters
    ----------
    kernel_size : int
        Size of the sliding window.
    stride : int or None, optional
        Step between successive windows.  Defaults to ``kernel_size`` when
        ``None``.
    padding : int, optional
        Zero-padding added to both ends of the input before pooling.
        Default: ``0``.
    dilation : int, optional
        Spacing between kernel elements (dilated / atrous pooling).
        ``dilation=1`` gives the standard contiguous window.
        Default: ``1``.
    return_indices : bool, optional
        If ``True``, the forward pass returns ``(output, indices)`` where
        ``indices`` holds the argmax locations for :class:`MaxUnpool1d`.
        Default: ``False``.
    ceil_mode : bool, optional
        When ``True``, use ceiling instead of floor to compute the output
        length.  Default: ``False``.

    Attributes
    ----------
    kernel_size : int
    stride : int or None
    padding : int
    dilation : int
    ceil_mode : bool

    Shape
    -----
    - Input:  ``(N, C, L_in)``
    - Output: ``(N, C, L_out)``

    Notes
    -----
    - Dilation inserts :math:`d - 1` gaps between consecutive kernel
      elements, effectively enlarging the receptive field without
      increasing the number of comparisons.
    - Only the largest value in each window is propagated; gradients flow
      back exclusively to the winning element.

    Examples
    --------
    Basic non-overlapping pooling (stride equals kernel_size by default):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> pool = nn.MaxPool1d(kernel_size=3)
    >>> x = lucid.ones((1, 4, 12))
    >>> y = pool(x)
    >>> y.shape
    (1, 4, 4)

    Keeping where each maximum came from, for
    :class:`~lucid.nn.MaxUnpool1d`:

    >>> pool = nn.MaxPool1d(kernel_size=2, return_indices=True)
    >>> y, idx = pool(lucid.randn(2, 8, 16))
    >>> tuple(idx.shape), idx.dtype
    ((2, 8, 8), lucid.int64)
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        """Initialise the MaxPool1d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.return_indices = return_indices
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    if TYPE_CHECKING:

        @override
        def forward(self, input: Tensor) -> Tensor: ...

    else:

        @override
        def forward(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
            """Apply the pooling operation to the input tensor."""
            return max_pool1d(
                input,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.ceil_mode,
                self.return_indices,
            )

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}"
        )


class MaxPool2d(Module):
    r"""Applies 2-D max pooling over a spatial feature map.

    For each position in the output, the module selects the maximum value
    from a rectangular ``kernel_size`` window applied over the height and
    width dimensions of the input.

    .. math::

        y[n, c, h, w] = \max_{\substack{0 \le k_h < k_H \\ 0 \le k_w < k_W}}
            x\!\left[n,\, c,\, h \cdot s_H + k_h \cdot d_H,\,
                          w \cdot s_W + k_w \cdot d_W\right]

    where :math:`(k_H, k_W)` is ``kernel_size``, :math:`(s_H, s_W)` is
    ``stride``, and :math:`(d_H, d_W)` is ``dilation``.

    Output spatial sizes are:

    .. math::

        H_{out} = \left\lfloor
            \frac{H_{in} + 2p_H - d_H(k_H - 1) - 1}{s_H} + 1
        \right\rfloor, \quad
        W_{out} = \left\lfloor
            \frac{W_{in} + 2p_W - d_W(k_W - 1) - 1}{s_W} + 1
        \right\rfloor

    Parameters
    ----------
    kernel_size : int or tuple[int, int]
        Size of the pooling window.  A single ``int`` is broadcast to
        ``(kernel_size, kernel_size)``.
    stride : int or tuple[int, int] or None, optional
        Step between successive windows.  Defaults to ``kernel_size``.
    padding : int or tuple[int, int], optional
        Zero-padding added to all four sides of the input.  Default: ``0``.
    dilation : int or tuple[int, int], optional
        Spacing between kernel elements.  Default: ``1``.
    return_indices : bool, optional
        If ``True``, returns ``(output, indices)``.  Default: ``False``.
    ceil_mode : bool, optional
        Use ceiling instead of floor for output size.  Default: ``False``.

    Attributes
    ----------
    kernel_size : int or tuple[int, int]
    stride : int or tuple[int, int] or None
    padding : int or tuple[int, int]
    dilation : int or tuple[int, int]
    ceil_mode : bool

    Shape
    -----
    - Input:  ``(N, C, H_in, W_in)``
    - Output: ``(N, C, H_out, W_out)``

    Notes
    -----
    - Max pooling is translation-equivariant and provides a form of local
      invariance to small spatial shifts.
    - When ``dilation > 1``, the effective receptive field grows without
      extra memory cost (atrous / dilated pooling).
    - A common configuration for a 2× spatial downsampling is
      ``MaxPool2d(kernel_size=2, stride=2)``.

    Examples
    --------
    Halving spatial resolution:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> pool = nn.MaxPool2d(kernel_size=2, stride=2)
    >>> x = lucid.ones((1, 3, 32, 32))
    >>> y = pool(x)
    >>> y.shape
    (1, 3, 16, 16)

    Asymmetric kernel with padding:

    >>> pool = nn.MaxPool2d(kernel_size=(3, 5), stride=1, padding=(1, 2))
    >>> x = lucid.ones((2, 16, 24, 24))
    >>> y = pool(x)
    >>> y.shape
    (2, 16, 24, 24)
    """

    def __init__(
        self,
        kernel_size: _Size2d,
        stride: _Size2d | None = None,
        padding: _Size2d = 0,
        dilation: _Size2d = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        """Initialise the MaxPool2d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.return_indices = return_indices
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    if TYPE_CHECKING:

        @override
        def forward(self, input: Tensor) -> Tensor: ...

    else:

        @override
        def forward(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
            """Apply the pooling operation to the input tensor."""
            return max_pool2d(
                input,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.ceil_mode,
                self.return_indices,
            )

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}"
        )


class AvgPool1d(Module):
    r"""Applies 1-D average pooling over a sequence.

    For each output position the module computes the arithmetic mean of
    the values in a sliding window of length ``kernel_size``:

    .. math::

        y[n, c, i] = \frac{1}{|W|} \sum_{k=0}^{k_s - 1}
            x\!\left[n,\, c,\, i \cdot s + k\right]

    where :math:`|W|` is the effective window size (see
    ``count_include_pad``) and :math:`s` is ``stride``.

    The output length follows:

    .. math::

        L_{out} = \left\lfloor
            \frac{L_{in} + 2p - k_s}{s} + 1
        \right\rfloor

    Parameters
    ----------
    kernel_size : int
        Size of the averaging window.
    stride : int or None, optional
        Step between successive windows.  Defaults to ``kernel_size``.
    padding : int, optional
        Zero-padding added to both ends of the input before pooling.
        Default: ``0``.
    ceil_mode : bool, optional
        Use ceiling instead of floor for the output length.
        Default: ``False``.
    count_include_pad : bool, optional
        If ``True`` (default), zero-padded values are counted in the
        denominator.  If ``False``, only real input values are averaged.

    Attributes
    ----------
    kernel_size : int
    stride : int or None
    padding : int
    ceil_mode : bool
    count_include_pad : bool

    Shape
    -----
    - Input:  ``(N, C, L_in)``
    - Output: ``(N, C, L_out)``

    Notes
    -----
    - When ``count_include_pad=True`` the denominator always equals
      ``kernel_size``, even for border windows that overlap the padding.
      Setting ``count_include_pad=False`` can produce slightly higher
      values near the boundaries.
    - Average pooling is commonly used in sequence models to obtain
      a fixed-size summary of variable-length inputs.

    Examples
    --------
    Non-overlapping average pooling:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> pool = nn.AvgPool1d(kernel_size=4)
    >>> x = lucid.ones((1, 8, 16))
    >>> y = pool(x)
    >>> y.shape
    (1, 8, 4)

    Overlapping pooling with padding, excluding pad from average:

    >>> pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1,
    ...                     count_include_pad=False)
    >>> x = lucid.ones((2, 4, 10))
    >>> y = pool(x)
    >>> y.shape
    (2, 4, 10)
    """

    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        """Initialise the AvgPool1d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the pooling operation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)` where :math:`*` are the
            spatial dimensions appropriate for this pooling layer.

        Returns
        -------
        Tensor
            Pooled output tensor.
        """
        return avg_pool1d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class AvgPool2d(Module):
    r"""Applies 2-D average pooling over a spatial feature map.

    For each output position the module computes the arithmetic mean of
    the values in a ``kernel_size`` window:

    .. math::

        y[n, c, h, w] = \frac{1}{|W|}
            \sum_{k_h=0}^{k_H-1} \sum_{k_w=0}^{k_W-1}
            x\!\left[n,\, c,\, h \cdot s_H + k_h,\, w \cdot s_W + k_w\right]

    where :math:`|W|` is the effective window area (governed by
    ``count_include_pad``).

    Output spatial sizes:

    .. math::

        H_{out} = \left\lfloor
            \frac{H_{in} + 2p_H - k_H}{s_H} + 1
        \right\rfloor, \quad
        W_{out} = \left\lfloor
            \frac{W_{in} + 2p_W - k_W}{s_W} + 1
        \right\rfloor

    Parameters
    ----------
    kernel_size : int or tuple[int, int]
        Size of the averaging window.
    stride : int or tuple[int, int] or None, optional
        Step between successive windows.  Defaults to ``kernel_size``.
    padding : int or tuple[int, int], optional
        Zero-padding added to all four sides.  Default: ``0``.
    ceil_mode : bool, optional
        Use ceiling instead of floor for output sizes.  Default: ``False``.
    count_include_pad : bool, optional
        Count zero-padded values in the denominator.  Default: ``True``.
    divisor_override : int or None, optional
        Divide each window's sum by this, never ``0``, instead of its size.

    Attributes
    ----------
    kernel_size : int or tuple[int, int]
    stride : int or tuple[int, int] or None
    padding : int or tuple[int, int]
    ceil_mode : bool
    count_include_pad : bool
    divisor_override : int or None

    Shape
    -----
    - Input:  ``(N, C, H_in, W_in)``
    - Output: ``(N, C, H_out, W_out)``

    Notes
    -----
    - Average pooling is differentiable everywhere and provides a smooth
      spatial summary, unlike max pooling which has sub-gradient issues at
      ties.
    - A ``kernel_size`` equal to the full spatial extent is called *global
      average pooling* (GAP) and is widely used as the penultimate layer
      in image classification networks.

    Examples
    --------
    Standard 2× downsampling:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> pool = nn.AvgPool2d(kernel_size=2, stride=2)
    >>> x = lucid.ones((1, 64, 56, 56))
    >>> y = pool(x)
    >>> y.shape
    (1, 64, 28, 28)

    Smoothing with padding (output same size as input):

    >>> pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
    >>> x = lucid.ones((4, 32, 16, 16))
    >>> y = pool(x)
    >>> y.shape
    (4, 32, 16, 16)
    """

    def __init__(
        self,
        kernel_size: _Size2d,
        stride: _Size2d | None = None,
        padding: _Size2d = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ) -> None:
        """Initialise the AvgPool2d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the pooling operation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)` where :math:`*` are the
            spatial dimensions appropriate for this pooling layer.

        Returns
        -------
        Tensor
            Pooled output tensor.
        """
        return avg_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class AdaptiveAvgPool1d(Module):
    r"""Applies adaptive 1-D average pooling to produce a fixed output length.

    Rather than specifying ``kernel_size`` and ``stride`` manually, you
    provide the desired output length and the module automatically derives
    the window parameters for each output position.  For output index
    :math:`i` the window boundaries are:

    .. math::

        \text{start}(i) = \left\lfloor \frac{i \cdot L_{in}}{L_{out}} \right\rfloor,
        \quad
        \text{end}(i)   = \left\lceil  \frac{(i+1) \cdot L_{in}}{L_{out}} \right\rceil

    so that :math:`y[n, c, i] = \text{mean}\bigl(x[n, c,
    \text{start}(i):\text{end}(i)]\bigr)`.

    Parameters
    ----------
    output_size : int or tuple[int, ...]
        Desired output length.  Pass ``None`` inside a tuple to keep that
        dimension unchanged.

    Attributes
    ----------
    output_size : int or tuple[int, ...]

    Shape
    -----
    - Input:  ``(N, C, L_in)``  (any ``L_in``)
    - Output: ``(N, C, output_size)``

    Notes
    -----
    - Adaptive pooling decouples the network architecture from the input
      size, which is particularly valuable when fine-tuning models on data
      with spatial dimensions different from the original training set.
    - The window sizes may be non-uniform across output positions; all
      averages are still exact (no overlap or gap).

    Examples
    --------
    Compress any sequence to length 1 (global average pooling):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> gap = nn.AdaptiveAvgPool1d(output_size=1)
    >>> x = lucid.ones((4, 512, 100))
    >>> y = gap(x)
    >>> y.shape
    (4, 512, 1)

    Downsample to a fixed length regardless of input:

    >>> pool = nn.AdaptiveAvgPool1d(output_size=8)
    >>> for l in [32, 64, 128]:
    ...     x = lucid.ones((1, 16, l))
    ...     assert pool(x).shape == (1, 16, 8)
    """

    def __init__(self, output_size: _OutSize) -> None:
        """Initialise the AdaptiveAvgPool1d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.output_size = output_size

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the pooling operation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)` where :math:`*` are the
            spatial dimensions appropriate for this pooling layer.

        Returns
        -------
        Tensor
            Pooled output tensor.
        """
        return adaptive_avg_pool1d(input, self.output_size)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"output_size={self.output_size}"


class AdaptiveAvgPool2d(Module):
    r"""Applies adaptive 2-D average pooling to produce a fixed output size.

    For each spatial output position :math:`(h, w)` the window boundaries
    are derived from the input-to-output ratio:

    .. math::

        h_s = \left\lfloor \frac{h \cdot H_{in}}{H_{out}} \right\rfloor,\quad
        h_e = \left\lceil  \frac{(h+1) H_{in}}{H_{out}} \right\rceil,\quad
        w_s = \left\lfloor \frac{w \cdot W_{in}}{W_{out}} \right\rfloor,\quad
        w_e = \left\lceil  \frac{(w+1) W_{in}}{W_{out}} \right\rceil

    and the output is:

    .. math::

        y[n, c, h, w] = \frac{1}{(h_e - h_s)(w_e - w_s)}
            \sum_{i=h_s}^{h_e-1}\sum_{j=w_s}^{w_e-1} x[n, c, i, j]

    Parameters
    ----------
    output_size : int or tuple[int, int]
        Target ``(H_out, W_out)``.  A scalar is broadcast to both
        dimensions.

    Attributes
    ----------
    output_size : int or tuple[int, int]

    Shape
    -----
    - Input:  ``(N, C, H_in, W_in)``  (any spatial size)
    - Output: ``(N, C, H_out, W_out)``

    Notes
    -----
    - Global average pooling (GAP) is a special case with
      ``output_size=(1, 1)`` and is the standard final pooling layer in
      modern classification backbones (ResNet, EfficientNet, etc.).
    - Unlike :class:`AvgPool2d`, the window dimensions may differ across
      output cells when :math:`H_{in}` is not divisible by :math:`H_{out}`.

    Examples
    --------
    Global average pooling for a classification head:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> gap = nn.AdaptiveAvgPool2d(output_size=1)
    >>> x = lucid.ones((8, 2048, 7, 7))
    >>> y = gap(x)
    >>> y.shape
    (8, 2048, 1, 1)

    Resize feature maps to a fixed ``4 × 4`` grid:

    >>> pool = nn.AdaptiveAvgPool2d(output_size=(4, 4))
    >>> for h, w in [(14, 14), (28, 28), (56, 56)]:
    ...     x = lucid.ones((2, 256, h, w))
    ...     assert pool(x).shape == (2, 256, 4, 4)
    """

    def __init__(self, output_size: _OutSize) -> None:
        """Initialise the AdaptiveAvgPool2d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.output_size = output_size

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the pooling operation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)` where :math:`*` are the
            spatial dimensions appropriate for this pooling layer.

        Returns
        -------
        Tensor
            Pooled output tensor.
        """
        return adaptive_avg_pool2d(input, self.output_size)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"output_size={self.output_size}"


class AdaptiveMaxPool2d(Module):
    r"""Applies adaptive 2-D max pooling to produce a fixed output size.

    The window boundaries for each output position are computed the same
    way as in :class:`AdaptiveAvgPool2d`, but instead of averaging the
    module returns the maximum value in each window:

    .. math::

        y[n, c, h, w] = \max_{i \in [h_s, h_e),\; j \in [w_s, w_e)}
            x[n, c, i, j]

    Parameters
    ----------
    output_size : int or tuple[int, int]
        Target ``(H_out, W_out)``.  A scalar is broadcast to both
        dimensions.
    return_indices : bool, optional
        If ``True``, returns ``(output, indices)``.  Default: ``False``.

    Attributes
    ----------
    output_size : int or tuple[int, int]

    Shape
    -----
    - Input:  ``(N, C, H_in, W_in)``
    - Output: ``(N, C, H_out, W_out)``

    Notes
    -----
    - Adaptive max pooling preserves the strongest activations in each
      spatial region, making it useful for detection and segmentation
      tasks where precise location of features matters.
    - Like :class:`AdaptiveAvgPool2d`, this layer accepts any input
      spatial size, which simplifies transfer learning from different
      input resolutions.

    Examples
    --------
    Reduce arbitrary feature map to a fixed ``3 × 3`` grid:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> pool = nn.AdaptiveMaxPool2d(output_size=(3, 3))
    >>> x = lucid.ones((1, 64, 20, 20))
    >>> y = pool(x)
    >>> y.shape
    (1, 64, 3, 3)

    Single-pixel summary (spatial global max):

    >>> pool = nn.AdaptiveMaxPool2d(output_size=1)
    >>> x = lucid.ones((4, 128, 13, 13))
    >>> y = pool(x)
    >>> y.shape
    (4, 128, 1, 1)
    """

    def __init__(self, output_size: _OutSize, return_indices: bool = False) -> None:
        """Initialise the AdaptiveMaxPool2d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.return_indices = return_indices
        self.output_size = output_size

    if TYPE_CHECKING:

        @override
        def forward(self, input: Tensor) -> Tensor: ...

    else:

        @override
        def forward(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
            """Apply the pooling operation to the input tensor."""
            return adaptive_max_pool2d(input, self.output_size, self.return_indices)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"output_size={self.output_size}"


class MaxPool3d(Module):
    r"""Applies 3-D max pooling over a volumetric feature map.

    Extends :class:`MaxPool2d` to three spatial dimensions (depth, height,
    width).  For each output position the maximum is taken over a
    :math:`k_D \times k_H \times k_W` window:

    .. math::

        y[n,c,d,h,w] = \max_{\substack{0 \le k_d < k_D \\
                                        0 \le k_h < k_H \\
                                        0 \le k_w < k_W}}
        x\!\left[n,\, c,\,
            d \cdot s_D + k_d,\,
            h \cdot s_H + k_h,\,
            w \cdot s_W + k_w
        \right]

    Output dimensions:

    .. math::

        D_{out} &= \left\lfloor
            \frac{D_{in} + 2p_D - k_D}{s_D} + 1
        \right\rfloor \\[4pt]
        H_{out} &= \left\lfloor
            \frac{H_{in} + 2p_H - k_H}{s_H} + 1
        \right\rfloor \\[4pt]
        W_{out} &= \left\lfloor
            \frac{W_{in} + 2p_W - k_W}{s_W} + 1
        \right\rfloor

    Parameters
    ----------
    kernel_size : int or tuple[int, int, int]
        Size of the pooling window along ``(D, H, W)``.
    stride : int or tuple[int, int, int] or None, optional
        Step between windows.  Defaults to ``kernel_size``.
    padding : int or tuple[int, int, int], optional
        Zero-padding applied to all six faces of the input.
        Default: ``0``.
    dilation : int or tuple[int, int, int], optional
        Spacing between kernel elements.  Default: ``1``.
    return_indices : bool, optional
        If ``True``, returns ``(output, indices)``.  Default: ``False``.
    ceil_mode : bool, optional
        Use ceiling for output size.  Default: ``False``.

    Attributes
    ----------
    kernel_size : int or tuple[int, int, int]
    stride : int or tuple[int, int, int] or None
    padding : int or tuple[int, int, int]
    dilation : int or tuple[int, int, int]
    ceil_mode : bool

    Shape
    -----
    - Input:  ``(N, C, D_in, H_in, W_in)``
    - Output: ``(N, C, D_out, H_out, W_out)``

    Notes
    -----
    - Typical use cases include video understanding (D = temporal frames)
      and 3-D medical imaging (CT / MRI volumes).
    - Computational cost is proportional to the product
      :math:`D_{out} \times H_{out} \times W_{out} \times k_D k_H k_W`.

    Examples
    --------
    Halving all three spatial dimensions:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> pool = nn.MaxPool3d(kernel_size=2, stride=2)
    >>> x = lucid.ones((1, 32, 16, 16, 16))
    >>> y = pool(x)
    >>> y.shape
    (1, 32, 8, 8, 8)

    Asymmetric window for video (different temporal/spatial strides):

    >>> pool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2))
    >>> x = lucid.ones((2, 64, 8, 28, 28))
    >>> y = pool(x)
    >>> y.shape
    (2, 64, 8, 13, 13)
    """

    def __init__(
        self,
        kernel_size: _Size3d,
        stride: _Size3d | None = None,
        padding: _Size3d = 0,
        dilation: _Size3d = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        """Initialise the MaxPool3d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.return_indices = return_indices
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    if TYPE_CHECKING:

        @override
        def forward(self, input: Tensor) -> Tensor: ...

    else:

        @override
        def forward(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
            """Apply the pooling operation to the input tensor."""
            return max_pool3d(
                input,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.ceil_mode,
                self.return_indices,
            )

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"kernel_size={self.kernel_size}, stride={self.stride}, " f"padding={self.padding}"


class AvgPool3d(Module):
    r"""Applies 3-D average pooling over a volumetric feature map.

    Computes the arithmetic mean of values in a
    :math:`k_D \times k_H \times k_W` sliding window:

    .. math::

        y[n,c,d,h,w] = \frac{1}{|W|}
            \sum_{k_d=0}^{k_D-1}\sum_{k_h=0}^{k_H-1}\sum_{k_w=0}^{k_W-1}
            x\!\left[n,\,c,\,
                d \cdot s_D + k_d,\,
                h \cdot s_H + k_h,\,
                w \cdot s_W + k_w
            \right]

    where :math:`|W| = k_D k_H k_W` when ``count_include_pad=True``.

    Output dimensions mirror those of :class:`MaxPool3d`.

    Parameters
    ----------
    kernel_size : int or tuple[int, int, int]
        Size of the averaging window along ``(D, H, W)``.
    stride : int or tuple[int, int, int] or None, optional
        Step between windows.  Defaults to ``kernel_size``.
    padding : int or tuple[int, int, int], optional
        Zero-padding on all six faces.  Default: ``0``.
    ceil_mode : bool, optional
        Use ceiling for output size.  Default: ``False``.
    count_include_pad : bool, optional
        Include zero-padded elements in the denominator.  Default: ``True``.
    divisor_override : int or None, optional
        If set, each window's sum is divided by this instead of its
        size.  Must not be ``0``.  Default: ``None``.

    Attributes
    ----------
    kernel_size : int or tuple[int, int, int]
    stride : int or tuple[int, int, int] or None
    padding : int or tuple[int, int, int]
    ceil_mode : bool
    count_include_pad : bool

    Shape
    -----
    - Input:  ``(N, C, D_in, H_in, W_in)``
    - Output: ``(N, C, D_out, H_out, W_out)``

    Notes
    -----
    - 3-D average pooling is the natural extension of 2-D GAP to video
      and volumetric data; global 3-D average pooling collapses
      ``(D, H, W)`` to ``(1, 1, 1)`` for a compact feature vector.
    - Gradients are distributed uniformly over all elements in a window,
      which can help maintain stable training compared to max pooling.

    Examples
    --------
    Reduce temporal resolution while keeping spatial size:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> pool = nn.AvgPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
    >>> x = lucid.ones((1, 16, 16, 8, 8))
    >>> y = pool(x)
    >>> y.shape
    (1, 16, 8, 8, 8)

    Halve all three dimensions simultaneously:

    >>> pool = nn.AvgPool3d(kernel_size=2, stride=2)
    >>> x = lucid.ones((2, 32, 8, 8, 8))
    >>> y = pool(x)
    >>> y.shape
    (2, 32, 4, 4, 4)
    """

    def __init__(
        self,
        kernel_size: _Size3d,
        stride: _Size3d | None = None,
        padding: _Size3d = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ) -> None:
        """Initialise the AvgPool3d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the pooling operation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)` where :math:`*` are the
            spatial dimensions appropriate for this pooling layer.

        Returns
        -------
        Tensor
            Pooled output tensor.
        """
        return avg_pool3d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, ceil_mode={self.ceil_mode}, "
            f"count_include_pad={self.count_include_pad}"
        )


class AdaptiveAvgPool3d(Module):
    r"""Applies adaptive 3-D average pooling to produce a fixed output volume.

    Automatically derives window parameters so that the output has the
    specified ``(D_out, H_out, W_out)`` shape regardless of input size.
    For output voxel :math:`(d, h, w)`:

    .. math::

        y[n,c,d,h,w] = \frac{1}{(d_e-d_s)(h_e-h_s)(w_e-w_s)}
            \sum_{i=d_s}^{d_e-1}
            \sum_{j=h_s}^{h_e-1}
            \sum_{k=w_s}^{w_e-1}
            x[n,c,i,j,k]

    where the boundary indices follow the same ratio formula as in
    :class:`AdaptiveAvgPool2d`, applied independently per dimension.

    Parameters
    ----------
    output_size : int or tuple[int, int, int]
        Target ``(D_out, H_out, W_out)``.  A scalar is broadcast to all
        three dimensions.

    Attributes
    ----------
    output_size : int or tuple[int, int, int]

    Shape
    -----
    - Input:  ``(N, C, D_in, H_in, W_in)``
    - Output: ``(N, C, D_out, H_out, W_out)``

    Notes
    -----
    - Pass ``output_size=(1, 1, 1)`` for 3-D global average pooling, which
      collapses a volumetric feature map to a single vector per channel —
      the standard head for 3-D classification networks.
    - Window sizes are determined per output cell and may differ, but all
      cells together partition the input without overlap or gap.

    Examples
    --------
    3-D global average pooling:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> gap = nn.AdaptiveAvgPool3d(output_size=1)
    >>> x = lucid.ones((2, 256, 4, 7, 7))
    >>> y = gap(x)
    >>> y.shape
    (2, 256, 1, 1, 1)

    Fixed spatial output with variable input:

    >>> pool = nn.AdaptiveAvgPool3d(output_size=(2, 4, 4))
    >>> for d in [4, 8, 16]:
    ...     x = lucid.ones((1, 64, d, 28, 28))
    ...     assert pool(x).shape == (1, 64, 2, 4, 4)
    """

    def __init__(self, output_size: _OutSize) -> None:
        """Initialise the AdaptiveAvgPool3d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.output_size = output_size

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the pooling operation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)` where :math:`*` are the
            spatial dimensions appropriate for this pooling layer.

        Returns
        -------
        Tensor
            Pooled output tensor.
        """
        return adaptive_avg_pool3d(input, self.output_size)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"output_size={self.output_size}"


class AdaptiveMaxPool1d(Module):
    r"""Applies adaptive 1-D max pooling to produce a fixed output length.

    For each output position :math:`i`, the window boundaries are:

    .. math::

        \text{start}(i) = \left\lfloor \frac{i \cdot L_{in}}{L_{out}} \right\rfloor,
        \quad
        \text{end}(i)   = \left\lceil  \frac{(i+1) L_{in}}{L_{out}} \right\rceil

    and the output is:

    .. math::

        y[n, c, i] = \max_{k \in [\text{start}(i),\; \text{end}(i))}
            x[n, c, k]

    Parameters
    ----------
    output_size : int or tuple[int, ...]
        Desired output length.
    return_indices : bool, optional
        If ``True``, returns ``(output, indices)``.  Default: ``False``.

    Attributes
    ----------
    output_size : int or tuple[int, ...]

    Shape
    -----
    - Input:  ``(N, C, L_in)``
    - Output: ``(N, C, output_size)``

    Notes
    -----
    - Adaptive max pooling retains the strongest activation in each
      adaptively-sized segment, combining the spatial invariance of max
      pooling with input-size flexibility.
    - This layer is frequently used in region-proposal networks and
      variable-length sequence encoders.

    Examples
    --------
    Compress sequence to fixed length 4:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> pool = nn.AdaptiveMaxPool1d(output_size=4)
    >>> x = lucid.ones((2, 32, 50))
    >>> y = pool(x)
    >>> y.shape
    (2, 32, 4)

    Global max pooling (output_size=1):

    >>> gmp = nn.AdaptiveMaxPool1d(output_size=1)
    >>> x = lucid.ones((4, 128, 200))
    >>> y = gmp(x)
    >>> y.shape
    (4, 128, 1)
    """

    def __init__(self, output_size: _OutSize, return_indices: bool = False) -> None:
        """Initialise the AdaptiveMaxPool1d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.return_indices = return_indices
        self.output_size = output_size

    if TYPE_CHECKING:

        @override
        def forward(self, input: Tensor) -> Tensor: ...

    else:

        @override
        def forward(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
            """Apply the pooling operation to the input tensor."""
            return adaptive_max_pool1d(input, self.output_size, self.return_indices)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"output_size={self.output_size}"


class AdaptiveMaxPool3d(Module):
    r"""Applies adaptive 3-D max pooling to produce a fixed output volume.

    Extends :class:`AdaptiveMaxPool2d` by one depth dimension.  For each
    output voxel :math:`(d, h, w)` the maximum is taken over the
    adaptively-sized 3-D window:

    .. math::

        y[n,c,d,h,w] = \max_{\substack{i \in [d_s, d_e) \\
                                        j \in [h_s, h_e) \\
                                        k \in [w_s, w_e)}}
        x[n,c,i,j,k]

    where the window bounds follow the same ratio formula as
    :class:`AdaptiveAvgPool3d`.

    Parameters
    ----------
    output_size : int or tuple[int, int, int]
        Target ``(D_out, H_out, W_out)``.  A scalar is broadcast.
    return_indices : bool, optional
        If ``True``, returns ``(output, indices)``.  Default: ``False``.

    Attributes
    ----------
    output_size : int or tuple[int, int, int]

    Shape
    -----
    - Input:  ``(N, C, D_in, H_in, W_in)``
    - Output: ``(N, C, D_out, H_out, W_out)``

    Notes
    -----
    - 3-D adaptive max pooling is useful for video classification models
      that must handle clips of arbitrary duration.
    - Like all adaptive pooling layers, the variable-sized windows can
      produce slightly different effective receptive fields per output
      cell when the input is not evenly divisible.

    Examples
    --------
    Compress volumetric feature map to ``2 × 2 × 2``:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> pool = nn.AdaptiveMaxPool3d(output_size=2)
    >>> x = lucid.ones((1, 32, 8, 8, 8))
    >>> y = pool(x)
    >>> y.shape
    (1, 32, 2, 2, 2)

    Mixed output sizes (keep depth, compress spatial):

    >>> pool = nn.AdaptiveMaxPool3d(output_size=(8, 1, 1))
    >>> x = lucid.ones((2, 64, 8, 14, 14))
    >>> y = pool(x)
    >>> y.shape
    (2, 64, 8, 1, 1)
    """

    def __init__(self, output_size: _OutSize, return_indices: bool = False) -> None:
        """Initialise the AdaptiveMaxPool3d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.return_indices = return_indices
        self.output_size = output_size

    if TYPE_CHECKING:

        @override
        def forward(self, input: Tensor) -> Tensor: ...

    else:

        @override
        def forward(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
            """Apply the pooling operation to the input tensor."""
            return adaptive_max_pool3d(input, self.output_size, self.return_indices)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"output_size={self.output_size}"


class LPPool1d(Module):
    r"""Applies 1-D power-average (Lp-norm) pooling over a sequence.

    For each window of length ``kernel_size`` the module computes the
    power average of the elements:

    .. math::

        y[n, c, i] = \left(
            \sum_{k=0}^{k_s - 1} x[n,\, c,\, i \cdot s + k]^p
        \right)^{1/p}

    where :math:`p` = ``norm_type``, :math:`k_s` = ``kernel_size``, and
    :math:`s` = ``stride``.  The power keeps the sign of :math:`x`; see
    :func:`lucid.nn.functional.lp_pool1d`.

    Special cases:

    * :math:`p = 1` — sum pooling.
    * :math:`p = 2` — square-root of sum of squares (L2 norm).
    * :math:`p = \infty` — max pooling of :math:`|x|`.

    Parameters
    ----------
    norm_type : float
        The exponent :math:`p`.  Must be positive; ``inf`` is allowed.
    kernel_size : int
        Size of the sliding window.
    stride : int or None, optional
        Step between windows.  Defaults to ``kernel_size``.
    ceil_mode : bool, optional
        Use ceiling for output length.  Default: ``False``.

    Attributes
    ----------
    norm_type : float
    kernel_size : int
    stride : int
    ceil_mode : bool

    Shape
    -----
    - Input:  ``(N, C, L_in)``
    - Output: ``(N, C, L_out)`` where
      :math:`L_{out} = \lfloor (L_{in} - k_s) / s \rfloor + 1`

    Notes
    -----
    - Lp pooling is a power-mean generalisation: it smoothly interpolates
      between sum pooling (:math:`p=1`), energy pooling (:math:`p=2`), and
      max pooling (:math:`p \to \infty`).
    - Because the operation is differentiable for finite :math:`p`, it can
      be used as a drop-in replacement for max or average pooling in
      gradient-based training.

    Examples
    --------
    L2 power pooling (energy pooling):

    >>> import lucid
    >>> import lucid.nn as nn
    >>> pool = nn.LPPool1d(norm_type=2, kernel_size=4)
    >>> x = lucid.ones((1, 8, 16))
    >>> y = pool(x)
    >>> y.shape
    (1, 8, 4)

    Sum pooling (p=1) with overlapping windows:

    >>> pool = nn.LPPool1d(norm_type=1, kernel_size=3, stride=1)
    >>> x = lucid.ones((2, 4, 12))
    >>> y = pool(x)
    >>> y.shape
    (2, 4, 10)
    """

    def __init__(
        self, norm_type: float, kernel_size: int, stride: int | None = None, ceil_mode: bool = False
    ) -> None:
        """Initialise the LPPool1d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.ceil_mode = ceil_mode

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the pooling operation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)` where :math:`*` are the
            spatial dimensions appropriate for this pooling layer.

        Returns
        -------
        Tensor
            Pooled output tensor.
        """
        return lp_pool1d(input, self.norm_type, self.kernel_size, self.stride, self.ceil_mode)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"norm_type={self.norm_type}, kernel_size={self.kernel_size}, stride={self.stride}"


class LPPool2d(Module):
    r"""Applies 2-D power-average (Lp-norm) pooling over a spatial feature map.

    For each :math:`k_H \times k_W` window the module computes:

    .. math::

        y[n,c,h,w] = \left(
            \sum_{k_h=0}^{k_H-1} \sum_{k_w=0}^{k_W-1}
            x\!\left[n,\,c,\,h \cdot s_H + k_h,\,w \cdot s_W + k_w\right]^p
        \right)^{1/p}

    with :math:`p` = ``norm_type``.  The power keeps the sign of :math:`x`,
    and the special cases :math:`p = 1, 2, \infty` are those of
    :class:`LPPool1d`.

    Parameters
    ----------
    norm_type : float
        The exponent :math:`p`.  Must be positive; ``inf`` is allowed.
    kernel_size : int or tuple[int, int]
        Window size ``(k_H, k_W)``.  A scalar is broadcast.
    stride : int or tuple[int, int] or None, optional
        Step between windows.  Defaults to ``kernel_size``.
    ceil_mode : bool, optional
        Use ceiling for output sizes.  Default: ``False``.

    Attributes
    ----------
    norm_type : float
    kh : int
        Kernel height.
    kw : int
        Kernel width.
    sh : int
        Stride along the height dimension.
    sw : int
        Stride along the width dimension.
    ceil_mode : bool

    Shape
    -----
    - Input:  ``(N, C, H_in, W_in)``
    - Output: ``(N, C, H_out, W_out)`` where

      .. math::

          H_{out} = \left\lfloor \frac{H_{in} - k_H}{s_H} \right\rfloor + 1, \quad
          W_{out} = \left\lfloor \frac{W_{in} - k_W}{s_W} \right\rfloor + 1

    Notes
    -----
    - Lp pooling with :math:`p = 2` appears in energy-based models and
      certain biologically-inspired convolutional architectures.

    Examples
    --------
    L2 energy pooling:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> pool = nn.LPPool2d(norm_type=2, kernel_size=2, stride=2)
    >>> x = lucid.ones((1, 32, 16, 16))
    >>> y = pool(x)
    >>> y.shape
    (1, 32, 8, 8)

    Asymmetric window with p=1 (sum pooling):

    >>> pool = nn.LPPool2d(norm_type=1, kernel_size=(3, 1), stride=(1, 1))
    >>> x = lucid.ones((2, 16, 10, 8))
    >>> y = pool(x)
    >>> y.shape
    (2, 16, 8, 8)
    """

    def __init__(
        self,
        norm_type: float,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        ceil_mode: bool = False,
    ) -> None:
        """Initialise the LPPool2d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.norm_type = norm_type
        self.kh, self.kw = _ntuple(kernel_size, 2, "LPPool2d", "kernel_size")
        step = kernel_size if stride is None else stride
        self.sh, self.sw = _ntuple(step, 2, "LPPool2d", "stride")
        self.ceil_mode = ceil_mode

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the pooling operation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)` where :math:`*` are the
            spatial dimensions appropriate for this pooling layer.

        Returns
        -------
        Tensor
            Pooled output tensor.
        """
        return lp_pool2d(
            input, self.norm_type, (self.kh, self.kw), (self.sh, self.sw), self.ceil_mode
        )

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"norm_type={self.norm_type}, kernel_size=({self.kh},{self.kw}), "
            f"stride=({self.sh},{self.sw})"
        )


class LPPool3d(Module):
    r"""Applies 3-D power-average (Lp-norm) pooling over a volumetric input.

    Extends :class:`LPPool2d` to three spatial dimensions.  For each
    :math:`k_D \times k_H \times k_W` window:

    .. math::

        y[n,c,d,h,w] = \left(
            \sum_{k_d}\sum_{k_h}\sum_{k_w}
            x\!\left[n,c,\,
                d \cdot s_D + k_d,\,
                h \cdot s_H + k_h,\,
                w \cdot s_W + k_w
            \right]^p
        \right)^{1/p}

    The power keeps the sign of :math:`x`, as in :class:`LPPool1d`.

    Parameters
    ----------
    norm_type : float
        The exponent :math:`p`.  Must be positive; ``inf`` is allowed.
    kernel_size : int or tuple[int, int, int]
        Window size ``(k_D, k_H, k_W)``.  A scalar is broadcast.
    stride : int or tuple[int, int, int] or None, optional
        Step between windows.  Defaults to ``kernel_size``.
    ceil_mode : bool, optional
        Use ceiling for output sizes.  Default: ``False``.

    Attributes
    ----------
    norm_type : float
    kernel_size : int or tuple[int, int, int]
    stride : int or tuple[int, int, int]
    ceil_mode : bool

    Shape
    -----
    - Input:  ``(N, C, D_in, H_in, W_in)``
    - Output: ``(N, C, D_out, H_out, W_out)``

    Notes
    -----
    - Thin wrapper around :func:`lucid.nn.functional.pooling.lp_pool3d`,
      providing the standard ``Module`` interface.
    - Special cases follow those of :class:`LPPool1d` and
      :class:`LPPool2d` — the exponent :math:`p` smoothly interpolates
      between sum (:math:`p=1`), energy (:math:`p=2`), and max pooling.

    Examples
    --------
    L2 energy pooling over a volumetric map:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> pool = nn.LPPool3d(norm_type=2, kernel_size=2, stride=2)
    >>> x = lucid.ones((1, 16, 8, 8, 8))
    >>> y = pool(x)
    >>> y.shape
    (1, 16, 4, 4, 4)

    Non-overlapping sum pooling (p=1):

    >>> pool = nn.LPPool3d(norm_type=1, kernel_size=(1, 2, 2))
    >>> x = lucid.ones((2, 32, 4, 16, 16))
    >>> y = pool(x)
    >>> y.shape
    (2, 32, 4, 8, 8)
    """

    def __init__(
        self,
        norm_type: float,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] | None = None,
        ceil_mode: bool = False,
    ) -> None:
        """Initialise the LPPool3d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.norm_type = norm_type
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.ceil_mode = ceil_mode

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Apply the pooling operation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)` where :math:`*` are the
            spatial dimensions appropriate for this pooling layer.

        Returns
        -------
        Tensor
            Pooled output tensor.
        """
        return lp_pool3d(
            input,
            norm_type=self.norm_type,
            kernel_size=self.kernel_size,
            stride=self.stride,
            ceil_mode=self.ceil_mode,
        )

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"norm_type={self.norm_type}, kernel_size={self.kernel_size}, " f"stride={self.stride}"
        )


# ── P4 fill: MaxUnpool / FractionalMaxPool ────────────────────────────────


class _MaxUnpoolNd[S = int](Module):
    r"""Shared base for :class:`MaxUnpool1d`, :class:`MaxUnpool2d`, and
    :class:`MaxUnpool3d`.

    Stores ``kernel_size``, ``stride``, and ``padding`` — the same
    parameters used by the corresponding `MaxPool{1,2,3}d` layer — and
    provides a shared :meth:`extra_repr`.  Concrete subclasses implement
    :meth:`forward` by delegating to the appropriate functional call.
    """

    def __init__(self, kernel_size: S, stride: S | None = None, padding: S | int = 0) -> None:
        """Initialise the _MaxUnpoolNd module. See the class docstring for parameter semantics."""
        super().__init__()
        self.kernel_size: S = kernel_size
        self.stride: S = stride if stride is not None else kernel_size
        self.padding: S | int = padding

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"kernel_size={self.kernel_size}, stride={self.stride}, " f"padding={self.padding}"


class MaxUnpool1d(_MaxUnpoolNd[int]):
    r"""Computes a partial inverse of :class:`MaxPool1d`.

    :class:`MaxPool1d` is not fully invertible because the non-maximum
    values in each window are discarded.  :class:`MaxUnpool1d` performs
    an approximate inversion by placing each pooled value back at its
    original position (recorded by the ``indices`` tensor) and filling
    all other locations with zero.

    Formally, if :math:`x_{\text{pool}}[n,c,i]` was the maximum at flat
    index :math:`\text{idx}[n,c,i]` within the original input, then:

    .. math::

        x_{\text{unpool}}[n, c, j] =
        \begin{cases}
            x_{\text{pool}}[n, c, i] & \text{if } j = \text{idx}[n,c,i] \\
            0 & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    kernel_size : int
        Must match the ``kernel_size`` of the paired :class:`MaxPool1d`.
    stride : int or None, optional
        Must match the ``stride`` of the paired pool.  Defaults to
        ``kernel_size``.
    padding : int, optional
        Must match the ``padding`` of the paired pool.  Default: ``0``.

    Attributes
    ----------
    kernel_size : int
    stride : int
    padding : int

    Shape
    -----
    - Input ``x``:       ``(N, C, L_pooled)``
    - Input ``indices``: ``(N, C, L_pooled)``  — flat indices from pool
    - Output:            ``(N, C, L_original)``

    Notes
    -----
    - The ``indices`` tensor is typically obtained from a paired
      :class:`MaxPool1d` with ``return_indices=True``.
    - The output is sparse — only the positions corresponding to maxima
      are non-zero.  This sparsity is intentional and matches the original
      architecture (SegNet, etc.).
    - Use the optional ``output_size`` argument to control the shape of
      the reconstructed tensor when ambiguity exists.

    Examples
    --------
    Typical encoder-decoder unpool:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> unpool = nn.MaxUnpool1d(kernel_size=2, stride=2)
    >>> x = lucid.ones((1, 4, 8))
    >>> indices = lucid.zeros((1, 4, 8), dtype="int32")
    >>> y = unpool(x, indices)
    >>> y.shape
    (1, 4, 16)

    With explicit output_size:

    >>> unpool = nn.MaxUnpool1d(kernel_size=2, stride=2)
    >>> x = lucid.ones((1, 4, 8))
    >>> indices = lucid.zeros((1, 4, 8), dtype="int32")
    >>> y = unpool(x, indices, output_size=(1, 4, 17))
    >>> y.shape
    (1, 4, 17)
    """

    @override
    def forward(
        self, input: Tensor, indices: Tensor, output_size: tuple[int, ...] | None = None
    ) -> Tensor:
        r"""Apply the pooling operation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)` where :math:`*` are the
            spatial dimensions appropriate for this pooling layer.

        Returns
        -------
        Tensor
            Pooled output tensor.
        """
        return max_unpool1d(
            input,
            indices,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_size=output_size,
        )


class MaxUnpool2d(_MaxUnpoolNd[_Size2d]):
    r"""Computes a partial inverse of :class:`MaxPool2d`.

    Reconstructs a sparse approximation of the original 2-D feature map
    by scattering each pooled value back to the spatial position where
    the maximum was found, with all other positions set to zero.

    For a pooled value at output cell :math:`(h, w)` with flat spatial
    index :math:`\text{idx}[n,c,h,w]`:

    .. math::

        x_{\text{unpool}}[n,c,i,j] =
        \begin{cases}
            x_{\text{pool}}[n,c,h,w]
                & \text{if } (i,j) \text{ maps to } \text{idx}[n,c,h,w] \\
            0 & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    kernel_size : int or tuple[int, int]
        Must match the ``kernel_size`` of the paired :class:`MaxPool2d`.
    stride : int or tuple[int, int] or None, optional
        Must match the ``stride`` of the paired pool.  Defaults to
        ``kernel_size``.
    padding : int or tuple[int, int], optional
        Must match the ``padding`` of the paired pool.  Default: ``0``.

    Attributes
    ----------
    kernel_size : int or tuple[int, int]
    stride : int or tuple[int, int]
    padding : int or tuple[int, int]

    Shape
    -----
    - Input ``x``:       ``(N, C, H_pooled, W_pooled)``
    - Input ``indices``: ``(N, C, H_pooled, W_pooled)``
    - Output:            ``(N, C, H_original, W_original)``

    Notes
    -----
    - Originally introduced as part of the SegNet encoder-decoder
      architecture, where max-pooling indices are passed across the skip
      connection to guide unpooling in the decoder.
    - The reconstructed tensor is *not* a true inverse — non-maximum
      values from the original pooling input are permanently lost.
    - An optional ``output_size`` parameter resolves ambiguity when the
      original input size cannot be uniquely recovered from ``stride`` and
      ``kernel_size`` alone.

    Examples
    --------
    Symmetric encoder-decoder pair:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
    >>> x = lucid.ones((1, 64, 14, 14))
    >>> indices = lucid.zeros((1, 64, 14, 14), dtype="int32")
    >>> y = unpool(x, indices)
    >>> y.shape
    (1, 64, 28, 28)

    With explicit output_size to handle odd input dimensions:

    >>> unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
    >>> x = lucid.ones((1, 16, 7, 7))
    >>> indices = lucid.zeros((1, 16, 7, 7), dtype="int32")
    >>> y = unpool(x, indices, output_size=(1, 16, 15, 15))
    >>> y.shape
    (1, 16, 15, 15)
    """

    @override
    def forward(
        self, input: Tensor, indices: Tensor, output_size: tuple[int, ...] | None = None
    ) -> Tensor:
        r"""Apply the pooling operation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)` where :math:`*` are the
            spatial dimensions appropriate for this pooling layer.

        Returns
        -------
        Tensor
            Pooled output tensor.
        """
        return max_unpool2d(
            input,
            indices,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_size=output_size,
        )


class MaxUnpool3d(_MaxUnpoolNd[_Size3d]):
    r"""Computes a partial inverse of :class:`MaxPool3d`.

    Extends :class:`MaxUnpool2d` to three spatial dimensions.  Each
    pooled value is scattered back to its 3-D source position (recorded
    in ``indices``) and all other voxels are set to zero.

    Parameters
    ----------
    kernel_size : int or tuple[int, int, int]
        Must match the ``kernel_size`` of the paired :class:`MaxPool3d`.
    stride : int or tuple[int, int, int] or None, optional
        Must match the ``stride`` of the paired pool.  Defaults to
        ``kernel_size``.
    padding : int or tuple[int, int, int], optional
        Must match the ``padding`` of the paired pool.  Default: ``0``.

    Attributes
    ----------
    kernel_size : int or tuple[int, int, int]
    stride : int or tuple[int, int, int]
    padding : int or tuple[int, int, int]

    Shape
    -----
    - Input ``x``:       ``(N, C, D_pooled, H_pooled, W_pooled)``
    - Input ``indices``: ``(N, C, D_pooled, H_pooled, W_pooled)``
    - Output:            ``(N, C, D_original, H_original, W_original)``

    Notes
    -----
    - 3-D unpooling is used in volumetric segmentation networks that
      mirror a pooling encoder with an unpooling decoder.
    - As with :class:`MaxUnpool2d`, the output is sparse — only voxels
      that were pool maxima receive non-zero values.
    - Pass ``output_size`` when the original volumetric shape cannot be
      uniquely determined from the pooling hyperparameters.

    Examples
    --------
    Reconstruct a volumetric map:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> unpool = nn.MaxUnpool3d(kernel_size=2, stride=2)
    >>> x = lucid.ones((1, 32, 4, 4, 4))
    >>> indices = lucid.zeros((1, 32, 4, 4, 4), dtype="int32")
    >>> y = unpool(x, indices)
    >>> y.shape
    (1, 32, 8, 8, 8)

    With explicit output_size:

    >>> unpool = nn.MaxUnpool3d(kernel_size=2, stride=2)
    >>> x = lucid.ones((1, 8, 3, 3, 3))
    >>> indices = lucid.zeros((1, 8, 3, 3, 3), dtype="int32")
    >>> y = unpool(x, indices, output_size=(1, 8, 7, 7, 7))
    >>> y.shape
    (1, 8, 7, 7, 7)
    """

    @override
    def forward(
        self, input: Tensor, indices: Tensor, output_size: tuple[int, ...] | None = None
    ) -> Tensor:
        r"""Apply the pooling operation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)` where :math:`*` are the
            spatial dimensions appropriate for this pooling layer.

        Returns
        -------
        Tensor
            Pooled output tensor.
        """
        return max_unpool3d(
            input,
            indices,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            output_size=output_size,
        )


class FractionalMaxPool2d(Module):
    r"""Applies fractional max-pooling over a 2-D spatial feature map.

    Implements the random fractional pooling scheme of Graham (2014).
    Instead of a fixed integer stride, the spatial extent is divided by
    randomly sampled fractional boundaries, so the effective downsampling
    ratio is a random variable in :math:`(0, 1)` during training.

    For a given random partition of the height axis into :math:`H_{out}`
    intervals :math:`[r_h^{(i)}, r_h^{(i+1)})` and the width axis into
    :math:`W_{out}` intervals :math:`[r_w^{(j)}, r_w^{(j+1)})`:

    .. math::

        y[n, c, i, j] =
            \max_{h \in [r_h^{(i)},\, r_h^{(i+1)}),\;
                  w \in [r_w^{(j)},\, r_w^{(j+1)})}
            x[n, c, h, w]

    The boundaries are drawn independently per batch-channel pair from
    ``_random_samples`` (or internally sampled when not provided).

    Parameters
    ----------
    kernel_size : int or tuple[int, int]
        Size of the max pooling window applied at each sampled position.
    output_size : int or tuple[int, int] or None, optional
        Fixed target output size ``(H_out, W_out)``.  Exactly one of
        ``output_size`` and ``output_ratio`` must be specified.
    output_ratio : float or tuple[float, float] or None, optional
        Target output size as a fraction of the input size.  Each value
        must lie in :math:`(0, 1)`.  Exactly one of ``output_size`` and
        ``output_ratio`` must be specified.
    return_indices : bool, optional
        If ``True``, also return the flat indices of the selected maxima.
        Default: ``False``.

    Attributes
    ----------
    kernel_size : int or tuple[int, int]
    output_size : int or tuple[int, int] or None
    output_ratio : float or tuple[float, float] or None
    return_indices : bool

    Shape
    -----
    - Input:  ``(N, C, H_in, W_in)``
    - Output: ``(N, C, H_out, W_out)``

    Notes
    -----
    - Because the boundaries are random, **each forward pass produces a
      different spatial partition during training**.  This acts as a form
      of stochastic spatial data augmentation and can improve
      generalisation.
    - At inference (``model.eval()``), the same random draw is applied; if
      deterministic output is required, fix the random seed before
      calling forward.
    - Original reference: B. Graham, "Fractional Max-Pooling", *arXiv*
      1412.6071, 2014.
    - Implemented as a pure-Python composite using
      :func:`lucid.nn.functional.pooling.fractional_max_pool2d`; gradients
      flow through :meth:`Tensor.max` automatically.

    Examples
    --------
    Downsample to a fixed target size:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> pool = nn.FractionalMaxPool2d(kernel_size=2, output_size=(7, 7))
    >>> x = lucid.ones((1, 64, 14, 14))
    >>> y = pool(x)
    >>> y.shape
    (1, 64, 7, 7)

    Downsample by a ratio (approximately halve each dimension):

    >>> pool = nn.FractionalMaxPool2d(kernel_size=2, output_ratio=0.5)
    >>> x = lucid.ones((2, 32, 20, 20))
    >>> y = pool(x)
    >>> y.shape[0], y.shape[1]
    (2, 32)
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        output_size: int | tuple[int, int] | None = None,
        output_ratio: float | tuple[float, float] | None = None,
        return_indices: bool = False,
    ) -> None:
        """Initialise the FractionalMaxPool2d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.kernel_size = kernel_size
        self.output_size = output_size
        self.output_ratio = output_ratio
        self.return_indices = return_indices

    @override
    def forward(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Pool ``input`` of shape ``(N, C, H, W)``, or unbatched.

        Returns the pooled tensor, and with ``return_indices`` also each
        window maximum's flat position in the input.
        """
        return fractional_max_pool2d(
            input,
            kernel_size=self.kernel_size,
            output_size=self.output_size,
            output_ratio=self.output_ratio,
            return_indices=self.return_indices,
        )


class FractionalMaxPool3d(Module):
    r"""Applies fractional max-pooling over a 3-D volumetric feature map.

    Extends :class:`FractionalMaxPool2d` to three spatial dimensions
    (depth, height, width) following the same random fractional boundary
    scheme of Graham (2014).  Independent random partitions are sampled
    for each of the three axes:

    .. math::

        y[n,c,d,i,j] =
            \max_{\substack{
                z \in [r_d^{(d)},\, r_d^{(d+1)}) \\
                h \in [r_h^{(i)},\, r_h^{(i+1)}) \\
                w \in [r_w^{(j)},\, r_w^{(j+1)})
            }}
            x[n,c,z,h,w]

    Parameters
    ----------
    kernel_size : int or tuple[int, int, int]
        Size of the max pooling window at each sampled position.
    output_size : int or tuple[int, int, int] or None, optional
        Fixed target output size ``(D_out, H_out, W_out)``.  Exactly one
        of ``output_size`` and ``output_ratio`` must be given.
    output_ratio : float or tuple[float, float, float] or None, optional
        Target size as a fraction of each input dimension.  Values must
        lie in :math:`(0, 1)`.
    return_indices : bool, optional
        If ``True``, also return flat indices of the selected maxima.
        Default: ``False``.

    Attributes
    ----------
    kernel_size : int or tuple[int, int, int]
    output_size : int or tuple[int, int, int] or None
    output_ratio : float or tuple[float, float, float] or None
    return_indices : bool

    Shape
    -----
    - Input:  ``(N, C, D_in, H_in, W_in)``
    - Output: ``(N, C, D_out, H_out, W_out)``

    Notes
    -----
    - The random partition is drawn independently per batch-channel pair,
      so the output shape is deterministic (fixed by ``output_size`` or
      ``output_ratio``) but the selected voxels differ across calls during
      training.
    - The stochastic sampling implicitly augments the data, which can
      reduce over-fitting on volumetric datasets.
    - Delegates to
      :func:`lucid.nn.functional.pooling.fractional_max_pool3d`.

    Examples
    --------
    Volumetric fractional pooling with fixed output:

    >>> import lucid
    >>> import lucid.nn as nn
    >>> pool = nn.FractionalMaxPool3d(kernel_size=2, output_size=(4, 4, 4))
    >>> x = lucid.ones((1, 32, 8, 8, 8))
    >>> y = pool(x)
    >>> y.shape
    (1, 32, 4, 4, 4)

    Using output_ratio for proportional downsampling:

    >>> pool = nn.FractionalMaxPool3d(kernel_size=2, output_ratio=0.5)
    >>> x = lucid.ones((2, 16, 12, 12, 12))
    >>> y = pool(x)
    >>> y.shape[0], y.shape[1]
    (2, 16)
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int, int],
        output_size: int | tuple[int, int, int] | None = None,
        output_ratio: float | tuple[float, float, float] | None = None,
        return_indices: bool = False,
    ) -> None:
        """Initialise the FractionalMaxPool3d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.kernel_size = kernel_size
        self.output_size = output_size
        self.output_ratio = output_ratio
        self.return_indices = return_indices

    @override
    def forward(self, input: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Pool ``input`` of shape ``(N, C, D, H, W)``, or unbatched.

        Returns the pooled tensor, and with ``return_indices`` also each
        window maximum's flat position in the input.
        """
        return fractional_max_pool3d(
            input,
            kernel_size=self.kernel_size,
            output_size=self.output_size,
            output_ratio=self.output_ratio,
            return_indices=self.return_indices,
        )
