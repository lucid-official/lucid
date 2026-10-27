"""
Padding modules: Constant, Reflection, Replication, Zero padding.
"""

from typing import override

from lucid._tensor.tensor import Tensor
from lucid._types import _Size2d
from lucid.nn.module import Module
from lucid.nn.functional.sampling import pad


class _ConstantPadNd(Module):
    r"""Private base class for all constant-value padding modules.

    Stores the ``padding`` tuple and scalar ``value``, then delegates to
    ``nn.functional.pad`` with ``mode='constant'``.  Concrete subclasses
    set the class attribute ``_dims`` to the expected number of padding
    values (2 for 1-D, 4 for 2-D, 6 for 3-D) so that a scalar ``padding``
    argument can be broadcast to the correct length via ``_make_tuple``.

    Parameters
    ----------
    padding : int or tuple[int, ...]
        Padding sizes in the order expected by ``nn.functional.pad``
        (innermost dimension first: left, right for 1-D; left, right, top,
        bottom for 2-D; left, right, top, bottom, front, back for 3-D).
        A single ``int`` is broadcast to all ``_dims`` positions.
    value : float
        Constant fill value (e.g. ``0.0`` for zero padding).

    Attributes
    ----------
    padding : tuple[int, ...]
        Normalised padding tuple of length ``_dims``.
    value : float
        Stored fill value.
    """

    def __init__(self, padding: int | tuple[int, ...], value: float) -> None:
        """Initialise the _ConstantPadNd module. See the class docstring for parameter semantics."""
        super().__init__()
        self.padding = (
            padding if isinstance(padding, tuple) else _make_tuple(padding, self._dims)  # type: ignore[arg-type]
        )
        self.value = value

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Pad the input tensor according to the configured padding.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)`.

        Returns
        -------
        Tensor
            Padded tensor with spatial dimensions expanded by the configured
            padding amounts.
        """
        return pad(input, self.padding, mode="constant", value=self.value)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"padding={self.padding}, value={self.value}"


def _make_tuple(v: int, n: int) -> tuple[int, ...]:
    return (v,) * n


class ConstantPad1d(_ConstantPadNd):
    r"""Pad a 3-D tensor (N, C, L) along the last dimension with a constant.

    Adds ``padding[0]`` values on the left and ``padding[1]`` values on the
    right of the sequence dimension, filling each new position with
    ``value``.

    .. math::

        \text{output}[n, c, i]
        = \begin{cases}
            \text{value} & i < p_{\text{left}}
              \text{ or } i \geq L + p_{\text{left}} \\
            x[n, c,\; i - p_{\text{left}}] & \text{otherwise}
          \end{cases}

    Parameters
    ----------
    padding : int or tuple[int, int]
        ``(left, right)`` padding sizes.  A single ``int`` applies the same
        amount on both sides.
    value : float
        Constant fill value.

    Attributes
    ----------
    padding : tuple[int, int]
        Normalised ``(left, right)`` padding.
    value : float
        Fill value.

    Shape
    -----
    - **Input:** :math:`(N, C, L)`.
    - **Output:** :math:`(N, C, L + p_{\text{left}} + p_{\text{right}})`.

    Examples
    --------
    **Causal temporal padding (left-only) for a 1-D causal convolution:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> # Pad 2 timesteps on the left so no future context leaks
    >>> causal_pad = nn.ConstantPad1d(padding=(2, 0), value=0.0)
    >>> x = lucid.zeros(4, 16, 100)     # (N, C, T)
    >>> causal_pad(x).shape
    (4, 16, 102)

    **Symmetric padding with a non-zero fill value:**

    >>> pad = nn.ConstantPad1d(padding=3, value=-1.0)
    >>> x = lucid.zeros(2, 8, 50)
    >>> pad(x).shape
    (2, 8, 56)
    """

    _dims = 2

    def __init__(self, padding: _Size2d, value: float) -> None:
        """Initialise the ConstantPad1d module. See the class docstring for parameter semantics."""
        super().__init__(padding, value)


class ConstantPad2d(_ConstantPadNd):
    r"""Pad a 4-D tensor (N, C, H, W) on all four spatial sides with a constant.

    Adds padding around the height and width dimensions.  The ``padding``
    tuple follows the convention ``(left, right, top, bottom)``:

    .. math::

        \text{H}_{\text{out}} = H + p_{\text{top}} + p_{\text{bottom}}
        \qquad
        \text{W}_{\text{out}} = W + p_{\text{left}} + p_{\text{right}}

    Parameters
    ----------
    padding : int or tuple[int, int, int, int]
        ``(left, right, top, bottom)`` padding sizes.  A single ``int``
        applies the same amount on all four sides.
    value : float
        Constant fill value.

    Attributes
    ----------
    padding : tuple[int, int, int, int]
        Normalised ``(left, right, top, bottom)`` padding.
    value : float
        Fill value.

    Shape
    -----
    - **Input:** :math:`(N, C, H, W)`.
    - **Output:**
      :math:`(N, C, H + p_{\text{top}} + p_{\text{bottom}},
      W + p_{\text{left}} + p_{\text{right}})`.

    Examples
    --------
    **Uniform 2-pixel border padding:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> pad = nn.ConstantPad2d(padding=2, value=0.0)
    >>> x = lucid.zeros(1, 3, 28, 28)
    >>> pad(x).shape
    (1, 3, 32, 32)

    **Asymmetric padding (e.g. to adjust receptive field alignment):**

    >>> pad = nn.ConstantPad2d(padding=(1, 2, 0, 3), value=-999.0)
    >>> x = lucid.zeros(2, 16, 10, 10)
    >>> pad(x).shape    # H: 10+0+3=13,  W: 10+1+2=13
    (2, 16, 13, 13)
    """

    _dims = 4

    def __init__(self, padding: int | tuple[int, int, int, int], value: float) -> None:
        """Initialise the ConstantPad2d module. See the class docstring for parameter semantics."""
        super().__init__(padding, value)


class ConstantPad3d(_ConstantPadNd):
    r"""Pad a 5-D tensor (N, C, D, H, W) on all six faces with a constant.

    Adds padding around the depth, height, and width dimensions.  The
    ``padding`` tuple follows the convention
    ``(left, right, top, bottom, front, back)``:

    .. math::

        D_{\text{out}} &= D + p_{\text{front}} + p_{\text{back}} \\
        H_{\text{out}} &= H + p_{\text{top}} + p_{\text{bottom}} \\
        W_{\text{out}} &= W + p_{\text{left}} + p_{\text{right}}

    Parameters
    ----------
    padding : int or tuple[int, int, int, int, int, int]
        ``(left, right, top, bottom, front, back)`` padding sizes.  A single
        ``int`` applies the same amount on all six faces.
    value : float
        Constant fill value.

    Attributes
    ----------
    padding : tuple[int, int, int, int, int, int]
        Normalised 6-element padding tuple.
    value : float
        Fill value.

    Shape
    -----
    - **Input:** :math:`(N, C, D, H, W)`.
    - **Output:** :math:`(N, C, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})`.

    Examples
    --------
    **Uniform 1-voxel border around a volumetric tensor:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> pad = nn.ConstantPad3d(padding=1, value=0.0)
    >>> x = lucid.zeros(1, 4, 16, 16, 16)
    >>> pad(x).shape
    (1, 4, 18, 18, 18)

    **Pad only the depth axis (front/back) for temporal video data:**

    >>> pad = nn.ConstantPad3d(padding=(0, 0, 0, 0, 2, 2), value=0.0)
    >>> x = lucid.zeros(2, 3, 8, 32, 32)    # (N, C, T, H, W)
    >>> pad(x).shape
    (2, 3, 12, 32, 32)
    """

    _dims = 6

    def __init__(self, padding: int | tuple[int, int, int, int, int, int], value: float) -> None:
        """Initialise the ConstantPad3d module. See the class docstring for parameter semantics."""
        super().__init__(padding, value)


class ZeroPad1d(ConstantPad1d):
    r"""Pad a 3-D tensor (N, C, L) with zeros along the sequence dimension.

    Equivalent to ``ConstantPad1d(padding, value=0.0)``.  Zero-padding is
    the most common padding mode for 1-D convolutional networks because it
    introduces no spurious signal at the boundaries and is implicit in
    most convolution implementations.

    Parameters
    ----------
    padding : int or tuple[int, int]
        ``(left, right)`` padding sizes.  A single ``int`` pads equally on
        both sides.

    Attributes
    ----------
    padding : tuple[int, int]
        Normalised ``(left, right)`` padding.
    value : float
        Always ``0.0``.

    Shape
    -----
    - **Input:** :math:`(N, C, L)`.
    - **Output:** :math:`(N, C, L + p_{\text{left}} + p_{\text{right}})`.

    Examples
    --------
    **Same-padding for a 1-D convolution with kernel size 5:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> # kernel=5 → same-padding = (kernel-1)//2 = 2 on each side
    >>> pad = nn.ZeroPad1d(padding=2)
    >>> x = lucid.zeros(8, 32, 100)
    >>> pad(x).shape
    (8, 32, 104)

    **Asymmetric padding for causal convolution:**

    >>> causal = nn.ZeroPad1d(padding=(4, 0))
    >>> x = lucid.zeros(4, 16, 50)
    >>> causal(x).shape
    (4, 16, 54)
    """

    def __init__(self, padding: _Size2d) -> None:
        """Initialise the ZeroPad1d module. See the class docstring for parameter semantics."""
        super().__init__(padding, value=0.0)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"padding={self.padding}"


class ZeroPad2d(ConstantPad2d):
    r"""Pad a 4-D tensor (N, C, H, W) with zeros on all four spatial sides.

    Equivalent to ``ConstantPad2d(padding, value=0.0)``.  The standard
    padding mode used before 2-D convolutions to preserve spatial resolution
    (``'same'`` padding) or to prevent aliasing at image boundaries.

    Parameters
    ----------
    padding : int or tuple[int, int, int, int]
        ``(left, right, top, bottom)`` padding sizes.  A single ``int``
        pads equally on all four sides.

    Attributes
    ----------
    padding : tuple[int, int, int, int]
        Normalised ``(left, right, top, bottom)`` padding.
    value : float
        Always ``0.0``.

    Shape
    -----
    - **Input:** :math:`(N, C, H, W)`.
    - **Output:**
      :math:`(N, C, H + p_{\text{top}} + p_{\text{bottom}},
      W + p_{\text{left}} + p_{\text{right}})`.

    Examples
    --------
    **1-pixel zero border for a 3×3 convolution:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> pad = nn.ZeroPad2d(1)
    >>> x = lucid.zeros(2, 64, 32, 32)
    >>> pad(x).shape
    (2, 64, 34, 34)

    **Use inside a Sequential block before strided convolution:**

    >>> block = nn.Sequential(
    ...     nn.ZeroPad2d(padding=(0, 1, 0, 1)),   # asymmetric for stride-2
    ...     nn.Conv2d(32, 64, kernel_size=3, stride=2),
    ... )
    """

    def __init__(self, padding: int | tuple[int, int, int, int]) -> None:
        """Initialise the ZeroPad2d module. See the class docstring for parameter semantics."""
        super().__init__(padding, value=0.0)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"padding={self.padding}"


class ZeroPad3d(ConstantPad3d):
    r"""Pad a 5-D tensor (N, C, D, H, W) with zeros on all six faces.

    Equivalent to ``ConstantPad3d(padding, value=0.0)``.  Used before 3-D
    convolutions operating on volumetric data (medical imaging, video, point
    clouds) to maintain spatial dimensions.

    Parameters
    ----------
    padding : int or tuple[int, int, int, int, int, int]
        ``(left, right, top, bottom, front, back)`` padding sizes.  A single
        ``int`` pads equally on all six faces.

    Attributes
    ----------
    padding : tuple[int, int, int, int, int, int]
        Normalised 6-element padding tuple.
    value : float
        Always ``0.0``.

    Shape
    -----
    - **Input:** :math:`(N, C, D, H, W)`.
    - **Output:** :math:`(N, C, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})`.

    Examples
    --------
    **Same-padding for a 3×3×3 volumetric convolution:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> pad = nn.ZeroPad3d(1)           # 1 voxel on every face
    >>> x = lucid.zeros(1, 8, 16, 16, 16)
    >>> pad(x).shape
    (1, 8, 18, 18, 18)

    **Temporal-only padding for video (depth = time axis):**

    >>> pad = nn.ZeroPad3d(padding=(0, 0, 0, 0, 1, 1))
    >>> x = lucid.zeros(2, 3, 8, 112, 112)
    >>> pad(x).shape
    (2, 3, 10, 112, 112)
    """

    def __init__(self, padding: int | tuple[int, int, int, int, int, int]) -> None:
        """Initialise the ZeroPad3d module. See the class docstring for parameter semantics."""
        super().__init__(padding, value=0.0)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"padding={self.padding}"


class ReflectionPad1d(Module):
    r"""Pad a 3-D tensor (N, C, L) by reflecting at the sequence boundaries.

    Unlike constant padding, reflection padding mirrors the input signal
    itself at the boundary rather than filling with a fixed value.  For a
    left pad of size :math:`p`, the new values at positions
    :math:`0, 1, \dots, p-1` are set to :math:`x[\dots, p], x[\dots, p-1],
    \dots, x[\dots, 1]` respectively (the boundary pixel is excluded).

    .. math::

        \text{output}[n, c, i]
        = x\!\left[n, c,\; |i - p_{\text{left}}|\right]
        \quad \text{for } i < p_{\text{left}}

    Parameters
    ----------
    padding : int or tuple[int, int]
        ``(left, right)`` reflection padding sizes.  Each value must be
        strictly less than the corresponding input dimension.  A single
        ``int`` pads equally on both sides.

    Attributes
    ----------
    padding : tuple[int, int]
        Normalised ``(left, right)`` padding.

    Shape
    -----
    - **Input:** :math:`(N, C, L)`.
    - **Output:** :math:`(N, C, L + p_{\text{left}} + p_{\text{right}})`.

    Notes
    -----
    - Reflection padding avoids the abrupt discontinuity introduced by zero
      or constant padding, making it preferable for tasks sensitive to
      border artefacts (image style transfer, super-resolution).
    - The padding size must be strictly less than the input size on the
      padded dimension.

    Examples
    --------
    **Mirror the ends of an audio waveform to avoid boundary clicks:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> pad = nn.ReflectionPad1d(padding=4)
    >>> x = lucid.zeros(2, 1, 1024)     # (N, channels, T)
    >>> pad(x).shape
    (2, 1, 1032)

    **Asymmetric reflection for causal-ish temporal convolution:**

    >>> pad = nn.ReflectionPad1d(padding=(8, 0))
    >>> x = lucid.zeros(4, 16, 200)
    >>> pad(x).shape
    (4, 16, 208)
    """

    def __init__(self, padding: _Size2d) -> None:
        """Initialise the ReflectionPad1d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Pad the input tensor according to the configured padding.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)`.

        Returns
        -------
        Tensor
            Padded tensor with spatial dimensions expanded by the configured
            padding amounts.
        """
        return pad(input, self.padding, mode="reflect")

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"padding={self.padding}"


class ReflectionPad2d(Module):
    r"""Pad a 4-D tensor (N, C, H, W) by reflecting at all four spatial edges.

    Reflection padding mirrors the pixel values at each boundary, producing a
    smooth, artefact-free extension of the image.  This is particularly
    effective for convolutional image processing tasks where constant-value
    borders would introduce ringing or aliasing.

    The reflected value at an out-of-bounds index :math:`i` (with respect to
    input size :math:`S` and left-pad :math:`p`) is:

    .. math::

        \text{output}[\dots, i] = x[\dots,\; |i - p|]
        \quad (i < p)

    Parameters
    ----------
    padding : int or tuple[int, int, int, int]
        ``(left, right, top, bottom)`` reflection padding sizes.  Each value
        must be strictly less than the corresponding spatial input size.
        A single ``int`` pads equally on all four sides.

    Attributes
    ----------
    padding : tuple[int, int, int, int]
        Normalised ``(left, right, top, bottom)`` padding.

    Shape
    -----
    - **Input:** :math:`(N, C, H, W)`.
    - **Output:**
      :math:`(N, C, H + p_{\text{top}} + p_{\text{bottom}},
      W + p_{\text{left}} + p_{\text{right}})`.

    Notes
    -----
    - Widely used in Neural Style Transfer and image-to-image translation
      architectures (e.g. Johnson et al., 2016) to avoid border colour
      bleeding caused by zero padding.

    Examples
    --------
    **Style-transfer encoder with reflection padding:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> pad = nn.ReflectionPad2d(padding=3)
    >>> conv = nn.Conv2d(3, 64, kernel_size=7)
    >>> x = lucid.zeros(1, 3, 256, 256)
    >>> conv(pad(x)).shape    # 256 + 2*3 - 6 = 256 (same-size output)
    (1, 64, 256, 256)

    **Asymmetric reflection:**

    >>> pad = nn.ReflectionPad2d(padding=(1, 2, 0, 3))
    >>> x = lucid.zeros(2, 8, 32, 32)
    >>> pad(x).shape
    (2, 8, 35, 35)
    """

    def __init__(self, padding: int | tuple[int, int, int, int]) -> None:
        """Initialise the ReflectionPad2d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.padding = (padding,) * 4 if isinstance(padding, int) else tuple(padding)

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Pad the input tensor according to the configured padding.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)`.

        Returns
        -------
        Tensor
            Padded tensor with spatial dimensions expanded by the configured
            padding amounts.
        """
        return pad(input, self.padding, mode="reflect")

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"padding={self.padding}"


class ReplicationPad1d(Module):
    r"""Pad a 3-D tensor (N, C, L) by replicating the edge values.

    Replication padding fills out-of-bounds positions with the value of the
    nearest valid element (edge replication):

    .. math::

        \text{output}[n, c, i]
        = x\!\left[n, c,\; \text{clip}(i - p_{\text{left}},\; 0,\; L-1)\right]

    The leftmost valid element is repeated to the left; the rightmost valid
    element is repeated to the right.

    Parameters
    ----------
    padding : int or tuple[int, int]
        ``(left, right)`` replication padding sizes.  A single ``int`` pads
        equally on both sides.

    Attributes
    ----------
    padding : tuple[int, int]
        Normalised ``(left, right)`` padding.

    Shape
    -----
    - **Input:** :math:`(N, C, L)`.
    - **Output:** :math:`(N, C, L + p_{\text{left}} + p_{\text{right}})`.

    Notes
    -----
    - Replication padding is particularly useful when the input represents a
      signal whose boundary should be extended with the last known value
      (e.g. padding a time series at the end with its final sample).
    - Unlike reflection padding, replication padding is valid even for
      padding sizes equal to or larger than the input length.

    Examples
    --------
    **Edge-extend a short time series before a convolution:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> pad = nn.ReplicationPad1d(padding=5)
    >>> x = lucid.zeros(2, 4, 20)
    >>> pad(x).shape
    (2, 4, 30)

    **One-sided extension (right-only):**

    >>> pad = nn.ReplicationPad1d(padding=(0, 10))
    >>> x = lucid.zeros(1, 8, 50)
    >>> pad(x).shape
    (1, 8, 60)
    """

    def __init__(self, padding: _Size2d) -> None:
        """Initialise the ReplicationPad1d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Pad the input tensor according to the configured padding.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)`.

        Returns
        -------
        Tensor
            Padded tensor with spatial dimensions expanded by the configured
            padding amounts.
        """
        return pad(input, self.padding, mode="replicate")

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"padding={self.padding}"


class ReplicationPad2d(Module):
    r"""Pad a 4-D tensor (N, C, H, W) by replicating the edge pixels.

    Each out-of-bounds position is filled with the value of the nearest
    in-bounds pixel.  Corner out-of-bounds positions replicate the nearest
    corner pixel.

    .. math::

        \text{output}[n, c, i, j]
        = x\!\left[n, c,\;
            \text{clip}(i - p_t, 0, H-1),\;
            \text{clip}(j - p_l, 0, W-1)
          \right]

    Parameters
    ----------
    padding : int or tuple[int, int, int, int]
        ``(left, right, top, bottom)`` replication padding sizes.  A single
        ``int`` pads equally on all four sides.

    Attributes
    ----------
    padding : tuple[int, int, int, int]
        Normalised ``(left, right, top, bottom)`` padding.

    Shape
    -----
    - **Input:** :math:`(N, C, H, W)`.
    - **Output:**
      :math:`(N, C, H + p_{\text{top}} + p_{\text{bottom}},
      W + p_{\text{left}} + p_{\text{right}})`.

    Notes
    -----
    - Replication padding is preferred over zero padding in networks that
      process images with meaningful content close to the borders (e.g.
      panoramic or satellite imagery) because it avoids introducing a dark
      border artefact that the network would need to learn to ignore.

    Examples
    --------
    **Replicate borders before a 5×5 convolution:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> pad = nn.ReplicationPad2d(padding=2)
    >>> conv = nn.Conv2d(3, 32, kernel_size=5)
    >>> x = lucid.zeros(1, 3, 64, 64)
    >>> conv(pad(x)).shape    # same-size output
    (1, 32, 64, 64)

    **Asymmetric replication:**

    >>> pad = nn.ReplicationPad2d(padding=(0, 0, 1, 1))   # top and bottom only
    >>> x = lucid.zeros(4, 16, 30, 30)
    >>> pad(x).shape
    (4, 16, 32, 30)
    """

    def __init__(self, padding: int | tuple[int, int, int, int]) -> None:
        """Initialise the ReplicationPad2d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.padding = (padding,) * 4 if isinstance(padding, int) else tuple(padding)

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Pad the input tensor according to the configured padding.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)`.

        Returns
        -------
        Tensor
            Padded tensor with spatial dimensions expanded by the configured
            padding amounts.
        """
        return pad(input, self.padding, mode="replicate")

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"padding={self.padding}"


class ReplicationPad3d(Module):
    r"""Pad a 5-D tensor (N, C, D, H, W) by replicating the boundary voxels.

    Extends the volume by repeating the nearest in-bounds voxel value along
    each of the six faces, with corners and edges replicated from the nearest
    valid voxel.

    Parameters
    ----------
    padding : int or tuple[int, int, int, int, int, int]
        ``(left, right, top, bottom, front, back)`` replication padding sizes.
        A single ``int`` pads equally on all six faces.

    Attributes
    ----------
    padding : tuple[int, int, int, int, int, int]
        Normalised 6-element padding tuple.

    Shape
    -----
    - **Input:** :math:`(N, C, D, H, W)`.
    - **Output:** :math:`(N, C, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})`.

    Examples
    --------
    **Replicate 1 voxel on every face of a medical volume:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> pad = nn.ReplicationPad3d(padding=1)
    >>> x = lucid.zeros(1, 1, 64, 64, 64)
    >>> pad(x).shape
    (1, 1, 66, 66, 66)

    **Pad only temporal (depth) faces for video models:**

    >>> pad = nn.ReplicationPad3d(padding=(0, 0, 0, 0, 2, 2))
    >>> x = lucid.zeros(2, 3, 8, 112, 112)
    >>> pad(x).shape
    (2, 3, 12, 112, 112)
    """

    def __init__(self, padding: int | tuple[int, int, int, int, int, int]) -> None:
        """Initialise the ReplicationPad3d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.padding = (padding,) * 6 if isinstance(padding, int) else tuple(padding)

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Pad the input tensor according to the configured padding.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)`.

        Returns
        -------
        Tensor
            Padded tensor with spatial dimensions expanded by the configured
            padding amounts.
        """
        return pad(input, self.padding, mode="replicate")

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"padding={self.padding}"


class ReflectionPad3d(Module):
    r"""Pad a 5-D tensor (N, C, D, H, W) by reflecting at the volume boundaries.

    Applies reflection padding simultaneously across the depth, height, and
    width dimensions of a volumetric tensor.  Each out-of-bounds voxel
    receives the value of the voxel that is its mirror image with respect to
    the nearest valid face, excluding the boundary voxel itself.

    Parameters
    ----------
    padding : int or tuple[int, int, int, int, int, int]
        ``(left, right, top, bottom, front, back)`` reflection padding sizes.
        Each value must be strictly less than the size of the corresponding
        input dimension.  A single ``int`` pads equally on all six faces.

    Attributes
    ----------
    padding : tuple[int, int, int, int, int, int]
        Normalised 6-element padding tuple.

    Shape
    -----
    - **Input:** :math:`(N, C, D, H, W)`.
    - **Output:** :math:`(N, C, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})`.

    Notes
    -----
    - Reflection padding in 3-D is useful for volumetric image analysis
      (CT/MRI scans) where zero-padding at volume boundaries would create
      artificial low-density regions that the network might spuriously learn
      to avoid.

    Examples
    --------
    **Pad a small 3-D patch for a 3×3×3 volumetric convolution:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> pad = nn.ReflectionPad3d(padding=1)
    >>> x = lucid.zeros(1, 4, 16, 16, 16)
    >>> pad(x).shape
    (1, 4, 18, 18, 18)

    **Asymmetric 3-D reflection:**

    >>> pad = nn.ReflectionPad3d(padding=(1, 1, 2, 2, 0, 0))
    >>> x = lucid.zeros(2, 8, 10, 10, 10)
    >>> pad(x).shape
    (2, 8, 10, 14, 12)
    """

    def __init__(self, padding: int | tuple[int, int, int, int, int, int]) -> None:
        """Initialise the ReflectionPad3d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.padding = (padding,) * 6 if isinstance(padding, int) else tuple(padding)

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Pad the input tensor according to the configured padding.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)`.

        Returns
        -------
        Tensor
            Padded tensor with spatial dimensions expanded by the configured
            padding amounts.
        """
        return pad(input, self.padding, mode="reflect")

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"padding={self.padding}"


class CircularPad1d(Module):
    r"""Pad a 3-D tensor (N, C, L) by wrapping the sequence around itself.

    Circular (or periodic) padding treats the input as if it were a
    circular buffer: values from the right end of the sequence appear on
    the left, and values from the left end appear on the right.

    .. math::

        \text{output}[n, c, i]
        = x\!\left[n, c,\; (i - p_{\text{left}}) \bmod L\right]

    Parameters
    ----------
    padding : int or tuple[int, int]
        ``(left, right)`` circular padding sizes.  A single ``int`` pads
        equally on both sides.  Values may be larger than the input length
        (the wrap-around simply repeats).

    Attributes
    ----------
    padding : tuple[int, int]
        Normalised ``(left, right)`` padding.

    Shape
    -----
    - **Input:** :math:`(N, C, L)`.
    - **Output:** :math:`(N, C, L + p_{\text{left}} + p_{\text{right}})`.

    Notes
    -----
    - Circular padding is the natural choice for signals that are inherently
      periodic: audio spectral features, astronomical data, or any domain
      with wrap-around semantics.
    - For 1-D convolutions on periodic signals it ensures that the
      convolution filter sees a seamless boundary, unlike zero or reflection
      padding.

    Examples
    --------
    **Periodic extension of an audio spectrogram feature:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> pad = nn.CircularPad1d(padding=8)
    >>> x = lucid.zeros(4, 64, 256)
    >>> pad(x).shape
    (4, 64, 272)

    **Asymmetric wrap-around:**

    >>> pad = nn.CircularPad1d(padding=(3, 0))
    >>> x = lucid.zeros(2, 16, 100)
    >>> pad(x).shape
    (2, 16, 103)
    """

    def __init__(self, padding: _Size2d) -> None:
        """Initialise the CircularPad1d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Pad the input tensor according to the configured padding.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)`.

        Returns
        -------
        Tensor
            Padded tensor with spatial dimensions expanded by the configured
            padding amounts.
        """
        return pad(input, self.padding, mode="circular")

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"padding={self.padding}"


class CircularPad2d(Module):
    r"""Pad a 4-D tensor (N, C, H, W) with wrap-around in both spatial axes.

    Circular padding in 2-D treats both the height and width axes as periodic:
    pixels from the right column wrap to the left, and pixels from the bottom
    row wrap to the top.

    .. math::

        \text{output}[n, c, i, j]
        = x\!\left[n, c,\;
            (i - p_{\text{top}}) \bmod H,\;
            (j - p_{\text{left}}) \bmod W
          \right]

    Parameters
    ----------
    padding : int or tuple[int, int, int, int]
        ``(left, right, top, bottom)`` circular padding sizes.  A single
        ``int`` pads equally on all four sides.

    Attributes
    ----------
    padding : tuple[int, int, int, int]
        Normalised ``(left, right, top, bottom)`` padding.

    Shape
    -----
    - **Input:** :math:`(N, C, H, W)`.
    - **Output:**
      :math:`(N, C, H + p_{\text{top}} + p_{\text{bottom}},
      W + p_{\text{left}} + p_{\text{right}})`.

    Notes
    -----
    - Useful for processing panoramic (360°) images where left and right edges
      are physically adjacent, or for convolutional processing of spherical
      data projected to a 2-D grid.
    - Combining circular padding with a stride-2 convolution on a toroidal
      feature map maintains the topology of the feature space.

    Examples
    --------
    **Seamless tiling convolution for a panoramic image:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> pad = nn.CircularPad2d(padding=(2, 2, 0, 0))  # horizontal wrap only
    >>> x = lucid.zeros(1, 3, 256, 512)
    >>> pad(x).shape
    (1, 3, 256, 516)

    **Uniform circular border:**

    >>> pad = nn.CircularPad2d(padding=1)
    >>> x = lucid.zeros(4, 16, 32, 32)
    >>> pad(x).shape
    (4, 16, 34, 34)
    """

    def __init__(self, padding: int | tuple[int, int, int, int]) -> None:
        """Initialise the CircularPad2d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.padding = (padding,) * 4 if isinstance(padding, int) else tuple(padding)

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Pad the input tensor according to the configured padding.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)`.

        Returns
        -------
        Tensor
            Padded tensor with spatial dimensions expanded by the configured
            padding amounts.
        """
        return pad(input, self.padding, mode="circular")

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"padding={self.padding}"


class CircularPad3d(Module):
    r"""Pad a 5-D tensor (N, C, D, H, W) with wrap-around in all three axes.

    Treats the depth, height, and width dimensions as periodic: the volume
    wraps around in all three spatial directions, like a 3-D torus.

    .. math::

        \text{output}[n, c, k, i, j]
        = x\!\left[n, c,\;
            (k - p_f) \bmod D,\;
            (i - p_t) \bmod H,\;
            (j - p_l) \bmod W
          \right]

    Parameters
    ----------
    padding : int or tuple[int, int, int, int, int, int]
        ``(left, right, top, bottom, front, back)`` circular padding sizes.
        A single ``int`` pads equally on all six faces.

    Attributes
    ----------
    padding : tuple[int, int, int, int, int, int]
        Normalised 6-element padding tuple.

    Shape
    -----
    - **Input:** :math:`(N, C, D, H, W)`.
    - **Output:** :math:`(N, C, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})`.

    Notes
    -----
    - Applicable to volumetric data with inherent periodicity, such as
      crystallographic unit cells or periodic boundary condition simulations
      in computational physics.

    Examples
    --------
    **Toroidal padding for a periodic 3-D simulation box:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> pad = nn.CircularPad3d(padding=1)
    >>> x = lucid.zeros(1, 4, 32, 32, 32)
    >>> pad(x).shape
    (1, 4, 34, 34, 34)

    **Temporal wrap-around for a looping video sequence:**

    >>> pad = nn.CircularPad3d(padding=(0, 0, 0, 0, 4, 4))  # time axis only
    >>> x = lucid.zeros(2, 3, 16, 112, 112)
    >>> pad(x).shape
    (2, 3, 24, 112, 112)
    """

    def __init__(self, padding: int | tuple[int, int, int, int, int, int]) -> None:
        """Initialise the CircularPad3d module. See the class docstring for parameter semantics."""
        super().__init__()
        self.padding = (padding,) * 6 if isinstance(padding, int) else tuple(padding)

    @override
    def forward(self, input: Tensor) -> Tensor:
        r"""Pad the input tensor according to the configured padding.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape :math:`(N, C, *)`.

        Returns
        -------
        Tensor
            Padded tensor with spatial dimensions expanded by the configured
            padding amounts.
        """
        return pad(input, self.padding, mode="circular")

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"padding={self.padding}"
