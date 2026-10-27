"""
Flatten, Unflatten, Unfold, and Fold modules.
"""

from typing import override

from lucid._tensor.tensor import Tensor
from lucid._types import _Size2d
from lucid.nn.module import Module
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap


class Flatten(Module):
    r"""Flatten a contiguous range of dimensions into a single dimension.

    Given an input tensor of shape :math:`(d_0, d_1, \dots, d_{n-1})`,
    ``Flatten`` collapses dimensions ``start_dim`` through ``end_dim``
    (inclusive) into one dimension whose size is the product of the
    collapsed sizes:

    .. math::

        \text{output size at } \texttt{start\_dim}
        = \prod_{i=\texttt{start\_dim}}^{\texttt{end\_dim}} d_i

    All dimensions outside the range are left unchanged.

    Parameters
    ----------
    start_dim : int, optional
        First dimension to flatten (default ``1``).  Negative indices are
        supported and follow the standard Python convention
        (``-1`` is the last dimension).
    end_dim : int, optional
        Last dimension to flatten, inclusive (default ``-1``, the last
        dimension).  Negative indices are supported.

    Attributes
    ----------
    start_dim : int
        Stored value of the ``start_dim`` constructor argument.
    end_dim : int
        Stored value of the ``end_dim`` constructor argument.

    Shape
    -----
    - **Input:** :math:`(N, d_1, d_2, \dots, d_k)` — any number of
      dimensions.
    - **Output** (defaults ``start_dim=1``, ``end_dim=-1``):
      :math:`(N, d_1 \cdot d_2 \cdots d_k)`.

    Notes
    -----
    - The most common use case is between convolutional feature extraction
      and a ``Linear`` classifier: spatial dimensions :math:`(C, H, W)` are
      merged into a flat vector of length :math:`C \cdot H \cdot W`.
    - Setting ``start_dim=0`` flattens the batch dimension as well — use
      with caution.
    - The operation is backed by the C++ engine's ``flatten`` kernel and
      is fully differentiable through autograd.

    Examples
    --------
    **Typical CNN → Linear transition (default behaviour):**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> flat = nn.Flatten()           # start_dim=1, end_dim=-1
    >>> x = lucid.zeros(8, 3, 32, 32)
    >>> flat(x).shape                 # 3*32*32 = 3072
    (8, 3072)

    **Flatten only the spatial dimensions, keeping channels separate:**

    >>> flat_hw = nn.Flatten(start_dim=2, end_dim=3)
    >>> x = lucid.zeros(8, 16, 14, 14)
    >>> flat_hw(x).shape              # 14*14 = 196
    (8, 16, 196)

    **Inside a Sequential pipeline:**

    >>> model = nn.Sequential(
    ...     nn.Conv2d(3, 64, kernel_size=3, padding=1),
    ...     nn.ReLU(),
    ...     nn.AdaptiveAvgPool2d((1, 1)),
    ...     nn.Flatten(),              # (N, 64, 1, 1) -> (N, 64)
    ...     nn.Linear(64, 10),
    ... )
    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        """Initialise the Flatten module. See the class docstring for parameter semantics."""
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Flatten (or unflatten) the specified dimensions of the input.

        Parameters
        ----------
        input : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Tensor with the configured dimensions flattened or unflattened.
        """
        return _wrap(_C_engine.flatten(_unwrap(input), self.start_dim, self.end_dim))

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"


class Unflatten(Module):
    r"""Expand one dimension of a tensor into multiple dimensions.

    ``Unflatten`` is the inverse of ``Flatten``: it takes a single dimension
    of the input and splits it into the shape given by ``unflattened_size``.
    The product of ``unflattened_size`` must equal the size of the target
    dimension.

    .. math::

        \text{If } x.\text{shape}[\texttt{dim}] = \prod_i s_i
        \quad \text{then} \quad
        \text{output}.\text{shape}[\texttt{dim}:\texttt{dim}+k]
        = (s_0, s_1, \dots, s_{k-1})

    where :math:`k = \text{len(unflattened\_size)}`.

    Parameters
    ----------
    dim : int
        The dimension to expand.  Negative indices are supported.
    unflattened_size : tuple[int, ...]
        The target shape for the expanded dimension.  The product of all
        elements must equal ``x.shape[dim]``.

    Attributes
    ----------
    dim : int
        Stored value of the ``dim`` constructor argument.
    unflattened_size : tuple[int, ...]
        Stored target shape for the expanded dimension.

    Shape
    -----
    - **Input:** :math:`(\dots, d, \dots)` where :math:`d` is at position
      ``dim``.
    - **Output:** :math:`(\dots, s_0, s_1, \dots, s_{k-1}, \dots)` where
      :math:`s_0 \cdot s_1 \cdots s_{k-1} = d`.

    Notes
    -----
    - Internally implemented via a reshape — no data is copied.
    - A common pattern is ``Flatten`` in the encoder and ``Unflatten`` in
      the decoder to reconstruct spatial structure from a flat bottleneck.

    Examples
    --------
    **Reconstruct (C, H, W) spatial structure from a flat feature vector:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> unflat = nn.Unflatten(dim=1, unflattened_size=(64, 4, 4))
    >>> x = lucid.zeros(8, 1024)      # 64*4*4 = 1024
    >>> unflat(x).shape
    (8, 64, 4, 4)

    **Paired Flatten / Unflatten round-trip (encoder–decoder bottleneck):**

    >>> encoder = nn.Sequential(
    ...     nn.Conv2d(1, 16, 3, padding=1),
    ...     nn.ReLU(),
    ...     nn.Flatten(start_dim=1),   # (N, 16, H, W) -> (N, 16*H*W)
    ...     nn.Linear(16 * 28 * 28, 128),
    ... )
    >>> decoder = nn.Sequential(
    ...     nn.Linear(128, 16 * 28 * 28),
    ...     nn.Unflatten(1, (16, 28, 28)),
    ...     nn.Conv2d(16, 1, 3, padding=1),
    ... )
    """

    def __init__(self, dim: int, unflattened_size: tuple[int, ...]) -> None:
        """Initialise the Unflatten module. See the class docstring for parameter semantics."""
        super().__init__()
        self.dim = dim
        self.unflattened_size = unflattened_size

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Flatten (or unflatten) the specified dimensions of the input.

        Parameters
        ----------
        input : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Tensor with the configured dimensions flattened or unflattened.
        """
        return input.unflatten(self.dim, self.unflattened_size)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"dim={self.dim}, unflattened_size={self.unflattened_size}"


class Unfold(Module):
    r"""Extract sliding local blocks (patches) from a batched 4-D input tensor.

    ``Unfold`` performs the *im2col* operation: it tiles a sliding window of
    shape :math:`(k_H, k_W)` across the spatial dimensions of the input and
    stacks all window contents into columns, producing a 3-D output.

    .. math::

        \text{input:}  \quad (N,\, C,\, H,\, W)
        \;\longrightarrow\;
        \text{output:} \quad (N,\, C \cdot k_H \cdot k_W,\, L)

    where :math:`L` is the total number of windows (blocks):

    .. math::

        L = \left\lfloor \frac{H + 2p_H - d_H(k_H - 1) - 1}{s_H} + 1 \right\rfloor
            \times
            \left\lfloor \frac{W + 2p_W - d_W(k_W - 1) - 1}{s_W} + 1 \right\rfloor

    The ``Fold`` module performs the inverse operation (col2im).

    Parameters
    ----------
    kernel_size : int or tuple[int, int]
        Size of the sliding window :math:`(k_H, k_W)`.  A single ``int``
        is broadcast to both dimensions.
    dilation : int or tuple[int, int], optional
        Spacing between kernel elements :math:`(d_H, d_W)` (default ``1``).
        Dilation ``> 1`` corresponds to an atrous (dilated) window.
    padding : int or tuple[int, int], optional
        Zero-padding added to both sides of each spatial dimension
        :math:`(p_H, p_W)` (default ``0``).
    stride : int or tuple[int, int], optional
        Stride of the sliding window :math:`(s_H, s_W)` (default ``1``).

    Attributes
    ----------
    kernel_size : int or tuple[int, int]
        Stored value of the ``kernel_size`` constructor argument.
    dilation : int or tuple[int, int]
        Stored value of the ``dilation`` constructor argument.
    padding : int or tuple[int, int]
        Stored value of the ``padding`` constructor argument.
    stride : int or tuple[int, int]
        Stored value of the ``stride`` constructor argument.

    Shape
    -----
    - **Input:** :math:`(N, C, H, W)` or :math:`(C, H, W)`.
    - **Output:** :math:`(N, C \cdot k_H \cdot k_W, L)` or
      :math:`(C \cdot k_H \cdot k_W, L)`.

    Notes
    -----
    - This module wraps ``nn.functional.unfold``.
    - A manual convolution can be implemented as
      ``(weight.view(C_out, -1) @ Unfold(kH, kW)(x)).view(N, C_out, L_H, L_W)``.
    - In Vision Transformers (ViT) ``Unfold`` is used to extract non-overlapping
      patches before the patch-embedding linear layer (``stride == kernel_size``).

    Examples
    --------
    **Patch extraction for a Vision Transformer-style encoder:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> patch_size = 16
    >>> unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
    >>> x = lucid.zeros(2, 3, 224, 224)
    >>> patches = unfold(x)
    >>> patches.shape   # 768 = 3*16*16, 196 = (224//16)^2
    (2, 768, 196)

    **Im2col for manual convolution:**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> unfold = nn.Unfold(kernel_size=3, padding=1)
    >>> x = lucid.randn(1, 1, 5, 5)
    >>> cols = unfold(x)
    >>> cols.shape   # 9 = 1*3*3, 25 = 5*5 windows (stride=1, pad=1)
    (1, 9, 25)
    """

    def __init__(
        self,
        kernel_size: _Size2d,
        dilation: _Size2d = 1,
        padding: _Size2d = 0,
        stride: _Size2d = 1,
    ) -> None:
        """Initialise the Unfold module. See the class docstring for parameter semantics."""
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Flatten (or unflatten) the specified dimensions of the input.

        Parameters
        ----------
        input : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Tensor with the configured dimensions flattened or unflattened.
        """
        from lucid.nn.functional.sampling import unfold as _unfold

        return _unfold(input, self.kernel_size, self.dilation, self.padding, self.stride)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            f"padding={self.padding}, stride={self.stride}"
        )


class Fold(Module):
    r"""Combine an array of sliding local blocks back into a batched tensor.

    ``Fold`` is the inverse of ``Unfold``: it reconstructs a spatial tensor
    from its column representation produced by the *im2col* (``Unfold``)
    operation.  This is also known as *col2im*.

    When multiple blocks overlap a single output position, their contributions
    are **summed** (accumulated), not averaged.  Use a companion ``Fold`` of
    all-ones to compute the overlap count if you need average pooling semantics.

    .. math::

        \text{input:}  \quad (N,\, C \cdot k_H \cdot k_W,\, L)
        \;\longrightarrow\;
        \text{output:} \quad (N,\, C,\, H_{\text{out}},\, W_{\text{out}})

    where :math:`H_{\text{out}}` and :math:`W_{\text{out}}` are given by
    ``output_size`` and the relationship

    .. math::

        L = \left\lfloor
              \frac{H_{\text{out}} + 2p_H - d_H(k_H-1) - 1}{s_H} + 1
            \right\rfloor
            \times
            \left\lfloor
              \frac{W_{\text{out}} + 2p_W - d_W(k_W-1) - 1}{s_W} + 1
            \right\rfloor

    must hold consistently with the ``Unfold`` parameters used.

    Parameters
    ----------
    output_size : int or tuple[int, int]
        Desired spatial shape :math:`(H_{\text{out}}, W_{\text{out}})` of the
        output tensor (excluding batch and channel dimensions).  A single
        ``int`` is broadcast to both dimensions.
    kernel_size : int or tuple[int, int]
        Size of the sliding window :math:`(k_H, k_W)` (must match the
        ``Unfold`` that produced the input).
    dilation : int or tuple[int, int], optional
        Dilation of the kernel :math:`(d_H, d_W)` (default ``1``).
    padding : int or tuple[int, int], optional
        Zero-padding that was applied during ``Unfold`` :math:`(p_H, p_W)`
        (default ``0``).
    stride : int or tuple[int, int], optional
        Stride used during ``Unfold`` :math:`(s_H, s_W)` (default ``1``).

    Attributes
    ----------
    output_size : tuple[int, int]
        Normalised ``(H_out, W_out)`` stored as a 2-tuple.
    kernel_size : tuple[int, int]
        Normalised kernel size stored as a 2-tuple.
    dilation : tuple[int, int]
        Normalised dilation stored as a 2-tuple.
    padding : tuple[int, int]
        Normalised padding stored as a 2-tuple.
    stride : tuple[int, int]
        Normalised stride stored as a 2-tuple.

    Shape
    -----
    - **Input:** :math:`(N, C \cdot k_H \cdot k_W, L)` or
      :math:`(C \cdot k_H \cdot k_W, L)`.
    - **Output:** :math:`(N, C, H_{\text{out}}, W_{\text{out}})` or
      :math:`(C, H_{\text{out}}, W_{\text{out}})`.

    Notes
    -----
    - Overlapping patches **accumulate** (sum) their contributions; this is
      the correct adjoint of ``Unfold`` for gradient computation.
    - To reconstruct the average value at each position, fold a tensor of
      ones with the same parameters and divide element-wise.
    - Backed by the C++ fold (col2im) op via ``nn.functional.fold``.

    Examples
    --------
    **Round-trip Unfold → Fold (non-overlapping patches):**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> H, W, kH, kW = 16, 16, 4, 4
    >>> unfold = nn.Unfold(kernel_size=(kH, kW), stride=(kH, kW))
    >>> fold   = nn.Fold(output_size=(H, W), kernel_size=(kH, kW), stride=(kH, kW))
    >>>
    >>> x       = lucid.randn(1, 3, H, W)
    >>> patches = unfold(x)          # (1, 48, 16)
    >>> x_hat   = fold(patches)      # (1, 3, 16, 16) — exact reconstruction
    >>>                               # (no overlap → no accumulation artefacts)

    **Attention-weighted patch reconstruction (overlapping):**

    >>> import lucid
    >>> import lucid.nn as nn
    >>>
    >>> unfold = nn.Unfold(kernel_size=3, padding=1)
    >>> fold   = nn.Fold(output_size=(8, 8), kernel_size=3, padding=1)
    >>> divisor_fold = nn.Fold(output_size=(8, 8), kernel_size=3, padding=1)
    >>>
    >>> x    = lucid.randn(1, 2, 8, 8)
    >>> cols = unfold(x)                        # (1, 18, 64)
    >>> # ... apply per-patch attention weights ...
    >>> ones = lucid.ones_like(cols)
    >>> out  = fold(cols) / divisor_fold(ones)  # average over overlaps
    """

    def __init__(
        self,
        output_size: _Size2d,
        kernel_size: _Size2d,
        dilation: _Size2d = 1,
        padding: _Size2d = 0,
        stride: _Size2d = 1,
    ) -> None:
        """Initialise the Fold module. See the class docstring for parameter semantics."""
        super().__init__()
        self.output_size = (
            output_size if isinstance(output_size, tuple) else (output_size, output_size)
        )
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)

    @override
    def forward(self, input: Tensor) -> Tensor:
        """Flatten (or unflatten) the specified dimensions of the input.

        Parameters
        ----------
        input : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Tensor with the configured dimensions flattened or unflattened.
        """
        from lucid.nn.functional.sampling import fold as _fold

        oH, oW = self.output_size
        kH, kW = self.kernel_size
        dH, dW = self.dilation
        pH, pW = self.padding
        sH, sW = self.stride
        return _fold(
            input,
            (oH, oW),
            (kH, kW),
            dilation=(dH, dW),
            padding=(pH, pW),
            stride=(sH, sW),
        )

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"output_size={self.output_size}, kernel_size={self.kernel_size}, "
            f"dilation={self.dilation}, padding={self.padding}, stride={self.stride}"
        )
