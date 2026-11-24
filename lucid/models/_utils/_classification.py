"""Image-classification–specific utility modules.

Shared building blocks for classification architectures that do not
belong in any single model family's private namespace.
"""

from typing import override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor


class LayerScale(nn.Module):
    """Per-channel learnable scaling applied to a residual branch output.

    Introduced in ConvNeXt (Liu et al., 2022) and used by InceptionNeXt
    and other modern classification architectures.  The parameter ``gamma``
    is initialised to a small constant (typically 1e-6) so that at the
    start of training every block behaves approximately as an identity
    mapping.

    For ``(B, C)``-shaped inputs ``gamma`` is broadcast directly.
    For ``(B, C, H, W)``-shaped inputs ``gamma`` is reshaped to
    ``(1, C, 1, 1)`` so it scales each channel independently across the
    spatial dimensions.

    Parameters
    ----------
    dim : int
        Number of channels :math:`C`.
    init_value : float, optional
        Initial value for every entry of ``gamma``.  Default ``1e-6``
        per the ConvNeXt training recipe — small enough that the block
        starts near identity and large enough that gradients flow.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._classification import LayerScale
    >>> ls = LayerScale(dim=96)
    >>> x = lucid.randn(2, 96, 7, 7)
    >>> ls(x).shape
    (2, 96, 7, 7)

    References
    ----------
    .. [1] Liu et al., *A ConvNet for the 2020s*, CVPR 2022.
    """

    def __init__(self, dim: int, init_value: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(lucid.full((dim,), init_value))

    @override
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            # (B, C, H, W) — reshape gamma to broadcast over spatial dims
            g: Tensor = self.gamma.reshape(1, -1, 1, 1)
        else:
            g = self.gamma
        return x * g


class DropPath(nn.Module):
    """Stochastic depth per sample (Huang et al., 2016 "Deep Networks with
    Stochastic Depth").

    Drops the entire residual branch of a sample with probability
    ``drop_prob`` during training; surviving samples are scaled by
    ``1 / (1 - drop_prob)`` so the expected output matches identity.  Acts
    as identity at inference and when ``drop_prob == 0``.

    Used by every modern classification architecture trained with the
    "stochastic depth" recipe — ConvNeXt, Swin, EfficientFormer, MaxViT,
    DeiT, and so on.

    Parameters
    ----------
    drop_prob : float, optional
        Probability of dropping the entire residual branch.  In most
        architectures this scales linearly with depth (deeper blocks
        see higher drop rates).  Must satisfy ``0 <= drop_prob < 1``.
        Default ``0.0`` (no dropping).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._classification import DropPath
    >>> dp = DropPath(drop_prob=0.1).train()
    >>> x = lucid.randn(8, 96, 7, 7)
    >>> y = dp(x)              # ~10 % of samples zeroed; survivors rescaled
    >>> y.shape
    (8, 96, 7, 7)

    References
    ----------
    .. [1] Huang et al., *Deep Networks with Stochastic Depth*, ECCV 2016.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(f"DropPath drop_prob must be in [0, 1), got {drop_prob}")
        self.drop_prob = drop_prob

    @override
    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Per-sample binary mask, broadcast across all non-batch dims.
        shape = (int(x.shape[0]),) + (1,) * (x.ndim - 1)
        random_tensor = lucid.rand(shape, device=x.device.type)
        binary_mask = (random_tensor < keep_prob).float()
        return x * binary_mask / keep_prob


class BasicConvBnReLU2d(nn.Module):
    """Bias-free ``Conv2d -> BatchNorm2d(eps=1e-3) -> ReLU`` primitive for GoogLeNet / Inception trunks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        padding: int | tuple[int, int] = 0,
        *,
        eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=eps)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return F.relu(self.bn(self.conv(x)))


def check_patch_grid(
    owner: str, height: int, width: int, *, image_size: int, patch_size: int
) -> None:
    """Refuse an image the position table of a fixed-grid patch model was not built for.

    A ViT-style model holds one position entry per patch of the grid it was
    built for, so it takes exactly one input size — ``image_size`` squared.
    The reference ViT, CLIP and I-JEPA towers refuse every other size unless
    a caller opts into position-table interpolation, which this zoo does not
    offer.  Without the check the patch embedding, a stride-``patch_size``
    convolution, failed in three ways:

    * a side that is not a multiple of the patch makes the convolution drop
      the remainder, so a 47x47 image answered as its 32x32 crop with the
      border unread and no error;
    * another size on the patch grid broadcast against the table and failed
      as an engine ``ShapeMismatch`` naming neither size;
    * the same patch count laid out another way (16x64 against a 32x32 build
      of 8 px patches) fit the table and answered with every patch reading
      another patch's position entry.

    The config already refuses an ``image_size`` that is not a multiple of
    ``patch_size``, so ``image_size`` here is always a whole grid.

    Parameters
    ----------
    owner : str
        The model's name, for the message.
    height, width : int
        The input's spatial size in pixels.
    image_size : int
        Side of the square input the position table was built for.
    patch_size : int
        Side of a patch.

    Raises
    ------
    ValueError
        If ``(height, width)`` is not ``(image_size, image_size)``.

    Examples
    --------
    >>> from lucid.models._utils._classification import check_patch_grid
    >>> check_patch_grid("ViT", 32, 32, image_size=32, patch_size=16)
    >>> check_patch_grid("ViT", 47, 47, image_size=32, patch_size=16)
    Traceback (most recent call last):
        ...
    ValueError: ViT is built for 32x32 inputs: its position table holds one entry per patch of a 2x2 grid of 16px patches.  Got 47x47 pixels, which is not a multiple of the 16px patch, so the patch embedding would silently drop 15 rows and 15 columns at the border.  Resize the input to 32x32.
    """
    if height == image_size and width == image_size:
        return
    p = patch_size
    side = image_size // p
    if height % p or width % p:
        got = (
            f"which is not a multiple of the {p}px patch, so the patch embedding "
            f"would silently drop {height % p} rows and {width % p} columns at the border"
        )
    else:
        got = f"which is {height // p}x{width // p} patches"
    rebuild = (
        f", or build the model with image_size={height}, whose position "
        "table is then sized for that grid"
        if height == width and height % p == 0
        else ""
    )
    raise ValueError(
        f"{owner} is built for {image_size}x{image_size} inputs: its position "
        f"table holds one entry per patch of a {side}x{side} grid of {p}px "
        f"patches.  Got {height}x{width} pixels, {got}.  "
        f"Resize the input to {image_size}x{image_size}{rebuild}."
    )
