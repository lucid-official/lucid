"""Padding an image up to a whole number of patches.

A patch embedding is a convolution whose kernel and stride are both the patch
size, and such a convolution silently drops whatever does not fill a last
patch: a 35-pixel side under 4-pixel patches loses its final three columns
before the network sees them.  The reference Swin pads the right and bottom
with zeros instead, so the border takes part and the grid the stages are
handed is ``ceil(side / patch_size)`` on each axis.  Swin and the Mask2Former
Swin backbone share this one rule.
"""

import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor

__all__ = ["pad_to_patch_multiple"]


def pad_to_patch_multiple(x: Tensor, patch_size: int) -> Tensor:
    """Zero-pad ``(B, C, H, W)`` on the bottom and right to multiples of ``patch_size``.

    An input already on the grid is returned as is, so the common case costs
    nothing and keeps its autograd graph untouched.

    Parameters
    ----------
    x : Tensor
        Images, ``(B, C, H, W)``.
    patch_size : int
        Side of the square patch the grid is a multiple of.

    Returns
    -------
    Tensor
        ``(B, C, ceil(H / patch_size) * patch_size, ceil(W / patch_size) * patch_size)``,
        in the dtype and on the device of ``x``.
    """
    pad_h = -int(x.shape[-2]) % patch_size
    pad_w = -int(x.shape[-1]) % patch_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x
