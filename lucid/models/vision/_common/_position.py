"""The sine position table the DETR lineage feeds its attention.

DETR, MaskFormer and Mask2Former each carried their own copy of the
reference ``PositionEmbeddingSine``, line for line the same.  One copy,
keyed to the feature map it embeds, is the only place a rule about the
table — its size, its device — has to hold.
"""

from typing import final, override

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor

__all__ = ["SinePositionEmbedding"]


@final
class SinePositionEmbedding(nn.Module):
    """Normalised 2-D sinusoidal positions for an unpadded feature map.

    The reference masks padding before its cumulative sums; every caller
    here embeds one unpadded map, so the not-mask is all ones and the sums
    are plain row and column counts.  Defaults are the reference's:
    ``temperature = 10000``, ``normalize = True``, ``scale = 2π``.

    Parameters
    ----------
    num_position_features : int
        Width per axis; the table is ``2 * num_position_features`` wide.
    temperature, scale, eps : float
        The reference's frequency base, normalised range and divisor guard.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision._common._position import SinePositionEmbedding
    >>> embed = SinePositionEmbedding(4)
    >>> embed(lucid.zeros(2, 3, 5, 6)).shape
    (2, 8, 5, 6)
    """

    def __init__(
        self,
        num_position_features: int,
        temperature: float = 10000.0,
        scale: float = 6.283185307179586,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_position_features = num_position_features
        self.temperature = temperature
        self.scale = scale
        self.eps = eps

    @override
    def forward(self, feature: Tensor) -> Tensor:
        batch = int(feature.shape[0])
        height = int(feature.shape[-2])
        width = int(feature.shape[-1])
        device = feature.device.type
        npf = self.num_position_features

        ones = lucid.ones(1, height, width, dtype=lucid.float32, device=device)
        y_embed = ones.cumsum(dim=1)
        x_embed = ones.cumsum(dim=2)
        y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = lucid.arange(0, npf, dtype=lucid.float32, device=device)
        dim_t = self.temperature ** (2.0 * lucid.floor(dim_t / 2.0) / npf)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = lucid.stack(
            [lucid.sin(pos_x[:, :, :, 0::2]), lucid.cos(pos_x[:, :, :, 1::2])], dim=4
        ).flatten(3)
        pos_y = lucid.stack(
            [lucid.sin(pos_y[:, :, :, 0::2]), lucid.cos(pos_y[:, :, :, 1::2])], dim=4
        ).flatten(3)
        pos = lucid.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)  # (1, d, H, W)
        if batch > 1:
            pos = pos.repeat(batch, 1, 1, 1)
        return pos
