"""RefineNet score network — the backbone NCSN v1/v2 actually specify.

Song & Ermon's Implementation Details name a 4-cascade RefineNet with
**CondInstanceNorm++** (v1) or **InstanceNorm++** (v2), **ELU** throughout,
and **dilated convolutions replacing subsampling** in the deeper cascades.
That is a different inductive bias from the DDPM U-Net the family reuses by
default — GroupNorm, SiLU, self-attention and a sinusoidal timestep
embedding, which is what NCSN++ later moved to.

The distinguishing piece is the normalisation.  InstanceNorm++ adds a term
the plain version drops: it normalises each channel spatially *and* rescales
by the per-sample mean of those channel means, so the network keeps
information about overall image brightness that instance norm would
otherwise throw away.  §3.3 of the v1 paper introduces it for exactly that
reason.  The conditional form keeps one gain/bias triple per noise level.
"""

from typing import final, override

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor

__all__ = ["RefineNetScoreNet"]


@final
class CondInstanceNormPlusPlus(nn.Module):
    r"""Conditional InstanceNorm++ (Song & Ermon 2019, §3.3).

    For input :math:`x` with per-channel spatial statistics
    :math:`\mu_k, s_k`, and :math:`m, v` the mean and standard deviation of
    :math:`\mu` across channels:

    .. math::

        z_k = \gamma_{i,k}\,\frac{x_k - \mu_k}{s_k}
            + \beta_{i,k}
            + \alpha_{i,k}\,\frac{\mu_k - m}{v}

    The third term is the "++".  Plain instance norm removes the per-channel
    mean and with it any notion of which channels were brighter than the
    others; the extra branch feeds that back, and the paper reports colour
    shifts in generated samples without it.  Each :math:`i` indexes a noise
    level, so the gains are conditional.

    Parameters
    ----------
    channels : int
        Feature channels.
    num_classes : int
        Number of noise levels the gains are conditioned on.
    eps : float, optional, default=1e-5
        Added to variances before the square root.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.ncsn._refinenet import (
    ...     CondInstanceNormPlusPlus,
    ... )
    >>> norm = CondInstanceNormPlusPlus(4, num_classes=3)
    >>> out = norm(lucid.randn(2, 4, 8, 8), lucid.tensor([0, 2]).long())
    >>> out.shape
    (2, 4, 8, 8)
    """

    def __init__(self, channels: int, num_classes: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        # Gains start at identity so an untrained network is a plain
        # normaliser rather than an arbitrary rescale.
        self.gamma = nn.Embedding(num_classes, channels)
        self.beta = nn.Embedding(num_classes, channels)
        self.alpha = nn.Embedding(num_classes, channels)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.alpha.weight)

    @override
    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        """Normalise ``x`` under the gains selected by ``labels``."""
        mu = x.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        var = ((x - mu) ** 2).mean(dim=(2, 3), keepdim=True)
        h = (x - mu) / (var + self.eps).sqrt()

        # Statistics *of* the per-channel means — the "++" branch.
        m = mu.mean(dim=1, keepdim=True)
        v = ((mu - m) ** 2).mean(dim=1, keepdim=True)
        mu_hat = (mu - m) / (v + self.eps).sqrt()

        g = self.gamma(labels).reshape(*labels.shape, -1, 1, 1)
        b = self.beta(labels).reshape(*labels.shape, -1, 1, 1)
        a = self.alpha(labels).reshape(*labels.shape, -1, 1, 1)
        return g * h + b + a * mu_hat


@final
class _CondResBlock(nn.Module):
    """Pre-activation residual block: norm → ELU → conv, twice.

    ELU rather than SiLU, and dilation rather than stride when the cascade
    needs a wider receptive field without shrinking the map — both are the
    paper's choices, and the dilation one is why the deeper cascades keep
    full resolution.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_classes: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.norm1 = CondInstanceNormPlusPlus(in_ch, num_classes)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False)
        self.norm2 = CondInstanceNormPlusPlus(out_ch, num_classes)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False)
        self.skip: nn.Conv2d | nn.Identity = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        )

    @override
    def forward(self, x: Tensor, labels: Tensor) -> Tensor:
        h = self.conv1(F.elu(self.norm1(x, labels)))
        h = self.conv2(F.elu(self.norm2(h, labels)))
        return h + self.skip(x)


@final
class _RefineBlock(nn.Module):
    """Fuse a coarser cascade output with the matching encoder features.

    RefineNet's decoder step: bring both paths to a common resolution, sum
    them, and pass the result through a residual block.  The upsample is
    driven by the *actual* shape of the finer tensor rather than a fixed
    factor, so an odd resolution does not drift the two grids apart.
    """

    def __init__(self, coarse_ch: int, fine_ch: int, num_classes: int) -> None:
        super().__init__()
        # The two paths arrive at different widths — cascades widen going
        # down — so both are projected to the finer one before they can be
        # summed.  The output keeps the fine width, which is what the next
        # refine step up expects.
        self.adapt_coarse = _CondResBlock(coarse_ch, fine_ch, num_classes)
        self.adapt_fine = _CondResBlock(fine_ch, fine_ch, num_classes)
        self.fuse = _CondResBlock(fine_ch, fine_ch, num_classes)

    @override
    def forward(self, coarse: Tensor, fine: Tensor, labels: Tensor) -> Tensor:
        up = F.interpolate(
            coarse,
            size=(int(fine.shape[2]), int(fine.shape[3])),
            mode="bilinear",
            align_corners=False,
        )
        merged = self.adapt_coarse(up, labels) + self.adapt_fine(fine, labels)
        return self.fuse(merged, labels)


@final
class RefineNetScoreNet(nn.Module):
    r"""4-cascade RefineNet score network with conditional InstanceNorm++.

    Signature-compatible with the DDPM U-Net this family uses by default —
    ``forward(sample, sigma_idx)`` returning a tensor shaped like
    ``sample`` — so :class:`~lucid.models.generative.ncsn.NCSNModel` selects
    between them without any other change.

    Parameters
    ----------
    in_channels : int
        Image channels.
    base_channels : int
        Width of the first cascade; the paper uses 128 for CIFAR-10.
    num_classes : int
        Noise levels the normalisation is conditioned on.
    channel_mult : tuple of int, optional
        Per-cascade width multipliers.
    dilations : tuple of int, optional
        Dilation per cascade.  The deeper entries replace subsampling, which
        is what keeps the score map at full resolution.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative.ncsn._refinenet import RefineNetScoreNet
    >>> net = RefineNetScoreNet(3, base_channels=16, num_classes=4)
    >>> out = net(lucid.randn(2, 3, 32, 32), lucid.tensor([0, 3]).long())
    >>> out.shape
    (2, 3, 32, 32)
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 128,
        num_classes: int = 10,
        channel_mult: tuple[int, ...] = (1, 2, 2, 2),
        dilations: tuple[int, ...] = (1, 1, 2, 4),
    ) -> None:
        super().__init__()
        if len(channel_mult) != len(dilations):
            raise ValueError(
                "channel_mult and dilations describe the same cascades, so "
                f"they must be the same length; got {len(channel_mult)} and "
                f"{len(dilations)}."
            )
        self.stem = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        widths = [base_channels * m for m in channel_mult]
        encoders: list[_CondResBlock] = []
        prev = base_channels
        for w, d in zip(widths, dilations, strict=True):
            encoders.append(_CondResBlock(prev, w, num_classes, dilation=d))
            prev = w
        self.encoders = nn.ModuleList(encoders)
        # Downsample only where dilation is 1; past that the paper widens the
        # receptive field with dilation instead, so the map keeps its size.
        self.pools = nn.ModuleList(
            [nn.AvgPool2d(2, stride=2) if d == 1 else nn.Identity() for d in dilations]
        )
        # One refine step per skip below the deepest cascade, pairing the
        # width coming down with the width waiting at that level.
        self.refines = nn.ModuleList(
            [
                _RefineBlock(widths[i + 1], widths[i], num_classes)
                for i in reversed(range(len(widths) - 1))
            ]
        )
        self.out_norm = CondInstanceNormPlusPlus(widths[0], num_classes)
        self.out_conv = nn.Conv2d(widths[0], in_channels, 3, padding=1)

    @override
    def forward(self, sample: Tensor, sigma_idx: Tensor) -> Tensor:
        """Predict the score at ``sample`` for the given noise-level indices."""
        labels = sigma_idx.long()
        h = self.stem(sample)

        skips: list[Tensor] = []
        for enc, pool in zip(self.encoders, self.pools, strict=True):
            h = enc(h, labels)
            skips.append(h)
            h = pool(h)

        # Decode from the coarsest cascade back up, fusing each skip.
        h = skips[-1]
        for refine, skip in zip(self.refines, reversed(skips[:-1]), strict=True):
            h = refine(h, skip, labels)

        h = F.elu(self.out_norm(h, labels))
        out = self.out_conv(h)
        if tuple(int(v) for v in out.shape[2:]) != tuple(int(v) for v in sample.shape[2:]):
            out = F.interpolate(
                out,
                size=(int(sample.shape[2]), int(sample.shape[3])),
                mode="bilinear",
                align_corners=False,
            )
        return out
