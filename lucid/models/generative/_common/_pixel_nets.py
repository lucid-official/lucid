"""Pixel encoder / decoder and dense head — shared by the world models.

PlaNet cites "the convolutional and deconvolutional networks from Ha &
Schmidhuber, 2018" for its observation model, and Dreamer says it uses the
same ones.  Two families, one stack, so it lives here beside ``_rssm.py``
rather than being copied into each.

Geometry, at 64x64 with no padding anywhere::

    encode:  64 -> 31 -> 14 -> 6 -> 2      four 4x4 convolutions, stride 2
             channels d, 2d, 4d, 8d        flatten to 8d * 2 * 2 = 32d

    decode:  1 -> 5 -> 13 -> 30 -> 64      kernels 5, 5, 6, 6, stride 2
             channels 4d, 2d, d, out

Both chains land on their targets exactly, which is why the kernel
schedule is irregular and why 64 is the only resolution these express.

Like :class:`RSSM`, these take plain integers rather than a config — the
families that share them have different config classes, and a shared
building block should not know which one it is being built from.
"""

from typing import final, override

import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.models._utils._generative import generative_activation
from lucid.models.generative._common._config import WORLD_MODEL_IMAGE_SIZE

__all__ = [
    "PixelEncoder",
    "PixelDecoder",
    "DenseHead",
    "pixel_embed_size",
    "PIXEL_IMAGE_SIZE",
]

#: The only frame size the kernel schedule above produces.  Owned by
#: :mod:`._config` so the configs can validate against it without
#: importing ``nn``; re-exported here because this is where it is decided.
PIXEL_IMAGE_SIZE = WORLD_MODEL_IMAGE_SIZE


def pixel_embed_size(cnn_depth: int) -> int:
    """Width of :class:`PixelEncoder`'s output — ``8 * depth`` over a 2x2 grid.

    Parameters
    ----------
    cnn_depth : int
        Channel width of the encoder's first convolution.

    Returns
    -------
    int
        ``32 * cnn_depth`` — 1024 at the papers' ``cnn_depth=32``.  Both
        the RSSM's embedding input and the decoder's lift are sized from
        this, so it lives here rather than being recomputed in each.

    Examples
    --------
    >>> from lucid.models.generative._common._pixel_nets import (
    ...     pixel_embed_size,
    ... )
    >>> pixel_embed_size(32)
    1024

    The flat width a pixel encoder hands the recurrent state, derived
    from the CNN depth rather than configured beside it, so the two
    cannot be set to disagree.
    """
    return 32 * cnn_depth


def _fold(x: Tensor) -> tuple[Tensor, int, int]:
    """``(B, T, ...) -> (B * T, ...)``, returning the split for :func:`_unfold`."""
    b, t = int(x.shape[0]), int(x.shape[1])
    rest = tuple(int(s) for s in x.shape[2:])
    return x.reshape(b * t, *rest), b, t


def _unfold(x: Tensor, b: int, t: int) -> Tensor:
    """``(B * T, ...) -> (B, T, ...)``."""
    rest = tuple(int(s) for s in x.shape[1:])
    return x.reshape(b, t, *rest)


@final
class PixelEncoder(nn.Module):
    r"""Four stride-2 convolutions widening ``d, 2d, 4d, 8d``.

    Takes ``(B, T, C, 64, 64)`` and returns ``(B, T, 32 * cnn_depth)``.  The
    time axis is folded into the batch because convolutions only ever see
    4-D input — the encoder is per-frame, and only the dynamics are
    sequential.

    Parameters
    ----------
    in_channels : int
        Observation channels.
    cnn_depth : int
        Width of the first convolution; the stack widens from there.
    act_fn : {"silu", "swish", "relu", "gelu"}
        Activation after every convolution.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative._common._pixel_nets import (
    ...     PixelEncoder,
    ...     pixel_embed_size,
    ... )
    >>> encoder = PixelEncoder(3, 32, "relu").eval()
    >>> frames = lucid.randn(2, 5, 3, 64, 64)
    >>> encoder(frames).shape
    (2, 5, 1024)

    Five frames of a two-episode batch, and the time axis is required
    rather than optional — a world model always sees sequences, so the
    convolutions fold it away and put it back.

    >>> pixel_embed_size(32)
    1024
    """

    def __init__(self, in_channels: int, cnn_depth: int, act_fn: str) -> None:
        """Initialise the encoder. See the class docstring for parameters."""
        super().__init__()
        self._act_name = act_fn
        self.embed_size = pixel_embed_size(cnn_depth)
        widths = (in_channels, cnn_depth, 2 * cnn_depth, 4 * cnn_depth, 8 * cnn_depth)
        self.convs = nn.ModuleList(
            [nn.Conv2d(widths[i], widths[i + 1], kernel_size=4, stride=2) for i in range(4)]
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        h, b, t = _fold(x)
        for conv in self.convs:
            h = generative_activation(self._act_name, conv(h))
        return _unfold(h.reshape(int(h.shape[0]), self.embed_size), b, t)


@final
class PixelDecoder(nn.Module):
    r"""Linear lift to a 1x1 grid, then four transposed convolutions.

    Takes ``(B, T, latent_size)`` and returns ``(B, T, out_channels, 64, 64)``.

    Parameters
    ----------
    latent_size : int
        Width of the state the decoder reads.
    out_channels : int
        Reconstruction channels.
    cnn_depth : int
        Matches the encoder's, so the two stacks mirror.
    act_fn : {"silu", "swish", "relu", "gelu"}
        Activation between transposed convolutions.

    Notes
    -----
    The last transposed convolution is deliberately unactivated: its output
    is the mean of a unit-variance Gaussian over pixels, not a hidden
    layer.  Activating it would clamp reconstructions to a half-line.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative._common._pixel_nets import PixelDecoder
    >>> decoder = PixelDecoder(1024, 3, 32, "relu").eval()
    >>> decoder(lucid.randn(2, 5, 1024)).shape
    (2, 5, 3, 64, 64)

    The encoder's inverse, back to frames. Reconstruction is what forces
    the latent to keep the pixels rather than only what the reward
    happened to need.
    """

    def __init__(self, latent_size: int, out_channels: int, cnn_depth: int, act_fn: str) -> None:
        """Initialise the decoder. See the class docstring for parameters."""
        super().__init__()
        self._act_name = act_fn
        self._lift_width = pixel_embed_size(cnn_depth)

        self.lift = nn.Linear(latent_size, self._lift_width)
        specs = (
            (self._lift_width, 4 * cnn_depth, 5),
            (4 * cnn_depth, 2 * cnn_depth, 5),
            (2 * cnn_depth, cnn_depth, 6),
            (cnn_depth, out_channels, 6),
        )
        self.deconvs = nn.ModuleList(
            [nn.ConvTranspose2d(cin, cout, kernel_size=k, stride=2) for cin, cout, k in specs]
        )

    @override
    def forward(self, feature: Tensor) -> Tensor:
        h, b, t = _fold(feature)
        h = self.lift(h)
        h = h.reshape(int(h.shape[0]), self._lift_width, 1, 1)
        last = len(self.deconvs) - 1
        for i, deconv in enumerate(self.deconvs):
            h = deconv(h)
            if i != last:
                h = generative_activation(self._act_name, h)
        return _unfold(h, b, t)


@final
class DenseHead(nn.Module):
    r"""A stack of dense layers reading a latent state.

    Every scalar or vector prediction these families make off the latent —
    reward, value, action parameters — is one of these.  Takes
    ``(B, T, in_features)`` and returns ``(B, T, out_features)``, or
    ``(B, T)`` when ``out_features == 1`` and ``squeeze`` is set.

    Parameters
    ----------
    in_features : int
        Width of the state read.
    hidden : int
        Width of each hidden layer.
    layers : int
        Number of hidden layers.
    out_features : int, default=1
        Width of the prediction.
    act_fn : {"silu", "swish", "relu", "gelu"}, default="relu"
        Activation between hidden layers; the output layer is unactivated.
    squeeze : bool, default=False
        Drop the trailing axis when ``out_features == 1``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative._common._pixel_nets import DenseHead
    >>> head = DenseHead(1024, 64, 2).eval()
    >>> head(lucid.randn(2, 5, 1024)).shape
    (2, 5, 1)

    One number per step by default — reward, value, or the probability
    an episode ended. The same shape serves all three, which is why they
    share a class.
    """

    def __init__(
        self,
        in_features: int,
        hidden: int,
        layers: int,
        out_features: int = 1,
        act_fn: str = "relu",
        squeeze: bool = False,
    ) -> None:
        """Initialise the head. See the class docstring for parameters."""
        super().__init__()
        if layers < 1:
            raise ValueError(f"layers must be at least 1, got {layers}")
        self._act_name = act_fn
        self._squeeze = squeeze and out_features == 1
        widths = [in_features] + [hidden] * layers
        self.layers = nn.ModuleList([nn.Linear(widths[i], widths[i + 1]) for i in range(layers)])
        self.out = nn.Linear(hidden, out_features)

    @override
    def forward(self, feature: Tensor) -> Tensor:
        h, b, t = _fold(feature)
        for layer in self.layers:
            h = generative_activation(self._act_name, layer(h))
        out = self.out(h)
        return out.reshape(b, t) if self._squeeze else _unfold(out, b, t)
