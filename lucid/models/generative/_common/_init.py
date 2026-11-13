r"""How the released DreamerV3 starts its dense layers, as a reusable initialiser.

The released implementation does not use a framework's default for a dense
layer.  Every kernel is a truncated normal scaled by the layer's fan-in and
every bias is zero (``winit: trunc_normal_in`` in ``configs.yaml``; the
distribution is ``Initializer`` in ``embodied/jax/nets.py``).  The earlier
Dreamers did not specify one, so this is opt-in: a block takes a
``dense_init`` and the default leaves it exactly as it was built.

The rule, in the released code's own terms (``embodied/jax/nets.py``)::

    x = jax.random.truncated_normal(seed, -2, 2, shape)       # line 169
    x *= 1.1368 * np.sqrt(1 / fan)                            # line 170

so a unit normal cut at two standard deviations, rescaled so that the
*truncated* draw has standard deviation :math:`1/\sqrt{\text{fan}_{in}}`
(a normal cut at :math:`\pm 2\sigma` has standard deviation
:math:`0.8796\,\sigma`, and :math:`1.1368 \times 0.8796 \approx 1`).  The
fan is the first axis of the ``(in, out)`` kernel (``nets.py:188-197``).
"""

import math
from typing import Final

import lucid
import lucid.nn as nn
from lucid._tensor.tensor import Tensor
from lucid.models.generative._common._config import DenseInit, check_dense_init

__all__ = ["fan_in_trunc_normal_", "initialise_linear_layers_"]

#: ``embodied/jax/nets.py:170`` — what a normal truncated at +-2 is multiplied by
#: so that the draw's standard deviation, rather than the underlying normal's,
#: is the target.
_TRUNC_NORMAL_RESCALE: Final = 1.1368

#: ``embodied/jax/nets.py:169`` — the truncation, in units of the underlying
#: normal's standard deviation, on both sides.
_TRUNCATION: Final = 2.0


def fan_in_trunc_normal_(weight: Tensor, fan_in: int, outscale: float = 1.0) -> Tensor:
    r"""Fill ``weight`` as the released ``trunc_normal_in`` does, then scale it.

    Parameters
    ----------
    weight : Tensor
        Any shape; filled in place.
    fan_in : int
        The layer's input fan.  Passed rather than read off ``weight``
        because the released code's fan depends on the layer: the first axis
        of a dense ``(in, out)`` kernel, but the *whole* input width for a
        block-diagonal kernel, whose own first axes are ``(blocks, in)``
        (``nets.py:196-197`` multiplies the within-block width back by the
        block count).
    outscale : float, default=1.0
        The released ``Linear``'s ``outscale``, multiplied into the draw
        (``nets.py:250-251``).  ``0.0`` gives exact zeros.  The draw is made
        either way, so a layer scaled by ``outscale`` and one that is not
        consume the same random numbers.

    Returns
    -------
    Tensor
        ``weight``.

    Raises
    ------
    ValueError
        If ``fan_in`` is not positive or ``outscale`` is negative.

    Notes
    -----
    Truncated at :math:`\pm 2` underlying standard deviations
    :math:`\sigma = 1.1368 / \sqrt{\text{fan}_{in}}`, which makes the draw's
    own standard deviation :math:`1 / \sqrt{\text{fan}_{in}}` — the draw is
    :func:`lucid.nn.init.trunc_normal_`, not a second implementation of it.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.generative._common._init import fan_in_trunc_normal_
    >>> w = fan_in_trunc_normal_(lucid.empty(256, 64), fan_in=64)
    >>> bool(w.abs().max().item() <= 2 * 1.1368 / 64 ** 0.5)  # inside the truncation
    True
    >>> round(float(w.std().item()) * 64 ** 0.5, 1)  # std is 1 / sqrt(fan_in)
    1.0
    >>> bool((fan_in_trunc_normal_(lucid.empty(4, 4), fan_in=4, outscale=0.0) == 0).all().item())
    True
    """
    if fan_in < 1:
        raise ValueError(f"fan_in must be positive, got {fan_in}")
    if outscale < 0.0:
        raise ValueError(f"outscale must be non-negative, got {outscale}")
    std = _TRUNC_NORMAL_RESCALE / math.sqrt(fan_in)
    nn.init.trunc_normal_(weight, 0.0, std, -_TRUNCATION * std, _TRUNCATION * std)
    if outscale != 1.0:
        with lucid.no_grad():
            weight[:] = weight * outscale
    return weight


def initialise_linear_layers_(module: nn.Module, dense_init: DenseInit) -> None:
    """Start every :class:`~lucid.nn.Linear` under ``module`` as ``dense_init`` says.

    Parameters
    ----------
    module : nn.Module
        A block whose constructor has already drawn its defaults; may be a
        single layer.  Only ``nn.Linear`` is touched — other layers, the
        block-diagonal ones included, start themselves.
    dense_init : {"default", "fan_in_trunc_normal"}
        The initialiser.  ``"default"`` leaves every layer alone;
        ``"fan_in_trunc_normal"`` redraws each kernel with
        :func:`fan_in_trunc_normal_` at the layer's ``in_features`` and
        zeroes its bias.

    Raises
    ------
    ValueError
        If ``dense_init`` names no initialiser.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.models.generative._common._init import initialise_linear_layers_
    >>> layer = nn.Linear(16, 8)
    >>> initialise_linear_layers_(layer, "fan_in_trunc_normal")
    >>> bool((layer.bias == 0).all().item())
    True
    """
    check_dense_init(dense_init)
    if dense_init == "default":
        return
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            fan_in_trunc_normal_(layer.weight, layer.in_features)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
