"""The shared reset of the layers that scale and shift by a learnable affine pair."""

import lucid.nn.init as init
from lucid.nn.module import Module


class _AffineParameters(Module):
    """A learnable scale ``weight`` and shift ``bias``, each optional, both identity at rest.

    What the norm layers share: :meth:`reset_parameters` puts ``weight`` to
    ones and ``bias`` to zeros, whichever of the two the layer has.
    """

    def reset_parameters(self) -> None:
        """Put ``weight`` to ones and ``bias`` to zeros, where the layer has them."""
        weight = self._parameters.get("weight")
        if weight is not None:
            init.ones_(weight)
        bias = self._parameters.get("bias")
        if bias is not None:
            init.zeros_(bias)
