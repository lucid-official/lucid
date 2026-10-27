"""``lucid.nn.qat`` — quantization-aware training modules.

Float-trainable layers that fake-quantize their weights and outputs (via the
straight-through estimator) so a network learns to be robust to int8
inference.  Produced by :func:`lucid.quantization.prepare_qat`; turned into
real quantized inference layers by :func:`lucid.quantization.convert`.
"""

from lucid.nn.qat.conv import Conv1d, Conv2d, Conv3d, ConvReLU1d, ConvReLU2d, ConvReLU3d
from lucid.nn.qat.linear import Linear, LinearReLU
from lucid.nn.qat.sparse import Embedding

__all__ = [
    "Linear",
    "LinearReLU",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "Embedding",
]


def __dir__() -> list[str]:
    """Restrict introspection to the public modules."""
    return list(__all__)
