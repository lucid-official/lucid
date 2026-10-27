"""``lucid.nn.quantized`` — quantized inference modules.

The int8-weight counterparts of the float ``lucid.nn`` layers, produced by
:func:`lucid.quantization.convert` from a calibrated model.  Under the
sidecar representation (design B) they store int8 weights + ``scale`` /
``zero_point`` buffers and compute in float (dequantize → op →
fake-quantize).  :class:`QuantizedLinearMLX` is the exception and runs
the engine's own low-precision GEMM instead; the two agree to 5e-07,
which is what makes the float path a usable reference for it.

Also holds the boundary markers ``QuantStub`` / ``DeQuantStub`` placed in a
float model and their runtime forms ``Quantize`` / ``DeQuantize``.
"""

from lucid.nn.quantized.activation import ELU, Hardsigmoid, Hardswish, LeakyReLU, Sigmoid, Tanh
from lucid.nn.quantized.conv import Conv1d, Conv2d, Conv3d
from lucid.nn.quantized.conv_transpose import ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
from lucid.nn.quantized.functional_module import FloatFunctional, QFunctional
from lucid.nn.quantized.intrinsic import ConvReLU1d, ConvReLU2d, ConvReLU3d, LinearReLU
from lucid.nn.quantized.linear import Linear
from lucid.nn.quantized.mlx_linear import QuantizedLinearMLX
from lucid.nn.quantized.modules import DeQuantize, DeQuantStub, Quantize, QuantStub, QuantWrapper
from lucid.nn.quantized.sparse import Embedding, EmbeddingBag

__all__ = [
    "Linear",
    "QuantizedLinearMLX",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "Embedding",
    "EmbeddingBag",
    "LinearReLU",
    "ConvReLU1d",
    "ConvReLU2d",
    "ConvReLU3d",
    "QuantStub",
    "DeQuantStub",
    "Quantize",
    "DeQuantize",
    "QuantWrapper",
    "FloatFunctional",
    "QFunctional",
    "Sigmoid",
    "Hardswish",
    "Hardsigmoid",
    "Tanh",
    "ELU",
    "LeakyReLU",
]


def __dir__() -> list[str]:
    """Restrict introspection to the public modules (hide ``_utils`` etc.)."""
    return list(__all__)
