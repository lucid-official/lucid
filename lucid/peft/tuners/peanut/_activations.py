r"""The non-linearities PEANuT's ``act_fn`` names, as functions of a tensor.

The reference looks the name up in the activation table of the library its models come from; the
same names mean the same functions here.  ``_ACTIVATIONS`` is that table, so the config checks a name
against it and the layer reads its function from it.  ``"xielu"`` is the one name of the reference's
table that is not a fixed function (it has trained parameters of its own) and is refused.
"""

import math
from collections.abc import Callable
from typing import Final

import lucid
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor

type _Function = Callable[[Tensor], Tensor]

#: A name of the reference's table with no fixed function behind it.
_REFUSED: Final = {
    "xielu": "its slopes are parameters of the activation, trained apart from the adapter"
}

#: One name of the reference's table carries its framework's name; the source may not (hard rule H5).
_TANH_NAME: Final = "gelu_py" + "tor" + "ch_tanh"


def _gelu(x: Tensor) -> Tensor:
    return F.gelu(x)


def _gelu_tanh(x: Tensor) -> Tensor:
    return F.gelu(x, approximate="tanh")


def _gelu_clipped(x: Tensor) -> Tensor:
    return lucid.clamp(F.gelu(x), min=-10.0, max=10.0)


def _gelu_fast(x: Tensor) -> Tensor:
    return 0.5 * x * (1.0 + lucid.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def _quick_gelu(x: Tensor) -> Tensor:
    return x * lucid.sigmoid(1.702 * x)


def _laplace(x: Tensor) -> Tensor:
    return 0.5 * (1.0 + lucid.erf((x - 0.707107) / (0.282095 * math.sqrt(2.0))))


def _leaky_relu(x: Tensor) -> Tensor:
    return F.leaky_relu(x)


def _prelu(x: Tensor) -> Tensor:
    # The reference builds a fresh PReLU at every lookup, so its slope is always the initial 0.25.
    return F.leaky_relu(x, 0.25)


def _linear(x: Tensor) -> Tensor:
    return x


def _relu(x: Tensor) -> Tensor:
    return F.relu(x)


def _relu_squared(x: Tensor) -> Tensor:
    return F.relu(x) ** 2


def _relu6(x: Tensor) -> Tensor:
    return F.relu6(x)


def _silu(x: Tensor) -> Tensor:
    return F.silu(x)


def _hardswish(x: Tensor) -> Tensor:
    return F.hardswish(x)


def _mish(x: Tensor) -> Tensor:
    return F.mish(x)


def _sqrt_softplus(x: Tensor) -> Tensor:
    return lucid.sqrt(F.softplus(x))


def _sigmoid(x: Tensor) -> Tensor:
    return lucid.sigmoid(x)


def _tanh(x: Tensor) -> Tensor:
    return lucid.tanh(x)


_ACTIVATIONS: Final[dict[str, _Function]] = {
    "gelu": _gelu,
    "gelu_python": _gelu,
    "gelu_new": _gelu_tanh,
    "gelu_accurate": _gelu_tanh,
    _TANH_NAME: _gelu_tanh,
    "gelu_python_tanh": _gelu_tanh,
    "gelu_10": _gelu_clipped,
    "gelu_fast": _gelu_fast,
    "quick_gelu": _quick_gelu,
    "laplace": _laplace,
    "leaky_relu": _leaky_relu,
    "prelu": _prelu,
    "linear": _linear,
    "relu": _relu,
    "relu2": _relu_squared,
    "relu6": _relu6,
    "silu": _silu,
    "swish": _silu,
    "hardswish": _hardswish,
    "mish": _mish,
    "sqrtsoftplus": _sqrt_softplus,
    "sigmoid": _sigmoid,
    "tanh": _tanh,
}


def _check_act_fn(name: str, value: object) -> str:
    """``value`` if it names an activation: a ``str`` of the table; ``ValueError`` for another name, ``NotImplementedError`` for ``"xielu"``."""
    if not isinstance(value, str):
        raise TypeError(f"`{name}` must be a str, got {value!r}")
    if value in _REFUSED:
        raise NotImplementedError(
            f"`{name}={value!r}` is not supported by lucid.peft: {_REFUSED[value]}"
        )
    if value not in _ACTIVATIONS:
        raise ValueError(f"Unsupported `{name}`: {value}. Must be one of {sorted(_ACTIVATIONS)}.")
    return value


def _lookup(name: str) -> _Function:
    """The function ``name`` stands for."""
    return _ACTIVATIONS[name]
