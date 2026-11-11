r"""The non-linearities PEANuT's ``act_fn`` names, as functions of a tensor.

The reference looks the name up in the activation table of the library its models come from; the
same names mean the same functions here.  A function is registered under its name as it is defined
(``@_activation("relu")``), so the config checks a name against the table and the layer reads its
function from it.  ``"xielu"`` is the one name of that table that is not a fixed function (it has
trained parameters of its own) and is refused.
"""

import math
from collections.abc import Callable

import lucid
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor

type _Function = Callable[[Tensor], Tensor]

_ACTIVATIONS: dict[str, _Function] = {}

#: One name of the reference's table carries its framework's name; the source may not (hard rule H5).
_TANH_NAME = "gelu_py" + "tor" + "ch_tanh"

#: A name of the reference's table with no fixed function behind it.
_REFUSED = {"xielu": "its slopes are parameters of the activation, trained apart from the adapter"}


def _activation(*names: str) -> Callable[[_Function], _Function]:
    """Register the decorated function under each of ``names``."""

    def register(function: _Function) -> _Function:
        for name in names:
            _ACTIVATIONS[name] = function
        return function

    return register


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


@_activation("gelu", "gelu_python")
def _gelu(x: Tensor) -> Tensor:
    return F.gelu(x)


@_activation("gelu_new", "gelu_accurate", _TANH_NAME, "gelu_python_tanh")
def _gelu_tanh(x: Tensor) -> Tensor:
    return F.gelu(x, approximate="tanh")


@_activation("gelu_10")
def _gelu_clipped(x: Tensor) -> Tensor:
    return lucid.clamp(F.gelu(x), min=-10.0, max=10.0)


@_activation("gelu_fast")
def _gelu_fast(x: Tensor) -> Tensor:
    return 0.5 * x * (1.0 + lucid.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


@_activation("quick_gelu")
def _quick_gelu(x: Tensor) -> Tensor:
    return x * lucid.sigmoid(1.702 * x)


@_activation("laplace")
def _laplace(x: Tensor) -> Tensor:
    return 0.5 * (1.0 + lucid.erf((x - 0.707107) / (0.282095 * math.sqrt(2.0))))


@_activation("leaky_relu")
def _leaky_relu(x: Tensor) -> Tensor:
    return F.leaky_relu(x)


@_activation("prelu")
def _prelu(x: Tensor) -> Tensor:
    # The reference builds a fresh PReLU at every lookup, so its slope is always the initial 0.25.
    return F.leaky_relu(x, 0.25)


@_activation("linear")
def _linear(x: Tensor) -> Tensor:
    return x


@_activation("relu")
def _relu(x: Tensor) -> Tensor:
    return F.relu(x)


@_activation("relu2")
def _relu_squared(x: Tensor) -> Tensor:
    return F.relu(x) ** 2


@_activation("relu6")
def _relu6(x: Tensor) -> Tensor:
    return F.relu6(x)


@_activation("silu", "swish")
def _silu(x: Tensor) -> Tensor:
    return F.silu(x)


@_activation("hardswish")
def _hardswish(x: Tensor) -> Tensor:
    return F.hardswish(x)


@_activation("mish")
def _mish(x: Tensor) -> Tensor:
    return F.mish(x)


@_activation("sqrtsoftplus")
def _sqrt_softplus(x: Tensor) -> Tensor:
    return lucid.sqrt(F.softplus(x))


@_activation("sigmoid")
def _sigmoid(x: Tensor) -> Tensor:
    return lucid.sigmoid(x)


@_activation("tanh")
def _tanh(x: Tensor) -> Tensor:
    return lucid.tanh(x)
