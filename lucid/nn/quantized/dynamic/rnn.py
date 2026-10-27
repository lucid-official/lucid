"""Dynamically quantized recurrent layers — ``LSTM``, ``GRU`` and the cells.

The recurrent weight matrices (``weight_ih*`` / ``weight_hh*``) are stored as
int8 (per-row symmetric); biases stay float.  A float module of the same kind
— the *shell*, held outside the module registry so it never enters the
``state_dict`` — is the compute engine: on each forward its weights are
overwritten with the dequantized values (which live on the same device as the
int8 buffers), so only the int8 form is the persistent, device-tracked state.
This is weight-quantized dynamic inference: the per-timestep activation
quantization inside the cell is not done here, and the engine's low-precision
kernel is reached through :class:`~lucid.nn.quantized.QuantizedLinearMLX`
rather than from this module.
"""

import copy
from typing import TYPE_CHECKING, ClassVar, Self, override

import lucid
import lucid.nn as nn
from lucid.nn.utils.rnn import PackedSequence
from lucid.nn.parameter import Parameter
from lucid.quantization._functional import dequantize, quantize
from lucid.quantization._qscheme import QDtype, per_channel_symmetric, qint8

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _buffer_key(name: str) -> str:
    """The flat buffer name of shell parameter ``name``.

    ``nn.GRU`` keeps each layer's weights in a cell child (``cell_l0.weight_ih``);
    the buffer takes the flat name a float GRU's ``state_dict`` uses
    (``weight_ih_l0``), since a buffer name cannot hold a dot.
    """
    owner, _, leaf = name.rpartition(".")
    return f"{leaf}_{owner.removeprefix('cell_')}" if owner else leaf


def _is_weight(name: str) -> bool:
    """Whether shell parameter ``name`` is a weight matrix (else a bias)."""
    return name.rpartition(".")[2].startswith("weight")


class _DynamicRecurrent[F: nn.Module](nn.Module):
    """int8 weights of a float recurrent module, run through that module.

    The shared machinery of every dynamic recurrent layer: :meth:`from_float`
    quantizes the float module's weights into buffers, and :meth:`_loaded`
    writes their dequantized values back into it before each forward.  A
    subclass names the float type it accepts and exposes its ``forward``.
    """

    _FLOAT: ClassVar[type[nn.Module]]

    def __init__(self, shell: nn.Module) -> None:
        super().__init__()
        # The shell holds config + is the compute engine; keep it OUT of the
        # module registry so its float weights never reach ``state_dict``.
        object.__setattr__(self, "_shell", shell)
        named = [n for n, _ in shell.named_parameters()]
        # (path inside the shell, buffer name) of each weight and bias.
        self._weight_names: list[tuple[str, str]] = [
            (n, _buffer_key(n)) for n in named if _is_weight(n)
        ]
        self._bias_names: list[tuple[str, str]] = [
            (n, _buffer_key(n)) for n in named if not _is_weight(n)
        ]

    def _buffer(self, key: str) -> Tensor:
        value = self._buffers[key]
        if value is None:
            raise RuntimeError(f"{type(self).__name__}: buffer '{key}' is unset")
        return value

    def _loaded(self) -> F:
        """The shell, its weights replaced by the dequantized int8 buffers."""
        shell: F = object.__getattribute__(self, "_shell")
        for path, key in self._weight_names:
            deq = dequantize(
                self._buffer(key + "_int8"),
                self._buffer(key + "_scale"),
                lucid.tensor(0.0),
                ch_axis=0,
            )
            self._put(shell, path, deq)
        for path, key in self._bias_names:
            self._put(shell, path, self._buffer(key))
        return shell

    @staticmethod
    def _put(shell: nn.Module, path: str, value: Tensor) -> None:
        owner, _, leaf = path.rpartition(".")
        shell.get_submodule(owner)._parameters[leaf] = Parameter(value, requires_grad=False)

    @classmethod
    def from_float(cls, mod: nn.Module, dtype: QDtype = qint8) -> Self:
        """Quantize a float recurrent module's weights to int8 (no calibration).

        ``mod`` must be the float type the layer runs — an ``nn.LSTM`` for
        :class:`LSTM`, an ``nn.GRUCell`` for :class:`GRUCell`.
        """
        from lucid.quantization.observer import PerChannelMinMaxObserver

        if not isinstance(mod, cls._FLOAT):
            raise TypeError(
                f"dynamic {cls.__name__}.from_float expects an "
                f"nn.{cls._FLOAT.__name__}, got {type(mod).__name__}"
            )
        # The shell is overwritten with dequantized weights on every forward;
        # a copy of it, so the caller's float module is left as it was given.
        shell = copy.deepcopy(mod)
        qmod = cls(shell)
        for name, param in shell.named_parameters():
            key = _buffer_key(name)
            if _is_weight(name):
                obs = PerChannelMinMaxObserver(
                    ch_axis=0, qscheme=per_channel_symmetric, qdtype=dtype
                )
                obs(param)
                scale, zero_point = obs.calculate_qparams()
                qmod.register_buffer(
                    key + "_int8", quantize(param, scale, zero_point, dtype, ch_axis=0)
                )
                qmod.register_buffer(key + "_scale", scale)
            else:
                qmod.register_buffer(key, param.detach())
        return qmod

    @override
    def extra_repr(self) -> str:
        shell: nn.Module = object.__getattribute__(self, "_shell")
        return f"{shell.extra_repr()}, dynamic=True"


class LSTM(_DynamicRecurrent[nn.LSTM]):
    """int8-weight LSTM built from a float :class:`~lucid.nn.LSTM`.

    A dynamically quantized recurrent layer: every recurrent weight matrix
    (``weight_ih_l*`` / ``weight_hh_l*``) is stored int8 (per-row symmetric) while
    the biases stay float.  A float ``nn.LSTM`` *shell*, held outside the module
    registry so its float weights never enter the ``state_dict``, is the actual
    compute engine — on each forward its weights are overwritten with the values
    dequantized from the int8 buffers, so only the int8 form is the persistent,
    device-tracked state.  No calibration is required; the per-timestep activation
    quantization inside the cell is left to the real low-precision kernel.  Build
    instances with :func:`lucid.quantization.quantize_dynamic`, the recommended
    path for LSTM-heavy inference.

    Parameters
    ----------
    shell : lucid.nn.LSTM
        The float LSTM whose configuration and cell arithmetic are reused; it holds
        the recurrent weights that :meth:`from_float` quantizes to int8.

    Notes
    -----
    The shell is stored via ``object.__setattr__`` so it stays out of the module
    registry, keeping the serialized state limited to the int8 weight buffers, the
    per-row scales, and the float biases.
    """

    _FLOAT = nn.LSTM

    @override
    def forward(
        self, input: Tensor | PackedSequence, hx: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor | PackedSequence, tuple[Tensor, Tensor]]:
        """Inject dequantized weights into the shell and run it.

        Packed input passes straight through — the shell is a real
        :class:`~lucid.nn.LSTM`, so it handles it."""
        return self._loaded().forward(input, hx)


class GRU(_DynamicRecurrent[nn.GRU]):
    """int8-weight GRU built from a float :class:`~lucid.nn.GRU`.

    The GRU counterpart of :class:`LSTM`: each layer's (and direction's)
    ``weight_ih`` / ``weight_hh`` is stored int8 per-row symmetric under the
    flat name a float GRU's ``state_dict`` uses (``weight_ih_l0_int8``,
    ``weight_ih_l0_scale``), the biases stay float, and a float ``nn.GRU``
    shell outside the module registry runs the recurrence on the dequantized
    weights.  Built by :func:`lucid.quantization.quantize_dynamic`.

    Parameters
    ----------
    shell : lucid.nn.GRU
        The float GRU whose configuration and recurrence are reused.
    """

    _FLOAT = nn.GRU

    @override
    def forward(
        self, input: Tensor | PackedSequence, hx: Tensor | None = None
    ) -> tuple[Tensor | PackedSequence, Tensor]:
        """Inject dequantized weights into the shell and run it."""
        return self._loaded().forward(input, hx)


class LSTMCell(_DynamicRecurrent[nn.LSTMCell]):
    """int8-weight LSTM cell built from a float :class:`~lucid.nn.LSTMCell`.

    ``weight_ih`` / ``weight_hh`` are stored int8 per-row symmetric
    (``weight_ih_int8``, ``weight_ih_scale``), the biases stay float, and a
    float ``nn.LSTMCell`` shell runs one step on the dequantized weights.
    Built by :func:`lucid.quantization.quantize_dynamic`.

    Parameters
    ----------
    shell : lucid.nn.LSTMCell
        The float cell whose configuration and arithmetic are reused.
    """

    _FLOAT = nn.LSTMCell

    @override
    def forward(
        self, input: Tensor, hx: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor]:
        """Inject dequantized weights into the shell and run one step."""
        return self._loaded().forward(input, hx)


class GRUCell(_DynamicRecurrent[nn.GRUCell]):
    """int8-weight GRU cell built from a float :class:`~lucid.nn.GRUCell`.

    Stored and run like :class:`LSTMCell`, with a float ``nn.GRUCell``
    shell.  Built by :func:`lucid.quantization.quantize_dynamic`.

    Parameters
    ----------
    shell : lucid.nn.GRUCell
        The float cell whose configuration and arithmetic are reused.
    """

    _FLOAT = nn.GRUCell

    @override
    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor:
        """Inject dequantized weights into the shell and run one step."""
        return self._loaded().forward(input, hx)


class RNNCell(_DynamicRecurrent[nn.RNNCell]):
    """int8-weight Elman cell built from a float :class:`~lucid.nn.RNNCell`.

    Stored and run like :class:`LSTMCell`, with a float ``nn.RNNCell``
    shell, so its ``nonlinearity`` (``tanh`` or ``relu``) is kept.  Built by
    :func:`lucid.quantization.quantize_dynamic`.

    Parameters
    ----------
    shell : lucid.nn.RNNCell
        The float cell whose configuration and arithmetic are reused.
    """

    _FLOAT = nn.RNNCell

    @override
    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor:
        """Inject dequantized weights into the shell and run one step."""
        return self._loaded().forward(input, hx)
