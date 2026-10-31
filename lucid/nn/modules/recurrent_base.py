"""
The bases the recurrent layers and cells share: :class:`RNNBase` and :class:`RNNCellBase`.
"""

from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._factories.creation import empty
import lucid.nn.init as init
from lucid.nn.modules._recurrent_layout import _check_dropout, _check_sizes, _init_bound


def _read_cell_keys(
    module: Module, state_dict: dict[str, Tensor], prefix: str, *_rest: object
) -> None:
    """Load a GRU / RNN checkpoint saved before the parameters were flat.

    The layers used to hold one cell sub-module per layer and direction, so
    an old checkpoint names ``weight_ih_l0`` as ``cell_l0.weight_ih``.  Each
    such key is read as the flat one it stands for; a checkpoint holding
    both keeps the flat one.  A pre-hook rather than a loader override, so
    ``load_state_dict(assign=True)`` reads the old keys too.
    """
    for name in list(module._parameters):
        param, _, tag = name.partition("_l")
        legacy = f"{prefix}cell_l{tag}.{param}"
        if legacy in state_dict:
            value = state_dict.pop(legacy)
            state_dict.setdefault(f"{prefix}{name}", value)


class _RecurrentParameters(Module):
    """The initialisation every recurrent layer and cell shares.

    Each parameter — input and hidden weights, biases, a projection — is
    drawn from ``U(-1/sqrt(H), 1/sqrt(H))``, ``H`` the hidden size, in the
    order ``parameters()`` yields them.
    """

    hidden_size: int

    def reset_parameters(self) -> None:
        """Re-draw every parameter from ``U(-1/sqrt(hidden_size), 1/sqrt(hidden_size))``."""
        stdv = _init_bound(self.hidden_size)
        for p in self.parameters():
            init.uniform_(p, -stdv, stdv)


#: Gate blocks stacked in one weight matrix, by the reference's mode name.
_MODE_GATES = {"LSTM": 4, "GRU": 3, "RNN_TANH": 1, "RNN_RELU": 1}


class RNNBase(_RecurrentParameters):
    r"""Base class of the multi-layer recurrent modules :class:`RNN`, :class:`LSTM` and :class:`GRU`.

    Holds what the three share: the configuration, the per-layer and
    per-direction weights, their initialisation and the checks on the
    configuration.  The recurrence itself is the subclass's.

    The parameters of layer ``k`` are ``weight_ih_l{k}``, ``weight_hh_l{k}``,
    ``bias_ih_l{k}`` and ``bias_hh_l{k}`` (``_reverse`` appended for the
    backward direction), plus ``weight_hr_l{k}`` for an LSTM with
    ``proj_size > 0`` — the names, shapes and order of the reference, and so
    of the ``state_dict`` keys.  The input and hidden weights stack one block
    per gate: 4 for an LSTM, 3 for a GRU, 1 for an RNN.

    Parameters
    ----------
    mode : {'RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU'}
        The recurrence.
    input_size : int
        Features of the input.
    hidden_size : int
        Features of the hidden state.
    num_layers : int, optional
        Number of stacked layers.  Default: ``1``.
    bias : bool, optional
        Whether the layers have ``bias_ih`` and ``bias_hh``.  Default: ``True``.
    batch_first : bool, optional
        Whether input and output are ``(batch, seq, feature)``.  Default: ``False``.
    dropout : float, optional
        Dropout between layers, in ``[0, 1]``.  Default: ``0.0``.
    bidirectional : bool, optional
        Whether each layer also runs backward.  Default: ``False``.
    proj_size : int, optional
        LSTM hidden-state projection, smaller than ``hidden_size``.  ``0``
        is none; anything else is refused for ``RNN`` and ``GRU``.  Default: ``0``.
    device : DeviceLike, optional
        Device for the parameters.
    dtype : DTypeLike, optional
        Dtype for the parameters.

    Attributes
    ----------
    mode, input_size, hidden_size, num_layers, bias, batch_first, bidirectional, proj_size
        The configuration, as given.
    dropout : float
        The inter-layer dropout probability.

    Raises
    ------
    ValueError
        For an unknown ``mode``, a non-positive size, a ``dropout`` outside
        ``[0, 1]``, or a ``proj_size`` that is negative, not below
        ``hidden_size``, or given to a mode other than ``LSTM``.
    """

    mode: str
    input_size: int
    hidden_size: int
    num_layers: int
    bias: bool
    batch_first: bool
    dropout_val: float
    bidirectional: bool
    proj_size: int

    def __init__(
        self,
        mode: str,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float | int = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the module. See the class docstring for parameter semantics."""
        super().__init__()
        if mode not in _MODE_GATES:
            raise ValueError(f"Unrecognized RNN mode: {mode}")
        _check_sizes(input_size, hidden_size, num_layers)
        if proj_size < 0:
            raise ValueError(f"proj_size must be >= 0, got {proj_size}")
        if proj_size > 0 and mode != "LSTM":
            raise ValueError("proj_size argument is only supported for LSTM, not RNN or GRU")
        if proj_size >= hidden_size and proj_size > 0:
            raise ValueError(
                f"proj_size ({proj_size}) must be smaller than hidden_size ({hidden_size})"
            )
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout_val = _check_dropout(dropout, num_layers)
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self._register_weights(_MODE_GATES[mode], device, dtype)
        if mode != "LSTM":
            self.register_load_state_dict_pre_hook(_read_cell_keys)

    @property
    def dropout(self) -> float:
        """The inter-layer dropout probability."""
        return self.dropout_val

    def _weight_kinds(self) -> tuple[str, ...]:
        """The parameters of one layer and direction, in registration order."""
        kinds: tuple[str, ...] = ("weight_ih", "weight_hh")
        if self.bias:
            kinds += ("bias_ih", "bias_hh")
        if self.proj_size > 0:
            kinds += ("weight_hr",)
        return kinds

    def _tags(self) -> list[str]:
        """``_l{layer}`` and ``_l{layer}_reverse`` for every layer and direction, in order."""
        suffixes = ("", "_reverse")[: 2 if self.bidirectional else 1]
        return [f"_l{layer}{suffix}" for layer in range(self.num_layers) for suffix in suffixes]

    def _register_weights(self, gates: int, device: DeviceLike, dtype: DTypeLike) -> None:
        """Allocate every layer's weights, in the reference's order, and
        initialise them."""
        dirs = 2 if self.bidirectional else 1
        rows = gates * self.hidden_size
        # The width the next layer reads: the projected state when there is a projection.
        recurrent = self.proj_size if self.proj_size > 0 else self.hidden_size
        for index, tag in enumerate(self._tags()):
            width = self.input_size if index < dirs else recurrent * dirs
            shapes: dict[str, tuple[int, ...]] = {
                "weight_ih": (rows, width),
                "weight_hh": (rows, recurrent),
                "bias_ih": (rows,),
                "bias_hh": (rows,),
                "weight_hr": (self.proj_size, self.hidden_size),
            }
            for kind in self._weight_kinds():
                p = Parameter(empty(*shapes[kind], dtype=dtype, device=device))
                self.register_parameter(f"{kind}{tag}", p)
        self.reset_parameters()

    @property
    def all_weights(self) -> list[list[Tensor]]:
        """Each layer and direction's weights, grouped, as the module holds them now."""
        return [
            [getattr(self, f"{kind}{tag}") for kind in self._weight_kinds()] for tag in self._tags()
        ]

    def flatten_parameters(self) -> None:
        """No-op for API compatibility with reference recurrent modules.

        Some external codepaths call ``flatten_parameters()`` to coalesce
        weights for cuDNN; Lucid's BLAS / MLX path has no such concern,
        so this is a placeholder that lets such code run unchanged.
        """


class RNNCellBase(_RecurrentParameters):
    r"""Base class of the single-step cells :class:`RNNCell`, :class:`LSTMCell` and :class:`GRUCell`.

    Holds the four parameters the cells share — ``weight_ih``, ``weight_hh``,
    ``bias_ih`` and ``bias_hh`` — each stacking ``num_chunks`` gate blocks,
    drawn from :math:`U(-1/\sqrt{H}, 1/\sqrt{H})`.

    Parameters
    ----------
    input_size : int
        Features of the input.
    hidden_size : int
        Features of the hidden state.
    bias : bool
        Whether the cell has ``bias_ih`` and ``bias_hh``.
    num_chunks : int
        Gate blocks per weight: 4 for an LSTM cell, 3 for a GRU cell, 1 for an RNN cell.
    device : DeviceLike, optional
        Device for the parameters.
    dtype : DTypeLike, optional
        Dtype for the parameters.

    Attributes
    ----------
    weight_ih : Parameter, shape ``(num_chunks * hidden_size, input_size)``
        The input-to-hidden weights.
    weight_hh : Parameter, shape ``(num_chunks * hidden_size, hidden_size)``
        The hidden-to-hidden weights.
    bias_ih, bias_hh : Parameter or None, shape ``(num_chunks * hidden_size,)``
        The biases; ``None`` when ``bias`` is ``False``.
    """

    input_size: int
    hidden_size: int
    bias: bool
    weight_ih: Parameter
    weight_hh: Parameter
    bias_ih: Parameter | None
    bias_hh: Parameter | None

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool,
        num_chunks: int,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the cell. See the class docstring for parameter semantics."""
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        rows = num_chunks * hidden_size
        self.weight_ih = Parameter(empty(rows, input_size, dtype=dtype, device=device))
        self.weight_hh = Parameter(empty(rows, hidden_size, dtype=dtype, device=device))
        self.bias_ih = Parameter(empty(rows, dtype=dtype, device=device)) if bias else None
        self.bias_hh = Parameter(empty(rows, dtype=dtype, device=device)) if bias else None
        self.reset_parameters()
