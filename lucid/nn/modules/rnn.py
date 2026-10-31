"""
Recurrent modules: LSTM, GRU, RNN.
"""

from typing import override

from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike, DTypeLike
from lucid._factories.creation import zeros
import lucid as _lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.nn.functional.linear import linear
from lucid.nn.utils.rnn import PackedSequence
from lucid.nn.modules.recurrent_base import RNNBase, RNNCellBase
from lucid.nn.modules._recurrent_layout import (
    _inter_layer_dropout,
    _cell_batch,
    _cell_unbatch,
    _lstm_cell_step,
    _packed_segments,
    _restore_layout,
    _sequence_major,
)
from lucid.nn.functional.activations import tanh, relu, sigmoid
from lucid import stack

# ``Tensor`` is imported at runtime above rather than under TYPE_CHECKING so
# the annotations on ``forward`` can be resolved.  PEP 649 keeps them lazy
# only until something asks for the value, and ``inspect.signature`` /
# ``help()`` / ``typing.get_type_hints`` all do — each raised ``NameError``.


# ── Helpers ───────────────────────────────────────────────────────────────────


def _cat_last(a: Tensor, b: Tensor) -> Tensor:
    """Concatenate two tensors along the last dimension."""
    return _lucid.cat([a, b], a.ndim - 1)


def _slice(x: Tensor, *index: slice) -> Tensor:
    """``x[index]`` — or ``x`` itself when every slice spans its whole dim.

    Indexing that selects everything returns a view of its own, a separate
    tensor over the same storage: one more dispatched op and, on Metal, one
    more allocation.  The recurrences slice a batch prefix off every state
    and input on every call so that a packed schedule can shrink it, but on
    a dense batch the prefix is the whole tensor and the slice only feeds
    the next op — nothing ever holds it as a tensor of its own.
    """
    for sl, n in zip(index, x.shape, strict=False):  # index slices leading axes only
        if sl.indices(int(n)) != (0, int(n), 1):
            return x[index]
    return x


def _carry(state: Tensor, updated: Tensor, bs: int) -> Tensor:
    """Splice a recomputed prefix back onto a carried state.

    ``state`` is ``(1, B, H)`` (or ``(B, H)`` for the cell-based modules)
    and ``updated`` covers only the first ``bs`` rows.  The untouched tail
    is exactly what makes the packed recurrence work in both directions:

    * running forwards the active set only shrinks, so the tail already
      holds each finished sequence's finished state — leave it and ``h``
      *is* ``h_n`` when the loop ends;
    * running backwards it only grows, so the tail still holds ``h_0``
      for sequences that have not started yet, which is what they need.
    """
    axis = 1 if state.ndim == 3 else 0
    if bs >= int(state.shape[axis]):
        return updated
    tail = state[:, bs:] if axis == 1 else state[bs:]
    return _lucid.cat([updated, tail], axis)


def _pad_rows(x: Tensor, total: int) -> Tensor:
    """Zero-pad the batch axis of a ``(bs, H)`` step output up to ``total``."""
    bs = int(x.shape[0])
    if bs >= total:
        return x
    pad = zeros(total - bs, *tuple(int(v) for v in x.shape[1:]), device=x.device, dtype=x.dtype)
    return _lucid.cat([x, pad], 0)


# A recurrent layer's weights for one layer and direction:
# ``(weight_ih, weight_hh, bias_ih, bias_hh)``.
type _Weights = tuple[Tensor, Tensor, Tensor | None, Tensor | None]


def _rnn_step(x: Tensor, h: Tensor, w: _Weights, nonlinearity: str) -> Tensor:
    """One Elman step, ``(B, I)`` and ``(B, H)`` to ``(B, H)``."""
    pre = linear(x, w[0], w[2]) + linear(h, w[1], w[3])
    return tanh(pre) if nonlinearity == "tanh" else relu(pre)


def _gru_step(x: Tensor, h: Tensor, w: _Weights) -> Tensor:
    """One GRU step, ``(B, I)`` and ``(B, H)`` to ``(B, H)``."""
    hs = int(h.shape[-1])
    gates_x = linear(x, w[0], w[2])
    gates_h = linear(h, w[1], w[3])
    r = sigmoid(gates_x[:, :hs] + gates_h[:, :hs])
    z = sigmoid(gates_x[:, hs : 2 * hs] + gates_h[:, hs : 2 * hs])
    n = tanh(gates_x[:, 2 * hs :] + r * gates_h[:, 2 * hs :])
    return (1.0 - z) * n + z * h


def _check_nonlinearity(nonlinearity: str) -> str:
    """Refuse an activation the Elman recurrence does not have."""
    if nonlinearity not in ("tanh", "relu"):
        raise _C_engine.InvalidArgument(
            f"Unknown nonlinearity {nonlinearity!r}. Select from 'tanh' or 'relu'."
        )
    return nonlinearity


class _FlatRecurrent(RNNBase):
    """GRU and RNN: the stacked recurrence over the flat per-layer parameters.

    The parameters are those :class:`RNNBase` registers.  Each subclass
    sets ``_step``.
    """

    def _weights(self, layer: int, reverse: bool) -> _Weights:
        """One layer and direction's ``(weight_ih, weight_hh, bias_ih, bias_hh)``."""
        tag = f"_l{layer}{'_reverse' if reverse else ''}"
        # Attribute reads, not ``_parameters``: a weight under weight norm, an
        # orthogonal parametrization or a prune mask is no longer in that
        # registry, and the attribute is what holds its current value.
        w_ih, w_hh = getattr(self, f"weight_ih{tag}"), getattr(self, f"weight_hh{tag}")
        return (
            w_ih,
            w_hh,
            getattr(self, f"bias_ih{tag}", None),
            getattr(self, f"bias_hh{tag}", None),
        )

    def _step(self, x: Tensor, h: Tensor, w: _Weights) -> Tensor:
        """One timestep of this layer's cell."""
        raise NotImplementedError

    def _run(
        self, kind: str, x: Tensor | PackedSequence, hx: Tensor | None
    ) -> tuple[Tensor | PackedSequence, Tensor]:
        """The stacked recurrence: each layer, each direction, each step.

        A layer's output goes through the inter-layer dropout to the next.
        A dense input runs every sequence the full length, so ``_carry``
        and ``_pad_rows`` both become identities.
        """
        dirs = 2 if self.bidirectional else 1
        layout, (h0,) = _sequence_major(
            kind, x, self.batch_first, (("h0", hx, self.hidden_size),), self.num_layers * dirs
        )
        seq, batch_sizes = layout.seq, layout.batch_sizes
        T: int = seq.shape[0]
        B: int = seq.shape[1]
        h_n: list[Tensor] = []
        inp = seq
        for layer in range(self.num_layers):
            dir_outs: list[list[Tensor]] = []
            for direction in range(dirs):
                w = self._weights(layer, direction == 1)
                h: Tensor = h0[layer * dirs + direction]
                outs: list[Tensor] = []
                for t in range(T - 1, -1, -1) if direction == 1 else range(T):
                    bs = batch_sizes[t]
                    step = self._step(_slice(inp[t], slice(0, bs)), _slice(h, slice(0, bs)), w)
                    h = _carry(h, step, bs)
                    outs.append(_pad_rows(step, B))
                if direction == 1:
                    outs.reverse()
                h_n.append(h)
                dir_outs.append(outs)
            if dirs == 2:
                layer_out = stack([_cat_last(dir_outs[0][t], dir_outs[1][t]) for t in range(T)], 0)
            else:
                layer_out = stack(dir_outs[0], 0)
            inp = _inter_layer_dropout(
                layer_out, self.dropout_val, layer, self.num_layers, self.training
            )
        out, (h_n_tensor,) = _restore_layout(layout, inp, [stack(h_n, 0)])
        return out, h_n_tensor

    @override
    def forward(
        self, input: Tensor | PackedSequence, hx: Tensor | None = None
    ) -> tuple[Tensor | PackedSequence, Tensor]:
        """Run the recurrent forward pass.

        Parameters
        ----------
        input : Tensor
            See the class docstring.
        hx : Tensor
            See the class docstring.

        Returns
        -------
        Tensor or tuple of Tensor
            Output and (optionally) the new hidden state; see the class docstring.
        """
        return self._run(type(self).__name__, input, hx)


# ── LSTM ──────────────────────────────────────────────────────────────────────


class LSTM(RNNBase):
    r"""Long Short-Term Memory (LSTM) recurrent layer.

    Applies a multi-layer LSTM over an input sequence.  At each time step
    :math:`t` the following gated update equations are evaluated:

    .. math::

        i_t &= \sigma(W_{ii}\,x_t + b_{ii} + W_{hi}\,h_{t-1} + b_{hi}) \\
        f_t &= \sigma(W_{if}\,x_t + b_{if} + W_{hf}\,h_{t-1} + b_{hf}) \\
        g_t &= \tanh(W_{ig}\,x_t + b_{ig} + W_{hg}\,h_{t-1} + b_{hg}) \\
        o_t &= \sigma(W_{io}\,x_t + b_{io} + W_{ho}\,h_{t-1} + b_{ho}) \\
        c_t &= f_t \odot c_{t-1} + i_t \odot g_t \\
        h_t &= o_t \odot \tanh(c_t)

    where :math:`\sigma` is the sigmoid function, :math:`\odot` is the
    element-wise (Hadamard) product, :math:`x_t` is the input at step
    :math:`t`, and :math:`h_{t-1}`, :math:`c_{t-1}` are the hidden and
    cell states carried from the previous step.

    The four gates have the following roles:

    * **Input gate** :math:`i_t` — controls how much new information
      enters the cell state.
    * **Forget gate** :math:`f_t` — controls how much of the previous
      cell state is retained.
    * **Cell gate** :math:`g_t` — candidate new content to add to the
      cell state.
    * **Output gate** :math:`o_t` — controls what portion of the cell
      state is exposed as the hidden state :math:`h_t`.

    When ``num_layers > 1`` the hidden state output of layer :math:`\ell`
    becomes the input to layer :math:`\ell + 1`.  Inter-layer dropout
    (probability ``dropout``) is applied between every pair of adjacent
    layers during training.

    When ``bidirectional=True``, two independent LSTMs process the
    sequence in opposite directions.  Their output hidden states are
    concatenated along the feature dimension at each time step, so the
    output feature size is ``2 * hidden_size``.  The hidden/cell state
    tensors ``h_n`` and ``c_n`` pack both directions along their leading
    axis in the order ``[fwd_l0, rev_l0, fwd_l1, rev_l1, ...]``.

    Parameters
    ----------
    input_size : int
        Number of expected features in each input vector :math:`x_t`.
    hidden_size : int
        Number of features in the hidden state :math:`h_t` (denoted
        :math:`H` below).
    num_layers : int, optional
        Number of stacked recurrent layers.  Default: ``1``.
    bias : bool, optional
        If ``False`` all bias terms are treated as zero (the parameter
        tensors still exist for API compatibility, but are set to zero
        and contribute nothing numerically).  Default: ``True``.
    batch_first : bool, optional
        If ``True`` the expected input/output shape is
        ``(N, L, input_size)`` / ``(N, L, H * D)`` rather than the
        default ``(L, N, input_size)`` / ``(L, N, H * D)``.
        Default: ``False``.
    dropout : float, optional
        Dropout probability applied to the outputs of every LSTM layer
        except the last.  ``0.0`` disables dropout.  Default: ``0.0``.
    bidirectional : bool, optional
        If ``True`` use a bidirectional LSTM; doubles the output feature
        size and the leading dimension of ``h_n`` / ``c_n``.
        Default: ``False``.
    proj_size : int, optional
        If ``> 0``, adds a learnable linear projection of size
        ``proj_size`` after each LSTM cell output, reducing the
        recurrent hidden dimension fed to the next time step.
        Default: ``0`` (disabled).
    device : DeviceLike, optional
        Device on which to allocate weight tensors.
    dtype : DTypeLike, optional
        Data type for weight tensors.

    Attributes
    ----------
    weight_ih_l0 : Parameter, shape ``(4H, I)``
        Input–hidden weight matrix for layer 0, forward direction.
        Stacks the four gate weight matrices :math:`[W_{ii}; W_{if};
        W_{ig}; W_{io}]`.  Here :math:`I` is ``input_size`` and
        :math:`H` is ``hidden_size``.
    weight_hh_l0 : Parameter, shape ``(4H, H)``
        Hidden–hidden weight matrix for layer 0, forward direction.
        Stacks :math:`[W_{hi}; W_{hf}; W_{hg}; W_{ho}]`.
        When ``proj_size > 0`` the second dimension shrinks to
        ``proj_size`` (the recurrent state is the projected output).
    bias_ih_l0 : Parameter, shape ``(4H,)``
        Input–hidden bias for layer 0, forward direction.  Present only
        when ``bias=True``.
    bias_hh_l0 : Parameter, shape ``(4H,)``
        Hidden–hidden bias for layer 0, forward direction.  Present only
        when ``bias=True``.

    For layer ``k`` in the forward direction substitute ``l0`` →
    ``l{k}``; for the backward direction append ``_reverse``,
    e.g. ``weight_ih_l1_reverse``.

    Shape
    -----
    * **Input** ``x``: ``(L, N, input_size)`` or ``(N, L, input_size)``
      when ``batch_first=True``.  ``L`` = sequence length,
      ``N`` = batch size.
    * **h_0**, **c_0** (optional): ``(D * num_layers, N, H)`` where
      ``D = 2`` if bidirectional else ``1``.  When omitted they default
      to zero tensors.
    * **output**: ``(L, N, D * H)`` or ``(N, L, D * H)`` when
      ``batch_first=True``.
    * **h_n**: ``(D * num_layers, N, H)`` — hidden state after the last
      time step for every layer and direction.
    * **c_n**: ``(D * num_layers, N, H)`` — cell state after the last
      time step for every layer and direction.
    * **Unbatched**: an ``(L, input_size)`` input runs as a batch of one
      whatever ``batch_first`` says.  Its states are ``(D * num_layers, H)``
      (``h`` is ``proj_size`` wide when projected), the output is
      ``(L, D * H)``, and a supplied state of any other shape raises
      ``ValueError``.

    Notes
    -----
    All weight matrices are initialised with a uniform distribution
    :math:`\mathcal{U}(-1/\sqrt{H},\, 1/\sqrt{H})`, matching the
    initialisation convention of the reference framework.

    The C++ engine processes a single layer and a single direction per
    call.  Multi-layer and bidirectional configurations are composed
    entirely in Python: the module loops over layers and directions,
    applies inter-layer dropout, and concatenates bidirectional outputs.

    :meth:`flatten_parameters` is a no-op provided purely for API
    compatibility with codebases that call it before the forward pass.

    A ``PackedSequence`` may be passed directly: padded steps are kept out
    of the recurrence, ``h_n`` holds each sequence's state at its *own*
    last step, and the output comes back packed.  ``enforce_sorted=False``
    is fine — the batch is returned in the order you passed it.

    Examples
    --------
    Basic sequence encoding (batch-first convention):

    >>> import lucid, lucid.nn as nn
    >>> lstm = nn.LSTM(input_size=10, hidden_size=20, batch_first=True)
    >>> x = lucid.randn(2, 5, 10)          # (N=2, L=5, I=10)
    >>> output, (h_n, c_n) = lstm(x)
    >>> output.shape, h_n.shape, c_n.shape
    ((2, 5, 20), (1, 2, 20), (1, 2, 20))

    Bidirectional, 2-layer LSTM with dropout:

    >>> lstm2 = nn.LSTM(
    ...     input_size=16, hidden_size=32,
    ...     num_layers=2, dropout=0.3,
    ...     bidirectional=True, batch_first=True,
    ... )
    >>> x2 = lucid.randn(4, 12, 16)        # (N=4, L=12, I=16)
    >>> out2, (h_n2, c_n2) = lstm2(x2)
    >>> out2.shape    # D*H = 2*32 = 64
    (4, 12, 64)
    >>> h_n2.shape    # D*num_layers = 4
    (4, 4, 32)

    See Also
    --------
    LSTMCell : Single time-step LSTM cell.
    GRU : Gated Recurrent Unit (simpler, no cell state).
    RNN : Vanilla Elman RNN.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the LSTM module. See the class docstring for parameter semantics."""
        # proj_size > 0 with multi-layer / bidirectional configurations is
        # composed in Python by looping single-layer engine calls — see
        # ``forward``.  The C++ engine itself still handles only the
        # single-layer single-direction case; the Python wrapper bridges
        # the gap.
        super().__init__(
            "LSTM",
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            proj_size,
            device,
            dtype,
        )

    def _reverse_along_time(self, x: Tensor) -> Tensor:
        """Flip a sequence-major tensor along the time (axis-0) dimension.

        Implemented via ``gather`` so the backward path works correctly —
        the engine's ``Tensor.flip`` backward is currently broken.
        """
        # Reverse range [T-1, T-2, …, 0] built on the engine.
        T: int = int(x.shape[0])
        rev_1d: Tensor = _lucid.arange(T - 1, -1, -1, dtype=_lucid.int32, device=x.device)
        target_shape: list[int] = [1] * x.ndim
        target_shape[0] = T
        bcast_shape: list[int] = [int(s) for s in x.shape]
        idx: Tensor = rev_1d.reshape(target_shape).broadcast_to(tuple(bcast_shape)).contiguous()
        return _lucid.gather(x, 0, idx)

    def _run_direction(
        self,
        layer_input: Tensor,
        h0_layer: Tensor,
        c0_layer: Tensor,
        layer: int,
        direction: int,
        batch_sizes: list[int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run one layer / direction, honouring a packed active-set schedule.

        The engine takes a whole ``(T, B, *)`` block, so rather than stepping
        per timestep this splits time into runs where the active batch size
        is constant and makes one call per run.  A dense input is a single
        run covering everything, which is byte-for-byte the previous single
        call — the packed schedule degenerates rather than branching.

        Args:
            layer_input: ``(T, B, *)`` input, padded if packed.
            h0_layer: ``(1, B, *)`` initial hidden state.
            c0_layer: ``(1, B, *)`` initial cell state.
            layer: Index of the stacked layer whose weights to use.
            direction: ``0`` forward, ``1`` backward.
            batch_sizes: Active-set size per timestep, non-increasing.

        Returns:
            ``(output, h_n, c_n)`` with the output padded back to ``B`` and
            the states holding each sequence's value at its *own* last step.
        """
        B = int(layer_input.shape[1])
        segments = _packed_segments(batch_sizes)
        if direction == 1:
            # The reverse pass starts at each sequence's own final element,
            # so walk the runs from the end — which is where the shortest
            # sequences have already dropped out.
            segments = list(reversed(segments))

        h, c = h0_layer, c0_layer
        blocks: list[tuple[int, Tensor]] = []
        for t0, t1, bs in segments:
            chunk = _slice(layer_input, slice(t0, t1), slice(0, bs))
            if direction == 1:
                chunk = self._reverse_along_time(chunk)
            out, h_new, c_new = self._run_single_layer_engine(
                chunk,
                _slice(h, slice(None), slice(0, bs)),
                _slice(c, slice(None), slice(0, bs)),
                layer,
                direction,
            )
            if direction == 1:
                out = self._reverse_along_time(out)
            h = _carry(h, h_new, bs)
            c = _carry(c, c_new, bs)

            if bs < B:
                pad = zeros(t1 - t0, B - bs, int(out.shape[2]), device=out.device, dtype=out.dtype)
                out = _lucid.cat([out, pad], 1)
            blocks.append((t0, out))

        blocks.sort(key=lambda pair: pair[0])
        stacked = blocks[0][1] if len(blocks) == 1 else _lucid.cat([blk for _, blk in blocks], 0)
        return stacked, h, c

    def _run_single_layer_engine(
        self, layer_input: Tensor, h0_layer: Tensor, c0_layer: Tensor, layer: int, direction: int
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run one ``lstm_forward`` engine call for a single layer / direction."""
        suffix: str = "_reverse" if direction == 1 else ""
        x_impl = _unwrap(layer_input)

        def param(name: str) -> _C_engine.TensorImpl:
            # An attribute read: see ``_FlatRecurrent._weights``.
            return _unwrap(getattr(self, f"{name}_l{layer}{suffix}"))

        if self.bias:
            biases = [param("bias_ih"), param("bias_hh")]
        else:
            # The engine requires every operand in the input's dtype.
            zero_b = _C_engine.zeros([4 * self.hidden_size], x_impl.dtype, x_impl.device)
            biases = [zero_b, zero_b]
        weights = [param("weight_ih"), param("weight_hh"), *biases]
        if self.proj_size > 0:
            weights.append(param("weight_hr"))

        # One layer, one direction, sequence-first: the loops above do the rest.
        out_impl, h_impl, c_impl = _C_engine.nn.lstm_forward(
            x_impl,
            _unwrap(h0_layer),
            _unwrap(c0_layer),
            weights,
            self.hidden_size,
            1,
            False,
            False,
            True,
            self.proj_size,
        )
        return _wrap(out_impl), _wrap(h_impl), _wrap(c_impl)

    @override
    def forward(
        self, input: Tensor | PackedSequence, hx: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor | PackedSequence, tuple[Tensor, Tensor]]:
        """Multi-layer × bidirectional forward.

        The C++ engine only handles a single layer in a single direction
        at a time, so this loops over layers and directions, applying
        inter-layer dropout and concatenating bidirectional outputs as
        the input to the next layer.
        """
        num_dirs: int = 2 if self.bidirectional else 1
        L: int = self.num_layers
        rec_size: int = self.proj_size if self.proj_size > 0 else self.hidden_size

        h_in, c_in = (None, None) if hx is None else hx
        # A dense input is one segment: one engine call per layer and direction.
        layout, (h0_full, c0_full) = _sequence_major(
            "LSTM",
            input,
            self.batch_first,
            (("h0", h_in, rec_size), ("c0", c_in, self.hidden_size)),
            L * num_dirs,
        )
        batch_sizes = layout.batch_sizes

        h_n_layers: list[Tensor] = []
        c_n_layers: list[Tensor] = []

        layer_input: Tensor = layout.seq

        for layer in range(L):
            dir_outs: list[Tensor] = []
            for direction in range(num_dirs):
                idx: int = layer * num_dirs + direction
                # Slice (1, B, *) initial state for this layer/direction.
                h0_slice: Tensor = _slice(h0_full, slice(idx, idx + 1))
                c0_slice: Tensor = _slice(c0_full, slice(idx, idx + 1))

                out, h_n, c_n = self._run_direction(
                    layer_input, h0_slice, c0_slice, layer, direction, batch_sizes
                )
                dir_outs.append(out)
                h_n_layers.append(h_n)
                c_n_layers.append(c_n)

            # Combine forward / reverse outputs along the feature axis.
            if num_dirs == 2:
                layer_input = _lucid.cat(dir_outs, 2)
            else:
                layer_input = dir_outs[0]

            layer_input = _inter_layer_dropout(
                layer_input, self.dropout_val, layer, L, self.training
            )

        result, (h_n_final, c_n_final) = _restore_layout(
            layout, layer_input, [_lucid.cat(h_n_layers, 0), _lucid.cat(c_n_layers, 0)]
        )
        return result, (h_n_final, c_n_final)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        s: str = (
            f"{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
            f"bias={self.bias}, batch_first={self.batch_first}, "
            f"dropout={self.dropout_val}, bidirectional={self.bidirectional}"
        )
        if self.proj_size > 0:
            s += f", proj_size={self.proj_size}"
        return s


# ── Pure-Python RNN cells ─────────────────────────────────────────────────────


class RNNCell(RNNCellBase):
    r"""Single time-step Elman RNN cell.

    Computes one recurrent update for a single time step:

    .. math::

        h_t = \phi\!\left(W_{ih}\,x_t + b_{ih} + W_{hh}\,h_{t-1} + b_{hh}\right)

    where :math:`\phi` is either :math:`\tanh` (default) or
    :math:`\text{ReLU}`, selected by ``nonlinearity``.

    Use this cell directly when you need fine-grained control over the
    time loop (e.g. teacher forcing, attention, custom stopping
    criteria).  For standard sequence processing prefer :class:`RNN`.

    Parameters
    ----------
    input_size : int
        Number of features in the input vector :math:`x_t`.
    hidden_size : int
        Number of features in the hidden state :math:`h_t`.
    bias : bool, optional
        If ``False``, no bias terms are used.  Default: ``True``.
    nonlinearity : {'tanh', 'relu'}, optional
        Activation function applied to the pre-activation.
        Default: ``'tanh'``.
    device : DeviceLike, optional
        Device for weight allocation.
    dtype : DTypeLike, optional
        Data type for weight tensors.

    Attributes
    ----------
    weight_ih : Parameter, shape ``(hidden_size, input_size)``
        Input–hidden weight matrix :math:`W_{ih}`.
    weight_hh : Parameter, shape ``(hidden_size, hidden_size)``
        Hidden–hidden weight matrix :math:`W_{hh}`.
    bias_ih : Parameter or None, shape ``(hidden_size,)``
        Input–hidden bias :math:`b_{ih}`.  ``None`` when ``bias=False``.
    bias_hh : Parameter or None, shape ``(hidden_size,)``
        Hidden–hidden bias :math:`b_{hh}`.  ``None`` when ``bias=False``.

    Shape
    -----
    * **x**: ``(N, input_size)`` — batch of input vectors.
    * **hx** (optional): ``(N, hidden_size)`` — initial hidden state.
      Defaults to zeros when ``None``.
    * **Output** ``h_t``: ``(N, hidden_size)``.
    * **Unbatched**: an ``(input_size,)`` input takes and returns
      ``(hidden_size,)`` states.

    Notes
    -----
    Weights are initialised from
    :math:`\mathcal{U}(-1/\sqrt{H},\, 1/\sqrt{H})` where
    :math:`H` = ``hidden_size``.

    For long sequences, vanilla RNNs suffer from the vanishing-gradient
    problem because gradients are multiplied by :math:`W_{hh}` at every
    step.  Prefer :class:`LSTMCell` or :class:`GRUCell` for sequences
    longer than ~20 steps.

    Examples
    --------
    Manual time loop with ``tanh`` nonlinearity:

    >>> import lucid, lucid.nn as nn
    >>> cell = nn.RNNCell(input_size=8, hidden_size=16)
    >>> x_seq = lucid.randn(5, 3, 8)   # (L=5, N=3, I=8)
    >>> h = lucid.zeros(3, 16)
    >>> for t in range(5):
    ...     h = cell(x_seq[t], h)      # (N=3, H=16)
    >>> h.shape
    (3, 16)

    Using ``relu`` nonlinearity (avoids gradient saturation at extremes):

    >>> cell_relu = nn.RNNCell(8, 16, nonlinearity='relu')
    >>> h2 = cell_relu(lucid.randn(4, 8))
    >>> h2.shape
    (4, 16)

    See Also
    --------
    RNN : Multi-layer, multi-step wrapper around :class:`RNNCell`.
    LSTMCell : Single-step LSTM cell (gated, better for long sequences).
    GRUCell : Single-step GRU cell.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the RNNCell module. See the class docstring for parameter semantics."""
        _check_nonlinearity(nonlinearity)
        super().__init__(input_size, hidden_size, bias, 1, device, dtype)
        self.nonlinearity = nonlinearity

    @override
    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor:
        """Run the recurrent forward pass.

        Parameters
        ----------
        input : Tensor
            See the class docstring.
        hx : Tensor
            See the class docstring.

        Returns
        -------
        Tensor or tuple of Tensor
            Output and (optionally) the new hidden state; see the class docstring.
        """
        unbatched, x, (h0,) = _cell_batch("RNNCell", input, (("hx", hx),), self.hidden_size)
        w = (self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        return _cell_unbatch(unbatched, _rnn_step(x, h0, w, self.nonlinearity))

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"{self.input_size}, {self.hidden_size}, nonlinearity={self.nonlinearity!r}"


class LSTMCell(RNNCellBase):
    r"""Single time-step Long Short-Term Memory (LSTM) cell.

    Computes one recurrent update using the full LSTM gating equations:

    .. math::

        \begin{aligned}
        i_t &= \sigma(W_{ii}\,x_t + b_{ii} + W_{hi}\,h_{t-1} + b_{hi})
              & &\text{(input gate)}\\
        f_t &= \sigma(W_{if}\,x_t + b_{if} + W_{hf}\,h_{t-1} + b_{hf})
              & &\text{(forget gate)}\\
        g_t &= \tanh(W_{ig}\,x_t + b_{ig} + W_{hg}\,h_{t-1} + b_{hg})
              & &\text{(cell gate)}\\
        o_t &= \sigma(W_{io}\,x_t + b_{io} + W_{ho}\,h_{t-1} + b_{ho})
              & &\text{(output gate)}\\
        c_t &= f_t \odot c_{t-1} + i_t \odot g_t\\
        h_t &= o_t \odot \tanh(c_t)
        \end{aligned}

    The four weight matrices for the gates are stored as a single
    vertically-stacked parameter of shape ``(4H, *)``, in gate order
    ``[i; f; g; o]`` (i.e. the first ``H`` rows correspond to the input
    gate, the next ``H`` to the forget gate, and so on).

    Parameters
    ----------
    input_size : int
        Number of features in the input vector :math:`x_t`.
    hidden_size : int
        Number of features in the hidden/cell states :math:`h_t, c_t`.
    bias : bool, optional
        If ``False``, all bias terms are omitted.  Default: ``True``.
    device : DeviceLike, optional
        Device for weight allocation.
    dtype : DTypeLike, optional
        Data type for weight tensors.

    Attributes
    ----------
    weight_ih : Parameter, shape ``(4 * hidden_size, input_size)``
        Stacked input–hidden weight matrices
        :math:`[W_{ii}; W_{if}; W_{ig}; W_{io}]`.
    weight_hh : Parameter, shape ``(4 * hidden_size, hidden_size)``
        Stacked hidden–hidden weight matrices
        :math:`[W_{hi}; W_{hf}; W_{hg}; W_{ho}]`.
    bias_ih : Parameter or None, shape ``(4 * hidden_size,)``
        Stacked input–hidden biases.  ``None`` when ``bias=False``.
    bias_hh : Parameter or None, shape ``(4 * hidden_size,)``
        Stacked hidden–hidden biases.  ``None`` when ``bias=False``.

    Shape
    -----
    * **x**: ``(N, input_size)`` — batch of input vectors.
    * **hx** (optional): tuple ``(h_0, c_0)`` each of shape
      ``(N, hidden_size)``.  Defaults to zero tensors when ``None``.
    * **Output**: tuple ``(h_t, c_t)`` each of shape
      ``(N, hidden_size)``.
    * **Unbatched**: an ``(input_size,)`` input takes and returns
      ``(hidden_size,)`` states.

    Notes
    -----
    Weights are initialised from
    :math:`\mathcal{U}(-1/\sqrt{H},\, 1/\sqrt{H})`.

    The forget-gate bias is **not** initialised to 1 by default
    (unlike some implementations).  If you observe vanishing gradients
    on medium-length sequences, consider manually setting
    ``cell.bias_ih.data[H:2H] = 1.0`` after construction.

    Examples
    --------
    Single-step update, carrying state across a manual loop:

    >>> import lucid, lucid.nn as nn
    >>> cell = nn.LSTMCell(input_size=10, hidden_size=20)
    >>> h, c = lucid.zeros(3, 20), lucid.zeros(3, 20)
    >>> x_seq = lucid.randn(7, 3, 10)    # (L=7, N=3, I=10)
    >>> for t in range(7):
    ...     h, c = cell(x_seq[t], (h, c))
    >>> h.shape, c.shape
    ((3, 20), (3, 20))

    Without providing an explicit initial state (zeros used):

    >>> cell2 = nn.LSTMCell(4, 8)
    >>> h2, c2 = cell2(lucid.randn(5, 4))
    >>> h2.shape
    (5, 8)

    See Also
    --------
    LSTM : Multi-layer, multi-step LSTM module.
    RNNCell : Simpler single-step cell without gating.
    GRUCell : Single-step GRU cell (fewer gates, no separate cell state).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the LSTMCell module. See the class docstring for parameter semantics."""
        super().__init__(input_size, hidden_size, bias, 4, device, dtype)

    @override
    def forward(
        self, input: Tensor, hx: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor]:
        """Run the recurrent forward pass.

        Parameters
        ----------
        input : Tensor
            See the class docstring.
        hx : Tensor
            See the class docstring.

        Returns
        -------
        Tensor or tuple of Tensor
            Output and (optionally) the new hidden state; see the class docstring.
        """
        h_in, c_in = (None, None) if hx is None else hx
        unbatched, x, (h0, c0) = _cell_batch(
            "LSTMCell", input, (("hx", h_in), ("cx", c_in)), self.hidden_size
        )
        w = (self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        h1, c1 = _lstm_cell_step(x, (h0, c0), w)
        return _cell_unbatch(unbatched, h1), _cell_unbatch(unbatched, c1)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"{self.input_size}, {self.hidden_size}"


class GRUCell(RNNCellBase):
    r"""Single time-step Gated Recurrent Unit (GRU) cell.

    Computes one recurrent update using the three-gate GRU equations:

    .. math::

        \begin{aligned}
        r_t &= \sigma(W_{ir}\,x_t + b_{ir} + W_{hr}\,h_{t-1} + b_{hr})
              & &\text{(reset gate)}\\
        z_t &= \sigma(W_{iz}\,x_t + b_{iz} + W_{hz}\,h_{t-1} + b_{hz})
              & &\text{(update gate)}\\
        n_t &= \tanh\!\left(W_{in}\,x_t + b_{in}
                + r_t \odot (W_{hn}\,h_{t-1} + b_{hn})\right)
              & &\text{(new gate)}\\
        h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot n_t
        \end{aligned}

    The **reset gate** :math:`r_t` controls how much of the previous
    hidden state leaks into the candidate :math:`n_t`; setting it near
    zero makes the cell ignore past context.  The **update gate**
    :math:`z_t` interpolates between the old hidden state and the
    candidate, allowing the cell to retain information over many steps
    without an explicit cell state.

    The three gate weight matrices are stacked into single parameters
    of shape ``(3H, *)`` in gate order ``[r; z; n]``.

    Parameters
    ----------
    input_size : int
        Number of features in the input vector :math:`x_t`.
    hidden_size : int
        Number of features in the hidden state :math:`h_t`.
    bias : bool, optional
        If ``False``, no bias terms are used.  Default: ``True``.
    device : DeviceLike, optional
        Device for weight allocation.
    dtype : DTypeLike, optional
        Data type for weight tensors.

    Attributes
    ----------
    weight_ih : Parameter, shape ``(3 * hidden_size, input_size)``
        Stacked input–hidden weight matrices
        :math:`[W_{ir}; W_{iz}; W_{in}]`.
    weight_hh : Parameter, shape ``(3 * hidden_size, hidden_size)``
        Stacked hidden–hidden weight matrices
        :math:`[W_{hr}; W_{hz}; W_{hn}]`.
    bias_ih : Parameter or None, shape ``(3 * hidden_size,)``
        Stacked input–hidden biases :math:`[b_{ir}; b_{iz}; b_{in}]`.
        ``None`` when ``bias=False``.
    bias_hh : Parameter or None, shape ``(3 * hidden_size,)``
        Stacked hidden–hidden biases :math:`[b_{hr}; b_{hz}; b_{hn}]`.
        ``None`` when ``bias=False``.

    Shape
    -----
    * **x**: ``(N, input_size)`` — batch of input vectors.
    * **hx** (optional): ``(N, hidden_size)`` — initial hidden state.
      Defaults to zeros when ``None``.
    * **Output** ``h_t``: ``(N, hidden_size)``.
    * **Unbatched**: an ``(input_size,)`` input takes and returns
      ``(hidden_size,)`` states.

    Notes
    -----
    Weights are initialised from
    :math:`\mathcal{U}(-1/\sqrt{H},\, 1/\sqrt{H})`.

    The GRU has fewer parameters than the LSTM (no cell state, three
    gates instead of four) and often converges faster on shorter
    sequences while matching LSTM quality on many benchmarks.

    Examples
    --------
    Manual sequence loop:

    >>> import lucid, lucid.nn as nn
    >>> cell = nn.GRUCell(input_size=8, hidden_size=16)
    >>> x_seq = lucid.randn(6, 4, 8)    # (L=6, N=4, I=8)
    >>> h = lucid.zeros(4, 16)
    >>> for t in range(6):
    ...     h = cell(x_seq[t], h)
    >>> h.shape
    (4, 16)

    No explicit initial state (defaults to zeros):

    >>> cell2 = nn.GRUCell(4, 12)
    >>> h2 = cell2(lucid.randn(3, 4))
    >>> h2.shape
    (3, 12)

    See Also
    --------
    GRU : Multi-layer, multi-step GRU module.
    LSTMCell : Single-step LSTM cell with separate cell state.
    RNNCell : Vanilla single-step cell without gating.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the GRUCell module. See the class docstring for parameter semantics."""
        super().__init__(input_size, hidden_size, bias, 3, device, dtype)

    @override
    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor:
        """Run the recurrent forward pass.

        Parameters
        ----------
        input : Tensor
            See the class docstring.
        hx : Tensor
            See the class docstring.

        Returns
        -------
        Tensor or tuple of Tensor
            Output and (optionally) the new hidden state; see the class docstring.
        """
        unbatched, x, (h0,) = _cell_batch("GRUCell", input, (("hx", hx),), self.hidden_size)
        w = (self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh)
        return _cell_unbatch(unbatched, _gru_step(x, h0, w))

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return f"{self.input_size}, {self.hidden_size}"


# ── Multi-step GRU ────────────────────────────────────────────────────────────


class GRU(_FlatRecurrent):
    r"""Multi-layer Gated Recurrent Unit (GRU) recurrent layer.

    Applies a stack of GRU cells over an input sequence.  At each time
    step :math:`t` the following equations are evaluated (see
    :class:`GRUCell` for the full derivation):

    .. math::

        \begin{aligned}
        r_t &= \sigma(W_{ir}\,x_t + W_{hr}\,h_{t-1} + b_r) \\
        z_t &= \sigma(W_{iz}\,x_t + W_{hz}\,h_{t-1} + b_z) \\
        n_t &= \tanh\!\left(W_{in}\,x_t + b_{in}
                + r_t \odot (W_{hn}\,h_{t-1} + b_{hn})\right) \\
        h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot n_t
        \end{aligned}

    The output of layer :math:`\ell` is used as the input of layer
    :math:`\ell + 1`.  When ``bidirectional=True`` two GRUs process the
    sequence in opposite directions and their outputs are concatenated
    along the feature axis at every time step.

    Inter-layer dropout (probability ``dropout``) is applied between
    adjacent layers during training, but not after the final layer.

    Parameters
    ----------
    input_size : int
        Number of expected features in the input :math:`x_t`.
    hidden_size : int
        Number of features in the hidden state :math:`h_t` (denoted
        :math:`H` below).
    num_layers : int, optional
        Number of stacked GRU layers.  Default: ``1``.
    bias : bool, optional
        If ``False``, all bias parameters are omitted.  Default: ``True``.
    batch_first : bool, optional
        If ``True`` the input/output tensors have shape
        ``(N, L, *)`` instead of the default ``(L, N, *)``.
        Default: ``False``.
    dropout : float, optional
        Dropout probability applied after every layer except the last.
        ``0.0`` disables dropout.  Default: ``0.0``.
    bidirectional : bool, optional
        If ``True``, a bidirectional GRU is used; the output feature
        dimension becomes ``2 * hidden_size``.  Default: ``False``.
    device : DeviceLike, optional
        Device for weight allocation.
    dtype : DTypeLike, optional
        Data type for weight tensors.

    Shape
    -----
    * **Input** ``x``: ``(L, N, input_size)`` or ``(N, L, input_size)``
      when ``batch_first=True``.
    * **h_0** (optional): ``(D * num_layers, N, H)`` where ``D = 2`` if
      bidirectional else ``1``.  Defaults to zeros.
    * **output**: ``(L, N, D * H)`` or ``(N, L, D * H)``.
    * **h_n**: ``(D * num_layers, N, H)`` — hidden state at the final
      time step for each layer and direction.
    * **Unbatched**: an ``(L, input_size)`` input runs as a batch of one
      whatever ``batch_first`` says.  Its states are ``(D * num_layers, H)``,
      the output is ``(L, D * H)``, and a supplied ``h_0`` of any other
      shape raises ``ValueError``.

    Notes
    -----
    The parameters are ``weight_ih_l{k}`` ``(3H, I)``, ``weight_hh_l{k}``
    ``(3H, H)``, ``bias_ih_l{k}`` and ``bias_hh_l{k}`` ``(3H,)``, with
    ``_reverse`` for the backward direction — as for :class:`LSTM`.  A
    checkpoint saved under the older ``cell_l{k}.weight_ih`` names loads.

    :meth:`flatten_parameters` is a no-op retained for API
    compatibility.

    A ``PackedSequence`` may be passed directly: padded steps are kept out
    of the recurrence, ``h_n`` holds each sequence's state at its *own*
    last step, and the output comes back packed.  ``enforce_sorted=False``
    is fine — the batch is returned in the order you passed it.

    Examples
    --------
    Two-layer GRU, batch-first:

    >>> import lucid, lucid.nn as nn
    >>> gru = nn.GRU(8, 16, num_layers=2, batch_first=True)
    >>> x = lucid.randn(2, 5, 8)       # (N=2, L=5, I=8)
    >>> out, h_n = gru(x)
    >>> out.shape, h_n.shape
    ((2, 5, 16), (2, 2, 16))

    Bidirectional GRU:

    >>> gru_bi = nn.GRU(8, 16, bidirectional=True, batch_first=True)
    >>> x2 = lucid.randn(3, 10, 8)
    >>> out2, h_n2 = gru_bi(x2)
    >>> out2.shape    # D*H = 2*16 = 32
    (3, 10, 32)
    >>> h_n2.shape    # D*num_layers = 2*1 = 2
    (2, 3, 16)

    See Also
    --------
    GRUCell : Single time-step GRU cell.
    LSTM : Long Short-Term Memory (carries a separate cell state).
    RNN : Vanilla Elman RNN (no gating).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the GRU module. See the class docstring for parameter semantics."""
        super().__init__(
            "GRU",
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            device=device,
            dtype=dtype,
        )

    @override
    def _step(self, x: Tensor, h: Tensor, w: _Weights) -> Tensor:
        return _gru_step(x, h, w)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
            f"batch_first={self.batch_first}, dropout={self.dropout_val}, "
            f"bidirectional={self.bidirectional}"
        )


# ── Multi-step RNN ────────────────────────────────────────────────────────────


class RNN(_FlatRecurrent):
    r"""Multi-layer Elman recurrent neural network (RNN).

    Applies a stack of Elman RNN cells over an input sequence.  At each
    time step :math:`t` the hidden state is updated by:

    .. math::

        h_t = \phi\!\left(W_{ih}\,x_t + b_{ih} + W_{hh}\,h_{t-1} + b_{hh}\right)

    where :math:`\phi` is either :math:`\tanh` (default) or
    :math:`\text{ReLU}`, controlled by ``nonlinearity``.

    The output of layer :math:`\ell` is fed as the input of layer
    :math:`\ell + 1`.  When ``bidirectional=True``, two RNNs process the
    sequence in opposite directions and their outputs are concatenated
    along the feature axis at each time step.

    Inter-layer dropout (probability ``dropout``) is applied between
    adjacent layers during training.

    .. warning::

        Vanilla RNNs are prone to the **vanishing gradient problem**:
        gradients are multiplied by :math:`W_{hh}` at every time step,
        causing them to shrink exponentially for sequences longer than
        ~10–20 steps.  For longer sequences, prefer :class:`LSTM` or
        :class:`GRU`, which use gating mechanisms to maintain gradient
        flow.

    Parameters
    ----------
    input_size : int
        Number of expected features in the input :math:`x_t`.
    hidden_size : int
        Number of features in the hidden state :math:`h_t` (denoted
        :math:`H` below).
    num_layers : int, optional
        Number of stacked RNN layers.  Default: ``1``.
    nonlinearity : {'tanh', 'relu'}, optional
        Activation function :math:`\phi`.  ``'tanh'`` is recommended for
        most use cases; ``'relu'`` can help when gradients vanish with
        ``tanh``.  Default: ``'tanh'``.
    bias : bool, optional
        If ``False``, all bias parameters are omitted.  Default: ``True``.
    batch_first : bool, optional
        If ``True``, input/output tensors are ``(N, L, *)`` instead of
        the default ``(L, N, *)``.  Default: ``False``.
    dropout : float, optional
        Dropout probability applied after each layer except the last.
        ``0.0`` disables dropout.  Default: ``0.0``.
    bidirectional : bool, optional
        If ``True``, use a bidirectional RNN; the output feature
        dimension becomes ``2 * hidden_size``.  Default: ``False``.
    device : DeviceLike, optional
        Device for weight allocation.
    dtype : DTypeLike, optional
        Data type for weight tensors.

    Shape
    -----
    * **Input** ``x``: ``(L, N, input_size)`` or ``(N, L, input_size)``
      when ``batch_first=True``.
    * **h_0** (optional): ``(D * num_layers, N, H)`` where ``D = 2`` if
      bidirectional else ``1``.  Defaults to zeros.
    * **output**: ``(L, N, D * H)`` or ``(N, L, D * H)``.
    * **h_n**: ``(D * num_layers, N, H)`` — hidden state at the final
      time step for each layer and direction.
    * **Unbatched**: an ``(L, input_size)`` input runs as a batch of one
      whatever ``batch_first`` says.  Its states are ``(D * num_layers, H)``,
      the output is ``(L, D * H)``, and a supplied ``h_0`` of any other
      shape raises ``ValueError``.

    Notes
    -----
    The parameters are ``weight_ih_l{k}`` ``(H, I)``, ``weight_hh_l{k}``
    ``(H, H)``, ``bias_ih_l{k}`` and ``bias_hh_l{k}`` ``(H,)``, with
    ``_reverse`` for the backward direction — as for :class:`LSTM`.  A
    checkpoint saved under the older ``cell_l{k}.weight_ih`` names loads.

    :meth:`flatten_parameters` is a no-op kept for API compatibility.

    A ``PackedSequence`` may be passed directly: padded steps are kept out
    of the recurrence, ``h_n`` holds each sequence's state at its *own*
    last step, and the output comes back packed.  ``enforce_sorted=False``
    is fine — the batch is returned in the order you passed it.

    Examples
    --------
    Two-layer RNN, batch-first:

    >>> import lucid, lucid.nn as nn
    >>> rnn = nn.RNN(8, 16, num_layers=2, batch_first=True)
    >>> x = lucid.randn(2, 5, 8)       # (N=2, L=5, I=8)
    >>> out, h_n = rnn(x)
    >>> out.shape, h_n.shape
    ((2, 5, 16), (2, 2, 16))

    Bidirectional RNN with ReLU activation:

    >>> rnn_bi = nn.RNN(
    ...     8, 16, nonlinearity='relu',
    ...     bidirectional=True, batch_first=True,
    ... )
    >>> x2 = lucid.randn(3, 7, 8)
    >>> out2, h_n2 = rnn_bi(x2)
    >>> out2.shape    # D*H = 2*16 = 32
    (3, 7, 32)
    >>> h_n2.shape    # D*num_layers = 2*1 = 2
    (2, 3, 16)

    See Also
    --------
    RNNCell : Single time-step Elman cell.
    LSTM : Gated RNN with a separate cell state (better for long seqs).
    GRU : Gated RNN without a separate cell state.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device: DeviceLike = None,
        dtype: DTypeLike = None,
    ) -> None:
        """Initialise the RNN module. See the class docstring for parameter semantics."""
        mode = "RNN_RELU" if _check_nonlinearity(nonlinearity) == "relu" else "RNN_TANH"
        super().__init__(
            mode,
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
            device=device,
            dtype=dtype,
        )
        self.nonlinearity = nonlinearity

    @override
    def _step(self, x: Tensor, h: Tensor, w: _Weights) -> Tensor:
        return _rnn_step(x, h, w, self.nonlinearity)

    @override
    def extra_repr(self) -> str:
        """Return a string representation of the layer's configuration."""
        return (
            f"{self.input_size}, {self.hidden_size}, num_layers={self.num_layers}, "
            f"nonlinearity={self.nonlinearity!r}, batch_first={self.batch_first}, "
            f"dropout={self.dropout_val}, bidirectional={self.bidirectional}"
        )
