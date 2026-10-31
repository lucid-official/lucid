"""Observers — collect activation / weight statistics to derive quant params.

An *observer* watches tensors flow through a model during a calibration
pass and accumulates the statistics (running min/max, or a histogram)
needed to choose ``(scale, zero_point)``.  Each observer is an
:class:`~lucid.nn.Module`, so its statistics live in **buffers** that
persist through ``state_dict`` and move with ``.to(device)``; its
``forward`` observes the input and returns it unchanged (identity), which
lets it be dropped into a model or wrapped by a :class:`FakeQuantize`.

Statistics are not differentiable.  Every statistics-gathering ``forward``
folds its batch in under :func:`lucid.no_grad`, and every buffer is written
through ``_store_running_stat``, which keeps no autograd history and
writes in place (re-registering only when the buffer changes shape or
device, as the scalar seed does on a per-channel observer's first batch).
A buffer that carried a step's graph would chain every later step onto
it and keep each step's activations alive until memory runs out.

All statistics are computed with the public ``lucid.*`` op surface — no
external libraries (H4); the :class:`HistogramObserver` uses the engine
``histc`` primitive rather than a numpy histogram.
"""

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, cast, override

import lucid
import lucid.nn as nn
from lucid._engine_scope import untraced
from lucid.quantization._qparams import calculate_qparams
from lucid.quantization._qscheme import (
    QDtype,
    QScheme,
    per_channel_symmetric,
    per_tensor_affine,
    qint8,
    quint8,
)

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


def _store_running_stat(module: nn.Module, name: str, value: Tensor) -> None:
    """Write ``value`` into ``module``'s statistics buffer ``name``, graph-free.

    The one place observer and fake-quantize statistics are stored.  A
    statistic computed from a grad-requiring input would otherwise carry
    that step's graph; the next step folds the buffer into its own update,
    so every step's activations stay reachable and a training loop grows
    until it runs out of memory.  The value is written in place, so whatever
    holds the buffer (a compile trace, a ``state_dict``) sees the update.
    It is re-registered only when the buffer changes shape or device: a
    per-channel observer's scalar seed becoming a ``(C,)`` vector, or a
    fixed qparam that lives on another device.
    """
    value = value.detach()
    current = module._buffers.get(name)
    if (
        current is not None
        and current.shape == value.shape
        and current.dtype == value.dtype
        and current.device == value.device
    ):
        with lucid.no_grad():
            current.copy_(value)
        return
    module.register_buffer(name, value)


def _observes[O: ObserverBase](
    forward: Callable[[O, Tensor], Tensor],
) -> Callable[[O, Tensor], Tensor]:
    """Run an observer's ``forward`` graph-free, on buffers aligned to ``x``'s device.

    Every statistics-gathering ``forward`` wears this, so none of them can
    record the autograd graph into a buffer; the gradient reaches ``x``
    only through whatever consumes the returned tensor.
    """

    @functools.wraps(forward)
    def observe(self: O, x: Tensor) -> Tensor:
        with lucid.no_grad():
            self._align_running_buffers(x)
            return forward(self, x)

    return observe


class ObserverBase(nn.Module):
    r"""Abstract base for calibration observers that yield ``(scale, zero_point)``.

    An *observer* is the calibration workhorse of the quantization subsystem: an
    :class:`~lucid.nn.Module` that watches a tensor stream during a calibration pass,
    accumulates a summary statistic (a running min/max, a channel-wise range, or a value
    histogram), and on demand distills that statistic into the ``(scale, zero_point)``
    pair a quantized tensor needs. Its ``forward`` is the *identity* — it observes ``x``
    and returns it untouched — so an observer can be spliced inline into a prepared graph
    or wrapped by a :class:`FakeQuantize` without perturbing numerics. Concrete
    subclasses override :meth:`forward` (to fold each batch into the running statistic)
    and :meth:`calculate_qparams` (to reduce that statistic to qparams).

    **What ``calculate_qparams`` computes.** Every min/max-family subclass funnels its
    observed range through the shared affine / symmetric calibration in
    :func:`~lucid.quantization.calculate_qparams`. The range is first widened to include
    real zero (so ``0.0`` is always representable), then mapped onto the integer grid
    ``[q_min, q_max]`` of :attr:`qdtype`:

    .. math::

        \text{affine:}\quad
        s = \frac{\max - \min}{q_{\max} - q_{\min}},\qquad
        z = \operatorname{clip}\!\bigl(q_{\min} - \operatorname{round}(\min/s),\,
            q_{\min},\, q_{\max}\bigr)

    .. math::

        \text{symmetric:}\quad
        s = \frac{\max(|\min|,\,|\max|)}{(q_{\max} - q_{\min})/2},\qquad
        z = \begin{cases} 0 & \text{signed} \\
             \lfloor (q_{\max}+q_{\min}+1)/2 \rfloor & \text{unsigned} \end{cases}

    with a floor :math:`s \leftarrow \max(s, \varepsilon)` guarding a degenerate
    (constant-input) range. Affine spends a free ``zero_point`` to pack an asymmetric
    range tightly (ideal for post-ReLU activations); symmetric pins ``zero_point`` so
    real 0 lands exactly on a code (the standard choice for signed weights).

    **Choosing a subclass.** Pick along three axes. *Granularity* — per-tensor
    (:class:`MinMaxObserver`, one pair for the whole tensor) vs per-channel
    (:class:`PerChannelMinMaxObserver`, one pair per output channel, far tighter for
    conv / linear weights whose channels differ in range). *Outlier robustness* — a hard
    running min/max grabs every spike, an EMA variant
    (:class:`MovingAverageMinMaxObserver`) smooths per-batch outliers, and
    :class:`HistogramObserver` clips the tails to minimise L2 error on heavy-tailed
    activations. *Fixed-range* — ops whose output range is known a priori use
    :class:`FixedQParamsObserver`, and dtype-only carriers use
    :class:`PlaceholderObserver` / :class:`NoopObserver`.

    Parameters
    ----------
    qscheme : QScheme
        Target quantization scheme — one of ``per_tensor_affine`` /
        ``per_tensor_symmetric`` / ``per_channel_affine`` / ``per_channel_symmetric``.
        Selects the affine-vs-symmetric branch above and whether qparams are scalar or
        per-channel vectors.
    qdtype : QDtype
        Target quantized dtype supplying the grid bounds ``quant_min`` / ``quant_max``
        (e.g. ``quint8`` → ``[0, 255]``, ``qint8`` → ``[-128, 127]``) and the on-device
        storage dtype that physically holds the integer codes.
    ch_axis : int, optional
        Channel axis quantized independently. Forced to ``None`` for per-tensor schemes
        regardless of the value passed.
    eps : float, default 1e-8
        Lower floor on the derived ``scale``, forwarded to
        :func:`~lucid.quantization.calculate_qparams` to avoid division by zero.

    Attributes
    ----------
    qscheme : QScheme
        The stored target scheme.
    qdtype : QDtype
        The stored target dtype.
    ch_axis : int or None
        The effective channel axis (``None`` unless the scheme is per-channel).
    eps : float
        The stored scale floor.

    Notes
    -----
    - **Identity forward.** ``forward`` returns its input unchanged; only the side effect
      of updating the running buffers matters. This is what lets an observer be spliced
      into a prepared model transparently.
    - **Statistics carry no graph.** A statistics-gathering ``forward`` is decorated
      with ``_observes``, which runs it under :func:`lucid.no_grad` on buffers aligned to
      ``x``'s device, and it writes each statistic through
      ``_store_running_stat``: in place when shape, dtype and device agree, re-registered
      otherwise (the scalar-seed → per-channel-vector change on the first observation).
      Gradients reach ``x`` only through whatever consumes the returned tensor — a
      :class:`FakeQuantize` straight-through estimator — never through the buffers, so a
      training loop does not keep one step's graph alive per iteration.
    - **Not instantiated directly.** ``ObserverBase`` is abstract; its
      :meth:`calculate_qparams` raises :class:`NotImplementedError`. Use a concrete
      subclass, or :meth:`with_args` to defer construction into a
      :class:`~lucid.quantization.QConfig`.
    - **H4 purity.** Every statistic is computed on the public ``lucid.*`` op surface —
      no external numeric libraries — so the same code path is correct on both the
      Accelerate (CPU) and MLX (GPU) streams.

    Examples
    --------
    >>> import lucid
    >>> import lucid.quantization as Q
    >>> obs = Q.MinMaxObserver()            # a concrete ObserverBase subclass
    >>> _ = obs(lucid.randn(64, 128))       # forward observes and returns x unchanged
    >>> scale, zero_point = obs.calculate_qparams()
    >>> scale.shape, zero_point.shape
    ((), ())

    Deferring construction into a :class:`~lucid.quantization.QConfig` recipe:

    >>> factory = Q.PerChannelMinMaxObserver.with_args(ch_axis=0)
    >>> obs2 = factory()                    # zero-arg factory builds a fresh observer
    >>> type(obs2).__name__
    'PerChannelMinMaxObserver'

    See Also
    --------
    lucid.quantization.MinMaxObserver : Per-tensor running min/max (the common default).
    lucid.quantization.PerChannelMinMaxObserver : Per-channel weight observer.
    lucid.quantization.HistogramObserver : Tail-clipping observer for activations.
    lucid.quantization.calculate_qparams : The shared range → ``(scale, zero_point)`` map.
    lucid.quantization.QConfig : Pairs activation + weight observer factories.
    lucid.quantization.FakeQuantize : Wraps an observer for quantization-aware training.
    """

    def __init__(
        self, qscheme: QScheme, qdtype: QDtype, ch_axis: int | None = None, eps: float = 1e-8
    ) -> None:
        super().__init__()
        self.qscheme = qscheme
        self.qdtype = qdtype
        self.ch_axis = ch_axis if qscheme.is_per_channel else None
        self.eps = eps

    def _align_running_buffers(self, ref: Tensor) -> None:
        """Adopt the observed tensor's device for every running buffer.

        Observers are seeded device-agnostically (``+inf`` / ``-inf`` on CPU) at
        construction because the calibration device is unknown then.  A model run
        on Metal feeds GPU activations, and a per-channel *weight* observer's seed
        does not ride along on ``module.to(device)`` (the weight fake-quant is
        reached through a path ``.to`` does not traverse), so the first
        ``minimum(seed_cpu, x_gpu)`` reduction ``DeviceMismatch``-es.  Re-register
        each buffer on ``ref``'s device on first sight; a no-op once aligned.
        """
        with untraced():
            ch = ref.shape[self.ch_axis] if self.ch_axis is not None and ref.ndim > 0 else None
            for name, buf in list(self.named_buffers(recurse=False)):
                want_shape = (ch,) if ch is not None and buf.ndim == 0 else buf.shape
                if buf.device != ref.device or buf.shape != want_shape:
                    aligned = buf.to(ref.device)
                    if aligned.shape != want_shape:
                        aligned = aligned.expand(want_shape).contiguous()
                    self.register_buffer(name, aligned)

    @override
    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Adopt the saved buffers' shapes before the default loader checks them.

        An observer's running buffers change shape as it calibrates: a
        per-channel observer is seeded with scalar ``+inf`` / ``-inf``
        because the channel count is unknown at construction, and grows to
        ``(C,)`` on the first batch it sees.  A freshly constructed
        observer is therefore scalar, and loading a calibrated checkpoint
        into it failed on a size mismatch — which meant a quantized model's
        calibration could be saved and never restored.

        Re-registering is how these buffers change shape or device
        everywhere else in this file (see :meth:`_align_running_buffers` and
        ``_store_running_stat``), so the load path uses it too rather than
        inventing a second mechanism.
        """
        for name, current in list(self.named_buffers(recurse=False)):
            saved = state_dict.get(prefix + name)
            if saved is None:
                continue
            if tuple(saved.shape) != tuple(current.shape):
                self.register_buffer(name, saved.to(current.device))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def calculate_qparams(self) -> tuple[Tensor, Tensor]:
        """Return ``(scale, zero_point)`` from the accumulated statistics."""
        raise NotImplementedError

    @classmethod
    def with_args(cls, **kwargs: object) -> functools.partial[ObserverBase]:
        """Return a zero-arg factory that builds this observer with ``kwargs``.

        Mirrors the reference framework's ``Observer.with_args`` ergonomic so
        a :class:`~lucid.quantization.QConfig` can carry an observer *recipe*
        rather than a live instance.
        """
        return functools.partial(cls, **kwargs)  # type: ignore[arg-type]


class MinMaxObserver(ObserverBase):
    r"""Per-tensor observer tracking the global running min / max of the input.

    The simplest and most common calibration statistic: on every batch it folds the
    tensor's overall ``min`` / ``max`` into a *cumulative* running range, and
    :meth:`calculate_qparams` distills that single ``[min_val, max_val]`` into one
    per-tensor ``(scale, zero_point)`` pair. It is the natural default for activations
    and a serviceable choice for weights, but because the running range only ever grows,
    a single spike in any calibration batch permanently widens it — wasting codes on a
    range that most values never occupy.

    **When to pick it.** Reach for ``MinMaxObserver`` when the tensor's distribution is
    well-behaved (roughly bounded, light-tailed) and you want the cheapest possible
    calibration. If a few calibration batches carry outliers, prefer
    :class:`MovingAverageMinMaxObserver`, whose EMA lets the range *relax* back toward
    the bulk of the data; for heavy-tailed activations where clipping the tails buys real
    accuracy, prefer :class:`HistogramObserver`. For weights, the per-channel
    :class:`PerChannelMinMaxObserver` is almost always tighter.

    Per batch the running range is the cumulative min/max:

    .. math::

        \text{min\_val} \leftarrow \min(\text{min\_val},\ \min_i x_i),\qquad
        \text{max\_val} \leftarrow \max(\text{max\_val},\ \max_i x_i)

    seeded from :math:`(+\infty, -\infty)`, then reduced to qparams by the affine map of
    :func:`~lucid.quantization.calculate_qparams` (range first widened to include 0):

    .. math::

        s = \frac{\text{max\_val} - \text{min\_val}}{q_{\max} - q_{\min}},\qquad
        z = \operatorname{clip}\!\bigl(q_{\min} -
            \operatorname{round}(\text{min\_val}/s),\ q_{\min},\ q_{\max}\bigr)

    (or the symmetric form of the base class if a symmetric ``qscheme`` is passed).

    Parameters
    ----------
    qscheme : QScheme, default ``per_tensor_affine``
        Target quantization scheme. Affine spends a free ``zero_point`` to fit an
        asymmetric range tightly (e.g. post-ReLU activations).
    qdtype : QDtype, default ``quint8``
        Target quantized dtype; ``quint8`` gives the unsigned grid ``[0, 255]``.
    eps : float, default 1e-8
        Lower floor on the derived ``scale`` — avoids a zero scale on a constant input
        (``min_val == max_val``).

    Attributes
    ----------
    min_val : Tensor
        Scalar (shape ``()``) running minimum, seeded ``+inf``.
    max_val : Tensor
        Scalar (shape ``()``) running maximum, seeded ``-inf``.

    Notes
    -----
    - **Cumulative, not per-batch.** ``min_val`` / ``max_val`` are monotone: they only
      widen across calibration, so the final range reflects the extreme of *all* data
      seen, not any single batch. This is the source of the outlier sensitivity.
    - **No autograd history.** The range is folded in under :func:`lucid.no_grad` and
      written in place, per the :class:`ObserverBase` contract.
    - **Symmetric vs affine.** With a symmetric ``qscheme`` the base-class math pins
      ``zero_point`` (0 for a signed dtype, mid-range for unsigned); with an affine
      scheme ``zero_point`` is free.
    - **Uncalibrated.** Before the first ``forward`` the range is ``[+inf, -inf]``;
      calling :meth:`calculate_qparams` on it yields a meaningless (infinite) scale —
      always calibrate first.
    - Not the default in :func:`~lucid.quantization.get_default_qconfig` (which uses
      :class:`HistogramObserver` for activations); it is the plain building block the EMA
      and per-channel variants specialise.

    Examples
    --------
    >>> import lucid
    >>> import lucid.quantization as Q
    >>> obs = Q.MinMaxObserver()
    >>> for _ in range(4):                       # feed several calibration batches
    ...     _ = obs(lucid.randn(32, 256))
    >>> scale, zero_point = obs.calculate_qparams()
    >>> bool(scale.item() > 0)
    True

    An *uncalibrated* observer still holds the ``[+inf, -inf]`` seed, so its range is
    degenerate until it has seen data:

    >>> fresh = Q.MinMaxObserver()
    >>> import math
    >>> math.isinf(fresh.min_val.item()), math.isinf(fresh.max_val.item())
    (True, True)

    See Also
    --------
    lucid.quantization.MovingAverageMinMaxObserver : EMA variant, robust to outliers.
    lucid.quantization.PerChannelMinMaxObserver : Per-channel granularity for weights.
    lucid.quantization.HistogramObserver : Tail-clipping for heavy-tailed activations.
    lucid.quantization.calculate_qparams : The range → ``(scale, zero_point)`` reduction.
    lucid.quantization.FakeQuantize : Wraps an observer for QAT.
    """

    min_val: Tensor
    max_val: Tensor

    def __init__(
        self, qscheme: QScheme = per_tensor_affine, qdtype: QDtype = quint8, eps: float = 1e-8
    ) -> None:
        super().__init__(qscheme, qdtype, None, eps)
        self.register_buffer("min_val", lucid.tensor(float("inf")))
        self.register_buffer("max_val", lucid.tensor(float("-inf")))

    @override
    @_observes
    def forward(self, x: Tensor) -> Tensor:
        """Fold ``x``'s global min/max into the running statistics."""
        _store_running_stat(self, "min_val", lucid.minimum(self.min_val, x.min()))
        _store_running_stat(self, "max_val", lucid.maximum(self.max_val, x.max()))
        return x

    @override
    def calculate_qparams(self) -> tuple[Tensor, Tensor]:
        return calculate_qparams(self.min_val, self.max_val, self.qscheme, self.qdtype, self.eps)


class MovingAverageMinMaxObserver(MinMaxObserver):
    r"""Per-tensor observer with an exponential moving average of min / max.

    An outlier-tolerant refinement of :class:`MinMaxObserver`. Instead of taking a hard
    cumulative min/max, it tracks an *exponential moving average* of each batch's range,
    so a single anomalous batch nudges the running range only slightly and is soon
    forgotten as later batches pull it back toward the bulk of the data. The result is a
    tighter, more representative grid than a plain min/max on noisy activation streams —
    at the cost of one hyperparameter, the averaging constant.

    **When to pick it.** Prefer this over :class:`MinMaxObserver` whenever calibration
    data may contain occasional spikes (most real activation streams do) and you want a
    range that reflects the *typical* extreme rather than the absolute worst case. It is
    the activation observer wrapped by :class:`FakeQuantize` in
    :func:`~lucid.quantization.get_default_qat_qconfig`, where smoothing also stabilises
    training. For heavy-tailed distributions where you would rather *clip* the tail than
    average it, :class:`HistogramObserver` is stronger; for weights, use
    :class:`MovingAveragePerChannelMinMaxObserver`.

    The first batch *seeds* the range directly (the ``±inf`` seed cannot be averaged);
    every later batch applies the EMA update with averaging constant :math:`c`:

    .. math::

        \text{min\_val} \leftarrow (1-c)\,\text{min\_val} + c\,(\min_i x_i)
            = \text{min\_val} + c\,(\min_i x_i - \text{min\_val}),

    .. math::

        \text{max\_val} \leftarrow (1-c)\,\text{max\_val} + c\,(\max_i x_i)
            = \text{max\_val} + c\,(\max_i x_i - \text{max\_val}).

    Smaller :math:`c` means heavier smoothing (slower to react, more outlier-robust);
    :math:`c = 1` degenerates to using only the latest batch. The reduction to
    ``(scale, zero_point)`` is the same affine / symmetric map inherited from
    :class:`MinMaxObserver`.

    Parameters
    ----------
    qscheme : QScheme, default ``per_tensor_affine``
        Target quantization scheme.
    qdtype : QDtype, default ``quint8``
        Target quantized dtype (unsigned ``[0, 255]`` grid).
    averaging_constant : float, default 0.01
        EMA weight :math:`c \in (0, 1]` of each new batch's range; smaller is smoother
        and more outlier-robust, larger reacts faster.
    eps : float, default 1e-8
        Lower floor on the derived ``scale``.

    Attributes
    ----------
    min_val : Tensor
        Scalar (shape ``()``) EMA of the batch minimum, seeded ``+inf``.
    max_val : Tensor
        Scalar (shape ``()``) EMA of the batch maximum, seeded ``-inf``.
    averaging_constant : float
        The stored EMA weight :math:`c`.

    Notes
    -----
    - **First-batch seeding.** ``forward`` detects the ``+inf`` seed and copies the first
      batch's range in verbatim; the EMA recurrence applies only from the second batch
      on. Without this the average would be corrupted by the infinite seed.
    - **Not cumulative.** Unlike :class:`MinMaxObserver`, the range can *shrink* between
      batches — this is exactly why a lone outlier does not permanently widen the grid.
    - **No autograd history.** The EMA update runs under :func:`lucid.no_grad` and is
      written back in place, per the :class:`ObserverBase` contract.
    - **Symmetric vs affine** behaves as in the base class; ``averaging_constant`` only
      affects the running range, not the range → qparams reduction.
    - Default activation observer of :func:`~lucid.quantization.get_default_qat_qconfig`
      (wrapped in :class:`FakeQuantize`).

    Examples
    --------
    >>> import lucid
    >>> import lucid.quantization as Q
    >>> obs = Q.MovingAverageMinMaxObserver(averaging_constant=0.1)
    >>> for _ in range(8):
    ...     _ = obs(lucid.randn(16, 128))
    >>> scale, zero_point = obs.calculate_qparams()
    >>> bool(scale.item() > 0)
    True

    A lone outlier batch barely moves the smoothed range (contrast the hard-min/max
    observer, which would absorb the full spike):

    >>> obs = Q.MovingAverageMinMaxObserver(averaging_constant=0.01)
    >>> _ = obs(lucid.zeros(1, 4))               # seed range ~ [0, 0]
    >>> _ = obs(lucid.full((1, 4), 1000.0))      # one huge batch
    >>> bool(obs.max_val.item() < 20.0)          # EMA absorbed only ~1% of the spike
    True

    See Also
    --------
    lucid.quantization.MinMaxObserver : Hard cumulative min/max (no smoothing).
    lucid.quantization.MovingAveragePerChannelMinMaxObserver : Per-channel EMA analogue.
    lucid.quantization.HistogramObserver : Tail-clipping alternative for outliers.
    lucid.quantization.get_default_qat_qconfig : Uses this observer for activations.
    lucid.quantization.FakeQuantize : Wraps it for quantization-aware training.
    """

    def __init__(
        self,
        qscheme: QScheme = per_tensor_affine,
        qdtype: QDtype = quint8,
        averaging_constant: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(qscheme, qdtype, eps)
        self.averaging_constant = averaging_constant

    @override
    @_observes
    def forward(self, x: Tensor) -> Tensor:
        """EMA-update the running min/max (seeded on the first batch)."""
        c = self.averaging_constant
        cur_min, cur_max = x.min(), x.max()
        # On the first batch (min == +inf) seed directly; else EMA toward cur.
        first = lucid.isinf(self.min_val)
        new_min = lucid.where(first, cur_min, self.min_val + c * (cur_min - self.min_val))
        new_max = lucid.where(first, cur_max, self.max_val + c * (cur_max - self.max_val))
        _store_running_stat(self, "min_val", new_min)
        _store_running_stat(self, "max_val", new_max)
        return x


class PerChannelMinMaxObserver(ObserverBase):
    r"""Per-channel observer tracking a running min / max along ``ch_axis``.

    The standard **weight** observer. Rather than one range for the whole tensor, it
    keeps an independent ``[min, max]`` for every slice along ``ch_axis`` — reducing over
    all *other* axes — so :meth:`calculate_qparams` returns a length-``C`` *vector* of
    ``(scale, zero_point)``, one per output channel. Because different output neurons /
    filters routinely span wildly different dynamic ranges, a single per-tensor scale is
    dominated by the widest channel and quantizes the rest coarsely; a per-channel scale
    tracks each one tightly, which is why per-channel symmetric ``qint8`` is the standard
    accuracy-preserving choice for convolution and linear weights.

    **When to pick it.** Use it for **weights** — it is the weight observer in both
    :func:`~lucid.quantization.get_default_qconfig` (static PTQ) and
    :func:`~lucid.quantization.get_default_qat_qconfig` (QAT, wrapped in
    :class:`FakeQuantize`). It is *not* used for activations, whose channel layout varies
    per op and which are cheaper to quantize per-tensor. For a smoothed per-channel range
    over noisy weight calibration, use :class:`MovingAveragePerChannelMinMaxObserver`.

    Writing the input with ``ch_axis`` moved to the front and the rest flattened,
    :math:`x^{(c)}` denotes the values of channel :math:`c`; each channel keeps its own
    cumulative range:

    .. math::

        \text{min\_val}_c \leftarrow \min\!\bigl(\text{min\_val}_c,\ \min x^{(c)}\bigr),
        \qquad
        \text{max\_val}_c \leftarrow \max\!\bigl(\text{max\_val}_c,\ \max x^{(c)}\bigr)

    then the (default symmetric, signed) reduction pins ``zero_point`` to 0 and scales
    each channel by its own peak magnitude:

    .. math::

        s_c = \frac{\max(|\text{min\_val}_c|,\ |\text{max\_val}_c|)}
                   {(q_{\max} - q_{\min})/2},\qquad z_c = 0

    (an affine ``qscheme`` instead yields per-channel ``(s_c, z_c)`` via the base-class
    affine map).

    Parameters
    ----------
    ch_axis : int, default 0
        Axis whose entries are quantized independently — the output-channel axis, which
        is 0 for both conv weights ``(out, in, kh, kw)`` and linear weights ``(out, in)``.
    qscheme : QScheme, default ``per_channel_symmetric``
        Target quantization scheme; the per-channel default pins ``zero_point`` per
        channel.
    qdtype : QDtype, default ``qint8``
        Target quantized dtype; ``qint8`` gives the signed grid ``[-128, 127]``.
    eps : float, default 1e-8
        Lower floor on each derived per-channel ``scale``.

    Attributes
    ----------
    min_val : Tensor
        Per-channel running minimum, shape ``(C,)`` after the first batch (scalar
        ``+inf`` seed before it).
    max_val : Tensor
        Per-channel running maximum, shape ``(C,)`` after the first batch (scalar
        ``-inf`` seed before it).

    Notes
    -----
    - **Scalar seed → per-channel vector.** The buffers start as scalar ``±inf``; the
      first ``forward`` broadcasts that seed against the length-``C`` batch range, so
      ``min_val`` / ``max_val`` acquire shape ``(C,)`` on the first observation; that one
      write re-registers the buffers, and every later one lands in place.
    - **Cumulative** per channel — like :class:`MinMaxObserver`, the ranges only widen,
      so the observer is outlier-sensitive; smooth with the moving-average variant.
    - **Symmetric vs affine.** The default per-channel-symmetric scheme fixes
      ``zero_point = 0`` (signed weights map 0 → 0 exactly); an affine scheme gives each
      channel a free zero-point.
    - **Reduction axes.** Every axis except ``ch_axis`` is reduced, so the number of
      qparams equals ``x.shape[ch_axis]``.
    - Weight observer of both default QConfigs
      (:func:`~lucid.quantization.get_default_qconfig` and
      :func:`~lucid.quantization.get_default_qat_qconfig`).

    Examples
    --------
    >>> import lucid
    >>> import lucid.quantization as Q
    >>> obs = Q.PerChannelMinMaxObserver(ch_axis=0)
    >>> _ = obs(lucid.randn(16, 8))              # 16 output channels
    >>> scale, zero_point = obs.calculate_qparams()
    >>> scale.shape                              # one scale per output channel
    (16,)
    >>> bool((zero_point == 0).all().item())     # symmetric ⇒ zero_point pinned to 0
    True

    Per-channel is far tighter than per-tensor when channels differ in range — here one
    row is ~1000x larger, yet each row keeps its own scale:

    >>> import lucid
    >>> w = lucid.stack([lucid.ones(4) * 0.001, lucid.ones(4) * 1.0])
    >>> obs = Q.PerChannelMinMaxObserver(ch_axis=0)
    >>> _ = obs(w)
    >>> s, _ = obs.calculate_qparams()
    >>> bool(s[0].item() < s[1].item())          # small-range row ⇒ smaller step
    True

    See Also
    --------
    lucid.quantization.MinMaxObserver : Per-tensor counterpart.
    lucid.quantization.MovingAveragePerChannelMinMaxObserver : Per-channel EMA variant.
    lucid.quantization.get_default_qconfig : Uses this as the weight observer.
    lucid.nn.quantized.Linear : Consumes per-channel weight qparams at convert time.
    lucid.quantization.calculate_qparams : The per-channel range → qparams reduction.
    """

    min_val: Tensor
    max_val: Tensor

    def __init__(
        self,
        ch_axis: int = 0,
        qscheme: QScheme = per_channel_symmetric,
        qdtype: QDtype = qint8,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(qscheme, qdtype, ch_axis, eps)
        self.register_buffer("min_val", lucid.tensor(float("inf")))
        self.register_buffer("max_val", lucid.tensor(float("-inf")))

    @override
    @_observes
    def forward(self, x: Tensor) -> Tensor:
        """Fold per-channel min/max (reduced over all non-channel axes)."""
        axis = self.ch_axis if self.ch_axis is not None else 0
        perm = lucid.moveaxis(x, axis, 0)
        num_channels = perm.shape[0]
        flat = perm.reshape(num_channels, -1)
        cur_min = lucid.amin(flat, dim=1)
        cur_max = lucid.amax(flat, dim=1)
        # A scalar +inf/-inf seed broadcasts to per-channel on the first call.
        _store_running_stat(self, "min_val", lucid.minimum(self.min_val, cur_min))
        _store_running_stat(self, "max_val", lucid.maximum(self.max_val, cur_max))
        return x

    @override
    def calculate_qparams(self) -> tuple[Tensor, Tensor]:
        return calculate_qparams(self.min_val, self.max_val, self.qscheme, self.qdtype, self.eps)


class HistogramObserver(ObserverBase):
    r"""Per-tensor observer that records a value histogram and picks a clip range.

    The accuracy-maximising activation observer. Instead of committing to the raw
    ``[min, max]`` — which on a heavy-tailed activation would stretch the grid over a
    handful of rare extremes and quantize the dense bulk coarsely — it accumulates a
    ``bins``-bucket histogram of every value seen (via the engine ``histc`` primitive),
    then at :meth:`calculate_qparams` time *searches* for the clip range
    ``[new_min, new_max]`` that minimises the expected L2 quantization error. Clipping
    the tails costs a little error on the rare outliers but wins far finer resolution on
    the common values, which is usually a large net accuracy gain.

    **When to pick it.** Use it for **activations** with skewed / heavy-tailed
    distributions — it is the activation observer in
    :func:`~lucid.quantization.get_default_qconfig` (static PTQ). It is more expensive
    than min/max (it maintains a histogram and runs a search) and applies to per-tensor
    affine schemes, so it is not used for per-channel weights (see
    :class:`PerChannelMinMaxObserver`). When calibration is cheap and distributions are
    light-tailed, plain :class:`MinMaxObserver` / :class:`MovingAverageMinMaxObserver`
    suffice.

    The running range widens with the data; when it grows the existing histogram is
    *rebinned* onto the wider support before the new batch is added. With bin centers
    :math:`c_i` and counts :math:`n_i`, the search evaluates candidate clip endpoints
    :math:`[a, b]` (drawn from a few tail quantiles) and picks the pair minimising the
    per-value clip error plus the uniform-quantization noise variance:

    .. math::

        \operatorname{err}(a, b) = \sum_i n_i
            \Bigl[\bigl(c_i - \operatorname{clip}(c_i, a, b)\bigr)^2
                  + \tfrac{\Delta^2}{12}\Bigr],
        \qquad \Delta = \frac{b - a}{q_{\max} - q_{\min}},

    where the first term is the squared clipping error (mass pushed onto the boundary)
    and :math:`\Delta^2/12` is the variance of rounding to a uniform grid of step
    :math:`\Delta`. The winning :math:`[a, b]` is then fed to
    :func:`~lucid.quantization.calculate_qparams` for the final affine ``(scale,
    zero_point)``.

    Parameters
    ----------
    bins : int, default 2048
        Number of histogram buckets over the running ``[min, max]``. More bins give a
        finer clip search at higher memory / compute cost.
    qscheme : QScheme, default ``per_tensor_affine``
        Target quantization scheme (the clip search assumes a per-tensor grid).
    qdtype : QDtype, default ``quint8``
        Target quantized dtype; ``quint8`` gives the unsigned ``[0, 255]`` grid.
    eps : float, default 1e-8
        Lower floor on the derived ``scale``.

    Attributes
    ----------
    min_val : Tensor
        Scalar (shape ``()``) running minimum of the histogram support, seeded ``+inf``.
    max_val : Tensor
        Scalar (shape ``()``) running maximum of the histogram support, seeded ``-inf``.
    histogram : Tensor
        The accumulated value histogram, shape ``(bins,)``, seeded to zeros.

    Notes
    -----
    - **Rebinning on growth.** When a batch widens ``[min, max]``, the stored histogram
      is redistributed onto the new, wider bin grid before the fresh counts are added, so
      accumulated mass is preserved rather than discarded.
    - **Search is a coarse scan.** The clip endpoints are candidate tail quantiles
      (``0, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2``), not a full exhaustive sweep — a deliberate
      speed / accuracy trade-off. The best-scoring pair wins.
    - **Data-dependent control flow.** ``forward`` reads ``x.min()/.max()`` via
      ``.item()`` and the search runs in Python over ``histogram.tolist()``; this is a
      sanctioned CPU round-trip (data-dependent output), still H4-pure (engine ``histc``,
      no external histogram library).
    - **No autograd history.** ``min_val`` / ``max_val`` / ``histogram`` are all
      updated under :func:`lucid.no_grad` and written in place, per the
      :class:`ObserverBase` contract.
    - **Symmetric vs affine.** The reduction honours ``qscheme``; the default per-tensor
      affine leaves ``zero_point`` free to fit the (clipped) asymmetric range.
    - Default activation observer of :func:`~lucid.quantization.get_default_qconfig`.

    Examples
    --------
    >>> import lucid
    >>> import lucid.quantization as Q
    >>> obs = Q.HistogramObserver(bins=512)
    >>> for _ in range(4):
    ...     _ = obs(lucid.randn(8, 1024))
    >>> obs.histogram.shape
    (512,)
    >>> scale, zero_point = obs.calculate_qparams()
    >>> bool(scale.item() > 0)
    True

    On a heavy-tailed input the search clips the rare extreme, so the chosen range is
    *narrower* than the raw min/max the histogram spans.  It does so only when the bulk
    is large enough that the finer grid outweighs the outlier's clipping error — at
    ``1e5`` values, keeping the outlier is the lower-error choice; at ``1e6`` it is not:

    >>> import lucid
    >>> _ = lucid.manual_seed(0)
    >>> x = lucid.randn(1, 1_000_000)
    >>> x[0, 0] = 50.0                           # a single far outlier
    >>> obs = Q.HistogramObserver(bins=2048)
    >>> _ = obs(x)
    >>> s, _ = obs.calculate_qparams()
    >>> bool(s.item() * 255 < 50.0)              # grid span clipped below the outlier
    True

    See Also
    --------
    lucid.quantization.MinMaxObserver : Cheaper per-tensor observer (no clip search).
    lucid.quantization.MovingAverageMinMaxObserver : EMA alternative for outliers.
    lucid.quantization.get_default_qconfig : Uses this as the activation observer.
    lucid.quantization.calculate_qparams : Reduces the chosen clip range to qparams.
    lucid.quantization.FakeQuantize : Wraps an observer for QAT.
    """

    min_val: Tensor
    max_val: Tensor
    histogram: Tensor

    def __init__(
        self,
        bins: int = 2048,
        qscheme: QScheme = per_tensor_affine,
        qdtype: QDtype = quint8,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(qscheme, qdtype, None, eps)
        self.bins = bins
        self.register_buffer("min_val", lucid.tensor(float("inf")))
        self.register_buffer("max_val", lucid.tensor(float("-inf")))
        self.register_buffer("histogram", lucid.zeros(bins))

    def _rebin(
        self, hist: Tensor, old_min: float, old_max: float, new_min: float, new_max: float
    ) -> Tensor:
        """Redistribute an existing histogram onto a wider ``[new_min,new_max]``."""
        bins = self.bins
        old_width = (old_max - old_min) / bins
        new_width = (new_max - new_min) / bins
        # Built where ``hist`` lives, not on the default device.
        #
        # These were plain ``arange`` / ``zeros``, so they landed wherever
        # the default device pointed and the scatter mixed devices.  It
        # never showed because ``histc`` returned a CPU tensor whatever it
        # was given — so on Metal every operand here was CPU and agreed
        # by accident.  Once ``histc`` answered on the input's device this
        # raised ``bad_variant_access``, which is the latent bug becoming
        # visible rather than a new one.
        device = hist.device
        centers = old_min + (lucid.arange(bins, device=device).to(lucid.float32) + 0.5) * old_width
        # Clip in the float domain (CPU has no integer clamp kernel) then cast.
        raw = lucid.clip((centers - new_min) / new_width, 0.0, float(bins - 1))
        dst = raw.to(lucid.int64)
        out = lucid.zeros(bins, device=device)
        return lucid.scatter_add(out, 0, dst, hist)

    @override
    @_observes
    def forward(self, x: Tensor) -> Tensor:
        """Update the running range and accumulate the histogram."""
        cur_min, cur_max = float(x.min().item()), float(x.max().item())
        old_min, old_max = float(self.min_val.item()), float(self.max_val.item())
        if old_min == float("inf"):
            new_min, new_max = cur_min, cur_max
            hist = lucid.histc(x, bins=self.bins, min=new_min, max=new_max)
        else:
            new_min, new_max = min(old_min, cur_min), max(old_max, cur_max)
            hist = lucid.histc(x, bins=self.bins, min=new_min, max=new_max)
            if new_min == old_min and new_max == old_max:
                hist = hist + self.histogram
            else:
                hist = hist + self._rebin(self.histogram, old_min, old_max, new_min, new_max)
        device = x.device
        _store_running_stat(self, "min_val", lucid.tensor(new_min, device=device))
        _store_running_stat(self, "max_val", lucid.tensor(new_max, device=device))
        _store_running_stat(self, "histogram", hist)
        return x

    def _search_clip_range(self) -> tuple[float, float]:
        """Scan candidate clip ranges; return the L2-error-minimising one."""
        lo, hi = float(self.min_val.item()), float(self.max_val.item())
        # ``histogram`` is a 1-D float buffer, so ``tolist`` yields ``list[float]``.
        counts: list[float] = cast(list[float], self.histogram.tolist())
        total = sum(counts)
        if total <= 0.0 or hi <= lo:
            return lo, hi
        bins = self.bins
        width = (hi - lo) / bins
        centers = [lo + (i + 0.5) * width for i in range(bins)]
        levels = self.qdtype.quant_max - self.qdtype.quant_min
        best_err, best = float("inf"), (lo, hi)
        # Coarse symmetric-ish scan: clip the tails at a handful of quantiles.
        for frac in (0.0, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2):
            cmin = _quantile_from_hist(counts, centers, frac)
            cmax = _quantile_from_hist(counts, centers, 1.0 - frac)
            if cmax <= cmin:
                continue
            step = (cmax - cmin) / max(levels, 1)
            err = 0.0
            for c, n in zip(centers, counts, strict=True):
                if n == 0.0:
                    continue
                cc = cmin if c < cmin else (cmax if c > cmax else c)
                # clip error + rounding error (uniform-quantization variance).
                err += n * ((c - cc) ** 2 + (step * step) / 12.0)
            if err < best_err:
                best_err, best = err, (cmin, cmax)
        return best

    @override
    def calculate_qparams(self) -> tuple[Tensor, Tensor]:
        new_min, new_max = self._search_clip_range()
        return calculate_qparams(
            lucid.tensor(float(new_min)),
            lucid.tensor(float(new_max)),
            self.qscheme,
            self.qdtype,
            self.eps,
        )


def _quantile_from_hist(counts: list[float], centers: list[float], q: float) -> float:
    """Return the bin center at cumulative mass fraction ``q``."""
    total = sum(counts)
    if total <= 0:
        return centers[0]
    target = q * total
    acc = 0.0
    for c, n in zip(centers, counts, strict=True):
        acc += n
        if acc >= target:
            return c
    return centers[-1]


class MovingAveragePerChannelMinMaxObserver(PerChannelMinMaxObserver):
    r"""Per-channel observer with an EMA of the per-channel min / max.

    The per-channel analogue of :class:`MovingAverageMinMaxObserver`, and the
    outlier-tolerant sibling of :class:`PerChannelMinMaxObserver`. It keeps an
    independent range per channel along ``ch_axis`` (reducing over every other axis) but
    updates each channel's ``[min, max]`` with an exponential moving average instead of a
    hard cumulative min/max — so a spike in one calibration batch nudges that channel's
    range only slightly and is relaxed back by later batches. :meth:`calculate_qparams`
    still returns a length-``C`` vector of ``(scale, zero_point)``.

    **When to pick it.** Use it for **weights** when weight calibration is noisy (e.g.
    running statistics observed during QAT rather than on static weights) and you want
    per-channel tightness *and* per-batch robustness. For static weights the plain
    :class:`PerChannelMinMaxObserver` is usually enough; for activations, use the
    per-tensor :class:`MovingAverageMinMaxObserver` or :class:`HistogramObserver`.

    With :math:`x^{(c)}` the values of channel :math:`c` in the current batch, the first
    batch seeds each channel directly and every later batch applies the per-channel EMA:

    .. math::

        \text{min\_val}_c \leftarrow \text{min\_val}_c
            + c\,\bigl(\min x^{(c)} - \text{min\_val}_c\bigr),\qquad
        \text{max\_val}_c \leftarrow \text{max\_val}_c
            + c\,\bigl(\max x^{(c)} - \text{max\_val}_c\bigr),

    where :math:`c` is ``averaging_constant``. The reduction to per-channel qparams is
    the (default symmetric, signed) map inherited from
    :class:`PerChannelMinMaxObserver`, pinning each ``z_c = 0``.

    Parameters
    ----------
    ch_axis : int, default 0
        Output-channel axis quantized independently (0 for conv / linear weights).
    qscheme : QScheme, default ``per_channel_symmetric``
        Target quantization scheme.
    qdtype : QDtype, default ``qint8``
        Target quantized dtype (signed ``[-128, 127]`` grid).
    averaging_constant : float, default 0.01
        EMA weight :math:`c \in (0, 1]` of each new batch's per-channel range; smaller is
        smoother / slower / more outlier-robust.
    eps : float, default 1e-8
        Lower floor on each derived per-channel ``scale``.

    Attributes
    ----------
    min_val : Tensor
        Per-channel EMA minimum, shape ``(C,)`` after the first batch (scalar ``+inf``
        seed before it).
    max_val : Tensor
        Per-channel EMA maximum, shape ``(C,)`` after the first batch (scalar ``-inf``
        seed before it).
    averaging_constant : float
        The stored EMA weight :math:`c`.

    Notes
    -----
    - **First-batch seeding is explicit.** Because the scalar ``±inf`` seed cannot be
      broadcast into an elementwise ``where`` against a length-``C`` batch range,
      ``forward`` special-cases the first observation and copies the per-channel range in
      directly; the EMA recurrence runs only from the second batch on.
    - **Not cumulative** — each channel's range can shrink between batches (the whole
      point of the EMA), unlike :class:`PerChannelMinMaxObserver`.
    - **No autograd history.** The scalar-seed → ``(C,)`` first write re-registers the
      buffers and every later EMA update lands in place under :func:`lucid.no_grad`, per
      the :class:`ObserverBase` contract.
    - **Symmetric vs affine** behaves as in the per-channel base class; the default pins
      each ``zero_point`` to 0.

    Examples
    --------
    >>> import lucid
    >>> import lucid.quantization as Q
    >>> obs = Q.MovingAveragePerChannelMinMaxObserver(ch_axis=0, averaging_constant=0.1)
    >>> for _ in range(8):
    ...     _ = obs(lucid.randn(12, 32))         # 12 output channels
    >>> scale, zero_point = obs.calculate_qparams()
    >>> scale.shape
    (12,)
    >>> bool((zero_point == 0).all().item())     # per-channel symmetric ⇒ z = 0
    True

    See Also
    --------
    lucid.quantization.PerChannelMinMaxObserver : Hard per-channel min/max (no smoothing).
    lucid.quantization.MovingAverageMinMaxObserver : Per-tensor EMA counterpart.
    lucid.quantization.get_default_qat_qconfig : QAT recipe using per-channel weights.
    lucid.quantization.calculate_qparams : The per-channel range → qparams reduction.
    lucid.quantization.FakeQuantize : Wraps an observer for QAT.
    """

    def __init__(
        self,
        ch_axis: int = 0,
        qscheme: QScheme = per_channel_symmetric,
        qdtype: QDtype = qint8,
        averaging_constant: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(ch_axis, qscheme, qdtype, eps)
        self.averaging_constant = averaging_constant

    @override
    @_observes
    def forward(self, x: Tensor) -> Tensor:
        """EMA-update the per-channel min/max (seeded on the first batch)."""
        c = self.averaging_constant
        axis = self.ch_axis if self.ch_axis is not None else 0
        perm = lucid.moveaxis(x, axis, 0)
        flat = perm.reshape(perm.shape[0], -1)
        cur_min, cur_max = lucid.amin(flat, dim=1), lucid.amax(flat, dim=1)
        # First batch: the ±inf seed is scalar, so seed the per-channel vectors
        # directly (a scalar `where` condition wouldn't broadcast to length C).
        if bool(lucid.isinf(self.min_val).all().item()):
            new_min, new_max = cur_min, cur_max
        else:
            new_min = self.min_val + c * (cur_min - self.min_val)
            new_max = self.max_val + c * (cur_max - self.max_val)
        _store_running_stat(self, "min_val", new_min)
        _store_running_stat(self, "max_val", new_max)
        return x


class FixedQParamsObserver(ObserverBase):
    r"""Observer with a fixed ``(scale, zero_point)`` — no statistics collected.

    For ops whose output range is *known a priori*, calibration is pointless: the grid
    can be fixed once and for all. The canonical cases are bounded activations —
    ``sigmoid`` → ``[0, 1]``, ``softmax`` → ``[0, 1]``, ``tanh`` → ``[-1, 1]`` — where
    the ideal ``(scale, zero_point)`` is a closed-form function of the dtype grid, not
    something the data can improve on. ``FixedQParamsObserver`` carries that fixed pair,
    makes ``forward`` a pure identity (no statistics gathered), and returns the supplied
    qparams verbatim from :meth:`calculate_qparams`.

    **When to pick it.** Use it on the output of a range-bounded op so calibration does
    not waste effort (or worse, under-shoot the true bound) estimating a range you
    already know. For a ``sigmoid``/``softmax`` output on the unsigned ``quint8`` grid
    ``[0, 255]`` the natural choice is ``scale = 1/256, zero_point = 0`` (mapping
    ``[0, 1]`` onto the full grid); a ``tanh`` output ``[-1, 1]`` uses ``scale = 2/256``
    with a mid-grid ``zero_point``. For every unbounded tensor, use a statistics-collecting
    observer (:class:`MinMaxObserver`, :class:`HistogramObserver`, …) instead.

    There is no statistic and no reduction — the qparams are the constants themselves:

    .. math::

        (\text{scale},\ \text{zero\_point}) = (s_{\text{fixed}},\ z_{\text{fixed}}),
        \qquad \hat{x} = \bigl(\operatorname{clip}(\operatorname{round}(x/s_{\text{fixed}})
            + z_{\text{fixed}},\ q_{\min},\ q_{\max}) - z_{\text{fixed}}\bigr)\,
            s_{\text{fixed}}

    independent of any observed data.

    Parameters
    ----------
    scale : float
        The fixed quantization step size :math:`s_{\text{fixed}}`.
    zero_point : int
        The fixed integer code that real value ``0`` maps to.
    qscheme : QScheme, default ``per_tensor_affine``
        Target quantization scheme — metadata only; the qparams are fixed regardless.
    qdtype : QDtype, default ``quint8``
        Target quantized dtype recorded for downstream consumers (grid bounds).
    eps : float, default 1e-8
        Retained for interface symmetry with :class:`ObserverBase`; unused because the
        ``scale`` is fixed (never derived, so never floored).

    Attributes
    ----------
    scale : Tensor
        Scalar (shape ``()``) fixed quantization step.
    zero_point : Tensor
        Scalar (shape ``()``) fixed zero-point (stored as an integer-valued float).

    Notes
    -----
    - **Identity forward.** ``forward`` returns ``x`` unchanged and collects nothing —
      the two buffers are set once at construction and never move.
    - **No calibration needed.** :meth:`calculate_qparams` never raises (unlike
      :class:`PlaceholderObserver`); it simply returns the stored ``(scale, zero_point)``,
      so it is valid to call immediately after construction with no forward passes.
    - **Buffers, not stats.** ``scale`` / ``zero_point`` are registered buffers, so they
      persist through ``state_dict`` and move with ``.to(device)`` like any other buffer.
    - **Metadata fields.** ``qscheme`` / ``eps`` are stored for interface symmetry but do
      not affect the returned qparams.

    Examples
    --------
    A fixed observer for a ``sigmoid`` output on the ``quint8`` grid — the ``[0, 1]``
    range maps onto ``[0, 255]`` with ``scale = 1/256``:

    >>> import lucid.quantization as Q
    >>> obs = Q.FixedQParamsObserver(scale=1.0 / 256, zero_point=0)
    >>> scale, zero_point = obs.calculate_qparams()   # valid with no calibration
    >>> round(scale.item(), 6), int(zero_point.item())
    (0.003906, 0)

    The forward is a pure identity — feeding data changes nothing:

    >>> import lucid
    >>> before = obs.scale.item()
    >>> _ = obs(lucid.randn(4, 4))
    >>> obs.scale.item() == before
    True

    See Also
    --------
    lucid.quantization.PlaceholderObserver : Dtype carrier whose qparams instead raise.
    lucid.quantization.MinMaxObserver : Statistics-collecting observer for unbounded ops.
    lucid.quantization.calculate_qparams : The range-based reduction this observer skips.
    lucid.quantization.FakeQuantize : Wraps an observer for QAT.
    """

    scale: Tensor
    zero_point: Tensor

    def __init__(
        self,
        scale: float,
        zero_point: int,
        qscheme: QScheme = per_tensor_affine,
        qdtype: QDtype = quint8,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(qscheme, qdtype, None, eps)
        self.register_buffer("scale", lucid.tensor(float(scale)))
        self.register_buffer("zero_point", lucid.tensor(float(zero_point)))

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Identity — the qparams are fixed, so no statistics are gathered."""
        return x

    @override
    def calculate_qparams(self) -> tuple[Tensor, Tensor]:
        return self.scale, self.zero_point


class PlaceholderObserver(ObserverBase):
    r"""Carries ``qdtype`` / ``qscheme`` metadata but collects no statistics.

    A pure metadata carrier for the cases where a range cannot — or should not — be
    calibrated ahead of time. The prototypical use is **dynamic quantization**, where an
    activation's ``(scale, zero_point)`` is computed *at runtime* from the actual input
    of each forward, so estimating a static range during a calibration pass would be
    meaningless. The observer still needs to advertise the *target* dtype and scheme to
    downstream consumers (the convert pass, the quantized module), which is exactly what
    it records — and nothing more. Its ``forward`` is the identity, and because there is
    no statistic to reduce, :meth:`calculate_qparams` deliberately **raises**.

    **When to pick it.** Use it wherever qparams are supplied elsewhere or produced
    later: the activation slot of a dynamic-quant recipe, or a tensor deliberately
    excluded from static calibration. If the range *is* fixed and known, use
    :class:`FixedQParamsObserver` (whose ``calculate_qparams`` returns the constants
    instead of raising); if the range should be *learned* from data, use a
    statistics-collecting observer such as :class:`MinMaxObserver` or
    :class:`HistogramObserver`.

    There is no statistic and no reduction — only the recorded metadata; asking for
    qparams is an error:

    .. math::

        \text{calculate\_qparams}() \;\Rightarrow\;
        \textbf{raise } \texttt{RuntimeError}

    Parameters
    ----------
    qscheme : QScheme, default ``per_tensor_affine``
        Target quantization scheme recorded for downstream consumers.
    qdtype : QDtype, default ``quint8``
        Target quantized dtype recorded for downstream consumers (grid + storage).
    eps : float, default 1e-8
        Retained for interface symmetry with :class:`ObserverBase`; unused because no
        range is ever derived.

    Attributes
    ----------
    qscheme : QScheme
        The recorded target scheme (metadata only).
    qdtype : QDtype
        The recorded target dtype (metadata only).

    Notes
    -----
    - **No buffers.** Unlike the min/max family, this observer registers no running
      statistic — there is nothing to persist beyond the metadata fields.
    - **Identity forward.** ``forward`` returns ``x`` unchanged and gathers nothing.
    - **``calculate_qparams`` raises.** It throws :class:`RuntimeError` by design — the
      qparams are produced at runtime (dynamic quantization) or supplied elsewhere, so
      there is no stored range to return. Callers must not treat this observer as a
      static-range source.
    - **Relation to** :class:`NoopObserver` — ``NoopObserver`` is a named subclass with
      identical behaviour, used to mark tensors that should pass through untouched.

    Examples
    --------
    >>> import lucid
    >>> import lucid.quantization as Q
    >>> obs = Q.PlaceholderObserver()
    >>> _ = obs(lucid.randn(4, 8))               # identity: observes nothing
    >>> try:
    ...     obs.calculate_qparams()              # no statistic ⇒ raises
    ... except RuntimeError as e:
    ...     print("raised")
    raised

    See Also
    --------
    lucid.quantization.NoopObserver : Named no-op subclass with the same behaviour.
    lucid.quantization.FixedQParamsObserver : Fixed qparams (returns, does not raise).
    lucid.quantization.MinMaxObserver : Statistics-collecting observer for static ranges.
    lucid.quantization.FakeQuantize : Wraps an observer for QAT.
    """

    def __init__(
        self, qscheme: QScheme = per_tensor_affine, qdtype: QDtype = quint8, eps: float = 1e-8
    ) -> None:
        super().__init__(qscheme, qdtype, None, eps)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Identity — no statistics are gathered."""
        return x

    @override
    def calculate_qparams(self) -> tuple[Tensor, Tensor]:
        raise RuntimeError(
            f"{type(self).__name__} collects no statistics; qparams are computed "
            "at runtime (dynamic quantization) or supplied elsewhere."
        )


class NoopObserver(PlaceholderObserver):
    r"""A no-op :class:`PlaceholderObserver` — pure identity, collects no statistics.

    The intent-revealing name for a :class:`PlaceholderObserver`: a tensor should flow
    through the observed graph **without any statistic gathering at all**. Whereas a bare
    ``PlaceholderObserver`` reads as "qparams come from elsewhere (dynamic quant)",
    ``NoopObserver`` reads as "this tensor is deliberately *not* quantized here" — an
    already-quantized input, a residual passed straight through, or a layer explicitly
    excluded from calibration. Behaviourally the two are identical: ``forward`` is the
    identity and :meth:`calculate_qparams` raises. Only the name (and the reader's
    intent) differs.

    **When to pick it.** Use it to annotate a passthrough tensor in a QConfig / mapping
    so the tooling records the "leave alone" intent explicitly rather than by omission.
    When qparams really are produced at runtime, prefer :class:`PlaceholderObserver`;
    when a fixed grid is known, use :class:`FixedQParamsObserver`.

    Like its parent it holds no statistic and its qparams query is an error:

    .. math::

        \text{forward}(x) = x,\qquad
        \text{calculate\_qparams}() \;\Rightarrow\;
        \textbf{raise } \texttt{RuntimeError}

    Parameters
    ----------
    qscheme : QScheme, default ``per_tensor_affine``
        Target quantization scheme recorded for downstream consumers (metadata only).
    qdtype : QDtype, default ``quint8``
        Target quantized dtype recorded for downstream consumers (metadata only).
    eps : float, default 1e-8
        Retained for interface symmetry; unused (no range is derived).

    Notes
    -----
    - **Pure alias.** Inherits the ``qscheme`` / ``qdtype`` / ``eps`` constructor and all
      behaviour of :class:`PlaceholderObserver` unchanged; it overrides nothing.
    - **Identity forward, raising qparams** — same contract as the parent; the class
      exists to make "no observation" a first-class, greppable choice.
    - **No buffers**, so nothing is added to the ``state_dict`` beyond the metadata.

    Examples
    --------
    >>> import lucid
    >>> import lucid.quantization as Q
    >>> obs = Q.NoopObserver()
    >>> x = lucid.randn(2, 3)
    >>> bool((obs(x) == x).all().item())         # identity passthrough
    True
    >>> isinstance(obs, Q.PlaceholderObserver)   # it *is* a PlaceholderObserver
    True

    See Also
    --------
    lucid.quantization.PlaceholderObserver : The parent — dtype carrier, qparams raise.
    lucid.quantization.FixedQParamsObserver : Fixed qparams for range-bounded ops.
    lucid.quantization.MinMaxObserver : Statistics-collecting observer for static ranges.
    lucid.quantization.FakeQuantize : Wraps an observer for QAT.
    """
