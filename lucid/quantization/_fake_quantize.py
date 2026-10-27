"""``FakeQuantize`` — an observer paired with a straight-through fake-quant.

During quantization-aware training a :class:`FakeQuantize` module sits on
a weight or activation: on every forward it (optionally) lets its
observer refresh ``(scale, zero_point)`` from the live statistics, then
(optionally) applies :func:`~lucid.quantization.fake_quantize` so the
downstream graph sees quantization rounding while gradients still flow
via the straight-through estimator.

The two toggles — ``enable_observer`` / ``enable_fake_quant`` — mirror the
reference framework and drive the standard QAT schedule: observe+fake-quant
early, then freeze the observer (and later the BN stats) as training
converges.
"""

import functools
from typing import TYPE_CHECKING, override

import lucid
import lucid.nn as nn
from lucid.quantization._functional import fake_quantize
from lucid.quantization._qscheme import QDtype, QScheme
from lucid.quantization.observer import (
    MovingAverageMinMaxObserver,
    ObserverBase,
    _store_running_stat,
)

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


class FakeQuantize(nn.Module):
    r"""Observer + straight-through fake-quantization, toggleable for QAT.

    The building block of **quantization-aware training** (QAT). A ``FakeQuantize`` sits
    on a weight or activation and, on every forward, does two things: it lets its wrapped
    observer refresh ``(scale, zero_point)`` from the live statistics, then it applies
    :func:`~lucid.quantization.fake_quantize` — rounding the tensor onto the integer grid
    and back to float — so the downstream graph *sees the quantization error* while
    training still runs in float. This is what lets a network learn weights that are
    robust to the eventual int8 inference numerics: the forward simulates quantization,
    and a straight-through estimator (STE) keeps the operation differentiable so the
    rounding does not block gradients.

    **Observer vs fake-quant, decoupled.** The two behaviours are independently gated by
    :meth:`enable_observer` / :meth:`enable_fake_quant`, which drives the standard QAT
    schedule: observe + fake-quant early so statistics settle and the net adapts, then
    :meth:`disable_observer` to freeze the grid once ranges converge (and, in fused
    stacks, freeze BN stats). With the observer off but fake-quant on, the layer keeps
    quantizing against the last-learned qparams; with fake-quant off it is a pure
    passthrough (the observer may still watch).

    **The straight-through fake-quant.** Writing :math:`S, Z` for the current
    ``(scale, zero_point)`` and :math:`[q_\min, q_\max]` for the grid, the forward
    computes the quantize→dequantize round-trip in the float domain:

    .. math::

        \hat{x} = \bigl(\operatorname{clip}(\operatorname{round}(x/S) + Z,\
            q_\min,\ q_\max) - Z\bigr)\,S.

    The :math:`\operatorname{round}` step has a zero (a.e.) derivative, which would kill
    training, so the backward substitutes the **straight-through estimator** — gradient
    passes through unchanged inside the grid and is zeroed where the code saturates:

    .. math::

        \frac{\partial \hat{x}}{\partial x} =
        \begin{cases}
            1 & q_\min \le \operatorname{round}(x/S) + Z \le q_\max \\
            0 & \text{otherwise}
        \end{cases}

    so :math:`\nabla_x = \nabla_{\hat{x}} \cdot \mathbb{1}[\,\text{code in range}\,]`.
    The qparams :math:`S, Z` are treated as constants of the step (they are refreshed by
    the observer, not learned through this gradient).

    Parameters
    ----------
    observer : type[ObserverBase], default MovingAverageMinMaxObserver
        Observer *class* (not instance) instantiated once to track the statistics from
        which ``(scale, zero_point)`` are derived. The default matches the activation
        recipe of :func:`~lucid.quantization.get_default_qat_qconfig`.
    **observer_kwargs : object
        Forwarded verbatim to the observer constructor — e.g. ``qscheme`` / ``qdtype`` /
        ``ch_axis`` / ``averaging_constant`` — so one :class:`FakeQuantize` can wrap any
        observer configuration (per-tensor activation or per-channel weight).

    Attributes
    ----------
    activation_post_process : ObserverBase
        The live wrapped observer instance; its :meth:`calculate_qparams` supplies the
        grid on each forward.
    scale : Tensor
        Scalar (or length-``C`` for per-channel) quantization step, refreshed from the
        observer while observing is enabled. Seeded to ``1.0``.
    zero_point : Tensor
        The matching zero-point, refreshed alongside ``scale``. Seeded to ``0.0``.
    qdtype : QDtype
        Target dtype, mirrored from the wrapped observer (supplies the grid bounds).
    qscheme : QScheme
        Target scheme, mirrored from the wrapped observer.
    ch_axis : int or None
        Per-channel axis, mirrored from the wrapped observer (``None`` for per-tensor),
        passed to :func:`~lucid.quantization.fake_quantize` so per-channel qparams
        broadcast correctly.

    Notes
    -----
    - **QAT schedule.** Typical use: start with both toggles on; call
      :meth:`disable_observer` once activation ranges stabilise to freeze the grid, so the
      final quantized weights are trained against fixed qparams.
    - **STE saturation mask.** The backward zeroes gradient for values whose code lands
      outside ``[q_min, q_max]`` — saturated activations receive no learning signal,
      which nudges the observer's range to cover them.
    - **Buffer re-registration.** Refreshed ``scale`` / ``zero_point`` are written back
      via ``register_buffer`` (same name), the same in-place-update contract the
      observers use, so the qparams persist through ``state_dict``.
    - **Per-channel weights.** When wrapping a per-channel observer (``ch_axis`` set), the
      fake-quant applies a distinct ``(scale, zero_point)`` per channel — the QAT analogue
      of the per-channel int8 weight used at inference by
      :class:`~lucid.nn.quantized.Linear`.
    - **Default QAT recipe.** :func:`~lucid.quantization.get_default_qat_qconfig` builds
      one ``FakeQuantize`` for activations (wrapping
      :class:`~lucid.quantization.MovingAverageMinMaxObserver`) and one for weights
      (wrapping :class:`~lucid.quantization.PerChannelMinMaxObserver`).
    - Use :meth:`with_args` to defer construction into a
      :class:`~lucid.quantization.QConfig` as a zero-arg factory.

    Examples
    --------
    >>> import lucid
    >>> import lucid.quantization as Q
    >>> fq = Q.FakeQuantize()                    # wraps MovingAverageMinMaxObserver
    >>> y = fq(lucid.randn(8, 16))               # observe, then round-trip through grid
    >>> y.shape
    (8, 16)
    >>> bool(fq.scale.item() > 0)                # qparams refreshed from the observer
    True

    Freezing the observer holds the grid fixed while fake-quant keeps applying it — the
    QAT "freeze ranges near convergence" step:

    >>> _ = fq.disable_observer()
    >>> frozen = fq.scale.item()
    >>> _ = fq(lucid.randn(8, 16) * 100.0)       # a wild batch cannot move the grid now
    >>> fq.scale.item() == frozen
    True

    Turning fake-quant off makes the module a pure passthrough (identity):

    >>> _ = fq.disable_fake_quant()
    >>> x = lucid.randn(2, 4)
    >>> bool((fq(x) == x).all().item())
    True

    See Also
    --------
    lucid.quantization.fake_quantize : The underlying STE round-trip primitive.
    lucid.quantization.MovingAverageMinMaxObserver : Default wrapped activation observer.
    lucid.quantization.PerChannelMinMaxObserver : Default wrapped weight observer.
    lucid.quantization.get_default_qat_qconfig : Builds the default QAT FakeQuantize pair.
    lucid.nn.quantized.Linear : The int8 inference layer QAT trains toward.
    """

    scale: Tensor
    zero_point: Tensor

    def __init__(
        self, observer: type[ObserverBase] = MovingAverageMinMaxObserver, **observer_kwargs: object
    ) -> None:
        super().__init__()
        self.activation_post_process: ObserverBase = observer(**observer_kwargs)  # type: ignore[arg-type]
        self.qdtype: QDtype = self.activation_post_process.qdtype
        self.qscheme: QScheme = self.activation_post_process.qscheme
        self.ch_axis: int | None = self.activation_post_process.ch_axis
        self.register_buffer("scale", lucid.tensor(1.0))
        self.register_buffer("zero_point", lucid.tensor(0.0))
        self._observer_enabled = True
        self._fake_quant_enabled = True

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Refresh qparams (if observing) then fake-quantize (if enabled)."""
        if self._observer_enabled:
            # The qparams are statistics, not a function of ``x`` to
            # differentiate: the gradient reaches ``x`` through the
            # straight-through estimator below, never through them.
            with lucid.no_grad():
                self.activation_post_process(x)
                scale, zero_point = self.activation_post_process.calculate_qparams()
                _store_running_stat(self, "scale", scale)
                _store_running_stat(self, "zero_point", zero_point)
        if self._fake_quant_enabled:
            return fake_quantize(
                x,
                self.scale,
                self.zero_point,
                self.qdtype.quant_min,
                self.qdtype.quant_max,
                self.ch_axis,
            )
        return x

    def enable_observer(self, enabled: bool = True) -> FakeQuantize:
        """Enable/disable statistic collection; returns ``self`` for chaining."""
        self._observer_enabled = enabled
        return self

    def disable_observer(self) -> FakeQuantize:
        """Freeze the observer (stop refreshing ``scale`` / ``zero_point``)."""
        return self.enable_observer(False)

    def enable_fake_quant(self, enabled: bool = True) -> FakeQuantize:
        """Enable/disable the fake-quant transform; returns ``self``."""
        self._fake_quant_enabled = enabled
        return self

    def disable_fake_quant(self) -> FakeQuantize:
        """Pass activations through unchanged (observer may still run)."""
        return self.enable_fake_quant(False)

    def calculate_qparams(self) -> tuple[Tensor, Tensor]:
        """Delegate to the wrapped observer."""
        return self.activation_post_process.calculate_qparams()

    @classmethod
    def with_args(cls, **kwargs: object) -> functools.partial[FakeQuantize]:
        """Return a zero-arg factory building this module with ``kwargs``."""
        return functools.partial(cls, **kwargs)  # type: ignore[arg-type]
