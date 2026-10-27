"""Shared ``from_float`` helpers for quantized modules.

Quantizing a calibrated (PTQ) or trained (QAT) float module is the same
two-step recipe for every weighted layer: derive the **weight** qparams and
read the **activation** qparams, then bake them in.  These helpers centralise
that so ``Linear`` / the ``Conv`` family stay tiny and cover both the PTQ and
QAT source modules.
"""

import warnings
from typing import TYPE_CHECKING, cast

import lucid
from lucid.nn.module import Module
from lucid.quantization._functional import fake_quantize, quantize
from lucid.quantization._qscheme import quint8

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

    import lucid.nn as nn
    from lucid.nn.parameter import Parameter
    from lucid.quantization._fake_quantize import FakeQuantize
    from lucid.quantization._qscheme import QDtype
    from lucid.quantization.observer import ObserverBase
    from lucid.quantization.qconfig import QConfig


class _QuantizedGridModule(Module):
    """Shared base for modules carrying an output quantization grid ``(scale, zero_point, qdtype)``."""

    scale: Tensor
    zero_point: Tensor

    def __init__(self, scale: Tensor, zero_point: Tensor, qdtype: QDtype = quint8) -> None:
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("zero_point", zero_point)
        self.qdtype = qdtype

    def _fake_quant(self, x: Tensor) -> Tensor:
        """Fake-quantize ``x`` onto ``(scale, zero_point, qdtype)``."""
        return fake_quantize(
            x, self.scale, self.zero_point, self.qdtype.quant_min, self.qdtype.quant_max
        )


def quantize_weight(mod: nn.Module) -> tuple[Tensor, Tensor, Tensor, int]:
    """Quantize a module's weight, returning ``(codes, scale, zero_point, ch_axis)``.

    Works for both sources: a **QAT** module carries a ``weight_fake_quant``
    whose observer already saw the (trained) weight — reuse its qparams; a
    **PTQ** float module instead runs its ``qconfig``'s weight observer over
    the static weight now.
    """
    weight = cast("Tensor | Parameter", mod.weight)
    wfq = getattr(mod, "weight_fake_quant", None)
    if wfq is not None:
        wfq_m = cast("FakeQuantize", wfq)
        wfq_m(weight)  # refresh the observer on the current weight
        scale, zero_point = wfq_m.calculate_qparams()
        qdtype, ch_axis = wfq_m.qdtype, wfq_m.ch_axis
    else:
        qconfig = cast("QConfig", mod.qconfig)
        wobs = cast("ObserverBase", qconfig.weight())
        wobs(weight)
        scale, zero_point = wobs.calculate_qparams()
        qdtype, ch_axis = wobs.qdtype, wobs.ch_axis
    axis = ch_axis if ch_axis is not None else 0
    # Bake the qparams on the weight's device so the quantized module is
    # single-device (a HistogramObserver derives its range from host floats, so
    # its qparams land on CPU even for a Metal weight — a mixed-device module is
    # both a runtime DeviceMismatch and an uncompilable mixed-device trace).
    scale, zero_point = _to_device(scale, zero_point, weight.device)
    codes = quantize(weight, scale, zero_point, qdtype, ch_axis=ch_axis)
    return codes, scale, zero_point, axis


def activation_qparams(mod: nn.Module) -> tuple[Tensor, Tensor, QDtype]:
    """Read the output ``(scale, zero_point, qdtype)`` from a module's observer.

    ``activation_post_process`` is a bare observer for PTQ and a
    :class:`FakeQuantize` for QAT — both expose ``calculate_qparams`` / ``qdtype``.
    """
    obs = cast("ObserverBase", mod.activation_post_process)
    # A PTQ observer that never saw calibration data keeps its ±inf seed, which
    # collapses ``scale`` to ``eps`` and makes the quantized output ~0.  Warn
    # loudly rather than fail silently.  (QAT's FakeQuantize has no ``min_val``,
    # so this only fires on the uncalibrated static path.)
    min_val = getattr(obs, "min_val", None)
    if min_val is not None and bool(lucid.isinf(cast("Tensor", min_val)).any().item()):
        warnings.warn(
            f"Quantizing {type(mod).__name__} whose activation observer never saw "
            "data — run calibration through the prepared model before convert(), "
            "or the quantized output collapses to a near-zero grid.",
            stacklevel=3,
        )
    scale, zero_point = obs.calculate_qparams()
    scale, zero_point = _to_device(scale, zero_point, _module_device(mod))
    return scale, zero_point, obs.qdtype


def _module_device(mod: nn.Module) -> lucid.device:
    """Best-effort device of ``mod``'s activations.

    Prefer the layer weight; a weightless activation module (Sigmoid, ELU, …)
    has none, so fall back to any parameter, then the activation observer's
    running buffer (a MinMax observer's ``min_val`` rides the observed device),
    then CPU.  Used to bake qparams onto the device the module runs on.
    """
    w = getattr(mod, "weight", None)
    if isinstance(w, lucid.Tensor):
        return w.device
    for p in mod.parameters():
        return p.device
    obs = getattr(mod, "activation_post_process", None)
    mv = getattr(obs, "min_val", None)
    if isinstance(mv, lucid.Tensor):
        return mv.device
    return lucid.tensor(0.0).device


def _to_device(scale: Tensor, zero_point: Tensor, dev: lucid.device) -> tuple[Tensor, Tensor]:
    """Move a ``(scale, zero_point)`` pair onto ``dev`` (no-op if already there)
    so a quantized module built from a Metal float module keeps every buffer on
    Metal — a HistogramObserver derives its range from host floats, so its
    qparams otherwise land on CPU and strand the module across two devices."""
    if scale.device != dev:
        scale = scale.to(dev)
    if zero_point.device != dev:
        zero_point = zero_point.to(dev)
    return scale, zero_point
