"""lucid.autograd — reverse-mode differentiation and the grad-mode controls.

``backward`` and ``grad`` walk the tape the engine records as ops execute.
``Function`` together with ``FunctionCtx`` lets Python define a new
differentiable op by writing its forward and backward explicitly, for cases
the composed ops cannot express.

Grad modes: ``no_grad``, ``enable_grad``, ``inference_mode``, and the
``set_grad_enabled`` / ``is_grad_enabled`` pair.  Note that
``set_grad_enabled(flag)`` is a plain function returning ``None``, *not* a
context manager — using it in a ``with`` statement turns grad off globally
and never restores it, which then affects every later computation.

Higher-order and verification: ``jacobian``, ``hessian``, ``vjp``, ``jvp``,
``hvp`` and ``vhp`` for derivatives of derivatives, and ``gradcheck`` /
``gradgradcheck`` to verify a hand-written backward against finite
differences — worth running on any new ``Function``, since a wrong backward produces plausible numbers
rather than an error.  ``checkpoint`` trades recomputation for activation
memory, and ``detect_anomaly`` traces the op that first produced a NaN.
"""

from lucid.autograd._grad_mode import (
    no_grad,
    enable_grad,
    set_grad_enabled,
    is_grad_enabled,
    inference_mode,
)
from lucid.autograd._backward import backward, grad
from lucid.autograd.function import Function, FunctionCtx
from lucid.autograd.gradcheck import gradcheck, gradgradcheck
from lucid.autograd._anomaly import detect_anomaly, is_anomaly_enabled, set_detect_anomaly
from lucid.autograd._functional import jacobian, hessian, vjp, jvp, hvp, vhp
from lucid.autograd.checkpoint import checkpoint
from lucid.autograd._hooks import RemovableHandle
from lucid.autograd import profiler as profiler
from lucid.autograd import graph as graph

__all__ = [
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "is_grad_enabled",
    "inference_mode",
    "backward",
    "grad",
    "Function",
    "FunctionCtx",
    "gradcheck",
    "gradgradcheck",
    "detect_anomaly",
    "set_detect_anomaly",
    "is_anomaly_enabled",
    "jacobian",
    "hessian",
    "vjp",
    "jvp",
    "hvp",
    "vhp",
    "checkpoint",
    "RemovableHandle",
    "profiler",
    "graph",
]
