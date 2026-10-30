r"""Automatic Mixed Precision (AMP) utilities for Lucid.

Mixed precision training keeps the master copy of parameters and the
loss in float32 while running the bulk of forward / backward compute in
a lower-precision format (float16 or bfloat16). On Apple Silicon this
typically yields a wall-clock speed-up of roughly two times on
compute-bound layers and roughly halves the activation memory
footprint, enabling larger batch sizes or longer context windows.

This subpackage exposes two cooperating tools:

- :class:`autocast` — a context manager that casts eligible ops to a
  lower-precision dtype while leaving reductions and accumulators in
  float32 for numerical safety.
- :class:`GradScaler` — a dynamic loss-scaling helper that prevents
  gradient underflow when training in float16. Tracks an exponential
  scale factor, halving it on detected ``inf`` / ``nan`` and growing
  it back up on successful steps.

and the pieces around them:

- :func:`custom_fwd` / :func:`custom_bwd` — decorators that make a custom
  :class:`~lucid.autograd.Function` follow the autocast state its ``forward``
  ran in, so its ``backward`` does not meet half-precision gradients with
  float32 saved tensors.
- :func:`is_autocast_available` — whether a device type supports autocast.

Examples
--------
>>> import lucid
>>> import lucid.nn as nn
>>> import lucid.optim as optim
>>> from lucid.amp import autocast, GradScaler
>>> model = nn.Linear(4, 2).to("metal")
>>> optimizer = optim.SGD(model.parameters(), lr=0.1)
>>> x = lucid.randn(8, 4, device="metal")
>>> scaler = GradScaler()
>>> with autocast(device_type="metal", dtype=lucid.float16):
...     out = model(x)  # forward pass, in float16
>>> out.dtype
lucid.float16
>>> loss = out.float().pow(2).mean()
>>> scaler.scale(loss).backward()
>>> scaler.step(optimizer)
>>> scaler.update()
"""

from lucid.amp._custom import custom_bwd, custom_fwd
from lucid.amp.autocast import autocast, is_autocast_available
from lucid.amp.grad_scaler import GradScaler

__all__ = ["autocast", "GradScaler", "custom_fwd", "custom_bwd", "is_autocast_available"]
