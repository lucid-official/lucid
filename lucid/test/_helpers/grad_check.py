"""Finite-difference gradient check.

Used by the canonical per-op test template — every differentiable op
gets a `gradcheck` call to ensure its registered backward agrees with
numerically-perturbed forward values to within a tolerance.

This is *the* numerical-correctness gate for autograd kernels; if a
backward formula is wrong, this catches it.
"""

from collections.abc import Callable
from typing import Sequence

import numpy as np

import lucid
from lucid._tensor.tensor import Tensor


def _flat_jacobian_finite_diff(
    fn: Callable[..., Tensor], inputs: Sequence[Tensor], *, eps: float, output_idx: int
) -> np.ndarray:
    """Numeric Jacobian of ``fn``'s ``output_idx``-th output w.r.t.
    ``inputs[0]`` via central differences.

    Returns shape ``(out_numel, in_numel)``.
    """
    base = fn(*inputs)
    if isinstance(base, tuple):
        base = base[output_idx]
    out_numel = int(base.numel())

    in_t = inputs[0]
    in_arr = in_t.numpy().reshape(-1).astype(np.float64).copy()
    in_numel = in_arr.size

    jac = np.zeros((out_numel, in_numel), dtype=np.float64)
    for j in range(in_numel):
        plus = in_arr.copy()
        minus = in_arr.copy()
        plus[j] += eps
        minus[j] -= eps

        in_plus = lucid.tensor(
            plus.reshape(in_t.shape).astype(np.float32), dtype=in_t.dtype, device=in_t.device
        )
        in_minus = lucid.tensor(
            minus.reshape(in_t.shape).astype(np.float32), dtype=in_t.dtype, device=in_t.device
        )
        rest = list(inputs[1:])
        out_plus = fn(in_plus, *rest)
        out_minus = fn(in_minus, *rest)
        if isinstance(out_plus, tuple):
            out_plus = out_plus[output_idx]
            out_minus = out_minus[output_idx]
        diff = out_plus.numpy().reshape(-1).astype(np.float64) - out_minus.numpy().reshape(
            -1
        ).astype(np.float64)
        jac[:, j] = diff / (2.0 * eps)
    return jac


def _flat_jacobian_autograd(
    fn: Callable[..., Tensor], inputs: Sequence[Tensor], *, output_idx: int
) -> np.ndarray:
    """Analytic Jacobian via autograd: reverse-mode VJP applied to each
    output basis vector."""
    inputs = [x.requires_grad_(True) if isinstance(x, Tensor) else x for x in inputs]
    out = fn(*inputs)
    if isinstance(out, tuple):
        out = out[output_idx]
    out_numel = int(out.numel())
    in_numel = int(inputs[0].numel())

    jac = np.zeros((out_numel, in_numel), dtype=np.float64)
    out.reshape(-1)
    for k in range(out_numel):
        # Re-run forward each time because autograd graphs are freed
        # after backward by default.
        fresh = [
            (
                lucid.tensor(
                    x.numpy().copy(), dtype=x.dtype, device=x.device, requires_grad=x.requires_grad
                )
                if isinstance(x, Tensor)
                else x
            )
            for x in inputs
        ]
        out_k = fn(*fresh)
        if isinstance(out_k, tuple):
            out_k = out_k[output_idx]
        flat_k = out_k.reshape(-1)
        seed = lucid.zeros_like(flat_k)
        seed_np = np.zeros(out_numel, dtype=np.float32)
        seed_np[k] = 1.0
        seed = lucid.tensor(seed_np.reshape(out_k.shape), dtype=out_k.dtype, device=out_k.device)
        out_k.backward(seed)
        g = fresh[0].grad
        if g is None:
            jac[k, :] = 0.0
        else:
            jac[k, :] = g.numpy().reshape(-1).astype(np.float64)
    return jac


def gradcheck(
    fn: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    *,
    eps: float = 1e-3,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    output_idx: int = 0,
) -> bool:
    """Compare finite-difference and autograd Jacobians of ``fn``.

    Inputs must be float tensors; only ``inputs[0]`` is differentiated
    against in this minimal check (most lucid ops only have one
    "primary" input — extend this when a test needs multi-input
    gradchecks).
    """
    fd = _flat_jacobian_finite_diff(fn, inputs, eps=eps, output_idx=output_idx)
    ag = _flat_jacobian_autograd(fn, inputs, output_idx=output_idx)
    np.testing.assert_allclose(
        ag,
        fd,
        atol=atol,
        rtol=rtol,
        err_msg=f"gradcheck mismatch for {getattr(fn, '__name__', fn)!r}",
    )
    return True
