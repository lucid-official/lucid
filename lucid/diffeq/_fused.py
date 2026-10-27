"""Thin wrappers over the fused ``_C_engine.diffeq`` primitives.

Both the fixed-grid and the adaptive solver reduce almost all of their tensor
arithmetic to two engine calls, so the wrappers live here rather than in
either solver.

Engine ops are strict about dtype — promotion is the Python layer's job,
exactly as it is for ``Tensor.__add__``.  Getting that wrong is not a corner
case: under ``autocast`` on Metal the right-hand side returns float16 while
the state stays float32, and the mismatch is invisible on CPU because
autocast folds float16 back to float32 there.
"""

from typing import Sequence

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._tensor.tensor import Tensor

__all__: list[str] = []


def _promote(terms: Sequence[Tensor]) -> tuple[Tensor, ...]:
    """Cast a group of tensors to their common dtype.

    Parameters
    ----------
    terms : sequence of Tensor
        Operands about to be handed to a strict engine op.

    Returns
    -------
    tuple of Tensor
        The same tensors, each at the promoted dtype.

    Notes
    -----
    Every stage of every step passes through here, and on all but the mixed
    dtype paths it has nothing to do — a solver's stages inherit the state's
    dtype.  So the dtypes are read once each and compared before any promotion
    is attempted, rather than read once to fold, once to compare, and folded
    pairwise in between.  ``dtype`` objects come from a fixed table, so
    identity settles the common case without calling ``__eq__``.
    """
    dtypes = [t.dtype for t in terms]
    target = dtypes[0]
    if all(d is target for d in dtypes):
        return tuple(terms)

    for d in dtypes[1:]:
        target = lucid.promote_types(target, d)
    return tuple(t if d == target else t.to(target) for t, d in zip(terms, dtypes, strict=True))


def combine(y0: Tensor, ks: Sequence[Tensor], coeffs: Sequence[float], dt: float) -> Tensor:
    """Evaluate ``y0 + dt * sum_i coeffs[i] * ks[i]`` in one fused engine op.

    Parameters
    ----------
    y0 : Tensor
        Base state.
    ks : sequence of Tensor
        Stage derivatives, or any tensors being combined affinely.
    coeffs : sequence of float
        One host-side weight per entry of ``ks``.
    dt : float
        Scale applied to the whole sum.

    Returns
    -------
    Tensor
        The combined tensor, or ``y0`` itself when nothing is added.

    Notes
    -----
    An all-zero row contributes nothing, so the engine call is skipped — that
    covers the first stage of every method, whose tableau row is empty.
    """
    if not ks or not any(coeffs):
        return y0
    base, *stages = _promote([y0, *ks])
    return _wrap(
        _C_engine.diffeq.rk_combine(_unwrap(base), [_unwrap(s) for s in stages], list(coeffs), dt)
    )


def lincomb(terms: Sequence[Tensor], weights: Sequence[float]) -> Tensor:
    """Evaluate ``sum_i weights[i] * terms[i]`` in one fused engine op.

    Parameters
    ----------
    terms : sequence of Tensor
        Tensors to combine; must be non-empty and share a shape.
    weights : sequence of float
        One host-side weight per term.

    Returns
    -------
    Tensor
        The weighted sum.

    Notes
    -----
    Routed through the same fused op as :func:`combine` by folding the first
    term's weight into the base: ``terms[0] + 1 * ((w0 - 1) * terms[0] + ...)``
    is the requested sum, so the interpolation math costs one engine call per
    coefficient rather than a chain of multiplies and adds.
    """
    return combine(terms[0], list(terms), [weights[0] - 1.0, *weights[1:]], 1.0)


def error_ratio(
    y0: Tensor,
    y1: Tensor,
    ks: Sequence[Tensor],
    coeffs: Sequence[float],
    dt: float,
    rtol: float,
    atol: float,
) -> float:
    """Root-mean-square ratio of the embedded error estimate to its tolerance.

    Parameters
    ----------
    y0, y1 : Tensor
        States at the start and the proposed end of the step.
    ks : sequence of Tensor
        Stage derivatives of the step.
    coeffs : sequence of float
        Embedded-error weights (``b - b_hat``).
    dt : float
        Step size.
    rtol, atol : float
        Relative and absolute tolerances.

    Returns
    -------
    float
        The error ratio, on the host.  At most ``1`` means accept the step.

    Notes
    -----
    The controller has to branch on this value, so it crosses to the host no
    matter how it is computed; the fused op makes that exactly one
    synchronisation per step and never materialises the error tensor.

    The engine kernel widens half-precision operands internally so step
    control accumulates in wide floating point without allocating promoted
    tensors in Python.
    """
    operands = _promote([y0, y1, *ks])
    if not operands[0].is_floating_point():
        operands = tuple(t.to(lucid.float32) for t in operands)
    base, end, *stages = operands
    return _C_engine.diffeq.rk_error_norm(
        _unwrap(base),
        _unwrap(end),
        [_unwrap(s) for s in stages],
        list(coeffs),
        dt,
        rtol,
        atol,
    )


def poly_eval(coeffs: Sequence[Tensor], t0: float, t1: float, t: float) -> Tensor:
    """Evaluate a step polynomial in descending powers of normalised time.

    Parameters
    ----------
    coeffs : sequence of Tensor
        Coefficients, highest power first.
    t0, t1 : float
        Ends of the step the polynomial was fitted on; either orientation.
    t : float
        Query time.

    Returns
    -------
    Tensor
        The interpolated value.

    Notes
    -----
    The powers of the normalised time are host floats, so the whole
    evaluation is one affine combination of the coefficient tensors — a
    single fused engine call rather than a chain of scalar multiplies.
    """
    x = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    last = len(coeffs) - 1
    return lincomb(list(coeffs), [x ** (last - i) for i in range(len(coeffs))])


def broyden_probe(
    residual: Tensor, step: Tensor, state: Tensor, info: Tensor | None = None
) -> tuple[float, float, float, float]:
    """Answer one Broyden iteration's questions in a single pass.

    Parameters
    ----------
    residual : Tensor
        Current residual vector.
    step : Tensor
        Proposed Newton step.  Same element count as ``residual``; its shape
        need not match, since the linear solve returns a column.
    state : Tensor
        Current iterate.  Same element count as ``residual``.
    info : Tensor or None, default=None
        Status scalar from the linear solve; non-zero means singular.  ``None``
        for the seeding call made before any solve, which reports zero.

    Returns
    -------
    tuple of float
        ``(sum(residual**2), sum(step**2), sum(state**2), info)``, on the host.

    Notes
    -----
    All of these are branched on every iteration, so they cross to the host no
    matter how they are computed.  Asking for them separately costs one device
    synchronisation each, and on the GPU a synchronisation is not the price of
    moving eight bytes -- MLX evaluates lazily, so it waits for everything
    queued behind it.  Fused, an iteration pays for one.

    The engine kernel widens half-precision operands internally, matching
    :func:`error_ratio`, so convergence norms accumulate in wide floating
    point without allocating promoted tensors in Python.
    """
    operands = _promote([residual, step, state])
    if not operands[0].is_floating_point():
        operands = tuple(t.to(lucid.float32) for t in operands)
    res, stp, cur = operands
    raw = None if info is None else _unwrap(info)
    return _C_engine.diffeq.broyden_probe(_unwrap(res), _unwrap(stp), _unwrap(cur), raw)
