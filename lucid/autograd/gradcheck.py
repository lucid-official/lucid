"""
Numerical gradient checker — implemented entirely with the Lucid C++ engine.

No numpy is used.  Finite differences are built by perturbing elements
via scatter_add, computing forward passes through the engine, and comparing
analytical gradients from backward() with the numerical Jacobian columns.
"""

from typing import Callable, Sequence, cast

import lucid
from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._dispatch import _impl_with_grad, _wrap, _unwrap
from lucid._tensor.tensor import Tensor

_Func = Callable[..., Tensor | tuple[Tensor, ...]]


def _differentiated(inputs: Sequence[Tensor]) -> list[int]:
    """Positions of the inputs a check differentiates: those that require grad.

    The others are constants, as the reference treats them, so an index or a
    mask can be among the inputs.

    Raises
    ------
    ValueError
        If no input requires grad.
    """
    diff = [i for i, t in enumerate(inputs) if isinstance(t, Tensor) and t.requires_grad]
    if not diff:
        raise ValueError(
            "gradcheck expects at least one input tensor to require gradient, "
            "but none of them have requires_grad=True"
        )
    return diff


def _first(out: Tensor | tuple[Tensor, ...]) -> Tensor:
    """The output a check differentiates: ``func``'s, or the first of a tuple."""
    return out if isinstance(out, Tensor) else out[0]


def _analytical(
    func: _Func, inputs: Sequence[Tensor], diff: list[int]
) -> list[_C_engine.TensorImpl]:
    """``d func / d inputs[i]`` by backward, for each ``i`` in ``diff``, in F64."""
    args = list(inputs)
    for i in diff:
        args[i] = _wrap(_impl_with_grad(_C_engine.contiguous(_unwrap(inputs[i])), True))
    out = _first(func(*args))
    if _unwrap(out).numel() != 1:
        raise ValueError(
            f"gradcheck requires a scalar-valued function "
            f"(got shape {tuple(_unwrap(out).shape)})"
        )
    out.backward()
    grads: list[_C_engine.TensorImpl] = []
    for i in diff:
        g = args[i].grad
        if g is None:
            raise RuntimeError(f"gradcheck: input {i} has no gradient after backward()")
        grads.append(_C_engine.astype(_unwrap(g), _C_engine.F64))
    return grads


def _shifted_inputs(
    inputs: Sequence[Tensor], i: int, shifted: _C_engine.TensorImpl, shape: list[int]
) -> list[Tensor]:
    """The inputs of one finite-difference evaluation of ``d / d inputs[i]``.

    ``inputs[i]`` is ``shifted``, which stays in F64 so the step is exactly
    ``eps``: a cast back to the input's dtype would round it.  The other
    floating inputs go to F64 without grad, fresh for each evaluation; an
    integer or bool input (an index, a mask) passes through unchanged.
    """
    args: list[Tensor] = []
    for j, t in enumerate(inputs):
        impl = _unwrap(t)
        if j == i:
            args.append(_wrap(_C_engine.reshape(shifted, shape)))
        elif _promotion.is_integral(impl.dtype):
            args.append(t)
        else:
            plain = _C_engine.contiguous(impl).clone_with_grad(False)
            args.append(_wrap(_C_engine.astype(plain, _C_engine.F64)))
    return args


def _numerical(func: _Func, inputs: Sequence[Tensor], i: int, eps: float) -> _C_engine.TensorImpl:
    """``d func / d inputs[i]`` by central differences, in F64."""
    from lucid._factories.converters import tensor as _tensor_fn

    base = _C_engine.contiguous(_unwrap(inputs[i])).clone_with_grad(False)
    dev, numel, shape = base.device, base.numel(), list(base.shape)
    flat = _C_engine.astype(_C_engine.reshape(base, [numel]), _C_engine.F64)
    eps_v = _C_engine.full([1], float(eps), _C_engine.F64, dev)
    values: list[float] = []
    for k in range(numel):
        # Perturbation vector: eps at position k, 0 elsewhere.
        k_idx = _C_engine.full([1], float(k), _C_engine.I32, dev)
        zeros = _C_engine.zeros([numel], _C_engine.F64, dev)
        e_k = _C_engine.scatter_add(zeros, k_idx, eps_v, 0)
        plus = _first(func(*_shifted_inputs(inputs, i, _C_engine.add(flat, e_k), shape)))
        minus = _first(func(*_shifted_inputs(inputs, i, _C_engine.sub(flat, e_k), shape)))
        values.append((float(plus.item()) - float(minus.item())) / (2.0 * eps))
    # A Python list crosses into the engine here (interop boundary).
    num = _unwrap(_tensor_fn(values, dtype=lucid.float64))
    return _C_engine.reshape(num, shape)


def _mismatch(
    i: int, an: _C_engine.TensorImpl, nu: _C_engine.TensorImpl, atol: float, rtol: float
) -> str | None:
    """Why input ``i``'s analytical and numerical gradients differ, or ``None``."""
    diff = _C_engine.abs(_C_engine.sub(an, nu))
    thresh = _C_engine.add(
        _C_engine.full(list(diff.shape), atol, _C_engine.F64, diff.device),
        _C_engine.mul(
            _C_engine.full(list(diff.shape), rtol, _C_engine.F64, diff.device), _C_engine.abs(nu)
        ),
    )
    if bool(_wrap(_C_engine.all(_C_engine.less_equal(diff, thresh))).item()):
        return None
    max_diff = float(_wrap(_C_engine.max(diff, [], False)).item())
    max_thresh = float(_wrap(_C_engine.max(thresh, [], False)).item())
    diff_flat = _C_engine.reshape(diff, [diff.numel()])
    worst_k = int(_wrap(_C_engine.argmax(diff_flat, 0, False)).item())
    k_idx = _C_engine.full([1], float(worst_k), _C_engine.I32, diff.device)
    an_val = float(_wrap(_C_engine.gather(_C_engine.reshape(an, [an.numel()]), k_idx, 0)).item())
    nu_val = float(_wrap(_C_engine.gather(_C_engine.reshape(nu, [nu.numel()]), k_idx, 0)).item())
    return (
        f"Gradient check failed for input {i} "
        f"(atol={atol}, rtol={rtol}):\n"
        f"  Max |diff|    = {max_diff:.6g}\n"
        f"  Max threshold = {max_thresh:.6g}\n"
        f"  Worst element [flat {worst_k}]: "
        f"analytical={an_val:.6g}, numerical={nu_val:.6g}, "
        f"|diff|={abs(an_val - nu_val):.6g}"
    )


def gradcheck(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    inputs: Sequence[Tensor],
    *,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    raise_exception: bool = True,
) -> bool:
    """Compare analytical gradients from backward() against finite-difference Jacobians.

    Implemented entirely with the Lucid C++ engine — no numpy.  Only the
    inputs that require grad are differentiated, as in the reference
    framework; the others are held constant, so an index or a mask can be
    passed among them.

    Parameters
    ----------
    func : callable
        Scalar-valued function mapping Tensor inputs to a single-element Tensor.
    inputs : sequence of Tensor
        Inputs to func.  Those with ``requires_grad=True`` (float tensors)
        are checked; the rest pass through unchanged.
    eps : float
        Finite-difference step size (default 1e-6).
    atol, rtol : float
        Tolerances for the gradient comparison.
    raise_exception : bool
        Raise AssertionError on mismatch (default True), else return False.

    Returns
    -------
    bool  True if all gradients agree within tolerance.

    Raises
    ------
    ValueError
        If no input requires grad, or ``func`` is not scalar-valued.

    Examples
    --------
    >>> import lucid
    >>> from lucid.autograd import gradcheck
    >>> x = lucid.randn(3, dtype=lucid.float64, requires_grad=True)
    >>> index = lucid.tensor([0, 2])
    >>> gradcheck(lambda x, i: (x.gather(0, i) ** 2).sum(), [x, index])
    True
    """
    diff = _differentiated(inputs)
    analytical = _analytical(func, inputs, diff)
    for i, an in zip(diff, analytical, strict=True):
        msg = _mismatch(i, an, _numerical(func, inputs, i, eps), atol, rtol)
        if msg is not None:
            if raise_exception:
                raise AssertionError(msg)
            return False
    return True


def gradgradcheck(
    func: Callable[..., Tensor | tuple[Tensor, ...]],
    inputs: Sequence[Tensor],
    grad_outputs: Sequence[Tensor] | None = None,
    *,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    raise_exception: bool = True,
) -> bool:
    r"""Verify second-order gradients via finite differences.

    Most bugs in custom :class:`~lucid.autograd.Function.backward`
    implementations show up at the *second*-derivative level —
    the first-order gradient is consistent but the gradient of
    the gradient is not. ``gradgradcheck`` constructs such a
    test by wrapping ``func`` in a scalar-valued helper

    .. math::

        \tilde f(x) = \sum_i (\nabla f(x))_i,

    differentiates it analytically with
    ``create_graph=True``, and then runs :func:`gradcheck` on
    :math:`\tilde f` so its gradient is compared against the
    central finite-difference estimate

    .. math::

        \frac{\tilde f(x + \varepsilon e_k)
              - \tilde f(x - \varepsilon e_k)}{2 \varepsilon}
        \approx \frac{\partial^2 f(x)}{\partial x_k^2}.

    Disagreement signals a bug in the analytic backward
    formula that ordinary :func:`gradcheck` would miss.

    Parameters
    ----------
    func : callable
        Function mapping ``Tensor`` inputs to a ``Tensor`` (or
        tuple of ``Tensor``). Must be twice differentiable.
    inputs : sequence of Tensor
        Input tensors at which to verify the gradient.  Those with
        ``requires_grad=True`` (floating) are differentiated; the
        rest are held constant, as in :func:`gradcheck`.
    grad_outputs : sequence of Tensor or None, optional
        Reserved for custom upstream gradients in the inner
        backward pass. Currently ignored — ``ones_like``
        upstream gradients are always used.
    eps : float, optional
        Finite-difference step size used by the underlying
        :func:`gradcheck`. Defaults to ``1e-6``.
    atol : float, optional
        Absolute tolerance for the comparison. Defaults to
        ``1e-5``.
    rtol : float, optional
        Relative tolerance for the comparison. Defaults to
        ``1e-3``.
    raise_exception : bool, optional
        If ``True`` (default) raise ``AssertionError`` on
        mismatch; if ``False`` return ``False`` silently.

    Returns
    -------
    bool
        ``True`` iff all second-order gradients agree with the
        finite-difference reference within the supplied
        tolerances.

    Notes
    -----
    The bound on the truncation error of central differences is

    .. math::

        \left|
            \frac{\tilde f(x + \varepsilon) - \tilde f(x - \varepsilon)}
                 {2 \varepsilon}
            - \tilde f'(x)
        \right|
        = O(\varepsilon^2),

    so tightening ``eps`` improves accuracy until round-off
    error dominates.

    Examples
    --------
    >>> import lucid
    >>> from lucid.autograd import gradgradcheck
    >>> x = lucid.randn(3, requires_grad=True, dtype=lucid.float64)
    >>> def f(x):
    ...     return (x ** 3).sum()
    >>> gradgradcheck(f, [x])
    True
    """
    if grad_outputs is not None:
        # The reference framework allows custom upstream gradients; we
        # accept the kwarg for source compatibility but the simple
        # ones-grad path is always sufficient for verifying ``backward``.
        # A future pass can wire the supplied tensors through.
        pass

    diff = _differentiated(inputs)

    def _scalar_grad_fn(*args: Tensor) -> Tensor:
        # gradcheck hands the inputs it finite-differences over without
        # ``requires_grad``, so set it again on fresh leaves before we
        # differentiate: ``func`` runs on them, the first-order grad is taken
        # with ``create_graph=True``, and the returned scalar depends on them
        # through the grad-of-grad graph.  The other inputs stay constants.
        leaves = list(args)
        for i in diff:
            if not leaves[i].requires_grad:
                leaves[i] = leaves[i].detach().requires_grad_(True)
        out = func(*leaves)
        parts = [out] if isinstance(out, Tensor) else list(out)
        scalar = parts[0].sum()
        for p in parts[1:]:
            scalar = scalar + p.sum()
        grads = lucid.autograd.grad(
            scalar, [leaves[i] for i in diff], create_graph=True, retain_graph=True
        )
        result: Tensor = cast(Tensor, grads[0]).sum()
        for g in grads[1:]:
            result = result + cast(Tensor, g).sum()
        return result

    return gradcheck(
        _scalar_grad_fn, inputs, eps=eps, atol=atol, rtol=rtol, raise_exception=raise_exception
    )
