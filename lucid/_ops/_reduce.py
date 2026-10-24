"""The numerics the norm and variance reductions share.

``var`` / ``std`` (registry adapters), ``var_mean`` / ``std_mean``,
``linalg.vector_norm`` / ``linalg.norm``, ``cdist`` and
``nn.functional.pairwise_distance`` all end in the same three places, and
each used to get one of them wrong on its own:

* **the root of a sum** — ``sqrt(sum)`` and ``sum ** (1/p)`` have an
  infinite slope at zero, so a zero norm or a constant slice's ``std``
  sent ``inf * 0 = nan`` back through autograd.  :func:`_zero_safe_root`
  gives zero there, the subgradient the reference picks;
* **the divisor of a variance** — ``max(N - correction, 0)``, so too few
  elements answer ``nan`` (no spread to divide) or ``inf``, never a
  variance that silently skipped its correction (:func:`_bessel_correct`);
* **the axes** — a dim named twice, or one that is not an integer, is
  refused in the called op's name, rather than counted twice in ``N``
  (:func:`_reduction_axes`; every reduction adapter takes its axes here).

The four variance functions also read their arguments in one place,
:func:`_variance_operands`.
"""

import math
import operator
import warnings
from collections.abc import Callable, Sequence

from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._deprecation import _PACKAGE_DIR
from lucid._dtype import _ENGINE_TO_DTYPE, to_engine_dtype
from lucid._ops._autocast import _autocast_suspended
from lucid._ops._dims import _to_axes, _wrap_dims
from lucid._types import DTypeLike

_Impl = _C_engine.TensorImpl


class _ValuesIndices[T](tuple[T, T]):
    """``(values, indices)`` pair returned by order-statistics reductions."""

    @property
    def values(self) -> T:
        return self[0]

    @property
    def indices(self) -> T:
        return self[1]


class _MinMax[T](tuple[T, T]):
    """``(min, max)`` pair returned by :func:`lucid.aminmax`."""

    @property
    def min(self) -> T:
        return self[0]

    @property
    def max(self) -> T:
        return self[1]


# ── Axes ─────────────────────────────────────────────────────────────────────


def _reduction_axes(x_impl: _Impl, dim: object, name: str) -> list[int]:
    """``dim`` as engine axes, refusing a non-integer one or an axis named twice.

    The engine reduces a repeated axis once, so ``sum`` gave the right
    number and ``var`` the wrong one — its element count multiplied the
    axis in twice.  The reference refuses the call; so does this, in the
    name of the public op that was called, by the engine's own dim rule
    (:func:`lucid._ops._dims._wrap_dims` — a 0-d tensor's 0 and -1 are one
    axis).  A float used to truncate (``int(1.5)``) or fail as "not
    iterable".  A 0-d tensor's notional axis is every axis it has (``[]``):
    ``var`` read ``shape[0]`` of it for its divisor and raised
    ``IndexError``.
    """
    axes = _to_axes(dim, name)
    ndim = len(x_impl.shape)
    if len(axes) > 1 or ndim == 0:
        axes = _wrap_dims(axes, ndim, name)
    # That one axis is the whole tensor: the engine's empty list, whose
    # element count the variance divisor reads as 1, not as shape[0].
    return [] if ndim == 0 else axes


# ── Roots and divisors ───────────────────────────────────────────────────────


def _zero_safe_root(s: _Impl, p: float) -> _Impl:
    """``s ** (1/p)`` for ``s >= 0``, with a zero gradient where ``s`` is 0.

    The root's slope is infinite at zero and the sum's gradient there is
    zero, so autograd multiplied them into ``nan``.  Both branches here are
    finite: the root is taken of 1 where ``s`` is 0, and that branch's
    result is then dropped for 0.  A ``nan`` or ``inf`` sum is not 0 and
    passes through as itself.  The values are the plain root's, so a sum
    no gradient will reach takes the plain root.
    """
    if not s.requires_grad:
        return _C_engine.sqrt(s) if p == 2 else _C_engine.pow_scalar(s, 1.0 / p)
    at_zero = _C_engine.equal(s, _C_engine.zeros_like(s))
    safe = _C_engine.where(at_zero, _C_engine.ones_like(s), s)
    # The root of an integer sum is floating, so its zero is taken from it.
    root = _C_engine.sqrt(safe) if p == 2 else _C_engine.pow_scalar(safe, 1.0 / p)
    return _C_engine.where(at_zero, _C_engine.zeros_like(root), root)


#: A set rather than a tuple: the membership test is on every norm's path.
_HALF = frozenset({_C_engine.F16, _C_engine.BF16})

#: The dtypes a real-valued distance or similarity is defined on.
_FLOATING = frozenset(
    {
        _C_engine.F16,
        _C_engine.BF16,
        _C_engine.F32,
        _C_engine.F64,
    }
)

#: The dtypes a norm is defined on.
_NORMABLE = _FLOATING | frozenset({_C_engine.C64, _C_engine.C128})


def _require_floating(impl: _Impl, op: str) -> _Impl:
    """``impl``, or a refusal if it is not a real floating-point input to ``op``.

    ``cdist``, ``pdist`` and ``cosine_similarity`` are defined on real
    vectors only; the reference refuses both integer/bool and complex
    inputs.  On Metal ``cdist`` and ``cosine_similarity`` used to run
    their real formulas on complex entries and return complex numbers,
    while the CPU failed later inside ``matmul`` or ``lp_normalize``.
    """
    if impl.dtype not in _FLOATING:
        raise _C_engine.DtypeMismatch(
            f"{op}: expected a floating point tensor as input, "
            f"got {_ENGINE_TO_DTYPE[impl.dtype]}"
        )
    return impl


def _require_normable(impl: _Impl, op: str) -> _Impl:
    """``impl``, or a refusal if it is an integer or bool input to the norm ``op``.

    The integer path answered by accident, differently per order: ``ord``
    2 and 0 came back float32, 1 and ``inf`` in the input's own dtype, and
    every other order raised from deep inside ``pow_scalar``.  A
    ``DtypeMismatch`` is a ``TypeError`` and a ``RuntimeError``, the class
    the reference raises.  ``dtype=`` does not lift the refusal: the
    reference checks the input before any cast.
    """
    if impl.dtype not in _NORMABLE:
        raise _C_engine.DtypeMismatch(
            f"{op}: expected a floating point or complex tensor as input, "
            f"got {_ENGINE_TO_DTYPE[impl.dtype]}"
        )
    return impl


#: The orders that reduce the magnitudes directly, with no power or root.
_ABS_REDUCTIONS = {
    math.inf: _C_engine.max,
    -math.inf: _C_engine.min,
    1.0: _C_engine.sum,
}


def _p_norm(r: _Impl, p: float, axes: list[int], keepdim: bool) -> _Impl:
    """``(sum |r|^p)^(1/p)`` over ``axes`` for a real ``r`` and ``p != 0``.

    Its gradient is zero wherever the norm is — the subgradient the
    reference picks — and nowhere ``nan``.  The root's slope is infinite at
    a zero sum (:func:`_zero_safe_root`), and for ``p < 1`` so is
    ``|r|^p``'s at a zero entry; both used to meet a zero factor in
    autograd and leave ``nan`` behind.

    A half input reduces in float32 and answers in its own dtype, as the
    reference's op-math does: the sum of squares of 65 536 ones is already
    past float16's largest finite value, so the norm 256 came back ``inf``.
    """
    if r.dtype in _HALF:
        out = _p_norm(_C_engine.astype(r, _C_engine.F32), p, axes, keepdim)
        return _C_engine.astype(out, r.dtype)
    if p in _ABS_REDUCTIONS:
        return _ABS_REDUCTIONS[p](_C_engine.abs(r), axes, keepdim)
    if p == 2:
        # r² = |x|² for a real r, one product cheaper than ``abs`` first.
        return _zero_safe_root(_C_engine.sum(_C_engine.mul(r, r), axes, keepdim), 2)
    magnitude = _C_engine.abs(r)
    if p > 1 or not magnitude.requires_grad:
        powered = _C_engine.pow_scalar(magnitude, p)
    else:
        # A zero entry contributes 0 to the sum (``inf`` for p < 0, which
        # makes the norm 0) and nothing to the gradient.
        nonzero = _C_engine.not_equal(magnitude, _C_engine.zeros_like(magnitude))
        safe = _C_engine.where(nonzero, magnitude, _C_engine.ones_like(magnitude))
        raised = _C_engine.pow_scalar(safe, p)
        fill = _C_engine.full_like(raised, 0.0 if p > 0 else math.inf)
        powered = _C_engine.where(nonzero, raised, fill)
    s = _C_engine.sum(powered, axes, keepdim)
    if p > 0:
        return _zero_safe_root(s, p)
    return _C_engine.pow_scalar(s, 1.0 / p)


def _bessel_correct(
    result_impl: _Impl,
    x_impl: _Impl,
    axes_list: Sequence[int],
    correction: float,
    name: str,
) -> _Impl:
    """Turn the engine's ``sum / N`` variance into ``sum / max(N - correction, 0)``.

    With ``N <= correction`` the divisor is 0, as the reference has it: a
    slice with no spread answers ``0 / 0 = nan`` and one with spread
    ``inf``.  The correction used to be skipped there instead, so
    ``var([5.])`` answered 0.  An empty reduction is ``nan`` for any
    correction — the engine's CPU kernel answers 0 for it, Metal ``nan``,
    and scaling by ``inf`` sends both to ``nan``.

    With ``N < correction`` the reference's gradient still divides by the
    negative ``N - correction``, so it is finite under a ``nan`` / ``inf``
    value.  The value here is the forward's and the slope is that one, so
    a loss summed over many slices is not poisoned by a ``nan`` gradient
    the reference does not give.  A divisor of 0 or less warns, as the
    reference's does, in the name of the op ``name`` that was called.
    """
    shape = x_impl.shape
    n = math.prod([shape[a] for a in axes_list] if axes_list else shape)
    if correction == 0 and n > 0:
        return result_impl
    dof = n - correction
    if dof > 0:
        return _scale(result_impl, n / dof)
    warnings.warn(
        f"{name}(): degrees of freedom is <= 0. Correction should be strictly "
        "less than the reduction factor (input numel divided by output numel).",
        UserWarning,
        skip_file_prefixes=(_PACKAGE_DIR,),
    )
    undivided = _scale(result_impl, math.inf)
    if dof == 0 or not result_impl.requires_grad:
        return undivided
    slope = _scale(result_impl, n / dof)
    detached = _C_engine.sub(undivided.data_alias(), slope.data_alias())
    return _C_engine.add(slope, detached)


def _scale(impl: _Impl, factor: float) -> _Impl:
    """``impl * factor``, the factor a 0-d constant of ``impl``'s dtype and device.

    0-d rather than ``impl``'s shape: it broadcasts, and a full-size
    constant cost an allocation and a fill the size of the result.
    """
    factor_t = _C_engine.full([], factor, impl.dtype, impl.device)
    return _C_engine.mul(impl, factor_t)


# ── var / std arguments ──────────────────────────────────────────────────────

_VAR_KEYWORDS = frozenset({"dim", "unbiased", "keepdim", "correction"})


def _as_correction(name: str, value: object) -> float:
    """``value`` as a ``correction``: a real number, a bool as 1 or 0."""
    num = _promotion.as_number(value)
    if isinstance(num, (int, float)):
        return float(num)
    raise TypeError(
        f"{name}(): correction must be a number, got {type(value).__name__}"
    )


def _variance_operands(
    name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> tuple[object, float, bool]:
    """``(dim, correction, keepdim)`` from any spelling of a variance call.

    The one reading of ``var`` / ``std`` / ``var_mean`` / ``std_mean``'s
    arguments.  The reference spellings are ``f(input, dim, unbiased,
    keepdim)``, ``f(input, unbiased)`` and keywords for the rest, so a bool
    where the ``dim`` goes is ``unbiased`` — it used to reduce axis 0 or 1 —
    and a third positional argument is ``unbiased``, never ``keepdim``.
    A number rather than a bool after ``dim`` is a ``correction``, as
    ``var_mean(x, dim, correction)`` has always been spelled.
    """
    unknown = sorted(set(kwargs) - _VAR_KEYWORDS)
    if unknown:
        raise TypeError(f"{name}() got an unexpected keyword argument {unknown[0]!r}")
    if len(args) > 3:
        raise TypeError(
            f"{name}() takes at most 4 positional arguments but "
            f"{len(args) + 1} were given"
        )
    names: tuple[str, ...]
    if args and isinstance(_promotion.as_number(args[0]), bool):
        if len(args) > 1:
            raise TypeError(
                f"{name}(input, unbiased) takes no further positional arguments; "
                f"pass dim and keepdim by keyword"
            )
        names = ("unbiased",)
    else:
        bias = (
            "unbiased"
            if len(args) < 2 or isinstance(_promotion.as_number(args[1]), bool)
            else "correction"
        )
        names = ("dim", bias, "keepdim")[: len(args)]
    bound = dict(kwargs)
    for key, value in zip(names, args, strict=True):
        if key in bound:
            raise TypeError(f"{name}() got multiple values for argument {key!r}")
        bound[key] = value
    unbiased = bound.get("unbiased")
    given = bound.get("correction")
    if unbiased is not None:
        correction = 1.0 if unbiased else 0.0
    else:
        correction = 1.0 if given is None else _as_correction(name, given)
    return bound.get("dim"), correction, bool(bound.get("keepdim", False))


def _corrected_variance(
    x_impl: _Impl, axes: list[int], correction: float, keepdim: bool, name: str
) -> _Impl:
    """The variance over ``axes`` divided by ``max(N - correction, 0)``."""
    biased = _C_engine.var(x_impl, axes, keepdim)
    return _bessel_correct(biased, x_impl, axes, correction, name)


def _variance(
    name: str, x_impl: _Impl, args: tuple[object, ...], kwargs: dict[str, object]
) -> _Impl:
    """The corrected variance a ``var`` / ``std`` call asks for."""
    if (
        not kwargs
        and len(args) < 2
        and not (args and isinstance(_promotion.as_number(args[0]), bool))
    ):
        # ``x.var()`` / ``x.var(dim)``: nothing to parse.
        dim: object = args[0] if args else None
        correction, keepdim = 1.0, False
    else:
        dim, correction, keepdim = _variance_operands(name, args, kwargs)
    axes = _reduction_axes(x_impl, dim, name)
    return _corrected_variance(x_impl, axes, correction, keepdim, name)


def _var_adapter(input: _Impl, *args: object, **kwargs: object) -> _Impl:
    """var(input, dim=None, unbiased=None, keepdim=False, *, correction=1).

    The parameters after ``input`` stay open because the reference's
    overloads bind the second one by type — a bool there is ``unbiased``,
    anything else ``dim`` — see :func:`_variance_operands`.
    """
    return _variance("var", input, args, kwargs)


def _std_adapter(input: _Impl, *args: object, **kwargs: object) -> _Impl:
    """std(input, dim=None, unbiased=None, keepdim=False, *, correction=1).

    The square root of :func:`_var_adapter`'s variance, with a zero
    gradient where that variance is 0 (a constant slice).
    """
    return _zero_safe_root(_variance("std", input, args, kwargs), 2)


def _run_with_dtype(
    x_impl: _Impl,
    dtype: DTypeLike | None,
    call: Callable[[_Impl], _Impl],
    *,
    op: str,
    inexact_only: bool = False,
    floating_only: bool = False,
) -> _Impl:
    """Cast ``x_impl`` to ``dtype`` before ``call`` when ``dtype`` is given.

    Suspends autocast while evaluating ``call`` so an explicit ``dtype=``
    overrides the active autocast policy, and narrows the output back to
    ``target`` when an integer reduction kernel widens narrow ints to int64.
    """
    if dtype is None:
        return call(x_impl)
    target = to_engine_dtype(dtype)
    if floating_only and target not in _FLOATING:
        raise _C_engine.NotImplementedError(
            f"{op}: not implemented for {_ENGINE_TO_DTYPE[target]}"
        )
    if inexact_only and _promotion.is_integral(target):
        raise _C_engine.DtypeMismatch(
            f"{op}(): could not infer output dtype. Optional dtype must be "
            f"either a floating point or complex dtype. Got: {_ENGINE_TO_DTYPE[target]}"
        )
    cast_impl = _C_engine.astype(x_impl, target) if x_impl.dtype != target else x_impl
    with _autocast_suspended(cast_impl.device):
        out = call(cast_impl)
    return _C_engine.astype(out, target) if out.dtype != target else out


def _sum_adapter(
    x_impl: _Impl,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    dtype: DTypeLike | None = None,
) -> _Impl:
    """sum(x, dim=None, keepdim=False, *, dtype=None)."""
    axes = _reduction_axes(x_impl, dim, "sum")
    return _run_with_dtype(
        x_impl, dtype, lambda xi: _C_engine.sum(xi, axes, bool(keepdim)), op="sum"
    )


def _mean_adapter(
    x_impl: _Impl,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    dtype: DTypeLike | None = None,
) -> _Impl:
    """mean(x, dim=None, keepdim=False, *, dtype=None)."""
    axes = _reduction_axes(x_impl, dim, "mean")
    return _run_with_dtype(
        x_impl,
        dtype,
        lambda xi: _C_engine.mean(xi, axes, bool(keepdim)),
        op="mean",
        inexact_only=True,
    )


def _prod_adapter(
    x_impl: _Impl,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
    *,
    dtype: DTypeLike | None = None,
) -> _Impl:
    """prod(x, dim=None, keepdim=False, *, dtype=None)."""
    axes = _reduction_axes(x_impl, dim, "prod")
    return _run_with_dtype(
        x_impl, dtype, lambda xi: _C_engine.prod(xi, axes, bool(keepdim)), op="prod"
    )


def _cumsum_adapter(
    x_impl: _Impl, dim: int = -1, *, dtype: DTypeLike | None = None
) -> _Impl:
    """cumsum(x, dim=-1, *, dtype=None)."""
    axis = operator.index(dim)
    return _run_with_dtype(
        x_impl, dtype, lambda xi: _C_engine.cumsum(xi, axis), op="cumsum"
    )


def _cumprod_adapter(
    x_impl: _Impl, dim: int = -1, *, dtype: DTypeLike | None = None
) -> _Impl:
    """cumprod(x, dim=-1, *, dtype=None)."""
    axis = operator.index(dim)
    return _run_with_dtype(
        x_impl, dtype, lambda xi: _C_engine.cumprod(xi, axis), op="cumprod"
    )


def _softmax_adapter(
    x_impl: _Impl, dim: int = -1, dtype: DTypeLike | None = None
) -> _Impl:
    """softmax(x, dim=-1, dtype=None)."""
    axis = operator.index(dim)
    return _run_with_dtype(
        x_impl,
        dtype,
        lambda xi: _C_engine.softmax(xi, axis),
        op="softmax",
        floating_only=True,
    )


def _softmin_adapter(
    x_impl: _Impl, dim: int = -1, dtype: DTypeLike | None = None
) -> _Impl:
    """softmin(x, dim=-1, dtype=None)."""
    axis = operator.index(dim)
    return _run_with_dtype(
        _C_engine.neg(x_impl),
        dtype,
        lambda xi: _C_engine.softmax(xi, axis),
        op="softmin",
        floating_only=True,
    )


def _log_softmax_adapter(
    x_impl: _Impl, dim: int = -1, dtype: DTypeLike | None = None
) -> _Impl:
    """log_softmax(x, dim=-1, dtype=None)."""
    axis = operator.index(dim)
    return _run_with_dtype(
        x_impl,
        dtype,
        lambda xi: _C_engine.log_softmax(xi, axis),
        op="log_softmax",
        floating_only=True,
    )


def _max_adapter(
    x_impl: _Impl,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> _Impl:
    """max(x, dim=None, keepdim=False)."""
    return _C_engine.max(x_impl, _reduction_axes(x_impl, dim, "max"), bool(keepdim))


def _min_adapter(
    x_impl: _Impl,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> _Impl:
    """min(x, dim=None, keepdim=False)."""
    return _C_engine.min(x_impl, _reduction_axes(x_impl, dim, "min"), bool(keepdim))


def _flat_arg_reduce(
    engine_fn: Callable[[_Impl, int, bool], _Impl],
    name: str,
    x_impl: _Impl,
    keepdim: bool,
) -> _Impl:
    """``dim=None``: the index into the flattened tensor, as documented."""
    shape = list(x_impl.shape)
    if math.prod(shape) == 0:
        raise IndexError(
            f"{name}(): Expected reduction dim to be specified for input.numel() == 0."
        )
    out = engine_fn(_C_engine.reshape(x_impl, [-1]), 0, False)
    return _C_engine.reshape(out, [1] * len(shape)) if keepdim else out


def _argmax_adapter(
    x_impl: _Impl,
    dim: int | None = None,
    keepdim: bool = False,
) -> _Impl:
    """argmax(x, dim=None, keepdim=False)."""
    if dim is None:
        return _flat_arg_reduce(_C_engine.argmax, "argmax", x_impl, bool(keepdim))
    return _C_engine.argmax(x_impl, int(dim), bool(keepdim))


def _argmin_adapter(
    x_impl: _Impl,
    dim: int | None = None,
    keepdim: bool = False,
) -> _Impl:
    """argmin(x, dim=None, keepdim=False)."""
    if dim is None:
        return _flat_arg_reduce(_C_engine.argmin, "argmin", x_impl, bool(keepdim))
    return _C_engine.argmin(x_impl, int(dim), bool(keepdim))


def _truth_count(a_impl: _Impl, axes: list[int], keepdim: bool, falsy: bool) -> _Impl:
    """How many entries along ``axes`` are non-zero (``falsy``: are zero)."""
    if a_impl.dtype in (_C_engine.C64, _C_engine.C128):
        truth = _C_engine.logical_or(
            _C_engine.astype(_C_engine.real(a_impl), _C_engine.Bool),
            _C_engine.astype(_C_engine.imag(a_impl), _C_engine.Bool),
        )
    else:
        truth = _C_engine.astype(a_impl, _C_engine.Bool)
    if falsy:
        truth = _C_engine.logical_not(truth)
    return _C_engine.sum(truth, axes, keepdim)


def _logical_reduce(
    name: str, a_impl: _Impl, dim: int | Sequence[int] | None, keepdim: bool
) -> _Impl:
    """``any`` / ``all`` over ``dim`` — ``all`` is "no entry is zero"."""
    if dim is None and not keepdim:
        return _C_engine.any(a_impl) if name == "any" else _C_engine.all(a_impl)
    axes = _reduction_axes(a_impl, dim, name)
    counted = _truth_count(a_impl, axes, bool(keepdim), falsy=name == "all")
    none = _C_engine.equal(counted, _C_engine.zeros_like(counted))
    return none if name == "all" else _C_engine.logical_not(none)


def _any_adapter(
    a_impl: _Impl, dim: int | Sequence[int] | None = None, keepdim: bool = False
) -> _Impl:
    """any(input, dim=None, keepdim=False) — whether some entry is non-zero."""
    return _logical_reduce("any", a_impl, dim, keepdim)


def _all_adapter(
    a_impl: _Impl, dim: int | Sequence[int] | None = None, keepdim: bool = False
) -> _Impl:
    """all(input, dim=None, keepdim=False) — whether every entry is non-zero."""
    return _logical_reduce("all", a_impl, dim, keepdim)


def _logsumexp_adapter(
    a_impl: _Impl,
    dim: int | Sequence[int] | None = None,
    keepdim: bool = False,
) -> _Impl:
    """logsumexp(a, dim=None, keepdim=False) — accept axis/None like sum/mean."""
    axes = _reduction_axes(a_impl, dim, "logsumexp")
    if _promotion.is_integral(a_impl.dtype):
        a_impl = _C_engine.astype(a_impl, to_engine_dtype(None))
    return _C_engine.logsumexp(a_impl, axes, bool(keepdim))
