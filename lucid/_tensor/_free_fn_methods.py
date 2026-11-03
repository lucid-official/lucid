"""``Tensor`` methods that are ``lucid`` composite functions with ``self`` first.

``x.expm1_()`` existed and ``x.expm1()`` did not: the in-place methods
(:mod:`lucid._tensor._methods`) were written one family at a time, and the
out-of-place method each one's free function implies was left out — 36 of
them, where the reference has every one.  A registry op gets its method
from its own entry (``OpEntry.method_name`` — ``greater`` / ``less`` and
the rest); the composites, which have no entry, are listed in
:data:`FREE_FN_METHODS`.  Each is ``lucid.<name>`` as a method: the same
callee, so the method and the free function cannot disagree, and the same
parameters after the first — names, kinds, defaults, annotations — so
``inspect.signature`` says what the method takes and ``tools/gen_pyi.py``
writes its stub from it.

The free functions resolve, and their annotations evaluate, only once
``import lucid`` has bound everything, so ``lucid/__init__.py`` attaches
these last (:func:`inject_free_fn_methods`) — which is also why ``Tensor``
can be imported here at run time.
"""

import annotationlib
import inspect
from collections.abc import Callable, Sequence

from lucid._C import engine as _C_engine

from lucid._tensor.tensor import Tensor

#: ``x.<name>(...)`` is ``lucid.<name>(x, ...)``: the composites among the
#: out-of-place twins of the in-place methods, and the free functions whose
#: reference has a method form too — ``frexp``, ``equal``, ``amax``, ``stft``,
#: the ``is_*`` predicates — none of which has an in-place form.
FREE_FN_METHODS: tuple[str, ...] = (
    "absolute",
    "acosh",
    "addbmm",
    "addmv",
    "addr",
    "adjoint",
    "allclose",
    "amax",
    "amin",
    "angle",
    "arccosh",
    "arcsinh",
    "arctan2",
    "arctanh",
    "argwhere",
    "asinh",
    "atanh",
    "baddbmm",
    "conj_physical",
    "copysign",
    "corrcoef",
    "count_nonzero",
    "cov",
    "deg2rad",
    "diag",
    "diag_embed",
    "diagonal_scatter",
    "digamma",
    "dist",
    "equal",
    "erfc",
    "expm1",
    "fix",
    "float_power",
    "fmax",
    "fmin",
    "frexp",
    "gcd",
    "ger",
    "heaviside",
    "histogram",
    "i0",
    "is_conj",
    "is_neg",
    "is_nonzero",
    "is_same_size",
    "is_signed",
    "isneginf",
    "isposinf",
    "isreal",
    "istft",
    "lcm",
    "ldexp",
    "lgamma",
    "logaddexp2",
    "logdet",
    "logit",
    "moveaxis",
    "multinomial",
    "multiply",
    "mv",
    "nanmean",
    "nansum",
    "negative",
    "positive",
    "put",
    "rad2deg",
    "resolve_conj",
    "resolve_neg",
    "rot90",
    "select_scatter",
    "sgn",
    "signbit",
    "sinc",
    "slice_scatter",
    "split_with_sizes",
    "stft",
    "subtract",
    "swapdims",
    "unsafe_chunk",
    "unsafe_split",
    "vdot",
    "xlogy",
)


def _as_method(name: str, fn: Callable[..., object]) -> Callable[..., object]:
    """``fn`` as a method: ``self`` in its first parameter's place."""

    def method(self: Tensor, *args: object, **kwargs: object) -> object:
        return fn(self, *args, **kwargs)

    own = inspect.signature(fn, annotation_format=annotationlib.Format.VALUE)
    _, *rest = own.parameters.values()
    self_param = inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    method.__dict__["__signature__"] = own.replace(parameters=[self_param, *rest])
    method.__name__ = name
    method.__qualname__ = f"Tensor.{name}"
    method.__doc__ = (
        f"Method form of :func:`lucid.{name}`: ``x.{name}(...)`` is " f"``lucid.{name}(x, ...)``."
    )
    return method




def _prop_mH(self: Tensor) -> Tensor:
    """Conjugate (Hermitian) transpose of the last two dimensions (`self.adjoint()`)."""
    if self.ndim < 2:
        raise _C_engine.ShapeMismatch(
            f"tensor.mH is only supported on matrices or batches of matrices. Got {self.ndim}-D tensor."
        )
    return self.adjoint()


def _prop_is_cpu(self: Tensor) -> bool:
    """Whether this tensor resides on the CPU device."""
    return self._impl.device == _C_engine.Device.CPU


def _prop_is_meta(_self: Tensor) -> bool:
    """Whether this tensor is a meta tensor (always `False` in Lucid)."""
    return False


def _prop_layout(_self: Tensor) -> object:
    """Memory layout descriptor of this tensor (`lucid.strided`)."""
    from lucid._dtype import strided

    return strided


def inject_free_fn_methods(tensor_cls: type) -> None:
    """Attach every :data:`FREE_FN_METHODS` name to ``tensor_cls``.

    Each is bound to the function ``lucid.<name>`` resolves to now, and
    takes its signature with the annotations evaluated, so it must run once
    both hold: :mod:`lucid` calls it as its last step.
    """
    import lucid

    for name in FREE_FN_METHODS:
        setattr(tensor_cls, name, _as_method(name, getattr(lucid, name)))
    for pname, pfn in (("mH", _prop_mH), ("is_cpu", _prop_is_cpu), ("is_meta", _prop_is_meta), ("layout", _prop_layout)):
        type.__setattr__(tensor_cls, pname, property(pfn))


def _ndimension(self: Tensor) -> int:
    """Alias for :attr:`Tensor.ndim`."""
    return self.ndim


def _nelement(self: Tensor) -> int:
    """Alias for :meth:`Tensor.numel`."""
    return self.numel()


def _polygamma(self: Tensor, n: int) -> Tensor:
    """Compute the `n`-th derivative of the digamma function on `self`."""
    from lucid.special import polygamma

    return polygamma(n, self)


def _parse_sum_to_size_target(
    size: tuple[int | Sequence[int], ...], src: tuple[int, ...]
) -> tuple[int, ...]:
    target = (
        tuple(int(s) for s in size[0])
        if len(size) == 1 and isinstance(size[0], Sequence)
        else tuple(int(s) for s in size if isinstance(s, int))
    )
    lead = len(src) - len(target)
    if lead < 0 or any(ts != ss and ts != 1 for ts, ss in zip(target, src[lead:], strict=True)):
        raise _C_engine.ShapeMismatch(
            f"size {list(target)} is not expandable to size {list(src)}."
        )
    return target


def _sum_to_size(self: Tensor, *size: int | Sequence[int]) -> Tensor:
    """Sum `self` to `size`, which must be broadcastable to `self.shape`."""
    import lucid

    src = tuple(int(d) for d in self.shape)
    target = _parse_sum_to_size_target(size, src)
    if target == src:
        return self
    lead = len(src) - len(target)
    out = lucid.sum(self, dim=list(range(lead)), keepdim=False) if lead > 0 else self
    keep_axes = [
        i for i, (ts, ss) in enumerate(zip(target, src[lead:], strict=True)) if ts == 1 and ss != 1
    ]
    if keep_axes:
        out = lucid.sum(out, dim=keep_axes, keepdim=True)
    return out if tuple(int(d) for d in out.shape) == target else out.reshape(target)


def _fill_diagonal_(self: Tensor, fill_value: float | int | bool | complex, wrap: bool = False) -> Tensor:
    """Fill the main diagonal of a tensor with at least 2 dimensions in place."""
    import lucid

    sh = tuple(int(d) for d in self.shape)
    if len(sh) < 2:
        raise _C_engine.ShapeMismatch("dimensions must be larger than 1")
    if len(sh) > 2 and len(set(sh)) > 1:
        raise _C_engine.ShapeMismatch("all dimensions of input must be of equal length")
    if 0 in sh:
        return self

    def result_of(base: Tensor) -> Tensor:
        flat = base.reshape(-1)
        if len(sh) == 2:
            rows, cols = sh
            step = cols + 1
            if wrap:
                indices = [r * cols + (r % cols) for r in range(rows) if (r % cols) == (r % (cols + 1))]
                # In reference wrap=True on 2D (rows, cols) steps by cols + 1 across flat index up to rows * cols
                indices = list(range(0, rows * cols, step))
            else:
                k = min(rows, cols)
                indices = [i * step for i in range(k)]
        else:
            n = sh[0]
            ndim = len(sh)
            step = sum(n**p for p in range(ndim))
            indices = [i * step for i in range(n)]
        idx = lucid.tensor(indices, dtype=lucid.int64, device=base.device)
        val = lucid.full((len(indices),), fill_value, dtype=base.dtype, device=base.device)
        return lucid.index_put(flat, (idx,), val).reshape(sh)

    from lucid._ops.composite.indexing import _write_inplace

    return _write_inplace(self, "fill_diagonal_", result_of)


def _rebind_view_inplace(self: Tensor, name: str, make_view: Callable[[Tensor], Tensor]) -> Tensor:
    """Rebind `self` to `make_view(self)` in place, bumping version and checking leaf grad."""
    from lucid._tensor._indexing import _take

    if _C_engine.grad_enabled() and self.requires_grad and self.is_leaf:
        raise RuntimeError(
            f"a leaf Variable that requires grad is being used in an in-place operation ({name})."
        )
    view = make_view(self)
    _take(self, view._impl)
    return self


def _t_(self: Tensor) -> Tensor:
    """In-place version of :meth:`Tensor.t`."""
    return _rebind_view_inplace(self, "t_", lambda x: x.t())


def _transpose_(self: Tensor, dim0: int, dim1: int) -> Tensor:
    """In-place version of :meth:`Tensor.transpose`."""
    return _rebind_view_inplace(self, "transpose_", lambda x: x.transpose(dim0, dim1))


def _swapaxes_(self: Tensor, axis0: int, axis1: int) -> Tensor:
    """In-place version of :meth:`Tensor.swapaxes`."""
    return _rebind_view_inplace(self, "swapaxes_", lambda x: x.swapaxes(axis0, axis1))


def _swapdims_(self: Tensor, dim0: int, dim1: int) -> Tensor:
    """In-place version of :meth:`Tensor.swapdims`."""
    return _rebind_view_inplace(self, "swapdims_", lambda x: x.swapdims(dim0, dim1))


def _squeeze_(self: Tensor, dim: int | Sequence[int] | None = None) -> Tensor:
    """In-place version of :meth:`Tensor.squeeze`."""
    return _rebind_view_inplace(
        self,
        "squeeze_",
        lambda x: x.squeeze() if dim is None else x.squeeze(dim if isinstance(dim, int) else list(dim)),
    )


def _unsqueeze_(self: Tensor, dim: int) -> Tensor:
    """In-place version of :meth:`Tensor.unsqueeze`."""
    return _rebind_view_inplace(self, "unsqueeze_", lambda x: x.unsqueeze(dim))


def _exponential_(
    self: Tensor, lambd: float = 1.0, *, generator: _C_engine.Generator | None = None
) -> Tensor:
    """Fill `self` with samples from the exponential distribution `Exp(lambd)`."""
    from lucid._factories.random import _exponential_like

    from lucid._tensor._methods import _fill_drawn

    return _fill_drawn(self, "exponential_", lambda base: _exponential_like(base, lambd, generator))


def _cauchy_(
    self: Tensor,
    median: float = 0.0,
    sigma: float = 1.0,
    *,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    """Fill `self` with samples from `Cauchy(median, sigma)`."""
    from lucid._factories.random import _cauchy_like

    from lucid._tensor._methods import _fill_drawn

    return _fill_drawn(
        self, "cauchy_", lambda base: _cauchy_like(base, median, sigma, generator)
    )


def _log_normal_(
    self: Tensor,
    mean: float = 1.0,
    std: float = 2.0,
    *,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    """Fill `self` with samples from `LogNormal(mean, std)`."""
    from lucid._factories.random import _log_normal_like

    from lucid._tensor._methods import _fill_drawn

    return _fill_drawn(
        self, "log_normal_", lambda base: _log_normal_like(base, mean, std, generator)
    )


def _geometric_(
    self: Tensor, p: float, *, generator: _C_engine.Generator | None = None
) -> Tensor:
    """Fill `self` with samples from `Geometric(p)` in `{1, 2, ...}`."""
    from lucid._factories.random import _geometric_like

    from lucid._tensor._methods import _fill_drawn

    return _fill_drawn(self, "geometric_", lambda base: _geometric_like(base, p, generator))


def _random_(
    self: Tensor,
    from_: int | None = None,
    to: int | None = None,
    *,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    """Fill `self` with discrete uniform samples over `[from_, to - 1]`."""
    from lucid._factories.random import _random_like

    from lucid._tensor._methods import _fill_drawn

    return _fill_drawn(self, "random_", lambda base: _random_like(base, from_, to, generator))

