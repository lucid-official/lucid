"""Statistical composite ops: quantile, cov, corrcoef, cdist, bincount,
multinomial, poisson and the histogram family."""

import math
from typing import Sequence, TYPE_CHECKING

import lucid
from lucid import _promotion
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid._ops._autocast import _autocast_entry
from lucid._types import DeviceLike, DTypeLike
from lucid._ops.composite.indexing import _scatter_add
from lucid._ops._dims import _wrap_dim
from lucid._ops._reduce import (
    _corrected_variance,
    _p_norm,
    _reduction_axes,
    _require_floating,
    _variance_operands,
    _zero_safe_root,
)

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

# ── quantile helpers ──────────────────────────────────────────────────────────


def _quantile_sorted(
    sorted_x: Tensor,
    q_list: Sequence[float],
    dim: int | None,
    keepdim: bool,
    interpolation: str,
    n: int,
) -> list[Tensor]:
    """Core quantile computation on an already-sorted tensor."""
    results = []
    for qi in q_list:
        idx_f = qi * (n - 1)
        lo = int(math.floor(idx_f))
        hi = min(lo + 1, n - 1)
        frac = idx_f - lo

        if dim is None:
            val_lo = sorted_x[lo]
            val_hi = sorted_x[hi]
        else:
            lo_t = lucid.tensor([lo], dtype=lucid.int64, device=sorted_x.device)
            hi_t = lucid.tensor([hi], dtype=lucid.int64, device=sorted_x.device)
            val_lo = sorted_x.index_select(dim, lo_t).squeeze(dim)
            val_hi = sorted_x.index_select(dim, hi_t).squeeze(dim)

        if interpolation == "linear":
            val = val_lo * (1.0 - frac) + val_hi * frac
        elif interpolation == "lower":
            val = val_lo
        elif interpolation == "higher":
            val = val_hi
        elif interpolation == "midpoint":
            val = (val_lo + val_hi) * 0.5
        elif interpolation == "nearest":
            val = (
                sorted_x[round(idx_f)]
                if dim is None
                else sorted_x.index_select(
                    dim,
                    lucid.tensor(
                        [round(idx_f)], dtype=lucid.int64, device=sorted_x.device
                    ),
                ).squeeze(dim)
            )
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation!r}")

        if keepdim and dim is not None:
            val = val.unsqueeze(dim)
        results.append(val)
    return results


def _nanquantile_along(
    sorted_x: Tensor,
    counts: Tensor,
    q_list: Sequence[float],
    dim: int,
    keepdim: bool,
    interpolation: str,
) -> list[Tensor]:
    """Quantile along ``dim`` where each slice has its own valid count.

    ``_quantile_sorted`` takes a single integer ``n``, which is all a
    NaN-free quantile needs.  A NaN-aware one does not have that: slice
    ``i`` may have five valid entries where slice ``j`` has six, and the
    position ``q * (n - 1)`` differs per slice.  Collapsing the counts to
    one number — the mean, rounded down — is what this replaces, and it
    was wrong for *every* slice rather than only the ones holding a NaN,
    since the mean is nobody's count.

    ``counts`` is the per-slice valid count with ``dim`` kept at size 1.
    The sort put every NaN after the numbers, so index ``n - 1`` is the
    largest real value — and a slice with no number reads its first NaN
    at index 0, which is its answer and where its gradient goes.
    """
    zero = lucid.zeros_like(counts)
    last = lucid.maximum(counts - 1.0, zero)

    def _gather(pos: Tensor) -> Tensor:
        return lucid.take_along_dim(sorted_x, pos.to(lucid.int32), dim=dim)

    results = []
    for qi in q_list:
        pos = last * qi
        lo = lucid.floor(pos)
        hi = lucid.minimum(lo + 1.0, last)

        if interpolation == "linear":
            frac = pos - lo
            val = _gather(lo) * (1.0 - frac) + _gather(hi) * frac
        elif interpolation == "lower":
            val = _gather(lo)
        elif interpolation == "higher":
            val = _gather(hi)
        elif interpolation == "midpoint":
            val = (_gather(lo) + _gather(hi)) * 0.5
        elif interpolation == "nearest":
            val = _gather(lucid.round(pos))
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation!r}")
        results.append(val if keepdim else val.squeeze(dim))
    return results


def _parse_q(q: float | Sequence[float] | Tensor) -> tuple[list[float], bool]:
    """Return (q_list of floats, scalar_flag)."""
    if isinstance(q, lucid.Tensor):
        if q.ndim == 0:
            return [float(q.item())], True
        return [float(q[i].item()) for i in range(q.shape[0])], False
    elif hasattr(q, "__iter__"):
        return [float(v) for v in q], False
    else:
        return [float(q)], True


# ── public API ────────────────────────────────────────────────────────────────


def quantile(
    input: Tensor,
    q: float | Sequence[float] | Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    interpolation: str = "linear",
) -> Tensor:
    """Compute the *q*-th quantile of ``input``.

    Parameters
    ----------
    input : Tensor
        Real-valued input.
    q : float, sequence of float, or Tensor
        Quantile(s) in :math:`[0, 1]`.  A single ``q`` returns a tensor
        without the leading quantile axis; a sequence/tensor adds a
        leading axis of length ``len(q)``.
    dim : int, optional
        Axis along which to reduce.  ``None`` (default) reduces over
        the flattened input.
    keepdim : bool, optional
        Retain the reduced axis as a size-1 dimension.  Default
        ``False``.
    interpolation : str, optional
        Rule for interpolating between the two samples bracketing the
        quantile.  One of ``'linear'`` (default), ``'lower'``,
        ``'higher'``, ``'midpoint'``, ``'nearest'``.

    Returns
    -------
    Tensor
        Same dtype as ``input``.  Shape depends on ``q`` / ``dim`` /
        ``keepdim``.

    See Also
    --------
    nanquantile : NaN-aware sibling.
    """
    if dim is not None:
        dim = _wrap_dim(dim, input.ndim, "quantile")
    if input.numel() == 0:
        raise _C_engine.InvalidArgument("quantile() input tensor must be non-empty")
    q_list, scalar_q = _parse_q(q)
    with _autocast_entry("quantile", _unwrap(input)) as (xi,):
        input = _wrap(xi)

        # A NaN anywhere in the slice poisons its quantile — that is the whole
        # difference from :func:`nanquantile`, and it has to be stated rather
        # than left to fall out of the arithmetic.  Sorting puts NaN at the end,
        # so it only reached the answer when the interpolation position happened
        # to land beside one (``nan * 0.0`` is still ``NaN``).  Anywhere else it
        # dropped out silently: 50 samples with three NaNs returned a perfectly
        # ordinary median where the reference and NumPy both return NaN.
        if dim is None:
            flat = input.reshape(-1)
            sorted_x = lucid.sort(flat).values
            n = flat.shape[0]
            results = _quantile_sorted(
                sorted_x, q_list, None, keepdim, interpolation, n
            )
            nans = lucid.isnan(flat).sum()
        else:
            if input.ndim == 0:
                # A 0-d tensor's notional axis is the whole of it, with or
                # without ``keepdim``: the quantile of its one-element view.
                return quantile(
                    input.reshape(1), q, 0, False, interpolation=interpolation
                )
            sorted_x = lucid.sort(input, dim).values
            n = input.shape[dim]
            results = _quantile_sorted(sorted_x, q_list, dim, keepdim, interpolation, n)
            nans = lucid.isnan(input).sum(dim=dim, keepdim=keepdim)
        # An integer input holds no NaN, and an ``interpolation="lower"`` result
        # of one is an integer tensor that could not hold the NaN fill anyway.
        if lucid.is_floating_point(input):
            results = [
                lucid.where(nans == 0, r, lucid.full_like(r, math.nan)) for r in results
            ]

        return _pack_quantile_results(results, scalar_q, input.ndim, dim, keepdim)


def _pack_quantile_results(
    results: list[Tensor],
    scalar_q: bool,
    ndim: int,
    dim: int | None,
    keepdim: bool,
) -> Tensor:
    """Stack quantile slices and restore ``(1,) * ndim`` when ``dim=None, keepdim=True``."""
    if dim is None and keepdim and ndim > 0:
        ones = (1,) * ndim
        results = [r.reshape(ones) for r in results]
    return results[0] if scalar_q else lucid.stack(results)


def nanquantile(
    input: Tensor,
    q: float | Sequence[float] | Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    *,
    interpolation: str = "linear",
) -> Tensor:
    """NaN-aware quantile — same interface as :func:`quantile`, but
    every ``NaN`` is excluded from the sort and the per-axis quantile
    is computed over the remaining values.

    Parameters
    ----------
    input : Tensor
        Real-valued input.  ``NaN`` entries are masked out before
        sorting.
    q : float, sequence of float, or Tensor
        Quantile(s) in :math:`[0, 1]`.  A single ``q`` returns a tensor
        without the leading quantile axis; a sequence/tensor adds a
        leading axis of length ``len(q)``.
    dim : int, optional
        Axis along which to reduce.  ``None`` (default) reduces over
        the flattened input.
    keepdim : bool, optional
        Retain the reduced axis as a size-1 dimension.  Default
        ``False``.
    interpolation : str, optional
        Interpolation rule between two surrounding samples.  Currently
        ``"linear"`` is supported (default).

    Returns
    -------
    Tensor
        Same dtype as ``input``; shape depends on ``q`` / ``dim`` /
        ``keepdim``.  Returns ``NaN`` for slices that contain only
        ``NaN`` values.

    See Also
    --------
    quantile : NaN-unaware version (a single ``NaN`` poisons the slice).
    """
    if dim is not None:
        dim = _wrap_dim(dim, input.ndim, "nanquantile")
    if input.numel() == 0:
        raise _C_engine.InvalidArgument("quantile() input tensor must be non-empty")
    if not lucid.is_floating_point(input):
        # No NaN to skip: the plain quantile is the answer.
        return quantile(input, q, dim, keepdim, interpolation=interpolation)
    q_list, scalar_q = _parse_q(q)
    with _autocast_entry("nanquantile", _unwrap(input)) as (xi,):
        input = _wrap(xi)
        # The whole tensor is the one slice of its flattening, so it takes the
        # per-slice path: the count stays a tensor on the device, and an all-NaN
        # input reads its first NaN with the gradient attached.
        x, along = (
            (input.reshape(-1), 0) if dim is None or input.ndim == 0 else (input, dim)
        )
        # The engine's sort already puts every NaN after the numbers.
        sorted_x = lucid.sort(x, along).values
        non_nan_count = (~lucid.isnan(x)).sum(dim=along, keepdim=True).to(x.dtype)
        results = _nanquantile_along(
            sorted_x,
            non_nan_count,
            q_list,
            along,
            keepdim and dim is not None and input.ndim > 0,
            interpolation,
        )

        return _pack_quantile_results(results, scalar_q, input.ndim, dim, keepdim)


# ── covariance & correlation ──────────────────────────────────────────────────


def _cov_weights(
    n_obs: int,
    dtype: DTypeLike,
    device: DeviceLike,
    fweights: Tensor | None,
    aweights: Tensor | None,
) -> tuple[Tensor, Tensor | None]:
    """Validate ``fweights`` and ``aweights`` for :func:`cov`."""
    w = lucid.ones((1, n_obs), dtype=dtype, device=device)
    if fweights is not None:
        if (
            fweights.is_floating_point()
            or fweights.is_complex()
            or fweights.dtype == lucid.bool_
        ):
            raise _C_engine.InvalidArgument(
                f"cov(): expected fweights to have integral dtype but got {fweights.dtype}"
            )
        if fweights.ndim > 1 or int(fweights.numel()) != n_obs:
            raise _C_engine.InvalidArgument(
                "cov(): fweights must be 1-D with length matching observation count"
            )
        if bool((fweights < 0).any().item()):
            raise _C_engine.InvalidArgument("cov(): fweights cannot be negative")
        w = w * fweights.to(dtype=dtype, device=device).reshape(1, n_obs)
    aw: Tensor | None = None
    if aweights is not None:
        if aweights.is_complex() or aweights.dtype == lucid.bool_:
            raise _C_engine.InvalidArgument(
                f"cov(): expected aweights to be real numeric but got {aweights.dtype}"
            )
        if aweights.ndim > 1 or int(aweights.numel()) != n_obs:
            raise _C_engine.InvalidArgument(
                "cov(): aweights must be 1-D with length matching observation count"
            )
        if bool((aweights < 0).any().item()):
            raise _C_engine.InvalidArgument("cov(): aweights cannot be negative")
        aw = aweights.to(dtype=dtype, device=device).reshape(1, n_obs)
        w = w * aw
    return w, aw


def cov(
    input: Tensor,
    *,
    correction: int = 1,
    fweights: Tensor | None = None,
    aweights: Tensor | None = None,
) -> Tensor:
    r"""Covariance matrix of ``input``.

    Parameters
    ----------
    input : Tensor
        Shape ``(N,)`` (single-variable degenerate case) or ``(N, M)``
        — *N* variables observed at *M* time points.
    correction : int, optional
        Bessel correction; the divisor is ``max(1, M - correction)``.
        Default ``1`` (unbiased sample covariance); pass ``0`` for the
        biased population covariance.
    fweights : Tensor, optional
        1-D integer tensor of observation repeat counts (length ``M``).
    aweights : Tensor, optional
        1-D numeric tensor of relative observation weights (length ``M``).

    Returns
    -------
    Tensor
        Shape ``(N, N)``.  Entry :math:`(i, j)` =
        :math:`\frac{1}{M - \text{correction}}\sum_k (x_{i,k} - \bar x_i)(x_{j,k} - \bar x_j)`.

    See Also
    --------
    corrcoef : normalised counterpart (Pearson correlation).
    """
    if input.ndim > 2:
        raise _C_engine.InvalidArgument(
            f"cov(): expected input to have two or fewer dimensions "
            f"but got an input with {input.ndim} dimensions"
        )
    if input.ndim == 0:
        x = input.reshape(1, 1)
    elif input.ndim == 1:
        x = input.unsqueeze(0)  # (1, M)
    else:
        x = input  # (N, M)
    if not lucid.is_floating_point(x) and not lucid.is_complex(x):
        x = x.to(lucid.get_default_dtype())

    n_obs = int(x.shape[1])
    if fweights is None and aweights is None:
        mean = x.sum(dim=1, keepdim=True) * (1.0 / n_obs)  # (N, 1)
        xc = x - mean  # (N, M)
        denom = float(max(1, n_obs - correction))
        return lucid.matmul(xc, xc.swapaxes(-1, -2)) * (1.0 / denom)  # (N, N)

    w, aw = _cov_weights(n_obs, x.dtype, x.device, fweights, aweights)
    w_sum = w.sum()
    mean = (x * w).sum(dim=1, keepdim=True) / w_sum
    xc = x - mean
    if aw is None:
        denom_t = w_sum - float(correction)
    elif correction == 0:
        denom_t = w_sum
    else:
        denom_t = w_sum - float(correction) * (w * aw).sum() / w_sum
    return lucid.matmul(xc * w, xc.swapaxes(-1, -2)) / denom_t


def corrcoef(input: Tensor) -> Tensor:
    r"""Pearson correlation-coefficient matrix.

    Composite of :func:`cov` divided by the outer product of the
    per-variable standard deviations:

    .. math::

        R_{ij} = \frac{C_{ij}}{\sqrt{C_{ii}} \sqrt{C_{jj}}}.

    Diagonal entries are ``1.0`` (up to floating-point); off-diagonal
    entries fall in :math:`[-1, 1]`.

    Parameters
    ----------
    input : Tensor
        Shape ``(N,)`` for a single-variable degenerate case, or
        ``(N, M)`` for ``N`` variables observed at ``M`` time points.

    Returns
    -------
    Tensor
        Shape ``(N, N)`` Pearson correlation matrix.

    See Also
    --------
    cov : raw covariance matrix without the std-normalisation.
    """
    c = cov(input, correction=1)  # (N, N)
    d = lucid.sqrt(lucid.diagonal(c))  # (N,) — std deviations
    # Outer product: d_col * d_row gives the normalisation matrix
    d_col = d.reshape(-1, 1)  # (N, 1)
    d_row = d.reshape(1, -1)  # (1, N)
    return c / (d_col * d_row)


# ── pairwise distances ────────────────────────────────────────────────────────

_CDIST_MODES: frozenset[str] = frozenset(
    {
        "use_mm_for_euclid_dist_if_necessary",
        "use_mm_for_euclid_dist",
        "donot_use_mm_for_euclid_dist",
    }
)


def cdist(
    x1: Tensor,
    x2: Tensor,
    p: float = 2.0,
    compute_mode: str = "use_mm_for_euclid_dist_if_necessary",
) -> Tensor:
    r"""Pairwise :math:`L_p` distance matrix between rows of ``x1`` and ``x2``.

    Specialised paths exist for ``p`` in :math:`\{1, 2, \infty, 0\}`;
    other ``p`` values fall back to the general
    :math:`(\sum_k |a_k - b_k|^p)^{1/p}` formula.  For ``p=2`` the
    numerically stable expansion
    :math:`\|a-b\|^2 = \|a\|^2 + \|b\|^2 - 2 a \cdot b^\top` is used
    so large coordinates don't lose precision.

    Parameters
    ----------
    x1 : Tensor
        Shape ``(..., P, M)``.
    x2 : Tensor
        Shape ``(..., R, M)``.  Leading batch dimensions are
        broadcast-compatible with ``x1``.
    p : float, optional
        :math:`L_p` exponent.  Default ``2.0`` (Euclidean).  Use
        ``float("inf")`` for Chebyshev, ``0.0`` for the Hamming-style
        non-zero count.
    compute_mode : str, optional
        Controls whether the matrix-multiplication identity is used when
        ``p == 2.0``.  One of ``'use_mm_for_euclid_dist_if_necessary'``
        (default), ``'use_mm_for_euclid_dist'``, or
        ``'donot_use_mm_for_euclid_dist'``.

    Returns
    -------
    Tensor
        Shape ``(..., P, R)``.  Entry :math:`(\ldots, i, j)` =
        :math:`\|x_1[\ldots, i, :] - x_2[\ldots, j, :]\|_p`.

    Raises
    ------
    DtypeMismatch
        If ``x1`` or ``x2`` is not a real floating-point tensor, as the
        reference refuses integer, bool and complex inputs — a ``TypeError``
        and a ``RuntimeError``.
    """
    if compute_mode not in _CDIST_MODES:
        raise _C_engine.InvalidArgument(
            f"possiblemodes: 'use_mm_for_euclid_dist_if_necessary', "
            f"'use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist', "
            f"was given: {compute_mode}"
        )
    for operand in (x1, x2):
        _require_floating(_unwrap(operand), "cdist")
    with _autocast_entry("cdist", _unwrap(x1), _unwrap(x2)) as (x1i, x2i):
        x1, x2 = _wrap(x1i), _wrap(x2i)
        if p == 2.0 and compute_mode != "donot_use_mm_for_euclid_dist":
            # ||a - b||² = ||a||² + ||b||² − 2 a·bᵀ, which never materialises
            # the (..., P, R, M) difference.
            x1_sq = (x1 * x1).sum(dim=-1, keepdim=True)  # (..., P, 1)
            x2_sq = (x2 * x2).sum(dim=-1, keepdim=True)  # (..., R, 1)
            cross = lucid.matmul(x1, x2.swapaxes(-1, -2))  # (..., P, R)
            sq = x1_sq + x2_sq.swapaxes(-1, -2) - 2.0 * cross
            sq = lucid.clamp(sq, 0.0, float("inf"))  # clamp numerical noise
            # Coincident points — every diagonal entry of cdist(x, x) — sit at
            # the root's infinite slope; the shared zero-safe root gives them
            # the 0 subgradient.
            return _wrap(_zero_safe_root(_unwrap(sq), 2))
        diff = _unwrap(x1.unsqueeze(-2) - x2.unsqueeze(-3))  # (..., P, R, M)
        last = [len(diff.shape) - 1]
        if p == 0.0:
            # Count the non-zero differences; NaN is not one.
            abs_diff = _C_engine.abs(diff)
            counted = _C_engine.logical_and(
                _C_engine.not_equal(abs_diff, _C_engine.zeros_like(abs_diff)),
                _C_engine.logical_not(_C_engine.isnan(abs_diff)),
            )
            return _wrap(
                _C_engine.astype(_C_engine.sum(counted, last, False), diff.dtype)
            )
        # Every other order — 1, inf, and the general p whose root used to give
        # coincident points a nan gradient — is the shared vector p-norm.
        return _wrap(_p_norm(diff, float(p), last, False))


# ── bincount ──────────────────────────────────────────────────────────────────


def bincount(
    input: Tensor,
    weights: Tensor | None = None,
    minlength: int = 0,
) -> Tensor:
    """Count occurrences of each integer value in ``input``.

    One ``scatter_add`` into a zero tensor of the output's length.  That
    length depends on the largest value, so it costs one read of the
    input's minimum and maximum back to the host — the device round trip
    every data-dependent output size takes on Metal.

    Parameters
    ----------
    input : Tensor
        1-D tensor of non-negative integers.
    weights : Tensor, optional
        1-D, as long as ``input``.  When supplied, ``output[i]`` is the
        *sum of weights* for entries with value ``i`` (rather than the
        plain count).  Default ``None``.
    minlength : int, optional
        Floor for the output length.  The output has length
        ``max(input.max() + 1, minlength)``.  Default ``0``.

    Returns
    -------
    Tensor
        1-D tensor on ``input``'s device: ``int64`` counts without
        ``weights``; with them, the weights' dtype for ``float32`` /
        ``float64`` weights and ``float64`` for any other (``float32`` on
        Metal, which holds no ``float64``).

    Raises
    ------
    TypeError
        ``input`` is not an integer tensor.
    ValueError
        ``input`` is not 1-D or holds a negative value, ``minlength`` is
        negative, or ``weights`` is not 1-D and as long as ``input``.

    Examples
    --------
    >>> import lucid
    >>> lucid.bincount(lucid.tensor([0, 1, 1, 3]))
    tensor([1, 2, 0, 1], dtype=lucid.int64)
    >>> lucid.bincount(lucid.tensor([0, 1, 1]), weights=lucid.tensor([1.0, 2.0, 3.0]))
    tensor([1., 5.])
    """
    if input.is_floating_point() or input.is_complex() or input.dtype == lucid.bool_:
        raise TypeError(f"bincount: input must be an integer tensor, got {input.dtype}")
    if input.ndim != 1:
        raise ValueError(f"bincount: input must be 1-D, got shape {tuple(input.shape)}")
    if minlength < 0:
        raise ValueError(f"bincount: minlength must be non-negative, got {minlength}")
    n = int(input.shape[0])
    if weights is not None and tuple(weights.shape) != (n,):
        raise ValueError(
            f"bincount: weights must be 1-D and as long as input ({n}), got "
            f"shape {tuple(weights.shape)}"
        )
    length = minlength
    if n:
        lowest, highest = _extremes(input)
        if lowest < 0:
            raise ValueError(
                f"bincount: input must hold non-negative integers, got {lowest}"
            )
        length = max(int(highest) + 1, minlength)
    device = input.device
    idx = input if input.dtype in (lucid.int32, lucid.int64) else input.to(lucid.int64)
    if weights is None:
        counts = lucid.zeros(length, dtype=lucid.int64, device=device)
        ones = lucid.ones(n, dtype=lucid.int64, device=device)
        return _scatter_add(counts, 0, idx, ones)
    dtype = _bincount_dtype(weights)
    totals = lucid.zeros(length, dtype=dtype, device=device)
    return _scatter_add(totals, 0, idx, weights.to(dtype))


def _extremes(values: Tensor) -> tuple[float, float]:
    """``values``' minimum and maximum, in one read back to the host."""
    pair = lucid.stack([values.min(), values.max()]).tolist()
    assert isinstance(pair, list)
    lowest, highest = pair
    assert isinstance(lowest, (int, float)) and isinstance(highest, (int, float))
    return float(lowest), float(highest)


def _bincount_dtype(weights: Tensor) -> DTypeLike:
    """The dtype weighted bin totals are summed in, as the reference sums them."""
    if weights.dtype in (lucid.float32, lucid.float64):
        return weights.dtype
    return lucid.float64 if weights.device.type == "cpu" else lucid.float32


# Elements of the (batch, samples, categories) comparison multinomial makes
# at once when drawing with replacement.
_MULTINOMIAL_CHUNK = 1 << 24


def _validate_multinomial_weights(
    rows: Tensor, num_samples: int, replacement: bool
) -> None:
    """Validate that ``rows`` contains finite, non-negative weights with sufficient support."""
    if rows._impl.is_phantom:
        return
    # The checks only decide whether to raise; nothing read here reaches
    # the draws, so a trace does not count them as data dependencies.
    # Imported here: ``lucid.compile`` imports the tensor this module builds on.
    from lucid.compile import _host_check

    with _host_check():
        lowest = float(rows.min().item())
        total = float(rows.sum().item())
        positive = int((rows > 0).sum(dim=-1).min().item())
    if lowest < 0 or not math.isfinite(total):
        raise ValueError("multinomial: weights must be finite and non-negative")
    if positive == 0:
        raise ValueError("multinomial: every distribution needs a positive weight")
    if not replacement and positive < num_samples:
        raise ValueError(
            f"multinomial: cannot draw {num_samples} samples without replacement "
            f"from a distribution with {positive} positive weights"
        )


def multinomial(
    input: Tensor,
    num_samples: int,
    replacement: bool = False,
    *,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    """Draw ``num_samples`` indices from the categorical distribution defined by ``input``.

    Non-differentiable — gradients of ``input`` are not propagated
    through the sampling step.  Pass an explicit ``generator`` for an
    isolated PRNG stream; without one, draws come from Lucid's
    :func:`manual_seed`-controlled default Philox generator.  The draws
    are made on ``input``'s device.

    With replacement, each draw inverts the cumulative weights at a
    uniform sample.  Without replacement, the categories are ranked by
    log-weight plus Gumbel noise and the first ``num_samples`` are kept,
    which has the same law as drawing one at a time and removing each
    pick.

    Parameters
    ----------
    input : Tensor
        Probability weights.  1-D ``(K,)`` for a single categorical
        distribution, or 2-D ``(B, K)`` for a batch of ``B``
        distributions.  Weights are renormalised internally; they need
        not sum to 1.
    num_samples : int
        Number of draws per distribution.
    replacement : bool, optional
        When ``True`` each draw is independent.  When ``False``
        (default) each chosen index is removed before the next draw
        within the same row.
    generator : engine.Generator, optional
        Override Lucid's default Philox generator with an isolated
        stream.

    Returns
    -------
    Tensor
        Integer tensor.  Shape ``(num_samples,)`` for 1-D input,
        ``(B, num_samples)`` for 2-D input.
    """

    if input.ndim not in (1, 2):
        raise ValueError(f"multinomial: input must be 1-D or 2-D, got {input.ndim}-D")
    k = int(input.shape[-1])
    if num_samples < 1:
        raise ValueError(
            f"multinomial: num_samples must be positive, got {num_samples}"
        )
    if not replacement and num_samples > k:
        raise ValueError(
            f"multinomial: cannot draw {num_samples} samples without replacement "
            f"from {k} categories"
        )
    rows = input.detach().reshape(-1, k)
    if rows.dtype not in (lucid.float32, lucid.float64):
        rows = rows.float()
    _validate_multinomial_weights(rows, num_samples, replacement)
    batch = int(rows.shape[0])

    if replacement:
        # Inverse CDF: a uniform draw belongs to the category whose interval
        # of the cumulative weights holds it, which is the count of cumulative
        # weights at or below it.  Dividing by the last one makes that exactly
        # 1.0, so a draw in [0, 1) never counts past the last category, and a
        # zero weight's empty interval is never hit.
        cdf = rows.cumsum(dim=-1)
        cdf = cdf / cdf[:, -1:]
        u = lucid.rand(
            batch,
            num_samples,
            dtype=rows.dtype,
            device=rows.device,
            generator=generator,
        )
        # The comparison is (batch, samples, k); chunking the samples keeps it
        # bounded however many draws are asked for.
        step = max(1, _MULTINOMIAL_CHUNK // (batch * k))
        parts = [
            (cdf.unsqueeze(1) <= u[:, s : s + step].unsqueeze(-1)).sum(dim=-1)
            for s in range(0, num_samples, step)
        ]
        idx = parts[0] if len(parts) == 1 else lucid.cat(parts, dim=1)
    else:
        # Gumbel-top-k: ranking the categories by log-weight plus independent
        # Gumbel noise and keeping the first num_samples draws them without
        # replacement, in the order one-at-a-time draws would pick them.  A
        # zero weight's key is -inf, so it is never kept.
        u = lucid.rand(
            batch, k, dtype=rows.dtype, device=rows.device, generator=generator
        )
        gumbel = -(-u.clamp(min=1e-30).log()).log()
        _, idx = lucid.topk(rows.log() + gumbel, num_samples, dim=-1)

    idx = idx.long()
    return idx.reshape(num_samples) if input.ndim == 1 else idx


def poisson(
    input: Tensor,
    generator: _C_engine.Generator | None = None,
) -> Tensor:
    r"""Sample element-wise from :math:`\mathrm{Poisson}(\text{rate} = \text{input}[i])`.

    Parameters
    ----------
    input : Tensor
        Non-negative rates :math:`\lambda_i` per output position.  Any
        shape.
    generator : engine.Generator, optional
        Override Lucid's default Philox generator with an isolated
        stream.  When ``None`` (default), draws are controlled by
        :func:`manual_seed`.

    Returns
    -------
    Tensor
        ``int64`` tensor with the same shape as ``input``.  Zero rates
        always return zero.

    Raises
    ------
    ValueError
        If any element of ``input`` is negative.

    Notes
    -----
    Two-branch implementation:

    * **Knuth's multiplication method** for ``rate < 30`` — exact but
      runtime grows with the rate.
    * **Normal approximation** :math:`\mathrm{Pois}(\lambda) \approx
      \lfloor \mathcal{N}(\lambda, \sqrt{\lambda}) + 0.5 \rfloor`
      for ``rate >= 30`` — bias < 0.05 standard deviations, orders of
      magnitude faster than Knuth in the tail.
    """
    from lucid._factories.random import _default_stream

    g: _C_engine.Generator = generator if generator is not None else _default_stream()

    flat: Tensor = input.reshape(-1)
    n: int = int(flat.numel())
    out: list[int] = []
    _SMALL_RATE_CUTOFF: float = 30.0
    _TWO_PI: float = 2.0 * math.pi

    for i in range(n):
        r: float = float(flat[i].item())
        if r < 0.0:
            raise ValueError(
                f"poisson: rate must be non-negative, got {r} at flat index {i}"
            )
        if r == 0.0:
            out.append(0)
            continue

        if r < _SMALL_RATE_CUTOFF:
            # Knuth: keep multiplying U(0,1) draws until product ≤ e^{-rate}.
            L: float = math.exp(-r)
            k: int = 0
            p: float = 1.0
            while True:
                k += 1
                u: float = g.next_uniform_float()
                # Guard against u==0 (Philox 24-bit mantissa never returns
                # exactly 0, but be defensive against future RNG changes).
                if u <= 0.0:
                    u = 1e-300
                p *= u
                if p <= L:
                    break
            out.append(k - 1)
        else:
            # Normal approximation: Box-Muller for one std-normal, scale
            # by √rate, shift by rate, round to nearest non-negative int.
            u1: float = g.next_uniform_float()
            u2: float = g.next_uniform_float()
            if u1 <= 0.0:
                u1 = 1e-30
            z: float = math.sqrt(-2.0 * math.log(u1)) * math.cos(_TWO_PI * u2)
            s: int = int(math.floor(r + math.sqrt(r) * z + 0.5))
            out.append(max(0, s))

    # Knuth sampling runs on the host, but the result must ride the input's
    # device — a CPU return device-mismatched every downstream op (e.g.
    # ``Poisson.log_prob`` combining the sample with a Metal ``rate``).
    return lucid.tensor(out, dtype=lucid.int64, device=input.device).reshape(
        input.shape
    )


def histogram2d(
    x: Tensor,
    y: Tensor,
    bins: int | tuple[int, int] = 10,
    range: tuple[tuple[float, float], tuple[float, float]] | None = None,
    density: bool = False,
) -> tuple[Tensor, Tensor, Tensor]:
    """Joint 2-D histogram of paired observations ``(x[i], y[i])``.

    Parameters
    ----------
    x : Tensor
        1-D x-coordinates.
    y : Tensor
        1-D y-coordinates; must have the same length as ``x``.
    bins : int or tuple[int, int], optional
        Bin count.  A single int applies to both axes; a 2-tuple sets
        each axis independently.  Default ``10`` (i.e. 10×10).
    range : tuple of (float, float), optional
        ``((x_lo, x_hi), (y_lo, y_hi))``.  When ``None`` (default)
        each axis uses its own ``(min, max)``.
    density : bool, optional
        When ``True`` divide by total count × bin area to yield a
        probability density (sums × bin-area = 1).  Default ``False``.

    Returns
    -------
    tuple of Tensor
        ``(counts, x_edges, y_edges)``.  ``counts`` shape
        ``(bins_x, bins_y)`` int64 (or float64 when ``density``).
        ``x_edges`` length ``bins_x + 1``, ``y_edges`` length
        ``bins_y + 1``.
    """
    from lucid._C import engine as _C_engine
    from lucid._dispatch import _unwrap, _wrap

    if isinstance(bins := _promotion.as_number(bins), int):
        nbins_a, nbins_b = bins, bins
    else:
        nbins_a, nbins_b = bins
    if range is None:
        x_flat = x.reshape(-1)
        y_flat = y.reshape(-1)
        lo_a, hi_a = float(x_flat.min().item()), float(x_flat.max().item())
        lo_b, hi_b = float(y_flat.min().item()), float(y_flat.max().item())
        if lo_a == hi_a:
            hi_a = lo_a + 1.0
        if lo_b == hi_b:
            hi_b = lo_b + 1.0
    else:
        (lo_a, hi_a), (lo_b, hi_b) = range
        lo_a, hi_a = float(lo_a), float(hi_a)
        lo_b, hi_b = float(lo_b), float(hi_b)

    counts_impl, edges_impl = _C_engine.histogram2d(
        _unwrap(x),
        _unwrap(y),
        int(nbins_a),
        int(nbins_b),
        lo_a,
        hi_a,
        lo_b,
        hi_b,
        density,
    )
    # Engine packs both edge arrays into a single (a+b+2,)-length tensor;
    # split it back into the ``(bins_a + 1, bins_b + 1)`` pair the user
    # expects from a NumPy/SciPy-like API.
    counts = _wrap(counts_impl)
    edges = _wrap(edges_impl)
    x_edges = edges[: nbins_a + 1]
    y_edges = edges[nbins_a + 1 :]
    return counts, x_edges, y_edges


def _histogramdd_weighted(
    input: Tensor,
    weight: Tensor,
    bins_seq: list[int],
    edges_list: list[Tensor],
    density: bool,
) -> Tensor:
    """Compute weighted N-D histogram counts over ``edges_list``."""
    n_samples, d_axes = int(input.shape[0]), int(input.shape[1])
    if tuple(weight.shape) != (n_samples,):
        raise ValueError(
            f"histogramdd: weight shape {tuple(weight.shape)} must match "
            f"sample count ({n_samples},)"
        )
    dtype = input.dtype if input.is_floating_point() else lucid.get_default_dtype()
    device = input.device
    flat_idx = lucid.zeros(n_samples, dtype=lucid.int64, device=device)
    inside = lucid.ones(n_samples, dtype=lucid.bool, device=device)
    total_bins = 1
    for d in range(d_axes):
        nb = bins_seq[d]
        total_bins *= nb
        col = input[:, d].to(dtype)
        edges = edges_list[d].to(dtype=dtype, device=device)
        inside = inside & (col >= edges[0]) & (col <= edges[-1])
        b_d = _uniform_bucket(col, edges, nb).clamp(0, nb - 1)
        flat_idx = flat_idx * nb + b_d
    mass = lucid.where(
        inside, weight.to(dtype), lucid.zeros(n_samples, dtype=dtype, device=device)
    )
    counts_flat = _scatter_add(
        lucid.zeros(total_bins, dtype=dtype, device=device), 0, flat_idx, mass
    )
    counts = counts_flat.reshape(bins_seq)
    if density:
        counts = counts / counts.sum()
        for d in range(d_axes):
            edges = edges_list[d].to(dtype=dtype, device=device)
            widths = edges[1:] - edges[:-1]
            shape = [bins_seq[d] if i == d else 1 for i in range(d_axes)]
            counts = counts / widths.reshape(shape)
    return counts


def histogramdd(
    input: Tensor,
    bins: int | Sequence[int] = 10,
    range: Sequence[tuple[float, float]] | None = None,
    weight: Tensor | None = None,
    density: bool = False,
) -> tuple[Tensor, list[Tensor]]:
    """N-dimensional histogram for arbitrary sample dimension.

    Parameters
    ----------
    input : Tensor
        Shape ``(N, D)`` — :math:`N` samples in :math:`D`
        dimensions.  Other ranks raise.
    bins : int or sequence of int, optional
        Bin count per axis.  A single int applies to every axis; a
        sequence of length :math:`D` sets each axis independently.
        Default ``10``.
    range : sequence of (float, float), optional
        Per-axis ``(lo, hi)`` ranges.  When ``None`` (default) each
        axis uses its own ``(min, max)`` of the samples.
    weight : Tensor, optional
        1-D tensor of shape ``(N,)`` holding per-sample weights.
    density : bool, optional
        When ``True`` divide by total count × bin volume to yield a
        density that integrates to 1.  Default ``False``.

    Returns
    -------
    tuple
        ``(counts, edges)``.  ``counts`` shape
        ``(bins_0, …, bins_{D-1})``; ``edges`` is a length-:math:`D`
        list of 1-D tensors of size ``bins_i + 1``.
    """
    from lucid._C import engine as _C_engine
    from lucid._dispatch import _unwrap, _wrap

    if input.ndim != 2:
        raise ValueError(
            f"histogramdd: expected (N, D) input, got shape {tuple(input.shape)}"
        )
    D: int = int(input.shape[1])
    if isinstance(bins := _promotion.as_number(bins), int):
        bins_seq: list[int] = [int(bins)] * D
    else:
        bins_seq = [int(b) for b in bins]
        if len(bins_seq) != D:
            raise ValueError(
                f"histogramdd: bins must be an int or a length-{D} sequence, "
                f"got length {len(bins_seq)}"
            )

    # ``range`` shadows the Python builtin in this function's signature,
    # so capture the builtin once for use below.
    py_range = (
        __builtins__["range"] if isinstance(__builtins__, dict) else __builtins__.range
    )

    if range is None:
        ranges: list[tuple[float, float]] = []
        for d in py_range(D):
            col = input[:, d]
            lo: float = float(col.min().item())
            hi: float = float(col.max().item())
            if lo == hi:
                hi = lo + 1.0
            ranges.append((lo, hi))
    else:
        ranges = [(float(lo), float(hi)) for (lo, hi) in range]
        if len(ranges) != D:
            raise ValueError(
                f"histogramdd: range must be a length-{D} sequence, "
                f"got length {len(ranges)}"
            )

    counts_impl, edges_impl = _C_engine.histogramdd(
        _unwrap(input), bins_seq, ranges, density and weight is None
    )
    counts = _wrap(counts_impl)
    edges_flat = _wrap(edges_impl)
    # Split the concatenated edges into D per-axis tensors.
    edges_list: list[Tensor] = []
    offset: int = 0
    for d in py_range(D):
        n = bins_seq[d] + 1
        edges_list.append(edges_flat[offset : offset + n])
        offset += n
    if weight is not None:
        counts = _histogramdd_weighted(input, weight, bins_seq, edges_list, density)
    return counts, edges_list


def histogram(
    input: Tensor,
    bins: int | Sequence[float] | Tensor = 100,
    *,
    range: tuple[float, float] | None = None,
    weight: Tensor | None = None,
    density: bool = False,
) -> tuple[Tensor, Tensor]:
    """Compute a 1-D histogram of ``input`` values.

    Everything happens on ``input``'s device, without reading a value back
    to the host: the range, when not given, is the input's extremes as
    tensors; each value's bin is its position on the uniform grid,
    corrected against the edges themselves, or a ``searchsorted`` over
    explicit edges; and the bins fill with one ``scatter_add``.

    Parameters
    ----------
    input : Tensor
        Values to bin.  Flattened internally — shape doesn't matter.
    bins : int, sequence of float or Tensor, optional
        When an int, the number of equal-width bins (default ``100``).
        Otherwise the increasing bin edges (length ``bins_n + 1``).
    range : tuple[float, float], optional
        ``(lo, hi)``.  Values outside the range are dropped.  Ignored
        when ``bins`` gives the edges.  ``None`` (default) →
        ``(input.min(), input.max())``.  A range of zero width is widened
        to ``(lo - 0.5, hi + 0.5)``.
    density : bool, optional
        When ``True`` divide by total count × bin width to yield a
        probability density that integrates to 1.  Default ``False``.
    weight : Tensor, optional
        As many elements as ``input``.  When supplied, sums weights per
        bin rather than counting.  Default ``None``.

    Returns
    -------
    tuple of Tensor
        ``(hist, bin_edges)``, both in ``input``'s floating dtype (the
        default float dtype for an integer ``input``) and on its device.
        ``hist`` holds counts, weight sums or densities; ``bin_edges`` has
        length ``bins + 1``.  Every bin is half-open but the last, which
        holds its right edge.  When the range comes from an ``input``
        holding a NaN or an infinity, ``hist`` is all NaN — the reference
        refuses there, which would take a read back to the host.

    Raises
    ------
    ValueError
        ``bins`` is not positive or its edges are fewer than two, the
        given ``range`` is not finite or its ``lo`` exceeds its ``hi``, or
        ``weight`` has another number of elements than ``input``.

    Examples
    --------
    >>> import lucid
    >>> hist, edges = lucid.histogram(lucid.tensor([1.0, 2.0, 1.0, 4.0]), bins=4)
    >>> hist
    tensor([2., 1., 0., 1.])
    >>> edges
    tensor([1., 1.75, 2.5, 3.25, 4.])
    """
    dtype = input.dtype if input.is_floating_point() else lucid.get_default_dtype()
    values = input.reshape(-1).to(dtype)
    device = input.device
    edges, bucket, finite = _histogram_bins(values, bins, range)
    n_bins = int(edges.numel()) - 1
    inside = (values >= edges[0]) & (values <= edges[-1])
    if weight is None:
        mass = lucid.ones(values.shape, dtype=dtype, device=device)
    elif weight.numel() != values.numel():
        raise ValueError(
            f"histogram: weight must have as many elements as input "
            f"({values.numel()}), got {weight.numel()}"
        )
    else:
        mass = weight.reshape(-1).to(dtype)
    mass = lucid.where(inside, mass, lucid.zeros_like(mass))
    bucket = bucket.clamp(0, n_bins - 1)
    hist = _scatter_add(
        lucid.zeros(n_bins, dtype=dtype, device=device), 0, bucket, mass
    )
    if density:
        hist = hist / (hist.sum() * (edges[1:] - edges[:-1]))
    if finite is not None:
        hist = lucid.where(finite, hist, lucid.full_like(hist, math.nan))
    return hist, edges


def _histogram_bins(
    values: Tensor,
    bins: int | Sequence[float] | Tensor,
    range: tuple[float, float] | None,
) -> tuple[Tensor, Tensor, Tensor | None]:
    """The edges, each value's bin, and — when the range was read from the
    values — whether that range is finite."""
    if not isinstance(bins := _promotion.as_number(bins), int):
        if isinstance(bins, lucid.Tensor):
            edges = bins.reshape(-1).to(dtype=values.dtype, device=values.device)
        else:
            edges = lucid.tensor(list(bins), dtype=values.dtype, device=values.device)
        if edges.numel() < 2:
            raise ValueError("histogram: bin edges must hold at least two values")
        return edges, lucid.searchsorted(edges, values, right=True) - 1, None
    if bins <= 0:
        raise ValueError(f"histogram: bins must be positive, got {bins}")
    lo, hi = _histogram_range(values, range)
    finite = None
    if isinstance(lo, lucid.Tensor) and isinstance(hi, lucid.Tensor):
        finite = lucid.isfinite(lo) & lucid.isfinite(hi)
    edges = _uniform_edges(lo, hi, bins, values)
    return edges, _uniform_bucket(values, edges, bins), finite


def _histogram_range(
    values: Tensor, range: tuple[float, float] | None
) -> tuple[float, float] | tuple[Tensor, Tensor]:
    """The ``(lo, hi)`` a uniform histogram spans.

    A given ``range`` is checked here, on the host, where it already is.
    Read from ``values`` it stays two 0-d tensors on their device, a range
    of zero width widened by 0.5 either side by ``where``.
    """
    if range is not None:
        lo, hi = float(range[0]), float(range[1])
        if not (math.isfinite(lo) and math.isfinite(hi)):
            raise ValueError(f"histogram: the range [{lo}, {hi}] is not finite")
        if lo > hi:
            raise ValueError(f"histogram: the range's lo {lo} exceeds its hi {hi}")
        return (lo - 0.5, hi + 0.5) if lo == hi else (lo, hi)
    if not values.numel():
        return 0.0, 1.0
    low, high = values.min(), values.max()
    flat = low == high
    return (
        lucid.where(flat, low - 0.5, low),
        lucid.where(flat, high + 0.5, high),
    )


def _uniform_edges(
    lo: float | Tensor, hi: float | Tensor, bins: int, like: Tensor
) -> Tensor:
    """``bins + 1`` equally spaced edges from ``lo`` to ``hi``, both exact."""
    if isinstance(lo, float) and isinstance(hi, float):
        return lucid.linspace(lo, hi, bins + 1, dtype=like.dtype, device=like.device)
    steps = lucid.arange(bins, dtype=like.dtype, device=like.device) / bins
    inner = lo + (hi - lo) * steps
    last = (hi + lucid.zeros(1, dtype=like.dtype, device=like.device)).reshape(1)
    return lucid.cat([inner, last], dim=0)


def _uniform_bucket(values: Tensor, edges: Tensor, bins: int) -> Tensor:
    """The bin of each value on ``bins`` equal bins between ``edges``' ends.

    The position on the grid, ``(v - lo) * bins / (hi - lo)``, rounds: a
    value on an edge can land one bin off from what the edges say.  So it
    is checked against the edges either side and moved by one where they
    disagree, as the reference corrects it — the edges are the truth.
    """
    lo, hi = edges[0], edges[-1]
    scaled = (values - lo) * (bins / (hi - lo))
    scaled = lucid.where(lucid.isfinite(scaled), scaled, lucid.zeros_like(scaled))
    bucket = scaled.floor().to(lucid.int64).clamp(0, bins - 1)
    below = values < edges[bucket]
    above = (values >= edges[(bucket + 1).clamp(max=bins)]) & (bucket < bins - 1)
    return bucket - below.to(lucid.int64) + above.to(lucid.int64)


def std_mean(input: Tensor, *args: object, **kwargs: object) -> tuple[Tensor, Tensor]:
    """Return ``(std, mean)`` along ``dim`` in a single call.

    Fused convenience over :func:`lucid.std` + :func:`lucid.mean` — the
    engine reduction still scans ``input`` twice, but the call site is
    shorter and the two outputs share an explicit ``correction`` /
    ``keepdim`` contract.

    Parameters
    ----------
    input : Tensor
        Input.
    dim : int, sequence of int, or None, optional
        Axis or axes to reduce.  ``None`` (default) reduces over every
        axis to a scalar pair.  A bool here is ``unbiased``:
        ``std_mean(x, False)``.
    unbiased : bool, optional
        ``True`` is ``correction=1``, ``False`` ``correction=0``.  A
        number in this place is the ``correction`` itself, as
        ``std_mean(x, dim, 0)`` has always been read.
    keepdim : bool, optional
        Retain the reduced axes as size-1 dimensions.  Default
        ``False``.
    correction : int or float, optional
        Bessel correction, as :func:`lucid.std` takes it.  ``1`` is the
        unbiased sample std (default); ``0`` the population std.

    Returns
    -------
    tuple of Tensor
        ``(std, mean)`` — same shape relationship as
        :func:`lucid.std` / :func:`lucid.mean` would produce
        individually.

    Notes
    -----
    ``correction`` and ``keepdim`` apply with ``dim=None`` too; they used
    to be dropped there.  Too few elements — no more than ``correction``
    — give ``nan`` or ``inf``, as :func:`lucid.var` explains.

    See Also
    --------
    var_mean : variance + mean variant.

    Examples
    --------
    >>> import lucid
    >>> lucid.std_mean(lucid.tensor([1.0, 2.0, 3.0, 4.0]), correction=0)
    (tensor(1.118), tensor(2.5))
    """
    return _moments("std_mean", input, args, kwargs)


def var_mean(input: Tensor, *args: object, **kwargs: object) -> tuple[Tensor, Tensor]:
    """Return ``(var, mean)`` along ``dim`` in a single call.

    Variance counterpart to :func:`std_mean`.  Useful inside
    normalisation layers where the squared deviation is the quantity
    actually wanted (e.g. BatchNorm / LayerNorm forward computes
    ``(x - mean) / sqrt(var + eps)``).

    Parameters
    ----------
    input : Tensor
        Input.
    dim : int, sequence of int, or None, optional
        Axis or axes to reduce.  ``None`` (default) → full-tensor
        reduction.  A bool here is ``unbiased``: ``var_mean(x, False)``.
    unbiased : bool, optional
        ``True`` is ``correction=1``, ``False`` ``correction=0``.  A
        number in this place is the ``correction`` itself, as
        ``var_mean(x, dim, 0)`` has always been read.
    keepdim : bool, optional
        Retain the reduced axes as size-1 dimensions.  Default
        ``False``.
    correction : int or float, optional
        Bessel correction, as :func:`lucid.var` takes it.  Default ``1``
        (unbiased sample variance).

    Returns
    -------
    tuple of Tensor
        ``(var, mean)``.

    Notes
    -----
    ``correction`` and ``keepdim`` apply with ``dim=None`` too; they used
    to be dropped there.  A single element has no spread to divide by
    :math:`n - 1`, so ``var_mean(tensor([5.0]))`` is ``(nan, 5)``.

    See Also
    --------
    std_mean : standard-deviation + mean variant.

    Examples
    --------
    >>> import lucid
    >>> lucid.var_mean(lucid.tensor([[1.0, 3.0], [2.0, 6.0]]), 1)
    (tensor([2., 8.]), tensor([2., 4.]))
    """
    return _moments("var_mean", input, args, kwargs)


def _moments(
    name: str, x: Tensor, args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[Tensor, Tensor]:
    """``(var or std, mean)``, the arguments read as ``var`` reads them.

    The positional contract has one owner, ``_variance_operands``; these
    two used to keep their own copies of it, which drifted — one refused a
    ``correction=True`` the released code accepted.  The third positional
    argument is ``unbiased`` or a ``correction``, never ``keepdim``.
    """
    from lucid._dispatch import _unwrap, _wrap

    dim, correction, keepdim = _variance_operands(name, args, kwargs)
    xi = _unwrap(x)
    axes = _reduction_axes(xi, dim, name)
    spread = _corrected_variance(xi, axes, correction, keepdim, name)
    if name == "std_mean":
        spread = _zero_safe_root(spread, 2)
    if spread.dtype != xi.dtype:
        spread = _C_engine.astype(spread, xi.dtype)
    return _wrap(spread), _wrap(_C_engine.mean(xi, axes, keepdim))


__all__ = [
    "quantile",
    "nanquantile",
    "cov",
    "corrcoef",
    "cdist",
    "bincount",
    "histogram",
    "histogram2d",
    "histogramdd",
    "multinomial",
    "poisson",
    "std_mean",
    "var_mean",
]
