"""Exact whole-tensor samplers for the Poisson and Binomial laws.

Each element picks its own algorithm from its own parameters, and no element
is visited from Python:

* **Inversion** for a small mean (``< 10``): the probability table
  ``P(X = 0..K)`` is built in log space with a recurrence, accumulated, and a
  uniform draw is placed in it by counting the cumulative sums below it.
* **Transformed rejection** for a large mean (Hoermann 1993): PTRS for the
  Poisson, BTRS for the Binomial.  A round draws one proposal per element and
  tests it; elements already accepted keep their draw while the rest go round
  again, until every element has one.

A rejection round never reads another round's output.  Traced code cannot
stop early, so it unrolls a fixed number of rounds and picks the first
accepting one with a ``where`` chain in which each node is read once -- the
shape the Metal graph compiler needs (see ``Gamma``'s sampler for why).

Arithmetic is float64 on the CPU stream and float32 on the GPU stream, where
a Binomial with ``n`` beyond about ``1e6`` loses accuracy in the acceptance
test.  The returned counts are ``int64``.
"""

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import lucid
from lucid._C import engine as _C_engine
from lucid._types import DTypeLike

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor

# Below this mean a count is drawn by inversion, at or above it by rejection.
_REJECTION_MEAN = 10.0
# Rounds the eager loop may spend (it leaves as soon as every element has a
# draw; reaching the limit is a probability below 1e-90) and the rounds a
# traced graph unrolls (the slowest regime accepts ~70 % per round, so 16
# rounds leave under 3e-9 of the elements on the fallback).
_EAGER_ROUNDS = 100
_TRACED_ROUNDS = 16
# Inversion tables hold at most this many elements, so a large batch is
# worked through in chunks.
_TABLE_ELEMENTS = 1 << 21
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def _work_dtype(device_type: str) -> DTypeLike:
    return lucid.float64 if device_type == "cpu" else lucid.float32


def _is_traced() -> bool:
    return _C_engine.compile.current_tracer() is not None


def _table_length(mean: float) -> int:
    """Entries of the pmf table that leave less than ``1e-15`` of the mass beyond it."""
    return int(math.ceil(mean + 10.0 * math.sqrt(mean) + 12.0))


def _first_accepted(one_round: Callable[[], tuple[Tensor, Tensor]], fallback: Tensor) -> Tensor:
    """The value of the first round that accepts, per element; ``fallback`` where none does."""
    if _is_traced():
        rounds = [one_round() for _ in range(_TRACED_ROUNDS)]
        draw = fallback
        for accepted, value in reversed(rounds):
            draw = lucid.where(accepted, value, draw)
        return draw
    draw = fallback
    done: Tensor | None = None
    for _ in range(_EAGER_ROUNDS):
        accepted, value = one_round()
        take = accepted if done is None else accepted & ~done
        draw = lucid.where(take, value, draw)
        done = accepted if done is None else done | accepted
        if bool(done.all().item()):
            break
    return draw


def _count_below(cdf: Tensor, u: Tensor) -> Tensor:
    """``#{k : cdf[k] < u}`` over the leading axis: the inverse CDF at ``u``."""
    return (cdf < u.unsqueeze(0)).to(u.dtype).sum(dim=0)


def _invert_in_chunks(
    columns: list[Tensor], u: Tensor, length: int, table: Callable[..., Tensor]
) -> Tensor:
    """Run ``table(*chunks, u_chunk) -> counts`` over flat chunks that bound the table size."""
    flat = [c.reshape(-1) for c in columns]
    uf = u.reshape(-1)
    total = int(uf.shape[0])
    step = max(1, _TABLE_ELEMENTS // length)
    parts = [
        table(*[c[s : s + step] for c in flat], uf[s : s + step], length)
        for s in range(0, total, step)
    ]
    out = parts[0] if len(parts) == 1 else lucid.cat(parts, dim=0)
    return out.reshape(u.shape)


def _rows(length: int, like: Tensor) -> Tensor:
    """``0 .. length-1`` as a column, to broadcast against ``(length, chunk)`` tables."""
    return lucid.arange(0, length, 1, dtype=like.dtype, device=like.device).reshape(length, 1)


def _pmf_cdf(log_p0: Tensor, log_step: Tensor) -> Tensor:
    """Cumulative probabilities from ``log P(0)`` and the log of each ratio ``P(k) / P(k-1)``."""
    return lucid.exp(log_p0.unsqueeze(0) + log_step.cumsum(dim=0)).cumsum(dim=0)


# ── Poisson ──────────────────────────────────────────────────────────────────


def _poisson_table(lam: Tensor, u: Tensor, length: int) -> Tensor:
    k = _rows(length, lam)
    log_lam = lam.log().unsqueeze(0)
    step = lucid.where(k > 0.0, log_lam - k.clip(1.0, float("inf")).log(), 0.0)
    return _count_below(_pmf_cdf(-lam, step), u)


def _stirling_correction(k: Tensor) -> Tensor:
    """``ln k! - (k ln k - k + 0.5 ln(2 pi k))`` for ``k >= 10`` (error below ``1e-12``)."""
    inv = 1.0 / k
    inv2 = inv * inv
    return inv * (1.0 / 12.0 - inv2 * (1.0 / 360.0 - inv2 * (1.0 / 1260.0 - inv2 / 1680.0)))


def _poisson_log_pmf(k: Tensor, lam: Tensor, log_lam: Tensor) -> Tensor:
    """``-lam + k ln lam - ln k!`` for ``k >= 0``, free of the cancellation in that form.

    Beyond ``k = 10`` the three terms are replaced by ``-lam * h(d) - ...`` with
    ``d = (k - lam) / lam`` and ``h(d) = (1 + d) ln(1 + d) - d``: both are of
    order ``lam ln lam``, and their difference is of order ``1``.
    """
    plain = -lam + k * log_lam - lucid.lgamma(k + 1.0)
    big = k.clip(10.0, float("inf"))
    d = (big - lam) / lam
    h = (1.0 + d) * d.log1p() - d
    stirling = -lam * h - (0.5 * (2.0 * math.pi * big).log()) - _stirling_correction(big)
    return lucid.where(k >= 10.0, stirling, plain)


def _poisson_ptrs(lam: Tensor, generator: _C_engine.Generator | None) -> Tensor:
    """Hoermann's PTRS for ``lam >= 10``."""
    shape = tuple(lam.shape)
    slam = lam.sqrt()
    log_lam = lam.log()
    b = 0.931 + 2.53 * slam
    a = -0.059 + 0.02483 * b
    inv_alpha = 1.1239 + 1.1328 / (b - 3.4)
    log_inv_alpha = inv_alpha.log()
    v_r = 0.9277 - 3.6224 / (b - 2.0)

    def one_round() -> tuple[Tensor, Tensor]:
        u = lucid.rand(*shape, dtype=lam.dtype, device=lam.device, generator=generator) - 0.5
        v = lucid.rand(*shape, dtype=lam.dtype, device=lam.device, generator=generator)
        us = (0.5 - u.abs()).clip(1e-30, float("inf"))
        k = ((2.0 * a / us + b) * u + lam + 0.43).floor()
        in_range = (us >= 0.013) | (v <= us)
        squeeze = (us >= 0.07) & (v <= v_r)
        k_safe = k.clip(0.0, float("inf"))
        log_v = v.clip(1e-30, float("inf")).log() + log_inv_alpha - (a / (us * us) + b).log()
        exact = log_v <= _poisson_log_pmf(k_safe, lam, log_lam)
        return (k >= 0.0) & (squeeze | (in_range & exact)), k_safe

    return _first_accepted(one_round, lam.floor())


def poisson_counts(rate: Tensor, generator: _C_engine.Generator | None = None) -> Tensor:
    """Exact ``Poisson(rate)`` draws, one per element of ``rate``.

    Parameters
    ----------
    rate : Tensor
        Finite, non-negative rates; any shape.
    generator : engine.Generator, optional
        An isolated stream; by default the :func:`manual_seed` stream.

    Returns
    -------
    Tensor
        ``int64`` counts, the shape of ``rate``.
    """
    f = _work_dtype(rate.device.type)
    lam = rate.detach().to(f)
    traced = _is_traced()
    out = lucid.zeros_like(lam)
    if lam.numel() == 0:
        return out.to(lucid.int64)

    large = lam >= _REJECTION_MEAN
    if traced or bool(large.any().item()):
        draw = _poisson_ptrs(lucid.where(large, lam, 20.0), generator)
        out = lucid.where(large, draw, out)

    small = (lam > 0.0) & ~large
    if traced or bool(small.any().item()):
        lam_s = lucid.where(small, lam, 1.0)
        length = _table_length(_REJECTION_MEAN if traced else float(lam_s.max().item()))
        u = lucid.rand(*lam.shape, dtype=f, device=lam.device, generator=generator)
        draw = _invert_in_chunks([lam_s], u, length, _poisson_table)
        out = lucid.where(small, draw, out)
    return out.to(lucid.int64)


# ── Binomial ─────────────────────────────────────────────────────────────────


def _binomial_table(n: Tensor, p: Tensor, u: Tensor, length: int) -> Tensor:
    k = _rows(length, n)
    left = n.unsqueeze(0) + 1.0 - k
    ratio = (
        lucid.where(left > 0.0, left.clip(1e-30, float("inf")).log(), float("-inf"))
        + (p / (1.0 - p)).log().unsqueeze(0)
        - k.clip(1.0, float("inf")).log()
    )
    step = lucid.where(k > 0.0, ratio, 0.0)
    return _count_below(_pmf_cdf(n * (-p).log1p(), step), u)


def _stirling_tail(k: Tensor) -> Tensor:
    """``ln k! - ((k + 0.5) ln(k + 1) - (k + 1) + 0.5 ln(2 pi))``: the Stirling series' tail."""
    x = k + 1.0
    inv2 = 1.0 / (x * x)
    series = (1.0 / 12.0 - (1.0 / 360.0 - (1.0 / 1260.0) * inv2) * inv2) / x
    small = lucid.lgamma(x) - ((k + 0.5) * x.log() - x + _HALF_LOG_2PI)
    return lucid.where(k <= 9.0, small, series)


def _binomial_btrs(n: Tensor, p: Tensor, generator: _C_engine.Generator | None) -> Tensor:
    """Hoermann's BTRS for ``n * p >= 10`` and ``p <= 0.5``."""
    shape = tuple(n.shape)
    q = 1.0 - p
    spq = (n * p * q).sqrt()
    b = 1.15 + 2.53 * spq
    a = -0.0873 + 0.0248 * b + 0.01 * p
    c = n * p + 0.5
    v_r = 0.92 - 4.2 / b
    log_r = (p / q).log()
    alpha = (2.83 + 5.1 / b) * spq
    m = ((n + 1.0) * p).floor()
    # The part of the acceptance bound that does not depend on the proposal.
    base = (
        (m + 0.5) * ((m + 1.0).log() - log_r - (n - m + 1.0).log())
        + (n + 1.0) * (n - m + 1.0).log()
        + _stirling_tail(m)
        + _stirling_tail(n - m)
    )

    def one_round() -> tuple[Tensor, Tensor]:
        u = lucid.rand(*shape, dtype=n.dtype, device=n.device, generator=generator) - 0.5
        v = lucid.rand(*shape, dtype=n.dtype, device=n.device, generator=generator)
        us = (0.5 - u.abs()).clip(1e-30, float("inf"))
        k = ((2.0 * a / us + b) * u + c).floor()
        valid = (k >= 0.0) & (k <= n)
        kc = k.clip(0.0, float("inf")).minimum(n)
        squeeze = (us >= 0.07) & (v <= v_r)
        log_v = (v.clip(1e-30, float("inf")) * alpha / (a / (us * us) + b)).log()
        bound = (
            base
            - (n + 1.0) * (n - kc + 1.0).log()
            + (kc + 0.5) * (log_r + (n - kc + 1.0).log() - (kc + 1.0).log())
            - _stirling_tail(kc)
            - _stirling_tail(n - kc)
        )
        return valid & (squeeze | (log_v <= bound)), kc

    return _first_accepted(one_round, m)


def binomial_counts(
    count: Tensor, prob: Tensor, generator: _C_engine.Generator | None = None
) -> Tensor:
    """Exact ``Binomial(count, prob)`` draws; ``count`` and ``prob`` share one shape.

    ``prob`` is folded onto ``[0, 0.5]`` first (``n - X`` is the count of the
    complementary trials), as the reference does.

    Parameters
    ----------
    count : Tensor
        Trial counts ``n``, integral and non-negative.
    prob : Tensor
        Success probabilities in ``[0, 1]``, the shape of ``count``.
    generator : engine.Generator, optional
        An isolated stream; by default the :func:`manual_seed` stream.

    Returns
    -------
    Tensor
        ``int64`` counts in ``[0, n]``, the shape of ``count``.
    """
    f = _work_dtype(count.device.type)
    n = count.detach().to(f)
    p = prob.detach().to(f)
    flip = p > 0.5
    q = lucid.where(flip, 1.0 - p, p)
    traced = _is_traced()
    out = lucid.zeros_like(n)
    if n.numel() == 0:
        return out.to(lucid.int64)

    live = (n > 0.0) & (q > 0.0)
    large = live & (n * q >= _REJECTION_MEAN)
    if traced or bool(large.any().item()):
        draw = _binomial_btrs(lucid.where(large, n, 100.0), lucid.where(large, q, 0.5), generator)
        out = lucid.where(large, draw, out)

    small = live & ~large
    if traced or bool(small.any().item()):
        n_s = lucid.where(small, n, 0.0)
        q_s = lucid.where(small, q, 0.25)
        mean = _REJECTION_MEAN if traced else float((n_s * q_s).max().item())
        u = lucid.rand(*n.shape, dtype=f, device=n.device, generator=generator)
        draw = _invert_in_chunks([n_s, q_s], u, _table_length(mean), _binomial_table)
        out = lucid.where(small, draw, out)
    return lucid.where(flip, n - out, out).to(lucid.int64)
