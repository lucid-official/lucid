"""The statistical composites, checked against NumPy's definitions.

``_ops/composite/statistics.py`` sat at 60%.  The dark part was the
tail of the file — ``nanquantile`` along an axis, explicit histogram
edges, weighted histograms, ``histogram2d`` / ``histogramdd``,
``std_mean`` / ``var_mean`` — and three of those were returning a
plausible number that was not the right one.

Statistics is a bad place for a quiet error: a quantile that is off by
one sample position, or a histogram binned onto the wrong grid, still
looks like a summary of the data.  So every expectation here is NumPy's,
computed independently.
"""

import numpy as np
import pytest

import lucid

RNG = np.random.default_rng(0)
X = RNG.standard_normal((6, 5))
Y = RNG.standard_normal((4, 5))
V = RNG.standard_normal(50)

V_NAN = V.copy()
V_NAN[[3, 17, 40]] = np.nan

X_NAN = X.copy()
X_NAN[0, 0] = np.nan
X_NAN[2, 3] = np.nan


def _t(a):
    return lucid.tensor(np.asarray(a, dtype=np.float64))


def _v(x):
    return np.asarray(x.numpy())


def _close(got, want, atol=1e-8):
    assert got.shape == want.shape, f"shape {got.shape} != {want.shape}"
    assert np.allclose(got, want, atol=atol, equal_nan=True)


# ── quantile ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("q", [0.0, 0.25, 0.5, 0.9, 1.0])
def test_quantile_of_a_vector(q):
    _close(_v(lucid.quantile(_t(V), q)), np.quantile(V, q))


def test_a_list_of_quantiles_adds_a_leading_axis():
    _close(_v(lucid.quantile(_t(V), [0.1, 0.5, 0.9])), np.quantile(V, [0.1, 0.5, 0.9]))


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_quantile_along_an_axis(dim, keepdim):
    _close(
        _v(lucid.quantile(_t(X), 0.5, dim=dim, keepdim=keepdim)),
        np.quantile(X, 0.5, axis=dim, keepdims=keepdim),
    )


@pytest.mark.parametrize("interpolation", ["linear", "lower", "higher", "nearest", "midpoint"])
def test_every_interpolation_rule(interpolation):
    """``0.37`` lands between two samples, so the five rules give five
    different answers — a constant ``q`` would hide four of them."""
    _close(
        _v(lucid.quantile(_t(V), 0.37, interpolation=interpolation)),
        np.quantile(V, 0.37, method=interpolation),
    )


def test_an_unknown_interpolation_rule_is_refused():
    with pytest.raises(ValueError, match="interpolation"):
        lucid.quantile(_t(V), 0.5, interpolation="cubic")


# ── nanquantile ───────────────────────────────────────────────────────────────


def test_nanquantile_ignores_the_nans():
    _close(_v(lucid.nanquantile(_t(V_NAN), 0.5)), np.nanquantile(V_NAN, 0.5))
    _close(_v(lucid.nanquantile(_t(V_NAN), [0.25, 0.75])), np.nanquantile(V_NAN, [0.25, 0.75]))


def test_a_single_nan_is_what_separates_it_from_quantile():
    """``quantile`` has to poison the slice; ``nanquantile`` has to not.

    ``quantile`` only propagated by accident: sorting puts NaN last, so a
    NaN reached the answer just when the interpolation position landed
    beside one and ``nan * 0.0`` carried it through.  Anywhere else it
    dropped out — these 50 samples with three NaNs returned an entirely
    ordinary median where NumPy and the reference both return NaN, which
    is a summary of data that silently isn't there.
    """
    assert np.isnan(_v(lucid.quantile(_t(V_NAN), 0.5)))
    assert np.isfinite(_v(lucid.nanquantile(_t(V_NAN), 0.5)))


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_quantile_poisons_only_the_slices_holding_a_nan(dim, keepdim):
    _close(
        _v(lucid.quantile(_t(X_NAN), 0.5, dim=dim, keepdim=keepdim)),
        np.quantile(X_NAN, 0.5, axis=dim, keepdims=keepdim),
    )


def test_quantile_poisons_per_slice_for_a_list_of_quantiles():
    _close(
        _v(lucid.quantile(_t(X_NAN), [0.25, 0.75], dim=0)), np.quantile(X_NAN, [0.25, 0.75], axis=0)
    )


@pytest.mark.parametrize("dim", [0, 1])
def test_nanquantile_along_an_axis_uses_each_slices_own_count(dim):
    """Slice ``i`` may have five valid entries where slice ``j`` has six,
    and ``q * (n - 1)`` differs accordingly.

    It used to collapse the counts to their mean, rounded down — a
    number that is nobody's count.  Two NaNs in a ``(6, 5)`` gave a mean
    of 5.6, so every column was read as if it had five entries and
    *three of the five columns holding no NaN at all* came back wrong.
    """
    _close(_v(lucid.nanquantile(_t(X_NAN), 0.5, dim=dim)), np.nanquantile(X_NAN, 0.5, axis=dim))


@pytest.mark.parametrize("interpolation", ["linear", "lower", "higher", "nearest", "midpoint"])
def test_nanquantile_honours_the_interpolation_rule_along_an_axis(interpolation):
    _close(
        _v(lucid.nanquantile(_t(X_NAN), 0.37, dim=0, interpolation=interpolation)),
        np.nanquantile(X_NAN, 0.37, axis=0, method=interpolation),
    )


def test_nanquantile_keepdim():
    _close(
        _v(lucid.nanquantile(_t(X_NAN), 0.5, dim=1, keepdim=True)),
        np.nanquantile(X_NAN, 0.5, axis=1, keepdims=True),
    )


def test_an_all_nan_slice_has_no_quantile():
    a = X.copy()
    a[:, 2] = np.nan
    got = _v(lucid.nanquantile(_t(a), 0.5, dim=0))
    assert np.isnan(got[2])
    assert np.isfinite(got[[0, 1, 3, 4]]).all()


def test_an_all_nan_input_gives_nan():
    assert np.isnan(_v(lucid.nanquantile(_t(np.full(5, np.nan)), 0.5)))


# ── cdist ─────────────────────────────────────────────────────────────────────


def _minkowski(a, b, p):
    diff = np.abs(a[:, None, :] - b[None, :, :])
    if np.isinf(p):
        return diff.max(axis=-1)
    return (diff**p).sum(axis=-1) ** (1.0 / p)


@pytest.mark.parametrize("p", [1.0, 2.0, 3.0, float("inf")])
def test_cdist_is_the_minkowski_distance(p):
    _close(_v(lucid.cdist(_t(X), _t(Y), p=p)), _minkowski(X, Y, p), atol=1e-6)


def test_cdist_of_a_set_with_itself_has_a_zero_diagonal():
    d = _v(lucid.cdist(_t(X), _t(X)))
    assert np.allclose(np.diag(d), 0.0, atol=1e-6)
    assert np.allclose(d, d.T, atol=1e-6)


# ── covariance ────────────────────────────────────────────────────────────────


def test_cov_matches_numpy():
    _close(_v(lucid.cov(_t(X))), np.cov(X))


def test_cov_without_the_bessel_correction():
    _close(_v(lucid.cov(_t(X), correction=0)), np.cov(X, bias=True))


def test_corrcoef_matches_numpy():
    _close(_v(lucid.corrcoef(_t(X))), np.corrcoef(X))


def test_corrcoef_has_a_unit_diagonal():
    assert np.allclose(np.diag(_v(lucid.corrcoef(_t(X)))), 1.0, atol=1e-8)


# ── std_mean / var_mean ───────────────────────────────────────────────────────


def test_std_mean_returns_both():
    std, mean = lucid.std_mean(_t(X))
    _close(_v(std), np.array(X.std(ddof=1)))
    _close(_v(mean), np.array(X.mean()))


@pytest.mark.parametrize("dim", [0, 1])
def test_std_mean_along_an_axis(dim):
    std, mean = lucid.std_mean(_t(X), dim=dim)
    _close(_v(std), X.std(axis=dim, ddof=1))
    _close(_v(mean), X.mean(axis=dim))


def test_std_mean_over_several_axes_without_correction():
    std, mean = lucid.std_mean(_t(X), dim=(0, 1), correction=0)
    _close(_v(std), np.array(X.std()))


def test_var_mean_is_std_mean_squared():
    var, mean = lucid.var_mean(_t(X), dim=1, keepdim=True)
    _close(_v(var), X.var(axis=1, ddof=1, keepdims=True))
    std, _ = lucid.std_mean(_t(X), dim=1, keepdim=True)
    _close(_v(var), _v(std) ** 2)


# ── bincount ──────────────────────────────────────────────────────────────────


COUNTS = np.array([0, 1, 1, 3, 3, 3, 7], dtype=np.int32)


def _ints(a):
    return lucid.tensor(np.asarray(a, dtype=np.int32), dtype=lucid.int32)


def test_bincount_counts_each_value():
    _close(_v(lucid.bincount(_ints(COUNTS))), np.bincount(COUNTS))


def test_bincount_pads_to_minlength():
    _close(_v(lucid.bincount(_ints(COUNTS), minlength=12)), np.bincount(COUNTS, minlength=12))


def test_bincount_sums_weights():
    w = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    _close(_v(lucid.bincount(_ints(COUNTS), weights=_t(w))), np.bincount(COUNTS, weights=w))


# ── histogram ─────────────────────────────────────────────────────────────────


def test_histogram_counts_and_edges():
    counts, edges = lucid.histogram(_t(V), bins=8)
    want_counts, want_edges = np.histogram(V, bins=8)
    _close(_v(counts), want_counts)
    _close(_v(edges), want_edges)


def test_histogram_over_an_explicit_range():
    counts, edges = lucid.histogram(_t(V), bins=8, range=(-2.0, 2.0))
    want_counts, want_edges = np.histogram(V, bins=8, range=(-2.0, 2.0))
    _close(_v(counts), want_counts)
    _close(_v(edges), want_edges)


def test_histogram_with_explicit_uneven_edges():
    """``(v - lo) / (hi - lo) * n_bins`` is the bin index only when the
    edges are equally spaced.  Given ``[-3, -1, 0, 1, 3]`` it silently
    re-binned onto a uniform grid of the same span: 50 standard normals
    came back as ``[1, 19, 23, 7]`` where the answer is ``[8, 12, 17,
    13]``.  The totals agree, which is why nothing looked wrong.
    """
    edges_in = [-3.0, -1.0, 0.0, 1.0, 3.0]
    counts, edges = lucid.histogram(_t(V), bins=edges_in)
    want_counts, want_edges = np.histogram(V, bins=edges_in)
    _close(_v(counts), want_counts)
    _close(_v(edges), want_edges)


def test_a_weighted_histogram_sums_weights_rather_than_counting():
    """Weighted sums are not counts, and were being emitted at ``int64``
    — every bin truncated toward zero, ``2.2035`` arriving as ``2``."""
    counts, _ = lucid.histogram(_t(V), bins=6, weight=_t(np.abs(V)))
    want, _ = np.histogram(V, bins=6, weights=np.abs(V))
    _close(_v(counts), want)
    assert not np.allclose(_v(counts), np.floor(want))


def test_a_density_histogram_integrates_to_one():
    counts, edges = lucid.histogram(_t(V), bins=8, density=True)
    want, _ = np.histogram(V, bins=8, density=True)
    _close(_v(counts), want)
    widths = np.diff(_v(edges))
    assert np.isclose((_v(counts) * widths).sum(), 1.0)


# ── histogram2d / histogramdd ─────────────────────────────────────────────────


def test_histogram2d_matches_numpy():
    xs, ys = V[:25], V[25:]
    counts, ex, ey = lucid.histogram2d(_t(xs), _t(ys), bins=5)
    want, want_ex, want_ey = np.histogram2d(xs, ys, bins=5)
    _close(_v(counts), want)
    _close(_v(ex), want_ex)
    _close(_v(ey), want_ey)


def test_histogram2d_with_a_different_count_per_axis():
    xs, ys = V[:25], V[25:]
    counts, _, _ = lucid.histogram2d(_t(xs), _t(ys), bins=(4, 6))
    want, _, _ = np.histogram2d(xs, ys, bins=(4, 6))
    assert _v(counts).shape == (4, 6)
    _close(_v(counts), want)


def test_histogramdd_matches_numpy():
    points = RNG.standard_normal((40, 3))
    counts, edges = lucid.histogramdd(_t(points), bins=4)
    want, want_edges = np.histogramdd(points, bins=4)
    _close(_v(counts), want)
    for got, expected in zip(edges, want_edges):
        _close(_v(got), expected)


def test_histogramdd_keeps_every_point():
    points = RNG.standard_normal((40, 3))
    counts, _ = lucid.histogramdd(_t(points), bins=4)
    assert _v(counts).sum() == 40
