"""CPU kernels rewritten to run as vector loops answer exactly as before.

Each of these was scalar for a reason the source did not show: a branch on a
data mask (``where``, ``masked_fill``), a per-element switch over the op
(comparisons, bitwise), a NaN test in a serial chain (``max``, ``argmax``),
a byte store that may alias the loop's own bound (casts), or vForce's exact
power for ``x ** 2``.  Rewritten branch-free they are 3-70 times faster; the
edge cases those branches handled — NaN, ±inf, signed zero, saturation, the
first of equal values — are what this file holds them to.  Sizes cross the
16-lane blocks and the per-core chunks, so tails and joins are exercised.
"""

import math

import numpy as np
import pytest

import lucid

_SIZES = [1, 3, 15, 16, 17, 63, 65537, 300_001]


def _data(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32)
    specials = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0], np.float32)
    k = min(n, len(specials))
    x[rng.choice(n, k, replace=False)] = specials[:k]
    return x


@pytest.mark.parametrize("n", _SIZES)
@pytest.mark.parametrize(
    "op, ref",
    [
        ("__eq__", np.equal),
        ("__ne__", np.not_equal),
        ("__gt__", np.greater),
        ("__ge__", np.greater_equal),
        ("__lt__", np.less),
        ("__le__", np.less_equal),
    ],
)
def test_comparisons(n: int, op: str, ref: np.ufunc) -> None:
    a, b = _data(n, 1), _data(n, 2)
    got = getattr(lucid.tensor(a), op)(lucid.tensor(b)).numpy()
    np.testing.assert_array_equal(got, ref(a, b))
    ia = np.where(np.isfinite(a), a * 4, 0).astype(np.int64)
    ib = np.where(np.isfinite(b), b * 4, 0).astype(np.int64)
    got = getattr(lucid.tensor(ia), op)(lucid.tensor(ib)).numpy()
    np.testing.assert_array_equal(got, ref(ia, ib))


@pytest.mark.parametrize("n", _SIZES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_where_and_masked_fill_select(n: int, dtype: type) -> None:
    a = _data(n, 3)
    x = np.nan_to_num(a, posinf=7, neginf=-7).astype(dtype)
    y = np.nan_to_num(_data(n, 4), posinf=5, neginf=-5).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        x, y = a.astype(dtype), _data(n, 4).astype(dtype)
    mask = np.random.default_rng(5).random(n) > 0.5
    got = lucid.where(lucid.tensor(mask), lucid.tensor(x), lucid.tensor(y)).numpy()
    np.testing.assert_array_equal(got, np.where(mask, x, y))
    got = lucid.tensor(x).masked_fill(lucid.tensor(mask), -3.0).numpy()
    np.testing.assert_array_equal(got, np.where(mask, dtype(-3), x))


@pytest.mark.parametrize("n", [1, 17, 65537])
@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_bitwise_ops(n: int, dtype: type) -> None:
    rng = np.random.default_rng(6)
    info = np.iinfo(dtype)
    a = rng.integers(info.min, info.max, n, dtype=dtype, endpoint=True)
    b = rng.integers(info.min, info.max, n, dtype=dtype, endpoint=True)
    la, lb = lucid.tensor(a), lucid.tensor(b)
    np.testing.assert_array_equal((la & lb).numpy(), a & b)
    np.testing.assert_array_equal((la | lb).numpy(), a | b)
    np.testing.assert_array_equal((la ^ lb).numpy(), a ^ b)
    bits = np.iinfo(dtype).bits
    shifts = rng.integers(-2, bits + 3, n).astype(dtype)
    ls = lucid.tensor(shifts)
    s64 = shifts.astype(np.int64)
    in_range = (s64 >= 0) & (s64 < bits)
    left = np.where(in_range, a << np.clip(shifts, 0, bits - 1), 0).astype(dtype)
    right = np.where(
        s64 < 0,
        0,
        np.where(
            s64 >= bits, np.where(a < 0, -1, 0), a >> np.clip(shifts, 0, bits - 1)
        ),
    ).astype(dtype)
    np.testing.assert_array_equal((la << ls).numpy(), left)
    np.testing.assert_array_equal((la >> ls).numpy(), right)


@pytest.mark.parametrize(
    "src, dst",
    [
        (np.float32, np.int32),
        (np.float32, np.int64),
        (np.float32, np.int16),
        (np.float32, np.int8),
        (np.float64, np.int32),
        (np.float64, np.int64),
    ],
)
def test_float_to_int_saturates_at_every_edge(src: type, dst: type) -> None:
    info = np.iinfo(dst)
    lo, hi = float(info.min), float(info.max)
    below_top = float(np.nextafter(src(2.0 ** (info.bits - 1)), src(0)))
    values = np.array(
        [
            math.nan,
            math.inf,
            -math.inf,
            0.0,
            -0.0,
            1.9,
            -1.9,
            lo,
            hi,
            lo - 1e3,
            hi + 1e3,
            below_top,
            -below_top,
            2.0 ** (info.bits - 1),
            -(2.0 ** (info.bits - 1)),
            3e30,
            -3e30,
        ],
        dtype=src,
    )

    def expect(v: float) -> int:
        if math.isnan(v):
            return 0
        if v >= hi:
            return int(info.max)
        if v <= lo:
            return int(info.min)
        return int(v)

    values = np.tile(values, 5)  # past one 16-lane block
    got = lucid.tensor(values).to(getattr(lucid, np.dtype(dst).name)).tolist()
    assert got == [expect(float(v)) for v in values]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 15, 16, 17, 31, 33, 1000])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_max_min_and_their_indices(n: int, dtype: type) -> None:
    rng = np.random.default_rng(n)
    x = rng.standard_normal(n).astype(dtype)
    if n > 2:
        x[n // 2] = x.max()  # a tie: the first index wins
    t = lucid.tensor(x)
    assert t.max().item() == x.max() and t.min().item() == x.min()
    assert t.argmax().item() == int(np.argmax(x))
    assert t.argmin().item() == int(np.argmin(x))
    for where in {0, n // 2, n - 1}:  # NaN in the vector body and in the tail
        y = x.copy()
        y[where] = np.nan
        u = lucid.tensor(y)
        assert math.isnan(u.max().item()) and math.isnan(u.min().item())
        assert u.argmax().item() == where and u.argmin().item() == where
    z = np.full(n, -np.inf, dtype)
    assert lucid.tensor(z).max().item() == -math.inf
    assert lucid.tensor(z).argmax().item() == 0


@pytest.mark.parametrize(
    "shape, axis", [((7, 33), 1), ((33, 7), 0), ((3, 17, 5), 1), ((4, 5, 6), 2)]
)
def test_axis_reductions_keep_their_answers(shape: tuple[int, ...], axis: int) -> None:
    x = np.random.default_rng(9).standard_normal(shape).astype(np.float32)
    x.flat[5] = np.nan
    t = lucid.tensor(x)
    np.testing.assert_array_equal(t.max(dim=axis).values.numpy(), np.max(x, axis=axis))
    np.testing.assert_array_equal(t.min(dim=axis).values.numpy(), np.min(x, axis=axis))
    np.testing.assert_allclose(t.sum(dim=axis).numpy(), np.sum(x, axis=axis), rtol=1e-6)
    np.testing.assert_allclose(
        t.prod(dim=axis).numpy(), np.prod(x, axis=axis), rtol=1e-5
    )
    np.testing.assert_array_equal(t.argmax(dim=axis).numpy(), np.argmax(x, axis=axis))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_scalar_powers_are_the_exact_operation(dtype: type) -> None:
    x = np.concatenate([_data(70_000, 7), [0.0, -0.0, 1.0, -1.0, 4.0]]).astype(dtype)
    t = lucid.tensor(x)
    with np.errstate(all="ignore"):
        np.testing.assert_array_equal((t**2).numpy(), x * x)
        np.testing.assert_array_equal((t**1).numpy(), x)
        np.testing.assert_array_equal((t**0).numpy(), np.ones_like(x))
        np.testing.assert_array_equal((t**-1).numpy(), dtype(1) / x)
        # IEEE pow, as before and as on Metal: +0 for -0, +inf for -inf
        # (numpy's power takes sqrt for 0.5 and answers -0 and nan there).
        ieee_root = np.where(x == -np.inf, np.inf, np.sqrt(x) + dtype(0))
        np.testing.assert_array_equal((t**0.5).numpy(), ieee_root)
        got, ref = (t**2.5).numpy(), np.power(x, dtype(2.5))
    finite = np.isfinite(ref)
    np.testing.assert_array_equal(np.isnan(got), np.isnan(ref))
    np.testing.assert_allclose(got[finite], ref[finite], rtol=4 * np.finfo(dtype).eps)
    # Signed zero and -inf, where sqrt and the power part ways.
    edge = lucid.tensor(np.array([-0.0, -np.inf, np.inf], dtype)) ** 0.5
    assert edge.tolist() == [0.0, math.inf, math.inf]
    assert math.copysign(1.0, edge.tolist()[0]) == 1.0


def test_scalar_power_keeps_dtype_promotion_and_gradient() -> None:
    i = lucid.tensor([1, 2, 3])
    assert (i**2).dtype == (i ** lucid.tensor(2)).dtype
    x = lucid.tensor([1.0, 2.0, 3.0], requires_grad=True)
    (x**3).sum().backward()
    assert x.grad.tolist() == [3.0, 12.0, 27.0]
    for form in (lambda t: t**2, lambda t: t.pow(2), lambda t: lucid.pow(t, 2)):
        x = lucid.tensor([1.0, -2.0], requires_grad=True)
        form(x).sum().backward()
        assert x.grad.tolist() == [2.0, -4.0]


def test_count_nonzero_is_an_exact_int64() -> None:
    x = lucid.tensor([[0.0, 1.0, -0.0], [math.nan, 2.0, 0.0]])
    assert lucid.count_nonzero(x).dtype == lucid.int64
    assert lucid.count_nonzero(x).item() == 3
    assert lucid.count_nonzero(x, dim=1).tolist() == [1, 2]
    assert lucid.count_nonzero(x, dim=[0, 1]).item() == 3


@pytest.mark.parametrize(
    "op",
    ["exp", "log", "sqrt", "tanh", "sin", "cos", "erf", "sigmoid", "arctan", "sinh"],
)
def test_transcendentals_split_across_cores_give_the_same_bits(op: str) -> None:
    # vForce calls (and erf, and the fused sigmoid) run in chunks across
    # cores now; a chunk computes exactly what the whole call would, so an
    # element's answer cannot depend on how large an array it came in.
    x = np.random.default_rng(3).standard_normal(300_001).astype(np.float32)
    if op in ("log", "sqrt"):
        x = np.abs(x) + 0.1
    fn = getattr(lucid, op)
    whole = fn(lucid.tensor(x)).numpy()
    for lo, hi in ((0, 17), (65_530, 65_545), (299_990, 300_001)):
        np.testing.assert_array_equal(fn(lucid.tensor(x[lo:hi])).numpy(), whole[lo:hi])
