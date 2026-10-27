"""Regression tests for RNG bit resolution, 64-bit randint bounds, and tie-free randperm (LCD-219)."""

import numpy as np
import pytest

import lucid
import lucid.nn as nn

DEVICES = ["cpu", "metal"]


class TestFloat64BitResolution:
    """CPU float64 random draws use full 53-bit IEEE-754 mantissas, not 24-bit float32 casts."""

    def test_rand_float64_has_53_bit_resolution(self) -> None:
        lucid.manual_seed(42)
        r = lucid.rand(100_000, dtype=lucid.float64).numpy()
        assert r.dtype == np.float64
        assert np.all((r >= 0.0) & (r < 1.0))
        # A 24-bit float32 draw satisfies (x * 2**24) % 1 == 0 for every element.
        # A 53-bit draw has 29 additional random bits below 2**-24, so almost every
        # sample has a non-zero fractional part after scaling by 2**24.
        frac_below_24 = np.mod(r * float(1 << 24), 1.0)
        assert np.count_nonzero(frac_below_24 != 0.0) > 99_900
        # Every sample is an exact multiple of 2**-53.
        assert np.all(np.mod(r * float(1 << 53), 1.0) == 0.0)

    def test_rand_like_and_init_uniform_float64_have_53_bit_resolution(self) -> None:
        lucid.manual_seed(123)
        proto = lucid.empty(10_000, dtype=lucid.float64)
        r_like = lucid.rand_like(proto).numpy()
        assert np.count_nonzero(np.mod(r_like * float(1 << 24), 1.0) != 0.0) > 9_900

        w = lucid.empty(10_000, dtype=lucid.float64)
        nn.init.uniform_(w, 0.0, 1.0)
        w_np = w.numpy()
        assert np.count_nonzero(np.mod(w_np * float(1 << 24), 1.0) != 0.0) > 9_900

    def test_randn_and_normal_float64_use_double_precision_box_muller(self) -> None:
        lucid.manual_seed(7)
        z = lucid.randn(50_000, dtype=lucid.float64).numpy()
        assert z.dtype == np.float64
        # Samples computed in double precision are not representable as exact float32 values.
        assert np.count_nonzero(z.astype(np.float32).astype(np.float64) != z) > 49_900
        assert abs(float(z.mean())) < 0.02
        assert abs(float(z.std()) - 1.0) < 0.02

        n = lucid.normal(mean=2.5, std=0.5, size=(20_000,), dtype=lucid.float64).numpy()
        assert np.count_nonzero(n.astype(np.float32).astype(np.float64) != n) > 19_900
        assert abs(float(n.mean()) - 2.5) < 0.02
        assert abs(float(n.std()) - 0.5) < 0.02

        z_like = lucid.randn_like(lucid.empty(10_000, dtype=lucid.float64)).numpy()
        assert np.count_nonzero(z_like.astype(np.float32).astype(np.float64) != z_like) > 9_900

        w = lucid.empty(10_000, dtype=lucid.float64)
        nn.init.normal_(w, mean=0.0, std=1.0)
        w_np = w.numpy()
        assert np.count_nonzero(w_np.astype(np.float32).astype(np.float64) != w_np) > 9_900

    def test_bernoulli_float64_compares_against_53_bit_uniform(self) -> None:
        # Find a generator state where the top 24 bits of word 0 are non-zero, and compare
        # the exact 24-bit vs 53-bit uniform draw from that state by setting p strictly
        # between them so a 24-bit comparison and a 53-bit comparison disagree.
        g = lucid.Generator(0)
        for counter in range(256):
            g.counter = counter
            w0, w1, _, _ = g.next_uint32x4()
            u24 = float(w0 >> 8) * (1.0 / float(1 << 24))
            u53 = float(((w0 << 32) | w1) >> 11) * (1.0 / float(1 << 53))
            if u53 > u24:
                p = 0.5 * (u24 + u53)
                g.counter = counter
                b = lucid.bernoulli(lucid.tensor([p], dtype=lucid.float64), generator=g).item()
                # Since u24 < p < u53, a 24-bit draw would yield 1.0 whereas the 53-bit draw yields 0.0.
                assert b == 0.0
                return
        pytest.fail("Expected u53 > u24 within 256 Philox blocks")


class TestRandintWideRangesAndDtypes:
    """randint supports 64-bit bounds and large spans on both CPU and Metal without truncation."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize(
        ("low", "high"),
        [
            (0, 1 << 31),
            (0, 1 << 32),
            (0, 1 << 40),
            (1 << 40, (1 << 40) + 10_000),
            (-(1 << 40), 1 << 40),
            (-(1 << 62), 1 << 62),
        ],
        ids=["0..2^31", "0..2^32", "0..2^40", "shifted-above-2^40", "-2^40..2^40", "-2^62..2^62"],
    )
    def test_wide_ranges_are_in_bounds_and_unquantized(
        self, device: str, low: int, high: int
    ) -> None:
        lucid.manual_seed(99)
        t = lucid.randint(low, high, (10_000,), dtype=lucid.int64, device=device)
        arr = t.numpy()
        assert arr.dtype == np.int64
        assert np.all((arr >= low) & (arr < high))
        # Must not collapse to all zeros or float32-quantized even numbers.
        assert np.any(arr != 0)
        odd_ratio = float(np.mean(arr & 1))
        assert 0.45 < odd_ratio < 0.55
        # Sample mean in float64 should be close to the interval midpoint.
        mid = 0.5 * (float(low) + float(high))
        span = float(high) - float(low)
        assert abs(float(arr.astype(np.float64).mean()) - mid) < 0.05 * span

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize(
        "dtype",
        [lucid.int8, lucid.int16, lucid.int32, lucid.int64],
        ids=["int8", "int16", "int32", "int64"],
    )
    def test_randint_integer_dtypes(self, device: str, dtype: lucid.dtype) -> None:
        lucid.manual_seed(17)
        t = lucid.randint(-20, 20, (512,), dtype=dtype, device=device)
        assert t.dtype == dtype
        arr = t.numpy()
        assert np.all((arr >= -20) & (arr < 20))
        assert len(set(arr.tolist())) > 20

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("shape", [(), (0,), (3, 0, 2)], ids=["scalar", "empty-1d", "empty-3d"])
    def test_randint_edge_shapes(self, device: str, shape: tuple[int, ...]) -> None:
        t = lucid.randint(0, 1 << 35, shape, device=device)
        assert t.shape == shape
        assert t.dtype == lucid.int64


class TestRandpermTieFree:
    """randperm uses 64-bit keys so million-element permutations do not suffer float32 key ties."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_randperm_million_is_valid_permutation_with_unbiased_adjacencies(
        self, device: str
    ) -> None:
        lucid.manual_seed(2026)
        n = 1_000_000
        perm = lucid.randperm(n, device=device).numpy()
        assert perm.shape == (n,)
        assert perm.dtype == np.int64
        # Every index 0..n-1 appears exactly once.
        assert np.array_equal(np.sort(perm), np.arange(n, dtype=np.int64))
        # Under 24-bit float32 keys, ~30,000 tied pairs sorted stably in ascending index order,
        # biasing P(perm[i] < perm[i+1]) upward. Verify adjacent ascent rate is 0.5 within 3 sigma.
        ascents = float(np.mean(perm[:-1] < perm[1:]))
        assert abs(ascents - 0.5) < 0.003

    @pytest.mark.parametrize("device", DEVICES)
    def test_64_bit_sort_keys_have_zero_ties_at_one_million(self, device: str) -> None:
        lucid.manual_seed(2026)
        n = 1_000_000
        keys = lucid.randint(
            -(1 << 63), (1 << 63) - 1, (n,), dtype=lucid.int64, device=device
        ).numpy()
        assert len(np.unique(keys)) == n
