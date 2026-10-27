"""Statistics-flavored ops: quantile / cov / corrcoef / cdist /
histogram* / multinomial / poisson."""

import numpy as np
import pytest

import lucid


class TestQuantile:
    def test_median(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
        out = lucid.quantile(t, 0.5).item()
        assert abs(out - 3.0) < 1e-6

    def test_p25(self, device: str) -> None:
        t = lucid.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
        out = lucid.quantile(t, 0.25).item()
        assert abs(out - 2.0) < 1e-6

    def test_with_dim(self, device: str) -> None:
        # quantile along a dim builds index tensors internally; those must land
        # on the input's device (regression: the metal path previously crashed
        # with bad_variant_access from a CPU index against a metal tensor).
        t = lucid.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
        out = lucid.quantile(t, 0.5, dim=1)
        assert out.shape == (2,)
        np.testing.assert_allclose(out.numpy(), [2.0, 5.0], atol=1e-6)

    def test_with_dim_all_nan(self, device: str) -> None:
        t = lucid.full((2, 3), float("nan"), device=device)
        out = lucid.quantile(t, 0.5, dim=1)
        assert out.shape == (2,)
        assert bool(lucid.isnan(out).all().item())


class TestCov:
    def test_basic(self, device: str) -> None:
        # Two-row matrix: (3, 4) — covariance is a 3×3 matrix.
        x = lucid.tensor(
            [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], [0.0, 1.0, 0.0, 1.0]], device=device
        )
        out = lucid.cov(x).numpy()
        # Variances on the diagonal must be non-negative.
        assert np.all(np.diag(out) >= 0)


class TestCorrcoef:
    def test_perfect_positive(self, device: str) -> None:
        # Two perfectly-correlated rows.
        x = lucid.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]], device=device)
        out = lucid.corrcoef(x).numpy()
        assert abs(out[0, 1] - 1.0) < 1e-3


class TestCdist:
    def test_l2(self, device: str) -> None:
        a = lucid.tensor([[0.0, 0.0]], device=device)
        b = lucid.tensor([[3.0, 4.0]], device=device)
        out = lucid.cdist(a, b).item()
        assert abs(out - 5.0) < 1e-4


class TestBincount:
    def test_basic(self, device: str) -> None:
        t = lucid.tensor([0, 1, 1, 2, 2, 2, 3], dtype=lucid.int64, device=device)
        out = lucid.bincount(t).numpy()
        np.testing.assert_array_equal(out, [1, 2, 3, 1])


class TestHistogram:
    # ``lucid.histogram`` has a known parameter-shadowing issue when
    # ``range=(lo, hi)`` is passed positionally — work around by
    # passing explicit bin edges via the ``bins`` argument.

    def test_basic(self, device: str) -> None:
        t = lucid.tensor([0.5, 1.5, 1.5, 2.5], device=device)
        counts, edges = lucid.histogram(t, bins=[0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(counts.numpy().astype(np.int32), [1, 2, 1])

    def test_density(self, device: str) -> None:
        t = lucid.tensor([0.5, 1.5, 2.5, 0.5], device=device)
        counts, edges = lucid.histogram(t, bins=[0.0, 1.0, 2.0, 3.0], density=True)
        bw = edges[1].item() - edges[0].item()
        assert abs(counts.sum().item() * bw - 1.0) < 1e-5


class TestHistogram2d:
    def test_basic(self, device: str) -> None:
        x = lucid.tensor([0.0, 1.0, 1.0, 2.0], device=device)
        y = lucid.tensor([0.0, 0.0, 1.0, 1.0], device=device)
        counts, xe, ye = lucid.histogram2d(x, y, bins=2, range=((0.0, 2.0), (0.0, 1.0)))
        assert counts.shape == (2, 2)
        assert xe.shape == (3,)
        assert ye.shape == (3,)


class TestHistogramdd:
    def test_3d(self, device: str) -> None:
        np.random.seed(0)
        data_np = np.random.uniform(-1.0, 1.0, size=(50, 3)).astype(np.float32)
        data = lucid.tensor(data_np.copy(), device=device)
        counts, edges = lucid.histogramdd(data, bins=3)
        assert counts.shape == (3, 3, 3)
        assert int(counts.sum().item()) == 50


class TestMultinomial:
    def test_shape(self, device: str) -> None:
        probs = lucid.tensor([0.2, 0.3, 0.5], device=device)
        out = lucid.multinomial(probs, num_samples=10, replacement=True)
        assert out.shape == (10,)
        arr = out.numpy()
        assert (arr >= 0).all() and (arr < 3).all()

    def test_stays_on_the_input_device_as_int64(self, device: str) -> None:
        for replacement in (True, False):
            probs = lucid.tensor([[1.0, 2.0, 3.0]] * 2, device=device)
            out = lucid.multinomial(probs, 2, replacement)
            assert out.shape == (2, 2)
            assert out.dtype == lucid.int64 and out.device == device

    @pytest.mark.parametrize("replacement", [True, False])
    def test_draws_follow_the_generator(self, device: str, replacement: bool) -> None:
        # Draws came from the host's own random module, which ignored both
        # the generator and manual_seed.
        probs = lucid.tensor([[0.1, 0.2, 0.7], [0.5, 0.25, 0.25]], device=device)

        def draw(seed: int) -> list[list[int]]:
            gen = lucid.Generator(seed=seed)
            return lucid.multinomial(probs, 3, replacement, generator=gen).tolist()

        assert draw(7) == draw(7)
        lucid.manual_seed(3)
        first = lucid.multinomial(probs, 3, replacement).tolist()
        lucid.manual_seed(3)
        assert lucid.multinomial(probs, 3, replacement).tolist() == first

    def test_frequencies_follow_the_weights(self, device: str) -> None:
        lucid.manual_seed(0)
        probs = lucid.tensor([1.0, 0.0, 3.0], device=device)
        arr = lucid.multinomial(probs, 4000, replacement=True).numpy()
        assert (arr != 1).all()  # a zero weight is never drawn
        assert abs((arr == 2).mean() - 0.75) < 0.03

    def test_without_replacement_draws_distinct_positive_categories(self, device: str) -> None:
        lucid.manual_seed(0)
        probs = lucid.tensor([[1.0, 0.0, 2.0, 3.0]] * 2000, device=device)
        rows = lucid.multinomial(probs, 3).tolist()
        assert all(sorted(row) == [0, 2, 3] for row in rows)
        # The first pick is an ordinary draw: category 3 holds half the weight.
        assert abs(sum(row[0] == 3 for row in rows) / len(rows) - 0.5) < 0.05

    def test_invalid_requests_raise(self, device: str) -> None:
        with pytest.raises(ValueError, match="without replacement"):
            lucid.multinomial(lucid.tensor([1.0, 0.0, 1.0], device=device), 3)
        with pytest.raises(ValueError, match="non-negative"):
            lucid.multinomial(lucid.tensor([1.0, -1.0], device=device), 1, True)
        with pytest.raises(ValueError, match="positive weight"):
            lucid.multinomial(lucid.tensor([0.0, 0.0], device=device), 1, True)


class TestPoissonOp:
    def test_zero_rate(self, device: str) -> None:
        rates = lucid.tensor([0.0, 0.0, 0.0], device=device)
        out = lucid.poisson(rates).numpy()
        np.testing.assert_array_equal(out, [0, 0, 0])

    def test_positive_samples(self, device: str) -> None:
        lucid.manual_seed(0)
        rates = lucid.tensor([5.0] * 200, device=device)
        out = lucid.poisson(rates).numpy()
        assert (out >= 0).all()
        # Mean should be near 5 with this many samples.
        assert abs(out.mean() - 5.0) < 1.0
