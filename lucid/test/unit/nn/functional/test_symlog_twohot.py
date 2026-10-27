"""Scale-free transforms: symlog / symexp and two-hot encoding.

These exist so one set of hyperparameters can cover targets whose scale is
not known in advance — a reward of 0.01 in one environment and 10000 in
another. So the tests are about *range* and *exactness*, not shapes: the
round trip has to hold across eight orders of magnitude, and a value
landing on a bin has to encode as a plain one-hot.

The pairing is what a distributional regression head uses. Encoding a
target through ``symlog`` onto a uniform grid, then reading the
prediction back as ``symexp`` of the grid's expectation, is DreamerV3's
reward and value loss; the arithmetic below is that loop.
"""

import math

import pytest

import lucid
import lucid.nn.functional as F


class TestSymlog:
    @pytest.mark.parametrize("value", [-1e6, -1234.0, -1.0, 0.0, 1.0, 1234.0, 1e6])
    def test_round_trip(self, value: float) -> None:
        drawn = lucid.tensor([value])
        recovered = float(F.symexp(F.symlog(drawn)).item())
        assert abs(recovered - value) <= 1e-5 * max(abs(value), 1.0)

    def test_zero_is_fixed(self) -> None:
        """Not approximately — a transform that shifted zero would move every
        reward of exactly nothing."""
        assert float(F.symlog(lucid.tensor([0.0])).item()) == 0.0
        assert float(F.symexp(lucid.tensor([0.0])).item()) == 0.0

    def test_matches_the_formula(self) -> None:
        for value in (0.5, 7.0, 1e6):
            expected = math.copysign(math.log1p(abs(value)), value)
            got = float(F.symlog(lucid.tensor([value])).item())
            assert abs(got - expected) < 1e-4 * max(abs(expected), 1.0)

    def test_it_is_odd(self) -> None:
        drawn = lucid.tensor([0.3, 5.0, 900.0])
        assert float((F.symlog(-drawn) + F.symlog(drawn)).abs().max().item()) < 1e-6

    def test_it_compresses(self) -> None:
        """Eight orders of magnitude in, one and a half out."""
        small = float(F.symlog(lucid.tensor([1.0])).item())
        large = float(F.symlog(lucid.tensor([1e6])).item())
        assert large / small < 25.0

    @pytest.mark.parametrize("value", [3.0, 99.0, 0.5])
    def test_gradient_matches_the_derivative(self, value: float) -> None:
        """``d/dx symlog = 1 / (1 + |x|)`` — near 1 near the origin, small far out."""
        drawn = lucid.tensor([value], requires_grad=True)
        F.symlog(drawn).backward()
        assert abs(float(drawn.grad.item()) - 1.0 / (1.0 + value)) < 1e-5

    def test_the_gradient_at_exactly_zero_is_zero(self) -> None:
        """Documented, not desired.

        The analytic derivative at the origin is 1 — ``symlog`` is the
        identity to first order there. Written as ``sign(x) * log1p(|x|)``
        it comes out 0 instead, because ``sign(0)`` is 0 and the product
        rule kills the term. It does not matter for what this is used for:
        the transform is applied to observations and to regression
        *targets*, neither of which is differentiated through. Asserted so
        that nobody discovers it by accident in a place where it would.
        """
        drawn = lucid.tensor([0.0], requires_grad=True)
        F.symlog(drawn).backward()
        assert float(drawn.grad.item()) == 0.0


class TestTwoHot:
    BINS = lucid.tensor([0.0, 1.0, 2.0, 3.0])

    def test_a_value_on_a_bin_is_one_hot(self) -> None:
        """The degenerate case has to stay degenerate."""
        encoded = F.two_hot(lucid.tensor([2.0]), self.BINS)
        assert [round(float(v), 6) for v in encoded[0]] == [0.0, 0.0, 1.0, 0.0]

    def test_a_midpoint_splits_evenly(self) -> None:
        encoded = F.two_hot(lucid.tensor([1.5]), self.BINS)
        assert [round(float(v), 6) for v in encoded[0]] == [0.0, 0.5, 0.5, 0.0]

    def test_weight_is_linear_in_position(self) -> None:
        encoded = F.two_hot(lucid.tensor([2.25]), self.BINS)
        assert abs(float(encoded[0][2]) - 0.75) < 1e-6
        assert abs(float(encoded[0][3]) - 0.25) < 1e-6

    @pytest.mark.parametrize("value", [-5.0, 0.0, 1.7, 3.0, 99.0])
    def test_it_is_a_distribution(self, value: float) -> None:
        encoded = F.two_hot(lucid.tensor([value]), self.BINS)
        assert abs(float(encoded.sum().item()) - 1.0) < 1e-6
        assert bool((encoded >= 0.0).all().item())

    @pytest.mark.parametrize("value", [-5.0, 1.7, 99.0])
    def test_at_most_two_bins_are_used(self, value: float) -> None:
        """The name is a claim; a softmax-like spread would pass the sum check."""
        encoded = F.two_hot(lucid.tensor([value]), self.BINS)
        assert int((encoded > 1e-9).to(lucid.float32).sum().item()) <= 2

    def test_out_of_range_clamps(self) -> None:
        below = F.two_hot(lucid.tensor([-100.0]), self.BINS)
        above = F.two_hot(lucid.tensor([100.0]), self.BINS)
        assert float(below[0][0]) == 1.0
        assert float(above[0][3]) == 1.0

    def test_batched(self) -> None:
        encoded = F.two_hot(lucid.randn((2, 5)) * 3.0, self.BINS)
        assert encoded.shape == (2, 5, 4)
        assert float((encoded.sum(dim=-1) - 1.0).abs().max().item()) < 1e-6

    def test_rejects_a_degenerate_grid(self) -> None:
        with pytest.raises(ValueError):
            F.two_hot(lucid.tensor([0.0]), lucid.tensor([1.0]))
        with pytest.raises(ValueError):
            F.two_hot(lucid.tensor([0.0]), lucid.tensor([[0.0, 1.0]]))


class TestSymexpTwoHotLoop:
    """The pairing as a distributional regression head uses it.

    Encode through ``symlog`` onto a uniform grid, decode as ``symexp`` of
    the grid's expectation. This is what makes one 41-bin head cover
    rewards spanning eight orders of magnitude, which is the claim worth
    testing.
    """

    GRID = lucid.linspace(-20.0, 20.0, 41)

    def _round_trip(self, value: float) -> float:
        encoded = F.two_hot(F.symlog(lucid.tensor([value])), self.GRID)
        return float(F.symexp((encoded * self.GRID).sum(dim=-1)).item())

    @pytest.mark.parametrize("value", [-1e6, -10000.0, -12.5, -1.0, 0.0, 0.5, 7.0, 3000.0, 1e6])
    def test_recovers_across_eight_orders_of_magnitude(self, value: float) -> None:
        recovered = self._round_trip(value)
        assert abs(recovered - value) <= 1e-4 * max(abs(value), 1.0)

    def test_the_decode_is_exact_on_any_grid(self) -> None:
        """What ``symlog`` does *not* buy, stated so the next test is read right.

        Two-hot interpolates linearly, so the grid's expectation equals
        the encoded value whatever the bins are — even absurd ones. The
        transform is not what makes the decode accurate.
        """
        absurd = lucid.linspace(-1e6, 1e6, 41)
        encoded = F.two_hot(lucid.tensor([7.0]), absurd)
        recovered = float((encoded * absurd).sum(dim=-1).item())
        assert abs(recovered - 7.0) < 1e-2

    def test_what_symlog_buys_is_resolution(self) -> None:
        """A coarse grid cannot *represent* small values apart.

        The decode is exact either way, but a head predicts the
        distribution, not the expectation. On a linear grid spanning six
        orders of magnitude, 1 and 100 encode to almost the same vector —
        no classifier can separate them. Under ``symlog`` they are bins
        apart.
        """
        linear = lucid.linspace(-1e6, 1e6, 41)
        near = F.two_hot(lucid.tensor([1.0]), linear)
        far = F.two_hot(lucid.tensor([100.0]), linear)
        flat = float((near - far).abs().max().item())

        near_log = F.two_hot(F.symlog(lucid.tensor([1.0])), self.GRID)
        far_log = F.two_hot(F.symlog(lucid.tensor([100.0])), self.GRID)
        separated = float((near_log - far_log).abs().max().item())

        # Measured: 0.002 against 1.0 — two orders of magnitude of
        # separation, from the transform alone.
        assert flat < 0.01
        assert separated > 0.5
        assert separated > 100.0 * flat

    def test_the_encoding_stays_a_distribution(self) -> None:
        values = lucid.tensor([[-1e5, -1.0, 0.0, 2.5, 1e5]])
        encoded = F.two_hot(F.symlog(values), self.GRID)
        assert float((encoded.sum(dim=-1) - 1.0).abs().max().item()) < 1e-5
