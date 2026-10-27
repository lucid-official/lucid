"""The distributional regression head, and what it is actually for.

The motivation is easy to state and easy to test badly. "A squared error
scales with the target" suggests a head fitted to large rewards should
fail outright, and with a plain optimiser it would — but Adam normalises
per parameter, and a squared-error head fitted to a *single* large
constant does fine. Measured: 0.6% relative error at a target of 100000,
against the two-hot's 0.00%. Better, not decisive.

Where it becomes decisive is one head covering several magnitudes at
once, which is the real case: a reward head sees the whole range inside a
single batch. There the squared error is dominated by the large entries
and the small ones are simply not fitted. That is the comparison below,
and the numbers in it are measured rather than assumed.
"""

import pytest

import lucid
import lucid.optim as optim
from lucid.models.generative._common._pixel_nets import DenseHead
from lucid.models.generative.dreamer_v3._heads import TwoHotHead


def _fit(head: object, inputs: lucid.Tensor, targets: lucid.Tensor, steps: int) -> None:
    optimiser = optim.Adam(head.parameters(), lr=3e-2)
    for _ in range(steps):
        if isinstance(head, TwoHotHead):
            loss = head.loss(inputs, targets)
        else:
            loss = (0.5 * (head(inputs) - targets) ** 2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()


class TestTwoHotHead:
    def test_shapes(self) -> None:
        head = TwoHotHead(8, 16, 2, num_bins=41)
        feature = lucid.zeros((2, 3, 8))
        assert head(feature).shape == (2, 3, 41)
        assert head.predict(feature).shape == (2, 3)

    def test_zero_init_predicts_zero(self) -> None:
        """A critic that starts by promising returns sends the actor after them.

        Not *exactly* zero at the default 255 bins.  A uniform
        distribution over a grid symmetric about the origin has mean zero
        in exact arithmetic, but the float32 sum of 255 terms leaves about
        1e-7 behind — at 41 bins the cancellation happened to be exact,
        which made the earlier equality test a statement about summation
        order rather than about the head.
        """
        head = TwoHotHead(8, 16, 2, zero_init=True)
        assert float(head.predict(lucid.randn((4, 5, 8))).abs().max().item()) < 1e-5

    def test_without_zero_init_it_does_not(self) -> None:
        """Guards the test above — otherwise it would pass on any head."""
        lucid.manual_seed(0)
        head = TwoHotHead(8, 16, 2, zero_init=False)
        assert float(head.predict(lucid.randn((4, 5, 8))).abs().max().item()) > 1e-3

    def test_the_loss_is_a_cross_entropy(self) -> None:
        """Non-negative, and minimised when the logits match the encoding."""
        head = TwoHotHead(4, 16, 2, zero_init=True)
        inputs = lucid.randn((16, 1, 4))
        targets = lucid.ones((16, 1)) * 5.0
        before = float(head.loss(inputs, targets).item())
        _fit(head, inputs, targets, 200)
        after = float(head.loss(inputs, targets).item())
        assert after >= 0.0 and after < before

    def test_the_target_is_not_differentiated(self) -> None:
        """It is data; a head able to move its own target fits nothing."""
        head = TwoHotHead(4, 16, 2)
        targets = lucid.ones((8, 1), requires_grad=True)
        head.loss(lucid.randn((8, 1, 4)), targets).backward()
        assert targets.grad is None or float(targets.grad.abs().sum().item()) == 0.0

    @pytest.mark.parametrize("scale", [0.01, 1.0, 1000.0, 100000.0])
    def test_one_configuration_fits_any_scale(self, scale: float) -> None:
        """The paper's claim, at four magnitudes, with the rate held fixed."""
        lucid.manual_seed(0)
        head = TwoHotHead(4, 32, 2, zero_init=True)
        inputs = lucid.randn((64, 1, 4))
        targets = lucid.ones((64, 1)) * scale
        _fit(head, inputs, targets, 300)
        predicted = float(head.predict(inputs).mean().item())
        assert abs(predicted - scale) / scale < 0.01

    @pytest.mark.parametrize("bins", [1, 0])
    def test_rejects_a_grid_too_small_to_interpolate(self, bins: int) -> None:
        with pytest.raises(ValueError):
            TwoHotHead(4, 8, 1, num_bins=bins)

    def test_rejects_a_degenerate_range(self) -> None:
        with pytest.raises(ValueError):
            TwoHotHead(4, 8, 1, bin_range=0.0)


class TestAgainstSquaredError:
    """Why the head exists, measured against the thing it replaces."""

    @staticmethod
    def _mixed() -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor]:
        """Half the batch wants 0.01, half wants 10000."""
        lucid.manual_seed(0)
        inputs = lucid.randn((256, 1, 4))
        large = (inputs[..., 0] > 0).to(lucid.float32)
        return inputs, large * 10000.0 + (1.0 - large) * 0.01, large

    def _errors(self, head: object, steps: int = 600) -> tuple[float, float]:
        inputs, targets, large = self._mixed()
        _fit(head, inputs, targets, steps)
        with lucid.no_grad():
            predicted = head.predict(inputs) if isinstance(head, TwoHotHead) else head(inputs)
            small_error = (
                float(
                    ((predicted - 0.01).abs() * (1.0 - large)).sum().item()
                    / float((1.0 - large).sum().item())
                )
                / 0.01
            )
            large_error = (
                float(
                    ((predicted - 10000.0).abs() * large).sum().item() / float(large.sum().item())
                )
                / 10000.0
            )
        return small_error, large_error

    def test_two_hot_fits_both_magnitudes(self) -> None:
        lucid.manual_seed(0)
        small, large = self._errors(TwoHotHead(4, 64, 2, zero_init=True))
        # Measured: 0.27 and 0.002.
        assert small < 1.0
        assert large < 0.05

    def test_squared_error_abandons_the_small_one(self) -> None:
        """The failure the transform exists to prevent, shown rather than argued.

        The loss is dominated by the entries near 10000, so the head
        predicts tens where the target is hundredths — measured at 4400x
        the target's own magnitude, against the two-hot's 0.27.
        """
        lucid.manual_seed(0)
        head = DenseHead(4, 64, 2, out_features=1, act_fn="silu", squeeze=True)
        small, large = self._errors(head)
        assert small > 100.0
        assert large < 0.05, "it fits the large one fine — that is the point"

    def test_a_single_large_target_is_not_enough_to_tell_them_apart(self) -> None:
        """Guards the framing above.

        With Adam, a squared-error head handles one large constant
        perfectly well. The mixed batch is what separates them, and a test
        using a single scale would have quietly proved nothing.
        """
        lucid.manual_seed(0)
        head = DenseHead(4, 32, 2, out_features=1, act_fn="silu", squeeze=True)
        inputs = lucid.randn((64, 1, 4))
        targets = lucid.ones((64, 1)) * 100000.0
        _fit(head, inputs, targets, 300)
        with lucid.no_grad():
            error = abs(float(head(inputs).mean().item()) - 100000.0) / 100000.0
        assert error < 0.05
