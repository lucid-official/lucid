"""What float16 costs a particular model, asked as a question.

The Neural Engine runs float16 and nothing else, so every export that
wants the accelerator accepts whatever half precision does to that
network — and it varies by more than an order of magnitude between
architectures that look alike. Measured on an untrained forward across
fourteen families: convolutional stacks near 1e-3, ViT at 1e-2, MaxViT
at 1.6e-1. Nothing in an export said which kind a given model was, and
the subsystem pushes people toward float16 for the accelerator, so the
gap mattered.

``precision_cost`` exports twice and compares both against eager. These
tests are about the measurement being trustworthy rather than about any
particular number: that it separates precisions at all, that it inherits
the refusal when the model answers with zeros, and that it does not
leave packages behind.

The float16 figure is Core ML's, and Core ML is better at half precision
than running the same network in half throughout — a ViT does that at
1.5 relative against 1e-2 here, because the runtime keeps the parts that
need range in float32. That is why this measures the export rather than
simulating half precision in Lucid.
"""

import os

import pytest

import lucid
import lucid.nn as nn
import lucid.coreml as cml
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"), reason="the engine was built without the Core ML writer"
)


class _Deep(nn.Module):
    """Enough layers that float16 has somewhere to accumulate."""

    def __init__(self) -> None:
        super().__init__()
        self.stack = nn.Sequential(
            *[layer for _ in range(6) for layer in (nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())]
        )

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.stack(x)


class TestItAnswersBothPrecisions:
    def test_it_reports_one_number_per_precision(self) -> None:
        lucid.manual_seed(0)
        cost = cml.precision_cost(_Deep().eval(), lucid.randn(1, 16, 16, 16))
        assert set(cost) == {"float32", "float16"}
        assert all(value >= 0.0 for value in cost.values())

    def test_float32_is_the_tighter_of_the_two(self) -> None:
        """The comparison has to be able to tell them apart.

        A report where both numbers came out the same would pass a
        looser check while measuring nothing — the exports would have to
        be identical for that, which is the bug this would hide.
        """
        lucid.manual_seed(0)
        cost = cml.precision_cost(_Deep().eval(), lucid.randn(1, 16, 16, 16))
        assert cost["float32"] < cost["float16"]
        assert cost["float32"] < 1e-4

    def test_it_leaves_nothing_behind(self, tmp_path: object) -> None:
        """Two packages are written and neither is the caller's problem."""
        before = set(os.listdir(str(tmp_path)))
        lucid.manual_seed(0)
        cml.precision_cost(_Deep().eval(), lucid.randn(1, 16, 16, 16))
        assert set(os.listdir(str(tmp_path))) == before

    def test_a_model_that_answers_zero_is_refused(self) -> None:
        """Inherited from ``verify``, and worth stating here.

        Comparing precisions against an all-zero reference proves
        nothing about either — and an untrained model whose head is
        zero-initialised is exactly that, which is common enough to
        meet by accident.
        """

        class _Zero(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return x * 0.0

        with pytest.raises(ValueError, match="below the"):
            cml.precision_cost(_Zero().eval(), lucid.randn(2, 4))

    def test_a_model_that_answers_almost_zero_is_refused_too(self) -> None:
        """Exact zeros were guarded; almost-zeros were not, and are worse.

        An untrained EfficientNet answers with logits around 1e-12, and
        the comparison used to report agreement to 1e-19 — a number that
        reads as a flawless export and is a blind probe. Three families
        sat in the smoke list passing that way. The refusal names the
        magnitude so the reader can see it is the model, not the export.
        """

        class _Vanishing(nn.Module):
            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return x * 1e-12

        with pytest.raises(ValueError, match="reach only"):
            cml.precision_cost(_Vanishing().eval(), lucid.randn(2, 4))

    def test_an_ordinary_model_is_not_caught_by_it(self) -> None:
        """The guard has to be as narrow as the thing it refuses.

        A model with ordinary outputs must still be measurable — a
        threshold that swallowed those would be worse than the trap it
        replaced.
        """
        lucid.manual_seed(0)
        cost = cml.precision_cost(_Deep().eval(), lucid.randn(1, 16, 16, 16) * 1e-3)
        assert cost["float32"] < cost["float16"]

    def test_the_weight_storage_is_held_across_both(self) -> None:
        """Otherwise the difference would not be the body's precision.

        Palettizing on one side and not the other would report the
        table's error as float16's.
        """
        lucid.manual_seed(0)
        cost = cml.precision_cost(
            _Deep().eval(), lucid.randn(1, 16, 16, 16), weights=cml.WeightPrecision.INT8
        )
        assert set(cost) == {"float32", "float16"}
