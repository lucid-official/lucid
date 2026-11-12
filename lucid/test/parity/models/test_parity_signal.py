"""The harness puts signal into a parity comparison, and refuses one that has none (LCD-960)."""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.test._helpers.compare import assert_close
from lucid.test.parity.models import _utils


class TestRequireSignal:
    """``_require_signal`` fails -- it never skips -- a comparison that cannot discriminate."""

    def test_a_zero_reference_answer_fails(self) -> None:
        with pytest.raises(pytest.fail.Exception, match="all-zeros"):
            _utils._require_signal("zero_head", 0.0, 5e-3)

    def test_a_reference_answer_inside_the_tolerance_fails(self) -> None:
        with pytest.raises(pytest.fail.Exception, match="under the 100x"):
            _utils._require_signal("mobilenet", 1.9e-9, 1e-3)

    def test_the_ratio_is_what_is_judged_not_the_atol_alone(self) -> None:
        # 0.05 is 50x its own tolerance of 1e-3: nearly vacuous, not merely small.
        with pytest.raises(pytest.fail.Exception, match="50x"):
            _utils._require_signal("small", 0.05, 1e-3)

    def test_an_answer_past_the_threshold_passes(self) -> None:
        _utils._require_signal("live", 0.11, 1e-3)

    def test_a_runaway_answer_fails_too(self) -> None:
        with pytest.raises(pytest.fail.Exception, match="float32"):
            _utils._require_signal("runaway", 5e3, 1.0)


def _zeroed_linear() -> nn.Linear:
    layer = nn.Linear(8, 4)
    with lucid.no_grad():
        layer.weight.mul_(0.0)
    return layer


class TestWakeZeroParameters:
    def test_zero_tensors_get_values_and_the_rest_are_left_alone(self) -> None:
        layer = _zeroed_linear()
        kept = layer.bias.numpy().copy()
        assert kept.any(), "the bias of this fixture must start non-zero"

        woken = _utils._wake_zero_parameters(layer)

        assert woken == 1
        assert layer.weight.numpy().any()
        assert_close(layer.bias, kept, atol=0.0, rtol=0.0)

    def test_it_is_deterministic(self) -> None:
        first, second = _zeroed_linear(), _zeroed_linear()
        _utils._wake_zero_parameters(first)
        _utils._wake_zero_parameters(second)
        assert_close(first.weight, second.weight, atol=0.0, rtol=0.0)


class TestCalibrateNormStatistics:
    def test_statistics_are_those_of_the_calibration_batch(self) -> None:
        conv, norm = nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4)
        model = nn.Sequential(conv, norm, nn.Dropout(0.5)).eval()
        x = np.random.default_rng(0).standard_normal((1, 3, 8, 8)).astype(np.float32)
        extra = np.random.default_rng(_utils._CALIBRATION_SEED).standard_normal(
            (_utils._CALIBRATION_EXTRA, 3, 8, 8)
        )
        batch = lucid.from_numpy(np.concatenate([x, extra.astype(np.float32)]))

        assert _utils._calibrate_norm_statistics(model, x) == 1

        features = conv(batch).numpy()
        assert_close(norm.running_mean, features.mean(axis=(0, 2, 3)), atol=1e-5, rtol=1e-4)
        assert_close(norm.running_var, features.var(axis=(0, 2, 3), ddof=1), atol=1e-5, rtol=1e-4)

    def test_modes_and_momentum_are_restored(self) -> None:
        norm = nn.BatchNorm2d(4, momentum=0.3)
        model = nn.Sequential(nn.Conv2d(3, 4, 3), norm, nn.Dropout(0.5)).eval()
        _utils._calibrate_norm_statistics(model, np.ones((1, 3, 8, 8), dtype=np.float32))

        assert norm.momentum == 0.3
        assert not any(m.training for m in model.modules())

    def test_a_model_with_no_norm_is_left_alone(self) -> None:
        model = nn.Sequential(nn.Linear(3, 3)).eval()
        assert _utils._calibrate_norm_statistics(model, np.ones((1, 3), dtype=np.float32)) == 0
