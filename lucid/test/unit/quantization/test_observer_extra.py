"""``lucid.quantization`` Phase-5 — observer completeness."""

import numpy as np
import pytest

import lucid
import lucid.quantization as Q


class TestMovingAveragePerChannel:
    def test_per_channel_scale_shape(self) -> None:
        lucid.manual_seed(0)
        obs = Q.MovingAveragePerChannelMinMaxObserver(ch_axis=0)
        for _ in range(6):
            obs(lucid.randn(4, 16))
        scale, zero_point = obs.calculate_qparams()
        assert scale.shape == (4,)
        assert bool((scale > 0).all().item())

    def test_ema_smooths_outlier(self) -> None:
        lucid.manual_seed(1)
        obs = Q.MovingAveragePerChannelMinMaxObserver(ch_axis=0, averaging_constant=0.1)
        for _ in range(5):
            obs(lucid.randn(3, 8))
        s_pre, _ = obs.calculate_qparams()
        obs(lucid.ones(3, 8) * 50.0)  # one outlier batch
        s_post, _ = obs.calculate_qparams()
        # EMA (0.1) moves the range only partway toward the spike, not fully.
        assert bool((s_post < 50.0).all().item())
        assert bool((s_post >= s_pre).all().item())


class TestFixedQParams:
    def test_fixed_scale_zero_point(self) -> None:
        obs = Q.FixedQParamsObserver(scale=1.0 / 256.0, zero_point=0)
        obs(lucid.randn(3, 3))  # identity, no stats
        scale, zero_point = obs.calculate_qparams()
        assert np.isclose(float(scale.item()), 1.0 / 256.0)
        assert float(zero_point.item()) == 0.0

    def test_ignores_data(self) -> None:
        obs = Q.FixedQParamsObserver(scale=0.5, zero_point=128)
        s0, _ = obs.calculate_qparams()
        obs(lucid.randn(10, 10) * 1000.0)
        s1, _ = obs.calculate_qparams()
        assert float(s0.item()) == float(s1.item())  # unaffected by data


class TestPlaceholderNoop:
    def test_placeholder_identity_and_raises(self) -> None:
        obs = Q.PlaceholderObserver()
        x = lucid.randn(2, 2)
        assert np.allclose(obs(x).numpy(), x.numpy())
        with pytest.raises(RuntimeError, match="no statistics"):
            obs.calculate_qparams()

    def test_noop_is_placeholder(self) -> None:
        assert issubclass(Q.NoopObserver, Q.PlaceholderObserver)
        assert np.allclose(Q.NoopObserver()(lucid.ones(3)).numpy(), lucid.ones(3).numpy())
