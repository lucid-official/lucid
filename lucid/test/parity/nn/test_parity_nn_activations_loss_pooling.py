"""Reference parity for nn / nn.functional."""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F
from lucid.test._helpers.compare import assert_close


@pytest.mark.parity
class TestActivationParity:
    @pytest.fixture
    def x_pair(self, ref: Any) -> tuple[lucid.Tensor, Any]:
        np.random.seed(0)
        x = np.random.uniform(-2.0, 2.0, size=(4, 5)).astype(np.float32)
        return lucid.tensor(x.copy()), ref.tensor(x.copy())

    def test_relu(self, x_pair, ref) -> None:  # type: ignore[no-untyped-def]
        l, r = x_pair
        assert_close(F.relu(l), ref.nn.functional.relu(r), atol=1e-6)

    def test_gelu(self, x_pair, ref) -> None:  # type: ignore[no-untyped-def]
        l, r = x_pair
        # ``gelu`` defaults to the exact formulation in both frameworks.
        assert_close(F.gelu(l), ref.nn.functional.gelu(r), atol=1e-4)

    def test_silu(self, x_pair, ref) -> None:  # type: ignore[no-untyped-def]
        l, r = x_pair
        assert_close(F.silu(l), ref.nn.functional.silu(r), atol=1e-5)

    def test_sigmoid(self, x_pair, ref) -> None:  # type: ignore[no-untyped-def]
        l, r = x_pair
        assert_close(l.sigmoid(), r.sigmoid(), atol=1e-5)

    def test_tanh(self, x_pair, ref) -> None:  # type: ignore[no-untyped-def]
        l, r = x_pair
        assert_close(l.tanh(), r.tanh(), atol=1e-5)

    def test_softmax(self, x_pair, ref) -> None:  # type: ignore[no-untyped-def]
        l, r = x_pair
        assert_close(F.softmax(l, dim=1), ref.nn.functional.softmax(r, dim=1), atol=1e-6)

    def test_log_softmax(self, x_pair, ref) -> None:  # type: ignore[no-untyped-def]
        l, r = x_pair
        assert_close(F.log_softmax(l, dim=1), ref.nn.functional.log_softmax(r, dim=1), atol=1e-5)


@pytest.mark.parity
class TestLossParity:
    def test_mse(self, ref: Any) -> None:
        np.random.seed(0)
        x = np.random.standard_normal(size=(4, 5)).astype(np.float32)
        y = np.random.standard_normal(size=(4, 5)).astype(np.float32)
        l = F.mse_loss(lucid.tensor(x.copy()), lucid.tensor(y.copy()))
        r = ref.nn.functional.mse_loss(ref.tensor(x.copy()), ref.tensor(y.copy()))
        assert_close(l, r, atol=1e-5)

    def test_l1(self, ref: Any) -> None:
        np.random.seed(0)
        x = np.random.standard_normal(size=(4, 5)).astype(np.float32)
        y = np.random.standard_normal(size=(4, 5)).astype(np.float32)
        l = F.l1_loss(lucid.tensor(x.copy()), lucid.tensor(y.copy()))
        r = ref.nn.functional.l1_loss(ref.tensor(x.copy()), ref.tensor(y.copy()))
        assert_close(l, r, atol=1e-5)

    def test_cross_entropy(self, ref: Any) -> None:
        np.random.seed(0)
        x = np.random.standard_normal(size=(8, 5)).astype(np.float32)
        y = np.random.randint(0, 5, size=(8,)).astype(np.int64)
        l = F.cross_entropy(lucid.tensor(x.copy()), lucid.tensor(y.copy()))
        r = ref.nn.functional.cross_entropy(ref.tensor(x.copy()), ref.tensor(y.copy()))
        assert_close(l, r, atol=1e-4)


@pytest.mark.parity
class TestPoolingParity:
    def test_max_pool2d(self, ref: Any) -> None:
        np.random.seed(0)
        x = np.random.standard_normal(size=(1, 3, 8, 8)).astype(np.float32)
        l = F.max_pool2d(lucid.tensor(x.copy()), kernel_size=2)
        r = ref.nn.functional.max_pool2d(ref.tensor(x.copy()), kernel_size=2)
        assert_close(l, r, atol=1e-6)

    def test_avg_pool2d(self, ref: Any) -> None:
        np.random.seed(0)
        x = np.random.standard_normal(size=(1, 3, 8, 8)).astype(np.float32)
        l = F.avg_pool2d(lucid.tensor(x.copy()), kernel_size=2)
        r = ref.nn.functional.avg_pool2d(ref.tensor(x.copy()), kernel_size=2)
        assert_close(l, r, atol=1e-6)

    def test_fractional_max_pool2d(self, ref: Any) -> None:
        np.random.seed(42)
        x = np.random.standard_normal(size=(2, 3, 8, 8)).astype(np.float32)
        samples = np.random.uniform(0.0, 1.0, size=(2, 3, 2)).astype(np.float32)
        lucid_out = F.fractional_max_pool2d(
            lucid.tensor(x.copy()),
            kernel_size=2,
            output_size=4,
            _random_samples=lucid.tensor(samples.copy()),
        )
        ref_out = ref.nn.functional.fractional_max_pool2d(
            ref.tensor(x.copy()),
            kernel_size=2,
            output_size=4,
            _random_samples=ref.tensor(samples.copy()),
        )
        assert_close(lucid_out, ref_out, atol=1e-5)

    def test_fractional_max_pool2d_return_indices(self, ref: Any) -> None:
        np.random.seed(7)
        x = np.random.standard_normal(size=(1, 2, 6, 6)).astype(np.float32)
        samples = np.random.uniform(0.0, 1.0, size=(1, 2, 2)).astype(np.float32)
        l_out, l_idx = F.fractional_max_pool2d(  # type: ignore[misc]
            lucid.tensor(x.copy()),
            kernel_size=2,
            output_size=3,
            return_indices=True,
            _random_samples=lucid.tensor(samples.copy()),
        )
        r_result = ref.nn.functional.fractional_max_pool2d(
            ref.tensor(x.copy()),
            kernel_size=2,
            output_size=3,
            return_indices=True,
            _random_samples=ref.tensor(samples.copy()),
        )
        assert_close(l_out, r_result[0], atol=1e-5)
        np.testing.assert_array_equal(l_idx.numpy(), r_result[1].numpy())

    def test_fractional_max_pool3d(self, ref: Any) -> None:
        np.random.seed(13)
        x = np.random.standard_normal(size=(1, 2, 8, 8, 8)).astype(np.float32)
        samples = np.random.uniform(0.0, 1.0, size=(1, 2, 3)).astype(np.float32)
        lucid_out = F.fractional_max_pool3d(
            lucid.tensor(x.copy()),
            kernel_size=2,
            output_size=4,
            _random_samples=lucid.tensor(samples.copy()),
        )
        ref_out = ref.nn.functional.fractional_max_pool3d(
            ref.tensor(x.copy()),
            kernel_size=2,
            output_size=4,
            _random_samples=ref.tensor(samples.copy()),
        )
        assert_close(lucid_out, ref_out, atol=1e-5)
