"""Unit and reference-parity tests for ``lucid.nn.init`` (LCD-187)."""

from collections.abc import Callable
import math
import warnings

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.init as init
from lucid.test._fixtures.devices import metal_available

DEVICES: list[str] = ["cpu", "metal"] if metal_available() else ["cpu"]


class TestViewsInPlace:
    """In-place initialisers mutate the base tensor when called on CPU or Metal views."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_constant_slice_view(self, device: str) -> None:
        b = nn.Parameter(lucid.zeros(8, device=device))
        out = init.constant_(b[2:4], 1.0)
        assert out is not b
        assert b.tolist() == [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

    @pytest.mark.parametrize("device", DEVICES)
    def test_ones_and_zeros_column_view(self, device: str) -> None:
        w = nn.Parameter(lucid.zeros(3, 4, device=device))
        init.ones_(w[:, 0])
        assert w[:, 0].tolist() == [1.0, 1.0, 1.0]
        assert w[:, 1:].tolist() == [[0.0, 0.0, 0.0]] * 3
        init.zeros_(w[:, 0])
        assert w[:, 0].tolist() == [0.0, 0.0, 0.0]

    @pytest.mark.parametrize("device", DEVICES)
    def test_normal_row_view(self, device: str) -> None:
        w = nn.Parameter(lucid.zeros(3, 8, device=device))
        init.normal_(w[1], mean=5.0, std=0.1)
        assert w[0].tolist() == [0.0] * 8
        assert w[2].tolist() == [0.0] * 8
        assert all(v > 3.0 for v in w[1].tolist())

    @pytest.mark.parametrize("device", DEVICES)
    def test_uniform_transpose_view(self, device: str) -> None:
        w = nn.Parameter(lucid.zeros(2, 4, device=device))
        init.uniform_(w.T, a=2.0, b=3.0)
        vals = w.reshape(-1).tolist()
        assert all(2.0 <= v <= 3.0 for v in vals)

    @pytest.mark.parametrize("device", DEVICES)
    def test_eye_submatrix_view(self, device: str) -> None:
        w = nn.Parameter(lucid.zeros(4, 4, device=device))
        init.eye_(w[:2, :2])
        assert w[:2, :2].tolist() == [[1.0, 0.0], [0.0, 1.0]]
        assert w[2:, :].tolist() == [[0.0] * 4, [0.0] * 4]

    @pytest.mark.parametrize("device", DEVICES)
    def test_dirac_subtensor_view(self, device: str) -> None:
        w = nn.Parameter(lucid.zeros(4, 4, 3, 3, device=device))
        init.dirac_(w[:2, :2])
        np.testing.assert_array_equal(w[:2, :2, 1, 1].numpy(), np.eye(2))
        assert float(w[2:].abs().sum().item()) == 0.0

    @pytest.mark.parametrize("device", DEVICES)
    def test_orthogonal_slice_view(self, device: str) -> None:
        w = nn.Parameter(lucid.zeros(4, 4, device=device))
        init.orthogonal_(w[:2, :])
        sub = w[:2, :].numpy()
        np.testing.assert_allclose(sub @ sub.T, np.eye(2), atol=1e-4)
        assert float(w[2:].abs().sum().item()) == 0.0

    @pytest.mark.parametrize("device", DEVICES)
    def test_sparse_slice_view(self, device: str) -> None:
        w = nn.Parameter(lucid.ones(6, 4, device=device))
        init.sparse_(w[:, :2], sparsity=0.5, std=0.1)
        sub = w[:, :2].numpy()
        assert ((sub == 0.0).sum(axis=0) == 3).all()
        assert (w[:, 2:].numpy() == 1.0).all()

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize(
        "fn",
        [
            init.xavier_uniform_,
            init.xavier_normal_,
            init.kaiming_uniform_,
            init.kaiming_normal_,
            lambda t: init.trunc_normal_(t, mean=2.0, std=0.1, a=1.5, b=2.5),
        ],
    )
    def test_variance_scaling_and_trunc_normal_on_slice_view(
        self, device: str, fn: Callable[[lucid.Tensor], lucid.Tensor]
    ) -> None:
        w = nn.Parameter(lucid.zeros(4, 8, device=device))
        fn(w[:2, :])
        assert float(w[:2, :].abs().sum().item()) > 0.0
        assert float(w[2:, :].abs().sum().item()) == 0.0


class TestOrthogonalHaar:
    """``orthogonal_`` applies ``sign(diag(R))`` so Q is Haar-distributed."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("shape", [(4, 4), (8, 4), (4, 8), (3, 2, 2)])
    def test_orthogonal_haar_sign_and_orthonormality(
        self, device: str, shape: tuple[int, ...]
    ) -> None:
        pos_q00 = 0
        trials = 40
        for seed in range(trials):
            g = lucid.Generator(seed)
            w = lucid.empty(*shape, device=device)
            init.orthogonal_(w, gain=1.5, generator=g)
            flat = w.reshape(shape[0], -1).numpy()
            if flat.shape[0] >= flat.shape[1]:
                np.testing.assert_allclose(
                    flat.T @ flat, (1.5**2) * np.eye(flat.shape[1]), atol=1e-3
                )
            else:
                np.testing.assert_allclose(
                    flat @ flat.T, (1.5**2) * np.eye(flat.shape[0]), atol=1e-3
                )
            if float(flat[0, 0]) > 0.0:
                pos_q00 += 1
        assert 8 <= pos_q00 <= trials - 8


class TestZeroElementTensors:
    """Zero-element tensors and 0-channel layers do not crash on initialisation."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_zero_channel_linear_and_conv2d(self, device: str) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lin = nn.Linear(0, 4, device=device)
            conv = nn.Conv2d(0, 4, 3, device=device)
        assert lin.weight.shape == (4, 0)
        assert lin.bias is not None and lin.bias.tolist() == [0.0, 0.0, 0.0, 0.0]
        assert conv.weight.shape == (4, 0, 3, 3)
        assert conv.bias is not None and conv.bias.tolist() == [0.0, 0.0, 0.0, 0.0]
        assert any("zero-element" in str(w.message) for w in caught)

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize(
        "fn",
        [init.kaiming_uniform_, init.kaiming_normal_, init.xavier_uniform_, init.xavier_normal_],
    )
    @pytest.mark.parametrize("shape", [(0, 4), (4, 0), (0, 0), (2, 2, 0)])
    def test_fan_initializers_zero_numel_warn_and_noop(
        self, device: str, fn: Callable[[lucid.Tensor], lucid.Tensor], shape: tuple[int, ...]
    ) -> None:
        t = lucid.zeros(*shape, device=device)
        with pytest.warns(UserWarning, match="Initializing zero-element tensors is a no-op"):
            out = fn(t)
        assert out is t
        assert out.shape == shape

    @pytest.mark.parametrize("device", DEVICES)
    def test_other_initializers_zero_numel_noop(self, device: str) -> None:
        for shape in [(0, 4), (4, 0), (0, 0)]:
            t2 = lucid.zeros(*shape, device=device)
            assert init.orthogonal_(t2) is t2
            assert init.sparse_(t2, 0.5) is t2
            assert init.eye_(t2) is t2
            assert init.trunc_normal_(t2) is t2
            assert init.uniform_(t2) is t2
            assert init.normal_(t2) is t2
            assert init.constant_(t2, 1.0) is t2
            assert init.ones_(t2) is t2
            assert init.zeros_(t2) is t2

        for shape3 in [(0, 4, 3), (4, 0, 3, 3), (4, 4, 0, 3)]:
            t3 = lucid.zeros(*shape3, device=device)
            assert init.dirac_(t3) is t3


class TestTruncNormal:
    """``trunc_normal_`` validates bounds, warns on far mean, and samples via erfinv."""

    @pytest.mark.parametrize("device", DEVICES)
    def test_inverted_bounds_raise_immediately(self, device: str) -> None:
        t = lucid.empty(8, device=device)
        with pytest.raises((ValueError, RuntimeError)):
            init.trunc_normal_(t, mean=0.0, std=1.0, a=2.0, b=1.0)

    @pytest.mark.parametrize("device", DEVICES)
    def test_equal_bounds_fill_constant(self, device: str) -> None:
        t = lucid.empty(8, device=device)
        init.trunc_normal_(t, mean=0.0, std=1.0, a=1.25, b=1.25)
        assert t.tolist() == [1.25] * 8

    @pytest.mark.parametrize("device", DEVICES)
    def test_far_tails_warn_and_stay_within_bounds(self, device: str) -> None:
        t = lucid.empty(64, device=device)
        with pytest.warns(UserWarning, match="mean is more than 2 std"):
            init.trunc_normal_(t, mean=0.0, std=1.0, a=7.0, b=8.0)
        vals = t.tolist()
        assert all(7.0 <= v <= 8.0 for v in vals)
        if device == "cpu":
            assert any(7.0 < v < 8.0 for v in vals)

        with pytest.warns(UserWarning, match="mean is more than 2 std"):
            init.trunc_normal_(t, mean=0.0, std=1.0, a=-8.0, b=-7.0)
        vals_neg = t.tolist()
        assert all(-8.0 <= v <= -7.0 for v in vals_neg)


class TestGeneratorAndGainAndSparse:
    """``generator=`` keyword, ``calculate_gain`` parity, and ``sparse_`` ceil count."""

    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize(
        "call",
        [
            lambda t, g: init.uniform_(t, -1.0, 1.0, generator=g),
            lambda t, g: init.normal_(t, 0.0, 1.0, generator=g),
            lambda t, g: init.xavier_uniform_(t, generator=g),
            lambda t, g: init.xavier_normal_(t, generator=g),
            lambda t, g: init.kaiming_uniform_(t, generator=g),
            lambda t, g: init.kaiming_normal_(t, generator=g),
            lambda t, g: init.trunc_normal_(t, generator=g),
            lambda t, g: init.orthogonal_(t, generator=g),
            lambda t, g: init.sparse_(t, 0.5, generator=g),
        ],
    )
    def test_generator_reproducibility(
        self, device: str, call: Callable[[lucid.Tensor, lucid.Generator], lucid.Tensor]
    ) -> None:
        a = lucid.empty(8, 8, device=device)
        b = lucid.empty(8, 8, device=device)
        call(a, lucid.Generator(1234))
        call(b, lucid.Generator(1234))
        np.testing.assert_array_equal(a.numpy(), b.numpy())

    def test_calculate_gain_conv_transpose_and_bool_rejection(self) -> None:
        for name in ("conv_transpose1d", "conv_transpose2d", "conv_transpose3d"):
            assert init.calculate_gain(name) == 1.0
        with pytest.raises(ValueError, match="not a valid number"):
            init.calculate_gain("leaky_relu", True)
        with pytest.raises(ValueError, match="not a valid number"):
            init.calculate_gain("leaky_relu", False)
        assert math.isclose(init.calculate_gain("leaky_relu", 0), math.sqrt(2.0), rel_tol=1e-6)

    @pytest.mark.parametrize("device", DEVICES)
    def test_sparse_uses_ceil_zeros_per_column(self, device: str) -> None:
        t = lucid.empty(10, 6, device=device)
        init.sparse_(t, sparsity=0.25)
        zeros_per_col = (t.numpy() == 0.0).sum(axis=0)
        assert (zeros_per_col == math.ceil(0.25 * 10)).all()
