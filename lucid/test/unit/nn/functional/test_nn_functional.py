"""nn.functional — numerical correctness against closed-form refs."""

import math

import numpy as np
import pytest

import lucid
import lucid.nn.functional as F


class TestActivationsF:
    def test_relu(self) -> None:
        out = F.relu(lucid.tensor([-1.0, 0.0, 1.0])).numpy()
        np.testing.assert_array_equal(out, [0.0, 0.0, 1.0])

    def test_softmax_uniform(self) -> None:
        out = F.softmax(lucid.tensor([[1.0, 1.0, 1.0]]), dim=1).numpy()
        np.testing.assert_allclose(out, [[1 / 3] * 3], atol=1e-6)

    def test_log_softmax_sum_exp_one(self) -> None:
        out = F.log_softmax(lucid.tensor([[1.0, 1.0, 1.0]]), dim=1).numpy()
        # exp of log_softmax should sum to 1.
        assert abs(np.exp(out).sum() - 1.0) < 1e-6

    def test_log_softmax_stable_at_large_logits(self) -> None:
        # Regression: log_softmax must route through the engine's max-subtracting
        # kernel, not the naive log(softmax(x)) — the latter underflows to
        # log(0) = -inf once a non-max logit's softmax rounds to 0.
        x = lucid.tensor([[1000.0, 995.0, -1000.0]])
        out = F.log_softmax(x, dim=1).numpy()
        assert np.isfinite(out).all()
        # closed form: row max is 1000, so the top entry ≈ -log(1 + e^-5 + e^-2000)
        np.testing.assert_allclose(out[0, 0], -math.log1p(math.exp(-5.0)), atol=1e-3)

    def test_cross_entropy_stable_at_large_logits(self) -> None:
        # Regression: cross_entropy (log_softmax + NLL) stays finite in both the
        # forward loss and the gradient even when logits are far outside [-88, 88].
        rng = np.random.default_rng(0)
        xn = (rng.standard_normal((8, 10)).astype("float32")) * 200.0
        x = lucid.tensor(xn)
        x.requires_grad = True
        y = lucid.tensor(rng.integers(0, 10, size=(8,)).astype("int64")).long()
        loss = F.cross_entropy(x, y)
        loss.backward()
        assert np.isfinite(float(loss.item()))
        assert np.isfinite(x.grad.numpy()).all()

    def test_gelu(self) -> None:
        # gelu(0) = 0.
        assert abs(F.gelu(lucid.tensor([0.0])).item()) < 1e-6

    def test_silu(self) -> None:
        # silu(x) = x * sigmoid(x); silu(0) = 0.
        assert abs(F.silu(lucid.tensor([0.0])).item()) < 1e-6

    def test_softplus(self) -> None:
        # softplus(0) = log 2.
        assert abs(F.softplus(lucid.tensor([0.0])).item() - math.log(2.0)) < 1e-5


class TestPooling:
    def test_max_pool2d_shape(self) -> None:
        out = F.max_pool2d(lucid.zeros(1, 1, 4, 4), kernel_size=2)
        assert out.shape == (1, 1, 2, 2)

    def test_avg_pool2d_value(self) -> None:
        # Average of [[1, 1], [1, 1]] over 2x2 → 1.0.
        x = lucid.ones(1, 1, 2, 2)
        out = F.avg_pool2d(x, kernel_size=2)
        assert out.item() == 1.0


class TestNormalization:
    def test_layer_norm_unit_variance(self) -> None:
        # After layer_norm the output dim has zero mean / unit variance.
        x = lucid.tensor([[1.0, 2.0, 3.0, 4.0]])
        gamma = lucid.ones(4)
        beta = lucid.zeros(4)
        out = F.layer_norm(x, [4], weight=gamma, bias=beta).numpy()
        assert abs(out.mean()) < 1e-5
        assert abs(out.std() - 1.0) < 1e-2  # bessel correction in some impls.


class TestLossesF:
    def test_mse(self) -> None:
        x = lucid.tensor([1.0, 2.0])
        y = lucid.tensor([2.0, 4.0])
        assert abs(F.mse_loss(x, y).item() - 2.5) < 1e-6

    def test_cross_entropy_uniform(self) -> None:
        x = lucid.tensor([[1.0, 1.0, 1.0]])
        y = lucid.tensor([0], dtype=lucid.int64)
        assert abs(F.cross_entropy(x, y).item() - math.log(3.0)) < 1e-5


class TestGumbelSoftmax:
    def test_soft_sums_to_one(self) -> None:
        out = F.gumbel_softmax(lucid.tensor([[1.0, 2.0, 3.0]]), tau=1.0, hard=False).numpy()
        assert abs(out.sum() - 1.0) < 1e-5

    def test_hard_one_hot(self) -> None:
        out = F.gumbel_softmax(lucid.tensor([[1.0, 2.0, 3.0]]), tau=1.0, hard=True).numpy()
        assert out.sum() == 1.0


class TestOneHot:
    def test_answers_int64(self, device: str) -> None:
        # The reference framework's dtype; this returned the engine's int8.
        out = F.one_hot(lucid.tensor([0, 2, 1], device=device), num_classes=3)
        assert out.dtype == lucid.int64
        expected = np.eye(3, dtype=np.int64)[[0, 2, 1]]
        np.testing.assert_array_equal(out.numpy(), expected)

    def test_infers_num_classes_from_the_largest_index(self) -> None:
        # ``-1`` is the documented default, and the engine refused it.
        out = F.one_hot(lucid.tensor([0, 3, 1]))
        assert out.shape == (3, 4)
        assert out.numpy()[1].tolist() == [0, 0, 0, 1]

    def test_cannot_infer_from_an_empty_tensor(self) -> None:
        with pytest.raises(ValueError, match="num_classes"):
            F.one_hot(lucid.tensor([], dtype=lucid.int64))


class TestTripletWithDistance:
    def test_zero_when_dpos_zero(self) -> None:
        a = lucid.tensor([[1.0, 0.0]])
        p = lucid.tensor([[1.0, 0.0]])
        n = lucid.tensor([[0.0, 1.0]])
        assert abs(F.triplet_margin_with_distance_loss(a, p, n, margin=1.0).item()) < 1e-6


class TestFractionalPool:
    def test_frac_pool2d_output_size_shape(self) -> None:
        x = lucid.ones(2, 3, 8, 8)
        out = F.fractional_max_pool2d(x, kernel_size=2, output_size=4)
        assert tuple(out.shape) == (2, 3, 4, 4)

    def test_frac_pool2d_output_ratio_shape(self) -> None:
        x = lucid.ones(1, 2, 8, 8)
        out = F.fractional_max_pool2d(x, kernel_size=2, output_ratio=0.5)
        assert tuple(out.shape) == (1, 2, 4, 4)

    def test_frac_pool2d_deterministic_value(self) -> None:
        # sample=0 → alpha=2 → starts=[0, 2] for both H and W
        # x = arange(16) reshaped to (1,1,4,4):
        #   [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]
        # pool[0,0]=max(0,1,4,5)=5  pool[0,1]=max(2,3,6,7)=7
        # pool[1,0]=max(8,9,12,13)=13  pool[1,1]=max(10,11,14,15)=15
        x = lucid.arange(16, dtype=lucid.float32).reshape(1, 1, 4, 4)
        samples = lucid.zeros(1, 1, 2)
        out = F.fractional_max_pool2d(x, kernel_size=2, output_size=2, _random_samples=samples)
        np.testing.assert_allclose(out.numpy().flatten(), [5.0, 7.0, 13.0, 15.0], atol=1e-6)

    def test_frac_pool2d_return_indices_shape(self) -> None:
        x = lucid.randn(1, 1, 6, 6)
        out, idx = F.fractional_max_pool2d(  # type: ignore[misc]
            x, kernel_size=2, output_size=3, return_indices=True
        )
        assert tuple(out.shape) == (1, 1, 3, 3)
        assert tuple(idx.shape) == (1, 1, 3, 3)

    def test_frac_pool2d_indices_in_valid_range(self) -> None:
        x = lucid.randn(1, 2, 6, 6)
        _, idx = F.fractional_max_pool2d(  # type: ignore[misc]
            x, kernel_size=2, output_size=3, return_indices=True
        )
        flat = idx.numpy().flatten()
        assert int(flat.min()) >= 0
        assert int(flat.max()) < 6 * 6

    def test_frac_pool2d_indices_point_to_max(self) -> None:
        # Verify each returned index actually points to the max of its window.
        x = lucid.randn(1, 1, 4, 4)
        samples = lucid.zeros(1, 1, 2)  # deterministic: starts=[0,2]
        out, idx = F.fractional_max_pool2d(  # type: ignore[misc]
            x,
            kernel_size=2,
            output_size=2,
            return_indices=True,
            _random_samples=samples,
        )
        x_np = x.numpy()[0, 0]  # (4, 4)
        out_np = out.numpy()[0, 0]  # (2, 2)
        idx_np = idx.numpy()[0, 0]  # (2, 2)
        for i in range(2):
            for j in range(2):
                flat = int(idx_np[i, j])
                r, c = flat // 4, flat % 4
                np.testing.assert_allclose(x_np[r, c], out_np[i, j], atol=1e-6)

    def test_frac_pool2d_backward(self) -> None:
        x = lucid.randn(1, 1, 4, 4, requires_grad=True)
        out = F.fractional_max_pool2d(x, kernel_size=2, output_size=2)
        out.sum().backward()
        assert x.grad is not None
        assert tuple(x.grad.shape) == (1, 1, 4, 4)

    def test_frac_pool2d_grad_sums_to_one_per_channel(self) -> None:
        # Each input element is the max of at most one window, so the
        # gradient of sum(output) w.r.t. x must sum to oH * oW.
        x = lucid.randn(1, 1, 4, 4, requires_grad=True)
        out = F.fractional_max_pool2d(x, kernel_size=2, output_size=2)
        out.sum().backward()
        assert x.grad is not None
        total = float(x.grad.numpy().sum())
        assert abs(total - 4.0) < 1e-5  # 2*2 output cells, each contributes 1

    def test_frac_pool2d_error_both_size_and_ratio(self) -> None:
        import pytest as _pytest

        x = lucid.ones(1, 1, 4, 4)
        with _pytest.raises(ValueError):
            F.fractional_max_pool2d(x, kernel_size=2, output_size=2, output_ratio=0.5)

    def test_frac_pool2d_error_neither_size_nor_ratio(self) -> None:
        import pytest as _pytest

        x = lucid.ones(1, 1, 4, 4)
        with _pytest.raises(ValueError):
            F.fractional_max_pool2d(x, kernel_size=2)

    def test_frac_pool3d_output_size_shape(self) -> None:
        x = lucid.ones(1, 2, 8, 8, 8)
        out = F.fractional_max_pool3d(x, kernel_size=2, output_size=4)
        assert tuple(out.shape) == (1, 2, 4, 4, 4)

    def test_frac_pool3d_output_ratio_shape(self) -> None:
        x = lucid.ones(1, 1, 8, 8, 8)
        out = F.fractional_max_pool3d(x, kernel_size=2, output_ratio=0.5)
        assert tuple(out.shape) == (1, 1, 4, 4, 4)

    def test_frac_pool3d_return_indices_shape_and_range(self) -> None:
        x = lucid.randn(1, 1, 6, 6, 6)
        out, idx = F.fractional_max_pool3d(  # type: ignore[misc]
            x, kernel_size=2, output_size=3, return_indices=True
        )
        assert tuple(out.shape) == (1, 1, 3, 3, 3)
        assert tuple(idx.shape) == (1, 1, 3, 3, 3)
        flat = idx.numpy().flatten()
        assert int(flat.min()) >= 0
        assert int(flat.max()) < 6 * 6 * 6

    def test_frac_pool3d_backward(self) -> None:
        x = lucid.randn(1, 1, 4, 4, 4, requires_grad=True)
        out = F.fractional_max_pool3d(x, kernel_size=2, output_size=2)
        out.sum().backward()
        assert x.grad is not None
        assert tuple(x.grad.shape) == (1, 1, 4, 4, 4)


class TestBicubicInterpolate:
    """``mode="bicubic"`` was documented on :func:`interpolate` and absent.

    Anyone who asked for it got ``ValueError: Unsupported interpolation
    mode``, and the model that needed it — DIAMOND's CS:GO upsampler,
    whose training pairs are built with bicubic — silently used nearest
    instead, feeding the network a blockier conditioning image than it
    was ever trained on.
    """

    def test_it_reproduces_a_hand_computed_kernel(self) -> None:
        """The cubic convolution kernel at A = -0.75, checked by hand.

        Upsampling a constant field must return that constant: the four
        taps of the kernel sum to one at every offset.  This is the one
        property that fails loudly if the coefficients are transcribed
        wrong, and it needs no reference implementation to state.
        """
        flat = lucid.ones((1, 1, 5, 7)) * 3.25
        out = F.interpolate(flat, size=(11, 13), mode="bicubic")
        assert out.shape == (1, 1, 11, 13)
        assert float((out - 3.25).abs().max().item()) < 1e-5

    def test_it_passes_through_the_samples_it_lands_on(self) -> None:
        """Bicubic interpolates: it does not merely approximate.

        The kernel is 1 at zero offset and 0 at every other integer, so
        an output pixel that falls exactly on an input pixel must carry
        that value unchanged.  With ``align_corners=True`` a 5 -> 9
        resample puts every even output on an input, which pins the tap
        alignment: an off-by-one in the index arithmetic moves the whole
        stencil and this stops holding.
        """
        x = lucid.tensor([[[[0.0, 5.0, -2.0, 7.0, 1.0]]]])
        up = F.interpolate(x, size=(1, 9), mode="bicubic", align_corners=True)
        row = up.reshape(-1)
        for source, target in enumerate(range(0, 9, 2)):
            assert abs(float(row[target].item()) - float(x.reshape(-1)[source].item())) < 1e-5

    def test_it_overshoots_where_nearest_clamps(self) -> None:
        """Bicubic is allowed to leave the input's range — that is the filter.

        Normalising or clipping the weights would make it a different
        kernel, and one the reference weights were not trained against.
        A step edge is where it shows.
        """
        step = lucid.tensor([[[[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]]])
        out = F.interpolate(step, size=(1, 24), mode="bicubic")
        assert float(out.min().item()) < -1e-3 or float(out.max().item()) > 1.001

    def test_the_kernel_is_the_A_minus_three_quarters_one(self) -> None:
        """Which cubic, not merely "a cubic".

        Both ``A = -0.75`` (what the reference vision stacks use) and
        ``A = -0.5`` (Keys' original) sum their taps to one, so a
        constant field and the interpolating property hold for either.
        They differ in how far they undershoot at an edge, which is the
        only thing that separates them — and the trained weights were
        fitted against the first.
        """
        impulse = lucid.tensor([[[[0.0, 1.0, 0.0, 0.0]]]])
        out = F.interpolate(impulse, size=(1, 8), mode="bicubic")
        row = out.reshape(-1)
        # A = -0.75 undershoots to -0.105469 here; A = -0.5 to -0.070312.
        assert abs(float(row[0].item()) - (-0.10546875)) < 1e-6
        assert abs(float(row[2].item()) - 0.87890625) < 1e-6

    def test_the_gradient_reaches_the_input(self) -> None:
        x = lucid.randn((1, 2, 4, 5), requires_grad=True)
        F.interpolate(x, size=(9, 11), mode="bicubic").sum().backward()
        assert x.grad is not None
        assert tuple(x.grad.shape) == (1, 2, 4, 5)


class TestGridSample:
    """Regression guard — the Python wrapper once dropped mode/padding_mode
    and passed align_corners into the mode slot, silently turning every
    ``mode="bilinear"`` call into nearest-neighbour sampling."""

    def test_bilinear_actually_interpolates(self) -> None:
        # 2x2 ramp [[0,1],[2,3]]; sampling the exact centre must average all
        # four corners (1.5), NOT snap to a single pixel (the old bug → 0.0).
        feat = lucid.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
        grid = lucid.tensor([[[[0.0, 0.0]]]])  # (gx, gy) = centre
        out = F.grid_sample(
            feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).item()
        assert abs(out - 1.5) < 1e-6

    def test_bilinear_quarter_point(self) -> None:
        feat = lucid.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
        # row=0.25, col=0.75 → align_corners=True normalised (gx, gy)=(0.5, -0.5)
        grid = lucid.tensor([[[[0.5, -0.5]]]])
        out = F.grid_sample(
            feat, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).item()
        exp = 0.75 * (0.25 * 0 + 0.75 * 1) + 0.25 * (0.25 * 2 + 0.75 * 3)
        assert abs(out - exp) < 1e-6

    def test_nearest_still_snaps(self) -> None:
        feat = lucid.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
        grid = lucid.tensor([[[[0.5, -0.5]]]])
        out = F.grid_sample(
            feat, grid, mode="nearest", padding_mode="zeros", align_corners=True
        ).item()
        # nearest of (row 0.25, col 0.75) → pixel (0, 1) = 1.0
        assert out == 1.0

    def test_unknown_mode_raises(self) -> None:
        feat = lucid.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
        grid = lucid.tensor([[[[0.0, 0.0]]]])
        try:
            F.grid_sample(feat, grid, mode="cubic")
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for unknown mode")


class TestSoftmaxAndReductionDtype:
    """Guard for LCD-510: dtype= on softmax / log_softmax / softmin and
    reduction ops (sum, mean, prod, cumsum, cumprod, nansum, nanmean), plus
    F.softmax / log_softmax / softmin input/dim=None/_stacklevel parity."""

    @pytest.mark.parametrize("device", ["cpu", "metal"])
    @pytest.mark.parametrize("fn_name", ["softmax", "log_softmax", "softmin"])
    def test_softmax_family_dtype_forward_and_backward(
        self, ref: object, device: str, fn_name: str
    ) -> None:
        if device == "metal" and not lucid.metal.is_available():
            pytest.skip("metal not available")
        ref_mod = ref  # type: ignore[assignment]
        np_x = np.linspace(-3.0, 3.0, 16, dtype=np.float16).reshape(4, 4)
        x = lucid.tensor(np_x, dtype=lucid.float16, device=device, requires_grad=True)
        rx = ref_mod.tensor(np_x, dtype=ref_mod.float16, requires_grad=True)

        fn = getattr(F, fn_name)
        rfn = getattr(ref_mod.nn.functional, fn_name)
        out = fn(input=x, dim=-1, dtype=lucid.float32)
        rout = rfn(input=rx, dim=-1, dtype=ref_mod.float32)

        assert out.dtype == lucid.float32
        np.testing.assert_allclose(out.cpu().numpy(), rout.detach().numpy(), rtol=1e-3, atol=1e-3)
        weights = lucid.arange(1, 17, dtype=lucid.float32, device=device).reshape(4, 4)
        rweights = ref_mod.arange(1, 17, dtype=ref_mod.float32).reshape(4, 4)
        (out * weights).sum().backward()
        (rout * rweights).sum().backward()
        assert x.grad is not None
        assert x.grad.dtype == lucid.float16
        np.testing.assert_allclose(x.grad.cpu().numpy(), rx.grad.numpy(), rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("device", ["cpu", "metal"])
    def test_top_level_and_tensor_method_softmax_dtype(self, ref: object, device: str) -> None:
        if device == "metal" and not lucid.metal.is_available():
            pytest.skip("metal not available")
        ref_mod = ref  # type: ignore[assignment]
        np_x = np.linspace(-2.0, 2.0, 16, dtype=np.float32).reshape(4, 4)
        x = lucid.tensor(np_x, dtype=lucid.float16, device=device)
        rx = ref_mod.tensor(np_x, dtype=ref_mod.float16)

        for l_out, r_out in [
            (
                lucid.softmax(x, dim=-1, dtype=lucid.float32),
                ref_mod.softmax(rx, dim=-1, dtype=ref_mod.float32),
            ),
            (
                lucid.log_softmax(x, dim=-1, dtype=lucid.float32),
                ref_mod.log_softmax(rx, dim=-1, dtype=ref_mod.float32),
            ),
            (
                x.softmax(dim=-1, dtype=lucid.float32),
                rx.softmax(dim=-1, dtype=ref_mod.float32),
            ),
            (
                x.log_softmax(dim=-1, dtype=lucid.float32),
                rx.log_softmax(dim=-1, dtype=ref_mod.float32),
            ),
        ]:
            assert l_out.dtype == lucid.float32
            np.testing.assert_allclose(l_out.cpu().numpy(), r_out.numpy(), rtol=1e-3, atol=1e-3)

    def test_softmax_dim_none_warns_and_selects_reference_dim(self, ref: object) -> None:
        ref_mod = ref  # type: ignore[assignment]
        for shape in [(16,), (4, 4), (2, 2, 4), (2, 2, 2, 2)]:
            np_x = np.arange(16, dtype=np.float32).reshape(shape)
            x = lucid.tensor(np_x)
            rx = ref_mod.tensor(np_x)
            for fn_name in ("softmax", "log_softmax", "softmin"):
                with pytest.warns(UserWarning, match="Implicit dimension choice"):
                    out = getattr(F, fn_name)(x)
                with pytest.warns(UserWarning):
                    rout = getattr(ref_mod.nn.functional, fn_name)(rx)
                np.testing.assert_allclose(out.numpy(), rout.numpy(), rtol=1e-6, atol=1e-6)

    def test_softmax_explicit_dtype_honored_under_autocast(self) -> None:
        if not lucid.metal.is_available():
            pytest.skip("metal not available")
        x = lucid.linspace(-2.0, 2.0, 16, dtype=lucid.float32, device="metal").reshape(4, 4)
        with lucid.amp.autocast("metal", dtype=lucid.float16):
            out_default = F.softmax(x, dim=-1)
            out_f16 = F.softmax(x, dim=-1, dtype=lucid.float16)
        assert out_default.dtype == lucid.float32
        assert out_f16.dtype == lucid.float16

    @pytest.mark.parametrize("device", ["cpu", "metal"])
    @pytest.mark.parametrize(
        "op_name",
        ["sum", "mean", "prod", "cumsum", "cumprod", "nansum", "nanmean"],
    )
    def test_reduction_dtype_keyword(self, ref: object, device: str, op_name: str) -> None:
        if device == "metal" and not lucid.metal.is_available():
            pytest.skip("metal not available")
        ref_mod = ref  # type: ignore[assignment]
        np_x = np.linspace(0.5, 2.0, 16, dtype=np.float16).reshape(4, 4)
        x = lucid.tensor(np_x, dtype=lucid.float16, device=device, requires_grad=True)
        rx = ref_mod.tensor(np_x, dtype=ref_mod.float16, requires_grad=True)

        fn = getattr(lucid, op_name)
        rfn = getattr(ref_mod, op_name)
        if op_name in ("cumsum", "cumprod"):
            out = fn(x, dim=1, dtype=lucid.float32)
            rout = rfn(rx, dim=1, dtype=ref_mod.float32)
        else:
            out = fn(x, dim=1, dtype=lucid.float32)
            rout = rfn(rx, dim=1, dtype=ref_mod.float32)
        assert out.dtype == lucid.float32
        np.testing.assert_allclose(out.cpu().numpy(), rout.detach().numpy(), rtol=1e-3, atol=1e-3)
        out.sum().backward()
        rout.sum().backward()
        assert x.grad is not None
        assert x.grad.dtype == lucid.float16
        np.testing.assert_allclose(x.grad.cpu().numpy(), rx.grad.numpy(), rtol=1e-2, atol=1e-2)

    def test_dtype_error_classes(self) -> None:
        x = lucid.ones((16,), dtype=lucid.float32)
        with pytest.raises(NotImplementedError):
            F.softmax(x, dim=0, dtype=lucid.int64)
        with pytest.raises(TypeError):
            lucid.mean(x, dtype=lucid.int64)
        with pytest.raises(TypeError):
            lucid.nanmean(x, dtype=lucid.int64)
