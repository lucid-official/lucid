"""nn.utils — clip_grad, parametrize, weight_norm, etc."""

import numpy as np
import pytest

import lucid
import lucid.nn as nn


class TestClipGradNorm:
    def test_clamps_to_max(self) -> None:
        x = lucid.tensor([3.0, 4.0], requires_grad=True)
        (x * x).sum().backward()
        # ||grad||_2 = ||[6, 8]||_2 = 10.
        total_norm = nn.utils.clip_grad_norm_([x], max_norm=5.0)
        # ``total_norm`` may be a Tensor or a Python float depending on
        # the implementation — accept either.
        tn = total_norm.item() if hasattr(total_norm, "item") else float(total_norm)
        assert abs(tn - 10.0) < 1e-4
        # After clipping, ||grad||_2 should be ~5.
        new_norm = float(np.sqrt((x.grad.numpy() ** 2).sum()))
        assert abs(new_norm - 5.0) < 1e-4

    def test_returns_a_0d_tensor(self) -> None:
        # Documented as a scalar and 0-d in the reference framework; it was
        # shape (1,), which puts a stray axis into whatever it is stacked or
        # broadcast with.
        from lucid.nn.utils.clip_grad import get_total_norm

        x = lucid.tensor([3.0, 4.0], requires_grad=True)
        (x * x).sum().backward()
        assert nn.utils.clip_grad_norm_([x], max_norm=5.0).shape == ()
        inf_norm = nn.utils.clip_grad_norm_([x], 5.0, norm_type=float("inf"))
        assert inf_norm.shape == ()
        assert get_total_norm([x]).shape == ()
        # Nothing to measure is still a 0-d zero, not a length-1 vector.
        no_grad = lucid.tensor([1.0], requires_grad=True)
        assert nn.utils.clip_grad_norm_([no_grad], 1.0).shape == ()
        assert nn.utils.clip_grad_norm_([], 1.0).shape == ()


class TestClipGradValue:
    def test_clamps_each_element(self) -> None:
        x = lucid.tensor([1.0, 2.0], requires_grad=True)
        (x * x * x).sum().backward()
        # grad = 3x² = [3, 12].
        nn.utils.clip_grad_value_([x], clip_value=5.0)
        np.testing.assert_array_equal(x.grad.numpy(), [3.0, 5.0])


class TestCopyParametersAndBuffers:
    def test_round_trip(self) -> None:
        src = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 2))
        dst = nn.Sequential(nn.Linear(4, 8), nn.BatchNorm1d(8), nn.Linear(8, 2))
        for _, p in src.named_parameters():
            p._impl.copy_from(lucid.ones(p.shape, dtype=p.dtype, device=p.device)._impl)
        nn.utils.copy_parameters_and_buffers(src, dst)
        for (_, ps), (_, pd) in zip(src.named_parameters(), dst.named_parameters()):
            np.testing.assert_array_equal(ps.numpy(), pd.numpy())
        for (_, bs), (_, bd) in zip(src.named_buffers(), dst.named_buffers()):
            np.testing.assert_array_equal(bs.numpy(), bd.numpy())

    def test_missing_param_raises(self) -> None:
        src = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
        dst = nn.Sequential(nn.Linear(4, 8))
        with pytest.raises(KeyError, match="present on source but missing on dest"):
            nn.utils.copy_parameters_and_buffers(src, dst)


class TestFuseConvBnEval:
    def test_matches_unfused_2d(self) -> None:
        np.random.seed(0)
        conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=True)
        bn = nn.BatchNorm2d(8)
        bn.eval()
        conv.eval()
        bn.running_mean._impl.copy_from((lucid.ones(8) * 0.5)._impl)
        bn.running_var._impl.copy_from((lucid.ones(8) * 1.5)._impl)

        x = lucid.tensor(np.random.randn(2, 3, 8, 8).astype(np.float32))
        ref = bn(conv(x)).numpy()
        fused = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        out = fused(x).numpy()
        np.testing.assert_allclose(ref, out, atol=1e-5)

    def test_with_bn_affine_off(self) -> None:
        # Affine-disabled BN: scale = 1 / sqrt(σ² + ε) (no γ/β).
        np.random.seed(0)
        conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=True)
        bn = nn.BatchNorm2d(4, affine=False)
        bn.eval()
        conv.eval()
        bn.running_mean._impl.copy_from((lucid.ones(4) * 0.2)._impl)

        x = lucid.tensor(np.random.randn(1, 3, 6, 6).astype(np.float32))
        ref = bn(conv(x)).numpy()
        fused = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        out = fused(x).numpy()
        np.testing.assert_allclose(ref, out, atol=1e-5)

    def test_no_bias_conv(self) -> None:
        # Conv2d(bias=False) is now accepted by the engine binding.
        np.random.seed(0)
        conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        bn = nn.BatchNorm2d(4)
        bn.eval()
        conv.eval()
        bn.running_mean._impl.copy_from((lucid.ones(4) * 0.2)._impl)

        x = lucid.tensor(np.random.randn(1, 3, 6, 6).astype(np.float32))
        ref = bn(conv(x)).numpy()
        fused = nn.utils.fusion.fuse_conv_bn_eval(conv, bn)
        out = fused(x).numpy()
        np.testing.assert_allclose(ref, out, atol=1e-5)


def test_get_total_norm_measures_the_tensors_it_is_given() -> None:
    # It used to read ``.grad`` from its arguments, so the usual call —
    # handing it the gradients themselves — answered 0.
    from lucid.nn.utils.clip_grad import clip_grad_norm_, get_total_norm

    assert get_total_norm([lucid.tensor([3.0, 4.0])]).item() == 5.0
    assert get_total_norm(lucid.tensor([3.0, 4.0])).item() == 5.0
    x = lucid.tensor([3.0, 4.0], requires_grad=True)
    (x * x).sum().backward()  # the gradient is 2x = [6, 8]
    assert x.grad is not None
    assert get_total_norm([x.grad]).item() == 10.0
    assert clip_grad_norm_([x], max_norm=100.0).item() == 10.0


@pytest.mark.parametrize("device", ["cpu", "metal"])
@pytest.mark.parametrize("amp_dtype", [lucid.float16, lucid.bfloat16])
@pytest.mark.parametrize("norm_type", [1.0, 2.0, float("inf")])
def test_clip_grad_norm_inside_autocast_scope_matches_reference(
    ref, device: str, amp_dtype: lucid.dtype, norm_type: float
) -> None:
    """LCD-79 (#151): clip_grad_norm_ inside autocast must not raise or cast grads."""
    rng = np.random.default_rng(42)
    w_np = rng.standard_normal((4, 4)).astype(np.float32)
    b_np = rng.standard_normal((4,)).astype(np.float32)
    x_np = rng.standard_normal((2, 4)).astype(np.float32)

    m = nn.Linear(4, 4).to(device)
    m.weight._impl.copy_from(lucid.tensor(w_np, device=device)._impl)
    m.bias._impl.copy_from(lucid.tensor(b_np, device=device)._impl)

    with lucid.amp.autocast(device_type=device, dtype=amp_dtype):
        y = m(lucid.tensor(x_np, device=device)).sum()
    y.backward()

    rm = ref.nn.Linear(4, 4)
    with ref.no_grad():
        rm.weight.copy_(ref.tensor(w_np))
        rm.bias.copy_(ref.tensor(b_np))
    rm.weight.grad = ref.tensor(m.weight.grad.cpu().numpy().copy())
    rm.bias.grad = ref.tensor(m.bias.grad.cpu().numpy().copy())

    with lucid.amp.autocast(device_type=device, dtype=amp_dtype):
        total_norm = nn.utils.clip_grad_norm_(m.parameters(), max_norm=0.1, norm_type=norm_type)
    ref_norm = ref.nn.utils.clip_grad_norm_(rm.parameters(), max_norm=0.1, norm_type=norm_type)

    assert total_norm.dtype == lucid.float32
    assert m.weight.grad.dtype == lucid.float32
    assert m.bias.grad.dtype == lucid.float32
    np.testing.assert_allclose(
        float(total_norm.cpu().item()), float(ref_norm.item()), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        m.weight.grad.cpu().numpy(), rm.weight.grad.numpy(), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        m.bias.grad.cpu().numpy(), rm.bias.grad.numpy(), rtol=1e-4, atol=1e-4
    )


@pytest.mark.parametrize("device", ["cpu", "metal"])
def test_clip_grad_norm_mixed_parameter_dtypes_under_autocast(device: str) -> None:
    """LCD-79: mixed-dtype parameter gradients (f32 + bf16/f16) clip in their own dtypes."""
    p_f32 = nn.Parameter(lucid.ones(4, dtype=lucid.float32, device=device))
    p_bf16 = nn.Parameter(lucid.ones(4, dtype=lucid.bfloat16, device=device))
    p_f32.grad = lucid.full((4,), 3.0, dtype=lucid.float32, device=device)
    p_bf16.grad = lucid.full((4,), 4.0, dtype=lucid.bfloat16, device=device)

    with lucid.amp.autocast(device_type=device, dtype=lucid.bfloat16):
        norm = nn.utils.clip_grad_norm_([p_f32, p_bf16], max_norm=1.0)

    assert abs(float(norm.cpu().item()) - 10.0) < 1e-2
    assert p_f32.grad.dtype == lucid.float32
    assert p_bf16.grad.dtype == lucid.bfloat16
