"""Unit tests for complex views (`real`, `imag`, `view_as_real`, `view_as_complex`)
and `nn.Parameter` view sharing on CPU and Metal (LCD-119).
"""

import pytest

import lucid
import lucid.nn as nn
from lucid.test._fixtures.devices import metal_available

_DEVICES: tuple[str, ...] = ("cpu", "metal") if metal_available() else ("cpu",)


@pytest.mark.parametrize("device", _DEVICES)
def test_parameter_of_metal_and_cpu_tensor_aliases_source(device: str) -> None:
    x = lucid.zeros(3, device=device)
    p = nn.Parameter(x)
    assert p.storage_offset() == 0
    assert p.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    with lucid.no_grad():
        p.add_(2.0)
    assert x.tolist() == [2.0, 2.0, 2.0]
    with lucid.no_grad():
        x[1] = 7.0
    assert p.tolist() == [2.0, 7.0, 2.0]


@pytest.mark.parametrize("device", _DEVICES)
def test_real_and_imag_are_strided_views_both_ways(device: str) -> None:
    c = lucid.tensor([1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j], device=device)
    re = lucid.real(c)
    im = lucid.imag(c)

    assert re.dtype == lucid.float32
    assert im.dtype == lucid.float32
    assert re.shape == (3,)
    assert im.shape == (3,)
    assert re.stride() == (2,)
    assert im.stride() == (2,)
    assert re.storage_offset() == 0
    assert im.storage_offset() == 1
    assert re.untyped_storage().data_ptr() == c.untyped_storage().data_ptr()
    assert im.untyped_storage().data_ptr() == c.untyped_storage().data_ptr()

    re.add_(10.0)
    assert c.tolist() == [11.0 + 2.0j, 13.0 + 4.0j, 15.0 + 6.0j]

    im[1] = -9.0
    assert c.tolist() == [11.0 + 2.0j, 13.0 - 9.0j, 15.0 + 6.0j]

    c.add_(1.0 + 1.0j)
    assert re.tolist() == [12.0, 14.0, 16.0]
    assert im.tolist() == [3.0, -8.0, 7.0]


@pytest.mark.parametrize("device", _DEVICES)
def test_view_as_real_and_view_as_complex_share_storage_both_ways(device: str) -> None:
    c = lucid.tensor([[1.0 + 2.0j, 3.0 + 4.0j], [5.0 + 6.0j, 7.0 + 8.0j]], device=device)
    vr = lucid.view_as_real(c)
    assert vr.dtype == lucid.float32
    assert vr.shape == (2, 2, 2)
    assert vr.stride() == (4, 2, 1)
    assert vr.storage_offset() == 0
    assert vr.is_contiguous()

    vr[0, 1, 0] = 30.0
    vr[1, 0, 1] = 60.0
    assert c.tolist() == [[1.0 + 2.0j, 30.0 + 4.0j], [5.0 + 60.0j, 7.0 + 8.0j]]

    vc = lucid.view_as_complex(vr)
    assert vc.dtype == lucid.complex64
    assert vc.shape == (2, 2)
    assert vc.stride() == (2, 1)
    assert vc.storage_offset() == 0
    vc[1, 1] = -7.0 - 8.0j
    assert vr[1, 1].tolist() == [-7.0, -8.0]
    assert c[1, 1].item() == -7.0 - 8.0j


@pytest.mark.parametrize("device", _DEVICES)
def test_view_as_complex_over_sliced_odd_length_real_root(device: str) -> None:
    full = lucid.arange(7.0, device=device)
    pairs = full[1:7].reshape(3, 2)
    assert pairs.storage_offset() == 1
    with pytest.raises(ValueError, match="storage_offset divisible by 2"):
        lucid.view_as_complex(pairs)

    even_pairs = full[:6].reshape(3, 2)
    vc = lucid.view_as_complex(even_pairs)
    assert vc.tolist() == [0.0 + 1.0j, 2.0 + 3.0j, 4.0 + 5.0j]
    vc[1] = 20.0 + 30.0j
    assert full.tolist() == [0.0, 1.0, 20.0, 30.0, 4.0, 5.0, 6.0]
    lucid.imag(vc).add_(100.0)
    assert full.tolist() == [0.0, 101.0, 20.0, 130.0, 4.0, 105.0, 6.0]


@pytest.mark.parametrize("device", _DEVICES)
def test_view_as_complex_validates_strides_and_shape(device: str) -> None:
    x = lucid.zeros(2, 3, device=device)
    with pytest.raises(ValueError, match="expected last dim == 2"):
        lucid.view_as_complex(x)

    t = lucid.zeros(2, 4, device=device).T
    assert t.shape == (4, 2)
    assert t.stride() == (1, 4)
    with pytest.raises(ValueError, match="last dimension with stride 1"):
        lucid.view_as_complex(t)

    # Size-1 dimension with odd stride is allowed on non-empty tensors (matches reference).
    base = lucid.arange(4.0, device=device)
    s1 = base.as_strided((1, 2), (3, 1), 0)
    vc = lucid.view_as_complex(s1)
    assert vc.shape == (1,)
    assert vc.tolist() == [0.0 + 1.0j]


@pytest.mark.parametrize("device", _DEVICES)
def test_autograd_rebases_across_complex_and_real_views(device: str) -> None:
    z0 = lucid.tensor([1.0 + 2.0j, 3.0 + 4.0j], device=device, requires_grad=True)
    z = z0 * (2.0 + 0.0j)
    re = lucid.real(z)
    im = lucid.imag(z)
    re.add_(lucid.tensor([5.0, 6.0], device=device))
    loss = (re * 3.0 + im * 7.0).sum()
    loss.backward()
    assert z0.grad is not None
    assert z0.grad.tolist() == [6.0 + 14.0j, 6.0 + 14.0j]

    r0 = lucid.tensor([[1.0, 2.0], [3.0, 4.0]], device=device, requires_grad=True)
    r = r0 * 2.0
    vc = lucid.view_as_complex(r)
    vc.add_(lucid.tensor([10.0 + 20.0j, 30.0 + 40.0j], device=device))
    loss2 = (lucid.real(vc) * 3.0 + lucid.imag(vc) * 5.0).sum()
    loss2.backward()
    assert r0.grad is not None
    assert r0.grad.tolist() == [[6.0, 10.0], [6.0, 10.0]]
