"""Reference surface parity for LCD-525 out-of-place ops and Tensor methods.

Covers:
- ``lucid.logcumsumexp`` and ``Tensor.logcumsumexp``
- ``lucid.aminmax`` and ``Tensor.aminmax`` (returning ``(min, max)`` with ``.min`` and ``.max``)
- ``lucid.renorm`` and ``Tensor.renorm``
- ``lucid.trapezoid`` and ``lucid.cumulative_trapezoid``
- ``lucid.select`` and ``Tensor.select`` (view semantics on CPU and Metal)
- ``lucid.clamp_min``, ``lucid.clamp_max``, ``Tensor.clamp_min``, ``Tensor.clamp_max``
- ``lucid.empty_strided``
- ``narrow`` and ``Tensor.narrow`` with negative ``start``
- ``Tensor.tensor_split``, ``Tensor.vsplit``, ``Tensor.hsplit``, ``Tensor.dsplit``
- ``Tensor.addcmul`` and ``Tensor.addcdiv``
- ``Tensor.is_set_to``
"""

import math
from typing import Any

import numpy as np
import pytest

import lucid

_DEVICES = ["cpu"] + (["metal"] if lucid.metal.is_available() else [])


@pytest.mark.parametrize("device", _DEVICES)
def test_logcumsumexp_matches_reference(ref: Any, device: str) -> None:
    vals = [
        [1.0, 2.0, -3.0, 4.0],
        [-1000.0, -999.0, 1.0, 2.0],
        [1.0, 1000.0, 1000.0, -5.0],
        [float("-inf"), float("-inf"), 0.0, float("inf")],
    ]
    lx = lucid.tensor(vals, dtype=lucid.float32, device=device)
    rx = ref.tensor(vals, dtype=ref.float32)
    for dim in (0, 1, -1):
        lo_fn = lucid.logcumsumexp(lx, dim=dim)
        lo_m = lx.logcumsumexp(dim=dim)
        ro = ref.logcumsumexp(rx, dim=dim)
        np.testing.assert_allclose(
            lo_fn.cpu().numpy(), ro.numpy(), rtol=1e-5, atol=1e-5, equal_nan=True
        )
        np.testing.assert_allclose(
            lo_m.cpu().numpy(), ro.numpy(), rtol=1e-5, atol=1e-5, equal_nan=True
        )

    # Half dtypes and autograd
    for ldt, rdt in [(lucid.float16, ref.float16), (lucid.bfloat16, ref.bfloat16)]:
        lh = lucid.tensor([[1.0, 2.0, 3.0]], dtype=ldt, device=device)
        rh = ref.tensor([[1.0, 2.0, 3.0]], dtype=rdt)
        out_h = lucid.logcumsumexp(lh, dim=-1)
        assert out_h.dtype == ldt
        np.testing.assert_allclose(
            out_h.to(lucid.float32).cpu().numpy(),
            ref.logcumsumexp(rh, dim=-1).to(ref.float32).numpy(),
            rtol=1e-2,
            atol=1e-2,
        )

    lg = lucid.tensor([1.0, 2.0, 3.0, 4.0], device=device, requires_grad=True)
    rg = ref.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    lucid.logcumsumexp(lg, dim=0).sum().backward()
    ref.logcumsumexp(rg, dim=0).sum().backward()
    assert lg.grad is not None
    np.testing.assert_allclose(lg.grad.cpu().numpy(), rg.grad.numpy(), rtol=1e-5, atol=1e-5)

    with pytest.raises(NotImplementedError):
        lucid.logcumsumexp(lucid.tensor([1, 2, 3], device=device), dim=0)


@pytest.mark.parametrize("device", _DEVICES)
def test_aminmax_matches_reference(ref: Any, device: str) -> None:
    vals = [[3.0, -1.0, 4.0], [2.0, 5.0, -2.0]]
    lx = lucid.tensor(vals, device=device)
    rx = ref.tensor(vals)

    res_all = lucid.aminmax(lx)
    ref_all = ref.aminmax(rx)
    assert isinstance(res_all, tuple) and len(res_all) == 2
    assert res_all.min.shape == () and res_all.max.shape == ()
    assert res_all.min.item() == ref_all.min.item()
    assert res_all.max.item() == ref_all.max.item()

    res_keep = lx.aminmax(keepdim=True)
    assert res_keep.min.shape == (1, 1) and res_keep.max.shape == (1, 1)

    res_dim = lx.aminmax(dim=1, keepdim=False)
    ref_dim = rx.aminmax(dim=1, keepdim=False)
    np.testing.assert_allclose(res_dim.min.cpu().numpy(), ref_dim.min.numpy())
    np.testing.assert_allclose(res_dim.max.cpu().numpy(), ref_dim.max.numpy())

    # NaN propagation and bool support
    lnan = lucid.tensor([1.0, float("nan"), 3.0], device=device)
    rnan = lnan.aminmax()
    assert math.isnan(rnan.min.item()) and math.isnan(rnan.max.item())

    lbool = lucid.tensor([[True, False], [True, True]], device=device)
    rbool = lbool.aminmax(dim=1)
    assert rbool.min.tolist() == [False, True]
    assert rbool.max.tolist() == [True, True]

    with pytest.raises(RuntimeError):
        lucid.aminmax(lucid.zeros((0,), device=device))


@pytest.mark.parametrize("device", _DEVICES)
def test_renorm_matches_reference(ref: Any, device: str) -> None:
    lx = lucid.arange(1.0, 7.0, device=device).reshape(2, 3).requires_grad_(True)
    rx = ref.arange(1.0, 7.0).reshape(2, 3).requires_grad_(True)
    for p in (1.0, 2.0, float("inf")):
        lo = lx.renorm(p=p, dim=0, maxnorm=2.0)
        ro = rx.renorm(p=p, dim=0, maxnorm=2.0)
        np.testing.assert_allclose(
            lo.detach().cpu().numpy(), ro.detach().numpy(), rtol=1e-5, atol=1e-5
        )
        lo.sum().backward()
        ro.sum().backward()
        assert lx.grad is not None
        np.testing.assert_allclose(lx.grad.cpu().numpy(), rx.grad.numpy(), rtol=1e-5, atol=1e-5)
        lx.grad = None
        rx.grad = None

    with pytest.raises(RuntimeError, match="non-positive-norm"):
        lucid.renorm(lx.detach(), 0.0, 0, 1.0)
    with pytest.raises(RuntimeError, match="maxnorm"):
        lucid.renorm(lx.detach(), 2.0, 0, -1.0)
    with pytest.raises(RuntimeError, match="at least 2 dimensions"):
        lucid.renorm(lucid.tensor([1.0, 2.0], device=device), 2.0, 0, 1.0)


@pytest.mark.parametrize("device", _DEVICES)
def test_trapezoid_and_cumulative_trapezoid_match_reference(ref: Any, device: str) -> None:
    y_vals = [[1.0, 2.0, 4.0, 7.0], [2.0, 4.0, 8.0, 14.0]]
    x_1d = [0.0, 1.0, 3.0, 6.0]
    ly = lucid.tensor(y_vals, device=device, requires_grad=True)
    ry = ref.tensor(y_vals, requires_grad=True)
    lx = lucid.tensor(x_1d, device=device)
    rx = ref.tensor(x_1d)

    for lfn, rfn in [
        (lucid.trapezoid, ref.trapezoid),
        (lucid.cumulative_trapezoid, ref.cumulative_trapezoid),
    ]:
        lo1 = lfn(ly, dx=0.5, dim=-1)
        ro1 = rfn(ry, dx=0.5, dim=-1)
        np.testing.assert_allclose(
            lo1.detach().cpu().numpy(), ro1.detach().numpy(), rtol=1e-5, atol=1e-5
        )
        lo1.sum().backward()
        ro1.sum().backward()
        assert ly.grad is not None
        np.testing.assert_allclose(ly.grad.cpu().numpy(), ry.grad.numpy(), rtol=1e-5, atol=1e-5)
        ly.grad = None
        ry.grad = None

        lo2 = lfn(ly.detach(), x=lx, dim=1)
        ro2 = rfn(ry.detach(), x=rx, dim=1)
        np.testing.assert_allclose(lo2.cpu().numpy(), ro2.numpy(), rtol=1e-5, atol=1e-5)

        # Integer input promotes to default float
        li = lucid.tensor([1, 2, 4], dtype=lucid.int64, device=device)
        ri = ref.tensor([1, 2, 4], dtype=ref.int64)
        loi = lfn(li)
        roi = rfn(ri)
        assert loi.dtype == lucid.float32
        np.testing.assert_allclose(loi.cpu().numpy(), roi.numpy(), rtol=1e-5, atol=1e-5)

        with pytest.raises(TypeError):
            lfn(ly.detach(), x=lx, dx=0.5)
        with pytest.raises(RuntimeError, match="bool"):
            lfn(lucid.tensor([True, False], device=device))


@pytest.mark.parametrize("device", _DEVICES)
def test_select_and_narrow_negative_start_views(ref: Any, device: str) -> None:
    x = lucid.arange(12.0, device=device).reshape(3, 4)
    rx = ref.arange(12.0).reshape(3, 4)

    s = lucid.select(x, 1, -1)
    rs = ref.select(rx, 1, -1)
    assert s.shape == tuple(rs.shape)
    assert s.stride() == rs.stride()
    assert s.storage_offset() == rs.storage_offset()
    np.testing.assert_allclose(s.cpu().numpy(), rs.numpy())

    # In-place write through select view reaches base on CPU and Metal
    x.select(dim=0, index=-1).add_(100.0)
    rx.select(dim=0, index=-1).add_(100.0)
    np.testing.assert_allclose(x.cpu().numpy(), rx.numpy())

    with pytest.raises(IndexError):
        lucid.select(lucid.tensor(1.0, device=device), 0, 0)
    with pytest.raises(IndexError):
        x.select(0, 5)

    # narrow with negative start
    b = lucid.arange(6.0, device=device)
    rb = ref.arange(6.0)
    nw = b.narrow(0, -2, 2)
    rnw = rb.narrow(0, -2, 2)
    assert nw.shape == tuple(rnw.shape)
    assert nw.stride() == rnw.stride()
    np.testing.assert_allclose(nw.cpu().numpy(), rnw.numpy())
    nw.add_(10.0)
    rnw.add_(10.0)
    np.testing.assert_allclose(b.cpu().numpy(), rb.numpy())

    with pytest.raises(IndexError):
        b.narrow(0, -7, 2)


@pytest.mark.parametrize("device", _DEVICES)
def test_clamp_min_max_and_tensor_methods(ref: Any, device: str) -> None:
    x = lucid.tensor([-2.0, 0.5, 3.0], device=device, requires_grad=True)
    rx = ref.tensor([-2.0, 0.5, 3.0], requires_grad=True)

    cmin = lucid.clamp_min(x, 0.0) + x.clamp_min(lucid.tensor([0.0, 1.0, 2.0], device=device))
    rcmin = ref.clamp_min(rx, 0.0) + rx.clamp_min(ref.tensor([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(cmin.detach().cpu().numpy(), rcmin.detach().numpy())

    cmax = lucid.clamp_max(x, 1.0) + x.clamp_max(lucid.tensor([0.0, 1.0, 2.0], device=device))
    rcmax = ref.clamp_max(rx, 1.0) + rx.clamp_max(ref.tensor([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(cmax.detach().cpu().numpy(), rcmax.detach().numpy())

    with pytest.raises(TypeError):
        lucid.clamp_min(x.detach(), None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        lucid.clamp_max(x.detach(), None)  # type: ignore[arg-type]

    # Tensor.tensor_split / vsplit / hsplit / dsplit / addcmul / addcdiv
    cube = lucid.arange(24.0, device=device).reshape(2, 3, 4)
    assert [p.shape for p in cube.tensor_split(2, dim=2)] == [(2, 3, 2), (2, 3, 2)]
    assert [p.shape for p in cube.vsplit(2)] == [(1, 3, 4), (1, 3, 4)]
    assert [p.shape for p in cube.hsplit(3)] == [(2, 1, 4), (2, 1, 4), (2, 1, 4)]
    assert [p.shape for p in cube.dsplit(2)] == [(2, 3, 2), (2, 3, 2)]
    cube.vsplit(2)[1].add_(100.0)
    assert cube[1, 0, 0].item() == 112.0

    a = lucid.tensor([1.0, 2.0], device=device)
    b = lucid.tensor([3.0, 4.0], device=device)
    c = lucid.tensor([2.0, 2.0], device=device)
    np.testing.assert_allclose(a.addcmul(b, c, value=2).cpu().numpy(), [13.0, 18.0])
    np.testing.assert_allclose(a.addcdiv(b, c, value=2).cpu().numpy(), [4.0, 6.0])


@pytest.mark.parametrize("device", _DEVICES)
def test_empty_strided_and_is_set_to(ref: Any, device: str) -> None:
    t = lucid.empty_strided((2, 3), (1, 2), dtype=lucid.float32, device=device, requires_grad=True)
    rt = ref.empty_strided((2, 3), (1, 2), dtype=ref.float32, requires_grad=True)
    assert t.shape == tuple(rt.shape)
    assert t.stride() == rt.stride()
    assert t.is_leaf and t.requires_grad

    t2 = lucid.empty_strided((2, 3), (1, 2), dtype=lucid.float32, device=device)
    t2.fill_(5.0)
    assert t2.tolist() == [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]
    assert t2.as_strided((3,), (2,), 0).tolist() == [5.0, 5.0, 5.0]

    with pytest.raises(RuntimeError):
        lucid.empty_strided((-1, 2), (2, 1), device=device)
    with pytest.raises(RuntimeError):
        lucid.empty_strided((2, 3), (3, -1), device=device)
    with pytest.raises(RuntimeError):
        lucid.empty_strided((2, 3), (3,), device=device)

    # is_set_to
    base = lucid.arange(6.0, device=device).reshape(2, 3)
    assert base.is_set_to(base)
    assert base.is_set_to(base.view(2, 3))
    assert not base.is_set_to(base.view(3, 2))
    assert not base.is_set_to(base.t())
    assert not base.is_set_to(base.clone())
    assert not lucid.zeros((0,), device=device).is_set_to(lucid.zeros((0,), device=device))
    assert base[:0].is_set_to(base[:0])


@pytest.mark.parametrize("device", _DEVICES)
def test_lcd685_elementwise_and_factories(ref: Any, device: str) -> None:
    assert isinstance(lucid.strided, lucid.layout)
    assert repr(lucid.strided) == "lucid.strided"
    x = lucid.tensor([0.0, 90.0, 180.0], device=device, requires_grad=True)
    rx = ref.tensor([0.0, 90.0, 180.0], requires_grad=True)
    np.testing.assert_allclose(
        lucid.deg2rad(x).detach().cpu().numpy(), ref.deg2rad(rx).detach().numpy(), rtol=1e-5
    )
    np.testing.assert_allclose(
        x.rad2deg().detach().cpu().numpy(), rx.rad2deg().detach().numpy(), rtol=1e-5
    )
    a = lucid.tensor([1.0, 50.0, -50.0], device=device, requires_grad=True)
    b = lucid.tensor([2.0, -50.0, 50.0], device=device, requires_grad=True)
    ra = ref.tensor([1.0, 50.0, -50.0], requires_grad=True)
    rb = ref.tensor([2.0, -50.0, 50.0], requires_grad=True)
    out = lucid.logaddexp2(a, b)
    rout = ref.logaddexp2(ra, rb)
    np.testing.assert_allclose(out.detach().cpu().numpy(), rout.detach().numpy(), rtol=1e-5)
    out.sum().backward()
    rout.sum().backward()
    np.testing.assert_allclose(a.grad.cpu().numpy(), ra.grad.numpy(), rtol=1e-5)

    z = lucid.tensor([3.0 + 4.0j, 0.0 + 0.0j], device=device)
    rz = ref.tensor([3.0 + 4.0j, 0.0 + 0.0j])
    np.testing.assert_allclose(lucid.sgn(z).cpu().numpy(), ref.sgn(rz).numpy(), rtol=1e-5)
    f = lucid.tensor([-2.7, 0.0, 2.7], device=device)
    assert lucid.fix(f).tolist() == [-2.0, 0.0, 2.0]
    assert f.clone().fix_().tolist() == [-2.0, 0.0, 2.0]

    # asarray and scalar_tensor
    base = lucid.tensor([1.0, 2.0], device=device, requires_grad=True)
    assert lucid.asarray(base) is base
    copied = lucid.asarray(base, copy=True)
    assert copied is not base and copied.requires_grad and copied.is_leaf
    sc = lucid.scalar_tensor(3.5, device=device, requires_grad=True)
    assert sc.ndim == 0 and sc.requires_grad and abs(sc.item() - 3.5) < 1e-6


@pytest.mark.parametrize("device", _DEVICES)
def test_lcd685_shape_scatter_reductions_and_linalg(ref: Any, device: str) -> None:
    v = lucid.tensor([1.0, 2.0, 3.0], device=device)
    rv = ref.tensor([1.0, 2.0, 3.0])
    np.testing.assert_allclose(lucid.diag(v, 1).cpu().numpy(), ref.diag(rv, 1).numpy())
    np.testing.assert_allclose(
        lucid.diagflat(v.reshape(1, 3)).cpu().numpy(), ref.diagflat(rv.reshape(1, 3)).numpy()
    )
    np.testing.assert_allclose(
        lucid.concatenate([v, v], dim=0).cpu().numpy(), ref.concatenate([rv, rv], dim=0).numpy()
    )
    assert [t.shape for t in lucid.split_with_sizes(v, [1, 2])] == [(1,), (2,)]
    assert [t.shape for t in lucid.unsafe_chunk(v, 2)] == [(2,), (1,)]
    assert [t.shape for t in lucid.unsafe_split(v, 2)] == [(2,), (1,)]
    np.testing.assert_allclose(
        lucid.constant_pad_nd(v, (1, 2), 9.0).cpu().numpy(),
        ref.constant_pad_nd(rv, (1, 2), 9.0).numpy(),
    )
    idx = lucid.tensor([0, 3, 5], device=device)
    ridx = ref.tensor([0, 3, 5])
    for lt, rt in zip(lucid.unravel_index(idx, (2, 3)), ref.unravel_index(ridx, (2, 3))):
        np.testing.assert_array_equal(lt.cpu().numpy(), rt.numpy())

    mat = lucid.zeros(3, 3, device=device, requires_grad=True)
    rmat = ref.zeros(3, 3, requires_grad=True)
    s1 = lucid.select_scatter(mat, lucid.ones(3, device=device), 0, 1)
    rs1 = ref.select_scatter(rmat, ref.ones(3), 0, 1)
    s2 = lucid.slice_scatter(s1, lucid.full((2, 3), 2.0, device=device), 0, 1, 3)
    rs2 = ref.slice_scatter(rs1, ref.full((2, 3), 2.0), 0, 1, 3)
    s3 = lucid.diagonal_scatter(s2, lucid.full((3,), 5.0, device=device))
    rs3 = ref.diagonal_scatter(rs2, ref.full((3,), 5.0))
    np.testing.assert_allclose(s3.detach().cpu().numpy(), rs3.detach().numpy())

    np.testing.assert_allclose(
        lucid.dist(v, lucid.zeros_like(v), p=2.0).cpu().numpy(),
        ref.dist(rv, ref.zeros_like(rv), p=2.0).numpy(),
        rtol=1e-5,
    )
    g_in = lucid.arange(6.0, device=device).reshape(2, 3)
    rg_in = ref.arange(6.0).reshape(2, 3)
    for lg, rg in zip(lucid.gradient(g_in), ref.gradient(rg_in)):
        np.testing.assert_allclose(lg.cpu().numpy(), rg.numpy(), rtol=1e-5)

    spd = lucid.tensor([[4.0, 1.0], [1.0, 3.0]], device=device)
    rspd = ref.tensor([[4.0, 1.0], [1.0, 3.0]])
    chol = lucid.linalg.cholesky(spd)
    rchol = ref.linalg.cholesky(rspd)
    np.testing.assert_allclose(
        lucid.linalg.cholesky_inverse(chol).cpu().numpy(),
        ref.cholesky_inverse(rchol).numpy(),
        rtol=1e-5,
    )
    rect = lucid.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device=device)
    lu_data, lu_piv = lucid.linalg.lu_factor(rect)
    p_mat, l_mat, u_mat = lucid.linalg.lu_unpack(lu_data, lu_piv)
    np.testing.assert_allclose((p_mat @ l_mat @ u_mat).cpu().numpy(), rect.cpu().numpy(), atol=1e-5)


@pytest.mark.parametrize("device", _DEVICES)
def test_lcd685_tensor_methods_and_signal_windows_kwargs(ref: Any, device: str) -> None:
    t = lucid.ones(2, 3, device=device)
    assert t.is_cpu == (device == "cpu")
    assert not t.is_meta
    assert t.layout is lucid.strided
    assert t.ndimension() == 2 and t.nelement() == 6
    assert t.mH.shape == (3, 2)
    assert t.sum_to_size(1, 3).shape == (1, 3)
    np.testing.assert_allclose(
        t.polygamma(1).cpu().numpy(), ref.ones(2, 3).polygamma(1).numpy(), rtol=1e-5
    )

    d = lucid.zeros(3, 3, device=device)
    d.fill_diagonal_(4.0)
    assert d.diagonal().tolist() == [4.0, 4.0, 4.0]

    v = lucid.zeros(2, 3, device=device)
    assert v.t_().shape == (3, 2)
    assert v.transpose_(0, 1).shape == (2, 3)
    assert v.swapaxes_(0, 1).shape == (3, 2)
    assert v.swapdims_(0, 1).shape == (2, 3)
    assert v.unsqueeze_(0).shape == (1, 2, 3)
    assert v.squeeze_(0).shape == (2, 3)

    r = lucid.empty(16, device=device)
    assert (r.exponential_(2.0) >= 0).all().item()
    assert r.cauchy_(0.0, 1.0).shape == (16,)
    assert (r.log_normal_(0.0, 1.0) > 0).all().item()
    assert (r.geometric_(0.5) >= 1).all().item()
    assert ((r.random_(0, 5) >= 0) & (r < 5)).all().item()

    w = lucid.signal.windows.hann(8, device=device, layout=lucid.strided, requires_grad=True)
    assert w.shape == (8,) and w.requires_grad and w.layout is lucid.strided
