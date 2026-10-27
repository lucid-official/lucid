"""Tests for 0-d CPU scalar tensor device mixing with Metal (GPU) operands.

Covers GitHub #317 (AE-13):
  * A 0-d CPU tensor (``ndim == 0`` and ``device == "cpu"``) may mix with a
    Metal operand in either operand order across binary arithmetic, comparison,
    bitwise, logical, composite, special, and ternary (``where``) ops.
  * The output lives on ``"metal"`` and matches the reference framework.
  * When ``requires_grad=True`` on either or both operands, ``.backward()``
    propagates gradients to both the Metal tensor (on ``"metal"``) and the 0-d
    CPU tensor (on ``"cpu"``, ``shape == ()``) via ``ToDeviceBackward``.
  * Every other cross-device combination — ``(0-d metal, shaped cpu)``,
    ``(shaped cpu, 0-d metal)``, ``(shaped cpu, shaped metal)``,
    ``(shaped metal, shaped cpu)`` — still raises ``DeviceMismatch``.
  * In-place binary ops ``self.op_(other)`` allow ``self`` on Metal with a 0-d
    CPU ``other``, but refuse ``self`` on CPU (even 0-d CPU) when ``other`` is
    on Metal.
"""

import numpy as np
import pytest

import lucid
from lucid._C.engine import DeviceMismatch

# ── Differentiable binary ops ───────────────────────────────────────────────

DIFF_BINARY_OPS = [
    ("add", lambda a, b: a + b, lambda r, a, b: a + b),
    ("sub", lambda a, b: a - b, lambda r, a, b: a - b),
    ("mul", lambda a, b: a * b, lambda r, a, b: a * b),
    ("div", lambda a, b: a / b, lambda r, a, b: a / b),
    ("pow", lambda a, b: a**b, lambda r, a, b: r.pow(a, b)),
    ("maximum", lambda a, b: lucid.maximum(a, b), lambda r, a, b: r.maximum(a, b)),
    ("minimum", lambda a, b: lucid.minimum(a, b), lambda r, a, b: r.minimum(a, b)),
    ("fmod", lambda a, b: lucid.fmod(a, b), lambda r, a, b: r.fmod(a, b)),
    ("remainder", lambda a, b: a % b, lambda r, a, b: r.remainder(a, b)),
    ("atan2", lambda a, b: lucid.atan2(a, b), lambda r, a, b: r.atan2(a, b)),
    ("hypot", lambda a, b: lucid.hypot(a, b), lambda r, a, b: r.hypot(a, b)),
    ("logaddexp", lambda a, b: lucid.logaddexp(a, b), lambda r, a, b: r.logaddexp(a, b)),
    ("xlogy", lambda a, b: lucid.xlogy(a, b), lambda r, a, b: r.xlogy(a, b)),
    ("xlog1py", lambda a, b: lucid.special.xlog1py(a, b), lambda r, a, b: r.special.xlog1py(a, b)),
]

# ── Non-differentiable / special binary ops ─────────────────────────────────

NONDIFF_FLOAT_BINARY_OPS = [
    ("floordiv", lambda a, b: a // b, lambda r, a, b: r.div(a, b, rounding_mode="floor")),
    ("equal", lambda a, b: a == b, lambda r, a, b: a == b),
    ("not_equal", lambda a, b: a != b, lambda r, a, b: a != b),
    ("less", lambda a, b: a < b, lambda r, a, b: a < b),
    ("less_equal", lambda a, b: a <= b, lambda r, a, b: a <= b),
    ("greater", lambda a, b: a > b, lambda r, a, b: a > b),
    ("greater_equal", lambda a, b: a >= b, lambda r, a, b: a >= b),
    ("nextafter", lambda a, b: lucid.nextafter(a, b), lambda r, a, b: r.nextafter(a, b)),
    ("isclose", lambda a, b: lucid.isclose(a, b), lambda r, a, b: r.isclose(a, b)),
    ("logical_and", lambda a, b: lucid.logical_and(a, b), lambda r, a, b: r.logical_and(a, b)),
    ("logical_or", lambda a, b: lucid.logical_or(a, b), lambda r, a, b: r.logical_or(a, b)),
    ("logical_xor", lambda a, b: lucid.logical_xor(a, b), lambda r, a, b: r.logical_xor(a, b)),
    ("igamma", lambda a, b: lucid.special.gammainc(a, b), lambda r, a, b: r.special.gammainc(a, b)),
    (
        "igammac",
        lambda a, b: lucid.special.gammaincc(a, b),
        lambda r, a, b: r.special.gammaincc(a, b),
    ),
    ("zeta", lambda a, b: lucid.special.zeta(a, b), lambda r, a, b: r.special.zeta(a, b)),
]

BITWISE_INT_OPS = [
    ("bitwise_and", lambda a, b: a & b, lambda r, a, b: a & b),
    ("bitwise_or", lambda a, b: a | b, lambda r, a, b: a | b),
    ("bitwise_xor", lambda a, b: a ^ b, lambda r, a, b: a ^ b),
    ("left_shift", lambda a, b: a << b, lambda r, a, b: a << b),
    ("right_shift", lambda a, b: a >> b, lambda r, a, b: a >> b),
    # Composites that compute on the host, off the engine's device check.
    ("gcd", lambda a, b: lucid.gcd(a, b), lambda r, a, b: r.gcd(a, b)),
    ("lcm", lambda a, b: lucid.lcm(a, b), lambda r, a, b: r.lcm(a, b)),
]

INPLACE_OPS = [
    ("add_", lambda a, b: a.add_(b), lambda r, a, b: a.add_(b)),
    ("sub_", lambda a, b: a.sub_(b), lambda r, a, b: a.sub_(b)),
    ("mul_", lambda a, b: a.mul_(b), lambda r, a, b: a.mul_(b)),
    ("div_", lambda a, b: a.div_(b), lambda r, a, b: a.div_(b)),
    ("pow_", lambda a, b: a.pow_(b), lambda r, a, b: a.pow_(b)),
    ("maximum_", lambda a, b: a.maximum_(b), lambda r, a, b: r.maximum(a, b)),
    ("minimum_", lambda a, b: a.minimum_(b), lambda r, a, b: r.minimum(a, b)),
]


# ── 1. Forward + backward for (0-d CPU, Metal) and (Metal, 0-d CPU) ────────


@pytest.mark.parametrize("name,l_fn,t_fn", DIFF_BINARY_OPS, ids=[x[0] for x in DIFF_BINARY_OPS])
@pytest.mark.parametrize("cpu_first", [True, False], ids=["cpu0_metal", "metal_cpu0"])
@pytest.mark.parametrize("metal_is_0d", [False, True], ids=["shaped_metal", "0d_metal"])
@pytest.mark.parametrize(
    "grad_cpu,grad_metal", [(False, False), (True, False), (False, True), (True, True)]
)
def test_diff_binary_zero_dim_cpu_with_metal(
    ref, name, l_fn, t_fn, cpu_first, metal_is_0d, grad_cpu, grad_metal
):
    c_val = np.array(2.25, dtype=np.float32)
    m_val = (
        np.array(1.5, dtype=np.float32)
        if metal_is_0d
        else np.array([1.2, 2.7, 3.5], dtype=np.float32)
    )

    c_l = lucid.tensor(c_val, dtype=lucid.float32, device="cpu", requires_grad=grad_cpu)
    m_l = lucid.tensor(m_val, dtype=lucid.float32, device="metal", requires_grad=grad_metal)

    c_t = ref.tensor(c_val, dtype=ref.float32, requires_grad=grad_cpu)
    m_t = ref.tensor(m_val, dtype=ref.float32, requires_grad=grad_metal)

    out_l = l_fn(c_l, m_l) if cpu_first else l_fn(m_l, c_l)
    out_t = t_fn(ref, c_t, m_t) if cpu_first else t_fn(ref, m_t, c_t)

    assert out_l.device.type == "metal"
    assert tuple(out_l.shape) == tuple(out_t.shape)
    np.testing.assert_allclose(out_l.numpy(), out_t.detach().numpy(), rtol=1e-5, atol=1e-5)

    if grad_cpu or grad_metal:
        out_l.sum().backward()
        out_t.sum().backward()

        if grad_cpu:
            assert c_l.grad is not None
            assert c_l.grad.device.type == "cpu"
            assert tuple(c_l.grad.shape) == ()
            np.testing.assert_allclose(
                c_l.grad.numpy(), c_t.grad.detach().numpy(), rtol=1e-4, atol=1e-4
            )
        else:
            assert c_l.grad is None

        if grad_metal:
            assert m_l.grad is not None
            assert m_l.grad.device.type == "metal"
            assert tuple(m_l.grad.shape) == tuple(m_t.shape)
            np.testing.assert_allclose(
                m_l.grad.numpy(), m_t.grad.detach().numpy(), rtol=1e-4, atol=1e-4
            )
        else:
            assert m_l.grad is None


@pytest.mark.parametrize(
    "name,l_fn,t_fn", NONDIFF_FLOAT_BINARY_OPS, ids=[x[0] for x in NONDIFF_FLOAT_BINARY_OPS]
)
@pytest.mark.parametrize("cpu_first", [True, False], ids=["cpu0_metal", "metal_cpu0"])
def test_nondiff_and_special_binary_zero_dim_cpu_with_metal(ref, name, l_fn, t_fn, cpu_first):
    c_val = np.array(2.5, dtype=np.float32)
    m_val = np.array([1.5, 2.5, 3.5], dtype=np.float32)

    c_l = lucid.tensor(c_val, dtype=lucid.float32, device="cpu")
    m_l = lucid.tensor(m_val, dtype=lucid.float32, device="metal")

    c_t = ref.tensor(c_val, dtype=ref.float32)
    m_t = ref.tensor(m_val, dtype=ref.float32)

    out_l = l_fn(c_l, m_l) if cpu_first else l_fn(m_l, c_l)
    out_t = t_fn(ref, c_t, m_t) if cpu_first else t_fn(ref, m_t, c_t)

    assert out_l.device.type == "metal"
    assert tuple(out_l.shape) == tuple(out_t.shape)
    np.testing.assert_allclose(out_l.numpy(), out_t.numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name,l_fn,t_fn", BITWISE_INT_OPS, ids=[x[0] for x in BITWISE_INT_OPS])
@pytest.mark.parametrize("cpu_first", [True, False], ids=["cpu0_metal", "metal_cpu0"])
def test_bitwise_int_zero_dim_cpu_with_metal(ref, name, l_fn, t_fn, cpu_first):
    c_val = np.array(2, dtype=np.int32)
    m_val = np.array([1, 2, 3, 6], dtype=np.int32)

    c_l = lucid.tensor(c_val, dtype=lucid.int32, device="cpu")
    m_l = lucid.tensor(m_val, dtype=lucid.int32, device="metal")

    c_t = ref.tensor(c_val, dtype=ref.int32)
    m_t = ref.tensor(m_val, dtype=ref.int32)

    out_l = l_fn(c_l, m_l) if cpu_first else l_fn(m_l, c_l)
    out_t = t_fn(ref, c_t, m_t) if cpu_first else t_fn(ref, m_t, c_t)

    assert out_l.device.type == "metal"
    assert tuple(out_l.shape) == tuple(out_t.shape)
    np.testing.assert_array_equal(out_l.numpy(), out_t.numpy())


# ── 2. Rejection of non-(0-d CPU) cross-device pairs ───────────────────────


@pytest.mark.parametrize("name,l_fn,_", DIFF_BINARY_OPS, ids=[x[0] for x in DIFF_BINARY_OPS])
@pytest.mark.parametrize(
    "a_spec,b_spec",
    [
        ((0, "metal"), (1, "cpu")),
        ((1, "cpu"), (0, "metal")),
        ((1, "cpu"), (1, "metal")),
        ((1, "metal"), (1, "cpu")),
    ],
    ids=["0d_metal_shaped_cpu", "shaped_cpu_0d_metal", "shaped_cpu_metal", "shaped_metal_cpu"],
)
def test_binary_rejects_disallowed_cross_device_pairs(name, l_fn, _, a_spec, b_spec):
    def _make(ndim: int, dev: str) -> lucid.Tensor:
        arr = (
            np.array(2.0, dtype=np.float32) if ndim == 0 else np.array([1.5, 2.5], dtype=np.float32)
        )
        return lucid.tensor(arr, dtype=lucid.float32, device=dev)

    a = _make(*a_spec)
    b = _make(*b_spec)
    with pytest.raises(DeviceMismatch):
        l_fn(a, b)


@pytest.mark.parametrize(
    "name,l_fn,_", NONDIFF_FLOAT_BINARY_OPS, ids=[x[0] for x in NONDIFF_FLOAT_BINARY_OPS]
)
@pytest.mark.parametrize(
    "a_spec,b_spec",
    [((0, "metal"), (1, "cpu")), ((1, "cpu"), (0, "metal")), ((1, "cpu"), (1, "metal"))],
    ids=["0d_metal_shaped_cpu", "shaped_cpu_0d_metal", "shaped_cpu_metal"],
)
def test_nondiff_binary_rejects_disallowed_cross_device_pairs(name, l_fn, _, a_spec, b_spec):
    def _make(ndim: int, dev: str) -> lucid.Tensor:
        arr = (
            np.array(2.0, dtype=np.float32) if ndim == 0 else np.array([1.5, 2.5], dtype=np.float32)
        )
        return lucid.tensor(arr, dtype=lucid.float32, device=dev)

    a = _make(*a_spec)
    b = _make(*b_spec)
    with pytest.raises(DeviceMismatch):
        l_fn(a, b)


@pytest.mark.parametrize("name,l_fn,_", BITWISE_INT_OPS, ids=[x[0] for x in BITWISE_INT_OPS])
@pytest.mark.parametrize(
    "a_spec,b_spec",
    [
        ((0, "metal"), (1, "cpu")),
        ((1, "cpu"), (0, "metal")),
        ((1, "cpu"), (1, "metal")),
        ((1, "metal"), (1, "cpu")),
    ],
    ids=["0d_metal_shaped_cpu", "shaped_cpu_0d_metal", "shaped_cpu_metal", "shaped_metal_cpu"],
)
def test_bitwise_int_rejects_disallowed_cross_device_pairs(name, l_fn, _, a_spec, b_spec):
    def _make(ndim: int, dev: str) -> lucid.Tensor:
        arr = np.array(2, dtype=np.int32) if ndim == 0 else np.array([1, 6], dtype=np.int32)
        return lucid.tensor(arr, dtype=lucid.int32, device=dev)

    with pytest.raises(DeviceMismatch):
        l_fn(_make(*a_spec), _make(*b_spec))


# ── 3. Ternary ``where`` ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "cond_dev,x_dev,y_dev",
    [
        ("metal", "metal", "cpu"),
        ("metal", "cpu", "metal"),
        ("metal", "cpu", "cpu"),
        ("cpu", "metal", "metal"),
        ("cpu", "metal", "cpu"),
        ("cpu", "cpu", "metal"),
    ],
)
def test_where_allows_zero_dim_cpu_operands_with_metal(ref, cond_dev, x_dev, y_dev):
    cond_l = lucid.tensor(
        np.array(True) if cond_dev == "cpu" else np.array([True, False, True]),
        dtype=lucid.bool_,
        device=cond_dev,
    )
    x_l = lucid.tensor(
        (
            np.array(3.0, dtype=np.float32)
            if x_dev == "cpu"
            else np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ),
        dtype=lucid.float32,
        device=x_dev,
        requires_grad=True,
    )
    y_l = lucid.tensor(
        (
            np.array(-4.0, dtype=np.float32)
            if y_dev == "cpu"
            else np.array([4.0, 5.0, 6.0], dtype=np.float32)
        ),
        dtype=lucid.float32,
        device=y_dev,
        requires_grad=True,
    )

    cond_t = ref.tensor(cond_l.numpy(), dtype=ref.bool)
    x_t = ref.tensor(x_l.numpy(), dtype=ref.float32, requires_grad=True)
    y_t = ref.tensor(y_l.numpy(), dtype=ref.float32, requires_grad=True)

    out_l = lucid.where(cond_l, x_l, y_l)
    out_t = ref.where(cond_t, x_t, y_t)

    assert out_l.device.type == "metal"
    assert tuple(out_l.shape) == tuple(out_t.shape)
    np.testing.assert_allclose(out_l.numpy(), out_t.detach().numpy(), rtol=1e-5, atol=1e-5)

    out_l.sum().backward()
    out_t.sum().backward()

    assert x_l.grad is not None and x_l.grad.device.type == x_dev
    assert y_l.grad is not None and y_l.grad.device.type == y_dev
    np.testing.assert_allclose(x_l.grad.numpy(), x_t.grad.detach().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(y_l.grad.numpy(), y_t.grad.detach().numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "cond_spec,x_spec,y_spec",
    [
        ((1, "cpu"), (1, "metal"), (1, "metal")),
        ((1, "metal"), (1, "cpu"), (1, "metal")),
        ((1, "metal"), (1, "metal"), (1, "cpu")),
        ((0, "metal"), (1, "cpu"), (1, "cpu")),
    ],
)
def test_where_rejects_shaped_cpu_with_metal(cond_spec, x_spec, y_spec):
    cond = lucid.tensor(
        np.array(True) if cond_spec[0] == 0 else np.array([True, False]),
        dtype=lucid.bool_,
        device=cond_spec[1],
    )
    x = lucid.tensor(
        (
            np.array(1.0, dtype=np.float32)
            if x_spec[0] == 0
            else np.array([1.0, 2.0], dtype=np.float32)
        ),
        dtype=lucid.float32,
        device=x_spec[1],
    )
    y = lucid.tensor(
        (
            np.array(3.0, dtype=np.float32)
            if y_spec[0] == 0
            else np.array([3.0, 4.0], dtype=np.float32)
        ),
        dtype=lucid.float32,
        device=y_spec[1],
    )
    with pytest.raises(DeviceMismatch):
        lucid.where(cond, x, y)


# ── 4. In-place binary ops ──────────────────────────────────────────────────


@pytest.mark.parametrize("name,l_fn,t_fn", INPLACE_OPS, ids=[x[0] for x in INPLACE_OPS])
def test_inplace_metal_target_with_zero_dim_cpu_rhs(ref, name, l_fn, t_fn):
    a_base = lucid.tensor([1.5, 2.5, 3.5], dtype=lucid.float32, device="metal", requires_grad=True)
    a_l = a_base * 1.0
    b_l = lucid.tensor(2.0, dtype=lucid.float32, device="cpu", requires_grad=True)

    a_t_base = ref.tensor([1.5, 2.5, 3.5], dtype=ref.float32, requires_grad=True)
    a_t = a_t_base * 1.0
    b_t = ref.tensor(2.0, dtype=ref.float32, requires_grad=True)

    res_l = l_fn(a_l, b_l)
    res_t = t_fn(ref, a_t, b_t)

    assert res_l is a_l
    assert a_l.device.type == "metal"
    np.testing.assert_allclose(a_l.numpy(), res_t.detach().numpy(), rtol=1e-5, atol=1e-5)

    a_l.sum().backward()
    res_t.sum().backward()

    assert a_base.grad is not None and a_base.grad.device.type == "metal"
    assert b_l.grad is not None and b_l.grad.device.type == "cpu" and tuple(b_l.grad.shape) == ()
    np.testing.assert_allclose(
        a_base.grad.numpy(), a_t_base.grad.detach().numpy(), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(b_l.grad.numpy(), b_t.grad.detach().numpy(), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("name,l_fn,_", INPLACE_OPS, ids=[x[0] for x in INPLACE_OPS])
@pytest.mark.parametrize(
    "a_spec,b_spec",
    [
        ((0, "cpu"), (0, "metal")),
        ((0, "cpu"), (1, "metal")),
        ((1, "cpu"), (0, "metal")),
        ((1, "metal"), (1, "cpu")),
    ],
    ids=[
        "0d_cpu_target_0d_metal",
        "0d_cpu_target_shaped_metal",
        "shaped_cpu_target",
        "shaped_cpu_rhs",
    ],
)
def test_inplace_rejects_cpu_target_with_metal_rhs_or_shaped_cpu_rhs(name, l_fn, _, a_spec, b_spec):
    a = lucid.tensor(
        (
            np.array(2.0, dtype=np.float32)
            if a_spec[0] == 0
            else np.array([1.0, 2.0], dtype=np.float32)
        ),
        dtype=lucid.float32,
        device=a_spec[1],
    )
    b = lucid.tensor(
        (
            np.array(3.0, dtype=np.float32)
            if b_spec[0] == 0
            else np.array([3.0, 4.0], dtype=np.float32)
        ),
        dtype=lucid.float32,
        device=b_spec[1],
    )
    with pytest.raises(DeviceMismatch):
        l_fn(a, b)


# ── 5. Half-precision opmath & complex ──────────────────────────────────────


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=["f16", "bf16"])
@pytest.mark.parametrize("cpu_first", [True, False], ids=["cpu0_metal", "metal_cpu0"])
def test_half_metal_with_zero_dim_cpu_float32_scalar(dtype, cpu_first):
    c0 = lucid.tensor(2.0, dtype=lucid.float32, device="cpu")
    m1 = lucid.tensor([1.5, 3.0], dtype=dtype, device="metal", requires_grad=True)

    out = (c0 * m1) if cpu_first else (m1 * c0)
    assert out.device.type == "metal"
    assert out.dtype == dtype
    np.testing.assert_allclose(out.to(lucid.float32).numpy(), np.array([3.0, 6.0]), rtol=1e-2)
    out.sum().backward()
    assert m1.grad is not None and m1.grad.device.type == "metal"


@pytest.mark.parametrize("cpu_first", [True, False], ids=["cpu0_metal", "metal_cpu0"])
def test_complex_allows_zero_dim_cpu_with_metal(cpu_first):
    c0 = lucid.tensor(1.5, dtype=lucid.float32, device="cpu", requires_grad=True)
    m1 = lucid.tensor([2.0, -3.0], dtype=lucid.float32, device="metal", requires_grad=True)

    z = lucid.complex(c0, m1) if cpu_first else lucid.complex(m1, c0)
    assert z.device.type == "metal"
    assert tuple(z.shape) == (2,)
    loss = (lucid.real(z) + lucid.imag(z)).sum()
    loss.backward()
    assert c0.grad is not None and c0.grad.device.type == "cpu" and tuple(c0.grad.shape) == ()
    assert m1.grad is not None and m1.grad.device.type == "metal"
