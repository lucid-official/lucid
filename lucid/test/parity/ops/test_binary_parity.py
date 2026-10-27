"""Reference parity for binary ops."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import lucid
from lucid.test._helpers.compare import assert_close

# ── Value parity: (name, lucid_fn, ref_attr, lo, hi, atol) ───────────────────
_BIN_OPS: list[tuple[str, Callable, str, float, float, float]] = [
    # arithmetic
    ("add", lambda a, b: a + b, "add", -2.0, 2.0, 1e-6),
    ("sub", lambda a, b: a - b, "sub", -2.0, 2.0, 1e-6),
    ("mul", lambda a, b: a * b, "mul", -2.0, 2.0, 1e-6),
    ("div", lambda a, b: a / b, "div", 0.5, 2.0, 1e-5),
    ("pow", lambda a, b: a**b, "pow", 0.5, 2.0, 1e-4),
    # min / max
    ("maximum", lambda a, b: lucid.maximum(a, b), "maximum", -2.0, 2.0, 0.0),
    ("minimum", lambda a, b: lucid.minimum(a, b), "minimum", -2.0, 2.0, 0.0),
    ("fmax", lambda a, b: lucid.fmax(a, b), "fmax", -2.0, 2.0, 0.0),
    ("fmin", lambda a, b: lucid.fmin(a, b), "fmin", -2.0, 2.0, 0.0),
    # angle / distance
    ("atan2", lambda a, b: lucid.atan2(a, b), "atan2", -2.0, 2.0, 1e-5),
    ("hypot", lambda a, b: lucid.hypot(a, b), "hypot", 0.1, 2.0, 1e-4),
    # log-domain
    ("logaddexp", lambda a, b: lucid.logaddexp(a, b), "logaddexp", -2.0, 2.0, 1e-5),
    # modulo / remainder
    ("fmod", lambda a, b: lucid.fmod(a, b), "fmod", 0.3, 3.0, 1e-5),
    ("remainder", lambda a, b: lucid.remainder(a, b), "remainder", 0.3, 3.0, 1e-5),
    # sign-related
    ("copysign", lambda a, b: lucid.copysign(a, b), "copysign", -2.0, 2.0, 1e-6),
    # special
    ("xlogy", lambda a, b: lucid.xlogy(a, b), "xlogy", 0.1, 2.0, 1e-5),
]

# ── Backward parity: (name, lucid_fn, ref_fn, lo, hi, atol) ──────────────────
_BIN_BACKWARD_OPS: list[tuple[str, Callable, Callable, float, float, float]] = [
    (
        "add_bwd",
        lambda a, b: (a + b).sum(),
        lambda r, a, b: (r.add(a, b)).sum(),
        -2.0,
        2.0,
        1e-5,
    ),
    (
        "mul_bwd",
        lambda a, b: (a * b).sum(),
        lambda r, a, b: (r.mul(a, b)).sum(),
        -2.0,
        2.0,
        1e-5,
    ),
    (
        "div_bwd",
        lambda a, b: (a / b).sum(),
        lambda r, a, b: (r.div(a, b)).sum(),
        0.5,
        2.0,
        1e-4,
    ),
    (
        "pow_bwd",
        lambda a, b: (a**b).sum(),
        lambda r, a, b: (r.pow(a, b)).sum(),
        0.5,
        2.0,
        1e-4,
    ),
    (
        "atan2_bwd",
        lambda a, b: lucid.atan2(a, b).sum(),
        lambda r, a, b: r.atan2(a, b).sum(),
        -2.0,
        2.0,
        1e-4,
    ),
]


@pytest.mark.parity
@pytest.mark.parametrize(
    "name,lucid_fn,ref_attr,low,high,atol",
    _BIN_OPS,
    ids=[op[0] for op in _BIN_OPS],
)
def test_binary_parity(
    name: str,
    lucid_fn: Callable,
    ref_attr: str,
    low: float,
    high: float,
    atol: float,
    ref: Any,
) -> None:
    np.random.seed(0)
    shape = (4, 5)
    a = np.random.uniform(low, high, size=shape).astype(np.float32)
    b = np.random.uniform(low, high, size=shape).astype(np.float32)
    la = lucid.tensor(a.copy())
    lb = lucid.tensor(b.copy())
    ra = ref.tensor(a.copy())
    rb = ref.tensor(b.copy())
    l_out = lucid_fn(la, lb)
    r_out = getattr(ref, ref_attr)(ra, rb)
    assert_close(l_out, r_out, atol=atol)


@pytest.mark.parity
def test_matmul_parity(ref: Any) -> None:
    np.random.seed(0)
    a = np.random.uniform(-1.0, 1.0, size=(8, 16)).astype(np.float32)
    b = np.random.uniform(-1.0, 1.0, size=(16, 4)).astype(np.float32)
    assert_close(lucid.tensor(a) @ lucid.tensor(b), ref.tensor(a) @ ref.tensor(b), atol=1e-4)


@pytest.mark.parity
def test_ldexp_parity(ref: Any) -> None:
    np.random.seed(2)
    m = np.random.uniform(0.5, 1.5, size=(4, 5)).astype(np.float32)
    e = np.random.randint(-4, 4, size=(4, 5)).astype(np.int32)
    # Lucid ldexp uses float exponent; reference uses int.  Cast before passing.
    l_out = lucid.ldexp(lucid.tensor(m.copy()), lucid.tensor(e.astype(np.float32)))
    r_out = ref.ldexp(ref.tensor(m.copy()), ref.tensor(e.copy()))
    assert_close(l_out, r_out, atol=1e-5)


@pytest.mark.parity
def test_gcd_parity(ref: Any) -> None:
    a = np.array([12, 18, 24, 100], dtype=np.int32)
    b = np.array([8, 12, 36, 75], dtype=np.int32)
    l_out = lucid.gcd(lucid.tensor(a.copy()), lucid.tensor(b.copy()))
    r_out = ref.gcd(ref.tensor(a.copy()), ref.tensor(b.copy()))
    assert_close(l_out, r_out, atol=0.0)


@pytest.mark.parity
def test_lcm_parity(ref: Any) -> None:
    a = np.array([4, 6, 8, 10], dtype=np.int32)
    b = np.array([6, 9, 12, 15], dtype=np.int32)
    l_out = lucid.lcm(lucid.tensor(a.copy()), lucid.tensor(b.copy()))
    r_out = ref.lcm(ref.tensor(a.copy()), ref.tensor(b.copy()))
    assert_close(l_out, r_out, atol=0.0)


@pytest.mark.parity
def test_heaviside_parity(ref: Any) -> None:
    np.random.seed(3)
    x = np.random.uniform(-2.0, 2.0, size=(4, 5)).astype(np.float32)
    v = np.full_like(x, 0.5)
    l_out = lucid.heaviside(lucid.tensor(x.copy()), lucid.tensor(v.copy()))
    r_out = ref.heaviside(ref.tensor(x.copy()), ref.tensor(v.copy()))
    assert_close(l_out, r_out, atol=0.0)


@pytest.mark.parity
def test_addcmul_parity(ref: Any) -> None:
    np.random.seed(4)
    inp = np.random.uniform(-1.0, 1.0, size=(4, 5)).astype(np.float32)
    t1 = np.random.uniform(-1.0, 1.0, size=(4, 5)).astype(np.float32)
    t2 = np.random.uniform(-1.0, 1.0, size=(4, 5)).astype(np.float32)
    l_out = lucid.addcmul(
        lucid.tensor(inp.copy()),
        lucid.tensor(t1.copy()),
        lucid.tensor(t2.copy()),
        value=0.5,
    )
    r_out = ref.addcmul(
        ref.tensor(inp.copy()), ref.tensor(t1.copy()), ref.tensor(t2.copy()), value=0.5
    )
    assert_close(l_out, r_out, atol=1e-5)


@pytest.mark.parity
def test_addcdiv_parity(ref: Any) -> None:
    np.random.seed(5)
    inp = np.random.uniform(-1.0, 1.0, size=(4, 5)).astype(np.float32)
    t1 = np.random.uniform(-1.0, 1.0, size=(4, 5)).astype(np.float32)
    t2 = np.random.uniform(0.5, 2.0, size=(4, 5)).astype(np.float32)
    l_out = lucid.addcdiv(
        lucid.tensor(inp.copy()),
        lucid.tensor(t1.copy()),
        lucid.tensor(t2.copy()),
        value=0.5,
    )
    r_out = ref.addcdiv(
        ref.tensor(inp.copy()), ref.tensor(t1.copy()), ref.tensor(t2.copy()), value=0.5
    )
    assert_close(l_out, r_out, atol=1e-5)


@pytest.mark.parity
@pytest.mark.parametrize(
    "name,lucid_fn,ref_fn,low,high,atol",
    _BIN_BACKWARD_OPS,
    ids=[op[0] for op in _BIN_BACKWARD_OPS],
)
def test_binary_backward_parity(
    name: str,
    lucid_fn: Callable,
    ref_fn: Callable,
    low: float,
    high: float,
    atol: float,
    ref: Any,
) -> None:
    np.random.seed(10)
    shape = (3, 4)
    a = np.random.uniform(low, high, size=shape).astype(np.float32)
    b = np.random.uniform(low, high, size=shape).astype(np.float32)

    al = lucid.tensor(a.copy(), requires_grad=True)
    bl = lucid.tensor(b.copy(), requires_grad=True)
    lucid_fn(al, bl).backward()

    ar = ref.tensor(a.copy(), requires_grad=True)
    br = ref.tensor(b.copy(), requires_grad=True)
    ref_fn(ref, ar, br).backward()

    assert_close(al.grad, ar.grad, atol=atol)
    assert_close(bl.grad, br.grad, atol=atol)
