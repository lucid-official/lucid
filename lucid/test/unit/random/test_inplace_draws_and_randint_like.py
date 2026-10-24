"""``Tensor.normal_`` / ``uniform_`` and ``lucid.randint_like`` (LCD-547).

None of the three existed.  The draws read the one default stream
``manual_seed`` controls (LCD-583) or a ``generator=``; the in-place ones
keep a tensor that requires grad requiring it without reading its old
values — the ``bernoulli_`` they share that path with turned an infinite or
NaN entry into NaN.
"""

import math
from types import ModuleType

import pytest

import lucid


@pytest.mark.parametrize("method", ["normal_", "uniform_"])
def test_an_inplace_draw_returns_self_and_follows_the_seed(
    device: str, method: str
) -> None:
    x = lucid.zeros(64, device=device)
    lucid.manual_seed(7)
    assert getattr(x, method)() is x
    first = x.tolist()
    lucid.manual_seed(7)
    getattr(x, method)()
    assert x.tolist() == first
    gen = lucid.Generator()
    gen.manual_seed(3)
    a = getattr(lucid.zeros(64, device=device), method)(generator=gen).tolist()
    gen.manual_seed(3)
    b = getattr(lucid.zeros(64, device=device), method)(generator=gen).tolist()
    assert a == b != first


def test_normal_draws_at_the_mean_and_spread(device: str) -> None:
    x = lucid.empty(20000, device=device).normal_(2.0, 0.5)
    assert abs(float(x.mean()) - 2.0) < 0.05
    assert abs(float(x.std()) - 0.5) < 0.05


def test_uniform_draws_in_its_interval(device: str) -> None:
    x = lucid.empty(5000, device=device).uniform_(-2.0, to=3.0)
    assert float(x.min()) >= -2.0 and float(x.max()) < 3.0
    assert abs(float(x.mean()) - 0.5) < 0.1
    assert lucid.empty(3, device=device).uniform_(1.5, 1.5).tolist() == [1.5] * 3


def test_a_complex_tensor_draws_both_parts() -> None:
    z = lucid.empty(20000, dtype=lucid.complex64).normal_(1.0, 2.0)
    for part in (z.real, z.imag):
        assert abs(float(part.mean()) - 1.0) < 0.05
        assert abs(float(part.std()) - math.sqrt(2.0)) < 0.05
    u = lucid.empty(2000, dtype=lucid.complex64).uniform_(1.0, 3.0)
    for part in (u.real, u.imag):
        assert float(part.min()) >= 1.0 and float(part.max()) < 3.0


def test_draws_refuse_what_they_cannot_fill(device: str) -> None:
    with pytest.raises(NotImplementedError, match="normal_"):
        lucid.zeros(3, dtype=lucid.int64, device=device).normal_()
    with pytest.raises(NotImplementedError, match="uniform_"):
        lucid.zeros(3, dtype=lucid.bool, device=device).uniform_()
    with pytest.raises(RuntimeError, match="std"):
        lucid.zeros(3, device=device).normal_(0.0, -1.0)
    with pytest.raises(RuntimeError, match="from=2"):
        lucid.zeros(3, device=device).uniform_(2.0, 1.0)
    leaf = lucid.zeros(3, device=device, requires_grad=True)
    with pytest.raises(RuntimeError, match="leaf"):
        leaf.normal_()


@pytest.mark.parametrize("method", ["normal_", "uniform_", "bernoulli_"])
def test_a_draw_into_a_computed_tensor_keeps_grad_and_reads_no_old_value(
    device: str, method: str
) -> None:
    leaf = lucid.tensor([math.inf, 1.0, math.nan], device=device, requires_grad=True)
    y = leaf * 1.0
    getattr(y, method)()
    assert y.requires_grad
    assert all(math.isfinite(v) for v in y.tolist())
    y.sum().backward()
    assert leaf.grad is not None and leaf.grad.tolist() == [0.0, 0.0, 0.0]


def test_randint_like_takes_the_input_shape_dtype_and_device(device: str) -> None:
    x = lucid.zeros(4, 5, device=device)
    r = lucid.randint_like(x, 10)
    assert (r.shape, r.dtype, r.device) == (x.shape, x.dtype, x.device)
    assert all(float(v).is_integer() and 0 <= v < 10 for v in r.reshape(-1).tolist())
    s = lucid.randint_like(x, 3, 5, dtype=lucid.int32)
    assert s.dtype == lucid.int32 and set(s.reshape(-1).tolist()) <= {3, 4}
    assert lucid.randint_like(x, high=1).tolist() == [[0.0] * 5] * 4
    b = lucid.randint_like(lucid.zeros(64, dtype=lucid.bool, device=device), 2)
    assert b.dtype == lucid.bool
    assert lucid.randint_like(x, 4, requires_grad=True).requires_grad


def test_randint_like_follows_the_seed_and_a_generator(device: str) -> None:
    x = lucid.zeros(32, dtype=lucid.int64, device=device)
    lucid.manual_seed(5)
    a = lucid.randint_like(x, 1000).tolist()
    lucid.manual_seed(5)
    assert lucid.randint_like(x, 1000).tolist() == a
    gen = lucid.Generator()
    gen.manual_seed(1)
    b = lucid.randint_like(x, 1000, generator=gen).tolist()
    gen.manual_seed(1)
    assert lucid.randint_like(x, 1000, generator=gen).tolist() == b


def test_randint_like_refuses_bad_bounds() -> None:
    x = lucid.zeros(3)
    with pytest.raises(RuntimeError, match="low must be less than high"):
        lucid.randint_like(x, 5, 2)
    with pytest.raises(TypeError, match="high"):
        lucid.randint_like(x, 5.5)
    with pytest.raises(TypeError, match="low"):
        lucid.randint_like(x, True, 3)
    with pytest.raises(RuntimeError):
        lucid.randint_like(lucid.zeros(3, dtype=lucid.int64), 4, requires_grad=True)


@pytest.mark.parity
def test_the_draws_exist_and_refuse_as_the_reference(ref: ModuleType) -> None:
    for name in ("normal_", "uniform_"):
        assert hasattr(ref.Tensor, name)
    assert ref.randint_like(ref.zeros(3), 10).dtype == ref.float32
    assert lucid.randint_like(lucid.zeros(3), 10).dtype == lucid.float32
    for ours, theirs in (
        (lambda: lucid.zeros(3, dtype=lucid.int64).normal_(), lambda: ref.zeros(3, dtype=ref.int64).normal_()),
        (lambda: lucid.zeros(3).normal_(0, -1), lambda: ref.zeros(3).normal_(0, -1)),
        (lambda: lucid.zeros(3).uniform_(2, 1), lambda: ref.zeros(3).uniform_(2, 1)),
        (lambda: lucid.randint_like(lucid.zeros(3), 5, 2), lambda: ref.randint_like(ref.zeros(3), 5, 2)),
    ):
        with pytest.raises(RuntimeError):
            theirs()
        with pytest.raises(RuntimeError):
            ours()
