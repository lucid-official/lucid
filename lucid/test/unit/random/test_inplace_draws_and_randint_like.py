"""``Tensor.normal_`` / ``uniform_`` and ``lucid.randint_like`` (LCD-547).

None of the three existed.  The draws read the one default stream
``manual_seed`` controls (LCD-583) or a ``generator=``; the in-place ones
keep a tensor that requires grad requiring it without reading its old
values — the ``bernoulli_`` they share that path with turned an infinite or
NaN entry into NaN.  ``nn.init.uniform_`` / ``normal_`` are the methods
under ``no_grad``, so the two cannot draw differently (they did: equal
bounds, a complex mean).  An integer draw lands only in a dtype that holds
it — ``randint_like``, ``randint`` and ``randperm`` wrapped or overflowed.
"""

import math
import warnings
from collections.abc import Callable
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
    # ``low=`` alone binds neither of the reference's overloads; one bound
    # by position, or ``high=`` alone, is the high bound.
    with pytest.raises(TypeError, match="low= was given without high="):
        lucid.randint_like(x, low=3)
    with pytest.raises(TypeError, match="high"):
        lucid.randint_like(x)
    assert lucid.randint_like(x, 1).tolist() == lucid.randint_like(x, high=1).tolist()
    assert set(lucid.randint_like(x, low=2, high=3).tolist()) == {2.0}


# ── the result's dtype holds the drawn integers ─────────────────────────────

#: ``(dtype, the largest high it takes, the least low it takes)``: an int
#: type's range, bool's ``{0, 1}``, a floating type's largest finite value.
_EDGES = [
    ("int8", 2**7, -(2**7)),
    ("int16", 2**15, -(2**15)),
    ("int32", 2**31, -(2**31)),
    ("bool", 2, 0),
    ("float16", 65505, -65504),
]


@pytest.mark.filterwarnings("ignore:randint_like.*not uniform:UserWarning")
@pytest.mark.parametrize("dtype,top,bottom", _EDGES, ids=[e[0] for e in _EDGES])
def test_randint_like_draws_only_what_the_dtype_holds(
    device: str, dtype: str, top: int, bottom: int
) -> None:
    x = lucid.zeros(64, dtype=getattr(lucid, dtype), device=device)
    hi = lucid.randint_like(x, top - 1, top, dtype=lucid.int64)
    assert set(hi.tolist()) == {top - 1}
    assert lucid.randint_like(x, top).dtype == x.dtype
    lo = lucid.randint_like(x, bottom, bottom + 1).to(lucid.int64).tolist()
    assert set(lo) == {bottom}
    with pytest.raises(RuntimeError, match="to - 1 .* out of bounds"):
        lucid.randint_like(x, top + 1)
    with pytest.raises(RuntimeError, match="from=.* out of bounds"):
        lucid.randint_like(x, bottom - 1, bottom)


def test_a_wide_floating_bound_stays_finite_and_says_it_rounds() -> None:
    with pytest.warns(UserWarning, match="not uniform"):
        out = lucid.randint_like(lucid.zeros(64, dtype=lucid.bfloat16), 2**40)
    assert all(math.isfinite(v) for v in out.tolist())
    half = lucid.zeros(4, dtype=lucid.float16)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lucid.randint_like(half, 2049)  # 2048 is the last exact integer
        lucid.randint(-(2**63), 2**63 - 1, (4,))  # an int type is exact throughout
    with pytest.warns(UserWarning, match=r"\[-2048, 2048\]"):
        lucid.randint_like(half, 2050)
    with pytest.raises(NotImplementedError, match="complex"):
        lucid.randint_like(lucid.zeros(3, dtype=lucid.complex64), 3)


@pytest.mark.filterwarnings("ignore:randint.*not uniform:UserWarning")
@pytest.mark.parametrize("dtype,top,bottom", _EDGES, ids=[e[0] for e in _EDGES])
def test_randint_draws_only_what_the_dtype_holds(
    device: str, dtype: str, top: int, bottom: int
) -> None:
    dt = getattr(lucid, dtype)
    edge = lucid.randint(top - 1, top, (8,), dtype=dt, device=device)
    assert set(edge.to(lucid.int64).tolist()) == {top - 1}
    with pytest.raises(RuntimeError, match="to - 1 .* out of bounds"):
        lucid.randint(0, top + 1, (8,), dtype=dt, device=device)
    with pytest.raises(RuntimeError, match="from=.* out of bounds"):
        lucid.randint(bottom - 1, 0, (8,), dtype=dt, device=device)
    with pytest.raises(RuntimeError, match="out of bounds"):
        lucid.randint(0, 5, (8,), dtype=lucid.bool, device=device)


@pytest.mark.parametrize(
    "dtype,most", [("int8", 128), ("int16", 2**15), ("float16", 2049)]
)
def test_randperm_refuses_what_the_dtype_cannot_hold_exactly(
    device: str, dtype: str, most: int
) -> None:
    dt = getattr(lucid, dtype)
    if most <= 4096:
        perm = lucid.randperm(most, dtype=dt, device=device).to(lucid.int64)
        assert sorted(perm.tolist()) == list(range(most))
    with pytest.raises(RuntimeError, match="too large"):
        lucid.randperm(most + 1, dtype=dt, device=device)
    with pytest.raises(NotImplementedError, match="randperm"):
        lucid.randperm(3, dtype=lucid.bool, device=device)


@pytest.mark.parity
@pytest.mark.filterwarnings("ignore::UserWarning")  # both round past 2048 in float16
@pytest.mark.parametrize("dtype,top,bottom", _EDGES, ids=[e[0] for e in _EDGES])
def test_the_dtype_edges_are_the_references(
    ref: ModuleType, dtype: str, top: int, bottom: int
) -> None:
    for mod in (lucid, ref):
        x = mod.zeros(3, dtype=getattr(mod, dtype))
        mod.randint_like(x, top)
        mod.randint_like(x, bottom, bottom + 1)
        with pytest.raises(RuntimeError):
            mod.randint_like(x, top + 1)
        with pytest.raises(RuntimeError):
            mod.randint_like(x, bottom - 1, bottom)
    for mod in (lucid, ref):
        with pytest.raises(TypeError):
            mod.randint_like(mod.zeros(3), low=3)
        with pytest.raises(RuntimeError):
            mod.randperm(129, dtype=mod.int8)


# ── one owner of the in-place draws ─────────────────────────────────────────


def _rand_after(seed: int, fill: Callable[[], object]) -> list[float]:
    """What ``rand`` draws next once ``fill()`` has drawn from the stream."""
    lucid.manual_seed(seed)
    fill()
    return lucid.rand(4).tolist()


@pytest.mark.parametrize("dtype,parts", [("float32", 1), ("complex64", 2)])
def test_equal_bounds_move_the_stream_as_any_bounds_do(
    device: str, dtype: str, parts: int
) -> None:
    dt = getattr(lucid, dtype)
    as_rand = _rand_after(9, lambda: lucid.rand(6 * parts, device=device))
    fills = (
        lambda: lucid.empty(6, dtype=dt, device=device).uniform_(1.5, 1.5),
        lambda: lucid.empty(6, dtype=dt, device=device).uniform_(0.0, 1.0),
        lambda: lucid.nn.init.uniform_(
            lucid.empty(6, dtype=dt, device=device), 1.5, 1.5
        ),
    )
    for fill in fills:
        assert _rand_after(9, fill) == as_rand
    z = lucid.empty(6, dtype=dt, device=device).uniform_(1.5, 1.5)
    assert set(z.tolist()) == {1.5 + 1.5j} if parts == 2 else {1.5}


@pytest.mark.parametrize(
    "name,args",
    [("uniform_", (-2.0, 3.0)), ("uniform_", (1.0, 1.0)), ("normal_", (1.0, 0.5))],
)
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_nn_init_draws_what_the_method_draws(
    device: str, name: str, args: tuple[float, float], dtype: str
) -> None:
    dt = getattr(lucid, dtype)
    lucid.manual_seed(4)
    method = getattr(lucid.empty(8, dtype=dt, device=device), name)(*args).tolist()
    lucid.manual_seed(4)
    param = lucid.nn.Parameter(lucid.empty(8, dtype=dt, device=device))
    init = getattr(lucid.nn.init, name)(param, *args)
    assert init is param and param.requires_grad and param.is_leaf
    assert param.tolist() == method


def test_a_complex_nn_init_normal_puts_the_mean_on_both_parts() -> None:
    z = lucid.nn.init.normal_(lucid.empty(20000, dtype=lucid.complex64), 1.0, 2.0)
    for part in (z.real, z.imag):
        assert abs(float(part.mean()) - 1.0) < 0.05
        assert abs(float(part.std()) - math.sqrt(2.0)) < 0.05


@pytest.mark.parity
def test_equal_bounds_move_the_reference_stream_too(ref: ModuleType) -> None:
    def after(fill: Callable[[], object]) -> list[float]:
        ref.manual_seed(9)
        fill()
        return ref.rand(4).tolist()

    assert after(lambda: ref.empty(6).uniform_(1.5, 1.5)) == after(lambda: ref.rand(6))
    assert after(lambda: ref.empty(6, dtype=ref.complex64).uniform_(1.5, 1.5)) == after(
        lambda: ref.rand(12)
    )
    z = ref.nn.init.normal_(ref.empty(20000, dtype=ref.complex64), 1.0, 2.0)
    assert abs(z.imag.mean().item() - 1.0) < 0.05


@pytest.mark.parity
def test_the_draws_exist_and_refuse_as_the_reference(ref: ModuleType) -> None:
    for name in ("normal_", "uniform_"):
        assert hasattr(ref.Tensor, name)
    assert ref.randint_like(ref.zeros(3), 10).dtype == ref.float32
    assert lucid.randint_like(lucid.zeros(3), 10).dtype == lucid.float32
    for ours, theirs in (
        (
            lambda: lucid.zeros(3, dtype=lucid.int64).normal_(),
            lambda: ref.zeros(3, dtype=ref.int64).normal_(),
        ),
        (lambda: lucid.zeros(3).normal_(0, -1), lambda: ref.zeros(3).normal_(0, -1)),
        (lambda: lucid.zeros(3).uniform_(2, 1), lambda: ref.zeros(3).uniform_(2, 1)),
        (
            lambda: lucid.randint_like(lucid.zeros(3), 5, 2),
            lambda: ref.randint_like(ref.zeros(3), 5, 2),
        ),
    ):
        with pytest.raises(RuntimeError):
            theirs()
        with pytest.raises(RuntimeError):
            ours()
