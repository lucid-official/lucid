"""A compiled addition of a zero constant keeps IEEE's signed zero (LCD-446).

``-0.0 + 0`` is ``+0.0``: under IEEE 754 only ``-0`` is the additive
identity. MPSGraph rewrites ``x + c`` and ``x - c`` to ``x`` whenever ``c``
is a constant zero of either sign, so ``compile(lambda t: t + 0)`` handed
back the ``-0.0`` it was given while eager gave ``+0.0``. Python's builtin
``sum`` starts at ``0``, so ``sum(losses)`` takes the same path, and so does
a zero that only passes through a reshape, expand or float cast. The add and
sub emitters now spell the sum so MPSGraph has no constant to fold
(``OpEmitters/_SignedZero.h``), in a form MPSGraph's autodiff passes an
infinite gradient through unchanged.

Both executable caches also compared a constant's value with ``==``, under
which ``0.0`` and ``-0.0`` are the same key — ``t + (-0.0)`` ran the
executable compiled for ``t + 0.0`` — and the Python one let ``1``, ``1.0``
and ``True`` share an entry although each promotes an integer tensor
differently.

Each case asserts that it really compiled, because an eager fallback would
match eager trivially.
"""

import math
from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid.compile import fused_step

from lucid.test.unit.compile._helpers import COMPILE_DEVICE

_FLOAT_DTYPES = [lucid.float32, lucid.float16, lucid.bfloat16]

#: Both zeros, ordinary values, and the non-finite ones a wrong
#: ``x * 0 -> 0`` fold would show.
_VALUES = [-0.0, 0.0, 1.0, -2.5, math.inf, -math.inf, math.nan]


def _zeros_like(t: lucid.Tensor) -> lucid.Tensor:
    return lucid.zeros(*t.shape, dtype=t.dtype, device=t.device)


def _full_zero_like(t: lucid.Tensor) -> lucid.Tensor:
    return lucid.full(tuple(t.shape), 0.0, dtype=t.dtype, device=t.device)


_ELEMENTWISE: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "x+0": lambda t: t + 0,
    "0+x": lambda t: 0 + t,
    "x+0.0": lambda t: t + 0.0,
    "x+(-0.0)": lambda t: t + (-0.0),
    "x*1+0": lambda t: t * 1 + 0,
    "x-0": lambda t: t - 0,
    "x-(-0.0)": lambda t: t - (-0.0),
    "0-x": lambda t: 0 - t,
    "x*1": lambda t: t * 1,
    "x*0": lambda t: t * 0,
    "x/1": lambda t: t / 1,
    "x+zeros": lambda t: t + _zeros_like(t),
    "x+full(0)": lambda t: t + _full_zero_like(t),
}

_REDUCTIONS: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "sum": lambda t: t.sum(),
    "mean": lambda t: t.mean(),
}


def _run_compiled(fn: Callable[[lucid.Tensor], lucid.Tensor], x: lucid.Tensor) -> lucid.Tensor:
    _C_engine.compile.session_cache_clear()
    out = lucid.compile(fn)(x)
    assert (
        _C_engine.compile.session_cache_size() >= 1
    ), "the function ran eagerly, so the comparison below proves nothing"
    return out


def _assert_same_bits(eager: lucid.Tensor, compiled: lucid.Tensor, label: str) -> None:
    # f16 / bf16 widen to f32 exactly, sign of zero included.
    e = eager.to(lucid.float32).numpy().reshape(-1)
    c = compiled.to(lucid.float32).numpy().reshape(-1)
    assert np.array_equal(np.isnan(e), np.isnan(c)), f"{label}: eager {e}, compiled {c}"
    finite = ~np.isnan(e)
    assert np.array_equal(
        e[finite].view(np.uint32), c[finite].view(np.uint32)
    ), f"{label}: eager {e} (signbit {np.signbit(e)}), compiled {c} (signbit {np.signbit(c)})"


@pytest.mark.parametrize("dtype", _FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("name", list(_ELEMENTWISE))
def test_identity_with_a_zero_constant_matches_eager(name: str, dtype: lucid.dtype) -> None:
    fn = _ELEMENTWISE[name]
    x = lucid.tensor(_VALUES, dtype=dtype).to(COMPILE_DEVICE)
    _assert_same_bits(fn(x), _run_compiled(fn, x), f"{name} {dtype}")


@pytest.mark.parametrize("dtype", _FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("name", list(_REDUCTIONS))
def test_reduction_of_negative_zero_matches_eager(name: str, dtype: lucid.dtype) -> None:
    fn = _REDUCTIONS[name]
    x = lucid.tensor([-0.0], dtype=dtype).to(COMPILE_DEVICE)
    _assert_same_bits(fn(x), _run_compiled(fn, x), f"{name} {dtype}")


def test_integer_add_of_zero_is_unchanged() -> None:
    x = lucid.tensor([-3, 0, 7], dtype=lucid.int32).to(COMPILE_DEVICE)
    out = _run_compiled(lambda t: t + 0, x)
    assert out.dtype == lucid.int32
    assert out.numpy().tolist() == [-3, 0, 7]


def _zeros(t: lucid.Tensor, *shape: int, dtype: lucid.dtype | None = None) -> lucid.Tensor:
    return lucid.zeros(*shape, dtype=dtype or t.dtype, device=t.device)


def _other_float(dtype: lucid.dtype) -> lucid.dtype:
    return lucid.float16 if dtype == lucid.float32 else lucid.float32


#: A zero that reaches the add only through ops that move or widen values —
#: each traces to the op in the name, and MPSGraph folds the result alike.
_N = len(_VALUES)
_ROUTED: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "reshape": lambda t: t + _zeros(t, _N, 1).reshape(_N),
    "view": lambda t: t + _zeros(t, _N, 1).view(_N),
    "flatten": lambda t: t + _zeros(t, _N, 1).flatten(),
    "squeeze": lambda t: t + _zeros(t, _N, 1).squeeze(1),
    "unsqueeze": lambda t: t.unsqueeze(0) + _zeros(t, _N).unsqueeze(0),
    "expand": lambda t: t + _zeros(t, 1).expand(_N),
    "broadcast_to": lambda t: t + lucid.broadcast_to(_zeros(t, 1), (_N,)),
    "permute": lambda t: t + _zeros(t, 1, _N).permute(1, 0).reshape(_N),
    "contiguous": lambda t: t + _zeros(t, 1, _N).permute(1, 0).contiguous().reshape(_N),
    "detach": lambda t: t + _zeros(t, _N).detach(),
    "tile": lambda t: t + lucid.tile(_zeros(t, 1), (_N,)),
    "flip": lambda t: t + lucid.flip(_zeros(t, _N), (0,)),
    "roll": lambda t: t + lucid.roll(_zeros(t, _N), 1),
    "astype": lambda t: t + _zeros(t, _N, dtype=_other_float(t.dtype)).to(t.dtype),
    "sub_neg_zero_reshape": lambda t: (
        t - lucid.full((_N, 1), -0.0, dtype=t.dtype, device=t.device).reshape(_N)
    ),
}


@pytest.mark.parametrize("dtype", _FLOAT_DTYPES, ids=str)
@pytest.mark.parametrize("name", list(_ROUTED))
def test_zero_routed_through_a_shape_op_matches_eager(name: str, dtype: lucid.dtype) -> None:
    fn = _ROUTED[name]
    x = lucid.tensor(_VALUES, dtype=dtype).to(COMPILE_DEVICE)
    _assert_same_bits(fn(x), _run_compiled(fn, x), f"{name} {dtype}")


def _add_argument(t: lucid.Tensor, c: object) -> lucid.Tensor:
    return t + c


@pytest.mark.parametrize("order", [(0.0, -0.0), (-0.0, 0.0)], ids=["pos-first", "neg-first"])
def test_python_cache_keeps_zero_arguments_apart(order: tuple[float, float]) -> None:
    x = lucid.tensor([-0.0, 1.0]).to(COMPILE_DEVICE)
    f = lucid.compile(_add_argument)
    for c in order:
        _assert_same_bits(x + c, f(x, c), f"t + {c!r}")
    assert f.cache_info()["entries"] == 2


def test_python_cache_keeps_int_float_and_bool_apart() -> None:
    x = lucid.tensor([1, 2], dtype=lucid.int32).to(COMPILE_DEVICE)
    f = lucid.compile(_add_argument)
    for c in (1, 1.0, True):
        got, want = f(x, c), x + c
        assert got.dtype == want.dtype, f"t + {c!r}"
        assert got.numpy().tolist() == want.numpy().tolist(), f"t + {c!r}"
    assert f.cache_info()["entries"] == 3


def test_python_cache_reuses_a_nan_argument() -> None:
    """A fresh NaN each call is one entry, not a recompile per call."""
    x = lucid.tensor([1.0, 2.0]).to(COMPILE_DEVICE)
    f = lucid.compile(_add_argument)
    _C_engine.compile.session_cache_clear()
    for _ in range(3):
        _assert_same_bits(x + float("nan"), f(x, float("nan")), "t + nan")
    assert f.cache_info()["entries"] == 1
    assert _C_engine.compile.session_cache_size() == 1


def test_zeros_of_either_sign_get_their_own_executable() -> None:
    x = lucid.tensor([-0.0, 1.0]).to(COMPILE_DEVICE)
    _C_engine.compile.session_cache_clear()
    pos = lucid.compile(lambda t: t + 0.0)(x)
    neg = lucid.compile(lambda t: t + (-0.0))(x)
    assert _C_engine.compile.session_cache_size() == 2
    _assert_same_bits(x + 0.0, pos, "t + 0.0")
    _assert_same_bits(x + (-0.0), neg, "t + (-0.0) after t + 0.0")


def test_a_nan_constant_hits_the_cache() -> None:
    """A NaN never equalled itself under ``==``, so it never hit."""
    x = lucid.tensor([1.0, 2.0]).to(COMPILE_DEVICE)
    _C_engine.compile.session_cache_clear()
    lucid.compile(lambda t: t + math.nan)(x)
    lucid.compile(lambda t: t + math.nan)(x)
    assert _C_engine.compile.session_cache_size() == 1


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b = nn.Linear(8, 2)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        # The ``+ 0`` puts a rewritten add on every parameter's path.
        return self.b(F.relu(self.a(x)) + 0)


def _sum_of_losses(out: lucid.Tensor, target: lucid.Tensor) -> lucid.Tensor:
    # Builtin ``sum`` starts at 0: the loss itself is ``0 + mse``.
    return sum([F.mse_loss(out, target)])


@pytest.mark.parametrize("manual_vjp", ["1", "0"], ids=["manual-vjp", "autodiff"])
def test_fused_step_with_zero_adds_matches_eager(
    manual_vjp: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``fused_step`` has no eager fallback: a step that returns compiled."""
    monkeypatch.setenv("LUCID_MANUAL_VJP", manual_vjp)
    monkeypatch.setenv("LUCID_MANUAL_VJP_REQUIRE", manual_vjp)
    lucid.manual_seed(0)
    eager_model = _MLP().to(COMPILE_DEVICE)
    comp_model = _MLP().to(COMPILE_DEVICE)
    with lucid.no_grad():
        for pe, pc in zip(eager_model.parameters(), comp_model.parameters(), strict=True):
            pc.copy_(pe.detach().clone())
    x = lucid.randn(5, 4).to(COMPILE_DEVICE)
    target = lucid.randn(5, 2).to(COMPILE_DEVICE)

    opt_eager = optim.SGD(list(eager_model.parameters()), lr=0.1)
    step = fused_step(comp_model, _sum_of_losses, optim.SGD(list(comp_model.parameters()), lr=0.1))
    for _ in range(3):
        opt_eager.zero_grad()
        eager_loss = _sum_of_losses(eager_model(x), target)
        eager_loss.backward()
        opt_eager.step()
        comp_loss = step(x, target)
        assert abs(float(eager_loss.item()) - float(comp_loss.item())) < 1e-5

    for pe, pc in zip(eager_model.parameters(), comp_model.parameters(), strict=True):
        assert float((pe - pc).abs().max().item()) < 1e-5


class _Scaled(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(lucid.tensor([100.0, 0.5]))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.w * x + 0


def _exp_sum(out: lucid.Tensor) -> lucid.Tensor:
    return out.exp().sum()


@pytest.mark.parametrize("manual_vjp", ["1", "0"], ids=["manual-vjp", "autodiff"])
def test_infinite_gradient_passes_the_zero_add(
    manual_vjp: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``exp(1000)`` overflows, so ``w[0]``'s gradient is +inf and SGD sends it to -inf.

    The ``+ 0`` adds a term whose derivative is 0; its gradient must not
    multiply that inf by 0 into NaN.
    """
    monkeypatch.setenv("LUCID_MANUAL_VJP", manual_vjp)
    monkeypatch.setenv("LUCID_MANUAL_VJP_REQUIRE", manual_vjp)
    x = lucid.tensor([10.0, 1.0]).to(COMPILE_DEVICE)
    eager_model = _Scaled().to(COMPILE_DEVICE)
    opt = optim.SGD(list(eager_model.parameters()), lr=1.0)
    opt.zero_grad()
    _exp_sum(eager_model(x)).backward()
    opt.step()

    comp_model = _Scaled().to(COMPILE_DEVICE)
    fused_step(comp_model, _exp_sum, optim.SGD(list(comp_model.parameters()), lr=1.0))(x)

    want = eager_model.w.numpy()
    got = comp_model.w.numpy()
    assert got[0] == want[0] == -math.inf, f"eager {want}, compiled {got}"
    assert abs(float(got[1]) - float(want[1])) < 1e-5
