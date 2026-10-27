"""A step through a custom ``Function`` trains with that ``Function``'s backward.

A compiled training step traces its model under ``no_grad``, where
``Function.apply`` only runs ``forward``: the recording held ``forward``'s own
ops, and the compiled backward differentiated those instead of running the
``backward`` the user wrote.  On Metal, where the step compiles, a gradient
scaled by 5 came back unscaled, a straight-through rounding came back zero,
and ``fake_quantize`` on a QAT weight gave it a gradient of exactly 0.0 —
nothing said so, and ``eager_only`` stayed empty.

Whether a ``backward`` is the derivative of its ``forward`` cannot be told, so
such a step now runs eagerly, records its signature as eager-only and warns
once (``custom-backward``).  ``compiled_step``, which has no eager step,
refuses.  A forward-only compile differentiates nothing and still compiles.
"""

import warnings
from collections.abc import Callable

import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
import lucid.quantization as Q
from lucid.autograd import Function
from lucid.compile import fused_step, make_step
from lucid.compile._core.fallback import _NOTICED, CompileFallbackWarning
from lucid.compile._entry.function import compiled_step


class _Scaled(Function):
    @staticmethod
    def forward(ctx, h):
        del ctx
        return h * 1

    @staticmethod
    def backward(ctx, g):
        del ctx
        return g * 5


class _StraightThrough(Function):
    @staticmethod
    def forward(ctx, h):
        del ctx
        return h.round()

    @staticmethod
    def backward(ctx, g):
        del ctx
        return g


class _Reversal(Function):
    @staticmethod
    def forward(ctx, h):
        del ctx
        return h.view_as(h)

    @staticmethod
    def backward(ctx, g):
        del ctx
        return -g


class _WithMask(Function):
    """``h * 2`` and a mask marked non-differentiable, both used."""

    @staticmethod
    def forward(ctx, h):
        mask = (h > 0).to(h.dtype)
        ctx.mark_non_differentiable(mask)
        return h * 2, mask

    @staticmethod
    def backward(ctx, g, g_mask):
        del ctx, g_mask
        return g * 3


class _TwoOutputs(Function):
    """Two outputs, of which the model reads only the first."""

    @staticmethod
    def forward(ctx, h):
        del ctx
        return h * 1, h * 2

    @staticmethod
    def backward(ctx, g, g_unused):
        del ctx
        return g * 5 if g_unused is None else g * 5 + g_unused * 7


class _Block(nn.Module):
    def __init__(self, apply: Callable[[lucid.Tensor], lucid.Tensor]) -> None:
        super().__init__()
        self.apply_fn = apply

    def forward(self, h: lucid.Tensor) -> lucid.Tensor:
        return self.apply_fn(h)


class _FakeQuantizedLinear(nn.Linear):
    """A QAT-style layer: its weight is fake-quantized on every forward."""

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        w = Q.fake_quantize(self.weight, 0.05, 0, -128, 127)
        return F.linear(x, w, self.bias)


def _first(h: lucid.Tensor) -> lucid.Tensor:
    return _TwoOutputs.apply(h)[0]


def _masked(h: lucid.Tensor) -> lucid.Tensor:
    y, mask = _WithMask.apply(h)
    return y + mask


#: name → (the Function the step goes through, its first layer, the block).
_CASES: dict[str, tuple[str, type[nn.Linear], Callable[..., lucid.Tensor]]] = {
    "scaled backward": ("_Scaled", nn.Linear, _Scaled.apply),
    "straight-through estimator": ("_StraightThrough", nn.Linear, _StraightThrough.apply),
    "gradient reversal": ("_Reversal", nn.Linear, _Reversal.apply),
    "non-differentiable output": ("_WithMask", nn.Linear, _masked),
    "second output unused": ("_TwoOutputs", nn.Linear, _first),
    "fake-quantized weight": ("_FakeQuantizeAffine", _FakeQuantizedLinear, lambda h: h),
}


def _model(case: str, device: str) -> nn.Sequential:
    _, first, block = _CASES[case]
    lucid.manual_seed(0)
    return nn.Sequential(first(4, 4), _Block(block), nn.Linear(4, 3)).to(device)


def _data(device: str) -> tuple[lucid.Tensor, lucid.Tensor]:
    lucid.manual_seed(1)
    return lucid.randn(5, 4).to(device), lucid.randn(5, 3).to(device)


def _grads(model: nn.Module) -> dict[str, list | None]:
    return {n: None if p.grad is None else p.grad.tolist() for n, p in model.named_parameters()}


def _close(a: list | None, b: list | None) -> bool:
    if a is None or b is None:
        return a is b
    flat_a = lucid.tensor(a).reshape(-1)
    flat_b = lucid.tensor(b).reshape(-1)
    return bool(lucid.allclose(flat_a, flat_b, rtol=1e-4, atol=1e-5))


def _assert_same(got: dict[str, list | None], want: dict[str, list | None]) -> None:
    assert got.keys() == want.keys()
    for name in want:
        assert _close(got[name], want[name]), name


def _eager(case: str, device: str) -> tuple[dict[str, list | None], nn.Module]:
    """Eager gradients, and the model after one SGD step on them."""
    model = _model(case, device)
    x, t = _data(device)
    F.mse_loss(model(x), t).backward()
    grads = _grads(model)
    optim.SGD(model.parameters(), lr=0.5).step()
    return grads, model


def _warns_on_metal(function: str, device: str):
    """The fallback's warning, naming ``function``, where the step compiles."""
    if device != "metal":
        return warnings.catch_warnings()
    _NOTICED.discard("custom-backward")
    return pytest.warns(CompileFallbackWarning, match=function)


def _step_and_update(
    model: nn.Module, step: Callable[..., lucid.Tensor], device: str
) -> dict[str, list | None]:
    x, t = _data(device)
    step(x, t).backward()
    grads = _grads(model)
    optim.SGD(model.parameters(), lr=0.5).step()
    return grads


@pytest.mark.parametrize("case", sorted(_CASES))
def test_make_step_trains_as_eager(case: str, device: str) -> None:
    want, eager = _eager(case, device)
    model = _model(case, device)
    step = make_step(model, F.mse_loss)
    with _warns_on_metal(_CASES[case][0], device):
        got = _step_and_update(model, step, device)
    _assert_same(got, want)
    _assert_same(
        {n: p.tolist() for n, p in model.named_parameters()},
        {n: p.tolist() for n, p in eager.named_parameters()},
    )
    assert len(step.eager_only) == 1
    assert not step.cache


@pytest.mark.parametrize("case", sorted(_CASES))
def test_compiled_module_step_trains_as_eager(case: str, device: str) -> None:
    want, _ = _eager(case, device)
    model = _model(case, device)
    compiled = lucid.compile(model)
    with _warns_on_metal(_CASES[case][0], device):
        got = _step_and_update(model, lambda x, t: compiled.step(x, t, loss_fn=F.mse_loss), device)
    _assert_same(got, want)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_segmented_step_trains_as_eager(case: str, device: str) -> None:
    want, _ = _eager(case, device)
    model = _model(case, device)
    step = make_step(model, F.mse_loss, segments=3)
    with _warns_on_metal(_CASES[case][0], device):
        got = _step_and_update(model, step, device)
    _assert_same(got, want)
    assert step.eager_only["flag"]


@pytest.mark.parametrize("case", sorted(_CASES))
def test_fused_step_updates_as_eager(case: str, device_gpu_only: str) -> None:
    device = device_gpu_only
    _, eager = _eager(case, device)
    model = _model(case, device)
    step = fused_step(model, F.mse_loss, optim.SGD(model.parameters(), lr=0.5))
    x, t = _data(device)
    with _warns_on_metal(_CASES[case][0], device):
        step(x, t)
    assert step.cache_info()["eager_only"]
    _assert_same(
        {n: p.tolist() for n, p in model.named_parameters()},
        {n: p.tolist() for n, p in eager.named_parameters()},
    )


@pytest.mark.parametrize("case", sorted(_CASES))
def test_compiled_step_refuses(case: str, device_gpu_only: str) -> None:
    model = _model(case, device_gpu_only)
    x, t = _data(device_gpu_only)
    with pytest.raises(NotImplementedError, match=_CASES[case][0]):
        compiled_step(model, x, lambda y: F.mse_loss(y, t))
    assert all(p.grad is None for p in model.parameters())


def test_a_function_off_the_gradient_path_still_compiles(device_gpu_only: str) -> None:
    """Applied to the target only, no gradient the step computes crosses it."""
    model = nn.Linear(4, 3).to(device_gpu_only)
    step = make_step(model, lambda y, t: F.mse_loss(y, _Scaled.apply(t)))
    x, t = _data(device_gpu_only)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        step(x, t).backward()
    assert len(step.cache) == 1
    assert not step.eager_only


@pytest.mark.parametrize("case", sorted(_CASES))
def test_a_forward_only_compile_is_unaffected(case: str, device_gpu_only: str) -> None:
    """Nothing is differentiated, so the forward the user wrote is the answer."""
    model = _model(case, device_gpu_only).eval()
    x, _ = _data(device_gpu_only)
    compiled = lucid.compile(model)
    with lucid.no_grad():
        want = model(x)
        compiled(x)
        got = compiled(x)
    info = compiled.cache_info()
    assert info["entries"] == 1
    assert not info["eager_only"]
    assert _close(got.tolist(), want.tolist())


def test_a_quantization_aware_model_trains_as_eager(device: str) -> None:
    """``prepare_qat``'s own fake-quantization, through ``make_step``."""

    def net() -> nn.Module:
        lucid.manual_seed(0)
        plain = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
        return Q.prepare_qat(plain, Q.get_default_qat_qconfig_mapping()).to(device)

    x, t = _data(device)
    eager = net()
    F.mse_loss(eager(x), t).backward()
    model = net()
    step = make_step(model, F.mse_loss)
    with _warns_on_metal("_FakeQuantizeAffine", device):
        step(x, t).backward()
    _assert_same(_grads(model), _grads(eager))


# ── each part of the rule, alone ──────────────────────────────────────────
#
# Every case above is caught by one clause of the rule: an output computed
# from a parameter.  These three are each caught by one other clause alone.


class _IgnoresWeight(Function):
    """Returns ``x``; ``forward`` never reads ``w``, ``backward`` still answers for it."""

    @staticmethod
    def forward(ctx, w, x):
        del ctx, w
        return x * 1

    @staticmethod
    def backward(ctx, g):
        del ctx
        return g.sum(dim=0), g


class _HandsBack(Function):
    """Returns its input itself, the trace's id for it not yet made."""

    @staticmethod
    def forward(ctx, w):
        del ctx
        return w

    @staticmethod
    def backward(ctx, g):
        del ctx
        return g * 3


#: name → (the Function, the block in front of a ``Linear``).  ``w`` is a
#: ``(4,)`` parameter, ``x`` the batch.
_CLAUSES: dict[str, tuple[str, Callable[..., lucid.Tensor]]] = {
    # The trace never records ``w``: only "an input is a leaf that requires
    # grad" sees that a gradient goes to it.
    "leaf input forward ignores": ("_IgnoresWeight", lambda w, x: _IgnoresWeight.apply(w, x)),
    # ``w * 2`` comes from a parameter, the output only from the batch: only
    # the inputs' side of "comes from a source" sees it.
    "input computed from a parameter": (
        "_IgnoresWeight",
        lambda w, x: _IgnoresWeight.apply(w * 2, x),
    ),
    # ``w`` has no id until ``x * w`` consumes it: only the second look-up of
    # the outputs' ids sees that one leads to the loss.  Eager hands ``w``
    # back with the node on it (LCD-558), so only a first step is compared.
    "parameter handed back": ("_HandsBack", lambda w, x: x * _HandsBack.apply(w)),
}


class _Weighted(nn.Module):
    def __init__(self, clause: str) -> None:
        super().__init__()
        self.block = _CLAUSES[clause][1]
        self.w = nn.Parameter(lucid.randn(4))
        self.head = nn.Linear(4, 3)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.head(self.block(self.w, x))


def _weighted(clause: str, device: str) -> _Weighted:
    lucid.manual_seed(0)
    return _Weighted(clause).to(device)


@pytest.mark.parametrize("clause", sorted(_CLAUSES))
def test_each_clause_alone_sends_the_step_eager(clause: str, device: str) -> None:
    x, t = _data(device)
    eager = _weighted(clause, device)
    F.mse_loss(eager(x), t).backward()
    model = _weighted(clause, device)
    step = make_step(model, F.mse_loss)
    with _warns_on_metal(_CLAUSES[clause][0], device):
        step(x, t).backward()
    _assert_same(_grads(model), _grads(eager))
    assert len(step.eager_only) == 1


@pytest.mark.parametrize("clause", sorted(_CLAUSES))
def test_each_clause_alone_makes_compiled_step_refuse(clause: str, device_gpu_only: str) -> None:
    model = _weighted(clause, device_gpu_only)
    x, t = _data(device_gpu_only)
    if clause == "leaf input forward ignores":
        # A parameter the trace never saw stops compiled_step first.
        with pytest.raises(RuntimeError, match="not observed in the trace"):
            compiled_step(model, x, lambda y: F.mse_loss(y, t))
    else:
        with pytest.raises(NotImplementedError, match=_CLAUSES[clause][0]):
            compiled_step(model, x, lambda y: F.mse_loss(y, t))
    assert all(p.grad is None for p in model.parameters())
