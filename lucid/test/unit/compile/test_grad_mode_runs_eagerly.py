"""A compiled call that needs gradients answers like eager (LCD-582).

The executable is a forward pass with no autograd graph.  On Metal,
``lucid.compile(model)(x)`` returned a result with ``requires_grad=False``
for a model with trainable parameters, without a word, so a plain training
loop failed at ``loss.backward()`` ("does not require grad") — or trained
nothing if it caught that.  The reference keeps autograd: under grad mode a
call that reads a tensor requiring grad gives a result that requires grad,
in train mode and in eval mode alike.

Such a call now runs eagerly, under a key of its own (``grad=True``), with a
one-time warning; under ``no_grad``, or when nothing it reads requires grad,
it stays compiled.  Each test holds the compiled answer to eager on the CPU
(which runs eagerly anyway) and on Metal.
"""

import warnings
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid.compile._core.fallback import _NOTICED, CompileFallbackWarning
from lucid.compile._entry.module import CompiledModule
from lucid.test._fixtures.devices import metal_available

_METAL = pytest.mark.skipif(not metal_available(), reason="metal unavailable")
_DEVICES = ["cpu", pytest.param("metal", marks=_METAL)]


def _data(shape: tuple[int, ...], device: str, seed: int = 0) -> lucid.Tensor:
    rng = np.random.default_rng(seed)
    return lucid.tensor(rng.standard_normal(shape).astype(np.float32)).to(device)


class _ConvNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.head = nn.Linear(4 * 6 * 6, 5)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.head(self.conv(x).relu().flatten(1))


def _mlp(device: str) -> nn.Module:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3)).to(device)


def _conv(device: str) -> nn.Module:
    lucid.manual_seed(0)
    return _ConvNet().to(device)


def _bn_train(device: str) -> nn.Module:
    lucid.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(4, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Linear(8, 3)
    )
    return model.to(device).train()


_MODELS: dict[str, tuple[Callable[[str], nn.Module], tuple[int, ...]]] = {
    "mlp": (_mlp, (6, 4)),
    "conv": (_conv, (2, 3, 6, 6)),
    "bn_train": (_bn_train, (6, 4)),
}


def _grads(params: list[nn.Parameter]) -> list[np.ndarray]:
    out = []
    for p in params:
        assert p.grad is not None, "backward did not reach a parameter"
        out.append(p.grad.numpy().copy())
        p.grad = None
    return out


def _assert_same(got: list[np.ndarray], want: list[np.ndarray]) -> None:
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_array_equal(g, w)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("kind", sorted(_MODELS))
def test_backward_reaches_every_parameter(kind: str, device: str) -> None:
    build, shape = _MODELS[kind]
    model = build(device)
    params = list(model.parameters())
    x = _data(shape, device)

    model(x).sum().backward()
    eager = _grads(params)

    compiled = lucid.compile(model)
    for _ in range(3):  # the first call and the cached path alike
        out = compiled(x)
        assert out.requires_grad and out.grad_fn is not None
        out.sum().backward()
        _assert_same(_grads(params), eager)


@pytest.mark.parametrize("device", _DEVICES)
def test_compiled_callable_closing_over_a_module(device: str) -> None:
    model = _mlp(device)
    params = list(model.parameters())
    x = _data((6, 4), device)

    def function(t: lucid.Tensor) -> lucid.Tensor:
        return model(t) * 2.0

    function(x).sum().backward()
    eager = _grads(params)

    compiled = lucid.compile(function)
    for _ in range(3):
        out = compiled(x)
        assert out.requires_grad
        out.sum().backward()
        _assert_same(_grads(params), eager)


@pytest.mark.parametrize("device", _DEVICES)
def test_closure_tensor_requiring_grad(device: str) -> None:
    """A tensor no module holds is found once the trace has read it."""
    weight = _data((4, 3), device).requires_grad_(True)
    x = _data((6, 4), device, seed=1)

    def function(t: lucid.Tensor) -> lucid.Tensor:
        return (t @ weight.exp()).tanh()

    function(x).sum().backward()
    assert weight.grad is not None
    eager = weight.grad.numpy().copy()
    weight.grad = None

    compiled = lucid.compile(function)
    for _ in range(3):
        out = compiled(x)
        assert out.requires_grad
        out.sum().backward()
        assert weight.grad is not None
        np.testing.assert_array_equal(weight.grad.numpy(), eager)
        weight.grad = None


@pytest.mark.parametrize("device", _DEVICES)
def test_gradient_reaches_an_input(device: str) -> None:
    """Frozen parameters and an input that requires grad: grad mode decides."""
    model = _mlp(device).requires_grad_(False)
    x = _data((6, 4), device).requires_grad_(True)

    model(x).sum().backward()
    assert x.grad is not None
    eager = x.grad.numpy().copy()
    x.grad = None

    compiled = lucid.compile(model)
    out = compiled(x)
    assert out.requires_grad
    out.sum().backward()
    assert x.grad is not None
    np.testing.assert_array_equal(x.grad.numpy(), eager)


@pytest.mark.parametrize("device", _DEVICES)
def test_eval_mode_still_differentiable(device: str) -> None:
    """Eval mode changes BatchNorm and dropout, not autograd."""
    model = _bn_train(device).eval()
    params = list(model.parameters())
    x = _data((6, 4), device)

    model(x).sum().backward()
    eager = _grads(params)

    out = lucid.compile(model)(x)
    assert out.requires_grad
    out.sum().backward()
    _assert_same(_grads(params), eager)


@_METAL
def test_no_grad_stays_compiled_and_quiet() -> None:
    model = _mlp("metal")
    x = _data((6, 4), "metal")
    compiled = lucid.compile(model)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        with lucid.no_grad():
            out = compiled(x)
            again = compiled(x)
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info
    assert not out.requires_grad
    # MPSGraph and the eager kernels order their sums differently.
    np.testing.assert_allclose(again.numpy(), model(x).detach().numpy(), atol=1e-6)


@_METAL
def test_nothing_requires_grad_stays_compiled() -> None:
    """Grad mode alone is not enough: the reference's frozen model is not differentiable."""
    model = _mlp("metal").requires_grad_(False)
    compiled = lucid.compile(model)
    out = compiled(_data((6, 4), "metal"))
    assert not out.requires_grad
    info = compiled.cache_info()
    assert info["entries"] == 1 and not info["eager_only"], info


@_METAL
def test_unfreezing_after_the_trace_is_seen() -> None:
    """``requires_grad`` moves no version: the cached call reads it every time."""
    model = _mlp("metal").requires_grad_(False)
    compiled = lucid.compile(model)
    x = _data((6, 4), "metal")
    assert not compiled(x).requires_grad

    model.requires_grad_(True)
    out = compiled(x)
    assert out.requires_grad
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


@_METAL
def test_alternating_grad_and_no_grad_traces_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces = 0
    trace = CompiledModule._trace_forward

    def counting(self: CompiledModule, *args: object) -> object:
        nonlocal traces
        traces += 1
        return trace(self, *args)

    monkeypatch.setattr(CompiledModule, "_trace_forward", counting)
    model = _mlp("metal")
    params = list(model.parameters())
    x = _data((6, 4), "metal")
    model(x).sum().backward()
    eager = _grads(params)

    compiled = lucid.compile(model)
    for _ in range(4):
        with lucid.no_grad():
            assert not compiled(x).requires_grad
        compiled(x).sum().backward()
        _assert_same(_grads(params), eager)

    assert traces == 1
    info = compiled.cache_info()
    assert info["entries"] == 1
    (grad_key,) = info["eager_only"]
    assert grad_key.grad
    assert sorted(info["n_calls"].values()) == [4, 4]


@_METAL
@pytest.mark.parametrize("first_call", ["no_grad", "grad"])
def test_a_first_run_mpsgraph_refuses_answers_eagerly(
    first_call: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MPSGraph may refuse a kernel only when it first runs: eager, remembered.

    A call that needed gradients found only by its trace (a closure tensor)
    caches the executable unrun; the first ``no_grad`` call then runs it first.
    """
    weight = _data((4, 3), "metal").requires_grad_(True)

    def function(t: lucid.Tensor) -> lucid.Tensor:
        return (t @ weight).tanh()

    runs = 0

    def refusing(exe: object, feeds: list[object]) -> list[object]:
        nonlocal runs
        runs += 1
        raise RuntimeError("run_executable: MPSGraph failed: simulated first run")

    monkeypatch.setattr(_C_engine.compile, "run_executable", refusing)
    monkeypatch.setattr("lucid.compile._core.fallback._NOTICED", set())
    compiled = lucid.compile(function)
    x = _data((6, 4), "metal", seed=1)
    if first_call == "grad":
        assert compiled(x).requires_grad
        assert runs == 0 and compiled.cache_info()["entries"] == 1
    with lucid.no_grad():
        with pytest.warns(CompileFallbackWarning, match="MPSGraph failed"):
            got = compiled(x)
        again = compiled(x)
        want = function(x)
    assert runs == 1
    assert compiled.cache_info()["entries"] == 0
    np.testing.assert_array_equal(got.numpy(), want.numpy())
    np.testing.assert_array_equal(again.numpy(), want.numpy())


def test_warning_names_the_way_out_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LUCID_COMPILE_VERBOSE", raising=False)
    monkeypatch.setattr(
        "lucid.compile._core.fallback._NOTICED", set(_NOTICED) - {"autograd"}
    )
    device = "metal" if metal_available() else "cpu"
    first, second = lucid.compile(_mlp(device)), lucid.compile(_conv(device))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            first(_data((6, 4), device))
            second(_data((2, 3, 6, 6), device))
    notices = [
        str(w.message)
        for w in caught
        if issubclass(w.category, CompileFallbackWarning)
        and "needs gradients" in str(w.message)
    ]
    assert len(notices) == 1, notices
    assert "lucid.no_grad()" in notices[0] and "make_step" in notices[0]


def test_reference_keeps_autograd_the_same_way(ref: ModuleType) -> None:
    """Whether the result requires grad, case by case, as the reference decides."""
    ref_model = ref.nn.Sequential(ref.nn.Linear(4, 3))
    ref_compiled = ref.compile(ref_model, backend="aot_eager")
    model = nn.Sequential(nn.Linear(4, 3))
    compiled = lucid.compile(model)

    def cases(lib: ModuleType, m: Any, call: Callable[[Any], Any]) -> list[bool]:
        x = lib.randn(2, 4)
        out = [call(x).requires_grad]
        m.eval()
        out.append(call(x).requires_grad)
        with lib.no_grad():
            out.append(call(x).requires_grad)
        m.requires_grad_(False)
        out.append(call(x).requires_grad)
        out.append(call(lib.randn(2, 4).requires_grad_(True)).requires_grad)
        return out

    want = cases(ref, ref_model, ref_compiled)
    assert want == [True, True, False, False, True]
    assert cases(lucid, model, compiled) == want
