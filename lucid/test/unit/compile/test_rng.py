"""RNG behaviour in compile mode.

A random draw inside a compiled function is drawn again on every call.
The RNG ops used to lower into the graph with the trace-time seed baked
into the descriptor, so every call of the executable drew the same values
— and a diffusion step compiled with ``make_step`` trained on one noise
sample and one timestep forever, with nothing failing.

Now a draw from the default generator becomes a feed that the run path
makes again through the same eager op, in trace order
(``lucid/_C/compile/RngFeeds.h``).  Under the same seed a compiled call
draws exactly what eager draws and leaves the generator where eager
would; the draw is taken from whichever generator ``lucid.manual_seed``
installed last, not the one the trace saw.  A draw from a caller's own
generator runs eager — the executable could not advance it.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn

from lucid.test.unit.compile._helpers import COMPILE_DEVICE


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")


class _RandModel(nn.Module):
    """Wraps an RNG op into an nn.Module so it can be ``lucid.compile``-d."""

    def __init__(self, mode: str = "randn") -> None:
        super().__init__()
        self._mode = mode

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        if self._mode == "randn":
            r = lucid.randn(*x.shape, device=x.device)
        elif self._mode == "rand":
            r = lucid.rand(*x.shape, device=x.device)
        else:
            raise ValueError(self._mode)
        return x + r


def test_compile_randn_draws_afresh_each_call() -> None:
    lucid.manual_seed(42)
    model = _RandModel("randn").to(COMPILE_DEVICE)
    cm = lucid.compile(model)

    x = lucid.zeros(4, 8).to(COMPILE_DEVICE)
    first = cm(x).numpy()
    second = cm(x).numpy()
    third = cm(x).numpy()
    assert not cm.cache_info()["eager_only"]
    assert not np.array_equal(first, second)
    assert not np.array_equal(second, third)


def test_compile_rand_uniform_within_bounds() -> None:
    """rand draws U(0, 1) — values must lie in ``[0, 1)`` after compile."""
    model = _RandModel("rand").to(COMPILE_DEVICE)
    cm = lucid.compile(model)

    x = lucid.zeros(64, 64).to(COMPILE_DEVICE)
    r = cm(x).detach()  # adds zeros so values == raw uniform draws
    lo = float(r.min().item())
    hi = float(r.max().item())
    assert 0.0 <= lo < hi <= 1.0, f"uniform out of [0, 1): [{lo}, {hi}]"
    # A 4096-sample draw should have non-trivial spread.
    assert hi - lo > 0.5, f"uniform draw is suspiciously narrow: spread={hi-lo}"


def test_compile_randn_normal_stats() -> None:
    """randn draws N(0, 1) — sample mean ≈ 0, std ≈ 1 within fp32 noise.

    Large sample size (16K) so the central-limit bound shrinks below
    the tolerance — this guards against silent dtype downcasts or
    range clipping in the emitter.
    """
    model = _RandModel("randn").to(COMPILE_DEVICE)
    cm = lucid.compile(model)
    x = lucid.zeros(128, 128).to(COMPILE_DEVICE)
    r = cm(x).detach()
    mean = float(r.mean().item())
    std = float(((r - mean) ** 2).mean().sqrt().item())
    assert abs(mean) < 0.05, f"randn mean = {mean}"
    assert abs(std - 1.0) < 0.05, f"randn std = {std}"


class _Draws(nn.Module):
    """Every traced RNG kind, each feeding a parameterised computation."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
        dev = x.device
        noise = lucid.randn(*x.shape, device=dev)
        t = lucid.randint(0, 1000, (x.shape[0], 1), device=dev)
        u = lucid.rand(x.shape[0], 1, device=dev)
        g = lucid.normal(2.0, 0.5, size=(1, x.shape[1]), device=dev)
        keep = lucid.bernoulli(u, 0.7)
        h = self.lin(x + noise) * (t.float() / 1000.0 + u) + g
        return h * keep, noise


def _model() -> _Draws:
    lucid.manual_seed(0)
    return _Draws().to(COMPILE_DEVICE).eval()


def _x() -> lucid.Tensor:
    lucid.manual_seed(1)
    return lucid.randn(4, 8).to(COMPILE_DEVICE)


@pytest.mark.usefixtures("inference")
def test_every_call_draws_afresh() -> None:
    model, x = _model(), _x()
    compiled = lucid.compile(model)
    outs = [compiled(x)[0].numpy() for _ in range(3)]
    assert not compiled.cache_info()["eager_only"]
    assert not np.array_equal(outs[1], outs[2])
    assert not np.array_equal(outs[0], outs[1])


@pytest.mark.usefixtures("inference")
def test_compiled_call_is_bit_identical_to_eager() -> None:
    model, x = _model(), _x()
    compiled = lucid.compile(model)
    for seed in (5, 6):
        lucid.manual_seed(seed)
        got_h, got_noise = (t.numpy() for t in compiled(x))
        after_compiled = lucid.rand(3).numpy()
        lucid.manual_seed(seed)
        want_h, want_noise = (t.numpy() for t in model(x))
        after_eager = lucid.rand(3).numpy()
        # The draws are eager's own, bit for bit; the returned one is the
        # fresh draw, not the trace's.  The arithmetic on them is MPSGraph's.
        assert np.array_equal(got_noise, want_noise)
        assert np.allclose(got_h, want_h, rtol=1e-5, atol=1e-6)
        # And the generator was left where eager leaves it.
        assert np.array_equal(after_compiled, after_eager)
    assert not compiled.cache_info()["eager_only"]


def test_make_step_draws_like_eager() -> None:
    model, x = _model(), _x()

    def loss_fn(out: tuple[lucid.Tensor, lucid.Tensor]) -> lucid.Tensor:
        return (out[0] * out[0]).sum()

    step = lucid.compile.make_step(model, loss_fn)
    for seed in (7, 8):
        for p in model.parameters():
            p.grad = None
        lucid.manual_seed(seed)
        got = step(x)
        got.backward()
        got_grads = [p.grad.numpy().copy() for p in model.parameters()]
        for p in model.parameters():
            p.grad = None
        lucid.manual_seed(seed)
        want = loss_fn(model(x))
        want.backward()
        want_grads = [p.grad.numpy() for p in model.parameters()]
        assert np.allclose(got.item(), want.item(), rtol=1e-5)
        for g, w in zip(got_grads, want_grads):
            assert np.allclose(g, w, rtol=1e-4, atol=1e-5)
    assert not step.eager_only


def test_own_generator_runs_eager() -> None:
    """The executable cannot advance a caller's generator — it declines."""
    gen = lucid.Generator(3)

    class _OwnGen(nn.Module):
        def forward(self, x: lucid.Tensor) -> lucid.Tensor:
            return x + lucid.randn(*x.shape, device=x.device, generator=gen)

    compiled = lucid.compile(_OwnGen())
    x = _x()
    a, b = compiled(x).numpy(), compiled(x).numpy()
    assert not np.array_equal(a, b)
    assert compiled.cache_info()["eager_only"]


class _ReshapedDraws(nn.Module):
    """RNG draws that have views taken during tracing (flow_matching pattern)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        t = lucid.rand((x.shape[0],), device=x.device)
        noise = lucid.randn(x.shape, device=x.device)
        t_view = t.reshape(-1, 1)
        xt = (1.0 - t_view) * noise + t_view * x
        return self.lin(xt) + t_view


def test_rng_draw_with_view_taken_redraws_like_eager() -> None:
    """Taking a view (e.g. reshape) of a traced RNG draw must not freeze it on replay."""
    lucid.manual_seed(0)
    model = _ReshapedDraws().to(COMPILE_DEVICE).eval()
    x = _x()

    def loss_fn(out: lucid.Tensor) -> lucid.Tensor:
        return (out * out).mean()

    step = lucid.compile.make_step(model, loss_fn)

    for seed in (7, 8):
        for p in model.parameters():
            p.grad = None
        lucid.manual_seed(seed)
        got = step(x)
        got.backward()
        got_grads = [p.grad.numpy().copy() for p in model.parameters()]

        for p in model.parameters():
            p.grad = None
        lucid.manual_seed(seed)
        want = loss_fn(model(x))
        want.backward()
        want_grads = [p.grad.numpy() for p in model.parameters()]

        assert np.allclose(got.item(), want.item(), rtol=1e-5)
        for g, w in zip(got_grads, want_grads):
            assert np.allclose(g, w, rtol=1e-4, atol=1e-5)
    assert not step.eager_only
