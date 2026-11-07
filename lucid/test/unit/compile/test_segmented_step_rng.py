"""A segmented step's backward differentiates the draw its forward made (LCD-896).

Each segment's backward executable recomputes the segment's forward, and a
compiled run draws its random values anew: the backward drew a second mask,
so the gradient belonged to a network the loss never measured, and the
stream ended two draws ahead of eager's.  The recompute now runs from the
position its forward started at (:mod:`lucid.compile._core.rng_replay`).

Every random op a forward can hold is a row of one table, run through a
segmented step and an eager twin over several steps: the loss, every
gradient, and the *next* draw from the stream must be eager's.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap, _wrap
from lucid.compile import make_step

from lucid.test.unit.compile._helpers import COMPILE_DEVICE, metal_tensor


def _metal_ok() -> bool:
    try:
        lucid.zeros(1).to(COMPILE_DEVICE)
    except Exception:  # noqa: BLE001 — any failure means no Metal here
        return False
    return True


pytestmark = pytest.mark.skipif(not _metal_ok(), reason="Metal unavailable")


class _Lambda(nn.Module):
    """A parameter-free stage built from one function."""

    def __init__(self, fn) -> None:  # type: ignore[no-untyped-def]  # a test stage
        super().__init__()
        self._fn = fn

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self._fn(x)


def _drop_path(x: lucid.Tensor) -> lucid.Tensor:
    return _wrap(_C_engine.nn.drop_path(_unwrap(x), 0.5, True))


# Tensor ops that draw from the default stream, applied to a (batch, 8) stage.
_SAMPLERS = {
    "randn_like": lambda x: x + lucid.randn_like(x) * 0.5,
    "rand_like": lambda x: x * lucid.rand_like(x),
    "rand": lambda x: x * lucid.rand(*x.shape, device=x.device),
    "normal": lambda x: x + lucid.normal(0.0, 0.5, size=x.shape, device=x.device),
    "bernoulli": lambda x: x * lucid.bernoulli(x.detach(), 0.7),
    "gumbel_softmax": lambda x: F.gumbel_softmax(x, tau=0.7),
    "rrelu": lambda x: F.rrelu(x, training=True),
}


def _stages(name: str) -> list[nn.Module]:
    """The stages around one random op, which sees a ``(batch, 8)`` activation
    (reshaped to a channel layout for the channel-wise ones)."""
    channel = {
        "dropout1d": (nn.Unflatten(1, (4, 2)), nn.Dropout1d(0.5)),
        "dropout2d": (nn.Unflatten(1, (2, 2, 2)), nn.Dropout2d(0.5)),
        "dropout3d": (nn.Unflatten(1, (2, 2, 2, 1)), nn.Dropout3d(0.5)),
        "feature_alpha_dropout": (nn.Unflatten(1, (4, 2)), nn.FeatureAlphaDropout(0.5)),
    }
    if name in channel:
        reshape, drop = channel[name]
        return [reshape, drop, nn.Flatten(1)]
    if name == "dropout":
        return [nn.Dropout(0.4)]
    if name == "alpha_dropout":
        return [nn.AlphaDropout(0.4)]
    if name == "drop_path":
        return [_Lambda(_drop_path)]
    return [_Lambda(_SAMPLERS[name])]


OPS = [
    "dropout",
    "dropout1d",
    "dropout2d",
    "dropout3d",
    "alpha_dropout",
    "feature_alpha_dropout",
    "drop_path",
    *_SAMPLERS,
]


def _model(name: str) -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(
        nn.Linear(6, 8),
        nn.ReLU(),
        nn.Linear(8, 8),
        *_stages(name),
        nn.Linear(8, 8),
        *_stages(name),
        nn.Linear(8, 3),
    ).to(COMPILE_DEVICE)


def _loss(y: lucid.Tensor) -> lucid.Tensor:
    return (y * y).mean()


def _grads(model: nn.Module) -> list[np.ndarray]:
    return [p.grad.numpy().copy() for p in model.parameters()]


def _run(call, model: nn.Module, x: lucid.Tensor, seed: int):  # type: ignore[no-untyped-def]
    """One step under ``seed``: (loss, gradients, the next three draws)."""
    for p in model.parameters():
        p.grad = None
    lucid.manual_seed(seed)
    loss = call(x)
    loss.backward()
    return float(loss.item()), _grads(model), lucid.rand(3).numpy()


@pytest.mark.parametrize("segments", [2, 3])
@pytest.mark.parametrize("name", OPS)
def test_segmented_step_matches_an_eager_twin_and_its_stream(name: str, segments: int) -> None:
    compiled, twin = _model(name), _model(name)
    step = make_step(compiled, _loss, segments=segments)
    lucid.manual_seed(1)
    batches = [metal_tensor(6, 6) * (1 + i) for i in range(3)]
    for i, x in enumerate(batches):
        got = _run(step, compiled, x, 70 + i)
        want = _run(lambda b: _loss(twin(b)), twin, x, 70 + i)
        assert got[0] == pytest.approx(want[0], rel=1e-4), f"{name} step {i}: loss"
        for g, w in zip(got[1], want[1], strict=True):
            np.testing.assert_allclose(g, w, rtol=1e-3, atol=1e-5, err_msg=f"{name} step {i}")
        np.testing.assert_array_equal(got[2], want[2], err_msg=f"{name} step {i}: next draw")


@pytest.mark.parametrize("segments", [2, 3])
def test_a_forward_that_draws_stays_compiled_and_one_without_stays_plain(segments: int) -> None:
    drawing = make_step(_model("dropout"), _loss, segments=segments)
    plain_model = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 8), nn.Linear(8, 3))
    plain = make_step(plain_model.to(COMPILE_DEVICE), _loss, segments=segments)
    x = metal_tensor(6, 6)
    drawing(x).backward()
    plain(x).backward()
    assert not drawing.eager_only["flag"] and not plain.eager_only["flag"]
    flags = [c.draws for cache in drawing.cache for c in cache.values()]
    assert flags and any(flags)
    assert not any(c.draws for cache in plain.cache for c in cache.values())


def test_a_backward_that_would_draw_otherwise_runs_eagerly() -> None:
    """A forward that draws a different amount the second time it is traced
    cannot have its draws replayed: the step refuses, and is eager's."""
    calls = [0]

    def uneven(x: lucid.Tensor) -> lucid.Tensor:
        calls[0] += 1
        if calls[0] == 2:
            lucid.rand(4)  # the backward's trace only
        return x * lucid.rand_like(x)

    def build() -> nn.Sequential:
        lucid.manual_seed(0)
        return nn.Sequential(nn.Linear(6, 8), _Lambda(uneven), nn.Linear(8, 3)).to(COMPILE_DEVICE)

    compiled, twin = build(), build()
    step = make_step(compiled, _loss, segments=2)
    x = metal_tensor(6, 6)
    with pytest.warns(UserWarning, match="draws a different number"):
        got = _run(step, compiled, x, 9)
    calls[0] = 5  # the twin draws exactly what the eager rerun did
    want = _run(lambda b: _loss(twin(b)), twin, x, 9)
    assert step.eager_only["flag"]
    assert got[0] == pytest.approx(want[0], rel=1e-5)
    for g, w in zip(got[1], want[1], strict=True):
        np.testing.assert_allclose(g, w, rtol=1e-4, atol=1e-6)
    np.testing.assert_array_equal(got[2], want[2])


def test_the_backward_may_run_twice_and_draws_the_same_mask() -> None:
    """``retain_graph`` runs one segment's backward again: eager reuses its mask."""
    model = nn.Sequential(nn.Linear(6, 8), nn.Dropout(0.5), nn.Linear(8, 3)).to(COMPILE_DEVICE)
    step = make_step(model, _loss, segments=2)
    x = metal_tensor(6, 6)
    lucid.manual_seed(3)
    loss = step(x)
    loss.backward(retain_graph=True)
    once = _grads(model)
    for p in model.parameters():
        p.grad = None
    loss.backward()
    for a, b in zip(once, _grads(model), strict=True):
        np.testing.assert_array_equal(a, b)


def test_replayed_draws_again_and_leaves_the_stream_where_it_stood() -> None:
    from lucid._factories._generator import _stream_states
    from lucid.compile._core.rng_replay import advance_since, replayed

    lucid.manual_seed(11)
    start = _stream_states()
    first = lucid.rand(4).numpy()
    assert advance_since(start)
    after = _stream_states()
    with replayed(start):
        np.testing.assert_array_equal(lucid.rand(4).numpy(), first)
    assert advance_since(after) == {}  # the replay's draw was not the caller's
    with pytest.raises(ZeroDivisionError), replayed(start):
        lucid.rand(4)
        1 / 0
    assert advance_since(after) == {}
    with replayed(None):
        lucid.rand(1)
    assert advance_since(after)
