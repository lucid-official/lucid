"""Every draw that names no generator reads the one stream ``manual_seed`` sets.

``manual_seed`` used to install a Python generator that ``rand`` / ``randn``
read, while everything that handed the engine ``None`` — ``rand_like``,
``randn_like``, ``nn.init``, the dropout masks — read the engine's own,
seeded alike.  The two streams then repeated each other: ``randn`` followed
by ``randn_like`` returned the same numbers, a dropout mask kept exactly
where the ``rand(n)`` drawn before it was below ``1 - p``, and
``get_rng_state`` / ``set_rng_state`` (and the checkpoint recompute built
on them) saw only one of the two.  These hold every spelling of a default
draw to one stream.
"""

from collections.abc import Callable

import numpy as np
import pytest

import lucid
import lucid.distributions as D
import lucid.nn.functional as F
from lucid.autograd.checkpoint import checkpoint
from lucid.nn import init

Draw = Callable[[str, lucid.Generator | None], lucid.Tensor]


def _kw(gen: lucid.Generator | None) -> dict[str, lucid.Generator]:
    # Leaving the keyword out is how a caller names no generator.
    return {} if gen is None else {"generator": gen}


# Each takes a device and an explicit generator, or ``None`` for the default.
_WITH_GENERATOR: dict[str, Draw] = {
    "rand": lambda d, g: lucid.rand(5, device=d, **_kw(g)),
    "randn": lambda d, g: lucid.randn(5, device=d, **_kw(g)),
    "randint": lambda d, g: lucid.randint(0, 1 << 20, (5,), device=d, **_kw(g)),
    "normal": lambda d, g: lucid.normal(2.0, 3.0, size=(5,), device=d, **_kw(g)),
    "bernoulli": lambda d, g: lucid.bernoulli(
        lucid.full((32,), 0.5, device=d), **_kw(g)
    ),
    "bernoulli_p": lambda d, g: lucid.bernoulli(
        lucid.zeros(32, device=d), 0.5, **_kw(g)
    ),
    "Tensor.bernoulli_": lambda d, g: lucid.zeros(32, device=d).bernoulli_(
        0.5, **_kw(g)
    ),
    "randperm": lambda d, g: lucid.randperm(16, device=d, **_kw(g)),
    "rand_like": lambda d, g: lucid.rand_like(lucid.zeros(5, device=d), **_kw(g)),
    "randn_like": lambda d, g: lucid.randn_like(lucid.zeros(5, device=d), **_kw(g)),
    "poisson": lambda d, g: lucid.poisson(lucid.full((8,), 4.0, device=d), **_kw(g)),
    "init.uniform_": lambda d, g: init.uniform_(
        lucid.empty(5, device=d), -1.0, 1.0, **_kw(g)
    ),
    "init.normal_": lambda d, g: init.normal_(lucid.empty(5, device=d), **_kw(g)),
    "init.trunc_normal_": lambda d, g: init.trunc_normal_(
        lucid.empty(5, device=d), **_kw(g)
    ),
    "init.xavier_uniform_": lambda d, g: init.xavier_uniform_(
        lucid.empty(4, 5, device=d), **_kw(g)
    ),
    "init.xavier_normal_": lambda d, g: init.xavier_normal_(
        lucid.empty(4, 5, device=d), **_kw(g)
    ),
    "init.kaiming_uniform_": lambda d, g: init.kaiming_uniform_(
        lucid.empty(4, 5, device=d), **_kw(g)
    ),
    "init.kaiming_normal_": lambda d, g: init.kaiming_normal_(
        lucid.empty(4, 5, device=d), **_kw(g)
    ),
    "init.orthogonal_": lambda d, g: init.orthogonal_(
        lucid.empty(4, 5, device=d), **_kw(g)
    ),
    "init.sparse_": lambda d, g: init.sparse_(
        lucid.empty(6, 5, device=d), 0.5, **_kw(g)
    ),
}

# These take no generator, as in the reference: the default stream is the
# only one they can read.
_DEFAULT_ONLY: dict[str, Callable[[str], lucid.Tensor]] = {
    "F.dropout": lambda d: F.dropout(lucid.ones(64, device=d), 0.5, training=True),
    "F.dropout2d": lambda d: F.dropout2d(
        lucid.ones(4, 8, 2, 2, device=d), 0.5, training=True
    ),
    "F.alpha_dropout": lambda d: F.alpha_dropout(
        lucid.ones(64, device=d), 0.5, training=True
    ),
    "F.feature_alpha_dropout": lambda d: F.feature_alpha_dropout(
        lucid.ones(4, 8, 2, 2, device=d), 0.5, training=True
    ),
    "F.rrelu": lambda d: F.rrelu(-lucid.ones(64, device=d), training=True),
    "F.gumbel_softmax": lambda d: F.gumbel_softmax(lucid.zeros(4, 8, device=d)),
    "Normal.sample": lambda d: D.Normal(
        lucid.zeros(5, device=d), lucid.ones(5, device=d)
    ).sample(),
}

_EVERY_DRAW: dict[str, Callable[[str], lucid.Tensor]] = {
    **{name: (lambda d, f=f: f(d, None)) for name, f in _WITH_GENERATOR.items()},
    **_DEFAULT_ONLY,
}


def _np(t: lucid.Tensor) -> np.ndarray:
    return t.detach().numpy()


@pytest.mark.parametrize("name", sorted(_EVERY_DRAW))
def test_a_default_draw_advances_the_seeded_stream(name: str, device: str) -> None:
    lucid.manual_seed(5)
    before = lucid.get_rng_state().tolist()
    _EVERY_DRAW[name](device)
    assert (
        lucid.get_rng_state().tolist() != before
    ), f"{name} drew from a stream get_rng_state does not report"


@pytest.mark.parametrize("name", sorted(_EVERY_DRAW))
def test_the_saved_state_replays_a_default_draw(name: str, device: str) -> None:
    draw = _EVERY_DRAW[name]
    lucid.manual_seed(5)
    lucid.rand(7)  # the stream is not at its start
    state = lucid.get_rng_state()
    first = _np(draw(device))
    second = _np(draw(device))
    lucid.set_rng_state(state)
    replayed = _np(draw(device))
    np.testing.assert_array_equal(replayed, first)
    assert not np.array_equal(first, second), f"{name} repeated itself"


def test_interleaved_draws_are_one_stream(device: str) -> None:
    # Every spelling, one after another from one seed, must draw what the
    # same calls draw from one explicit generator given that seed.
    lucid.manual_seed(5)
    default = [_np(f(device, None)) for f in _WITH_GENERATOR.values()]
    g = lucid.Generator(5)
    explicit = [_np(f(device, g)) for f in _WITH_GENERATOR.values()]
    for name, a, b in zip(_WITH_GENERATOR, default, explicit, strict=True):
        np.testing.assert_array_equal(a, b, err_msg=name)
    assert lucid.get_rng_state().tolist() == g.get_state().tolist()


def test_a_checkpoint_recompute_draws_the_same_dropout_mask(device: str) -> None:
    # The recompute restores the state ``get_rng_state`` saved: it must be
    # the state dropout drew from, even after a factory draw moved it.
    lucid.manual_seed(0)
    lucid.randn(37, device=device)
    x = lucid.ones(256, device=device, requires_grad=True)
    y = checkpoint(lambda t: F.dropout(t, 0.5, training=True), x)
    y.sum().backward()
    assert x.grad is not None
    np.testing.assert_array_equal(_np(x.grad) > 0, _np(y) > 0)
