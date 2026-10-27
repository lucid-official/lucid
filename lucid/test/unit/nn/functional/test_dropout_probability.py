"""Every dropout entry point takes ``p`` in ``[0, 1]`` and answers ``p = 1``.

The reference accepts ``p = 1`` and returns ``x * 0`` in training — zeros
that keep the autograd edge.  Lucid's ``nn.Dropout*`` constructors accepted
``p = 1`` too, then the first training forward raised from the engine
kernel (``p must be in [0, 1)``), and the functional forms let an
out-of-range ``p`` through to the same engine error instead of a
``ValueError``.  ``feature_alpha_dropout`` also zeroed dropped channels
and left kept ones unscaled, where the reference maps both through the
alpha-dropout affine.

The cases run p in {0, 0.5, 1} x {train, eval} for each function: exact
reference values where no mask is drawn, and at ``p = 0.5`` the value
structure the reference's own output has (which values a kept and a
dropped element take, and whether a mask is per element or per channel).
"""

from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

# function → (input shape, module name, mask granularity)
_FUNCTIONS: dict[str, tuple[tuple[int, ...], str, str]] = {
    "dropout": ((4, 6, 5), "Dropout", "element"),
    "dropout1d": ((4, 6, 5), "Dropout1d", "channel"),
    "dropout2d": ((4, 6, 3, 3), "Dropout2d", "channel"),
    "dropout3d": ((2, 6, 2, 3, 3), "Dropout3d", "channel"),
    "alpha_dropout": ((4, 6, 5), "AlphaDropout", "element"),
    "feature_alpha_dropout": ((4, 6, 3, 3), "FeatureAlphaDropout", "channel"),
}
_ALPHA = 1.7580993408473766


def _input(name: str) -> np.ndarray:
    shape = _FUNCTIONS[name][0]
    # Shifted away from zero so a kept element is never mistaken for a
    # dropped one.
    return np.random.default_rng(128).standard_normal(shape).astype(np.float32) + 3.0


def _kept_and_dropped(name: str, x: np.ndarray, p: float) -> tuple[np.ndarray, np.ndarray]:
    """The value each element takes if kept, and if dropped."""
    if "alpha" not in name:
        return x / (1.0 - p), np.zeros_like(x)
    a = 1.0 / np.sqrt((_ALPHA * _ALPHA * p + 1.0) * (1.0 - p))
    b = _ALPHA * a * p
    return a * x + b, np.full_like(x, -_ALPHA * a * (1.0 - p))


def _assert_mask_structure(name: str, x: np.ndarray, y: np.ndarray, p: float) -> None:
    kept_v, dropped_v = _kept_and_dropped(name, x, p)
    kept = np.isclose(y, kept_v, rtol=1e-5, atol=1e-5)
    dropped = np.isclose(y, dropped_v, rtol=1e-5, atol=1e-5)
    assert np.all(kept | dropped), f"{name}: values outside the kept/dropped pair"
    assert kept.any() and dropped.any(), f"{name}: no mask was drawn"
    if _FUNCTIONS[name][2] == "channel":
        per_channel = kept.reshape(x.shape[0], x.shape[1], -1)
        assert np.all(
            per_channel.all(-1) | ~per_channel.any(-1)
        ), f"{name}: a channel is partly dropped"


def _call(fw: ModuleType, name: str, x: object, p: float, train: bool) -> object:
    fn: Callable[..., object] = getattr(fw.nn.functional, name)
    return fn(x, p, train)


@pytest.mark.parity
@pytest.mark.parametrize("train", [True, False], ids=["train", "eval"])
@pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("name", list(_FUNCTIONS))
def test_dropout_matches_reference(
    ref: ModuleType, name: str, p: float, train: bool, device: str
) -> None:
    x = _input(name)
    got = _call(lucid, name, lucid.tensor(x, device=device), p, train).numpy()
    want = _call(ref, name, ref.tensor(x), p, train).numpy()
    assert got.shape == want.shape
    if p == 0.5 and train:
        _assert_mask_structure(name, x, want, p)
        _assert_mask_structure(name, x, got, p)
    else:
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("name", list(_FUNCTIONS))
def test_p_one_keeps_the_gradient_edge(name: str, device: str) -> None:
    x = lucid.tensor(_input(name), device=device, requires_grad=True)
    y = _call(lucid, name, x, 1.0, True)
    assert np.all(y.numpy() == 0.0)
    y.sum().backward()
    assert x.grad is not None
    assert np.all(x.grad.numpy() == 0.0)


@pytest.mark.parametrize("name", list(_FUNCTIONS))
def test_the_module_runs_at_p_one(name: str, device: str) -> None:
    module = getattr(nn, _FUNCTIONS[name][1])(1.0)
    module.train()
    out = module(lucid.tensor(_input(name), device=device))
    assert np.all(out.numpy() == 0.0)


@pytest.mark.parametrize("p", [-0.1, 1.5, float("nan")])
@pytest.mark.parametrize("name", list(_FUNCTIONS))
def test_a_probability_outside_the_unit_interval_is_refused(name: str, p: float) -> None:
    x = lucid.tensor(_input(name))
    for train in (True, False):
        with pytest.raises(ValueError, match="between 0 and 1"):
            getattr(F, name)(x, p, train)
    with pytest.raises(ValueError, match="between 0 and 1"):
        getattr(nn, _FUNCTIONS[name][1])(p)
