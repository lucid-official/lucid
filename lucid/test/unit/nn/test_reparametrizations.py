"""Weight norm, spectral norm and the parametrization machinery.

``nn/utils/weight_norm.py`` sat at 14.9% and ``parametrize.py`` at
19.7% — both were exported and essentially never constructed.

A reparameterisation is invisible when it fails.  The module still has
parameters, still runs forward, still trains; it just is not the
reparameterisation it claims to be.  So each is checked on the identity
that defines it — ``w == g · v / ‖v‖`` for weight norm, a top singular
value of 1 for spectral norm — rather than on the presence of the
tensors it installs.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.utils as U
from lucid.nn.utils.parametrize import (
    is_parametrized,
    register_parametrization,
    remove_parametrizations,
)


def _v(x):
    return np.asarray(x.numpy())


def _x(*shape):
    return lucid.tensor(np.ones(shape, dtype=np.float32))


# ── weight norm ───────────────────────────────────────────────────────────────


def test_weight_norm_installs_the_magnitude_and_direction():
    layer = U.weight_norm(nn.Linear(6, 4))
    names = dict(layer.named_parameters())
    assert "weight_g" in names and "weight_v" in names
    assert "weight" not in names  # no longer a leaf


def test_the_weight_really_is_g_times_v_over_its_norm():
    """The identity the whole reparameterisation exists for.  A module
    that installed ``g`` and ``v`` and then ignored them would pass every
    structural check and train the wrong thing."""
    layer = U.weight_norm(nn.Linear(6, 4))
    names = dict(layer.named_parameters())
    direction = _v(names["weight_v"])
    magnitude = _v(names["weight_g"])
    want = magnitude * direction / np.linalg.norm(direction, axis=1, keepdims=True)
    assert np.allclose(_v(layer.weight), want, atol=1e-5)


def test_weight_norm_leaves_the_forward_unchanged_at_initialisation():
    """``g`` starts at ``‖w‖`` and ``v`` at ``w``, so wrapping a trained
    layer must not move it."""
    base = nn.Linear(6, 4)
    x = _x(2, 6)
    before = _v(base(x))
    assert np.allclose(before, _v(U.weight_norm(base)(x)), atol=1e-4)


def test_gradients_reach_both_the_magnitude_and_the_direction():
    layer = U.weight_norm(nn.Linear(6, 4))
    layer(_x(2, 6)).sum().backward()
    for name, param in layer.named_parameters():
        assert param.grad is not None, name


def test_a_step_on_the_magnitude_scales_the_weight_without_turning_it():
    """``g`` and ``v`` are not interchangeable: moving ``g`` has to change
    the weight's norm and leave its direction alone.

    ``weight`` is recomputed by a forward pre-hook, so it is stale between
    calls — the assertion has to come after a forward, not after the
    assignment.
    """
    layer = U.weight_norm(nn.Linear(6, 4))
    x = _x(2, 6)
    layer(x)
    before = _v(layer.weight).copy()

    layer.weight_g = nn.Parameter(dict(layer.named_parameters())["weight_g"] * 2.0)
    assert np.allclose(_v(layer.weight), before), "stale until the next forward"
    layer(x)
    after = _v(layer.weight)

    assert np.isclose(np.linalg.norm(after), 2 * np.linalg.norm(before), rtol=1e-5)
    unit_before = before / np.linalg.norm(before, axis=1, keepdims=True)
    unit_after = after / np.linalg.norm(after, axis=1, keepdims=True)
    assert np.allclose(unit_before, unit_after, atol=1e-5)


@pytest.mark.parametrize(
    "optimiser",
    [lucid.optim.SGD, lucid.optim.Adam, lucid.optim.AdamW, lucid.optim.RMSprop],
    ids=["sgd", "adam", "adamw", "rmsprop"],
)
def test_every_optimiser_actually_moves_the_magnitude(optimiser):
    """A reparameterisation whose ``g`` never moved would train like a
    plain layer and look identical from the outside."""
    layer = U.weight_norm(nn.Linear(6, 4))
    opt = optimiser(layer.parameters(), lr=0.5)
    x = _x(4, 6)
    before = _v(dict(layer.named_parameters())["weight_g"]).copy()
    for _ in range(3):
        opt.zero_grad()
        (layer(x) ** 2).mean().backward()
        opt.step()
    assert not np.allclose(before, _v(dict(layer.named_parameters())["weight_g"]))


def test_the_cached_weight_makes_g_and_v_read_only_in_place():
    """Recorded, not a defect — but a trap worth naming.

    ``weight`` is cached as an attribute derived from ``g`` and ``v``, and
    the engine refuses an in-place write to any tensor that has a live
    derived view over its storage.  So the canonical
    ``with no_grad(): p.add_(...)`` — EMA updates, warmup surgery, a
    hand-rolled optimiser — succeeds on a plain parameter and fails here.

    The message says "call .clone() first", which if followed mutates a
    copy and drops the update silently.  Optimisers are unaffected: they
    go through ``step()``, which does not take this path.
    """
    plain = nn.Linear(6, 4)
    with lucid.no_grad():
        plain.weight.add_(1.0)  # fine

    layer = U.weight_norm(nn.Linear(6, 4))
    with pytest.raises(Exception, match="shares storage"):
        with lucid.no_grad():
            layer.weight_g.add_(1.0)


@pytest.mark.parametrize("dim", [0, 1])
def test_weight_norm_honours_the_dim_it_normalises_over(dim):
    layer = U.weight_norm(nn.Linear(6, 4), dim=dim)
    assert _v(layer.weight).shape == (4, 6)
    assert np.isfinite(_v(layer(_x(2, 6)))).all()


def test_weight_norm_on_a_convolution():
    layer = U.weight_norm(nn.Conv2d(3, 8, kernel_size=3))
    out = _v(layer(_x(1, 3, 8, 8)))
    assert out.shape == (1, 8, 6, 6)
    assert np.isfinite(out).all()


def test_removing_weight_norm_keeps_the_weight_it_had():
    base = nn.Linear(6, 4)
    x = _x(2, 6)
    layer = U.weight_norm(base)
    before = _v(layer(x))
    layer = U.remove_weight_norm(layer)
    names = dict(layer.named_parameters())
    assert "weight" in names and "weight_g" not in names
    assert np.allclose(before, _v(layer(x)), atol=1e-5)


def test_a_weight_normed_layer_still_trains():
    layer = U.weight_norm(nn.Linear(6, 4))
    optimiser = lucid.optim.SGD(layer.parameters(), lr=0.1)
    x, target = _x(4, 6), _x(4, 4)
    first = float(_v(((layer(x) - target) ** 2).mean()))
    for _ in range(20):
        optimiser.zero_grad()
        ((layer(x) - target) ** 2).mean().backward()
        optimiser.step()
    assert float(_v(((layer(x) - target) ** 2).mean())) < first


# ── spectral norm ─────────────────────────────────────────────────────────────


def test_spectral_norm_drives_the_top_singular_value_to_one():
    """That is the Lipschitz bound it exists to impose; a wrapper that
    installed ``u`` and ``v`` without dividing would look identical."""
    layer = U.spectral_norm(nn.Linear(8, 8), n_power_iterations=50)
    x = _x(2, 8)
    for _ in range(20):
        layer(x)
    largest = np.linalg.svd(_v(layer.weight), compute_uv=False)[0]
    assert abs(largest - 1.0) < 0.02


def test_one_power_iteration_gets_most_of_the_way_there():
    layer = U.spectral_norm(nn.Linear(8, 8))
    layer(_x(2, 8))
    largest = np.linalg.svd(_v(layer.weight), compute_uv=False)[0]
    assert abs(largest - 1.0) < 0.35


def test_spectral_norm_stops_iterating_in_eval():
    """The power iteration is a training-time estimate; continuing it at
    inference would make the same input give different answers."""
    layer = U.spectral_norm(nn.Linear(8, 8))
    x = _x(2, 8)
    for _ in range(10):
        layer(x)
    layer.eval()
    assert np.allclose(_v(layer(x)), _v(layer(x)))


def test_spectral_norm_bounds_how_much_it_can_stretch_a_vector():
    lucid.manual_seed(0)
    layer = U.spectral_norm(nn.Linear(8, 8), n_power_iterations=50)
    for _ in range(20):
        layer(_x(2, 8))
    rng = np.random.default_rng(0)
    for _ in range(10):
        vec = rng.standard_normal((1, 8)).astype(np.float32)
        out = _v(layer(lucid.tensor(vec))) - _v(layer.bias)
        assert np.linalg.norm(out) <= np.linalg.norm(vec) * 1.05


def test_removing_spectral_norm_restores_a_plain_weight():
    layer = U.remove_spectral_norm(U.spectral_norm(nn.Linear(8, 8)))
    assert "weight" in dict(layer.named_parameters())
    assert np.isfinite(_v(layer(_x(2, 8)))).all()


# ── the general parametrization machinery ─────────────────────────────────────


class _Symmetric(nn.Module):
    def forward(self, X):
        return X.T @ X


class _Doubled(nn.Module):
    def forward(self, X):
        return X * 2.0


def test_a_parametrization_applies_on_every_read():
    layer = nn.Linear(5, 5)
    register_parametrization(layer, "weight", _Symmetric())
    assert is_parametrized(layer, "weight")
    weight = _v(layer.weight)
    assert np.allclose(weight, weight.T, atol=1e-5)


def test_the_constraint_survives_a_parameter_update():
    """The point of a parametrization rather than a one-off projection:
    the invariant has to hold after the optimiser has moved things."""
    layer = nn.Linear(5, 5)
    register_parametrization(layer, "weight", _Symmetric())
    optimiser = lucid.optim.SGD(layer.parameters(), lr=0.05)
    for _ in range(5):
        optimiser.zero_grad()
        layer(_x(2, 5)).sum().backward()
        optimiser.step()
    weight = _v(layer.weight)
    assert np.allclose(weight, weight.T, atol=1e-5)


def test_a_parametrized_module_still_receives_gradients():
    layer = nn.Linear(4, 4)
    register_parametrization(layer, "weight", _Doubled())
    layer(_x(2, 4)).sum().backward()
    assert any(p.grad is not None for _, p in layer.named_parameters())


def test_removing_a_parametrization_freezes_the_current_value():
    layer = nn.Linear(4, 4)
    register_parametrization(layer, "weight", _Doubled())
    before = _v(layer.weight)
    remove_parametrizations(layer, "weight")
    assert not is_parametrized(layer, "weight")
    assert np.allclose(before, _v(layer.weight), atol=1e-6)


def test_an_unparametrized_module_says_so():
    assert not is_parametrized(nn.Linear(4, 4), "weight")


# ── gradient clipping ─────────────────────────────────────────────────────────


def _with_gradients(scale=1.0):
    layer = nn.Linear(4, 4)
    (layer(_x(2, 4)).sum() * scale).backward()
    return layer


def test_clip_grad_norm_brings_the_total_norm_under_the_cap():
    layer = _with_gradients(scale=100.0)
    U.clip_grad_norm_(layer.parameters(), max_norm=0.1)
    total = np.sqrt(sum((_v(p.grad) ** 2).sum() for p in layer.parameters()))
    assert total <= 0.1 + 1e-5


def test_clip_grad_norm_reports_the_norm_before_clipping():
    layer = _with_gradients(scale=100.0)
    before = np.sqrt(sum((_v(p.grad) ** 2).sum() for p in layer.parameters()))
    reported = U.clip_grad_norm_(layer.parameters(), max_norm=0.1)
    assert np.isclose(float(reported.item()), before, rtol=1e-4)


def test_the_reported_norm_is_a_scalar_as_in_the_reference():
    """``clip_grad_norm_`` and ``get_total_norm`` hand back a 0-d tensor.

    They used to hand back a ``(1,)`` one, pinned here as recorded rather
    than endorsed: the number was the same and ``.item()`` worked on both,
    but code that stacks the results or checks ``norm.shape == ()`` saw
    the stray axis.  Both now match their docstrings and the reference.
    """
    layer = _with_gradients(scale=100.0)
    assert U.clip_grad_norm_(layer.parameters(), max_norm=0.1).shape == ()
    assert U.get_total_norm([p.grad for p in layer.parameters()]).shape == ()


def test_clipping_preserves_the_direction():
    """A clip that rescaled each parameter separately would change which
    way the step points, which is the one thing it must not do."""
    layer = _with_gradients(scale=100.0)
    before = np.concatenate([_v(p.grad).ravel() for p in layer.parameters()])
    U.clip_grad_norm_(layer.parameters(), max_norm=0.1)
    after = np.concatenate([_v(p.grad).ravel() for p in layer.parameters()])
    cosine = before @ after / (np.linalg.norm(before) * np.linalg.norm(after))
    assert np.isclose(cosine, 1.0, atol=1e-5)


def test_a_gradient_already_under_the_cap_is_left_alone():
    layer = _with_gradients()
    before = [_v(p.grad).copy() for p in layer.parameters()]
    U.clip_grad_norm_(layer.parameters(), max_norm=1e6)
    for old, param in zip(before, layer.parameters()):
        assert np.allclose(old, _v(param.grad))


@pytest.mark.parametrize("norm_type", [1.0, 2.0, float("inf")])
def test_clip_grad_norm_honours_the_norm_type(norm_type):
    layer = _with_gradients(scale=100.0)
    flat = np.concatenate([_v(p.grad).ravel() for p in layer.parameters()])
    reported = float(U.clip_grad_norm_(layer.parameters(), 0.1, norm_type).item())
    assert np.isclose(reported, np.linalg.norm(flat, ord=norm_type), rtol=1e-4)


def test_clip_grad_value_caps_each_entry():
    layer = _with_gradients(scale=100.0)
    U.clip_grad_value_(layer.parameters(), 0.01)
    for param in layer.parameters():
        assert np.abs(_v(param.grad)).max() <= 0.01 + 1e-6


# ── parameter vectors ─────────────────────────────────────────────────────────


def test_parameters_round_trip_through_a_flat_vector():
    source = nn.Linear(5, 3)
    target = nn.Linear(5, 3)
    U.vector_to_parameters(U.parameters_to_vector(source.parameters()), target.parameters())
    x = _x(2, 5)
    assert np.allclose(_v(source(x)), _v(target(x)), atol=1e-6)


def test_the_flat_vector_has_one_entry_per_parameter():
    layer = nn.Linear(5, 3)
    flat = _v(U.parameters_to_vector(layer.parameters()))
    assert flat.shape == (5 * 3 + 3,)


# ── chained parametrizations ──────────────────────────────────────────────────


class _Scale(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, W):
        return W * self.k


class _Symmetric(nn.Module):
    def forward(self, W):
        return 0.5 * (W + W.mT)


def _chained():
    layer = nn.Linear(4, 4, bias=False)
    base = _v(layer.weight).copy()
    register_parametrization(layer, "weight", _Symmetric())
    first = _v(layer.parametrizations["weight"]()).copy()
    register_parametrization(layer, "weight", _Scale(3.0))
    return layer, base, first


def test_a_second_parametrization_composes_onto_the_first():
    """Registering twice used to raise.  The risk in allowing it is that
    the second silently *replaces* the first, which looks fine — the shape
    is right and the values move — until the first constraint quietly
    stops holding."""
    layer, base, first = _chained()
    chained = _v(layer.parametrizations["weight"]())

    assert np.allclose(chained, 3.0 * first, atol=1e-6)
    # If Symmetric had been dropped, this would be 3 * the raw weight.
    assert not np.allclose(chained, 3.0 * base, atol=1e-4)


def test_the_earlier_constraint_still_holds_after_chaining():
    layer, _, _ = _chained()
    chained = _v(layer.parametrizations["weight"]())
    assert np.allclose(chained, chained.T, atol=1e-6)


def test_chaining_keeps_exactly_one_trainable_leaf():
    """Composing must not mint a second copy of the weight."""
    layer, _, _ = _chained()
    container = layer.parametrizations["weight"]
    assert len(container) == 2
    assert isinstance(container.original, nn.Parameter)
    assert [n for n, _ in layer.named_parameters()] == ["parametrizations.weight.original"]

    layer(_x(2, 4)).sum().backward()
    assert np.abs(_v(container.original.grad)).max() > 0


def test_chaining_rejects_a_shape_change_unless_asked():
    class _Widen(nn.Module):
        def forward(self, W):
            return lucid.cat([W, W], 1)

    layer = nn.Linear(4, 4, bias=False)
    register_parametrization(layer, "weight", _Symmetric())
    with pytest.raises(ValueError, match="shape"):
        register_parametrization(layer, "weight", _Widen())
    register_parametrization(layer, "weight", _Widen(), unsafe=True)
    assert _v(layer.parametrizations["weight"]()).shape == (4, 8)


def test_the_chain_keeps_registration_order():
    layer, _, _ = _chained()
    container = layer.parametrizations["weight"]
    assert isinstance(container[0], _Symmetric)
    assert isinstance(container[1], _Scale)
