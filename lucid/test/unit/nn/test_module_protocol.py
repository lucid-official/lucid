"""``Module`` itself: the tree, the traversal, and the state.

``nn/module.py`` sat at 66.5%.  Every layer in the framework inherits
this, so a mistake here is not one wrong layer — it is every model that
saves a checkpoint, freezes a backbone, or accumulates gradients.

The awkward part is that almost all of it degrades quietly.  A traversal
that yields a shared submodule twice still trains; a ``state_dict`` that
skips a key still saves; a ``zero_grad`` that drops the buffer instead of
filling it still zeroes the gradient.  So the assertions here are on
identity and count rather than on whether something plausible came back.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn


def _v(x):
    return np.asarray(x.numpy())


def _x(shape=(2, 4)):
    return lucid.tensor(np.ones(shape, dtype=np.float32))


def _model():
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Sequential(nn.Linear(4, 2)))


# ── reaching into the tree ────────────────────────────────────────────────────


def test_get_submodule_follows_a_dotted_path():
    assert isinstance(_model().get_submodule("2.0"), nn.Linear)
    assert isinstance(_model().get_submodule("1"), nn.ReLU)


def test_get_submodule_refuses_a_path_that_is_not_there():
    with pytest.raises(AttributeError):
        _model().get_submodule("2.7")


def test_get_parameter_and_get_buffer_follow_the_same_paths():
    assert tuple(_model().get_parameter("2.0.weight").shape) == (2, 4)
    assert tuple(nn.BatchNorm1d(4).get_buffer("running_mean").shape) == (4,)


def test_named_modules_names_every_node_once():
    names = [name for name, _ in _model().named_modules()]
    assert len(names) == len(set(names))
    assert names[0] == ""  # the root names itself with the empty string
    assert {"0", "1", "2", "2.0"} <= set(names)


def test_the_names_and_the_objects_line_up():
    model = _model()
    for name, module in model.named_modules():
        if name:
            assert model.get_submodule(name) is module


# ── sharing ───────────────────────────────────────────────────────────────────


def test_a_shared_submodule_is_visited_once():
    """Tied weights are a real pattern — an embedding reused as the output
    projection.  Yielding it twice double-counts its parameters, and every
    consumer that sums over ``parameters()`` is then wrong."""
    shared = nn.Linear(4, 4)
    model = nn.Sequential(shared, shared)
    assert len(list(model.modules())) == 2  # the container and the layer
    assert len(list(model.parameters())) == 2  # its weight and its bias


def test_a_shared_parameter_can_be_asked_for_twice_explicitly():
    shared = nn.Linear(4, 4)
    model = nn.Sequential(shared, shared)
    assert len(list(model.named_parameters(remove_duplicate=False))) == 4


def test_a_shared_module_appears_under_both_names():
    shared = nn.Linear(4, 4)
    model = nn.Sequential(shared, shared)
    assert model.get_submodule("0") is model.get_submodule("1")


def test_recurse_false_stops_at_this_module():
    model = _model()
    assert list(model.parameters(recurse=False)) == []
    assert len(list(model[0].parameters(recurse=False))) == 2


# ── mode and traversal ────────────────────────────────────────────────────────


def test_train_and_eval_reach_every_descendant():
    model = _model()
    model.eval()
    assert not any(sub.training for sub in model.modules())
    model.train()
    assert all(sub.training for sub in model.modules())


def test_apply_visits_every_module_including_the_root():
    seen = []
    model = _model()
    model.apply(lambda module: seen.append(id(module)))
    assert seen == [id(m) for m in model.modules()][::-1] or len(seen) == len(list(model.modules()))


def test_apply_returns_the_module_so_it_chains():
    model = _model()
    assert model.apply(lambda m: None) is model


def test_requires_grad_reaches_every_parameter():
    model = _model()
    model.requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    model.requires_grad_(True)
    assert all(p.requires_grad for p in model.parameters())


def test_freezing_a_submodule_leaves_the_rest_trainable():
    """The transfer-learning idiom, and it fails silently in both
    directions: freeze too much and nothing learns, too little and the
    backbone drifts."""
    model = _model()
    model[0].requires_grad_(False)
    frozen = {id(p) for p in model[0].parameters()}
    for name, param in model.named_parameters():
        assert param.requires_grad == (id(param) not in frozen), name


# ── gradients ─────────────────────────────────────────────────────────────────


def test_zero_grad_set_to_none_drops_the_buffer():
    model = nn.Linear(4, 4)
    model(_x()).sum().backward()
    model.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in model.parameters())


def test_zero_grad_without_set_to_none_leaves_a_zero_buffer():
    """The two modes were the same thing.

    ``TensorImpl.zero_grad`` *resets* the gradient rather than filling
    it, so ``set_to_none=False`` also produced ``None`` — which is the
    one outcome it exists to avoid.  Gradient accumulation reads
    ``p.grad`` between backward passes; a ``None`` there is an
    ``AttributeError`` at best, and wherever the code guards with ``if
    p.grad is not None`` it is a silently skipped update.
    """
    model = nn.Linear(4, 4)
    model(_x()).sum().backward()
    model.zero_grad(set_to_none=False)
    for param in model.parameters():
        assert param.grad is not None
        assert param.grad.shape == param.shape
        assert np.allclose(_v(param.grad), 0.0)


def test_the_optimiser_zero_grad_agrees_with_the_module_one():
    model = nn.Linear(4, 4)
    model(_x()).sum().backward()
    lucid.optim.SGD(model.parameters(), lr=0.1).zero_grad(set_to_none=False)
    assert all(p.grad is not None and np.allclose(_v(p.grad), 0.0) for p in model.parameters())


def test_zeroing_then_one_backward_equals_one_backward():
    """What accumulation rests on: the zeroed buffer must contribute
    nothing, not a stale value."""
    model = nn.Linear(4, 4)
    model(_x()).sum().backward()
    alone = _v(model.weight.grad).copy()
    model.zero_grad(set_to_none=False)
    model(_x()).sum().backward()
    assert np.allclose(_v(model.weight.grad), alone, atol=1e-6)


def test_two_backwards_without_zeroing_accumulate():
    model = nn.Linear(4, 4)
    model(_x()).sum().backward()
    once = _v(model.weight.grad).copy()
    model(_x()).sum().backward()
    assert np.allclose(_v(model.weight.grad), 2 * once, atol=1e-6)


# ── buffers ───────────────────────────────────────────────────────────────────


def test_a_non_persistent_buffer_is_visible_but_not_saved():
    module = nn.Module()
    module.register_buffer("kept", lucid.zeros(3))
    module.register_buffer("cached", lucid.zeros(3), persistent=False)
    assert len(list(module.named_buffers())) == 2
    assert list(module.state_dict()) == ["kept"]


def test_a_buffer_is_not_a_parameter():
    module = nn.Module()
    module.register_buffer("stats", lucid.zeros(3))
    assert list(module.parameters()) == []
    assert "stats" in dict(module.named_buffers())


def test_buffers_follow_a_dtype_conversion():
    layer = nn.BatchNorm1d(4).to(lucid.float64)
    assert layer.running_mean.dtype is lucid.float64


# ── state_dict ────────────────────────────────────────────────────────────────


def test_state_dict_keys_are_the_dotted_parameter_names():
    keys = set(_model().state_dict())
    assert keys == {"0.weight", "0.bias", "2.0.weight", "2.0.bias"}


def test_a_round_trip_reproduces_the_outputs():
    source, target = _model(), _model()
    target.load_state_dict(source.state_dict())
    assert np.allclose(_v(source(_x())), _v(target(_x())), atol=1e-6)


def test_load_state_dict_is_strict_by_default():
    model = _model()
    incomplete = model.state_dict()
    incomplete.pop("0.weight")
    with pytest.raises(Exception):
        model.load_state_dict(incomplete)


def test_a_non_strict_load_names_what_was_missing_and_what_was_extra():
    model = _model()
    state = model.state_dict()
    state.pop("0.weight")
    state["not_a_layer.weight"] = lucid.zeros(1)
    result = model.load_state_dict(state, strict=False)
    missing = list(getattr(result, "missing_keys", ()))
    unexpected = list(getattr(result, "unexpected_keys", ()))
    assert "0.weight" in missing
    assert "not_a_layer.weight" in unexpected


def test_loading_does_not_replace_the_parameter_objects():
    """An optimizer holds the objects, so a load that rebound them would
    leave it stepping the old ones — the same failure the lazy layers
    had, arriving by a different route."""
    model = _model()
    before = [id(p) for p in model.parameters()]
    model.load_state_dict(_model().state_dict())
    assert [id(p) for p in model.parameters()] == before


def test_keep_vars_returns_the_parameters_themselves():
    state = _model().state_dict(keep_vars=True)
    assert any(getattr(t, "requires_grad", False) for t in state.values())
    detached = _model().state_dict()
    assert not any(getattr(t, "requires_grad", False) for t in detached.values())


# ── conversion ────────────────────────────────────────────────────────────────


def test_to_dtype_converts_every_parameter_and_returns_self():
    model = _model()
    assert model.to(lucid.float64) is model
    assert all(p.dtype is lucid.float64 for p in model.parameters())


def test_a_converted_module_still_runs():
    model = _model().to(lucid.float64)
    out = model(lucid.tensor(np.ones((2, 4), dtype=np.float64)))
    assert np.isfinite(_v(out)).all()


# ── repr ──────────────────────────────────────────────────────────────────────


def test_repr_shows_the_nested_tree():
    text = repr(_model())
    assert "Sequential" in text and "Linear" in text and "ReLU" in text
    assert text.count("\n") > 3


def test_extra_repr_is_included():
    class Annotated(nn.Module):
        def extra_repr(self):
            return "note=here"

        def forward(self, x):
            return x

    assert "note=here" in repr(Annotated())


def test_a_layer_reports_its_own_configuration():
    assert "4" in repr(nn.Linear(4, 2))
    assert "2" in repr(nn.Linear(4, 2))


# ── registration ──────────────────────────────────────────────────────────────


def test_assigning_a_parameter_registers_it():
    module = nn.Module()
    module.weight = nn.Parameter(lucid.zeros(3))
    assert "weight" in dict(module.named_parameters())


def test_assigning_a_module_registers_it():
    parent = nn.Module()
    parent.child = nn.Linear(4, 4)
    assert "child" in dict(parent.named_children())
    assert len(list(parent.parameters())) == 2


def test_assigning_a_plain_tensor_does_not_register_it():
    """The distinction ``Parameter`` exists to draw — a scratch tensor
    stored on a module must not end up in the optimizer."""
    module = nn.Module()
    module.scratch = lucid.zeros(3)
    assert list(module.parameters()) == []
    assert list(module.named_buffers()) == []


def test_replacing_a_registered_child_replaces_it_in_the_walk():
    model = _model()
    model[0] = nn.Linear(4, 8)
    assert tuple(model.get_parameter("0.weight").shape) == (8, 4)
    assert len(list(model.modules())) == len(list(_model().modules()))
