"""The legacy ``weight_norm`` / ``spectral_norm`` hooks against the reference (LCD-781).

Four things were wrong, each invisible to a test that applies the hook once and
runs one training forward: ``eval -> train`` raised (the power iteration wrote
into buffers the eval graph still held), a module carrying any of the legacy
reparametrisations could not be pickled (the hooks were closures), ``dim=None``
and ``dim=-1`` did not mean what the reference means by them, and a
parametrised ``weight`` was skipped by the norm layers' ``reset_parameters``.
"""

import copy
import importlib
import pickle

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.nn import utils as U
from lucid.nn.modules._affine import _AffineParameters
from lucid.nn.utils import parametrize, prune
from lucid.nn.utils.spectral_norm import SpectralNorm
from lucid.nn.utils.weight_norm import WeightNorm
from lucid.test._fixtures.devices import metal_available

_METAL = pytest.param(
    "metal", marks=pytest.mark.skipif(not metal_available(), reason="needs Metal")
)
_DEVICES = ["cpu", _METAL]


def _np(t: lucid.Tensor) -> np.ndarray:
    return t.detach().to("cpu").numpy()


def _ref_utils(ref, name: str):
    return importlib.import_module(f"{ref.__name__}.nn.utils.{name}")


def _pair(ref, shape: tuple[int, ...], seed: int = 0):
    """A Lucid module and a reference module whose ``weight`` is the same array."""
    w = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    ours, theirs = nn.Module(), ref.nn.Module()
    ours.weight = nn.Parameter(lucid.tensor(w))
    theirs.weight = ref.nn.Parameter(ref.tensor(w))
    return ours, theirs


def _linear_pair(ref, out: int = 4, inp: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((out, inp)).astype(np.float32)
    b = rng.standard_normal(out).astype(np.float32)
    ours, theirs = nn.Linear(inp, out), ref.nn.Linear(inp, out)
    with lucid.no_grad():
        ours.weight.copy_(lucid.tensor(w))
        ours.bias.copy_(lucid.tensor(b))
    theirs.weight.data = ref.tensor(w)
    theirs.bias.data = ref.tensor(b)
    return ours, theirs


# ── weight_norm: which axis g keeps ─────────────────────────────────────

_SHAPES = [(4, 3), (2, 3, 5), (6,)]
_DIMS = [0, 1, 2, -1, -2, -3, None]


@pytest.mark.parity
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dim", _DIMS)
def test_weight_norm_dim_table_matches_reference(ref, shape: tuple[int, ...], dim) -> None:
    ours, theirs = _pair(ref, shape)
    try:
        wn = _ref_utils(ref, "weight_norm").WeightNorm
        wn.apply(theirs, "weight", dim)
    except IndexError:
        with pytest.raises(IndexError):
            U.weight_norm(ours, "weight", dim=dim)
        assert "weight" in ours._parameters and not ours._forward_pre_hooks
        return
    U.weight_norm(ours, "weight", dim=dim)
    assert tuple(ours.weight_g.shape) == tuple(theirs.weight_g.shape)
    np.testing.assert_allclose(_np(ours.weight_g), theirs.weight_g.detach().numpy(), rtol=1e-5)
    np.testing.assert_allclose(_np(ours.weight_v), theirs.weight_v.detach().numpy(), rtol=1e-6)
    np.testing.assert_allclose(
        _np(ours.weight), theirs.weight.detach().numpy(), rtol=1e-5, atol=1e-6
    )


def test_none_and_minus_one_are_one_scalar_magnitude() -> None:
    for dim in (None, -1):
        m = U.weight_norm(nn.Linear(3, 2), dim=dim)
        assert tuple(m.weight_g.shape) == ()
        assert m.weight_g.shape == ()
        np.testing.assert_allclose(
            float(m.weight_g.item()), float(np.linalg.norm(_np(m.weight_v))), rtol=1e-6
        )
    # ``-2`` still counts from the end.
    assert tuple(U.weight_norm(nn.Linear(3, 2), dim=-2).weight_g.shape) == (2, 1)


@pytest.mark.parity
@pytest.mark.parametrize("dim", [0, 1, -1, None])
def test_weight_norm_forward_and_gradients_match_reference(ref, dim) -> None:
    ours, theirs = _linear_pair(ref)
    U.weight_norm(ours, "weight", dim=dim)
    _ref_utils(ref, "weight_norm").WeightNorm.apply(theirs, "weight", dim)
    x = np.random.default_rng(3).standard_normal((5, 6)).astype(np.float32)
    got, want = ours(lucid.tensor(x)), theirs(ref.tensor(x))
    np.testing.assert_allclose(_np(got), want.detach().numpy(), rtol=1e-5, atol=1e-5)
    got.sum().backward()
    want.sum().backward()
    for name in ("weight_g", "weight_v", "bias"):
        np.testing.assert_allclose(
            _np(getattr(ours, name).grad),
            getattr(theirs, name).grad.numpy(),
            rtol=1e-4,
            atol=1e-5,
            err_msg=name,
        )


@pytest.mark.parity
@pytest.mark.parametrize("dim", [0, 1, None])
def test_remove_weight_norm_matches_reference(ref, dim) -> None:
    ours, theirs = _linear_pair(ref)
    U.weight_norm(ours, "weight", dim=dim)
    ref.nn.utils.weight_norm(theirs, "weight", dim=dim)
    U.remove_weight_norm(ours)
    ref.nn.utils.remove_weight_norm(theirs)
    assert [k for k, _ in ours.named_parameters()] == [k for k, _ in theirs.named_parameters()]
    np.testing.assert_allclose(_np(ours.weight), theirs.weight.detach().numpy(), rtol=1e-5)
    assert not ours._forward_pre_hooks and isinstance(ours.weight, nn.Parameter)


def test_applying_weight_norm_twice_is_refused() -> None:
    m = U.weight_norm(nn.Linear(3, 2))
    with pytest.raises(RuntimeError, match="Cannot register two weight_norm hooks"):
        U.weight_norm(m)
    assert len(m._forward_pre_hooks) == 1 and sorted(m._parameters) == [
        "bias",
        "weight_g",
        "weight_v",
    ]
    # Another parameter of the same module is a different registration.
    U.weight_norm(m, "bias", dim=0)
    assert len(m._forward_pre_hooks) == 2


def test_removing_weight_norm_that_is_not_there_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        U.remove_weight_norm(nn.Linear(3, 2))
    m = U.weight_norm(nn.Linear(3, 2))
    U.remove_weight_norm(m)
    with pytest.raises(ValueError):
        U.remove_weight_norm(m)


def test_weight_norm_of_a_one_dimensional_weight() -> None:
    m = nn.Module()
    m.weight = nn.Parameter(lucid.tensor([3.0, -4.0, 12.0]))
    U.weight_norm(m, "weight", dim=0)
    np.testing.assert_allclose(_np(m.weight), [3.0, -4.0, 12.0], rtol=1e-6)
    assert tuple(m.weight_g.shape) == (3,)


# ── spectral_norm ───────────────────────────────────────────────────────


def _seeded_pair(ref, **kwargs):
    """Layers with the same weight, ``u`` and ``v`` -- the iteration is then deterministic."""
    ours, theirs = _linear_pair(ref, out=5, inp=7)
    U.spectral_norm(ours, **kwargs)
    ref.nn.utils.spectral_norm(theirs, **kwargs)
    with lucid.no_grad():
        ours.weight_u.copy_(lucid.tensor(theirs.weight_u.numpy()))
        ours.weight_v.copy_(lucid.tensor(theirs.weight_v.numpy()))
    return ours, theirs


@pytest.mark.parity
@pytest.mark.parametrize("n_iter", [1, 3])
def test_spectral_norm_modes_cycle_like_the_reference(ref, n_iter: int) -> None:
    ours, theirs = _seeded_pair(ref, n_power_iterations=n_iter)
    x = np.random.default_rng(5).standard_normal((4, 7)).astype(np.float32)
    # Every transition, with the graph kept and dropped: the old code raised on eval -> train.
    for mode, grad in [
        ("train", True),
        ("eval", True),
        ("train", True),
        ("eval", False),
        ("train", False),
        ("eval", True),
        ("train", True),
        ("train", True),
    ]:
        ours.train(mode == "train")
        theirs.train(mode == "train")
        if grad:
            got, want = ours(lucid.tensor(x)), theirs(ref.tensor(x))
            got.sum().backward()
            want.sum().backward()
            np.testing.assert_allclose(
                _np(ours.weight_orig.grad), theirs.weight_orig.grad.numpy(), rtol=1e-4, atol=1e-5
            )
            ours.weight_orig.grad = None
            theirs.weight_orig.grad = None
        else:
            with lucid.no_grad():
                got = ours(lucid.tensor(x))
            with ref.no_grad():
                want = theirs(ref.tensor(x))
        np.testing.assert_allclose(_np(got), want.detach().numpy(), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(
            _np(ours.weight_u), theirs.weight_u.numpy(), rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            _np(ours.weight_v), theirs.weight_v.numpy(), rtol=1e-4, atol=1e-5
        )


@pytest.mark.parametrize("device", _DEVICES)
def test_eval_then_train_then_backward_does_not_write_into_a_saved_buffer(device: str) -> None:
    layer = U.spectral_norm(nn.Linear(4, 3).to(device))
    x = lucid.ones(2, 4, device=device)
    layer.eval()
    held = layer(x)  # an eval graph that still references u and v
    layer.train()
    layer(x).sum().backward()  # the power iteration writes u and v in place
    held.sum().backward()
    assert layer.weight_orig.grad is not None


def test_spectral_norm_buffers_stay_the_same_tensors_in_every_mode() -> None:
    layer = U.spectral_norm(nn.Linear(4, 3))
    u, v = layer.weight_u, layer.weight_v
    x = lucid.ones(1, 4)
    for training in (True, False, True, False):
        layer.train(training)
        layer(x)
        assert layer.weight_u is u and layer.weight_v is v


def test_spectral_norm_keeps_the_parameter_object() -> None:
    layer = nn.Linear(4, 3)
    weight = layer.weight
    U.spectral_norm(layer)
    assert layer.weight_orig is weight and isinstance(weight, nn.Parameter)
    assert list(layer._parameters) == ["bias", "weight_orig"]
    assert list(layer._buffers) == ["weight_u", "weight_v"]
    U.remove_spectral_norm(layer)
    assert layer.weight is weight


def test_a_frozen_weight_stays_frozen_under_spectral_norm() -> None:
    layer = nn.Linear(4, 3)
    layer.weight.requires_grad_(False)
    U.spectral_norm(layer)
    assert not layer.weight_orig.requires_grad
    layer.train()
    layer(lucid.ones(1, 4)).sum().backward()
    assert layer.weight_orig.grad is None


def test_spectral_norm_refuses_what_the_reference_refuses() -> None:
    layer = U.spectral_norm(nn.Linear(4, 3))
    with pytest.raises(RuntimeError, match="Cannot register two spectral_norm hooks"):
        U.spectral_norm(layer)
    for bad in (0, -1):
        with pytest.raises(ValueError):
            U.spectral_norm(nn.Linear(4, 3), n_power_iterations=bad)
    with pytest.raises(ValueError):
        U.spectral_norm(nn.Linear(4, 3), dim=5)
    with pytest.raises(ValueError):
        U.remove_spectral_norm(nn.Linear(4, 3))
    refused = nn.Linear(4, 3)
    with pytest.raises(ValueError):
        U.spectral_norm(refused, n_power_iterations=0)
    assert "weight" in refused._parameters and not refused._forward_pre_hooks


def test_spectral_norm_vectors_follow_the_weight_dtype() -> None:
    layer = nn.Linear(4, 3).to(lucid.float16)
    U.spectral_norm(layer)
    assert layer.weight_u.dtype == lucid.float16 and layer.weight_v.dtype == lucid.float16
    layer(lucid.ones(1, 4, dtype=lucid.float16))


def test_conv_transpose_spectral_norm_takes_the_output_channels_as_rows() -> None:
    layer = U.spectral_norm(nn.ConvTranspose2d(3, 4, 2))
    assert tuple(layer.weight_u.shape) == (4,) and tuple(layer.weight_v.shape) == (3 * 2 * 2,)


# ── pickle and deepcopy: the hook is an object, not a closure ───────────


def _weight_normed() -> nn.Module:
    return U.weight_norm(nn.Linear(5, 4))


def _weight_normed_scalar() -> nn.Module:
    return U.weight_norm(nn.Linear(5, 4), dim=None)


def _spectral() -> nn.Module:
    return U.spectral_norm(nn.Linear(5, 4))


def _pruned() -> nn.Module:
    return prune.l1_unstructured(nn.Linear(5, 4), "weight", amount=0.4)


def _pruned_twice() -> nn.Module:
    return prune.ln_structured(_pruned(), "weight", amount=1, n=2, dim=0)


def _custom_pruned() -> nn.Module:
    return prune.custom_from_mask(nn.Linear(5, 4), "weight", lucid.ones(4, 5))


def _all_three() -> nn.Module:
    m = nn.Sequential(U.weight_norm(nn.Linear(5, 4)), U.spectral_norm(nn.Linear(4, 4)))
    prune.l1_unstructured(m[1], "bias", amount=1)
    return m


_BUILDERS = {
    "weight_norm": _weight_normed,
    "weight_norm_scalar": _weight_normed_scalar,
    "spectral_norm": _spectral,
    "pruned": _pruned,
    "pruned_twice": _pruned_twice,
    "custom_from_mask": _custom_pruned,
    "all_three": _all_three,
}


@pytest.mark.parametrize("name", list(_BUILDERS))
@pytest.mark.parametrize("ran_forward", [False, True])
@pytest.mark.parametrize("how", ["pickle", "deepcopy"])
def test_a_module_with_a_legacy_reparametrization_pickles_and_copies(
    name: str, ran_forward: bool, how: str
) -> None:
    m = _BUILDERS[name]()
    x = lucid.ones(2, 5)
    m.eval()
    if ran_forward:
        m(x)
    clone = pickle.loads(pickle.dumps(m)) if how == "pickle" else copy.deepcopy(m)
    clone.eval()
    np.testing.assert_allclose(_np(clone(x)), _np(m(x)), rtol=1e-6, atol=1e-6)
    assert type(clone) is type(m)
    # The copy is its own module: its hooks are its own objects ...
    assert {id(h) for h in clone._forward_pre_hooks.values()}.isdisjoint(
        id(h) for h in m._forward_pre_hooks.values()
    )
    # ... and it still trains.
    clone.train()
    clone(x).sum().backward()


def test_a_copied_spectral_norm_module_keeps_iterating_independently() -> None:
    m = _spectral()
    m(lucid.ones(1, 5))
    clone = copy.deepcopy(m)
    u_before = _np(m.weight_u).copy()
    clone.train()
    for _ in range(3):
        clone(lucid.ones(1, 5))
    np.testing.assert_array_equal(_np(m.weight_u), u_before)
    assert not np.array_equal(_np(clone.weight_u), u_before)


def test_the_installed_hooks_are_the_method_objects() -> None:
    (wn,) = _weight_normed()._forward_pre_hooks.values()
    (sn,) = _spectral()._forward_pre_hooks.values()
    assert isinstance(wn, WeightNorm) and (wn.name, wn.dim) == ("weight", 0)
    assert isinstance(sn, SpectralNorm) and (sn.name, sn.dim, sn.n_power_iterations) == (
        "weight",
        0,
        1,
    )
    assert WeightNorm.find(_weight_normed(), "weight") is not None
    assert WeightNorm.find(_spectral(), "weight") is None
    assert SpectralNorm.find(_spectral(), "weight") is not None


def test_none_normalises_as_minus_one_in_the_hook() -> None:
    (hook,) = _weight_normed_scalar()._forward_pre_hooks.values()
    assert hook.dim == -1


def test_reparametrization_hooks_are_listed_for_the_tracer() -> None:
    from lucid.nn.utils._reparam import _ReparamHook

    m = _all_three()
    assert [type(h).__name__ for _, h in _ReparamHook.installed(m[0])] == ["WeightNorm"]
    assert [type(h).__name__ for _, h in _ReparamHook.installed(m[1])] == [
        "SpectralNorm",
        "L1Unstructured",
    ]
    assert list(_ReparamHook.installed(nn.Linear(2, 2))) == []
    handle = m[0].register_forward_pre_hook(lambda mod, args: None)
    assert len(list(_ReparamHook.installed(m[0]))) == 1  # a user's hook is not one of them
    handle.remove()


# ── reset_parameters reads the weight the module answers with ───────────


class _Same(nn.Module):
    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return x


def _affine_subclasses() -> list[type]:
    found: list[type] = []

    def walk(cls: type) -> None:
        for sub in cls.__subclasses__():
            if sub not in found:
                found.append(sub)
                walk(sub)

    walk(_AffineParameters)
    return found


_AFFINE_BUILD = {
    "LayerNorm": lambda: nn.LayerNorm(4),
    "RMSNorm": lambda: nn.RMSNorm(4),
    "GroupNorm": lambda: nn.GroupNorm(2, 4),
}


@pytest.mark.parametrize("name", list(_AFFINE_BUILD))
@pytest.mark.parametrize("target", ["weight", "bias"])
def test_reset_parameters_reaches_a_parametrized_parameter(name: str, target: str) -> None:
    layer = _AFFINE_BUILD[name]()
    if getattr(layer, target, None) is None:
        pytest.skip(f"{name} has no {target}")
    with lucid.no_grad():
        getattr(layer, target).copy_(lucid.full((4,), 7.0))
    parametrize.register_parametrization(layer, target, _Same())
    layer.reset_parameters()
    original = getattr(layer.parametrizations, target).original
    np.testing.assert_array_equal(_np(original), np.full(4, 1.0 if target == "weight" else 0.0))


def test_reset_parameters_still_skips_a_layer_without_the_pair() -> None:
    layer = nn.LayerNorm(4, elementwise_affine=False)
    layer.reset_parameters()
    assert not list(layer.parameters())


def test_every_class_resetting_through_the_shared_base_is_in_the_table() -> None:
    names = {c.__name__ for c in _affine_subclasses() if not c.__name__.startswith("_")}
    # The batch and instance norms reset their running statistics too and are
    # covered by their own tests; the three here are the ones that use the base as is.
    assert set(_AFFINE_BUILD) <= names


# ── the compile tracer leaves these pre-hooks to the graph ──────────────


@pytest.mark.skipif(not metal_available(), reason="compiled calls are traced on Metal")
@pytest.mark.parametrize("name", ["weight_norm", "spectral_norm", "pruned", "all_three"])
def test_a_compiled_call_traces_through_the_legacy_hooks(name: str) -> None:
    import warnings

    from lucid.compile._core.fallback import CompileFallbackWarning

    m = _BUILDERS[name]().to("metal").eval()
    x = lucid.randn(2, 5, device="metal")
    compiled = lucid.compile(m)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CompileFallbackWarning)
        with lucid.no_grad():
            got, want = compiled(x), m(x)
    np.testing.assert_allclose(_np(got), _np(want), rtol=1e-5, atol=1e-5)
