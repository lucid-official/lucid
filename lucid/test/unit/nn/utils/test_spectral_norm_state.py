"""``spectral_norm``'s removal and checkpoint versioning against the reference (LCD-792).

``remove_spectral_norm`` handed back the unnormalised ``W_orig`` (and said so
in its docstring) where the reference bakes ``W_orig / sigma`` into a new leaf
Parameter, so a module computed something else the moment the reparametrisation
was removed.  And ``spectral_norm`` recorded no version in the checkpoint
metadata and could not load a pre-v1 checkpoint (no ``weight_v``), because the
module had no state-dict hook API to hang either on.
"""

import copy
import pickle

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.nn import utils as U
from lucid.nn.utils.spectral_norm import (
    SpectralNorm,
    SpectralNormLoadStateDictPreHook,
    SpectralNormStateDictHook,
)


def _np(t) -> np.ndarray:
    return t.detach().to("cpu").numpy() if hasattr(t, "to") else t.detach().numpy()


def _normed_pair(ref, out: int = 4, inp: int = 6, seed: int = 0):
    """A spectral-normed Linear on each framework with the same weight, ``u`` and ``v``."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((out, inp)).astype(np.float32)
    u = rng.standard_normal(out).astype(np.float32)
    v = rng.standard_normal(inp).astype(np.float32)
    ours, theirs = nn.Linear(inp, out), ref.nn.Linear(inp, out)
    b = rng.standard_normal(out).astype(np.float32)
    with lucid.no_grad():
        ours.weight.copy_(lucid.tensor(w))
        ours.bias.copy_(lucid.tensor(b))
    theirs.weight.data = ref.tensor(w)
    theirs.bias.data = ref.tensor(b)
    U.spectral_norm(ours)
    ref.nn.utils.spectral_norm(theirs)
    with lucid.no_grad():
        ours.weight_u.copy_(lucid.tensor(u / np.linalg.norm(u)))
        ours.weight_v.copy_(lucid.tensor(v / np.linalg.norm(v)))
    theirs.weight_u.data = ref.tensor(u / np.linalg.norm(u))
    theirs.weight_v.data = ref.tensor(v / np.linalg.norm(v))
    return ours, theirs


@pytest.mark.parity
@pytest.mark.parametrize("training", [True, False])
def test_remove_spectral_norm_bakes_the_normalised_weight_like_the_reference(ref, training) -> None:
    ours, theirs = _normed_pair(ref)
    x = np.random.default_rng(1).standard_normal((3, 6)).astype(np.float32)
    for m in (ours, theirs):
        m.train(training)
    ours(lucid.tensor(x))
    theirs(ref.tensor(x))
    U.remove_spectral_norm(ours)
    ref.nn.utils.remove_spectral_norm(theirs)

    assert [k for k, _ in ours.named_parameters()] == [k for k, _ in theirs.named_parameters()]
    assert [k for k, _ in ours.named_buffers()] == [k for k, _ in theirs.named_buffers()] == []
    assert isinstance(ours.weight, nn.Parameter) and ours.weight.requires_grad
    np.testing.assert_allclose(_np(ours.weight), theirs.weight.detach().numpy(), rtol=1e-5)
    np.testing.assert_allclose(
        _np(ours(lucid.tensor(x))), theirs(ref.tensor(x)).detach().numpy(), rtol=1e-5, atol=1e-6
    )
    assert not ours._forward_pre_hooks


def test_removal_leaves_the_module_computing_what_it_computed() -> None:
    layer = U.spectral_norm(nn.Linear(6, 4))
    layer.eval()
    x = lucid.randn(3, 6)
    before = _np(layer(x))
    U.remove_spectral_norm(layer)
    np.testing.assert_allclose(_np(layer(x)), before, rtol=1e-5, atol=1e-6)


def test_removal_drops_the_state_dict_hooks_so_the_weight_can_be_normalised_again() -> None:
    layer = U.spectral_norm(nn.Linear(6, 4))
    assert len(layer._state_dict_hooks) == 1 and len(layer._load_state_dict_pre_hooks) == 1
    U.remove_spectral_norm(layer)
    assert not layer._state_dict_hooks and not layer._load_state_dict_pre_hooks
    assert "spectral_norm" not in layer.state_dict()._metadata[""]
    U.spectral_norm(layer)
    assert layer.state_dict()._metadata[""]["spectral_norm"] == {"weight.version": 1}


@pytest.mark.parity
def test_the_checkpoint_records_the_version_where_the_reference_does(ref) -> None:
    ours, theirs = _normed_pair(ref)
    assert dict(ours.state_dict()._metadata) == dict(theirs.state_dict()._metadata)
    assert list(ours.state_dict()) == list(theirs.state_dict())
    assert isinstance(
        ours._state_dict_hooks[next(iter(ours._state_dict_hooks))], SpectralNormStateDictHook
    )


def test_a_checkpoint_round_trips_through_save_and_load(tmp_path) -> None:
    src = U.spectral_norm(nn.Linear(6, 4))
    path = str(tmp_path / "sn.lt")
    lucid.save(src.state_dict(), path)
    dst = U.spectral_norm(nn.Linear(6, 4))
    assert dst.load_state_dict(lucid.load(path)).missing_keys == []
    np.testing.assert_array_equal(_np(dst.weight_v), _np(src.weight_v))


def _pre_v1_checkpoint(ref, ours, theirs):
    """What an older release saved: the normalised ``weight`` and ``_orig`` / ``_u``, no ``_v``."""
    ours.eval()
    theirs.eval()
    ours(lucid.ones(1, ours.in_features))
    theirs(ref.ones(1, ours.in_features))
    sd = {
        "bias": theirs.bias.detach().clone(),
        "weight": theirs.weight.detach().clone(),
        "weight_orig": theirs.weight_orig.detach().clone(),
        "weight_u": theirs.weight_u.detach().clone(),
    }
    return sd


@pytest.mark.parity
def test_a_pre_v1_checkpoint_gets_its_v_rebuilt_as_the_reference_rebuilds_it(ref) -> None:
    # Full column rank: on a rank-deficient Gram matrix both pseudo-inverses amplify rounding noise.
    ours, theirs = _normed_pair(ref, out=6, inp=4)
    sd = _pre_v1_checkpoint(ref, ours, theirs)
    fresh_ours, fresh_theirs = _normed_pair(ref, out=6, inp=4, seed=5)
    assert "spectral_norm" not in getattr(sd, "_metadata", {})

    theirs_result = fresh_theirs.load_state_dict(sd, strict=False)
    ours_sd = {k: lucid.tensor(v.numpy()) for k, v in sd.items()}
    ours_result = fresh_ours.load_state_dict(ours_sd, strict=False)

    assert ours_result.missing_keys == list(theirs_result.missing_keys) == []
    assert ours_result.unexpected_keys == list(theirs_result.unexpected_keys) == []
    np.testing.assert_allclose(
        _np(fresh_ours.weight_v), fresh_theirs.weight_v.detach().numpy(), rtol=1e-3, atol=1e-5
    )
    np.testing.assert_allclose(
        _np(fresh_ours.weight_u), fresh_theirs.weight_u.detach().numpy(), rtol=1e-6
    )


@pytest.mark.parity
def test_a_pre_v1_checkpoint_missing_a_key_reports_it_under_strict(ref) -> None:
    ours, theirs = _normed_pair(ref)
    sd = _pre_v1_checkpoint(ref, ours, theirs)
    del sd["weight_u"]
    ours_sd = {k: lucid.tensor(v.numpy()) for k, v in sd.items()}
    with pytest.raises(RuntimeError) as theirs_err:
        _normed_pair(ref, seed=5)[1].load_state_dict(sd)
    with pytest.raises(RuntimeError) as ours_err:
        _normed_pair(ref, seed=5)[0].load_state_dict(ours_sd)
    assert "weight_u" in str(theirs_err.value) and "weight_u" in str(ours_err.value)


def test_a_current_checkpoint_without_metadata_is_taken_as_current() -> None:
    src = U.spectral_norm(nn.Linear(6, 4))
    plain = dict(src.state_dict())
    dst = U.spectral_norm(nn.Linear(6, 4))
    assert dst.load_state_dict(plain).missing_keys == []
    np.testing.assert_array_equal(_np(dst.weight_v), _np(src.weight_v))


def test_the_hooks_are_plain_objects_so_the_module_still_pickles_and_copies() -> None:
    layer = U.spectral_norm(nn.Linear(6, 4))
    for twin in (pickle.loads(pickle.dumps(layer)), copy.deepcopy(layer)):
        (hook,) = twin._state_dict_hooks.values()
        (wrapped,) = twin._load_state_dict_pre_hooks.values()
        assert isinstance(hook, SpectralNormStateDictHook)
        assert isinstance(wrapped.hook, SpectralNormLoadStateDictPreHook)
        assert hook.fn is wrapped.hook.fn is SpectralNorm.find(twin, "weight")
        assert twin.state_dict()._metadata[""]["spectral_norm"] == {"weight.version": 1}
