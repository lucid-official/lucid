"""``nn.utils.parametrizations.*`` are built on ``register_parametrization`` (LCD-646).

``weight_norm`` and ``spectral_norm`` here used to be plain re-exports of
the legacy hook-based functions (``<name>_g`` / ``<name>_v`` / ``<name>_orig``
leaves, installed by a forward pre-hook), and ``orthogonal`` did not exist
at all.  The reference's ``parametrizations.*`` build all three on
``register_parametrization`` instead, so the trainable leaves land at
``parametrizations.<name>.original0`` / ``original1`` / ``original`` and a
checkpoint saved against one framework loads into the other.  Each check
below is one way that would be invisible if silently wrong: the registry's
own names, a loaded checkpoint's forward value, a step actually moving the
originals, and — spectral norm only — eval mode actually freezing the
power iteration.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.nn.utils import parametrizations as P
from lucid.test._fixtures.devices import metal_available

_METAL = pytest.param(
    "metal", marks=pytest.mark.skipif(not metal_available(), reason="needs Metal")
)


def _as_lucid_state_dict(ref_module: object) -> dict[str, lucid.Tensor]:
    """A reference module's ``state_dict``, converted to Lucid tensors."""
    return {
        k: lucid.tensor(v.detach().numpy())
        for k, v in ref_module.state_dict().items()  # type: ignore[attr-defined]
    }


def _pair(ref, kind: str):
    """A matching (reference, Lucid) module, unparametrized, same weights."""
    if kind == "linear":
        ref_m, our_m = ref.nn.Linear(6, 4), nn.Linear(6, 4)
    else:
        ref_m, our_m = ref.nn.Conv2d(3, 8, 3), nn.Conv2d(3, 8, 3)
    our_m.load_state_dict(_as_lucid_state_dict(ref_m))
    return ref_m, our_m


def _input(kind: str, n: int) -> np.ndarray:
    shape = (n, 6) if kind == "linear" else (n, 3, 8, 8)
    return np.random.default_rng(0).standard_normal(shape).astype("float32")


_KINDS = ["linear", "conv2d"]

#: (our constructor, the reference's, under ``nn.utils.parametrizations``)
_CTORS = {
    "weight_norm": (P.weight_norm, "weight_norm"),
    "spectral_norm": (P.spectral_norm, "spectral_norm"),
    "orthogonal": (P.orthogonal, "orthogonal"),
}


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_CTORS))
@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize("device", ["cpu", _METAL])
def test_names_and_forward_match_the_reference(ref, name, kind, device):
    our_ctor, ref_name = _CTORS[name]
    ref_ctor = getattr(ref.nn.utils.parametrizations, ref_name)

    ref_m, our_m = _pair(ref, kind)
    ref_wrapped = ref_ctor(ref_m)
    our_wrapped = our_ctor(our_m).to(device=device)

    assert sorted(n for n, _ in our_wrapped.named_parameters()) == sorted(
        n for n, _ in ref_wrapped.named_parameters()
    )
    assert sorted(n for n, _ in our_wrapped.named_buffers()) == sorted(
        n for n, _ in ref_wrapped.named_buffers()
    )
    assert sorted(our_wrapped.state_dict()) == sorted(ref_wrapped.state_dict())

    # Load the reference's own state (post-registration, so _u/_v/base are
    # the reference's actual values) and check the forward value agrees.
    our_wrapped.load_state_dict(
        {k: v.to(device=device) for k, v in _as_lucid_state_dict(ref_wrapped).items()}
    )
    x = _input(kind, 3)
    out_ref = ref_wrapped(ref.tensor(x)).detach().numpy()
    out_our = our_wrapped(lucid.tensor(x).to(device=device)).numpy()
    assert np.allclose(out_ref, out_our, atol=1e-3), name


@pytest.mark.parametrize("name", list(_CTORS))
def test_a_training_step_moves_the_originals(name):
    """Every one of the three has to actually train, not just install the
    right names — a reparametrization that silently froze its originals
    would pass every structural check above and still be a dead end."""
    our_ctor, _ = _CTORS[name]
    layer = our_ctor(nn.Linear(6, 6))
    opt = lucid.optim.SGD(layer.parameters(), lr=0.1)
    before = {n: p.numpy().copy() for n, p in layer.named_parameters()}

    x = lucid.tensor(_input("linear", 3))
    for _ in range(3):
        opt.zero_grad()
        layer(x).sum().backward()
        opt.step()

    after = {n: p.numpy() for n, p in layer.named_parameters()}
    assert any(not np.allclose(before[n], after[n]) for n in before), name


@pytest.mark.parity
def test_spectral_norm_eval_vs_train_matches_the_reference(ref):
    """Training-mode reads keep advancing the power iteration (so two
    reads differ); eval freezes it (so two reads agree) — both sides of
    that behaviour have to match the reference, not just the values."""
    ref_m, our_m = _pair(ref, "linear")
    ref_sn = ref.nn.utils.parametrizations.spectral_norm(ref_m, n_power_iterations=1)
    our_sn = P.spectral_norm(our_m, n_power_iterations=1)
    our_sn.load_state_dict(_as_lucid_state_dict(ref_sn))

    x = _input("linear", 2)
    ref_sn.train()
    our_sn.train()
    for _ in range(3):
        out_ref = ref_sn(ref.tensor(x)).detach().numpy()
        out_our = our_sn(lucid.tensor(x)).numpy()
        assert np.allclose(out_ref, out_our, atol=1e-4)

    ref_sn.eval()
    our_sn.eval()
    ref_eval_1 = ref_sn(ref.tensor(x)).detach().numpy()
    ref_eval_2 = ref_sn(ref.tensor(x)).detach().numpy()
    our_eval_1 = our_sn(lucid.tensor(x)).numpy()
    our_eval_2 = our_sn(lucid.tensor(x)).numpy()
    assert np.allclose(ref_eval_1, ref_eval_2)
    assert np.allclose(our_eval_1, our_eval_2)
    assert np.allclose(ref_eval_1, our_eval_1, atol=1e-4)
