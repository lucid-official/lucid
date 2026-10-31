"""``nn.utils.prune`` against the reference (LCD-779).

The module was a thin layer on forward pre-hooks and answered differently from
the reference in a way that only shows once pruning is used the way a pruning
loop uses it: the original ``Parameter`` was *replaced* (an optimizer built
before pruning stopped updating, a frozen weight came back trainable), pruning
twice raised, an ``int`` amount was refused, ties over-pruned, the random mask
was Bernoulli, ``is_pruned`` ignored submodules, and half the methods did not
exist.  So the table is method x module x amount, compared mask for mask with
the reference, plus the invariants the reference states rather than computes.
"""

import importlib
import math
import pickle

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.nn.utils import prune
from lucid.test._fixtures.devices import metal_available

_METAL = pytest.param(
    "metal", marks=pytest.mark.skipif(not metal_available(), reason="needs Metal")
)
_DEVICES = ["cpu", _METAL]


def _rp(ref):
    """The reference's ``nn.utils.prune`` module (a submodule the package does not import)."""
    return importlib.import_module(ref.__name__ + ".nn.utils.prune")


def _np(t: lucid.Tensor) -> np.ndarray:
    return t.detach().to("cpu").numpy()


def _values(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Distinct magnitudes, either sign -- no ties, so both frameworks prune the same set."""
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    magnitude = (rng.permutation(n).astype(np.float32) + 1.0) / n
    return (magnitude * rng.choice([-1.0, 1.0], size=n).astype(np.float32)).reshape(shape)


def _make(ref, kind: str, seed: int = 0, device: str = "cpu", scale: float = 1.0):
    """A Lucid module and the reference module holding the same weights."""
    if kind == "linear":
        ours, theirs = nn.Linear(5, 6), ref.nn.Linear(5, 6)
    elif kind == "conv":
        ours, theirs = nn.Conv2d(3, 4, 2), ref.nn.Conv2d(3, 4, 2)
    else:
        raise ValueError(kind)
    # Weight and bias magnitudes are offset from each other (and from another module's
    # by ``scale``) so a ranking across tensors has no ties.
    w = _values(tuple(ours.weight.shape), seed) * scale
    b = _values(tuple(ours.bias.shape), seed + 1) * 0.9137 * scale
    ours = ours.to(device)
    with lucid.no_grad():
        ours.weight.copy_(lucid.tensor(w).to(device))
        ours.bias.copy_(lucid.tensor(b).to(device))
    theirs.weight.data = ref.tensor(w)
    theirs.bias.data = ref.tensor(b)
    return ours, theirs


def _same_mask(ours, theirs, name: str = "weight") -> None:
    np.testing.assert_array_equal(
        _np(getattr(ours, name + "_mask")), getattr(theirs, name + "_mask").numpy()
    )
    np.testing.assert_allclose(
        _np(getattr(ours, name)), getattr(theirs, name).detach().numpy(), rtol=1e-6
    )


# ── the deterministic methods, mask for mask ────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("kind", ["linear", "conv"])
@pytest.mark.parametrize("amount", [0, 0.0, 0.3, 0.5, 1.0, 1, 7, 30])
def test_l1_unstructured_matches_reference(ref, kind: str, amount) -> None:
    ours, theirs = _make(ref, kind)
    prune.l1_unstructured(ours, "weight", amount=amount)
    _rp(ref).l1_unstructured(theirs, "weight", amount=amount)
    _same_mask(ours, theirs)


@pytest.mark.parity
@pytest.mark.parametrize("amount", [0.4, 5])
def test_l1_unstructured_with_importance_scores_matches_reference(ref, amount) -> None:
    ours, theirs = _make(ref, "linear")
    scores = _values((6, 5), seed=9)
    prune.l1_unstructured(ours, "weight", amount, importance_scores=lucid.tensor(scores))
    _rp(ref).l1_unstructured(theirs, "weight", amount, importance_scores=ref.tensor(scores))
    _same_mask(ours, theirs)


@pytest.mark.parity
@pytest.mark.parametrize("n", [1, 2, 3, 0.5, math.inf, -math.inf])
@pytest.mark.parametrize("dim", [0, 1, -1, -2])
@pytest.mark.parametrize("amount", [0.5, 1])
def test_ln_structured_matches_reference(ref, n, dim: int, amount) -> None:
    ours, theirs = _make(ref, "conv")
    prune.ln_structured(ours, "weight", amount=amount, n=n, dim=dim)
    _rp(ref).ln_structured(theirs, "weight", amount=amount, n=n, dim=dim)
    _same_mask(ours, theirs)


@pytest.mark.parity
@pytest.mark.parametrize("dim", [0, 1])
def test_ln_structured_frobenius_norm_of_matrix_channels_matches_reference(ref, dim: int) -> None:
    ours, theirs = _make(ref, "linear")
    prune.ln_structured(ours, "weight", amount=2, n="fro", dim=dim)
    _rp(ref).ln_structured(theirs, "weight", amount=2, n="fro", dim=dim)
    _same_mask(ours, theirs)


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_frobenius_norm_of_a_convolution_channel_is_the_two_norm(dim: int) -> None:
    # The reference's own "fro" refuses more than two axes besides ``dim``.
    a, b = nn.Conv2d(3, 4, 2), nn.Conv2d(3, 4, 2)
    with lucid.no_grad():
        b.weight.copy_(a.weight)
    prune.ln_structured(a, "weight", amount=1, n="fro", dim=dim)
    prune.ln_structured(b, "weight", amount=1, n=2, dim=dim)
    np.testing.assert_array_equal(_np(a.weight_mask), _np(b.weight_mask))


@pytest.mark.parity
@pytest.mark.parametrize("dim", [0, 1, 2, -1])
def test_ln_structured_nuclear_norm_of_matrix_channels_matches_reference(ref, dim: int) -> None:
    w = _values((4, 3, 5), seed=3)
    ours, theirs = nn.Module(), ref.nn.Module()
    ours.weight = nn.Parameter(lucid.tensor(w))
    theirs.weight = ref.nn.Parameter(ref.tensor(w))
    prune.ln_structured(ours, "weight", amount=1, n="nuc", dim=dim)
    _rp(ref).ln_structured(theirs, "weight", amount=1, n="nuc", dim=dim)
    _same_mask(ours, theirs)


@pytest.mark.parity
@pytest.mark.parametrize("kind", ["linear", "conv"])
def test_ln_structured_with_importance_scores_matches_reference(ref, kind: str) -> None:
    ours, theirs = _make(ref, kind)
    scores = _values(tuple(ours.weight.shape), seed=7)
    prune.ln_structured(
        ours, "weight", amount=0.5, n=2, dim=0, importance_scores=lucid.tensor(scores)
    )
    _rp(ref).ln_structured(
        theirs, "weight", amount=0.5, n=2, dim=0, importance_scores=ref.tensor(scores)
    )
    _same_mask(ours, theirs)


@pytest.mark.parity
def test_custom_from_mask_and_identity_match_reference(ref) -> None:
    ours, theirs = _make(ref, "linear")
    mask = (_values((6, 5), seed=4) > 0).astype(np.float32)
    prune.custom_from_mask(ours, "weight", lucid.tensor(mask))
    _rp(ref).custom_from_mask(theirs, "weight", ref.tensor(mask))
    _same_mask(ours, theirs)
    ours, theirs = _make(ref, "linear")
    prune.identity(ours, "bias")
    _rp(ref).identity(theirs, "bias")
    _same_mask(ours, theirs, "bias")


@pytest.mark.parity
def test_the_bias_prunes_like_the_weight(ref) -> None:
    ours, theirs = _make(ref, "linear")
    prune.l1_unstructured(ours, "bias", amount=2)
    _rp(ref).l1_unstructured(theirs, "bias", amount=2)
    _same_mask(ours, theirs, "bias")
    assert [k for k, _ in ours.named_parameters()] == [k for k, _ in theirs.named_parameters()]


# ── iterative pruning composes through a PruningContainer ───────────────

_ROUNDS = {
    "l1+l1": [
        lambda p, m: p.l1_unstructured(m, "weight", amount=0.3),
        lambda p, m: p.l1_unstructured(m, "weight", amount=5),
    ],
    "l1+l1+l1": [
        lambda p, m: p.l1_unstructured(m, "weight", amount=0.2),
        lambda p, m: p.l1_unstructured(m, "weight", amount=0.2),
        lambda p, m: p.l1_unstructured(m, "weight", amount=0.2),
    ],
    "structured+l1": [
        lambda p, m: p.ln_structured(m, "weight", amount=1, n=2, dim=0),
        lambda p, m: p.l1_unstructured(m, "weight", amount=4),
    ],
    "l1+structured": [
        lambda p, m: p.l1_unstructured(m, "weight", amount=0.4),
        lambda p, m: p.ln_structured(m, "weight", amount=1, n=1, dim=0),
    ],
    "structured+structured_other_dim": [
        lambda p, m: p.ln_structured(m, "weight", amount=1, n=2, dim=0),
        lambda p, m: p.ln_structured(m, "weight", amount=2, n=2, dim=1),
    ],
    "structured+structured_same_dim": [
        lambda p, m: p.ln_structured(m, "weight", amount=1, n=2, dim=0),
        lambda p, m: p.ln_structured(m, "weight", amount=1, n=2, dim=0),
    ],
    "identity+l1": [
        lambda p, m: p.identity(m, "weight"),
        lambda p, m: p.l1_unstructured(m, "weight", amount=0.5),
    ],
}


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_ROUNDS))
def test_iterative_pruning_matches_reference(ref, name: str) -> None:
    ours, theirs = _make(ref, "linear")
    for step in _ROUNDS[name]:
        step(prune, ours)
        step(_rp(ref), theirs)
        _same_mask(ours, theirs)
    assert isinstance(next(iter(ours._forward_pre_hooks.values())), prune.PruningContainer)
    assert len(next(iter(ours._forward_pre_hooks.values()))) == len(_ROUNDS[name])


@pytest.mark.parity
def test_a_custom_mask_then_l1_matches_reference(ref) -> None:
    ours, theirs = _make(ref, "linear")
    mask = np.tril(np.ones((6, 5), np.float32))
    prune.custom_from_mask(ours, "weight", lucid.tensor(mask))
    _rp(ref).custom_from_mask(theirs, "weight", ref.tensor(mask))
    prune.l1_unstructured(ours, "weight", amount=6)
    _rp(ref).l1_unstructured(theirs, "weight", amount=6)
    _same_mask(ours, theirs)


@pytest.mark.parity
def test_a_pruned_module_computes_what_the_reference_computes(ref) -> None:
    ours, theirs = _make(ref, "linear")
    for step in _ROUNDS["l1+l1"]:
        step(prune, ours)
        step(_rp(ref), theirs)
    x = _values((3, 5), seed=11)
    np.testing.assert_allclose(
        _np(ours(lucid.tensor(x))), theirs(ref.tensor(x)).detach().numpy(), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parity
@pytest.mark.parametrize("kind", ["linear", "conv"])
def test_global_unstructured_matches_reference(ref, kind: str) -> None:
    o1, t1 = _make(ref, kind, seed=0)
    o2, t2 = _make(ref, "linear", seed=5, scale=1.0173)
    for amount in (7, 0.4):
        sites_o = [(o1, "weight"), (o2, "weight"), (o2, "bias")]
        sites_t = [(t1, "weight"), (t2, "weight"), (t2, "bias")]
        prune.global_unstructured(sites_o, pruning_method=prune.L1Unstructured, amount=amount)
        _rp(ref).global_unstructured(sites_t, pruning_method=_rp(ref).L1Unstructured, amount=amount)
        for o, t, name in ((o1, t1, "weight"), (o2, t2, "weight"), (o2, t2, "bias")):
            _same_mask(o, t, name)


@pytest.mark.parity
def test_global_unstructured_with_importance_scores_matches_reference(ref) -> None:
    o1, t1 = _make(ref, "linear", seed=0)
    o2, t2 = _make(ref, "linear", seed=5, scale=1.0173)
    s1, s2 = _values((6, 5), 21), _values((6, 5), 22) * 1.0173
    prune.global_unstructured(
        [(o1, "weight"), (o2, "weight")],
        prune.L1Unstructured,
        importance_scores={(o1, "weight"): lucid.tensor(s1), (o2, "weight"): lucid.tensor(s2)},
        amount=11,
    )
    _rp(ref).global_unstructured(
        [(t1, "weight"), (t2, "weight")],
        _rp(ref).L1Unstructured,
        importance_scores={(t1, "weight"): ref.tensor(s1), (t2, "weight"): ref.tensor(s2)},
        amount=11,
    )
    _same_mask(o1, t1)
    _same_mask(o2, t2)


@pytest.mark.parity
def test_remove_matches_reference(ref) -> None:
    ours, theirs = _make(ref, "linear")
    for step in _ROUNDS["l1+l1"]:
        step(prune, ours)
        step(_rp(ref), theirs)
    prune.remove(ours, "weight")
    _rp(ref).remove(theirs, "weight")
    np.testing.assert_allclose(_np(ours.weight), theirs.weight.detach().numpy(), rtol=1e-6)
    assert [k for k, _ in ours.named_parameters()] == [k for k, _ in theirs.named_parameters()]
    assert not ours._forward_pre_hooks and not ours._buffers


@pytest.mark.parity
def test_state_dict_keys_match_reference(ref) -> None:
    ours, theirs = _make(ref, "linear")
    prune.l1_unstructured(ours, "weight", amount=0.5)
    _rp(ref).l1_unstructured(theirs, "weight", amount=0.5)
    assert list(ours.state_dict()) == list(theirs.state_dict())


@pytest.mark.parity
@pytest.mark.parametrize("amount", [0.5, 3])
def test_prune_standalone_matches_reference(ref, amount) -> None:
    w = _values((6, 5), seed=2)
    got = prune.L1Unstructured(amount).prune(lucid.tensor(w))
    want = _rp(ref).L1Unstructured(amount).prune(ref.tensor(w))
    np.testing.assert_array_equal(_np(got), want.numpy())


@pytest.mark.parity
@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda p, m: p.l1_unstructured(m, "weight", amount=-1), ValueError),
        (lambda p, m: p.l1_unstructured(m, "weight", amount=1.5), ValueError),
        (lambda p, m: p.l1_unstructured(m, "weight", amount=31), ValueError),
        (lambda p, m: p.l1_unstructured(m, "weight", amount="a"), TypeError),
        (lambda p, m: p.ln_structured(m, "bias", amount=1, n=2, dim=0), ValueError),
        (lambda p, m: p.ln_structured(m, "weight", amount=7, n=2, dim=0), ValueError),
        (lambda p, m: p.random_structured(m, "bias", amount=1, dim=0), ValueError),
        (lambda p, m: p.remove(m, "weight"), ValueError),
    ],
)
def test_bad_arguments_raise_what_the_reference_raises(ref, call, error) -> None:
    ours, theirs = _make(ref, "linear")
    with pytest.raises(error):
        call(_rp(ref), theirs)
    with pytest.raises(error):
        call(prune, ours)
    # A refused call leaves the module as it was.
    assert sorted(ours._parameters) == ["bias", "weight"] and not ours._buffers
    assert not ours._forward_pre_hooks


# ── the invariants the reference states ─────────────────────────────────


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    "method", ["l1_unstructured", "random_unstructured", "identity", "custom_from_mask"]
)
def test_pruning_keeps_the_parameter_object(device: str, method: str) -> None:
    m = nn.Linear(5, 6).to(device)
    weight = m.weight
    if method == "identity":
        prune.identity(m, "weight")
    elif method == "custom_from_mask":
        prune.custom_from_mask(m, "weight", lucid.ones(6, 5, device=device))
    else:
        getattr(prune, method)(m, "weight", amount=0.5)
    assert m.weight_orig is weight
    assert isinstance(m.weight_orig, nn.Parameter)
    assert "weight" not in m._parameters and "weight_mask" in m._buffers
    assert list(m._parameters) == ["bias", "weight_orig"]


def test_an_optimizer_built_before_pruning_keeps_updating() -> None:
    m = nn.Linear(5, 6)
    opt = optim.SGD(m.parameters(), lr=0.1)
    prune.l1_unstructured(m, "weight", amount=0.5)
    before = _np(m.weight_orig).copy()
    m(lucid.ones(2, 5)).sum().backward()
    opt.step()
    after = _np(m.weight_orig)
    assert not np.array_equal(before, after)
    # The step moved the very parameter the pruned weight is derived from ...
    m(lucid.ones(2, 5))
    np.testing.assert_allclose(_np(m.weight), after * _np(m.weight_mask), rtol=1e-6)
    # ... and the mask is what keeps pruned entries at zero in what the layer uses.
    assert (_np(m.weight)[_np(m.weight_mask) == 0] == 0).all()


def test_a_frozen_weight_stays_frozen_and_unfreezing_works() -> None:
    m = nn.Linear(5, 6)
    m.weight.requires_grad_(False)
    prune.l1_unstructured(m, "weight", amount=0.5)
    assert not m.weight_orig.requires_grad and not m.weight.requires_grad
    prune.remove(m, "weight")
    assert not m.weight.requires_grad
    prune.l1_unstructured(m, "weight", amount=0.5)
    m.weight_orig.requires_grad_(True)
    m(lucid.ones(1, 5)).sum().backward()
    assert m.weight_orig.grad is not None


def test_remove_hands_back_the_same_parameter_with_the_mask_baked_in() -> None:
    m = nn.Linear(5, 6)
    weight = m.weight
    before = _np(weight).copy()
    prune.l1_unstructured(m, "weight", amount=10)
    mask = _np(m.weight_mask).copy()
    prune.remove(m, "weight")
    assert m.weight is weight and weight.requires_grad
    np.testing.assert_array_equal(_np(weight), before * mask)
    assert list(m._parameters) == ["bias", "weight"]
    assert not m._buffers and not m._forward_pre_hooks and "weight" not in m.__dict__


def test_remove_after_two_rounds_removes_the_whole_container() -> None:
    m = nn.Linear(5, 6)
    prune.l1_unstructured(m, "weight", amount=5)
    prune.l1_unstructured(m, "weight", amount=5)
    prune.remove(m, "weight")
    assert int((m.weight == 0).sum().item()) == 10
    assert not m._forward_pre_hooks
    assert not prune.is_pruned(m)


def test_each_parameter_of_a_module_prunes_on_its_own() -> None:
    m = nn.Linear(5, 6)
    prune.l1_unstructured(m, "weight", amount=5)
    prune.l1_unstructured(m, "bias", amount=2)
    assert len(m._forward_pre_hooks) == 2
    prune.remove(m, "weight")
    assert prune.is_pruned(m)
    prune.remove(m, "bias")
    assert not prune.is_pruned(m)


def test_is_pruned_looks_into_submodules() -> None:
    net = nn.Sequential(nn.Linear(4, 4), nn.Sequential(nn.ReLU(), nn.Linear(4, 2)))
    assert not prune.is_pruned(net)
    prune.l1_unstructured(net[1][1], "weight", amount=0.5)
    assert prune.is_pruned(net) and prune.is_pruned(net[1]) and not prune.is_pruned(net[0])


def test_a_failed_second_round_leaves_the_first_in_place() -> None:
    m = nn.Linear(5, 6)
    prune.l1_unstructured(m, "weight", amount=5)
    hook = next(iter(m._forward_pre_hooks.values()))
    mask = _np(m.weight_mask).copy()
    with pytest.raises(ValueError):
        prune.l1_unstructured(m, "weight", amount=26)
    assert next(iter(m._forward_pre_hooks.values())) is hook and len(m._forward_pre_hooks) == 1
    np.testing.assert_array_equal(_np(m.weight_mask), mask)
    prune.l1_unstructured(m, "weight", amount=1)
    assert int((m.weight_mask == 0).sum().item()) == 6


def test_the_pre_hook_is_the_method_object() -> None:
    m = nn.Linear(5, 6)
    prune.l1_unstructured(m, "weight", amount=0.5)
    (hook,) = m._forward_pre_hooks.values()
    assert isinstance(hook, prune.BasePruningMethod) and isinstance(hook, prune.L1Unstructured)
    assert hook._tensor_name == "weight" and hook.PRUNING_TYPE == "unstructured"
    prune.random_structured(m, "weight", amount=1, dim=0)
    (container,) = m._forward_pre_hooks.values()
    assert isinstance(container, prune.PruningContainer)
    assert [type(h) for h in container] == [prune.L1Unstructured, prune.RandomStructured]


# ── exact counts: the three ways the old mask missed ────────────────────


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("amount", [0, 1, 17, 0.5, 0.31, 1.0, 30, 12])
def test_l1_prunes_exactly_the_count_even_when_every_value_ties(device: str, amount) -> None:
    m = nn.Linear(5, 6).to(device)
    with lucid.no_grad():
        m.weight.copy_(lucid.full((6, 5), 0.25, device=device))
    prune.l1_unstructured(m, "weight", amount=amount)
    want = amount if isinstance(amount, int) else round(amount * 30)
    assert int((m.weight_mask == 0).sum().item()) == want


def test_l1_prunes_the_smallest_magnitudes_among_ties() -> None:
    m = nn.Linear(4, 3)
    w = np.array([[3, -1, 1, 5], [1, -1, 2, 7], [-1, 9, 1, 4]], dtype=np.float32)  # six |w| = 1
    with lucid.no_grad():
        m.weight.copy_(lucid.tensor(w))
    prune.l1_unstructured(m, "weight", amount=5)  # five of the six |w|=1 entries, no more
    pruned = _np(m.weight_mask) == 0
    assert pruned.sum() == 5 and (np.abs(w)[pruned] == 1).all()
    m2 = nn.Linear(4, 3)
    with lucid.no_grad():
        m2.weight.copy_(lucid.tensor(w))
    prune.l1_unstructured(m2, "weight", amount=7)  # all six, then the next smallest, |w|=2
    pruned = _np(m2.weight_mask) == 0
    assert pruned.sum() == 7 and np.abs(w)[pruned].max() == 2 and (np.abs(w)[~pruned] > 2).all()


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("amount", [0, 1, 13, 30, 0.5, 0.31, 1.0])
def test_random_unstructured_prunes_exactly_the_count(device: str, amount) -> None:
    for _ in range(3):
        m = nn.Linear(5, 6).to(device)
        prune.random_unstructured(m, "weight", amount=amount)
        want = amount if isinstance(amount, int) else round(amount * 30)
        assert int((m.weight_mask == 0).sum().item()) == want


def test_random_unstructured_draws_a_different_mask_each_time() -> None:
    masks = set()
    for _ in range(4):
        m = nn.Linear(5, 6)
        prune.random_unstructured(m, "weight", amount=15)
        masks.add(_np(m.weight_mask).tobytes())
    assert len(masks) > 1


@pytest.mark.parametrize("dim", [0, 1, -1])
@pytest.mark.parametrize("amount", [0, 1, 2, 0.5])
def test_random_structured_prunes_whole_channels(dim: int, amount) -> None:
    m = nn.Conv2d(3, 4, 2)
    prune.random_structured(m, "weight", amount=amount, dim=dim)
    mask = _np(m.weight_mask)
    axis = dim % 4
    others = tuple(a for a in range(4) if a != axis)
    dead = (mask.sum(axis=others) == 0).sum()
    want = amount if isinstance(amount, int) else round(amount * mask.shape[axis])
    assert dead == want
    assert set(np.unique(mask)) <= {0.0, 1.0}
    # A channel is all-or-nothing.
    per_channel = np.moveaxis(mask, axis, 0).reshape(mask.shape[axis], -1)
    assert ((per_channel.min(axis=1) == per_channel.max(axis=1))).all()


def test_structured_and_unstructured_rounds_respect_each_other() -> None:
    m = nn.Conv2d(3, 4, 2)
    prune.random_structured(m, "weight", amount=1, dim=0)
    dead_filter = np.flatnonzero(_np(m.weight_mask).reshape(4, -1).sum(axis=1) == 0)
    prune.random_unstructured(m, "weight", amount=10)
    mask = _np(m.weight_mask)
    assert (mask[dead_filter] == 0).all()
    assert int((mask == 0).sum()) == 12 + 10
    prune.random_structured(m, "weight", amount=1, dim=1)
    assert (_np(m.weight_mask)[dead_filter] == 0).all()


# ── the entry points and their refusals ─────────────────────────────────


def test_random_structured_and_ln_structured_check_the_axis() -> None:
    m = nn.Linear(5, 6)
    for dim in (2, -3):
        with pytest.raises(IndexError):
            prune.random_structured(m, "weight", amount=1, dim=dim)
        with pytest.raises(IndexError):
            prune.ln_structured(m, "weight", amount=1, n=2, dim=dim)
    assert not m._forward_pre_hooks and sorted(m._parameters) == ["bias", "weight"]


@pytest.mark.parametrize("bad", [True, False, None, "0.5", [1]])
def test_amount_of_the_wrong_type_is_a_type_error(bad) -> None:
    with pytest.raises(TypeError):
        prune.l1_unstructured(nn.Linear(3, 3), "weight", amount=bad)


def test_pruning_something_that_is_not_a_parameter_is_refused() -> None:
    m = nn.Linear(3, 3, bias=False)
    with pytest.raises(TypeError):
        prune.l1_unstructured(m, "bias", amount=1)
    with pytest.raises(AttributeError):
        prune.l1_unstructured(m, "nope", amount=1)
    with pytest.raises(TypeError):
        prune.l1_unstructured(object(), "weight", amount=1)


def test_importance_scores_of_the_wrong_shape_are_refused() -> None:
    m = nn.Linear(3, 3)
    with pytest.raises(ValueError):
        prune.l1_unstructured(m, "weight", amount=1, importance_scores=lucid.ones(2, 2))
    assert not m._forward_pre_hooks


def test_global_unstructured_refuses_a_structured_method_and_bad_containers() -> None:
    m = nn.Linear(3, 3)
    with pytest.raises(TypeError):
        prune.global_unstructured([(m, "weight")], prune.LnStructured, amount=1, n=2, dim=0)
    with pytest.raises(TypeError):
        prune.global_unstructured(3, prune.L1Unstructured, amount=1)
    with pytest.raises(TypeError):
        prune.global_unstructured([(m, "weight")], prune.L1Unstructured, [1], amount=1)
    assert not m._forward_pre_hooks


def test_global_unstructured_counts_across_modules() -> None:
    net = nn.Sequential(nn.Linear(10, 4), nn.Linear(4, 1))
    prune.global_unstructured(
        [(net[0], "weight"), (net[1], "weight")], prune.L1Unstructured, amount=10
    )
    assert sum(int((m.weight_mask == 0).sum().item()) for m in net) == 10


def test_container_methods_must_share_a_parameter() -> None:
    a, b = prune.L1Unstructured(1), prune.L1Unstructured(1)
    a._tensor_name, b._tensor_name = "weight", "bias"
    container = prune.PruningContainer(a)
    with pytest.raises(ValueError):
        container.add_pruning_method(b)
    with pytest.raises(TypeError):
        container.add_pruning_method(object())  # type: ignore[arg-type]


def test_custom_mask_shape_is_checked_before_the_module_changes() -> None:
    m = nn.Linear(3, 3)
    with pytest.raises(ValueError):
        prune.custom_from_mask(m, "weight", lucid.ones(2, 2))
    assert sorted(m._parameters) == ["bias", "weight"] and not m._buffers


def test_a_user_defined_method_plugs_in() -> None:
    class EveryOther(prune.BasePruningMethod):
        PRUNING_TYPE = "unstructured"

        def compute_mask(self, t, default_mask):
            flat = default_mask.clone().reshape([-1])
            flat[lucid.tensor([0, 2, 4])] = 0
            return flat.reshape(list(t.shape))

    m = nn.Linear(3, 3)
    EveryOther.apply(m, "weight")
    assert int((m.weight_mask == 0).sum().item()) == 3
    prune.l1_unstructured(m, "weight", amount=2)
    assert int((m.weight_mask == 0).sum().item()) == 5


# ── a pruned module is an ordinary module ───────────────────────────────


@pytest.mark.parametrize("rounds", [1, 2])
def test_a_pruned_module_pickles_and_copies(rounds: int) -> None:
    import copy

    m = nn.Linear(5, 6)
    for _ in range(rounds):
        prune.l1_unstructured(m, "weight", amount=5)
    x = lucid.ones(2, 5)
    want = _np(m(x))
    for clone in (pickle.loads(pickle.dumps(m)), copy.deepcopy(m)):
        assert prune.is_pruned(clone)
        np.testing.assert_allclose(_np(clone(x)), want, rtol=1e-6)
        assert clone.weight_orig is not m.weight_orig
    prune.remove(copy.deepcopy(m), "weight")


def test_a_pruned_module_saves_and_loads_its_mask() -> None:
    m = nn.Linear(5, 6)
    prune.l1_unstructured(m, "weight", amount=10)
    fresh = nn.Linear(5, 6)
    prune.identity(fresh, "weight")
    fresh.load_state_dict(m.state_dict())
    np.testing.assert_array_equal(_np(fresh.weight_mask), _np(m.weight_mask))
    x = lucid.ones(2, 5)
    np.testing.assert_allclose(_np(fresh(x)), _np(m(x)), rtol=1e-6)


@pytest.mark.parametrize("device", _DEVICES)
def test_the_masked_weight_is_recomputed_on_every_forward(device: str) -> None:
    m = nn.Linear(5, 6).to(device)
    prune.l1_unstructured(m, "weight", amount=0.5)
    with lucid.no_grad():
        m.weight_orig.copy_(m.weight_orig * 3.0)
    m(lucid.ones(1, 5, device=device))
    np.testing.assert_allclose(_np(m.weight), _np(m.weight_orig) * _np(m.weight_mask), rtol=1e-6)
    assert m.weight.device == m.weight_orig.device


def test_a_second_round_ranks_the_weights_as_they_are_now() -> None:
    m = nn.Linear(4, 3)
    prune.l1_unstructured(m, "weight", amount=2)
    # Train without a forward, then prune again: the ranking sees the new values.
    w = np.arange(1, 13, dtype=np.float32).reshape(3, 4)[::-1].copy()
    with lucid.no_grad():
        m.weight_orig.copy_(lucid.tensor(w))
    mask_before = _np(m.weight_mask).copy()
    prune.l1_unstructured(m, "weight", amount=3)
    mask = _np(m.weight_mask)
    alive = mask_before == 1
    ranked = np.argsort(np.where(alive, np.abs(w), np.inf).reshape(-1))[:3]
    assert (mask.reshape(-1)[ranked] == 0).all() and int((mask == 0).sum()) == 5
