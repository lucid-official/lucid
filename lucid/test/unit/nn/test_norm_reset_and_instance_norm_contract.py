"""The norm layers' shared state: resetting it, and InstanceNorm's contract (LCD-502).

The reference keeps BatchNorm's and InstanceNorm's state in one base, and
three things it does there Lucid did not:

* ``reset_running_stats()`` / ``reset_parameters()`` were missing.
* ``InstanceNorm(momentum=None, track_running_stats=True)`` raised a
  ``TypeError``.  The reference passes ``None`` on as ``0`` — the running
  statistics stay where they are — and counts no batches.
* A training InstanceNorm over one spatial element was accepted.  One
  element is its own mean, so the output is ``bias`` whatever the input;
  the reference refuses it with a ``ValueError``.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

_C = 3

# (class, constructor keywords, input shape).  Every norm whose state the
# shared base keeps, with that state switched on.
_TRACKING = {
    "batch_norm_1d": ("BatchNorm1d", {}, (4, _C, 5)),
    "batch_norm_2d": ("BatchNorm2d", {}, (4, _C, 2, 3)),
    "batch_norm_3d": ("BatchNorm3d", {}, (2, _C, 2, 2, 2)),
    "instance_norm_1d": ("InstanceNorm1d", {"affine": True}, (4, _C, 5)),
    "instance_norm_2d": ("InstanceNorm2d", {"affine": True}, (4, _C, 2, 3)),
    "instance_norm_3d": ("InstanceNorm3d", {"affine": True}, (2, _C, 2, 2, 2)),
}


def _make(lib, name, device=None):
    cls, kw, _ = _TRACKING[name]
    kw = {**kw, "track_running_stats": True}
    if lib is lucid:
        return getattr(nn, cls)(_C, device=device, **kw)
    return getattr(lib.nn, cls)(_C, **kw)


def _input(shape, step=0):
    rng = np.random.default_rng(step)
    return (rng.standard_normal(shape) * (1 + step) + step).astype(np.float32)


def _host(t):
    return np.asarray(t.detach().cpu().numpy(), dtype=np.float64)


def _train(m, shape, to_tensor):
    m.train()
    for step in range(3):
        m(to_tensor(_input(shape, step)))
    for p in m.parameters():
        p.data.fill_(3.0)


# ── reset_running_stats / reset_parameters ───────────────────────────────────


@pytest.mark.parametrize("name", list(_TRACKING))
def test_reset_running_stats_writes_a_fresh_layers_values_in_place(name, device):
    m = _make(lucid, name, device)
    _train(m, _TRACKING[name][2], lambda a: lucid.tensor(a, device=device))
    held = dict(m.named_buffers())
    m.reset_running_stats()
    assert dict(m.named_buffers()) == held  # the same objects, not new ones
    assert _host(held["running_mean"]).tolist() == [0.0] * _C
    assert _host(held["running_var"]).tolist() == [1.0] * _C
    assert int(held["num_batches_tracked"].item()) == 0
    # The parameters are not running statistics.
    assert _host(m.weight).tolist() == [3.0] * _C


@pytest.mark.parametrize("name", list(_TRACKING))
def test_reset_parameters_also_resets_weight_and_bias(name, device):
    m = _make(lucid, name, device)
    _train(m, _TRACKING[name][2], lambda a: lucid.tensor(a, device=device))
    weight = m.weight
    m.reset_parameters()
    assert m.weight is weight
    assert _host(m.weight).tolist() == [1.0] * _C
    assert _host(m.bias).tolist() == [0.0] * _C
    assert _host(m.running_mean).tolist() == [0.0] * _C


@pytest.mark.parametrize(
    "build",
    [
        lambda: nn.BatchNorm1d(_C, affine=False, track_running_stats=False),
        lambda: nn.InstanceNorm1d(_C),
        lambda: nn.LazyBatchNorm1d(),
        lambda: nn.LazyInstanceNorm1d(affine=True, track_running_stats=True),
    ],
    ids=["no-state", "instance-norm-default", "lazy", "lazy-instance"],
)
def test_a_layer_with_nothing_to_reset_resets_nothing(build):
    m = build()
    m.reset_running_stats()
    m.reset_parameters()
    # A lazy layer is still pending, and sizes itself on its first input.
    assert m(lucid.ones(4, _C, 5)).shape == (4, _C, 5)


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_TRACKING))
@pytest.mark.parametrize("reset", ["reset_running_stats", "reset_parameters"])
def test_a_reset_leaves_the_reference_state(ref, name, reset):
    shape = _TRACKING[name][2]
    ours, theirs = _make(lucid, name), _make(ref, name)
    _train(ours, shape, lucid.tensor)
    _train(theirs, shape, ref.tensor)
    getattr(ours, reset)()
    getattr(theirs, reset)()
    expected = theirs.state_dict()
    got = ours.state_dict()
    assert list(got) == list(expected)
    for key, value in expected.items():
        np.testing.assert_allclose(_host(got[key]), value.double().numpy(), err_msg=key)


# ── InstanceNorm(momentum=None) ──────────────────────────────────────────────


@pytest.mark.parametrize("unbatched", [False, True], ids=["batched", "unbatched"])
def test_instance_norm_momentum_none_leaves_the_running_statistics(device, unbatched):
    m = nn.InstanceNorm1d(_C, momentum=None, track_running_stats=True, device=device)
    shape = (_C, 5) if unbatched else (4, _C, 5)
    for step in range(2):
        m(lucid.tensor(_input(shape, step), device=device))
    assert _host(m.running_mean).tolist() == [0.0] * _C
    assert _host(m.running_var).tolist() == [1.0] * _C
    assert int(m.num_batches_tracked.item()) == 0
    m.eval()
    assert m(lucid.tensor(_input(shape), device=device)).shape == shape


def test_lazy_instance_norm_takes_momentum_none():
    m = nn.LazyInstanceNorm2d(momentum=None, track_running_stats=True)
    m(lucid.tensor(_input((2, _C, 2, 3))))
    assert m.momentum is None
    assert _host(m.running_mean).tolist() == [0.0] * _C


@pytest.mark.parity
@pytest.mark.parametrize("momentum", [None, 0.1, 0.5])
def test_instance_norm_running_statistics_match_the_reference(ref, momentum):
    ours = nn.InstanceNorm1d(_C, momentum=momentum, track_running_stats=True)
    theirs = ref.nn.InstanceNorm1d(_C, momentum=momentum, track_running_stats=True)
    for step in range(3):
        x = _input((4, _C, 5), step)
        np.testing.assert_allclose(
            _host(ours(lucid.tensor(x))),
            theirs(ref.tensor(x)).detach().numpy(),
            rtol=1e-4,
            atol=1e-5,
        )
    for key, value in theirs.state_dict().items():
        np.testing.assert_allclose(
            _host(ours.state_dict()[key]),
            value.double().numpy(),
            rtol=1e-5,
            err_msg=key,
        )


# ── one spatial element ──────────────────────────────────────────────────────

# (case, module or None for the functional, input shape, refused).
_ONE_ELEMENT = [
    ("train-batched", lambda: nn.InstanceNorm1d(_C), (4, _C, 1), True),
    ("train-unbatched", lambda: nn.InstanceNorm1d(_C), (_C, 1), True),
    ("train-2d", lambda: nn.InstanceNorm2d(_C), (2, _C, 1, 1), True),
    ("train-3d", lambda: nn.InstanceNorm3d(_C), (2, _C, 1, 1, 1), True),
    ("eval-untracked", lambda: nn.InstanceNorm1d(_C).eval(), (4, _C, 1), True),
    (
        "eval-tracked",
        lambda: nn.InstanceNorm1d(_C, track_running_stats=True).eval(),
        (4, _C, 1),
        False,
    ),
    ("lazy-train", lambda: nn.LazyInstanceNorm1d(), (4, _C, 1), True),
    ("two-elements", lambda: nn.InstanceNorm1d(_C), (4, _C, 2), False),
    ("no-elements", lambda: nn.InstanceNorm1d(_C), (4, _C, 0), False),
]
_ONE_IDS = [c[0] for c in _ONE_ELEMENT]


# Zero elements have no variance either, and say so; only one is refused.
@pytest.mark.filterwarnings("ignore:var\\(\\)")
@pytest.mark.parametrize("case,build,shape,refused", _ONE_ELEMENT, ids=_ONE_IDS)
def test_one_spatial_element_is_refused_when_normalised_by_itself(
    case, build, shape, refused, device
):
    del case
    m = build().to(device)
    x = lucid.ones(*shape, device=device)
    if refused:
        with pytest.raises(ValueError, match="more than 1 spatial element"):
            m(x)
    else:
        assert m(x).shape == shape


def test_the_functional_owns_the_check(device):
    x = lucid.ones(4, _C, 1, device=device)
    with pytest.raises(ValueError, match=r"got input size \(4, 3, 1\)"):
        F.instance_norm(x)
    rm, rv = lucid.zeros(_C, device=device), lucid.ones(_C, device=device)
    assert F.instance_norm(x, rm, rv, use_input_stats=False).shape == (4, _C, 1)
    for mean_arg, var_arg in [(None, None), (rm, None), (None, rv)]:
        with pytest.raises(
            RuntimeError,
            match="Expected running_mean and running_var to be defined when use_input_stats is false",
        ):
            F.instance_norm(x, mean_arg, var_arg, use_input_stats=False)


@pytest.mark.parity
@pytest.mark.parametrize(
    "cls_name,shape",
    [
        ("LazyInstanceNorm1d", (2, _C, 4)),
        ("LazyInstanceNorm2d", (2, _C, 3, 4)),
        ("LazyInstanceNorm3d", (2, _C, 2, 3, 4)),
    ],
)
def test_lazy_instance_norm_defaults_match_reference(ref, cls_name, shape, device):
    ours = getattr(nn, cls_name)().to(device)
    theirs = getattr(ref.nn, cls_name)()
    assert (ours.affine, ours.track_running_stats) == (True, True)
    assert (ours.affine, ours.track_running_stats) == (
        theirs.affine,
        theirs.track_running_stats,
    )
    x_np = _input(shape)
    out = ours(lucid.tensor(x_np, device=device))
    ref_out = theirs(ref.tensor(x_np))
    np.testing.assert_allclose(_host(out), ref_out.detach().numpy(), rtol=1e-5, atol=1e-5)
    assert set(ours.state_dict()) == set(theirs.state_dict())


@pytest.mark.parity
def test_instance_norm_refuses_missing_buffers_when_use_input_stats_is_false(ref):
    x_ours, x_ref = lucid.ones(2, _C, 4), ref.ones(2, _C, 4)
    rm_ours, rv_ours = lucid.zeros(_C), lucid.ones(_C)
    rm_ref, rv_ref = ref.zeros(_C), ref.ones(_C)
    msg = "Expected running_mean and running_var to be defined when use_input_stats is false"
    for (m_o, v_o), (m_r, v_r) in [
        ((None, None), (None, None)),
        ((rm_ours, None), (rm_ref, None)),
        ((None, rv_ours), (None, rv_ref)),
    ]:
        with pytest.raises(RuntimeError, match=msg):
            F.instance_norm(x_ours, m_o, v_o, use_input_stats=False)
        with pytest.raises(RuntimeError, match=msg):
            ref.nn.functional.instance_norm(x_ref, m_r, v_r, use_input_stats=False)


@pytest.mark.parity
@pytest.mark.filterwarnings("ignore:var\\(\\)")
@pytest.mark.parametrize("case,build,shape,refused", _ONE_ELEMENT, ids=_ONE_IDS)
def test_the_reference_refuses_the_same_inputs(ref, case, build, shape, refused):
    ours = build()
    cls = type(ours).__name__
    kw = {"track_running_stats": ours.track_running_stats}
    theirs = getattr(ref.nn, cls)(**kw) if cls.startswith("Lazy") else None
    if theirs is None:
        theirs = getattr(ref.nn, cls)(_C, **kw)
    theirs.train(ours.training)
    x = ref.ones(*shape)
    if refused:
        with pytest.raises(ValueError, match="more than 1 spatial element"):
            theirs(x)
    else:
        assert tuple(theirs(x).shape) == shape, case
