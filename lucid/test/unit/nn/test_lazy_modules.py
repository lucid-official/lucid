"""The lazy layers, and the way they silently do not train.

``nn/modules/conv.py`` sat at 52.9% and almost all of the dark part was
the lazy family — thirteen classes across three files that infer their
input size from the first forward.

A lazy layer that gets its shape wrong fails loudly, so that is not the
risk.  The risk is the parameters not existing yet: read
``model.parameters()`` before the first forward and the list is missing
them, an optimiser built from it never touches them, and the loss still
goes down because every other layer is learning.  No error, no shape
complaint, and a model that simply trains worse than it should.

So the shape and gradient checks here are the easy half.  The half worth
having is at the bottom.
"""

import warnings

import numpy as np
import pytest

import lucid
import lucid.nn as nn

CONVS = [
    ("LazyConv1d", dict(out_channels=8, kernel_size=3), (2, 3, 16)),
    ("LazyConv2d", dict(out_channels=8, kernel_size=3), (2, 3, 8, 8)),
    ("LazyConv3d", dict(out_channels=8, kernel_size=3), (2, 3, 4, 4, 4)),
    ("LazyConvTranspose1d", dict(out_channels=8, kernel_size=3), (2, 3, 16)),
    ("LazyConvTranspose2d", dict(out_channels=8, kernel_size=3), (2, 3, 8, 8)),
    ("LazyConvTranspose3d", dict(out_channels=8, kernel_size=3), (2, 3, 4, 4, 4)),
]

NORMS = [
    ("LazyBatchNorm1d", (4, 6, 8)),
    ("LazyBatchNorm2d", (4, 6, 5, 5)),
    ("LazyBatchNorm3d", (4, 6, 3, 3, 3)),
    ("LazyInstanceNorm1d", (4, 6, 8)),
    ("LazyInstanceNorm2d", (4, 6, 5, 5)),
    ("LazyInstanceNorm3d", (4, 6, 3, 3, 3)),
]

ALL_LAZY = (
    [(n, kw, s) for n, kw, s in CONVS]
    + [(n, {}, s) for n, s in NORMS]
    + [("LazyLinear", dict(out_features=3), (4, 6))]
)
IDS = [c[0] for c in ALL_LAZY]


def _v(x):
    return np.asarray(x.numpy())


def _x(shape):
    return lucid.tensor(np.ones(shape, dtype=np.float32))


def _noisy(shape):
    return lucid.tensor(np.random.default_rng(len(shape)).standard_normal(shape).astype(np.float32))


def _params(module):
    """``parameters()`` without the warning, for the checks that are not
    about the warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", nn.UninitializedParameterWarning)
        return list(module.parameters())


# ── inferring the input size ──────────────────────────────────────────────────


@pytest.mark.parametrize("name,kw,shape", ALL_LAZY, ids=IDS)
def test_a_lazy_layer_infers_its_input_size(name, kw, shape):
    layer = getattr(nn, name)(**kw)
    out = layer(_noisy(shape))
    assert np.isfinite(_v(out)).all()
    assert not layer.has_uninitialized_parameters()


@pytest.mark.parametrize("name,kw,shape", CONVS, ids=[c[0] for c in CONVS])
def test_the_inferred_layer_matches_the_eager_one(name, kw, shape):
    """Same weights shape, same output shape as writing ``in_channels``
    out by hand — otherwise the inference is doing something else."""
    lazy = getattr(nn, name)(**kw)
    lazy(_x(shape))
    eager = getattr(nn, name.replace("Lazy", ""))(in_channels=shape[1], **kw)
    assert tuple(lazy.weight.shape) == tuple(eager.weight.shape)
    assert _v(lazy(_x(shape))).shape == _v(eager(_x(shape))).shape


def test_lazy_linear_infers_in_features():
    layer = nn.LazyLinear(out_features=3)
    assert _v(layer(_x((4, 6)))).shape == (4, 3)
    assert tuple(layer.weight.shape) == (3, 6)


@pytest.mark.parametrize("name,shape", NORMS, ids=[c[0] for c in NORMS])
def test_a_lazy_norm_infers_num_features(name, shape):
    layer = getattr(nn, name)(affine=True)
    out = layer(_noisy(shape))
    assert _v(out).shape == shape
    eager = getattr(nn, name.replace("Lazy", ""))(shape[1], affine=True)
    assert tuple(layer.weight.shape) == tuple(eager.weight.shape)


def test_an_instance_norm_without_affine_has_nothing_to_infer():
    """With ``affine=False, track_running_stats=False``, ``weight`` stays
    ``None`` — and that is not the same thing as being uninitialised."""
    layer = nn.LazyInstanceNorm2d(affine=False, track_running_stats=False)
    assert layer.weight is None
    layer(_noisy((4, 6, 5, 5)))
    assert layer.weight is None
    assert not layer.has_uninitialized_parameters()


@pytest.mark.parametrize("name,kw,shape", ALL_LAZY, ids=IDS)
def test_a_second_forward_does_not_re_initialise(name, kw, shape):
    layer = getattr(nn, name)(**kw)
    first = _v(layer(_noisy(shape)))
    assert np.allclose(first, _v(layer(_noisy(shape))))


@pytest.mark.parametrize("name,kw,shape", CONVS, ids=[c[0] for c in CONVS])
def test_a_different_input_size_afterwards_is_refused(name, kw, shape):
    layer = getattr(nn, name)(**kw)
    layer(_x(shape))
    wider = (shape[0], shape[1] + 2, *shape[2:])
    with pytest.raises(Exception):
        layer(_x(wider))


@pytest.mark.parametrize("name,kw,shape", ALL_LAZY, ids=IDS)
def test_gradients_reach_every_inferred_parameter(name, kw, shape):
    layer = getattr(nn, name)(**kw)
    out = layer(_noisy(shape))
    if not any(p.requires_grad for p in layer.parameters()):
        # InstanceNorm defaults to affine=False: nothing to train, so the
        # output carries no graph and backward() refuses it.
        assert not out.requires_grad
        return
    out.sum().backward()
    for param_name, param in layer.named_parameters():
        assert param.grad is not None, param_name


def test_chained_lazy_layers_resolve_in_one_pass():
    model = nn.Sequential(
        nn.LazyConv2d(out_channels=4, kernel_size=3),
        nn.ReLU(),
        nn.LazyConv2d(out_channels=2, kernel_size=3),
    )
    assert _v(model(_x((4, 3, 8, 8)))).shape == (4, 2, 4, 4)


# ── checkpoints ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name,kw,shape", ALL_LAZY, ids=IDS)
def test_a_checkpoint_materialises_an_uninitialised_layer(name, kw, shape):
    """Loading into a lazy layer has to work without a forward pass first,
    or a checkpoint can only be restored by first inventing an input."""
    source = getattr(nn, name)(**kw)
    x = _noisy(shape)
    source(x)

    target = getattr(nn, name)(**kw)
    assert target.has_uninitialized_parameters()
    checkpoint = source.state_dict()
    target.load_state_dict(checkpoint)

    if checkpoint:
        assert not target.has_uninitialized_parameters()
    else:
        # ``InstanceNorm`` defaults to no affine and no running stats, so
        # there is nothing in the checkpoint to materialise *from*.  The
        # layer stays pending and infers on the next forward, which is the
        # only thing it could correctly do.
        assert target.has_uninitialized_parameters()
    assert np.allclose(_v(source(x)), _v(target(x)), atol=1e-5)


@pytest.mark.parametrize(
    "kw",
    [
        dict(affine=True),
        dict(track_running_stats=True),
        dict(affine=True, track_running_stats=True),
    ],
    ids=["affine", "running-stats", "both"],
)
def test_a_stateful_instance_norm_materialises_from_its_checkpoint(kw):
    """The configurations that do put something in the checkpoint."""
    x = _noisy((4, 6, 5, 5))
    source = nn.LazyInstanceNorm2d(**kw)
    source(x)
    target = nn.LazyInstanceNorm2d(**kw)
    target.load_state_dict(source.state_dict())
    assert not target.has_uninitialized_parameters()
    assert np.allclose(_v(source(x)), _v(target(x)), atol=1e-5)


def test_state_dict_before_the_first_forward_is_empty_rather_than_wrong():
    """The parameters exist as placeholders, but a placeholder has no
    values to save — and an entry holding its zero-element buffer would
    look like a real one and restore to shape ``(0,)``."""
    layer = nn.LazyConv2d(out_channels=4, kernel_size=3)
    assert len(_params(layer)) == 2  # the objects are there ...
    assert list(layer.state_dict()) == []  # ... and the checkpoint is not


def test_repr_works_before_the_first_forward():
    assert "Lazy" in repr(nn.LazyConv2d(out_channels=4, kernel_size=3))


# ── the part that matters ─────────────────────────────────────────────────────


def test_reading_parameters_too_early_says_so():
    """What is left to warn about, now that the training defect is gone.

    The objects are real and survive materialisation, so an optimizer is
    fine.  Their *shapes* are not: they are ``(0,)`` until the first
    forward, so counting elements, reading ``.shape``, or flattening
    them into a vector reads nothing.  The suite promotes the warning to
    an error (``filterwarnings`` in ``pyproject.toml``), so the gate is
    strict where a running program is merely told.
    """
    model = nn.Sequential(nn.LazyConv2d(out_channels=4, kernel_size=3))
    with pytest.warns(nn.UninitializedParameterWarning, match="reading nothing"):
        list(model.parameters())


def test_the_warning_stops_once_the_shapes_are_known():
    model = nn.Sequential(nn.LazyConv2d(out_channels=4, kernel_size=3))
    with warnings.catch_warnings():
        warnings.simplefilter("error", nn.UninitializedParameterWarning)
        model(_x((2, 3, 8, 8)))
        list(model.parameters())  # must not raise


@pytest.mark.parametrize(
    "build",
    [
        lambda: nn.Linear(4, 4),
        lambda: nn.Linear(4, 4, bias=False),
        lambda: nn.Conv2d(3, 4, 3),
        lambda: nn.InstanceNorm2d(6),  # affine=False: no parameters, ever
        lambda: nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU()),
    ],
    ids=["linear", "linear-no-bias", "conv", "instancenorm-no-affine", "sequential"],
)
def test_an_eager_module_never_warns(build):
    """A parameter that is legitimately absent — no bias, no affine — is
    not an uninitialised one, and a gate that confused the two would be
    turned off within a week."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", nn.UninitializedParameterWarning)
        list(build().parameters())


def test_a_lazy_layer_mixed_with_eager_ones_contributes_its_placeholders():
    """The realistic case, and the one an empty-list check misses.

    The list used to hold only the eager layer's weights, so the
    optimiser received something perfectly plausible that happened to be
    missing exactly the lazy layer.  Now every layer is represented.
    """
    model = nn.Sequential(
        nn.LazyConv2d(out_channels=4, kernel_size=3), nn.Flatten(), nn.Linear(144, 2)
    )
    with pytest.warns(nn.UninitializedParameterWarning):
        params = list(model.parameters())
    assert len(params) == 4  # the conv's two and the Linear's two


def test_an_optimiser_over_nothing_is_refused():
    """It would step nothing and report no error.  The reference refuses
    it for the same reason."""
    with pytest.raises(ValueError, match="empty parameter list"):
        lucid.optim.SGD([], lr=0.1)


def test_an_optimiser_built_before_the_first_forward_still_trains():
    """What (b) is for, and what used to be impossible.

    Placeholder parameters mean the list is not empty; the optimizer's
    deferred engine binding means the impls it eventually steps are the
    materialized ones.  Either half alone leaves the layer frozen.
    """
    model = nn.Sequential(nn.LazyConv2d(out_channels=4, kernel_size=3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", nn.UninitializedParameterWarning)
        optimiser = lucid.optim.SGD(model.parameters(), lr=0.5)

    x = _noisy((4, 3, 8, 8))
    model(x)  # the parameters take their shape here, after the optimizer exists
    before = _v(model[0].weight).copy()
    for _ in range(3):
        optimiser.zero_grad()
        (model(x) ** 2).mean().backward()
        optimiser.step()
    assert not np.allclose(before, _v(model[0].weight))


def test_the_placeholder_objects_are_the_ones_that_get_filled():
    """Identity, not equality — a list captured early has to name the
    same objects afterwards or nothing holding it benefits."""
    model = nn.Sequential(nn.LazyConv2d(out_channels=4, kernel_size=3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", nn.UninitializedParameterWarning)
        early = list(model.parameters())
    assert [tuple(p.shape) for p in early] == [(0,), (0,)]
    model(_noisy((4, 3, 8, 8)))
    assert [id(p) for p in early] == [id(p) for p in model.parameters()]
    assert [tuple(p.shape) for p in early] == [(4, 3, 3, 3), (4,)]


def test_a_dry_run_first_makes_the_layer_train():
    """The documented fix, asserted to actually work rather than assumed."""
    model = nn.Sequential(nn.LazyConv2d(out_channels=4, kernel_size=3))
    x = _noisy((4, 3, 8, 8))
    model(x)  # the dry run
    optimiser = lucid.optim.SGD(model.parameters(), lr=0.5)
    before = _v(model[0].weight).copy()
    for _ in range(3):
        optimiser.zero_grad()
        (model(x) ** 2).mean().backward()
        optimiser.step()
    assert not np.allclose(before, _v(model[0].weight))


def test_has_uninitialized_parameters_sees_through_a_container():
    lazy = nn.Sequential(nn.Conv2d(3, 4, 3), nn.LazyConv2d(out_channels=2, kernel_size=3))
    eager = nn.Sequential(nn.Conv2d(3, 4, 3), nn.Conv2d(4, 2, 3))
    assert lazy.has_uninitialized_parameters()
    assert not eager.has_uninitialized_parameters()
    assert not lazy.has_uninitialized_parameters(recurse=False)


def test_zero_grad_still_works_on_an_uninitialised_model():
    """Why this warns rather than raises: ``zero_grad`` and
    ``requires_grad_`` legitimately walk a tree that has no parameters
    yet, and turning those into errors would be a worse trade."""
    model = nn.Sequential(nn.LazyConv2d(out_channels=4, kernel_size=3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", nn.UninitializedParameterWarning)
        model.zero_grad()
        model.requires_grad_(False)
