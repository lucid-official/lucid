"""A lazy layer moved before its first input builds its weights where it was moved (LCD-502).

``nn.LazyLinear(3).to('metal')(x_metal)`` raised ``DeviceMismatch``:
``Module.to`` moved the zero-element placeholders, but the layer built its
real weights from the device and dtype it was *constructed* with, so they
landed on the CPU.  The reference's lazy layers materialise on the device
and dtype their placeholders were last converted to.  The record now lives
in one place every lazy class inherits (``nn/modules/_lazy.py``), and the
checkpoint path that sizes a layer reads it too.

The checkpoint path of ``LazyLinear`` also assigned a new ``weight``
instead of filling the placeholder, so an optimizer built before
``load_state_dict`` held an object the layer no longer had.
"""

import warnings

import numpy as np
import pytest

import lucid
import lucid.nn as nn

# (class, constructor keywords, input shape) — all thirteen lazy classes,
# with every parameter and buffer they can have switched on.
_LAZY = {
    "LazyLinear": (dict(out_features=3), (4, 6)),
    "LazyConv1d": (dict(out_channels=4, kernel_size=3), (2, 3, 8)),
    "LazyConv2d": (dict(out_channels=4, kernel_size=3), (2, 3, 6, 6)),
    "LazyConv3d": (dict(out_channels=4, kernel_size=3), (2, 3, 4, 4, 4)),
    "LazyConvTranspose1d": (dict(out_channels=4, kernel_size=3), (2, 3, 8)),
    "LazyConvTranspose2d": (dict(out_channels=4, kernel_size=3), (2, 3, 6, 6)),
    "LazyConvTranspose3d": (dict(out_channels=4, kernel_size=3), (2, 3, 4, 4, 4)),
    "LazyBatchNorm1d": ({}, (4, 6, 8)),
    "LazyBatchNorm2d": ({}, (4, 6, 5, 5)),
    "LazyBatchNorm3d": ({}, (4, 6, 3, 3, 3)),
    "LazyInstanceNorm1d": (dict(affine=True, track_running_stats=True), (4, 6, 8)),
    "LazyInstanceNorm2d": (dict(affine=True, track_running_stats=True), (4, 6, 5, 5)),
    "LazyInstanceNorm3d": (dict(affine=True, track_running_stats=True), (4, 6, 3, 3, 3)),
}
_NAMES = list(_LAZY)


def _build(lib, name, **extra):
    kw, _ = _LAZY[name]
    return getattr(lib.nn, name)(**kw, **extra)


def _input(name, lib=lucid, **to):
    shape = _LAZY[name][1]
    data = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    return lib.tensor(data).to(**to)


def _state(module):
    return [*module.named_parameters(), *module.named_buffers()]


def _float_state(module):
    return [(k, t) for k, t in _state(module) if k != "num_batches_tracked"]


@pytest.fixture(autouse=True)
def _quiet_placeholder_reads():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", nn.UninitializedParameterWarning)
        yield


@pytest.mark.parametrize("name", _NAMES)
def test_moved_before_the_first_input_it_materialises_there(name, device):
    layer = _build(lucid, name).to(device)
    out = layer(_input(name, device=device))
    assert out.device == lucid.device(device)
    assert {str(t.device) for _, t in _state(layer)} == {str(lucid.device(device))}


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize(
    "convert",
    [lambda m: m.double(), lambda m: m.to(lucid.float64), lambda m: m.type(lucid.float64)],
    ids=["double", "to-dtype", "type"],
)
def test_cast_before_the_first_input_it_materialises_in_that_dtype(name, convert):
    layer = convert(_build(lucid, name))
    out = layer(_input(name, dtype=lucid.float64))
    assert out.dtype == lucid.float64
    assert {t.dtype for _, t in _float_state(layer)} == {lucid.float64}
    counter = layer._buffers.get("num_batches_tracked")
    assert counter is None or counter.dtype == lucid.int64


@pytest.mark.parametrize("name", _NAMES)
def test_its_placeholders_sit_where_it_was_made(name, device):
    layer = _build(lucid, name, device=device)
    params = [p for _, p in layer.named_parameters()]
    assert params and {str(p.device) for p in params} == {str(lucid.device(device))}


@pytest.mark.parametrize("name", ["LazyLinear", "LazyConv2d", "LazyBatchNorm2d"])
def test_the_latest_move_wins(name, cross_device_pair):
    first, second = cross_device_pair
    layer = _build(lucid, name).to(first).to(second)
    layer(_input(name, device=second))
    assert {str(t.device) for _, t in _state(layer)} == {str(lucid.device(second))}


@pytest.mark.parametrize("name", _NAMES)
def test_a_checkpoint_loaded_after_the_move_lands_there(name, device):
    source = _build(lucid, name)
    source(_input(name))
    layer = _build(lucid, name).to(device)
    layer.load_state_dict(source.state_dict())
    assert not layer.has_uninitialized_parameters()
    assert {str(t.device) for _, t in _state(layer)} == {str(lucid.device(device))}
    source.eval()
    layer.eval()
    np.testing.assert_allclose(
        layer(_input(name, device=device)).cpu().numpy(),
        source(_input(name)).numpy(),
        rtol=1e-4,
        atol=1e-5,
    )


@pytest.mark.parametrize("name", _NAMES)
def test_a_checkpoint_fills_the_placeholders_an_optimizer_holds(name):
    source = _build(lucid, name)
    source(_input(name))
    layer = _build(lucid, name)
    early = [id(p) for p in layer.parameters()]
    layer.load_state_dict(source.state_dict())
    assert [id(p) for p in layer.parameters()] == early


@pytest.mark.parity
@pytest.mark.parametrize("name", _NAMES)
def test_the_dtype_it_materialises_in_is_the_references(ref, name):
    ours = _build(lucid, name).double()
    theirs = _build(ref, name).double()
    ours(_input(name, dtype=lucid.float64))
    theirs(_input(name, ref, dtype=ref.float64))

    def dtypes(module):
        return {k: str(t.dtype).split(".")[-1] for k, t in _state(module)}

    assert dtypes(ours) == dtypes(theirs)


@pytest.mark.parity
@pytest.mark.parametrize("name", _NAMES)
def test_the_reference_also_materialises_where_it_was_moved(ref, name):
    if not ref.backends.mps.is_available():
        pytest.skip("the reference has no Apple GPU device here")
    theirs = _build(ref, name).to("mps")
    theirs(_input(name, ref, device="mps"))
    assert {t.device.type for _, t in _state(theirs)} == {"mps"}
    ours = _build(lucid, name).to("metal")
    ours(_input(name, device="metal"))
    assert {str(t.device) for _, t in _state(ours)} == {str(lucid.device("metal"))}
