"""``nn.Module.type`` casts every parameter and buffer with ``Tensor.type``.

It handed its argument to ``Module.to``, which reads a string as a device:
``Linear.type("lucid.DoubleTensor")`` raised "Unknown device", and the
integer buffers ``to`` spares (``num_batches_tracked``) kept their dtype
where the reference casts them too (LCD-499).  The reference's
``Module.type(dst)`` is ``_apply(lambda t: t.type(dst))``: what
``Tensor.type`` takes, on every tensor.
"""

import pytest

import lucid
import lucid.nn as nn
from lucid.test._fixtures.devices import devices_supporting


def _dtypes(module):
    return {name: str(t.dtype).split(".")[-1] for name, t in module.state_dict().items()}


def _model(lib):
    return lib.nn.Sequential(lib.nn.Linear(2, 2), lib.nn.BatchNorm1d(2))


# (id, target spelled for a library, the dtype it names).  A legacy name is
# written with the library's own prefix.
_TARGETS = [
    ("float16", lambda lib: lib.float16, lucid.float16),
    ("half name", lambda lib: f"{lib.__name__}.HalfTensor", lucid.float16),
    ("complex64", lambda lib: lib.complex64, lucid.complex64),
    ("double name", lambda lib: f"{lib.__name__}.DoubleTensor", lucid.float64),
]
_ON_DEVICES = [
    pytest.param(device, target, id=f"{device}-{case}")
    for case, target, dtype in _TARGETS
    for device in devices_supporting(dtype)
]


@pytest.mark.parametrize("device,target", _ON_DEVICES)
def test_every_tensor_is_cast_and_parameters_are_kept(device, target):
    model = _model(lucid).to(device)
    params = list(model.parameters())
    assert model.type(target(lucid)) is model
    assert list(model.parameters()) == params
    assert all(type(p) is nn.Parameter and p.requires_grad for p in params)
    state = model.state_dict().values()
    assert len({t.dtype for t in state}) == 1, _dtypes(model)
    assert all(t.device == params[0].device for t in state)


@pytest.mark.parity
@pytest.mark.parametrize("target", [t for _, t, _ in _TARGETS], ids=[c for c, _, _ in _TARGETS])
def test_the_dtypes_match_the_reference(ref, target):
    ours, theirs = _model(lucid), _model(ref)
    ours.type(target(lucid))
    theirs.type(target(ref))
    assert _dtypes(ours) == _dtypes(theirs)


def test_a_name_moves_the_module_to_its_device(cross_device_pair):
    # A legacy name carries a device, as the reference's do (LCD-504).
    cpu, metal = cross_device_pair
    model = _model(lucid).type("lucid.metal.HalfTensor")
    assert {t.device for t in model.state_dict().values()} == {lucid.device(metal)}
    model.type("lucid.FloatTensor")
    assert {t.device for t in model.state_dict().values()} == {lucid.device(cpu)}
    model.to(metal).type(lucid.float16)
    assert {t.device for t in model.state_dict().values()} == {lucid.device(metal)}


def test_a_gradient_is_cast_with_its_parameter(device):
    layer = nn.Linear(2, 2).to(device)
    layer(lucid.ones(1, 2, device=device)).sum().backward()
    layer.type(lucid.float16)
    assert layer.weight.grad.dtype == lucid.float16


def test_an_integer_type_is_refused_while_a_parameter_trains():
    model = _model(lucid)
    before = _dtypes(model)
    with pytest.raises(RuntimeError, match="requires grad"):
        model.type(lucid.int64)
    # Refused before anything is cast; the reference fails on the first
    # parameter it reaches, here before it too has cast anything.
    assert _dtypes(model) == before


@pytest.mark.parity
def test_the_reference_refuses_it_too(ref):
    with pytest.raises(RuntimeError, match="floating point or complex"):
        _model(ref).type(ref.int64)


def _trains_nothing(lib, frozen):
    if frozen:
        return lib.nn.Linear(2, 2).requires_grad_(False)
    module = lib.nn.Module()
    module.register_buffer("b", lib.ones(2))
    return module


@pytest.mark.parametrize("frozen", [True, False], ids=["frozen", "buffers only"])
def test_an_integer_type_casts_a_module_that_trains_nothing(frozen):
    module = _trains_nothing(lucid, frozen)
    module.type(lucid.int64)
    assert set(_dtypes(module).values()) == {"int64"}


@pytest.mark.parity
@pytest.mark.parametrize("frozen", [True, False], ids=["frozen", "buffers only"])
def test_the_reference_casts_it_too(ref, frozen):
    ours, theirs = _trains_nothing(lucid, frozen), _trains_nothing(ref, frozen)
    ours.type(lucid.int64)
    theirs.type(ref.int64)
    assert _dtypes(ours) == _dtypes(theirs)


@pytest.mark.parametrize(
    "spelling,error",
    [("float64", ValueError), (float, ValueError), (3, TypeError)],
    ids=["dtype name", "float", "int value"],
)
def test_what_tensor_type_refuses_is_refused(spelling, error):
    model = _model(lucid)
    before = _dtypes(model)
    with pytest.raises(error):
        model.type(spelling)
    assert _dtypes(model) == before
