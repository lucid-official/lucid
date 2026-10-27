"""``Tensor.type`` takes a dtype or a legacy type name, as the reference does.

It took eight legacy strings only: ``t.type(lucid.float64)`` raised, a
bfloat16 tensor called itself ``'lucid.lucid.bfloat16'`` and could not be
cast back to that name, and ``int8`` answered ``ByteTensor``, the
reference's *unsigned* byte.  ``nn.Module.type`` casts through it, so it
inherited each gap (LCD-499).  Every dtype now has the reference's name,
and a cast takes either spelling.
"""

import pytest

import lucid
from lucid.test._fixtures.devices import device_dtype_params

_DTYPES = [
    lucid.float32,
    lucid.float64,
    lucid.float16,
    lucid.bfloat16,
    lucid.int8,
    lucid.int16,
    lucid.int32,
    lucid.int64,
    lucid.bool_,
    lucid.complex64,
    lucid.complex128,
]
_IDS = [str(d) for d in _DTYPES]


def _short(dtype):
    return str(dtype).removeprefix("lucid.")


@pytest.mark.parametrize("device,dtype", device_dtype_params(_DTYPES))
def test_a_name_casts_back_to_its_dtype(device, dtype):
    t = lucid.zeros(2, dtype=dtype, device=device)
    name = t.type()
    assert t.type(name) is t
    cast = lucid.zeros(2, device=device).type(name)
    assert cast.dtype == dtype and cast.device == t.device


# Off the CPU a name says the device as well (LCD-504): the reference names
# its Apple GPU tensors ``<lib>.mps.FloatTensor``, and a name casts to its
# device, a plain one to the CPU.  Metal's names are ``lucid.metal.*`` and,
# as the reference's other accelerators' names are, are taken back.
@pytest.mark.parametrize("device,dtype", device_dtype_params(_DTYPES))
def test_the_name_says_the_device(device, dtype):
    name = lucid.zeros(2, dtype=dtype, device=device).type()
    prefix = "lucid.metal." if device == "metal" else "lucid."
    assert name.startswith(prefix) and name.count(".") == prefix.count(".")


def test_a_name_casts_to_its_device_and_a_dtype_keeps_it(cross_device_pair):
    cpu, metal = cross_device_pair
    on_cpu, on_metal = lucid.ones(2, device=cpu), lucid.ones(2, device=metal)
    moved = on_cpu.type("lucid.metal.HalfTensor")
    assert moved.device == lucid.device(metal) and moved.dtype == lucid.float16
    back = on_metal.type("lucid.HalfTensor")
    assert back.device == lucid.device(cpu) and back.dtype == lucid.float16
    assert on_metal.type(lucid.float16).device == lucid.device(metal)
    assert on_metal.type(on_metal.type()) is on_metal
    assert on_cpu.type(on_metal.type()).device == lucid.device(metal)


@pytest.mark.parity
def test_the_reference_names_and_casts_its_gpu_tensors_alike(ref):
    if not ref.backends.mps.is_available():
        pytest.skip("the reference has no Apple GPU device here")
    lib = ref.__name__
    theirs = ref.zeros(2, device="mps")
    ours = lucid.zeros(2, device="metal")
    assert theirs.type() == f"{lib}.mps.FloatTensor"
    assert ours.type() == "lucid.metal.FloatTensor"
    assert theirs.type(f"{lib}.HalfTensor").device.type == "cpu"
    assert ours.type("lucid.HalfTensor").device == lucid.device("cpu")
    assert theirs.type(ref.float16).device.type == "mps"
    assert ours.type(lucid.float16).device == lucid.device("metal")


@pytest.mark.parametrize("device,dtype", device_dtype_params(_DTYPES))
def test_a_dtype_casts_as_to_does(device, dtype):
    t = lucid.ones(2, device=device)
    assert t.type(dtype).dtype == dtype
    assert t.type(dtype).tolist() == t.to(dtype).tolist()


@pytest.mark.parity
@pytest.mark.parametrize("dtype", _DTYPES, ids=_IDS)
def test_each_name_is_the_reference_name(ref, dtype):
    ours = lucid.zeros(2, dtype=dtype).type()
    theirs = ref.zeros(2, dtype=getattr(ref, _short(dtype))).type()
    assert ours.removeprefix("lucid.") == theirs.removeprefix(f"{ref.__name__}.")


def test_a_cast_keeps_the_graph(device):
    x = lucid.ones(2, requires_grad=True, device=device)
    y = x.type(lucid.float16)
    assert y.grad_fn is not None
    y.float().sum().backward()
    assert x.grad.tolist() == [1.0, 1.0]


def test_non_blocking_is_accepted(device):
    t = lucid.ones(2, device=device)
    assert t.type(lucid.float16, True).dtype == lucid.float16
    assert t.type("lucid.HalfTensor", non_blocking=True).dtype == lucid.float16
    assert t.type(None, True) == t.type()


# What each spelling raises, the same class in both libraries.  A string is
# written with ``{lib}`` for the library's own prefix.
_REFUSED = [
    ("float64", ValueError),
    ("{lib}.float64", ValueError),
    ("bogus", ValueError),
    (float, ValueError),
    (int, ValueError),
    (3, TypeError),
    ([1], TypeError),
]
_REFUSED_IDS = [
    "dtype name",
    "prefixed dtype name",
    "bogus",
    "float",
    "int",
    "int value",
    "list",
]


def _spelled(spelling, prefix):
    return spelling.format(lib=prefix) if isinstance(spelling, str) else spelling


@pytest.mark.parametrize("spelling,error", _REFUSED, ids=_REFUSED_IDS)
def test_what_names_no_type_is_refused(spelling, error):
    t = lucid.ones(2)
    with pytest.raises(error):
        t.type(_spelled(spelling, "lucid"))


@pytest.mark.parity
@pytest.mark.parametrize("spelling,error", _REFUSED, ids=_REFUSED_IDS)
def test_the_reference_refuses_the_same_spellings(ref, spelling, error):
    with pytest.raises(error):
        ref.ones(2).type(_spelled(spelling, ref.__name__))


def test_byte_tensor_names_nothing():
    # The reference's ByteTensor is uint8, which Lucid does not have; giving
    # int8 under that name would flip the sign of every value above 127.
    with pytest.raises(ValueError, match="invalid type"):
        lucid.ones(2).type("lucid.ByteTensor")
    assert lucid.ones(2, dtype=lucid.int8).type() == "lucid.CharTensor"


def test_type_as_moves_device_and_casts_dtype(cross_device_pair):
    cpu, metal = cross_device_pair
    x_cpu = lucid.tensor([1.0, 2.0], dtype=lucid.float32, device=cpu, requires_grad=True)
    proto_metal = lucid.zeros(2, dtype=lucid.float16, device=metal)
    y = x_cpu.type_as(proto_metal)
    assert y.device == lucid.device(metal) and y.dtype == lucid.float16
    assert y.type_as(proto_metal) is y
    back = y.type_as(x_cpu)
    assert back.device == lucid.device(cpu) and back.dtype == lucid.float32
    back.sum().backward()
    assert x_cpu.grad is not None and x_cpu.grad.device == lucid.device(cpu)
    assert x_cpu.grad.tolist() == [1.0, 1.0]


@pytest.mark.parity
def test_type_as_matches_reference_across_devices(ref):
    if not ref.backends.mps.is_available():
        pytest.skip("the reference has no Apple GPU device here")
    ours_cpu = lucid.ones(2, dtype=lucid.int32, device="cpu")
    ours_gpu = lucid.zeros(2, dtype=lucid.float16, device="metal")
    theirs_cpu = ref.ones(2, dtype=ref.int32, device="cpu")
    theirs_gpu = ref.zeros(2, dtype=ref.float16, device="mps")

    ours_to_gpu = ours_cpu.type_as(ours_gpu)
    theirs_to_gpu = theirs_cpu.type_as(theirs_gpu)
    assert ours_to_gpu.device == lucid.device("metal")
    assert theirs_to_gpu.device.type == "mps"
    assert ours_to_gpu.dtype == lucid.float16
    assert theirs_to_gpu.dtype == ref.float16

    ours_to_cpu = ours_gpu.type_as(ours_cpu)
    theirs_to_cpu = theirs_gpu.type_as(theirs_cpu)
    assert ours_to_cpu.device == lucid.device("cpu")
    assert theirs_to_cpu.device.type == "cpu"
    assert ours_to_cpu.dtype == lucid.int32
    assert theirs_to_cpu.dtype == ref.int32
