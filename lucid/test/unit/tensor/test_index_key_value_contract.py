"""Index keys and assigned values mean one thing on every path (API-02).

``x[key]`` and ``x[key] = v`` normalise ``key`` once
(``_tensor/_indexing.py`` ``_normalize_key``) and ``v`` once
(``_normalize_value``).  Each path used to decide for itself, and the
decisions drifted apart:

* ``x[True]`` had shape ``x.shape[1:]`` rather than ``(1, *x.shape)``, and
  ``x[False]`` the same rather than ``(0, *x.shape)``;
* a list key raised ``Unsupported index type: list``;
* an int64 index was narrowed to int32 first, so ``x[[2**40]]`` read
  ``x[0]`` (LCD-273), and on Metal ``x[[1, 99]] = v`` wrote ``x[0]``
  (LCD-209);
* a value went through ``float()``: a complex value raised and an int64
  past ``2**53`` lost its low bits;
* a value on the other device raised ``DeviceMismatch``;
* two slice assignments into one buffer made backward refuse the second,
  ``p.grad[0] = v`` left the gradient as it was, and ``retain_grad`` did
  not survive an assignment (LCD-296).

The grid below crosses every key kind with every value kind, on both
devices, read and written, against the reference.  The single-case tests
after it pin each reported repro without the reference.
"""

from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

import lucid
from lucid.test._fixtures.devices import metal_available

_OTHER = {"cpu": "metal", "metal": "cpu"}

# ── keys ──────────────────────────────────────────────────────────────────────

#: name -> builder of the key, given a function that turns a nested list
#: into a tensor of the library at hand (index tensors live on the CPU).
_KEYS: dict[str, Callable[[Callable[[object], object]], object]] = {
    "true": lambda T: True,
    "false": lambda T: False,
    "true-and-int": lambda T: (1, True),
    "bool-0d-tensor": lambda T: T(True),
    "bool-0d-tensor-false": lambda T: (slice(None), T(False)),
    "bool-mask-rows": lambda T: T([True, False, True]),
    "bool-mask-trailing": lambda T: (slice(None), T([False, True, True, False])),
    "bool-mask-full": lambda T: T([[True, False] * 2, [False] * 4, [True] * 4]),
    "int64-wide": lambda T: T([2, -1]),
    "list": lambda T: [0, 2],
    "nested-list": lambda T: [[0, 2], [1, 0]],
    "empty-list": lambda T: [],
    "ndarray": lambda T: np.array([2, 0]),
    "slice-and-list": lambda T: (slice(None), [3, 0]),
    "int-ellipsis-tensor": lambda T: (0, ..., T([1, 3])),
    "none-mask-slice": lambda T: (None, T([True, True, False]), slice(1, 3)),
    "two-arrays-negative": lambda T: (T([0, 2]), T([-1, -4])),
    "int64-past-2**31": lambda T: T([1, 2**40]),
}

#: The reference still reads a nested list as a tuple of indices, with a
#: deprecation warning.  Lucid reads it as one index, as NumPy does and as
#: the reference has announced it will, so the reference is handed that.
_REF_KEYS = {"nested-list": lambda T: T([[0, 2], [1, 0]])}

#: An index past the end: the CPU refuses it on both sides, and Metal
#: isolates it (LCD-228 policy B) — a read answers NaN, a write is dropped —
#: so there the reference runs without it.
_OUT_OF_RANGE = {"int64-past-2**31": lambda T: T([1])}

# ── values ────────────────────────────────────────────────────────────────────

#: name -> (dtype name, value builder given the library and the device)
_VALUES: dict[str, tuple[str, Callable[[ModuleType, str], object]]] = {
    "float": ("float32", lambda lib, d: 2.5),
    "int-past-2**53": ("int64", lambda lib, d: 2**53 + 1),
    "complex": ("complex64", lambda lib, d: 2 + 3j),
    "0d-tensor": ("float32", lambda lib, d: lib.tensor(7.0, device=d)),
    "other-device-tensor": ("float32", lambda lib, d: lib.tensor(-1.5, device=d)),
}


def _lucid_tensor(data: object) -> lucid.Tensor:
    if isinstance(data, list) and data and isinstance(data[0], list):
        is_bool = isinstance(data[0][0], bool)
    else:
        is_bool = isinstance(data, bool) or (
            isinstance(data, list) and bool(data) and isinstance(data[0], bool)
        )
    return lucid.tensor(data, dtype=lucid.bool if is_bool else lucid.int64)


def _base(lib: ModuleType, dtype: str, device: str) -> object:
    return lib.arange(12).reshape(3, 4).to(getattr(lib, dtype)).to(device)


def _value(lib: ModuleType, name: str, device: str) -> object:
    _, build = _VALUES[name]
    if lib is lucid and name == "other-device-tensor":
        return build(lib, _OTHER[device])
    return build(lib, device if lib is lucid else "cpu")


def _key(lib: ModuleType, name: str, device: str) -> object:
    if lib is lucid:
        return _KEYS[name](_lucid_tensor)
    if device == "metal" and name in _OUT_OF_RANGE:
        return _OUT_OF_RANGE[name](lib.tensor)
    return _REF_KEYS.get(name, _KEYS[name])(lib.tensor)


def _run(lib: ModuleType, key: str, value: str, op: str, device: str) -> object:
    dtype, _ = _VALUES[value]
    x = _base(lib, dtype, device if lib is lucid else "cpu")
    if op == "get":
        return x[_key(lib, key, device)].tolist()  # type: ignore[index]
    x[_key(lib, key, device)] = _value(lib, value, device)  # type: ignore[index]
    return x.tolist()  # type: ignore[attr-defined]


@pytest.mark.parity
@pytest.mark.parametrize("op", ["get", "set"])
@pytest.mark.parametrize("value", list(_VALUES))
@pytest.mark.parametrize("key", list(_KEYS))
def test_matches_the_reference(ref: ModuleType, device: str, key: str, value: str, op: str) -> None:
    if value == "other-device-tensor" and not metal_available():
        pytest.skip("needs a second device")
    if op == "get" and value not in ("float", "complex"):
        pytest.skip("a read takes no value; the float and complex dtypes cover it")
    if key in _OUT_OF_RANGE and device == "cpu":
        with pytest.raises(IndexError):
            _run(lucid, key, value, op, device)
        with pytest.raises(IndexError):
            _run(ref, key, value, op, device)
        return
    got = _run(lucid, key, value, op, device)
    if key in _OUT_OF_RANGE and op == "get":
        # Metal reads the row past the end as NaN.
        missing = got.pop()  # type: ignore[attr-defined]
        assert all(v != v for v in missing)
    assert got == _run(ref, key, value, op, device)


# ── each reported case, without the reference ────────────────────────────────


def test_bool_scalars_insert_an_axis(device: str) -> None:
    x = lucid.zeros(2, 3, device=device)
    assert x[True].shape == (1, 2, 3)
    assert x[False].shape == (0, 2, 3)
    assert x[lucid.tensor(True)].shape == (1, 2, 3)
    assert x[lucid.tensor(False)].shape == (0, 2, 3)
    assert x[0, True, None].shape == (1, 1, 3)


def test_a_bool_scalar_write(device: str) -> None:
    x = lucid.zeros(2, device=device)
    x[False] = 1.0
    assert x.tolist() == [0.0, 0.0]
    x[True] = 1.0
    assert x.tolist() == [1.0, 1.0]


def test_a_mask_must_match_the_dims_it_covers(device: str) -> None:
    with pytest.raises(IndexError, match="mask"):
        lucid.zeros(4, device=device)[lucid.tensor([True, False])]


def test_list_and_array_keys_are_indices(device: str) -> None:
    x = lucid.arange(5.0, device=device)
    assert x[[0, 1]].tolist() == [0.0, 1.0]
    assert x[np.array([4, 0])].tolist() == [4.0, 0.0]
    assert x[np.int64(3)].item() == 3.0
    x[[1, 3]] = 9.0
    assert x.tolist() == [0.0, 9.0, 2.0, 9.0, 4.0]


@pytest.mark.parametrize("bad", [1.5, "a", lucid.tensor([1.0])])
def test_what_is_not_an_index_is_refused(device: str, bad: object) -> None:
    with pytest.raises(IndexError):
        lucid.zeros(3, device=device)[bad]  # type: ignore[index]


def test_a_coordinate_index_is_checked_per_dim(device: str) -> None:
    # Folded into one flat index, x[[0], [-1]] read element 7 and x[[0], [5]]
    # read element 5.
    x = lucid.arange(8.0, device=device).reshape(2, 4)
    assert x[lucid.tensor([0]), lucid.tensor([-1])].tolist() == [3.0]
    past = (lucid.tensor([0]), lucid.tensor([5]))
    if device == "cpu":
        with pytest.raises(IndexError):
            x[past]
    else:
        assert np.isnan(x[past].tolist()[0])


def test_an_int64_index_is_not_narrowed(device: str) -> None:
    x = lucid.arange(5.0, device=device)
    idx = lucid.tensor([2**40])
    if device == "cpu":
        with pytest.raises(IndexError):
            x[idx]
        with pytest.raises(IndexError):
            x[idx] = 1.0
        return
    assert np.isnan(x[idx].tolist()[0])
    x[lucid.tensor([1, 2**40])] = 7.0
    assert x.tolist() == [0.0, 7.0, 2.0, 3.0, 4.0]


def test_python_scalars_keep_their_kind(device: str) -> None:
    c = lucid.zeros(3, dtype=lucid.complex64, device=device)
    c[1] = 2 + 3j
    assert c.tolist() == [0j, 2 + 3j, 0j]
    i = lucid.zeros(3, dtype=lucid.int64, device=device)
    i[1] = 2**60 + 1
    assert i[1].item() == 2**60 + 1
    i[2] = -2.7
    assert i[2].item() == -2
    with pytest.raises(TypeError):
        lucid.zeros(3, device=device)[0] = 1j
    with pytest.raises(OverflowError):
        lucid.zeros(3, dtype=lucid.int32, device=device)[0] = 2**40
    for non_finite in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            lucid.zeros(3, dtype=lucid.int64, device=device)[0] = non_finite


def test_a_value_moves_to_the_destination(device: str) -> None:
    x = lucid.zeros(4, device=device)
    x[0:2] = lucid.ones(2, device=_OTHER[device]) if metal_available() else 1.0
    assert x.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert x.device.type == lucid.device(device).type


def test_slice_assignments_into_one_buffer_backpropagate(device: str) -> None:
    # pad_packed_sequence builds its output this way.
    t = lucid.randn(2, 3, device=device, requires_grad=True)
    out = lucid.zeros(2, 3, device=device)
    out[0] = t[0]
    out[1] = t[1] * 2
    out.sum().backward()
    assert t.grad is not None
    assert t.grad.tolist() == [[1.0] * 3, [2.0] * 3]


def test_a_write_into_grad_writes_the_gradient(device: str) -> None:
    p = lucid.ones(3, device=device, requires_grad=True)
    (p * 2).sum().backward()
    assert p.grad is not None
    p.grad[0] = 5.0
    assert p.grad.tolist() == [5.0, 2.0, 2.0]


def test_retain_grad_survives_an_assignment(device: str) -> None:
    x = lucid.ones(3, device=device, requires_grad=True)
    y = x * 2
    y.retain_grad()
    y[0] = lucid.tensor(7.0, device=device)
    (y * 5).sum().backward()
    assert y.grad is not None and y.grad.tolist() == [5.0, 5.0, 5.0]
    assert x.grad is not None and x.grad.tolist() == [0.0, 10.0, 10.0]


def test_index_put_reads_a_bool_mask_as_a_mask(device: str) -> None:
    out = lucid.zeros(4, device=device).index_put(
        (lucid.tensor([False, False, True, True], device=device),),
        lucid.tensor(5.0, device=device),
    )
    assert out.tolist() == [0.0, 0.0, 5.0, 5.0]


def test_pad_packed_sequence_backpropagates(device: str) -> None:
    # It fills its output one step at a time, by slice assignment.
    from lucid.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

    x = lucid.randn(3, 2, 4, device=device, requires_grad=True)
    out, _ = pad_packed_sequence(pack_padded_sequence(x, [3, 2]))
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.sum().item() == 20.0


@pytest.mark.parity
@pytest.mark.parametrize("key", [slice(None), Ellipsis, 0], ids=["slice", "ellipsis", "int"])
@pytest.mark.parametrize("value", ["scalar", "tensor"])
def test_a_constant_write_cuts_the_gradient(
    ref: ModuleType, device: str, key: object, value: str
) -> None:
    # Copied into the buffer, ``x`` kept the graph of the values the write
    # replaced: ``w * 2`` assigned 5 sent ``w`` a gradient of 2.
    def run(lib: ModuleType, dev: str) -> object:
        w = lib.ones(3, device=dev, requires_grad=True)
        x = w * 2
        x[key] = 5.0 if value == "scalar" else lib.tensor(5.0, device=dev)
        x.sum().backward()
        return w.grad.tolist()

    assert run(lucid, device) == run(ref, "cpu")


def test_a_list_holding_slices_or_none_is_refused(device: str) -> None:
    x = lucid.zeros(2, 2, device=device)
    for key in ([0, slice(None)], [None, 0]):
        with pytest.raises(IndexError, match="tuple"):
            x[key]  # type: ignore[index]


def test_the_cpu_names_the_index_out_of_range() -> None:
    x = lucid.zeros(2, 4)
    with pytest.raises(IndexError, match="index 5 .* size 4"):
        x[lucid.tensor([0]), lucid.tensor([5])]


_SCALAR_KEY_SPELLINGS: dict[str, Callable[[Callable[[object], object]], object]] = {
    "int": lambda T: 1,
    "numpy-int": lambda T: np.int64(1),
    "numpy-0d-array": lambda T: np.array(1),
    "0d-int64-tensor": lambda T: T(1),
    "0d-neg-int-tensor": lambda T: T(-2),
    "tuple-two-0d-ints": lambda T: (T(1), T(2)),
    "tuple-0d-int-slice": lambda T: (T(1), slice(1, 3)),
    "ellipsis-0d-int": lambda T: (..., T(2)),
    "none-0d-int-ellipsis": lambda T: (None, T(1), ...),
    "0d-bool-true": lambda T: T(True),
    "0d-bool-false": lambda T: T(False),
    "tuple-0d-int-0d-bool": lambda T: (T(1), T(True)),
}


@pytest.mark.parity
@pytest.mark.parametrize("name", list(_SCALAR_KEY_SPELLINGS))
def test_scalar_key_spellings_view_and_write_through_match_reference(
    ref: ModuleType, device: str, name: str
) -> None:
    def run(lib: ModuleType, dev: str) -> tuple[tuple[int, ...], object, object]:
        x = lib.arange(12.0, device=dev).reshape(3, 4)
        make_t = (lambda v: _lucid_tensor(v).to(dev)) if lib is lucid else (lambda v: lib.tensor(v))
        v = x[_SCALAR_KEY_SPELLINGS[name](make_t)]
        read = v.tolist()
        v.add_(10.0)
        return tuple(v.shape), read, x.tolist()

    assert run(lucid, device) == run(ref, "cpu")


@pytest.mark.parametrize("dtype", [lucid.int8, lucid.int16, lucid.int32, lucid.int64])
def test_0d_integer_tensor_writes_through_across_integer_widths(
    device: str, dtype: lucid.dtype
) -> None:
    x = lucid.arange(4.0, device=device)
    v = x[lucid.tensor(1, dtype=dtype, device=device)]
    v.add_(10.0)
    assert x.tolist() == [0.0, 11.0, 2.0, 3.0]
