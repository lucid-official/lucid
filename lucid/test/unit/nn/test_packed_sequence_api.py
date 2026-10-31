"""LCD-750: ``PackedSequence`` builds from ``(data, batch_sizes)``, converts like a tensor.

Swept against the reference framework: constructor arity, the derived
inverse permutation, ``.to()`` / dtype shortcuts / ``cpu()`` with the index
tensors following ``data`` to its device, and ``pad_sequence(padding_side=)``,
``unpad_sequence``, ``unpack_sequence``.
"""

from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.nn.utils.rnn import (
    PackedSequence,
    pack_padded_sequence,
    pack_sequence,
    pad_sequence,
    unpack_sequence,
    unpad_sequence,
)
from lucid.test._fixtures.devices import metal_available

_DATA = np.arange(6, dtype=np.float32).reshape(3, 2)
_BATCH_SIZES = [2, 1]
_SORTED = [1, 2, 0]
_UNSORTED = [2, 0, 1]


def _np(t: lucid.Tensor) -> np.ndarray:
    return t.to("cpu").numpy()


def _ours(
    device: str, sorted_: list[int] | None = None, unsorted: list[int] | None = None
) -> PackedSequence:
    def idx(v: list[int] | None) -> lucid.Tensor | None:
        return None if v is None else lucid.tensor(v, dtype=lucid.int64, device=device)

    return PackedSequence(
        lucid.tensor(_DATA, device=device),
        lucid.tensor(_BATCH_SIZES, dtype=lucid.int64, device="cpu"),
        idx(sorted_),
        idx(unsorted),
    )


# ── constructor arity ─────────────────────────────────────────────────────────

#: case -> (sorted_indices, unsorted_indices given, unsorted_indices expected)
_ARITY = {
    "data+sizes": (None, None, None),
    "sorted": (_SORTED, None, _UNSORTED),
    "sorted+unsorted": (_SORTED, _UNSORTED, _UNSORTED),
    "unsorted only": (None, _UNSORTED, _UNSORTED),
}


@pytest.mark.parametrize("case", list(_ARITY))
def test_constructor_arity_and_derived_inverse(ref: ModuleType, case: str, device: str) -> None:
    sorted_, unsorted, expected = _ARITY[case]
    got = _ours(device, sorted_, unsorted)
    t = ref.tensor
    want = ref.nn.utils.rnn.PackedSequence(
        t(_DATA),
        t(_BATCH_SIZES),
        None if sorted_ is None else t(sorted_),
        None if unsorted is None else t(unsorted),
    )
    np.testing.assert_array_equal(_np(got.data), want.data.numpy())
    np.testing.assert_array_equal(_np(got.batch_sizes), want.batch_sizes.numpy())
    assert (got.sorted_indices is None) == (want.sorted_indices is None)
    assert (got.unsorted_indices is None) == (want.unsorted_indices is None)
    if expected is not None:
        assert got.unsorted_indices is not None
        assert got.unsorted_indices.device == lucid.device(device)
        np.testing.assert_array_equal(_np(got.unsorted_indices), want.unsorted_indices.numpy())
        assert _np(got.unsorted_indices).tolist() == expected


def test_a_packed_sequence_is_still_a_named_tuple() -> None:
    packed = _ours("cpu", _SORTED)
    assert isinstance(packed, tuple) and len(packed) == 4
    assert packed._fields == ("data", "batch_sizes", "sorted_indices", "unsorted_indices")
    data, batch_sizes, sorted_indices, unsorted_indices = packed
    assert data is packed.data and batch_sizes is packed.batch_sizes
    assert sorted_indices is packed.sorted_indices and unsorted_indices is packed.unsorted_indices
    replaced = packed._replace(sorted_indices=None)
    assert isinstance(replaced, PackedSequence) and replaced.sorted_indices is None


@pytest.mark.parametrize("bad", [[0, 0, 1], [0, 1, 3], [-1, 0, 1]])
def test_sorted_indices_must_be_a_permutation(bad: list[int]) -> None:
    with pytest.raises(ValueError, match="permutation"):
        _ours("cpu", bad)


def test_batch_sizes_must_live_on_the_cpu() -> None:
    if not metal_available():
        pytest.skip("Metal device not available on this host")
    with pytest.raises(ValueError, match="CPU"):
        PackedSequence(
            lucid.tensor(_DATA, device="metal"),
            lucid.tensor(_BATCH_SIZES, dtype=lucid.int64, device="metal"),
        )


@pytest.mark.parametrize("kind", ["RNN", "GRU", "LSTM"])
def test_a_recurrent_layer_runs_on_a_hand_built_packed_sequence(kind: str, device: str) -> None:
    lucid.manual_seed(0)
    layer = getattr(nn, kind)(2, 3).to(device)
    x = lucid.randn(3, 4, 2, device=device)
    packed = pack_padded_sequence(x, [4, 3, 1], batch_first=True)
    built = PackedSequence(packed.data, packed.batch_sizes)
    want = layer(packed)[0]
    got = layer(built)[0]
    np.testing.assert_allclose(_np(got.data), _np(want.data), rtol=0, atol=0)


# ── conversions ───────────────────────────────────────────────────────────────

_CASTS = {
    "float": lucid.float32,
    "double": lucid.float64,
    "half": lucid.float16,
    "long": lucid.int64,
    "int": lucid.int32,
    "short": lucid.int16,
    "char": lucid.int8,
}


@pytest.mark.parametrize("name", list(_CASTS))
def test_dtype_shortcuts_cast_data_and_leave_the_indices(
    ref: ModuleType, name: str, device: str
) -> None:
    packed = _ours(device, _SORTED)
    if device == "metal" and name == "double":
        with pytest.raises(NotImplementedError, match="float64"):
            packed.double()
        return
    got = getattr(packed, name)()
    assert got.data.dtype == _CASTS[name]
    assert got.data.device == lucid.device(device)
    assert got.sorted_indices.dtype == lucid.int64 and got.unsorted_indices.dtype == lucid.int64
    assert got.batch_sizes is packed.batch_sizes
    theirs = getattr(
        ref.nn.utils.rnn.PackedSequence(
            ref.tensor(_DATA), ref.tensor(_BATCH_SIZES), ref.tensor(_SORTED)
        ),
        name,
    )()
    np.testing.assert_array_equal(_np(got.data), theirs.data.numpy())
    np.testing.assert_array_equal(_np(got.sorted_indices), theirs.sorted_indices.numpy())


def test_a_no_op_conversion_returns_the_same_object(device: str) -> None:
    packed = _ours(device, _SORTED)
    assert packed.float() is packed
    assert packed.to(device) is packed
    assert packed.to(dtype=lucid.float32) is packed
    assert packed.to(device, copy=True) is not packed


def test_to_moves_the_indices_with_the_data() -> None:
    if not metal_available():
        pytest.skip("Metal device not available on this host")
    cpu = _ours("cpu", _SORTED)
    for moved in (cpu.to("metal"), cpu.metal(), cpu.to(lucid.zeros(1, device="metal"))):
        assert moved.is_metal and moved.data.is_metal
        assert moved.sorted_indices.is_metal and moved.unsorted_indices.is_metal
        assert not moved.batch_sizes.is_metal
        np.testing.assert_array_equal(_np(moved.unsorted_indices), _UNSORTED)
    back = cpu.to("metal").cpu()
    assert not back.is_metal and not back.sorted_indices.is_metal
    np.testing.assert_array_equal(_np(back.data), _DATA)
    both = cpu.to("metal", dtype=lucid.float16)
    assert both.data.dtype == lucid.float16 and both.sorted_indices.dtype == lucid.int64


def test_cpu_on_a_cpu_packed_sequence_is_the_identity() -> None:
    packed = _ours("cpu")
    assert packed.cpu() is packed and not packed.is_metal


def test_none_indices_stay_none_through_a_conversion(device: str) -> None:
    got = _ours(device).half()
    assert got.sorted_indices is None and got.unsorted_indices is None


def test_pin_memory_is_the_identity_on_unified_memory(device: str) -> None:
    packed = _ours(device, _SORTED)
    pinned = packed.pin_memory()
    assert isinstance(pinned, PackedSequence)
    np.testing.assert_array_equal(_np(pinned.data), _DATA)
    np.testing.assert_array_equal(_np(pinned.unsorted_indices), _UNSORTED)
    assert pinned.is_pinned() is False
    assert _ours(device).pin_memory().sorted_indices is None


# ── pad_sequence(padding_side=) / unpad_sequence / unpack_sequence ────────────

_LENS = {"ragged": [4, 2, 3], "equal": [3, 3, 3], "single": [5]}


def _seqs(lens: list[int]) -> list[np.ndarray]:
    rng = np.random.default_rng(len(lens))
    return [rng.standard_normal((n, 2)).astype(np.float32) for n in lens]


@pytest.mark.parametrize("case", list(_LENS))
@pytest.mark.parametrize("batch_first", [False, True], ids=["TB", "BT"])
@pytest.mark.parametrize("side", ["right", "left"])
def test_pad_sequence_padding_side(
    ref: ModuleType, case: str, batch_first: bool, side: str, device: str
) -> None:
    arrays = _seqs(_LENS[case])
    got = pad_sequence(
        [lucid.tensor(a, device=device) for a in arrays],
        batch_first=batch_first,
        padding_value=-1.0,
        padding_side=side,
    )
    want = ref.nn.utils.rnn.pad_sequence(
        [ref.tensor(a) for a in arrays],
        batch_first=batch_first,
        padding_value=-1.0,
        padding_side=side,
    )
    np.testing.assert_array_equal(_np(got), want.numpy())


def test_pad_sequence_left_keeps_the_gradient(device: str) -> None:
    a = lucid.ones(2, 2, device=device, requires_grad=True)
    b = lucid.ones(4, 2, device=device, requires_grad=True)
    pad_sequence([a, b], padding_side="left").sum().backward()
    np.testing.assert_array_equal(_np(a.grad), np.ones((2, 2)))


@pytest.mark.parametrize("side", ["both", "Right", "", None])
def test_pad_sequence_refuses_an_unknown_side(side: object) -> None:
    with pytest.raises(ValueError, match="padding_side"):
        pad_sequence([lucid.ones(2, 2)], padding_side=side)  # type: ignore[arg-type]


@pytest.mark.parametrize("case", list(_LENS))
@pytest.mark.parametrize("batch_first", [False, True], ids=["TB", "BT"])
@pytest.mark.parametrize("as_tensor", [False, True], ids=["list", "tensor"])
def test_unpad_sequence(
    ref: ModuleType, case: str, batch_first: bool, as_tensor: bool, device: str
) -> None:
    lens = _LENS[case]
    arrays = _seqs(lens)
    padded = pad_sequence([lucid.tensor(a, device=device) for a in arrays], batch_first=batch_first)
    got_lens = lucid.tensor(lens, device=device) if as_tensor else lens
    got = unpad_sequence(padded, got_lens, batch_first=batch_first)
    theirs_padded = ref.nn.utils.rnn.pad_sequence(
        [ref.tensor(a) for a in arrays], batch_first=batch_first
    )
    want = ref.nn.utils.rnn.unpad_sequence(theirs_padded, ref.tensor(lens), batch_first=batch_first)
    assert len(got) == len(want) == len(lens)
    for g, w, a in zip(got, want, arrays, strict=True):
        assert g.device == lucid.device(device)
        np.testing.assert_array_equal(_np(g), w.numpy())
        np.testing.assert_array_equal(_np(g), a)


@pytest.mark.parametrize(
    "lens, match",
    [([4, 2], "len\\(lengths\\)"), ([4, 5, 3], "must lie"), ([4, -1, 3], "must lie")],
)
def test_unpad_sequence_validates_lengths(lens: list[int], match: str) -> None:
    padded = pad_sequence([lucid.ones(n, 2) for n in (4, 2, 3)], batch_first=True)
    with pytest.raises(ValueError, match=match):
        unpad_sequence(padded, lens, batch_first=True)


def test_unpad_sequence_allows_an_empty_sequence() -> None:
    padded = pad_sequence([lucid.ones(3, 2), lucid.ones(2, 2)], batch_first=True)
    first, second = unpad_sequence(padded, [3, 0], batch_first=True)
    assert tuple(first.shape) == (3, 2) and tuple(second.shape) == (0, 2)


def test_unpad_sequence_refuses_a_one_axis_input() -> None:
    with pytest.raises(ValueError, match="batch axis"):
        unpad_sequence(lucid.ones(3), [3])


def test_unpad_sequence_follows_the_gradient() -> None:
    padded = lucid.ones(2, 4, 2, requires_grad=True)
    parts = unpad_sequence(padded, [4, 2], batch_first=True)
    (parts[0].sum() + parts[1].sum()).backward()
    assert padded.grad.numpy().sum() == 12.0


@pytest.mark.parametrize("case", list(_LENS))
@pytest.mark.parametrize("enforce_sorted", [True, False], ids=["enforce", "sort"])
def test_unpack_sequence(ref: ModuleType, case: str, enforce_sorted: bool, device: str) -> None:
    lens = _LENS[case]
    if enforce_sorted and lens != sorted(lens, reverse=True):
        pytest.skip("an unsorted batch cannot be packed with enforce_sorted=True")
    arrays = _seqs(lens)
    got = unpack_sequence(
        pack_sequence([lucid.tensor(a, device=device) for a in arrays], enforce_sorted)
    )
    want = ref.nn.utils.rnn.unpack_sequence(
        ref.nn.utils.rnn.pack_sequence([ref.tensor(a) for a in arrays], enforce_sorted)
    )
    assert len(got) == len(want) == len(arrays)
    for g, w, a in zip(got, want, arrays, strict=True):
        np.testing.assert_array_equal(_np(g), w.numpy())
        np.testing.assert_array_equal(_np(g), a)


def test_unpack_sequence_of_a_hand_built_packed_sequence(device: str) -> None:
    arrays = _seqs([4, 2, 3])
    packed = pack_sequence([lucid.tensor(a, device=device) for a in arrays], enforce_sorted=False)
    built = PackedSequence(packed.data, packed.batch_sizes, packed.sorted_indices)
    for g, a in zip(unpack_sequence(built), arrays, strict=True):
        np.testing.assert_array_equal(_np(g), a)


def test_byte_is_refused_as_the_tensor_method_is() -> None:
    with pytest.raises(NotImplementedError, match="uint8"):
        _ours("cpu", _SORTED).byte()
