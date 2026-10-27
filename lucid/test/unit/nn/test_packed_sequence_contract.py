"""The PackedSequence helpers validate lengths and keep indices on the input's device.

Two halves of one contract, checked against the reference:

* **lengths at the boundary** — a length of 0 or less, longer than the
  padded time axis, the wrong count, or a 2-D ``lengths`` used to build a
  ``batch_sizes`` that no longer described ``data`` (``[3, -1]`` packed
  without complaint).  Every helper now refuses before it gathers.
* **device-follow** — the sort permutation was built on the default
  device, so ``enforce_sorted=False`` on a Metal input raised
  ``DeviceMismatch (gather)``.  Every index a helper builds now lives on
  the input's device; ``batch_sizes`` and the returned lengths stay CPU
  ``int64``, as the reference keeps them.

Sweep: {pack_padded_sequence, pad_packed_sequence, pack_sequence,
pad_sequence} × lengths {sorted, unsorted, contains 0, negative, > T,
wrong count, 2-D} × batch_first × device, plus gradients through
pack → LSTM → pad.
"""

from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.nn.utils.rnn import (
    PackedSequence,
    pack_padded_sequence,
    pack_sequence,
    pad_packed_sequence,
    pad_sequence,
)
from lucid.test._fixtures.devices import metal_available

_T, _F = 4, 2

#: Per-sample lengths for a batch of three padded to ``_T`` steps.
_FLAT: dict[str, list[int]] = {
    "sorted": [4, 3, 1],
    "unsorted": [3, 4, 1],
    "ties": [3, 1, 3],
    "zero": [4, 0, 2],
    "negative": [4, -1, 2],
    "too-long": [5, 3, 1],
    "wrong-count": [4, 3],
}
_LENGTHS: dict[str, list[int] | list[list[int]]] = {**_FLAT, "two-d": [[4, 3, 1]]}
_VALID = {"sorted", "unsorted", "ties"}
#: ``(case, enforce_sorted)`` the reference packs instead of refusing: a
#: length past ``T`` yields a ``batch_sizes`` that outruns ``data``, and a
#: short ``lengths`` with sorting on silently drops the trailing samples.
_REFERENCE_PACKS = {("too-long", True), ("too-long", False), ("wrong-count", False)}

_SeqFn = Callable[[list[lucid.Tensor]], object]


def _padded(batch_first: bool, seed: int = 0) -> np.ndarray:
    x = np.random.default_rng(seed).standard_normal((3, _T, _F)).astype(np.float32)
    return x if batch_first else np.ascontiguousarray(x.transpose(1, 0, 2))


def _np(t: lucid.Tensor) -> np.ndarray:
    return t.to("cpu").numpy()


def _assert_index(got: lucid.Tensor | None, want: object, device: str) -> None:
    if want is None:
        assert got is None
        return
    assert got is not None
    assert got.device == lucid.device(device)
    assert got.dtype == lucid.int64
    np.testing.assert_array_equal(_np(got), want.numpy())


def _assert_packed(
    got: PackedSequence, want: object, device: str, check_order: bool = True
) -> None:
    assert got.data.device == lucid.device(device)
    assert got.batch_sizes.device == lucid.device("cpu")
    assert got.batch_sizes.dtype == lucid.int64
    np.testing.assert_array_equal(_np(got.batch_sizes), want.batch_sizes.numpy())
    if check_order:
        np.testing.assert_allclose(_np(got.data), want.data.numpy(), rtol=0, atol=0)
        _assert_index(got.sorted_indices, want.sorted_indices, device)
        _assert_index(got.unsorted_indices, want.unsorted_indices, device)


@pytest.mark.parametrize("as_tensor", [False, True], ids=["list", "tensor"])
@pytest.mark.parametrize("enforce_sorted", [True, False], ids=["enforce", "sort"])
@pytest.mark.parametrize("batch_first", [False, True], ids=["TB", "BT"])
@pytest.mark.parametrize("case", list(_LENGTHS))
def test_pack_padded_sequence(
    ref: ModuleType,
    case: str,
    batch_first: bool,
    enforce_sorted: bool,
    as_tensor: bool,
    device: str,
) -> None:
    lengths = _LENGTHS[case]
    x = _padded(batch_first)
    # ``lengths`` may live anywhere for us; the reference wants it on CPU.
    ours_len = lucid.tensor(lengths, device=device) if as_tensor else lengths
    theirs_len = ref.tensor(lengths) if as_tensor else lengths
    refused = case not in _VALID or (enforce_sorted and case != "sorted")
    if refused:
        with pytest.raises(ValueError):
            pack_padded_sequence(
                lucid.tensor(x, device=device), ours_len, batch_first, enforce_sorted
            )
        if (case, enforce_sorted) not in _REFERENCE_PACKS:
            with pytest.raises((RuntimeError, IndexError)):
                ref.nn.utils.rnn.pack_padded_sequence(
                    ref.tensor(x), theirs_len, batch_first, enforce_sorted
                )
        return
    got = pack_padded_sequence(
        lucid.tensor(x, device=device), ours_len, batch_first, enforce_sorted
    )
    want = ref.nn.utils.rnn.pack_padded_sequence(
        ref.tensor(x), theirs_len, batch_first, enforce_sorted
    )
    # How a sort breaks ties is unspecified; the round trip below still holds.
    _assert_packed(got, want, device, check_order=case != "ties")


@pytest.mark.parametrize("total", [None, "max", "longer", "shorter"])
@pytest.mark.parametrize("batch_first", [False, True], ids=["TB", "BT"])
@pytest.mark.parametrize(
    "case, enforce_sorted",
    # Only a sorted batch packs with enforce_sorted=True.
    [("sorted", True)] + [(case, False) for case in sorted(_VALID)],
)
def test_pad_packed_sequence_round_trip(
    ref: ModuleType,
    case: str,
    batch_first: bool,
    enforce_sorted: bool,
    total: str | None,
    device: str,
) -> None:
    lengths = _FLAT[case]
    longest = max(lengths)
    total_length = {None: None, "max": longest, "longer": longest + 2}.get(total, longest - 1)
    x = _padded(batch_first, seed=1)
    packed = pack_padded_sequence(
        lucid.tensor(x, device=device), lengths, batch_first, enforce_sorted
    )
    rpacked = ref.nn.utils.rnn.pack_padded_sequence(
        ref.tensor(x), lengths, batch_first, enforce_sorted
    )
    if total == "shorter":
        with pytest.raises(ValueError, match="total_length"):
            pad_packed_sequence(packed, batch_first, -1.0, total_length)
        with pytest.raises(ValueError, match="total_length"):
            ref.nn.utils.rnn.pad_packed_sequence(rpacked, batch_first, -1.0, total_length)
        return
    out, out_len = pad_packed_sequence(packed, batch_first, -1.0, total_length)
    rout, rlen = ref.nn.utils.rnn.pad_packed_sequence(rpacked, batch_first, -1.0, total_length)
    assert out.device == lucid.device(device)
    assert out_len.device == lucid.device("cpu") and out_len.dtype == lucid.int64
    np.testing.assert_array_equal(_np(out), rout.numpy())
    np.testing.assert_array_equal(_np(out_len), rlen.numpy())


@pytest.mark.parametrize("enforce_sorted", [True, False], ids=["enforce", "sort"])
@pytest.mark.parametrize("case", ["sorted", "unsorted", "zero"])
def test_pack_sequence(ref: ModuleType, case: str, enforce_sorted: bool, device: str) -> None:
    lengths = _FLAT[case]
    rng = np.random.default_rng(2)
    seqs = [rng.standard_normal((n, _F)).astype(np.float32) for n in lengths]
    ours = [lucid.tensor(s, device=device) for s in seqs]
    theirs = [ref.tensor(s) for s in seqs]
    if case == "zero" or (enforce_sorted and case != "sorted"):
        with pytest.raises(ValueError):
            pack_sequence(ours, enforce_sorted)
        with pytest.raises(RuntimeError):
            ref.nn.utils.rnn.pack_sequence(theirs, enforce_sorted)
        return
    got = pack_sequence(ours, enforce_sorted)
    _assert_packed(got, ref.nn.utils.rnn.pack_sequence(theirs, enforce_sorted), device)
    out, _ = pad_packed_sequence(got, batch_first=True)
    rout, _ = ref.nn.utils.rnn.pad_packed_sequence(
        ref.nn.utils.rnn.pack_sequence(theirs, enforce_sorted), batch_first=True
    )
    np.testing.assert_array_equal(_np(out), rout.numpy())


@pytest.mark.parametrize("batch_first", [False, True], ids=["TB", "BT"])
@pytest.mark.parametrize("case", ["sorted", "unsorted", "zero"])
def test_pad_sequence(ref: ModuleType, case: str, batch_first: bool, device: str) -> None:
    lengths = _FLAT[case]
    rng = np.random.default_rng(3)
    seqs = [rng.standard_normal((n, _F)).astype(np.float32) for n in lengths]
    got = pad_sequence([lucid.tensor(s, device=device) for s in seqs], batch_first, 7.0)
    want = ref.nn.utils.rnn.pad_sequence([ref.tensor(s) for s in seqs], batch_first, 7.0)
    assert got.device == lucid.device(device)
    np.testing.assert_array_equal(_np(got), want.numpy())


@pytest.mark.parametrize(
    "build, error",
    [
        (lambda d: [], ValueError),
        (lambda d: [lucid.tensor(1.0, device=d), lucid.tensor(2.0, device=d)], ValueError),
        (lambda d: [lucid.zeros(2, 3, device=d), lucid.zeros(3, 4, device=d)], ValueError),
        (lambda d: [lucid.zeros(2, 3, device=d), lucid.zeros(3, device=d)], ValueError),
    ],
    ids=["empty", "zero-d", "trailing-shape", "rank"],
)
@pytest.mark.parametrize("fn", [pad_sequence, pack_sequence], ids=["pad", "pack"])
def test_sequence_list_refusals(
    fn: _SeqFn, build: Callable[[str], list[lucid.Tensor]], error: type, device: str
) -> None:
    with pytest.raises(error):
        fn(build(device))


@pytest.mark.skipif(not metal_available(), reason="needs a second device")
@pytest.mark.parametrize("fn", [pad_sequence, pack_sequence], ids=["pad", "pack"])
def test_sequence_list_refuses_mixed_devices(fn: _SeqFn) -> None:
    with pytest.raises(ValueError, match="sequence 1 is on"):
        fn([lucid.zeros(3, 2), lucid.zeros(2, 2, device="metal")])


@pytest.mark.parametrize(
    "batch_sizes, rows",
    [([[2, 1]], 3), ([], 0), ([1, 2], 3), ([2, 0], 2), ([2, 1], 4)],
    ids=["two-d", "empty", "increasing", "zero", "row-count"],
)
def test_pad_packed_sequence_refuses_a_corrupt_batch_sizes(
    batch_sizes: list[int], rows: int, device: str
) -> None:
    packed = PackedSequence(
        lucid.zeros(rows, _F, device=device),
        lucid.tensor(batch_sizes, dtype=lucid.int64),
        None,
        None,
    )
    with pytest.raises(ValueError, match="batch_sizes"):
        pad_packed_sequence(packed)


@pytest.mark.parametrize("batch_first", [False, True], ids=["TB", "BT"])
@pytest.mark.parametrize("case", ["sorted", "unsorted"])
def test_gradients_through_pack_lstm_pad(
    ref: ModuleType, case: str, batch_first: bool, device: str
) -> None:
    lengths = _FLAT[case]
    enforce_sorted = case == "sorted"
    # Metal holds no float64; float32 there is checked to float32 precision.
    dtype, tol = (lucid.float64, 1e-10) if device == "cpu" else (lucid.float32, 1e-5)
    np_dt = np.float64 if dtype is lucid.float64 else np.float32
    x_np = _padded(batch_first, seed=4).astype(np_dt)
    w_np = (
        np.random.default_rng(5)
        .standard_normal((3, _T, 5) if batch_first else (_T, 3, 5))
        .astype(np_dt)
    )

    ours = nn.LSTM(_F, 5, batch_first=batch_first, dtype=dtype)
    theirs = ref.nn.LSTM(_F, 5, batch_first=batch_first).to(
        ref.float64 if dtype is lucid.float64 else ref.float32
    )
    for p, q in zip(ours.parameters(), theirs.parameters(), strict=True):
        q.data = ref.tensor(p.numpy())
    ours = ours.to(device)

    x = lucid.tensor(x_np, device=device, requires_grad=True)
    packed = pack_padded_sequence(x, lengths, batch_first, enforce_sorted)
    out, _ = pad_packed_sequence(ours(packed)[0], batch_first)
    (out * lucid.tensor(w_np, device=device)).sum().backward()

    rx = ref.tensor(x_np, requires_grad=True)
    rpacked = ref.nn.utils.rnn.pack_padded_sequence(rx, lengths, batch_first, enforce_sorted)
    rout, _ = ref.nn.utils.rnn.pad_packed_sequence(theirs(rpacked)[0], batch_first)
    (rout * ref.tensor(w_np)).sum().backward()

    np.testing.assert_allclose(_np(out), rout.detach().numpy(), rtol=tol, atol=tol)
    np.testing.assert_allclose(_np(x.grad), rx.grad.numpy(), rtol=tol, atol=tol)
    for p, q in zip(ours.parameters(), theirs.parameters(), strict=True):
        np.testing.assert_allclose(_np(p.grad), q.grad.numpy(), rtol=tol, atol=tol)
