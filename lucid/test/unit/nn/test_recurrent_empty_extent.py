"""Recurrent modules at an empty extent: no timesteps, or no sequences.

A zero-length sequence is refused by the reference, on both its CPU and
its GPU path (RuntimeError "Expected sequence length to be larger than 0").
Lucid used to fail on it too, by accident: the recurrences stack their
per-step outputs, and with no steps ``stack`` / ``cat`` raised an internal
"empty input".  On Metal, the engine's own ``lstm`` entry reached the
kernel and failed in ``mx::stack``.  Both doors now refuse with one
ValueError, the module door and the engine entry.  An empty batch
(``B == 0`` with ``T > 0``) is valid in the reference.  It runs here
and is compared against the reference: shapes, values and gradients.
"""

from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid._C import engine as _C_engine
from lucid._dispatch import _unwrap
from lucid.nn.utils.rnn import PackedSequence

_I, _H, _P, _B, _T = 4, 3, 2, 2, 3
_REFUSAL = "expected sequence length to be larger than 0"
_KINDS = ("RNN", "GRU", "LSTM")

# name → (constructor kwargs, unbatched input).  ``proj`` is LSTM-only.
_LAYOUTS: dict[str, tuple[dict[str, object], bool]] = {
    "dense": ({}, False),
    "batch_first": ({"batch_first": True}, False),
    "bidir_2layer": ({"num_layers": 2, "bidirectional": True}, False),
    "proj": ({"proj_size": _P}, False),
    "unbatched": ({}, True),
}
_BATCH_LAYOUTS = ("dense", "batch_first", "bidir_2layer", "proj")


def _cases(layouts: tuple[str, ...]) -> list[object]:
    cases: list[object] = []
    for kind in _KINDS:
        for layout in layouts:
            if layout == "proj" and kind != "LSTM":
                continue
            cases.append(pytest.param(kind, layout, id=f"{kind}-{layout}"))
    return cases


def _sequence_shape(layout: str, *, steps: int, batch: int) -> tuple[int, ...]:
    kwargs, unbatched = _LAYOUTS[layout]
    if unbatched:
        return (steps, _I)
    return (batch, steps, _I) if kwargs.get("batch_first") else (steps, batch, _I)


# ── T == 0 is refused ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind,layout", _cases(tuple(_LAYOUTS)))
def test_empty_sequence_is_refused(kind: str, layout: str, device: str) -> None:
    kwargs, _ = _LAYOUTS[layout]
    module = getattr(nn, kind)(_I, _H, **kwargs).to(device)
    shape = _sequence_shape(layout, steps=0, batch=_B)
    x = lucid.randn(*shape, device=device, requires_grad=True)
    with pytest.raises(ValueError, match=_REFUSAL):
        module(x)


@pytest.mark.parametrize("kind,layout", _cases(tuple(_LAYOUTS)))
def test_reference_refuses_empty_sequence(ref: ModuleType, kind: str, layout: str) -> None:
    # Why the refusal above is the contract and not a gap.
    kwargs, _ = _LAYOUTS[layout]
    module = getattr(ref.nn, kind)(_I, _H, **kwargs)
    with pytest.raises(RuntimeError, match="sequence length to be larger than 0"):
        module(ref.randn(*_sequence_shape(layout, steps=0, batch=_B)))


@pytest.mark.parametrize("kind", _KINDS)
def test_empty_packed_sequence_is_refused(kind: str, device: str) -> None:
    module = getattr(nn, kind)(_I, _H).to(device)
    packed = PackedSequence(
        lucid.randn(0, _I, device=device), lucid.tensor([], dtype=lucid.int64), None, None
    )
    with pytest.raises(ValueError, match=_REFUSAL):
        module(packed)


@pytest.mark.parametrize("requires_grad", [False, True], ids=["inference", "train"])
def test_engine_lstm_refuses_empty_sequence(device: str, requires_grad: bool) -> None:
    # The engine entry is reachable without the module; on Metal it used to
    # reach the kernel's ``mx::stack`` of no steps.
    x = lucid.randn(0, _B, _I, device=device)
    weights = [
        lucid.randn(4 * _H, _I, device=device, requires_grad=requires_grad),
        lucid.randn(4 * _H, _H, device=device),
        lucid.randn(4 * _H, device=device),
        lucid.randn(4 * _H, device=device),
    ]
    with pytest.raises(ValueError, match=_REFUSAL) as info:
        _C_engine.nn.lstm_forward(
            _unwrap(x), None, None, [_unwrap(w) for w in weights], _H, 1, False, False, True, 0
        )
    assert isinstance(info.value, _C_engine.InvalidArgument)


# ── B == 0 runs, as in the reference ─────────────────────────────────────────


def _run_empty_batch(fw: ModuleType, kind: str, layout: str, device: str) -> dict[str, object]:
    """Run ``kind`` over ``_T`` steps of an empty batch, with explicit initial
    states, in Lucid or the reference (``fw``); backpropagate every output."""
    kwargs, _ = _LAYOUTS[layout]
    module = getattr(fw.nn, kind)(_I, _H, **kwargs).to(device)
    rows = int(kwargs.get("num_layers", 1)) * (2 if kwargs.get("bidirectional") else 1)
    width = _P if layout == "proj" else _H

    shape = _sequence_shape(layout, steps=_T, batch=0)
    leaves = {
        "x": fw.randn(*shape, device=device, requires_grad=True),
        "h0": fw.randn(rows, 0, width, device=device, requires_grad=True),
    }
    if kind == "LSTM":
        leaves["c0"] = fw.randn(rows, 0, _H, device=device, requires_grad=True)
        out, (h_n, c_n) = module(leaves["x"], (leaves["h0"], leaves["c0"]))
        states = [h_n, c_n]
    else:
        out, h_n = module(leaves["x"], leaves["h0"])
        states = [h_n]

    loss = out.sum()
    for state in states:
        loss = loss + state.sum()
    loss.backward()
    return {
        "out": out,
        "states": states,
        "leaf_grads": {name: leaf.grad for name, leaf in leaves.items()},
        "param_grads": {n: p.grad for n, p in module.named_parameters()},
    }


def _assert_same(got: object, want: object) -> None:
    """Same shape and values; ``want`` is a reference tensor or None."""
    if want is None:
        assert got is None
        return
    assert got is not None
    assert tuple(got.shape) == tuple(want.shape)
    np.testing.assert_array_equal(got.numpy(), want.detach().numpy())


@pytest.mark.parametrize("kind,layout", _cases(_BATCH_LAYOUTS))
def test_empty_batch_forward_matches_reference(
    ref: ModuleType, kind: str, layout: str, device: str
) -> None:
    kwargs, _ = _LAYOUTS[layout]
    shape = _sequence_shape(layout, steps=_T, batch=0)
    with lucid.no_grad():
        out, states = getattr(nn, kind)(_I, _H, **kwargs).to(device)(
            lucid.randn(*shape, device=device)
        )
    with ref.no_grad():
        want_out, want_states = getattr(ref.nn, kind)(_I, _H, **kwargs)(ref.randn(*shape))
    _assert_same(out, want_out)
    ours = list(states) if isinstance(states, tuple) else [states]
    theirs = list(want_states) if isinstance(want_states, tuple) else [want_states]
    for got, want in zip(ours, theirs, strict=True):
        _assert_same(got, want)


@pytest.mark.parametrize("kind,layout", _cases(_BATCH_LAYOUTS))
def test_empty_batch_gradients_match_reference(
    ref: ModuleType, kind: str, layout: str, device: str
) -> None:
    ours = _run_empty_batch(lucid, kind, layout, device)
    theirs = _run_empty_batch(ref, kind, layout, "cpu")
    _assert_same(ours["out"], theirs["out"])
    for got, want in zip(ours["states"], theirs["states"], strict=True):
        _assert_same(got, want)
    for group in ("leaf_grads", "param_grads"):
        got_grads, want_grads = ours[group], theirs[group]
        assert got_grads.keys() == want_grads.keys()
        for name, grad in got_grads.items():
            _assert_same(grad, want_grads[name])
