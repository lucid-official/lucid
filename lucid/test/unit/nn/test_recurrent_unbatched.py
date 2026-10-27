"""Recurrent modules and cells on an unbatched input.

The reference takes a sequence without a batch axis — ``(L, H_in)`` for
RNN / GRU / LSTM, ``(H_in,)`` for the cells — and answers without one:
``(L, D * H_out)`` and ``(D * num_layers, H_out)`` states, ``(H_out,)`` from
a cell.  Lucid used to run that ``(L, H_in)`` as if it were ``(T, B, H_in)``:
RNN answered ``(L, H_in, H_out)``, GRU raised IndexError, LSTM a shape
mismatch, and the cells went wrong the same way one rank down.

Every recurrent forward now goes through one prologue that adds the batch
axis of 1 and one epilogue that takes it off, and every cell forward
through one shared cell prologue.  The cases below compare outputs, final
states and the gradients to the input, the initial states and every
parameter against the reference, on CPU and Metal.

An initial state whose rank does not match the input's is refused, as the
reference refuses it.  Lucid raises ValueError naming the expected shape,
where the reference raises RuntimeError.
"""

import re
from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn as nn

_I, _H, _P, _T, _B = 4, 3, 2, 5, 2
_KINDS = ("RNN", "GRU", "LSTM")
_CELLS = ("RNNCell", "GRUCell", "LSTMCell")

# name → (constructor kwargs, supply initial states).  ``proj`` is LSTM-only.
_CONFIGS: dict[str, tuple[dict[str, object], bool]] = {
    "no_h0": ({}, False),
    "h0": ({}, True),
    "batch_first": ({"batch_first": True}, True),
    "bidir_2layer": ({"num_layers": 2, "bidirectional": True}, True),
    "proj": ({"proj_size": _P}, True),
}


def _module_cases() -> list[object]:
    return [
        pytest.param(kind, config, id=f"{kind}-{config}")
        for kind in _KINDS
        for config in _CONFIGS
        if config != "proj" or kind == "LSTM"
    ]


def _pair(
    ref: ModuleType, kind: str, kwargs: dict[str, object], device: str
) -> tuple[object, object]:
    """A Lucid module on ``device`` and a reference module, same weights."""
    ours = getattr(nn, kind)(_I, _H, **kwargs)
    theirs = getattr(ref.nn, kind)(_I, _H, **kwargs)
    named = dict(theirs.named_parameters())
    for name, p in ours.named_parameters():
        named[name].data = ref.tensor(p.numpy())
    return ours.to(device), theirs


def _close(got: object, want: object) -> None:
    """Same shape and values; ``want`` is a reference tensor."""
    assert want is not None and got is not None
    assert tuple(got.shape) == tuple(want.shape)
    np.testing.assert_allclose(got.numpy(), want.detach().numpy(), rtol=1e-4, atol=1e-5)


def _leaves(fw: ModuleType, arrays: dict[str, np.ndarray], device: str) -> dict[str, object]:
    return {
        name: fw.tensor(a, device=device, requires_grad=not name.startswith("w_"))
        for name, a in arrays.items()
    }


def _run_module(
    fw: ModuleType, module: object, kind: str, leaves: dict[str, object]
) -> tuple[list[object], dict[str, object]]:
    """Forward ``module`` and backpropagate a loss over every output; return
    the outputs and the gradients of every leaf and parameter."""
    states: tuple[object, ...] = ()
    if "h0" in leaves:
        states = (leaves["h0"], leaves["c0"]) if kind == "LSTM" else (leaves["h0"],)
    hx = (states if kind == "LSTM" else states[0]) if states else None
    out, final = module(leaves["x"], hx)
    finals = list(final) if kind == "LSTM" else [final]
    loss = (out * leaves["w_out"]).sum()
    for i, state in enumerate(finals):
        loss = loss + (state * leaves[f"w_s{i}"]).sum()
    loss.backward()
    grads = {n: t.grad for n, t in leaves.items() if not n.startswith("w_")}
    for name, p in module.named_parameters():
        grads[name] = p.grad
    return [out, *finals], grads


@pytest.mark.parity
@pytest.mark.parametrize("kind,config", _module_cases())
def test_unbatched_module_matches_reference(
    ref: ModuleType, kind: str, config: str, device: str
) -> None:
    kwargs, with_h0 = _CONFIGS[config]
    ours, theirs = _pair(ref, kind, kwargs, device)
    rows = int(kwargs.get("num_layers", 1)) * (2 if kwargs.get("bidirectional") else 1)
    dirs = 2 if kwargs.get("bidirectional") else 1
    rec = _P if config == "proj" else _H

    rng = np.random.default_rng(370)
    arrays = {"x": rng.standard_normal((_T, _I)).astype(np.float32)}
    if with_h0:
        arrays["h0"] = rng.standard_normal((rows, rec)).astype(np.float32)
        if kind == "LSTM":
            arrays["c0"] = rng.standard_normal((rows, _H)).astype(np.float32)
    arrays["w_out"] = rng.standard_normal((_T, dirs * rec)).astype(np.float32)
    arrays["w_s0"] = rng.standard_normal((rows, rec)).astype(np.float32)
    if kind == "LSTM":
        arrays["w_s1"] = rng.standard_normal((rows, _H)).astype(np.float32)

    got, got_grads = _run_module(lucid, ours, kind, _leaves(lucid, arrays, device))
    want, want_grads = _run_module(ref, theirs, kind, _leaves(ref, arrays, "cpu"))

    assert tuple(got[0].shape) == (_T, dirs * rec)
    for g, w in zip(got, want, strict=True):
        _close(g, w)
    assert got_grads.keys() == want_grads.keys()
    for name, grad in got_grads.items():
        _close(grad, want_grads[name])


@pytest.mark.parity
@pytest.mark.parametrize("with_hx", [False, True], ids=["no_hx", "hx"])
@pytest.mark.parametrize("kind", _CELLS)
def test_unbatched_cell_matches_reference(
    ref: ModuleType, kind: str, with_hx: bool, device: str
) -> None:
    ours, theirs = _pair(ref, kind, {}, device)
    lstm = kind == "LSTMCell"
    rng = np.random.default_rng(371)
    arrays = {"x": rng.standard_normal((_I,)).astype(np.float32)}
    if with_hx:
        arrays["h0"] = rng.standard_normal((_H,)).astype(np.float32)
        if lstm:
            arrays["c0"] = rng.standard_normal((_H,)).astype(np.float32)
    arrays["w_s0"] = rng.standard_normal((_H,)).astype(np.float32)
    if lstm:
        arrays["w_s1"] = rng.standard_normal((_H,)).astype(np.float32)

    def run(fw: ModuleType, cell: object, dev: str) -> tuple[list[object], dict]:
        leaves = _leaves(fw, arrays, dev)
        hx = None
        if with_hx:
            hx = (leaves["h0"], leaves["c0"]) if lstm else leaves["h0"]
        result = cell(leaves["x"], hx)
        outs = list(result) if lstm else [result]
        loss = sum((o * leaves[f"w_s{i}"]).sum() for i, o in enumerate(outs))
        loss.backward()
        grads = {n: t.grad for n, t in leaves.items() if not n.startswith("w_")}
        grads.update({n: p.grad for n, p in cell.named_parameters()})
        return outs, grads

    got, got_grads = run(lucid, ours, device)
    want, want_grads = run(ref, theirs, "cpu")
    assert all(tuple(g.shape) == (_H,) for g in got)
    for g, w in zip(got, want, strict=True):
        _close(g, w)
    assert got_grads.keys() == want_grads.keys()
    for name, grad in got_grads.items():
        _close(grad, want_grads[name])


# ── a state whose rank does not match the input's is refused ────────────────

# name → (input shape, state shape) — the state is right for the other rank.
_MISMATCH = {
    "unbatched_x_batched_h": ((_T, _I), (1, 1, _H)),
    "batched_x_unbatched_h": ((_T, _B, _I), (1, _H)),
}


def _states(fw: ModuleType, kind: str, shape: tuple[int, ...]) -> object:
    if kind.startswith("LSTM"):
        return (fw.zeros(*shape), fw.zeros(*shape))
    return fw.zeros(*shape)


@pytest.mark.parametrize("case", list(_MISMATCH))
@pytest.mark.parametrize("kind", _KINDS)
def test_mismatched_state_rank_is_refused(kind: str, case: str, device: str) -> None:
    x_shape, h_shape = _MISMATCH[case]
    module = getattr(nn, kind)(_I, _H).to(device)
    hx = _states(lucid, kind, h_shape)
    hx = tuple(t.to(device) for t in hx) if isinstance(hx, tuple) else hx.to(device)
    expected = (1, _H) if len(x_shape) == 2 else (1, _B, _H)
    with pytest.raises(ValueError, match=re.escape(f"h0 of shape {expected}")):
        module(lucid.randn(*x_shape, device=device), hx)


@pytest.mark.parity
@pytest.mark.parametrize("case", list(_MISMATCH))
@pytest.mark.parametrize("kind", _KINDS)
def test_reference_refuses_mismatched_state_rank(ref: ModuleType, kind: str, case: str) -> None:
    # Why the refusal above is the contract and not a gap.  The reference
    # raises RuntimeError; Lucid's taxonomy makes a bad shape a ValueError.
    x_shape, h_shape = _MISMATCH[case]
    module = getattr(ref.nn, kind)(_I, _H)
    with pytest.raises(RuntimeError, match="should also be"):
        module(ref.randn(*x_shape), _states(ref, kind, h_shape))


@pytest.mark.parametrize("rank", [1, 4])
@pytest.mark.parametrize("kind", _KINDS)
def test_input_rank_outside_two_and_three_is_refused(kind: str, rank: int, device: str) -> None:
    module = getattr(nn, kind)(_I, _H).to(device)
    shape = (_I,) if rank == 1 else (1, _T, _B, _I)
    with pytest.raises(ValueError, match=r"expected input to be 2-D .* or 3-D"):
        module(lucid.randn(*shape, device=device))


# name → (input shape, state shape, expected state shape)
_CELL_MISMATCH = {
    "unbatched_x_batched_h": ((_I,), (1, _H), (_H,)),
    "batched_x_unbatched_h": ((_B, _I), (_H,), (_B, _H)),
}


@pytest.mark.parametrize("case", list(_CELL_MISMATCH))
@pytest.mark.parametrize("kind", _CELLS)
def test_cell_mismatched_state_rank_is_refused(kind: str, case: str, device: str) -> None:
    x_shape, h_shape, expected = _CELL_MISMATCH[case]
    cell = getattr(nn, kind)(_I, _H).to(device)
    hx = _states(lucid, kind, h_shape)
    hx = tuple(t.to(device) for t in hx) if isinstance(hx, tuple) else hx.to(device)
    with pytest.raises(ValueError, match=re.escape(f"hx of shape {expected}")):
        cell(lucid.randn(*x_shape, device=device), hx)


@pytest.mark.parity
@pytest.mark.parametrize("case", list(_CELL_MISMATCH))
@pytest.mark.parametrize("kind", _CELLS)
def test_reference_cell_refuses_mismatched_state_rank(
    ref: ModuleType, kind: str, case: str
) -> None:
    x_shape, h_shape, _ = _CELL_MISMATCH[case]
    cell = getattr(ref.nn, kind)(_I, _H)
    with pytest.raises(RuntimeError):
        cell(ref.randn(*x_shape), _states(ref, kind, h_shape))


@pytest.mark.parametrize("kind", _CELLS)
def test_cell_input_rank_three_is_refused(kind: str, device: str) -> None:
    cell = getattr(nn, kind)(_I, _H).to(device)
    with pytest.raises(ValueError, match=r"expected input to be 1-D .* or 2-D"):
        cell(lucid.randn(1, _B, _I, device=device))
