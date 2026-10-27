"""A caller's initial state meets a packed batch in the caller's order.

A packed batch runs sorted by descending length, so ``hx`` — given in the
caller's batch order — has to be permuted by ``sorted_indices`` on the way
in, as the final state is permuted back by ``unsorted_indices`` on the way
out.  All three recurrent layers skipped the first half: with an unsorted
batch every sequence started from another sequence's state, and ``h_n``
came back wrong (GRU off by 0.29, RNN by 0.86) with nothing to show for it.

Each case compares outputs, final state and every gradient with the
reference: {RNN, GRU, LSTM} × {explicit hx, none} × {sorted, unsorted} ×
device.
"""

from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.nn.utils.rnn import PackedSequence, pack_padded_sequence
from lucid.test._fixtures.devices import _device_params

_T, _B, _I, _H = 5, 3, 4, 6
_ORDERS = {"sorted": [5, 3, 2], "unsorted": [3, 5, 2]}


def _pack(x: lucid.Tensor, lengths: list[int], device: str) -> PackedSequence:
    return pack_padded_sequence(x.to(device), lengths, enforce_sorted=lengths == _ORDERS["sorted"])


def _state(state: object) -> list[object]:
    return list(state) if isinstance(state, tuple) else [state]


@pytest.mark.parametrize("device", _device_params())
@pytest.mark.parametrize("order", list(_ORDERS))
@pytest.mark.parametrize("given", [True, False], ids=["hx", "no-hx"])
@pytest.mark.parametrize("kind", ["RNN", "GRU", "LSTM"])
def test_packed_hx_follows_the_callers_batch_order(
    ref: ModuleType, kind: str, given: bool, order: str, device: str
) -> None:
    lengths = _ORDERS[order]
    # Metal holds no float64; float32 there is checked to float32 precision.
    dtype, tol = (lucid.float64, 1e-10) if device == "cpu" else (lucid.float32, 1e-5)
    np_dt = np.float64 if dtype is lucid.float64 else np.float32
    rng = np.random.default_rng(11)
    x_np = rng.standard_normal((_T, _B, _I)).astype(np_dt)
    for b, n in enumerate(lengths):
        x_np[n:, b] = 0.0
    hx_np = [
        rng.standard_normal((1, _B, _H)).astype(np_dt) for _ in range(2 if kind == "LSTM" else 1)
    ]
    w_out = rng.standard_normal((sum(lengths), _H)).astype(np_dt)
    w_state = rng.standard_normal((1, _B, _H)).astype(np_dt)

    ours = getattr(nn, kind)(_I, _H, dtype=dtype)
    theirs = getattr(ref.nn, kind)(_I, _H).to(
        ref.float64 if dtype is lucid.float64 else ref.float32
    )
    for p, q in zip(ours.parameters(), theirs.parameters(), strict=True):
        q.data = ref.tensor(p.numpy())
    ours = ours.to(device)

    x = lucid.tensor(x_np, requires_grad=True)
    hx = [lucid.tensor(h, device=device, requires_grad=True) for h in hx_np]
    feed_hx = (tuple(hx) if kind == "LSTM" else hx[0]) if given else None
    out, state = ours(_pack(x, lengths, device), feed_hx)
    loss = (out.data * lucid.tensor(w_out, device=device)).sum()
    for s in _state(state):
        loss = loss + (s * lucid.tensor(w_state, device=device)).sum()
    loss.backward()

    rx = ref.tensor(x_np, requires_grad=True)
    rhx = [ref.tensor(h, requires_grad=True) for h in hx_np]
    rfeed = (tuple(rhx) if kind == "LSTM" else rhx[0]) if given else None
    rpacked = ref.nn.utils.rnn.pack_padded_sequence(rx, lengths, enforce_sorted=order == "sorted")
    rout, rstate = theirs(rpacked, rfeed)
    rloss = (rout.data * ref.tensor(w_out)).sum()
    for s in _state(rstate):
        rloss = rloss + (s * ref.tensor(w_state)).sum()
    rloss.backward()

    def close(got: lucid.Tensor, want: object, what: str) -> None:
        np.testing.assert_allclose(
            got.to("cpu").numpy(), want.detach().numpy(), rtol=tol, atol=tol, err_msg=what
        )

    close(out.data, rout.data, "output")
    for i, (s, rs) in enumerate(zip(_state(state), _state(rstate), strict=True)):
        close(s, rs, f"state {i}")
    close(x.grad, rx.grad, "x.grad")
    if given:
        for i, (h, rh) in enumerate(zip(hx, rhx, strict=True)):
            close(h.grad, rh.grad, f"hx[{i}].grad")
    for p, q in zip(ours.parameters(), theirs.parameters(), strict=True):
        close(p.grad, q.grad, "parameter grad")
