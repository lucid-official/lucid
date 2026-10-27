"""RNN, GRU and LSTM apply their ``dropout`` between stacked layers.

GRU stored ``dropout`` and never read it, and RNN did not even store it:
both ran identically in train and eval mode whatever ``dropout`` said, while
LSTM applied it.  All three now go through one helper — dropout on each
layer's output except the last, in training only — and validate ``dropout``
the way the reference does: a number in ``[0, 1]``, with a warning when a
single layer leaves nowhere to apply it.
"""

from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn as nn

_KINDS = ("RNN", "GRU", "LSTM")
_I, _H, _T, _B = 4, 6, 7, 3


def _x(device: str) -> lucid.Tensor:
    rng = np.random.default_rng(128)
    return lucid.tensor(rng.standard_normal((_T, _B, _I)).astype(np.float32), device=device)


def _final_hidden(kind: str, state: object) -> np.ndarray:
    h = state[0] if kind == "LSTM" else state
    return np.asarray(h.numpy())


@pytest.mark.parametrize("kind", _KINDS)
def test_train_mode_draws_a_fresh_mask_per_call(kind: str, device: str) -> None:
    module = getattr(nn, kind)(_I, _H, num_layers=2, dropout=0.5).to(device)
    module.train()
    x = _x(device)
    out_a, state_a = module(x)
    out_b, state_b = module(x)
    assert not np.allclose(out_a.numpy(), out_b.numpy())
    # The mask sits between the layers: layer 0's own state never sees it.
    np.testing.assert_array_equal(_final_hidden(kind, state_a)[0], _final_hidden(kind, state_b)[0])


@pytest.mark.parametrize("kind", _KINDS)
def test_the_last_layer_output_is_not_dropped(kind: str, device: str) -> None:
    module = getattr(nn, kind)(_I, _H, num_layers=2, dropout=0.9).to(device)
    module.train()
    out, _ = module(_x(device))
    assert np.count_nonzero(out.numpy() == 0.0) == 0


@pytest.mark.parametrize("kind", _KINDS)
def test_eval_mode_is_dropout_zero(kind: str, device: str) -> None:
    dropped = getattr(nn, kind)(_I, _H, num_layers=2, dropout=0.5)
    plain = getattr(nn, kind)(_I, _H, num_layers=2, dropout=0.0)
    plain.load_state_dict(dropped.state_dict())
    dropped.to(device).eval()
    plain.to(device).eval()
    x = _x(device)
    np.testing.assert_array_equal(dropped(x)[0].numpy(), plain(x)[0].numpy())


def test_the_mask_is_per_element_and_rescales_survivors(device: str) -> None:
    # Layer 1 is relu(I @ x): it passes the dropped layer-0 output through
    # unchanged, so the output is exactly dropout(layer-0 output).
    width, p = 8, 0.5
    module = nn.RNN(width, width, num_layers=2, nonlinearity="relu", dropout=p)
    single = nn.RNN(width, width, num_layers=1, nonlinearity="relu")
    state = module.state_dict()
    for name, value in state.items():
        if name.endswith("_l1"):
            eye = np.eye(width, dtype=np.float32) if "weight_ih" in name else None
            state[name] = lucid.tensor(
                eye if eye is not None else np.zeros(value.shape, np.float32)
            )
    module.load_state_dict(state)
    single.load_state_dict({k: v for k, v in state.items() if k.endswith("_l0")})
    module.to(device).train()
    single.to(device).eval()

    rng = np.random.default_rng(1)
    x = lucid.tensor(rng.standard_normal((40, 16, width)).astype(np.float32) + 1.0)
    x = x.to(device)
    layer0 = single(x)[0].numpy()
    out = module(x)[0].numpy()
    live = layer0 > 0
    kept = np.isclose(out, layer0 / (1.0 - p), rtol=1e-5, atol=1e-6)
    zeroed = out == 0.0
    assert np.all(kept | zeroed)
    fraction = np.count_nonzero(zeroed & live) / np.count_nonzero(live)
    assert 0.4 < fraction < 0.6


@pytest.mark.parametrize("kind", _KINDS)
def test_a_single_layer_warns_that_dropout_has_nowhere_to_go(kind: str) -> None:
    with pytest.warns(UserWarning, match="num_layers greater than 1"):
        getattr(nn, kind)(_I, _H, num_layers=1, dropout=0.5)


@pytest.mark.parametrize("dropout", [-0.1, 1.5, float("nan"), True])
@pytest.mark.parametrize("kind", _KINDS)
def test_dropout_outside_the_unit_interval_is_refused(kind: str, dropout: float) -> None:
    with pytest.raises(ValueError, match="dropout"):
        getattr(nn, kind)(_I, _H, num_layers=2, dropout=dropout)


@pytest.mark.parity
@pytest.mark.parametrize("bidirectional", [False, True], ids=["uni", "bidir"])
@pytest.mark.parametrize("kind", _KINDS)
def test_dropout_one_matches_the_reference(
    ref: ModuleType, kind: str, bidirectional: bool, device: str
) -> None:
    # p = 1 drops all of layer 0's output, so the result is deterministic
    # and has to match the reference exactly — including that layer 0's
    # final state is untouched.
    kwargs = dict(num_layers=2, dropout=1.0, bidirectional=bidirectional)
    ours = getattr(nn, kind)(_I, _H, **kwargs)
    theirs = getattr(ref.nn, kind)(_I, _H, **kwargs)
    named = dict(theirs.named_parameters())
    for name, p in ours.named_parameters():
        named[name].data = ref.tensor(p.numpy())
    ours.to(device).train()
    theirs.train()

    x = _x("cpu").numpy()
    got_out, got_state = ours(lucid.tensor(x, device=device))
    want_out, want_state = theirs(ref.tensor(x))
    np.testing.assert_allclose(got_out.numpy(), want_out.detach().numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        _final_hidden(kind, got_state),
        (want_state[0] if kind == "LSTM" else want_state).detach().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
