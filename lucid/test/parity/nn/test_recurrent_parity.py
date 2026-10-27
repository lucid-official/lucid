"""The recurrent layers, against the reference.

``nn/modules/rnn.py`` sat at 74.5%, and what was dark was every option
that changes the recurrence: a second layer, a backward direction, no
bias, ``batch_first=False``, a supplied initial state, ``proj_size``,
inter-layer dropout.

A recurrence is unusually good at hiding a mistake.  The output has the
right shape whether or not the backward direction was concatenated the
right way round, whether or not layer two received layer one's output,
and whether or not the state handed back is the one at the last step.
So the reference is the oracle for the values, and this file holds those
checks; the properties that do not need one — that carrying the state
across a split equals running the sequence whole — stay in
``lucid/test/unit/nn/test_recurrent.py`` so the fast tier runs them.
"""

import numpy as np
import pytest

import lucid
import lucid.nn as nn

pytestmark = pytest.mark.parity

X = np.random.default_rng(0).standard_normal((3, 5, 4)).astype(np.float32)


def _v(x):
    return np.asarray(x.numpy())


def _t(a):
    return lucid.tensor(np.asarray(a, dtype=np.float32))


def _mirror(lucid_module, ref_module, ref):
    """Give the reference module Lucid's weights, so only the recurrence
    is under test rather than two different initialisations."""
    target = ref_module.state_dict()
    for name, tensor in lucid_module.state_dict().items():
        if name in target:
            target[name].copy_(ref.tensor(_v(tensor)))
    ref_module.load_state_dict(target)


def _first(out):
    return out[0] if isinstance(out, tuple) else out


OPTIONS = [
    ("default", {}),
    ("two layers", {"num_layers": 2}),
    ("bidirectional", {"bidirectional": True}),
    ("two layers bidirectional", {"num_layers": 2, "bidirectional": True}),
    ("no bias", {"bias": False}),
    ("sequence first", {"batch_first": False}),
]


@pytest.mark.parametrize("kind", ["RNN", "LSTM", "GRU"])
@pytest.mark.parametrize("options", [o[1] for o in OPTIONS], ids=[o[0] for o in OPTIONS])
def test_the_recurrence_matches_the_reference(kind, options, ref):
    kwargs = dict(options)
    batch_first = kwargs.pop("batch_first", True)
    built = dict(input_size=4, hidden_size=6, batch_first=batch_first, **kwargs)

    lucid.manual_seed(0)
    mine = getattr(nn, kind)(**built)
    theirs = getattr(ref.nn, kind)(**built)
    _mirror(mine, theirs, ref)

    inputs = X if batch_first else X.transpose(1, 0, 2)
    got = _v(_first(mine(_t(inputs))))
    want = _first(theirs(ref.tensor(inputs))).detach().numpy()
    assert got.shape == want.shape
    assert np.allclose(got, want, atol=1e-4)


@pytest.mark.parametrize("nonlinearity", ["tanh", "relu"])
def test_the_plain_rnn_honours_its_nonlinearity(nonlinearity, ref):
    lucid.manual_seed(0)
    mine = nn.RNN(4, 6, batch_first=True, nonlinearity=nonlinearity)
    theirs = ref.nn.RNN(4, 6, batch_first=True, nonlinearity=nonlinearity)
    _mirror(mine, theirs, ref)
    assert np.allclose(
        _v(_first(mine(_t(X)))), _first(theirs(ref.tensor(X))).detach().numpy(), atol=1e-4
    )


def test_lstm_projects_its_hidden_state_when_asked(ref):
    """``proj_size`` makes the output narrower than the cell, which is
    the one option that changes the output *shape* as well as its
    values."""
    lucid.manual_seed(0)
    mine = nn.LSTM(4, 6, proj_size=3, batch_first=True)
    theirs = ref.nn.LSTM(4, 6, proj_size=3, batch_first=True)
    _mirror(mine, theirs, ref)
    got = _v(_first(mine(_t(X))))
    assert got.shape == (3, 5, 3)
    assert np.allclose(got, _first(theirs(ref.tensor(X))).detach().numpy(), atol=1e-4)


@pytest.mark.parametrize("kind", ["RNNCell", "LSTMCell", "GRUCell"])
def test_a_single_cell_matches_the_reference(kind, ref):
    lucid.manual_seed(0)
    mine = getattr(nn, kind)(4, 6)
    theirs = getattr(ref.nn, kind)(4, 6)
    _mirror(mine, theirs, ref)
    got, want = mine(_t(X[:, 0])), theirs(ref.tensor(X[:, 0]))
    if isinstance(got, tuple):
        for a, b in zip(got, want):
            assert np.allclose(_v(a), b.detach().numpy(), atol=1e-4)
    else:
        assert np.allclose(_v(got), want.detach().numpy(), atol=1e-4)
