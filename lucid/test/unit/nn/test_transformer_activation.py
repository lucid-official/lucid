"""Transformer layers use the ``activation`` they are given.

The encoder and decoder layers picked ``gelu`` when ``activation == "gelu"``
and ``relu`` for everything else: a callable, ``"silu"``, ``"tanh"`` and a
misspelling all ran relu, silently.  Both layers now resolve ``activation``
through one helper — the reference's strings ``"relu"`` / ``"gelu"``, or a
callable used as-is — and refuse any other string when they are built,
naming the accepted values.  The reference raises RuntimeError there;
Lucid raises ValueError, its class for a value outside the domain.
"""

from collections.abc import Callable
from types import ModuleType

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

_D, _NHEAD, _FF = 8, 2, 16


def _swish(t: object) -> object:
    return t * t.sigmoid()


# id → (Lucid activation, its counterpart built from the reference module)
_ACTIVATIONS: dict[str, tuple[object, Callable[[ModuleType], object]]] = {
    "relu": ("relu", lambda ref: "relu"),
    "gelu": ("gelu", lambda ref: "gelu"),
    "callable_gelu": (F.gelu, lambda ref: ref.nn.functional.gelu),
    "lambda": (_swish, lambda ref: _swish),
    "module": (nn.SiLU(), lambda ref: ref.nn.SiLU()),
}


def _layer(fw: ModuleType, which: str, activation: object) -> object:
    cls = getattr(
        fw.nn, "TransformerEncoderLayer" if which == "encoder" else "TransformerDecoderLayer"
    )
    return cls(_D, _NHEAD, _FF, dropout=0.0, activation=activation, batch_first=True)


def _run(fw: ModuleType, layer: object, which: str, device: str) -> np.ndarray:
    rng = np.random.default_rng(128)
    src = fw.tensor(rng.standard_normal((2, 5, _D)).astype(np.float32), device=device)
    if which == "encoder":
        out = layer(src)
    else:
        memory = fw.tensor(rng.standard_normal((2, 6, _D)).astype(np.float32), device=device)
        out = layer(src, memory)
    return np.asarray(out.detach().numpy())


@pytest.mark.parity
@pytest.mark.parametrize("act", list(_ACTIVATIONS))
@pytest.mark.parametrize("which", ["encoder", "decoder"])
def test_the_activation_matches_the_reference(
    ref: ModuleType, which: str, act: str, device: str
) -> None:
    mine_act, ref_act = _ACTIVATIONS[act]
    ours = _layer(lucid, which, mine_act)
    theirs = _layer(ref, which, ref_act(ref))
    theirs_params = list(theirs.parameters())
    ours_params = list(ours.parameters())
    assert len(ours_params) == len(theirs_params)
    with lucid.no_grad():
        for mine, src in zip(ours_params, theirs_params, strict=True):
            mine.copy_(lucid.tensor(src.detach().numpy()))
    ours.to(device).eval()
    theirs.eval()
    got = _run(lucid, ours, which, device)
    want = _run(ref, theirs, which, "cpu")
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("which", ["encoder", "decoder"])
def test_a_callable_is_not_replaced_by_relu(which: str) -> None:
    lucid.manual_seed(0)
    relu_layer = _layer(lucid, which, "relu")
    zero_layer = _layer(lucid, which, lambda t: t * 0.0)
    zero_layer.load_state_dict(relu_layer.state_dict())
    relu_layer.eval()
    zero_layer.eval()
    assert not np.allclose(
        _run(lucid, relu_layer, which, "cpu"), _run(lucid, zero_layer, which, "cpu")
    )


@pytest.mark.parametrize("activation", ["bogus", "silu", "tanh", "RELU"])
@pytest.mark.parametrize(
    "build",
    [
        lambda a: _layer(lucid, "encoder", a),
        lambda a: _layer(lucid, "decoder", a),
        lambda a: nn.Transformer(_D, _NHEAD, 1, 1, _FF, activation=a),
    ],
    ids=["encoder", "decoder", "transformer"],
)
def test_an_unknown_string_is_refused_when_built(
    build: Callable[[str], object], activation: str
) -> None:
    with pytest.raises(ValueError, match=r"\['gelu', 'relu'\]"):
        build(activation)


@pytest.mark.parametrize("which", ["encoder", "decoder"])
def test_a_non_callable_is_refused(which: str) -> None:
    with pytest.raises(TypeError, match="string or a callable"):
        _layer(lucid, which, 3)


def test_a_stack_keeps_the_prototype_activation() -> None:
    def act(t: lucid.Tensor) -> lucid.Tensor:
        return t * 0.5

    encoder = nn.TransformerEncoder(_layer(lucid, "encoder", act), 2)
    decoder = nn.TransformerDecoder(_layer(lucid, "decoder", act), 2)
    assert all(layer.activation is act for layer in encoder.layers)
    assert all(layer.activation is act for layer in decoder.layers)
