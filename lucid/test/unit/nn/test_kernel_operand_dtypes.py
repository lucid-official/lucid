"""No engine kernel reads one operand in another operand's dtype (LCD-385).

A CPU kernel is handed raw bytes and a dtype for the call, and reads every
operand as that dtype.  An operand in any other dtype was read as such:
a float16 target as float32 bytes, a float32 weight past its end as
float64.  Nothing raised and the answers were wrong.  The holes were the
loss and normalisation doors with no dtype-agreement check (``huber``,
``bce``, ``bce_with_logits``, the class weight of ``cross_entropy`` and
``nll``, ``global_response_norm``, the attention mask) and half-widening
paths that widened some operands but not others (the attention mask, the
class weight in backward).  Metal promoted some of the same calls and
refused others, so the devices disagreed as well.

Each cell below calls one engine kernel with one floating operand in a
dtype other than its siblings', on every device.  The call must either be
refused (a ``TypeError``, or ``NotImplementedError`` for a kernel
with no half path) or return the float64 answer for the same
values.  A wrong number is the bug.
"""

import itertools
from collections.abc import Callable, Sequence

import pytest

import lucid
from lucid._C import engine as _C_engine
from lucid._dispatch import _wrap
from lucid.test._fixtures.devices import metal_available

_DEVICES = ["cpu", "metal"] if metal_available() else ["cpu"]
_DTYPES = {"f16": lucid.float16, "bf16": lucid.bfloat16, "f32": lucid.float32, "f64": lucid.float64}

_Impl = _C_engine.TensorImpl
_Call = Callable[..., _Impl]


def _wave(*shape: int) -> lucid.Tensor:
    """Deterministic values in [-1, 1] with no repeats along any axis."""
    n = 1
    for s in shape:
        n *= s
    return lucid.sin(lucid.arange(n, dtype=lucid.float64) * 0.7 + 0.3).reshape(*shape)


def _normal(*shape: int) -> lucid.Tensor:
    return _wave(*shape) * 1.5


def _unit(*shape: int) -> lucid.Tensor:
    """Values in [0.1, 0.9]: probabilities and weights."""
    return _wave(*shape) * 0.4 + 0.5


_MASK = lucid.zeros(2, 1, 3, 3).double()
_MASK[..., 0, 1] = -5.0

# name -> (floating operands, non-floating extras, call).  Extras are moved
# to the device but keep their dtype; the call gets floats then extras.
_CASES: dict[str, tuple[list[lucid.Tensor], list[lucid.Tensor], _Call]] = {
    "huber_loss": (
        [_normal(3, 4), _normal(3, 4)],
        [],
        lambda a, b: _C_engine.nn.huber_loss(a, b, 1.0, 0),
    ),
    "mse_loss": ([_normal(3, 4), _normal(3, 4)], [], lambda a, b: _C_engine.nn.mse_loss(a, b, 0)),
    "bce_loss": (
        [_unit(3, 4), _unit(3, 4), _unit(3, 4)],
        [],
        lambda a, b, w: _C_engine.nn.bce_loss(a, b, w, 0),
    ),
    "bce_with_logits": (
        [_normal(3, 4), _unit(3, 4), _unit(3, 4), _unit(4)],
        [],
        lambda a, b, w, pw: _C_engine.nn.bce_with_logits(a, b, w, pw, 0),
    ),
    "cross_entropy_weight": (
        [_normal(3, 4), _unit(4)],
        [lucid.tensor([0, 2, 1])],
        lambda a, w, t: _C_engine.nn.cross_entropy_loss(a, t, w, 0),
    ),
    "nll_loss_weight": (
        [_normal(3, 4), _unit(4)],
        [lucid.tensor([0, 2, 1])],
        lambda a, w, t: _C_engine.nn.nll_loss(a, t, w, 0),
    ),
    "linear": ([_normal(3, 4), _normal(5, 4), _normal(5)], [], _C_engine.nn.linear),
    "bilinear_layer": (
        [_normal(3, 4), _normal(3, 4), _normal(2, 4, 4)],
        [],
        _C_engine.nn.bilinear_layer,
    ),
    "conv1d": (
        [_normal(2, 3, 8), _normal(4, 3, 3), _normal(4)],
        [],
        lambda x, w, b: _C_engine.nn.conv1d(x, w, b, 1, 1, 1, 1),
    ),
    "conv2d": (
        [_normal(2, 3, 6, 6), _normal(4, 3, 3, 3), _normal(4)],
        [],
        lambda x, w, b: _C_engine.nn.conv2d(x, w, b, 1, 1, 1, 1, 1, 1, 1),
    ),
    "conv_transpose2d": (
        [_normal(2, 3, 5, 5), _normal(3, 4, 3, 3), _normal(4)],
        [],
        lambda x, w, b: _C_engine.nn.conv_transpose2d(x, w, b, 1, 1, 1, 1, 0, 0, 1, 1, 1),
    ),
    "layer_norm": (
        [_normal(3, 4), _normal(4), _normal(4)],
        [],
        lambda x, w, b: _C_engine.nn.layer_norm(x, w, b, 1e-5),
    ),
    "group_norm": (
        [_normal(2, 4, 3), _normal(4), _normal(4)],
        [],
        lambda x, w, b: _C_engine.nn.group_norm(x, w, b, 2, 1e-5),
    ),
    "rms_norm": ([_normal(3, 4), _normal(4)], [], lambda x, w: _C_engine.nn.rms_norm(x, w, 1e-5)),
    "batch_norm": (
        [_normal(4, 3, 2, 2), _normal(3), _normal(3)],
        [],
        lambda x, w, b: _C_engine.nn.batch_norm(x, w, b, 1e-5),
    ),
    "batch_norm_eval": (
        [_normal(4, 3, 2, 2), _unit(3), _unit(3), _normal(3), _normal(3)],
        [],
        lambda x, rm, rv, w, b: _C_engine.nn.batch_norm_eval(x, rm, rv, w, b, 1e-5),
    ),
    "global_response_norm": (
        [_normal(2, 4, 3, 3), _normal(4), _normal(4)],
        [],
        lambda x, g, b: _C_engine.nn.global_response_norm(x, g, b, 1e-6),
    ),
    "grid_sample": (
        [_normal(1, 2, 4, 4), _unit(1, 3, 3, 2) * 2.0 - 1.0],
        [],
        lambda x, g: _C_engine.nn.grid_sample(x, g, 0, 0, True),
    ),
    "sdpa_mask": (
        [_normal(2, 1, 3, 4), _normal(2, 1, 3, 4), _normal(2, 1, 3, 4), _MASK],
        [],
        lambda q, k, v, m: _C_engine.nn.scaled_dot_product_attention(q, k, v, m, 0.5, False),
    ),
    "sdpa_weights_mask": (
        [_normal(2, 1, 3, 4), _normal(2, 1, 3, 4), _normal(2, 1, 3, 4), _MASK],
        [],
        lambda q, k, v, m: _C_engine.nn.scaled_dot_product_attention_with_weights(
            q, k, v, m, 0.5, False
        )[0],
    ),
}


def _run(name: str, dtypes: Sequence[str], device: str) -> lucid.Tensor:
    floats, extras, call = _CASES[name]
    args = [f.to(_DTYPES[d]).to(device)._impl for f, d in zip(floats, dtypes, strict=True)]
    args += [e.to(device)._impl for e in extras]
    return _wrap(call(*args))


def _cells() -> list[object]:
    cells = []
    for name, (floats, _, _) in _CASES.items():
        for device, base in itertools.product(_DEVICES, ("f32", "f16")):
            for i, odd in itertools.product(range(len(floats)), _DTYPES):
                if odd == base or (device == "metal" and odd == "f64"):
                    continue
                dtypes = [base] * len(floats)
                dtypes[i] = odd
                cells.append(
                    pytest.param(
                        name, device, tuple(dtypes), id=f"{name}-{device}-{'-'.join(dtypes)}"
                    )
                )
    return cells


@pytest.mark.parametrize(("name", "device", "dtypes"), _cells())
def test_a_mixed_operand_is_refused_or_read_right(
    name: str, device: str, dtypes: tuple[str, ...]
) -> None:
    want = _run(name, ["f64"] * len(dtypes), "cpu")
    try:
        got = _run(name, dtypes, device)
    except (TypeError, NotImplementedError):
        # Refused: a dtype mismatch, or a half kernel the CPU lacks.
        return
    err = (got.cpu().double() - want).abs().max().item()
    scale = max(1.0, want.abs().max().item())
    assert err <= 0.06 * scale, f"{name} {dtypes} on {device}: off by {err:.3g}"
