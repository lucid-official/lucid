"""A half-precision attention reads its mask as the mask's own dtype (LCD-385).

The CPU kernel has no half path, so it widened q, k and v to float32 and
reused the float kernel.  The mask went through unwidened, and the float
kernel read its float16 bytes as float32: the scores got garbage added,
and the read ran past the end of the mask's buffer.  Every masked
float16 / bfloat16 attention on the CPU was wrong by up to 1.6.  That
included a boolean mask, since ``F.scaled_dot_product_attention`` turns
one into an additive mask in the query's dtype.  It was also the source
of the LCD-129 flake: the out-of-bounds read differed from run to run.

Each cell runs {SDPA, MultiheadAttention} x {bool, additive, zero} mask x
{float16, bfloat16} on every device.  A float mask in another dtype than the
queries (float32 or float16 against half queries) is cast at the public
door, and its gradient comes back in its own dtype.  The output and every gradient,
including the additive mask's, must match the same call in float32 on the
same rounded values.  The ``parity`` cells hold them to the reference too.
"""

import copy
import math
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._C import engine as _C_engine
from lucid.test._fixtures.devices import metal_available

_DEVICES = ["cpu", "metal"] if metal_available() else ["cpu"]
_HALVES = {"float16": lucid.float16, "bfloat16": lucid.bfloat16}
_TOL = {"float16": 2e-2, "bfloat16": 6e-2}
_KINDS = ["bool", "additive", "zero"]

N, H, L, S, E = 2, 2, 3, 4, 8
D = E // H


def _values(seed: int, *shape: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _rounded(a: np.ndarray, dtype: str) -> np.ndarray:
    """``a`` rounded through ``dtype``, so every computation sees one input."""
    return lucid.tensor(a).to(_HALVES[dtype]).to(lucid.float32).numpy()


def _mask_values(kind: str, shape: tuple[int, ...]) -> np.ndarray:
    """``True`` attends for ``bool``; an additive bias otherwise."""
    if kind == "bool":
        keep = np.ones(shape, dtype=bool)
        keep[..., 0, 1] = False
        keep[..., -1, 0] = False
        return keep
    if kind == "zero":
        return np.zeros(shape, dtype=np.float32)
    bias = _values(7, *shape) * 2.0
    bias[..., 0, 1] = -5.0
    return bias


def _close(got: lucid.Tensor | None, want: Any, dtype: str, what: str) -> None:
    assert got is not None, f"{what}: no gradient"
    g = got.detach().to(lucid.float32).cpu().numpy()
    w = np.asarray(want, dtype=np.float32)
    assert np.isfinite(g).all(), f"{what}: non-finite"
    np.testing.assert_allclose(
        g, w, rtol=0, atol=_TOL[dtype] * max(1.0, float(np.abs(w).max())), err_msg=what
    )


def _sdpa(
    arrays: list[np.ndarray], mask: np.ndarray, kind: str, dtype: Any, device: str
) -> tuple[lucid.Tensor, list[lucid.Tensor]]:
    """Output and (q, k, v[, mask]) gradients for one SDPA call."""
    ins = [lucid.tensor(a, dtype=dtype, device=device, requires_grad=True) for a in arrays]
    if kind == "bool":
        m = lucid.tensor(mask, device=device)
    else:
        m = lucid.tensor(mask, dtype=dtype, device=device, requires_grad=True)
        ins.append(m)
    out = F.scaled_dot_product_attention(*ins[:3], attn_mask=m)
    upstream = lucid.tensor(_values(5, *out.shape), dtype=dtype, device=device)
    (out * upstream).sum().backward()
    return out, [t.grad for t in ins]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", list(_HALVES))
@pytest.mark.parametrize("kind", _KINDS)
def test_sdpa_matches_float32(device: str, dtype: str, kind: str) -> None:
    arrays = [_rounded(_values(i, N, H, n, D), dtype) for i, n in enumerate((L, S, S))]
    mask = _mask_values(kind, (N, H, L, S))
    if kind != "bool":
        mask = _rounded(mask, dtype)
    got, got_grads = _sdpa(arrays, mask, kind, _HALVES[dtype], device)
    want, want_grads = _sdpa(arrays, mask, kind, lucid.float32, device)
    _close(got, want.numpy(), dtype, "output")
    names = ["query", "key", "value", "mask"]
    for name, g, w in zip(names, got_grads, want_grads, strict=False):
        assert w is not None
        _close(g, w.cpu().numpy(), dtype, f"{name} gradient")


def test_a_zero_mask_is_no_mask() -> None:
    q, k, v = (
        lucid.tensor(_values(i, N, H, n, D), dtype=lucid.float16) for i, n in enumerate((L, S, S))
    )
    with_mask = F.scaled_dot_product_attention(
        q, k, v, attn_mask=lucid.zeros(N, H, L, S, dtype=lucid.float16)
    )
    without = F.scaled_dot_product_attention(q, k, v)
    assert (with_mask.numpy() == without.numpy()).all()


def _mha(
    module: nn.MultiheadAttention,
    arrays: list[np.ndarray],
    mask: np.ndarray,
    kind: str,
    dtype: Any,
    device: str,
) -> tuple[lucid.Tensor, list[lucid.Tensor]]:
    ins = [lucid.tensor(a, dtype=dtype, device=device, requires_grad=True) for a in arrays]
    if kind == "bool":
        m = lucid.tensor(~mask, device=device)  # MHA's bool mask drops on True
    else:
        m = lucid.tensor(mask, dtype=dtype, device=device)
    out, _ = module(*ins, attn_mask=m, need_weights=False)
    upstream = lucid.tensor(_values(6, *out.shape), dtype=dtype, device=device)
    (out * upstream).sum().backward()
    return out, [t.grad for t in ins]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", list(_HALVES))
@pytest.mark.parametrize("kind", _KINDS)
def test_mha_matches_float32(device: str, dtype: str, kind: str) -> None:
    half = nn.MultiheadAttention(E, H, batch_first=True).to(device, dtype=_HALVES[dtype])
    full = copy.deepcopy(half).to(dtype=lucid.float32)
    arrays = [_rounded(_values(10 + i, N, n, E), dtype) for i, n in enumerate((L, S, S))]
    mask = _mask_values(kind, (L, S))
    if kind != "bool":
        mask = _rounded(mask, dtype)
    got, got_grads = _mha(half, arrays, mask, kind, _HALVES[dtype], device)
    want, want_grads = _mha(full, arrays, mask, kind, lucid.float32, device)
    _close(got, want.numpy(), dtype, "output")
    for name, g, w in zip(("query", "key", "value"), got_grads, want_grads, strict=True):
        assert w is not None
        _close(g, w.cpu().numpy(), dtype, f"{name} gradient")


@pytest.mark.parity
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", list(_HALVES))
@pytest.mark.parametrize("kind", _KINDS)
def test_sdpa_matches_the_reference(ref: Any, device: str, dtype: str, kind: str) -> None:
    arrays = [_rounded(_values(i, N, H, n, D), dtype) for i, n in enumerate((L, S, S))]
    mask = _mask_values(kind, (N, H, L, S))
    if kind != "bool":
        mask = _rounded(mask, dtype)
    got, got_grads = _sdpa(arrays, mask, kind, _HALVES[dtype], device)

    theirs = [ref.from_numpy(a).requires_grad_() for a in arrays]
    ref_mask = ref.from_numpy(mask)
    if kind != "bool":
        ref_mask.requires_grad_()
        theirs.append(ref_mask)
    want = ref.nn.functional.scaled_dot_product_attention(
        *theirs[:3], attn_mask=ref_mask, scale=1.0 / math.sqrt(D)
    )
    (want * ref.from_numpy(_values(5, *want.shape))).sum().backward()

    _close(got, want.detach().numpy(), dtype, "output")
    names = ["query", "key", "value", "mask"]
    for name, g, t in zip(names, got_grads, theirs, strict=False):
        _close(g, t.grad.numpy(), dtype, f"{name} gradient")


def _ref_mha(ref: Any, ours: nn.MultiheadAttention) -> Any:
    """The reference module holding ``ours``'s parameters."""
    theirs = ref.nn.MultiheadAttention(E, H, batch_first=True)
    with ref.no_grad():
        for mine, name in (
            ("in_proj_weight", "in_proj_weight"),
            ("in_proj_bias", "in_proj_bias"),
            ("out_proj_weight", "out_proj.weight"),
            ("out_proj_bias", "out_proj.bias"),
        ):
            value = getattr(ours, mine).detach().to(lucid.float32).cpu().numpy()
            theirs.get_parameter(name).copy_(ref.from_numpy(value))
    return theirs


@pytest.mark.parity
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", list(_HALVES))
@pytest.mark.parametrize("kind", _KINDS)
def test_mha_matches_the_reference(ref: Any, device: str, dtype: str, kind: str) -> None:
    ours = nn.MultiheadAttention(E, H, batch_first=True).to(device, dtype=_HALVES[dtype])
    theirs = _ref_mha(ref, ours)
    arrays = [_rounded(_values(10 + i, N, n, E), dtype) for i, n in enumerate((L, S, S))]
    mask = _mask_values(kind, (L, S))
    if kind != "bool":
        mask = _rounded(mask, dtype)
    got, got_grads = _mha(ours, arrays, mask, kind, _HALVES[dtype], device)

    ref_in = [ref.from_numpy(a).requires_grad_() for a in arrays]
    ref_mask = ref.from_numpy(~mask if kind == "bool" else mask)
    want, _ = theirs(*ref_in, attn_mask=ref_mask, need_weights=False)
    (want * ref.from_numpy(_values(6, *want.shape))).sum().backward()

    _close(got, want.detach().numpy(), dtype, "output")
    for name, g, t in zip(("query", "key", "value"), got_grads, ref_in, strict=True):
        _close(g, t.grad.numpy(), dtype, f"{name} gradient")


# ── a float mask in another dtype than the queries ─────────────────────────
#
# The engine refuses an additive mask whose dtype is not the queries'; the
# public door casts it (the reference takes a float32 bias against half
# queries).  The cast is recorded, so a learnable mask's gradient comes
# back in the mask's own dtype.

_MASK_DTYPES = {"float32": lucid.float32, "float16": lucid.float16}
_ENTRIES = ["sdpa", "mha"]


def _mixed_shapes(entry: str) -> tuple[list[tuple[int, ...]], tuple[int, ...]]:
    if entry == "sdpa":
        return [(N, H, n, D) for n in (L, S, S)], (N, H, L, S)
    return [(N, n, E) for n in (L, S, S)], (L, S)


def _mixed_inputs(entry: str, dtype: str, mask_dtype: str) -> tuple[list[np.ndarray], np.ndarray]:
    """Query/key/value rounded through ``dtype``; the bias through its own."""
    shapes, mask_shape = _mixed_shapes(entry)
    arrays = [_rounded(_values(20 + i, *sh), dtype) for i, sh in enumerate(shapes)]
    bias = _mask_values("additive", mask_shape)
    bias = lucid.tensor(bias).to(_MASK_DTYPES[mask_dtype]).to(lucid.float32).numpy()
    return arrays, bias


def _mixed_run(
    module: nn.MultiheadAttention | None,
    arrays: list[np.ndarray],
    bias: np.ndarray,
    dtype: Any,
    mask_dtype: Any,
    device: str,
) -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor]:
    """Output, query gradient and mask gradient."""
    q, k, v = (lucid.tensor(a, dtype=dtype, device=device, requires_grad=True) for a in arrays)
    m = lucid.tensor(bias, dtype=mask_dtype, device=device, requires_grad=True)
    if module is None:
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
    else:
        out, _ = module(q, k, v, attn_mask=m, need_weights=False)
    upstream = lucid.tensor(_values(8, *out.shape), dtype=dtype, device=device)
    (out * upstream).sum().backward()
    assert q.grad is not None and m.grad is not None
    return out, q.grad, m.grad


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", list(_HALVES))
@pytest.mark.parametrize("mask_dtype", list(_MASK_DTYPES))
@pytest.mark.parametrize("entry", _ENTRIES)
def test_a_float_mask_in_another_dtype_is_cast(
    entry: str, mask_dtype: str, dtype: str, device: str
) -> None:
    arrays, bias = _mixed_inputs(entry, dtype, mask_dtype)
    half = None
    full = None
    if entry == "mha":
        half = nn.MultiheadAttention(E, H, batch_first=True).to(device, dtype=_HALVES[dtype])
        full = copy.deepcopy(half).to(dtype=lucid.float32)
    got, gq, gm = _mixed_run(half, arrays, bias, _HALVES[dtype], _MASK_DTYPES[mask_dtype], device)
    want, wq, wm = _mixed_run(full, arrays, bias, lucid.float32, lucid.float32, device)
    assert got.dtype == _HALVES[dtype]
    assert gm.dtype == _MASK_DTYPES[mask_dtype], "mask gradient left its dtype"
    _close(got, want.cpu().numpy(), dtype, "output")
    _close(gq, wq.cpu().numpy(), dtype, "query gradient")
    _close(gm, wm.cpu().numpy(), dtype, "mask gradient")


@pytest.mark.parity
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("dtype", list(_HALVES))
@pytest.mark.parametrize("mask_dtype", list(_MASK_DTYPES))
@pytest.mark.parametrize("entry", _ENTRIES)
def test_a_float_mask_in_another_dtype_matches_the_reference(
    ref: Any, entry: str, mask_dtype: str, dtype: str, device: str
) -> None:
    arrays, bias = _mixed_inputs(entry, dtype, mask_dtype)
    ours = None
    if entry == "mha":
        ours = nn.MultiheadAttention(E, H, batch_first=True).to(device, dtype=_HALVES[dtype])
    got, gq, gm = _mixed_run(ours, arrays, bias, _HALVES[dtype], _MASK_DTYPES[mask_dtype], device)

    q, k, v = (ref.from_numpy(a).requires_grad_() for a in arrays)
    m = ref.from_numpy(bias).requires_grad_()
    if ours is None:
        want = ref.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=m)
    else:
        want, _ = _ref_mha(ref, ours)(q, k, v, attn_mask=m, need_weights=False)
    (want * ref.from_numpy(_values(8, *want.shape))).sum().backward()

    _close(got, want.detach().numpy(), dtype, "output")
    _close(gq, q.grad.numpy(), dtype, "query gradient")
    _close(gm, m.grad.numpy(), dtype, "mask gradient")


def test_the_cpu_kernel_refuses_a_mask_it_would_read_past() -> None:
    """The kernel reads one score slice of mask or all of them; anything else
    was read past its end.  The engine is reachable directly, so it refuses."""
    q = lucid.tensor(_values(0, N, H, L, D))
    padding = lucid.zeros(N, 1, 1, S)  # broadcasts over heads and queries
    with pytest.raises(ValueError, match="attn_mask has"):
        _C_engine.nn.scaled_dot_product_attention(
            q._impl, q._impl, q._impl, padding._impl, 0.5, False
        )
