"""Parity: multi-head attention's key-appending and projection options.

``add_bias_kv`` and ``add_zero_attn`` append key rows; the reference widens
the attention and key-padding masks to match, and Lucid's module used to
leave them, failing on every mask.  The functional form now reaches the
same module with ``add_zero_attn`` and separate Q / K / V weights.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as lnn
import lucid.nn.functional as LF

E, H, L, S, N = 8, 2, 3, 5, 2

_FUSED = [
    ("in_proj_weight", "in_proj_weight"),
    ("in_proj_bias", "in_proj_bias"),
    ("out_proj_weight", "out_proj.weight"),
    ("out_proj_bias", "out_proj.bias"),
]
_CONFIGS = {
    "bias_kv": {"add_bias_kv": True},
    "zero_attn": {"add_zero_attn": True},
    "both": {"add_bias_kv": True, "add_zero_attn": True},
}
_MASKS = {
    "none": (False, False),
    "key_padding": (True, False),
    "attn": (False, True),
    "both": (True, True),
}


def _mask_arrays() -> tuple[np.ndarray, np.ndarray]:
    key_padding = np.zeros((N, S), dtype=bool)
    key_padding[0, -1] = True
    attn = np.zeros((L, S), dtype=bool)
    attn[0, 1] = True
    return key_padding, attn


@pytest.mark.parity
@pytest.mark.parametrize("masks", list(_MASKS), ids=list(_MASKS))
@pytest.mark.parametrize("config", list(_CONFIGS), ids=list(_CONFIGS))
def test_the_module_matches_with_any_mask(ref: Any, config: str, masks: str) -> None:
    cfg = _CONFIGS[config]
    rng = np.random.default_rng(0)
    theirs = ref.nn.MultiheadAttention(E, H, **cfg).eval()
    ours = lnn.MultiheadAttention(E, H, **cfg).eval()
    state = theirs.state_dict()
    pairs = _FUSED + (
        [("bias_k", "bias_k"), ("bias_v", "bias_v")] if cfg.get("add_bias_kv") else []
    )
    for mine, theirs_name in pairs:
        getattr(ours, mine)._impl = lucid.from_numpy(
            state[theirs_name].detach().numpy().copy()
        )._impl
    q = rng.standard_normal((L, N, E)).astype(np.float32)
    kv = rng.standard_normal((S, N, E)).astype(np.float32)
    key_padding, attn = _mask_arrays()
    use_kpm, use_attn = _MASKS[masks]
    mine_kw: dict[str, Any] = {}
    ref_kw: dict[str, Any] = {}
    if use_kpm:
        mine_kw["key_padding_mask"] = lucid.from_numpy(key_padding)
        ref_kw["key_padding_mask"] = ref.from_numpy(key_padding)
    if use_attn:
        mine_kw["attn_mask"] = lucid.from_numpy(attn)
        ref_kw["attn_mask"] = ref.from_numpy(attn)
    with ref.no_grad():
        want, want_w = theirs(ref.from_numpy(q), ref.from_numpy(kv), ref.from_numpy(kv), **ref_kw)
    got, got_w = ours(lucid.from_numpy(q), lucid.from_numpy(kv), lucid.from_numpy(kv), **mine_kw)
    np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=0, atol=1e-5)
    np.testing.assert_allclose(got_w.numpy(), want_w.numpy(), rtol=0, atol=1e-6)


_FUNCTIONAL = {
    "zero_attn": {"add_zero_attn": True},
    "zero_attn_bias_kv": {"add_zero_attn": True, "bias_kv": True},
    "separate_kdim6_vdim4": {"separate": True, "kdim": 6, "vdim": 4},
    "separate_stacked": {"separate": True},
    "everything": {"separate": True, "kdim": 6, "vdim": 4, "add_zero_attn": True, "bias_kv": True},
}


@pytest.mark.parity
@pytest.mark.parametrize("case", list(_FUNCTIONAL), ids=list(_FUNCTIONAL))
def test_the_functional_form_matches(ref: Any, case: str) -> None:
    spec = _FUNCTIONAL[case]
    separate = bool(spec.get("separate", False))
    kdim, vdim = int(spec.get("kdim", E)), int(spec.get("vdim", E))
    rng = np.random.default_rng(0)

    def draw(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float32)

    q, k, v = draw(L, N, E), draw(S, N, kdim), draw(S, N, vdim)
    weights = {
        "in_proj_weight": None if separate else draw(3 * E, E),
        "in_proj_bias": draw(3 * E),
        "out_proj_weight": draw(E, E),
        "out_proj_bias": draw(E),
        "bias_k": draw(1, 1, E) if spec.get("bias_kv") else None,
        "bias_v": draw(1, 1, E) if spec.get("bias_kv") else None,
        "q_proj_weight": draw(E, E) if separate else None,
        "k_proj_weight": draw(E, kdim) if separate else None,
        "v_proj_weight": draw(E, vdim) if separate else None,
    }
    key_padding, attn = _mask_arrays()
    common = {
        "add_zero_attn": bool(spec.get("add_zero_attn", False)),
        "dropout_p": 0.0,
        "training": False,
        "need_weights": True,
        "use_separate_proj_weight": separate,
    }
    got, got_w = LF.multi_head_attention_forward(
        lucid.from_numpy(q),
        lucid.from_numpy(k),
        lucid.from_numpy(v),
        E,
        H,
        key_padding_mask=lucid.from_numpy(key_padding),
        attn_mask=lucid.from_numpy(attn),
        **{n: None if t is None else lucid.from_numpy(t) for n, t in weights.items()},
        **common,
    )
    with ref.no_grad():
        want, want_w = ref.nn.functional.multi_head_attention_forward(
            ref.from_numpy(q),
            ref.from_numpy(k),
            ref.from_numpy(v),
            E,
            H,
            key_padding_mask=ref.from_numpy(key_padding),
            attn_mask=ref.from_numpy(attn),
            **{n: None if t is None else ref.from_numpy(t) for n, t in weights.items()},
            **common,
        )
    assert got_w is not None
    np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=0, atol=1e-5)
    np.testing.assert_allclose(got_w.numpy(), want_w.numpy(), rtol=0, atol=1e-6)
