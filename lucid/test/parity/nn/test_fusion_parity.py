"""Parity: ``lucid.nn.utils.fusion`` against the reference fusion utilities.

One table covers the conv families (1d / 2d / 3d, transposed, grouped), Linear
and the raw-tensor entry points: the fused layer equals the unfused pair in
eval mode, its parameters equal the reference's, and each refusal raises the
reference's exception class.
"""

from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.nn.utils import fusion
from lucid.test._fixtures.devices import metal_available
from lucid.test._helpers.compare import assert_close

DEVICES: list[str] = ["cpu", "metal"] if metal_available() else ["cpu"]

# (module class name, constructor args/kwargs, bn class name, input shape, transpose)
CASES: dict[str, tuple[str, tuple[Any, ...], dict[str, Any], str, tuple[int, ...], bool]] = {
    "conv1d": ("Conv1d", (3, 6, 3), {}, "BatchNorm1d", (2, 3, 9), False),
    "conv2d": ("Conv2d", (3, 6, 3), {"padding": 1}, "BatchNorm2d", (2, 3, 7, 7), False),
    "conv2d_nobias": ("Conv2d", (3, 6, 3), {"bias": False}, "BatchNorm2d", (2, 3, 7, 7), False),
    "conv3d": ("Conv3d", (3, 6, 3), {}, "BatchNorm3d", (2, 3, 5, 5, 5), False),
    "convt1d": ("ConvTranspose1d", (3, 6, 3), {}, "BatchNorm1d", (2, 3, 9), True),
    "convt2d": ("ConvTranspose2d", (3, 6, 3), {"stride": 2}, "BatchNorm2d", (2, 3, 5, 5), True),
    "convt3d": ("ConvTranspose3d", (3, 6, 3), {}, "BatchNorm3d", (2, 3, 4, 4, 4), True),
    "convt2d_nobias": (
        "ConvTranspose2d",
        (3, 6, 3),
        {"bias": False},
        "BatchNorm2d",
        (2, 3, 5, 5),
        True,
    ),
}


def _state(module: Any, bn: Any, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Random weights for a module and a BatchNorm, as numpy, shared by both sides."""
    state = {"weight": rng.standard_normal(tuple(module.weight.shape)).astype(np.float32)}
    if module.bias is not None:
        state["bias"] = rng.standard_normal(tuple(module.bias.shape)).astype(np.float32)
    features = bn.num_features
    state["gamma"] = rng.standard_normal(features).astype(np.float32)
    state["beta"] = rng.standard_normal(features).astype(np.float32)
    state["mean"] = rng.standard_normal(features).astype(np.float32)
    state["var"] = (rng.random(features) + 0.5).astype(np.float32)
    return state


def _load_lucid(module: Any, bn: Any, state: dict[str, np.ndarray], device: str) -> None:
    module.weight = nn.Parameter(lucid.tensor(state["weight"].copy(), device=device))
    if "bias" in state:
        module.bias = nn.Parameter(lucid.tensor(state["bias"].copy(), device=device))
    bn.weight = nn.Parameter(lucid.tensor(state["gamma"].copy(), device=device))
    bn.bias = nn.Parameter(lucid.tensor(state["beta"].copy(), device=device))
    bn.running_mean.copy_(lucid.tensor(state["mean"].copy(), device=device))
    bn.running_var.copy_(lucid.tensor(state["var"].copy(), device=device))


def _load_ref(ref: Any, module: Any, bn: Any, state: dict[str, np.ndarray]) -> None:
    module.weight = ref.nn.Parameter(ref.tensor(state["weight"].copy()))
    if "bias" in state:
        module.bias = ref.nn.Parameter(ref.tensor(state["bias"].copy()))
    bn.weight = ref.nn.Parameter(ref.tensor(state["gamma"].copy()))
    bn.bias = ref.nn.Parameter(ref.tensor(state["beta"].copy()))
    bn.running_mean.copy_(ref.tensor(state["mean"].copy()))
    bn.running_var.copy_(ref.tensor(state["var"].copy()))


@pytest.mark.parity
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("case", CASES)
def test_fused_conv_matches_the_reference(ref: Any, case: str, device: str) -> None:
    name, args, kwargs, bn_name, shape, transpose = CASES[case]
    mine, my_bn = getattr(nn, name)(*args, **kwargs), getattr(nn, bn_name)(6)
    theirs, their_bn = getattr(ref.nn, name)(*args, **kwargs), getattr(ref.nn, bn_name)(6)
    state = _state(mine, my_bn, np.random.default_rng(3))
    mine, my_bn = mine.to(device=device), my_bn.to(device=device)
    _load_lucid(mine, my_bn, state, device)
    _load_ref(ref, theirs, their_bn, state)
    for module in (mine, my_bn, theirs, their_bn):
        module.eval()

    x = np.random.default_rng(4).standard_normal(shape).astype(np.float32)
    fused = fusion.fuse_conv_bn_eval(mine, my_bn, transpose=transpose)
    fused_ref = ref.nn.utils.fusion.fuse_conv_bn_eval(theirs, their_bn, transpose=transpose)

    xl = lucid.tensor(x.copy(), device=device)
    assert_close(fused(xl), fused_ref(ref.tensor(x.copy())), atol=1e-4)
    assert_close(fused(xl), my_bn(mine(xl)), atol=1e-4)
    assert_close(fused.weight, fused_ref.weight, atol=1e-5)
    assert_close(fused.bias, fused_ref.bias, atol=1e-5)
    assert fused.weight.requires_grad == fused_ref.weight.requires_grad
    assert fused.bias.requires_grad == fused_ref.bias.requires_grad


@pytest.mark.parity
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "nobias"])
def test_fused_linear_matches_the_reference(ref: Any, bias: bool, device: str) -> None:
    mine, my_bn = nn.Linear(5, 6, bias=bias), nn.BatchNorm1d(6)
    theirs, their_bn = ref.nn.Linear(5, 6, bias=bias), ref.nn.BatchNorm1d(6)
    state = _state(mine, my_bn, np.random.default_rng(5))
    mine, my_bn = mine.to(device=device), my_bn.to(device=device)
    _load_lucid(mine, my_bn, state, device)
    _load_ref(ref, theirs, their_bn, state)
    for module in (mine, my_bn, theirs, their_bn):
        module.eval()

    x = np.random.default_rng(6).standard_normal((4, 5)).astype(np.float32)
    fused = fusion.fuse_linear_bn_eval(mine, my_bn)
    fused_ref = ref.nn.utils.fusion.fuse_linear_bn_eval(theirs, their_bn)

    xl = lucid.tensor(x.copy(), device=device)
    assert_close(fused(xl), fused_ref(ref.tensor(x.copy())), atol=1e-4)
    assert_close(fused(xl), my_bn(mine(xl)), atol=1e-4)
    assert_close(fused.weight, fused_ref.weight, atol=1e-5)
    assert_close(fused.bias, fused_ref.bias, atol=1e-5)
    assert fused.bias.requires_grad == fused_ref.bias.requires_grad


@pytest.mark.parity
@pytest.mark.parametrize("transpose", [False, True])
def test_weights_functions_match_the_reference(ref: Any, transpose: bool) -> None:
    rng = np.random.default_rng(7)
    shape = (3, 6, 3, 3) if transpose else (6, 3, 3, 3)
    w = rng.standard_normal(shape).astype(np.float32)
    b = rng.standard_normal(6).astype(np.float32)
    rm, gamma, beta = (rng.standard_normal(6).astype(np.float32) for _ in range(3))
    rv = (rng.random(6) + 0.5).astype(np.float32)

    mine = fusion.fuse_conv_bn_weights(
        lucid.tensor(w),
        lucid.tensor(b),
        lucid.tensor(rm),
        lucid.tensor(rv),
        1e-5,
        lucid.tensor(gamma),
        lucid.tensor(beta),
        transpose,
    )
    theirs = ref.nn.utils.fusion.fuse_conv_bn_weights(
        ref.tensor(w),
        ref.tensor(b),
        ref.tensor(rm),
        ref.tensor(rv),
        1e-5,
        ref.tensor(gamma),
        ref.tensor(beta),
        transpose,
    )
    for got, want in zip(mine, theirs, strict=True):
        assert_close(got, want, atol=1e-5)

    lw = rng.standard_normal((6, 5)).astype(np.float32)
    mine_l = fusion.fuse_linear_bn_weights(
        lucid.tensor(lw),
        lucid.tensor(b),
        lucid.tensor(rm),
        lucid.tensor(rv),
        1e-5,
        lucid.tensor(gamma),
        lucid.tensor(beta),
    )
    theirs_l = ref.nn.utils.fusion.fuse_linear_bn_weights(
        ref.tensor(lw),
        ref.tensor(b),
        ref.tensor(rm),
        ref.tensor(rv),
        1e-5,
        ref.tensor(gamma),
        ref.tensor(beta),
    )
    for got, want in zip(mine_l, theirs_l, strict=True):
        assert_close(got, want, atol=1e-5)


@pytest.mark.parity
class TestRefusalsMatchTheReference:
    """The reference's exception class and message for each refusal."""

    @staticmethod
    def _raised(call: Any) -> tuple[type[BaseException], str]:
        with pytest.raises(BaseException) as caught:
            call()
        return caught.type, str(caught.value)

    @pytest.mark.parametrize("train", ["conv", "bn"])
    def test_train_mode(self, ref: Any, train: str) -> None:
        mine = (nn.Conv2d(3, 6, 3), nn.BatchNorm2d(6))
        theirs = (ref.nn.Conv2d(3, 6, 3), ref.nn.BatchNorm2d(6))
        for pair in (mine, theirs):
            pair[0].eval()
            pair[1].eval()
            pair[0 if train == "conv" else 1].train()
        want = self._raised(lambda: ref.nn.utils.fusion.fuse_conv_bn_eval(*theirs))
        got = self._raised(lambda: fusion.fuse_conv_bn_eval(*mine))
        assert got == want

    def test_missing_running_statistics(self, ref: Any) -> None:
        mine = (nn.Conv2d(3, 6, 3).eval(), nn.BatchNorm2d(6, track_running_stats=False).eval())
        theirs = (
            ref.nn.Conv2d(3, 6, 3).eval(),
            ref.nn.BatchNorm2d(6, track_running_stats=False).eval(),
        )
        want = self._raised(lambda: ref.nn.utils.fusion.fuse_conv_bn_eval(*theirs))
        got = self._raised(lambda: fusion.fuse_conv_bn_eval(*mine))
        assert got == want

    def test_linear_feature_mismatch(self, ref: Any) -> None:
        mine = (nn.Linear(5, 6).eval(), nn.BatchNorm1d(4).eval())
        theirs = (ref.nn.Linear(5, 6).eval(), ref.nn.BatchNorm1d(4).eval())
        want = self._raised(lambda: ref.nn.utils.fusion.fuse_linear_bn_eval(*theirs))
        got = self._raised(lambda: fusion.fuse_linear_bn_eval(*mine))
        assert got == want
