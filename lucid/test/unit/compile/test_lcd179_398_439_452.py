"""Regression tests for LCD-179, LCD-398, LCD-439, and LCD-452."""

import struct
from collections.abc import Callable

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
import lucid.optim as optim
from lucid._C import engine as _C_engine
from lucid.amp import GradScaler, autocast
from lucid.compile import fused_step
from lucid.coreml._emit import TracedValue, _grid_sample

from lucid.test.unit.compile._helpers import COMPILE_DEVICE

if TYPE_CHECKING:
    from lucid.coreml._build import Builder


@dataclass
class _StubOp:
    name: str
    attrs: dict[str, object]
    outputs: list[TracedValue] = field(default_factory=list)


def _f32_bits(t: lucid.Tensor) -> list[int]:
    arr = t.to(lucid.float32).numpy().reshape(-1)
    return [struct.unpack("<I", struct.pack("<f", float(v)))[0] for v in arr]


class TestFusedStepAutocastScalerAndBatchChange:
    """LCD-179: fused_step under autocast(float16) + an overflowing GradScaler and batch change.

    The scaled gradients must overflow float16 on the first step so both
    paths skip it and halve the scale, and a later step must go through.
    65536 used to overflow because the scale multiply itself ran in float16
    — ``mul`` was LOWER.  It has no row in the policy table now (LCD-500),
    so the scaled loss is float32 and the overflow happens where the
    reference's does, when the float32 gradient reaches the float16 linear
    layers.  From 2**19 this seed's steps skip, skip and go through, so the
    changed batch is the one that updates.  The seed was 42 until the
    model's initialisers and ``lucid.randn`` drew from one stream (LCD-583):
    the batches now follow the init's draws, and 42's batch change skipped."""

    def test_grad_scaler_overflowing_init_scale_autocast_f16_and_batch_change(self) -> None:
        lucid.manual_seed(12)
        eager_model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).to(
            COMPILE_DEVICE
        )
        comp_model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).to(COMPILE_DEVICE)
        with lucid.no_grad():
            for pe, pc in zip(eager_model.parameters(), comp_model.parameters(), strict=True):
                pc.copy_(pe.detach().clone())

        opt_eager = optim.AdamW(list(eager_model.parameters()), lr=1e-2)
        opt_comp = optim.AdamW(list(comp_model.parameters()), lr=1e-2)
        scaler_eager = GradScaler(init_scale=2.0**19)
        scaler_comp = GradScaler(init_scale=2.0**19)
        scales = (2.0**18, 2.0**17, 2.0**17)
        step = fused_step(comp_model, F.mse_loss, opt_comp, grad_scaler=scaler_comp)

        params_before = [p.detach().clone() for p in comp_model.parameters()]

        for step_idx, bsz in enumerate((6, 6, 2)):
            xb = lucid.randn(bsz, 8, device=COMPILE_DEVICE)
            yb = lucid.randn(bsz, 4, device=COMPILE_DEVICE)
            with autocast("metal", dtype=lucid.float16):
                opt_eager.zero_grad()
                loss_e = F.mse_loss(eager_model(xb), yb)
                scaled = scaler_eager.scale(loss_e)
                assert isinstance(scaled, lucid.Tensor)
                scaled.backward()
                scaler_eager.step(opt_eager)
                scaler_eager.update()

                loss_c = step(xb, yb)

            assert loss_c.dtype == loss_e.dtype == lucid.float32
            assert np.isfinite(float(loss_c.item()))
            assert abs(float(loss_c.item()) - float(loss_e.item())) < 1e-3
            assert scaler_comp.get_scale() == scaler_eager.get_scale() == scales[step_idx]
            if step_idx == 0:
                for p_old, p_now in zip(params_before, comp_model.parameters(), strict=True):
                    assert float((p_old - p_now).abs().max().item()) == 0.0

        assert step.cache_info()["entries"] == 2
        for pe, pc in zip(eager_model.parameters(), comp_model.parameters(), strict=True):
            assert float((pe - pc).abs().max().item()) < 1e-3


class TestCoreMLGridSampleValidation:
    """LCD-398: Core ML _grid_sample emitter rejects modes and padding_modes outside {0, 1}."""

    @pytest.mark.parametrize("bad_mode", [-1, 2, 3])
    def test_invalid_mode_raises(self, bad_mode: int) -> None:
        op = _StubOp(
            name="grid_sample", attrs={"mode": bad_mode, "padding_mode": 0, "align_corners": False}
        )
        with pytest.raises(ValueError, match="mode"):
            _grid_sample(cast("Builder", object()), op, ["x", "grid"])

    @pytest.mark.parametrize("bad_pad", [-1, 2, 3])
    def test_invalid_padding_mode_raises(self, bad_pad: int) -> None:
        op = _StubOp(
            name="grid_sample", attrs={"mode": 0, "padding_mode": bad_pad, "align_corners": False}
        )
        with pytest.raises(ValueError, match="padding_mode"):
            _grid_sample(cast("Builder", object()), op, ["x", "grid"])


_SUBNORMAL_OPS: dict[str, Callable[[lucid.Tensor], lucid.Tensor]] = {
    "x*1": lambda t: t * 1,
    "1*x": lambda t: 1 * t,
    "x*(-1)": lambda t: t * (-1),
    "(-1)*x": lambda t: (-1) * t,
    "x/1": lambda t: t / 1,
    "x/(-1)": lambda t: t / (-1),
    "x-0": lambda t: t - 0,
    "x+(-0.0)": lambda t: t + (-0.0),
    "(-0.0)+x": lambda t: (-0.0) + t,
    "(-0.0)-x": lambda t: (-0.0) - t,
    "x*ones_reshape": lambda t: t * lucid.ones(1, 2, dtype=t.dtype, device=t.device).reshape(2),
    "mean": lambda t: t[:1].mean(),
}


class TestCompiledSubnormalIdentityFlush:
    """LCD-452: compiled identity ops flush f32/bf16 subnormals and keep f16 subnormals."""

    @pytest.mark.parametrize(
        ("dtype", "vals"),
        [
            (lucid.float32, [1e-40, -1e-40]),
            (lucid.bfloat16, [1e-40, -1e-40]),
            (lucid.float16, [1e-7, -1e-7]),
        ],
    )
    @pytest.mark.parametrize("name", list(_SUBNORMAL_OPS))
    def test_subnormal_identity_matches_eager_bits(
        self, dtype: lucid.dtype, vals: list[float], name: str
    ) -> None:
        fn = _SUBNORMAL_OPS[name]
        x = lucid.tensor(vals, dtype=dtype, device=COMPILE_DEVICE)
        _C_engine.compile.session_cache_clear()
        compiled_fn = lucid.compile(fn)
        got = compiled_fn(x)
        assert _C_engine.compile.session_cache_size() >= 1
        want = fn(x)
        assert _f32_bits(got) == _f32_bits(want), f"{name} ({dtype})"


class TestCompiledTopkFastPath:
    """LCD-439: compiled topk uses fast selection without full-axis sort."""

    def test_topk_64x1000_k5_matches_eager_and_backward(self) -> None:
        lucid.manual_seed(0)
        x = lucid.randn(64, 1000, device=COMPILE_DEVICE)
        c_topk = lucid.compile(lambda t: lucid.topk(t, 5, dim=-1))
        v_c, i_c = c_topk(x)
        v_e, i_e = lucid.topk(x, 5, dim=-1)
        assert (v_c.to("cpu") == v_e.to("cpu")).all().item()
        assert (i_c.to("cpu") == i_e.to("cpu")).all().item()
