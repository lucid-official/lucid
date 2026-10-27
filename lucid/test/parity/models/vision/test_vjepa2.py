"""Reference parity for what V-JEPA 2 adds to V-JEPA.

The tubelet embedding and the L1 are checked in ``test_vjepa.py``; what
is new here is the geometry — a three-axis rotary attention carrying a
layout quirk the released weights were trained with, the action model's
frame-causal mask, and the gated feed-forward's width rule.  Each of
these is a place where a plausible implementation is numerically wrong
against a published checkpoint while every shape still agrees.
"""

from typing import Any

import numpy as np
import pytest

import lucid
from lucid.models.vision.vjepa2 import VJEPA2Config
from lucid.models.vision.vjepa2._model import (
    _action_attention_mask,
    _axis_positions,
    _FeedForward,
    _RoPEAttention,
    _rotate_queries_or_keys,
)


def _np(tensor: Any) -> Any:
    """Lucid tensor to numpy, for comparison."""
    return tensor.numpy()


@pytest.mark.parity
class TestRotaryParity:
    """The rotation the released encoder was trained with."""

    def test_the_rotation_matches_the_reference(self, ref: Any) -> None:
        rng = np.random.RandomState(0)
        head_dim = 8
        x = rng.randn(2, 3, 5, head_dim).astype(np.float32)
        positions = np.arange(5, dtype=np.float32)

        ours = _np(
            _rotate_queries_or_keys(
                lucid.tensor(x.copy()), lucid.tensor(positions.copy()), 10_000.0
            )
        )

        with ref.no_grad():
            value = ref.tensor(x.copy())
            pos = ref.tensor(positions.copy())
            omega = ref.arange(head_dim // 2, dtype=ref.float32)
            omega /= head_dim / 2.0
            omega = 1.0 / 10000**omega
            freq = pos.unsqueeze(-1) * omega
            # The released code repeats the half as a block rather than
            # interleaving it, which pairs each rotation with the wrong
            # frequency — and the published weights were trained that way,
            # so reproducing the checkpoint means reproducing this.
            sin = freq.sin().repeat(1, 2)
            cos = freq.cos().repeat(1, 2)
            pair = value.unflatten(-1, (-1, 2))
            first, second = pair.unbind(dim=-1)
            rotated = ref.stack((-second, first), dim=-1).flatten(-2)
            theirs = ((value * cos) + (rotated * sin)).numpy()

        assert ours.shape == theirs.shape
        assert np.abs(ours - theirs).max() < 1e-5

    def test_the_layout_is_not_the_interleaved_one(self, ref: Any) -> None:
        """Guards the test above: the corrected layout is a different model."""
        rng = np.random.RandomState(1)
        head_dim = 8
        x = rng.randn(1, 1, 4, head_dim).astype(np.float32)
        positions = np.arange(4, dtype=np.float32)

        ours = _np(
            _rotate_queries_or_keys(
                lucid.tensor(x.copy()), lucid.tensor(positions.copy()), 10_000.0
            )
        )
        with ref.no_grad():
            value = ref.tensor(x.copy())
            omega = ref.arange(head_dim // 2, dtype=ref.float32)
            omega /= head_dim / 2.0
            omega = 1.0 / 10000**omega
            freq = ref.tensor(positions.copy()).unsqueeze(-1) * omega
            sin = freq.sin().repeat_interleave(2, dim=-1)
            cos = freq.cos().repeat_interleave(2, dim=-1)
            pair = value.unflatten(-1, (-1, 2))
            first, second = pair.unbind(dim=-1)
            rotated = ref.stack((-second, first), dim=-1).flatten(-2)
            corrected = ((value * cos) + (rotated * sin)).numpy()

        assert np.abs(ours - corrected).max() > 1e-3

    def test_position_zero_is_the_identity(self, ref: Any) -> None:
        x = np.random.RandomState(2).randn(1, 1, 3, 6).astype(np.float32)
        positions = np.zeros(3, dtype=np.float32)
        ours = _np(
            _rotate_queries_or_keys(
                lucid.tensor(x.copy()), lucid.tensor(positions.copy()), 10_000.0
            )
        )
        assert np.abs(ours - x).max() < 1e-6


@pytest.mark.parity
class TestAxisSplitParity:
    """One third of a head per axis, rounded down to an even width."""

    @pytest.mark.parametrize(("head_dim", "axis"), [(64, 20), (48, 16), (32, 10), (16, 4), (8, 2)])
    def test_each_axis_takes_the_released_width(self, ref: Any, head_dim: int, axis: int) -> None:
        attention = _RoPEAttention(
            head_dim * 2, 2, grid_size=4, rope_base=10_000.0, qkv_bias=True, attn_drop_rate=0.0
        )
        assert attention.depth_dim == axis
        assert attention.height_dim == axis
        assert attention.width_dim == axis
        assert 3 * axis <= head_dim

    def test_the_tail_of_a_head_is_left_alone(self, ref: Any) -> None:
        """``3 * axis`` rarely fills a head; the remainder must not rotate."""
        config = VJEPA2Config(
            image_size=16,
            patch_size=8,
            tubelet_size=2,
            num_frames=4,
            dim=32,
            depth=1,
            num_heads=2,
            predictor_dim=16,
            predictor_depth=1,
            predictor_heads=2,
        )
        attention = _RoPEAttention(
            config.dim,
            config.num_heads,
            grid_size=2,
            rope_base=config.rope_base,
            qkv_bias=True,
            attn_drop_rate=0.0,
        ).eval()
        head_dim = config.dim // config.num_heads
        assert 3 * attention.depth_dim < head_dim

        x = lucid.rand(1, 4, config.dim)
        with lucid.no_grad():
            first = attention(x, temporal=1, height=2, width=2)
            second = attention(x, temporal=1, height=2, width=2)
        assert float((first - second).abs().max().item()) == 0.0

    def test_the_axes_decompose_a_flat_index(self, ref: Any) -> None:
        ids = lucid.arange(24)
        depth, row, col = _axis_positions(ids, 3, 2, grid_size=3)
        with ref.no_grad():
            flat = ref.arange(24)
            want_depth = (flat // 6).float().numpy()
            want_row = ((flat % 6) // 2).float().numpy() * (3.0 / 3.0)
            want_col = ((flat % 6) % 2).float().numpy() * (3.0 / 3.0)
        assert np.abs(_np(depth) - want_depth).max() == 0.0
        assert np.abs(_np(row) - want_row).max() == 0.0
        assert np.abs(_np(col) - want_col).max() == 0.0

    def test_a_shorter_grid_is_snapped_onto_the_trained_one(self, ref: Any) -> None:
        """The released attention scales the spatial axes by grid / H."""
        ids = lucid.arange(4)
        _depth, row, col = _axis_positions(ids, 2, 2, grid_size=4)
        assert _np(row).tolist() == [0.0, 0.0, 2.0, 2.0]
        assert _np(col).tolist() == [0.0, 2.0, 0.0, 2.0]


@pytest.mark.parity
class TestActionMaskParity:
    """Frame-causal attention: a step sees its own frame and the past."""

    def test_the_mask_matches_the_reference_construction(self, ref: Any) -> None:
        steps, per_step = 3, 5
        ours = _np(_action_attention_mask(steps, per_step, "cpu", True))

        with ref.no_grad():
            total = steps * per_step
            theirs = ref.zeros(total, total).bool()
            block = ref.ones(per_step, per_step).bool()
            for first in range(steps):
                for second in range(0, first + 1):
                    theirs[
                        first * per_step : (first + 1) * per_step,
                        second * per_step : (second + 1) * per_step,
                    ] = block
            expected = theirs.numpy()

        assert ours.shape == expected.shape
        assert (ours == expected).all()

    def test_the_mask_is_a_keep_mask_not_a_block_mask(self, ref: Any) -> None:
        """Inverted, this would hide the past and show only the future."""
        mask = _np(_action_attention_mask(2, 2, "cpu", True))
        assert bool(mask[0][0]) and not bool(mask[0][2])
        assert bool(mask[3][0]) and bool(mask[3][3])

    def test_without_causality_there_is_no_mask(self, ref: Any) -> None:
        assert _action_attention_mask(3, 5, "cpu", False) is None


@pytest.mark.parity
class TestGatedFeedForwardParity:
    """The optional SiLU path, whose width rule the release rounds to eight."""

    @pytest.mark.parametrize(
        ("hidden", "inner"), [(3072, 2048), (4096, 2736), (1024, 688), (96, 64)]
    )
    def test_the_wide_silu_width_matches_the_release(
        self, ref: Any, hidden: int, inner: int
    ) -> None:
        layer = _FeedForward(64, hidden, use_silu=True, wide_silu=True)
        assert int(layer.fc1.weight.shape[0]) == inner
        assert int(layer.fc3.weight.shape[1]) == inner

    def test_the_gate_is_applied_to_the_first_branch(self, ref: Any) -> None:
        lucid.manual_seed(5)
        layer = _FeedForward(8, 16, use_silu=True, wide_silu=False).eval()
        x = np.random.RandomState(5).randn(2, 3, 8).astype(np.float32)

        ours = _np(layer(lucid.tensor(x.copy())))

        with ref.no_grad():
            value = ref.tensor(x.copy())

            def linear(module: Any, tensor: Any) -> Any:
                return tensor @ ref.tensor(module.weight.numpy()).T + ref.tensor(
                    module.bias.numpy()
                )

            gated = ref.nn.functional.silu(linear(layer.fc1, value)) * linear(layer.fc2, value)
            theirs = linear(layer.fc3, gated).numpy()

        assert np.abs(ours - theirs).max() < 1e-5
