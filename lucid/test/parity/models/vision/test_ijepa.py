"""Reference parity for the pieces I-JEPA is built out of.

There is no I-JEPA in any reference package, so this cannot be a
model-level comparison.  What it can be — and what the released
checkpoint cannot give on a machine without 10 GB to spare — is a
numerical check that each piece computes what it claims: the fixed
position table, a pre-norm block with its fused QKV, the moving average,
and the discrepancy the paper and its code disagree about.

The one comparison this file cannot make is recorded instead of skipped.
Against the released ViT-H/14 checkpoint, loaded into this encoder, the
outputs agree to 2.2e-03 at the worst position and 1.4e-05 on average
(relative 1e-04), and the average-pooled representation the paper
evaluates to 7.2e-05.  The position table this builds reproduces the
released *code*'s table exactly; the table stored inside the checkpoint
differs from both by 1.4e-04, which is a property of that file.  See
``obsidian/retro/retro-ijepa.md`` for how to reproduce it.
"""

import math
from typing import Any

import numpy as np
import pytest

import lucid
from lucid.models.vision.ijepa import IJEPAConfig, IJEPAModel
from lucid.models.vision._common._transformers import Block
from lucid.models.vision.ijepa._model import _sincos_2d


def _np(tensor: Any) -> Any:
    """Lucid tensor to numpy, for comparison."""
    return tensor.numpy()


def _copy_into(ref: Any, source: Any, target: Any) -> None:
    """Give a reference module Lucid's weights."""
    with ref.no_grad():
        for name, parameter in source.named_parameters():
            getattr(target, name).copy_(ref.tensor(parameter.numpy()))


def _linear(ref: Any, layer: Any, x: Any) -> Any:
    """``x @ W.T + b`` with Lucid's weights, in the reference framework."""
    weight = ref.tensor(layer.weight.numpy())
    bias = ref.tensor(layer.bias.numpy())
    return x @ weight.T + bias


def _tiny() -> IJEPAConfig:
    return IJEPAConfig(
        image_size=32,
        patch_size=8,
        dim=16,
        depth=1,
        num_heads=2,
        predictor_dim=8,
        predictor_depth=1,
        min_keep=1,
    )


@pytest.mark.parity
class TestPositionTableParity:
    """The table is fixed, so a wrong one is wrong for the whole run."""

    def test_the_table_is_the_sine_cosine_one(self, ref: Any) -> None:
        # Half the width encodes the column and half the row; within each
        # half, sines then cosines of position times a geometric sweep of
        # frequencies.
        dim, grid = 16, 4
        ours = _np(_sincos_2d(dim, grid))[0]

        with ref.no_grad():
            half = dim // 2
            omega = 1.0 / (10000.0 ** (ref.arange(half // 2, dtype=ref.float64) * (2.0 / half)))
            coords = ref.arange(grid, dtype=ref.float64)
            cols = coords.repeat(grid)
            rows = coords.repeat_interleave(grid)

            def block(pos: Any) -> Any:
                angles = pos.reshape(-1, 1) * omega.reshape(1, -1)
                return ref.cat([ref.sin(angles), ref.cos(angles)], dim=1)

            theirs = ref.cat([block(cols), block(rows)], dim=1).float().numpy()

        assert np.abs(ours - theirs).max() < 1e-6

    def test_the_first_position_is_all_zeros_and_ones(self, ref: Any) -> None:
        """Guards the test above: position 0 has a closed form."""
        table = _np(_sincos_2d(16, 4))[0, 0]
        assert np.abs(table[:4]).max() == 0.0  # sin(0)
        assert np.abs(table[4:8] - 1.0).max() < 1e-6  # cos(0)

    def test_the_table_is_not_learned(self) -> None:
        # The released code marks it as requiring no gradient; a learnable
        # table is a different model that trains just as happily.
        model = IJEPAModel(_tiny())
        assert "encoder.pos_embed" not in dict(model.named_parameters())
        assert "pos_embed" in dict(model.encoder.named_buffers())


@pytest.mark.parity
class TestBlockParity:
    """A pre-norm block with one fused QKV projection."""

    def test_the_block_matches_the_reference(self, ref: Any) -> None:
        lucid.manual_seed(0)
        block = Block(dim=12, num_heads=3, hidden=24, eps=1e-6).eval()
        x = np.random.RandomState(0).randn(2, 5, 12).astype(np.float32)

        ours = _np(block(lucid.tensor(x.copy())))

        with ref.no_grad():
            current = ref.tensor(x.copy())
            norm1, norm2 = ref.nn.LayerNorm(12, eps=1e-6), ref.nn.LayerNorm(12, eps=1e-6)
            _copy_into(ref, block.norm1, norm1)
            _copy_into(ref, block.norm2, norm2)

            b, n, c = current.shape
            heads, head_dim = 3, 4
            qkv = _linear(ref, block.attn.qkv, norm1(current))
            qkv = qkv.reshape(b, n, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
            scores = qkv[0] @ qkv[1].transpose(-2, -1) / math.sqrt(head_dim)
            attended = ref.softmax(scores, dim=-1) @ qkv[2]
            merged = attended.permute(0, 2, 1, 3).reshape(b, n, c)
            current = current + _linear(ref, block.attn.proj, merged)

            hidden = ref.nn.functional.gelu(_linear(ref, block.mlp.fc1, norm2(current)))
            theirs = (current + _linear(ref, block.mlp.fc2, hidden)).numpy()

        assert np.abs(ours - theirs).max() < 1e-5

    def test_the_qkv_projection_is_fused(self, ref: Any) -> None:
        """Guards the copy above — the shapes it moves must be the fused ones."""
        block = Block(dim=12, num_heads=3, hidden=24, eps=1e-6)
        assert tuple(int(d) for d in block.attn.qkv.weight.shape) == (36, 12)
        assert tuple(int(d) for d in block.attn.proj.weight.shape) == (12, 12)


@pytest.mark.parity
class TestMovingAverageParity:
    """The update the target encoder follows."""

    def test_the_average_matches_the_reference(self, ref: Any) -> None:
        model = IJEPAModel(_tiny())
        with lucid.no_grad():
            for parameter in model.encoder.parameters():
                parameter[:] = parameter * 2.0 + 0.5

        live = [p.numpy().copy() for p in model.encoder.parameters()]
        averaged = [p.numpy().copy() for p in model.target_encoder.parameters()]
        model.update_target(0.996)

        with ref.no_grad():
            theirs = [
                (0.996 * ref.tensor(old) + 0.004 * ref.tensor(new)).numpy()
                for old, new in zip(averaged, live)
            ]
        for got, want in zip(model.target_encoder.parameters(), theirs):
            assert np.abs(got.numpy() - want).max() < 1e-6


@pytest.mark.parity
class TestObjectiveParity:
    """Smooth L1 is the released code's; the paper writes L2."""

    def test_smooth_l1_matches_the_reference(self, ref: Any) -> None:
        model = IJEPAModel(_tiny())
        rng = np.random.RandomState(1)
        prediction = rng.randn(4, 6).astype(np.float32)
        target = rng.randn(4, 6).astype(np.float32) * 3.0

        ours = float(
            model._discrepancy(lucid.tensor(prediction.copy()), lucid.tensor(target.copy())).item()
        )
        with ref.no_grad():
            theirs = ref.nn.functional.smooth_l1_loss(
                ref.tensor(prediction.copy()), ref.tensor(target.copy())
            ).item()
        assert abs(ours - theirs) < 1e-6

    def test_the_paper_s_l2_matches_the_reference(self, ref: Any) -> None:
        model = IJEPAModel(
            IJEPAConfig(
                image_size=32,
                patch_size=8,
                dim=16,
                depth=1,
                num_heads=2,
                predictor_dim=8,
                predictor_depth=1,
                min_keep=1,
                objective="l2",
            )
        )
        rng = np.random.RandomState(2)
        prediction = rng.randn(4, 6).astype(np.float32)
        target = rng.randn(4, 6).astype(np.float32)

        ours = float(
            model._discrepancy(lucid.tensor(prediction.copy()), lucid.tensor(target.copy())).item()
        )
        with ref.no_grad():
            theirs = ref.nn.functional.mse_loss(
                ref.tensor(prediction.copy()), ref.tensor(target.copy())
            ).item()
        assert abs(ours - theirs) < 1e-6

    def test_the_two_disagree_on_large_errors(self, ref: Any) -> None:
        """Guards the pair: on small errors smooth L1 *is* half the square."""
        model = IJEPAModel(_tiny())
        far = lucid.tensor([[5.0]])
        zero = lucid.tensor([[0.0]])
        smooth = float(model._discrepancy(far, zero).item())
        assert smooth == pytest.approx(4.5)  # 5 - 0.5, the linear regime
        assert smooth != pytest.approx(25.0)


@pytest.mark.parity
class TestTargetNormalisationParity:
    """The layer norm the paper never mentions."""

    def test_the_targets_are_non_affine_layer_normed(self, ref: Any) -> None:
        lucid.manual_seed(3)
        config = _tiny()
        model = IJEPAModel(config).eval()
        images = lucid.rand(2, 3, 32, 32)

        with lucid.no_grad():
            raw = model.target_encoder(images)
        out = model(images)

        with ref.no_grad():
            normed = ref.nn.functional.layer_norm(
                ref.tensor(_np(raw)), (int(raw.shape[-1]),)
            ).numpy()
        # Every target row must be a row of the normalised tokens.
        targets = _np(out.target)
        indices = _np(out.target_indices)
        for image in range(targets.shape[0]):
            for block in range(targets.shape[1]):
                for slot, patch in enumerate(indices[image, block]):
                    gap = np.abs(targets[image, block, slot] - normed[image, int(patch)])
                    assert gap.max() < 1e-5
