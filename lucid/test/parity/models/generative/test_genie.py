"""Reference parity for the pieces Genie is built out of.

There is no reference-framework Genie to diff against, and there is no
official one at all: the paper says outright that no checkpoints, no
training data and no code were released.  The open reimplementations are
JAX (Jafar, Jasmine) or take deliberate liberties with the architecture
(open-genie), so a model-level comparison has nothing to compare to.

What can be checked is the same thing the Dreamer families check —
whether each *mechanism* computes what it claims — and for Genie that
matters more than usual, because the paper leaves the feed-forward
width, the norm placement, the positional encoding and the commitment
weight unstated.  A reader can only tell whether the parts were
assembled correctly if the parts themselves are right.

The pieces here are the ones where a disagreement would be invisible
downstream: an attention whose inner width is *decoupled* from the model
width (36 heads of 128 inside a 5120-wide model, which is what Table 12
says and what a normal implementation gets wrong), the block order the
paper spells out, the patch grid, the masked objective, and the
quantiser both codebooks share.
"""

import math
from typing import Any

import numpy as np
import pytest

import lucid
import lucid.nn as nn
from lucid.models.generative.genie import GenieConfig, GenieModel
from lucid.models.generative.genie._model import (
    _Attention,
    _patchify,
    _STBlock,
    _unpatchify,
)


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


def _ref_attention(ref: Any, module: Any, x: Any, causal: bool) -> Any:
    """The attention of :class:`_Attention`, rebuilt from its weights.

    Written out rather than delegated to the reference's own attention
    module, because the thing under test is precisely that the heads are
    ``heads * head_dim`` wide rather than ``dim`` wide.
    """
    batch, length = int(x.shape[0]), int(x.shape[1])
    heads, head_dim = module.heads, module.head_dim

    def split(layer: Any) -> Any:
        projected = _linear(ref, layer, x)
        return projected.reshape(batch, length, heads, head_dim).permute(0, 2, 1, 3)

    query, key, value = split(module.query), split(module.key), split(module.value)
    if module.query_norm is not None:
        query_norm = ref.nn.LayerNorm(head_dim)
        key_norm = ref.nn.LayerNorm(head_dim)
        _copy_into(ref, module.query_norm, query_norm)
        _copy_into(ref, module.key_norm, key_norm)
        query, key = query_norm(query), key_norm(key)
    scores = query @ key.transpose(-2, -1) / math.sqrt(head_dim)
    if causal:
        forbidden = ref.triu(ref.ones(length, length), diagonal=1).bool()
        scores = scores.masked_fill(forbidden, float("-inf"))
    attended = ref.softmax(scores, dim=-1) @ value
    merged = attended.permute(0, 2, 1, 3).reshape(batch, length, heads * head_dim)
    return _linear(ref, module.out, merged)


def _tiny(**overrides: object) -> GenieConfig:
    base: dict[str, object] = dict(
        sample_size=(8, 8),
        num_frames=4,
        num_codes=16,
        code_dim=4,
        tokenizer_encoder_layers=1,
        tokenizer_encoder_dim=16,
        tokenizer_encoder_heads=2,
        tokenizer_encoder_head_dim=8,
        tokenizer_decoder_layers=1,
        tokenizer_decoder_dim=16,
        tokenizer_decoder_heads=2,
        tokenizer_decoder_head_dim=8,
        action_patch_size=4,
        action_dim=4,
        action_encoder_layers=1,
        action_encoder_dim=16,
        action_encoder_heads=2,
        action_decoder_layers=1,
        action_decoder_dim=16,
        action_decoder_heads=2,
        dynamics_layers=1,
        dynamics_dim=16,
        dynamics_heads=2,
        dynamics_head_dim=8,
        maskgit_steps=4,
    )
    base.update(overrides)
    return GenieConfig(**base)  # type: ignore[arg-type]


@pytest.mark.parity
class TestAttentionParity:
    """Table 12's heads are 36 x 128 inside a 5120-wide model.

    Tying the inner width to the model width is the silent version of
    this defect: it trains, it attends, and it is a different network
    from the paper's.
    """

    @pytest.mark.parametrize("causal", [False, True])
    def test_attention_matches_the_reference(self, ref: Any, causal: bool) -> None:
        lucid.manual_seed(0)
        module = _Attention(dim=12, heads=3, head_dim=5, qk_norm=False).eval()
        x = np.random.RandomState(0).randn(2, 6, 12).astype(np.float32)

        ours = _np(module(lucid.tensor(x.copy()), causal=causal))
        with ref.no_grad():
            theirs = _ref_attention(ref, module, ref.tensor(x.copy()), causal).numpy()
        assert np.abs(ours - theirs).max() < 1e-5

    def test_the_inner_width_is_heads_times_head_dim(self, ref: Any) -> None:
        """Guards the test above: the shapes it copies must be the odd ones."""
        module = _Attention(dim=12, heads=3, head_dim=5, qk_norm=False)
        assert tuple(int(d) for d in module.query.weight.shape) == (15, 12)
        assert tuple(int(d) for d in module.out.weight.shape) == (12, 15)

    def test_qk_norm_matches_the_reference(self, ref: Any) -> None:
        """Section 3 normalises queries and keys per head, not per token."""
        lucid.manual_seed(1)
        module = _Attention(dim=12, heads=3, head_dim=5, qk_norm=True).eval()
        x = np.random.RandomState(1).randn(2, 6, 12).astype(np.float32)

        ours = _np(module(lucid.tensor(x.copy()), causal=True))
        with ref.no_grad():
            theirs = _ref_attention(ref, module, ref.tensor(x.copy()), True).numpy()
        assert np.abs(ours - theirs).max() < 1e-5

    def test_the_causal_mask_is_the_one_that_hides_the_future(self, ref: Any) -> None:
        """Guards the mask: causal and free attention must differ."""
        lucid.manual_seed(2)
        module = _Attention(dim=12, heads=3, head_dim=5, qk_norm=False).eval()
        x = lucid.tensor(np.random.RandomState(2).randn(2, 6, 12).astype(np.float32))
        free, masked = _np(module(x, causal=False)), _np(module(x, causal=True))
        assert np.abs(free - masked).max() > 1e-3


@pytest.mark.parity
class TestBlockParity:
    """Spatial attention, causal temporal attention, then one FFW.

    Section 2: *"we include only one FFW after both spatial and temporal
    components, omitting the post-spatial FFW"*.  Rebuilt here in the
    stated order; a block that attends over the wrong axis still produces
    a tensor of the right shape.
    """

    def test_the_block_matches_the_reference(self, ref: Any) -> None:
        lucid.manual_seed(3)
        block = _STBlock(
            dim=12, heads=3, head_dim=5, hidden=24, activation=nn.GELU, qk_norm=False
        ).eval()
        b, t, n, d = 2, 3, 4, 12
        x = np.random.RandomState(3).randn(b, t, n, d).astype(np.float32)

        ours = _np(block(lucid.tensor(x.copy())))

        with ref.no_grad():
            current = ref.tensor(x.copy())
            spatial_norm = ref.nn.LayerNorm(d)
            temporal_norm = ref.nn.LayerNorm(d)
            ffw_norm = ref.nn.LayerNorm(d)
            _copy_into(ref, block.spatial_norm, spatial_norm)
            _copy_into(ref, block.temporal_norm, temporal_norm)
            _copy_into(ref, block.ffw_norm, ffw_norm)

            # Space: the N tokens of one frame, free to attend both ways.
            flat = spatial_norm(current).reshape(b * t, n, d)
            spatial = _ref_attention(ref, block.spatial, flat, False)
            current = current + spatial.reshape(b, t, n, d)

            # Time: the T tokens at one position, each seeing only the past.
            moved = temporal_norm(current).permute(0, 2, 1, 3).reshape(b * n, t, d)
            temporal = _ref_attention(ref, block.temporal, moved, True)
            current = current + temporal.reshape(b, n, t, d).permute(0, 2, 1, 3)

            first, second = [m for m in block.ffw.modules() if isinstance(m, nn.Linear)]
            hidden = ref.nn.functional.gelu(_linear(ref, first, ffw_norm(current)))
            theirs = (current + _linear(ref, second, hidden)).numpy()

        assert np.abs(ours - theirs).max() < 1e-5

    def test_attending_over_the_wrong_axis_would_show(self, ref: Any) -> None:
        """Guards the test above.

        Spatial and temporal attention are the same shape, so a block that
        swapped them would still run.  It would not agree.
        """
        lucid.manual_seed(4)
        block = _STBlock(
            dim=12, heads=3, head_dim=5, hidden=24, activation=nn.GELU, qk_norm=False
        ).eval()
        b, t, n, d = 2, 3, 4, 12
        x = lucid.tensor(np.random.RandomState(4).randn(b, t, n, d).astype(np.float32))

        ours = _np(block(x))
        swapped = _np(block(x.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3).contiguous())
        assert np.abs(ours - swapped).max() > 1e-3


@pytest.mark.parity
class TestPatchParity:
    """The patch grid, against the reference's own sliding window.

    A patch order that disagrees is invisible: the model still learns,
    against a permutation of the image.
    """

    def test_patchify_matches_unfold(self, ref: Any) -> None:
        rng = np.random.RandomState(5)
        video = rng.randn(2, 3, 3, 10, 12).astype(np.float32)

        ours = _np(_patchify(lucid.tensor(video.copy()), 4))

        with ref.no_grad():
            frames = ref.tensor(video.copy()).reshape(6, 3, 10, 12)
            padded = ref.nn.functional.pad(frames, (0, 0, 0, 2))
            windows = ref.nn.functional.unfold(padded, kernel_size=4, stride=4)
            theirs = windows.transpose(1, 2).reshape(2, 3, 9, 48).numpy()

        assert np.abs(ours - theirs).max() < 1e-6

    def test_unpatchify_inverts_it(self, ref: Any) -> None:
        rng = np.random.RandomState(6)
        video = lucid.tensor(rng.randn(2, 3, 3, 10, 12).astype(np.float32))
        restored = _unpatchify(_patchify(video, 4), 4, (3, 3), 3, (10, 12))
        assert np.abs(_np(restored) - _np(video)).max() == 0.0


@pytest.mark.parity
class TestObjectiveParity:
    """The masked objective, and the quantiser both codebooks share."""

    def test_the_masked_loss_is_cross_entropy_over_what_was_hidden(self, ref: Any) -> None:
        # At a rate of 1 the masked mean equals the plain mean over frames
        # 2..T, and the comparison says nothing; at 0.5 it separates them.
        lucid.manual_seed(5)
        config = _tiny(mask_ratio_min=0.5, mask_ratio_max=0.5)
        model = GenieModel(config)
        tokens = lucid.randint(0, config.num_codes, (2, 4, 4))
        actions = lucid.randn(2, 3, config.action_dim)

        loss, logits, mask = model._dynamics_objective(tokens, actions)

        with ref.no_grad():
            flat = ref.tensor(_np(logits).reshape(-1, config.num_codes))
            targets = ref.tensor(_np(tokens).reshape(-1)).long()
            weights = ref.tensor(_np(mask).reshape(-1))
            per_token = ref.nn.functional.cross_entropy(flat, targets, reduction="none")
            theirs = ((per_token * weights).sum() / weights.sum()).item()
            everything = ref.nn.functional.cross_entropy(flat, targets).item()

        assert abs(float(loss.item()) - theirs) < 1e-5
        assert abs(float(loss.item()) - everything) > 1e-3

    def test_the_quantiser_matches_the_reference(self, ref: Any) -> None:
        """Nearest code, straight-through, and the two VQ-VAE terms."""
        lucid.manual_seed(6)
        quantizer = nn.VectorQuantizer(8, 4, commitment_cost=0.25)
        x = np.random.RandomState(7).randn(3, 5, 4).astype(np.float32)

        out = quantizer(lucid.tensor(x.copy()))

        with ref.no_grad():
            values = ref.tensor(x.copy())
            codebook = ref.tensor(quantizer.weight.numpy())
            flat = values.reshape(-1, 4)
            distances = ref.cdist(flat, codebook)
            indices = distances.argmin(dim=-1)
            hard = codebook[indices].reshape(3, 5, 4)
            codebook_loss = ((hard - values) ** 2).mean().item()
            commitment = ((values - hard) ** 2).mean().item()
            theirs_indices = indices.reshape(3, 5).numpy()
            theirs_quantized = hard.numpy()

        assert (_np(out.indices) == theirs_indices).all()
        assert np.abs(_np(out.quantized) - theirs_quantized).max() < 1e-5
        assert abs(float(out.codebook_loss.item()) - codebook_loss) < 1e-5
        assert abs(float(out.commitment_loss.item()) - commitment) < 1e-5
        assert abs(float(quantizer.loss(out).item()) - (codebook_loss + 0.25 * commitment)) < 1e-5
