"""Reference parity for what V-JEPA adds to I-JEPA.

The transformer the two families share is checked in ``test_ijepa.py``;
this file covers the video half — the three-axis position table, the
tubelet embedding, the L1 the paper and its code agree on, and the
attentive probe that is the difference between comparing against the
paper's numbers and comparing against something else.
"""

from typing import Any

import numpy as np
import pytest

import lucid
from lucid.models.vision.vjepa import VJEPAConfig, VJEPAModel
from lucid.models.vision.vjepa._model import _sincos_3d, _TubeletEmbed


def _np(tensor: Any) -> Any:
    """Lucid tensor to numpy, for comparison."""
    return tensor.numpy()


def _tiny(**overrides: object) -> VJEPAConfig:
    base: dict[str, object] = dict(
        image_size=32,
        patch_size=8,
        tubelet_size=2,
        num_frames=4,
        dim=24,
        depth=1,
        num_heads=2,
        predictor_dim=12,
        predictor_depth=1,
        short_range_blocks=2,
        long_range_blocks=1,
    )
    base.update(overrides)
    return VJEPAConfig(**base)  # type: ignore[arg-type]


@pytest.mark.parity
class TestPositionTableParity:
    """Three tables concatenated, in time-row-column order."""

    def test_the_table_matches_the_reference(self, ref: Any) -> None:
        dim, grid = 24, (2, 4, 4)
        ours = _np(_sincos_3d(dim, grid, uniform_power=True))[0]

        with ref.no_grad():
            each = 2 * -(-dim // 6)  # ceil(dim / 6) * 2, as the code does
            duration, rows, cols = grid

            def table(positions: Any, width: int) -> Any:
                omega = 1.0 / (
                    10000.0 ** (ref.arange(width // 2, dtype=ref.float64) * (2.0 / width))
                )
                angles = positions.reshape(-1, 1) * omega.reshape(1, -1)
                return ref.cat([ref.sin(angles), ref.cos(angles)], dim=1)

            time = ref.arange(duration, dtype=ref.float64).repeat_interleave(rows * cols)
            down = ref.arange(rows, dtype=ref.float64).repeat_interleave(cols).repeat(duration)
            across = ref.arange(cols, dtype=ref.float64).repeat(duration * rows)
            theirs = (
                ref.cat([table(time, each), table(down, each), table(across, each)], dim=1)[:, :dim]
                .float()
                .numpy()
            )

        assert np.abs(ours - theirs).max() < 1e-6

    def test_the_even_split_is_not_the_weighted_one(self, ref: Any) -> None:
        """Guards the test above — the two splits must be distinguishable."""
        even = _np(_sincos_3d(24, (2, 4, 4), uniform_power=True))
        weighted = _np(_sincos_3d(24, (2, 4, 4), uniform_power=False))
        assert np.abs(even - weighted).max() > 1e-3


@pytest.mark.parity
class TestTubeletParity:
    """A clip cut into tubelets, projected by one 3-D convolution."""

    def test_the_tubelet_embedding_matches_the_reference(self, ref: Any) -> None:
        lucid.manual_seed(0)
        embed = _TubeletEmbed(3, tubelet_size=2, patch_size=8, dim=12).eval()
        clip = np.random.RandomState(0).randn(2, 4, 3, 32, 32).astype(np.float32)

        ours = _np(embed(lucid.tensor(clip.copy())))

        with ref.no_grad():
            conv = ref.nn.Conv3d(3, 12, (2, 8, 8), stride=(2, 8, 8))
            conv.weight.copy_(ref.tensor(embed.proj.weight.numpy()))
            conv.bias.copy_(ref.tensor(embed.proj.bias.numpy()))
            # (B, T, C, H, W) is this zoo's clip shape; the convolution
            # wants the channel axis second.
            tokens = conv(ref.tensor(clip.copy()).permute(0, 2, 1, 3, 4))
            theirs = tokens.reshape(2, 12, -1).permute(0, 2, 1).numpy()

        assert ours.shape == theirs.shape == (2, 2 * 4 * 4, 12)
        assert np.abs(ours - theirs).max() < 1e-5

    def test_the_token_order_is_time_then_row_then_column(self, ref: Any) -> None:
        """Guards the position table: the table assumes this order.

        A clip whose second temporal half is zero must leave the first
        half of the tokens untouched.
        """
        lucid.manual_seed(1)
        embed = _TubeletEmbed(3, tubelet_size=2, patch_size=8, dim=12).eval()
        clip = lucid.rand(1, 4, 3, 32, 32)
        halved = clip.detach().clone()
        halved[:, 2:] = lucid.zeros(1, 2, 3, 32, 32)
        with lucid.no_grad():
            whole, cut = embed(clip), embed(halved)
        assert float((whole[:, :16] - cut[:, :16]).abs().max().item()) < 1e-6
        assert float((whole[:, 16:] - cut[:, 16:]).abs().max().item()) > 1e-6


@pytest.mark.parity
class TestObjectiveParity:
    """L1, which both the paper and the released code use."""

    def test_l1_matches_the_reference(self, ref: Any) -> None:
        model = VJEPAModel(_tiny())
        rng = np.random.RandomState(2)
        prediction = rng.randn(4, 6).astype(np.float32)
        target = rng.randn(4, 6).astype(np.float32) * 2.0

        ours = float(
            model._discrepancy(lucid.tensor(prediction.copy()), lucid.tensor(target.copy())).item()
        )
        with ref.no_grad():
            theirs = ref.nn.functional.l1_loss(
                ref.tensor(prediction.copy()), ref.tensor(target.copy())
            ).item()
        assert abs(ours - theirs) < 1e-6

    def test_l1_is_not_the_squared_error(self, ref: Any) -> None:
        """Guards the choice: on these numbers the two differ by 2x."""
        model = VJEPAModel(_tiny())
        a, b = lucid.zeros(2, 2), lucid.zeros(2, 2) + 2.0
        assert float(model._discrepancy(a, b).item()) == pytest.approx(2.0)
        model.config = _tiny(objective="l2")
        assert float(model._discrepancy(a, b).item()) == pytest.approx(4.0)


@pytest.mark.parity
class TestAttentiveProbeParity:
    """One query, cross-attending the frozen feature map."""

    def test_the_pooler_matches_the_reference(self, ref: Any) -> None:
        lucid.manual_seed(3)
        config = _tiny(num_classes=5)
        model = VJEPAModel(config)
        from lucid.models.vision.vjepa._model import _AttentivePooler

        pooler = _AttentivePooler(
            config.dim, config.num_heads, config.mlp_ratio, config.layer_norm_eps
        ).eval()
        tokens = np.random.RandomState(3).randn(2, 7, config.dim).astype(np.float32)

        ours = _np(pooler(lucid.tensor(tokens.copy())))

        with ref.no_grad():
            x = ref.tensor(tokens.copy())
            norm_keys = ref.nn.LayerNorm(config.dim, eps=config.layer_norm_eps)
            norm_out = ref.nn.LayerNorm(config.dim, eps=config.layer_norm_eps)
            for source, target in ((pooler.norm_keys, norm_keys), (pooler.norm_out, norm_out)):
                for name, parameter in source.named_parameters():
                    getattr(target, name).copy_(ref.tensor(parameter.numpy()))

            def linear(layer: Any, value: Any) -> Any:
                return value @ ref.tensor(layer.weight.numpy()).T + ref.tensor(layer.bias.numpy())

            batch, count = 2, 7
            heads, head_dim = config.num_heads, config.dim // config.num_heads
            keys_in = norm_keys(x)
            query_in = ref.tensor(pooler.query_token.numpy()).expand(batch, 1, -1)
            q = (
                linear(pooler.query, query_in)
                .reshape(batch, 1, heads, head_dim)
                .permute(0, 2, 1, 3)
            )
            k = (
                linear(pooler.key, keys_in)
                .reshape(batch, count, heads, head_dim)
                .permute(0, 2, 1, 3)
            )
            v = (
                linear(pooler.value, keys_in)
                .reshape(batch, count, heads, head_dim)
                .permute(0, 2, 1, 3)
            )
            scores = q @ k.transpose(-2, -1) / float(np.sqrt(head_dim))
            attended = (ref.softmax(scores, dim=-1) @ v).permute(0, 2, 1, 3)
            attended = attended.reshape(batch, 1, config.dim)

            pooled = query_in + linear(pooler.proj, attended)
            hidden = ref.nn.functional.gelu(linear(pooler.mlp.fc1, norm_out(pooled)))
            theirs = (pooled + linear(pooler.mlp.fc2, hidden)).reshape(batch, config.dim).numpy()

        assert np.abs(ours - theirs).max() < 1e-5
        assert model.config.dim == config.dim

    def test_pooling_is_not_averaging(self, ref: Any) -> None:
        """Guards the point of the probe — the paper prices this at 17 points."""
        lucid.manual_seed(4)
        config = _tiny()
        from lucid.models.vision.vjepa._model import _AttentivePooler

        pooler = _AttentivePooler(
            config.dim, config.num_heads, config.mlp_ratio, config.layer_norm_eps
        ).eval()
        tokens = lucid.rand(2, 9, config.dim)
        with lucid.no_grad():
            pooled = _np(pooler(tokens))
            averaged = _np(tokens.mean(dim=1))
        assert np.abs(pooled - averaged).max() > 1e-3
