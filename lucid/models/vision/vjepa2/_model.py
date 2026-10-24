"""V-JEPA 2 model family and its private transformer building blocks.

The family keeps its checkpoint-sensitive encoder, predictor and rotary
attention implementation in this module.  Only the model, task wrapper and
output are public; the action-conditioned family imports private building
blocks without creating a second public component module.
"""

import math
from dataclasses import dataclass
from typing import ClassVar, TypedDict, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._dtype import _float32_or_wider
from lucid._engine_scope import untraced
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import ImageClassificationOutput, ModelOutput
from lucid.models._tasks import ImageClassificationModel
from lucid.models._utils._classification import DropPath
from lucid.models._utils._common import update_averages
from lucid.models.vision.vjepa2._config import VJEPA2Config

__all__ = [
    "VJEPA2Model",
    "VJEPA2ForVideoClassification",
    "VJEPA2Output",
]


class _EncoderArguments(TypedDict):
    image_size: int
    patch_size: int
    tubelet_size: int
    in_channels: int
    dim: int
    depth: int
    num_heads: int
    mlp_ratio: float
    layer_norm_eps: float
    qkv_bias: bool
    init_std: float
    use_silu: bool
    wide_silu: bool
    use_rope: bool
    rope_base: float
    uniform_power: bool
    drop_rate: float
    attn_drop_rate: float
    drop_path_rate: float


def _gather_tokens(tokens: Tensor, indices: Tensor) -> Tensor:
    """Gather a per-batch token index matrix from ``(B, N, D)`` tokens."""
    batch, count, width = (
        int(indices.shape[0]),
        int(indices.shape[1]),
        int(tokens.shape[2]),
    )
    expanded = indices.unsqueeze(-1) + lucid.zeros(
        batch, count, width, dtype=indices.dtype, device=tokens.device
    )
    return lucid.gather(tokens, 1, expanded)


def _rotate_queries_or_keys(x: Tensor, positions: Tensor, base: float) -> Tensor:
    """Apply the upstream three-axis rotary pairing, including its layout.

    The released implementation repeats the frequency half as a block before
    rotating adjacent feature pairs.  That layout is intentionally retained:
    changing it would make a released checkpoint numerically incompatible.
    """
    head_dim = int(x.shape[-1])
    if head_dim % 2 != 0:
        raise ValueError(f"rotary head dimension must be even, got {head_dim}")
    half = head_dim // 2
    omega = 1.0 / (
        base ** (lucid.arange(half, device=positions.device).to(x.dtype) / float(half))
    )
    omega = omega.reshape((1,) * positions.ndim + (half,))
    freq = positions.to(x.dtype).unsqueeze(-1) * omega
    sin = lucid.sin(freq).repeat(*(1 for _ in range(positions.ndim)), 2)
    cos = lucid.cos(freq).repeat(*(1 for _ in range(positions.ndim)), 2)
    paired = x.reshape(*x.shape[:-1], half, 2)
    first = paired[..., 0]
    second = paired[..., 1]
    rotated = lucid.stack([-second, first], dim=-1).reshape(*x.shape)
    return x * cos + rotated * sin


def _axis_positions(
    ids: Tensor, height: int, width: int, grid_size: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Convert flattened token ids to depth, row and column coordinates."""
    per_frame = height * width
    depth = ids // per_frame
    remainder = ids - depth * per_frame
    row = remainder // width
    col = remainder - row * width
    spatial_scale = float(grid_size) / float(height)
    return (
        depth.to(lucid.float32),
        row.to(lucid.float32) * spatial_scale,
        col.to(lucid.float32) * spatial_scale,
    )


class _RoPEAttention(nn.Module):
    """Fused-QKV attention with the released three-axis rotary layout."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid_size: int,
        rope_base: float,
        qkv_bias: bool,
        attn_drop_rate: float,
        proj_drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim must be divisible by num_heads, got {dim} and {num_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.grid_size = grid_size
        self.rope_base = rope_base
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop_rate = attn_drop_rate
        self.proj_drop = nn.Dropout(proj_drop_rate)
        # The upstream attention gives one third of a head to each axis,
        # rounded down to an even number, and leaves a residual tail alone.
        axis_dim = 2 * ((self.head_dim // 3) // 2)
        self.depth_dim = axis_dim
        self.height_dim = axis_dim
        self.width_dim = axis_dim

    @override
    def forward(
        self,
        x: Tensor,
        token_indices: Tensor | None = None,
        *,
        temporal: int,
        height: int,
        width: int,
        attn_mask: Tensor | None = None,
        action_tokens: int = 0,
    ) -> Tensor:
        batch, count, dim = (int(s) for s in x.shape)
        qkv = self.qkv(x).reshape(batch, count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, value = qkv[0], qkv[1], qkv[2]

        if action_tokens:
            return self._forward_action_conditioned(
                q,
                k,
                value,
                temporal,
                height,
                width,
                action_tokens,
                attn_mask,
                batch,
                count,
                dim,
            )

        if token_indices is None:
            positions = lucid.arange(temporal * height * width, device=x.device)
            positions = positions[:count]
        else:
            positions = token_indices
            if positions.ndim == 2:
                positions = positions.unsqueeze(1).repeat(1, self.num_heads, 1)
        depth, row, col = _axis_positions(positions, height, width, self.grid_size)
        offset = 0
        q_parts: list[Tensor] = []
        k_parts: list[Tensor] = []
        for coordinate, span in (
            (depth, self.depth_dim),
            (row, self.height_dim),
            (col, self.width_dim),
        ):
            if span:
                q_parts.append(
                    _rotate_queries_or_keys(
                        q[..., offset : offset + span], coordinate, self.rope_base
                    )
                )
                k_parts.append(
                    _rotate_queries_or_keys(
                        k[..., offset : offset + span], coordinate, self.rope_base
                    )
                )
                offset += span
        if offset < self.head_dim:
            q_parts.append(q[..., offset:])
            k_parts.append(k[..., offset:])
        q = lucid.cat(q_parts, dim=-1)
        k = lucid.cat(k_parts, dim=-1)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            value,
            attn_mask=attn_mask,
            dropout_p=self._dropout_p(),
        )
        attended = attended.permute(0, 2, 1, 3).reshape(batch, count, dim)
        return self.proj_drop(self.proj(attended))

    def _dropout_p(self) -> float:
        """Attention dropout, which is an inference-time no-op.

        ``F.scaled_dot_product_attention`` draws its mask whenever the
        probability is positive, so a module that forwards its rate
        unconditionally keeps dropping in ``eval()`` and makes inference
        non-deterministic.  The gate belongs here, as it does in BERT.
        """
        return self.attn_drop_rate if self.training else 0.0

    def _forward_action_conditioned(
        self,
        q: Tensor,
        k: Tensor,
        value: Tensor,
        temporal: int,
        height: int,
        width: int,
        action_tokens: int,
        attn_mask: Tensor | None,
        batch: int,
        count: int,
        dim: int,
    ) -> Tensor:
        """Rotate and merge action/state tokens with each spatial grid."""
        del dim
        heads, head_dim = int(q.shape[1]), int(q.shape[-1])
        frame_tokens = height * width
        expected = temporal * (action_tokens + frame_tokens)
        if count != expected:
            raise ValueError(
                f"action-conditioned sequence has {count} tokens, expected {expected}"
            )
        q = q.reshape(batch, heads, temporal, action_tokens + frame_tokens, head_dim)
        k = k.reshape(batch, heads, temporal, action_tokens + frame_tokens, head_dim)
        value = value.reshape(
            batch, heads, temporal, action_tokens + frame_tokens, head_dim
        )
        action_q: list[Tensor] = []
        action_k: list[Tensor] = []
        action_v: list[Tensor] = []
        temporal_positions = lucid.arange(temporal, device=q.device)
        for index in range(action_tokens):
            q_action = q[:, :, :, index, :]
            k_action = k[:, :, :, index, :]
            q_depth = _rotate_queries_or_keys(
                q_action[..., : self.depth_dim], temporal_positions, self.rope_base
            )
            k_depth = _rotate_queries_or_keys(
                k_action[..., : self.depth_dim], temporal_positions, self.rope_base
            )
            action_q.append(
                lucid.cat([q_depth, q_action[..., self.depth_dim :]], dim=-1)
            )
            action_k.append(
                lucid.cat([k_depth, k_action[..., self.depth_dim :]], dim=-1)
            )
            action_v.append(value[:, :, :, index, :])
        frame_q = q[:, :, :, action_tokens:, :].reshape(
            batch, heads, temporal * frame_tokens, head_dim
        )
        frame_k = k[:, :, :, action_tokens:, :].reshape(
            batch, heads, temporal * frame_tokens, head_dim
        )
        ids = lucid.arange(temporal * frame_tokens, device=q.device)
        depth, row, col = _axis_positions(ids, height, width, self.grid_size)
        offset = 0
        frame_q_parts: list[Tensor] = []
        frame_k_parts: list[Tensor] = []
        for coordinate, span in (
            (depth, self.depth_dim),
            (row, self.height_dim),
            (col, self.width_dim),
        ):
            if span:
                frame_q_parts.append(
                    _rotate_queries_or_keys(
                        frame_q[..., offset : offset + span], coordinate, self.rope_base
                    )
                )
                frame_k_parts.append(
                    _rotate_queries_or_keys(
                        frame_k[..., offset : offset + span], coordinate, self.rope_base
                    )
                )
                offset += span
        if offset < head_dim:
            frame_q_parts.append(frame_q[..., offset:])
            frame_k_parts.append(frame_k[..., offset:])
        frame_q = lucid.cat(frame_q_parts, dim=-1).reshape(
            batch, heads, temporal, frame_tokens, head_dim
        )
        frame_k = lucid.cat(frame_k_parts, dim=-1).reshape(
            batch, heads, temporal, frame_tokens, head_dim
        )
        action_q_tensor = lucid.stack(action_q, dim=3)
        action_k_tensor = lucid.stack(action_k, dim=3)
        action_v_tensor = lucid.stack(action_v, dim=3)
        q = lucid.cat([action_q_tensor, frame_q], dim=3).reshape(
            batch, heads, count, head_dim
        )
        k = lucid.cat([action_k_tensor, frame_k], dim=3).reshape(
            batch, heads, count, head_dim
        )
        value = lucid.cat(
            [action_v_tensor, value[:, :, :, action_tokens:, :]], dim=3
        ).reshape(batch, heads, count, head_dim)
        attended = F.scaled_dot_product_attention(
            q, k, value, attn_mask=attn_mask, dropout_p=self._dropout_p()
        )
        attended = attended.permute(0, 2, 1, 3).reshape(batch, count, heads * head_dim)
        return self.proj_drop(self.proj(attended))


class _FeedForward(nn.Module):
    """The GELU or wide-SiLU feed-forward used by the released blocks."""

    def __init__(
        self,
        dim: int,
        hidden: int,
        use_silu: bool,
        wide_silu: bool,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_silu = use_silu
        if use_silu:
            inner = hidden
            if wide_silu:
                inner = (int(2 * hidden / 3) + 7) // 8 * 8
            self.fc1 = nn.Linear(dim, inner)
            self.fc2 = nn.Linear(dim, inner)
            self.fc3 = nn.Linear(inner, dim)
        else:
            self.fc1 = nn.Linear(dim, hidden)
            self.fc2 = nn.Linear(hidden, dim)
        # The gated path takes no dropout upstream; the plain one takes it
        # on both sides of the output projection.
        self.drop = nn.Dropout(0.0 if use_silu else drop_rate)

    @override
    def forward(self, x: Tensor) -> Tensor:
        if self.use_silu:
            first = self.fc1(x)
            second = self.fc2(x)
            return self.fc3(F.silu(first) * second)
        hidden = self.drop(F.gelu(self.fc1(x)))
        return self.drop(self.fc2(hidden))


class _Block(nn.Module):
    """Pre-normalised attention plus feed-forward residual block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        hidden: int,
        grid_size: int,
        rope_base: float,
        layer_norm_eps: float,
        qkv_bias: bool,
        use_silu: bool,
        wide_silu: bool,
        attn_drop_rate: float,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.attn = _RoPEAttention(
            dim,
            num_heads,
            grid_size,
            rope_base,
            qkv_bias,
            attn_drop_rate,
            drop_rate,
        )
        self.drop_path = DropPath(drop_path_rate)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.mlp = _FeedForward(dim, hidden, use_silu, wide_silu, drop_rate)

    @override
    def forward(
        self,
        x: Tensor,
        token_indices: Tensor | None = None,
        *,
        temporal: int,
        height: int,
        width: int,
        attn_mask: Tensor | None = None,
        action_tokens: int = 0,
    ) -> Tensor:
        normalized = self.norm1(x)
        if token_indices is None:
            attended = self.attn(
                normalized,
                temporal=temporal,
                height=height,
                width=width,
                attn_mask=attn_mask,
                action_tokens=action_tokens,
            )
        else:
            attended = self.attn(
                normalized,
                token_indices,
                temporal=temporal,
                height=height,
                width=width,
                attn_mask=attn_mask,
                action_tokens=action_tokens,
            )
        attended = attended
        x = x + self.drop_path(attended)
        residual = self.mlp(self.norm2(x))
        return x + self.drop_path(residual)


class _RoPEVideoEncoder(nn.Module):
    """Checkpoint-compatible tubelet encoder used by both families."""

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        tubelet_size: int,
        in_channels: int,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        layer_norm_eps: float,
        qkv_bias: bool,
        init_std: float,
        use_silu: bool,
        wide_silu: bool,
        use_rope: bool,
        rope_base: float,
        uniform_power: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.grid_size = image_size // patch_size
        self.dim = dim
        self.use_rope = use_rope
        self.uniform_power = uniform_power
        self.patch_embed = nn.Conv3d(
            in_channels,
            dim,
            (tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )
        hidden = int(dim * mlp_ratio)
        rates = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        blocks = [
            _Block(
                dim,
                num_heads,
                hidden,
                self.grid_size,
                rope_base,
                layer_norm_eps,
                qkv_bias,
                use_silu,
                wide_silu,
                attn_drop_rate,
                drop_rate,
                rate,
            )
            for rate in rates
        ]
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(dim, eps=layer_norm_eps)
        _init_vjepa2_weights(self, init_std)
        for index, block in enumerate(blocks, start=1):
            _scale_projection(block.attn.proj, block.mlp, index, init_std)

    @override
    def forward(self, x: Tensor, indices: Tensor | None = None) -> Tensor:
        if x.ndim != 5:
            raise ValueError(f"video must have rank 5 (B, T, C, H, W), got {x.shape}")
        batch, frames, channels, height, width = (int(s) for s in x.shape)
        if channels != int(self.patch_embed.in_channels):
            raise ValueError(
                f"video has {channels} channels, expected {self.patch_embed.in_channels}"
            )
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("video height and width must be divisible by patch_size")
        if frames % self.tubelet_size != 0:
            raise ValueError("video frame count must be divisible by tubelet_size")
        tokens = self.patch_embed(x.permute(0, 2, 1, 3, 4))
        time = frames // self.tubelet_size
        rows = height // self.patch_size
        cols = width // self.patch_size
        tokens = tokens.reshape(batch, self.dim, -1).permute(0, 2, 1)
        if not self.use_rope:
            positions = _sincos_3d(
                self.dim,
                (time, rows, cols),
                self.uniform_power,
                device=tokens.device,
                dtype=tokens.dtype,
            )
            tokens = tokens + positions
        if indices is not None:
            tokens = _gather_tokens(tokens, indices)
        for block in self.blocks:
            if indices is None:
                tokens = block(tokens, temporal=time, height=rows, width=cols)
            else:
                tokens = block(tokens, indices, temporal=time, height=rows, width=cols)
        return self.norm(tokens)


class _RoPEVideoPredictor(nn.Module):
    """Masked-token predictor used during V-JEPA 2 pretraining."""

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        grid_depth: int,
        dim: int,
        predictor_dim: int,
        predictor_out_dim: int,
        depth: int,
        num_heads: int,
        num_mask_tokens: int,
        mlp_ratio: float,
        layer_norm_eps: float,
        qkv_bias: bool,
        init_std: float,
        use_silu: bool,
        wide_silu: bool,
        use_rope: bool,
        rope_base: float,
        zero_init_mask_tokens: bool,
        return_all_tokens: bool = False,
        uniform_power: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.grid_size = image_size // patch_size
        # Tubelet positions along time, fixed by the config as the reference's
        # ``grid_depth`` is.  Reading it off the largest token index instead
        # was a host sync inside forward, which no trace can follow (LCD-373).
        self.grid_depth = grid_depth
        self.predictor_dim = predictor_dim
        self.return_all_tokens = return_all_tokens
        self.uniform_power = uniform_power
        self.num_mask_tokens = num_mask_tokens
        self.use_rope = use_rope
        self.predictor_embed = nn.Linear(dim, predictor_dim)
        self.mask_tokens = nn.ParameterList(
            [
                nn.Parameter(lucid.zeros(1, 1, predictor_dim))
                for _ in range(num_mask_tokens)
            ]
        )
        hidden = int(predictor_dim * mlp_ratio)
        rates = [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]
        blocks = [
            _Block(
                predictor_dim,
                num_heads,
                hidden,
                self.grid_size,
                rope_base,
                layer_norm_eps,
                qkv_bias,
                use_silu,
                wide_silu,
                attn_drop_rate,
                drop_rate,
                rate,
            )
            for rate in rates
        ]
        self.predictor_blocks = nn.ModuleList(blocks)
        self.predictor_norm = nn.LayerNorm(predictor_dim, eps=layer_norm_eps)
        self.predictor_proj = nn.Linear(predictor_dim, predictor_out_dim)
        _init_vjepa2_weights(self, init_std)
        if zero_init_mask_tokens:
            for token in self.mask_tokens:
                nn.init.zeros_(token)
        else:
            for token in self.mask_tokens:
                nn.init.trunc_normal_(token, std=init_std)
        for index, block in enumerate(blocks, start=1):
            _scale_projection(block.attn.proj, block.mlp, index, init_std)

    @override
    def forward(
        self,
        context: Tensor,
        context_indices: Tensor,
        target_indices: Tensor,
        *,
        mask_index: int = 1,
    ) -> Tensor:
        if not 0 <= mask_index < self.num_mask_tokens:
            raise ValueError(
                f"mask_index must lie in [0, {self.num_mask_tokens}), got {mask_index}"
            )
        batch, context_count = int(context.shape[0]), int(context.shape[1])
        target_count = int(target_indices.shape[1])
        tokens = self.predictor_embed(context)
        queries = self.mask_tokens[mask_index].repeat(batch, target_count, 1)
        if not self.use_rope:
            positions = _sincos_3d(
                self.predictor_dim,
                (self.grid_depth, self.grid_size, self.grid_size),
                self.uniform_power,
                device=tokens.device,
                dtype=tokens.dtype,
            ).repeat(batch, 1, 1)
            tokens = tokens + _gather_tokens(positions, context_indices)
            queries = queries + _gather_tokens(positions, target_indices)
        hidden = lucid.cat([tokens, queries], dim=1)
        masks = lucid.cat([context_indices, target_indices], dim=1)
        order = masks.argsort(dim=1)
        gather_order = order.unsqueeze(-1) + lucid.zeros(
            batch,
            context_count + target_count,
            self.predictor_dim,
            dtype=order.dtype,
            device=hidden.device,
        )
        hidden = lucid.gather(hidden, 1, gather_order)
        sorted_masks = lucid.gather(masks, 1, order)
        rows = self.grid_size
        for block in self.predictor_blocks:
            hidden = block(
                hidden,
                sorted_masks,
                temporal=self.grid_depth,
                height=rows,
                width=rows,
            )
        hidden = self.predictor_norm(hidden)
        inverse = order.argsort(dim=1)
        inverse_gather = inverse.unsqueeze(-1) + lucid.zeros(
            batch,
            context_count + target_count,
            self.predictor_dim,
            dtype=inverse.dtype,
            device=hidden.device,
        )
        hidden = lucid.gather(hidden, 1, inverse_gather)
        if self.return_all_tokens:
            selected = hidden
        else:
            selected = hidden[:, context_count:]
        return self.predictor_proj(selected)


class _ActionConditionedPredictor(nn.Module):
    """Frame-causal predictor that interleaves action and state tokens."""

    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        predictor_dim: int,
        depth: int,
        num_heads: int,
        action_dim: int,
        state_dim: int,
        extrinsics_dim: int,
        use_extrinsics: bool,
        frame_causal: bool,
        mlp_ratio: float,
        layer_norm_eps: float,
        qkv_bias: bool,
        init_std: float,
        use_silu: bool,
        wide_silu: bool,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.grid_size = image_size // patch_size
        self.use_extrinsics = use_extrinsics
        self.frame_causal = frame_causal
        self.predictor_dim = predictor_dim
        self.predictor_embed = nn.Linear(embed_dim, predictor_dim)
        self.action_encoder = nn.Linear(action_dim, predictor_dim)
        self.state_encoder = nn.Linear(state_dim, predictor_dim)
        # The upstream predictor owns this projection even when the default
        # release does not feed extrinsics at runtime.  Keeping the parameter
        # present makes the checkpoint layout strict-loadable for both modes.
        self.extrinsics_encoder = nn.Linear(extrinsics_dim, predictor_dim)
        hidden = int(predictor_dim * mlp_ratio)
        blocks = [
            _Block(
                predictor_dim,
                num_heads,
                hidden,
                self.grid_size,
                rope_base,
                layer_norm_eps,
                qkv_bias,
                use_silu,
                wide_silu,
                0.0,
            )
            for _ in range(depth)
        ]
        self.predictor_blocks = nn.ModuleList(blocks)
        self.predictor_norm = nn.LayerNorm(predictor_dim, eps=layer_norm_eps)
        self.predictor_proj = nn.Linear(predictor_dim, embed_dim)
        _init_vjepa2_weights(self, init_std)
        for index, block in enumerate(blocks, start=1):
            _scale_projection(block.attn.proj, block.mlp, index, init_std)

    @override
    def forward(
        self,
        x: Tensor,
        actions: Tensor,
        states: Tensor,
        extrinsics: Tensor | None = None,
    ) -> Tensor:
        batch, token_count, _ = (int(s) for s in x.shape)
        spatial = self.grid_size * self.grid_size
        if token_count % spatial != 0:
            raise ValueError(
                f"context token count {token_count} is not divisible by spatial "
                f"grid size {spatial}"
            )
        steps = token_count // spatial
        for name, tensor in (("actions", actions), ("states", states)):
            if (
                tensor.ndim != 3
                or int(tensor.shape[0]) != batch
                or int(tensor.shape[1]) != steps
            ):
                raise ValueError(
                    f"{name} must have shape (B, {steps}, width), got {tensor.shape}"
                )
        if self.use_extrinsics:
            if (
                extrinsics is None
                or extrinsics.ndim != 3
                or int(extrinsics.shape[1]) != steps
            ):
                raise ValueError(
                    f"extrinsics must have shape (B, {steps}, width) when enabled"
                )
        hidden = self.predictor_embed(x)
        hidden = hidden.reshape(batch, steps, spatial, self.predictor_dim)
        action = self.action_encoder(actions).unsqueeze(2)
        state = self.state_encoder(states).unsqueeze(2)
        pieces = [action, state]
        if self.use_extrinsics:
            assert extrinsics is not None
            pieces.append(self.extrinsics_encoder(extrinsics).unsqueeze(2))
        pieces.append(hidden)
        hidden = lucid.cat(pieces, dim=2).reshape(
            batch, steps * (spatial + len(pieces) - 1), self.predictor_dim
        )
        cond = len(pieces) - 1
        total_per_step = spatial + cond
        masks = _action_attention_mask(
            steps, total_per_step, hidden.device, self.frame_causal
        )
        # The conditioning tokens are not part of the spatial rotary grid.  A
        # token index of the same frame is sufficient for the causal attention
        # path; spatial token positions are reconstructed inside each block.
        for block in self.predictor_blocks:
            hidden = block(
                hidden,
                temporal=steps,
                height=self.grid_size,
                width=self.grid_size,
                attn_mask=masks,
                action_tokens=cond,
            )
        hidden = hidden.reshape(batch, steps, total_per_step, self.predictor_dim)
        hidden = hidden[:, :, cond:, :].reshape(batch, token_count, self.predictor_dim)
        return self.predictor_proj(self.predictor_norm(hidden))


def _action_attention_mask(
    steps: int, tokens_per_step: int, device: DeviceLike, causal: bool
) -> Tensor | None:
    if not causal:
        return None
    mask = lucid.zeros(steps * tokens_per_step, steps * tokens_per_step, device=device)
    for current in range(steps):
        begin = current * tokens_per_step
        end = begin + tokens_per_step
        past_end = end
        mask[begin:end, :past_end] = 1.0
    return mask.to(lucid.bool_)


def _sincos_3d(
    dim: int,
    grid: tuple[int, int, int],
    uniform_power: bool,
    *,
    device: lucid.device,
    dtype: lucid.dtype,
) -> Tensor:
    """Build a fixed 3-D sine/cosine table for the optional non-RoPE path.

    Built where the tokens it is added to live and cast once to their dtype:
    a CPU table met Metal tokens with a device mismatch (LCD-639).  In
    float64 on the CPU, as the reference builds it, so a checkpoint sees the
    positions it was trained on; in float32 on Metal, which has no float64.
    It depends
    on the grid alone, so it is built outside any active trace — a recording
    holds it as a constant feed, never the ops that make it (LCD-627).
    """
    time, rows, cols = grid
    if uniform_power:
        each = int(math.ceil(dim / 6) * 2)
        widths = (each, each, each)
    else:
        widths = (dim // 2, dim // 4, dim // 4)
    work = lucid.float64 if device.type == "cpu" else _float32_or_wider(dtype)
    with untraced():
        zeros = lucid.zeros(time, rows, cols, dtype=work, device=device)
        coords = (
            lucid.arange(time, dtype=work, device=device).reshape(time, 1, 1) + zeros,
            lucid.arange(rows, dtype=work, device=device).reshape(1, rows, 1) + zeros,
            lucid.arange(cols, dtype=work, device=device).reshape(1, 1, cols) + zeros,
        )
        axes: list[Tensor] = []
        for coordinate, width in zip(coords, widths, strict=True):
            flat = coordinate.reshape(-1, 1)
            exponent = lucid.arange(width // 2, dtype=work, device=device)
            omega = 1.0 / (10_000.0 ** (exponent * (2.0 / width)))
            angles = flat * omega.reshape(1, -1)
            encoded = lucid.cat([lucid.sin(angles), lucid.cos(angles)], dim=1)
            if int(encoded.shape[1]) < width:
                pad = lucid.zeros(
                    int(encoded.shape[0]),
                    width - int(encoded.shape[1]),
                    dtype=work,
                    device=device,
                )
                encoded = lucid.cat([encoded, pad], dim=1)
            axes.append(encoded[:, :width])
        table = lucid.cat(axes, dim=1)[:, :dim]
        return table.reshape(1, time * rows * cols, dim).to(dtype)


def _scale_projection(
    attention_proj: nn.Linear,
    mlp: _FeedForward,
    depth_index: int,
    init_std: float,
) -> None:
    factor = init_std / math.sqrt(2.0 * float(depth_index))
    nn.init.trunc_normal_(attention_proj.weight, std=factor)
    # The upstream implementation rescales ``fc2`` even for the optional
    # gated feed-forward path; retain that checkpoint convention.
    nn.init.trunc_normal_(mlp.fc2.weight, std=factor)


def _init_vjepa2_weights(module: nn.Module, std: float = 0.02) -> None:
    """Apply the official truncated-normal and LayerNorm initialisation."""

    def initialize(layer: nn.Module) -> None:
        if isinstance(layer, (nn.Linear, nn.Conv3d)):
            nn.init.trunc_normal_(layer.weight, std=std)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            if layer.weight is not None:
                nn.init.ones_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    module.apply(initialize)


@dataclass(slots=True)
class VJEPA2Output(ModelOutput):
    r"""Output of a representation pass or masked prediction step.

    ``tokens`` is populated for a plain representation pass.  When both mask
    index matrices are supplied, ``prediction``, ``target`` and ``loss`` hold
    the predictor result and the representation-space training objective.
    """

    tokens: Tensor | None = None
    prediction: Tensor | None = None
    target: Tensor | None = None
    loss: Tensor | None = None
    context_indices: Tensor | None = None
    target_indices: Tensor | None = None


class _SelfAttentionBlock(nn.Module):
    """A plain pre-norm block, the probe's own tower above the backbone.

    The probe's blocks carry no rotary geometry: the released code builds
    them from the position-free ``Attention``, not from the encoder's
    ``RoPEAttention``, because the tokens reaching them already carry the
    encoder's positional information.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, eps: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = _FeedForward(dim, int(dim * mlp_ratio), False, False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        batch, count, width = (int(s) for s in x.shape)
        qkv = self.qkv(self.norm1(x))
        qkv = qkv.reshape(batch, count, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        attended = attended.permute(0, 2, 1, 3).reshape(batch, count, width)
        x = x + self.proj(attended)
        return x + self.mlp(self.norm2(x))


class _AttentivePooler(nn.Module):
    r"""The paper's evaluation probe: a tower, then one learned query.

    V-JEPA 2 evaluates a frozen backbone through this rather than through
    a linear map on averaged tokens.  Two details separate it from the
    pooler V-JEPA 1 used, and both change the parameter set:

    * the cross-attention has **no output projection** — the released
      implementation carries that ``Linear`` commented out, and the
      published classifier checkpoints have no tensor for it;
    * ``num_pooler_layers`` self-attention blocks run *before* the query
      attends, three in every released classifier.
    """

    def __init__(self, config: VJEPA2Config) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.blocks = nn.ModuleList(
            [
                _SelfAttentionBlock(
                    config.dim,
                    config.num_heads,
                    config.mlp_ratio,
                    config.layer_norm_eps,
                )
                for _ in range(config.num_pooler_layers)
            ]
        )
        self.query_token = nn.Parameter(lucid.zeros(1, 1, config.dim))
        self.norm_keys = nn.LayerNorm(config.dim, eps=config.layer_norm_eps)
        self.query = nn.Linear(config.dim, config.dim)
        self.key = nn.Linear(config.dim, config.dim)
        self.value = nn.Linear(config.dim, config.dim)
        self.norm_out = nn.LayerNorm(config.dim, eps=config.layer_norm_eps)
        self.mlp = _FeedForward(
            config.dim, int(config.dim * config.mlp_ratio), False, False
        )

        # The probe is a tower and is initialised as one — the same
        # truncated normal and the same depth-scaled residual outputs the
        # encoder gets.  Left out, it starts at the framework's fan-in
        # default, which no released number was measured at.
        _init_vjepa2_weights(self, config.init_std)
        nn.init.trunc_normal_(self.query_token, std=config.init_std)
        for depth, block in enumerate(self.blocks, start=1):
            tower = block
            _scale_projection(tower.proj, tower.mlp, depth, config.init_std)
        # The cross-attention carries no projection to narrow, and upstream
        # narrows its feed-forward by the last block's factor rather than
        # by one past it — with no blocks at all that factor is the first.
        cross_depth = max(config.num_pooler_layers, 1)
        nn.init.trunc_normal_(
            self.mlp.fc2.weight,
            std=config.init_std / math.sqrt(2.0 * float(cross_depth)),
        )

    @override
    def forward(self, tokens: Tensor) -> Tensor:
        batch, _, width = (int(s) for s in tokens.shape)
        for block in self.blocks:
            tokens = block(tokens)
        count = int(tokens.shape[1])
        keys = self.norm_keys(tokens)

        def split_heads(value: Tensor, length: int) -> Tensor:
            return value.reshape(batch, length, self.num_heads, self.head_dim).permute(
                0, 2, 1, 3
            )

        query_input = self.query_token.repeat(batch, 1, 1)
        query = split_heads(self.query(query_input), 1)
        key = split_heads(self.key(keys), count)
        value = split_heads(self.value(keys), count)
        attended = F.scaled_dot_product_attention(query, key, value)
        pooled = query_input + attended.permute(0, 2, 1, 3).reshape(batch, 1, width)
        pooled = pooled + self.mlp(self.norm_out(pooled))
        return pooled.reshape(batch, width)


class VJEPA2Model(PretrainedModel, BackboneMixin):
    r"""V-JEPA 2 context encoder, EMA target encoder and predictor.

    Clips use ``(B, T, C, H, W)``.  A plain call returns the target encoder's
    token map.  Supplying both ``context_indices`` and ``target_indices`` runs
    one masked representation-prediction step, which is useful for training or
    checking the predictor in isolation.  Indices are flattened tubelet
    positions in row-major ``(time, height, width)`` order.
    """

    config_class: ClassVar[type[VJEPA2Config]] = VJEPA2Config

    def __init__(self, config: VJEPA2Config) -> None:
        super().__init__(config)
        self.config: VJEPA2Config = config
        common: _EncoderArguments = {
            "image_size": config.image_size,
            "patch_size": config.patch_size,
            "tubelet_size": config.tubelet_size,
            "in_channels": config.in_channels,
            "dim": config.dim,
            "depth": config.depth,
            "num_heads": config.num_heads,
            "mlp_ratio": config.mlp_ratio,
            "layer_norm_eps": config.layer_norm_eps,
            "qkv_bias": config.qkv_bias,
            "init_std": config.init_std,
            "use_silu": config.use_silu,
            "wide_silu": config.wide_silu,
            "use_rope": config.use_rope,
            "rope_base": config.rope_base,
            "uniform_power": config.uniform_power,
            "drop_rate": config.drop_rate,
            "attn_drop_rate": config.attn_drop_rate,
            "drop_path_rate": config.drop_path_rate,
        }
        self.encoder = _RoPEVideoEncoder(**common)
        self.target_encoder = _RoPEVideoEncoder(**common)
        self.predictor = _RoPEVideoPredictor(
            image_size=config.image_size,
            patch_size=config.patch_size,
            grid_depth=config.token_grid[0],
            dim=config.dim,
            predictor_dim=config.predictor_dim,
            predictor_out_dim=config.predictor_output_dim,
            depth=config.predictor_depth,
            num_heads=config.predictor_heads,
            num_mask_tokens=config.predictor_num_mask_tokens,
            mlp_ratio=config.predictor_mlp_ratio,
            layer_norm_eps=config.layer_norm_eps,
            qkv_bias=config.qkv_bias,
            init_std=config.init_std,
            use_silu=config.use_silu,
            wide_silu=config.wide_silu,
            use_rope=config.use_rope,
            rope_base=config.rope_base,
            zero_init_mask_tokens=config.zero_init_mask_tokens,
            return_all_tokens=config.predictor_return_all_tokens,
            uniform_power=config.uniform_power,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            drop_path_rate=config.drop_path_rate,
        )
        nn.utils.copy_parameters_and_buffers(self.encoder, self.target_encoder)
        self.target_encoder.requires_grad_(False)
        self._feature_info = [
            FeatureInfo(stage=1, num_channels=config.dim, reduction=config.patch_size)
        ]

    @override
    @property
    def feature_info(self) -> list[FeatureInfo]:
        """The single tubelet-token feature stage."""
        return self._feature_info

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        """Return the mean target representation for backbone consumers."""
        return self.encode(x)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return context-encoder and predictor parameters only."""
        return list(self.encoder.parameters()) + list(self.predictor.parameters())

    def momentum(self, step: int, total_steps: int) -> float:
        """The target encoder's momentum at a training step.

        Parameters
        ----------
        step : int
            Steps already taken.
        total_steps : int
            Steps the run will take in total.

        Returns
        -------
        float
            Linear from ``config.ema[0]`` toward ``config.ema[1]`` over
            ``total_steps * config.ema_schedule_scale`` steps.

        Notes
        -----
        The released pretraining configuration sets both ends of ``ema``
        to ``0.99925``, so this is a constant for the published runs —
        unlike V-JEPA 1, which ramps.  The schedule is still built from
        the two endpoints, because a method that takes a step and a
        horizon and reads neither cannot be given a different recipe.
        """
        if total_steps < 1:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        start, end = self.config.ema
        horizon = total_steps * self.config.ema_schedule_scale
        fraction = min(max(step / horizon, 0.0), 1.0)
        return start + (end - start) * fraction

    def update_target(self, momentum: float) -> None:
        """Update the non-gradient target encoder after an optimiser step."""
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"momentum must lie in [0, 1], got {momentum}")
        update_averages(self.encoder, [(self.target_encoder, momentum)])

    def _check_clip(self, x: Tensor) -> None:
        expected = (
            self.config.num_frames,
            self.config.in_channels,
            self.config.image_size,
            self.config.image_size,
        )
        if x.ndim != 5 or tuple(int(s) for s in x.shape[1:]) != expected:
            raise ValueError(
                "clips must have shape (B, T, C, H, W) = "
                f"(B, {expected[0]}, {expected[1]}, {expected[2]}, {expected[3]}), "
                f"got {tuple(x.shape)}"
            )

    def tokens(self, x: Tensor, *, target: bool = True) -> Tensor:
        """Return the full ``(B, N, dim)`` token map."""
        self._check_clip(x)
        encoder = self.target_encoder if target else self.encoder
        return encoder(x)

    def encode(self, x: Tensor) -> Tensor:
        """Return mean-pooled target-encoder representations ``(B, dim)``."""
        return self.tokens(x).mean(dim=1)

    def _objective(self, prediction: Tensor, target: Tensor) -> Tensor:
        if self.config.objective == "l2":
            return F.mse_loss(prediction, target)
        if self.config.objective == "smooth_l1":
            return F.smooth_l1_loss(prediction, target)
        return F.l1_loss(prediction, target)

    @override
    def forward(
        self,
        x: Tensor,
        context_indices: Tensor | None = None,
        target_indices: Tensor | None = None,
        *,
        mask_index: int = 1,
    ) -> VJEPA2Output:
        """Run representation extraction or a masked prediction step."""
        self._check_clip(x)
        if (context_indices is None) != (target_indices is None):
            raise ValueError(
                "context_indices and target_indices must be supplied together"
            )
        if context_indices is None or target_indices is None:
            return VJEPA2Output(tokens=self.tokens(x))

        with lucid.no_grad():
            target = self.target_encoder(x)
            if self.config.normalize_targets:
                target = F.layer_norm(target, (int(target.shape[-1]),))
            target = _gather_tokens(target, target_indices)
        context = self.encoder(x, context_indices)
        prediction = self.predictor(
            context,
            context_indices,
            target_indices,
            mask_index=mask_index,
        )
        # With ``predictor_return_all_tokens`` the predictor answers at the
        # context positions too, and only the held-out ones have a target to
        # be scored against — the objective takes that tail, while the output
        # carries the whole map the caller asked for.
        predicted = (
            prediction[:, -int(target_indices.shape[1]) :]
            if self.config.predictor_return_all_tokens
            else prediction
        )
        return VJEPA2Output(
            tokens=None,
            prediction=prediction,
            target=target,
            loss=self._objective(predicted, target),
            context_indices=context_indices,
            target_indices=target_indices,
        )


class VJEPA2ForVideoClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""V-JEPA 2 read through the probe the paper evaluates with.

    Parameters
    ----------
    config : VJEPA2Config
        Frozen configuration; ``num_classes`` sizes the classifier and
        ``num_pooler_layers`` the probe's tower.

    Attributes
    ----------
    vjepa2 : VJEPA2Model
        The pretraining networks.  Its target encoder is frozen, and that
        is the tower this reads.
    pooler : Module
        The attentive probe: self-attention blocks, then one learned query
        cross-attending the feature map.
    head : nn.Linear
        The classifier on the pooled vector.

    Notes
    -----
    Registered under ``image-classification`` because that is the task
    this zoo has; the input is a clip rather than an image, which is what
    the class name says.  Averaging tokens and fitting a linear map would
    measure something else — the paper prices the difference in points.
    """

    config_class: ClassVar[type[VJEPA2Config]] = VJEPA2Config

    def __init__(self, config: VJEPA2Config) -> None:
        super().__init__(config)
        self.config: VJEPA2Config = config
        self.vjepa2 = VJEPA2Model(config)
        self.pooler = _AttentivePooler(config)
        self.head = nn.Linear(config.dim, config.num_classes)

    @override
    def forward(
        self, x: Tensor, labels: Tensor | None = None
    ) -> ImageClassificationOutput:
        """Classify a clip and optionally return cross-entropy loss.

        The backbone is the frozen target encoder, which is what the
        paper's probe reads.  Freezing is a property of those parameters,
        not of this call: wrapping the pass in ``no_grad`` would also
        stop a caller who unfroze them deliberately.
        """
        pooled = self.pooler(self.vjepa2.tokens(x))
        logits = self.head(pooled)
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return ImageClassificationOutput(logits=logits, loss=loss)
