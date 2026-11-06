"""Drifting's feature space: the ResNet-style MAE encoder and ConvNeXt-V2 (A.3–A.5).

The frozen network the drifting loss is computed in, one module apart from the generator
it reads the samples of.  It only reads the family config, so nothing here knows the
model holding it.
"""

from typing import override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._dtype import _float32_or_wider
from lucid._tensor.tensor import Tensor
from lucid.models.generative.drifting._config import DriftingConfig

__all__ = ["_DriftingFeatureEncoder"]

# ImageNet statistics the release normalises by before ConvNeXt-V2.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
# ConvNeXt-V2 reads 224-pixel images; the pixel generators emit 256.
_CONVNEXT_SIZE = 224


def _gn_groups(channels: int, max_groups: int = 32) -> int:
    """The largest group count ``<= max_groups`` that divides ``channels``."""
    g = min(max_groups, channels)
    while g > 1 and channels % g != 0:
        g -= 1
    return max(g, 1)


def _group_norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with the release's group rule and Flax's ``epsilon=1e-6``."""
    return nn.GroupNorm(_gn_groups(channels), channels, eps=1e-6)


def _safe_std(x: Tensor, dims: tuple[int, ...], eps: float = 1e-6) -> Tensor:
    """Population std over ``dims`` in float32-or-wider, ``sqrt(max(var, 0) + eps)``."""
    x = x.to(_float32_or_wider(x.dtype))
    mean = x.mean(dim=dims, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=dims)
    return lucid.sqrt(lucid.clamp(var, min=0.0) + eps)


def _tokens(feat: Tensor) -> Tensor:
    """``(B, C, H, W) -> (B, H*W, C)``, row-major locations."""
    b, c, h, w = (int(s) for s in feat.shape)
    return feat.reshape(b, c, h * w).permute(0, 2, 1)


def _space_to_depth(x: Tensor, patch: int) -> Tensor:
    """``b c (h p) (w p) -> b (p p c) h w`` with the release's ``(h2, w2, c)`` order."""
    if patch == 1:
        return x
    b, c, h, w = (int(s) for s in x.shape)
    x = x.reshape(b, c, h // patch, patch, w // patch, patch).permute(0, 3, 5, 1, 2, 4)
    return x.reshape(b, patch * patch * c, h // patch, w // patch)


def _add_feature_sets(
    out: dict[str, Tensor], name: str, feat: Tensor, config: DriftingConfig
) -> None:
    """Add one ``(B, C, H, W)`` map's feature sets to ``out``, in the release's order.

    Its locations as ``name``; its global ``_mean`` and ``_std``; then the
    means and stds pooled over each ``patch_mean_size`` / ``patch_std_size``
    square that tiles the map.
    """
    b, c, h, w = (int(s) for s in feat.shape)
    out[name] = _tokens(feat)
    if config.use_mean:
        out[f"{name}_mean"] = feat.mean(dim=(2, 3)).unsqueeze(1)
    if config.use_std:
        out[f"{name}_std"] = _safe_std(feat, (2, 3)).unsqueeze(1)
    for size in config.patch_mean_size:
        if h % size == 0 and w % size == 0:
            grouped = feat.reshape(b, c, h // size, size, w // size, size)
            out[f"{name}_mean_{size}"] = _tokens(grouped.mean(dim=(3, 5)))
    for size in config.patch_std_size:
        if h % size == 0 and w % size == 0:
            grouped = feat.reshape(b, c, h // size, size, w // size, size)
            out[f"{name}_std_{size}"] = _tokens(_safe_std(grouped, (3, 5)))


class _MAEBasicBlock(nn.Module):
    """ResNet basic block with GroupNorm: two 3x3 convs, projected skip on change."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        """Initialise the block. See the class docstring."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.gn1 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.gn2 = _group_norm(out_channels)
        # The release builds these in every block and Flax only materialises
        # them where the shape changes; building them only there is the same
        # parameter set.
        self.has_proj = stride != 1 or in_channels != out_channels
        if self.has_proj:
            self.proj_conv = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            self.proj_gn = _group_norm(out_channels)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """``relu(skip(x) + gn2(conv2(relu(gn1(conv1(x))))))``."""
        y = self.gn2(self.conv2(F.relu(self.gn1(self.conv1(x)))))
        skip = self.proj_gn(self.proj_conv(x)) if self.has_proj else x
        return F.relu(skip + y)


class _MAEResNetEncoder(nn.Module):
    """The latent-/pixel-MAE's encoder: a GroupNorm ResNet-34 at width ``C``."""

    def __init__(self, config: DriftingConfig) -> None:
        """Build the encoder from the family config's ``mae_*`` fields."""
        super().__init__()
        self.input_patch_size = config.mae_input_patch_size
        base = config.mae_base_channels
        in_ch = config.in_channels * self.input_patch_size**2
        self.conv1 = nn.Conv2d(in_ch, base, 3, stride=1, padding=1, bias=False)
        self.gn1 = _group_norm(base)
        stages: list[nn.ModuleList[_MAEBasicBlock]] = []
        norms: list[nn.GroupNorm] = []
        ch = base
        for stage, num_blocks in enumerate(config.mae_layers):
            out_ch = base * (2**stage)
            blocks = [_MAEBasicBlock(ch, out_ch, 2 if stage > 0 else 1)]
            blocks += [_MAEBasicBlock(out_ch, out_ch, 1) for _ in range(1, num_blocks)]
            stages.append(nn.ModuleList(blocks))
            norms.append(_group_norm(out_ch))
            ch = out_ch
        self.stages = nn.ModuleList(stages)
        self.stage_norms = nn.ModuleList(norms)

    def activations(self, x: Tensor, config: DriftingConfig) -> dict[str, Tensor]:
        """The release's ``MAEResNetJAX.get_activations``: every feature set, ``(B, T, D)``.

        Keys: ``norm_x``; ``conv1`` and ``layer{1-4}`` (stage outputs after
        their GroupNorm); ``layer{i}_blk{j}`` every ``every_k_block`` blocks
        (pre-norm); and for each, ``_mean``/``_std`` and the pooled
        ``_mean_{s}``/``_std_{s}`` sets.
        """
        x = _space_to_depth(x, self.input_patch_size)
        out: dict[str, Tensor] = {}
        work = _float32_or_wider(x.dtype)
        out["norm_x"] = lucid.sqrt((x.to(work) ** 2).mean(dim=(2, 3)) + 1e-6).unsqueeze(1)

        x = F.relu(self.gn1(self.conv1(x)))
        stage_feats: dict[str, Tensor] = {"conv1": x}
        block_feats: list[tuple[str, Tensor]] = []
        k = config.every_k_block
        for i, (blocks, norm) in enumerate(zip(self.stages, self.stage_norms, strict=True)):
            for j, block in enumerate(blocks, start=1):
                x = block(x)
                if k is not None and j % k == 0:
                    block_feats.append((f"layer{i + 1}_blk{j}", x))
            x = norm(x)
            stage_feats[f"layer{i + 1}"] = x
        for name, feat in [*stage_feats.items(), *block_feats]:
            _add_feature_sets(out, name, feat, config)
        return out


class _ChannelsLastLayerNorm(nn.Module):
    """ConvNeXt's LayerNorm over the channel axis of an ``NCHW`` map, eps 1e-6."""

    def __init__(self, dim: int) -> None:
        """Initialise the norm. See the class docstring."""
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(lucid.ones(dim))
        self.bias = nn.Parameter(lucid.zeros(dim))

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Normalise each location's channel vector."""
        y = F.layer_norm(x.permute(0, 2, 3, 1), (self.dim,), self.weight, self.bias, eps=1e-6)
        return y.permute(0, 3, 1, 2)


class _GRN(nn.Module):
    """ConvNeXt-V2 Global Response Normalisation on a channels-last map.

    The release's form: ``gx = sqrt(sum_hw x^2 + eps)``,
    ``nx = gx / (mean_c gx + eps)``, ``gamma * (x * nx) + beta + x``.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialise the GRN. See the class docstring."""
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(lucid.zeros(dim))
        self.beta = nn.Parameter(lucid.zeros(dim))

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Apply GRN to ``(B, H, W, C)``."""
        gx = lucid.sqrt((x * x).sum(dim=(1, 2), keepdim=True) + self.eps)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class _ConvNeXtV2Block(nn.Module):
    """ConvNeXt-V2 block: depthwise 7x7, LN, 4x MLP with GRN, residual."""

    def __init__(self, dim: int) -> None:
        """Initialise the block. See the class docstring."""
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.grn = _GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Advance an ``NCHW`` map."""
        y = self.dwconv(x).permute(0, 2, 3, 1)
        y = self.pwconv2(self.grn(F.gelu(self.pwconv1(self.norm(y)))))
        return x + y.permute(0, 3, 1, 2)


class _ConvNeXtV2(nn.Module):
    """ConvNeXt-V2-Base (``depths=(3, 3, 27, 3)``, ``dims=(128, 256, 512, 1024)``)."""

    def __init__(
        self, depths: tuple[int, ...] = (3, 3, 27, 3), dims: tuple[int, ...] = (128, 256, 512, 1024)
    ) -> None:
        """Initialise the backbone. See the class docstring."""
        super().__init__()
        layers: list[nn.Module] = [
            nn.Sequential(nn.Conv2d(3, dims[0], 4, stride=4), _ChannelsLastLayerNorm(dims[0]))
        ]
        layers.extend(
            nn.Sequential(
                _ChannelsLastLayerNorm(dims[i]), nn.Conv2d(dims[i], dims[i + 1], 2, stride=2)
            )
            for i in range(3)
        )
        self.downsample_layers = nn.ModuleList(layers)
        self.stages = nn.ModuleList(
            [
                nn.Sequential(*[_ConvNeXtV2Block(dims[i]) for _ in range(depths[i])])
                for i in range(4)
            ]
        )
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

    def activations(self, x: Tensor) -> dict[str, Tensor]:
        """The release's ``ConvNextV2.get_activations`` on ImageNet-normalised ``x``.

        The keys keep the release's spelling (``convenxt_*``).
        """
        x = F.interpolate(
            x,
            size=(_CONVNEXT_SIZE, _CONVNEXT_SIZE),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        work = _float32_or_wider(x.dtype)

        def normalize(y: Tensor) -> Tensor:
            y32 = y.to(work)
            mean = y32.mean(dim=1, keepdim=True)
            std = lucid.sqrt(((y32 - mean) ** 2).mean(dim=1, keepdim=True))
            return ((y32 - mean) / (std + 1e-3)).to(y.dtype)

        out: dict[str, Tensor] = {}
        for i in range(4):
            x = self.stages[i](self.downsample_layers[i](x))
            tokens = _tokens(normalize(x))
            if i > 0:
                out[f"convenxt_stage_{i}"] = tokens
            out[f"convenxt_stage_{i}_mean"] = tokens.mean(dim=1).unsqueeze(1)
            out[f"convenxt_stage_{i}_std"] = _safe_std(tokens, (1,)).unsqueeze(1)
        out["global_mean"] = self.norm(x.mean(dim=(2, 3))).unsqueeze(1)
        out["global_std"] = _safe_std(_tokens(normalize(x)), (1,)).unsqueeze(1)
        return out


class _DriftingFeatureEncoder(nn.Module):
    r"""The frozen feature map :math:`\phi` the drifting loss is computed in.

    Parameters
    ----------
    config : DriftingConfig
        Reads ``use_mae``, ``use_convnext``, the ``mae_*`` architecture and
        the A.5 feature options.
    build : bool, default=True
        Construct the encoders now.  ``False`` leaves an empty shell, which
        :meth:`DriftingForImageGeneration.load_feature_encoder` replaces —
        the MAE-640 encoder is far larger than the generator, and sampling
        never needs it.

    Attributes
    ----------
    mae : _MAEResNetEncoder
        Present when ``use_mae`` and built.
    convnext : _ConvNeXtV2
        Present when ``use_convnext`` and built.

    Notes
    -----
    Reference: Deng, Li, Li, Du, and He, arXiv:2602.04770, 2026,
    Appendices A.3–A.5.  The output is the release's ``activation_fn``: a
    ``"global"`` set holding the raw sample (the vanilla drifting loss
    without :math:`\phi`), then the MAE's sets, then ConvNeXt-V2's, which
    read ``(x + 1) / 2`` normalised by ImageNet statistics.

    Examples
    --------
    >>> import lucid
    >>> import lucid.models as models
    >>> model = models.create_model("drifting_base_2_gen", mae_base_channels=16,
    ...                             mae_layers=(1, 1, 1, 1), sample_size=8,
    ...                             hidden_size=32, cond_dim=32, depth=1, num_heads=4)
    >>> feats = model.load_feature_encoder()(lucid.randn(2, 4, 8, 8))
    >>> feats["global"].shape, feats["layer1"].shape
    ((2, 1, 256), (2, 64, 16))
    """

    def __init__(self, config: DriftingConfig, *, build: bool = True) -> None:
        """Initialise the encoder. See the class docstring for parameters."""
        super().__init__()
        self.config = config
        self.built = False
        if build:
            self.build()

    @property
    def needed(self) -> bool:
        """Whether the config asks for any encoder beyond the raw sample."""
        return self.config.use_mae or self.config.use_convnext

    def build(self) -> None:
        """Construct the configured encoders (idempotent)."""
        if self.built:
            return
        if self.config.use_mae:
            self.mae = _MAEResNetEncoder(self.config)
        if self.config.use_convnext:
            self.convnext = _ConvNeXtV2()
        self.built = True

    @override
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Every feature set of ``(B, C, H, W)`` samples, each ``(B, T, D)``."""
        if self.needed and not self.built:
            raise RuntimeError(
                "the drifting loss is configured in a learned feature space "
                "(use_mae / use_convnext) but no encoder has been built; call "
                "DriftingForImageGeneration.load_feature_encoder() first, or set "
                "use_mae=False and use_convnext=False to drift in the raw space"
            )
        config = self.config
        b = int(x.shape[0])
        feats: dict[str, Tensor] = {"global": x.reshape(b, 1, -1)}
        if config.use_mae:
            feats.update(self.mae.activations(x, config))
        if config.use_convnext:
            mean = lucid.tensor(list(_IMAGENET_MEAN), dtype=x.dtype, device=x.device.type)
            std = lucid.tensor(list(_IMAGENET_STD), dtype=x.dtype, device=x.device.type)
            rgb = ((x + 1.0) / 2.0 - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1)
            feats.update(self.convnext.activations(rgb))
        return feats
