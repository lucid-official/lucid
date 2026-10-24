"""Stable Diffusion weight converter — diffusers layout → Lucid.

Two sub-models, converted together because they ship together and a
half-converted pair is useless.  The mapping is mechanical but not a
rename: three shapes of transformation are involved.

**Nesting versus flat lists.**  Upstream groups by resolution —
``down_blocks.1.resnets.0`` — where Lucid keeps one flat ``ModuleList``
per role and indexes it ``level * layers_per_block + i``.  The orders
correspond exactly, so the mapping is arithmetic rather than a lookup
table, and the arithmetic is stated once in :func:`_flat_index`.

**Fused attention.**  ``attn1`` upstream is three separate ``to_q`` /
``to_k`` / ``to_v`` matrices; Lucid's self-attention keeps one fused
``in_proj_weight``.  ``attn2`` is *not* fused here, because its keys and
values are the conditioning's width rather than the query's, so those
three stay three.  That asymmetry is why the tensor counts differ —
654 here against 686 upstream, exactly the 32 saved by fusing
``attn1`` across sixteen blocks.

**The output bias lives elsewhere.**  Upstream carries no bias on q/k/v
and one on ``to_out.0``; Lucid's ``MultiheadAttention`` controls both
with one flag, so the attention is built bias-free and the block holds
``attnN_out_bias`` beside it.  ``to_out.0.bias`` maps there.

The autoencoder needs one more: upstream's mid-block attention uses
separate ``to_q`` / ``to_k`` / ``to_v`` (named ``query`` / ``key`` /
``value`` in older exports), which fuse the same way ``attn1`` does.
"""

import dataclasses
from typing import Any, cast

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file

from lucid.models.generative.stable_diffusion import (
    StableDiffusionConfig,
    StableDiffusionModel,
)
from lucid.nn import Module
from tools.convert_weights._base import (
    Architecture,
    ConversionSpec,
    UnknownSourceKey,
    register_arch,
)

_CITATION = (
    "@inproceedings{rombach2022high,\n"
    "  title={High-Resolution Image Synthesis with Latent Diffusion "
    "Models},\n"
    "  author={Rombach, Robin and Blattmann, Andreas and Lorenz, "
    "Dominik and Esser, Patrick and Ommer, Bj{\\\"o}rn},\n"
    "  booktitle={CVPR},\n"
    "  pages={10684--10695},\n"
    "  year={2022}\n"
    "}"
)

_RELEASES: dict[str, tuple[str, dict[str, object]]] = {
    "stable_diffusion": ("stable-diffusion-v1-5/stable-diffusion-v1-5", {}),
}


def _flat_index(level: int, block: int, layers: int) -> int:
    """Position of one resnet in Lucid's flat list.

    Parameters
    ----------
    level : int
        Resolution index.
    block : int
        Block within the resolution.
    layers : int
        ``layers_per_block``.

    Returns
    -------
    int
        Index into the flat ``ModuleList``.
    """
    return level * layers + block


class StableDiffusionArch(Architecture):
    """Conversion recipe for one released Stable Diffusion pair."""

    def __init__(self, model_name: str, tag: str = "CompVis_LAION") -> None:
        """Bind a factory name to its upstream repo.

        Parameters
        ----------
        model_name : str
            A key of ``_RELEASES``.
        tag : str, default="CompVis_LAION"
            Weight tag.
        """
        if model_name not in _RELEASES:
            raise ValueError(
                f"no release for {model_name!r}; known: {sorted(_RELEASES)}"
            )
        self.model_name = model_name
        self.tag = tag
        self.repo, self.overrides = _RELEASES[model_name]

    def _config(self) -> StableDiffusionConfig:
        return StableDiffusionConfig(**cast(dict[str, Any], self.overrides))

    def source_state_dict(self) -> dict[str, object]:
        """Download both sub-models and return one namespaced dict.

        Returns
        -------
        dict
            ``{"unet.<key>": array}`` and ``{"vae.<key>": array}``, with
            ``attn1`` / mid-block attention already fused.
        """
        out: dict[str, object] = {}
        for prefix, folder in (("unet", "unet"), ("vae", "vae")):
            path = hf_hub_download(
                self.repo, f"{folder}/diffusion_pytorch_model.safetensors"
            )
            for key, value in load_file(path).items():
                out[f"{prefix}.{key}"] = value
        return _fuse_attention(out)

    def target_model(self) -> Module:
        """Build the empty Lucid model the weights load into."""
        return StableDiffusionModel(self._config())

    def map_key(self, src_key: str) -> str | None:
        """Map one upstream key to its Lucid name.

        Parameters
        ----------
        src_key : str
            Namespaced upstream name, post-fusion.

        Returns
        -------
        str or None
            The Lucid name.  Nothing is dropped: every U-Net and
            autoencoder tensor has a home.

        Raises
        ------
        UnknownSourceKey
            For a key no rule here covers.
        """
        if src_key.startswith("unet."):
            mapped = _map_unet(src_key[len("unet.") :], self._config())
            # The U-Net hangs off ``StableDiffusionModel.unet``, so every
            # name it owns is prefixed once here rather than in each of
            # the dozen branches below.
            if mapped is None:
                raise UnknownSourceKey(src_key)
            return f"unet.{mapped}"
        if src_key.startswith("vae."):
            mapped = _map_vae(src_key[len("vae.") :], self._config())
            if mapped is None:
                raise UnknownSourceKey(src_key)
            return mapped
        raise UnknownSourceKey(src_key)

    def spec(self) -> ConversionSpec:
        """Return the static metadata written beside the weights."""
        config = self._config()
        return ConversionSpec(
            model_name=self.model_name,
            architecture="Stable Diffusion",
            repo_id=f"lucid-dl/{self.model_name.replace('_', '-')}",
            tag=self.tag,
            task="base",
            model_type="stable_diffusion",
            source=f"https://huggingface.co/{self.repo}",
            license="creativeml-openrail-m",
            num_classes=0,
            config=dataclasses.asdict(config),
            preprocessing={
                "resize": config.sample_size,
                "rescale": 1 / 127.5,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            citation=_CITATION,
            title="High-Resolution Image Synthesis with Latent Diffusion Models",
            paper_url="https://arxiv.org/abs/2112.10752",
            categories=["generative", "diffusion", "text-to-image"],
            datasets=["LAION-5B"],
        )


def _fuse_attention(state: dict[str, object]) -> dict[str, object]:
    """Concatenate q/k/v where Lucid keeps one fused projection.

    Parameters
    ----------
    state : dict
        Upstream tensors, namespaced.

    Returns
    -------
    dict
        The same, with each fused triple replaced by one
        ``in_proj_weight`` and its three sources removed.

    Notes
    -----
    Only ``attn1`` and the autoencoder's mid-block attention fuse.
    ``attn2`` keeps three because its key and value widths are the
    conditioning's, not the query's.
    """
    out: dict[str, object] = {}
    consumed: set[str] = set()
    for key in state:
        for trio in (("to_q", "to_k", "to_v"), ("query", "key", "value")):
            marker = f".{trio[0]}."
            if marker not in key or ".attn2." in key:
                continue
            stem, suffix = key.split(marker)
            parts = [
                cast(np.ndarray, state[f"{stem}.{name}.{suffix}"]) for name in trio
            ]
            # ``suffix`` is "weight" or "bias"; both fuse, into their own
            # names.  Dropping it here silently overwrote the fused
            # weight with the fused bias.
            out[f"{stem}.in_proj_{suffix}"] = np.concatenate(parts, axis=0)
            consumed.update(f"{stem}.{name}.{suffix}" for name in trio)
    for key, value in state.items():
        if key not in consumed:
            out.setdefault(key, value)
    return out


def _map_unet(key: str, config: StableDiffusionConfig) -> str | None:
    """Map one U-Net key, or ``None`` when no rule covers it.

    See :meth:`StableDiffusionArch.map_key`.
    """
    layers = config.unet_layers_per_block
    parts = key.split(".")

    if key.startswith("conv_in.") or key.startswith("conv_out."):
        return key
    if key.startswith("conv_norm_out."):
        return "norm_out." + parts[-1]
    if key.startswith("time_embedding.linear_1."):
        return "time_mlp.0." + parts[-1]
    if key.startswith("time_embedding.linear_2."):
        return "time_mlp.2." + parts[-1]

    if key.startswith("mid_block."):
        rest = ".".join(parts[1:])
        if rest.startswith("resnets.0."):
            return "mid_block_1." + _resnet_leaf(".".join(parts[3:]))
        if rest.startswith("resnets.1."):
            return "mid_block_2." + _resnet_leaf(".".join(parts[3:]))
        if rest.startswith("attentions.0."):
            return "mid_attn." + _attn_leaf(".".join(parts[3:]))
        return None

    for side, block_name, attn_name, sampler in (
        ("down_blocks", "down_blocks", "down_attns", "downsamplers"),
        ("up_blocks", "up_blocks", "up_attns", "upsamplers"),
    ):
        if not key.startswith(side + "."):
            continue
        level = int(parts[1])
        kind = parts[2]
        per = layers if side == "down_blocks" else layers + 1
        if kind == "resnets":
            index = _flat_index(level, int(parts[3]), per)
            return f"{block_name}.{index}." + _resnet_leaf(".".join(parts[4:]))
        if kind == "attentions":
            index = _flat_index(level, int(parts[3]), per)
            return f"{attn_name}.{index}." + _attn_leaf(".".join(parts[4:]))
        if kind in ("downsamplers", "upsamplers"):
            # Upstream nests ``.0.conv.``; Lucid holds one per level.
            return f"{sampler}.{level}." + parts[-1]
        return None
    return None


def _resnet_leaf(leaf: str) -> str:
    """Rename the inside of a residual block."""
    return {
        "norm1.weight": "norm1.weight",
        "norm1.bias": "norm1.bias",
        "norm2.weight": "norm2.weight",
        "norm2.bias": "norm2.bias",
        "conv1.weight": "conv1.weight",
        "conv1.bias": "conv1.bias",
        "conv2.weight": "conv2.weight",
        "conv2.bias": "conv2.bias",
        "time_emb_proj.weight": "time_proj.weight",
        "time_emb_proj.bias": "time_proj.bias",
        "conv_shortcut.weight": "shortcut.weight",
        "conv_shortcut.bias": "shortcut.bias",
    }.get(leaf, leaf)


def _attn_leaf(leaf: str) -> str:
    """Rename the inside of a spatial transformer."""
    if leaf.startswith(("norm.", "proj_in.", "proj_out.")):
        return leaf
    if not leaf.startswith("transformer_blocks."):
        return leaf
    rest = leaf.split(".", 2)[2]
    index = leaf.split(".")[1]
    table = {
        "attn1.in_proj_weight": "attn1.in_proj_weight",
        "attn1.to_out.0.weight": "attn1.out_proj_weight",
        "attn1.to_out.0.bias": "attn1_out_bias",
        "attn2.to_q.weight": "attn2.q_proj_weight",
        "attn2.to_k.weight": "attn2.k_proj_weight",
        "attn2.to_v.weight": "attn2.v_proj_weight",
        "attn2.to_out.0.weight": "attn2.out_proj_weight",
        "attn2.to_out.0.bias": "attn2_out_bias",
        "ff.net.0.proj.weight": "ff.0.proj.weight",
        "ff.net.0.proj.bias": "ff.0.proj.bias",
        "ff.net.2.weight": "ff.1.weight",
        "ff.net.2.bias": "ff.1.bias",
    }
    mapped = table.get(rest, rest)
    return f"blocks.{index}.{mapped}"


def _map_vae(key: str, config: StableDiffusionConfig) -> str | None:
    """Map one autoencoder key, or ``None`` when no rule covers it.

    See :meth:`StableDiffusionArch.map_key`.
    """
    layers = config.vae_layers_per_block
    parts = key.split(".")

    if key.startswith("quant_conv.") or key.startswith("post_quant_conv."):
        return "vae." + key
    if key.startswith("encoder.conv_in."):
        return "vae.conv_in." + parts[-1]
    if key.startswith("encoder.conv_out."):
        return "vae.conv_out." + parts[-1]
    if key.startswith("encoder.conv_norm_out."):
        return "vae.norm_out." + parts[-1]
    if key.startswith("decoder.conv_in."):
        return "vae.decoder_conv_in." + parts[-1]
    if key.startswith("decoder.conv_out."):
        return "vae.decoder_conv_out." + parts[-1]
    if key.startswith("decoder.conv_norm_out."):
        return "vae.decoder_norm_out." + parts[-1]

    if key.startswith("encoder.mid_block."):
        rest = ".".join(parts[2:])
        if rest.startswith("resnets.0."):
            return "vae.mid_block_1." + _resnet_leaf(".".join(parts[4:]))
        if rest.startswith("resnets.1."):
            return "vae.mid_block_2." + _resnet_leaf(".".join(parts[4:]))
        if rest.startswith("attentions.0."):
            return "vae.mid_attn." + _vae_attn_leaf(".".join(parts[4:]))
        return None
    if key.startswith("decoder.mid_block."):
        rest = ".".join(parts[2:])
        if rest.startswith("resnets.0."):
            return "vae.decoder_mid_block_1." + _resnet_leaf(".".join(parts[4:]))
        if rest.startswith("resnets.1."):
            return "vae.decoder_mid_block_2." + _resnet_leaf(".".join(parts[4:]))
        if rest.startswith("attentions.0."):
            return "vae.decoder_mid_attn." + _vae_attn_leaf(".".join(parts[4:]))
        return None

    if key.startswith("encoder.down_blocks."):
        level, kind = int(parts[2]), parts[3]
        if kind == "resnets":
            index = _flat_index(level, int(parts[4]), layers)
            # Lucid interleaves the stride-2 conv into the same list.
            index += level
            return f"vae.down.{index}." + _resnet_leaf(".".join(parts[5:]))
        if kind == "downsamplers":
            index = (level + 1) * layers + level
            # Lucid wraps the stride-2 conv in ``_Downsample2d`` for the
            # asymmetric padding, so the weight sits one level deeper.
            return f"vae.down.{index}.conv." + parts[-1]
        return None

    if key.startswith("decoder.up_blocks."):
        level, kind = int(parts[2]), parts[3]
        # No flip: diffusers already reverses ``block_out_channels`` for
        # the decoder, so its ``up_blocks[0]`` is the widest stage and so
        # is Lucid's.  Flipping here put every shortcut on the wrong
        # block — 0 and 5 instead of 10 and 15.
        per = layers + 1
        stride = per + 2  # resnets, then Upsample and its conv
        if kind == "resnets":
            return f"vae.up.{level * stride + int(parts[4])}." + _resnet_leaf(
                ".".join(parts[5:])
            )
        if kind == "upsamplers":
            return f"vae.up.{level * stride + per + 1}." + parts[-1]
        return None
    return None


def _vae_attn_leaf(leaf: str) -> str:
    """Rename the inside of the autoencoder's attention block."""
    return {
        "group_norm.weight": "norm.weight",
        "group_norm.bias": "norm.bias",
        "in_proj_weight": "attn.in_proj_weight",
        "in_proj_bias": "attn.in_proj_bias",
        "proj_attn.weight": "attn.out_proj_weight",
        "proj_attn.bias": "attn.out_proj_bias",
        "to_out.0.weight": "attn.out_proj_weight",
        "to_out.0.bias": "attn.out_proj_bias",
    }.get(leaf, leaf)


@register_arch("stable_diffusion")
def _stable_diffusion(tag: str) -> StableDiffusionArch:
    """Build the v1 recipe."""
    return StableDiffusionArch("stable_diffusion", tag)
