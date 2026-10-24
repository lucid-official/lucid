"""DiT weight converter — diffusers layout → Lucid.

The published DiT checkpoints ship in the diffusers ``DiTPipeline``
layout, and three things about it need undoing.

**The shared embedders are stored twenty-eight times.**  DiT computes
the timestep and label embeddings once and hands the result to every
block.  diffusers hangs an ``AdaLayerNormZero`` off each block and each
one carries its own ``emb`` sub-module, so the same two embedders are
written out per block — 2.78M parameters × 27 redundant copies, which
is why a 675M model arrives as a 750M file.  Block zero's copy is the
one that maps; the rest are dropped after
:func:`_check_shared_embedders` confirms they are identical, because
"identical" is the assumption that makes dropping them safe and a
future re-export could quietly break it.

**Attention is three matrices upstream and one here.**  ``to_q`` /
``to_k`` / ``to_v`` concatenate into Lucid's fused ``in_proj_weight``,
the same fusion :mod:`tools.convert_weights.stable_diffusion` performs.
Both sides carry q/k/v biases, so those fuse too.

**The final layer is split across two names.**  ``proj_out_1`` is the
modulation that produces the last shift and scale; ``proj_out_2`` is
the decode back to patch space.  Lucid keeps them in one module, so
they land as ``final.ada_ln.1`` and ``final.proj``.

The autoencoder that ships beside the transformer is not DiT's — the
paper uses "the off-the-shelf pre-trained VAE from Stable Diffusion",
here the ``sd-vae-ft-ema`` finetune.  Its architecture is exactly what
:class:`~lucid.models.generative.stable_diffusion.StableDiffusionConfig`
already describes, so it converts through that family's mapper rather
than a second copy of it here.
"""

import dataclasses
from typing import Any, cast

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from lucid.models.generative.dit import DiTConfig, DiTModel
from lucid.nn import Module
from tools.convert_weights._base import (
    Architecture,
    ConversionSpec,
    UnknownSourceKey,
    register_arch,
)

_CITATION = (
    "@inproceedings{peebles2023scalable,\n"
    "  title={Scalable Diffusion Models with Transformers},\n"
    "  author={Peebles, William and Xie, Saining},\n"
    "  booktitle={Proceedings of the IEEE/CVF International Conference "
    "on Computer Vision (ICCV)},\n"
    "  pages={4195--4205},\n"
    "  year={2023}\n"
    "}"
)

# Both released checkpoints are the XL backbone at patch 2 — the paper
# only trained the scaling sweep's other eleven configurations, it did
# not release them.  They differ solely in the latent they work on: a
# 256-pixel image is a 32-side latent after the VAE's factor of eight,
# a 512-pixel one is 64.
_RELEASES: dict[str, tuple[str, dict[str, object]]] = {
    "IMAGENET1K_256": ("facebook/DiT-XL-2-256", {"sample_size": 32}),
    "IMAGENET1K_512": ("facebook/DiT-XL-2-512", {"sample_size": 64}),
}


def _check_shared_embedders(state: dict[str, Any], depth: int) -> None:
    """Fail if the per-block embedder copies are not all identical.

    Parameters
    ----------
    state : dict
        The upstream state dict.
    depth : int
        Number of transformer blocks.

    Raises
    ------
    RuntimeError
        If any block's timestep or label embedder differs from block
        zero's — the conversion keeps only block zero and would
        silently discard real weights otherwise.
    """
    prefix = "transformer_blocks.{}.norm1.emb."
    base = {
        k[len(prefix.format(0)) :]: v
        for k, v in state.items()
        if k.startswith(prefix.format(0))
    }
    for index in range(1, depth):
        for leaf, want in base.items():
            got = state.get(prefix.format(index) + leaf)
            if got is None or not np.array_equal(want, got):
                raise RuntimeError(
                    f"convert(dit): block {index}'s {leaf!r} differs from "
                    f"block 0's.  The conversion keeps only block 0 because "
                    f"DiT shares one embedder across blocks; this export "
                    f"does not, so dropping the copies would lose weights."
                )


def _fuse_attention(state: dict[str, Any], depth: int) -> dict[str, Any]:
    """Concatenate ``to_q`` / ``to_k`` / ``to_v`` into one matrix each.

    Parameters
    ----------
    state : dict
        Upstream state dict.
    depth : int
        Number of transformer blocks.

    Returns
    -------
    dict
        The same mapping with each block's three projections replaced by
        a single ``attn1.in_proj.{weight,bias}``.
    """
    out = dict(state)
    for index in range(depth):
        stem = f"transformer_blocks.{index}.attn1."
        for leaf in ("weight", "bias"):
            parts = [out.pop(f"{stem}to_{p}.{leaf}") for p in ("q", "k", "v")]
            out[f"{stem}in_proj.{leaf}"] = np.concatenate(parts, axis=0)
    return out


class DiTArch(Architecture):
    """Conversion recipe for one released DiT checkpoint."""

    def __init__(self, model_name: str, tag: str = "IMAGENET1K_256") -> None:
        """Bind a factory name to its upstream repo.

        Parameters
        ----------
        model_name : str
            Lucid factory the weights load into.
        tag : str, default="IMAGENET1K_256"
            A key of ``_RELEASES``.

        Raises
        ------
        ValueError
            If ``tag`` names no released checkpoint.
        """
        if tag not in _RELEASES:
            raise ValueError(f"no release tagged {tag!r}; known: {sorted(_RELEASES)}")
        self.model_name = model_name
        self.tag = tag
        self.repo, self.overrides = _RELEASES[tag]

    def _config(self) -> DiTConfig:
        return DiTConfig(**cast(dict[str, Any], self.overrides))

    def source_state_dict(self) -> dict[str, object]:
        """Download the transformer and return it with attention fused.

        Returns
        -------
        dict
            Upstream names to arrays, with each block's q/k/v already
            concatenated and the redundant embedder copies still
            present (:meth:`map_key` drops them).
        """
        path = hf_hub_download(self.repo, "transformer/diffusion_pytorch_model.bin")
        raw = torch.load(path, map_location="cpu", weights_only=True)
        state = {k: v.detach().cpu().numpy() for k, v in raw.items()}
        depth = self._config().depth
        _check_shared_embedders(state, depth)
        return cast(dict[str, object], _fuse_attention(state, depth))

    def target_model(self) -> Module:
        """Build the empty Lucid model the weights load into."""
        return DiTModel(self._config())

    def map_key(self, src_key: str) -> str | None:
        """Map one upstream key to its Lucid name.

        Parameters
        ----------
        src_key : str
            Upstream name, post-fusion.

        Returns
        -------
        str or None
            The Lucid name, or ``None`` for the redundant per-block
            embedder copies.
        """
        if src_key.startswith("pos_embed.proj."):
            return "patch_embed." + src_key.rsplit(".", 1)[-1]
        if src_key.startswith("proj_out_1."):
            return "final.ada_ln.1." + src_key.rsplit(".", 1)[-1]
        if src_key.startswith("proj_out_2."):
            return "final.proj." + src_key.rsplit(".", 1)[-1]
        if src_key.startswith("transformer_blocks."):
            return _map_block(src_key)
        raise UnknownSourceKey(src_key)

    def spec(self) -> ConversionSpec:
        """Return the static metadata written beside the weights."""
        config = self._config()
        pixels = 256 if config.sample_size == 32 else 512
        return ConversionSpec(
            model_name=self.model_name,
            architecture="dit_xlarge_2",
            repo_id="lucid-dl/dit-xlarge-2",
            tag=self.tag,
            task="image-generation",
            model_type="dit",
            source=f"https://huggingface.co/{self.repo}",
            # The released weights are non-commercial; the architecture
            # is not, but a converted checkpoint carries its source's
            # terms and the card has to say so.
            license="cc-by-nc-4.0",
            num_classes=config.num_classes,
            config=dataclasses.asdict(config),
            preprocessing={
                "resize": pixels,
                "rescale": 1 / 127.5,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
            },
            citation=_CITATION,
            title=f"DiT-XL/2 ({pixels}x{pixels})",
            paper_url="https://arxiv.org/abs/2212.09748",
            categories=["generative", "diffusion", "class-conditional"],
            datasets=["imagenet-1k"],
            meta={
                "num_params": 674_834_720,
                "metrics": {"imagenet-1k": {"fid": 2.27 if pixels == 256 else 3.04}},
                "latent_size": config.sample_size,
                "vae": "stabilityai/sd-vae-ft-ema",
                "sampler": "250 DDPM steps (the paper's FID protocol)",
            },
        )


def _map_block(key: str) -> str | None:
    """Map one transformer-block key. See :meth:`DiTArch.map_key`."""
    index, leaf = key.split(".", 2)[1], key.split(".", 2)[2]
    tail = leaf.rsplit(".", 1)[-1]

    # The shared embedders: block zero's copy becomes the model's one,
    # every other block's is a duplicate already verified identical.
    if leaf.startswith("norm1.emb."):
        if index != "0":
            # A duplicate of block zero's, checked identical on load.
            return None
        if "timestep_embedder.linear_1." in leaf:
            return f"time_mlp.0.{tail}"
        if "timestep_embedder.linear_2." in leaf:
            return f"time_mlp.2.{tail}"
        if "class_embedder.embedding_table." in leaf:
            return "label_embed.weight"
        raise UnknownSourceKey(key)

    if leaf.startswith("norm1.linear."):
        return f"blocks.{index}.ada_ln.1.{tail}"
    if leaf.startswith("attn1.in_proj."):
        return f"blocks.{index}.attn.in_proj_{tail}"
    if leaf.startswith("attn1.to_out.0."):
        return f"blocks.{index}.attn.out_proj_{tail}"
    if leaf.startswith("ff.net.0.proj."):
        return f"blocks.{index}.mlp.0.{tail}"
    if leaf.startswith("ff.net.2."):
        return f"blocks.{index}.mlp.2.{tail}"
    raise UnknownSourceKey(key)


@register_arch("dit_xlarge_2")
def _dit(tag: str) -> DiTArch:
    """Build the recipe the CLI dispatches to, for either resolution."""
    return DiTArch("dit_xlarge_2", tag)
