"""CLIP weight converter — Hugging Face ``openai/clip-*`` → Lucid.

The upstream checkpoints are the original OpenAI releases, re-hosted in
the ``transformers`` layout.  That layout differs from Lucid's in three
ways, and only one of them is a rename:

**Q/K/V arrive split.**  ``transformers`` stores ``q_proj`` / ``k_proj``
/ ``v_proj`` as three ``Linear`` layers; Lucid's
:class:`~lucid.nn.MultiheadAttention` keeps them fused in a single
``in_proj_weight`` of width ``3 * width``.  The three are concatenated
along axis 0, in that order — which is the order the fused layout
expects and the order the original OpenAI checkpoint shipped them in
before ``transformers`` split them.

**The projections are transposed.**  ``visual_projection.weight`` and
``text_projection.weight`` are ``Linear`` weights, so ``(out, in)``.
Lucid stores both as bare matrices applied on the right
(``pooled @ proj``), so ``(in, out)``.  Loading either without the
transpose would still produce a tensor of the right *rank*, and for
``text_projection`` in the base models it is even square — a silent,
shape-clean corruption, which is why the check below is per-key.

**``logit_scale`` is a scalar upstream** and a one-element vector here.

===============================================  ==================================
transformers                                     Lucid
===============================================  ==================================
``vision_model.embeddings.patch_embedding.w``    ``visual.conv1.weight``
``vision_model.embeddings.class_embedding``      ``visual.class_embedding``
``vision_model.embeddings.position_embedding.w`` ``visual.positional_embedding``
``vision_model.pre_layrnorm.{w,b}``              ``visual.ln_pre.{w,b}``
``vision_model.post_layernorm.{w,b}``            ``visual.ln_post.{w,b}``
``visual_projection.weight``                     ``visual.proj`` (transposed)
``text_model.embeddings.token_embedding.w``      ``textual.token_embedding.weight``
``text_model.embeddings.position_embedding.w``   ``textual.positional_embedding``
``text_model.final_layer_norm.{w,b}``            ``textual.ln_final.{w,b}``
``text_projection.weight``                       ``textual.text_projection`` (T)
``*.encoder.layers.N.layer_norm1.{w,b}``         ``*.transformer.resblocks.N.ln_1``
``*.encoder.layers.N.layer_norm2.{w,b}``         ``*.transformer.resblocks.N.ln_2``
``*.encoder.layers.N.self_attn.{q,k,v}_proj``    ``…N.attn.in_proj_{weight,bias}``
``*.encoder.layers.N.self_attn.out_proj.{w,b}``  ``…N.attn.out_proj_{weight,bias}``
``*.encoder.layers.N.mlp.fc1.{w,b}``             ``…N.mlp.0.{weight,bias}``
``*.encoder.layers.N.mlp.fc2.{w,b}``             ``…N.mlp.2.{weight,bias}``
===============================================  ==================================

``position_ids`` is dropped: it is a buffer holding ``arange``, not a
learned tensor, and Lucid indexes its positional table directly.

The upstream configs confirm two choices this family made from the
paper rather than from code — ``hidden_act`` is ``quick_gelu`` in both
towers, and ``logit_scale_init_value`` is 2.6592, which is
:math:`\\log(1/0.07)`.

Only the ViT checkpoints are convertible.  Lucid ships no ResNet image
tower for CLIP (see ``multimodal/clip/_pretrained.py``), so the five
``RN*`` releases have nothing to load into.
"""

import dataclasses
from typing import Any, cast

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from lucid.models.multimodal.clip import CLIPModel, CLIPConfig
from lucid.nn import Module
from tools.convert_weights._base import (
    Architecture,
    ConversionSpec,
    UnknownSourceKey,
    register_arch,
)

_CITATION = (
    "@inproceedings{radford2021learning,\n"
    "  title={Learning Transferable Visual Models From Natural Language "
    "Supervision},\n"
    "  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and "
    "Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, "
    "Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and "
    "Krueger, Gretchen and Sutskever, Ilya},\n"
    "  booktitle={ICML},\n"
    "  pages={8748--8763},\n"
    "  year={2021}\n"
    "}"
)

# factory name -> (upstream repo, Lucid config overrides)
_RELEASES: dict[str, tuple[str, dict[str, object]]] = {
    "clip_vit_base_32": ("openai/clip-vit-base-patch32", {}),
    "clip_vit_base_16": ("openai/clip-vit-base-patch16", {"patch_size": 16}),
    "clip_vit_large_14": (
        "openai/clip-vit-large-patch14",
        {
            "embed_dim": 768,
            "patch_size": 14,
            "vision_layers": 24,
            "vision_width": 1024,
            "vision_heads": 16,
            "text_width": 768,
            "text_heads": 12,
        },
    ),
    "clip_vit_large_14_336": (
        "openai/clip-vit-large-patch14-336",
        {
            "embed_dim": 768,
            "image_size": 336,
            "patch_size": 14,
            "vision_layers": 24,
            "vision_width": 1024,
            "vision_heads": 16,
            "text_width": 768,
            "text_heads": 12,
        },
    ),
}

_TOWER = {"vision_model": "visual", "text_model": "textual"}


class CLIPArch(Architecture):
    """Conversion recipe for one ``openai/clip-vit-*`` release."""

    def __init__(self, model_name: str, tag: str = "OPENAI_WIT400M") -> None:
        """Bind a factory name to its upstream repo.

        Parameters
        ----------
        model_name : str
            One of the keys of ``_RELEASES``.
        tag : str, default="OPENAI_WIT400M"
            Weight tag; the dataset is the paper's 400M image-text pairs
            and there is exactly one release per architecture.
        """
        if model_name not in _RELEASES:
            raise ValueError(
                f"no CLIP release for {model_name!r}; known: "
                f"{sorted(_RELEASES)}"
            )
        self.model_name = model_name
        self.tag = tag
        self.repo, self.overrides = _RELEASES[model_name]

    def _config(self) -> CLIPConfig:
        return CLIPConfig(**cast(dict[str, Any], self.overrides))

    def source_state_dict(self) -> dict[str, object]:
        """Download the upstream checkpoint and fuse its Q/K/V.

        Returns
        -------
        dict
            ``{name: numpy.ndarray}`` in the upstream naming, except
            that each block's three projections have been replaced by
            one fused ``in_proj_weight`` / ``in_proj_bias``.

        Notes
        -----
        The fusion happens here rather than in ``transform_value``
        because it is three-to-one: that hook rewrites a value in place
        and has no way to consume its siblings.
        """
        path = hf_hub_download(self.repo, "pytorch_model.bin")
        raw = torch.load(path, map_location="cpu", weights_only=True)
        state = {k: v.numpy() for k, v in raw.items()}

        out: dict[str, object] = {}
        fused: set[str] = set()
        for key, value in state.items():
            if ".self_attn.q_proj." in key:
                stem, suffix = key.split(".self_attn.q_proj.")
                triple = [
                    state[f"{stem}.self_attn.{part}_proj.{suffix}"]
                    for part in ("q", "k", "v")
                ]
                out[f"{stem}.self_attn.in_proj_{suffix}"] = np.concatenate(
                    triple, axis=0
                )
                fused.update(
                    f"{stem}.self_attn.{part}_proj.{suffix}"
                    for part in ("q", "k", "v")
                )
            elif ".self_attn.k_proj." in key or ".self_attn.v_proj." in key:
                continue
            else:
                out[key] = value
        return out

    def target_model(self) -> Module:
        """Build the empty Lucid model the weights load into."""
        return CLIPModel(self._config())

    def map_key(self, src_key: str) -> str | None:
        """Rename one upstream key, or drop it.

        Parameters
        ----------
        src_key : str
            Upstream name, after Q/K/V fusion.

        Returns
        -------
        str or None
            The Lucid name, or ``None`` to drop.

        Raises
        ------
        UnknownSourceKey
            For a key no rule here covers.
        """
        if src_key.endswith("position_ids"):
            # An ``arange`` buffer, not a weight.
            return None
        if src_key == "logit_scale":
            return "logit_scale"
        if src_key == "visual_projection.weight":
            return "visual.proj"
        if src_key == "text_projection.weight":
            return "textual.text_projection"

        for upstream, lucid_name in _TOWER.items():
            if not src_key.startswith(f"{upstream}."):
                continue
            rest = src_key[len(upstream) + 1 :]
            if rest.startswith("embeddings."):
                leaf = rest[len("embeddings.") :]
                if leaf == "class_embedding":
                    return f"{lucid_name}.class_embedding"
                if leaf == "patch_embedding.weight":
                    return f"{lucid_name}.conv1.weight"
                if leaf == "position_embedding.weight":
                    return f"{lucid_name}.positional_embedding"
                if leaf == "token_embedding.weight":
                    return f"{lucid_name}.token_embedding.weight"
                raise UnknownSourceKey(src_key)
            if rest.startswith("pre_layrnorm."):  # upstream's own spelling
                return f"{lucid_name}.ln_pre.{rest.split('.')[-1]}"
            if rest.startswith("post_layernorm."):
                return f"{lucid_name}.ln_post.{rest.split('.')[-1]}"
            if rest.startswith("final_layer_norm."):
                return f"{lucid_name}.ln_final.{rest.split('.')[-1]}"
            if rest.startswith("encoder.layers."):
                parts = rest.split(".")
                index = parts[2]
                leaf = ".".join(parts[3:])
                block = f"{lucid_name}.transformer.resblocks.{index}"
                table = {
                    "layer_norm1.weight": "ln_1.weight",
                    "layer_norm1.bias": "ln_1.bias",
                    "layer_norm2.weight": "ln_2.weight",
                    "layer_norm2.bias": "ln_2.bias",
                    "self_attn.in_proj_weight": "attn.in_proj_weight",
                    "self_attn.in_proj_bias": "attn.in_proj_bias",
                    "self_attn.out_proj.weight": "attn.out_proj_weight",
                    "self_attn.out_proj.bias": "attn.out_proj_bias",
                    "mlp.fc1.weight": "mlp.0.weight",
                    "mlp.fc1.bias": "mlp.0.bias",
                    "mlp.fc2.weight": "mlp.2.weight",
                    "mlp.fc2.bias": "mlp.2.bias",
                }
                mapped = table.get(leaf)
                if mapped is None:
                    raise UnknownSourceKey(src_key)
                return f"{block}.{mapped}"
        raise UnknownSourceKey(src_key)

    def transform_value(self, src_key: str, arr: object) -> object:
        """Transpose the two projections and give ``logit_scale`` a shape.

        Parameters
        ----------
        src_key : str
            Upstream name.
        arr : object
            The raw array.

        Returns
        -------
        object
            The array as Lucid stores it.
        """
        array = cast(np.ndarray, arr)
        if src_key in ("visual_projection.weight", "text_projection.weight"):
            # Linear stores (out, in); Lucid applies these on the right.
            return array.T.copy()
        if src_key == "logit_scale":
            return array.reshape(1)
        return array

    def spec(self) -> ConversionSpec:
        """Return the static metadata written beside the weights."""
        config = self._config()
        return ConversionSpec(
            model_name=self.model_name,
            architecture="CLIPModel",
            repo_id=f"lucid-dl/{self.model_name.replace('_', '-')}",
            tag=self.tag,
            task="base",
            model_type="clip",
            source=f"https://huggingface.co/{self.repo}",
            license="mit",
            num_classes=0,
            config=dataclasses.asdict(config),
            preprocessing={
                "resize": config.image_size,
                "center_crop": config.image_size,
                "rescale": 1 / 255,
                "mean": [0.48145466, 0.4578275, 0.40821073],
                "std": [0.26862954, 0.26130258, 0.27577711],
                "interpolation": "bicubic",
            },
            citation=_CITATION,
            title="Learning Transferable Visual Models From Natural "
            "Language Supervision",
            paper_url="https://arxiv.org/abs/2103.00020",
            categories=["multimodal", "contrastive", "zero-shot"],
            datasets=["WIT-400M"],
        )


@register_arch("clip_vit_base_32")
def _clip_vit_base_32(tag: str) -> CLIPArch:
    """Build the ViT-B/32 recipe."""
    return CLIPArch("clip_vit_base_32", tag)


@register_arch("clip_vit_base_16")
def _clip_vit_base_16(tag: str) -> CLIPArch:
    """Build the ViT-B/16 recipe."""
    return CLIPArch("clip_vit_base_16", tag)


@register_arch("clip_vit_large_14")
def _clip_vit_large_14(tag: str) -> CLIPArch:
    """Build the ViT-L/14 recipe."""
    return CLIPArch("clip_vit_large_14", tag)


@register_arch("clip_vit_large_14_336")
def _clip_vit_large_14_336(tag: str) -> CLIPArch:
    """Build the ViT-L/14@336px recipe."""
    return CLIPArch("clip_vit_large_14_336", tag)
