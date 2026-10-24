"""CvT weight converter — Microsoft / HuggingFace transformers → Lucid.

Three paper-cited variants (Wu et al., ICCV 2021): ``cvt_13`` /
``cvt_21`` / ``cvt_w24``.  Source = ``transformers.CvtForImageClassification``
from the ``microsoft/cvt-<variant>`` hub repos.

Key naming difference summary (full mapping is in :meth:`map_key`):

============================================================  =================================
transformers                                                  Lucid
============================================================  =================================
``cvt.encoder.stages.S.embedding.convolution_embeddings.``    ``stages.S.embed.{proj,norm}.``
``cvt.encoder.stages.S.layers.B.layernorm_before.``           ``stages.S.blocks.B.norm1.``
``cvt.encoder.stages.S.layers.B.layernorm_after.``            ``stages.S.blocks.B.norm2.``
``cvt.encoder.stages.S.layers.B.attention.attention.``        ``stages.S.blocks.B.attn.``
``  convolution_projection_query.convolution_projection.``    ``  proj_q.{dw,bn}.``
``  projection_query.``                                       ``  proj_q.proj.``
``  (same for _key, _value)``                                 ``  (proj_k, proj_v)``
``cvt.encoder.stages.S.layers.B.attention.output.dense.``     ``stages.S.blocks.B.attn.out_proj.``
``cvt.encoder.stages.S.layers.B.intermediate.dense.``         ``stages.S.blocks.B.mlp.fc1.``
``cvt.encoder.stages.S.layers.B.output.dense.``               ``stages.S.blocks.B.mlp.fc2.``
``layernorm.``                                                ``head_norm.``
``classifier.``                                               ``classifier.``
============================================================  =================================

A single ``cls_token`` lives on the final transformer stage in the HF
checkpoint (``cvt.encoder.stages.2.cls_token``); Lucid's last stage owns
a matching learnable CLS token, so the converter carries that key
through verbatim.
"""

import dataclasses
import re

from lucid.nn import Module
from tools.convert_weights._base import (
    Architecture,
    ConversionSpec,
    UnknownSourceKey,
    register_arch,
)

_CVT_CITATION = (
    "@inproceedings{wu2021cvt,\n"
    "  title={CvT: Introducing Convolutions to Vision Transformers},\n"
    "  author={Wu, Haiping and Xiao, Bin and Codella, Noel and Liu, "
    "Mengchen and Dai, Xiyang and Yuan, Lu and Zhang, Lei},\n"
    "  booktitle={ICCV}, year={2021}\n"
    "}"
)

_CVT_VARIANTS: dict[str, tuple[str, str, str, str]] = {
    # arch -> (lucid_cls_factory, repo_id, title, hf_model_id)
    "cvt_13": ("cvt_13_cls", "lucid-dl/cvt-13", "CvT-13", "microsoft/cvt-13"),
    "cvt_21": ("cvt_21_cls", "lucid-dl/cvt-21", "CvT-21", "microsoft/cvt-21"),
    "cvt_w24": (
        "cvt_w24_cls",
        "lucid-dl/cvt-w24",
        "CvT-W24",
        "microsoft/cvt-w24-384-22k",
    ),
}


def _datasets_for(arch: str, tag: str) -> list[str]:
    """Tag-aware dataset list (CvT line)."""
    t = tag.lower()
    ds: list[str] = []
    if "in22k" in t or "22k" in arch:
        ds.append("imagenet-22k")
    if "in1k" in t or arch in ("cvt_13", "cvt_21"):
        # MS HF repos at ``cvt-13`` / ``cvt-21`` are 1k-class finetunes.
        ds.append("imagenet-1k")
    if not ds:
        ds.append("imagenet-1k")
    return ds


class CvTArch(Architecture):
    """Converter for one CvT variant + tag (``transformers``-sourced)."""

    def __init__(self, arch: str, tag: str) -> None:
        from transformers import CvtForImageClassification

        if arch not in _CVT_VARIANTS:
            raise KeyError(f"CvTArch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        self._hf_id = _CVT_VARIANTS[arch][3]
        self._model = CvtForImageClassification.from_pretrained(self._hf_id)
        self._model.eval()
        import lucid.models as models

        self._lucid_factory = _CVT_VARIANTS[arch][0]
        self._lucid_model = getattr(models, self._lucid_factory)()

    def source_state_dict(self) -> dict[str, object]:
        return {
            k: v.detach().cpu().numpy() for k, v in self._model.state_dict().items()
        }

    def target_model(self) -> Module:
        return self._lucid_model

    def map_key(self, src_key: str) -> str | None:
        # Head: ``classifier.{w,b}`` carries through; ``layernorm`` → ``head_norm``.
        if src_key.startswith("classifier."):
            return src_key
        if src_key.startswith("layernorm."):
            return "head_norm." + src_key[len("layernorm.") :]
        # Strip the ``cvt.encoder.`` prefix used by HF.
        if not src_key.startswith("cvt.encoder."):
            raise UnknownSourceKey(src_key)
        s = src_key[len("cvt.encoder.") :]
        # ``stages.S.cls_token`` → ``stages.S.cls_token`` (carried through;
        # Lucid's last stage owns a matching learnable CLS token).
        if s.endswith(".cls_token"):
            return s
        # Embedding (stem + per-stage patch embed).
        s = s.replace(
            ".embedding.convolution_embeddings.projection.",
            ".embed.proj.",
        )
        s = s.replace(
            ".embedding.convolution_embeddings.normalization.",
            ".embed.norm.",
        )
        # ``stages.S.layers.B.layernorm_before/after`` → ``stages.S.blocks.B.norm{1,2}``
        s = re.sub(
            r"^stages\.(\d+)\.layers\.(\d+)\.layernorm_before\.",
            r"stages.\1.blocks.\2.norm1.",
            s,
        )
        s = re.sub(
            r"^stages\.(\d+)\.layers\.(\d+)\.layernorm_after\.",
            r"stages.\1.blocks.\2.norm2.",
            s,
        )
        # Convolutional Q/K/V projections.
        s = s.replace(
            "attention.attention.convolution_projection_query.convolution_projection.convolution.",
            "attn.proj_q.dw.",
        )
        s = s.replace(
            "attention.attention.convolution_projection_query.convolution_projection.normalization.",
            "attn.proj_q.bn.",
        )
        s = s.replace(
            "attention.attention.convolution_projection_key.convolution_projection.convolution.",
            "attn.proj_k.dw.",
        )
        s = s.replace(
            "attention.attention.convolution_projection_key.convolution_projection.normalization.",
            "attn.proj_k.bn.",
        )
        s = s.replace(
            "attention.attention.convolution_projection_value.convolution_projection.convolution.",
            "attn.proj_v.dw.",
        )
        s = s.replace(
            "attention.attention.convolution_projection_value.convolution_projection.normalization.",
            "attn.proj_v.bn.",
        )
        # Final linear projections of Q/K/V (``projection_query/key/value``).
        s = s.replace("attention.attention.projection_query.", "attn.proj_q.proj.")
        s = s.replace("attention.attention.projection_key.", "attn.proj_k.proj.")
        s = s.replace("attention.attention.projection_value.", "attn.proj_v.proj.")
        # Attention output dense.
        s = s.replace("attention.output.dense.", "attn.out_proj.")
        # MLP: ``intermediate.dense`` → ``mlp.fc1``, ``output.dense`` → ``mlp.fc2``.
        s = re.sub(
            r"^stages\.(\d+)\.layers\.(\d+)\.intermediate\.dense\.",
            r"stages.\1.blocks.\2.mlp.fc1.",
            s,
        )
        s = re.sub(
            r"^stages\.(\d+)\.layers\.(\d+)\.output\.dense\.",
            r"stages.\1.blocks.\2.mlp.fc2.",
            s,
        )
        # Catch-all: any ``stages.S.layers.B.<rest>`` left over (e.g. the
        # remapped ``.attn.*`` paths from the per-projection rewrites)
        # gets a final ``layers → blocks`` rename.
        s = re.sub(r"^stages\.(\d+)\.layers\.(\d+)\.", r"stages.\1.blocks.\2.", s)
        return s

    def spec(self) -> ConversionSpec:
        factory_name, repo_id, title, _ = _CVT_VARIANTS[self.arch]
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(self._lucid_model.config).items()
        }

        from lucid.utils.transforms import ImageClassification

        # CvT inputs at 224 (cvt-13/cvt-21) or 384 (cvt-w24-384-22k).  HF's
        # AutoImageProcessor returns mean/std at ImageNet defaults; we
        # use those directly.
        is_384 = "384" in self._hf_id
        crop = 384 if is_384 else 224
        resize = int(round(crop / 0.875))
        preset = ImageClassification(
            crop_size=crop,
            resize_size=resize,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            interpolation="bicubic",
        )
        preprocessing = preset.to_dict()

        n_params = int(sum(p.numel() for p in self._model.parameters()))
        _acc_table = {
            "cvt_13": 81.6,
            "cvt_21": 82.5,
            "cvt_w24": 87.7,
        }
        meta = {
            "num_params": n_params,
            "recipe": f"HuggingFace/{self._hf_id}",
            "metrics": {"ImageNet-1k": {"acc@1": _acc_table.get(self.arch, 0.0)}},
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="cvt",
            source=f"transformers/{self._hf_id}",
            license="apache-2.0",
            num_classes=int(self._lucid_model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_CVT_CITATION,
            title=title,
            paper_url="Wu et al., 2021 — *CvT: Introducing Convolutions "
            "to Vision Transformers* (arXiv:2103.15808)",
            categories=[],
            datasets=_datasets_for(self.arch, self.tag),
            meta=meta,
        )


@register_arch("cvt_13")
def _13(tag: str) -> Architecture:
    return CvTArch("cvt_13", tag)


@register_arch("cvt_21")
def _21(tag: str) -> Architecture:
    return CvTArch("cvt_21", tag)


@register_arch("cvt_w24")
def _w24(tag: str) -> Architecture:
    return CvTArch("cvt_w24", tag)
