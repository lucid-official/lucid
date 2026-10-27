"""DETR weight converter — Facebook DETR (torch.hub) → Lucid.

Two paper-cited variants (Carion et al., ECCV 2020): ``detr_resnet50`` /
``detr_resnet101``.  Source = the original Facebook research checkpoints
loaded via ``torch.hub.load('facebookresearch/detr:main', ...)`` (the
cleanest key names of the available DETR ports), each at COCO
``num_classes = 91`` (91 foreground + 1 no-object → ``class_embed`` is
``(92, 256)``).

Lucid's DETR was rebuilt to mirror the reference module layout verbatim,
so the mapping is almost a pure identity.  Three families of keys differ:

========================================  =================================
Facebook DETR                             Lucid
========================================  =================================
``backbone.0.body.<rest>``                ``backbone.<rest>``  (strip prefix)
``bbox_embed.layers.{i}.<wb>``            ``bbox_embed.net.{2*i}.<wb>``
``transformer....out_proj.{weight,bias}`` ``transformer....out_proj_{weight,bias}``
everything else                           identity
========================================  =================================

The backbone uses ``_FrozenBatchNorm2d`` (4 buffers, no
``num_batches_tracked``), matching the reference ``FrozenBatchNorm2d``
key-set exactly, so backbone BN keys are an identity map after the prefix
strip.  The ``out_proj`` rename reflects Lucid's fused
:class:`nn.MultiheadAttention`, which stores the output projection as flat
``out_proj_weight`` / ``out_proj_bias`` attributes (the fused
``in_proj_weight`` / ``in_proj_bias`` already match the reference).
"""

import dataclasses

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_DETR_CITATION = (
    "@inproceedings{carion2020detr,\n"
    "  title={End-to-End Object Detection with Transformers},\n"
    "  author={Carion, Nicolas and Massa, Francisco and Synnaeve, "
    "Gabriel and Usunier, Nicolas and Kirillov, Alexander and "
    "Zagoruyko, Sergey},\n"
    "  booktitle={European Conference on Computer Vision (ECCV)},\n"
    "  year={2020}\n"
    "}"
)

_DETR_VARIANTS: dict[str, tuple[str, str, str, str, float]] = {
    # arch -> (lucid_factory, repo_id, title, hub_entry, COCO box AP)
    "detr_resnet50": (
        "detr_resnet50",
        "lucid-dl/detr-resnet-50",
        "DETR (ResNet-50)",
        "detr_resnet50",
        42.0,
    ),
    "detr_resnet101": (
        "detr_resnet101",
        "lucid-dl/detr-resnet-101",
        "DETR (ResNet-101)",
        "detr_resnet101",
        43.5,
    ),
}


class DETRArch(Architecture):
    """Converter for one Facebook DETR variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        import torch

        if arch not in _DETR_VARIANTS:
            raise KeyError(f"DETRArch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        _, _, _, hub_entry, _ = _DETR_VARIANTS[arch]
        self._hub_entry = hub_entry
        self._model = torch.hub.load("facebookresearch/detr:main", hub_entry, pretrained=True)
        self._model.eval()

        import lucid.models as models

        self._lucid_factory = _DETR_VARIANTS[arch][0]
        # COCO checkpoint = 91 foreground classes.
        self._lucid_model = getattr(models, self._lucid_factory)(num_classes=91)

    def source_state_dict(self) -> dict[str, object]:
        return {k: v.detach().cpu().numpy() for k, v in self._model.state_dict().items()}

    def target_model(self) -> Module:
        return self._lucid_model

    def map_key(self, src_key: str) -> str | None:
        # Backbone: drop the ``0.body.`` joiner prefix.
        if src_key.startswith("backbone.0.body."):
            return "backbone." + src_key[len("backbone.0.body.") :]
        # Box-head MLP: reference ``layers.{i}`` → Lucid ``net.{2*i}``
        # (Lucid interleaves ReLU at the odd indices).
        if src_key.startswith("bbox_embed.layers."):
            rest = src_key[len("bbox_embed.layers.") :]
            idx = int(rest.split(".")[0])
            wb = rest.split(".", 1)[1]
            return f"bbox_embed.net.{2 * idx}.{wb}"
        # Transformer: identity except the MHA output projection, which
        # Lucid stores as flat ``out_proj_weight`` / ``out_proj_bias``.
        if src_key.startswith("transformer."):
            return src_key.replace(".out_proj.weight", ".out_proj_weight").replace(
                ".out_proj.bias", ".out_proj_bias"
            )
        # class_embed / query_embed / input_proj → identity.
        return src_key

    def spec(self) -> ConversionSpec:
        factory_name, repo_id, title, hub_entry, box_ap = _DETR_VARIANTS[self.arch]
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(self._lucid_model.config).items()
        }

        from lucid.utils.transforms import Detection

        # Reference DETR eval pipeline: longest-side 1333 resize, padded on the
        # bottom and right (the image stays top-left) without rounding the
        # canvas to a stride multiple; ImageNet normalisation (boxes ride with
        # the image).  Matches detr/_weights.py.
        preset = Detection(max_size=1333, size_divisible=1, pad_position="top_left", pad_value=0.0)
        preprocessing = preset.to_dict()

        meta = {
            "num_params": int(sum(p.numel() for p in self._model.parameters())),
            "recipe": "facebookresearch/detr (COCO 2017, 500 epochs)",
            "metrics": {"COCO": {"box mAP": box_ap}},
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="object-detection",
            model_type="detr",
            source=f"facebookresearch/detr/{hub_entry}",
            license="apache-2.0",
            num_classes=91,
            config=config,
            preprocessing=preprocessing,
            citation=_DETR_CITATION,
            title=title,
            paper_url="Carion et al., 2020 — *End-to-End Object Detection "
            "with Transformers* (arXiv:2005.12872)",
            categories=[],
            datasets=["coco"],
            meta=meta,
        )


@register_arch("detr_resnet50")
def _r50(tag: str) -> Architecture:
    return DETRArch("detr_resnet50", tag)


@register_arch("detr_resnet101")
def _r101(tag: str) -> Architecture:
    return DETRArch("detr_resnet101", tag)
