"""Mask R-CNN weight converter — torchvision → Lucid.

Lucid's Mask R-CNN was rebuilt to mirror the reference
``maskrcnn_resnet50_fpn`` submodule layout verbatim.  It reuses Faster
R-CNN's ``backbone.body.*`` (ResNet-50 with FrozenBatchNorm2d, eps=0),
``backbone.fpn.inner_blocks / layer_blocks``, ``rpn.head.conv.0.0`` /
``cls_logits`` / ``bbox_pred``, and ``roi_heads.box_head`` (TwoMLPHead) +
``roi_heads.box_predictor`` (FastRCNNPredictor), and adds the mask branch
``roi_heads.mask_head.{i}.0`` (MaskRCNNHeads) + ``roi_heads.mask_predictor``
(``conv5_mask`` ConvTranspose2d + ``mask_fcn_logits`` Conv1×1).  A full
key/shape diff shows 307 keys on each side with an **identical** key-set
(295 shared with Faster R-CNN + 12 mask-branch keys) and zero shape
mismatches, so ``map_key`` is a pure identity.  The FrozenBatchNorm2d
buffers carry no ``num_batches_tracked``, so there is nothing to drop.

Source: ``torchvision.models.detection.maskrcnn_resnet50_fpn`` with the
``COCO_V1`` tag (91 classes; box AP 37.9 / mask AP 34.6; BSD-3 license).
"""

import dataclasses

import torchvision.models.detection as tvd

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_MASK_RCNN_CITATION = (
    "@inproceedings{he2017mask,\n"
    "  title={Mask R-CNN},\n"
    "  author={He, Kaiming and Gkioxari, Georgia and Doll{\\'a}r, Piotr and "
    "Girshick, Ross},\n"
    "  booktitle={Proceedings of the IEEE International Conference on "
    "Computer Vision (ICCV)},\n"
    "  pages={2961--2969},\n"
    "  year={2017}\n"
    "}"
)

# arch -> (lucid_factory, repo_id, title, COCO box AP, COCO mask AP)
_MASK_RCNN_VARIANTS: dict[str, tuple[str, str, str, float, float]] = {
    "mask_rcnn_resnet50_fpn": (
        "mask_rcnn_resnet50_fpn",
        "lucid-dl/mask-rcnn-resnet-50-fpn",
        "Mask R-CNN (ResNet-50-FPN)",
        37.9,
        34.6,
    )
}


class MaskRCNNArch(Architecture):
    """Converter for the torchvision Mask R-CNN ResNet-50-FPN variant."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _MASK_RCNN_VARIANTS:
            raise KeyError(f"MaskRCNNArch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        self._weights_enum = tvd.MaskRCNN_ResNet50_FPN_Weights
        self._tv_weights = self._weights_enum[tag]

    def source_state_dict(self) -> dict[str, object]:
        model = tvd.maskrcnn_resnet50_fpn(weights=self._tv_weights)
        model.eval()
        return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    def target_model(self) -> Module:
        import lucid.models as models

        factory = _MASK_RCNN_VARIANTS[self.arch][0]
        return models.create_model(factory, num_classes=91)

    def map_key(self, src_key: str) -> str | None:
        # Identical key layout — pure identity map.
        return src_key

    def spec(self) -> ConversionSpec:
        import lucid.models as models

        factory_name, repo_id, title, box_ap, mask_ap = _MASK_RCNN_VARIANTS[self.arch]
        model = models.create_model(factory_name, num_classes=91)
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(model.config).items()
        }

        from lucid.utils.transforms import Detection

        # The preset the weights module ships (mask_rcnn/_weights.py):
        # longest side 1333, the image in the top-left corner of a 1344 square
        # (a multiple of 32, as the reference pads its batches), ImageNet
        # normalisation (boxes / masks ride with the image).
        preset = Detection(max_size=1333, size_divisible=32, pad_position="top_left", pad_value=0.0)
        preprocessing = preset.to_dict()

        tv_meta = dict(self._tv_weights.meta)
        meta = {
            "num_params": int(tv_meta.get("num_params", 0)),
            "gflops": float(tv_meta.get("_ops", 0.0)),
            "recipe": str(tv_meta.get("recipe", "")),
            "metrics": {"COCO": {"box mAP": box_ap, "mask mAP": mask_ap}},
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="instance-segmentation",
            model_type="mask_rcnn",
            source=f"torchvision/{self._weights_enum.__name__}.{self.tag}",
            license="bsd-3-clause",
            num_classes=91,
            config=config,
            preprocessing=preprocessing,
            citation=_MASK_RCNN_CITATION,
            title=title,
            paper_url="He et al., 2017 — *Mask R-CNN* (arXiv:1703.06870)",
            categories=[],
            datasets=["coco"],
            meta=meta,
        )


@register_arch("mask_rcnn_resnet50_fpn")
def _mask_rcnn_r50_fpn(tag: str) -> Architecture:
    return MaskRCNNArch("mask_rcnn_resnet50_fpn", tag)
