"""GoogLeNet (Inception v1) weight converter — torchvision → Lucid.

Single paper-cited architecture (Szegedy et al., CVPR 2015).  Source =
torchvision's ``GoogLeNet_Weights.IMAGENET1K_V1``.

Lucid's GoogLeNet mirrors torchvision's module layout *almost* exactly:
top-level ``conv1`` / ``conv2`` / ``conv3`` stem blocks (each a
Conv → BatchNorm → ReLU ``_BasicConv2d``), ``inceptionNX.branch{1,2,3,4}``
Inception blocks, and ``aux1`` / ``aux2`` auxiliary classifiers.  The
only key rewrite is the head:

==============================  =========================
torchvision                     Lucid
==============================  =========================
``conv{1,2,3}.*``               ``conv{1,2,3}.*``       (identity)
``inceptionNX.branch{1..4}.*``  same                    (identity)
``aux{1,2}.*``                  same                    (identity)
``fc.{weight,bias}``            ``classifier.{weight,bias}``
==============================  =========================

The checkpoint is built with ``aux_logits=True`` so the converted
state-dict carries real (non-zero) ``aux1`` / ``aux2`` weights that
match Lucid's classifier, and ``transform_input=False`` so neither side
applies the (non-learned) input re-normalisation — keeping the numeric
parity test about the weights alone.
"""

import dataclasses

import torchvision.models as tvm

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_GOOGLENET_CITATION = (
    "@inproceedings{szegedy2015going,\n"
    "  title={Going Deeper with Convolutions},\n"
    "  author={Szegedy, Christian and Liu, Wei and Jia, Yangqing and "
    "Sermanet, Pierre and Reed, Scott and Anguelov, Dragomir and Erhan, "
    "Dumitru and Vanhoucke, Vincent and Rabinovich, Andrew},\n"
    "  booktitle={CVPR}, pages={1--9}, year={2015}\n"
    "}"
)

# arch -> (lucid_cls_factory, repo_id, title)
_GOOGLENET_VARIANTS: dict[str, tuple[str, str, str]] = {
    "googlenet": ("googlenet_cls", "lucid-dl/googlenet", "GoogLeNet")
}


class GoogLeNetArch(Architecture):
    """Converter for the torchvision GoogLeNet checkpoint + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _GOOGLENET_VARIANTS:
            raise KeyError(f"GoogLeNetArch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        self._weights_enum = tvm.GoogLeNet_Weights
        self._tv_weights = self._weights_enum[tag]

    def source_state_dict(self) -> dict[str, object]:
        # aux_logits=True ships the real auxiliary-head weights so the
        # key-set matches Lucid; transform_input=False disables the
        # (non-learned) input re-normalisation for a clean parity test.
        model = tvm.googlenet(weights=self._tv_weights, aux_logits=True, transform_input=False)
        model.eval()
        return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    def target_model(self) -> Module:
        import lucid.models as models

        factory = _GOOGLENET_VARIANTS[self.arch][0]
        return getattr(models, factory)()

    def map_key(self, src_key: str) -> str | None:
        # Head: torchvision ``fc`` → Lucid ``classifier``; everything else
        # is an identity map (Lucid mirrors the torchvision layout).
        if src_key.startswith("fc."):
            return "classifier." + src_key[len("fc.") :]
        return src_key

    def spec(self) -> ConversionSpec:
        import lucid.models as models

        factory_name, repo_id, title = _GOOGLENET_VARIANTS[self.arch]
        model = getattr(models, factory_name)()
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(model.config).items()
        }

        tv_meta = dict(self._tv_weights.meta)
        categories = list(tv_meta.get("categories", []))
        from lucid.utils.transforms import ImageClassification

        tf = self._tv_weights.transforms()
        preset = ImageClassification(
            crop_size=int(tf.crop_size[0]),
            resize_size=int(tf.resize_size[0]),
            mean=tuple(float(m) for m in tf.mean),
            std=tuple(float(s) for s in tf.std),
            interpolation=str(tf.interpolation.value),
        )
        preprocessing = preset.to_dict()
        meta = {
            "num_params": int(model.num_parameters()),
            "gflops": float(tv_meta.get("_ops", 0.0)),
            "recipe": str(tv_meta.get("recipe", "")),
            "metrics": dict(tv_meta.get("_metrics", {})),
        }

        return ConversionSpec(
            model_name=factory_name,
            architecture=self.arch,
            repo_id=repo_id,
            tag=self.tag,
            task="image-classification",
            model_type="googlenet",
            source=f"torchvision/{self._weights_enum.__name__}.{self.tag}",
            license="bsd-3-clause",
            num_classes=int(model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_GOOGLENET_CITATION,
            title=title,
            paper_url="Szegedy et al., 2015 — *Going Deeper with "
            "Convolutions* (arXiv:1409.4842)",
            categories=categories,
            datasets=["imagenet-1k"],
            meta=meta,
        )


@register_arch("googlenet")
def _googlenet(tag: str) -> Architecture:
    return GoogLeNetArch("googlenet", tag)
