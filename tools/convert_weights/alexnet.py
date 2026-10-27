"""AlexNet weight converter — reference framework → Lucid.

Maps torchvision's AlexNet ``state_dict`` keys onto Lucid's module
layout.  The conv trunk is byte-identical (features.{0,3,6,8,10}); the
three head linears (``classifier.{1,4,6}`` in a single Sequential)
split into ``fc6``/``fc7``/``classifier`` on the Lucid side:

==============================  =========================
torchvision                     Lucid
==============================  =========================
``features.N.{weight,bias}``    ``features.N.{weight,bias}``  (identical)
``classifier.1.{weight,bias}``  ``fc6.{weight,bias}``
``classifier.4.{weight,bias}``  ``fc7.{weight,bias}``
``classifier.6.{weight,bias}``  ``classifier.{weight,bias}``
==============================  =========================
"""

import dataclasses

import torchvision.models as tvm

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_ALEXNET_CITATION = (
    "@inproceedings{krizhevsky2012imagenet,\n"
    "  title={ImageNet Classification with Deep Convolutional Neural Networks},\n"
    "  author={Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E.},\n"
    "  booktitle={NIPS}, year={2012}\n"
    "}\n"
    "@article{krizhevsky2014oneweirdtrick,\n"
    "  title={One weird trick for parallelizing convolutional neural networks},\n"
    "  author={Krizhevsky, Alex},\n"
    "  journal={arXiv preprint arXiv:1404.5997}, year={2014}\n"
    "}"
)

_ALEXNET_VARIANTS: dict[str, tuple[str, str, str]] = {
    # arch -> (lucid_cls_factory, repo_id, title)
    "alexnet_cls": ("alexnet_cls", "lucid-dl/alexnet", "AlexNet")
}

_TV_BUILDERS = {"alexnet_cls": (tvm.alexnet, tvm.AlexNet_Weights)}


class AlexNetArch(Architecture):
    """Converter for AlexNet."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _ALEXNET_VARIANTS:
            raise KeyError(f"AlexNetArch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        self._builder, self._weights_enum = _TV_BUILDERS[arch]
        self._tv_weights = self._weights_enum[tag]

    def source_state_dict(self) -> dict[str, object]:
        model = self._builder(weights=self._tv_weights)
        model.eval()
        return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}

    def target_model(self) -> Module:
        import lucid.models as models

        factory = _ALEXNET_VARIANTS[self.arch][0]
        return getattr(models, factory)()

    def map_key(self, src_key: str) -> str | None:
        # Head: classifier.{1,4,6}.* → fc6/fc7/classifier
        if src_key.startswith("classifier.1."):
            return "fc6." + src_key[len("classifier.1.") :]
        if src_key.startswith("classifier.4."):
            return "fc7." + src_key[len("classifier.4.") :]
        if src_key.startswith("classifier.6."):
            return "classifier." + src_key[len("classifier.6.") :]
        # Conv trunk: features.{0,3,6,8,10}.* passes through unchanged.
        return src_key

    def spec(self) -> ConversionSpec:
        import lucid.models as models

        factory_name, repo_id, title = _ALEXNET_VARIANTS[self.arch]
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
            "num_params": int(tv_meta.get("num_params", 0)),
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
            model_type="alexnet",
            source=f"torchvision/{self._weights_enum.__name__}.{self.tag}",
            license="bsd-3-clause",
            num_classes=int(model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_ALEXNET_CITATION,
            title=title,
            paper_url="Krizhevsky et al., 2012 — *ImageNet Classification "
            "with Deep Convolutional Neural Networks* (NIPS); single-stream "
            "channel widths from Krizhevsky 2014 (arXiv:1404.5997).",
            categories=categories,
            meta=meta,
        )


@register_arch("alexnet_cls")
def _build_alexnet_cls(tag: str) -> Architecture:
    return AlexNetArch("alexnet_cls", tag)
