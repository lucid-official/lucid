"""MobileNet v3 weight converter — torchvision → Lucid.

torchvision wraps every conv + norm + activation triple inside a
``Conv2dNormActivation`` sub-module and every squeeze-excite inside a
``SqueezeExcitation`` sub-module, so its keys are deeply nested
(``features.N.block.M.0.weight`` for the conv, ``...M.1.*`` for the
norm, ``...M.fcK.*`` for SE).  Lucid's :class:`_InvertedResidual` instead
flattens each block into a single ``nn.Sequential`` whose integer indices
encode position (conv / bn / activation / SE / project), and the
activations carry no parameters, so the index spacing differs.

The mapping is therefore *role-based*: each torchvision block child is
classified as ``expand`` / ``dw`` / ``se`` / ``proj`` from the live module
graph, then remapped onto the Lucid block's flattened index for that
role.  The stem and head ``Conv2dNormActivation`` wrappers collapse the
same way, and the two classifier members shift index (Lucid inserts the
adaptive-pool / flatten / dropout as parameter-free members between them).

One value reshape is needed: torchvision's first classifier projection is
a ``Linear`` (``(out, in)``) operating on the pooled-and-flattened
features, whereas Lucid keeps it as a ``1×1`` ``Conv2d`` (``(out, in, 1,
1)``) applied before the flatten.  :meth:`transform_value` adds the two
trailing singleton dims for that one weight.

Both paper-cited variants ship the standard ImageNet eval preset (224
crop / 256 resize / bilinear / ImageNet stats), matching torchvision's
``MobileNet_V3_*_Weights.IMAGENET1K_V1`` transforms exactly.
"""

import dataclasses

import torchvision.models as tvm

from lucid.nn import Module
from tools.convert_weights._base import (
    Architecture,
    ConversionSpec,
    UnknownSourceKey,
    register_arch,
)

_MOBILENET_V3_CITATION = (
    "@inproceedings{howard2019searching,\n"
    "  title={Searching for MobileNetV3},\n"
    "  author={Howard, Andrew and Sandler, Mark and Chu, Grace and "
    "Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun "
    "and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and others},\n"
    "  booktitle={ICCV}, year={2019}\n"
    "}"
)

_MOBILENET_V3_PAPER_URL = (
    "Howard et al., 2019 — *Searching for MobileNetV3* (arXiv:1905.02244)"
)

# arch -> (lucid_cls_factory, repo_id, title)
_MOBILENET_V3_VARIANTS: dict[str, tuple[str, str, str]] = {
    "mobilenet_v3_large": (
        "mobilenet_v3_large_cls",
        "lucid-dl/mobilenet-v3-large",
        "MobileNet V3 Large",
    ),
    "mobilenet_v3_small": (
        "mobilenet_v3_small_cls",
        "lucid-dl/mobilenet-v3-small",
        "MobileNet V3 Small",
    ),
}

_TV_BUILDERS = {
    "mobilenet_v3_large": (
        tvm.mobilenet_v3_large,
        tvm.MobileNet_V3_Large_Weights,
    ),
    "mobilenet_v3_small": (
        tvm.mobilenet_v3_small,
        tvm.MobileNet_V3_Small_Weights,
    ),
}


class MobileNetV3Arch(Architecture):
    """Converter for one torchvision MobileNet v3 variant + tag."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _MOBILENET_V3_VARIANTS:
            raise KeyError(f"MobileNetV3Arch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        self._builder, self._weights_enum = _TV_BUILDERS[arch]
        self._tv_weights = self._weights_enum[tag]
        # Built lazily so the (heavy) source model is created at most once.
        self._key_map: dict[str, str] | None = None
        self._source_model: object | None = None

    # -- source ---------------------------------------------------------------

    def _build_source(self) -> object:
        if self._source_model is None:
            model = self._builder(weights=self._tv_weights)
            model.eval()
            self._source_model = model
        return self._source_model

    def source_state_dict(self) -> dict[str, object]:
        model = self._build_source()
        return {
            k: v.detach().cpu().numpy()
            for k, v in model.state_dict().items()  # type: ignore[attr-defined]
        }

    def target_model(self) -> Module:
        import lucid.models as models

        factory = _MOBILENET_V3_VARIANTS[self.arch][0]
        return getattr(models, factory)()

    # -- key mapping ----------------------------------------------------------

    def _build_key_map(self) -> dict[str, str]:
        """Derive the src→dst key map from the live torchvision graph.

        torchvision feature layout::

            features.0          stem Conv2dNormActivation (conv .0 / bn .1)
            features.1..N-2     InvertedResidual blocks
            features.N-1        head Conv2dNormActivation (conv .0 / bn .1)

        Each ``InvertedResidual.block`` is an ``nn.Sequential`` whose
        children are, in order: optional expand ``Conv2dNormActivation``,
        depthwise ``Conv2dNormActivation``, optional ``SqueezeExcitation``,
        project ``Conv2dNormActivation``.

        Lucid's flattened block indices::

            expand:  conv 0  / bn 1   (activation at 2, no params)
            dw:      conv 3  / bn 4   (activation at 5, no params)   [expand]
                     conv 0  / bn 1   (activation at 2, no params)   [no expand]
            se:      6                [expand]   /   3   [no expand]
            proj:    conv (7 or 6) / bn (+1)     [expand, with/without SE]
                     conv (4 or 3) / bn (+1)     [no expand, with/without SE]
        """
        model = self._build_source()
        feats = model.features  # type: ignore[attr-defined]
        n = len(feats)
        head_idx = n - 1
        nblocks = n - 2
        lucid_head_conv = 3 + nblocks
        lucid_head_bn = lucid_head_conv + 1

        out: dict[str, str] = {}
        src = model.state_dict()  # type: ignore[attr-defined]
        for src_key in src:
            parts = src_key.split(".")
            if parts[0] != "features":
                continue
            fi = int(parts[1])
            if fi == 0:
                sub = int(parts[2])
                rest = ".".join(parts[3:])
                out[src_key] = f"features.{0 if sub == 0 else 1}.{rest}"
            elif fi == head_idx:
                sub = int(parts[2])
                rest = ".".join(parts[3:])
                idx = lucid_head_conv if sub == 0 else lucid_head_bn
                out[src_key] = f"features.{idx}.{rest}"
            else:
                lfi = fi + 2
                children = list(feats[fi].block)
                nse = [type(c).__name__ for c in children].count("SqueezeExcitation")
                has_expand = (len(children) - nse) == 3
                roles = (
                    (["expand"] if has_expand else [])
                    + ["dw"]
                    + (["se"] if nse else [])
                    + ["proj"]
                )
                role = roles[int(parts[3])]
                if role == "se":
                    rest = ".".join(parts[4:])
                    li = 6 if has_expand else 3
                else:
                    sub2 = int(parts[4])
                    rest = ".".join(parts[5:])
                    if role == "expand":
                        li = 0 if sub2 == 0 else 1
                    elif role == "dw":
                        if has_expand:
                            li = 3 if sub2 == 0 else 4
                        else:
                            li = 0 if sub2 == 0 else 1
                    else:  # proj
                        if has_expand:
                            base = 7 if nse else 6
                        else:
                            base = 4 if nse else 3
                        li = base if sub2 == 0 else base + 1
                out[src_key] = f"features.{lfi}.block.{li}.{rest}"

        # Classifier: torchvision Linear/Dropout/Linear (0, 2-drop, 3) vs
        # Lucid AvgPool/Conv2d/h-swish/Flatten/Dropout/Linear (1, 5).
        for src_key in src:
            if src_key.startswith("classifier.0."):
                out[src_key] = "classifier.1." + src_key[len("classifier.0.") :]
            elif src_key.startswith("classifier.3."):
                out[src_key] = "classifier.5." + src_key[len("classifier.3.") :]
        return out

    def map_key(self, src_key: str) -> str | None:
        if self._key_map is None:
            self._key_map = self._build_key_map()
        mapped = self._key_map.get(src_key)
        if mapped is None:
            raise UnknownSourceKey(src_key)
        return mapped

    def transform_value(self, src_key: str, arr: object) -> object:
        # torchvision's first classifier projection is a Linear (out, in);
        # Lucid keeps it as a 1x1 Conv2d (out, in, 1, 1) applied before the
        # flatten, so add the two trailing singleton spatial dims.
        if src_key == "classifier.0.weight":
            return arr.reshape(arr.shape[0], arr.shape[1], 1, 1)  # type: ignore[attr-defined]
        return arr

    # -- spec -----------------------------------------------------------------

    def spec(self) -> ConversionSpec:
        import lucid.models as models

        factory_name, repo_id, title = _MOBILENET_V3_VARIANTS[self.arch]
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
            model_type="mobilenet_v3",
            source=f"torchvision/{self._weights_enum.__name__}.{self.tag}",
            license="bsd-3-clause",
            num_classes=int(model.config.num_classes),
            config=config,
            preprocessing=preprocessing,
            citation=_MOBILENET_V3_CITATION,
            title=title,
            paper_url=_MOBILENET_V3_PAPER_URL,
            categories=categories,
            datasets=["imagenet-1k"],
            meta=meta,
        )


@register_arch("mobilenet_v3_large")
def _build_mobilenet_v3_large(tag: str) -> Architecture:
    return MobileNetV3Arch("mobilenet_v3_large", tag)


@register_arch("mobilenet_v3_small")
def _build_mobilenet_v3_small(tag: str) -> Architecture:
    return MobileNetV3Arch("mobilenet_v3_small", tag)
