"""Every classifier checkpoint's preset is the pipeline its source evaluated it with.

A preset one pixel off still loads, still matches its source on random
tensors, and still scores within half a point — so nothing noticed until
``tools/check_pretrained_accuracy.py`` scored the checkpoints on
ImageNet-V2 next to their sources and ten of them stopped naming the same
class as their source on 3-7% of the images.  Every one rounded
``crop / crop_pct`` where the source floors it: PVT v2 and Swin-L 249 for
248, ResNeSt-101 293 for 292, EfficientFormer and InceptionNeXt-Base 236
for 235.  CSPNet's weights file had recorded the same lesson a month
earlier, in a comment nothing else read.

Compared field by field against the source's own configuration, with no
weights downloaded and no model built.
"""

import math

import pytest

import lucid.models  # noqa: F401 — populates the registry
from lucid.models._registry import _REGISTRY
from lucid.test._fixtures.ref_framework import ref_vision_module, zoo_module
from lucid.weights._registry import _WEIGHTS_BY_MODEL


def _classifier_checkpoints() -> list[object]:
    found = []
    for name, enum in sorted(_WEIGHTS_BY_MODEL.items()):
        if getattr(_REGISTRY.get(name), "task", None) != "image-classification":
            continue
        for tag, member in enum.__members__.items():
            source = str(member.entry.meta.get("source", ""))
            if tag != "DEFAULT" and source.startswith(("timm/", "reference_vision/")):
                found.append(pytest.param(name, tag, source, id=f"{name}-{tag}"))
    return found


def _rounded(values: object) -> tuple[float, ...]:
    return tuple(round(float(v), 4) for v in values)  # type: ignore[attr-defined]


def _source_pipeline(source: str) -> dict[str, object]:
    """The preprocessing the source evaluates this checkpoint with."""
    kind, identifier = source.split("/", 1)
    if kind == "timm":
        timm = zoo_module()
        if timm is None:
            pytest.skip("the model-zoo oracle is not installed")
        pretrained = timm.models.get_pretrained_cfg(identifier)
        assert pretrained is not None, f"the oracle has no configuration for {identifier}"
        cfg = timm.data.resolve_data_config(pretrained_cfg=pretrained.to_dict())
        assert cfg.get("crop_mode", "center") == "center", cfg
        crop = int(cfg["input_size"][-1])
        return {
            "crop": crop,
            # Floored, as the oracle's own transform factory does.
            "resize": math.floor(crop / cfg["crop_pct"]),
            "interpolation": cfg["interpolation"],
            "mean": _rounded(cfg["mean"]),
            "std": _rounded(cfg["std"]),
        }
    vision = ref_vision_module()
    if vision is None:
        pytest.skip("the reference vision package is not installed")
    enum_name, tag = identifier.split(".", 1)
    ref = getattr(getattr(vision.models, enum_name), tag).transforms()
    return {
        "crop": int(ref.crop_size[0]),
        "resize": int(ref.resize_size[0]),
        "interpolation": str(ref.interpolation.value),
        "mean": _rounded(ref.mean),
        "std": _rounded(ref.std),
    }


def _single(size: object) -> int:
    return int(size if isinstance(size, int) else size[0])  # type: ignore[index]


@pytest.mark.parametrize("name,tag,source", _classifier_checkpoints())
def test_a_preset_is_its_sources_pipeline(name: str, tag: str, source: str) -> None:
    preset = getattr(_WEIGHTS_BY_MODEL[name], tag).transforms()
    ours = {
        "crop": _single(preset.crop_size),
        "resize": _single(preset.resize_size),
        "interpolation": preset.interpolation,
        "mean": _rounded(preset.mean),
        "std": _rounded(preset.std),
    }
    assert ours == _source_pipeline(source)


def test_every_classifier_with_a_known_source_is_covered() -> None:
    # An empty parametrisation would pass by checking nothing.
    assert len(_classifier_checkpoints()) >= 90
