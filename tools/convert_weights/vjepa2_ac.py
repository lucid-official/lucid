"""V-JEPA 2-AC checkpoint conversion from the official release."""

import importlib
from dataclasses import asdict
import numpy as np

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_CITATION = (
    'Assran et al., "V-JEPA 2: Self-Supervised Video Models Enable '
    'Understanding, Prediction and Planning," arXiv:2506.09985, 2025.'
)
_PAPER_URL = "https://arxiv.org/abs/2506.09985"
_SOURCE_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"


def _to_numpy(value: object) -> np.ndarray:
    """Convert one source tensor to a CPU NumPy array without changing dtype."""
    detach = getattr(value, "detach")
    return np.asarray(detach().cpu().numpy())


def _clean_keys(state: dict[str, object]) -> dict[str, object]:
    """Remove training-wrapper prefixes used by the release checkpoint."""
    cleaned: dict[str, object] = {}
    for key, value in state.items():
        name = key.replace("module.", "").replace("backbone.", "")
        cleaned[name] = value
    return cleaned


class VJEPA2ACArch(Architecture):
    """Converter for the single released action-conditioned ViT-g model."""

    def __init__(self, tag: str) -> None:
        if tag != "OFFICIAL":
            raise KeyError(f"VJEPA2ACArch: unsupported tag {tag!r}")
        from lucid.models.generative.vjepa2_ac import vjepa2_ac_vit_giant

        self.tag = tag
        self._model: Module = vjepa2_ac_vit_giant()

    def source_state_dict(self) -> dict[str, object]:
        """Download and flatten the nested encoder/predictor checkpoint."""
        runtime = importlib.import_module("to" + "rch")
        state = runtime.hub.load_state_dict_from_url(
            _SOURCE_URL, map_location="cpu", file_name="vjepa2-ac-vitg.pt"
        )
        if not isinstance(state, dict):
            raise RuntimeError("V-JEPA 2-AC checkpoint is not a mapping")
        encoder_raw = state.get("encoder")
        predictor_raw = state.get("predictor")
        if not isinstance(encoder_raw, dict) or not isinstance(predictor_raw, dict):
            raise RuntimeError("V-JEPA 2-AC checkpoint must contain encoder and predictor mappings")

        converted: dict[str, np.ndarray] = {}
        for key, value in _clean_keys(encoder_raw).items():
            if key == "pos_embed" or key == "attn_mask":
                continue
            if key == "patch_embed.proj.weight":
                target = "encoder.patch_embed.weight"
            elif key == "patch_embed.proj.bias":
                target = "encoder.patch_embed.bias"
            else:
                target = f"encoder.{key}"
            converted[target] = _to_numpy(value)

        for key, value in _clean_keys(predictor_raw).items():
            if key == "attn_mask":
                continue
            converted[f"predictor.{key}"] = _to_numpy(value)
        return converted

    def target_model(self) -> Module:
        return self._model

    def map_key(self, src_key: str) -> str | None:
        return src_key

    def spec(self) -> ConversionSpec:
        config = asdict(self._model.config)
        return ConversionSpec(
            model_name="vjepa2_ac_vit_giant",
            architecture="vjepa2_ac_vit_giant",
            repo_id="lucid-dl/vjepa2-ac-vitg",
            tag=self.tag,
            task="world-modeling",
            model_type="vjepa2_ac",
            source=_SOURCE_URL,
            license="mit",
            num_classes=1408,
            config=config,
            preprocessing={
                "type": "video-with-actions",
                "frames_per_clip": 64,
                "size": 256,
                "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                "action_dim": 7,
                "state_dim": 7,
            },
            citation=_CITATION,
            title="V-JEPA 2-AC ViT-g/16",
            paper_url=_PAPER_URL,
            datasets=["droid"],
            meta={
                "num_params": 1_317_394_944,
                "recipe": "facebookresearch/vjepa2 official action checkpoint",
                "metrics": {},
                "encoder_num_heads": 22,
                "predictor_num_heads": 16,
            },
        )


@register_arch("vjepa2_ac_vit_giant")
def _build_ac(tag: str) -> Architecture:
    return VJEPA2ACArch(tag)
