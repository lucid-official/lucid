"""HF diffusers ``UNet2DModel`` (google/ddpm-*) → Lucid ``ddpm_*`` weights.

Lucid's ``_DDPMUNet`` is structurally identical to the diffusers ``UNet2DModel``
(both implement Ho 2020), but lays its modules out differently:

* diffusers groups blocks by stage — ``down_blocks[i].resnets[j]``,
  ``down_blocks[i].attentions[j]``, ``down_blocks[i].downsamplers[0]`` — while
  Lucid uses flat parallel ``ModuleList``\\s indexed by a global block counter
  (``down_res[k]`` / ``down_attn[k]`` / ``down_sample[i]``).  The flat index is
  ``k = i * num_res_blocks + j`` (encoder) / ``m * (num_res_blocks + 1) + j``
  (decoder).
* diffusers self-attention uses **split** ``to_q`` / ``to_k`` / ``to_v`` /
  ``to_out.0`` (``Linear``); Lucid uses a **fused** ``qkv`` + ``proj``
  (``Conv2d`` 1×1).  We concat ``[q; k; v]`` along the output axis and reshape
  ``(out, in) → (out, in, 1, 1)``.
* ResnetBlock2D fields rename ``time_emb_proj → time_proj`` and
  ``conv_shortcut → skip``; ``conv_norm_out → norm_out``; ``time_embedding →
  time_mlp``.

``source_state_dict`` emits keys already in Lucid naming (with the ``unet.``
prefix of :class:`DDPMModel`), so ``map_key`` / ``transform_value`` are identity
and the engine's 1:1 key-set + strict-load gate verifies the whole mapping.
"""

import dataclasses
import re
from collections.abc import Callable

import numpy as np

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_CITATION = (
    'Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising Diffusion '
    'Probabilistic Models." Advances in Neural Information Processing Systems, '
    "2020, pp. 6840–6851."
)
_PAPER_URL = "https://arxiv.org/abs/2006.11239"

# arch_key -> (lucid_factory, repo_slug, title, hf_model_id, dataset)
_VARIANTS: dict[str, tuple[str, str, str, str, str]] = {
    "ddpm_cifar": (
        "ddpm_cifar",
        "ddpm-cifar10",
        "DDPM (CIFAR-10 32×32)",
        "google/ddpm-cifar10-32",
        "cifar10",
    ),
    "ddpm_lsun": (
        "ddpm_lsun",
        "ddpm-church",
        "DDPM (LSUN-Church 256×256)",
        "google/ddpm-church-256",
        "lsun-churches",
    ),
}


def _np(t: object) -> np.ndarray:
    import torch

    assert isinstance(t, torch.Tensor)
    return t.detach().cpu().numpy()


def _jsonable(value: object) -> object:
    import enum

    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, tuple):
        return list(value)
    return value


class DDPMArch(Architecture):
    """diffusers ``UNet2DModel`` → Lucid ``DDPMModel`` (``unet.*``) converter."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _VARIANTS:
            raise KeyError(f"DDPMArch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        factory, slug, title, hf_id, dataset = _VARIANTS[arch]
        self._factory = factory
        self._slug = slug
        self._title = title
        self._hf_id = hf_id
        self._dataset = dataset

        from diffusers import UNet2DModel

        self._src = UNet2DModel.from_pretrained(hf_id).eval()

        import lucid.models as models

        self._model: Module = getattr(models, factory)()
        self._nrb = int(self._model.config.num_res_blocks)

    # ── key remap helpers ──────────────────────────────────────────────────
    @staticmethod
    def _resnet_field(field: str) -> str:
        return {"time_emb_proj": "time_proj", "conv_shortcut": "skip"}.get(field, field)

    def source_state_dict(self) -> dict[str, object]:
        sd = {k: _np(v) for k, v in self._src.state_dict().items()}
        nrb = self._nrb
        out: dict[str, np.ndarray] = {}

        def emit(lucid_key: str, arr: np.ndarray) -> None:
            out[f"unet.{lucid_key}"] = arr

        # Fixed singletons.
        emit("conv_in.weight", sd["conv_in.weight"])
        emit("conv_in.bias", sd["conv_in.bias"])
        emit("conv_out.weight", sd["conv_out.weight"])
        emit("conv_out.bias", sd["conv_out.bias"])
        emit("norm_out.weight", sd["conv_norm_out.weight"])
        emit("norm_out.bias", sd["conv_norm_out.bias"])
        for li in ("linear_1", "linear_2"):
            emit(f"time_mlp.{li}.weight", sd[f"time_embedding.{li}.weight"])
            emit(f"time_mlp.{li}.bias", sd[f"time_embedding.{li}.bias"])

        # Attention fuse: collect block prefixes, emit qkv/proj/norm.
        attn_prefixes = sorted(
            {m.group(0) for k in sd for m in [re.match(r".*attentions\.\d+(?=\.)", k)] if m}
        )

        def emit_attention(diff_prefix: str, lucid_prefix: str) -> None:
            wq, wk, wv = (
                sd[f"{diff_prefix}.to_q.weight"],
                sd[f"{diff_prefix}.to_k.weight"],
                sd[f"{diff_prefix}.to_v.weight"],
            )
            c = wq.shape[0]
            qkv_w = np.concatenate([wq, wk, wv], axis=0).reshape(3 * c, c, 1, 1)
            qkv_b = np.concatenate(
                [
                    sd[f"{diff_prefix}.to_q.bias"],
                    sd[f"{diff_prefix}.to_k.bias"],
                    sd[f"{diff_prefix}.to_v.bias"],
                ],
                axis=0,
            )
            emit(f"{lucid_prefix}.qkv.weight", qkv_w)
            emit(f"{lucid_prefix}.qkv.bias", qkv_b)
            emit(
                f"{lucid_prefix}.proj.weight",
                sd[f"{diff_prefix}.to_out.0.weight"].reshape(c, c, 1, 1),
            )
            emit(f"{lucid_prefix}.proj.bias", sd[f"{diff_prefix}.to_out.0.bias"])
            emit(f"{lucid_prefix}.norm.weight", sd[f"{diff_prefix}.group_norm.weight"])
            emit(f"{lucid_prefix}.norm.bias", sd[f"{diff_prefix}.group_norm.bias"])

        def lucid_attn_target(diff_prefix: str) -> str:
            mid = re.match(r"mid_block\.attentions\.0", diff_prefix)
            if mid:
                return "mid_attn"
            md = re.match(r"down_blocks\.(\d+)\.attentions\.(\d+)", diff_prefix)
            if md:
                i, j = int(md.group(1)), int(md.group(2))
                return f"down_attn.{i * nrb + j}"
            mu = re.match(r"up_blocks\.(\d+)\.attentions\.(\d+)", diff_prefix)
            assert mu, diff_prefix
            i, j = int(mu.group(1)), int(mu.group(2))
            return f"up_attn.{i * (nrb + 1) + j}"

        for p in attn_prefixes:
            emit_attention(p, lucid_attn_target(p))

        # ResnetBlock2D + downsample/upsample conv (1:1 field renames).
        for k, arr in sd.items():
            if "attentions" in k or k.startswith(
                ("conv_in", "conv_out", "conv_norm_out", "time_embedding")
            ):
                continue  # handled above

            md = re.match(r"down_blocks\.(\d+)\.resnets\.(\d+)\.(.+)", k)
            if md:
                i, j, field = int(md.group(1)), int(md.group(2)), md.group(3)
                emit(f"down_res.{i * nrb + j}.{self._resnet_field_path(field)}", arr)
                continue
            md = re.match(r"down_blocks\.(\d+)\.downsamplers\.0\.conv\.(\w+)", k)
            if md:
                i, wb = int(md.group(1)), md.group(2)
                emit(f"down_sample.{i}.op.{wb}", arr)
                continue
            mm = re.match(r"mid_block\.resnets\.(\d+)\.(.+)", k)
            if mm:
                ri, field = int(mm.group(1)), mm.group(2)
                tgt = "mid_block1" if ri == 0 else "mid_block2"
                emit(f"{tgt}.{self._resnet_field_path(field)}", arr)
                continue
            mu = re.match(r"up_blocks\.(\d+)\.resnets\.(\d+)\.(.+)", k)
            if mu:
                i, j, field = int(mu.group(1)), int(mu.group(2)), mu.group(3)
                emit(f"up_res.{i * (nrb + 1) + j}.{self._resnet_field_path(field)}", arr)
                continue
            mp = re.match(r"up_blocks\.(\d+)\.upsamplers\.0\.conv\.(\w+)", k)
            if mp:
                i, wb = int(mp.group(1)), mp.group(2)
                emit(f"up_sample.{i}.op.{wb}", arr)
                continue
            raise RuntimeError(f"DDPMArch: unmapped diffusers key {k!r}")

        return dict(out)

    @staticmethod
    def _resnet_field_path(field: str) -> str:
        # field is e.g. "norm1.weight" / "time_emb_proj.bias" / "conv_shortcut.weight"
        head, _, rest = field.partition(".")
        return f"{DDPMArch._resnet_field(head)}.{rest}" if rest else field

    def target_model(self) -> Module:
        return self._model

    def map_key(self, src_key: str) -> str | None:
        return src_key  # source_state_dict already emits Lucid names

    def spec(self) -> ConversionSpec:
        cfg = self._model.config
        config = {k: _jsonable(v) for k, v in dataclasses.asdict(cfg).items()}
        size = cfg.sample_size if isinstance(cfg.sample_size, int) else cfg.sample_size[0]
        preprocessing = {
            "sample_size": size,
            "in_channels": cfg.in_channels,
            "normalize": "[-1, 1]",
            "num_train_timesteps": cfg.num_train_timesteps,
        }
        meta: dict[str, object] = {
            "num_params": int(sum(p.numel() for p in self._model.parameters())),
            "recipe": f"HuggingFace/{self._hf_id}",
            "metrics": {},
        }
        return ConversionSpec(
            model_name=self._factory,
            architecture=self.arch,
            repo_id=f"lucid-dl/{self._slug}",
            tag=self.tag,
            task="unconditional-image-generation",
            model_type="ddpm",
            source=f"diffusers/{self._hf_id}",
            license="apache-2.0",
            num_classes=cfg.in_channels,
            config=config,
            preprocessing=preprocessing,
            citation=_CITATION,
            title=self._title,
            paper_url=_PAPER_URL,
            datasets=[self._dataset],
            meta=meta,
        )


def _make(arch: str) -> Callable[[str], Architecture]:
    def _builder(tag: str) -> Architecture:
        return DDPMArch(arch, tag)

    return _builder


for _arch in _VARIANTS:
    register_arch(_arch)(_make(_arch))
