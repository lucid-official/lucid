"""HF GPT-2 checkpoints → Lucid ``gpt2_*`` weights.

GPT-2's parameter naming maps one-for-one onto Lucid (``wte`` / ``wpe`` /
``h.N.*`` / ``ln_f``), so :meth:`map_key` is the identity.  The one real
transform is the **Conv1D → Linear weight transpose**: HF implements the
attention/MLP projections with ``Conv1D``, which stores its weight as
``(in, out)``; Lucid uses ``nn.Linear`` (``(out, in)``).  Every
``c_attn`` / ``c_proj`` / ``c_fc`` weight is therefore transposed
(``c_proj`` is square, so the shape gate would *not* catch a missing
transpose — only forward parity does).

Covers the four sizes (Radford et al. 2019) as bare transformers
(``GPT2Model``) and as causal-LM heads (``GPT2LMHeadModel``, ``lm_head``
tied to ``wte``).
"""

import dataclasses
import enum
from collections.abc import Callable

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_CITATION = (
    'Radford et al., "Language Models are Unsupervised Multitask Learners", ' "2019 (GPT-2)."
)
_PAPER_URL = "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf"

# Conv1D weights whose value must be transposed for nn.Linear.
_TRANSPOSE_SUFFIXES = (".c_attn.weight", ".c_proj.weight", ".c_fc.weight")

# arch_key -> (lucid_factory, repo_slug, title, hf_model_id, kind)
_VARIANTS: dict[str, tuple[str, str, str, str, str]] = {
    "gpt2_small": ("gpt2_small", "gpt2-small", "GPT-2 Small", "gpt2", "base"),
    "gpt2_medium": ("gpt2_medium", "gpt2-medium", "GPT-2 Medium", "gpt2-medium", "base"),
    "gpt2_large": ("gpt2_large", "gpt2-large", "GPT-2 Large", "gpt2-large", "base"),
    "gpt2_xlarge": ("gpt2_xlarge", "gpt2-xlarge", "GPT-2 XL", "gpt2-xl", "base"),
    "gpt2_small_lm": ("gpt2_small_lm", "gpt2-small-lm", "GPT-2 Small (causal-LM)", "gpt2", "lm"),
    "gpt2_medium_lm": (
        "gpt2_medium_lm",
        "gpt2-medium-lm",
        "GPT-2 Medium (causal-LM)",
        "gpt2-medium",
        "lm",
    ),
    "gpt2_large_lm": (
        "gpt2_large_lm",
        "gpt2-large-lm",
        "GPT-2 Large (causal-LM)",
        "gpt2-large",
        "lm",
    ),
    "gpt2_xlarge_lm": ("gpt2_xlarge_lm", "gpt2-xlarge-lm", "GPT-2 XL (causal-LM)", "gpt2-xl", "lm"),
}


def _jsonable(value: object) -> object:
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, tuple):
        return list(value)
    return value


class GPT2Arch(Architecture):
    """Identity-key, Conv1D-transpose converter for the GPT-2 family."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _VARIANTS:
            raise KeyError(f"GPT2Arch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        factory, slug, title, hf_id, kind = _VARIANTS[arch]
        self._factory = factory
        self._slug = slug
        self._title = title
        self._hf_id = hf_id
        self._kind = kind

        if kind == "lm":
            from transformers import GPT2LMHeadModel

            self._src = GPT2LMHeadModel.from_pretrained(hf_id).eval()
        else:
            from transformers import GPT2Model

            self._src = GPT2Model.from_pretrained(hf_id).eval()

        import lucid.models as models

        self._model: Module = getattr(models, factory)()

    def source_state_dict(self) -> dict[str, object]:
        return {k: v.detach().cpu().numpy() for k, v in self._src.state_dict().items()}

    def target_model(self) -> Module:
        return self._model

    def map_key(self, src_key: str) -> str | None:
        return src_key

    def transform_value(self, src_key: str, arr: object) -> object:
        if src_key.endswith(_TRANSPOSE_SUFFIXES):
            return arr.T  # type: ignore[attr-defined]  # numpy ndarray in the converter (tools/, outside the H4 boundary)
        return arr

    def spec(self) -> ConversionSpec:
        cfg = self._model.config
        config = {k: _jsonable(v) for k, v in dataclasses.asdict(cfg).items()}
        task = "causal-lm" if self._kind == "lm" else "base"
        num_classes = cfg.vocab_size if self._kind == "lm" else cfg.hidden_size

        preprocessing = {
            "tokenizer_class": "GPT2Tokenizer",
            "vocab_size": cfg.vocab_size,
            "max_length": cfg.max_position_embeddings,
        }
        meta: dict[str, object] = {
            "num_params": int(sum(p.numel() for p in self._src.parameters())),
            "recipe": f"HuggingFace/{self._hf_id}",
            "metrics": {},
        }
        return ConversionSpec(
            model_name=self._factory,
            architecture=self.arch,
            repo_id=f"lucid-dl/{self._slug}",
            tag=self.tag,
            task=task,
            model_type="gpt2",
            source=f"transformers/{self._hf_id}",
            license="mit",
            num_classes=num_classes,
            config=config,
            preprocessing=preprocessing,
            citation=_CITATION,
            title=self._title,
            paper_url=_PAPER_URL,
            datasets=["webtext"],
            meta=meta,
        )


def _make(arch: str) -> Callable[[str], Architecture]:
    def _builder(tag: str) -> Architecture:
        return GPT2Arch(arch, tag)

    return _builder


for _arch in _VARIANTS:
    register_arch(_arch)(_make(_arch))
