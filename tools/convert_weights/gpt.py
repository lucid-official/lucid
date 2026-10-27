"""HF GPT-1 (OpenAI GPT) checkpoints → Lucid ``gpt`` / ``gpt_lm`` weights.

Like GPT-2, GPT-1's parameter names map one-for-one onto Lucid
(``tokens_embed`` / ``positions_embed`` / ``h.N.*``), and the projections
are HF ``Conv1D`` layers whose weights must be transposed for Lucid's
``nn.Linear`` (``c_attn`` / ``c_proj`` / ``c_fc``).  GPT-1 differs from
GPT-2 in being **post-LN** with no final ``ln_f`` — handled by the model,
not the converter.
"""

import dataclasses
import enum
from collections.abc import Callable

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_CITATION = (
    'Radford et al., "Improving Language Understanding by Generative ' 'Pre-Training", 2018 (GPT).'
)
_PAPER_URL = (
    "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf"
)
_HF_ID = "openai-community/openai-gpt"
_TRANSPOSE_SUFFIXES = (".c_attn.weight", ".c_proj.weight", ".c_fc.weight")

# arch_key -> (lucid_factory, repo_slug, title, kind)
_VARIANTS: dict[str, tuple[str, str, str, str]] = {
    "gpt": ("gpt", "gpt", "GPT", "base"),
    "gpt_lm": ("gpt_lm", "gpt-lm", "GPT (causal-LM)", "lm"),
}


def _jsonable(value: object) -> object:
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, tuple):
        return list(value)
    return value


class GPTArch(Architecture):
    """Identity-key, Conv1D-transpose converter for GPT-1."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _VARIANTS:
            raise KeyError(f"GPTArch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        factory, slug, title, kind = _VARIANTS[arch]
        self._factory = factory
        self._slug = slug
        self._title = title
        self._kind = kind

        if kind == "lm":
            from transformers import OpenAIGPTLMHeadModel

            self._src = OpenAIGPTLMHeadModel.from_pretrained(_HF_ID).eval()
        else:
            from transformers import OpenAIGPTModel

            self._src = OpenAIGPTModel.from_pretrained(_HF_ID).eval()

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
            "tokenizer_class": "OpenAIGPTTokenizer",
            "vocab_size": cfg.vocab_size,
            "max_length": cfg.max_position_embeddings,
        }
        meta: dict[str, object] = {
            "num_params": int(sum(p.numel() for p in self._src.parameters())),
            "recipe": f"HuggingFace/{_HF_ID}",
            "metrics": {},
        }
        return ConversionSpec(
            model_name=self._factory,
            architecture=self.arch,
            repo_id=f"lucid-dl/{self._slug}",
            tag=self.tag,
            task=task,
            model_type="gpt",
            source=f"transformers/{_HF_ID}",
            license="mit",
            num_classes=num_classes,
            config=config,
            preprocessing=preprocessing,
            citation=_CITATION,
            title=self._title,
            paper_url=_PAPER_URL,
            datasets=["bookcorpus"],
            meta=meta,
        )


def _make(arch: str) -> Callable[[str], Architecture]:
    def _builder(tag: str) -> Architecture:
        return GPTArch(arch, tag)

    return _builder


for _arch in _VARIANTS:
    register_arch(_arch)(_make(_arch))
