"""Google's T5 checkpoints (HF ``google-t5/*``) → Lucid ``t5_*`` weights.

The released layout nests each sub-layer under ``block.N.layer.M`` with
module names from the Mesh TensorFlow port (``SelfAttention``,
``EncDecAttention``, ``DenseReluDense``).  Lucid names the sub-layers by
role, so :meth:`map_key` renames — the tensors themselves are copied
unchanged (``nn.Linear`` on both sides, no transposes):

====================================================  ==========================================
source                                                Lucid
====================================================  ==========================================
``shared.weight``                                     ``shared.weight``
``{enc,dec}.block.0.layer.0.SelfAttention.``          ``{enc,dec}.relative_attention_bias.weight``
``relative_attention_bias.weight``
``{enc,dec}.block.N.layer.0.SelfAttention.{q,k,v,o}``  ``….blocks.N.self_attention.{q,k,v,o}``
``decoder.block.N.layer.1.EncDecAttention.*``          ``decoder.blocks.N.cross_attention.*``
``….layer.M.DenseReluDense.{wi,wo}``                  ``….blocks.N.feed_forward.{wi,wo}``
``….layer.M.layer_norm.weight``                        ``….blocks.N.<sub-layer>_norm.weight``
``{enc,dec}.final_layer_norm.weight``                 unchanged
``{enc,dec}.embed_tokens.weight``                      dropped — aliases of ``shared``
``lm_head.weight``                                    unchanged (tied to ``shared``)
====================================================  ==========================================

The relative-position table lives on block 0 in the release and on the
stack in Lucid: the release computes it in block 0 and hands the result
to every later block, so it was always one table per stack.

Text-to-text heads nest the trunk under ``t5.``.  Covers the paper's five
sizes as trunks (``T5Model``) and as text-to-text models
(``T5ForConditionalGeneration`` → ``T5ForSeq2SeqLM``).
"""

import dataclasses
import enum
from collections.abc import Callable

from lucid.nn import Module
from lucid.peft._convert import t5_reference_name
from tools.convert_weights._base import (
    Architecture,
    ConversionSpec,
    UnknownSourceKey,
    register_arch,
)

_CITATION = (
    'Raffel et al., "Exploring the Limits of Transfer Learning with a Unified '
    'Text-to-Text Transformer", JMLR 21(140), 2020 (T5).'
)
_PAPER_URL = "https://arxiv.org/abs/1910.10683"

# arch_key -> (lucid_factory, repo_slug, title, hf_model_id, kind)
_VARIANTS: dict[str, tuple[str, str, str, str, str]] = {
    "t5_small": ("t5_small", "t5-small", "T5-Small", "google-t5/t5-small", "base"),
    "t5_base": ("t5_base", "t5-base", "T5-Base", "google-t5/t5-base", "base"),
    "t5_large": ("t5_large", "t5-large", "T5-Large", "google-t5/t5-large", "base"),
    "t5_3b": ("t5_3b", "t5-3b", "T5-3B", "google-t5/t5-3b", "base"),
    "t5_11b": ("t5_11b", "t5-11b", "T5-11B", "google-t5/t5-11b", "base"),
    "t5_small_seq2seq": (
        "t5_small_seq2seq",
        "t5-small-seq2seq",
        "T5-Small (text-to-text)",
        "google-t5/t5-small",
        "seq2seq",
    ),
    "t5_base_seq2seq": (
        "t5_base_seq2seq",
        "t5-base-seq2seq",
        "T5-Base (text-to-text)",
        "google-t5/t5-base",
        "seq2seq",
    ),
    "t5_large_seq2seq": (
        "t5_large_seq2seq",
        "t5-large-seq2seq",
        "T5-Large (text-to-text)",
        "google-t5/t5-large",
        "seq2seq",
    ),
    "t5_3b_seq2seq": (
        "t5_3b_seq2seq",
        "t5-3b-seq2seq",
        "T5-3B (text-to-text)",
        "google-t5/t5-3b",
        "seq2seq",
    ),
    "t5_11b_seq2seq": (
        "t5_11b_seq2seq",
        "t5-11b-seq2seq",
        "T5-11B (text-to-text)",
        "google-t5/t5-11b",
        "seq2seq",
    ),
}


def t5_key(src_key: str) -> str | None:
    """Lucid trunk key for a released T5 key; ``None`` drops a tied alias.

    The map is ``lucid.peft._convert.t5_reference_name`` — the one adapter
    files are renamed by, so the two cannot disagree.  Raises
    :class:`UnknownSourceKey` for a key the layout above does not describe.
    """
    if src_key.endswith(".embed_tokens.weight"):
        return None
    mapped = t5_reference_name(src_key)
    if mapped is None:
        raise UnknownSourceKey(src_key)
    return mapped


def _jsonable(value: object) -> object:
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, tuple):
        return list(value)
    return value


class T5Arch(Architecture):
    """Renaming converter for the T5 family (no tensor transforms)."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _VARIANTS:
            raise KeyError(f"T5Arch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        factory, slug, title, hf_id, kind = _VARIANTS[arch]
        self._factory = factory
        self._slug = slug
        self._title = title
        self._hf_id = hf_id
        self._kind = kind

        if kind == "seq2seq":
            from transformers import T5ForConditionalGeneration

            self._src = T5ForConditionalGeneration.from_pretrained(hf_id).eval()
        else:
            from transformers import T5Model

            self._src = T5Model.from_pretrained(hf_id).eval()

        import lucid.models as models

        self._model: Module = getattr(models, factory)()

    def source_state_dict(self) -> dict[str, object]:
        return {k: v.detach().cpu().numpy() for k, v in self._src.state_dict().items()}

    def target_model(self) -> Module:
        return self._model

    def map_key(self, src_key: str) -> str | None:
        if src_key == "lm_head.weight":
            return src_key if self._kind == "seq2seq" else None
        key = t5_key(src_key)
        if key is None or self._kind == "base":
            return key
        return f"t5.{key}"

    def transform_value(self, src_key: str, arr: object) -> object:
        return arr

    def spec(self) -> ConversionSpec:
        cfg = self._model.config
        config = {k: _jsonable(v) for k, v in dataclasses.asdict(cfg).items()}
        task = "text2text-generation" if self._kind == "seq2seq" else "base"
        num_classes = cfg.vocab_size if self._kind == "seq2seq" else cfg.hidden_size
        preprocessing = {
            "tokenizer_class": "UnigramTokenizer",
            "vocab_size": cfg.vocab_size,
            "task_prefixes": ["translate English to German: ", "summarize: "],
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
            model_type="t5",
            source=f"transformers/{self._hf_id}",
            license="apache-2.0",
            num_classes=num_classes,
            config=config,
            preprocessing=preprocessing,
            citation=_CITATION,
            title=self._title,
            paper_url=_PAPER_URL,
            datasets=["c4"],
            meta=meta,
        )


def _make(arch: str) -> Callable[[str], Architecture]:
    def _builder(tag: str) -> Architecture:
        return T5Arch(arch, tag)

    return _builder


for _arch in _VARIANTS:
    register_arch(_arch)(_make(_arch))
