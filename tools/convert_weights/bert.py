"""HF BERT checkpoints → Lucid ``bert_*`` weights.

BERT is a *pure identity map*: Lucid mirrors the upstream ``BertModel``
parameter naming one-for-one (verified: 199/199 keys, same names + shapes
for ``bert-base-uncased``), so the base encoders need no key renames and no
value transforms.

Two checkpoint families are covered:

* **Base encoders** (``bert_tiny`` … ``bert_large``) — the bare ``BertModel``
  (embeddings + encoder + pooler).  Sourced from ``AutoModel``.  The four
  miniatures are the Turc et al. 2019 "Well-Read Students" pre-distilled
  sizes; ``base`` / ``large`` are Devlin et al. 2018.
* **Masked-LM heads** (``bert_base_mlm`` / ``bert_large_mlm``) — base encoder
  + ``cls.predictions`` head.  Sourced from upstream ``BertForPreTraining`` (so
  the pooler weights, which Lucid keeps but the upstream ``BertForMaskedLM``
  drops, come from the real checkpoint).  Two upstream redundancies are dropped:
  the NSP head
  (``cls.seq_relationship.*``) and the duplicate ``cls.predictions.decoder.bias``
  (a tied alias of ``cls.predictions.bias``, which Lucid stores once).
"""

import dataclasses
import enum
from collections.abc import Callable

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_CITATION = (
    'Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers '
    'for Language Understanding", NAACL 2019. Miniatures: Turc et al., '
    '"Well-Read Students Learn Better", 2019.'
)
_PAPER_URL = "https://arxiv.org/abs/1810.04805"

# arch_key -> (lucid_factory, repo_slug, title, hf_model_id, kind)
_VARIANTS: dict[str, tuple[str, str, str, str, str]] = {
    "bert_tiny": (
        "bert_tiny",
        "bert-tiny",
        "BERT-Tiny",
        "google/bert_uncased_L-2_H-128_A-2",
        "base",
    ),
    "bert_mini": (
        "bert_mini",
        "bert-mini",
        "BERT-Mini",
        "google/bert_uncased_L-4_H-256_A-4",
        "base",
    ),
    "bert_small": (
        "bert_small",
        "bert-small",
        "BERT-Small",
        "google/bert_uncased_L-4_H-512_A-8",
        "base",
    ),
    "bert_medium": (
        "bert_medium",
        "bert-medium",
        "BERT-Medium",
        "google/bert_uncased_L-8_H-512_A-8",
        "base",
    ),
    "bert_base": ("bert_base", "bert-base", "BERT-Base", "google-bert/bert-base-uncased", "base"),
    "bert_large": (
        "bert_large",
        "bert-large",
        "BERT-Large",
        "google-bert/bert-large-uncased",
        "base",
    ),
    "bert_base_mlm": (
        "bert_base_mlm",
        "bert-base-mlm",
        "BERT-Base (Masked-LM)",
        "google-bert/bert-base-uncased",
        "mlm",
    ),
    "bert_large_mlm": (
        "bert_large_mlm",
        "bert-large-mlm",
        "BERT-Large (Masked-LM)",
        "google-bert/bert-large-uncased",
        "mlm",
    ),
    "bert_base_qa": (
        "bert_base_qa",
        "bert-base-squad",
        "BERT-Base (SQuAD v1.1)",
        "csarron/bert-base-uncased-squad-v1",
        "qa",
    ),
    "bert_large_qa": (
        "bert_large_qa",
        "bert-large-squad",
        "BERT-Large WWM (SQuAD v1.1)",
        "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",
        "qa",
    ),
    "bert_base_token_cls": (
        "bert_base_token_cls",
        "bert-base-ner",
        "BERT-Base (CoNLL-2003 NER)",
        "dslim/bert-base-NER",
        "token_cls",
    ),
}

# Per-task provenance for the fine-tuned heads (license / datasets / metrics /
# tokenizer-casing).  These are *full* fine-tuned checkpoints, unlike the
# encoder + MLM tags.
_TASK_INFO: dict[str, dict[str, object]] = {
    "bert_base_qa": {
        "task": "question-answering",
        "license": "mit",
        "datasets": ["squad"],
        "metrics": {"squad": {"exact_match": 80.9, "f1": 88.1}},
        "do_lower_case": True,
    },
    "bert_large_qa": {
        "task": "question-answering",
        "license": "apache-2.0",
        "datasets": ["squad"],
        "metrics": {"squad": {"exact_match": 86.9, "f1": 93.2}},
        "do_lower_case": True,
    },
    "bert_base_token_cls": {
        "task": "token-classification",
        "license": "mit",
        "datasets": ["conll2003"],
        "metrics": {"conll2003": {"f1": 91.3}},
        "do_lower_case": False,
    },
}


def _jsonable(value: object) -> object:
    """Make a config field JSON-serialisable (enum → its value, tuple → list)."""
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, tuple):
        return list(value)
    return value


class BERTArch(Architecture):
    """Identity-map converter for the BERT family (base + masked-LM)."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _VARIANTS:
            raise KeyError(f"BERTArch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        factory, slug, title, hf_id, kind = _VARIANTS[arch]
        self._factory = factory
        self._slug = slug
        self._title = title
        self._hf_id = hf_id
        self._kind = kind

        import lucid.models as models

        if kind == "mlm":
            from transformers import BertForPreTraining

            self._src = BertForPreTraining.from_pretrained(hf_id).eval()
            self._model: Module = getattr(models, factory)()
        elif kind in ("qa", "token_cls"):
            from transformers import AutoModel, BertForQuestionAnswering, BertForTokenClassification

            head_cls = BertForQuestionAnswering if kind == "qa" else BertForTokenClassification
            # Fine-tuned head + encoder come from the task model; the pooler
            # (dropped by add_pooling_layer=False, but present in the file) is
            # recovered via AutoModel so Lucid's BERTModel pooler slot fills.
            self._src = head_cls.from_pretrained(hf_id).eval()
            self._trunk = AutoModel.from_pretrained(hf_id).eval()
            if kind == "token_cls":
                vocab = int(self._src.config.vocab_size)
                labels = int(self._src.config.num_labels)
                self._model = getattr(models, factory)(vocab_size=vocab, num_labels=labels)
            else:
                self._model = getattr(models, factory)()
        else:
            from transformers import AutoModel

            self._src = AutoModel.from_pretrained(hf_id).eval()
            self._model = getattr(models, factory)()

    def source_state_dict(self) -> dict[str, object]:
        if self._kind in ("qa", "token_cls"):
            # Head model: bert.* (no pooler) + task head (qa_outputs/classifier).
            combined = dict(self._src.state_dict())
            # Recover the checkpoint's own pooler from the AutoModel trunk.
            for k, v in self._trunk.state_dict().items():
                if k.startswith("pooler."):
                    combined[f"bert.{k}"] = v
            return {k: v.detach().cpu().numpy() for k, v in combined.items()}
        return {k: v.detach().cpu().numpy() for k, v in self._src.state_dict().items()}

    def target_model(self) -> Module:
        return self._model

    def map_key(self, src_key: str) -> str | None:
        if self._kind == "mlm":
            # NSP head is unused by Lucid's masked-LM model.
            if src_key.startswith("cls.seq_relationship."):
                return None
            # HF stores the decoder bias twice (it is tied to predictions.bias);
            # Lucid keeps only the standalone `cls.predictions.bias`.
            if src_key == "cls.predictions.decoder.bias":
                return None
        return src_key

    def spec(self) -> ConversionSpec:
        cfg = self._model.config
        config = {k: _jsonable(v) for k, v in dataclasses.asdict(cfg).items()}

        info = _TASK_INFO.get(self.arch)
        if info is not None:
            task = str(info["task"])
            license_id = str(info["license"])
            datasets = list(info["datasets"])  # type: ignore[arg-type]
            metrics = info["metrics"]
            do_lower_case = bool(info["do_lower_case"])
            # QA span head = 2; token-cls = num_labels.
            num_classes = 2 if self._kind == "qa" else int(cfg.num_labels)
        elif self._kind == "mlm":
            task = "masked-lm"
            license_id = "apache-2.0"
            datasets = ["wikipedia", "bookcorpus"]
            metrics = {}
            do_lower_case = True
            num_classes = cfg.vocab_size
        else:
            task = "base"
            license_id = "apache-2.0"
            datasets = ["wikipedia", "bookcorpus"]
            metrics = {}
            do_lower_case = True
            num_classes = cfg.hidden_size

        preprocessing = {
            "tokenizer_class": "BERTTokenizer",
            "vocab_size": cfg.vocab_size,
            "do_lower_case": do_lower_case,
            "max_length": cfg.max_position_embeddings,
        }
        meta: dict[str, object] = {
            "num_params": int(sum(p.numel() for p in self._model.parameters())),
            "recipe": f"HuggingFace/{self._hf_id}",
            "metrics": metrics,
        }
        return ConversionSpec(
            model_name=self._factory,
            architecture=self.arch,
            repo_id=f"lucid-dl/{self._slug}",
            tag=self.tag,
            task=task,
            model_type="bert",
            source=f"transformers/{self._hf_id}",
            license=license_id,
            num_classes=num_classes,
            config=config,
            preprocessing=preprocessing,
            citation=_CITATION,
            title=self._title,
            paper_url=_PAPER_URL,
            datasets=datasets,
            meta=meta,
        )


def _make(arch: str) -> Callable[[str], Architecture]:
    def _builder(tag: str) -> Architecture:
        return BERTArch(arch, tag)

    return _builder


for _arch in _VARIANTS:
    register_arch(_arch)(_make(_arch))
