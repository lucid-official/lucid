"""HF RoFormer checkpoints → Lucid ``roformer`` / ``roformer_mlm`` weights.

RoFormer (Su et al., 2021) is the *original* rotary-position-embedding paper,
so its checkpoints were trained with the **interleaved** RoPE convention.
Lucid's RoFormer was rebuilt to use that interleaved layout (opt-in
``apply_rotary_emb(..., interleaved=True)`` + a family-local rotary table),
giving forward parity ≤6.4e-6 vs the reference framework.

Parameter naming maps one-for-one (split q/k/v, no Conv1D transpose), with
two upstream-only keys handled:

* ``*encoder.embed_positions.weight`` — HF's fixed sinusoidal RoPE table;
  Lucid recomputes RoPE from ``base`` so this is dropped.
* ``cls.predictions.decoder.bias`` (masked-LM only) — a tied alias of
  ``cls.predictions.bias`` which Lucid stores once; dropped.

The pooler (``*pooler.dense.*``) is **not present** in the upstream checkpoint
(it is untrained there too) but Lucid's ``RoFormerModel`` owns one, so the
converter injects the freshly-initialised pooler weights to satisfy the strict
key-set gate.  It affects only ``pooler_output`` (meaningless until fine-tuned),
never ``last_hidden_state``.  The task heads that read the raw hidden states
(``roformer_cls`` and friends) build no pooler, so a target without
``pooler.dense.*`` gets no injection.

Sequence classification (LCD-709): Lucid's ``classifier`` is the reference's
``RoFormerClassificationHead``, so a reference ``classifier.dense.*`` /
``classifier.out_proj.*`` checkpoint maps through ``map_key`` unchanged.  No
fine-tuned classification checkpoint ships, hence no ``roformer_cls`` entry in
``_VARIANTS``.

The only published canonical checkpoint is Chinese
(``junnyu/roformer_chinese_base``, CLUECorpusSmall).
"""

import dataclasses
import enum
from collections.abc import Callable

from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

_CITATION = 'Su et al., "RoFormer: Enhanced Transformer with Rotary Position ' 'Embedding", 2021.'
_PAPER_URL = "https://arxiv.org/abs/2104.09864"
_HF_ID = "junnyu/roformer_chinese_base"

# arch_key -> (lucid_factory, repo_slug, title, kind)
_VARIANTS: dict[str, tuple[str, str, str, str]] = {
    "roformer": ("roformer", "roformer-chinese-base", "RoFormer (Chinese base)", "base"),
    "roformer_mlm": (
        "roformer_mlm",
        "roformer-chinese-base-mlm",
        "RoFormer (Chinese base, Masked-LM)",
        "mlm",
    ),
}


def _jsonable(value: object) -> object:
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, tuple):
        return list(value)
    return value


class RoFormerArch(Architecture):
    """Converter for the RoFormer family (interleaved RoPE, pooler injection)."""

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _VARIANTS:
            raise KeyError(f"RoFormerArch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        factory, slug, title, kind = _VARIANTS[arch]
        self._factory = factory
        self._slug = slug
        self._title = title
        self._kind = kind

        if kind == "mlm":
            from transformers import RoFormerForMaskedLM

            self._src = RoFormerForMaskedLM.from_pretrained(_HF_ID).eval()
        else:
            from transformers import RoFormerModel

            self._src = RoFormerModel.from_pretrained(_HF_ID).eval()

        import lucid.models as models

        self._model: Module = getattr(models, factory)()

    def source_state_dict(self) -> dict[str, object]:
        out: dict[str, object] = {}
        for k, v in self._src.state_dict().items():
            if k.endswith("embed_positions.weight"):
                continue  # HF sinusoidal RoPE table — Lucid recomputes it
            if k == "cls.predictions.decoder.bias":
                continue  # tied dup of cls.predictions.bias
            out[k] = v.detach().cpu().numpy()
        # Inject the untrained pooler (absent upstream) from Lucid's init so the
        # strict key-set gate passes; it does not affect last_hidden_state.
        target = self._model.state_dict()
        for k, t in target.items():
            if "pooler.dense." in k and k not in out:
                out[k] = t.numpy()
        return out

    def target_model(self) -> Module:
        return self._model

    def map_key(self, src_key: str) -> str | None:
        return src_key

    def spec(self) -> ConversionSpec:
        cfg = self._model.config
        config = {k: _jsonable(v) for k, v in dataclasses.asdict(cfg).items()}
        task = "masked-lm" if self._kind == "mlm" else "base"
        num_classes = cfg.vocab_size if self._kind == "mlm" else cfg.hidden_size

        preprocessing = {
            "tokenizer_class": "RoFormerTokenizer",
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
            model_type="roformer",
            source=f"transformers/{_HF_ID}",
            license="apache-2.0",
            num_classes=num_classes,
            config=config,
            preprocessing=preprocessing,
            citation=_CITATION,
            title=self._title,
            paper_url=_PAPER_URL,
            datasets=["cluecorpussmall"],
            meta=meta,
        )


def _make(arch: str) -> Callable[[str], Architecture]:
    def _builder(tag: str) -> Architecture:
        return RoFormerArch(arch, tag)

    return _builder


for _arch in _VARIANTS:
    register_arch(_arch)(_make(_arch))
