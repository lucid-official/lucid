"""Fixed-schema renderers for ``config.json`` and the Hub model card.

Both artifacts are generated from a :class:`~tools.convert_weights._base.
ConversionSpec` so every converted checkpoint carries identical,
predictable structure.
"""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.convert_weights._base import ConversionSpec


def render_config_json(spec: "ConversionSpec") -> str:
    """Render the per-tag ``config.json`` (HF-compatible, self-contained).

    Includes the Lucid architecture config, the inference preprocessing
    spec, a ``weights`` provenance/metrics block, and ``id2label`` /
    ``label2id`` maps (the HF vision-classification convention).
    """
    id2label = {str(i): name for i, name in enumerate(spec.categories)}
    label2id = {name: i for i, name in enumerate(spec.categories)}

    payload: dict[str, object] = {
        "library_name": "lucid",
        "architecture": spec.architecture,
        "model_type": spec.model_type,
        "task": spec.task,
        "config": spec.config,
        "preprocessing": spec.preprocessing,
        "weights": {
            "tag": spec.tag,
            "num_classes": spec.num_classes,
            "license": spec.license,
            "source": spec.source,
            **spec.meta,
        },
    }
    if spec.categories:
        payload["id2label"] = id2label
        payload["label2id"] = label2id

    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


# Lucid task string → Hugging Face Hub ``pipeline_tag`` (HF has a single
# segmentation pipeline; Lucid distinguishes semantic vs instance).
_HF_PIPELINE_TAG: dict[str, str] = {
    "image-classification": "image-classification",
    "video-classification": "video-classification",
    "object-detection": "object-detection",
    "semantic-segmentation": "image-segmentation",
    "instance-segmentation": "image-segmentation",
    # World models. The Hub has no latent-dynamics pipeline, and the card
    # generator passes an unmapped task straight through as the tag, which
    # would be invalid; "reinforcement-learning" is the closest valid one.
    "world-modeling": "reinforcement-learning",
    # Same shape of problem: the Hub has no class-conditional generation
    # pipeline, and "image-generation" is not on its list at all, so it
    # passed straight through and the Hub rejected the frontmatter.
    # "unconditional-image-generation" is the diffusion entry it does have.
    "image-generation": "unconditional-image-generation",
    # Text tasks (Lucid task string → HF Hub pipeline_tag).
    "base": "feature-extraction",
    "masked-lm": "fill-mask",
    "causal-lm": "text-generation",
    "seq2seq-lm": "text2text-generation",
    "sequence-classification": "text-classification",
    "token-classification": "token-classification",
    "question-answering": "question-answering",
}

# Lucid task strings that operate on token-id inputs (no image preprocessing).
_TEXT_TASKS: frozenset[str] = frozenset(
    {
        "base",
        "masked-lm",
        "causal-lm",
        "seq2seq-lm",
        "sequence-classification",
        "token-classification",
        "question-answering",
    }
)


def _hf_pipeline_tag(task: str) -> str:
    """Map a Lucid ``task`` string to a valid HF Hub ``pipeline_tag``."""
    return _HF_PIPELINE_TAG.get(task, task)


def _first_metrics(spec: "ConversionSpec") -> dict[str, object]:
    """Return the metric dict of the first dataset in ``meta['metrics']``."""
    metrics = spec.meta.get("metrics", {})
    if isinstance(metrics, dict):
        for _dataset, vals in metrics.items():
            if isinstance(vals, dict):
                return dict(vals)
            break
    return {}


def _source_label(source: str) -> str:
    """Short provenance label for the weights table.

    ``source`` is either a framework path (``torchvision/ResNet18_Weights…``)
    or a Hub URL.  Splitting on the first slash suits the former and turns
    the latter into the useless ``https:`` — so a URL is reduced to its
    ``owner/repo`` instead.
    """
    if source.startswith(("http://", "https://")):
        parts = [p for p in source.split("/") if p]
        return "/".join(parts[-2:]) if len(parts) >= 2 else source
    return source.split("/")[0]


def _metrics_table(spec: "ConversionSpec", tags: "list[str] | None" = None) -> str:
    """Build the full markdown tag-comparison table.

    The metric columns are taken from whatever keys the converter put in
    ``meta['metrics'][<dataset>]`` (``acc@1``/``acc@5`` for classification,
    ``box mAP``/``mask mAP`` for detection, ``mIoU`` for segmentation, …),
    so the table is task-agnostic.

    Parameters
    ----------
    spec : ConversionSpec
        The tag currently being written.
    tags : list of str or None, optional
        Every tag the repo holds, when more than this one exists.  Rows
        for the others are read back from their ``config.json``, which
        is the only place a previous conversion's numbers survive — the
        spec in hand describes one tag.
    """
    vals = _first_metrics(spec)
    metric_keys = list(vals.keys())
    headers = ["Tag", *metric_keys, "Params", "GFLOPs", "Size", "Source"]

    def _row(tag: str, meta: dict[str, object], source: str) -> str:
        params = meta.get("num_params", "—")
        params_m = f"{params / 1e6:.1f}M" if isinstance(params, (int, float)) else "—"
        size = meta.get("file_size_mb", "—")
        metrics = meta.get("metrics", {})
        first: dict[str, object] = {}
        if isinstance(metrics, dict):
            for _dataset, entry in metrics.items():
                if isinstance(entry, dict):
                    first = dict(entry)
                break
        cells = [
            f"`{tag}`" + (" *(default)*" if tag == _DEFAULT_OF(tags, spec) else ""),
            *[str(first.get(k, "—")) for k in metric_keys],
            params_m,
            str(meta.get("gflops", "—")),
            f"{size} MB" if size != "—" else "—",
            _source_label(source),
        ]
        return "| " + " | ".join(cells) + " |"

    rows = []
    for tag in tags or [spec.tag]:
        if tag == spec.tag:
            rows.append(_row(tag, dict(spec.meta), spec.source))
            continue
        sibling = _SIBLING_META(tag)
        rows.append(_row(tag, sibling[0], sibling[1]))

    header_row = "| " + " | ".join(headers) + " |"
    sep_row = "|" + "|".join(["---"] * len(headers)) + "|"
    return "\n".join([header_row, sep_row, *rows])


# Filled in by :func:`render_model_card` before the table is built, so the
# row builder can reach the sibling tags' recorded numbers without the
# renderer having to thread a directory through every helper.
_SIBLING_LOOKUP: "dict[str, tuple[dict[str, object], str]]" = {}


def _SIBLING_META(tag: str) -> "tuple[dict[str, object], str]":
    """Recorded meta + source for a tag written by an earlier conversion."""
    return _SIBLING_LOOKUP.get(tag, ({}, ""))


def _DEFAULT_OF(tags: "list[str] | None", spec: "ConversionSpec") -> str:
    """Which tag the card marks as the default.

    With one tag it is that tag.  With several the card cannot know the
    enum's ``DEFAULT``, and guessing "whichever was written last" is how
    the first version of this card advertised the wrong one — so the
    lowest-sorted tag is used, which is stable across write order.
    """
    return spec.tag if not tags or len(tags) == 1 else sorted(tags)[0]


def _text_usage_snippet(spec: "ConversionSpec", enum_name: str) -> str:
    """Build the usage example body for a token-id (text) model."""
    preamble = (
        "import lucid\n"
        "import lucid.models as models\n"
        f"from lucid.models.weights import {enum_name}\n\n"
        "# default tag\n"
        f"model = models.{spec.model_name}(pretrained=True)\n\n"
        "# explicit tag (enum or string)\n"
        f"model = models.{spec.model_name}(weights={enum_name}.{spec.tag})\n"
        f'model = models.{spec.model_name}(pretrained="{spec.tag}")\n\n'
        "# feed token ids (tokenize with the matching lucid.utils.tokenizer)\n"
        "input_ids = lucid.tensor([[101, 7592, 2088, 102]], dtype=lucid.int64)\n"
        "out = model(input_ids)\n"
    )
    if spec.task == "base":
        tail = "hidden = out.last_hidden_state  # (B, T, hidden_size)\n"
    elif spec.task in ("masked-lm", "causal-lm", "seq2seq-lm"):
        tail = "logits = out.logits  # (B, T, vocab_size)\n"
    elif spec.task == "question-answering":
        tail = "start, end = out.start_logits, out.end_logits  # (B, T) each\n"
    else:  # sequence-classification / token-classification
        tail = "logits = out.logits  # classification logits\n"
    return preamble + tail


def _clip_frames(spec: "ConversionSpec") -> int:
    """Frames one clip carries, from the tag's own preprocessing block."""
    value = spec.preprocessing.get("frames_per_clip", 64)
    return int(value) if isinstance(value, (int, float, str)) else 64


def _clip_size(spec: "ConversionSpec") -> int:
    """Spatial side of one frame, from the tag's own preprocessing block."""
    value = spec.preprocessing.get("size", 256)
    return int(value) if isinstance(value, (int, float, str)) else 256


def _usage_snippet(spec: "ConversionSpec", enum_name: str) -> str:
    """Build the task-appropriate Python usage example body."""
    if spec.task in _TEXT_TASKS:
        return _text_usage_snippet(spec, enum_name)
    is_video = spec.task == "video-classification"
    # A world model is conditioned on more than pixels, and on what varies
    # by family.  The one shape this generator can write down is the
    # action-conditioned video model's, which its preprocessing declares.
    is_action_video = (
        spec.task == "world-modeling" and spec.preprocessing.get("type") == "video-with-actions"
    )
    preamble = (
        ("import lucid\n" if is_video or is_action_video else "")
        + "import lucid.models as models\n"
        + f"from lucid.models.weights import {enum_name}\n\n"
        "# default tag\n"
        f"model = models.{spec.model_name}(pretrained=True)\n\n"
        "# explicit tag (enum or string)\n"
        f"model = models.{spec.model_name}(weights={enum_name}.{spec.tag})\n"
        f'model = models.{spec.model_name}(pretrained="{spec.tag}")\n\n'
        "# preprocessing travels with the weights\n"
        f"weights = {enum_name}.{spec.tag}\n"
        "preprocess = weights.transforms()\n"
    )
    # Video models take a decoded clip rather than one preprocessed image,
    # so they supply their own call; every other task shares the image one.
    if is_video:
        preamble += (
            "# the model consumes a decoded (B, T, C, H, W) clip\n"
            f"video = lucid.rand(1, {_clip_frames(spec)}, 3,"
            f" {_clip_size(spec)}, {_clip_size(spec)})\n"
            "out = model(video)\n"
        )
    elif is_action_video:
        frames = _clip_frames(spec)
        side = _clip_size(spec)
        # One step is one frame, so a clip of T frames has T - 1
        # transitions and that many conditioning rows.
        steps = max(frames - 1, 1)
        preamble += (
            "# a clip, plus one action and one state per transition\n"
            f"video = lucid.rand(1, {frames}, 3, {side}, {side})\n"
            f"actions = lucid.rand(1, {steps},"
            f" {spec.preprocessing.get('action_dim', 7)})\n"
            f"states = lucid.rand(1, {steps},"
            f" {spec.preprocessing.get('state_dim', 7)})\n"
            "out = model(video, actions, states)\n"
        )
    elif spec.task == "world-modeling":
        preamble += (
            "# a world model's conditioning differs by family — see the\n"
            "# factory's docstring for the signature this one takes\n"
        )
    else:
        preamble += "out = model(preprocess(image)[None])\n"

    if is_action_video:
        tail = "latents = out.prediction  # (B, steps * tokens_per_step, dim)\n"
    elif spec.task == "world-modeling":
        tail = ""
    elif spec.task == "object-detection":
        tail = (
            "# ObjectDetectionOutput: per-query/proposal class logits + boxes\n"
            "logits, boxes = out.logits, out.pred_boxes\n"
        )
    elif spec.task == "instance-segmentation":
        tail = (
            "# InstanceSegmentationOutput: class logits + boxes + per-instance masks\n"
            "logits, boxes, masks = out.logits, out.pred_boxes, out.pred_masks\n"
        )
    elif spec.task == "semantic-segmentation":
        tail = (
            "# SemanticSegmentationOutput: per-pixel class logits (B, C, H, W)\n"
            "seg = out.logits.argmax(axis=1)  # (B, H, W) class indices\n"
        )
    elif spec.task == "image-generation":
        tail = (
            "# A latent diffusion backbone predicts noise, not labels.\n"
            "eps = out[:, : model.config.in_channels]  # (B, C, H, W)\n"
        )
    else:  # image-classification / video-classification
        tail = "logits = out.logits  # (B, num_classes)\n"
    return preamble + tail


def _conversion_evidence(spec: "ConversionSpec") -> str:
    """State what was actually checked, not what conversions usually check.

    Every conversion passes the same gate — a 1:1 key set against a
    freshly built model, a per-key shape check and a real strict load —
    so that sentence is always true.  A numerical comparison against the
    source needs a runnable reference implementation, which some families
    have and some do not, so it is claimed only when the converter
    recorded one in ``meta['parity']``.
    """
    base = (
        "Key set, tensor shapes and a strict load verified against a " "freshly built Lucid model."
    )
    parity = spec.meta.get("parity")
    if isinstance(parity, str) and parity:
        return f"{base}\nNumerical parity against the source: {parity}."
    return base


def _model_index(spec: "ConversionSpec") -> str:
    """Build the YAML ``model-index`` block from metrics."""
    metrics = spec.meta.get("metrics", {})
    if not isinstance(metrics, dict) or not metrics:
        return ""
    dataset, vals = next(iter(metrics.items()))
    if not isinstance(vals, dict):
        return ""
    metric_lines = "\n".join(f"          - {{ type: {k}, value: {v} }}" for k, v in vals.items())
    ds_type = str(dataset).lower().replace(" ", "-")
    return (
        "model-index:\n"
        f"  - name: {spec.architecture.replace('_', '-')}\n"
        "    results:\n"
        f"      - task: {{ type: {spec.task} }}\n"
        f"        dataset: {{ name: {dataset}, type: {ds_type} }}\n"
        "        metrics:\n"
        f"{metric_lines}\n"
    )


def render_model_card(spec: "ConversionSpec", tags: "list[str] | None" = None) -> str:
    """Render the repo-level ``README.md`` model card.

    YAML frontmatter (``library_name``, license, tags, datasets,
    ``pipeline_tag``, ``model-index``) + a tag comparison table + usage
    snippet + conversion provenance + license + citation.

    Parameters
    ----------
    spec : ConversionSpec
        The tag being written.  Its title and provenance head the card.
    tags : list of str or None, optional
        Every tag the repo holds.  The card is repo-level but rendered
        from one tag, so without this a second tag's write silently
        replaced the first and the table named the wrong default.  When
        given, rows are emitted for all of them.
    """
    metrics = spec.meta.get("metrics", {})
    # spec.datasets wins (lets a converter list every dataset the checkpoint
    # touched — e.g. ``["imagenet-22k", "imagenet-1k"]`` for an in22k → in1k
    # finetune); empty falls back to the metrics' first key for the simple
    # single-dataset case.
    dataset_names: list[str] = list(spec.datasets)
    if not dataset_names and isinstance(metrics, dict) and metrics:
        dataset_names = [str(next(iter(metrics))).lower().replace(" ", "-")]

    # The enum class name has irregular casing (SEResNet / ResNeXt / ViT /
    # PVTv2 / MaxViT …) that cannot be derived from the factory string, so
    # resolve the *actual* registered class instead of guessing.
    from lucid.weights import weights_for

    _enum = weights_for(spec.model_name)
    enum_name = (
        _enum.__name__
        if _enum is not None
        else "".join(p.capitalize() for p in spec.architecture.split("_")) + "Weights"
    )
    paper_line = f"\n> {spec.paper_url}\n" if spec.paper_url else ""

    frontmatter = [
        "---",
        "library_name: lucid",
        f"license: {spec.license}",
        "tags:",
        f"  - {spec.task}",
        f"  - {spec.architecture.split('_')[0]}",
        "  - lucid",
    ]
    if dataset_names:
        frontmatter.append("datasets:")
        frontmatter += [f"  - {d}" for d in dataset_names]
    frontmatter += [f"pipeline_tag: {_hf_pipeline_tag(spec.task)}"]
    mi = _model_index(spec)
    if mi:
        frontmatter.append(mi.rstrip("\n"))
    frontmatter.append("---")

    body = f"""
# {spec.title}
{paper_line}
[Lucid](https://github.com/lucid-official/lucid) port of `{spec.source}`,
converted to Lucid-native safetensors.

## Available weights

{_metrics_table(spec, tags)}

## Usage

```python
{_usage_snippet(spec, enum_name)}```

## Conversion

Converted from `{spec.source}` via
`python -m tools.convert_weights {spec.architecture} --tag {spec.tag}`.
{_conversion_evidence(spec)}

## License

`{spec.license}` — inherited from the original weights.

## Citation

```
{spec.citation}
```
"""
    return "\n".join(frontmatter) + "\n" + body
