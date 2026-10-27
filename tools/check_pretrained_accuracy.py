#!/usr/bin/env python3
"""tools/check_pretrained_accuracy.py — what the published classifiers name correctly.

``check_weight_fit`` asks whether a checkpoint loads; ``check_pretrained_parity``
whether it computes what its source computes, on inputs with no labels.
Neither ever showed a model a photograph through its own preprocessing and
counted what it got right — which is the path a user takes:
``weights.transforms()`` on a decoded image, then the argmax.  A preset with
the wrong mean, a crop ratio off by one resize, an interpolation the weights
were never trained with: each loads, matches its source on random tensors,
and loses accuracy only on real images.

This scores every ImageNet-1k classifier on ImageNet-V2 (matched frequency;
Recht et al., 2019, "Do ImageNet Classifiers Generalize to ImageNet?"), a
re-collection of the validation set released for exactly this: 10,000
labelled images, ten per class, under the MIT licence.  Each checkpoint is
scored twice on the same images — Lucid through its own preset, the source it
was converted from through the source's own preprocessing — and judged on
the two together:

* **agreement** — the share of images both name the same class, which only
  the weights and the preprocessing can move;
* **gap** — Lucid's top-1 minus the source's.

ImageNet-V2 is harder than the validation set published numbers come from —
every model scores 8 to 12 points lower on it — so a published ``acc@1`` is
shown beside the result for orientation and never compared.

Downloads go to a private temporary cache, one checkpoint at a time, and are
deleted as each comparison ends; nothing in a shared cache is touched.

Run::

    python -m tools.check_pretrained_accuracy --data ~/.cache/lucid/datasets
    python -m tools.check_pretrained_accuracy --data DIR --model resnet_50_cls
    python -m tools.check_pretrained_accuracy --data DIR --source timm --per-class 10

Exit codes
----------
0 — every checkpoint compared agrees with its source within the bounds below.
1 — at least one does not.
2 — nothing could be compared (no reference installed, no data).
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

import lucid
import lucid.models  # noqa: F401 — populates the registry
from lucid.models import create_model
from lucid.models._registry import _REGISTRY
from lucid.test._fixtures.ref_framework import _REF_NAME, ref_module, ref_vision_module, zoo_module
from lucid.weights._registry import _WEIGHTS_BY_MODEL
from tools.check_pretrained_parity import _reference_for, _reference_io

#: The extracted archive's directory: one folder per class index, 0 to 999.
DATASET = "imagenetv2-matched-frequency-format-val"

#: Share of images on which Lucid and the source must name the same class.
#: Two implementations of one resize differ in the last bits of a few
#: pixels, and that moves the argmax of an image the model was unsure of —
#: this bound leaves room for that and no more.
MIN_AGREEMENT = 0.97

#: Largest top-1 gap, in points, Lucid may show against the source.
MAX_GAP = 1.0

#: Images through a model at once.
BATCH = 50


def _images(root: Path, per_class: int) -> list[tuple[Path, int]]:
    """The first ``per_class`` images of every class, as (path, label)."""
    chosen: list[tuple[Path, int]] = []
    for label in range(1000):
        folder = root / str(label)
        files = sorted(folder.glob("*.jpeg")) + sorted(folder.glob("*.jpg"))
        if not files:
            raise SystemExit(f"{folder} holds no images — is --data the extracted archive?")
        chosen.extend((path, label) for path in files[:per_class])
    return chosen


def _decode(paths: list[Path]) -> list[object]:
    from PIL import Image  # noqa: PLC0415 — a bridge to the outside world

    decoded = []
    for path in paths:
        with Image.open(path) as image:
            decoded.append(image.convert("RGB"))
    return decoded


def _lucid_top1(name: str, tag: str, images: list[object], device: str) -> np.ndarray:
    """Lucid's predicted class per image, through the checkpoint's own preset."""
    enum = _WEIGHTS_BY_MODEL[name]
    preset = getattr(enum, tag).transforms()
    model = create_model(name, pretrained=tag).eval().to(device)
    predictions = []
    with lucid.no_grad():
        for start in range(0, len(images), BATCH):
            batch = []
            for image in images[start : start + BATCH]:
                # The decoded 8-bit pixels, as a user's image hands them
                # over; the preset resizes them the way PIL does.  Scaled to
                # float first, they were resized with the fractions PIL
                # rounds away, and convnext_xlarge agreed on 96.4% (CHA-13).
                pixels = np.asarray(image)
                batch.append(preset(lucid.tensor(pixels).permute(2, 0, 1)))
            out = model(lucid.stack(batch).to(device))
            logits = getattr(out, "logits", out)
            predictions.append(np.asarray(logits.argmax(dim=-1).numpy()))
    del model
    gc.collect()
    return np.concatenate(predictions)


def _source_top1(kind: str, identifier: str, images: list[object]) -> np.ndarray:
    """The source's predicted class per image, through its own preprocessing."""
    ref = ref_module()
    assert ref is not None
    if kind == "timm":
        timm = zoo_module()
        assert timm is not None
        model = timm.create_model(identifier, pretrained=True).eval()
        config = timm.data.resolve_data_config({}, model=model)
        transform = timm.data.create_transform(**config)
    else:
        vision = ref_vision_module()
        assert vision is not None
        enum_name, tag = identifier.split(".", 1)
        weights = getattr(getattr(vision.models, enum_name), tag)
        factory = getattr(vision.models, enum_name.removesuffix("_Weights").lower())
        model = factory(weights=weights).eval()
        transform = weights.transforms()
    device = "mps" if ref.backends.mps.is_available() else "cpu"
    model = model.to(device)
    predictions = []
    with ref.inference_mode():
        for start in range(0, len(images), BATCH):
            batch = ref.stack([transform(image) for image in images[start : start + BATCH]])
            predictions.append(model(batch.to(device)).argmax(dim=-1).cpu().numpy())
    del model
    gc.collect()
    return np.concatenate(predictions)


def _targets(args: argparse.Namespace) -> list[tuple[str, str, str, str]]:
    """(factory, tag, source kind, source identifier) for every classifier."""
    found = []
    for name, enum in sorted(_WEIGHTS_BY_MODEL.items()):
        if getattr(_REGISTRY.get(name), "task", None) != "image-classification":
            continue
        if args.model and name not in args.model.split(","):
            continue
        for tag, member in enum.__members__.items():
            if tag == "DEFAULT" or member.entry.num_classes != 1000:
                continue
            reference = _reference_for(str(member.entry.meta.get("source", "")))
            if reference is None or reference[0] not in ("timm", "vision"):
                continue
            if args.source and reference[0] != args.source:
                continue
            found.append((name, tag, reference[0], reference[1]))
    return found[: args.limit] if args.limit else found


def _published(name: str, tag: str) -> float | None:
    metrics = getattr(_WEIGHTS_BY_MODEL[name], tag).entry.meta.get("metrics", {})
    value = (metrics or {}).get("ImageNet-1k", {}).get("acc@1")
    return float(value) if value is not None else None


def _check(args: argparse.Namespace) -> int:
    root = Path(args.data).expanduser() / DATASET
    if not root.is_dir():
        print(f"no {DATASET} under {args.data}", file=sys.stderr)
        return 2
    if ref_module() is None:
        print("the reference framework is not installed — nothing to compare", file=sys.stderr)
        return 2
    chosen = _images(root, args.per_class)
    labels = np.array([label for _, label in chosen])
    started = time.time()
    images = _decode([path for path, _ in chosen])
    print(f"{len(images)} images decoded in {time.time() - started:.0f} s", flush=True)

    targets = _targets(args)
    report: list[dict[str, object]] = []
    bad = 0
    for index, (name, tag, kind, identifier) in enumerate(targets, 1):
        started = time.time()
        row: dict[str, object] = {"model": name, "tag": tag, "source": f"{kind}/{identifier}"}
        try:
            ours = _lucid_top1(name, tag, images, args.device)
            with _reference_io():
                theirs = _source_top1(kind, identifier, images)
        except Exception as exc:  # noqa: BLE001 — one checkpoint must not end the run
            row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            bad += 1
            print(f"  [{index}/{len(targets)}] {name} [{tag}]  ERROR {row['error']}", flush=True)
            report.append(row)
            continue
        finally:
            args.clean()
        top1_ours = float((ours == labels).mean() * 100)
        top1_theirs = float((theirs == labels).mean() * 100)
        agreement = float((ours == theirs).mean())
        gap = top1_ours - top1_theirs
        ok = agreement >= MIN_AGREEMENT and abs(gap) <= MAX_GAP
        bad += not ok
        row.update(
            top1=round(top1_ours, 2),
            source_top1=round(top1_theirs, 2),
            agreement=round(agreement, 4),
            gap=round(gap, 2),
            published_val_top1=_published(name, tag),
            ok=ok,
            seconds=round(time.time() - started, 1),
        )
        report.append(row)
        print(
            f"  [{index}/{len(targets)}] {name:28s} {tag:22s} top-1 {top1_ours:5.1f} "
            f"vs {top1_theirs:5.1f}  agree {agreement:6.1%}  {'ok' if ok else 'DIFFERS'}",
            flush=True,
        )
    if args.json:
        Path(args.json).write_text(json.dumps({"images": len(images), "rows": report}, indent=1))
    compared = sum("top1" in row for row in report)
    print(f"\n{compared} checkpoints compared, {bad} not within bounds")
    if not compared:
        return 2
    return 1 if bad else 0


def _isolated(arguments: list[str]) -> int:
    """Run in a private download cache, so deleting each download is safe."""
    with tempfile.TemporaryDirectory(prefix="lucid-pretrained-accuracy-") as directory:
        root = Path(directory)
        environment = dict(os.environ)
        environment.update(
            LUCID_HOME=str(root / "lucid"),
            HF_HOME=str(root / "hf"),
            HF_HUB_CACHE=str(root / "hf" / "hub"),
            HUGGINGFACE_HUB_CACHE=str(root / "hf" / "hub"),
        )
        # The reference framework's own download directory, named after it.
        environment[f"{_REF_NAME.upper()}_HOME"] = str(root / "ref")
        # A progress bar per download would bury the one line per checkpoint.
        environment["TQDM_DISABLE"] = "1"
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "tools.check_pretrained_accuracy",
                *arguments,
                "--_owned-cache",
                str(root),
            ],
            env=environment,
            check=False,
        ).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--data", required=True, help="directory holding the extracted archive")
    parser.add_argument("--per-class", type=int, default=2, help="images per class (max 10)")
    parser.add_argument("--model", help="only these factories, comma-separated")
    parser.add_argument("--source", choices=("timm", "vision"), help="only this source")
    parser.add_argument("--limit", type=int, default=0, help="stop after N checkpoints")
    parser.add_argument("--device", default="metal", help="Lucid's device")
    parser.add_argument("--json", help="write every row here")
    parser.add_argument(
        "--keep-downloads",
        action="store_true",
        help="use the shared caches and keep what is downloaded",
    )
    parser.add_argument("--_owned-cache", dest="owned", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not args.keep_downloads and not args.owned:
        return _isolated(sys.argv[1:])

    def clean() -> None:
        # Inside the private cache only: whatever the last comparison
        # downloaded goes, so the run never holds more than one pair.
        if args.owned:
            for child in Path(args.owned).iterdir():
                subprocess.run(["rm", "-rf", str(child)], check=False)

    args.clean = clean
    return _check(args)


if __name__ == "__main__":
    sys.exit(main())
