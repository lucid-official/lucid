"""YOLO weight converter — darknet ``.weights`` → Lucid safetensors.

Unlike every other converter in this package, the upstream here is not a
Python framework's ``state_dict`` but darknet's own binary format: a small
header followed by one flat ``float32`` stream holding every layer's
parameters in ``.cfg`` declaration order.  There are no keys, so the whole
job is getting the *order* right — and then getting the channel order of
each concatenation right, because darknet's ``[route]`` lists its inputs in
an order Lucid does not always reproduce.

Blob layout, per convolution, in cfg order::

    batch_normalize=1:  bn.bias, bn.weight, bn.running_mean, bn.running_var,
                        conv.weight
    batch_normalize=0:  conv.bias, conv.weight

``conv.weight`` is stored ``(out, in, k, k)`` — Lucid's layout already — so
no transposition is needed anywhere.

Ordering
--------
For ``yolo_v2`` / ``yolo_v3`` / ``yolo_v3_tiny`` a pre-order walk of the
Lucid module tree reproduces the cfg order exactly.  ``yolo_v4`` does not:
darknet emits each detection head *immediately* after the PAN stage feeding
it, whereas Lucid keeps the three heads on the model and the whole PAN
inside ``neck``, so the natural walk groups all three heads at the end.
:data:`_V4_TAIL` writes out the cfg order for that tail.

Concatenation order
-------------------
``[route] layers = -1,-7`` concatenates in the order listed, and for every
CSP stage that is ``(transition, skip)`` while Lucid's ``_CSPBlock.forward``
builds ``cat([skip, transition])``.  The merge convolution that consumes the
concat therefore expects its input channels in the opposite arrangement, so
the converter permutes them.  :data:`_V4_PERMUTED` lists every convolution
in YOLOv4 whose input channels need reordering; v2, v3 and v3-tiny agree
with darknet at every concat and need none.

Sources
-------
``yolov2`` / ``yolov3`` / ``yolov3-tiny`` from pjreddie.com (the original
darknet release, public domain); ``yolov4`` from the AlexeyAB darknet
GitHub release.  All four are the COCO 80-class detectors.
"""

import dataclasses
import hashlib
import os
import struct
import urllib.request
from pathlib import Path

import numpy as np

import lucid.nn as nn
from lucid.nn import Module
from tools.convert_weights._base import Architecture, ConversionSpec, register_arch

# ---------------------------------------------------------------------------
# Upstream checkpoints
# ---------------------------------------------------------------------------

_PJREDDIE = "https://pjreddie.com/media/files"
_ALEXEYAB = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal"


@dataclasses.dataclass(frozen=True)
class _Source:
    """One upstream darknet checkpoint."""

    url: str
    sha256: str
    factory: str
    repo_id: str
    title: str
    tag: str
    image_size: int
    map50: float
    """COCO mAP@0.5, as reported by the release that published the weights."""
    paper_url: str
    citation: str
    license: str


_YOLOV2_CITATION = (
    "@inproceedings{redmon2017yolo9000,\n"
    "  title={YOLO9000: Better, Faster, Stronger},\n"
    "  author={Redmon, Joseph and Farhadi, Ali},\n"
    "  booktitle={IEEE Conference on Computer Vision and Pattern "
    "Recognition (CVPR)},\n"
    "  year={2017}\n"
    "}"
)
_YOLOV3_CITATION = (
    "@article{redmon2018yolov3,\n"
    "  title={YOLOv3: An Incremental Improvement},\n"
    "  author={Redmon, Joseph and Farhadi, Ali},\n"
    "  journal={arXiv preprint arXiv:1804.02767},\n"
    "  year={2018}\n"
    "}"
)
_YOLOV4_CITATION = (
    "@article{bochkovskiy2020yolov4,\n"
    "  title={YOLOv4: Optimal Speed and Accuracy of Object Detection},\n"
    "  author={Bochkovskiy, Alexey and Wang, Chien-Yao and Liao, "
    "Hong-Yuan Mark},\n"
    "  journal={arXiv preprint arXiv:2004.10934},\n"
    "  year={2020}\n"
    "}"
)

_SOURCES: dict[str, _Source] = {
    "yolo_v2": _Source(
        url=f"{_PJREDDIE}/yolov2.weights",
        sha256="d9945162ed6f54ce1a901e3ec537bdba4d572ecae7873087bd730e5a7942df3f",
        factory="yolo_v2",
        repo_id="lucid-dl/yolo-v2",
        title="YOLOv2 (Darknet-19)",
        tag="COCO_2014",
        image_size=608,
        map50=48.1,
        paper_url="Redmon & Farhadi, 2017 — *YOLO9000: Better, Faster, "
        "Stronger* (arXiv:1612.08242)",
        citation=_YOLOV2_CITATION,
        license="other",
    ),
    "yolo_v3": _Source(
        url=f"{_PJREDDIE}/yolov3.weights",
        sha256="523e4e69e1d015393a1b0a441cef1d9c7659e3eb2d7e15f793f060a21b32f297",
        factory="yolo_v3",
        repo_id="lucid-dl/yolo-v3",
        title="YOLOv3 (Darknet-53)",
        tag="COCO_2014",
        image_size=416,
        map50=55.3,
        paper_url="Redmon & Farhadi, 2018 — *YOLOv3: An Incremental "
        "Improvement* (arXiv:1804.02767)",
        citation=_YOLOV3_CITATION,
        license="other",
    ),
    "yolo_v3_tiny": _Source(
        url=f"{_PJREDDIE}/yolov3-tiny.weights",
        sha256="dccea06f59b781ec1234ddf8d1e94b9519a97f4245748a7d4db75d5b7080a42c",
        factory="yolo_v3_tiny",
        repo_id="lucid-dl/yolo-v3-tiny",
        title="YOLOv3-Tiny",
        tag="COCO_2014",
        image_size=416,
        map50=33.1,
        paper_url="Redmon & Farhadi, 2018 — *YOLOv3: An Incremental "
        "Improvement* (arXiv:1804.02767)",
        citation=_YOLOV3_CITATION,
        license="other",
    ),
    "yolo_v4": _Source(
        url=f"{_ALEXEYAB}/yolov4.weights",
        sha256="e8a4f6c62188738d86dc6898d82724ec0964d0eb9d2ae0f0a9d53d65d108d562",
        factory="yolo_v4",
        repo_id="lucid-dl/yolo-v4",
        title="YOLOv4 (CSPDarknet-53 + SPP + PAN)",
        tag="COCO_2017",
        image_size=608,
        map50=65.7,
        paper_url="Bochkovskiy, Wang & Liao, 2020 — *YOLOv4: Optimal Speed "
        "and Accuracy of Object Detection* (arXiv:2004.10934)",
        citation=_YOLOV4_CITATION,
        license="other",
    ),
}

# ---------------------------------------------------------------------------
# YOLOv4 ordering + channel permutations
# ---------------------------------------------------------------------------

#: Convolutions from ``neck.p3_down`` onwards, in ``yolov4.cfg`` order.
#: Everything before this point is already in cfg order under a pre-order
#: module walk; only the tail interleaves the heads with the PAN stages.
_V4_TAIL: tuple[str, ...] = (
    "p3_head.0.conv",
    "p3_head.1",
    "neck.p3_down.conv",
    "neck.p4_bu.0.conv",
    "neck.p4_bu.1.conv",
    "neck.p4_bu.2.conv",
    "neck.p4_bu.3.conv",
    "neck.p4_bu.4.conv",
    "p4_head.0.conv",
    "p4_head.1",
    "neck.p4_down.conv",
    "neck.p5_bu.0.conv",
    "neck.p5_bu.1.conv",
    "neck.p5_bu.2.conv",
    "neck.p5_bu.3.conv",
    "neck.p5_bu.4.conv",
    "p5_head.0.conv",
    "p5_head.1",
)

#: ``{conv module name: source-order of Lucid's concatenated groups}``.
#:
#: Each value lists, for one convolution, where each equal-width slice of
#: *Lucid's* input comes from in *darknet's* concatenation.  ``(1, 0)`` means
#: Lucid's first half is darknet's second half.  Applied as a gather over the
#: input-channel axis.
_V4_PERMUTED: dict[str, tuple[int, ...]] = {
    # [route] -1,-7 = (transition, skip); Lucid cat([skip, transition]).
    "backbone.csp1.merge.conv": (1, 0),
    "backbone.csp2.merge.conv": (1, 0),
    "backbone.csp3.merge.conv": (1, 0),
    "backbone.csp4.merge.conv": (1, 0),
    "backbone.csp5.merge.conv": (1, 0),
    # SPP [route] -1,-3,-5,-6 = (pool13, pool9, pool5, identity);
    # Lucid cat([identity, pool5, pool9, pool13]).
    "neck.spp.post.0.conv": (3, 2, 1, 0),
    # PAN top-down [route] -1,-3 = (lateral, upsample);
    # Lucid cat([upsample, lateral]).
    "neck.p4_td.0.conv": (1, 0),
    "neck.p3_td.0.conv": (1, 0),
}


def _permute_in_channels(w: np.ndarray, order: tuple[int, ...]) -> np.ndarray:
    """Reorder ``w``'s input-channel axis by equal-width groups.

    Parameters
    ----------
    w : numpy.ndarray
        Convolution weight, ``(out, in, k, k)``.
    order : tuple of int
        ``order[i]`` is the darknet group that becomes Lucid's group ``i``.

    Returns
    -------
    numpy.ndarray
        Weight with its input channels rearranged.
    """
    groups = len(order)
    in_ch = w.shape[1]
    if in_ch % groups:
        raise ValueError(f"_permute_in_channels: {in_ch} not divisible by {groups}")
    width = in_ch // groups
    return np.concatenate([w[:, g * width : (g + 1) * width] for g in order], axis=1)


# ---------------------------------------------------------------------------
# Module ordering
# ---------------------------------------------------------------------------


def _natural_order(model: Module) -> list[tuple[str, str | None]]:
    """Pre-order walk pairing each conv with the norm that follows it.

    Returns
    -------
    list of (str, str or None)
        ``(conv module name, batch-norm module name)`` in module-tree order;
        the norm is ``None`` for the bias-carrying prediction convolutions.
    """
    rows: list[tuple[str, str | None]] = []
    pending: str | None = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            if pending is not None:
                rows.append((pending, None))
            pending = name
        elif isinstance(mod, nn.BatchNorm2d) and pending is not None:
            rows.append((pending, name))
            pending = None
    if pending is not None:
        rows.append((pending, None))
    return rows


def _darknet_order(model: Module, arch: str) -> list[tuple[str, str | None]]:
    """Return ``(conv, norm)`` pairs in darknet ``.cfg`` declaration order."""
    rows = _natural_order(model)
    if arch != "yolo_v4":
        return rows

    by_name = dict(rows)
    missing = [n for n in _V4_TAIL if n not in by_name]
    if missing:
        raise RuntimeError(f"_darknet_order: YOLOv4 tail names not found: {missing}")
    tail = set(_V4_TAIL)
    head = [(c, b) for c, b in rows if c not in tail]
    return head + [(c, by_name[c]) for c in _V4_TAIL]


# ---------------------------------------------------------------------------
# Darknet blob reader
# ---------------------------------------------------------------------------


def _fetch(url: str, sha256: str) -> Path:
    """Download ``url`` into the conversion cache and verify its digest.

    Not :func:`lucid.weights.download` — that one is the *runtime* fetcher
    for Lucid's own Hub-hosted checkpoints, and pjreddie.com answers its
    default user agent with ``403 Forbidden``.  Sources here are upstream
    third-party files, so the tool carries its own fetch.

    Parameters
    ----------
    url : str
        Upstream ``.weights`` URL.
    sha256 : str
        Expected hex digest; checked on download *and* on every cache hit.

    Returns
    -------
    pathlib.Path
        Local path to the verified file.

    Raises
    ------
    RuntimeError
        If the downloaded bytes do not match ``sha256``.
    """
    root = Path(os.environ.get("LUCID_HOME", Path.home() / ".cache" / "lucid"))
    cache = root / "darknet-src"
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / Path(url).name

    def digest(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    if dest.is_file() and digest(dest) == sha256:
        return dest

    req = urllib.request.Request(url, headers={"User-Agent": "lucid-convert-weights"})
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(req, timeout=30) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    got = digest(tmp)
    if got != sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"_fetch: {url} digest {got}, expected {sha256}")
    tmp.replace(dest)
    return dest


def _read_darknet(path: Path) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Read a darknet ``.weights`` file into its float32 payload.

    Parameters
    ----------
    path : pathlib.Path
        Local ``.weights`` file.

    Returns
    -------
    (numpy.ndarray, tuple of int)
        The flat ``float32`` parameter stream and the ``(major, minor,
        revision)`` version triple.

    Notes
    -----
    ``seen`` — the image counter that follows the version triple — widened
    from ``int32`` to ``int64`` at darknet version 0.2, so the header is 16
    bytes for the YOLOv2-era files and 20 bytes from YOLOv3 onward.
    """
    with open(path, "rb") as f:
        major, minor, revision = struct.unpack("<iii", f.read(12))
        f.read(8 if (major * 10 + minor) >= 2 else 4)  # seen
        payload = np.frombuffer(f.read(), dtype="<f4")
    return payload, (major, minor, revision)


class DarknetYOLOArch(Architecture):
    """Converter for one darknet YOLO checkpoint.

    ``source_state_dict`` emits keys already in Lucid naming, so
    :meth:`map_key` is the identity.
    """

    def __init__(self, arch: str, tag: str) -> None:
        if arch not in _SOURCES:
            raise KeyError(f"DarknetYOLOArch: unknown arch {arch!r}")
        self.arch = arch
        self.tag = tag
        self.src = _SOURCES[arch]
        if tag != self.src.tag:
            raise KeyError(f"DarknetYOLOArch: {arch!r} has tag {self.src.tag!r}, got {tag!r}")

    def target_model(self) -> Module:
        import lucid.models as models

        model = models.create_model(self.src.factory)
        return model

    def source_state_dict(self) -> dict[str, object]:
        path = _fetch(self.src.url, self.src.sha256)
        payload, version = _read_darknet(path)

        model = self.target_model()
        params = dict(model.named_parameters())
        buffers = dict(model.named_buffers())
        order = _darknet_order(model, self.arch)

        out: dict[str, object] = {}
        pos = 0

        def take(n: int, shape: tuple[int, ...]) -> np.ndarray:
            nonlocal pos
            if pos + n > payload.size:
                raise RuntimeError(
                    f"source_state_dict: {self.arch} blob exhausted at float "
                    f"{pos} (need {n}, have {payload.size - pos})"
                )
            arr = payload[pos : pos + n].reshape(shape).astype(np.float32)
            pos += n
            return arr

        for conv, norm in order:
            w_shape = tuple(int(s) for s in params[f"{conv}.weight"].shape)
            out_ch = w_shape[0]
            if norm is not None:
                out[f"{norm}.bias"] = take(out_ch, (out_ch,))
                out[f"{norm}.weight"] = take(out_ch, (out_ch,))
                out[f"{norm}.running_mean"] = take(out_ch, (out_ch,))
                out[f"{norm}.running_var"] = take(out_ch, (out_ch,))
                out[f"{norm}.num_batches_tracked"] = np.zeros((), dtype=np.int64)
            else:
                out[f"{conv}.bias"] = take(out_ch, (out_ch,))
            n = int(np.prod(w_shape))
            weight = take(n, w_shape)
            perm = _V4_PERMUTED.get(conv)
            if perm is not None:
                weight = _permute_in_channels(weight, perm)
            out[f"{conv}.weight"] = weight

        if pos != payload.size:
            raise RuntimeError(
                f"source_state_dict: {self.arch} consumed {pos} of "
                f"{payload.size} floats — layer order does not match "
                f"{self.src.url} (darknet version {version})"
            )

        # Every remaining buffer is a norm counter the blob does not carry.
        for name in buffers:
            out.setdefault(name, np.zeros((), dtype=np.int64))
        return out

    def map_key(self, src_key: str) -> str | None:
        return src_key

    def spec(self) -> ConversionSpec:
        import lucid.models as models
        from lucid.utils.transforms import Detection

        src = self.src
        model = models.create_model(src.factory)
        config = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in dataclasses.asdict(model.config).items()
        }

        # Darknet letterboxes to a square canvas (image centred) and feeds raw
        # [0, 1] pixels — no dataset mean/std is subtracted anywhere in its
        # pipeline.  Matches yolo/_weights.py.
        preprocessing = Detection(
            max_size=src.image_size,
            pad_position="center",
            pad_value=0.5,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        ).to_dict()

        n_params = sum(int(np.prod([int(s) for s in p.shape])) for p in model.parameters())
        meta = {
            "num_params": n_params,
            "recipe": src.url,
            "metrics": {"COCO": {"mAP@0.5": src.map50}},
        }

        return ConversionSpec(
            model_name=src.factory,
            architecture=self.arch,
            repo_id=src.repo_id,
            tag=self.tag,
            task="object-detection",
            model_type=self.arch,
            source=f"darknet/{Path(src.url).name}",
            license=src.license,
            num_classes=80,
            config=config,
            preprocessing=preprocessing,
            citation=src.citation,
            title=src.title,
            paper_url=src.paper_url,
            categories=[],
            datasets=["coco"],
            meta=meta,
        )


@register_arch("yolo_v2")
def _yolo_v2(tag: str) -> Architecture:
    return DarknetYOLOArch("yolo_v2", tag)


@register_arch("yolo_v3")
def _yolo_v3(tag: str) -> Architecture:
    return DarknetYOLOArch("yolo_v3", tag)


@register_arch("yolo_v3_tiny")
def _yolo_v3_tiny(tag: str) -> Architecture:
    return DarknetYOLOArch("yolo_v3_tiny", tag)


@register_arch("yolo_v4")
def _yolo_v4(tag: str) -> Architecture:
    return DarknetYOLOArch("yolo_v4", tag)
