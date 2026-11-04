"""The upstream darknet YOLO releases, and the order their parameters come in.

``tools.check_pretrained_parity`` gives darknet's own ``.cfg`` and ``.weights``
to an independent reader and compares its raw prediction convolutions with the
published Lucid checkpoint's.  That takes two things from here: where each
release is, and which Lucid convolution each layer of the file is.

A ``.weights`` file has no keys, only a small header and one flat ``float32``
stream holding every layer's parameters in ``.cfg`` declaration order::

    batch_normalize=1:  bn.bias, bn.weight, bn.running_mean, bn.running_var,
                        conv.weight
    batch_normalize=0:  conv.bias, conv.weight

For ``yolo_v2`` / ``yolo_v3`` / ``yolo_v3_tiny`` a pre-order walk of the Lucid
module tree reproduces the cfg order exactly.  ``yolo_v4`` does not: darknet
emits each detection head *immediately* after the PAN stage feeding it, whereas
Lucid keeps the three heads on the model and the whole PAN inside ``neck``, so
the natural walk groups all three heads at the end.  :data:`_V4_TAIL` writes out
the cfg order for that tail.
"""

import dataclasses
import hashlib
import os
import urllib.request
from pathlib import Path

import lucid.nn as nn
from lucid.nn import Module


@dataclasses.dataclass(frozen=True)
class Release:
    """One upstream ``.weights`` file."""

    url: str
    sha256: str


_PJREDDIE = "https://pjreddie.com/media/files"
_ALEXEYAB = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal"

#: By zoo factory.  ``yolov2`` / ``yolov3`` / ``yolov3-tiny`` are the original
#: darknet release; ``yolov4`` is the AlexeyAB GitHub release.  All four are the
#: COCO 80-class detectors.
RELEASES: dict[str, Release] = {
    "yolo_v2": Release(
        url=f"{_PJREDDIE}/yolov2.weights",
        sha256="d9945162ed6f54ce1a901e3ec537bdba4d572ecae7873087bd730e5a7942df3f",
    ),
    "yolo_v3": Release(
        url=f"{_PJREDDIE}/yolov3.weights",
        sha256="523e4e69e1d015393a1b0a441cef1d9c7659e3eb2d7e15f793f060a21b32f297",
    ),
    "yolo_v3_tiny": Release(
        url=f"{_PJREDDIE}/yolov3-tiny.weights",
        sha256="dccea06f59b781ec1234ddf8d1e94b9519a97f4245748a7d4db75d5b7080a42c",
    ),
    "yolo_v4": Release(
        url=f"{_ALEXEYAB}/yolov4.weights",
        sha256="e8a4f6c62188738d86dc6898d82724ec0964d0eb9d2ae0f0a9d53d65d108d562",
    ),
}

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


def darknet_order(model: Module, arch: str) -> list[tuple[str, str | None]]:
    """Return ``(conv, norm)`` pairs in darknet ``.cfg`` declaration order."""
    rows = _natural_order(model)
    if arch != "yolo_v4":
        return rows

    by_name = dict(rows)
    missing = [n for n in _V4_TAIL if n not in by_name]
    if missing:
        raise RuntimeError(f"darknet_order: YOLOv4 tail names not found: {missing}")
    tail = set(_V4_TAIL)
    head = [(c, b) for c, b in rows if c not in tail]
    return head + [(c, by_name[c]) for c in _V4_TAIL]


def fetch(url: str, sha256: str) -> Path:
    """Download ``url`` into the darknet cache and verify its digest.

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

    req = urllib.request.Request(url, headers={"User-Agent": "lucid-pretrained-parity"})
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(req, timeout=30) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    got = digest(tmp)
    if got != sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"fetch: {url} digest {got}, expected {sha256}")
    tmp.replace(dest)
    return dest
