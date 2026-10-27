"""G4f — `lucid.utils.transforms` vs Albumentations head-to-head.

Records per-transform timing for the deterministic pipeline most
production code hits (Resize → CenterCrop → Normalize → ToGray,
plus HSV / CLAHE / Posterize) on a standard imagenet-shaped tensor.

Auto-skipped when ``albumentations`` / ``cv2`` aren't importable
(opt-in tier).  Run explicitly with::

    python3 -m pytest lucid/test/perf/microbench/test_transforms_bench.py \\
        -m perf -s

The ``-s`` flag shows the per-test print; ``pytest-benchmark`` enriches
those with min / mean / stddev when installed.

No hard thresholds yet — the file establishes the baseline
(record-only).  Add entries to ``_thresholds.json`` once a
representative number lands on the reference hardware (M-series).
"""

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

import lucid
import lucid.utils.transforms as T
import lucid.utils.transforms.functional as F
from lucid.test._fixtures.perf import load_thresholds

A = pytest.importorskip("albumentations")
pytest.importorskip("cv2")


_THRESHOLDS = load_thresholds(Path(__file__).parent)

# Standard ImageNet-ish working size.
_H, _W = 224, 224


def _make_inputs(seed: int = 0) -> tuple[lucid.Tensor, np.ndarray]:
    """Matched ``(C, H, W)`` Lucid tensor + ``(H, W, C)`` numpy array."""
    rng = np.random.default_rng(seed)
    hwc = rng.random((_H, _W, 3), dtype=np.float32)
    chw = lucid.tensor(np.transpose(hwc, (2, 0, 1)).tolist())
    return chw, hwc


def _run_lucid(tf: T.Transform, chw: lucid.Tensor) -> lucid.Tensor:
    """Keep the result so the common fixture can submit lazy work."""
    return tf(T.Image(chw)).data


def _run_albu(aug: object, hwc: np.ndarray) -> None:
    aug(image=hwc)  # type: ignore[operator]


def _report(name: str, lucid_s: float, albu_s: float) -> None:
    print(
        f"\n  {name:30s}  lucid_observed={lucid_s * 1e3:7.2f} ms  "
        f"albu_10_call_mean={albu_s * 1e3:7.2f} ms  "
        "(different sampling protocols; no speedup claim)"
    )


def _bench_pair(
    bench: Callable[..., object],
    name: str,
    lucid_fn: Callable[[], object],
    albu_fn: Callable[[], object],
) -> None:
    """Time the Lucid path; also time Albu inline (one-shot) and report.

    Only the ``bench(lucid_fn)`` call participates in pytest-benchmark
    statistics — Albu is a single ``perf_counter`` reading next to it,
    which is enough to flag order-of-magnitude divergence.
    """
    import time as _t

    # Albu single-shot
    t0 = _t.perf_counter()
    for _ in range(10):
        albu_fn()
    albu_s = (_t.perf_counter() - t0) / 10.0

    # Lucid through the bench fixture
    bench(lucid_fn)
    lucid_s = getattr(bench, "last_elapsed", None)
    if lucid_s is None:
        # pytest-benchmark mode — fish the stats out.
        stats = getattr(bench, "stats", None)
        lucid_s = float(stats.stats.mean) if stats is not None else albu_s
    _report(name, float(lucid_s), albu_s)


# ── deterministic transforms — head-to-head ─────────────────────────


@pytest.mark.perf
class TestDeterministicBench:
    def test_resize_bilinear(self, bench: Callable[..., object]) -> None:
        chw, hwc = _make_inputs()
        _bench_pair(
            bench,
            "Resize 224→128 (bilinear)",
            lambda: _run_lucid(T.Resize(128, 128, p=1.0), chw),
            lambda: _run_albu(A.Resize(128, 128, interpolation=1, p=1.0), hwc),
        )

    def test_resize_nearest(self, bench: Callable[..., object]) -> None:
        chw, hwc = _make_inputs()
        _bench_pair(
            bench,
            "Resize 224→128 (nearest)",
            lambda: _run_lucid(T.Resize(128, 128, interpolation="nearest", p=1.0), chw),
            lambda: _run_albu(A.Resize(128, 128, interpolation=0, p=1.0), hwc),
        )

    def test_center_crop(self, bench: Callable[..., object]) -> None:
        chw, hwc = _make_inputs()
        _bench_pair(
            bench,
            "CenterCrop 224→160",
            lambda: _run_lucid(T.CenterCrop(160, 160, p=1.0), chw),
            lambda: _run_albu(A.CenterCrop(160, 160, p=1.0), hwc),
        )

    def test_hflip(self, bench: Callable[..., object]) -> None:
        chw, hwc = _make_inputs()
        _bench_pair(
            bench,
            "HorizontalFlip 224",
            lambda: _run_lucid(T.HorizontalFlip(p=1.0), chw),
            lambda: _run_albu(A.HorizontalFlip(p=1.0), hwc),
        )

    def test_normalize(self, bench: Callable[..., object]) -> None:
        chw, hwc = _make_inputs()
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        _bench_pair(
            bench,
            "Normalize 224",
            lambda: _run_lucid(T.Normalize(mean, std, max_pixel_value=1.0, p=1.0), chw),
            lambda: _run_albu(A.Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0), hwc),
        )

    def test_to_gray(self, bench: Callable[..., object]) -> None:
        chw, hwc = _make_inputs()
        _bench_pair(
            bench,
            "ToGray 224",
            lambda: _run_lucid(T.ToGray(p=1.0), chw),
            lambda: _run_albu(A.ToGray(p=1.0), hwc),
        )

    def test_invert(self, bench: Callable[..., object]) -> None:
        chw, hwc = _make_inputs()
        _bench_pair(
            bench,
            "InvertImg 224",
            lambda: _run_lucid(T.InvertImg(p=1.0), chw),
            lambda: _run_albu(A.InvertImg(p=1.0), hwc),
        )

    def test_posterize(self, bench: Callable[..., object]) -> None:
        chw, hwc = _make_inputs()
        _bench_pair(
            bench,
            "Posterize 3-bit",
            lambda: _run_lucid(T.Posterize(num_bits=3, mode="uint8_mask", p=1.0), chw),
            lambda: _run_albu(A.Posterize(num_bits=3, p=1.0), hwc),
        )


# ── cv2-accuracy transforms — where Lucid pays the float-tensor cost ──


@pytest.mark.perf
class TestCv2AccuracyBench:
    def test_hsv_shift(self, bench: Callable[..., object]) -> None:
        chw, hwc = _make_inputs()
        _bench_pair(
            bench,
            "adjust_hsv (10,20,5)",
            lambda: F.adjust_hsv(chw, 10.0, 20.0, 5.0),
            lambda: _run_albu(
                A.HueSaturationValue(
                    hue_shift_limit=(10, 10),
                    sat_shift_limit=(20, 20),
                    val_shift_limit=(5, 5),
                    p=1.0,
                ),
                hwc,
            ),
        )

    def test_clahe_default(self, bench: Callable[..., object]) -> None:
        chw, hwc = _make_inputs()
        _bench_pair(
            bench,
            "CLAHE 8x8 tiles",
            lambda: F.clahe(chw, 4.0, (8, 8)),
            lambda: _run_albu(A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0), hwc),
        )


# ── full classification preprocessing pipeline ──────────────────────


@pytest.mark.perf
class TestPipelineBench:
    def test_imagenet_eval_pipeline(self, bench: Callable[..., object]) -> None:
        """Resize 256 → CenterCrop 224 → Normalize: the canonical eval
        pipeline shipped with every reference-style preset."""
        chw, hwc = _make_inputs()
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        lucid_tf = T.Compose(
            [
                T.SmallestMaxSize(256, p=1.0),
                T.CenterCrop(224, 224, p=1.0),
                T.Normalize(mean, std, max_pixel_value=1.0, p=1.0),
            ]
        )
        albu_tf = A.Compose(
            [
                A.SmallestMaxSize(256, interpolation=1, p=1.0),
                A.CenterCrop(224, 224, p=1.0),
                A.Normalize(mean=mean, std=std, max_pixel_value=1.0, p=1.0),
            ]
        )
        _bench_pair(
            bench,
            "ImageNet eval pipeline",
            lambda: lucid_tf(T.Image(chw)).data,
            lambda: albu_tf(image=hwc),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "perf"])
