"""Published darknet weights and legacy user checkpoints need different layouts."""

from dataclasses import asdict

import numpy as np
import pytest

import lucid
from lucid.models.vision.yolo import _v2
from lucid.nn._shadow import shadow_alloc


@pytest.mark.parametrize("shape", [(1, 4, 4, 6), (2, 8, 6, 4), (1, 64, 14, 14)])
def test_original_flat_index_reorg_and_backward(shape: tuple[int, ...]) -> None:
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    batch, channels, height, width = shape
    indices = []
    for b in range(batch):
        for k in range(channels):
            for j in range(height):
                for i in range(width):
                    out_channels = channels // 4
                    offset = k // out_channels
                    w2 = i * 2 + offset % 2
                    h2 = j * 2 + offset // 2
                    indices.append(
                        w2 + width * 2 * (h2 + height * 2 * (k % out_channels + out_channels * b))
                    )
    wanted = values.reshape(-1)[indices].reshape(batch, channels * 4, height // 2, width // 2)
    x = lucid.tensor(values, requires_grad=True)
    got = _v2._darknet_reorg(x, 2)
    np.testing.assert_array_equal(got.numpy(), wanted)
    assert not np.array_equal(_v2._space_to_depth(x, 2).numpy(), wanted)
    covector = np.arange(values.size, dtype=np.float32).reshape(wanted.shape) / values.size
    (got * lucid.tensor(covector)).sum().backward()
    expected_grad = np.empty(values.size, dtype=np.float32)
    expected_grad[indices] = covector.reshape(-1)
    np.testing.assert_array_equal(x.grad.numpy(), expected_grad.reshape(shape))


def test_published_factory_selects_reorg_without_changing_legacy_config(
    monkeypatch,
) -> None:
    monkeypatch.setattr(_v2.weights_mod, "load_weight_entry", lambda *a, **k: None)
    with shadow_alloc():
        published = _v2.yolo_v2(pretrained=True)
        legacy = _v2.yolo_v2()
    assert published.config.darknet_reorg is True
    assert legacy.config.darknet_reorg is False
    saved = asdict(published.config)
    saved.pop("model_type", None)
    assert _v2.YOLOV2Config(**saved).darknet_reorg is True
    saved.pop("darknet_reorg")
    assert _v2.YOLOV2Config(**saved).darknet_reorg is False
    with pytest.raises(ValueError, match="require darknet_reorg=True"):
        _v2.yolo_v2(pretrained=True, darknet_reorg=False)


@pytest.mark.parametrize("shape,block", [((1, 3, 4, 4), 2), ((1, 4, 3, 4), 2), ((1, 4, 4, 4), 0)])
def test_invalid_reorg_dimensions_are_rejected(shape, block) -> None:
    with pytest.raises(ValueError, match="divisible"):
        _v2._darknet_reorg(lucid.zeros(shape), block)
