"""Preset round-trip + multi-target contract tests.

Each of the 6 registered presets is exercised on three axes:

1. :meth:`to_dict` / :meth:`from_dict` is a fixed-point — repeat
   round-trips do not drift.
2. :class:`AutoTransformsPreset.from_dict` resolves the right
   subclass.
3. The preset, called on a multi-target sample dict (image + mask /
   image + boxes / image + keypoints), threads each target through
   the inner geometric stages with consistent params so coords /
   labels stay aligned.

These are the contract the pretrained-weights system relies on.
"""

import pytest

import lucid
import lucid.utils.transforms as T

# ── round-trip ──────────────────────────────────────────────────────


def _round_trip(preset: T.TransformsPreset) -> None:
    cfg = preset.to_dict()
    back = T.AutoTransformsPreset.from_dict(cfg)
    assert type(back) is type(
        preset
    ), f"{type(preset).__name__}: AutoTransformsPreset returned {type(back).__name__}"
    assert (
        back.to_dict() == cfg
    ), f"{type(preset).__name__}: round-trip drifted\n  before={cfg}\n  after={back.to_dict()}"
    # Fixed-point: a second round-trip must equal the first.
    cfg2 = back.to_dict()
    back2 = T.AutoTransformsPreset.from_dict(cfg2)
    assert back2.to_dict() == cfg2


class TestRoundTrip:
    def test_image_classification(self) -> None:
        _round_trip(T.ImageClassification(crop_size=224, resize_size=256))

    def test_image_classification_custom_stats(self) -> None:
        _round_trip(
            T.ImageClassification(
                crop_size=299,
                resize_size=342,
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
                interpolation="bicubic",
            )
        )

    def test_image_classification_augment(self) -> None:
        _round_trip(T.ImageClassificationAugment(crop_size=224))

    def test_image_classification_augment_disabled_aux(self) -> None:
        # color_jitter=0 + hflip_prob=0 should still round-trip cleanly.
        _round_trip(T.ImageClassificationAugment(crop_size=224, color_jitter=0.0, hflip_prob=0.0))

    def test_video_classification(self) -> None:
        _round_trip(T.VideoClassification(crop_size=256))

    def test_video_classification_high_resolution(self) -> None:
        # The 384 tag derives resize 438; the round-trip must carry the
        # derived value, not recompute it from a different rule.
        _round_trip(T.VideoClassification(crop_size=384))

    def test_detection(self) -> None:
        _round_trip(T.Detection(max_size=1333, min_area=4.0, min_visibility=0.3))

    def test_segmentation(self) -> None:
        _round_trip(T.Segmentation(crop_size=480, resize_size=520))

    def test_pose(self) -> None:
        _round_trip(T.Pose(crop_size=192, resize_size=224))


# ── registry / resolver ─────────────────────────────────────────────


class TestAutoResolver:
    def test_registered_names(self) -> None:
        names = T.AutoTransformsPreset.registered()
        assert names == [
            "Detection",
            "ImageClassification",
            "ImageClassificationAugment",
            "Pose",
            "Segmentation",
            "VideoClassification",
        ]

    def test_unknown_preset_raises_keyerror(self) -> None:
        with pytest.raises(KeyError):
            T.AutoTransformsPreset.from_dict(
                {"preprocessor_type": "DoesNotExist", "init_kwargs": {}}
            )

    def test_missing_preprocessor_type_raises(self) -> None:
        with pytest.raises(ValueError):
            T.AutoTransformsPreset.from_dict({"init_kwargs": {}})

    def test_non_dict_init_kwargs_raises(self) -> None:
        with pytest.raises(TypeError):
            T.AutoTransformsPreset.from_dict(
                {"preprocessor_type": "ImageClassification", "init_kwargs": []}
            )

    def test_concrete_class_from_dict_validates_type(self) -> None:
        # ImageClassification.from_dict on a Detection cfg must raise.
        cfg = T.Detection().to_dict()
        with pytest.raises(ValueError):
            T.ImageClassification.from_dict(cfg)


# ── multi-target threading ──────────────────────────────────────────


class TestMultiTarget:
    def test_classification_passes_lone_tensor(self) -> None:
        # ImageClassification is image-only; passing a bare tensor
        # returns a bare tensor (no dict wrapping).
        img = lucid.rand(3, 300, 400)
        out = T.ImageClassification(crop_size=224)(img)
        assert tuple(out.shape) == (3, 224, 224)  # type: ignore[union-attr]

    def test_segmentation_image_and_mask(self) -> None:
        # Mask must follow the geometric chain — same final HxW, no
        # interpolation-induced label leakage.
        lucid.manual_seed(0)
        img = lucid.rand(3, 300, 400)
        # 5-label mask
        m = lucid.floor(lucid.rand(1, 300, 400) * 5.0)
        sample = {"image": T.Image(img), "mask": T.Mask(m)}
        out = T.Segmentation(crop_size=224, resize_size=256)(sample)
        assert tuple(out["image"].data.shape) == (3, 224, 224)
        assert tuple(out["mask"].data.shape) == (1, 224, 224)
        # No synthetic labels (nearest interp only).
        before = set(int(round(v)) for v in m.numpy().reshape(-1).tolist())
        after = set(int(round(v)) for v in out["mask"].data.numpy().reshape(-1).tolist())
        assert after <= before, f"mask gained labels {after - before}"

    def test_detection_image_and_boxes(self) -> None:
        # Boxes must update canvas + drop nothing in this safe case.
        img = lucid.rand(3, 400, 600)
        boxes = T.BoundingBoxes(
            lucid.tensor([[50.0, 50.0, 250.0, 250.0], [300.0, 200.0, 500.0, 380.0]]),
            "xyxy",
            (400, 600),
            labels=lucid.tensor([1.0, 2.0]),
        )
        sample = {"image": T.Image(img), "boxes": boxes}
        out = T.Detection(max_size=512, min_area=1.0)(sample)
        # All boxes survive (both well-inside the canvas before scaling).
        assert int(out["boxes"].data.shape[0]) == 2
        assert out["boxes"].labels.numpy().tolist() == [1.0, 2.0]

    def test_detection_drops_tiny_box(self) -> None:
        # A 1-pixel box should fall below min_area=10.
        img = lucid.rand(3, 256, 256)
        boxes = T.BoundingBoxes(
            lucid.tensor([[100.0, 100.0, 101.0, 101.0]]),
            "xyxy",
            (256, 256),
            labels=lucid.tensor([7.0]),
        )
        # Make pipeline a no-op size-wise then assert the filter
        # kicks the 1-pixel box.
        out = T.Detection(max_size=256, min_area=10.0)({"image": T.Image(img), "boxes": boxes})
        assert int(out["boxes"].data.shape[0]) == 0

    def test_pose_image_and_keypoints(self) -> None:
        img = lucid.rand(3, 400, 600)
        kps = T.Keypoints(
            lucid.tensor([[100.0, 50.0, 1.0], [300.0, 200.0, 1.0]]),
            canvas_size=(400, 600),
        )
        sample = {"image": T.Image(img), "kps": kps}
        out = T.Pose(crop_size=224, resize_size=256)(sample)
        # Count + visibility column preserved.
        assert int(out["kps"].data.shape[0]) == 2
        assert int(out["kps"].data.shape[1]) == 3
        extras = out["kps"].data.numpy()[:, 2].tolist()
        assert all(abs(v - 1.0) < 1e-5 for v in extras)


class TestImageClassificationAugmentStrong:
    """Phase 8 — `ImageClassificationAugment` extended kwargs.

    Verifies the new `auto_augment` and `random_erasing` knobs:
    pipeline assembly, round-trip, runtime correctness.
    """

    def test_no_optional_stages_by_default(self) -> None:
        # The default is the baseline recipe — no AutoAugment, no RandomErasing.
        preset = T.ImageClassificationAugment(crop_size=224)
        x = lucid.rand(3, 256, 256)
        y = preset(x)
        assert tuple(y.shape) == (3, 224, 224)

    def test_auto_augment_ta_wide_runs(self) -> None:
        preset = T.ImageClassificationAugment(crop_size=64, auto_augment="ta_wide")
        y = preset(lucid.rand(3, 96, 96))
        assert tuple(y.shape) == (3, 64, 64)

    def test_auto_augment_ra_with_params(self) -> None:
        preset = T.ImageClassificationAugment(crop_size=64, auto_augment="ra-m9-n2")
        y = preset(lucid.rand(3, 96, 96))
        assert tuple(y.shape) == (3, 64, 64)

    def test_auto_augment_aa_imagenet(self) -> None:
        preset = T.ImageClassificationAugment(crop_size=64, auto_augment="aa_imagenet")
        y = preset(lucid.rand(3, 96, 96))
        assert tuple(y.shape) == (3, 64, 64)

    def test_random_erasing_changes_output(self) -> None:
        # With random_erasing=1.0 the erase fires every call.
        lucid.manual_seed(0)
        preset = T.ImageClassificationAugment(
            crop_size=64,
            random_erasing=1.0,
            color_jitter=0.0,
            hflip_prob=0.0,
        )
        y = preset(lucid.ones(3, 64, 64))
        # Some pixels should be replaced (the erase region != input).
        assert float(y.min().item()) != float(y.max().item())

    def test_unknown_auto_augment_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown auto_augment"):
            T.ImageClassificationAugment(crop_size=224, auto_augment="bogus")

    def test_invalid_random_erasing_raises(self) -> None:
        with pytest.raises(ValueError, match="random_erasing"):
            T.ImageClassificationAugment(crop_size=224, random_erasing=1.5)

    def test_round_trip_with_strong_aug(self) -> None:
        # to_dict / from_dict must preserve the new kwargs.
        preset = T.ImageClassificationAugment(
            crop_size=224,
            auto_augment="ra-m9-n2",
            random_erasing=0.25,
        )
        cfg = preset.to_dict()
        assert cfg["init_kwargs"]["auto_augment"] == "ra-m9-n2"
        assert cfg["init_kwargs"]["random_erasing"] == 0.25
        restored = T.AutoTransformsPreset.from_dict(cfg)
        assert isinstance(restored, T.ImageClassificationAugment)
        assert restored.to_dict() == cfg

    def test_round_trip_defaults_yield_neutral_kwargs(self) -> None:
        # Default kwargs (no AutoAugment, no RandomErasing) survive round-trip.
        preset = T.ImageClassificationAugment(crop_size=224)
        cfg = preset.to_dict()
        assert cfg["init_kwargs"]["auto_augment"] is None
        assert cfg["init_kwargs"]["random_erasing"] == 0.0


class TestDetectionCanvas:
    """Canvas size and placement.

    The defaults reproduce what a config saved before either option existed
    did (pad to exactly ``max_size``, image centred); the R-CNN weights opt
    in to the reference's rounding to 32 and top-left placement.
    """

    def test_an_old_config_keeps_its_canvas_and_placement(self) -> None:
        # The init_kwargs a Detection preset serialised before this change.
        old = {
            "preprocessor_type": "Detection",
            "init_kwargs": {
                "max_size": 1333,
                "min_size": None,
                "min_area": 1.0,
                "min_visibility": 0.0,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "interpolation": "bilinear",
            },
        }
        tf = T.AutoTransformsPreset.from_dict(old)
        assert isinstance(tf, T.Detection)
        assert tf.canvas_size == 1333
        assert tf.pad_position == "center"

    def test_default_pads_to_exactly_max_size(self) -> None:
        tf = T.Detection()
        assert tf.canvas_size == 1333
        out = tf(T.Image(lucid.rand(3, 32, 32)))
        assert tuple(out.data.shape) == (3, 1333, 1333)  # type: ignore[union-attr]

    @pytest.mark.parametrize(
        "max_size, canvas", [(1333, 1344), (1344, 1344), (800, 800), (1000, 1024)]
    )
    def test_size_divisible_rounds_the_canvas_up(self, max_size: int, canvas: int) -> None:
        tf = T.Detection(max_size=max_size, size_divisible=32)
        assert tf.canvas_size == canvas
        # A non-square image still lands on the square canvas.
        out = tf(T.Image(lucid.rand(3, 24, 32)))
        assert tuple(out.data.shape) == (3, canvas, canvas)  # type: ignore[union-attr]

    def test_top_left_placement_leaves_the_image_at_the_origin(self) -> None:
        from lucid.utils.transforms._datatypes import to_xyxy

        tf = T.Detection(size_divisible=32, pad_position="top_left")
        boxes = T.BoundingBoxes(
            lucid.tensor([[50.0, 50.0, 250.0, 250.0]]),
            "xyxy",
            (400, 600),
            labels=lucid.tensor([1.0]),
        )
        # A flat image stays flat through the resize, so image and padding
        # are told apart by value after normalisation.
        sample = {"image": T.Image(lucid.ones(3, 400, 600)), "boxes": boxes}
        out = tf(sample)
        h, w = tf.image_size(400, 600)
        assert (h, w) == (889, 1333)
        img = out["image"].data
        inside = (1.0 - 0.485) / 0.229
        pad = (0.0 - 0.485) / 0.229
        for (row, col), want in [
            ((0, 0), inside),
            ((h - 1, w - 1), inside),
            ((h, 0), pad),
            ((0, w), pad),
            ((1343, 1343), pad),
        ]:
            assert float(img[0, row, col].item()) == pytest.approx(want, abs=1e-4)
        # Boxes are only scaled -- each axis by the size the resize gave it,
        # which is what ``image_size`` reports -- and never shifted.
        sx, sy = w / 600, h / 400
        got = to_xyxy(out["boxes"]).numpy().reshape(-1).tolist()
        assert got == pytest.approx([50 * sx, 50 * sy, 250 * sx, 250 * sy])

    def test_image_size_follows_the_resize_rule(self) -> None:
        # Shortest side to 800 unless the longest would pass 1333.
        tf = T.Detection(min_size=800, max_size=1333)
        assert tf.image_size(480, 640) == (800, 1067)
        assert tf.image_size(300, 900) == (444, 1333)

    def test_options_are_serialised(self) -> None:
        tf = T.Detection(size_divisible=32, pad_position="top_left")
        kwargs = tf.to_dict()["init_kwargs"]
        assert kwargs["size_divisible"] == 32
        assert kwargs["pad_position"] == "top_left"
        _round_trip(tf)

    def test_bad_options_are_refused(self) -> None:
        with pytest.raises(ValueError, match="size_divisible"):
            T.Detection(size_divisible=0)
        with pytest.raises(ValueError, match="position"):
            T.Detection(pad_position="middle")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_the_padding_reaches_the_model_at_pad_value() -> None:
    img = T.Image(lucid.rand(3, 16, 32))
    for pad_value in (0.0, 0.5):
        tf = T.Detection(max_size=64, pad_value=pad_value, pad_position="top_left")
        out = tf(img).data
        # The resized image fills rows [0, 32); rows [32, 64) are padding.
        assert float((out[:, 32:, :] - pad_value).abs().max().item()) < 1e-6


@pytest.mark.parametrize(
    ("mean", "std"),
    [((0.5, 0.5), None), ((0.5, 0.5, 0.5), (0.25, 0.25))],
    ids=["2-3", "3-2"],
)
def test_a_pad_value_with_unpaired_mean_and_std_names_both(
    mean: tuple[float, ...], std: tuple[float, ...] | None
) -> None:
    """The fill is one value per channel, so incompatible ``mean`` and ``std`` must raise.

    They were zipped to build it, which named neither once ``zip`` turned
    strict — and before that dropped the unpaired channels from the fill.
    """
    with pytest.raises(ValueError, match="mean and std .* got (2 and 3|3 and 2)"):
        T.Detection(max_size=64, pad_value=0.0, mean=mean, std=std)


@pytest.mark.parametrize(
    ("mean", "std"),
    [((0.5,), (0.25,)), ((0.5, 0.25, 0.75), (0.5,)), ((0.5,), (0.25, 0.5, 0.125))],
    ids=["1-1", "3-1", "1-3"],
)
def test_broadcastable_mean_and_std_normalize_and_pad_channels(
    mean: tuple[float, ...], std: tuple[float, ...]
) -> None:
    img = lucid.tensor([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]])
    norm = T.Normalize(mean=mean, std=std, max_pixel_value=1.0)
    out3 = norm(T.Image(img)).data
    out4 = norm(T.Image(img.unsqueeze(0))).data
    m_exp = mean * 3 if len(mean) == 1 else mean
    s_exp = std * 3 if len(std) == 1 else std
    for c in range(3):
        expected = (img[c] - m_exp[c]) / s_exp[c]
        assert lucid.allclose(out3[c], expected)
        assert lucid.allclose(out4[0, c], expected)
    det = T.Detection(max_size=64, pad_value=0.25, pad_position="top_left", mean=mean, std=std)
    padded = det(T.Image(lucid.rand(3, 16, 32))).data
    assert float((padded[:, 32:, :] - 0.25).abs().max().item()) < 1e-6


def test_without_pad_value_the_padding_is_a_normalised_black_pixel() -> None:
    # What a config saved before pad_value existed produced, unchanged.
    tf = T.Detection(max_size=64, pad_position="top_left")
    out = tf(T.Image(lucid.rand(3, 16, 32))).data
    expect = [(0.0 - m) / s for m, s in zip(tf.mean, tf.std)]
    assert [float(out[c, -1, 0].item()) for c in range(3)] == pytest.approx(expect, abs=1e-6)


def test_to_image_boxes_undoes_the_letterbox_and_the_resize() -> None:
    letterbox = T.Detection(max_size=64)
    assert letterbox.image_size(16, 32) == (32, 64)
    on_canvas = lucid.tensor([[0.0, 16.0, 64.0, 48.0]])
    assert letterbox.to_image_boxes(on_canvas, 16, 32).tolist() == [[0.0, 0.0, 32.0, 16.0]]
    top_left = T.Detection(max_size=64, pad_position="top_left")
    on_canvas = lucid.tensor([[0.0, 0.0, 64.0, 32.0]])
    assert top_left.to_image_boxes(on_canvas, 16, 32).tolist() == [[0.0, 0.0, 32.0, 16.0]]


def test_a_per_axis_canvas_rounds_each_side_up_to_the_divisor() -> None:
    # The reference gives a single image a canvas each side of which is the
    # next multiple of 32; the square canvas pads the short side to the long.
    img = T.Image(lucid.rand(3, 20, 30))
    square = T.Detection(max_size=100, size_divisible=32, pad_position="top_left")
    fitted = T.Detection(max_size=100, size_divisible=32, pad_position="top_left", square=False)
    assert tuple(square(img).data.shape) == (3, 128, 128)
    assert tuple(fitted(img).data.shape) == (3, 96, 128)


def test_pad_if_needed_pads_up_to_a_divisor() -> None:
    tf = T.PadIfNeeded(None, None, pad_height_divisor=32, pad_width_divisor=16)
    assert tuple(tf(T.Image(lucid.rand(3, 33, 16))).data.shape) == (3, 64, 16)
