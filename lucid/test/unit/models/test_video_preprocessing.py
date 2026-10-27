"""What the video preset does, and which convention it takes.

The V-JEPA families ship a preprocessing preset rather than the no-op
they carried at first: without one a caller holding a decoded clip has
no way to reach the numbers the published weights expect.

Compared against the released ``VJEPA2VideoProcessor``, a float clip
through this preset agrees to ``1.1e-4`` — the pipeline is the same
computation. Two things had to be right for that:

* the resize target truncates. ``int(256 * 256 / 224)`` is 292 and
  ``round`` gives 293, one pixel that moves the crop window and
  rescales everything inside it;
* the centre crop places an **odd** margin *down*, not to nearest.
  A 292x519 frame cropped to 256 starts at column 131 under Hugging
  Face's fast processors and 132 under the rounding that the reference
  framework's vision package and Albumentations use. These weights are published through the
  former. A square frame cannot see the difference — its margin is
  even — which is why the first version of this file missed it.

Handing the preset ``uint8`` and the reference the same leaves a
uniform ``1.74e-2``, which is ``1/255`` divided by the blue channel's
std: the reference rounds back to ``uint8`` between stages and Lucid
stays in float. That is Lucid computing the same thing more precisely,
so it is recorded rather than matched.
"""

import numpy as np
import pytest

import lucid
import lucid.utils.transforms as T
from lucid.models.generative.vjepa2_ac import VJEPA2ACWeights
from lucid.models.vision.vjepa2 import (
    VJEPA2ViTGiant384Weights,
    VJEPA2ViTLargeWeights,
)


class TestReleasedGeometry:
    """Numbers taken from the released processor's own config."""

    @pytest.mark.parametrize(("crop", "resize"), [(256, 292), (384, 438)])
    def test_the_resize_target_truncates(self, crop: int, resize: int) -> None:
        """``int(256 * 256 / 224)`` is 292; rounding 292.57 gives 293."""
        assert T.VideoClassification(crop_size=crop).resize_size == resize

    @staticmethod
    def _ramp(height: int, width: int) -> lucid.Tensor:
        """A clip whose pixels encode ``row * 1000 + col``.

        Reading a corner back says which window a crop took, which beats
        reasoning about offsets.
        """
        rows = np.arange(height)[None, :, None] * 1000
        cols = np.arange(width)[None, None, :]
        frame = (rows + cols).astype(np.float32)
        clip = np.ascontiguousarray(
            np.broadcast_to(frame, (3, height, width))[None].repeat(2, axis=0)
        )
        return lucid.from_numpy(clip)

    @staticmethod
    def _window(cropped: lucid.Tensor) -> tuple[int, int, int, int]:
        first = int(cropped[0, 0, 0, 0].item())
        last = int(cropped[0, 0, -1, -1].item())
        return first // 1000, last // 1000, first % 1000, last % 1000

    def test_an_even_margin_places_the_same_either_way(self) -> None:
        """292 -> 256 leaves 36, so both conventions start at 18.

        This is the case that cannot tell them apart — pinned so the
        next reader knows a green square-frame test proves nothing about
        the placement rule.
        """
        clip = self._ramp(292, 292)
        for offset in ("round", "floor"):
            window = self._window(T.CenterCrop(256, 256, offset=offset)(T.Image(clip)).data)
            assert window == (18, 273, 18, 273)

    def test_an_odd_margin_is_placed_down(self) -> None:
        """519 -> 256 leaves 263; the release starts at 131, not 132."""
        clip = self._ramp(292, 519)
        floored = self._window(T.CenterCrop(256, 256, offset="floor")(T.Image(clip)).data)
        rounded = self._window(T.CenterCrop(256, 256, offset="round")(T.Image(clip)).data)
        assert floored == (18, 273, 131, 386)
        assert rounded == (18, 273, 132, 387)

    def test_the_preset_takes_the_released_placement(self) -> None:
        """The weights come from Hugging Face, so the preset floors.

        Fed a clip already at the resize target the resize is a no-op,
        which leaves the crop as the only thing that can move a pixel.
        The preset normalises on the way out, so the ramp is read back
        through the inverse.
        """
        preset = T.VideoClassification(crop_size=256)
        out = preset(T.Image(self._ramp(292, 519))).data
        first = out[0, 0, 0, 0].item() * preset.std[0] + preset.mean[0]
        assert round(first) % 1000 == 131

    def test_the_statistics_are_imagenet_s(self) -> None:
        preset = T.VideoClassification(crop_size=256)
        assert preset.mean == (0.485, 0.456, 0.406)
        assert preset.std == (0.229, 0.224, 0.225)

    def test_a_constant_frame_normalises_by_hand(self) -> None:
        """Isolates the arithmetic from anything spatial."""
        preset = T.VideoClassification(crop_size=64)
        clip = lucid.ones(2, 3, 100, 100) * (128.0 / 255.0)
        out = preset(T.Image(clip)).data.numpy()
        wanted = (128 / 255 - np.array(preset.mean)) / np.array(preset.std)
        for channel in range(3):
            assert np.abs(out[:, channel] - wanted[channel]).max() < 1e-5


class TestClipContract:
    """A clip in, a clip out, and nothing done across time."""

    def test_the_frame_count_survives(self) -> None:
        preset = T.VideoClassification(crop_size=64)
        out = preset(T.Image(lucid.rand(7, 3, 90, 120))).data
        assert tuple(out.shape) == (7, 3, 64, 64)

    def test_each_frame_is_handled_alone(self) -> None:
        """No temporal mixing: one frame's pixels cannot reach another."""
        preset = T.VideoClassification(crop_size=64)
        clip = lucid.rand(3, 3, 90, 120)
        together = preset(T.Image(clip)).data.numpy()
        alone = preset(T.Image(clip[1:2])).data.numpy()
        assert np.abs(together[1] - alone[0]).max() == 0.0

    def test_a_non_square_frame_keeps_its_aspect(self) -> None:
        """Shortest side to the target, then the square crop."""
        preset = T.VideoClassification(crop_size=256)
        out = preset(T.Image(lucid.rand(2, 3, 180, 320))).data
        assert tuple(out.shape) == (2, 3, 256, 256)

    def test_the_preset_round_trips_through_its_config(self) -> None:
        preset = T.VideoClassification(crop_size=384)
        back = T.AutoTransformsPreset.from_dict(preset.to_dict())
        assert isinstance(back, T.VideoClassification)
        assert back.crop_size == 384 and back.resize_size == 438


class TestWeightsCarryIt:
    """A published tag has to hand the caller its preprocessing."""

    @pytest.mark.parametrize(
        ("entry", "crop"),
        [
            (VJEPA2ViTLargeWeights.DEFAULT, 256),
            (VJEPA2ViTGiant384Weights.DEFAULT, 384),
            (VJEPA2ACWeights.DEFAULT, 256),
        ],
    )
    def test_the_tag_ships_the_released_preset(self, entry: object, crop: int) -> None:
        preset = getattr(entry, "entry").transforms
        assert isinstance(preset, T.VideoClassification), (
            "the tag ships no video preprocessing — a caller with a decoded "
            "clip cannot reach the numbers these weights expect"
        )
        assert preset.crop_size == crop

    def test_the_preset_agrees_with_the_recorded_metadata(self) -> None:
        """``meta`` and the preset must not describe different pipelines."""
        for (entry,) in (
            (VJEPA2ViTLargeWeights.DEFAULT,),
            (VJEPA2ViTGiant384Weights.DEFAULT,),
            (VJEPA2ACWeights.DEFAULT,),
        ):
            record = getattr(entry, "entry")
            preprocessing = record.meta["preprocessing"]
            assert isinstance(preprocessing, dict)
            assert record.transforms.crop_size == preprocessing["size"]
            assert list(record.transforms.mean) == preprocessing["mean"]
            assert list(record.transforms.std) == preprocessing["std"]
