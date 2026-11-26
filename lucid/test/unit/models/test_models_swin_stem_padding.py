"""The Swin stems pad an image to a whole number of patches (LCD-1115).

A patch embedding is a stride-``patch_size`` convolution, which drops a border
that does not fill a last patch.  The reference zero-pads the right and bottom
first, so a 35-pixel side under 4-pixel patches is a 9-patch side, not an
8-patch one.  The comparison against the reference's own numbers is in
``lucid/test/parity/models/vision/test_swin_stem_padding.py``; what is pinned
here needs no reference: the padded grid, that padding is exactly zeros on the
bottom and right, and that both Swin stems (the classifier trunk and the
Mask2Former backbone) agree, since they share the rule.
"""

from typing import Any

import pytest

import lucid
import lucid.models as M
import lucid.optim as optim
from lucid.models.vision._common._padding import pad_to_patch_multiple
from lucid.models.vision.mask2former._model import _SwinPatchEmbeddings
from lucid.models.vision.swin._model import _PatchEmbed

_PATCH = 4
_DIM = 8

_SWIN_SIZES: dict[str, Any] = {
    "image_size": 32,
    "patch_size": _PATCH,
    "embed_dim": 16,
    "depths": (1, 1, 1, 1),
    "num_heads": (1, 2, 2, 2),
    "window_size": 2,
    "num_classes": 3,
}


def _ceil_div(side: int, patch: int) -> int:
    return -(-side // patch)


def _zero_padded(image: lucid.Tensor, height: int, width: int) -> lucid.Tensor:
    batch, channels = int(image.shape[0]), int(image.shape[1])
    padded = lucid.zeros(batch, channels, height, width)
    padded[:, :, : int(image.shape[2]), : int(image.shape[3])] = image
    return padded


def _swin_stem() -> _PatchEmbed:
    return _PatchEmbed(3, _PATCH, _DIM)


def _mask2former_stem() -> _SwinPatchEmbeddings:
    return _SwinPatchEmbeddings(3, _DIM, _PATCH)


def _grid(stem: Any, image: lucid.Tensor) -> tuple[int, int]:
    """``(rows, cols)`` of patches the stem cut ``image`` into."""
    out = stem(image)
    if isinstance(out, tuple):  # Mask2Former hands back (tokens, (rows, cols))
        rows, cols = out[1]
        return int(rows), int(cols)
    return int(out.shape[1]), int(out.shape[2])  # Swin's (B, rows, cols, C)


class TestPadToPatchMultiple:
    @pytest.mark.parametrize(
        ("height", "width", "padded"),
        [
            (35, 35, (36, 36)),
            (33, 37, (36, 40)),
            (36, 36, (36, 36)),
            (1, 4, (4, 4)),
            (5, 7, (8, 8)),
        ],
    )
    def test_pads_up_to_the_next_multiple(
        self, height: int, width: int, padded: tuple[int, int]
    ) -> None:
        image = lucid.randn(2, 3, height, width)
        out = pad_to_patch_multiple(image, _PATCH)
        assert tuple(int(d) for d in out.shape) == (2, 3, *padded)

    def test_the_border_is_zeros_on_the_bottom_and_right_only(self) -> None:
        image = lucid.rand(1, 2, 5, 6) + 1.0  # strictly positive, so a zero is a pad
        out = pad_to_patch_multiple(image, _PATCH)
        assert float((out[:, :, :5, :6] - image).abs().max().item()) == 0.0
        assert float(out[:, :, 5:, :].abs().max().item()) == 0.0
        assert float(out[:, :, :, 6:].abs().max().item()) == 0.0

    def test_an_input_on_the_grid_is_returned_untouched(self) -> None:
        image = lucid.randn(1, 3, 8, 12)
        assert pad_to_patch_multiple(image, _PATCH) is image

    def test_keeps_dtype(self) -> None:
        image = lucid.randn(1, 3, 5, 5).to(lucid.float16)
        assert pad_to_patch_multiple(image, _PATCH).dtype == lucid.float16


class TestStemsShareTheRule:
    """Class-wide guard: every Swin stem cuts a ragged image into ``ceil(side / patch)``."""

    @pytest.mark.parametrize("make", [_swin_stem, _mask2former_stem], ids=["swin", "mask2former"])
    @pytest.mark.parametrize("shape", [(35, 35), (33, 37), (36, 36), (30, 41), (5, 5)])
    def test_the_grid_is_the_ceiling(self, make: Any, shape: tuple[int, int]) -> None:
        stem = make().eval()
        image = lucid.randn(1, 3, *shape)
        assert _grid(stem, image) == (_ceil_div(shape[0], _PATCH), _ceil_div(shape[1], _PATCH))

    @pytest.mark.parametrize("make", [_swin_stem, _mask2former_stem], ids=["swin", "mask2former"])
    def test_a_ragged_image_equals_its_zero_padded_one(self, make: Any) -> None:
        stem = make().eval()
        image = lucid.randn(1, 3, 35, 35)
        got = stem(image)
        want = stem(_zero_padded(image, 36, 36))
        got_t, want_t = (got[0], want[0]) if isinstance(got, tuple) else (got, want)
        assert float((got_t - want_t).abs().max().item()) == 0.0


class TestSwinEndToEnd:
    def _model(self) -> Any:
        lucid.manual_seed(0)
        return M.create_model("swin_tiny_cls", **_SWIN_SIZES).eval()

    @pytest.mark.parametrize("shape", [(35, 35), (33, 37)])
    def test_a_ragged_image_equals_its_zero_padded_one(self, shape: tuple[int, int]) -> None:
        model = self._model()
        image = lucid.rand(1, 3, *shape)
        padded = _zero_padded(
            image, _ceil_div(shape[0], _PATCH) * _PATCH, _ceil_div(shape[1], _PATCH) * _PATCH
        )
        with lucid.no_grad():
            got, want = model(image).logits, model(padded).logits
        assert float((got - want).abs().max().item()) < 1e-5

    def test_the_border_reaches_the_answer(self) -> None:
        # Cropping the three dropped columns gave the same logits before the fix.
        model = self._model()
        image = lucid.rand(1, 3, 35, 35)
        with lucid.no_grad():
            full, cropped = model(image).logits, model(image[:, :, :32, :32]).logits
        assert float((full - cropped).abs().max().item()) > 1e-6

    def test_the_backbone_pools_a_ragged_image(self) -> None:
        lucid.manual_seed(0)
        backbone = M.create_model("swin_tiny", **{**_SWIN_SIZES, "num_classes": 1000}).eval()
        image = lucid.rand(1, 3, 35, 35)
        padded = _zero_padded(image, 36, 36)
        with lucid.no_grad():
            got, want = backbone.forward_features(image), backbone.forward_features(padded)
        assert float((got - want).abs().max().item()) < 1e-5

    def test_it_trains_on_a_ragged_image(self) -> None:
        # The stem is the first layer, so its weight only gets a gradient if the
        # padded input is on the autograd path.
        lucid.manual_seed(0)
        model = M.create_model("swin_tiny_cls", **_SWIN_SIZES).train()
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        before = model.patch_embed.proj.weight.numpy().copy()
        images = lucid.randn(2, 3, 35, 35)
        out = model(images, labels=lucid.tensor([0, 2]))
        assert out.loss is not None
        assert bool(lucid.isfinite(out.loss).item())
        out.loss.backward()
        grad = model.patch_embed.proj.weight.grad
        assert grad is not None and bool(lucid.isfinite(grad).all().item())
        optimizer.step()
        assert float(abs(model.patch_embed.proj.weight.numpy() - before).max()) > 0.0
