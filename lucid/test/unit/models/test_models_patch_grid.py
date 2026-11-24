"""A fixed-grid patch model takes exactly the image size it was built for (LCD-1107).

ViT, I-JEPA and CLIP's image tower hold one position entry per patch of the
grid they were built on.  Before the shared ``check_patch_grid`` they failed
in two unrelated ways:

* a size from ``N`` to ``N + patch - 1`` answered as its ``N x N`` crop — the
  stride-``patch`` convolution dropped the border, the position table still
  fit, and nothing complained (ViT's own check divided with ``//``, I-JEPA
  and CLIP had none);
* any other size broadcast against the table and failed as an engine
  ``ShapeMismatch`` that named neither size (I-JEPA, CLIP).

The reference ViT, CLIP and I-JEPA embeddings refuse every size but the
configured one unless the caller asks for position interpolation; so does
this zoo.  The sweep at the bottom holds every family with a patch
size to a stated answer for the same question, so a family added later
cannot skip it.
"""

import dataclasses
from collections.abc import Callable

import pytest

import lucid
import lucid.models as M
from lucid._C import engine as _C_engine
from lucid.models._base import ModelConfig
from lucid.models._utils._classification import check_patch_grid

_BUILT = 32
_PATCH = 16

_VIT = dict(
    image_size=_BUILT, patch_size=_PATCH, dim=32, depth=1, num_heads=2, num_classes=3
)
_IJEPA = dict(
    image_size=_BUILT,
    patch_size=_PATCH,
    dim=32,
    depth=1,
    num_heads=2,
    predictor_dim=16,
    predictor_depth=1,
    num_classes=3,
    min_keep=1,
)
_CLIP = dict(
    image_size=_BUILT,
    patch_size=_PATCH,
    vision_layers=1,
    vision_width=32,
    vision_heads=2,
    text_width=16,
    text_heads=2,
    text_layers=1,
    context_length=8,
    vocab_size=64,
    embed_dim=8,
)


def _run_vit(model: M.PretrainedModel, images: lucid.Tensor) -> lucid.Tensor:
    return model(images).logits


def _run_ijepa_probe(model: M.PretrainedModel, images: lucid.Tensor) -> lucid.Tensor:
    return model(images).logits


def _run_ijepa_encode(model: M.PretrainedModel, images: lucid.Tensor) -> lucid.Tensor:
    return model.encode(images)


def _run_clip_encode(model: M.PretrainedModel, images: lucid.Tensor) -> lucid.Tensor:
    return model.encode_image(images)


def _run_clip_forward(model: M.PretrainedModel, images: lucid.Tensor) -> lucid.Tensor:
    captions = lucid.zeros(int(images.shape[0]), 8, dtype=lucid.int64)
    return model(images, captions).logits_per_image


#: ``(id, factory, overrides, entry point)`` — every public way into the tower.
_ENTRY_POINTS: list[
    tuple[str, str, dict[str, object], Callable[[M.PretrainedModel, lucid.Tensor], lucid.Tensor]]
] = [
    ("vit", "vit_base_16", _VIT, lambda m, x: m(x).last_hidden_state),
    ("vit_cls", "vit_base_16_cls", _VIT, _run_vit),
    ("ijepa_encode", "ijepa_base_16", _IJEPA, _run_ijepa_encode),
    ("ijepa_probe", "ijepa_base_16_cls", _IJEPA, _run_ijepa_probe),
    ("clip_encode_image", "clip_vit_base_32", _CLIP, _run_clip_encode),
    ("clip_forward", "clip_vit_base_32", _CLIP, _run_clip_forward),
]
_IDS = [row[0] for row in _ENTRY_POINTS]

#: Sides that used to answer as the 32x32 crop (``N .. N + patch - 1``).
_BORDER_DROPPED = (33, 40, 47)
#: A size one short of the build, whose last patch is partial.
_PARTIAL_BELOW = (31,)
#: Whole numbers of patches on another grid — the old engine ``ShapeMismatch``.
_OTHER_GRID = (16, 48, 64)


def _build(factory: str, overrides: dict[str, object]) -> M.PretrainedModel:
    lucid.manual_seed(0)
    return M.create_model(factory, **overrides).eval()


def _refusal(run: Callable[..., lucid.Tensor], model: M.PretrainedModel, h: int, w: int) -> str:
    with lucid.no_grad(), pytest.raises(ValueError) as caught:
        run(model, lucid.rand(2, 3, h, w))
    assert not isinstance(caught.value, _C_engine.LucidError), (
        f"{h}x{w} reached the engine instead of the grid check: {caught.value}"
    )
    return str(caught.value)


class TestCheckPatchGrid:
    """The shared check itself, without a model."""

    def test_the_built_size_passes(self) -> None:
        check_patch_grid("ViT", 32, 32, image_size=32, patch_size=16)

    @pytest.mark.parametrize("side", [33, 40, 47])
    def test_a_border_the_patch_would_drop_is_named(self, side: int) -> None:
        with pytest.raises(ValueError) as caught:
            check_patch_grid("ViT", side, side, image_size=32, patch_size=16)
        message = str(caught.value)
        assert "ViT is built for 32x32 inputs" in message
        assert f"Got {side}x{side} pixels" in message
        assert "not a multiple of the 16px patch" in message
        assert f"drop {side % 16} rows and {side % 16} columns" in message
        assert "image_size=" not in message  # the config refuses an off-grid build

    def test_another_whole_grid_names_both_grids(self) -> None:
        with pytest.raises(ValueError) as caught:
            check_patch_grid("CLIP", 64, 64, image_size=32, patch_size=16)
        message = str(caught.value)
        assert "CLIP is built for 32x32 inputs" in message
        assert "a 2x2 grid of 16px patches" in message
        assert "Got 64x64 pixels, which is 4x4 patches" in message
        assert "or build the model with image_size=64" in message

    def test_a_rectangle_is_refused_even_on_a_whole_grid(self) -> None:
        with pytest.raises(ValueError) as caught:
            check_patch_grid("I-JEPA", 32, 64, image_size=32, patch_size=16)
        message = str(caught.value)
        assert "Got 32x64 pixels, which is 2x4 patches" in message
        assert "image_size=" not in message

    def test_the_same_patch_count_on_another_layout_is_refused(self) -> None:
        # 2x8 patches is the 4x4 build's 16: the table fit and answered wrong.
        with pytest.raises(ValueError, match="Got 16x64 pixels"):
            check_patch_grid("ViT", 16, 64, image_size=32, patch_size=8)

    def test_one_off_side_of_a_rectangle_counts_the_other_as_zero(self) -> None:
        with pytest.raises(ValueError, match="drop 0 rows and 7 columns"):
            check_patch_grid("ViT", 32, 39, image_size=32, patch_size=8)


@pytest.mark.parametrize(("_id", "factory", "overrides", "run"), _ENTRY_POINTS, ids=_IDS)
class TestFixedGridRefusesOtherSizes:
    """Every public entry into ViT, I-JEPA and CLIP's tower, at sizes that used to slip by."""

    def test_the_built_size_answers(
        self,
        _id: str,
        factory: str,
        overrides: dict[str, object],
        run: Callable[..., lucid.Tensor],
    ) -> None:
        model = _build(factory, overrides)
        with lucid.no_grad():
            out = run(model, lucid.rand(2, 3, _BUILT, _BUILT))
        assert int(out.shape[0]) == 2

    @pytest.mark.parametrize("side", _BORDER_DROPPED + _PARTIAL_BELOW)
    def test_an_off_grid_side_is_refused_not_cropped(
        self,
        _id: str,
        factory: str,
        overrides: dict[str, object],
        run: Callable[..., lucid.Tensor],
        side: int,
    ) -> None:
        message = _refusal(run, _build(factory, overrides), side, side)
        assert f"built for {_BUILT}x{_BUILT}" in message
        assert f"Got {side}x{side} pixels" in message
        assert f"not a multiple of the {_PATCH}px patch" in message

    @pytest.mark.parametrize("side", _OTHER_GRID)
    def test_another_whole_grid_is_refused_with_both_sizes(
        self,
        _id: str,
        factory: str,
        overrides: dict[str, object],
        run: Callable[..., lucid.Tensor],
        side: int,
    ) -> None:
        message = _refusal(run, _build(factory, overrides), side, side)
        assert f"built for {_BUILT}x{_BUILT}" in message
        assert f"Got {side}x{side} pixels, which is {side // _PATCH}x{side // _PATCH}" in message

    @pytest.mark.parametrize("shape", [(32, 48), (48, 32), (32, 47), (47, 32)])
    def test_a_rectangle_is_refused(
        self,
        _id: str,
        factory: str,
        overrides: dict[str, object],
        run: Callable[..., lucid.Tensor],
        shape: tuple[int, int],
    ) -> None:
        message = _refusal(run, _build(factory, overrides), *shape)
        assert f"Got {shape[0]}x{shape[1]} pixels" in message


def test_an_off_grid_input_no_longer_equals_its_crop() -> None:
    """The defect itself: 47x47 returned the logits of its 32x32 crop."""
    model = _build("vit_base_16_cls", _VIT)
    image = lucid.rand(1, 3, 47, 47)
    with lucid.no_grad():
        crop = model(image[:, :, :_BUILT, :_BUILT]).logits
        with pytest.raises(ValueError, match="not a multiple of the 16px patch"):
            model(image)
    assert tuple(crop.shape) == (1, 3)


def test_a_training_step_runs_at_the_built_size_and_not_off_grid() -> None:
    """Forward, backward and an optimiser step still work; the refusal is before the graph."""
    model = _build("vit_base_16_cls", _VIT).train()
    opt = lucid.optim.SGD(model.parameters(), lr=0.1)
    labels = lucid.zeros(2, dtype=lucid.int64)
    with pytest.raises(ValueError, match="built for 32x32"):
        model(lucid.rand(2, 3, 47, 47), labels=labels)
    loss = model(lucid.rand(2, 3, _BUILT, _BUILT), labels=labels).loss
    assert loss is not None
    loss.backward()
    opt.step()


# ── the sweep: every config with a patch size has a stated answer ────────────

#: Refused by ``check_patch_grid`` at the patch embedding (this file).
_REFUSED_HERE = {"ViTConfig", "IJEPAConfig", "CLIPConfig"}

#: Refuse every size but the built one in the model, with their own message.
_OWN_EXACT_CHECK = {
    "VJEPAConfig",  # ``_check_clip``: frames, channels and the square side
    "VJEPA2Config",
    "VJEPA2ACConfig",
    "GenieConfig",  # ``_check_video``
    "DiTConfig",  # ``check_latent_size``
    "MeanFlowConfig",
    "TVMConfig",
    "DriftingConfig",
}

#: Accept another size on purpose.  CrossViT resizes both branches to the
#: sizes its tables were built for; Swin has no position table, its windows
#: and its patch merging pad.
_VARIABLE_RESOLUTION = {"CrossViTConfig", "SwinConfig"}


def _patch_config_classes() -> set[str]:
    found: set[str] = set()
    stack: list[type[ModelConfig]] = [ModelConfig]
    while stack:
        for sub in stack.pop().__subclasses__():
            stack.append(sub)
            if not sub.__module__.startswith("lucid.models.") or sub.model_type == "base":
                continue
            if any("patch_size" in f.name for f in dataclasses.fields(sub)):
                found.add(sub.__name__)
    return found


def test_every_config_with_a_patch_size_has_a_stated_answer() -> None:
    """A new patch-embedding family has to say what it does with another size."""
    stated = _REFUSED_HERE | _OWN_EXACT_CHECK | _VARIABLE_RESOLUTION
    found = _patch_config_classes()
    assert found == stated, (
        f"unclassified: {sorted(found - stated)}; stale: {sorted(stated - found)} — "
        "a fixed-grid model calls check_patch_grid, one that resizes or pads says so here"
    )


@pytest.mark.xfail(
    strict=True,
    reason="LCD-1115: Swin stem drops a non-multiple border; the reference pads",
)
def test_swin_stem_pads_a_non_multiple_like_the_reference() -> None:
    lucid.manual_seed(0)
    model = M.create_model(
        "swin_tiny_cls",
        image_size=32,
        patch_size=4,
        embed_dim=16,
        depths=(1, 1, 1, 1),
        num_heads=(1, 2, 2, 2),
        window_size=2,
        num_classes=3,
    ).eval()
    image = lucid.rand(1, 3, 35, 35)
    padded = lucid.zeros(1, 3, 36, 36)
    padded[:, :, :35, :35] = image
    with lucid.no_grad():
        got, want = model(image).logits, model(padded).logits
    assert float((got - want).abs().max().item()) < 1e-5
