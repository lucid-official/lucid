"""Unit tests for the ``lucid.weights`` pretrained-weight system.

Covers the runtime surface that does not require network access:
  * WeightsEnum structure + DEFAULT aliasing + member accessors
  * Discovery: list_pretrained / get_weight
  * resolve_weights selection logic (bool / str / enum / errors)
  * ImageClassification transform shape + normalization behavior
  * Factory wiring: resnet_18_cls(pretrained=...) selects/loads correctly
    (random path; the download path is exercised by the parity suite once
    real checkpoints are uploaded)
"""

import pytest

import lucid
import lucid.weights as W
from lucid.utils.transforms import ImageClassification
from lucid.weights import WeightEntry, WeightsEnum
from lucid.models.vision.resnet import ResNet18Weights

# ── WeightsEnum structure ───────────────────────────────────────────


class TestWeightsEnum:
    def test_default_aliases_canonical(self) -> None:
        assert ResNet18Weights.DEFAULT is ResNet18Weights.IMAGENET1K_V1

    def test_member_accessors(self) -> None:
        w = ResNet18Weights.IMAGENET1K_V1
        assert w.tag == "IMAGENET1K_V1"
        assert w.num_classes == 1000
        assert w.url.endswith("/IMAGENET1K_V1/model.safetensors")
        assert isinstance(w.entry, WeightEntry)

    def test_meta_present(self) -> None:
        meta = ResNet18Weights.IMAGENET1K_V1.meta
        assert meta["source"].startswith("reference_vision/")
        assert meta["license"] == "bsd-3-clause"
        assert meta["metrics"]["ImageNet-1k"]["acc@1"] == pytest.approx(69.758)

    def test_transforms_returns_callable(self) -> None:
        tf = ResNet18Weights.IMAGENET1K_V1.transforms()
        assert isinstance(tf, ImageClassification)

    def test_iteration_skips_default_alias(self) -> None:
        # Enum iteration yields canonical members only.
        names = [m.name for m in ResNet18Weights]
        assert "IMAGENET1K_V1" in names
        assert "DEFAULT" not in names


# ── Discovery ───────────────────────────────────────────────────────


class TestDiscovery:
    def test_list_pretrained(self) -> None:
        # These are 1000-class ImageNet weights including the fc layer, so
        # they belong to the classifier factory -- the key every other
        # classification enum in the zoo registers under.
        assert W.list_pretrained("resnet_18_cls") == ["IMAGENET1K_V1"]
        assert W.list_pretrained("resnet_18") == []

    def test_list_pretrained_unknown(self) -> None:
        assert W.list_pretrained("does_not_exist") == []

    def test_get_weight(self) -> None:
        w = W.get_weight("ResNet18Weights.IMAGENET1K_V1")
        assert w is ResNet18Weights.IMAGENET1K_V1

    def test_get_weight_default_alias(self) -> None:
        w = W.get_weight("ResNet18Weights.DEFAULT")
        assert w is ResNet18Weights.IMAGENET1K_V1

    def test_get_weight_bad_format(self) -> None:
        with pytest.raises(ValueError, match="EnumName.TAG"):
            W.get_weight("ResNet18Weights")

    def test_get_weight_unknown_enum(self) -> None:
        with pytest.raises(ValueError, match="unknown weights enum"):
            W.get_weight("NopeWeights.TAG")

    def test_get_weight_unknown_tag(self) -> None:
        with pytest.raises(ValueError, match="no tag"):
            W.get_weight("ResNet18Weights.IMAGENET1K_V99")


# ── resolve_weights ─────────────────────────────────────────────────


class TestResolveWeights:
    def test_false_returns_none(self) -> None:
        assert W.resolve_weights(ResNet18Weights, False, None) is None

    def test_true_returns_default(self) -> None:
        assert W.resolve_weights(ResNet18Weights, True, None) is ResNet18Weights.IMAGENET1K_V1

    def test_string_tag(self) -> None:
        assert (
            W.resolve_weights(ResNet18Weights, "IMAGENET1K_V1", None)
            is ResNet18Weights.IMAGENET1K_V1
        )

    def test_explicit_weights_wins(self) -> None:
        # weights= takes precedence over pretrained.
        out = W.resolve_weights(ResNet18Weights, False, ResNet18Weights.IMAGENET1K_V1)
        assert out is ResNet18Weights.IMAGENET1K_V1

    def test_unknown_string_tag_raises(self) -> None:
        with pytest.raises(ValueError, match="no tag"):
            W.resolve_weights(ResNet18Weights, "NOPE", None)

    def test_wrong_enum_member_raises(self) -> None:
        class OtherWeights(WeightsEnum):
            X = WeightEntry(
                url="http://x",
                sha256="",
                num_classes=1,
                transforms=ImageClassification(crop_size=1),
            )

        with pytest.raises(TypeError, match="not a member"):
            W.resolve_weights(ResNet18Weights, False, OtherWeights.X)

    def test_bad_pretrained_type_raises(self) -> None:
        with pytest.raises(TypeError, match="bool or str"):
            W.resolve_weights(ResNet18Weights, 3.5, None)  # type: ignore[arg-type]


# ── ImageClassification transform ───────────────────────────────────


class TestImageClassification:
    def test_unbatched_shape(self) -> None:
        tf = ImageClassification(crop_size=224, resize_size=256)
        out = tf(lucid.rand(3, 300, 400))
        assert tuple(out.shape) == (3, 224, 224)

    def test_batched_shape(self) -> None:
        tf = ImageClassification(crop_size=224, resize_size=256)
        out = tf(lucid.rand(2, 3, 300, 400))
        assert tuple(out.shape) == (2, 3, 224, 224)

    def test_normalization_applied(self) -> None:
        # A constant image equal to the mean normalizes to ~0.
        tf = ImageClassification(crop_size=4, resize_size=4, mean=(0.5,), std=(0.5,))
        x = lucid.ones(1, 8, 8) * 0.5
        out = tf(x)
        assert abs(float(out.mean().item())) < 1e-5

    def test_square_crop(self) -> None:
        tf = ImageClassification(crop_size=128, resize_size=160)
        out = tf(lucid.rand(3, 200, 200))
        assert tuple(out.shape) == (3, 128, 128)


# ── Preset ↔ pretrained pipeline integration (G0) ───────────────────


class TestPresetIntegration:
    """Pins the contract between WeightEntry.transforms, the on-Hub
    config.json preprocessing block, and AutoTransformsPreset round-trip.

    The preset's to_dict() shape is what the conversion tool now
    emits into config.json; if that schema drifts here the published
    metadata silently becomes unreadable.
    """

    def test_resnet18_entry_transforms_is_preset(self) -> None:
        # WeightEntry.transforms must be a registered preset subclass
        # (not a raw Compose) so AutoTransformsPreset can round-trip it.
        from lucid.utils.transforms import TransformsPreset

        tf = ResNet18Weights.IMAGENET1K_V1.transforms()
        assert isinstance(tf, TransformsPreset)
        assert tf.preset_type == "ImageClassification"

    def test_resnet18_to_dict_matches_expected_schema(self) -> None:
        tf = ResNet18Weights.IMAGENET1K_V1.transforms()
        cfg = tf.to_dict()
        assert set(cfg) == {"preprocessor_type", "init_kwargs"}
        kw = cfg["init_kwargs"]
        assert isinstance(kw, dict)
        # Pinned ResNet-18 ImageNet hyperparams.
        assert kw["crop_size"] == 224
        assert kw["resize_size"] == 256
        assert kw["mean"] == [0.485, 0.456, 0.406]
        assert kw["std"] == [0.229, 0.224, 0.225]
        assert kw["interpolation"] == "bilinear"

    def test_resnet18_round_trip_via_auto_resolver(self) -> None:
        from lucid.utils.transforms import AutoTransformsPreset

        tf = ResNet18Weights.IMAGENET1K_V1.transforms()
        back = AutoTransformsPreset.from_dict(tf.to_dict())
        assert type(back) is type(tf)
        assert back.to_dict() == tf.to_dict()

    def test_resnet18_preset_threads_multitarget_sample(self) -> None:
        # An ImageClassification preset called on a {image, mask}
        # sample must thread both through the inner geometric stages
        # (Normalize touches only the image, mask survives Resize +
        # CenterCrop unchanged in label set).
        import lucid.utils.transforms as T

        lucid.manual_seed(0)
        img = lucid.rand(3, 300, 400)
        mask_raw = lucid.floor(lucid.rand(1, 300, 400) * 5.0)
        sample = {"image": T.Image(img), "mask": T.Mask(mask_raw)}
        tf = ResNet18Weights.IMAGENET1K_V1.transforms()
        out = tf(sample)
        assert tuple(out["image"].data.shape) == (3, 224, 224)
        assert tuple(out["mask"].data.shape) == (1, 224, 224)
        before = {int(round(v)) for v in mask_raw.numpy().reshape(-1).tolist()}
        after = {int(round(v)) for v in out["mask"].data.numpy().reshape(-1).tolist()}
        assert after <= before, (
            f"mask gained synthetic labels {sorted(after - before)} — "
            "ImageClassification leaked the geometric chain past nearest."
        )


# ── Factory wiring ──────────────────────────────────────────────────


class TestFactoryWiring:
    def test_random_init_default(self) -> None:
        from lucid.models.vision.resnet import resnet_18_cls

        model = resnet_18_cls()  # pretrained=False
        assert type(model).__name__ == "ResNetForImageClassification"

    def test_pretrained_false_no_download(self) -> None:
        from lucid.models.vision.resnet import resnet_18_cls

        # Must not attempt any network access when pretrained is falsy.
        model = resnet_18_cls(pretrained=False)
        assert model is not None

    def test_overrides_still_apply(self) -> None:
        from lucid.models.vision.resnet import resnet_18_cls

        model = resnet_18_cls(num_classes=10)
        assert model.config.num_classes == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ── WeightEntry.key_map ─────────────────────────────────────────────


class TestARenamedCheckpointLoads:
    """A converted checkpoint can be right and still not fit.

    ResNeSt-200/269 put a dropout before the classifier because §4.2
    asks for one, which makes the head a ``Sequential`` and moves the
    Linear to ``classifier.1``.  timm's head is a bare ``Linear``, so
    the converted checkpoint carries the *same* weights under
    ``classifier`` — dropout has no parameters — and the load failed on
    two names.  ``key_map`` renames, and does nothing else.
    """

    @staticmethod
    def _entry(key_map: dict[str, str]) -> WeightEntry:
        return WeightEntry(
            url="https://example.invalid/model.safetensors",
            sha256="0" * 64,
            num_classes=2,
            transforms=ImageClassification(crop_size=4, resize_size=4),
            key_map=key_map,
        )

    @staticmethod
    def _patch(monkeypatch: pytest.MonkeyPatch, state: dict) -> None:
        import lucid.serialization as serialization
        import lucid.weights._loading as loading

        monkeypatch.setattr(loading, "download", lambda *a, **k: "unused")
        monkeypatch.setattr(serialization, "load_safetensors", lambda _p: state)

    def test_the_rename_lands_the_weights(self, monkeypatch: pytest.MonkeyPatch) -> None:
        model = lucid.nn.Sequential(lucid.nn.Dropout(0.2), lucid.nn.Linear(3, 2))
        flat = {"1.weight": lucid.ones(2, 3), "1.bias": lucid.zeros(2)}
        self._patch(monkeypatch, {"weight": flat["1.weight"], "bias": flat["1.bias"]})
        W.load_weight_entry(
            model,
            self._entry({"weight": "1.weight", "bias": "1.bias"}),
            name="renamed",
        )
        assert float(model[1].weight.sum().item()) == 6.0

    def test_an_entry_without_a_map_is_untouched(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The overwhelming majority of entries, and they must not change."""
        model = lucid.nn.Linear(3, 2)
        self._patch(monkeypatch, {"weight": lucid.ones(2, 3), "bias": lucid.zeros(2)})
        W.load_weight_entry(model, self._entry({}), name="plain")
        assert float(model.weight.sum().item()) == 6.0

    def test_a_map_the_checkpoint_outgrew_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Silently skipping a stale rename is the dangerous reading.

        If the checkpoint is re-uploaded with corrected names, a map
        left behind would quietly do nothing and keep passing — right
        for the wrong reason, until some later layout change makes it
        wrong for good.
        """
        model = lucid.nn.Linear(3, 2)
        self._patch(monkeypatch, {"weight": lucid.ones(2, 3), "bias": lucid.zeros(2)})
        with pytest.raises(RuntimeError, match="does not contain"):
            W.load_weight_entry(model, self._entry({"gone": "weight"}), name="stale")


class TestTheTwoEntriesThatNeedIt:
    """Structural, so it holds without reaching the network."""

    def test_only_the_deep_resnests_rename(self) -> None:
        from lucid.models.vision.resnest import (
            ResNeSt50Weights,
            ResNeSt101Weights,
            ResNeSt200Weights,
            ResNeSt269Weights,
        )

        assert ResNeSt50Weights.DEFAULT.entry.key_map == {}
        assert ResNeSt101Weights.DEFAULT.entry.key_map == {}
        for deep in (ResNeSt200Weights, ResNeSt269Weights):
            assert deep.DEFAULT.entry.key_map == {
                "classifier.weight": "classifier.1.weight",
                "classifier.bias": "classifier.1.bias",
            }


# ── WeightEntry.requires_config ─────────────────────────────────────


class TestAnEntryDeclaresTheArchitectureItNeeds:
    """The drift that shapes cannot see.

    A checkpoint is trained against one configuration and the config
    that describes it is written somewhere else — a factory, a paper
    citation, someone reading a reference implementation. When they
    part company, what happens depends on whether the field touches a
    parameter. If it does, the load fails on a shape. If it does not,
    the load is clean and the model computes a different function:
    SE-ResNet's checkpoints were trained with an activation the code did
    not apply, and CSPNet's with two cross-stages the code left leaky.
    Both loaded without complaint and changed every prediction.
    """

    @staticmethod
    def _entry(requires: dict[str, object]) -> WeightEntry:
        return WeightEntry(
            url="https://example.invalid/model.safetensors",
            sha256="0" * 64,
            num_classes=2,
            transforms=ImageClassification(crop_size=4, resize_size=4),
            requires_config=requires,
        )

    @staticmethod
    def _model(**fields: object) -> lucid.nn.Module:
        module = lucid.nn.Linear(2, 2)
        module.config = type("Config", (), dict(fields))()  # type: ignore[assignment]
        return module

    def test_a_mismatch_is_refused_before_anything_is_downloaded(self) -> None:
        """The URL is unreachable, so reaching it would raise differently.

        That is the assertion: a config error costs nothing to find, and
        settling it after several hundred megabytes have moved teaches
        the same thing for a worse price.
        """
        model = self._model(min_attn_channels=32)
        with pytest.raises(RuntimeError, match="built as min_attn_channels=32"):
            W.load_weight_entry(model, self._entry({"min_attn_channels": 16}), name="drifted")

    def test_a_field_the_config_lost_is_an_error_too(self) -> None:
        """A declaration matching nothing reads as a check while being none."""
        model = self._model(something_else=1)
        with pytest.raises(RuntimeError, match="does not have"):
            W.load_weight_entry(model, self._entry({"min_attn_channels": 16}), name="renamed")

    def test_a_model_without_a_config_cannot_be_checked(self) -> None:
        with pytest.raises(RuntimeError, match="no config"):
            W.load_weight_entry(
                lucid.nn.Linear(2, 2),
                self._entry({"min_attn_channels": 16}),
                name="configless",
            )

    def test_a_tuple_and_its_list_describe_the_same_architecture(self) -> None:
        """Configs round-trip through JSON, which has no tuples."""
        model = self._model(cross_linear=[True, True, True, True])
        entry = self._entry({"cross_linear": (True, True, True, True)})
        # Gets past the config check and fails on the unreachable URL.
        with pytest.raises(Exception) as excinfo:
            W.load_weight_entry(model, entry, name="tuples")
        assert "built as" not in str(excinfo.value)

    def test_the_entries_that_carry_one_declare_what_was_measured(self) -> None:
        """Structural, so it holds without the network."""
        from lucid.models.vision.cspnet import CSPResNet50Weights
        from lucid.models.vision.maskformer import MaskFormerResNet50Weights
        from lucid.models.vision.sknet import SKResNet18Weights

        assert SKResNet18Weights.DEFAULT.entry.requires_config == {"min_attn_channels": 16}
        assert MaskFormerResNet50Weights.DEFAULT.entry.requires_config == {"num_encoder_layers": 0}
        assert CSPResNet50Weights.DEFAULT.entry.requires_config == {
            "cross_linear": (True, True, True, True)
        }

    def test_almost_every_entry_declares_nothing(self) -> None:
        """The field earns its place by being rare.

        If it spread to every entry it would be a second copy of the
        configs, drifting on its own.
        """
        from lucid.models.vision.resnet import ResNet18Weights

        assert ResNet18Weights.DEFAULT.entry.requires_config == {}
