"""Phase-0 foundation tests: ModelConfig / PretrainedModel / Output / Registry / Auto / Hub."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

import lucid
import lucid.nn as nn
from lucid.models import (
    AutoConfig,
    AutoModel,
    AutoModelForImageClassification,
    BaseModelOutput,
    ImageClassificationOutput,
    ModelConfig,
    ModelOutput,
    PretrainedModel,
    create_model,
    is_model,
    list_models,
    register_model,
)
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models import _registry as _reg_mod
from lucid.weights import download


@dataclass(frozen=True)
class _DummyConfig(ModelConfig):
    """Config used only for foundation tests."""

    model_type: ClassVar[str] = "dummy"
    in_features: int = 4
    out_features: int = 2
    hidden_dim: int = 8


class _DummyBackbone(PretrainedModel):
    """Backbone that just runs a single Linear."""

    config_class: ClassVar[type[ModelConfig]] = _DummyConfig

    def __init__(self, config: _DummyConfig) -> None:
        super().__init__(config)
        self.proj = nn.Linear(config.in_features, config.hidden_dim)

    def forward(self, x: lucid.Tensor) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=self.proj(x))


class _DummyForClassification(PretrainedModel, ClassificationHeadMixin):
    """Backbone + classifier head."""

    config_class: ClassVar[type[ModelConfig]] = _DummyConfig

    def __init__(self, config: _DummyConfig) -> None:
        super().__init__(config)
        self.backbone = _DummyBackbone(config)
        self._build_classifier(config.hidden_dim, config.out_features)

    def forward(self, x: lucid.Tensor) -> ImageClassificationOutput:
        features = self.backbone(x).last_hidden_state
        return ImageClassificationOutput(logits=self.classifier(features))


@pytest.fixture(autouse=True)
def _clean_registry() -> object:
    """Snapshot/restore the global registry around every test."""
    snap = dict(_reg_mod._REGISTRY)
    try:
        yield
    finally:
        _reg_mod._REGISTRY.clear()
        _reg_mod._REGISTRY.update(snap)


# ─── ModelOutput ─────────────────────────────────────────────────────────────


class TestModelOutput:
    def test_iter_skips_none(self) -> None:
        out = ImageClassificationOutput(logits=lucid.zeros(2, 3))
        assert len(list(out)) == 1

    def test_getitem_int_and_str(self) -> None:
        out = ImageClassificationOutput(logits=lucid.zeros(1, 2))
        assert out[0] is out.logits
        assert out["logits"] is out.logits

    def test_getitem_str_missing_field_raises(self) -> None:
        out = ImageClassificationOutput(logits=lucid.zeros(1, 2))
        with pytest.raises(KeyError):
            _ = out["nonexistent"]

    def test_keys_values_items_skip_none(self) -> None:
        out = ImageClassificationOutput(logits=lucid.zeros(1, 2))
        assert out.keys() == ["logits"]
        assert len(out.values()) == 1
        assert [k for k, _ in out.items()] == ["logits"]

    def test_to_tuple_unpack(self) -> None:
        out = ImageClassificationOutput(logits=lucid.zeros(1, 2))
        (only,) = out.to_tuple()
        assert only is out.logits

    def test_contains_str(self) -> None:
        out = ImageClassificationOutput(logits=lucid.zeros(1, 2))
        assert "logits" in out
        assert "loss" not in out

    def test_isinstance_modeloutput(self) -> None:
        out = ImageClassificationOutput(logits=lucid.zeros(1, 2))
        assert isinstance(out, ModelOutput)


# ─── ModelConfig ─────────────────────────────────────────────────────────────


class TestModelConfig:
    def test_to_dict_includes_model_type(self) -> None:
        cfg = _DummyConfig()
        d = cfg.to_dict()
        assert d["model_type"] == "dummy"
        assert d["in_features"] == 4

    def test_from_dict_round_trip(self) -> None:
        cfg = _DummyConfig(in_features=8, out_features=3, hidden_dim=16)
        d = cfg.to_dict()
        rebuilt = _DummyConfig.from_dict(d)
        assert rebuilt == cfg

    def test_from_dict_warns_on_unknown_field(self) -> None:
        # Unknown fields in the checkpoint (from a newer version of the config)
        # should warn and be silently dropped — not raise — so old code can
        # still load newer checkpoints that have gained extra fields.
        with pytest.warns(UserWarning, match="unrecognised fields"):
            cfg = _DummyConfig.from_dict({"in_features": 4, "bogus": 1})
        assert cfg == _DummyConfig(in_features=4)

    def test_from_dict_missing_required_field_raises(self) -> None:
        # Missing *required* fields (no default) must still raise TypeError.
        @dataclass(frozen=True)
        class _RequiredFieldConfig(ModelConfig):
            model_type: ClassVar[str] = "required_test"
            required_field: int  # no default — must be present

        with pytest.raises(TypeError):
            _RequiredFieldConfig.from_dict({})

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        cfg = _DummyConfig(in_features=12)
        path = tmp_path / "config.json"
        cfg.save(str(path))
        loaded = _DummyConfig.load(str(path))
        assert loaded == cfg


# ─── PretrainedModel ─────────────────────────────────────────────────────────


class TestPretrainedModel:
    def test_init_rejects_wrong_config_type(self) -> None:
        @dataclass(frozen=True)
        class _OtherConfig(ModelConfig):
            model_type: ClassVar[str] = "other"

        with pytest.raises(TypeError, match="expects _DummyConfig"):
            _DummyBackbone(_OtherConfig())  # type: ignore[arg-type]

    def test_forward_shape(self) -> None:
        cfg = _DummyConfig(in_features=4, hidden_dim=8)
        model = _DummyBackbone(cfg)
        x = lucid.randn(3, 4)
        out = model(x)
        assert tuple(out.last_hidden_state.shape) == (3, 8)

    def test_save_pretrained_then_from_pretrained(self, tmp_path: Path) -> None:
        cfg = _DummyConfig(in_features=4, hidden_dim=8, out_features=2)
        model = _DummyForClassification(cfg)
        # Run once so any lazy params materialise
        x = lucid.randn(2, 4)
        original = model(x).logits.numpy()
        save_dir = tmp_path / "ckpt"
        model.save_pretrained(str(save_dir))

        assert (save_dir / "config.json").exists()
        assert (save_dir / "weights.lucid").exists()

        restored = _DummyForClassification.from_pretrained(str(save_dir))
        again = restored(x).logits.numpy()
        # Same weights → identical output
        import numpy as np

        np.testing.assert_allclose(again, original, atol=1e-6)

    def test_num_parameters_counts_correctly(self) -> None:
        cfg = _DummyConfig(in_features=4, hidden_dim=8, out_features=2)
        model = _DummyForClassification(cfg)
        # backbone Linear(4→8) = 4*8 + 8 = 40
        # classifier Linear(8→2) = 8*2 + 2 = 18
        assert model.num_parameters() == 40 + 18


# ─── Registry ────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_register_and_lookup(self) -> None:
        @register_model(task="base", family="dummy", model_type="dummy")
        def dummy_small(pretrained: bool = False) -> _DummyBackbone:
            return _DummyBackbone(_DummyConfig())

        assert is_model("dummy_small")
        assert "dummy_small" in list_models()
        model = create_model("dummy_small")
        assert isinstance(model, _DummyBackbone)

    def test_name_normalization_hyphen_equiv(self) -> None:
        @register_model(family="dummy", model_type="dummy")
        def dummy_tiny(pretrained: bool = False) -> _DummyBackbone:
            return _DummyBackbone(_DummyConfig())

        assert is_model("dummy-tiny")
        assert is_model("DUMMY_TINY")
        assert create_model("dummy-tiny").config == _DummyConfig()

    def test_duplicate_registration_raises(self) -> None:
        @register_model(family="dummy")
        def dummy_dup(pretrained: bool = False) -> _DummyBackbone:
            return _DummyBackbone(_DummyConfig())

        with pytest.raises(ValueError, match="already registered"):

            @register_model(family="dummy")
            def dummy_dup(pretrained: bool = False) -> _DummyBackbone:  # noqa: F811
                return _DummyBackbone(_DummyConfig())

    def test_typo_suggestion_in_error(self) -> None:
        # resnet_50 is already registered; confirm the typo suggestion fires.
        with pytest.raises(ValueError, match="resnet_50"):
            create_model("resnet50")  # missing underscore → suggestion

    def test_list_models_filters(self) -> None:
        @register_model(task="base", family="dummy")
        def dummy_a(pretrained: bool = False) -> _DummyBackbone:
            return _DummyBackbone(_DummyConfig())

        @register_model(task="image-classification", family="dummy")
        def dummy_a_for_classification(pretrained: bool = False) -> _DummyForClassification:
            return _DummyForClassification(_DummyConfig())

        assert "dummy_a" in list_models(task="base")
        assert "dummy_a_for_classification" in list_models(task="image-classification")
        assert list_models(family="dummy") == sorted(["dummy_a", "dummy_a_for_classification"])


# ─── Auto classes ────────────────────────────────────────────────────────────


class TestAutoClasses:
    def test_auto_cannot_instantiate(self) -> None:
        with pytest.raises(EnvironmentError):
            AutoModel()
        with pytest.raises(EnvironmentError):
            AutoConfig()

    def test_automodel_dispatches_by_task_base(self) -> None:
        @register_model(task="base", family="dummy", model_type="dummy")
        def dummy_back(pretrained: bool = False) -> _DummyBackbone:
            return _DummyBackbone(_DummyConfig())

        @register_model(task="image-classification", family="dummy", model_type="dummy")
        def dummy_cls(pretrained: bool = False) -> _DummyForClassification:
            return _DummyForClassification(_DummyConfig())

        assert isinstance(AutoModel.from_pretrained("dummy_back"), _DummyBackbone)
        assert isinstance(
            AutoModelForImageClassification.from_pretrained("dummy_cls"), _DummyForClassification
        )

    def test_auto_wrong_task_raises(self) -> None:
        @register_model(task="base", family="dummy", model_type="dummy")
        def dummy_back2(pretrained: bool = False) -> _DummyBackbone:
            return _DummyBackbone(_DummyConfig())

        with pytest.raises(ValueError, match="registered for task"):
            AutoModelForImageClassification.from_pretrained("dummy_back2")

    def test_auto_from_directory(self, tmp_path: Path) -> None:
        @register_model(task="image-classification", family="dummy", model_type="dummy")
        def dummy_dir(pretrained: bool = False) -> _DummyForClassification:
            return _DummyForClassification(_DummyConfig())

        model = create_model("dummy_dir")
        save_dir = tmp_path / "saved"
        model.save_pretrained(str(save_dir))
        loaded = AutoModelForImageClassification.from_pretrained(str(save_dir))
        assert isinstance(loaded, _DummyForClassification)

    def test_autoconfig_returns_config_for_registered_name(self) -> None:
        @register_model(task="base", family="dummy", model_type="dummy")
        def dummy_for_config(pretrained: bool = False) -> _DummyBackbone:
            return _DummyBackbone(_DummyConfig(in_features=16))

        cfg = AutoConfig.from_pretrained("dummy_for_config")
        assert isinstance(cfg, _DummyConfig)
        assert cfg.in_features == 16


# ─── Mixins ──────────────────────────────────────────────────────────────────


class TestClassificationHeadMixin:
    def test_build_creates_linear_classifier(self) -> None:
        cfg = _DummyConfig(hidden_dim=8, out_features=4)
        m = _DummyForClassification(cfg)
        assert isinstance(m.classifier, nn.Linear)
        assert m.classifier.out_features == 4

    def test_reset_classifier_changes_output(self) -> None:
        cfg = _DummyConfig(hidden_dim=8, out_features=4)
        m = _DummyForClassification(cfg)
        m.reset_classifier(num_classes=10)
        assert isinstance(m.classifier, nn.Linear)
        assert m.classifier.out_features == 10
        # New head still receives same input dim
        assert m.classifier.in_features == 8


class TestBackboneMixin:
    def test_concrete_backbone_satisfies_interface(self) -> None:
        @dataclass(frozen=True)
        class _BConfig(ModelConfig):
            model_type: ClassVar[str] = "back"
            channels: int = 3

        class _MiniBackbone(PretrainedModel, BackboneMixin):
            config_class: ClassVar[type[ModelConfig]] = _BConfig

            def __init__(self, config: _BConfig) -> None:
                super().__init__(config)
                self.conv = nn.Conv2d(config.channels, 16, kernel_size=3, padding=1)

            def forward_features(self, x: lucid.Tensor) -> lucid.Tensor:
                return self.conv(x)

            @property
            def feature_info(self) -> list[FeatureInfo]:
                return [FeatureInfo(stage=0, num_channels=16, reduction=1)]

            def forward(self, x: lucid.Tensor) -> BaseModelOutput:
                return BaseModelOutput(last_hidden_state=self.forward_features(x))

        m = _MiniBackbone(_BConfig())
        out = m.forward_features(lucid.randn(1, 3, 8, 8))
        assert tuple(out.shape) == (1, 16, 8, 8)
        assert m.feature_info[0].num_channels == 16


# ─── Hub (download + SHA256 verify with file:// URL) ─────────────────────────


class TestHub:
    def test_download_verifies_and_caches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Point the cache somewhere temporary so we don't pollute ~/.cache.
        monkeypatch.setenv("LUCID_HOME", str(tmp_path))

        # Create a fake remote file and serve it via file:// URL.
        remote = tmp_path / "remote.bin"
        payload = b"hello-lucid-models-zoo"
        remote.write_bytes(payload)
        sha = hashlib.sha256(payload).hexdigest()

        url = remote.as_uri()
        local = download(url, sha, name="dummy_model")
        assert local.exists()
        assert local.read_bytes() == payload

        # Second call should hit the cache (no exception, same path).
        local2 = download(url, sha, name="dummy_model")
        assert local2 == local

    def test_download_sha_mismatch_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LUCID_HOME", str(tmp_path))
        remote = tmp_path / "remote.bin"
        remote.write_bytes(b"actual-content")
        wrong_sha = "0" * 64
        with pytest.raises(RuntimeError, match="SHA256 mismatch"):
            download(remote.as_uri(), wrong_sha, name="dummy_model")


# ─────────────────────────────────────────────────────────────────────────────
# AutoModel coverage — new task-level dispatchers (2026-05)
# ─────────────────────────────────────────────────────────────────────────────


class TestAutoModelTaskCoverage:
    """Every Auto class must resolve at least one registered factory.

    Catches the regression where a model factory is registered under a task
    string for which no ``AutoModel`` subclass exists (silent dispatch hole).
    """

    def test_sequence_classification_dispatch(self) -> None:
        from lucid.models import AutoModelForSequenceClassification, BERTForSequenceClassification
        from lucid.models._registry import _registry_lookup

        entry = _registry_lookup("bert_base_cls", task=AutoModelForSequenceClassification._task)
        assert entry.model_class is BERTForSequenceClassification

    def test_token_classification_dispatch(self) -> None:
        from lucid.models import AutoModelForTokenClassification, BERTForTokenClassification
        from lucid.models._registry import _registry_lookup

        entry = _registry_lookup("bert_base_token_cls", task=AutoModelForTokenClassification._task)
        assert entry.model_class is BERTForTokenClassification

    def test_question_answering_dispatch(self) -> None:
        """Span extraction is a sequence-classification model here.

        It had its own task until the taxonomy was collapsed; the head
        reads a pooled representation and scores it, which is what the
        surviving task describes.  The output type stays its own.
        """
        from lucid.models import AutoModelForSequenceClassification, BERTForQuestionAnswering
        from lucid.models._registry import _registry_lookup

        entry = _registry_lookup("bert_base_qa", task=AutoModelForSequenceClassification._task)
        assert entry.model_class is BERTForQuestionAnswering

    def test_seq2seq_lm_dispatch(self) -> None:
        from lucid.models import AutoModelForLanguageModeling, TransformerForSeq2SeqLM
        from lucid.models._registry import _registry_lookup

        entry = _registry_lookup(
            "transformer_base_seq2seq", task=AutoModelForLanguageModeling._task
        )
        assert entry.model_class is TransformerForSeq2SeqLM

    def test_no_task_has_zero_models(self) -> None:
        """Every Auto class's task tag has at least one registered factory.

        The list is derived rather than written out.  It used to be
        hardcoded and had already gone stale twice — ImageGeneration and
        WorldModeling were both missing, so the guard passed while not
        covering them.  A list of the things to check, maintained by hand
        beside the things themselves, drifts silently in exactly the way
        the check is supposed to catch.
        """
        import lucid.models as models
        from lucid.models._auto import _BaseAutoClass
        from lucid.models._registry import list_models

        autos = [
            getattr(models, n)
            for n in models.__all__
            if isinstance(getattr(models, n), type)
            and issubclass(getattr(models, n), _BaseAutoClass)
            and getattr(models, n) is not _BaseAutoClass
        ]
        assert autos, "no Auto classes found — the derivation itself broke"
        for auto_cls in autos:
            registered = list_models(task=auto_cls._task)
            assert len(registered) > 0, (
                f"{auto_cls.__name__} has task={auto_cls._task!r} but no models "
                f"are registered under it — dispatch hole."
            )

    def test_every_registered_wrapper_has_a_task_base(self) -> None:
        """A model registered under a task inherits that task's base.

        This is the invariant the eight task bases exist to hold. Before
        them every wrapper subclassed ``PretrainedModel`` directly, so
        "what is this model for" was answerable only from the class name
        — and the families predating the ``For<Task>`` convention
        (``GPTLMHeadModel``, ``GPT2LMHeadModel``) answered it differently
        from the rest.
        """
        from lucid.models._registry import _REGISTRY
        from lucid.models._tasks import TaskModel

        offenders = sorted(
            {
                entry.model_class.__name__
                for entry in _REGISTRY.values()
                if entry.task != "base"
                and entry.model_class is not None
                and not issubclass(entry.model_class, TaskModel)
            }
        )
        assert not offenders, f"registered under a task but not a TaskModel: {offenders}"

    def test_class_task_matches_the_tag_it_registers_under(self) -> None:
        """The base's ``task`` and the registry tag are the same string.

        They were three separate lists (class-name suffix, registry tag,
        Auto shell) that nothing forced to agree.  Now the base carries
        the tag and this asserts the factory did not override it with
        another.
        """
        from lucid.models._registry import _REGISTRY

        mismatched = [
            (name, entry.model_class.__name__, entry.task, entry.model_class.task)
            for name, entry in _REGISTRY.items()
            if entry.task != "base"
            and entry.model_class is not None
            and getattr(entry.model_class, "task", None) != entry.task
        ]
        assert not mismatched, f"class.task disagrees with registry tag: {mismatched[:5]}"
