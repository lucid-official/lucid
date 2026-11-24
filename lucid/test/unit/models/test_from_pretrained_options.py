"""``from_pretrained``: config overrides, ``ignore_mismatched_sizes`` and a short error (LCD-1111).

``from_pretrained(dir)`` took ``strict`` and nothing else.  Loading a checkpoint into a model
with another head (the transfer-learning case) meant rebuilding the config by hand, and loading
it into the wrong class raised an error that named every missing and unexpected key -- hundreds
of lines.

The loader is owned by ``lucid.models._checkpoint`` and shared by ``PretrainedModel`` and the
``AutoModelFor*`` classes, so each behaviour is asserted through both entry points:

* keyword arguments change config fields (only fields: an unknown one raises ``TypeError``);
* with ``ignore_mismatched_sizes=True`` a weight of another shape keeps its fresh
  initialisation, every other weight loads, and a warning names what was skipped;
* without it the mismatch is a ``RuntimeError`` that gives the count and the first few names;
* a registered name takes the same options, building the model at the overridden config and
  taking from the published weights whatever still fits.
"""

import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import lucid
from lucid.models import (
    AutoModelForImageClassification,
    PretrainedModel,
    create_model,
    register_model,
)
from lucid.models._registry import _REGISTRY
from lucid.models.text.bert import BERTForMaskedLM, BERTForSequenceClassification
from lucid.models.vision.lenet import LeNetForImageClassification

_TINY_BERT = dict(
    vocab_size=30,
    hidden_size=16,
    num_hidden_layers=1,
    num_attention_heads=2,
    intermediate_size=32,
    max_position_embeddings=16,
)

type Loader = Callable[..., PretrainedModel]

_LOADERS: dict[str, Loader] = {
    "PretrainedModel": LeNetForImageClassification.from_pretrained,
    "AutoModel": AutoModelForImageClassification.from_pretrained,
}


@pytest.fixture
def lenet_dir(tmp_path: Path) -> Path:
    lucid.manual_seed(0)
    create_model("lenet_5_cls").save_pretrained(str(tmp_path))
    return tmp_path


def _weights(model: PretrainedModel) -> dict[str, list[float]]:
    return {name: t.reshape(-1).tolist() for name, t in model.state_dict().items()}


@pytest.fixture(params=sorted(_LOADERS))
def load(request: pytest.FixtureRequest) -> Loader:
    return _LOADERS[request.param]


# ── config overrides ───────────────────────────────────────────────────────────


def test_a_keyword_changes_a_config_field_without_changing_a_weight(
    lenet_dir: Path, load: Loader
) -> None:
    model = load(str(lenet_dir), activation="relu")
    assert model.config.activation == "relu"
    assert _weights(model) == _weights(LeNetForImageClassification.from_pretrained(str(lenet_dir)))


@pytest.mark.parametrize("field", ["foo", "model_type", "architectures", "name_or_path"])
def test_a_keyword_that_is_not_a_config_field_is_refused(
    lenet_dir: Path, load: Loader, field: str
) -> None:
    with pytest.raises(TypeError, match=field):
        load(str(lenet_dir), **{field: 1})


def test_a_value_the_config_refuses_is_refused_by_the_config(lenet_dir: Path, load: Loader) -> None:
    with pytest.raises(ValueError):
        load(str(lenet_dir), activation="not_an_activation")


def test_a_pathlib_path_reaches_the_directory(lenet_dir: Path) -> None:
    model = LeNetForImageClassification.from_pretrained(lenet_dir, num_classes=10)
    assert model.config.num_classes == 10 and model.name_or_path == str(lenet_dir)


# ── ignore_mismatched_sizes ───────────────────────────────────────────────────


def test_a_mismatched_head_keeps_its_fresh_init_and_the_rest_loads(
    lenet_dir: Path, load: Loader
) -> None:
    saved = _weights(LeNetForImageClassification.from_pretrained(str(lenet_dir)))
    with pytest.warns(UserWarning, match="classifier.weight.*classifier.bias"):
        model = load(str(lenet_dir), num_classes=5, ignore_mismatched_sizes=True)
    assert model.config.num_classes == 5
    assert tuple(model.classifier.weight.shape) == (5, 84)
    for name, values in _weights(model).items():
        if not name.startswith("classifier."):
            assert values == saved[name], name
    # A head of its own: not the saved head's first rows, which a partial load would have left.
    assert _weights(model)["classifier.weight"] != saved["classifier.weight"][: 5 * 84]


def test_a_mismatch_is_an_error_that_counts_and_names(lenet_dir: Path, load: Loader) -> None:
    with pytest.raises(RuntimeError) as caught:
        load(str(lenet_dir), num_classes=5)
    message = str(caught.value)
    assert "2 weight(s) differ in shape" in message
    assert "'classifier.weight' (checkpoint (10, 84), model (5, 84))" in message
    assert "ignore_mismatched_sizes=True" in message


def test_no_warning_without_a_mismatch(lenet_dir: Path, load: Loader) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load(str(lenet_dir), ignore_mismatched_sizes=True)


@pytest.mark.parametrize("strict", [True, False])
def test_an_ignored_weight_is_not_a_missing_one(
    lenet_dir: Path, load: Loader, strict: bool
) -> None:
    with pytest.warns(UserWarning):
        load(str(lenet_dir), strict=strict, num_classes=3, ignore_mismatched_sizes=True)


def test_a_mismatch_it_may_not_ignore_is_not_tolerated_by_strict_false(
    lenet_dir: Path, load: Loader
) -> None:
    with pytest.raises(RuntimeError, match="differ in shape"):
        load(str(lenet_dir), strict=False, num_classes=5)


# ── the error stays short ──────────────────────────────────────────────────────


@pytest.fixture
def bert_dir(tmp_path: Path) -> Path:
    create_model("bert_base_mlm", **_TINY_BERT).save_pretrained(str(tmp_path))
    return tmp_path


def test_a_checkpoint_of_the_wrong_size_is_a_few_lines_not_hundreds(bert_dir: Path) -> None:
    with pytest.raises(RuntimeError) as caught:
        BERTForMaskedLM.from_pretrained(str(bert_dir), hidden_size=32, intermediate_size=64)
    message = str(caught.value)
    assert len(message.splitlines()) == 1 and len(message) < 1500
    assert "weight(s) differ in shape" in message and " more" in message


def test_the_wrong_class_names_the_class_the_checkpoint_came_from(bert_dir: Path) -> None:
    with pytest.raises(RuntimeError) as caught:
        BERTForSequenceClassification.from_pretrained(str(bert_dir))
    message = str(caught.value)
    assert "saved from BERTForMaskedLM, not BERTForSequenceClassification" in message
    assert "not in the checkpoint" in message and "match nothing in the model" in message
    assert len(message.splitlines()) == 1 and len(message) < 1500


def test_strict_false_still_loads_what_the_wrong_class_has_in_common(bert_dir: Path) -> None:
    model = BERTForSequenceClassification.from_pretrained(str(bert_dir), strict=False)
    saved = _weights(BERTForMaskedLM.from_pretrained(str(bert_dir)))
    assert (
        _weights(model)["bert.embeddings.word_embeddings.weight"]
        == saved["bert.embeddings.word_embeddings.weight"]
    )


# ── a registered name ─────────────────────────────────────────────────────────

_PUBLISHED = "lenet_probe_published"
_BUILT: list[tuple[bool, tuple[str, ...]]] = []


@pytest.fixture
def published() -> Iterator[PretrainedModel]:
    """A registered factory whose ``pretrained=True`` weights are a fixed LeNet's, as a hub's are."""
    lucid.manual_seed(7)
    weights = create_model("lenet_5_cls")
    _BUILT.clear()

    @register_model(
        task="image-classification",
        family="lenet",
        model_type="lenet",
        model_class=LeNetForImageClassification,
    )
    def lenet_probe_published(pretrained: bool = False, **overrides: object) -> PretrainedModel:
        _BUILT.append((pretrained, tuple(sorted(overrides))))
        model = create_model("lenet_5_cls", **overrides)
        if pretrained:
            model.load_state_dict(weights.state_dict())
        return model

    try:
        yield weights
    finally:
        _REGISTRY.pop(_PUBLISHED, None)
        _BUILT.clear()


def test_a_registered_name_without_options_loads_as_before(published: PretrainedModel) -> None:
    model = LeNetForImageClassification.from_pretrained(_PUBLISHED)
    assert _BUILT == [(True, ())] and _weights(model) == _weights(published)


@pytest.mark.parametrize("entry", ["PretrainedModel", "AutoModel"])
def test_a_registered_name_takes_overrides_and_keeps_what_fits(
    published: PretrainedModel, entry: str
) -> None:
    with pytest.warns(UserWarning, match="classifier.weight"):
        model = _LOADERS[entry](_PUBLISHED, num_classes=4, ignore_mismatched_sizes=True)
    assert model.config.num_classes == 4 and model.name_or_path == ""
    for name, values in _weights(model).items():
        if not name.startswith("classifier."):
            assert values == _weights(published)[name], name


def test_a_registered_name_refuses_the_mismatch_it_may_not_ignore(
    published: PretrainedModel,
) -> None:
    with pytest.raises(RuntimeError, match="'classifier.weight'.*ignore_mismatched_sizes=True"):
        LeNetForImageClassification.from_pretrained(_PUBLISHED, num_classes=4)


def test_an_unknown_override_is_refused_before_anything_is_downloaded(
    published: PretrainedModel,
) -> None:
    with pytest.raises(TypeError, match="foo"):
        LeNetForImageClassification.from_pretrained(_PUBLISHED, foo=1)
    assert all(not pretrained for pretrained, _ in _BUILT)  # the weights were never asked for


@pytest.mark.filterwarnings("ignore:Some weights:UserWarning")
def test_the_class_check_still_applies_to_a_registered_name(published: PretrainedModel) -> None:
    with pytest.raises(TypeError, match="not a subclass of BERTForMaskedLM"):
        BERTForMaskedLM.from_pretrained(_PUBLISHED, num_classes=4, ignore_mismatched_sizes=True)
