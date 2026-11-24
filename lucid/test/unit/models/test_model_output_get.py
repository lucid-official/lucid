"""``ModelOutput.get``, and the one rule ``keys``, ``in``, ``[]`` and ``get`` share (LCD-1111).

The reference's ``ModelOutput`` is a dict, so ``out.get("loss")`` works there; here there was
``keys`` / ``items`` / ``[]`` and no ``get``.  A *key* is a dataclass field that holds a value:
a ``None`` field is not one.  ``[]`` and ``in`` used to decide that by attribute lookup, which
also answers for the class's methods -- ``"keys" in out`` was ``True`` and ``out["keys"]``
returned the bound method.  All four now go through one lookup (``ModelOutput._present``), and
this file holds every output class in the zoo to the rule, so a new one cannot opt out.
"""

import dataclasses

import pytest

import lucid
import lucid.models  # registers every family, hence every output class
from lucid.models import ImageClassificationOutput
from lucid.models._output import ModelOutput


def _subclasses(root: type[ModelOutput]) -> list[type[ModelOutput]]:
    found: list[type[ModelOutput]] = []
    for sub in root.__subclasses__():
        found.append(sub)
        found.extend(_subclasses(sub))
    return found


_OUTPUTS = sorted(
    {cls for cls in _subclasses(ModelOutput) if dataclasses.is_dataclass(cls)},
    key=lambda cls: f"{cls.__module__}.{cls.__qualname__}",
)

_NOT_FIELDS = ("keys", "values", "items", "get", "to_tuple", "_present", "__class__", "__len__", "")


def _build(cls: type[ModelOutput], **given: object) -> ModelOutput:
    """``cls`` with every field that has no default set to a tensor, and ``given`` on top."""
    required = {
        f.name: lucid.zeros(1)
        for f in dataclasses.fields(cls)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    }
    return cls(**{**required, **given})


def _names(out: ModelOutput) -> list[str]:
    return [f.name for f in dataclasses.fields(type(out))]


def test_the_sweep_sees_the_zoos_output_classes() -> None:
    # Guards the parametrised tests below against matching nothing.
    assert len(_OUTPUTS) >= 20
    assert ImageClassificationOutput in _OUTPUTS


@pytest.mark.parametrize("cls", _OUTPUTS, ids=lambda c: c.__name__)
def test_get_and_the_other_lookups_agree_on_what_a_key_is(cls: type[ModelOutput]) -> None:
    out = _build(cls)
    names = _names(out)
    present = {n for n in names if getattr(out, n) is not None}
    assert set(out.keys()) == present
    assert {n for n in names if n in out} == present
    assert {n for n in names if out.get(n) is not None} == present
    for name in names:
        if name in present:
            assert out.get(name) is getattr(out, name) is out[name]
        else:
            with pytest.raises(KeyError):
                out[name]
            assert out.get(name, "fallback") == "fallback"


@pytest.mark.parametrize("cls", _OUTPUTS, ids=lambda c: c.__name__)
def test_a_none_field_is_not_a_key_whatever_it_is_called(cls: type[ModelOutput]) -> None:
    optional = [f.name for f in dataclasses.fields(cls) if f.default is None]
    for name in optional:
        out = _build(cls, **{name: None})
        assert name not in out.keys()
        assert name not in out
        assert out.get(name) is None
        marker = object()
        assert out.get(name, marker) is marker
        filled = _build(cls, **{name: lucid.ones(1)})
        assert name in filled and filled.get(name) is filled[name]


@pytest.mark.parametrize("cls", _OUTPUTS, ids=lambda c: c.__name__)
def test_a_name_that_is_not_a_field_is_not_a_key(cls: type[ModelOutput]) -> None:
    out = _build(cls)
    marker = object()
    for name in _NOT_FIELDS:
        assert name not in out
        assert out.get(name) is None
        assert out.get(name, marker) is marker
        with pytest.raises(KeyError):
            out[name]


@pytest.mark.parametrize("key", [0, 1, 2.5, None, ("logits",)], ids=lambda k: type(k).__name__)
def test_get_with_a_key_that_is_not_a_string_is_the_default(key: object) -> None:
    out = ImageClassificationOutput(logits=lucid.zeros(1, 3))
    assert key not in out
    assert out.get(key) is None
    assert out.get(key, 7) == 7


def test_a_present_value_that_is_falsy_is_still_the_value() -> None:
    loss = lucid.zeros(())
    out = ImageClassificationOutput(logits=lucid.zeros(1, 3), loss=loss)
    assert out.get("loss", 5.0) is loss


def test_get_defaults_to_none_and_takes_the_default_by_position_or_keyword() -> None:
    out = ImageClassificationOutput(logits=lucid.zeros(1, 3))
    assert out.get("loss") is None
    assert out.get("loss", 1) == 1
    assert out.get("loss", default=2) == 2
    assert out.get("logits") is out.logits


def test_len_iteration_and_keys_still_skip_none_fields() -> None:
    out = ImageClassificationOutput(logits=lucid.zeros(1, 3))
    assert len(out) == 1 and out.keys() == ["logits"]
    assert [v is out.logits for v in out] == [True]
    assert out.items() == [("logits", out.logits)]
