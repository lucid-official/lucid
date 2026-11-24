"""What the registry does with a name that is not a string (LCD-1111).

``create_model(5)`` and ``is_model(5)`` failed with ``AttributeError: 'int' object has no
attribute 'replace'``, from the first line of name normalisation.  The invariant is owned by
``_normalize``, the one place a name is checked, so this file drives every public entry
point that takes one and asserts they agree:

* a lookup (``create_model``, ``model_entrypoint``) refuses it with ``TypeError``.  The name
  is a wrong argument rather than an unknown model, which is what ``ValueError`` already
  says for a string no entry has;
* the predicate (``is_model``) answers ``False``, as it does for any name that is not
  registered -- which is also what lets a ``pathlib.Path`` fall through to the directory
  branch of ``from_pretrained``;
* the listing filters (``task=``, ``family=``) refuse it, since a filter that can match
  nothing returns ``[]`` and reads as "no such models".
"""

from pathlib import Path

import pytest

from lucid.models import AutoConfig, create_model, is_model, list_models, model_entrypoint
from lucid.models._registry import _REGISTRY, _registry_lookup

# Things that are not a ``str``, including the near misses: a ``Path`` to a real
# directory, and ``bytes`` of a registered name.
_NOT_NAMES: list[object] = [5, 2.5, None, True, b"resnet_50", ("resnet_50",), Path("resnet_50")]
_IDS = [type(v).__name__ for v in _NOT_NAMES]

_LOOKUPS = {
    "create_model": create_model,
    "model_entrypoint": model_entrypoint,
    "_registry_lookup": lambda name: _registry_lookup(name, task="base"),
}


@pytest.mark.parametrize("name", _NOT_NAMES, ids=_IDS)
@pytest.mark.parametrize("entry_point", sorted(_LOOKUPS))
def test_a_lookup_refuses_a_name_that_is_not_a_string(entry_point: str, name: object) -> None:
    with pytest.raises(TypeError, match="model name must be a str"):
        _LOOKUPS[entry_point](name)


@pytest.mark.parametrize("name", _NOT_NAMES, ids=_IDS)
def test_is_model_answers_false_for_a_name_that_is_not_a_string(name: object) -> None:
    assert is_model(name) is False


def test_the_lookups_still_agree_with_is_model_on_strings() -> None:
    registered = next(iter(_REGISTRY))
    assert is_model(registered) and is_model(registered.upper().replace("_", "-"))
    assert callable(model_entrypoint(registered))
    with pytest.raises(ValueError, match="Unknown model"):
        create_model("not_a_registered_model")
    assert is_model("not_a_registered_model") is False


@pytest.mark.parametrize("keyword", ["task", "family"])
@pytest.mark.parametrize("value", [5, 2.5, b"resnet", ("resnet",)], ids=lambda v: type(v).__name__)
def test_a_listing_filter_that_is_not_a_string_is_refused(keyword: str, value: object) -> None:
    with pytest.raises(TypeError, match=f"{keyword} must be a str or None"):
        list_models(**{keyword: value})


def test_a_listing_filter_may_still_be_omitted_or_none() -> None:
    assert list_models(task=None, family=None) == list_models()
    assert list_models(family="resnet")


def test_a_registered_name_in_another_case_still_resolves() -> None:
    name = next(iter(_REGISTRY))
    assert model_entrypoint(name.upper().replace("_", "-")) is _REGISTRY[name].factory


def test_auto_config_refuses_a_name_that_is_not_a_string_or_path() -> None:
    name: object = 5
    with pytest.raises(TypeError):
        AutoConfig.from_pretrained(name)
