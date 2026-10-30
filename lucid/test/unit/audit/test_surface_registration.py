"""A sub-package has to be declared to the audit, or declared out of it.

``lucid.peft`` landed in ``_SUBPKG_NAMES`` and nowhere in the audit's
subsystem table.  The surface did not skip it — it enumerated it under
the ``lucid`` catch-all, so ``lora.Linear`` was filed at the top level beside
``nn.Linear``, its model-rewriting functions were probed as ops over
tensors, and a release cut then would have frozen those spellings in
``released_surface.json``.  What failed was a construction count three
checks away.  These tests make the omission the failure.
"""

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import LoraModel, PeftModel
from lucid.peft.tuners import lora
from lucid.test.audit import _surface
from lucid.test.audit._builders import BUILDERS


def test_every_lazy_subpackage_is_audited_or_excluded() -> None:
    missing = _surface.unregistered_subpackages()
    assert not missing, (
        f"lucid.{missing} load on demand but are neither in _surface.SUBSYSTEMS nor in "
        "_surface.EXCLUDED — add a SUBSYSTEMS key (the kind says how its symbols are probed) "
        "or an EXCLUDED entry with the reason"
    )


def test_the_rule_can_go_red() -> None:
    """A check that cannot fail passes for free."""
    assert _surface.unregistered_subpackages(["nn", "brand_new", "signal", "test"]) == ["brand_new"]


def test_the_rule_reads_the_package_list_lucid_itself_uses() -> None:
    assert "peft" in lucid._SUBPKG_NAMES
    assert _surface.unregistered_subpackages(lucid._SUBPKG_NAMES) == []


def test_peft_symbols_carry_the_spelling_a_user_writes() -> None:
    by_name = {s.qualname: s for s in _surface.enumerate_surface(["peft", "peft.tuners.lora"])}
    for name in (
        "lucid.peft.LoraConfig",
        "lucid.peft.LoraModel",
        "lucid.peft.PeftModel",
        "lucid.peft.get_peft_model",
        "lucid.peft.tuners.lora.Linear",
        "lucid.peft.tuners.lora.LoraLayer",
    ):
        assert name in by_name, name
    assert not {"lucid.Linear", "lucid.LoraModel", "lucid.PeftModel"} & {
        s.qualname for s in _surface.enumerate_surface(["lucid"])
    }


@pytest.mark.parametrize("name", ["get_peft_model", "inject_adapter_in_model"])
def test_a_function_that_rewrites_its_model_is_never_surveyed_bare(name: str) -> None:
    symbol = {s.qualname: s for s in _surface.enumerate_surface(["peft"])}[f"lucid.peft.{name}"]
    assert not symbol.inert


@pytest.mark.parametrize("cls", [lora.Linear, LoraModel, PeftModel], ids=lambda c: c.__qualname__)
def test_a_registered_builder_makes_a_fresh_working_instance(cls: type) -> None:
    first, second = BUILDERS[cls](), BUILDERS[cls]()
    assert type(first) is cls
    # PeftModel rewrites the model it is handed; a builder that reused one
    # would hand the next caller an adapted model.
    assert {id(p) for p in first.parameters()}.isdisjoint(id(p) for p in second.parameters())
    assert first(lucid.randn(2, 4)).shape == (2, 3 if cls is lora.Linear else 4)


def test_a_builder_is_not_taken_for_a_layer_of_the_same_name() -> None:
    assert nn.Linear not in BUILDERS
    assert lora.Linear is not nn.Linear
