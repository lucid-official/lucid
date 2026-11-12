"""``PeftMixedModel`` against the reference's mixed model, on adapter files the reference wrote.

The reference cannot write a mixed file (its ``save_pretrained`` raises), so the comparison is the
realistic one: on one two-linear-layer model whose weights are copied to both sides, the reference
writes a plain adapter file per method; the reference's ``PeftMixedModel.from_pretrained`` +
``load_adapter`` and this library's read the same files; and the logits agree within ``1e-5`` with
each adapter in use alone and with both, in both load orders (and with ``set_adapter`` listing the
two the other way).  Every adapter is moved off its start the way ``perturb_adapter`` moves one, and
every comparison fails unless each adapter's own effect on the logits is at least a hundred times
the tolerance, so an agreement that held because an adapter did nothing cannot pass.

The pairs (the order of load matters where the methods do not commute: the reference's own two
orders of LoHa + OFT differ by 0.13 on the logits, and a reference stack opened in the other order
than this library's fails the test; LoRA + LoHa and LoRA + LoKr add, and agree to 5e-7 either way):

* LoRA + LoHa, LoRA + LoKr, LoHa + OFT, compared on the CPU and on Metal.
* LoRA + IA3 is compared only if the reference's mixed model takes IA3.  It does not: IA3 is in its
  ``from_pretrained`` mapping but not in the list its constructor and ``add_adapter`` accept
  (``LORA, LOHA, LOKR, ADALORA, OFT, SHIRA``), so it raises ``ValueError``.  This library accepts
  the pair, and the test holds *its* half to the same signal floor.  OFT is the reverse: the reference's
  constructor takes it and ``from_pretrained`` does not, so as the first adapter the reference side
  is opened with the constructor and the file's values loaded into it (the same files either way).

Two things measured on the reference (release 0.21.2) and pinned here, because the library differs
from, or is asked to match, the first:

* **A reload of the first file after stacking is ignored there**: the stack's inner layer holds its
  adapter at ``0.base_layer.lora_A.<name>.weight`` while the file says ``0.lora_A.weight``, so
  ``load_adapter(file, "lora")`` after a LoHa layer was stacked on it matches nothing: no warning,
  logits unchanged to the last bit (max difference 0.0), only the returned load result lists the
  keys (4 unexpected, 4 missing).  This library files an adapter of a stack as a single-method model
  would, so the same call loads the file (the logits move by 8.8 on the CPU) and agrees with the
  reference that loaded the file *before* stacking.  An intended difference.
* **Merging one of two active adapters of one LoRA layer drops the other** from the reference's
  forward pass (LCD-971): a layer that is merged runs its base layer only, so after
  ``merge_adapter(adapter_names=["one"])`` with ``["one", "two"]`` in use the logits equal ``one``
  alone (difference 7.2e-7) and differ from the pair by ``two``'s whole effect (4.31 here).  This
  library's single-method model gives the same logits to the last bit (difference 0.0, measured
  but not pinned: it is the open question LCD-971 asks, not parity), so that issue is whether to
  refuse what the reference silently drops, not a divergence from it.
"""

import warnings
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    PEFT_TYPE_TO_MIXED_MODEL_MAPPING,
    LoHaConfig,
    LoraConfig,
    PeftMixedModel,
    PeftType,
)
from lucid.test.unit.peft._parity import across, adapter_library, gap
from lucid.test.unit.peft._tuner_recipes import RECIPES

_F32 = lucid.float32
_TOLERANCE = 1e-5
#: What every comparison's smallest effect must reach: a hundred times the tolerance.
_SIGNAL = 100 * _TOLERANCE
#: The families the reference's mixed model refuses (measured: its constructor and ``add_adapter``
#: accept ``LORA, LOHA, LOKR, ADALORA, OFT, SHIRA`` only).  A pair with one is held to Lucid's half.
_REFERENCE_REFUSES = frozenset({"ia3"})

_ReferenceConfig = Callable[[ModuleType], ModuleType]
#: How the reference builds each family's adapter on the two linear layers (sites ``"0"``, ``"2"``).
_FAMILIES: dict[str, _ReferenceConfig] = {
    "lora": lambda library: library.LoraConfig(r=2, target_modules=["0", "2"]),
    "loha": lambda library: library.LoHaConfig(r=2, target_modules=["0", "2"]),
    "lokr": lambda library: library.LoKrConfig(r=2, target_modules=["0", "2"]),
    # the blocks must divide both layers' inputs (6 and 8)
    "oft": lambda library: library.OFTConfig(r=0, oft_block_size=2, target_modules=["0", "2"]),
    "ia3": lambda library: library.IA3Config(target_modules=["0", "2"], feedforward_modules=["2"]),
}


class _Pair(NamedTuple):
    a: str
    b: str

    @property
    def id(self) -> str:
        return f"{self.a}+{self.b}"


_PAIRS = [_Pair("lora", "loha"), _Pair("lora", "lokr"), _Pair("loha", "oft"), _Pair("lora", "ia3")]


def _net(device: str) -> nn.Module:
    """The model every test adapts, ``Linear(6, 8)``, ``ReLU``, ``Linear(8, 4)``, the same each call."""
    return RECIPES[PeftType.LORA].base(_F32, device)


def _batch(device: str) -> lucid.Tensor:
    lucid.manual_seed(3)
    return lucid.randn(5, 6, device=device)


def _twin(ref: ModuleType, net: nn.Module) -> ModuleType:
    """The model in the reference framework, weights copied from ``net`` (no adapters on it)."""
    layers: list[ModuleType] = []
    for module in net:
        if isinstance(module, nn.Linear):
            layers.append(ref.nn.Linear(module.in_features, module.out_features))
        else:
            assert isinstance(module, nn.ReLU)
            layers.append(ref.nn.ReLU())
    twin = ref.nn.Sequential(*layers)
    with ref.no_grad():
        for mine, theirs in zip(net, twin, strict=True):
            if isinstance(mine, nn.Linear):
                theirs.weight.copy_(across(ref, mine.weight))
                theirs.bias.copy_(across(ref, mine.bias))
    return twin


def _move(ref: ModuleType, peft: ModuleType, seed: int) -> None:
    """Move every adapter tensor, trainable or not, well off its start, as ``perturb_adapter`` does.

    A LoRA ``B`` starts at zero, and an adapter added beside the first is frozen: both would stay at
    their start if only trainable tensors moved.  The base layers' own weights stay.
    """
    generator = ref.Generator().manual_seed(seed)
    with ref.no_grad():
        for name, p in peft.named_parameters():
            if "base_layer" not in name:
                p.add_(0.3 * ref.randn(tuple(p.shape), generator=generator))


def _write(ref: ModuleType, library: ModuleType, family: str, directory: Path, seed: int) -> None:
    """The reference writes ``family``'s adapter, moved off its start, as a plain adapter file."""
    peft = library.get_peft_model(_twin(ref, _net("cpu")), _FAMILIES[family](library))
    _move(ref, peft, seed)
    peft.save_pretrained(directory, safe_serialization=True)


def _open_reference(ref: ModuleType, library: ModuleType, directory: Path, name: str) -> ModuleType:
    """The reference's mixed model holding the adapter in ``directory`` as ``name``.

    ``from_pretrained`` is the way in unless the reference refuses the file's method there (OFT: in
    the constructor's list, not in the mapping ``from_pretrained`` reads); the constructor then
    builds the adapter from the file's config and ``load_adapter`` puts the file's values in.
    """
    twin = _twin(ref, _net("cpu"))
    try:
        mixed = library.PeftMixedModel.from_pretrained(twin, str(directory), adapter_name=name)
    except ValueError as refused:
        if "not supported for mixed models" not in str(refused):
            raise
        config = library.PeftConfig.from_pretrained(str(directory))
        mixed = library.PeftMixedModel(twin, config, name)
        mixed.load_adapter(str(directory), name)
    return mixed.eval()


def _active(first: str, second: str) -> dict[str, list[str]]:
    """What is in use for each output: each adapter alone, both, both listed the other way."""
    return {
        "first": [first],
        "second": [second],
        "both": [first, second],
        "both, the other way": [second, first],
    }


def _far(ref: ModuleType, a: ModuleType, b: ModuleType) -> float:
    return float(ref.sub(a, b).detach().abs().max())


def _smallest_effect(ref: ModuleType, clean: ModuleType, outputs: dict[str, ModuleType]) -> float:
    """The least any adapter moves the logits: alone off the base, and on top of the other."""
    return min(
        _far(ref, outputs["first"], clean),
        _far(ref, outputs["second"], clean),
        _far(ref, outputs["both"], outputs["first"]),
        _far(ref, outputs["both"], outputs["second"]),
    )


class _Run(NamedTuple):
    #: The largest difference from the reference over the four outputs; ``None`` if it refused.
    worst_gap: float | None
    #: The least any adapter moved the logits, on the side with the least.
    smallest_effect: float
    #: The reference's refusal, if it refused.
    refusal: str | None


def _compare(
    ref: ModuleType, library: ModuleType, device: str, pair: _Pair, reverse: bool, directory: Path
) -> _Run:
    first, second = (pair.b, pair.a) if reverse else (pair.a, pair.b)
    for seed, family in enumerate((pair.a, pair.b), start=7):
        _write(ref, library, family, directory / family, seed)
    x = _batch(device)
    clean = across(ref, _net(device)(x))
    mixed = PeftMixedModel.from_pretrained(_net(device), directory / first, adapter_name=first)
    mixed.load_adapter(directory / second, second)
    mine: dict[str, lucid.Tensor] = {}
    for label, active in _active(first, second).items():
        mixed.set_adapter(active)
        with lucid.no_grad():
            mine[label] = mixed(x)
    effect = _smallest_effect(ref, clean, {k: across(ref, v) for k, v in mine.items()})
    try:
        theirs = _reference_outputs(ref, library, x, directory, first, second)
    except ValueError as refused:
        if "not compatible with the `PeftMixedModel`" not in str(refused):
            raise
        return _Run(None, effect, str(refused))
    effect = min(effect, _smallest_effect(ref, clean, theirs))
    return _Run(max(gap(ref, mine[label], theirs[label]) for label in mine), effect, None)


def _reference_outputs(
    ref: ModuleType, library: ModuleType, x: lucid.Tensor, directory: Path, first: str, second: str
) -> dict[str, ModuleType]:
    mixed = _open_reference(ref, library, directory / first, first)
    mixed.load_adapter(str(directory / second), second)
    outputs: dict[str, ModuleType] = {}
    for label, active in _active(first, second).items():
        mixed.set_adapter(active)
        with ref.no_grad():
            outputs[label] = mixed(across(ref, x))
    return outputs


@pytest.mark.parity
@pytest.mark.parametrize("reverse", [False, True], ids=["as-listed", "reversed"])
@pytest.mark.parametrize("pair", _PAIRS, ids=lambda p: p.id)
def test_two_files_stacked_compute_what_the_reference_computes(
    ref: ModuleType, device: str, pair: _Pair, reverse: bool, tmp_path: Path
) -> None:
    run = _compare(ref, adapter_library(), device, pair, reverse, tmp_path)
    assert run.smallest_effect >= _SIGNAL, f"an adapter moved the logits by {run.smallest_effect}"
    if run.worst_gap is None:
        assert {pair.a, pair.b} & _REFERENCE_REFUSES, run.refusal
        return
    assert run.worst_gap < _TOLERANCE, f"{pair.id}: {run.worst_gap}"


def _drive(ref: ModuleType, mixed: ModuleType, x: ModuleType, names: list[str]) -> ModuleType:
    mixed.set_adapter(names)
    with ref.no_grad():
        return mixed(x)


@pytest.mark.parity
def test_a_reload_of_the_first_file_after_stacking_is_ignored_there_and_loaded_here(
    ref: ModuleType, device: str, tmp_path: Path
) -> None:
    library = adapter_library()
    for seed, name, family in ((7, "lora", "lora"), (9, "newer", "lora"), (8, "loha", "loha")):
        _write(ref, library, family, tmp_path / name, seed)
    x = _batch(device)
    xs = across(ref, x)
    both = ["lora", "loha"]

    # the reference: the second file of the first's method, loaded before stacking, is the answer
    wanted = _open_reference(ref, library, tmp_path / "newer", "lora")
    wanted.load_adapter(str(tmp_path / "loha"), "loha")
    want_alone, want_both = _drive(ref, wanted, xs, ["lora"]), _drive(ref, wanted, xs, both)

    # the reference: loaded after the stacking, it is not
    theirs = _open_reference(ref, library, tmp_path / "lora", "lora")
    theirs.load_adapter(str(tmp_path / "loha"), "loha")
    before_alone, before_both = _drive(ref, theirs, xs, ["lora"]), _drive(ref, theirs, xs, both)
    assert _far(ref, before_alone, want_alone) >= _SIGNAL  # the two files really differ
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = theirs.load_adapter(str(tmp_path / "newer"), "lora")
    assert not [str(w.message) for w in caught if "keys" in str(w.message)]
    assert result.unexpected_keys and result.missing_keys
    assert _far(ref, _drive(ref, theirs, xs, ["lora"]), before_alone) < _TOLERANCE
    assert _far(ref, _drive(ref, theirs, xs, both), before_both) < _TOLERANCE

    # here: the same calls load the file
    mixed = PeftMixedModel.from_pretrained(_net(device), tmp_path / "lora", adapter_name="lora")
    mixed.load_adapter(tmp_path / "loha", "loha")
    mixed.set_adapter(["lora"])
    with lucid.no_grad():
        mine_before = mixed(x)
    loaded = mixed.load_adapter(tmp_path / "newer", "lora")
    assert loaded.unexpected_keys == []  # every key of the file found its tensor
    for names, expected in ((["lora"], want_alone), (both, want_both)):
        mixed.set_adapter(names)
        with lucid.no_grad():
            assert gap(ref, mixed(x), expected) < _TOLERANCE, names
    assert gap(ref, mine_before, want_alone) >= _SIGNAL  # it was not the file's values before


@pytest.mark.parity
def test_merging_one_of_two_active_adapters_drops_the_other_from_the_reference_forward_pass(
    ref: ModuleType,
) -> None:
    """LCD-971, on the reference: one layer, two adapters in use, one merged.

    Pins the reference only.  Lucid departs from it on purpose: a merged layer there goes on
    applying the active adapters it did not fold, so merging one of two does not change the
    function (``test_multi_adapter_merge_contract``, "merging some of the active adapters").
    """
    library = adapter_library()
    lucid.manual_seed(0)
    net = nn.Sequential(nn.Linear(6, 4))
    x = across(ref, _batch("cpu"))
    clean = _twin(ref, net)(x)
    config = library.LoraConfig(r=2, target_modules=["0"])
    peft = library.get_peft_model(_twin(ref, net), config, adapter_name="one")
    peft.add_adapter("two", library.LoraConfig(r=2, target_modules=["0"]))
    _move(ref, peft, seed=3)
    peft.eval()

    def logits(names: list[str]) -> ModuleType:
        peft.base_model.set_adapter(names)
        with ref.no_grad():
            return peft(x)

    one, two, both = logits(["one"]), logits(["two"]), logits(["one", "two"])
    assert min(_far(ref, one, clean), _far(ref, two, clean), _far(ref, both, one)) >= _SIGNAL
    peft.base_model.merge_adapter(adapter_names=["one"])
    with ref.no_grad():  # no ``set_adapter``: on a merged model it unmerges first
        after = peft(x)
    assert peft.base_model.active_adapters == ["one", "two"]
    assert peft.base_model.model[0].merged_adapters == ["one"]
    assert _far(ref, after, one) < _TOLERANCE  # `two` is gone from the forward pass
    assert _far(ref, after, both) >= _SIGNAL  # by its whole effect


@pytest.mark.parity
@pytest.mark.parametrize("peft_type", list(PEFT_TYPE_TO_MIXED_MODEL_MAPPING), ids=lambda t: t.name)
def test_every_method_either_reference_door_takes_starts_and_stacks_here(
    peft_type: PeftType, tmp_path: Path
) -> None:
    """The reference's constructor takes six methods and its ``from_pretrained`` mapping nine (OFT
    only at the first door; IA3, BEFT, HIRA and ROAD only at the second, where the constructor then
    refuses them).  This library's mapping holds both lists' union, and each of its ten methods
    starts a mixed model and takes another method's adapter on top."""
    library = adapter_library()
    constructor = {t.value for t in pytest.importorskip("peft.tuners.mixed").COMPATIBLE_TUNER_TYPES}
    by_file = {t.value for t in library.mapping.PEFT_TYPE_TO_MIXED_MODEL_MAPPING}
    assert constructor | by_file <= {t.value for t in PEFT_TYPE_TO_MIXED_MODEL_MAPPING}
    recipe = RECIPES[peft_type]
    mixed = PeftMixedModel(recipe.base(_F32, "cpu"), recipe.config(_F32, "cpu", tmp_path), "first")
    later = (
        LoHaConfig(r=2, target_modules=["0", "2"])
        if peft_type is PeftType.LORA
        else LoraConfig(r=2, target_modules=["0", "2"])
    )
    mixed.add_adapter("later", later)
    assert sorted(mixed.peft_config) == ["first", "later"]
