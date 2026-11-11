"""Stackable layers: one method's ``Linear`` layer standing on another method's, on one ``nn.Linear``.

A tuner layer takes ``base_layer`` to be the user's module.  Two methods on one module need the layer
of the second built around the layer of the first, and that is safe only for a layer that calls
``self.base_layer(x)`` once and changes the weights additively or by a transform of them.  A layer that
computes from ``self.weight`` (BOFT, HRA, MiSS, BONE, OFT's convolution) would run, and the layer
beneath it would be silently ignored; one that reads the weight (HiRA, DoRA) merges into a weight the
layers beneath it have changed.  So standing on a layer is a declaration of the class (``_stackable``),
refused otherwise, and this file holds each declaration to the contract by building every ordered pair
by hand, with no host:

* the forward pass equals the layers' merged values applied to a plain ``nn.Linear`` one after the other
  (a dense probe: no stacked forward is involved), and equals the model merged, inner layer first;
* unmerging, outer layer first, gives the weights back, and the module the layers stood on is the user's;
* gradients reach the adapters of both layers and none of the base.

A class that does not pass is not declared, and a class nobody has decided on is refused, so the sweep at
the end reads the whole package.
"""

import copy
import inspect
import itertools
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    AdaLoraConfig,
    BeftConfig,
    BOFTConfig,
    BoneConfig,
    HiraConfig,
    HRAConfig,
    IA3Config,
    LNTuningConfig,
    LoHaConfig,
    LoKrConfig,
    MissConfig,
    OFTConfig,
    RoadConfig,
    ShiraConfig,
)
from lucid.peft.tuners import (
    adalora,
    beft,
    boft,
    bone,
    hira,
    hra,
    ia3,
    ln_tuning,
    loha,
    lokr,
    lora,
    miss,
    oft,
    road,
    shira,
)
from lucid.peft.tuners._layer import _BaseTunerLayer, _qualified, _tuner_layers
from lucid.peft.tuners.lora._inits import _Init
from lucid.test.unit.peft._tuner_recipes import finite, perturb_adapter, same, span

_IN, _OUT = 8, 6
#: One float32 sum of a few products on values of order one; Metal measures under a tenth of it.
_TOLERANCE = 1e-5
_TARGETS = ["x"]

type _Build = Callable[[nn.Module, str], _BaseTunerLayer]


@dataclass(frozen=True)
class _Method:
    """A layer class and how to build one, with its adapter, on a given base."""

    name: str
    layer: type[_BaseTunerLayer]
    build: _Build


def _lora(base: nn.Module, name: str) -> _BaseTunerLayer:
    return lora.Linear(base, name, r=2, lora_alpha=4)


def _loha(base: nn.Module, name: str) -> _BaseTunerLayer:
    return loha.Linear(base, name, LoHaConfig(target_modules=_TARGETS), r=2, alpha=4)


def _lokr(base: nn.Module, name: str) -> _BaseTunerLayer:
    return lokr.Linear(base, name, LoKrConfig(target_modules=_TARGETS), r=2, alpha=4)


def _adalora(base: nn.Module, name: str) -> _BaseTunerLayer:
    config = AdaLoraConfig(total_step=10, target_modules=_TARGETS)
    return adalora.SVDLinear(base, name, config, r=3, lora_alpha=6)


def _shira(base: nn.Module, name: str) -> _BaseTunerLayer:
    return shira.Linear(base, name, ShiraConfig(r=1, random_seed=3, target_modules=_TARGETS))


def _oft(base: nn.Module, name: str) -> _BaseTunerLayer:
    return oft.Linear(base, name, OFTConfig(oft_block_size=4, target_modules=_TARGETS))


def _ia3(base: nn.Module, name: str) -> _BaseTunerLayer:
    return ia3.Linear(base, name, IA3Config(target_modules=_TARGETS))


def _ia3_feedforward(base: nn.Module, name: str) -> _BaseTunerLayer:
    return ia3.Linear(base, name, IA3Config(target_modules=_TARGETS), is_feedforward=True)


def _road(base: nn.Module, name: str) -> _BaseTunerLayer:
    return road.Linear(base, name, RoadConfig(group_size=2, target_modules=_TARGETS))


def _beft(base: nn.Module, name: str) -> _BaseTunerLayer:
    return beft.Linear(base, name, BeftConfig(target_modules=_TARGETS))


def _hira(base: nn.Module, name: str) -> _BaseTunerLayer:
    return hira.Linear(base, name, HiraConfig(r=2, target_modules=_TARGETS))


#: The nine stackable classes, the IA3 one in both its forms (it scales the input or the output).
_METHODS = [
    _Method("lora", lora.Linear, _lora),
    _Method("loha", loha.Linear, _loha),
    _Method("lokr", lokr.Linear, _lokr),
    _Method("adalora", adalora.SVDLinear, _adalora),
    _Method("shira", shira.Linear, _shira),
    _Method("oft", oft.Linear, _oft),
    _Method("ia3", ia3.Linear, _ia3),
    _Method("ia3-feedforward", ia3.Linear, _ia3_feedforward),
    _Method("road", road.Linear, _road),
    _Method("beft", beft.Linear, _beft),
]
#: HiRA reads the weight, so it may be the layer at the bottom and nothing else: no row of ``_METHODS``.
_HIRA = _Method("hira", hira.Linear, _hira)
#: What may stand beneath a layer here: any stackable layer, and HiRA.
_INNERS = [*_METHODS, _HIRA]

#: The classes that may be built on a layer, as ``_qualified`` spells them.
_DECLARED = frozenset(
    {
        "lora._layer.Linear",
        "loha._layer.Linear",
        "lokr._layer.Linear",
        "adalora._layer.SVDLinear",
        "shira._layer.Linear",
        "oft._layer.Linear",
        "ia3._layer.Linear",
        "road._layer.Linear",
        "beft._layer.Linear",
    }
)


class _Stack(NamedTuple):
    base: nn.Linear
    inner: _BaseTunerLayer
    outer: _BaseTunerLayer
    x: lucid.Tensor


def _stack(outer: _Method, inner: _Method, device: str) -> _Stack:
    """``outer``'s layer on ``inner``'s on a frozen ``nn.Linear``, both adapters moved well off their starts."""
    lucid.manual_seed(0)
    base = nn.Linear(_IN, _OUT).to(device)
    base.requires_grad_(False)
    low = inner.build(base, "i")
    high = outer.build(low, "o")
    perturb_adapter(low, "i", seed=1)
    perturb_adapter(high, "o", seed=2)
    return _Stack(base, low, high, lucid.randn(4, _IN, device=device))


def _chain(top: _BaseTunerLayer) -> list[_BaseTunerLayer]:
    """The layers from the one standing on the user's module up to ``top``."""
    layers = [top]
    while isinstance(below := layers[-1].base_layer, _BaseTunerLayer):
        layers.append(below)
    return layers[::-1]


def _own_adapter(layer: _BaseTunerLayer) -> str:
    (name,) = layer._adapter_names()
    return name


def _dense_probe(top: _BaseTunerLayer, x: lucid.Tensor) -> lucid.Tensor:
    """``x`` through a plain ``nn.Linear`` that takes each layer's merged values in turn, the lowest first.

    The stack is copied so the layers read the copy's weights; nothing of the stacked forward pass runs.
    """
    twin = copy.deepcopy(top)
    probe = twin._root()
    with lucid.no_grad():
        for layer in _chain(twin):
            for key, value in layer._merged_values(_own_adapter(layer)).items():
                getattr(probe, key.rpartition(".")[2]).copy_(value)
        out: lucid.Tensor = probe(x)
    return out


def _counted(module: nn.Module) -> list[int]:
    """A counter of how many times ``module``'s forward has run."""
    calls = [0]

    def count(_module: nn.Module, _args: tuple[object, ...], _out: object) -> None:
        calls[0] += 1

    module.register_forward_hook(count)
    return calls


_PAIRS = [
    pytest.param(outer, inner, id=f"{outer.name}-on-{inner.name}")
    for outer, inner in itertools.product(_METHODS, _INNERS)
]


@pytest.mark.filterwarnings("ignore:Unmerge result can be inaccurate:UserWarning")
@pytest.mark.parametrize(("outer", "inner"), _PAIRS)
class TestPairs:
    def test_every_level_reads_the_users_module_through_the_root(
        self, outer: _Method, inner: _Method, device: str
    ) -> None:
        base, low, high, _ = _stack(outer, inner, device)
        assert high._root() is base and low._root() is base
        assert high.get_base_layer() is low  # one step
        assert low.get_base_layer() is base
        assert high._base_path() == "base_layer.base_layer" and low._base_path() == "base_layer"
        users = {id(base.weight), id(base.bias)}
        for layer in (low, high):
            targets = layer._merge_targets()
            assert targets and {id(p) for p in targets.values()} <= users
            assert all(key.startswith(layer._base_path() + ".") for key in targets)
            assert set(layer._merged_values(_own_adapter(layer))) == set(targets)

    def test_the_forward_is_the_layers_merged_values_on_a_plain_linear(
        self, outer: _Method, inner: _Method, device: str
    ) -> None:
        _, _, high, x = _stack(outer, inner, device)
        with lucid.no_grad():
            adapted = high(x)
        assert span(adapted, _dense_probe(high, x)) < _TOLERANCE
        with lucid.no_grad():  # the adapters do something beyond what the base gives
            high.enable_adapters(False)
            assert span(adapted, high(x)) > 1e-3

    def test_each_layer_calls_the_layer_beneath_it_once(
        self, outer: _Method, inner: _Method, device: str
    ) -> None:
        base, low, high, x = _stack(outer, inner, device)
        base_calls, inner_calls = _counted(base), _counted(low)
        with lucid.no_grad():
            high(x)
        assert (inner_calls[0], base_calls[0]) == (1, 1)

    def test_merging_the_inner_layer_then_the_outer_is_the_forward_pass(
        self, outer: _Method, inner: _Method, device: str
    ) -> None:
        _, low, high, x = _stack(outer, inner, device)
        with lucid.no_grad():
            adapted = high(x)
            low.merge()
            high.merge()
            assert low.merged and high.merged
            assert span(high(x), adapted) < _TOLERANCE

    def test_unmerging_the_outer_layer_then_the_inner_gives_the_weights_back(
        self, outer: _Method, inner: _Method, device: str
    ) -> None:
        base, low, high, x = _stack(outer, inner, device)
        weight, bias = base.weight.detach().clone(), base.bias.detach().clone()
        with lucid.no_grad():
            adapted = high(x)
            low.merge()
            high.merge()
            high.unmerge()
            low.unmerge()
            assert not (low.merged or high.merged)
            assert span(base.weight, weight) < _TOLERANCE and span(base.bias, bias) < _TOLERANCE
            assert span(high(x), adapted) < _TOLERANCE

    def test_unwrapping_returns_the_users_own_linear_with_the_weights_it_had(
        self, outer: _Method, inner: _Method, device: str
    ) -> None:
        base, low, high, _ = _stack(outer, inner, device)
        weight = base.weight.detach().clone()
        with lucid.no_grad():
            low.merge()
            high.merge()
            assert high._unwrap(keep_trained=False) is base
            assert not (low.merged or high.merged)
            assert span(base.weight, weight) < _TOLERANCE

    def test_unwrapping_a_merged_stack_keeps_what_was_merged_when_asked(
        self, outer: _Method, inner: _Method, device: str
    ) -> None:
        base, low, high, x = _stack(outer, inner, device)
        with lucid.no_grad():
            adapted = high(x)
            low.merge()
            high.merge()
            assert high._unwrap(keep_trained=True) is base
            assert span(base(x), adapted) < _TOLERANCE

    def test_gradients_reach_the_adapters_of_both_layers_and_not_the_base(
        self, outer: _Method, inner: _Method, device: str
    ) -> None:
        base, low, high, x = _stack(outer, inner, device)
        high(x).sum().backward()
        for layer in (low, high):
            grads = [p.grad for p in layer._adapter_parameters(_own_adapter(layer))]
            assert grads and all(g is not None and finite(g) for g in grads)
        assert base.weight.grad is None and base.bias.grad is None


@pytest.mark.filterwarnings("ignore:Unmerge result can be inaccurate:UserWarning")
@pytest.mark.parametrize("outer", _METHODS, ids=lambda m: m.name)
class TestAHiraLayerBeneath:
    def test_unmerging_gives_the_weight_back_bit_for_bit(self, outer: _Method, device: str) -> None:
        """HiRA stashes the weight it merges into, so the layers above it cannot leave a trace in it."""
        base, low, high, _ = _stack(outer, _HIRA, device)
        weight = base.weight.detach().clone()
        with lucid.no_grad():
            low.merge()
            high.merge()
            high.unmerge()
            low.unmerge()
        assert same(base.weight, weight)

    def test_a_layer_above_it_reads_the_weight_it_has_merged(
        self, outer: _Method, device: str
    ) -> None:
        """The order is the contract: the inner layer merges first, so the outer one folds into its result."""
        base, low, high, x = _stack(outer, _HIRA, device)
        with lucid.no_grad():
            adapted = high(x)
            weight = base.weight.detach().clone()
            low.merge()
            assert span(base.weight, weight) > 1e-3  # HiRA moved the weight the outer layer reads
            high.merge()
            assert span(high(x), adapted) < _TOLERANCE


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=["float16", "bfloat16"])
@pytest.mark.parametrize(("outer", "inner"), _PAIRS)
class TestHalfPrecision:
    def test_the_tuners_widening_reaches_the_layer_beneath(
        self, outer: _Method, inner: _Method, dtype: lucid.dtype, device: str
    ) -> None:
        lucid.manual_seed(0)
        base = nn.Linear(_IN, _OUT).to(device).to(dtype)
        base.requires_grad_(False)
        low = inner.build(base, "i")
        high = outer.build(low, "o")
        high.to(dtype)  # adapters left in half precision, as a layer that allocated them so would
        halved = [
            p for layer in (low, high) for p in layer._adapter_parameters(_own_adapter(layer))
        ]
        assert {p.dtype for p in halved} == {dtype}
        # What ``_BaseTuner._inject`` does for each adapter it builds: every layer in the tree.
        for _, layer in _tuner_layers(high):
            layer._widen_adapter("i")
            layer._widen_adapter("o")
        widened = [
            p for layer in (low, high) for p in layer._adapter_parameters(_own_adapter(layer))
        ]
        assert {p.dtype for p in widened} == {lucid.float32}
        assert all(a is b for a, b in zip(halved, widened, strict=True))  # the Parameters stay
        out = high(lucid.randn(4, _IN, device=device).to(dtype))
        assert out.dtype == dtype and finite(out)


# ── the classes that may not stand on a layer ─────────────────────────────


@dataclass(frozen=True)
class _Refused:
    """A class that computes from, or reads, the base weight; how to build one and a module of its own kind."""

    name: str
    layer: type[_BaseTunerLayer]
    build: _Build
    own: Callable[[], nn.Module]


def _boft(base: nn.Module, name: str) -> _BaseTunerLayer:
    return boft.Linear(base, name, BOFTConfig(target_modules=_TARGETS))


def _hra(base: nn.Module, name: str) -> _BaseTunerLayer:
    return hra.Linear(base, name, HRAConfig(target_modules=_TARGETS))


def _oft_conv(base: nn.Module, name: str) -> _BaseTunerLayer:
    return oft.Conv2d(base, name, OFTConfig(oft_block_size=2, target_modules=_TARGETS))


def _miss(base: nn.Module, name: str) -> _BaseTunerLayer:
    return miss.Linear(base, name, MissConfig(r=2, target_modules=_TARGETS))


def _bone(base: nn.Module, name: str) -> _BaseTunerLayer:
    return bone.Linear(base, name, BoneConfig(r=2, target_modules=_TARGETS))


def _lora_conv(base: nn.Module, name: str) -> _BaseTunerLayer:
    return lora.Conv2d(base, name, r=2)


def _lora_embedding(base: nn.Module, name: str) -> _BaseTunerLayer:
    return lora.Embedding(base, name, r=2)


def _linear() -> nn.Module:
    return nn.Linear(_IN, _OUT)


def _conv() -> nn.Module:
    return nn.Conv2d(4, _OUT, 2)


def _embedding() -> nn.Module:
    return nn.Embedding(10, _OUT)


_REFUSED = [
    _Refused("boft", boft.Linear, _boft, _linear),
    _Refused("hra", hra.Linear, _hra, _linear),
    _Refused("oft-conv2d", oft.Conv2d, _oft_conv, _conv),
    _Refused("miss", miss.Linear, _miss, _linear),
    _Refused("bone", bone.Linear, _bone, _linear),
    _Refused("hira", hira.Linear, _hira, _linear),
    _Refused("lora-conv2d", lora.Conv2d, _lora_conv, _conv),
    _Refused("lora-embedding", lora.Embedding, _lora_embedding, _embedding),
]


@pytest.mark.parametrize("refused", _REFUSED, ids=lambda r: r.name)
class TestRefusedClasses:
    def test_it_builds_on_the_module_of_its_own_kind(self, refused: _Refused) -> None:
        base = refused.own()
        layer = refused.build(base, "x")
        assert type(layer) is refused.layer and layer.base_layer is base and layer._root() is base

    @pytest.mark.parametrize("inner", _INNERS, ids=lambda m: m.name)
    def test_it_is_refused_a_layer_beneath_it_and_names_itself(
        self, refused: _Refused, inner: _Method
    ) -> None:
        base = nn.Linear(_IN, _OUT)
        below = inner.build(base, "i")
        before = list(below.state_dict())
        with pytest.raises(TypeError, match=re.escape(_qualified(refused.layer))) as caught:
            refused.build(below, "x")
        assert _qualified(inner.layer) in str(caught.value) and "stackable" in str(caught.value)
        assert list(below.state_dict()) == before and below.base_layer is base


class TestWhyMissIsRefusedWhole:
    """MiSS adds to the base output in its default and ``mini`` forms, so a pair of them would pass; ``bat`` does not."""

    def test_its_bat_form_computes_from_the_weight_and_drops_the_layer_beneath_it(
        self, monkeypatch: pytest.MonkeyPatch, device: str
    ) -> None:
        monkeypatch.setattr(miss.Linear, "_stackable", True)
        base = nn.Linear(_IN, _OUT).to(device)
        base.requires_grad_(False)
        low = _lora(base, "i")
        high = miss.Linear(low, "o", MissConfig(r=2, init_weights="bat", target_modules=_TARGETS))
        perturb_adapter(low, "i", seed=1)
        perturb_adapter(high, "o", seed=2)
        x = lucid.randn(4, _IN, device=device)
        with lucid.no_grad():
            beneath_on = high(x)
            low.enable_adapters(False)
            beneath_off = high(x)
        assert same(beneath_on, beneath_off)  # the LoRA layer is neither run nor missed


class TestAStackableClassKeepsItsOtherChecks:
    def test_the_kind_is_read_off_the_root(self) -> None:
        convolution = nn.Conv2d(4, _OUT, 2)
        below = lora.Conv2d(convolution, "i", r=2)
        with pytest.raises(TypeError, match="adapts nn.Linear.*got Conv2d"):
            lora.Linear(below, "o", r=2)

    def test_dora_reads_the_weight_so_it_must_stand_on_the_users_module(self) -> None:
        base = nn.Linear(_IN, _OUT)
        below = lora.Linear(base, "i", r=2)
        with pytest.raises(ValueError, match="use_dora=True reads the base weight"):
            lora.Linear(below, "o", r=2, use_dora=True)
        assert lora.Linear(base, "d", r=2, use_dora=True).use_dora == {"d": True}

    def test_a_subclass_of_a_stackable_class_is_refused_until_it_declares_itself(self) -> None:
        class Plain(lora.Linear):
            pass

        class Declared(lora.Linear):
            _stackable = True

        assert lora.Linear._stackable and Declared._stackable
        assert not Plain._stackable and not lora.QuantizedLinear._stackable
        assert not Plain._stands_on(lora.Linear(nn.Linear(_IN, _OUT), "i", r=2))

    @pytest.mark.parametrize(
        "spelling", [cls.spelling for cls in _Init._strategies.values() if cls.factorises]
    )
    @pytest.mark.parametrize("inner", _INNERS, ids=lambda m: m.name)
    def test_a_factorising_init_would_rewrite_the_weight_beneath_so_it_is_refused(
        self, spelling: str, inner: _Method, device: str
    ) -> None:
        base = nn.Linear(_IN, _OUT).to(device)
        base.requires_grad_(False)
        below = inner.build(base, "i")
        perturb_adapter(below, "i", seed=1)
        weight, bias = base.weight.detach().clone(), base.bias.detach().clone()
        state = {key: value.detach().clone() for key, value in below.state_dict().items()}
        with pytest.raises(ValueError, match="writes the residual back"):
            lora.Linear(below, "o", r=2, init_lora_weights=spelling)
        assert same(base.weight, weight) and same(base.bias, bias)
        assert list(below.state_dict()) == list(state)
        assert all(same(value, state[key]) for key, value in below.state_dict().items())

    def test_the_init_registry_has_factorising_spellings_to_hold_to_that(self) -> None:
        spellings = [cls.spelling for cls in _Init._strategies.values() if cls.factorises]
        assert {"pissa", "olora", "lora_ga"} <= set(spellings)


# ── nothing changes for a layer on the user's own module ─────────────────


@pytest.mark.parametrize("method", _METHODS + [_HIRA], ids=lambda m: m.name)
class TestALayerOnTheUsersModule:
    def test_the_base_is_the_root_and_the_keys_are_the_ones_they_were(
        self, method: _Method
    ) -> None:
        base = nn.Linear(_IN, _OUT)
        layer = method.build(base, "a")
        assert layer._root() is base and layer.get_base_layer() is base
        assert layer._base_path() == "base_layer"
        assert set(layer._merge_targets()) <= {"base_layer.weight", "base_layer.bias"}
        assert "base_layer.weight" in layer._merge_targets() or method.layer is beft.Linear

    def test_unwrapping_hands_the_module_back(self, method: _Method) -> None:
        base = nn.Linear(_IN, _OUT)
        assert method.build(base, "a")._unwrap(keep_trained=False) is base


class _Gate(nn.Module):
    """A module with two raw parameters, each adapted by its own wrapper."""

    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Parameter(lucid.randn(4, 6))
        self.second = nn.Parameter(lucid.randn(4, 6))

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        out: lucid.Tensor = x @ self.first.mT + x @ self.second.mT
        return out


class TestWrappersOfOneModuleNestAsBefore:
    def test_a_wrapper_per_parameter_still_stands_on_the_last(self) -> None:
        gate = _Gate()
        inner = lora.ParamWrapper(gate, "a", "first", r=2, lora_alpha=4)
        outer = lora.ParamWrapper(inner, "a", "second", r=2, lora_alpha=4)
        assert outer._root() is gate and outer.get_base_layer() is gate
        assert outer._base_path() == "base_layer.base_layer"
        assert set(outer._merge_targets()) == {"base_layer.base_layer.second"}
        assert outer._unwrap(keep_trained=False) is gate

    def test_a_wrapper_may_not_stand_on_a_layer_of_a_module(self) -> None:
        below = lora.Linear(nn.Linear(_IN, _OUT), "i", r=2)
        with pytest.raises(TypeError, match=re.escape(_qualified(lora.ParamWrapper))):
            lora.ParamWrapper(below, "o", "weight", r=2, lora_alpha=4)

    def test_a_layer_that_adapts_any_module_is_no_wrapper_and_may_not_nest(self) -> None:
        config = LNTuningConfig(target_modules=_TARGETS)
        below = ln_tuning.LNTuningLayer(nn.LayerNorm(_OUT), "a", config)
        with pytest.raises(TypeError, match=re.escape(_qualified(ln_tuning.LNTuningLayer))):
            ln_tuning.LNTuningLayer(below, "b", config)


# ── the sweep: no class is left undecided ─────────────────────────────────


def _layer_classes() -> list[type[_BaseTunerLayer]]:
    """Every ``_BaseTunerLayer`` subclass in ``lucid.peft``, abstract ones too."""
    found: list[type[_BaseTunerLayer]] = []
    pending: list[type[_BaseTunerLayer]] = [_BaseTunerLayer]
    while pending:
        for sub in pending.pop().__subclasses__():
            if sub not in found:
                found.append(sub)
                pending.append(sub)
    return [cls for cls in found if cls.__module__.startswith("lucid.peft.")]


_CLASSES = _layer_classes()
_DECIDED = [*(m.layer for m in _METHODS), *(refused.layer for refused in _REFUSED)]


class TestEveryLayerClassIsDecided:
    def test_the_sweep_finds_the_classes_the_pair_tests_build(self) -> None:
        assert set(_DECIDED) <= set(_CLASSES)

    def test_the_stackable_classes_are_the_declared_ones(self) -> None:
        assert {_qualified(cls) for cls in _CLASSES if cls._stackable} == _DECLARED
        assert {m.layer for m in _METHODS} == {cls for cls in _CLASSES if cls._stackable}

    @pytest.mark.parametrize("cls", _CLASSES, ids=_qualified)
    def test_a_class_is_refused_a_layer_unless_it_is_declared(
        self, cls: type[_BaseTunerLayer]
    ) -> None:
        """A new layer class lands refused; to stack, add it to ``_METHODS`` and ``_DECLARED`` and pass the pairs."""
        below = lora.Linear(nn.Linear(_IN, _OUT), "i", r=2)
        declared = _qualified(cls) in _DECLARED
        assert cls._stackable is declared
        assert cls._stands_on(below) is declared  # a layer of another class

    @pytest.mark.parametrize(
        "cls", [c for c in _CLASSES if not inspect.isabstract(c)], ids=_qualified
    )
    def test_a_class_is_refused_a_layer_of_its_own_class_too(
        self, cls: type[_BaseTunerLayer]
    ) -> None:
        """Only ``ParamWrapper`` nests on its own kind (a wrapper per parameter); a class that is not abstract is probed with an instance made without its constructor."""
        own = object.__new__(cls)
        nests = cls is lora.ParamWrapper
        assert cls._stands_on(own) is (_qualified(cls) in _DECLARED or nests)
