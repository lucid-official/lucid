"""``MixedModel``: the class-wide contract of a model that holds adapters of several methods.

A method is mixable when its tuner says so (``mixed=True``), and what the declaration promises is
what these tests hold every declared method to, against every other, so a method added later is
checked without anyone remembering to:

* **The partition.**  Every registered method is declared mixable or is listed in ``_NOT_MIXED``
  with its reason, and a method outside the declared set is refused where it is added, with the
  model as it was.
* **The pair contract.**  Every ordered pair of declared methods (the one below may be of the
  innermost-only kind, the one above must stand on a layer), a few three-deep stacks, both orders
  of ``set_adapter``, on the CPU and on Metal: the stack is in the order the adapters were added;
  the forward pass equals a dense probe (a plain ``nn.Linear`` that takes each layer's merged values
  in turn, lowest first — no stacked forward is involved); the adapter of the layer below alone is
  the single-method model; ``merge_adapter`` is the forward pass and ``unmerge_adapter`` gives the
  weights back; ``merge_and_unload`` is the forward pass; ``disable_adapter_layers`` is the base.
  Each layer's own effect on the output is far above the tolerance, so an equality that held
  because a layer did nothing would fail.
* **What a stack refuses**, all before the model is touched.
* **Half precision and placement**: an adapter is float32 beside a half-precision base, and is on
  the base weight's device whatever the default device is.
* **The files**: an adapter of a stack is filed as a single-method model would file it, and a
  single-method file loads into a stack before and after the stacking.
"""

import copy
import dataclasses
import inspect
import itertools
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import NamedTuple

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    HiraConfig,
    IA3Config,
    LoHaConfig,
    LoraConfig,
    LoraModel,
    OFTConfig,
    PeftConfig,
    PeftType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from lucid.peft.tuners import MixedModel
from lucid.peft.tuners._base import _TUNERS
from lucid.peft.tuners._layer import _BaseTunerLayer, _tuner_layers
from lucid.peft.tuners._mixed import _chain, _family, _family_of
from lucid.peft.tuners._plan import _set_submodule
from lucid.peft.tuners._tuner import _MIXED_TUNERS, _BaseTuner
from lucid.test._default_device import default_device, metal_available
from lucid.test.unit.peft._tuner_recipes import RECIPES, magnitude, perturb_adapter, same, span

_F32 = lucid.float32
_SITES = ("0", "2")
#: A float32 sum of a few products on values of order one; Metal sums in another order, to ~1e-6.
_TOLERANCE = 1e-5
#: What each layer's own adapter must change the output by: 5000 times the tolerance (the smallest,
#: a HiRA layer's, is 0.097 on the CPU: its update is the weight times a product of two small matrices).
_EFFECT = 0.05

_REFUSAL = (ValueError, "is not compatible with the `PeftMixedModel`")

#: Every method registered and not declared ``mixed=True``, with why a stack has no place for it.
_PROMPT = (
    "prompt learning keeps one learned prompt beside the model and puts no layer in it to stack"
)
_NOT_MIXED: dict[PeftType, str] = {
    **dict.fromkeys(
        [
            PeftType.PROMPT_TUNING,
            PeftType.PREFIX_TUNING,
            PeftType.P_TUNING,
            PeftType.MULTITASK_PROMPT_TUNING,
            PeftType.CPT,
            PeftType.CARTRIDGE,
        ],
        _PROMPT,
    ),
    PeftType.XLORA: "mixes LoRA experts it loads by name through a classifier and holds one adapter",
    PeftType.ADAPTION_PROMPT: "attends to a prompt inside one attention module and holds one adapter",
    PeftType.LN_TUNING: "swaps a norm layer for a trained copy, so no layer stands on it and one adapter",
    PeftType.TRAINABLE_TOKENS: "rewrites the table rows it reads from its own base weight, so a layer beneath is dropped",
    PeftType.POLY: "routes low-rank skills per input, which no change of the weight holds",
    PeftType.LILY: "routes low-rank experts per input through tables the tuner shares",
    **dict.fromkeys(
        [
            PeftType.VERA,
            PeftType.PVERA,
            PeftType.RANDLORA,
            PeftType.VBLORA,
            PeftType.UNILORA,
            PeftType.TINYLORA,
            PeftType.FROD,
        ],
        "keeps a shared table or a settled basis in the tuner instance, which a mixed host lacks",
    ),
    **dict.fromkeys(
        [PeftType.C3A, PeftType.DELORA, PeftType.FOURIERFT, PeftType.GRALORA],
        "additive per-layer update not yet held to the pair contract",
    ),
    **dict.fromkeys(
        [PeftType.PSOFT, PeftType.OSF, PeftType.ADAMSS],
        "builds from the current weight's SVD, which a stack would change under it",
    ),
    **dict.fromkeys(
        [PeftType.SUPERTUNING, PeftType.WAVEFT],
        "additive update whose tuner overrides instance hooks a mixed host cannot call",
    ),
    **dict.fromkeys(
        [PeftType.DEFT, PeftType.GLORA, PeftType.PEANUT],
        "reads the weight like HiRA, so innermost only, and not yet held to the pair contract",
    ),
    **dict.fromkeys(
        [PeftType.BOFT, PeftType.HRA, PeftType.MISS, PeftType.BONE],
        "computes from the weight and not through its base layer, so a stack would drop the layer beneath",
    ),
}

_DECLARED = list(_MIXED_TUNERS)


def _linear_class(peft_type: PeftType) -> type[_BaseTunerLayer]:
    return _MIXED_TUNERS[peft_type]._layer_types[nn.Linear]


#: The declared methods whose ``Linear`` layer may be the lowest and nothing more (HiRA reads the weight).
_INNERMOST_ONLY = frozenset(p for p in _DECLARED if not _linear_class(p)._stackable)

type _Stack = tuple[PeftType, ...]

_PAIRS: list[_Stack] = [
    (inner, outer)
    for inner, outer in itertools.permutations(_DECLARED, 2)
    if outer not in _INNERMOST_ONLY
]
#: Three layers on a module: a reader of the weight at the bottom, a rotation, a scale and sums.
_TRIPLES: list[_Stack] = [
    (PeftType.HIRA, PeftType.LORA, PeftType.OFT),
    (PeftType.LORA, PeftType.LOHA, PeftType.IA3),
    (PeftType.BEFT, PeftType.ROAD, PeftType.LOKR),
    (PeftType.SHIRA, PeftType.ADALORA, PeftType.OFT),
    (PeftType.OFT, PeftType.IA3, PeftType.ROAD),
    (PeftType.HIRA, PeftType.SHIRA, PeftType.ROAD),
]
_STACKS = [*_PAIRS, *_TRIPLES]
_ORDERS = ["added", "reversed"]


def _id(stack: _Stack) -> str:
    return "-on-".join(p.name.lower() for p in reversed(stack))


def _name(peft_type: PeftType) -> str:
    return peft_type.name.lower()


def _config(
    peft_type: PeftType, device: str, scratch: Path, dtype: lucid.dtype = _F32
) -> PeftConfig:
    return RECIPES[peft_type].config(dtype, device, scratch)


def _call(model: nn.Module, device: str, dtype: lucid.dtype = _F32) -> lucid.Tensor:
    return RECIPES[PeftType.LORA].call(model, dtype, device)


def _move_off_the_start(base: nn.Module, name: str, seed: int) -> None:
    """Move ``name``'s tensors well off their start, three times: a HiRA update is a product of two small matrices, and an IA3 vector starts at one."""
    for step in range(3):
        perturb_adapter(base, name, 10 * seed + step)


class _Built(NamedTuple):
    """A model with a stack of adapters on its two linear layers, and the output it had with none."""

    base: nn.Module
    host: MixedModel
    names: list[str]
    clean: lucid.Tensor


def _build(
    stack: _Stack,
    device: str,
    scratch: Path,
    dtype: lucid.dtype = _F32,
    *,
    order: str = "added",
    perturb: bool = True,
    configs: Mapping[PeftType, PeftConfig] | None = None,
) -> _Built:
    """``stack``'s adapters, lowest first, each moved well off its start, all active in ``order``.

    A method's config is its recipe's unless ``configs`` names another (a variant of the method).
    """
    base = RECIPES[stack[0]].base(dtype, device)
    clean = _call(base, device, dtype)
    names = [_name(p) for p in stack]
    chosen = {p: (configs or {}).get(p) or _config(p, device, scratch, dtype) for p in stack}
    host = MixedModel(base, chosen[stack[0]], names[0])
    for peft_type, name in zip(stack[1:], names[1:], strict=True):
        host.add_adapter(name, chosen[peft_type])
    if perturb:
        for seed, name in enumerate(names, start=1):
            _move_off_the_start(base, name, seed)
    host.set_adapter(names if order == "added" else names[::-1])
    return _Built(base, host, names, clean)


def _root_of(top: nn.Module) -> nn.Module:
    """The user's module under a stack of layers, or ``top`` when nothing stands on it."""
    layers = _chain(top)
    return layers[0]._root() if layers else top


def _roots(base: nn.Module) -> list[nn.Module]:
    """The user's modules at the two sites."""
    return [_root_of(base.get_submodule(site)) for site in _SITES]


def _snapshot(base: nn.Module) -> list[lucid.Tensor]:
    return [t.detach().clone() for root in _roots(base) for t in (root.weight, root.bias)]


def _probe(base: nn.Module, device: str) -> lucid.Tensor:
    """The output of ``base``'s stacks on a plain twin: each layer's merged values in turn, lowest layer first.

    Only what each layer has active is applied; nothing of the stacked forward pass runs.
    """
    twin = copy.deepcopy(base)
    with lucid.no_grad():
        for site in _SITES:
            top = twin.get_submodule(site)
            root = _root_of(top)
            for layer in _chain(top)[::-1]:
                held = layer._adapter_names()
                for adapter in (a for a in layer.active_adapters if a in held):
                    for key, value in layer._merged_values(adapter).items():
                        getattr(root, key.rpartition(".")[2]).copy_(value)
            _set_submodule(twin, site, root)
        return _call(twin, device)


def _effect(built: _Built, device: str, only: str) -> float:
    """How far the adapter ``only``, alone, moves the output from the base's."""
    built.host.set_adapter([only])
    with lucid.no_grad():
        out = _call(built.base, device)
    return span(out, built.clean)


# ── the partition ─────────────────────────────────────────────────────────


class TestThePartition:
    def test_every_registered_method_is_declared_or_has_its_reason(self) -> None:
        undecided = set(_TUNERS) - set(_MIXED_TUNERS) - set(_NOT_MIXED)
        assert not undecided, (
            f"{sorted(t.name for t in undecided)} are registered but neither declared mixed=True "
            "nor listed in _NOT_MIXED: decide, and if declared pass the pair contract below"
        )
        assert set(_TUNERS) == set(_MIXED_TUNERS) | set(_NOT_MIXED)

    def test_no_method_is_both_and_every_declared_one_is_registered(self) -> None:
        assert set(_MIXED_TUNERS) & set(_NOT_MIXED) == set()
        assert set(_MIXED_TUNERS) <= set(_TUNERS)
        assert all(_TUNERS[p] is tuner for p, tuner in _MIXED_TUNERS.items())

    @pytest.mark.parametrize("peft_type", list(_NOT_MIXED), ids=lambda t: t.name)
    def test_a_reason_says_why_in_words(self, peft_type: PeftType) -> None:
        assert len(_NOT_MIXED[peft_type].split()) >= 5

    def test_the_declared_methods_are_these(self) -> None:
        """The decision of the ADR; a method joins or leaves by changing this line and its contract."""
        names = ["LORA", "LOHA", "LOKR", "ADALORA", "SHIRA", "OFT", "IA3", "BEFT", "ROAD", "HIRA"]
        assert {p.name for p in _DECLARED} == set(names)
        assert _INNERMOST_ONLY == {PeftType.HIRA}

    def test_a_declaration_needs_a_peft_type(self) -> None:
        with pytest.raises(TypeError, match="mixed=True without a peft_type"):
            type("Nameless", (_BaseTuner,), {}, mixed=True)

    @pytest.mark.parametrize("peft_type", list(_NOT_MIXED), ids=lambda t: t.name)
    def test_a_method_outside_the_declared_set_is_refused_as_the_first_adapter(
        self, peft_type: PeftType, tmp_path: Path
    ) -> None:
        base = RECIPES[PeftType.LORA].base(_F32, "cpu")
        before = _structure(base)
        error, message = _REFUSAL
        with pytest.raises(error, match=message):
            MixedModel(base, _config(peft_type, "cpu", tmp_path), "x")
        _unchanged(base, None, before)

    @pytest.mark.parametrize("peft_type", list(_NOT_MIXED), ids=lambda t: t.name)
    def test_and_as_a_later_one_with_the_model_as_it_was(
        self, peft_type: PeftType, tmp_path: Path
    ) -> None:
        built = _build((PeftType.LORA, PeftType.OFT), "cpu", tmp_path)
        before = _structure(built.base, built.host)
        error, message = _REFUSAL
        with pytest.raises(error, match=message):
            built.host.add_adapter("x", _config(peft_type, "cpu", tmp_path))
        _unchanged(built.base, built.host, before)

    def test_something_that_is_no_config_is_a_type_error_not_an_attribute_error(self) -> None:
        for given in ("lora", 3, None):
            with pytest.raises(TypeError, match="config must be a PeftConfig"):
                _family(given)

    def test_the_message_names_the_compatible_types(self, tmp_path: Path) -> None:
        built = _build((PeftType.LORA,), "cpu", tmp_path)
        with pytest.raises(ValueError, match="Compatible types are: ADALORA, BEFT, HIRA") as caught:
            built.host.add_adapter("x", _config(PeftType.VERA, "cpu", tmp_path))
        assert "`peft_type` 'VERA'" in str(caught.value)


class _Structure(NamedTuple):
    """Everything a refused ``add_adapter`` must leave alone."""

    tree: list[tuple[str, str]]
    state: dict[str, lucid.Tensor]
    grads: dict[str, bool]
    adapters: list[str]
    active: list[str]
    targets: list[str]


def _structure(base: nn.Module, host: MixedModel | None = None) -> _Structure:
    return _Structure(
        [(name, type(module).__name__) for name, module in base.named_modules()],
        {k: v.detach().clone() for k, v in base.state_dict().items()},
        {name: p.requires_grad for name, p in base.named_parameters()},
        sorted(host.peft_config) if host is not None else [],
        host.active_adapters if host is not None else [],
        list(host.targeted_module_names) if host is not None else [],
    )


def _unchanged(base: nn.Module, host: MixedModel | None, before: _Structure) -> None:
    after = _structure(base, host)
    assert after.tree == before.tree
    assert after.grads == before.grads
    assert (after.adapters, after.active, after.targets) == (
        before.adapters,
        before.active,
        before.targets,
    )
    assert after.state.keys() == before.state.keys()
    assert all(same(after.state[k], v) for k, v in before.state.items())


# ── the pair contract ─────────────────────────────────────────────────────


class TestTheStackIsTheOrderAdaptersWereAdded:
    @pytest.mark.parametrize("stack", _STACKS, ids=_id)
    def test_each_module_holds_one_layer_per_method_lowest_first(
        self, stack: _Stack, tmp_path: Path
    ) -> None:
        built = _build(stack, "cpu", tmp_path)
        for site in _SITES:
            layers = _chain(built.base.get_submodule(site))[::-1]
            assert [_family_of(layer) for layer in layers] == [_MIXED_TUNERS[p] for p in stack]
            assert [layer._adapter_names() for layer in layers] == [[n] for n in built.names]
        assert built.host.targeted_module_names == [*_SITES]

    @pytest.mark.parametrize("stack", _PAIRS, ids=_id)
    def test_a_second_adapter_of_a_method_goes_in_its_layer_not_in_a_new_one(
        self, stack: _Stack, tmp_path: Path
    ) -> None:
        built = _build(stack, "cpu", tmp_path)
        before = [len(_chain(built.base.get_submodule(site))) for site in _SITES]
        built.host.add_adapter("again", _config(stack[0], "cpu", tmp_path))
        layers = _chain(built.base.get_submodule(_SITES[0]))[::-1]
        assert [len(_chain(built.base.get_submodule(site))) for site in _SITES] == before
        assert layers[0]._adapter_names() == [built.names[0], "again"]
        assert all(
            layer._adapter_names() == [name]
            for layer, name in zip(layers[1:], built.names[1:], strict=True)
        )

    def test_the_first_method_added_is_the_lowest_whichever_it_is(self, tmp_path: Path) -> None:
        for stack in ((PeftType.LORA, PeftType.OFT), (PeftType.OFT, PeftType.LORA)):
            built = _build(stack, "cpu", tmp_path)
            top = built.base.get_submodule("0")
            assert _family_of(top) is _MIXED_TUNERS[stack[1]]
            assert _family_of(top.base_layer) is _MIXED_TUNERS[stack[0]]


@pytest.mark.parametrize("order", _ORDERS)
@pytest.mark.parametrize("stack", _STACKS, ids=_id)
class TestTheForwardPass:
    def test_it_is_the_dense_probe(
        self, stack: _Stack, order: str, device: str, tmp_path: Path
    ) -> None:
        built = _build(stack, device, tmp_path, order=order)
        with lucid.no_grad():
            got = _call(built.base, device)
        assert span(got, _probe(built.base, device)) < _TOLERANCE * max(1.0, magnitude(got))
        assert span(got, built.clean) > _EFFECT

    def test_every_layer_changes_the_output_by_far_more_than_the_tolerance(
        self, stack: _Stack, order: str, device: str, tmp_path: Path
    ) -> None:
        built = _build(stack, device, tmp_path, order=order)
        for name in built.names:
            assert _effect(built, device, name) > _EFFECT, name

    def test_the_adapters_of_the_lower_layers_alone_are_the_stack_without_the_rest(
        self, stack: _Stack, order: str, device: str, tmp_path: Path
    ) -> None:
        """Layers above whose adapters are not active pass their input through."""
        built = _build(stack, device, tmp_path, order=order)
        for count in range(1, len(stack)):
            built.host.set_adapter(
                built.names[:count] if order == "added" else built.names[:count][::-1]
            )
            with lucid.no_grad():
                got = _call(built.base, device)
            assert span(got, _probe(built.base, device)) < _TOLERANCE * max(1.0, magnitude(got))

    def test_disabling_the_adapters_gives_the_base_model(
        self, stack: _Stack, order: str, device: str, tmp_path: Path
    ) -> None:
        built = _build(stack, device, tmp_path, order=order)
        built.host.disable_adapter_layers()
        with lucid.no_grad():
            assert same(_call(built.base, device), built.clean)
        built.host.enable_adapter_layers()
        with lucid.no_grad():
            assert span(_call(built.base, device), built.clean) > _EFFECT


@pytest.mark.filterwarnings("ignore:Unmerge result can be inaccurate:UserWarning")
@pytest.mark.parametrize("order", _ORDERS)
@pytest.mark.parametrize("stack", _STACKS, ids=_id)
class TestMerging:
    def test_merging_is_the_forward_pass_and_unmerging_gives_the_weights_back(
        self, stack: _Stack, order: str, device: str, tmp_path: Path
    ) -> None:
        built = _build(stack, device, tmp_path, order=order)
        weights = _snapshot(built.base)
        with lucid.no_grad():
            adapted = _call(built.base, device)
            built.host.merge_adapter()
            assert all(layer.merged for _, layer in _tuner_layers(built.base))
            assert span(_call(built.base, device), adapted) < _TOLERANCE * max(
                1.0, magnitude(adapted)
            )
            built.host.unmerge_adapter()
            assert not any(layer.merged for _, layer in _tuner_layers(built.base))
            assert span(_call(built.base, device), adapted) < _TOLERANCE * max(
                1.0, magnitude(adapted)
            )
        for before, after in zip(weights, _snapshot(built.base), strict=True):
            assert span(before, after) < _TOLERANCE

    def test_a_layer_that_stashes_what_it_merges_into_gives_it_back_exactly(
        self, stack: _Stack, order: str, device: str, tmp_path: Path
    ) -> None:
        """HiRA stashes the weight it merges into and BEFT the bias; the layers above cannot leave a trace in it."""
        exact = {PeftType.HIRA: (0,), PeftType.BEFT: (1,)}.get(stack[0])
        if exact is None:
            pytest.skip("the lowest layer does not stash")
        built = _build(stack, device, tmp_path, order=order)
        weights = _snapshot(built.base)
        with lucid.no_grad():
            built.host.merge_adapter()
            built.host.unmerge_adapter()
        for index, (before, after) in enumerate(zip(weights, _snapshot(built.base), strict=True)):
            if index % 2 in exact:
                assert same(
                    before, after
                ), f"{'weight' if index % 2 == 0 else 'bias'} of site {index // 2}"

    def test_merge_and_unload_is_a_plain_model_computing_the_forward_pass(
        self, stack: _Stack, order: str, device: str, tmp_path: Path
    ) -> None:
        built = _build(stack, device, tmp_path, order=order)
        with lucid.no_grad():
            adapted = _call(built.base, device)
            plain = built.host.merge_and_unload()
            assert plain is built.base
            assert not any(isinstance(m, _BaseTunerLayer) for m in plain.modules())
            assert all(type(plain.get_submodule(site)) is nn.Linear for site in _SITES)
            assert span(_call(plain, device), adapted) < _TOLERANCE * max(1.0, magnitude(adapted))
        assert not hasattr(plain, "peft_config")

    def test_unloading_restores_the_model_it_was_given(
        self, stack: _Stack, order: str, device: str, tmp_path: Path
    ) -> None:
        built = _build(stack, device, tmp_path, order=order)
        weights = _snapshot(built.base)
        with lucid.no_grad():
            built.host.merge_adapter()
            plain = built.host.unload()
            assert plain is built.base
            assert not any(isinstance(m, _BaseTunerLayer) for m in plain.modules())
            for before, after in zip(weights, _snapshot(plain), strict=True):
                assert span(before, after) < _TOLERANCE
            assert span(_call(plain, device), built.clean) < _TOLERANCE
        assert all(p.requires_grad for p in plain.parameters())


# ── the host ──────────────────────────────────────────────────────────────


class TestTheHost:
    def test_it_is_not_the_tuner_of_any_method(self) -> None:
        assert MixedModel not in _TUNERS.values()
        assert MixedModel.__module__ == "lucid.peft.tuners._mixed"

    def test_all_linear_reaches_the_modules_another_method_already_adapted(
        self, tmp_path: Path
    ) -> None:
        built = _build((PeftType.LORA,), "cpu", tmp_path)
        built.host.add_adapter("loha", LoHaConfig(r=2, target_modules="all-linear"))
        for site in _SITES:
            chain = _chain(built.base.get_submodule(site))
            assert [_family_of(layer) for layer in chain] == [
                _MIXED_TUNERS[PeftType.LOHA],
                _MIXED_TUNERS[PeftType.LORA],
            ]

    def test_set_adapter_trains_exactly_the_adapters_it_names(self, tmp_path: Path) -> None:
        built = _build((PeftType.LORA, PeftType.OFT, PeftType.IA3), "cpu", tmp_path)
        for named in (["oft"], ["lora", "ia3"], ["ia3"]):
            built.host.set_adapter(named)
            trained = {name for name in built.names if _trains(built.base, name)}
            assert trained == set(named)
            assert built.host.active_adapters == named
            assert not any(
                p.requires_grad for root in _roots(built.base) for p in root.parameters()
            )

    def test_set_adapter_in_inference_mode_trains_none(self, tmp_path: Path) -> None:
        built = _build((PeftType.LORA, PeftType.OFT), "cpu", tmp_path)
        built.host.set_adapter(built.names, inference_mode=True)
        assert not any(_trains(built.base, name) for name in built.names)

    def test_deleting_the_adapter_of_a_lower_layer_leaves_the_stack_and_the_layer_is_reused(
        self, tmp_path: Path
    ) -> None:
        built = _build((PeftType.LORA, PeftType.OFT), "cpu", tmp_path)
        built.host.delete_adapter("lora")
        assert [layer._adapter_names() for layer in _chain(built.base.get_submodule("0"))] == [
            ["oft"],
            [],
        ]
        with lucid.no_grad():
            assert span(_call(built.base, "cpu"), _probe(built.base, "cpu")) < _TOLERANCE
        built.host.add_adapter("lora", _config(PeftType.LORA, "cpu", tmp_path))
        assert len(_chain(built.base.get_submodule("0"))) == 2

    def test_an_adapter_that_is_merged_cannot_be_deleted_and_nothing_is_lost(
        self, tmp_path: Path
    ) -> None:
        built = _build((PeftType.LORA, PeftType.LOHA), "cpu", tmp_path)
        built.host.merge_adapter()
        before = _structure(built.base, built.host)
        with pytest.raises(ValueError, match="while they are merged"):
            built.host.delete_adapter("lora")
        _unchanged(built.base, built.host, before)

    @pytest.mark.filterwarnings("ignore:Unmerge result can be inaccurate:UserWarning")
    def test_the_modules_may_hold_stacks_of_different_depths(self, tmp_path: Path) -> None:
        base = RECIPES[PeftType.LORA].base(_F32, "cpu")
        host = MixedModel(base, _lora(), "lora")
        host.add_adapter("ia3", IA3Config(target_modules=["0"], feedforward_modules=[]))
        host.add_adapter("oft", OFTConfig(oft_block_size=2, target_modules=["2"]))
        for seed, name in enumerate(["lora", "ia3", "oft"], start=1):
            _move_off_the_start(base, name, seed)
        host.set_adapter(["lora", "ia3", "oft"])
        depths = [
            [layer._adapter_names() for layer in _chain(base.get_submodule(s))[::-1]]
            for s in _SITES
        ]
        assert depths == [[["lora"], ["ia3"]], [["lora"], ["oft"]]]
        with lucid.no_grad():
            adapted = _call(base, "cpu")
            assert span(adapted, _probe(base, "cpu")) < _TOLERANCE * max(1.0, magnitude(adapted))
            host.merge_adapter()
            assert span(_call(base, "cpu"), adapted) < _TOLERANCE * max(1.0, magnitude(adapted))
            host.unmerge_adapter()
            plain = host.merge_and_unload()
            assert span(_call(plain, "cpu"), adapted) < _TOLERANCE * max(1.0, magnitude(adapted))

    def test_a_module_trained_in_full_stays_beside_the_stacks(self, tmp_path: Path) -> None:
        base = RECIPES[PeftType.LORA].base(_F32, "cpu")
        host = MixedModel(
            base, LoraConfig(r=2, target_modules=["0"], modules_to_save=["2"]), "lora"
        )
        host.add_adapter("loha", LoHaConfig(r=2, target_modules=["0"]))
        host.set_adapter(["lora", "loha"])
        assert type(base.get_submodule("2")).__name__ == "ModulesToSaveWrapper"
        assert [len(_chain(base.get_submodule(s))) for s in _SITES] == [2, 0]
        plain = host.unload()
        assert (
            type(plain.get_submodule("2")) is nn.Linear
            and type(plain.get_submodule("0")) is nn.Linear
        )

    def test_adalora_is_not_driven_by_the_host(self) -> None:
        """Its rank allocator and orthogonality loss belong to ``AdaLoraModel``; a mixed model is for inference."""
        assert not hasattr(MixedModel, "update_and_allocate")
        assert PeftType.ADALORA in _MIXED_TUNERS


def _trains(base: nn.Module, name: str) -> bool:
    """Whether every tensor of adapter ``name`` requires a gradient (and there is one)."""
    held = [p for _, layer in _tuner_layers(base) for p in layer._adapter_parameters(name)]
    return bool(held) and all(p.requires_grad for p in held)


@pytest.mark.parametrize("stack", _STACKS, ids=_id)
def test_gradients_reach_the_adapters_in_use_and_not_the_base(
    stack: _Stack, device: str, tmp_path: Path
) -> None:
    built = _build(stack, device, tmp_path)
    built.host.set_adapter(built.names)
    _call(built.base, device).sum().backward()
    for name in built.names:
        grads = [
            p.grad
            for _, layer in _tuner_layers(built.base)
            for p in layer._adapter_parameters(name)
        ]
        assert grads and all(
            g is not None and bool(lucid.isfinite(g).all().item()) for g in grads
        ), name
    assert all(root.weight.grad is None and root.bias.grad is None for root in _roots(built.base))


# ── what a stack refuses ──────────────────────────────────────────────────


def _lora(**fields: object) -> LoraConfig:
    """A rank-2 LoRA config on both sites, with ``fields`` changed."""
    return dataclasses.replace(LoraConfig(r=2, target_modules=list(_SITES)), **fields)


class TestWhatAStackRefuses:
    """Each refusal is made while planning, so the tree, the state and the flags are as they were."""

    def _refused(
        self, built: _Built, name: str, config: PeftConfig, error: type[Exception], match: str
    ) -> None:
        before = _structure(built.base, built.host)
        with pytest.raises(error, match=match):
            built.host.add_adapter(name, config)
        _unchanged(built.base, built.host, before)

    def test_a_method_that_stands_alone_is_not_put_on_another(self, tmp_path: Path) -> None:
        built = _build((PeftType.LORA, PeftType.OFT), "cpu", tmp_path)
        self._refused(
            built, "h", HiraConfig(r=2, target_modules=list(_SITES)), ValueError, "_stackable"
        )

    def test_it_says_what_the_module_already_holds_and_what_to_do(self, tmp_path: Path) -> None:
        built = _build((PeftType.LORA,), "cpu", tmp_path)
        with pytest.raises(
            ValueError, match=r"already holds \['lora'\].*before the others"
        ) as caught:
            built.host.add_adapter("h", HiraConfig(r=2, target_modules=list(_SITES)))
        assert "hira._layer.Linear" in str(caught.value)

    def test_the_method_that_stands_alone_may_be_the_lowest(self, tmp_path: Path) -> None:
        built = _build((PeftType.HIRA, PeftType.LORA), "cpu", tmp_path)
        assert _family_of(built.base.get_submodule("0").base_layer) is _MIXED_TUNERS[PeftType.HIRA]

    def test_a_layer_class_that_cannot_stand_on_another_is_refused_in_any_method(
        self, tmp_path: Path
    ) -> None:
        """OFT adapts a convolution too, but its convolution layer is not stackable."""
        net = nn.Sequential(nn.Conv2d(2, 4, 2))
        host = MixedModel(net, LoraConfig(r=2, target_modules=["0"]), "lora")
        built = _Built(net, host, ["lora"], lucid.zeros(1))
        self._refused(
            built, "o", OFTConfig(oft_block_size=2, target_modules=["0"]), ValueError, "_stackable"
        )

    def test_dora_is_refused_on_a_layer_that_stands_on_another(self, tmp_path: Path) -> None:
        built = _build((PeftType.LOHA,), "cpu", tmp_path)
        self._refused(
            built, "d", _lora(use_dora=True), ValueError, "use_dora reads the base weight"
        )

    @pytest.mark.parametrize("spelling", ["pissa", "olora", "lora_ga"])
    def test_a_start_that_rewrites_the_weight_is_refused_on_a_layer_that_stands_on_another(
        self, spelling: str, tmp_path: Path
    ) -> None:
        built = _build((PeftType.IA3,), "cpu", tmp_path)
        self._refused(
            built, "p", _lora(init_lora_weights=spelling), ValueError, "factorises the base weight"
        )

    @pytest.mark.filterwarnings("ignore:`init_lora_weights` is 'eva' but `eva_config`:UserWarning")
    def test_eva_is_refused_on_a_layer_that_stands_on_another(self, tmp_path: Path) -> None:
        built = _build((PeftType.LOHA,), "cpu", tmp_path)
        self._refused(built, "e", _lora(init_lora_weights="eva"), ValueError, "'eva'")

    def test_the_lowest_layer_may_be_dora_and_a_layer_above_it_merges_to_the_forward_pass(
        self, tmp_path: Path
    ) -> None:
        """DoRA reads the weight, so it is the lowest layer: the order is all the contract it needs."""
        base = RECIPES[PeftType.LORA].base(_F32, "cpu")
        host = MixedModel(base, _lora(use_dora=True), "dora")
        host.add_adapter("loha", _config(PeftType.LOHA, "cpu", tmp_path))
        for seed, name in enumerate(["dora", "loha"], start=1):
            _move_off_the_start(base, name, seed)
        host.set_adapter(["dora", "loha"])
        with lucid.no_grad():
            adapted = _call(base, "cpu")
            host.merge_adapter()
            assert span(_call(base, "cpu"), adapted) < _TOLERANCE * max(1.0, magnitude(adapted))
            host.unmerge_adapter()
            assert span(_call(base, "cpu"), adapted) < _TOLERANCE * max(1.0, magnitude(adapted))

    def test_a_module_that_a_second_method_adapts_whole_is_not_adapted_by_parameter(
        self, tmp_path: Path
    ) -> None:
        built = _build((PeftType.LORA,), "cpu", tmp_path)
        self._refused(
            built,
            "t",
            LoraConfig(r=2, target_parameters=["weight"], target_modules=list(_SITES)),
            ValueError,
            "target_parameters",
        )

    def test_a_second_adapter_that_trains_a_bias_is_refused(self, tmp_path: Path) -> None:
        base = RECIPES[PeftType.LORA].base(_F32, "cpu")
        host = MixedModel(base, _lora(bias="lora_only"), "a")
        built = _Built(base, host, ["a"], lucid.zeros(1))
        self._refused(
            built, "b", _lora(bias="lora_only"), ValueError, "supports only 1 adapter with bias"
        )
        host.add_adapter("c", _config(PeftType.LOHA, "cpu", tmp_path))  # any number without one

    def test_no_module_matching_the_config_is_an_error_and_changes_nothing(
        self, tmp_path: Path
    ) -> None:
        built = _build((PeftType.LORA,), "cpu", tmp_path)
        self._refused(
            built, "x", LoHaConfig(r=2, target_modules=["nothing"]), ValueError, "not found"
        )

    def test_a_model_that_already_holds_adapters_is_refused(self, tmp_path: Path) -> None:
        base = RECIPES[PeftType.LORA].base(_F32, "cpu")
        get_peft_model(base, _lora())
        before = _structure(base)
        with pytest.raises(ValueError, match="already holds adapters.*unload"):
            MixedModel(base, _config(PeftType.LOHA, "cpu", tmp_path), "x")
        _unchanged(base, None, before)

    def test_a_duplicate_adapter_name_is_refused(self, tmp_path: Path) -> None:
        built = _build((PeftType.LORA,), "cpu", tmp_path)
        self._refused(
            built, "lora", _config(PeftType.LOHA, "cpu", tmp_path), ValueError, "already exists"
        )

    def test_weighted_adapters_are_not_supported(self, tmp_path: Path) -> None:
        built = _build((PeftType.LORA, PeftType.LOHA), "cpu", tmp_path)
        with pytest.raises(NotImplementedError, match="combine the adapters of one method"):
            built.host.add_weighted_adapter(["lora"], [1.0], "w")
        with pytest.raises(NotImplementedError, match="combine the adapters of one method"):
            built.host.add_weighted_adapter(
                ["lora"], [1.0], "w", combination_type="cat", svd_rank=2, density=0.5
            )

    def test_the_refused_weighted_adapter_takes_the_arguments_of_the_one_it_stands_for(
        self,
    ) -> None:
        """A call written for ``LoraModel`` reaches the refusal, not a ``TypeError``."""
        mixed = inspect.signature(MixedModel.add_weighted_adapter).parameters
        lora = inspect.signature(LoraModel.add_weighted_adapter).parameters
        assert [(n, p.default) for n, p in mixed.items()] == [
            (n, p.default) for n, p in lora.items()
        ]

    @pytest.mark.parametrize(
        ("fields", "match"),
        [
            ({"use_dora": True}, "use_dora reads the base weight"),
            ({"init_lora_weights": "pissa"}, "factorises the base weight"),
            ({"init_lora_weights": "olora"}, "factorises the base weight"),
            ({"init_lora_weights": "lora_ga"}, "factorises the base weight"),
            ({"init_lora_weights": "eva"}, "'eva'"),
        ],
        ids=["dora", "pissa", "olora", "lora_ga", "eva"],
    )
    @pytest.mark.filterwarnings("ignore:`init_lora_weights` is 'eva' but `eva_config`:UserWarning")
    def test_a_second_adapter_in_a_lora_layer_that_stands_on_another_is_refused_as_a_new_one_is(
        self, fields: dict[str, object], match: str, tmp_path: Path
    ) -> None:
        """The layer exists already, so the refusal is not the layer's own at construction."""
        built = _build((PeftType.LOHA, PeftType.LORA), "cpu", tmp_path)
        assert len(_chain(built.base.get_submodule("0"))) == 2
        self._refused(built, "again", _lora(**fields), ValueError, match)

    def test_a_second_adapter_may_be_dora_in_the_lowest_lora_layer_under_others(
        self, tmp_path: Path
    ) -> None:
        built = _build((PeftType.LORA, PeftType.LOHA), "cpu", tmp_path)
        built.host.add_adapter("again", _lora(use_dora=True))
        _move_off_the_start(built.base, "again", 9)
        built.host.set_adapter(["again", "loha"])
        with lucid.no_grad():
            adapted = _call(built.base, "cpu")
            assert span(adapted, _probe(built.base, "cpu")) < _TOLERANCE * max(
                1.0, magnitude(adapted)
            )
            built.host.merge_adapter()
            assert span(_call(built.base, "cpu"), adapted) < _TOLERANCE * max(
                1.0, magnitude(adapted)
            )

    def test_a_quantized_base_cannot_be_mixed(self) -> None:
        import lucid.nn.quantized.dynamic as nnqd

        net = nn.Sequential(nnqd.Linear.from_float(nn.Linear(32, 16)))
        before = _structure(net)
        with pytest.raises(ValueError, match="quantized base cannot be mixed"):
            MixedModel(net, LoraConfig(r=2, target_modules=["0"]), "q")
        _unchanged(net, None, before)


# ── the order layers merge in ─────────────────────────────────────────────


@pytest.mark.filterwarnings("ignore:Unmerge result can be inaccurate:UserWarning")
class TestTheOrderOfAMerge:
    """A layer's update is computed from the weight the layers beneath it have left."""

    def _non_additive_pair(self, tmp_path: Path) -> _Built:
        return _build((PeftType.LORA, PeftType.OFT), "cpu", tmp_path)

    def _unchanged_by(self, built: _Built, names: list[str], match: str) -> None:
        before = _structure(built.base, built.host)
        merged = [layer.merged for _, layer in _tuner_layers(built.base)]
        with pytest.raises(ValueError, match=match):
            built.host.merge_adapter(adapter_names=names)
        _unchanged(built.base, built.host, before)
        assert [layer.merged for _, layer in _tuner_layers(built.base)] == merged

    def test_a_layer_is_merged_after_the_layers_beneath_it(self, tmp_path: Path) -> None:
        built = self._non_additive_pair(tmp_path)
        order = [name for name, _ in built.host._merge_order(_tuner_layers(built.base))]
        assert order == ["0.base_layer", "0", "2.base_layer", "2"]

    def test_the_tree_order_of_one_method_is_left_as_it_is(self) -> None:
        net = RECIPES[PeftType.LORA].base(_F32, "cpu")
        lora_model = get_peft_model(net, _lora()).base_model
        layers = _tuner_layers(net)
        assert lora_model._merge_order(layers) == layers

    def test_merging_a_layer_above_one_that_keeps_adapters_in_use_is_refused(
        self, tmp_path: Path
    ) -> None:
        """The rotation would transform the weight the LoRA update is folded into."""
        self._unchanged_by(self._non_additive_pair(tmp_path), ["oft"], r"\['lora'\] stay unmerged")

    def test_an_additive_layer_over_a_layer_that_rotates_is_refused_too(
        self, tmp_path: Path
    ) -> None:
        built = _build((PeftType.OFT, PeftType.LORA), "cpu", tmp_path)
        self._unchanged_by(built, ["lora"], r"\['oft'\] stay unmerged")

    def test_a_layer_below_may_be_merged_and_the_one_above_stays_in_use(
        self, tmp_path: Path
    ) -> None:
        built = self._non_additive_pair(tmp_path)
        with lucid.no_grad():
            adapted = _call(built.base, "cpu")
            built.host.merge_adapter(adapter_names=["lora"])
            assert span(_call(built.base, "cpu"), adapted) < _TOLERANCE * max(
                1.0, magnitude(adapted)
            )
            built.host.unmerge_adapter()
            assert span(_call(built.base, "cpu"), adapted) < _TOLERANCE * max(
                1.0, magnitude(adapted)
            )

    def test_two_layers_that_add_to_the_output_may_be_merged_in_either_order(
        self, tmp_path: Path
    ) -> None:
        built = _build((PeftType.LORA, PeftType.LOHA), "cpu", tmp_path)
        weights = _snapshot(built.base)
        with lucid.no_grad():
            adapted = _call(built.base, "cpu")
            built.host.merge_adapter(
                adapter_names=["loha"]
            )  # the layer above, the one beneath in use
            assert span(_call(built.base, "cpu"), adapted) < _TOLERANCE * max(
                1.0, magnitude(adapted)
            )
            built.host.merge_adapter(adapter_names=["lora"])
            assert span(_call(built.base, "cpu"), adapted) < _TOLERANCE * max(
                1.0, magnitude(adapted)
            )
            built.host.unmerge_adapter()
        for before, after in zip(weights, _snapshot(built.base), strict=True):
            assert span(before, after) < _TOLERANCE

    def test_a_layer_cannot_be_merged_after_the_layer_above_it_was(self, tmp_path: Path) -> None:
        """Reached by merging with the lower layer idle, then naming it: it would fold after the layer above."""
        built = _build((PeftType.OFT, PeftType.LORA), "cpu", tmp_path)
        built.host.set_adapter(["lora"])
        with lucid.no_grad():
            built.host.merge_adapter()  # the active one only: the layer beneath is not in use
        self._unchanged_by(built, ["oft"], "is merged already")

    def test_unmerging_by_any_entry_point_takes_the_layers_out_highest_first(
        self, tmp_path: Path
    ) -> None:
        """Out of order, the weights would come back corrupted: the pair above is one that rotates."""
        entries: dict[str, Callable[[MixedModel], None]] = {
            "unmerge_adapter": lambda host: host.unmerge_adapter(),
            "set_adapter": lambda host: host.set_adapter(["lora", "oft"]),
            "disable_adapter_layers": lambda host: host.disable_adapter_layers(),
            "unload": lambda host: host.unload(),
        }
        for entry, act in entries.items():
            built = self._non_additive_pair(tmp_path)
            weights = _snapshot(built.base)
            with lucid.no_grad(), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                built.host.merge_adapter()
                act(built.host)
            for before, after in zip(weights, _snapshot(built.base), strict=True):
                assert span(before, after) < _TOLERANCE, entry

    def test_a_layer_unmerged_beneath_a_merged_one_corrupts_the_weights_which_is_why_the_host_does_not(
        self, tmp_path: Path
    ) -> None:
        """What the order guards against, measured on the layers directly."""
        built = self._non_additive_pair(tmp_path)
        top = built.base.get_submodule("0")
        weights = _snapshot(built.base)
        with lucid.no_grad():
            top.base_layer.merge()
            top.merge()
            top.base_layer.unmerge()  # out of order: the rotation is still folded in
            top.unmerge()
            assert span(weights[0], _snapshot(built.base)[0]) > 1e-2


def _both_add_to_output(stack: _Stack) -> bool:
    return all(_MIXED_TUNERS[p]._adds_to_output for p in stack)


@pytest.mark.filterwarnings("ignore:Unmerge result can be inaccurate:UserWarning")
@pytest.mark.parametrize("stack", _PAIRS, ids=_id)
class TestMergingOneLayerOfAPair:
    """What ``_adds_to_output`` and the rule of ``merge_adapter`` claim, held for every pair."""

    def test_the_layer_below_may_be_merged_and_the_one_above_stays_in_use(
        self, stack: _Stack, device: str, tmp_path: Path
    ) -> None:
        built = _build(stack, device, tmp_path)
        weights = _snapshot(built.base)
        with lucid.no_grad():
            adapted = _call(built.base, device)
            built.host.merge_adapter(adapter_names=built.names[:1])
            assert span(_call(built.base, device), adapted) < _TOLERANCE * max(
                1.0, magnitude(adapted)
            )
            built.host.unmerge_adapter()
            assert span(_call(built.base, device), adapted) < _TOLERANCE * max(
                1.0, magnitude(adapted)
            )
        for before, after in zip(weights, _snapshot(built.base), strict=True):
            assert span(before, after) < _TOLERANCE

    def test_merging_and_unloading_the_layer_below_leaves_the_model_with_only_its_adapter(
        self, stack: _Stack, device: str, tmp_path: Path
    ) -> None:
        """The adapters not named are dropped with their layers, so what is left is the model without them."""
        built = _build(stack, device, tmp_path)
        with lucid.no_grad():
            built.host.set_adapter(built.names[:1])
            alone = _call(built.base, device)
            built.host.set_adapter(built.names)
            plain = built.host.merge_and_unload(adapter_names=built.names[:1])
            assert not any(isinstance(m, _BaseTunerLayer) for m in plain.modules())
            assert span(_call(plain, device), alone) < _TOLERANCE * max(1.0, magnitude(alone))
            assert span(alone, built.clean) > _EFFECT

    def test_the_layer_above_is_merged_alone_only_where_both_layers_add_to_the_output(
        self, stack: _Stack, device: str, tmp_path: Path
    ) -> None:
        built = _build(stack, device, tmp_path)
        before = _structure(built.base, built.host)
        if not _both_add_to_output(stack):
            with pytest.raises(ValueError, match="stay unmerged beneath it"):
                built.host.merge_adapter(adapter_names=built.names[1:])
            _unchanged(built.base, built.host, before)
            return
        with lucid.no_grad():
            adapted = _call(built.base, device)
            built.host.merge_adapter(adapter_names=built.names[1:])
            assert span(_call(built.base, device), adapted) < _TOLERANCE * max(
                1.0, magnitude(adapted)
            )
            built.host.merge_adapter(adapter_names=built.names[:1])  # then the one beneath it
            assert span(_call(built.base, device), adapted) < _TOLERANCE * max(
                1.0, magnitude(adapted)
            )


# ── a DoRA layer beneath: a variant of a method that reads the weight ──────


#: Every declared method that may stand on another's layer, bar LoRA itself (a method has one layer).
_UPPERS = [p for p in _DECLARED if p not in _INNERMOST_ONLY and p is not PeftType.LORA]
_DORA = {PeftType.LORA: _lora(use_dora=True)}


def _dora_below(upper: PeftType, device: str, scratch: Path) -> _Built:
    return _build((PeftType.LORA, upper), device, scratch, configs=_DORA)


@pytest.mark.filterwarnings("ignore:Unmerge result can be inaccurate:UserWarning")
@pytest.mark.parametrize("upper", _UPPERS, ids=lambda p: f"{p.name.lower()}-on-dora")
class TestADoraLayerBeneath:
    """DoRA divides by the norm of the weight it reads, so its layer is no additive term though LoRA is one."""

    def test_the_layer_is_not_additive_while_it_holds_dora_and_is_once_it_does_not(
        self, upper: PeftType, tmp_path: Path
    ) -> None:
        built = _dora_below(upper, "cpu", tmp_path)
        low = _chain(built.base.get_submodule("0"))[-1]
        assert _family_of(low) is _MIXED_TUNERS[PeftType.LORA]
        assert not _MIXED_TUNERS[PeftType.LORA]._layer_adds_to_output(low)
        plain = _build((PeftType.LORA, upper), "cpu", tmp_path)
        plain_low = _chain(plain.base.get_submodule("0"))[-1]
        assert _MIXED_TUNERS[PeftType.LORA]._layer_adds_to_output(plain_low)

    def test_the_forward_pass_is_the_probe_and_merging_is_the_forward_pass(
        self, upper: PeftType, device: str, tmp_path: Path
    ) -> None:
        built = _dora_below(upper, device, tmp_path)
        weights = _snapshot(built.base)
        with lucid.no_grad():
            adapted = _call(built.base, device)
            tolerance = _TOLERANCE * max(1.0, magnitude(adapted))
            assert span(adapted, _probe(built.base, device)) < tolerance
            assert _effect(built, device, "lora") > _EFFECT
            built.host.set_adapter(built.names)
            built.host.merge_adapter()
            assert span(_call(built.base, device), adapted) < tolerance
            built.host.unmerge_adapter()
            assert span(_call(built.base, device), adapted) < tolerance
        for before, after in zip(weights, _snapshot(built.base), strict=True):
            assert span(before, after) < _TOLERANCE

    def test_the_layer_above_is_not_merged_alone_and_nothing_changes(
        self, upper: PeftType, device: str, tmp_path: Path
    ) -> None:
        built = _dora_below(upper, device, tmp_path)
        before = _structure(built.base, built.host)
        merged = [layer.merged for _, layer in _tuner_layers(built.base)]
        with pytest.raises(ValueError, match=r"\['lora'\] stay unmerged beneath it"):
            built.host.merge_adapter(adapter_names=[_name(upper)])
        _unchanged(built.base, built.host, before)
        assert [layer.merged for _, layer in _tuner_layers(built.base)] == merged

    def test_dora_alone_and_then_the_layer_above_is_the_forward_pass_and_unmerges_clean(
        self, upper: PeftType, device: str, tmp_path: Path
    ) -> None:
        built = _dora_below(upper, device, tmp_path)
        weights = _snapshot(built.base)
        with lucid.no_grad():
            adapted = _call(built.base, device)
            tolerance = _TOLERANCE * max(1.0, magnitude(adapted))
            built.host.merge_adapter(adapter_names=["lora"])
            assert span(_call(built.base, device), adapted) < tolerance
            built.host.merge_adapter(adapter_names=[_name(upper)])
            assert span(_call(built.base, device), adapted) < tolerance
            built.host.unmerge_adapter()
            assert span(_call(built.base, device), adapted) < tolerance
        for before, after in zip(weights, _snapshot(built.base), strict=True):
            assert span(before, after) < _TOLERANCE

    def test_merging_and_unloading_dora_alone_is_the_model_with_only_dora(
        self, upper: PeftType, device: str, tmp_path: Path
    ) -> None:
        built = _dora_below(upper, device, tmp_path)
        with lucid.no_grad():
            alone = _only(built, "lora", device)
            built.host.set_adapter(built.names)
            plain = built.host.merge_and_unload(adapter_names=["lora"])
            assert span(_call(plain, device), alone) < _TOLERANCE * max(1.0, magnitude(alone))


@pytest.mark.filterwarnings("ignore:Unmerge result can be inaccurate:UserWarning")
def test_dora_under_two_additive_layers_is_merged_from_the_bottom_up(tmp_path: Path) -> None:
    """With more than one layer above it the order is the same: nothing above is merged first."""
    stack = (PeftType.LORA, PeftType.LOHA, PeftType.SHIRA)
    built = _build(stack, "cpu", tmp_path, configs=_DORA)
    with lucid.no_grad():
        adapted = _call(built.base, "cpu")
        tolerance = _TOLERANCE * max(1.0, magnitude(adapted))
        for names in (["shira"], ["loha"], ["loha", "shira"]):
            before = _structure(built.base, built.host)
            with pytest.raises(ValueError, match="stay unmerged beneath it"):
                built.host.merge_adapter(adapter_names=names)
            _unchanged(built.base, built.host, before)
        for names in (["lora"], ["loha"], ["shira"]):
            built.host.merge_adapter(adapter_names=names)
            assert span(_call(built.base, "cpu"), adapted) < tolerance, names


# ── half precision and placement ──────────────────────────────────────────


def _partner(peft_type: PeftType) -> PeftType:
    return PeftType.LOHA if peft_type is PeftType.LORA else PeftType.LORA


def _with(peft_type: PeftType) -> list[_Stack]:
    """``peft_type`` below and above a partner (above only where it may stand on a layer)."""
    partner = _partner(peft_type)
    stacks = [(partner, peft_type)] if peft_type not in _INNERMOST_ONLY else []
    return [(peft_type, partner), *stacks]


_EACH: list[_Stack] = list(dict.fromkeys(stack for p in _DECLARED for stack in _with(p)))


@pytest.mark.parametrize("dtype", [lucid.float16, lucid.bfloat16], ids=["float16", "bfloat16"])
@pytest.mark.parametrize("stack", _EACH, ids=_id)
def test_adapters_beside_a_half_precision_base_are_float32(
    stack: _Stack, dtype: lucid.dtype, device: str, tmp_path: Path
) -> None:
    built = _build(stack, device, tmp_path, dtype)
    adapters = {
        p.dtype
        for _, layer in _tuner_layers(built.base)
        for name in built.names
        for p in layer._adapter_parameters(name)
    }
    assert adapters == {_F32}
    with lucid.no_grad():
        out = _call(built.base, device, dtype)
    assert out.dtype == dtype and bool(lucid.isfinite(out.to(_F32)).all().item())


#: ``(default device, base device)``: where the two differ, and Metal under both.
_PLACEMENTS = [("metal", "cpu"), ("cpu", "metal"), ("metal", "metal")]


@pytest.mark.skipif(not metal_available(), reason="compares the CPU with Metal")
@pytest.mark.parametrize(("default", "where"), _PLACEMENTS, ids=lambda p: str(p))
@pytest.mark.parametrize("stack", _EACH, ids=_id)
def test_every_tensor_of_a_stack_is_on_the_base_weights_device(
    stack: _Stack, default: str, where: str, tmp_path: Path
) -> None:
    with default_device(default):
        built = _build(stack, where, tmp_path)
        held = [*built.base.named_parameters(), *built.base.named_buffers()]
        assert sorted(f"{n}: {t.device.type}" for n, t in held if t.device.type != where) == []
        out = _call(built.base, where)
        assert out.device.type == where
        out.sum().backward()


# ── the files ─────────────────────────────────────────────────────────────


def _alone(
    peft_type: PeftType, device: str, scratch: Path, seed: int
) -> tuple[dict[str, lucid.Tensor], lucid.Tensor]:
    """The file a single-method model of ``peft_type`` writes (keyed as a stack's host reads it) and its output."""
    recipe = RECIPES[peft_type]
    single = get_peft_model(recipe.base(_F32, device), recipe.config(_F32, device, scratch))
    _move_off_the_start(single, "default", seed)
    with lucid.no_grad():
        out = recipe.call(single, _F32, device)
    file = get_peft_model_state_dict(single)
    return {k.replace("base_model.model.", "model.", 1): v for k, v in file.items()}, out


def _only(built: _Built, name: str, device: str) -> lucid.Tensor:
    built.host.set_adapter([name])
    with lucid.no_grad():
        return _call(built.base, device)


@pytest.mark.parametrize("stack", _PAIRS, ids=_id)
class TestTheFiles:
    def test_an_adapter_of_a_stack_is_filed_as_the_method_alone_files_it(
        self, stack: _Stack, tmp_path: Path
    ) -> None:
        built = _build(stack, "cpu", tmp_path)
        for peft_type, name in zip(stack, built.names, strict=True):
            alone, _ = _alone(peft_type, "cpu", tmp_path, seed=1)
            filed = get_peft_model_state_dict(built.host, adapter_name=name)
            assert set(filed) == set(alone)
            assert all(tuple(filed[k].shape) == tuple(alone[k].shape) for k in alone)

    @pytest.mark.parametrize("when", ["before", "after"])
    def test_a_single_method_file_loads_into_a_stack_before_and_after_it_is_stacked(
        self, stack: _Stack, when: str, tmp_path: Path
    ) -> None:
        files = [_alone(p, "cpu", tmp_path, seed=7 + i) for i, p in enumerate(stack)]
        base = RECIPES[stack[0]].base(_F32, "cpu")
        clean = _call(base, "cpu")
        names = [_name(p) for p in stack]
        host = MixedModel(base, _config(stack[0], "cpu", tmp_path), names[0])
        if when == "before":
            set_peft_model_state_dict(host, files[0][0], adapter_name=names[0])
        for peft_type, name in zip(stack[1:], names[1:], strict=True):
            host.add_adapter(name, _config(peft_type, "cpu", tmp_path))
        built = _Built(base, host, names, clean)
        for index, (peft_type, name) in enumerate(zip(stack, names, strict=True)):
            if index > 0 or when == "after":
                set_peft_model_state_dict(host, files[index][0], adapter_name=name)
        for (_, wanted), name in zip(files, names, strict=True):
            assert span(_only(built, name, "cpu"), wanted) < _TOLERANCE * max(
                1.0, magnitude(wanted)
            ), name
            assert span(wanted, clean) > _EFFECT

    def test_what_a_stack_writes_is_what_it_was_given(self, stack: _Stack, tmp_path: Path) -> None:
        file, _ = _alone(stack[0], "cpu", tmp_path, seed=7)
        built = _build(stack, "cpu", tmp_path, perturb=False)
        set_peft_model_state_dict(built.host, file, adapter_name=built.names[0])
        written = get_peft_model_state_dict(built.host, adapter_name=built.names[0])
        assert written.keys() == file.keys()
        assert all(same(written[k], file[k]) for k in file)


#: What a stack's host reads a ``PeftModel`` file's keys as, and the other way round.
def _to_host(file: Mapping[str, lucid.Tensor]) -> dict[str, lucid.Tensor]:
    return {k.replace("base_model.model.", "model.", 1): v for k, v in file.items()}


def _to_single(file: Mapping[str, lucid.Tensor]) -> dict[str, lucid.Tensor]:
    return {k.replace("model.", "base_model.model.", 1): v for k, v in file.items()}


def _move_the_biases(base: nn.Module, seed: int) -> None:
    """Move every bias of the user's modules, as training them would."""
    lucid.manual_seed(seed)
    with lucid.no_grad():
        for root in _roots(base):
            root.bias.copy_(root.bias + 0.5 * lucid.randn(*root.bias.shape))


_BIASED = [
    (bias, position, partner)
    for bias in ("all", "lora_only")
    for position in ("lowest", "upper")
    for partner in _DECLARED
    if partner is not PeftType.LORA and (position == "upper" or partner not in _INNERMOST_ONLY)
]


def _biased_stack(bias: str, position: str, partner: PeftType, tmp_path: Path) -> _Built:
    stack = (PeftType.LORA, partner) if position == "lowest" else (partner, PeftType.LORA)
    return _build(stack, "cpu", tmp_path, configs={PeftType.LORA: _lora(bias=bias)})


def _alone_with_bias(bias: str, seed: int) -> tuple[dict[str, lucid.Tensor], lucid.Tensor]:
    """The file of a single-method LoRA model that trains ``bias``, keyed for a host, and its output."""
    recipe = RECIPES[PeftType.LORA]
    single = get_peft_model(recipe.base(_F32, "cpu"), _lora(bias=bias))
    _move_off_the_start(single, "default", seed)
    _move_the_biases(single.get_base_model(), seed)
    with lucid.no_grad():
        out = recipe.call(single, _F32, "cpu")
    return _to_host(get_peft_model_state_dict(single)), out


@pytest.mark.parametrize(
    ("bias", "position", "partner"), _BIASED, ids=lambda v: str(getattr(v, "name", v)).lower()
)
class TestATrainedBiasIsFiledBesideItsLayer:
    """A bias that ``bias="all"`` / ``"lora_only"`` trains is a base parameter: filed where a single-method file has it."""

    def test_the_keys_are_those_of_a_single_method_model(
        self, bias: str, position: str, partner: PeftType, tmp_path: Path
    ) -> None:
        alone, _ = _alone_with_bias(bias, seed=7)
        built = _biased_stack(bias, position, partner, tmp_path)
        filed = get_peft_model_state_dict(built.host, adapter_name="lora")
        assert sorted(filed) == sorted(alone)
        assert any(key.endswith(".bias") for key in filed)
        assert all(tuple(filed[k].shape) == tuple(alone[k].shape) for k in alone)

    def test_a_single_method_file_loads_into_the_stack_with_the_same_output(
        self, bias: str, position: str, partner: PeftType, tmp_path: Path
    ) -> None:
        alone, wanted = _alone_with_bias(bias, seed=7)
        built = _biased_stack(bias, position, partner, tmp_path)
        set_peft_model_state_dict(built.host, alone, adapter_name="lora")
        assert span(_only(built, "lora", "cpu"), wanted) < _TOLERANCE * max(1.0, magnitude(wanted))
        written = get_peft_model_state_dict(built.host, adapter_name="lora")
        assert written.keys() == alone.keys() and all(same(written[k], alone[k]) for k in alone)

    def test_a_stacks_file_loads_into_a_single_method_model_with_the_same_output(
        self, bias: str, position: str, partner: PeftType, tmp_path: Path
    ) -> None:
        built = _biased_stack(bias, position, partner, tmp_path)
        _move_the_biases(built.base, 11)
        wanted = _only(built, "lora", "cpu")
        file = get_peft_model_state_dict(built.host, adapter_name="lora")
        recipe = RECIPES[PeftType.LORA]
        single = get_peft_model(recipe.base(_F32, "cpu"), _lora(bias=bias))
        set_peft_model_state_dict(single, _to_single(file))
        with lucid.no_grad():
            assert span(recipe.call(single, _F32, "cpu"), wanted) < _TOLERANCE * max(
                1.0, magnitude(wanted)
            )
        written = _to_host(get_peft_model_state_dict(single))
        assert written.keys() == file.keys() and all(same(written[k], file[k]) for k in file)


def _bias_flags(base: nn.Module) -> list[bool]:
    """Whether each of the user's biases requires a gradient."""
    return [root.bias.requires_grad for root in _roots(base)]


@pytest.mark.parametrize(
    ("bias", "position", "partner"), _BIASED, ids=lambda v: str(getattr(v, "name", v)).lower()
)
def test_the_biases_a_method_trains_follow_the_adapters_in_use(
    bias: str, position: str, partner: PeftType, tmp_path: Path
) -> None:
    """``set_adapter`` is how a mixed model decides what trains: the host asks the method which parameters."""
    built = _biased_stack(bias, position, partner, tmp_path)
    assert _bias_flags(built.base) == [True, True]  # the adapters in use include LoRA's
    built.host.set_adapter([_name(partner)])
    assert _bias_flags(built.base) == [False, False]
    built.host.set_adapter(["lora"], inference_mode=True)
    assert _bias_flags(built.base) == [False, False]
    built.host.set_adapter(["lora"])
    assert _bias_flags(built.base) == [True, True]
    _call(built.base, "cpu").sum().backward()
    assert all(root.bias.grad is not None for root in _roots(built.base))


def test_the_biases_train_from_the_first_adapter_of_a_mixed_model() -> None:
    base = RECIPES[PeftType.LORA].base(_F32, "cpu")
    MixedModel(base, _lora(bias="all"), "lora")
    assert _bias_flags(base) == [True, True]


def test_only_lora_among_the_declared_methods_can_train_a_bias() -> None:
    """A declared method that learns to train one must have its file keys held like LoRA's above."""
    trained = []
    for peft_type, tuner in _MIXED_TUNERS.items():
        config = RECIPES[peft_type].config(_F32, "cpu", Path("."))
        if "bias" not in {f.name for f in dataclasses.fields(tuner._config_class)}:
            continue
        try:
            dataclasses.replace(config, bias="all")
        except (
            ValueError,
            NotImplementedError,
        ):  # a refused value: AdaLoRA trains its tensors only
            continue
        trained.append(peft_type)
    assert trained == [PeftType.LORA]


# ── lucid.compile ─────────────────────────────────────────────────────────


@pytest.mark.skipif(not lucid.metal.is_available(), reason="Metal unavailable")
@pytest.mark.parametrize("stack", _STACKS, ids=_id)
def test_switching_the_active_adapters_changes_the_compiled_output(
    stack: _Stack, tmp_path: Path
) -> None:
    built = _build(stack, "metal", tmp_path)
    built.base.eval()
    compiled = lucid.compile(built.base)
    x = lucid.randn(3, 6, device="metal")
    outputs = []
    for active in (built.names, built.names[:1], built.names[1:]):
        built.host.set_adapter(active)
        with lucid.no_grad():
            got, want = compiled(x), built.base(x)
        assert span(got, want) < _TOLERANCE * max(1.0, magnitude(want)), active
        outputs.append(got)
    assert span(outputs[0], outputs[1]) > _EFFECT and span(outputs[1], outputs[2]) > _EFFECT
    info = compiled.cache_info()
    assert not info["eager_only"], "the compiled model fell back to eager"
