"""What every tuner built on the registry promises, held for (IA)^3, LoHa and LoKr alike.

A method is a config, a tuner and layer classes registered by ``PeftType``;
nothing above the registry names LoRA.  So the entry points, the multi-adapter
machinery, the state-dict helpers, the task wrappers and ``lucid.compile``
must work for a method that was added as subclasses only.  Each test runs on
a tiny MLP (CPU and Metal) with the update made non-zero where the test needs
one, and the on-disk fixtures are written by hand in the reference's layout.
"""

import json
import types
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    IA3Config,
    IA3Model,
    LoHaConfig,
    LoHaModel,
    LoKrConfig,
    LoKrModel,
    PeftConfig,
    PeftModel,
    PeftModelForCausalLM,
    PeftType,
    get_peft_model,
    get_peft_model_state_dict,
    inject_adapter_in_model,
    set_peft_model_state_dict,
)
from lucid.peft._config import _CONFIG_CLASSES
from lucid.peft.tuners._base import _TUNERS, _BaseTunerLayer, _tuner_for
from lucid.peft.tuners.adalora import AdaLoraConfig, AdaLoraLayer, AdaLoraModel
from lucid.peft.tuners.ia3 import IA3Layer
from lucid.peft.tuners.loha import LoHaLayer
from lucid.peft.tuners.lokr import LoKrLayer
from lucid.peft.tuners.vera import VeraConfig, VeraLayer, VeraModel
from lucid.serialization import save_safetensors
from lucid.test.audit._builders import BUILDERS
from lucid.test._fixtures.devices import metal_available


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def _close(a: lucid.Tensor, b: lucid.Tensor, atol: float = 1e-5) -> bool:
    return bool(lucid.allclose(a, b, atol=atol))


def _mlp(device: str = "cpu") -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4)).to(device)


@dataclass(frozen=True)
class _Method:
    name: str
    peft_type: PeftType
    config: Callable[..., PeftConfig]
    tuner: type
    layer_root: type[_BaseTunerLayer]
    #: The adapter file's tensors for ``_mlp`` targeted at ``["0", "2"]``, with the shape of each.
    files: dict[str, tuple[int, ...]]
    #: Config fields of an adapter added beside the first (AdaLoRA: only one may train).
    added: dict[str, object] = field(default_factory=dict)

    def make(
        self, targets: tuple[str, ...] = ("0", "2"), random: bool = True, **kw: object
    ) -> PeftConfig:
        """A config; ``random`` makes the update non-zero from the start."""
        return self.config(targets, random, **kw)

    def add(self, targets: tuple[str, ...] = ("0",)) -> PeftConfig:
        """The config of a second adapter."""
        return self.make(targets=targets, **self.added)


def _ia3(targets: tuple[str, ...], random: bool, **kw: object) -> IA3Config:
    return IA3Config(
        target_modules=list(targets),
        feedforward_modules=[t for t in targets if t == "2"],
        init_ia3_weights=not random,
        **kw,  # type: ignore[arg-type]
    )


def _loha(targets: tuple[str, ...], random: bool, **kw: object) -> LoHaConfig:
    return LoHaConfig(
        r=2,
        alpha=4,
        target_modules=list(targets),
        init_weights=not random,
        **kw,  # type: ignore[arg-type]
    )


def _lokr(targets: tuple[str, ...], random: bool, **kw: object) -> LoKrConfig:
    return LoKrConfig(
        r=2,
        alpha=4,
        target_modules=list(targets),
        init_weights=not random,
        **kw,  # type: ignore[arg-type]
    )


def _adalora(targets: tuple[str, ...], random: bool, **kw: object) -> AdaLoraConfig:
    return AdaLoraConfig(
        init_r=3,
        target_r=2,
        lora_alpha=6,
        total_step=10,
        target_modules=list(targets),
        init_lora_weights=not random,
        **kw,  # type: ignore[arg-type]
    )


def _vera(targets: tuple[str, ...], random: bool, **kw: object) -> VeraConfig:
    return VeraConfig(
        r=3,
        target_modules=list(targets),
        init_weights=not random,
        **kw,  # type: ignore[arg-type]
    )


_PREFIX = "base_model.model."
_METHODS = [
    _Method(
        "ia3",
        PeftType.IA3,
        _ia3,
        IA3Model,
        IA3Layer,
        {f"{_PREFIX}0.ia3_l": (8, 1), f"{_PREFIX}2.ia3_l": (1, 8)},
    ),
    _Method(
        "loha",
        PeftType.LOHA,
        _loha,
        LoHaModel,
        LoHaLayer,
        {
            f"{_PREFIX}0.hada_w1_a": (8, 2),
            f"{_PREFIX}0.hada_w1_b": (2, 6),
            f"{_PREFIX}0.hada_w2_a": (8, 2),
            f"{_PREFIX}0.hada_w2_b": (2, 6),
            f"{_PREFIX}2.hada_w1_a": (4, 2),
            f"{_PREFIX}2.hada_w1_b": (2, 8),
            f"{_PREFIX}2.hada_w2_a": (4, 2),
            f"{_PREFIX}2.hada_w2_b": (2, 8),
        },
    ),
    _Method(
        "lokr",
        PeftType.LOKR,
        _lokr,
        LoKrModel,
        LoKrLayer,
        {
            f"{_PREFIX}0.lokr_w1": (2, 2),
            f"{_PREFIX}0.lokr_w2": (4, 3),
            f"{_PREFIX}2.lokr_w1": (2, 2),
            f"{_PREFIX}2.lokr_w2": (2, 4),
        },
    ),
    _Method(
        "adalora",
        PeftType.ADALORA,
        _adalora,
        AdaLoraModel,
        AdaLoraLayer,
        {
            f"{_PREFIX}0.lora_A": (3, 6),
            f"{_PREFIX}0.lora_B": (8, 3),
            f"{_PREFIX}0.lora_E": (3, 1),
            f"{_PREFIX}2.lora_A": (3, 8),
            f"{_PREFIX}2.lora_B": (4, 3),
            f"{_PREFIX}2.lora_E": (3, 1),
        },
        added={"inference_mode": True},
    ),
    _Method(
        "vera",
        PeftType.VERA,
        _vera,
        VeraModel,
        VeraLayer,
        {
            f"{_PREFIX}0.vera_lambda_b": (8,),
            f"{_PREFIX}0.vera_lambda_d": (3,),
            f"{_PREFIX}2.vera_lambda_b": (4,),
            f"{_PREFIX}2.vera_lambda_d": (3,),
            "base_model.vera_A": (3, 8),
            "base_model.vera_B": (8, 3),
        },
    ),
]


@pytest.fixture(params=_METHODS, ids=lambda m: m.name)
def method(request: pytest.FixtureRequest) -> _Method:
    assert isinstance(request.param, _Method)
    return request.param


def _perturb(model: nn.Module, seed: int = 3) -> None:
    """Move every trainable parameter off its starting value, as a few optimizer steps would."""
    lucid.manual_seed(seed)
    with lucid.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.copy_(p + 0.3 * lucid.randn(*p.shape, device=p.device))


class TestRegistration:
    def test_the_method_registers_its_config_tuner_and_layers(self, method: _Method) -> None:
        assert _TUNERS[method.peft_type] is method.tuner
        assert _CONFIG_CLASSES[method.peft_type] is type(method.make())
        assert method.tuner._layer_types is method.layer_root._layer_types
        assert method.layer_root._layer_types  # at least the linear layer
        assert method.layer_root._layer_types[nn.Linear].__mro__[1] is method.layer_root

    def test_each_method_has_a_table_of_its_own(self) -> None:
        tables = [m.layer_root._layer_types for m in _METHODS]
        assert len({id(t) for t in tables}) == len(tables)
        assert all(nn.Linear in t for t in tables)

    def test_the_entry_points_dispatch_on_the_config(self, method: _Method) -> None:
        config = method.make()
        assert _tuner_for(config) is method.tuner
        assert type(get_peft_model(_mlp(), config).base_model) is method.tuner
        net = inject_adapter_in_model(config, _mlp())
        assert any(isinstance(m, method.layer_root) for m in net.modules())

    def test_a_file_names_its_method(self, method: _Method, tmp_path: Path) -> None:
        method.make().save_pretrained(tmp_path)
        read = PeftConfig.from_pretrained(tmp_path)
        assert type(read) is type(method.make()) and read == method.make()
        assert json.loads((tmp_path / "adapter_config.json").read_text())["peft_type"] == str(
            method.peft_type
        )

    def test_a_default_of_raw_parameters_is_not_taken_for_a_module_method(
        self, method: _Method
    ) -> None:
        # LoRA's table says ``neural_ode`` is adapted through raw parameters; a method that
        # adapts modules only has no default for it and must say so, not adapt nothing.
        net = nn.Sequential(nn.Linear(4, 4))
        net.config = types.SimpleNamespace(model_type="neural_ode")  # type: ignore[assignment]
        config = method.make(targets=())
        config.target_modules = None  # type: ignore[attr-defined]
        with pytest.raises(ValueError, match="Please specify `target_modules`"):
            method.tuner(net, config)

    def test_a_config_of_another_method_is_refused(self, method: _Method) -> None:
        other = next(m for m in _METHODS if m is not method)
        with pytest.raises(TypeError, match="config must be a"):
            method.tuner(_mlp(), other.make())

    def test_the_audit_builds_a_fresh_working_instance(
        self, method: _Method, request: pytest.FixtureRequest
    ) -> None:
        if method.tuner not in BUILDERS:
            # LCD-714b: the audit's builder rows for AdaLoRA / VeRA land with lucid/test/audit/_builders.py.
            request.applymarker(
                pytest.mark.xfail(strict=True, raises=KeyError, reason="no audit builder row yet")
            )
        first, second = BUILDERS[method.tuner](), BUILDERS[method.tuner]()
        assert type(first) is method.tuner
        assert {id(p) for p in first.parameters()}.isdisjoint(id(p) for p in second.parameters())
        assert first(lucid.randn(2, 4)).shape == (2, 4)


class TestForward:
    def test_the_default_start_is_the_base_bit_for_bit(self, method: _Method, device: str) -> None:
        net = _mlp(device)
        x = lucid.randn(5, 6, device=device)
        base = net(x)
        peft = get_peft_model(net, method.make(random=False))
        assert _same(peft(x), base)

    def test_a_random_start_changes_the_output(self, method: _Method, device: str) -> None:
        net = _mlp(device)
        x = lucid.randn(5, 6, device=device)
        base = net(x)
        peft = get_peft_model(net, method.make())
        assert not _close(peft(x), base, atol=1e-3)
        with peft.disable_adapter():
            assert _same(peft(x), base)

    def test_only_the_adapter_trains_and_only_it_gets_a_gradient(
        self, method: _Method, device: str
    ) -> None:
        net = _mlp(device)
        peft = get_peft_model(net, method.make())
        trainable = {n for n, p in peft.named_parameters() if p.requires_grad}
        assert trainable and all(any(t in n for t in ("ia3_", "hada_", "lokr_", "lora_", "vera_lambda")) for n in trainable)
        peft(lucid.randn(5, 6, device=device)).sum().backward()
        for name, p in peft.named_parameters():
            if name in trainable:
                assert p.grad is not None and float(p.grad.abs().max().item()) > 0, name
            else:
                assert p.grad is None, name

    def test_the_base_parameters_keep_their_identity(self, method: _Method) -> None:
        net = _mlp()
        weight = net[0].weight
        peft = get_peft_model(net, method.make())
        assert peft.base_model.model[0].base_layer.weight is weight


class TestMerging:
    def test_merged_equals_unmerged(self, method: _Method, device: str) -> None:
        net = _mlp(device)
        peft = get_peft_model(net, method.make())
        _perturb(peft)
        x = lucid.randn(5, 6, device=device)
        with lucid.no_grad():
            adapted = peft(x)
            peft.base_model.merge_adapter()
            assert all(layer.merged for _, layer in _layers(peft))
            assert _close(peft(x), adapted)
            peft.base_model.unmerge_adapter()
            assert not any(layer.merged for _, layer in _layers(peft))
            assert _close(peft(x), adapted)

    def test_merge_and_unload_gives_the_plain_adapted_model(
        self, method: _Method, device: str
    ) -> None:
        net = _mlp(device)
        peft = get_peft_model(net, method.make())
        _perturb(peft)
        x = lucid.randn(5, 6, device=device)
        with lucid.no_grad():
            adapted = peft(x)
            merged = peft.merge_and_unload()
            assert merged is net and type(net[0]) is nn.Linear
            assert _close(net(x), adapted)
        assert all(p.requires_grad for p in net.parameters())

    def test_unmerging_restores_the_weights(self, method: _Method) -> None:
        net = _mlp()
        before = [p.clone() for layer in (net[0], net[2]) for p in layer.parameters()]
        peft = get_peft_model(net, method.make())
        _perturb(peft)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # (IA)^3 warns that unmerging is approximate
            peft.base_model.merge_adapter()
            peft.base_model.unmerge_adapter()
        after = [p for _, layer in _layers(peft) for p in layer.get_base_layer().parameters()]
        assert len(before) == len(after) == 4
        for old, new in zip(before, after, strict=True):
            assert _close(old, new, atol=1e-5)

    def test_safe_merge_refuses_a_non_finite_adapter_and_writes_nothing(
        self, method: _Method
    ) -> None:
        net = _mlp()
        peft = get_peft_model(net, method.make())
        _perturb(peft)
        layer = peft.base_model.model[0]
        assert isinstance(layer, _BaseTunerLayer)
        weight = layer.get_base_layer().weight.clone()
        with lucid.no_grad():
            for p in layer._adapter_parameters("default"):
                p.copy_(p * float("inf"))
        with pytest.raises(ValueError, match="NaNs detected"):
            peft.base_model.merge_adapter(safe_merge=True)
        assert not layer.merged
        assert _same(layer.get_base_layer().weight, weight)

    def test_a_shared_weight_is_not_merged_into(self, method: _Method) -> None:
        net = nn.Sequential(nn.Linear(6, 6), nn.Linear(6, 6))
        net[1].weight = net[0].weight
        peft = get_peft_model(net, method.make(targets=("0",)))
        with pytest.raises(RuntimeError, match="cannot merge into"):
            peft.base_model.merge_adapter()

    def test_merging_a_disabled_adapter_is_refused(self, method: _Method) -> None:
        peft = get_peft_model(_mlp(), method.make())
        with peft.disable_adapter():
            with pytest.raises(RuntimeError, match="disabled"):
                peft.base_model.merge_adapter()


def _layers(peft: PeftModel) -> list[tuple[str, _BaseTunerLayer]]:
    return [(n, m) for n, m in peft.named_modules() if isinstance(m, _BaseTunerLayer)]


class TestMultiAdapter:
    def _two(self, method: _Method, device: str = "cpu") -> tuple[PeftModel, lucid.Tensor]:
        net = _mlp(device)
        peft = get_peft_model(net, method.make(), adapter_name="a")
        lucid.manual_seed(11)
        peft.add_adapter("b", method.add())
        _perturb(peft)  # only the trainable ones move: "a"
        return peft, lucid.randn(5, 6, device=device)

    def test_a_new_adapter_is_frozen_and_inactive(self, method: _Method, device: str) -> None:
        peft, x = self._two(method, device)
        assert peft.active_adapters == ["a"]
        layer = peft.base_model.model[0]
        assert isinstance(layer, _BaseTunerLayer)
        assert not any(p.requires_grad for p in layer._adapter_parameters("b"))
        assert all(p.requires_grad for p in layer._adapter_parameters("a"))

    def test_switching_follows_the_chosen_adapter(self, method: _Method, device: str) -> None:
        peft, x = self._two(method, device)
        with lucid.no_grad():
            on_a = peft(x)
            peft.set_adapter("b")
            on_b = peft(x)
            assert not _close(on_a, on_b, atol=1e-3)
            layer = peft.base_model.model[0]
            assert isinstance(layer, _BaseTunerLayer)
            assert all(p.requires_grad for p in layer._adapter_parameters("b"))
            assert not any(p.requires_grad for p in layer._adapter_parameters("a"))
            peft.set_adapter("a")
            assert _same(peft(x), on_a)

    def test_an_adapter_of_one_layer_leaves_the_other_layers_on_their_base(
        self, method: _Method, device: str
    ) -> None:
        # "b" reaches module 0 only; with "b" active, module 2 runs its base alone.
        peft, x = self._two(method, device)
        peft.set_adapter("b")
        last = peft.base_model.model[2]
        assert isinstance(last, _BaseTunerLayer)
        h = peft.base_model.model[1](peft.base_model.model[0](x))
        with lucid.no_grad():
            assert _same(peft(x), last.get_base_layer()(h))

    def test_deleting_the_active_adapter_hands_over(self, method: _Method) -> None:
        peft, x = self._two(method)
        with pytest.warns(UserWarning, match="was active which is now deleted"):
            peft.delete_adapter("a")
        assert peft.active_adapters == ["b"] and sorted(peft.peft_config) == ["b"]
        layer = peft.base_model.model[2]
        assert isinstance(layer, _BaseTunerLayer) and layer._adapter_names() == []

    def test_a_taken_name_is_refused(self, method: _Method) -> None:
        peft, _ = self._two(method)
        with pytest.raises(ValueError, match="already exists"):
            peft.add_adapter("b", method.make())

    def test_a_config_of_another_method_cannot_be_added(self, method: _Method) -> None:
        peft, _ = self._two(method)
        other = next(m for m in _METHODS if m is not method)
        with pytest.raises(ValueError, match="different peft types"):
            peft.add_adapter("c", other.make())

    def test_a_failed_add_leaves_the_model_as_it_was(self, method: _Method) -> None:
        peft, x = self._two(method)
        before = sorted(peft.state_dict())
        with pytest.raises(ValueError, match="not found"):
            peft.add_adapter("c", method.add(("nowhere",)))
        assert sorted(peft.state_dict()) == before and "c" not in peft.peft_config


class TestFiles:
    def test_the_state_dict_keys_are_the_reference_s(self, method: _Method) -> None:
        peft = get_peft_model(_mlp(), method.make())
        state = get_peft_model_state_dict(peft)
        assert {k: tuple(v.shape) for k, v in state.items()} == method.files

    def test_save_then_load_reproduces_the_adapter(
        self, method: _Method, device: str, tmp_path: Path
    ) -> None:
        trained = get_peft_model(_mlp(device), method.make())
        _perturb(trained)
        x = lucid.randn(4, 6, device=device)
        with lucid.no_grad():
            expected = trained(x)
        trained.save_pretrained(tmp_path)
        loaded = PeftModel.from_pretrained(_mlp(device), tmp_path)
        assert type(loaded.base_model) is method.tuner
        with lucid.no_grad():
            assert _same(loaded(x), expected)
        assert not any(p.requires_grad for p in loaded.parameters())
        assert sorted(f.name for f in tmp_path.iterdir()) == [
            "adapter_config.json",
            "adapter_model.safetensors",
        ]

    def test_a_named_second_adapter_round_trips(self, method: _Method, tmp_path: Path) -> None:
        peft = get_peft_model(_mlp(), method.make(), adapter_name="a")
        peft.add_adapter("b", method.add())
        peft.set_adapter("b")
        _perturb(peft)
        peft.save_pretrained(tmp_path)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["a", "b"]
        loaded = PeftModel.from_pretrained(_mlp(), tmp_path / "b", adapter_name="b")
        x = lucid.randn(4, 6)
        with lucid.no_grad():
            assert _same(loaded(x), peft(x))

    def test_set_refuses_what_does_not_fit_and_changes_nothing(self, method: _Method) -> None:
        peft = get_peft_model(_mlp(), method.make())
        before = get_peft_model_state_dict(peft)
        key = next(iter(before))
        wrong = {**before, key: lucid.zeros(*before[key].shape, 2)}
        with pytest.raises(ValueError, match="shape"):
            set_peft_model_state_dict(peft, wrong)
        for k, v in get_peft_model_state_dict(peft).items():
            assert _same(v, before[k])

    def test_the_task_type_picks_the_wrapper_class(self, method: _Method, tmp_path: Path) -> None:
        peft = get_peft_model(_mlp(), method.make(task_type="CAUSAL_LM"))
        assert type(peft) is PeftModelForCausalLM and type(peft.base_model) is method.tuner
        peft.save_pretrained(tmp_path)
        loaded = PeftModel.from_pretrained(_mlp(), tmp_path)
        assert type(loaded) is PeftModelForCausalLM

    def test_modules_to_save_trains_a_copy_and_saves_it(
        self, method: _Method, tmp_path: Path
    ) -> None:
        peft = get_peft_model(_mlp(), method.make(targets=("0",), modules_to_save=["2"]))
        with lucid.no_grad():
            peft.base_model.model[2].modules_to_save["default"].weight.add_(1.0)
        peft.save_pretrained(tmp_path)
        loaded = PeftModel.from_pretrained(_mlp(), tmp_path)
        x = lucid.randn(3, 6)
        with lucid.no_grad():
            assert _same(loaded(x), peft(x))

    def test_unload_restores_the_model(self, method: _Method) -> None:
        net = _mlp()
        before = {k: v.clone() for k, v in net.state_dict().items()}
        peft = get_peft_model(net, method.make())
        assert peft.unload() is net
        assert sorted(net.state_dict()) == sorted(before)
        assert all(_same(v, before[k]) for k, v in net.state_dict().items())


# ── adapter files written by hand, in the reference's layout ─────────────────

#: ``adapter_config.json`` as the reference library writes it, one per method.
_REFERENCE_CONFIGS: dict[str, dict[str, object]] = {
    "adalora": {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": "tiny",
        "beta1": 0.85,
        "beta2": 0.85,
        "bias": "none",
        "corda_config": None,
        "deltaT": 1,
        "eva_config": None,
        "exclude_modules": None,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "init_r": 3,
        "layer_replication": None,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": 6,
        "lora_bias": False,
        "lora_dropout": 0.0,
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "orth_reg_weight": 0.5,
        "peft_type": "ADALORA",
        "peft_version": "0.21.2",
        "r": 8,
        "rank_pattern": None,
        "revision": None,
        "target_modules": ["0", "2"],
        "target_parameters": None,
        "target_r": 2,
        "task_type": None,
        "tfinal": 0,
        "tinit": 0,
        "total_step": 10,
        "trainable_token_indices": None,
        "use_dora": False,
        "use_qalora": False,
        "use_rslora": False,
    },
    "vera": {
        "auto_mapping": None,
        "base_model_name_or_path": "tiny",
        "bias": "none",
        "d_initial": 0.1,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "modules_to_save": None,
        "peft_type": "VERA",
        "peft_version": "0.21.2",
        "projection_prng_key": 0,
        "r": 3,
        "revision": None,
        "save_projection": True,
        "target_modules": ["0", "2"],
        "task_type": None,
    },
    "ia3": {
        "auto_mapping": None,
        "base_model_name_or_path": "tiny",
        "exclude_modules": None,
        "fan_in_fan_out": False,
        "feedforward_modules": ["2"],
        "inference_mode": True,
        "init_ia3_weights": True,
        "modules_to_save": None,
        "peft_type": "IA3",
        "peft_version": "0.21.2",
        "revision": None,
        "target_modules": ["0", "2"],
        "task_type": None,
    },
    "loha": {
        "alpha": 4,
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": "tiny",
        "exclude_modules": None,
        "inference_mode": True,
        "init_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "module_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LOHA",
        "peft_version": "0.21.2",
        "r": 2,
        "rank_dropout": 0.0,
        "rank_pattern": {},
        "revision": None,
        "target_modules": ["0", "2"],
        "task_type": None,
        "use_effective_conv2d": False,
    },
    "lokr": {
        "alpha": 4,
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": "tiny",
        "decompose_both": False,
        "decompose_factor": -1,
        "exclude_modules": None,
        "inference_mode": True,
        "init_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "module_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LOKR",
        "peft_version": "0.21.2",
        "r": 2,
        "rank_dropout": 0.0,
        "rank_dropout_scale": False,
        "rank_pattern": {},
        "revision": None,
        "target_modules": ["0", "2"],
        "task_type": None,
        "use_effective_conv2d": False,
    },
}


def _reference_tensors(method: _Method) -> dict[str, lucid.Tensor]:
    """Small integer-valued tensors under the reference's keys (so the closed form is exact)."""
    lucid.manual_seed(7)
    return {
        key: lucid.randint(-2, 3, shape).to(lucid.float32) * 0.5
        for key, shape in method.files.items()
    }


def _reference_forward(
    method: _Method, t: dict[str, lucid.Tensor], x: lucid.Tensor
) -> lucid.Tensor:
    """``_mlp(x)`` with the adapter of tensors ``t`` applied the way the reference's formulas say."""
    net = _mlp()
    relu = nn.functional.relu
    if method.name == "ia3":
        # Output-side on layer 0 (bias included), input-side on layer 2.
        v0, v2 = t[f"{_PREFIX}0.ia3_l"].flatten(), t[f"{_PREFIX}2.ia3_l"].flatten()
        hidden = relu((x @ net[0].weight.mT + net[0].bias) * v0)
        return (hidden * v2) @ net[2].weight.mT + net[2].bias
    weights = []
    for index in (0, 2):
        p = f"{_PREFIX}{index}."
        if method.name == "adalora":
            # B diag(E) A, scaled by lora_alpha / ranknum = 6 / 3.
            delta = (t[p + "lora_B"] @ (t[p + "lora_A"] * t[p + "lora_E"])) * (6 / 3)
        elif method.name == "vera":
            out, inn = net[index].weight.shape
            rows = t[p + "vera_lambda_b"].unsqueeze(-1) * t["base_model.vera_B"][:out, :3]
            columns = t[p + "vera_lambda_d"].unsqueeze(-1) * t["base_model.vera_A"][:3, :inn]
            delta = rows @ columns
        elif method.name == "loha":
            delta = (t[p + "hada_w1_a"] @ t[p + "hada_w1_b"]) * (
                t[p + "hada_w2_a"] @ t[p + "hada_w2_b"]
            )
        else:
            w1, w2 = t[p + "lokr_w1"], t[p + "lokr_w2"]
            delta = lucid.cat(
                [
                    lucid.cat([w1[i, j] * w2 for j in range(w1.shape[1])], dim=1)
                    for i in range(w1.shape[0])
                ],
                dim=0,
            )
        scale = 4 / 2 if method.name in ('loha', 'lokr') else 1.0  # alpha / r
        weights.append(net[index].weight + delta * scale)
    hidden = relu(x @ weights[0].mT + net[0].bias)
    return hidden @ weights[1].mT + net[2].bias


def test_a_reference_written_adapter_loads(method: _Method, tmp_path: Path) -> None:
    tensors = _reference_tensors(method)
    save_safetensors(
        tensors, str(tmp_path / "adapter_model.safetensors"), metadata={"format": "pt"}
    )
    (tmp_path / "adapter_config.json").write_text(json.dumps(_REFERENCE_CONFIGS[method.name]))
    loaded = PeftModel.from_pretrained(_mlp(), tmp_path)
    assert type(loaded.base_model) is method.tuner
    assert loaded.peft_config["default"].base_model_name_or_path == "tiny"
    x = lucid.randn(6, 6)
    with lucid.no_grad():
        assert _close(loaded(x), _reference_forward(method, tensors, x), atol=1e-4)


# ── compile ──────────────────────────────────────────────────────────────────

metal_only = pytest.mark.skipif(not metal_available(), reason="compiled executables run on Metal")


@metal_only
def test_compiled_follows_set_adapter_disable_and_merge(method: _Method) -> None:
    peft = get_peft_model(_mlp("metal"), method.make(), adapter_name="a")
    peft.add_adapter("b", method.add())
    _perturb(peft)
    compiled = lucid.compile(peft)
    x = lucid.randn(4, 6, device="metal")
    with lucid.no_grad():
        on_a = peft(x)
        assert _close(compiled(x), on_a)
        peft.set_adapter("b")
        on_b = peft(x)
        assert not _close(on_a, on_b, atol=1e-3)
        assert _close(compiled(x), on_b)
        with peft.disable_adapter():
            base = peft(x)
            assert _close(compiled(x), base)
        assert _close(compiled(x), on_b)
        peft.set_adapter("a")
        peft.base_model.merge_adapter()
        assert _close(compiled(x), on_a, atol=1e-4)
        peft.base_model.unmerge_adapter()
        assert _close(compiled(x), on_a, atol=1e-4)
