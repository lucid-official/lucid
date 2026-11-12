"""Uni-LoRA: the vector, its projection and the update in closed form, the file, and the reference's numbers.

Uni-LoRA (Li et al., 2025) writes every LoRA factor of every layer as a read of one vector through a
fixed projection whose rows are one-hot and whose columns are scaled by ``1 / sqrt(n_j)``.  The tests
expect the paper's objects from plain loops (``_bank_forms``): the isometry ``P^T P = I`` (Theorem 1),
the update ``B A`` of a layer, and the file layout of the reference library.  The ``parity`` tests
build the same adapter in the reference library and compare.
"""

import warnings
from pathlib import Path
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
import lucid.optim as optim
from lucid.peft import (
    PeftModel,
    UniLoraConfig,
    UniLoraModel,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from lucid.peft.tuners import unilora
from lucid.test.unit.peft import _bank_forms as forms
from lucid.test.unit.peft._parity import adapter_library


def _config(**kw: object) -> UniLoraConfig:
    fields: dict[str, object] = {"r": 2, "theta_d_length": 8, "target_modules": ["0", "2"]} | kw
    return UniLoraConfig.from_dict(fields)


def _mlp(device: str = "cpu") -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4)).to(device)


def _span(a: lucid.Tensor, b: lucid.Tensor) -> float:
    return float((a - b).abs().max().item())


def _same(a: lucid.Tensor, b: lucid.Tensor) -> bool:
    return bool((a == b).all().item())


def _move(model: nn.Module, seed: int = 3) -> None:
    """Move every trainable parameter well off its start, as training would."""
    lucid.manual_seed(seed)
    with lucid.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.copy_(p + 0.5 * lucid.randn(*p.shape, device=p.device))


def _layer(peft: PeftModel, index: str = "0") -> unilora.UniLoraLayer:
    layer = peft.base_model.model.get_submodule(index)
    assert isinstance(layer, unilora.UniLoraLayer)
    return layer


class TestConfig:
    def test_the_defaults_are_the_references(self) -> None:
        config = UniLoraConfig(target_modules=["a"])
        assert (config.r, config.proj_seed, config.theta_d_length) == (4, 42, 256)
        assert (config.unilora_dropout, config.init_theta_d_bound) == (0.0, 0.02)
        assert (config.init_weights, config.save_indices, config.bias) == (True, False, "none")

    def test_fields_are_checked(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            UniLoraConfig(theta_d_length=0, target_modules=["a"])
        with pytest.raises(ValueError, match="positive"):
            UniLoraConfig(init_theta_d_bound=0.0, target_modules=["a"])
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            UniLoraConfig(unilora_dropout=1.0, target_modules=["a"])

    def test_a_file_round_trips(self) -> None:
        config = _config(save_indices=True, proj_seed=7)
        assert UniLoraConfig.from_dict(config.to_dict()) == config

    def test_a_bias_the_adapter_does_not_train_is_refused(self) -> None:
        with pytest.raises(NotImplementedError, match="bias='all'"):
            _config(bias="all")


class TestProjection:
    """The indices and scales: balanced, seeded, and an isometry."""

    @pytest.mark.parametrize(("count", "length"), [(100, 7), (21, 7), (3, 8), (0, 4), (64, 64)])
    def test_every_slot_is_read_floor_or_ceil_times(self, count: int, length: int) -> None:
        indices = UniLoraModel.generate_index(count, length, 5).tolist()
        assert len(indices) == count and all(0 <= i < length for i in indices)
        counts = [indices.count(j) for j in range(length)]
        assert set(counts) <= {count // length, -(-count // length)}
        assert sum(counts) == count

    def test_the_seed_decides_the_assignment(self) -> None:
        first = UniLoraModel.generate_index(200, 16, 1).tolist()
        assert UniLoraModel.generate_index(200, 16, 1).tolist() == first
        assert UniLoraModel.generate_index(200, 16, 2).tolist() != first

    def test_the_draw_is_an_int64_tensor(self) -> None:
        indices = UniLoraModel.generate_index(10, 4, 0)
        assert indices.dtype == lucid.int64 and tuple(indices.shape) == (10,)

    def test_a_malformed_draw_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            UniLoraModel.generate_index(-1, 4, 0)
        with pytest.raises(ValueError, match="positive"):
            UniLoraModel.generate_index(4, 0, 0)

    def test_the_projection_is_an_isometry(self) -> None:
        """Theorem 1: with columns scaled by ``1 / sqrt(n_j)``, ``P^T P = I_d``, so ``|P(x - y)| = |x - y|``."""
        peft = get_peft_model(_mlp(), _config(save_indices=True))
        indices: list[int] = []
        scales: dict[int, float] = {}
        for index in ("0", "2"):
            layer = _layer(peft, index)
            for name in ("A", "B"):
                chosen = getattr(layer, f"unilora_indices_{name}")["default"].reshape(-1).tolist()
                kept = getattr(layer, f"unilora_scales_{name}")["default"].reshape(-1).tolist()
                indices += chosen
                scales.update(zip(chosen, kept, strict=True))
        p, column = forms.unilora_projection(indices, 8)
        gram = [
            [sum(p[row][i] * p[row][j] for row in range(len(p))) for j in range(8)]
            for i in range(8)
        ]
        for i in range(8):
            for j in range(8):
                assert gram[i][j] == pytest.approx(1.0 if i == j else 0.0, abs=1e-6)
        assert all(scales[j] == pytest.approx(column[j], abs=1e-6) for j in scales)

    def test_one_draw_serves_all_the_layers_of_an_adapter(self) -> None:
        """The slots are shared across layers: ``D`` is the whole model's, not a layer's."""
        peft = get_peft_model(_mlp(), _config(theta_d_length=4, save_indices=True))
        read = [
            i
            for index in ("0", "2")
            for name in ("A", "B")
            for i in getattr(_layer(peft, index), f"unilora_indices_{name}")["default"]
            .reshape(-1)
            .tolist()
        ]
        total = 2 * 6 + 8 * 2 + 2 * 8 + 4 * 2
        assert len(read) == total
        assert sorted({read.count(j) for j in range(4)}) == [total // 4]


class TestUpdate:
    def test_the_update_is_b_a_of_the_vector_read_through_the_projection(self) -> None:
        peft = get_peft_model(_mlp(), _config(save_indices=True))
        _move(peft)
        theta = peft.base_model.unilora_theta_d["default"].tolist()
        for index in ("0", "2"):
            layer = _layer(peft, index)
            expected = forms.unilora_delta(
                theta,
                layer.unilora_indices_A["default"].tolist(),
                layer.unilora_indices_B["default"].tolist(),
                layer.unilora_scales_A["default"].tolist(),
                layer.unilora_scales_B["default"].tolist(),
            )
            assert _span(layer.get_delta_weight("default"), lucid.tensor(expected)) < 1e-5

    def test_the_forward_adds_the_update_to_the_base(self) -> None:
        net = _mlp()
        x = lucid.randn(5, 6)
        base = net[0](x)
        peft = get_peft_model(net, _config())
        _move(peft)
        layer = _layer(peft)
        expected = base + x @ layer.get_delta_weight("default").mT
        assert _span(layer(x), expected) < 1e-5

    def test_the_vector_starts_uniform_in_its_bound(self) -> None:
        peft = get_peft_model(_mlp(), _config(theta_d_length=64, init_theta_d_bound=0.05))
        theta = peft.base_model.unilora_theta_d["default"]
        assert float(theta.abs().max().item()) <= 0.05 and float(theta.abs().max().item()) > 0.0
        random = get_peft_model(_mlp(), _config(theta_d_length=64, init_weights=False))
        assert float(random.base_model.unilora_theta_d["default"].abs().max().item()) > 0.5

    def test_dropout_acts_on_the_input_of_the_update_only(self) -> None:
        peft = get_peft_model(_mlp(), _config(unilora_dropout=0.5))
        _move(peft)
        peft.eval()
        x = lucid.randn(5, 6)
        with lucid.no_grad():
            assert _same(peft(x), peft(x))  # off in eval
        peft.train()
        with peft.disable_adapter():
            assert _same(peft(x), peft(x))  # the base has none


class TestTraining:
    def test_the_adapter_of_the_whole_model_is_one_vector(self) -> None:
        peft = get_peft_model(_mlp(), _config(theta_d_length=11))
        trainable = [p for p in peft.parameters() if p.requires_grad]
        assert len(trainable) == 1 and tuple(trainable[0].shape) == (11,)
        assert trainable[0] is peft.base_model.unilora_theta_d["default"]

    def test_only_the_vector_gets_a_gradient(self) -> None:
        peft = get_peft_model(_mlp(), _config())
        peft(lucid.randn(5, 6)).sum().backward()
        for name, p in peft.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and float(p.grad.abs().max().item()) > 0, name
            else:
                assert p.grad is None, name

    def test_a_step_moves_the_vector(self) -> None:
        peft = get_peft_model(_mlp(), _config())
        theta = peft.base_model.unilora_theta_d["default"]
        before = theta.detach().clone()
        optimizer = optim.SGD([theta], lr=0.1)
        peft(lucid.randn(5, 6)).sum().backward()
        optimizer.step()
        assert _span(theta, before) > 0


class TestMerging:
    def test_merged_is_the_adapted_model_and_unmerging_restores_the_weights(self) -> None:
        net = _mlp()
        own = list(net.parameters())  # the adapter is built into ``net`` itself
        weights = [p.detach().clone() for p in own]
        peft = get_peft_model(net, _config())
        _move(peft)
        x = lucid.randn(5, 6)
        with lucid.no_grad():
            adapted = peft(x)
            peft.base_model.merge_adapter()
            assert _span(peft(x), adapted) < 1e-5
            peft.base_model.unmerge_adapter()
            assert _span(peft(x), adapted) < 1e-5
        for old, new in zip(weights, own, strict=True):
            assert _span(old, new) < 1e-6 * max(1.0, float(old.abs().max().item()))

    def test_merge_and_unload_returns_the_plain_adapted_model(self) -> None:
        net = _mlp()
        peft = get_peft_model(net, _config())
        _move(peft)
        x = lucid.randn(5, 6)
        with lucid.no_grad():
            adapted = peft(x)
            merged = peft.merge_and_unload()
            assert merged is net and type(net[0]) is nn.Linear
            assert _span(net(x), adapted) < 1e-5


class TestMultipleAdapters:
    def _two(self) -> tuple[PeftModel, lucid.Tensor]:
        peft = get_peft_model(_mlp(), _config(), adapter_name="a")
        peft.add_adapter("b", _config(r=3, theta_d_length=5, target_modules=["0"], proj_seed=9))
        _move(peft)
        peft.set_adapter("b")
        _move(peft, seed=4)
        return peft, lucid.randn(5, 6)

    def test_each_adapter_has_a_vector_of_its_own_length(self) -> None:
        peft, _ = self._two()
        table = peft.base_model.unilora_theta_d
        assert (tuple(table["a"].shape), tuple(table["b"].shape)) == ((8,), (5,))

    def test_two_active_adapters_add_and_merge_to_the_same_output(self) -> None:
        peft, x = self._two()
        peft.set_adapter(["a", "b"])
        with lucid.no_grad():
            both = peft(x)
            peft.set_adapter("a")
            only_a = peft(x)
            assert not _same(both, only_a)
            peft.set_adapter(["a", "b"])
            peft.base_model.merge_adapter()
            assert _span(peft(x), both) < 1e-5

    def test_deleting_an_adapter_takes_its_vector_and_projection(self) -> None:
        peft, x = self._two()
        peft.set_adapter("a")
        with lucid.no_grad():
            before = peft(x)
        peft.delete_adapter("b")
        assert "b" not in peft.base_model.unilora_theta_d
        assert "b" not in _layer(peft).unilora_indices_A
        assert sorted(peft.peft_config) == ["a"]
        with lucid.no_grad():
            assert _same(peft(x), before)

    def test_disabling_runs_the_base(self) -> None:
        net = _mlp()
        x = lucid.randn(5, 6)
        base = net(x)
        peft = get_peft_model(net, _config(init_weights=False))
        with peft.disable_adapter():
            assert _same(peft(x), base)


class TestFiles:
    def test_the_file_holds_one_vector_and_nothing_per_layer_by_default(self) -> None:
        state = get_peft_model_state_dict(get_peft_model(_mlp(), _config()))
        assert {k: tuple(v.shape) for k, v in state.items()} == {"base_model.unilora_theta_d": (8,)}

    def test_save_indices_adds_each_layers_projection(self) -> None:
        state = get_peft_model_state_dict(get_peft_model(_mlp(), _config(save_indices=True)))
        assert {k: tuple(v.shape) for k, v in state.items()} == {
            "base_model.unilora_theta_d": (8,),
            "base_model.model.0.unilora_indices_A": (2, 6),
            "base_model.model.0.unilora_indices_B": (8, 2),
            "base_model.model.0.unilora_scales_A": (2, 6),
            "base_model.model.0.unilora_scales_B": (8, 2),
            "base_model.model.2.unilora_indices_A": (2, 8),
            "base_model.model.2.unilora_indices_B": (4, 2),
            "base_model.model.2.unilora_scales_A": (2, 8),
            "base_model.model.2.unilora_scales_B": (4, 2),
        }
        assert state["base_model.model.0.unilora_indices_A"].dtype == lucid.int64

    @pytest.mark.parametrize("save_indices", [False, True])
    def test_save_then_load_reproduces_the_adapter(
        self, save_indices: bool, device: str, tmp_path: Path
    ) -> None:
        trained = get_peft_model(_mlp(device), _config(save_indices=save_indices))
        _move(trained)
        x = lucid.randn(4, 6, device=device)
        with lucid.no_grad():
            expected = trained(x)
        trained.save_pretrained(tmp_path)
        loaded = PeftModel.from_pretrained(_mlp(device), tmp_path)
        assert type(loaded.base_model) is UniLoraModel
        with lucid.no_grad():
            assert _same(loaded(x), expected)

    def test_a_file_that_carries_its_projection_does_not_depend_on_the_seed(self) -> None:
        """The reference writes ``save_indices`` for a file to outlive how indices are drawn."""
        trained = get_peft_model(_mlp(), _config(save_indices=True, proj_seed=1))
        _move(trained)
        state = get_peft_model_state_dict(trained)
        other = get_peft_model(_mlp(), _config(save_indices=True, proj_seed=2))
        x = lucid.randn(4, 6)
        with lucid.no_grad():
            assert not _same(other(x), trained(x))
            set_peft_model_state_dict(other, state)
            assert _same(other(x), trained(x))

    def test_a_vector_of_another_length_is_refused_and_nothing_changes(self) -> None:
        peft = get_peft_model(_mlp(), _config())
        before = get_peft_model_state_dict(peft)
        with pytest.raises(ValueError, match="shape"):
            set_peft_model_state_dict(peft, {"base_model.unilora_theta_d": lucid.zeros(9)})
        assert _same(
            get_peft_model_state_dict(peft)["base_model.unilora_theta_d"],
            before["base_model.unilora_theta_d"],
        )


_THETA = "base_model.unilora_theta_d"
_PROJECTION = (
    "unilora_indices_A",
    "unilora_indices_B",
    "unilora_scales_A",
    "unilora_scales_B",
)

#: (layers of the first adapter, layers of the second): the second one's reach among the first's.
_LAYOUTS = {
    "same_layers": (("0", "2"), ("0", "2")),
    "second_on_a_subset": (("0", "2"), ("2",)),
    "second_on_more": (("0",), ("0", "2")),
    "disjoint": (("0",), ("2",)),
}
#: (save_indices of the first adapter, of the second)
_CHOICES = [(False, False), (False, True), (True, False), (True, True)]


def _projection_keys(*layers: str) -> list[str]:
    """The file keys of the indices and scales of ``layers``, layer by layer."""
    return [f"base_model.model.{layer}.{name}" for layer in layers for name in _PROJECTION]


def _expected_keys(layers: tuple[str, ...], saved: bool) -> list[str]:
    """An adapter's file: the vector, then each layer's projection when it saves them."""
    return [_THETA, *(_projection_keys(*layers) if saved else [])]


def _two(
    layout: tuple[tuple[str, ...], tuple[str, ...]], choices: tuple[bool, bool], device: str = "cpu"
) -> PeftModel:
    """Adapter ``a`` and adapter ``b``, each on its own layers and choosing for itself whether to save its indices."""
    (first, second), (save_a, save_b) = layout, choices
    peft = get_peft_model(
        _mlp(device), _config(save_indices=save_a, target_modules=list(first)), adapter_name="a"
    )
    peft.add_adapter("b", _config(save_indices=save_b, proj_seed=9, target_modules=list(second)))
    return peft


class TestSaveIndicesPerAdapter:
    """``save_indices`` is each adapter's own choice, as the reference has it (LCD-984).

    The reference lets adapters of one model differ: the file of each carries its projection if its own
    config asks for it.  The buffers are persistent per adapter, not per layer.
    """

    @pytest.mark.parametrize("choices", _CHOICES)
    @pytest.mark.parametrize("layout", _LAYOUTS.values(), ids=_LAYOUTS.keys())
    def test_each_file_holds_what_its_own_config_asks(
        self, layout: tuple[tuple[str, ...], tuple[str, ...]], choices: tuple[bool, bool]
    ) -> None:
        peft = _two(layout, choices)
        for name, layers, saved in zip("ab", layout, choices, strict=True):
            assert list(get_peft_model_state_dict(peft, adapter_name=name)) == _expected_keys(
                layers, saved
            )

    @pytest.mark.parametrize("choices", _CHOICES)
    def test_the_model_holds_the_buffers_of_the_adapters_that_save_them(
        self, choices: tuple[bool, bool]
    ) -> None:
        peft = _two(_LAYOUTS["same_layers"], choices)
        held = {k for k in peft.state_dict() if "unilora_indices_A" in k}
        assert held == {
            f"base_model.model.{layer}.unilora_indices_A.{name}"
            for name, saved in zip("ab", choices, strict=True)
            if saved
            for layer in ("0", "2")
        }

    @pytest.mark.parametrize("choices", [(False, True), (True, False)])
    @pytest.mark.parametrize("layout", _LAYOUTS.values(), ids=_LAYOUTS.keys())
    def test_save_then_load_of_each_adapter_reproduces_it(
        self,
        layout: tuple[tuple[str, ...], tuple[str, ...]],
        choices: tuple[bool, bool],
        device: str,
        tmp_path: Path,
    ) -> None:
        peft = _two(layout, choices, device)
        _move(peft)
        peft.set_adapter("b")
        _move(peft, seed=4)
        x = lucid.randn(4, 6, device=device)
        expected: dict[str, lucid.Tensor] = {}
        with lucid.no_grad():
            for name in "ab":
                peft.set_adapter(name)
                expected[name] = peft(x)
        peft.save_pretrained(tmp_path)
        for name in "ab":
            loaded = PeftModel.from_pretrained(_mlp(device), tmp_path / name, adapter_name=name)
            with lucid.no_grad():
                assert _same(loaded(x), expected[name]), name

    def test_the_adapter_that_saves_carries_its_indices_whatever_the_other_chose(self) -> None:
        trained = _two(_LAYOUTS["same_layers"], (False, True))
        trained.set_adapter("b")
        _move(trained)
        state = get_peft_model_state_dict(trained, adapter_name="b")
        other = get_peft_model(_mlp(), _config(save_indices=True, proj_seed=2), adapter_name="b")
        x = lucid.randn(4, 6)
        with lucid.no_grad():
            assert not _same(other(x), trained(x))
            set_peft_model_state_dict(other, state, adapter_name="b")
            assert _same(other(x), trained(x))

    def test_an_adapter_added_again_follows_its_new_choice(self) -> None:
        peft = get_peft_model(_mlp(), _config(save_indices=True), adapter_name="a")
        peft.add_adapter("b", _config(save_indices=False, proj_seed=9))
        assert list(get_peft_model_state_dict(peft, adapter_name="b")) == [_THETA]
        peft.delete_adapter("b")
        peft.add_adapter("b", _config(save_indices=True, proj_seed=9))
        assert list(get_peft_model_state_dict(peft, adapter_name="b")) == _expected_keys(
            ("0", "2"), True
        )

    def test_a_layer_first_built_by_a_later_adapter_follows_that_adapter(self) -> None:
        peft = _two((("0",), ("0", "2")), (False, True))
        layer = _layer(peft, "2")
        assert layer.unilora_indices_A.persists("b") and "a" not in layer.unilora_indices_A
        assert not _layer(peft, "0").unilora_indices_A.persists("a")
        assert _layer(peft, "0").unilora_indices_A.persists("b")


class TestSaveIndicesFileKeyOrder:
    """The order of the keys of a file: the vector, then each layer's projection in layer order.

    The files of the adapters that agree (all saving or none) are what they were before the buffers
    became persistent per adapter, key for key.
    """

    def test_the_layers_projections_follow_the_vector_in_layer_order(self) -> None:
        state = get_peft_model_state_dict(get_peft_model(_mlp(), _config(save_indices=True)))
        assert list(state) == [
            "base_model.unilora_theta_d",
            "base_model.model.0.unilora_indices_A",
            "base_model.model.0.unilora_indices_B",
            "base_model.model.0.unilora_scales_A",
            "base_model.model.0.unilora_scales_B",
            "base_model.model.2.unilora_indices_A",
            "base_model.model.2.unilora_indices_B",
            "base_model.model.2.unilora_scales_A",
            "base_model.model.2.unilora_scales_B",
        ]

    @pytest.mark.parametrize("layout", _LAYOUTS.values(), ids=_LAYOUTS.keys())
    def test_adapters_that_save_list_only_their_own_layers(
        self, layout: tuple[tuple[str, ...], tuple[str, ...]]
    ) -> None:
        peft = _two(layout, (True, True))
        for name, layers in zip("ab", layout, strict=True):
            assert list(get_peft_model_state_dict(peft, adapter_name=name)) == [
                _THETA,
                *_projection_keys(*layers),
            ]

    def test_adapters_that_do_not_save_hold_the_vector_alone(self) -> None:
        peft = _two(_LAYOUTS["second_on_a_subset"], (False, False))
        for name in "ab":
            assert list(get_peft_model_state_dict(peft, adapter_name=name)) == [_THETA]


class TestErrors:
    def test_a_convolution_is_not_adapted(self) -> None:
        net = nn.Sequential(nn.Conv2d(2, 3, 3))
        with pytest.raises(ValueError, match="not supported"):
            get_peft_model(net, _config(target_modules=["0"]))

    def test_fan_in_fan_out_is_reset_with_a_warning(self) -> None:
        with pytest.warns(UserWarning, match="fan_in_fan_out"):
            peft = get_peft_model(_mlp(), _config(fan_in_fan_out=True))
        assert peft.peft_config["default"].fan_in_fan_out is False  # type: ignore[attr-defined]

    def test_the_layer_alone_refuses_fan_in_fan_out(self) -> None:
        with pytest.raises(ValueError, match="fan_in_fan_out"):
            unilora.Linear(nn.Linear(4, 3), "default", _config(fan_in_fan_out=True))

    def test_the_layer_alone_works_without_a_tuner(self) -> None:
        layer = unilora.Linear(nn.Linear(4, 3), "default", _config(target_modules=["x"]))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert tuple(layer(lucid.randn(2, 4)).shape) == (2, 3)
        assert tuple(layer.get_delta_weight("default").shape) == (3, 4)
        assert not any("unilora_indices" in k for k in layer.state_dict())


@pytest.mark.parity
def test_an_adapter_the_reference_wrote_loads_and_computes_the_same(
    ref: ModuleType, tmp_path: Path
) -> None:
    library = adapter_library()
    twin = ref.nn.Sequential(ref.nn.Linear(6, 8), ref.nn.ReLU(), ref.nn.Linear(8, 4))
    net = _mlp()
    with lucid.no_grad():
        for i in (0, 2):
            net[i].weight.copy_(lucid.tensor(twin[i].weight.tolist()))
            net[i].bias.copy_(lucid.tensor(twin[i].bias.tolist()))
    theirs = library.get_peft_model(
        twin,
        library.UniLoraConfig(r=2, theta_d_length=8, target_modules=["0", "2"], save_indices=True),
    )
    with ref.no_grad():
        for p in theirs.parameters():
            if p.requires_grad:
                p.add_(0.5 * ref.randn_like(p))
    theirs.save_pretrained(tmp_path)
    x = ref.randn(5, 6)
    with ref.no_grad():
        want = theirs(x).tolist()
    loaded = PeftModel.from_pretrained(net, tmp_path)
    with lucid.no_grad():
        assert _span(loaded(lucid.tensor(x.tolist())), lucid.tensor(want)) < 1e-5
    held = {k: tuple(v.shape) for k, v in get_peft_model_state_dict(loaded).items()}
    assert held == {k: tuple(v.shape) for k, v in library.get_peft_model_state_dict(theirs).items()}
