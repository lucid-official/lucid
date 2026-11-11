"""TinyLoRA: the frozen SVD, the vector-to-update map in closed form, weight tying, the file and the reference's numbers.

TinyLoRA (Morris et al., 2026) freezes the truncated SVD of each weight and trains a vector ``v`` of ``u`` numbers;
``R = sum_i v_i P_i`` for fixed random ``P_i`` sits between the frozen factors, ``Delta W = B R A``, with the singular
values split over ``A = S^(1/2) V^T`` and ``B = U S^(1/2)``.  The tests expect those objects from plain loops
(``_bank_forms``): a weight whose SVD is known by construction, the update of a layer, the grouping of modules that
share a vector, and the file layout of the reference library.  The ``parity`` tests build the same adapter in the
reference library and compare.
"""

import math
import warnings
from pathlib import Path
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import (
    PeftModel,
    TinyLoraConfig,
    TinyLoraModel,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from lucid.peft.tuners import tinylora
from lucid.test.unit.peft import _bank_forms as forms
from lucid.test.unit.peft._parity import adapter_library


def _config(**kw: object) -> TinyLoraConfig:
    fields: dict[str, object] = {"r": 2, "u": 3, "target_modules": ["0", "2"]} | kw
    return TinyLoraConfig.from_dict(fields)


def _mlp(device: str = "cpu") -> nn.Sequential:
    lucid.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4)).to(device)


def _stack(depth: int, device: str = "cpu") -> nn.Sequential:
    """``depth`` linear layers of one shape, so every one is a target."""
    lucid.manual_seed(0)
    return nn.Sequential(*[nn.Linear(6, 6) for _ in range(depth)]).to(device)


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


def _layer(peft: PeftModel, index: str = "0") -> tinylora.TinyLoraLayer:
    layer = peft.base_model.model.get_submodule(index)
    assert isinstance(layer, tinylora.TinyLoraLayer)
    return layer


class TestConfig:
    def test_the_defaults_are_the_references(self) -> None:
        config = TinyLoraConfig(target_modules=["a"])
        assert (config.r, config.u, config.weight_tying, config.projection_seed) == (2, 64, 0.0, 42)
        assert (config.save_projection, config.init_v_bound) == (True, 0.02)
        assert (config.tinylora_dropout, config.bias, config.init_weights) == (0.0, "none", True)

    def test_fields_are_checked(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            TinyLoraConfig(u=0, target_modules=["a"])
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            TinyLoraConfig(weight_tying=1.5, target_modules=["a"])
        with pytest.raises(ValueError, match="uniform"):
            TinyLoraConfig(init_weights="normal", target_modules=["a"])

    def test_a_bias_the_adapter_does_not_train_is_refused(self) -> None:
        with pytest.raises(NotImplementedError, match="bias='all'"):
            _config(bias="all")


class TestFrozenFactors:
    def test_the_singular_values_split_evenly_over_the_two_factors(self) -> None:
        """For ``W`` with singular values 3, 2, 1: ``A A^T = B^T B = diag(3, 2, 1)`` and ``B A = W``."""
        base = nn.Linear(4, 5, bias=False)
        with lucid.no_grad():
            base.weight.copy_(
                lucid.tensor(
                    [
                        [3.0, 0.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                )
            )
        layer = tinylora.Linear(base, "default", _config(r=3, target_modules=["x"]))
        a, b = layer.tinylora_A["default"], layer.tinylora_B["default"]
        assert (tuple(a.shape), tuple(b.shape)) == ((3, 4), (5, 3))
        assert _span(a @ a.mT, lucid.tensor([[3.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]])) < 1e-5
        assert _span(b.mT @ b, lucid.tensor([[3.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]])) < 1e-5
        assert _span(b @ a, base.weight) < 1e-5

    def test_a_rank_above_the_weights_is_cut_to_it(self) -> None:
        layer = tinylora.Linear(nn.Linear(3, 5), "default", _config(r=9, target_modules=["x"]))
        assert layer.r["default"] == 3
        assert tuple(layer.tinylora_P["default"].shape) == (3, 3, 3)

    def test_the_factors_are_frozen_buffers(self) -> None:
        peft = get_peft_model(_mlp(), _config())
        trainable = {n for n, p in peft.named_parameters() if p.requires_grad}
        assert all("tinylora_v" in n for n in trainable)
        layer = _layer(peft)
        assert not layer.tinylora_A["default"].requires_grad
        assert not layer.tinylora_B["default"].requires_grad
        assert not layer.tinylora_P["default"].requires_grad


class TestProjection:
    def test_p_is_normal_with_standard_deviation_one_over_root_r(self) -> None:
        layer = tinylora.Linear(
            nn.Linear(8, 8), "default", _config(r=4, u=400, target_modules=["x"])
        )
        p = layer.tinylora_P["default"]
        assert tuple(p.shape) == (400, 4, 4)
        assert abs(float(p.std().item()) - 1 / math.sqrt(4)) < 0.03
        assert abs(float(p.mean().item())) < 0.03

    def test_the_seed_plus_the_position_decides_it(self) -> None:
        first = get_peft_model(_mlp(), _config(projection_seed=5))
        again = get_peft_model(_mlp(), _config(projection_seed=5))
        other = get_peft_model(_mlp(), _config(projection_seed=6))

        def drawn(peft: PeftModel, index: str) -> lucid.Tensor:
            return _layer(peft, index).tinylora_P["default"]

        assert _same(drawn(first, "0"), drawn(again, "0"))
        assert not _same(drawn(first, "0"), drawn(other, "0"))
        assert not _same(
            drawn(first, "0"), drawn(first, "2")
        )  # the layer's index is part of the seed


class TestUpdate:
    def test_the_update_is_b_r_a_with_r_the_vector_through_the_projection(self) -> None:
        peft = get_peft_model(_mlp(), _config(weight_tying=1.0))
        _move(peft)
        v = peft.base_model.tinylora_v["default"]["0"].tolist()
        for index in ("0", "2"):
            layer = _layer(peft, index)
            expected = forms.tinylora_delta(
                layer.tinylora_A["default"].tolist(),
                layer.tinylora_B["default"].tolist(),
                layer.tinylora_P["default"].tolist(),
                v,
            )
            assert _span(layer.get_delta_weight("default"), lucid.tensor(expected)) < 1e-5

    def test_the_forward_adds_the_update_to_the_base(self) -> None:
        net = _mlp()
        x = lucid.randn(5, 6)
        base = net[0](x)
        peft = get_peft_model(net, _config())
        _move(peft)
        layer = _layer(peft)
        assert _span(layer(x), base + x @ layer.get_delta_weight("default").mT) < 1e-5

    def test_the_default_start_is_the_base_bit_for_bit(self) -> None:
        net = _mlp()
        x = lucid.randn(5, 6)
        base = net(x)
        peft = get_peft_model(net, _config())
        assert _same(peft(x), base)

    def test_the_vector_starts_as_init_weights_says(self) -> None:
        zero = get_peft_model(_mlp(), _config())
        assert float(zero.base_model.tinylora_v["default"]["0"].abs().max().item()) == 0.0
        bounded = get_peft_model(_mlp(), _config(init_weights="uniform", init_v_bound=0.1, u=64))
        v = bounded.base_model.tinylora_v["default"]["0"]
        assert 0.0 < float(v.abs().max().item()) <= 0.1
        random = get_peft_model(_mlp(), _config(init_weights=False, u=64))
        assert float(random.base_model.tinylora_v["default"]["0"].std().item()) > 0.5

    def test_an_embedding_is_updated_through_the_rows_it_looks_up(self) -> None:
        base = nn.Embedding(10, 4)
        layer = tinylora.Embedding(
            base, "default", _config(init_weights=False, target_modules=["x"], r=2, u=3)
        )
        assert (
            tuple(layer.tinylora_A["default"].shape),
            tuple(layer.tinylora_B["default"].shape),
        ) == ((2, 4), (10, 2))
        ids = lucid.tensor([[1, 2, 3], [4, 4, 9]])
        delta = layer.get_delta_weight("default")
        assert tuple(delta.shape) == (10, 4)
        assert _span(layer(ids), base(ids) + delta[ids]) < 1e-5

    def test_the_layer_merges_into_an_embedding_too(self) -> None:
        net = nn.Sequential(nn.Embedding(10, 4))
        peft = get_peft_model(net, _config(init_weights=False, target_modules=["0"]))
        ids = lucid.tensor([[1, 2, 3], [4, 4, 9]])
        with lucid.no_grad():
            adapted = peft(ids)
            peft.base_model.merge_adapter()
            assert _span(peft(ids), adapted) < 1e-5


class TestWeightTying:
    @pytest.mark.parametrize(
        ("tying", "groups"), [(0.0, 4), (0.25, 3), (0.5, 2), (0.75, 1), (1.0, 1)]
    )
    def test_the_modules_are_cut_into_consecutive_groups(self, tying: float, groups: int) -> None:
        peft = get_peft_model(
            _stack(4), _config(weight_tying=tying, target_modules=["0", "1", "2", "3"])
        )
        vectors = peft.base_model.tinylora_v["default"]
        assert sorted(vectors.keys()) == [str(g) for g in range(groups)]
        size = max(1, 4 // groups)
        for index in range(4):
            expected = str(min(index // size, groups - 1))
            assert _layer(peft, str(index))._spec("default").group == expected

    def test_a_group_shares_one_vector_and_the_model_trains_one_per_group(self) -> None:
        peft = get_peft_model(
            _stack(4), _config(weight_tying=0.5, u=5, target_modules=["0", "1", "2", "3"])
        )
        trainable = [p for p in peft.parameters() if p.requires_grad]
        assert [tuple(p.shape) for p in trainable] == [(5,), (5,)]

    def test_full_tying_trains_u_numbers_for_the_whole_model(self) -> None:
        peft = get_peft_model(
            _stack(4), _config(weight_tying=1.0, u=7, target_modules=["0", "1", "2", "3"])
        )
        assert sum(int(p.numel()) for p in peft.parameters() if p.requires_grad) == 7

    def test_every_vector_gets_a_gradient(self) -> None:
        peft = get_peft_model(
            _stack(4),
            _config(weight_tying=0.5, init_weights="uniform", target_modules=["0", "1", "2", "3"]),
        )
        peft(lucid.randn(5, 6)).sum().backward()
        grads = [p.grad for p in peft.parameters() if p.requires_grad]
        assert len(grads) == 2 and all(
            g is not None and float(g.abs().max().item()) > 0 for g in grads
        )


class TestMerging:
    def test_merged_is_the_adapted_model_and_unmerging_restores_the_weights(self) -> None:
        net = _mlp()
        own = list(net.parameters())
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


class TestMultipleAdapters:
    def _two(self) -> tuple[PeftModel, lucid.Tensor]:
        peft = get_peft_model(_mlp(), _config(), adapter_name="a")
        peft.add_adapter("b", _config(u=5, weight_tying=1.0, target_modules=["0"]))
        _move(peft)
        peft.set_adapter("b")
        _move(peft, seed=4)
        return peft, lucid.randn(5, 6)

    def test_each_adapter_has_vectors_of_its_own(self) -> None:
        peft, _ = self._two()
        table = peft.base_model.tinylora_v
        assert {k: tuple(v.shape) for k, v in table["a"].items()} == {"0": (3,), "1": (3,)}
        assert {k: tuple(v.shape) for k, v in table["b"].items()} == {"0": (5,)}

    def test_two_active_adapters_merge_to_the_adapted_output(self) -> None:
        peft, x = self._two()
        peft.set_adapter(["a", "b"])
        with lucid.no_grad():
            both = peft(x)
            peft.base_model.merge_adapter()
            assert _span(peft(x), both) < 1e-5

    def test_deleting_an_adapter_takes_its_vectors_and_factors(self) -> None:
        peft, x = self._two()
        peft.set_adapter("a")
        with lucid.no_grad():
            before = peft(x)
        peft.delete_adapter("b")
        assert "b" not in peft.base_model.tinylora_v
        layer = _layer(peft)
        assert all(
            "b" not in table for table in (layer.tinylora_A, layer.tinylora_B, layer.tinylora_P)
        )
        with lucid.no_grad():
            assert _same(peft(x), before)

    def test_adapters_must_all_save_the_projection_or_none(self) -> None:
        peft = get_peft_model(_mlp(), _config())
        with pytest.raises(ValueError, match="all adapters or none"):
            peft.add_adapter("b", _config(save_projection=False))
        assert sorted(peft.peft_config) == ["default"]


class TestFiles:
    def test_the_file_is_the_references(self) -> None:
        state = get_peft_model_state_dict(get_peft_model(_mlp(), _config()))
        assert {k: tuple(v.shape) for k, v in state.items()} == {
            "base_model.model.0.tinylora_A": (2, 6),
            "base_model.model.0.tinylora_B": (8, 2),
            "base_model.model.0.tinylora_P": (3, 2, 2),
            "base_model.model.2.tinylora_A": (2, 8),
            "base_model.model.2.tinylora_B": (4, 2),
            "base_model.model.2.tinylora_P": (3, 2, 2),
            "base_model.tinylora_v.0": (3,),
            "base_model.tinylora_v.1": (3,),
        }

    @pytest.mark.filterwarnings("ignore:Specified to not")
    def test_without_the_projection_the_file_holds_the_factors_and_vectors_only(self) -> None:
        state = get_peft_model_state_dict(get_peft_model(_mlp(), _config(save_projection=False)))
        assert not any(key.endswith("tinylora_P") for key in state)
        assert (
            any(key.endswith("tinylora_A") for key in state) and "base_model.tinylora_v.0" in state
        )

    @pytest.mark.parametrize("save_projection", [True, False])
    @pytest.mark.filterwarnings("ignore:Specified to not")
    def test_save_then_load_reproduces_the_adapter(
        self, save_projection: bool, device: str, tmp_path: Path
    ) -> None:
        trained = get_peft_model(_mlp(device), _config(save_projection=save_projection))
        _move(trained)
        x = lucid.randn(4, 6, device=device)
        with lucid.no_grad():
            expected = trained(x)
        trained.save_pretrained(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded = PeftModel.from_pretrained(_mlp(device), tmp_path)
        assert type(loaded.base_model) is TinyLoraModel
        with lucid.no_grad():
            assert _same(loaded(x), expected)

    def test_a_file_without_its_vectors_is_refused(self) -> None:
        peft = get_peft_model(_mlp(), _config())
        state = {k: v for k, v in get_peft_model_state_dict(peft).items() if "tinylora_v" not in k}
        with pytest.raises(ValueError, match="tinylora_v"):
            set_peft_model_state_dict(peft, state)

    @pytest.mark.filterwarnings("ignore:Found missing adapter keys")
    def test_a_file_without_the_projection_it_asks_for_regenerates_it_and_warns(self) -> None:
        peft = get_peft_model(_mlp(), _config())
        state = {k: v for k, v in get_peft_model_state_dict(peft).items() if "tinylora_P" not in k}
        with pytest.warns(UserWarning, match="regenerated from the projection_seed"):
            set_peft_model_state_dict(peft, state)


class TestErrors:
    def test_a_convolution_is_not_adapted(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            get_peft_model(nn.Sequential(nn.Conv2d(2, 3, 3)), _config(target_modules=["0"]))

    def test_the_layer_alone_refuses_fan_in_fan_out(self) -> None:
        with pytest.raises(ValueError, match="fan_in_fan_out"):
            tinylora.Linear(nn.Linear(4, 3), "default", _config(fan_in_fan_out=True))

    def test_the_constructor_takes_the_references_keywords_only(self) -> None:
        with pytest.raises(TypeError, match="unexpected keyword"):
            TinyLoraModel(_mlp(), _config(), "default", mystery=1)
        with pytest.raises(TypeError, match="state_dict"):
            TinyLoraModel(_mlp(), _config(), "default", state_dict=3)

    def test_the_layer_alone_works_without_a_tuner(self) -> None:
        layer = tinylora.Linear(nn.Linear(4, 3), "default", _config(target_modules=["x"]))
        assert tuple(layer(lucid.randn(2, 4)).shape) == (2, 3)
        assert tuple(layer.get_delta_weight("default").shape) == (3, 4)


@pytest.mark.parity
@pytest.mark.parametrize("tying", [0.0, 1.0])
def test_an_adapter_the_reference_wrote_loads_and_computes_the_same(
    ref: ModuleType, tmp_path: Path, tying: float
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
        library.TinyLoraConfig(
            r=2, u=3, weight_tying=tying, target_modules=["0", "2"], init_weights="uniform"
        ),
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
