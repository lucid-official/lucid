"""OSF: the split in closed form, the gradient projection and what it preserves, the starts, the targets and the reference adapter library's numbers.

The adapted weight is expected from ``_spectral_forms`` (``U_high S_high V_high^T`` of the base plus the
file's low part, the SVD from the primitive: the products are the same whichever signs it picks).  The
projection of section 3.5 is checked against ``I - U_high U_high^T`` applied to the gradient the same loss
gives without it, and by what the paper wants of it: after real optimizer steps the frozen directions of
the weight are still the frozen ones.  The ``parity`` tests compare the forward pass, the update, the
merged weight and the projected gradients with the reference library's layer.
"""

import copy
from types import ModuleType

import pytest

import lucid
import lucid.linalg as linalg
import lucid.nn as nn
import lucid.optim as optim
from lucid.peft import OSFConfig, OSFModel, PeftModel, get_peft_model, get_peft_model_state_dict
from lucid.peft.tuners import osf
from lucid.test.unit.peft import _spectral_forms as forms
from lucid.test.unit.peft._parity import adapter_library, gap
from lucid.test.unit.peft._tuner_recipes import magnitude, span

#: Relative to the largest value compared: a float32 product and an SVD round.
_TOLERANCE = 1e-5


def _layer(
    in_features: int = 8, out_features: int = 6, **kw: object
) -> tuple[osf.Linear, nn.Linear]:
    lucid.manual_seed(1)
    base = nn.Linear(in_features, out_features)
    fields: dict[str, object] = {"effective_rank": 2, "target_modules": ["x"]} | kw
    return osf.Linear(base, "default", OSFConfig(**fields)), base  # type: ignore[arg-type]


def _top(weight: lucid.Tensor, frozen: int) -> tuple[lucid.Tensor, lucid.Tensor, lucid.Tensor]:
    """The frozen top ``(U_high, S_high, V_high^T)`` of ``weight``, from the SVD primitive."""
    u, s, vt = linalg.svd(weight.detach(), full_matrices=False)
    return u[:, :frozen], s[:frozen], vt[:frozen]


class TestConfig:
    def test_the_defaults_are_the_references(self) -> None:
        config = OSFConfig(target_modules=["a"])
        assert (config.effective_rank, config.rank_pattern, config.init_weights) == (
            None,
            None,
            True,
        )

    def test_fields_are_checked(self) -> None:
        with pytest.raises(TypeError, match="effective_rank"):
            OSFConfig(effective_rank="half", target_modules=["a"])  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="init_weights"):
            OSFConfig(init_weights="yes", target_modules=["a"])  # type: ignore[arg-type]

    def test_a_file_round_trips(self) -> None:
        config = OSFConfig(effective_rank=0.5, rank_pattern={"q": 2}, target_modules=["q", "v"])
        assert OSFConfig.from_dict(config.to_dict()) == config
        assert config.to_dict()["peft_type"] == "OSF"


class TestSplit:
    def test_the_trained_tensors_are_the_low_triplets_and_the_rest_is_frozen(self) -> None:
        layer, base = _layer(in_features=8, out_features=6, effective_rank=2)
        params = layer.osf_svd_params["default"]
        assert {n: tuple(p.shape) for n, p in params.items()} == {
            "U_low": (6, 4),
            "S_low": (4,),
            "V_low": (4, 8),
        }
        assert layer.effective_rank == {"default": 2}
        _, s, _ = linalg.svd(base.weight.detach(), full_matrices=False)
        assert span(params["S_low"], s[2:]) <= _TOLERANCE * magnitude(s)
        # Nothing but those three trains; the frozen copies are buffers the file does not hold.
        assert [n for n, _ in layer.named_parameters() if "osf" in n] == [
            "osf_svd_params.default.U_low",
            "osf_svd_params.default.S_low",
            "osf_svd_params.default.V_low",
        ]
        assert not any("init" in k or "high" in k for k in layer.state_dict())

    @pytest.mark.parametrize(
        ("value", "frozen"),
        [(None, 3), (2, 2), (0.5, 3), (1.0, 6), (0.34, 2), (4.0, 4), (100, 6)],
        ids=str,
    )
    def test_the_rank_is_a_count_a_fraction_or_half(self, value: object, frozen: int) -> None:
        layer, _ = _layer(effective_rank=value)
        assert layer.effective_rank["default"] == frozen
        assert layer.osf_svd_params["default"]["S_low"].shape[0] == 6 - frozen

    def test_a_high_factor_is_kept_only_where_the_low_one_cannot_stand_in(self) -> None:
        tall, _ = _layer(in_features=4, out_features=6)
        wide, _ = _layer(in_features=6, out_features=4)
        square, _ = _layer(in_features=5, out_features=5)
        assert (list(tall._osf_U_high), list(tall._osf_V_high)) == (["default"], [])
        assert (list(wide._osf_U_high), list(wide._osf_V_high)) == ([], ["default"])
        assert (list(square._osf_U_high), list(square._osf_V_high)) == ([], [])

    def test_a_rank_that_comes_out_as_zero_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            _layer(effective_rank=0)
        with pytest.raises(ValueError, match="positive integer"):
            _layer(effective_rank=0.01)

    def test_freezing_everything_leaves_an_adapter_with_nothing_to_train_or_change(self) -> None:
        layer, base = _layer(effective_rank=1.0)
        assert layer.osf_svd_params["default"]["S_low"].shape[0] == 0
        x = lucid.randn(3, 8)
        assert bool((layer(x) == base(x)).all().item())
        assert tuple(layer.get_delta_weight("default").shape) == (6, 8)


class TestStarts:
    def test_the_adapter_starts_as_the_identity(self, device: str) -> None:
        layer, base = _layer()
        layer.to(device)
        x = lucid.randn(5, 8, device=device)
        assert bool((layer(x) == base(x)).all().item())
        assert bool((layer.get_delta_weight("default") == 0).all().item())

    def test_a_random_start_moves_the_output_and_keeps_the_frozen_copies(self) -> None:
        layer, base = _layer(init_weights=False)
        before = layer._osf_U_low_init["default"].clone()
        x = lucid.randn(5, 8)
        assert span(layer(x), base(x)) > 1e-2
        assert span(layer._osf_U_low_init["default"], before) == 0.0


class TestUpdate:
    @pytest.mark.parametrize("shape", [(8, 6), (6, 8), (7, 7)], ids=["tall", "wide", "square"])
    def test_the_layer_computes_with_the_high_part_of_the_weight_and_the_trained_low_part(
        self, shape: tuple[int, int], device: str
    ) -> None:
        layer, base = _layer(in_features=shape[0], out_features=shape[1], init_weights=False)
        u, s, vt = linalg.svd(base.weight.detach(), full_matrices=False)
        params = layer.osf_svd_params["default"]
        expected = lucid.tensor(
            forms.osf_adapted(
                u.tolist(),
                s.tolist(),
                vt.tolist(),
                params["U_low"].tolist(),
                params["S_low"].tolist(),
                params["V_low"].tolist(),
                2,
            )
        )
        layer.to(device)
        x = lucid.randn(4, shape[0], device=device)
        with lucid.no_grad():
            adapted = layer(x)
            assert span(adapted, base.bias + x @ expected.to(device).mT) <= _TOLERANCE * magnitude(
                adapted
            )
        assert span(base.weight + layer.get_delta_weight("default"), expected.to(device)) <= (
            _TOLERANCE * magnitude(expected)
        )

    def test_the_difference_of_the_two_products_is_zero_where_they_are_equal(self) -> None:
        layer, base = _layer()
        x = lucid.randn(3, 8)
        with lucid.no_grad():
            assert float(layer._contribution("default", x).abs().max().item()) == 0.0

    def test_two_adapters_add_their_updates_to_the_weight(self) -> None:
        net = nn.Sequential(nn.Linear(8, 6))
        config = OSFConfig(effective_rank=2, init_weights=False, target_modules=["0"])
        peft = get_peft_model(net, config)
        peft.add_adapter("b", config)
        peft.set_adapter(["default", "b"])
        layer = peft.base_model.model[0]
        x = lucid.randn(4, 8)
        weight = layer.base_layer.weight
        expected = weight + layer.get_delta_weight("default") + layer.get_delta_weight("b")
        with lucid.no_grad():
            both = peft(x)
        assert span(both, layer.base_layer.bias + x @ expected.mT) <= _TOLERANCE * magnitude(both)

    def test_an_adapter_cannot_be_added_to_a_merged_layer(self) -> None:
        net = nn.Sequential(nn.Linear(8, 6))
        peft = get_peft_model(net, OSFConfig(effective_rank=2, target_modules=["0"]))
        peft.base_model.merge_adapter()
        with pytest.raises(ValueError, match="merged layer"):
            peft.add_adapter("b", OSFConfig(effective_rank=2, target_modules=["0"]))
        assert sorted(peft.peft_config) == ["default"]

    def test_merging_adds_the_delta_and_unmerging_takes_it_out(self) -> None:
        layer, base = _layer(init_weights=False)
        weight = base.weight.clone()
        delta = layer.get_delta_weight("default")
        layer.merge()
        assert span(base.weight, weight + delta) <= _TOLERANCE * magnitude(weight + delta)
        layer.unmerge()
        assert span(base.weight, weight) <= _TOLERANCE * magnitude(weight + delta)


class TestGradientProjection:
    """Section 3.5, Eq. 4: the gradients of the low factors stay out of the frozen directions."""

    @staticmethod
    def _loss(layer: osf.Linear, x: lucid.Tensor) -> lucid.Tensor:
        return (layer(x) ** 2).sum()

    @pytest.mark.parametrize("shape", [(8, 6), (6, 8), (7, 7)], ids=["tall", "wide", "square"])
    def test_the_gradient_of_each_factor_is_the_papers_projection(
        self, shape: tuple[int, int], device: str
    ) -> None:
        layer, base = _layer(in_features=shape[0], out_features=shape[1], init_weights=False)
        u_high, _, v_high = _top(base.weight, 2)
        layer.to(device)
        x = lucid.randn(5, shape[0], device=device)
        u, s, v = layer._trained("default")
        # The gradient the same loss gives when the weight is built from the factors directly.
        initial = (
            layer._osf_U_low_init["default"] * layer._osf_S_low_init["default"]
        ) @ layer._osf_V_low_init["default"]
        weight = base.weight + (u * s) @ v - initial
        ((x @ weight.mT + base.bias) ** 2).sum().backward()
        raw_u, raw_s, raw_v = u.grad.clone(), s.grad.clone(), v.grad.clone()
        for p in layer.osf_svd_params["default"].parameters():
            p.grad = None
        self._loss(layer, x).backward()
        u_high, v_high = u_high.to(device), v_high.to(device)
        left = raw_u - u_high @ (u_high.mT @ raw_u)
        right = raw_v - (raw_v @ v_high.mT) @ v_high
        assert span(u.grad, left) <= _TOLERANCE * magnitude(raw_u)
        assert span(v.grad, right) <= _TOLERANCE * magnitude(raw_v)
        # The singular values are not constrained.
        assert span(s.grad, raw_s) <= _TOLERANCE * magnitude(raw_s)
        assert magnitude(u_high.mT @ u.grad) <= _TOLERANCE * magnitude(raw_u)

    def test_training_leaves_the_frozen_directions_of_the_weight_where_they_were(self) -> None:
        layer, base = _layer(in_features=8, out_features=6, effective_rank=2)
        u_high, s_high, v_high = _top(base.weight, 2)
        optimizer = optim.SGD(list(layer.osf_svd_params.parameters()), lr=0.05)
        lucid.manual_seed(4)
        x = lucid.randn(16, 8)
        target = lucid.randn(16, 6)
        for _ in range(25):
            optimizer.zero_grad()
            ((layer(x) - target) ** 2).mean().backward()
            optimizer.step()
        trained = base.weight + layer.get_delta_weight("default")
        # The weight moved a lot ...
        assert span(trained, base.weight) > 1e-2
        # ... and the frozen part of it did not: U_high^T W' = S_high V_high^T and W' V_high^T = U_high S_high.
        assert span(u_high.mT @ trained, s_high.unsqueeze(1) * v_high) <= 1e-4
        assert span(trained @ v_high.mT, u_high * s_high) <= 1e-4

    def test_without_the_projection_the_frozen_directions_drift(self) -> None:
        layer, base = _layer(in_features=8, out_features=6, effective_rank=2)
        u_high, s_high, v_high = _top(base.weight, 2)
        optimizer = optim.SGD(list(layer.osf_svd_params.parameters()), lr=0.05)
        lucid.manual_seed(4)
        x = lucid.randn(16, 8)
        target = lucid.randn(16, 6)
        for _ in range(25):
            optimizer.zero_grad()
            # The weight built from the factors, which no hook sits on.
            u, s, v = layer._trained("default")
            weight = (
                base.weight
                + (u * s) @ v
                - (
                    (layer._osf_U_low_init["default"] * layer._osf_S_low_init["default"])
                    @ layer._osf_V_low_init["default"]
                )
            )
            ((x @ weight.mT + base.bias - target) ** 2).mean().backward()
            optimizer.step()
        trained = base.weight + layer.get_delta_weight("default")
        assert span(u_high.mT @ trained, s_high.unsqueeze(1) * v_high) > 1e-3

    def test_the_projection_follows_the_model_to_another_device_and_into_a_copy(
        self, device: str
    ) -> None:
        layer, base = _layer(init_weights=False)
        u_high, _, _ = _top(base.weight, 2)
        clone = copy.deepcopy(layer)
        layer.to(device)
        for model, high in ((layer, u_high.to(device)), (clone, u_high)):
            x = lucid.randn(5, 8, device=model.weight.device)
            (model(x) ** 2).sum().backward()
            u = model.osf_svd_params["default"]["U_low"]
            assert magnitude(high.mT @ u.grad) <= _TOLERANCE * magnitude(u.grad)

    def test_nothing_is_hooked_without_a_backward_pass_to_come(self) -> None:
        layer, _ = _layer(init_weights=False)
        x = lucid.randn(3, 8)
        with lucid.no_grad():
            u, _, _ = layer._guarded("default")
        assert u is layer.osf_svd_params["default"]["U_low"]
        layer.osf_svd_params["default"].requires_grad_(False)
        u, _, v = layer._guarded("default")
        assert not u.requires_grad and not v.requires_grad
        assert layer(x).shape == (3, 6)


class TestTargets:
    def test_a_model_type_without_a_default_takes_every_linear_layer(self) -> None:
        net = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.Linear(6, 6), nn.Linear(6, 2))
        peft = get_peft_model(net, OSFConfig(effective_rank=1))
        assert peft.base_model.targeted_module_names == ["0", "2", "3"]

    def test_gpt2_has_a_row_of_its_own(self) -> None:
        from lucid.models import create_model

        lucid.manual_seed(0)
        model = create_model(
            "gpt2_small_lm",
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=16,
            tie_word_embeddings=False,
        )
        peft = get_peft_model(model, OSFConfig(effective_rank=2))
        assert {n.rpartition(".")[2] for n in peft.base_model.targeted_module_names} == {
            "c_attn",
            "c_proj",
        }

    def test_the_first_matching_pattern_sets_the_rank_of_a_module(self) -> None:
        net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8))
        config = OSFConfig(
            effective_rank=2, rank_pattern={"^2$": 0.75, "^[02]$": 5}, target_modules=["0", "2"]
        )
        peft = get_peft_model(net, config)
        first, last = peft.base_model.model[0], peft.base_model.model[2]
        # ``re.search``, dictionary order: "2" matches the first pattern, "0" only the second.
        assert (first.effective_rank, last.effective_rank) == ({"default": 5}, {"default": 6})
        assert peft.peft_config["default"].rank_pattern == {"^2$": 0.75, "^[02]$": 5}

    def test_the_pattern_also_reaches_an_adapter_added_later(self) -> None:
        net = nn.Sequential(nn.Linear(8, 8))
        peft = get_peft_model(
            net, OSFConfig(effective_rank=2, target_modules=["0"]), adapter_name="a"
        )
        peft.add_adapter("b", OSFConfig(rank_pattern={"0": 3}, target_modules=["0"]))
        assert peft.base_model.model[0].effective_rank == {"a": 2, "b": 3}

    def test_only_a_linear_is_adapted(self) -> None:
        net = nn.Sequential(nn.Conv2d(2, 4, 3))
        with pytest.raises(ValueError, match="not supported"):
            get_peft_model(net, OSFConfig(effective_rank=1, target_modules=["0"]))

    def test_a_refused_adapter_leaves_the_model_as_it_was(self) -> None:
        net = nn.Sequential(nn.Linear(4, 6))
        with pytest.raises(ValueError, match="positive integer"):
            get_peft_model(net, OSFConfig(effective_rank=0, target_modules=["0"]))
        assert type(net[0]) is nn.Linear and all(p.requires_grad for p in net.parameters())


def test_unmerging_takes_no_argument_and_undoes_the_merge() -> None:
    net = nn.Sequential(nn.Linear(4, 6))
    weight = net[0].weight.clone()
    peft = get_peft_model(
        net, OSFConfig(effective_rank=1, init_weights=False, target_modules=["0"])
    )
    peft.base_model.merge_adapter()
    merged = peft.base_model.model[0].base_layer.weight.clone()
    with pytest.raises(TypeError, match="takes no arguments"):
        peft.base_model.unmerge_adapter(1)
    assert peft.base_model.model[0].merged
    peft.base_model.unmerge_adapter()
    assert not peft.base_model.model[0].merged
    scale = _TOLERANCE * max(magnitude(weight), magnitude(merged))
    assert span(peft.base_model.model[0].base_layer.weight, weight) <= scale


def test_the_constructor_names_its_config_as_the_reference_does() -> None:
    net = nn.Sequential(nn.Linear(4, 6))
    tuner = OSFModel(
        net, config=OSFConfig(effective_rank=1, target_modules=["0"]), adapter_name="a"
    )
    assert tuner.targeted_module_names == ["0"]


def test_the_tuner_wraps_the_linear_layers_and_unloads_them() -> None:
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    peft = get_peft_model(net, OSFConfig(effective_rank=1, target_modules=["0", "2"]))
    assert isinstance(peft, PeftModel) and isinstance(peft.base_model, OSFModel)
    assert type(net[0]) is osf.Linear
    assert peft.unload() is net and type(net[0]) is nn.Linear


# ── the reference adapter library ────────────────────────────────────────────


@pytest.mark.parity
@pytest.mark.parametrize("shape", [(6, 8), (8, 6), (7, 7)], ids=["wide", "tall", "square"])
@pytest.mark.parametrize("effective_rank", [1, 2, 0.5, None], ids=str)
@pytest.mark.parametrize("random", [False, True], ids=["start", "random"])
def test_a_layer_matches_the_reference(
    ref: ModuleType, shape: tuple[int, int], effective_rank: object, random: bool
) -> None:
    library = adapter_library()
    lucid.manual_seed(5)
    net = nn.Sequential(nn.Linear(*shape))
    twin = ref.nn.Sequential(ref.nn.Linear(*shape))
    x = lucid.randn(4, shape[0])
    config = {"effective_rank": effective_rank, "init_weights": not random}
    peft = get_peft_model(net, OSFConfig(target_modules=["0"], **config))  # type: ignore[arg-type]
    mine = peft.base_model.model[0]
    with ref.no_grad():
        twin[0].weight.copy_(ref.tensor(mine.base_layer.weight.tolist()))
        twin[0].bias.copy_(ref.tensor(mine.base_layer.bias.tolist()))
    theirs = library.get_peft_model(twin, library.OSFConfig(target_modules=["0"], **config))
    other = theirs.base_model.model[0]
    # What the two compute with, once the trained tensors are the same, does not depend on the SVD's signs.
    with lucid.no_grad():
        for name, value in other.osf_svd_params["default"].items():
            mine.osf_svd_params["default"][name].copy_(lucid.tensor(value.tolist()))
    assert gap(ref, peft(x), theirs(ref.tensor(x.tolist()))) < 1e-5 * max(1.0, magnitude(peft(x)))
    assert gap(ref, mine.get_delta_weight("default"), other.get_delta_weight("default")) < 1e-4
    (peft(x) ** 2).sum().backward()
    (theirs(ref.tensor(x.tolist())) ** 2).sum().backward()
    for name, value in other.osf_svd_params["default"].items():
        scale = max(1.0, float(value.grad.abs().max()))
        assert gap(ref, mine.osf_svd_params["default"][name].grad, value.grad) < 1e-4 * scale
    peft.base_model.merge_adapter()
    theirs.base_model.merge_adapter()
    assert gap(ref, mine.base_layer.weight, other.base_layer.weight) < 1e-4


@pytest.mark.parity
def test_the_state_dict_has_the_references_keys_and_shapes(ref: ModuleType) -> None:
    library = adapter_library()
    for shape in ((10, 12), (12, 10), (9, 9)):
        mine = get_peft_model(
            nn.Sequential(nn.Linear(*shape)), OSFConfig(effective_rank=3, target_modules=["0"])
        )
        theirs = library.get_peft_model(
            ref.nn.Sequential(ref.nn.Linear(*shape)),
            library.OSFConfig(effective_rank=3, target_modules=["0"]),
        )
        expected = {k: tuple(v.shape) for k, v in library.get_peft_model_state_dict(theirs).items()}
        assert {k: tuple(v.shape) for k, v in get_peft_model_state_dict(mine).items()} == expected


@pytest.mark.parity
def test_the_config_file_has_the_references_keys() -> None:
    library = adapter_library()
    ours = OSFConfig(effective_rank=0.5, rank_pattern={"q": 2}, target_modules=["a"]).to_dict()
    theirs = library.OSFConfig(
        effective_rank=0.5, rank_pattern={"q": 2}, target_modules=["a"]
    ).to_dict()
    assert set(ours) == set(theirs) - {"peft_version"}
    assert all(
        ours[k] == (sorted(v) if isinstance(v, set) else v) for k, v in theirs.items() if k in ours
    )
