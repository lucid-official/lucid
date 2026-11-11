"""PVeRA: the Gaussian latent in closed form, the shared projections, the merge at the mean, the file and the reference's numbers.

PVeRA (Fillioux et al., 2025) is VeRA whose rank-``r`` latent is a Gaussian: ``mu, logvar = split(x A * lambda_d)``,
``z = mu + eps * sigma`` while training, ``z = mu`` for inference, and the update ``lambda_b * (z B)``.  The tests
expect those objects from plain loops (``_bank_forms``): the split latent (Eq. 5), the merged update at the mean
(Eq. 10) and, with a seeded sampler, the reparameterised draw (Eq. 6).  The ``parity`` tests build the same adapter
in the reference library and compare.
"""

import warnings
from pathlib import Path
from types import ModuleType

import pytest

import lucid
import lucid.nn as nn
from lucid.peft import PeftModel, PveraConfig, PveraModel, get_peft_model, get_peft_model_state_dict
from lucid.peft.tuners import pvera
from lucid.test.unit.peft import _bank_forms as forms
from lucid.test.unit.peft._parity import adapter_library


def _config(**kw: object) -> PveraConfig:
    fields: dict[str, object] = {"r": 3, "target_modules": ["0", "2"]} | kw
    return PveraConfig.from_dict(fields)


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


def _layer(peft: PeftModel, index: str = "0") -> pvera.PveraLayer:
    layer = peft.base_model.model.get_submodule(index)
    assert isinstance(layer, pvera.PveraLayer)
    return layer


class TestConfig:
    def test_the_defaults_are_the_references(self) -> None:
        config = PveraConfig(target_modules=["a"])
        assert (config.r, config.projection_prng_key, config.save_projection) == (256, 0, True)
        assert (config.pvera_dropout, config.d_initial, config.bias) == (0.0, 0.1, "none")
        assert (config.init_weights, config.sample_at_inference) == (True, False)
        assert config.generator_seed is None

    def test_fields_are_checked(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            PveraConfig(r=0, target_modules=["a"])
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            PveraConfig(pvera_dropout=1.0, target_modules=["a"])
        with pytest.raises(TypeError, match="generator_seed"):
            PveraConfig(generator_seed="x", target_modules=["a"])  # type: ignore[arg-type]

    def test_not_saving_the_projections_warns(self) -> None:
        with pytest.warns(UserWarning, match="pvera_A and pvera_B"):
            _config(save_projection=False)

    def test_a_bias_the_adapter_does_not_train_is_refused(self) -> None:
        with pytest.raises(NotImplementedError, match="bias='all'"):
            _config(bias="all")


class TestProjections:
    def test_a_is_twice_the_rank_wide_and_b_is_one_rank_wide(self) -> None:
        peft = get_peft_model(_mlp(), _config(r=3))
        table = peft.base_model
        # sized for the largest layer: 8 out, 8 in
        assert tuple(table.pvera_A["default"].shape) == (6, 8)
        assert tuple(table.pvera_B["default"].shape) == (8, 3)

    def test_the_key_draws_the_same_projections_on_every_model(self) -> None:
        first = get_peft_model(_mlp(), _config(projection_prng_key=5))
        second = get_peft_model(_mlp(), _config(projection_prng_key=5))
        other = get_peft_model(_mlp(), _config(projection_prng_key=6))
        assert _same(first.base_model.pvera_A["default"], second.base_model.pvera_A["default"])
        assert not _same(first.base_model.pvera_A["default"], other.base_model.pvera_A["default"])

    def test_the_projections_are_frozen_and_the_vectors_alone_train(self) -> None:
        peft = get_peft_model(_mlp(), _config())
        trainable = {n for n, p in peft.named_parameters() if p.requires_grad}
        assert trainable == {
            f"base_model.model.{i}.pvera_lambda_{v}.default" for i in "02" for v in "bd"
        }
        assert not peft.base_model.pvera_A["default"].requires_grad

    def test_adapters_alias_the_first_adapters_projections(self) -> None:
        peft = get_peft_model(_mlp(), _config(), adapter_name="a")
        peft.add_adapter("b", _config(r=2))
        tuner = peft.base_model
        assert tuner.pvera_A["b"] is tuner.pvera_A["a"] and tuner.pvera_B["b"] is tuner.pvera_B["a"]
        assert tuple(_layer(peft).pvera_lambda_d["b"].shape) == (4,)

    def test_an_adapter_of_a_higher_rank_does_not_fit_the_projections(self) -> None:
        peft = get_peft_model(_mlp(), _config(r=2))
        with pytest.raises(ValueError, match="higher rank"):
            peft.add_adapter("b", _config(r=4))
        assert sorted(peft.peft_config) == ["default"]

    def test_adapters_must_share_the_key(self) -> None:
        peft = get_peft_model(_mlp(), _config(projection_prng_key=1))
        with pytest.raises(ValueError, match="PRNG initialisation key"):
            peft.add_adapter("b", _config(projection_prng_key=2))


class TestLatent:
    """Eq. 5-7: ``mu, logvar`` are the halves of ``x A * lambda_d``; ``z = mu + eps * sigma``; the update is ``lambda_b * (z B)``."""

    def _setup(self, **kw: object) -> tuple[PveraConfig, nn.Linear, pvera.Linear]:
        lucid.manual_seed(0)  # the same base and the same vectors every time
        config = _config(init_weights=False, target_modules=["x"], **kw)
        base = nn.Linear(8, 4)
        return config, base, pvera.Linear(base, "default", config)

    def test_the_latent_is_the_two_halves_of_the_scaled_projection(self) -> None:
        _, _, layer = self._setup()
        x = lucid.randn(5, 8)
        proj_a = layer._projection("default").pvera_A
        lambda_d = layer.pvera_lambda_d["default"]
        mu, logvar = forms.pvera_latent(x.tolist(), proj_a.tolist(), lambda_d.tolist(), 8)
        hidden = (x @ proj_a[:, :8].mT) * lambda_d
        assert _span(hidden[..., :3], lucid.tensor(mu)) < 1e-5
        assert _span(hidden[..., 3:], lucid.tensor(logvar)) < 1e-5

    def test_evaluation_uses_the_mean_and_is_deterministic(self) -> None:
        _, base, layer = self._setup()
        layer.eval()
        x = lucid.randn(5, 8)
        proj_a, proj_b = layer._projection("default")
        lambda_d, lambda_b = layer.pvera_lambda_d["default"], layer.pvera_lambda_b["default"]
        mu = ((x @ proj_a[:, :8].mT) * lambda_d)[..., :3]
        expected = base(x) + lambda_b * (mu @ proj_b[:4].mT)
        assert _span(layer(x), expected) < 1e-5
        assert _same(layer(x), layer(x))

    def test_training_draws_the_reparameterised_sample(self) -> None:
        _, base, layer = self._setup(generator_seed=11)
        layer.train()
        x = lucid.randn(5, 8)
        proj_a, proj_b = layer._projection("default")
        lambda_d, lambda_b = layer.pvera_lambda_d["default"], layer.pvera_lambda_b["default"]
        hidden = (x @ proj_a[:, :8].mT) * lambda_d
        mu, logvar = hidden[..., :3], hidden[..., 3:]
        noise = lucid.randn(5, 3, generator=lucid.Generator(11, device="cpu"))
        expected = base(x) + lambda_b * ((mu + noise * (0.5 * logvar).exp()) @ proj_b[:4].mT)
        assert _span(layer(x), expected) < 1e-5

    def test_a_seeded_sampler_repeats_across_models_and_moves_between_calls(self) -> None:
        _, _, first = self._setup(generator_seed=3)
        _, _, second = self._setup(generator_seed=3)
        x = lucid.randn(5, 8)
        for layer in (first, second):
            layer.train()
        out = first(x)
        assert _same(out, second(x))
        assert not _same(first(x), out)  # the stream moves

    def test_sample_at_inference_samples_in_evaluation_too(self) -> None:
        _, _, layer = self._setup(sample_at_inference=True)
        layer.eval()
        x = lucid.randn(5, 8)
        assert not _same(layer(x), layer(x))
        _, _, quiet = self._setup()
        quiet.eval()
        assert _same(quiet(x), quiet(x))

    def test_the_merged_update_is_the_one_at_the_mean(self) -> None:
        _, _, layer = self._setup()
        proj_a, proj_b = layer._projection("default")
        expected = forms.pvera_mean_delta(
            [row[:8] for row in proj_a.tolist()],
            proj_b.tolist()[:4],
            layer.pvera_lambda_d["default"].tolist(),
            layer.pvera_lambda_b["default"].tolist(),
            (4, 8),
        )
        assert _span(layer.get_delta_weight("default"), lucid.tensor(expected)) < 1e-5
        layer.eval()
        x = lucid.randn(5, 8)
        assert (
            _span(
                layer(x), nn.functional.linear(x, layer.weight + lucid.tensor(expected), layer.bias)
            )
            < 1e-5
        )


class TestStart:
    def test_the_default_start_is_the_base_even_while_sampling(self) -> None:
        net = _mlp()
        x = lucid.randn(5, 6)
        base = net(x)
        peft = get_peft_model(net, _config())
        peft.train()
        assert _same(peft(x), base)  # lambda_b starts at zero

    def test_the_vectors_start_where_the_reference_puts_them(self) -> None:
        peft = get_peft_model(_mlp(), _config(d_initial=0.25))
        layer = _layer(peft)
        assert float(layer.pvera_lambda_b["default"].abs().max().item()) == 0.0
        assert bool((layer.pvera_lambda_d["default"] == 0.25).all().item())
        random = _layer(get_peft_model(_mlp(), _config(init_weights=False)))
        assert bool((random.pvera_lambda_b["default"] == 1.0).all().item())
        assert float(random.pvera_lambda_d["default"].std().item()) > 0.1


class TestMerging:
    def test_merged_is_the_adapted_model_at_the_mean_and_unmerging_restores_the_weights(
        self,
    ) -> None:
        net = _mlp()
        own = list(net.parameters())
        weights = [p.detach().clone() for p in own]
        peft = get_peft_model(net, _config())
        _move(peft)
        peft.eval()
        x = lucid.randn(5, 6)
        with lucid.no_grad():
            adapted = peft(x)
            peft.base_model.merge_adapter()
            assert _span(peft(x), adapted) < 1e-5
            peft.base_model.unmerge_adapter()
            assert _span(peft(x), adapted) < 1e-5
        for old, new in zip(weights, own, strict=True):
            assert _span(old, new) < 1e-6 * max(1.0, float(old.abs().max().item()))


class TestFiles:
    def test_the_file_is_the_references(self) -> None:
        state = get_peft_model_state_dict(get_peft_model(_mlp(), _config()))
        assert {k: tuple(v.shape) for k, v in state.items()} == {
            "base_model.model.0.pvera_lambda_b": (8,),
            "base_model.model.0.pvera_lambda_d": (6,),
            "base_model.model.2.pvera_lambda_b": (4,),
            "base_model.model.2.pvera_lambda_d": (6,),
            "base_model.pvera_A": (6, 8),
            "base_model.pvera_B": (8, 3),
        }

    @pytest.mark.filterwarnings("ignore:Specified to not save")
    def test_without_the_projections_the_file_is_small_and_the_key_restores_them(
        self, tmp_path: Path
    ) -> None:
        with pytest.warns(UserWarning, match="pvera_A and pvera_B"):
            trained = get_peft_model(_mlp(), _config(save_projection=False, projection_prng_key=4))
        _move(trained)
        trained.eval()
        assert sorted(get_peft_model_state_dict(trained)) == [
            f"base_model.model.{i}.pvera_lambda_{v}" for i in "02" for v in "bd"
        ]
        trained.save_pretrained(tmp_path)
        x = lucid.randn(4, 6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded = PeftModel.from_pretrained(_mlp(), tmp_path)
        loaded.eval()
        with lucid.no_grad():
            assert _same(loaded(x), trained(x))

    def test_save_then_load_reproduces_the_adapter(self, device: str, tmp_path: Path) -> None:
        trained = get_peft_model(_mlp(device), _config(init_weights=False))
        trained.eval()
        x = lucid.randn(4, 6, device=device)
        with lucid.no_grad():
            expected = trained(x)
        trained.save_pretrained(tmp_path)
        loaded = PeftModel.from_pretrained(_mlp(device), tmp_path)
        loaded.eval()
        assert type(loaded.base_model) is PveraModel
        with lucid.no_grad():
            assert _same(loaded(x), expected)

    def test_a_file_without_the_projections_it_asks_for_is_refused(self) -> None:
        from lucid.peft import set_peft_model_state_dict

        peft = get_peft_model(_mlp(), _config())
        state = get_peft_model_state_dict(peft)
        del state["base_model.pvera_A"]
        with pytest.raises(ValueError, match="pvera_A and pvera_B"):
            set_peft_model_state_dict(peft, state)


class TestErrors:
    def test_a_convolution_is_not_adapted(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            get_peft_model(nn.Sequential(nn.Conv2d(2, 3, 3)), _config(target_modules=["0"]))

    def test_the_layer_alone_refuses_fan_in_fan_out(self) -> None:
        with pytest.raises(ValueError, match="fan_in_fan_out"):
            pvera.Linear(nn.Linear(4, 3), "default", _config(fan_in_fan_out=True))

    def test_the_layer_alone_works_without_a_tuner(self) -> None:
        layer = pvera.Linear(nn.Linear(4, 3), "default", _config(target_modules=["x"], r=2))
        assert tuple(layer(lucid.randn(2, 4)).shape) == (2, 3)
        assert not any("pvera_A" in k for k in layer.state_dict())


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
    theirs = library.get_peft_model(twin, library.PveraConfig(r=3, target_modules=["0", "2"]))
    with ref.no_grad():
        for p in theirs.parameters():
            if p.requires_grad:
                p.add_(0.5 * ref.randn_like(p))
    theirs.eval()
    theirs.save_pretrained(tmp_path)
    x = ref.randn(5, 6)
    with ref.no_grad():
        want = theirs(x).tolist()
    loaded = PeftModel.from_pretrained(net, tmp_path)
    loaded.eval()
    with lucid.no_grad():
        assert _span(loaded(lucid.tensor(x.tolist())), lucid.tensor(want)) < 1e-5
    held = {k: tuple(v.shape) for k, v in get_peft_model_state_dict(loaded).items()}
    assert held == {k: tuple(v.shape) for k, v in library.get_peft_model_state_dict(theirs).items()}
