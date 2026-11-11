"""PSOFT: the update in closed form, the paper's guarantees, the starts, the SVD options and the reference adapter library's numbers.

The update ``A' (C - I) B'`` is expected from ``_spectral_forms`` (the Cayley transform by Gauss-Jordan on
plain lists, the factors from the SVD primitive's output), the paper's Theorem 4.1 is checked on the
merged weight (strict orthogonality keeps the Gram matrix of the principal part), and the ``parity``
tests copy the factors of the reference library's layer across and compare everything built on them.
"""

import warnings
from types import ModuleType

import pytest

import lucid
import lucid.linalg as linalg
import lucid.nn as nn
from lucid.peft import PeftModel, PsoftConfig, PsoftModel, get_peft_model, get_peft_model_state_dict
from lucid.peft.tuners import psoft
from lucid.test.unit.peft import _spectral_forms as forms
from lucid.test.unit.peft._parity import adapter_library, gap
from lucid.test.unit.peft._tuner_recipes import magnitude, span

#: Relative to the largest value compared: a float32 product, a solve and an SVD round.
_TOLERANCE = 1e-5


def _layer(
    in_features: int = 8, out_features: int = 6, **kw: object
) -> tuple[psoft.Linear, nn.Linear]:
    lucid.manual_seed(1)
    base = nn.Linear(in_features, out_features)
    fields: dict[str, object] = {"r": 3, "target_modules": ["x"]} | kw
    return psoft.Linear(base, "default", PsoftConfig(**fields)), base  # type: ignore[arg-type]


def _svd(weight: lucid.Tensor) -> tuple[list[list[float]], list[float], list[list[float]]]:
    """The thin SVD of ``weight`` as lists, from the linear-algebra primitive (the layer's own call)."""
    u, s, vt = linalg.svd(weight.detach(), full_matrices=False)
    return u.tolist(), s.tolist(), vt.tolist()


def _closed_inner(layer: psoft.PsoftLayer, adapter: str = "default") -> list[list[float]]:
    """:math:`C` from the layer's trained tensors, by the formulas."""
    orth = layer._orth(adapter)
    size = orth.size
    rotation = forms.cayley(orth.weight.tolist(), size) if orth.orth else orth.weight.tolist()
    return forms.psoft_inner(
        rotation,
        orth.vector_b.tolist() if orth.vector_b is not None else None,
        orth.vector_a.tolist() if orth.vector_a is not None else None,
    )


def _move(layer: psoft.PsoftLayer, seed: int = 3) -> None:
    """Move every trained tensor well off its start, as training would."""
    lucid.manual_seed(seed)
    with lucid.no_grad():
        for p in layer._adapter_parameters("default"):
            p.copy_(p + 0.3 * lucid.randn(*p.shape, device=p.device))


class TestConfig:
    def test_the_defaults_are_the_references(self) -> None:
        config = PsoftConfig(target_modules=["a"])
        assert (config.r, config.psoft_alpha, config.ab_svd_init, config.psoft_svd) == (
            32,
            32,
            "psoft_init",
            "full",
        )
        assert (config.psoft_orth, config.psoft_mag_a, config.psoft_mag_b) == (True, True, True)
        assert (config.use_cayley_neumann, config.num_cayley_neumann_terms) == (False, 5)

    def test_fields_are_checked(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            PsoftConfig(r=0, target_modules=["a"])
        with pytest.raises(ValueError, match="psoft_init"):
            PsoftConfig(ab_svd_init="svd", target_modules=["a"])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="lowrank"):
            PsoftConfig(psoft_svd="exact", target_modules=["a"])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="positive"):
            PsoftConfig(use_cayley_neumann=True, num_cayley_neumann_terms=0, target_modules=["a"])
        with pytest.raises(ValueError, match="in .0, 1."):
            PsoftConfig(use_cayley_neumann=True, cayley_neumann_eps=1.5, target_modules=["a"])
        with pytest.raises(TypeError, match="psoft_orth"):
            PsoftConfig(psoft_orth="yes", target_modules=["a"])  # type: ignore[arg-type]

    def test_an_option_the_choice_ignores_is_warned_about(self) -> None:
        with pytest.warns(UserWarning, match="psoft_svd_lowrank_niter"):
            PsoftConfig(psoft_svd_lowrank_niter=3, target_modules=["a"])
        with pytest.warns(UserWarning, match="num_cayley_neumann_terms"):
            PsoftConfig(num_cayley_neumann_terms=7, target_modules=["a"])
        with pytest.warns(UserWarning, match="cayley_neumann_eps"):
            PsoftConfig(cayley_neumann_eps=0.5, target_modules=["a"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PsoftConfig(psoft_svd="lowrank", psoft_svd_lowrank_niter=3, target_modules=["a"])
            PsoftConfig(use_cayley_neumann=True, cayley_neumann_eps=0.5, target_modules=["a"])

    def test_a_file_round_trips(self) -> None:
        config = PsoftConfig(r=8, psoft_mag_a=False, target_modules=["q", "v"])
        assert PsoftConfig.from_dict(config.to_dict()) == config
        assert config.to_dict()["peft_type"] == "PSOFT"


class TestStarts:
    def test_the_adapter_starts_as_the_identity(self, device: str) -> None:
        layer, base = _layer()
        layer.to(device)
        x = lucid.randn(5, 8, device=device)
        assert bool((layer(x) == base(x)).all().item())
        assert bool((layer.get_delta_weight("default") == 0).all().item())

    def test_the_trained_tensors_are_the_papers(self) -> None:
        layer, _ = _layer(r=3)
        orth = layer._orth("default")
        # r (r - 1) / 2 entries of the generator and two vectors of r: the paper's count (section 4.3).
        assert tuple(orth.weight.shape) == (3,) and orth.weight.shape[0] == 3 * 2 // 2
        assert orth.vector_b.tolist() == [1.0] * 3 and orth.vector_a.tolist() == [1.0] * 3
        assert sum(p.numel() for p in layer._adapter_parameters("default")) == 3 + 2 * 3

    def test_a_free_matrix_starts_at_the_identity(self) -> None:
        layer, _ = _layer(psoft_orth=False, psoft_mag_a=False, psoft_mag_b=False)
        assert layer._orth("default").weight.tolist() == [
            [1.0 if i == j else 0.0 for j in range(3)] for i in range(3)
        ]
        assert [n for n, _ in layer.psoft_R["default"].named_parameters()] == ["weight"]

    def test_a_random_start_moves_the_output(self) -> None:
        layer, base = _layer(init_weights=False)
        x = lucid.randn(5, 8)
        assert span(layer(x), base(x)) > 1e-3

    def test_the_factors_are_the_top_singular_directions(self) -> None:
        layer, base = _layer(r=3)
        u, s, vt = _svd(base.weight)
        factor_a = layer._psoft_A_cache["default"]
        factor_b = layer._psoft_B_cache["default"]
        assert tuple(factor_a.shape) == (3, 8) and tuple(factor_b.shape) == (6, 3)
        # A has orthonormal rows (what Theorem 4.1 needs), and scaling * B A is the principal part.
        assert span(factor_a @ factor_a.mT, lucid.eye(3)) <= _TOLERANCE
        principal = [
            [sum(u[i][k] * s[k] * vt[k][j] for k in range(3)) for j in range(8)] for i in range(6)
        ]
        scaling = layer.scaling["default"]
        assert span(
            scaling * (factor_b @ factor_a), lucid.tensor(principal)
        ) <= _TOLERANCE * magnitude(lucid.tensor(principal))


class TestUpdate:
    @pytest.mark.parametrize("init", ["psoft_init", "pissa_init"])
    @pytest.mark.parametrize("shape", [(8, 6), (6, 8)], ids=["tall", "wide"])
    def test_the_delta_is_the_papers_a_c_minus_i_b(
        self, init: str, shape: tuple[int, int], device: str
    ) -> None:
        layer, base = _layer(in_features=shape[0], out_features=shape[1], ab_svd_init=init)
        _move(layer)
        u, s, vt = _svd(base.weight)
        layer.to(device)
        expected = lucid.tensor(forms.psoft_delta(u, s, vt, _closed_inner(layer), 3, init))
        delta = layer.get_delta_weight("default")
        assert span(delta, expected.to(device)) <= _TOLERANCE * magnitude(expected)
        x = lucid.randn(4, shape[0], device=device)
        with lucid.no_grad():
            assert span(layer(x), base(x) + x @ expected.to(device).mT) <= _TOLERANCE * magnitude(
                layer(x)
            )

    def test_the_scale_alpha_does_not_change_what_the_adapter_computes(self) -> None:
        first, base = _layer(psoft_alpha=3.0)
        second, _ = _layer(psoft_alpha=30.0)
        _move(first)
        with lucid.no_grad():
            for p, q in zip(
                first._adapter_parameters("default"),
                second._adapter_parameters("default"),
                strict=True,
            ):
                q.copy_(p)
        x = lucid.randn(4, 8)
        assert span(first(x), second(x)) <= _TOLERANCE * magnitude(first(x))

    def test_dropout_acts_on_the_input_in_training_only(self) -> None:
        layer, base = _layer(psoft_dropout=0.5, init_weights=False)
        x = lucid.randn(64, 8)
        layer.eval()
        first = layer(x)
        assert span(layer(x), first) == 0.0
        layer.train()
        assert span(layer(x), first) > 1e-4

    def test_the_neumann_series_is_the_truncated_transform(self) -> None:
        layer, _ = _layer(use_cayley_neumann=True, num_cayley_neumann_terms=4)
        _move(layer)
        orth = layer._orth("default")
        q = forms.skew(orth.weight.tolist(), 3)
        eye = lucid.eye(3)
        q_t = lucid.tensor(q)
        # I + 2Q + 2Q^2 + Q^3: the last of the four terms is halved.
        series = eye + 2 * q_t + 2 * (q_t @ q_t) + q_t @ q_t @ q_t
        expected = lucid.tensor(
            forms.psoft_inner(series.tolist(), orth.vector_b.tolist(), orth.vector_a.tolist())
        )
        assert span(orth.get_matrix(), expected) <= _TOLERANCE * magnitude(expected)

    def test_a_long_series_approaches_the_exact_transform_of_a_small_generator(self) -> None:
        exact, _ = _layer(psoft_mag_a=False, psoft_mag_b=False)
        series, _ = _layer(
            psoft_mag_a=False,
            psoft_mag_b=False,
            use_cayley_neumann=True,
            num_cayley_neumann_terms=30,
        )
        with lucid.no_grad():
            for layer in (exact, series):
                layer._orth("default").weight.copy_(lucid.tensor([0.05, -0.04, 0.03]))
        assert (
            span(exact._orth("default").get_matrix(), series._orth("default").get_matrix()) <= 1e-5
        )

    def test_the_generator_is_rescaled_to_the_bound_when_it_exceeds_it(self) -> None:
        layer, _ = _layer(
            psoft_mag_a=False,
            psoft_mag_b=False,
            use_cayley_neumann=True,
            num_cayley_neumann_terms=2,
            cayley_neumann_eps=0.5,
        )
        with lucid.no_grad():
            layer._orth("default").weight.copy_(lucid.tensor([3.0, -4.0, 1.0]))
        q = lucid.tensor(forms.skew([3.0, -4.0, 1.0], 3))
        norm = float((q * q).sum().sqrt().item())
        bounded = q * (0.5 / (norm + 1e-12))
        # Two terms: I + 2Q of the rescaled generator.
        expected = lucid.eye(3) + 2 * bounded
        assert span(layer._orth("default").get_matrix(), expected) <= _TOLERANCE
        # Below the bound nothing is rescaled.
        with lucid.no_grad():
            layer._orth("default").weight.copy_(lucid.tensor([0.1, 0.0, 0.0]))
        small = lucid.eye(3) + 2 * lucid.tensor(forms.skew([0.1, 0.0, 0.0], 3))
        assert span(layer._orth("default").get_matrix(), small) <= _TOLERANCE

    def test_a_rank_of_one_has_no_generator_and_scales_only(self) -> None:
        layer, base = _layer(r=1)
        assert tuple(layer._orth("default").weight.shape) == (0,)
        with lucid.no_grad():
            layer._orth("default").vector_b.copy_(lucid.tensor([2.0]))
        x = lucid.randn(3, 8)
        # R = [[1]], C = [[2]]: the update doubles the principal component.
        u, s, vt = _svd(base.weight)
        principal = lucid.tensor([[u[i][0] * s[0] * vt[0][j] for j in range(8)] for i in range(6)])
        assert span(layer(x), base(x) + x @ principal.mT) <= _TOLERANCE * magnitude(layer(x))


class TestOrthogonality:
    """The paper's guarantee (section 4.2, Theorem 4.1): angles and norms of the principal part survive."""

    def test_strict_orthogonality_keeps_the_gram_matrix_of_the_principal_part(self) -> None:
        layer, base = _layer(psoft_mag_a=False, psoft_mag_b=False)
        _move(layer)
        weight = base.weight
        u, s, vt = _svd(weight)
        principal = lucid.tensor(
            [[sum(u[i][k] * s[k] * vt[k][j] for k in range(3)) for j in range(8)] for i in range(6)]
        )
        tuned = principal + layer.get_delta_weight("default")
        # The paper's columns are the rows of this layout's weight (it writes ``h = W^T x``): their
        # norms are the diagonal of the Gram matrix and their angles the rest of it.
        gram = principal @ principal.mT
        assert span(tuned @ tuned.mT, gram) <= _TOLERANCE * magnitude(gram)

    def test_the_inner_matrix_is_orthogonal_until_the_vectors_move(self) -> None:
        layer, _ = _layer(psoft_mag_a=False, psoft_mag_b=False)
        with lucid.no_grad():
            layer._orth("default").weight.copy_(lucid.tensor([0.4, -0.3, 0.2]))
        matrix = layer._orth("default").get_matrix()
        assert span(matrix.mT @ matrix, lucid.eye(3)) <= _TOLERANCE
        relaxed, _ = _layer()
        with lucid.no_grad():
            relaxed._orth("default").weight.copy_(lucid.tensor([0.4, -0.3, 0.2]))
            relaxed._orth("default").vector_a.copy_(lucid.tensor([1.5, 1.0, 0.5]))
        loose = relaxed._orth("default").get_matrix()
        assert span(loose.mT @ loose, lucid.eye(3)) > 1e-2

    def test_a_free_matrix_is_not_kept_orthogonal(self) -> None:
        layer, _ = _layer(psoft_orth=False, psoft_mag_a=False, psoft_mag_b=False)
        with lucid.no_grad():
            layer._orth("default").weight.copy_(lucid.eye(3) * 2.0)
        matrix = layer._orth("default").get_matrix()
        assert span(matrix.mT @ matrix, lucid.eye(3)) > 1.0


class TestSvd:
    def test_the_factors_do_not_depend_on_the_device_the_weight_is_on(self, device: str) -> None:
        # A singular pair's sign is the backend's to choose, and the trained matrix is relative to it.
        lucid.manual_seed(1)
        cpu_base = nn.Linear(8, 6)
        there = nn.Linear(8, 6).to(device)
        with lucid.no_grad():
            there.weight.copy_(cpu_base.weight.to(device))
            there.bias.copy_(cpu_base.bias.to(device))
        config = PsoftConfig(r=3, target_modules=["x"])
        here = psoft.Linear(cpu_base, "default", config)
        moved = psoft.Linear(there, "default", config)
        for name in ("_psoft_A_cache", "_psoft_B_cache"):
            a, b = getattr(here, name)["default"], getattr(moved, name)["default"].to("cpu")
            assert span(a, b) == 0.0

    def test_the_low_rank_svd_is_reproducible_and_leaves_the_global_generator_alone(self) -> None:
        first, _ = _layer(psoft_svd="lowrank", psoft_svd_lowrank_niter=4, random_seed=5)
        second, _ = _layer(psoft_svd="lowrank", psoft_svd_lowrank_niter=4, random_seed=5)
        different, _ = _layer(psoft_svd="lowrank", psoft_svd_lowrank_niter=0, random_seed=6)
        # ``_layer`` seeds the base weight itself, so the draw of the SVD is what differs.
        assert span(first._psoft_A_cache["default"], second._psoft_A_cache["default"]) == 0.0
        assert span(first._psoft_A_cache["default"], different._psoft_A_cache["default"]) > 0.0

    def test_the_global_generator_is_where_it_was_after_the_svd(self) -> None:
        lucid.manual_seed(2)
        base = nn.Linear(8, 6)
        config = PsoftConfig(r=3, psoft_svd="lowrank", random_seed=11, target_modules=["x"])
        before = lucid.get_rng_state()
        psoft.Linear(base, "default", config)
        assert lucid.get_rng_state().tolist() == before.tolist()

    def test_many_iterations_agree_with_the_exact_svd(self) -> None:
        exact, base = _layer(psoft_svd="full")
        near, _ = _layer(psoft_svd="lowrank", psoft_svd_lowrank_niter=40, random_seed=3)
        product = exact._psoft_B_cache["default"] @ exact._psoft_A_cache["default"]
        approx = near._psoft_B_cache["default"] @ near._psoft_A_cache["default"]
        assert span(product, approx) <= 1e-3 * magnitude(product)


class TestMerging:
    def test_merging_adds_the_delta_and_unmerging_takes_it_out(self) -> None:
        layer, base = _layer(init_weights=False)
        weight = base.weight.clone()
        delta = layer.get_delta_weight("default")
        x = lucid.randn(5, 8)
        with lucid.no_grad():
            adapted = layer(x)
        layer.merge()
        assert span(base.weight, weight + delta) <= _TOLERANCE * magnitude(weight)
        assert span(layer(x), adapted) <= _TOLERANCE * magnitude(adapted)
        layer.unmerge()
        assert span(base.weight, weight) <= _TOLERANCE * magnitude(weight)


class TestErrors:
    def test_a_rank_beyond_the_smaller_side_is_refused(self) -> None:
        with pytest.raises(ValueError, match="min.in_features, out_features"):
            _layer(in_features=8, out_features=6, r=7)

    def test_only_a_linear_is_adapted(self) -> None:
        net = nn.Sequential(nn.Conv2d(2, 4, 3))
        with pytest.raises(ValueError, match="not supported"):
            get_peft_model(net, PsoftConfig(r=2, target_modules=["0"]))

    def test_a_refused_adapter_leaves_the_model_as_it_was(self) -> None:
        net = nn.Sequential(nn.Linear(4, 6))
        with pytest.raises(ValueError, match="min.in_features, out_features"):
            get_peft_model(net, PsoftConfig(r=5, target_modules=["0"]))
        assert type(net[0]) is nn.Linear and all(p.requires_grad for p in net.parameters())

    def test_fan_in_fan_out_is_reset_with_a_warning(self) -> None:
        net = nn.Sequential(nn.Linear(4, 4))
        with pytest.warns(UserWarning, match="fan_in_fan_out"):
            peft = get_peft_model(net, PsoftConfig(r=2, fan_in_fan_out=True, target_modules=["0"]))
        assert peft.peft_config["default"].fan_in_fan_out is False  # type: ignore[attr-defined]


class TestAdapters:
    def test_the_factors_are_not_in_the_file_and_leave_with_their_adapter(self) -> None:
        net = nn.Sequential(nn.Linear(6, 6))
        peft = get_peft_model(net, PsoftConfig(r=2, target_modules=["0"]), adapter_name="a")
        peft.add_adapter("b", PsoftConfig(r=3, target_modules=["0"]))
        layer = peft.base_model.model[0]
        assert not any("cache" in key for key in peft.state_dict())
        assert layer.r == {"a": 2, "b": 3}
        peft.delete_adapter("b")
        assert list(layer._psoft_A_cache) == ["a"] and list(layer._psoft_B_cache) == ["a"]


def test_the_tuner_wraps_the_linear_layers_and_unloads_them() -> None:
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    peft = get_peft_model(net, PsoftConfig(r=2, target_modules=["0", "2"]))
    assert isinstance(peft, PeftModel) and isinstance(peft.base_model, PsoftModel)
    assert type(net[0]) is psoft.Linear
    assert peft.unload() is net and type(net[0]) is nn.Linear


# ── the reference adapter library ────────────────────────────────────────────

_PARITY_OPTIONS = [
    {},
    {"psoft_orth": False},
    {"psoft_mag_a": False},
    {"psoft_mag_a": False, "psoft_mag_b": False},
    {"ab_svd_init": "pissa_init"},
    {"use_cayley_neumann": True},
    {"use_cayley_neumann": True, "num_cayley_neumann_terms": 3, "cayley_neumann_eps": 0.5},
    {"psoft_alpha": 7.0},
]


@pytest.mark.parity
@pytest.mark.parametrize("options", _PARITY_OPTIONS, ids=lambda o: ",".join(o) or "default")
@pytest.mark.parametrize("shape", [(10, 12), (12, 8)], ids=["wide", "tall"])
def test_a_layer_matches_the_reference(
    ref: ModuleType, options: dict[str, object], shape: tuple[int, int]
) -> None:
    library = adapter_library()
    lucid.manual_seed(5)
    net = nn.Sequential(nn.Linear(*shape))
    twin = ref.nn.Sequential(ref.nn.Linear(*shape))
    x = lucid.randn(4, shape[0])
    config = {"r": 3, "init_weights": False, **options}
    peft = get_peft_model(net, PsoftConfig(target_modules=["0"], **config))  # type: ignore[arg-type]
    mine = peft.base_model.model[0]
    with ref.no_grad():
        twin[0].weight.copy_(ref.tensor(mine.base_layer.weight.tolist()))
        twin[0].bias.copy_(ref.tensor(mine.base_layer.bias.tolist()))
    theirs = library.get_peft_model(twin, library.PsoftConfig(target_modules=["0"], **config))
    other = theirs.base_model.model[0]
    factor_a, factor_b = other._get_psoft_ab_cache_buffers("default")
    # What the two compute with: the principal part, which does not depend on the SVD's signs.
    assert (
        gap(
            ref,
            mine._psoft_B_cache["default"] @ mine._psoft_A_cache["default"],
            ref.matmul(factor_b, factor_a),
        )
        < 1e-5
    )
    # The signs of a singular pair are the decomposition's to choose; the trained matrix is relative to them.
    with lucid.no_grad():
        mine._psoft_A_cache["default"].copy_(lucid.tensor(factor_a.tolist()))
        mine._psoft_B_cache["default"].copy_(lucid.tensor(factor_b.tolist()))
    with ref.no_grad():
        for name, value in other.psoft_R["default"].named_parameters():
            value.copy_(ref.tensor(getattr(mine._orth("default"), name).tolist()))
    assert gap(ref, peft(x), theirs(ref.tensor(x.tolist()))) < 1e-5
    assert gap(ref, mine.get_delta_weight("default"), other.get_delta_weight("default")) < 1e-5
    peft.base_model.merge_adapter()
    theirs.base_model.merge_adapter()
    assert gap(ref, mine.base_layer.weight, other.base_layer.weight) < 1e-5
    peft.base_model.unmerge_adapter()
    theirs.base_model.unmerge_adapter()
    assert gap(ref, mine.base_layer.weight, other.base_layer.weight) < 1e-5


@pytest.mark.parity
@pytest.mark.parametrize(
    "options", [{}, {"psoft_orth": False, "psoft_mag_a": False}], ids=["default", "free"]
)
def test_the_state_dict_has_the_references_keys_and_shapes(
    ref: ModuleType, options: dict[str, object]
) -> None:
    library = adapter_library()
    mine = get_peft_model(
        nn.Sequential(nn.Linear(10, 12)),
        PsoftConfig(r=4, target_modules=["0"], **options),  # type: ignore[arg-type]
    )
    theirs = library.get_peft_model(
        ref.nn.Sequential(ref.nn.Linear(10, 12)),
        library.PsoftConfig(r=4, target_modules=["0"], **options),
    )
    expected = {k: tuple(v.shape) for k, v in library.get_peft_model_state_dict(theirs).items()}
    assert {k: tuple(v.shape) for k, v in get_peft_model_state_dict(mine).items()} == expected


@pytest.mark.parity
def test_the_config_file_has_the_references_keys() -> None:
    library = adapter_library()
    ours = PsoftConfig(r=4, psoft_mag_a=False, target_modules=["a"]).to_dict()
    theirs = library.PsoftConfig(r=4, psoft_mag_a=False, target_modules=["a"]).to_dict()
    assert set(ours) == set(theirs) - {"peft_version"}
    assert all(
        ours[k] == (sorted(v) if isinstance(v, set) else v) for k, v in theirs.items() if k in ours
    )
